"""Plaintext matrix contraction in fully split finite-field NTT slots."""

from functools import lru_cache

import torch

from ._number_theory import odd_prime_power, unit_generator
from ._transform_tables import device_powers, resolve_device
from .cyclotomic import CyclotomicNTTPlan
from .modular import arithmetic_context, residues
from .multivariate_poly import MultivariatePoly
from .ntt import NTTPlan, _digit_order
from .rings import PolynomialRing
from .rns import RNSPoly
from .twisted import GaussianNTTPlan


def _reorder_branches(values, dim, permutations):
    positive, negative = values.unbind(-1)
    return torch.stack(
        (positive.index_select(dim, permutations[0]), negative.index_select(dim, permutations[1])),
        dim=-1,
    )


def _matrix_product_residues(left, right, arithmetic):
    """Exact modular GEMM with two 15-bit parts and one float64 bmm.

    Shapes are (W,row,column,split). Each partial dot product is a sum of
    nonnegative integer products smaller than 2**30. Under the checked bound,
    all partial sums and their cross sum are exact float64 integers, including
    FMA evaluation. Recombination reduces before shifts, avoiding int64 overflow.
    Inputs and output remain int32 Montgomery residues.
    """
    n = left.shape[2]
    part_bits, part_mask = 15, (1 << 15) - 1
    if 2 * n * part_mask**2 > 1 << 53:
        raise ValueError("matrix dimension exceeds the exact float64 partial-sum bound")
    left_matrices = left.permute(0, 3, 1, 2).reshape(-1, n, n)
    right_matrices = right.permute(0, 3, 1, 2).reshape(-1, n, n)
    left_packed = torch.cat(
        (
            (left_matrices & part_mask).to(torch.float64),
            (left_matrices >> part_bits).to(torch.float64),
        ),
        dim=1,
    )
    right_packed = torch.cat(
        (
            (right_matrices & part_mask).to(torch.float64),
            (right_matrices >> part_bits).to(torch.float64),
        ),
        dim=2,
    )
    parts = torch.bmm(left_packed, right_packed)
    low = parts[:, :n, :n].to(torch.int64).remainder(arithmetic.q)
    cross = (parts[:, :n, n:] + parts[:, n:, :n]).to(torch.int64).remainder(arithmetic.q)
    high = parts[:, n:, n:].to(torch.int64).remainder(arithmetic.q)
    # Each reduced part is <q<2**30: the complete expression is <2**61.
    combined = (
        (low + (cross << part_bits) + (high << (2 * part_bits)))
        .remainder(arithmetic.q)
        .to(torch.int32)
    )
    # GEMM multiplies two Montgomery operands (R**2); one REDC restores R.
    output = arithmetic.decode(combined)
    return output.reshape(left.shape[0], 2, n, n).permute(0, 2, 3, 1).contiguous()


class MatrixNTTPlan:
    """NTT/INTT between coefficients and (phi(p),n,n,2) matrix slots.

    Last channels are evaluation at +I and -I, not Gaussian real/imaginary.
    Positive roots follow powers of five (X/Y) and a unit generator (W).
    The negative branch uses their inverse roots. Public tensors are normal
    int32 residues; contraction stays in Montgomery form internally.
    """

    def __init__(self, ring: PolynomialRing, *, device: torch.device | str = "cpu"):
        if not isinstance(ring, PolynomialRing):
            raise TypeError("ring must be a PolynomialRing")
        names = tuple(axis.name for axis in ring.axes)
        if set(names) != {"X", "Y", "W"}:
            raise ValueError("matrix NTT requires X, Y, W axes")
        self.ring, self.device = ring, resolve_device(device)
        X_dim, Y_dim, W_dim = (names.index(name) for name in ("X", "Y", "W"))
        plans = ring.transform_plans
        if (
            not isinstance(plans[X_dim], GaussianNTTPlan)
            or not isinstance(plans[Y_dim], GaussianNTTPlan)
            or not isinstance(plans[W_dim], CyclotomicNTTPlan)
            or ring.axes[X_dim].degree != ring.axes[Y_dim].degree
        ):
            raise ValueError("requires equal native X/Y degrees and a native W NTT")
        self.n, self.p = plans[X_dim].length, plans[W_dim].p
        self.slot_shape = (plans[W_dim].length, self.n, self.n, 2)
        # Match MatrixEncoder: W first, remaining axes in ring order. This
        # matters when X/Y are reordered, since transposition reverses products.
        self._slot_dims = (W_dim,) + tuple(dim for dim in range(3) if dim != W_dim) + (3,)
        self._coefficient_dims = tuple(self._slot_dims.index(dim) for dim in range(4))
        self._transpose_dims = (X_dim, Y_dim)
        self._arithmetic = arithmetic_context(ring.q)
        root = plans[X_dim].root
        self.imaginary_root = pow(root, self.n, ring.q)
        self.W_root = plans[W_dim].root
        self.X_root = root
        self._forward_steps, self._inverse_steps = [], []
        for dim, plan in enumerate(plans):
            twist = inverse_twist = None
            if dim in (X_dim, Y_dim):
                shared_plan = NTTPlan(ring.q, self.n, pow(root, 4, ring.q))
                shape = [1, 1, 1, 2]
                shape[dim] = self.n
                roots = (root, pow(root, -1, ring.q))
                twist = device_powers(ring.q, roots, self.n, str(self.device)).T.reshape(shape)
                inverse_twist = device_powers(
                    ring.q,
                    roots[::-1],
                    self.n,
                    str(self.device),
                    scale=pow(self.n, -1, ring.q),
                ).T.reshape(shape)
                inverse_kernel = shared_plan._kernel(
                    self.device, dim=dim, inverse=True, permuted=True, normalize=False
                )
                positive_order = [(pow(5, slot, 4 * self.n) - 1) // 4 for slot in range(self.n)]
                negative_order = [(-index) % self.n for index in positive_order]
                digit_order = _digit_order(self.n)
            else:
                shared_plan = plan
                inverse_kernel = shared_plan._kernel(
                    self.device, dim=dim, inverse=True, permuted=True
                )
                prime = odd_prime_power(self.p)[0]
                stride = self.p // prime

                def compact_index(exponent, stride=stride, prime=prime):
                    if prime != 3:
                        return exponent - 1
                    root_class = exponent % 3
                    return (root_class - 1) * stride + (exponent - root_class) // 3

                exponents = [
                    pow(unit_generator(self.p), slot, self.p) for slot in range(plan.length)
                ]
                positive_order = [compact_index(exponent) for exponent in exponents]
                negative_order = [compact_index((-exponent) % self.p) for exponent in exponents]
                digit_order = (
                    tuple(range(plan.length))
                    if prime != 3
                    else tuple(
                        branch * stride + frequency
                        for branch in range(2)
                        for frequency in _digit_order(stride)
                    )
                )
            # Fold digit reversal into the already-required canonical slot
            # permutation, rather than adding a separate reorder pass.
            natural_to_digit = [0] * len(digit_order)
            for position, frequency in enumerate(digit_order):
                natural_to_digit[frequency] = position
            permutations = tuple(
                torch.tensor([natural_to_digit[index] for index in order], device=self.device)
                for order in (positive_order, negative_order)
            )
            inverse_permutations = tuple(torch.argsort(order) for order in permutations)
            self._forward_steps.append(
                (
                    dim,
                    shared_plan._kernel(self.device, dim=dim, permuted=True),
                    twist,
                    permutations,
                )
            )
            self._inverse_steps.append(
                (
                    dim,
                    inverse_kernel,
                    inverse_twist,
                    inverse_permutations,
                )
            )
        self._inverse_steps.reverse()
        self._encoded_imaginary = self._arithmetic.constant(self.imaginary_root)
        self._inverse_two = self._arithmetic.constant(pow(2, -1, ring.q))
        self._inverse_two_imaginary = self._arithmetic.constant(
            pow(2 * self.imaginary_root, -1, ring.q)
        )
        self._W_forward = plans[W_dim]._kernel(self.device, dim=W_dim, permuted=True)
        self._W_inverse = plans[W_dim]._kernel(self.device, dim=W_dim, inverse=True, permuted=True)
        self._trace_factor = self._arithmetic.constant(self.n)
        self._split_imaginary = torch.tensor(
            (
                self._encoded_imaginary,
                self._arithmetic.constant(-self.imaginary_root),
            ),
            dtype=torch.int32,
            device=self.device,
        )

    def _check_polynomial(self, polynomial):
        if not isinstance(polynomial, MultivariatePoly):
            raise TypeError("expected a MultivariatePoly")
        if polynomial.ring != self.ring or polynomial.device != self.device:
            raise ValueError("polynomial must have the plan's ring and device")

    def _forward_residues(self, coefficients, *, canonical=True):
        arithmetic = self._arithmetic
        real, imag = arithmetic.encode(coefficients).unbind(-1)
        imaginary_part = arithmetic.multiply(imag, self._encoded_imaginary)
        values = torch.stack(
            (arithmetic.add(real, imaginary_part), arithmetic.subtract(real, imaginary_part)), -1
        )
        for dim, transform, twist, permutations in self._forward_steps:
            if twist is not None:
                values = arithmetic.multiply(values, twist)
            values = transform(values)
            if canonical:
                values = _reorder_branches(values, dim, permutations)
        return values.permute(self._slot_dims)

    def _inverse_residues(self, slots, *, canonical=True):
        arithmetic = self._arithmetic
        values = slots.permute(self._coefficient_dims)
        for dim, transform, twist, permutations in self._inverse_steps:
            if canonical:
                values = _reorder_branches(values, dim, permutations)
            values = transform(values)
            if twist is not None:
                values = arithmetic.multiply(values, twist)
        positive, negative = values.unbind(-1)
        real = arithmetic.multiply(arithmetic.add(positive, negative), self._inverse_two)
        imag = arithmetic.multiply(
            arithmetic.subtract(positive, negative), self._inverse_two_imaginary
        )
        coefficients = arithmetic.decode(torch.stack((real, imag), -1))
        return MultivariatePoly._from_owned(self.ring, coefficients)

    def forward(self, polynomial: MultivariatePoly) -> torch.Tensor:
        self._check_polynomial(polynomial)
        return self._arithmetic.decode(self._forward_residues(polynomial._coefficients))

    def inverse(self, slots: torch.Tensor) -> MultivariatePoly:
        if tuple(slots.shape) != self.slot_shape or slots.device != self.device:
            raise ValueError("slots must have the plan's shape and device")
        return self._inverse_residues(self._arithmetic.encode(residues(slots, self.ring.q)))

    def multiply(self, left: MultivariatePoly, right: MultivariatePoly) -> MultivariatePoly:
        self._check_polynomial(left)
        self._check_polynomial(right)
        # Contract X/Y coefficient matrices, transforming W alone. For the
        # ordinary slot product, sum_k zeta_k**(s+t) is n at s+t=0 and
        # n*i at s+t=n. Thus C=n*L*prepared_R, with prepared_R[0]=R[0]
        # and prepared_R[s]=i*R[n-s] for 1<=s<n.
        left_slots = self._coefficient_slots(left._coefficients)
        right_slots = self._right_coefficient_slots(right._coefficients)
        return self._product_from_coefficient_slots(left_slots, right_slots)

    def _right_coefficient_slots(self, coefficients):
        right_slots = self._coefficient_slots(coefficients)
        return torch.cat(
            (
                right_slots[:, :1],
                self._arithmetic.multiply(right_slots[:, 1:].flip(1), self._split_imaginary),
            ),
            dim=1,
        )

    def _multiply_ciphertext_components(self, left_b, left_a, right_b, right_a):
        """Reuse four W transforms for the four encrypted product components."""
        left_b_slots = self._coefficient_slots(left_b._coefficients)
        left_a_slots = self._coefficient_slots(left_a._coefficients)
        right_b_slots = self._right_coefficient_slots(
            right_b._coefficients.transpose(*self._transpose_dims)
        )
        right_a_slots = self._right_coefficient_slots(
            right_a._coefficients.transpose(*self._transpose_dims)
        )
        return (
            self._product_from_coefficient_slots(left_b_slots, right_b_slots),
            self._product_from_coefficient_slots(left_a_slots, right_b_slots),
            self._product_from_coefficient_slots(left_b_slots, right_a_slots),
            self._product_from_coefficient_slots(left_a_slots, right_a_slots),
        )

    def _product_from_coefficient_slots(self, left_slots, right_slots):
        slots = _matrix_product_residues(
            left_slots,
            right_slots,
            self._arithmetic,
        )
        values = self._W_inverse(
            self._arithmetic.multiply(slots, self._trace_factor).permute(self._coefficient_dims)
        )
        positive, negative = values.unbind(-1)
        real = self._arithmetic.multiply(
            self._arithmetic.add(positive, negative), self._inverse_two
        )
        imag = self._arithmetic.multiply(
            self._arithmetic.subtract(positive, negative), self._inverse_two_imaginary
        )
        coefficients = self._arithmetic.decode(torch.stack((real, imag), -1))
        return MultivariatePoly._from_owned(self.ring, coefficients)

    def _coefficient_slots(self, coefficients):
        """W digit-order slots; X/Y remain coefficient indices for contraction."""
        real, imag = self._arithmetic.encode(coefficients).unbind(-1)
        component = self._arithmetic.multiply(imag, self._encoded_imaginary)
        values = torch.stack(
            (
                self._arithmetic.add(real, component),
                self._arithmetic.subtract(real, component),
            ),
            -1,
        )
        return self._W_forward(values).permute(self._slot_dims).contiguous()


@lru_cache(maxsize=32)
def _matrix_plan(ring, device):
    return MatrixNTTPlan(ring, device=device)


def matrix_multiply(left: RNSPoly, right: RNSPoly) -> RNSPoly:
    """Plaintext W-batched matrix multiplication through each prime's NTT."""
    if not isinstance(left, RNSPoly) or not isinstance(right, RNSPoly):
        raise TypeError("matrix_multiply requires two RNSPoly values")
    if left.ring != right.ring or left.device != right.device:
        raise ValueError("operands must have the same RNS ring and device")
    return RNSPoly(
        left.ring,
        tuple(
            _matrix_plan(limb_left.ring, left.device).multiply(limb_left, limb_right)
            for limb_left, limb_right in zip(left.limbs, right.limbs, strict=True)
        ),
    )


def matrix_transpose(polynomial: RNSPoly) -> RNSPoly:
    """Substitute (X,Y,W)->(Y,X,W), without coefficient conjugation."""
    if not isinstance(polynomial, RNSPoly):
        raise TypeError("matrix_transpose requires an RNSPoly")
    names = tuple(axis.name for axis in polynomial.ring.axes)
    if "X" not in names or "Y" not in names:
        raise ValueError("transpose requires X and Y axes")
    X_dim, Y_dim = names.index("X"), names.index("Y")
    if polynomial.ring.axes[X_dim].modulus != polynomial.ring.axes[Y_dim].modulus:
        raise ValueError("transpose requires identical X/Y quotient relations")
    return RNSPoly(
        polynomial.ring,
        tuple(
            MultivariatePoly._from_owned(
                limb.ring, limb._coefficients.transpose(X_dim, Y_dim).contiguous()
            )
            for limb in polynomial.limbs
        ),
    )
