"""Plaintext matrix contraction in fully split finite-field NTT slots."""

from functools import lru_cache

import torch

from ._transform_tables import device_powers, resolve_device
from .cyclotomic import CyclotomicNTTPlan
from .modular import arithmetic_context, residues
from .multivariate_poly import MultivariatePoly
from .ntt import NTTPlan
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
    """Batched modular matmul, reducing after each inner-index product.

    Shapes are (W, row, column, split). No unreduced int64 dot-product sums:
    even a short dot product of 30-bit residues can overflow int64.
    """
    output = torch.zeros_like(left)
    for inner_index in range(left.shape[2]):
        term = arithmetic.multiply(
            left[:, :, inner_index, :].unsqueeze(2),
            right[:, inner_index, :, :].unsqueeze(1),
        )
        output = arithmetic.add(output, term)
    return output


class MatrixNTTPlan:
    """NTT/INTT between coefficients and (phi(p),n,n,2) matrix slots.

    Last channels are evaluation at +I and -I, not Gaussian real/imaginary.
    Positive roots follow powers of five (X/Y) and powers of two (W).
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
            raise ValueError("requires equal native X/Y degrees and a power-of-three W NTT")
        self.n, self.p = plans[X_dim].length, plans[W_dim].p
        self.slot_shape = (plans[W_dim].length, self.n, self.n, 2)
        # Match MatrixEncoder: W first, remaining axes in ring order. This
        # matters when X/Y are reordered, since transposition reverses products.
        self._slot_dims = (W_dim,) + tuple(dim for dim in range(3) if dim != W_dim) + (3,)
        self._coefficient_dims = tuple(self._slot_dims.index(dim) for dim in range(4))
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
                    ring.q, roots[::-1], self.n, str(self.device)
                ).T.reshape(shape)
                positive_order = [(pow(5, slot, 4 * self.n) - 1) // 4 for slot in range(self.n)]
                negative_order = [(-index) % self.n for index in positive_order]
            else:
                shared_plan = plan
                stride = self.p // 3

                def compact_index(exponent, stride=stride):
                    root_class = exponent % 3
                    return (root_class - 1) * stride + (exponent - root_class) // 3

                exponents = [pow(2, slot, self.p) for slot in range(plan.length)]
                positive_order = [compact_index(exponent) for exponent in exponents]
                negative_order = [compact_index((-exponent) % self.p) for exponent in exponents]
            permutations = tuple(
                torch.tensor(order, device=self.device)
                for order in (positive_order, negative_order)
            )
            inverse_permutations = tuple(torch.argsort(order) for order in permutations)
            self._forward_steps.append(
                (dim, shared_plan._kernel(self.device, dim=dim), twist, permutations)
            )
            self._inverse_steps.append(
                (
                    dim,
                    shared_plan._kernel(self.device, dim=dim, inverse=True),
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

    def _check_polynomial(self, polynomial):
        if not isinstance(polynomial, MultivariatePoly):
            raise TypeError("expected a MultivariatePoly")
        if polynomial.ring != self.ring or polynomial.device != self.device:
            raise ValueError("polynomial must have the plan's ring and device")

    def _forward_residues(self, coefficients):
        arithmetic = self._arithmetic
        real, imag = arithmetic.encode(coefficients).unbind(-1)
        imaginary_part = arithmetic.multiply(imag, self._encoded_imaginary)
        values = torch.stack(
            (arithmetic.add(real, imaginary_part), arithmetic.subtract(real, imaginary_part)), -1
        )
        for dim, transform, twist, permutations in self._forward_steps:
            if twist is not None:
                values = arithmetic.multiply(values, twist)
            values = _reorder_branches(transform(values), dim, permutations)
        return values.permute(self._slot_dims)

    def _inverse_residues(self, slots):
        arithmetic = self._arithmetic
        values = slots.permute(self._coefficient_dims)
        for dim, transform, twist, permutations in self._inverse_steps:
            values = transform(_reorder_branches(values, dim, permutations))
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
        slots = _matrix_product_residues(
            self._forward_residues(left._coefficients),
            self._forward_residues(right._coefficients),
            self._arithmetic,
        )
        return self._inverse_residues(slots)


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
