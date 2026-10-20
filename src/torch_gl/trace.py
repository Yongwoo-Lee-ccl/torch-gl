"""Matrix multiplication by an exact algebra trace in GL's auxiliary ring."""

from functools import lru_cache

import torch

from ._native_transform import NativeTransform
from ._transform_tables import resolve_device
from .matrix import _matrix_product_residues
from .multivariate_poly import MultivariatePoly
from .rings import PolynomialRing
from .rns import RNSPoly


class AuxiliaryMatrixPlan:
    """W/V matrix contraction, batched over X (Theorem 8, 2026/956).

    For Phi_p with p=d^k, Tr(U^r) is phi(p) if p|r, -p/d if
    (p/d)|r but p does not divide r, and 0 otherwise. These are Ramanujan
    sums, rather than the orthogonal-power-basis trace used for X^n-i.
    Only X is NTT-transformed. A sparse trace Gram matrix contracts W/V
    coefficients, followed by the inverse X transform. All data stays int32
    Montgomery form except the checked int64 sparse summation.
    """

    def __init__(self, ring: PolynomialRing, *, device="cpu"):
        if {a.name for a in ring.axes} != {"X", "W", "V"}:
            raise ValueError("auxiliary multiplication requires X/W/V axes")
        self.ring, self.device = ring, resolve_device(device)
        names = tuple(a.name for a in ring.axes)
        self._dims = (names.index("W"), names.index("V"), names.index("X"), 3)
        self._inverse_dims = tuple(self._dims.index(i) for i in range(4))
        row, col = ring.axes[names.index("W")], ring.axes[names.index("V")]
        if row.modulus != col.modulus or not row.cyclotomic_order:
            raise ValueError("W/V must have the same cyclotomic relation")
        p, width = row.cyclotomic_order, row.degree
        stride = p - width
        spectator = PolynomialRing(ring.q, (ring.axes[names.index("X")],))
        self._transform = NativeTransform(spectator, self.device, batch_ndim=2)
        self._arithmetic = self._transform.arithmetic
        sources, targets, weights = [], [], []
        for alpha in range(width):
            for beta in range(width):
                residue = (alpha + beta) % p
                weight = width if residue == 0 else -stride if residue % stride == 0 else 0
                if weight:
                    sources.append(beta)
                    targets.append(alpha)
                    weights.append(self._arithmetic.constant(weight))
        self._sources = torch.tensor(sources, device=self.device)
        self._targets = torch.tensor(targets, device=self.device)
        self._weights = torch.tensor(weights, dtype=torch.int32, device=self.device).reshape(
            1, -1, 1, 1
        )

    def _slots(self, polynomial):
        if not isinstance(polynomial, MultivariatePoly):
            raise TypeError("expected a MultivariatePoly")
        if polynomial.ring != self.ring or polynomial.device != self.device:
            raise ValueError("polynomial must have the auxiliary plan's ring and device")
        return (
            self._transform.forward(polynomial._coefficients.permute(self._dims))
            .permute(2, 0, 1, 3)
            .contiguous()
        )

    def multiply(self, left, right):
        left_slots, right_slots = self._slots(left), self._slots(right)
        terms = self._arithmetic.multiply(right_slots.index_select(1, self._sources), self._weights)
        prepared = torch.zeros_like(right_slots, dtype=torch.int64).index_add_(
            1, self._targets, terms.to(torch.int64)
        )
        prepared = prepared.remainder_(self.ring.q).to(torch.int32)
        slots = _matrix_product_residues(left_slots, prepared, self._arithmetic)
        coefficients = self._transform.inverse(slots.permute(1, 2, 0, 3))
        return MultivariatePoly._from_owned(
            self.ring, coefficients.permute(self._inverse_dims).contiguous()
        )


@lru_cache(maxsize=32)
def _auxiliary_plan(ring, device):
    return AuxiliaryMatrixPlan(ring, device=device)


def auxiliary_matrix_multiply(left: RNSPoly, right: RNSPoly) -> RNSPoly:
    """W/V slot-matrix product A @ B, independently at every X slot."""
    if not isinstance(left, RNSPoly) or not isinstance(right, RNSPoly):
        raise TypeError("auxiliary matrix multiplication requires two RNSPoly values")
    if left.ring != right.ring or left.device != right.device:
        raise ValueError("operands must have the same auxiliary ring and device")
    return RNSPoly(
        left.ring,
        tuple(
            _auxiliary_plan(a.ring, left.device).multiply(a, b)
            for a, b in zip(left.limbs, right.limbs, strict=True)
        ),
    )
