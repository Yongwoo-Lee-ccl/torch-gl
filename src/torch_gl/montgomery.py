"""Single-limb Montgomery arithmetic and explicitly tagged tensor conversions.

R=2**31 and q<2**30. Ordinary products are below 2**60, the cancellation
product is below 2**62, and the reduction accumulator is below 3*2**60.
Wide accumulators fit signed int64. Cancellation intentionally uses int32
low-word wraparound before masking to 31 bits.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import torch

from ._compilation import maybe_product
from ._number_theory import MAX_MODULUS

_RADIX_BITS = 31
_RADIX = 1 << _RADIX_BITS
_RADIX_MASK = _RADIX - 1
MONTGOMERY_RADIX = _RADIX


def _montgomery_product(
    left: torch.Tensor, right: torch.Tensor | int, q: int, negative_inverse: int
) -> torch.Tensor:
    """Return left*right*R^-1 mod q for canonical operands in [0,q)."""
    compiled = maybe_product(left, right, q, negative_inverse)
    if compiled is not None:
        return compiled
    # Stored operands are int32. Only the reduction's local intermediates
    # are widened; fused compiler kernels can keep these in registers.
    product = left.to(torch.int64) * right
    # Cancellation needs only the low 31 bits. Int32 multiplication wraps
    # intentionally; all other wide arithmetic is bounded without overflow.
    cancellation = product.to(torch.int32)
    cancellation.mul_(negative_inverse).bitwise_and_(_RADIX_MASK)
    product.add_(cancellation, alpha=q).bitwise_right_shift_(_RADIX_BITS)
    reduced = product.to(torch.int32)
    return reduced.add_(reduced >= q, alpha=-q)


@dataclass(frozen=True)
class MontgomeryContext:
    """Modulus constants for exact arithmetic with R=2**31."""

    q: int
    negative_inverse: int = field(init=False)
    radix_squared: int = field(init=False)

    def __post_init__(self) -> None:
        if type(self.q) is not int or not 3 <= self.q <= MAX_MODULUS or self.q % 2 == 0:
            raise ValueError(f"Montgomery modulus must be odd and between 3 and {MAX_MODULUS}")
        object.__setattr__(self, "negative_inverse", -pow(self.q, -1, _RADIX) & _RADIX_MASK)
        object.__setattr__(self, "radix_squared", _RADIX * _RADIX % self.q)

    def multiply(self, left: torch.Tensor, right: torch.Tensor | int) -> torch.Tensor:
        return _montgomery_product(left, right, self.q, self.negative_inverse)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.multiply(values, self.radix_squared)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return self.multiply(values, 1)

    def encode_integer(self, value: int) -> int:
        return value * _RADIX % self.q


@lru_cache(maxsize=128)
def montgomery_context(q: int) -> MontgomeryContext:
    return MontgomeryContext(q)


class MontgomeryTensor:
    """A tagged Montgomery value; construct it through to_montgomery_form.

    The modulus travels with the tensor, preventing accidental arithmetic
    between representations or moduli. Exported values are defensive copies.
    """

    __slots__ = ("_values", "_q")

    def __init__(self, values: torch.Tensor, q: int, *, _encoded: bool = False):
        if not _encoded:
            raise TypeError("use to_montgomery_form(values, q) to construct a MontgomeryTensor")
        self._values, self._q = values, q

    @property
    def q(self) -> int:
        return self._q

    @property
    def values(self) -> torch.Tensor:
        return self._values.clone()

    @property
    def device(self) -> torch.device:
        return self._values.device

    def __repr__(self) -> str:
        return (
            f"MontgomeryTensor(q={self.q}, shape={tuple(self._values.shape)}, device={self.device})"
        )


def to_montgomery_form(values: torch.Tensor, q: int) -> MontgomeryTensor:
    """Normalize integer residues and encode once, preserving shape and device."""
    from .modular import residues

    if is_montgomery_form(values):
        raise TypeError("values are already in Montgomery form")
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    context = montgomery_context(q)
    encoded = context.encode(residues(values, q))
    return MontgomeryTensor(encoded, q, _encoded=True)


def to_normal_form(values: MontgomeryTensor) -> torch.Tensor:
    """Decode a tagged Montgomery tensor into canonical torch.int32 residues."""
    if not is_montgomery_form(values):
        raise TypeError("values must be a MontgomeryTensor")
    return montgomery_context(values.q).decode(values._values)


def is_montgomery_form(values: object) -> bool:
    """Inspect the explicit representation tag; never guess from integer values."""
    return isinstance(values, MontgomeryTensor)


def montgomery_mul(left: MontgomeryTensor, right: MontgomeryTensor) -> MontgomeryTensor:
    """Elementwise product in Montgomery form, with normal Torch broadcasting."""
    if not is_montgomery_form(left) or not is_montgomery_form(right):
        raise TypeError("both operands must be MontgomeryTensor values")
    if left.q != right.q or left.device != right.device:
        raise ValueError("Montgomery operands must have the same modulus and device")
    product = montgomery_context(left.q).multiply(left._values, right._values)
    return MontgomeryTensor(product, left.q, _encoded=True)
