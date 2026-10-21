"""Int32 residue storage, with local int64 intermediates in Montgomery products.

Internal operands lie in [0,q), with q<2**30. Addition and subtraction
fit signed int32. Transform arithmetic stays in Montgomery form; only
public boundaries convert. Larger moduli belong in a higher-level RNS layer.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import torch

from .montgomery import MontgomeryContext, montgomery_context

_INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}


def residues(value: torch.Tensor, q: int) -> torch.Tensor:
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError("coefficients must use an integer dtype (not bool, float, or complex)")
    if value.device.type not in {"cpu", "cuda"}:
        raise ValueError("exact integer arithmetic supports CPU and CUDA devices only")
    # Reduce before narrowing: arbitrary signed int64 input must not truncate.
    normalized_input = value if value.dtype in {torch.int32, torch.int64} else value.to(torch.int32)
    return normalized_input.remainder(q).to(dtype=torch.int32)


def mul_mod(left: torch.Tensor, right: torch.Tensor | int, q: int) -> torch.Tensor:
    context = montgomery_context(q)
    encoded_right = (
        context.encode(right)
        if isinstance(right, torch.Tensor)
        else (context.encode_integer(right))
    )
    return context.multiply(left, encoded_right)


def add_mod(left: torch.Tensor, right: torch.Tensor | int, q: int) -> torch.Tensor:
    total = left + right
    return total.add_(total >= q, alpha=-q)


def sub_mod(left: torch.Tensor, right: torch.Tensor | int, q: int) -> torch.Tensor:
    difference = left - right
    return difference.add_(difference < 0, alpha=q)


@dataclass(frozen=True)
class ResidueArithmetic:
    """Montgomery-form int32 arithmetic shared by an entire transform or product."""

    q: int
    context: MontgomeryContext = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", montgomery_context(self.q))

    def constant(self, value: int) -> int:
        return self.context.encode_integer(value)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.context.encode(values)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return self.context.decode(values)

    def multiply(self, left: torch.Tensor, right: torch.Tensor | int) -> torch.Tensor:
        return self.context.multiply(left, right)

    def add(self, left: torch.Tensor, right: torch.Tensor | int) -> torch.Tensor:
        return add_mod(left, right, self.q)

    def subtract(self, left: torch.Tensor, right: torch.Tensor | int) -> torch.Tensor:
        return sub_mod(left, right, self.q)


@lru_cache(maxsize=128)
def arithmetic_context(q: int) -> ResidueArithmetic:
    return ResidueArithmetic(q)


def gaussian_mul(left: torch.Tensor, right: torch.Tensor, q: int) -> torch.Tensor:
    """Multiply (..., 2) coefficient pairs using i**2 = -1."""
    arithmetic = arithmetic_context(q)
    encoded = gaussian_mul_residues(arithmetic.encode(left), arithmetic.encode(right), arithmetic)
    return arithmetic.decode(encoded)


def gaussian_mul_residues(
    left: torch.Tensor, right: torch.Tensor, arithmetic: ResidueArithmetic
) -> torch.Tensor:
    """Gaussian product within the chosen internal representation."""
    left_real, left_imag = left.unbind(-1)
    right_real, right_imag = right.unbind(-1)
    real = arithmetic.subtract(
        arithmetic.multiply(left_real, right_real), arithmetic.multiply(left_imag, right_imag)
    )
    imag = arithmetic.add(
        arithmetic.multiply(left_real, right_imag), arithmetic.multiply(left_imag, right_real)
    )
    return torch.stack((real, imag), dim=-1)
