"""Exact single-limb arithmetic on canonical torch.int64 residues.

Internal operands lie in [0,q), with q<2**30. Ordinary products fit
signed int64 directly. Larger moduli belong in a higher-level RNS layer.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch

_INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}


def residues(value: torch.Tensor, q: int) -> torch.Tensor:
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError("coefficients must use an integer dtype (not bool, float, or complex)")
    if value.device.type not in {"cpu", "cuda"}:
        raise ValueError("exact int64 arithmetic supports CPU and CUDA devices only")
    return value.to(dtype=torch.int64).remainder(q)


def mul_mod(left: torch.Tensor, right: torch.Tensor | int, q: int) -> torch.Tensor:
    return (left * right).remainder(q)


def add_mod(left: torch.Tensor, right: torch.Tensor | int, q: int) -> torch.Tensor:
    total = left + right
    return torch.where(total >= q, total - q, total)


def sub_mod(left: torch.Tensor, right: torch.Tensor | int, q: int) -> torch.Tensor:
    difference = left - right
    return torch.where(difference < 0, difference + q, difference)


@dataclass(frozen=True)
class ResidueArithmetic:
    """Normal-form arithmetic shared by an entire transform or product."""

    q: int

    def constant(self, value: int) -> int:
        return value % self.q

    def multiply(self, left: torch.Tensor, right: torch.Tensor | int) -> torch.Tensor:
        return (left * right).remainder(self.q)

    def add(self, left: torch.Tensor, right: torch.Tensor | int) -> torch.Tensor:
        if left.device.type == "cpu":
            return (left + right).remainder(self.q)
        return add_mod(left, right, self.q)

    def subtract(self, left: torch.Tensor, right: torch.Tensor | int) -> torch.Tensor:
        if left.device.type == "cpu":
            return (left - right).remainder(self.q)
        return sub_mod(left, right, self.q)


@lru_cache(maxsize=128)
def arithmetic_context(q: int) -> ResidueArithmetic:
    return ResidueArithmetic(q)


def gaussian_mul(left: torch.Tensor, right: torch.Tensor, q: int) -> torch.Tensor:
    """Multiply (..., 2) coefficient pairs using i**2 = -1."""
    arithmetic = arithmetic_context(q)
    return gaussian_mul_residues(left, right, arithmetic)


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
