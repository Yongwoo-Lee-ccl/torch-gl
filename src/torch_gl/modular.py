"""Overflow-safe arithmetic on canonical torch.int64 residues.

Internal operands must lie in [0, q). At q <= 2**62-1, sums of two
residues fit int64. Large products use binary add-and-double instead
of an overflowing native product. These routines are not constant-time.
"""

import torch

_DIRECT_PRODUCT_LIMIT = 3_037_000_500
_INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}


def residues(value: torch.Tensor, q: int) -> torch.Tensor:
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError("coefficients must use an integer dtype (not bool, float, or complex)")
    if value.device.type not in {"cpu", "cuda"}:
        raise ValueError("exact int64 arithmetic supports CPU and CUDA devices only")
    return value.to(dtype=torch.int64).remainder(q)


def mul_mod(left: torch.Tensor, right: torch.Tensor | int, q: int) -> torch.Tensor:
    if q <= _DIRECT_PRODUCT_LIMIT:
        return (left * right).remainder(q)
    if isinstance(right, int):
        right %= q
        result = torch.zeros_like(left)
        while right:
            if right & 1:
                result = (result + left).remainder(q)
            right >>= 1
            if right:
                left = (left + left).remainder(q)
        return result
    left, right = torch.broadcast_tensors(left, right)
    result = torch.zeros_like(left)
    for _ in range(q.bit_length()):
        result = (result + torch.where((right & 1).bool(), left, 0)).remainder(q)
        right = right >> 1
        left = (left + left).remainder(q)
    return result


def gaussian_mul(left: torch.Tensor, right: torch.Tensor, q: int) -> torch.Tensor:
    """Multiply (..., 2) coefficient pairs using i**2 = -1."""
    left_real, left_imag = left.unbind(-1)
    right_real, right_imag = right.unbind(-1)
    real = (mul_mod(left_real, right_real, q) - mul_mod(left_imag, right_imag, q)).remainder(q)
    imag = (mul_mod(left_real, right_imag, q) + mul_mod(left_imag, right_real, q)).remainder(q)
    return torch.stack((real, imag), dim=-1)
