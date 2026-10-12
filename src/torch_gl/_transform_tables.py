"""Bounded device table caches. No host tensors are rebuilt on a warm transform."""

from functools import lru_cache

import torch

from .modular import arithmetic_context


def resolve_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cpu":
        return torch.device("cpu")
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        return torch.device("cuda", index)
    raise ValueError("exact integer arithmetic supports CPU and CUDA devices only")


@lru_cache(maxsize=256)
def powers(q: int, root: int, length: int) -> tuple[int, ...]:
    values = [1]
    for _ in range(1, length):
        values.append(values[-1] * root % q)
    return tuple(values)


@lru_cache(maxsize=128)
def device_powers(q: int, roots: tuple[int, ...], length: int, device: str) -> torch.Tensor:
    arithmetic = arithmetic_context(q)
    rows = [
        tuple(arithmetic.constant(value) for value in powers(q, root, length)) for root in roots
    ]
    return torch.tensor(rows, dtype=torch.int32, device=device)


@lru_cache(maxsize=128)
def stage_tables(q: int, root: int, length: int, device: str) -> tuple[torch.Tensor, ...]:
    """One row per nontrivial branch, stored in the transform's representation."""
    tables = []
    while length > 1:
        radix = 2 if length % 2 == 0 else 3
        branch_length = length // radix
        roots = tuple(pow(root, branch, q) for branch in range(1, radix))
        tables.append(device_powers(q, roots, branch_length, device))
        root = pow(root, radix, q)
        length = branch_length
    return tuple(tables)


def clear_transform_cache() -> None:
    """Release cached CPU/CUDA tensor references, e.g. before a memory benchmark."""
    stage_tables.cache_clear()
    device_powers.cache_clear()
    from ._multiplication import native_multiplier
    from .cyclotomic import CyclotomicNTTPlan
    from .matrix import _matrix_plan
    from .ntt import NTTPlan

    native_multiplier.cache_clear()
    _matrix_plan.cache_clear()
    NTTPlan._kernel.cache_clear()
    CyclotomicNTTPlan._kernel.cache_clear()
