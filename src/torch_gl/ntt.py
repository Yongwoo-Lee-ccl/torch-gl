"""Exact mixed-radix (2 and 3) number-theoretic transforms."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from ._number_theory import root_of_unity, smooth_factors
from .modular import mul_mod, residues


@lru_cache(maxsize=256)
def _powers(q: int, root: int, length: int) -> tuple[int, ...]:
    values = [1]
    for _ in range(1, length):
        values.append(values[-1] * root % q)
    return tuple(values)


def _dft(values: torch.Tensor, q: int, root: int) -> torch.Tensor:
    """Decimation in time with vectorized radix-2 and radix-3 butterflies."""
    length = values.shape[-1]
    if length == 1:
        return values.clone()
    radix = 2 if length % 2 == 0 else 3
    branch_length = length // radix
    branches = values.reshape(*values.shape[:-1], branch_length, radix).transpose(-2, -1)
    branches = _dft(branches, q, pow(root, radix, q))
    twiddle = torch.tensor(_powers(q, root, branch_length), dtype=torch.int64, device=values.device)
    first_branch = branches[..., 0, :]
    second_branch = mul_mod(branches[..., 1, :], twiddle, q)
    if radix == 2:
        output = (
            (first_branch + second_branch).remainder(q),
            (first_branch - second_branch).remainder(q),
        )
    else:
        third_branch = mul_mod(branches[..., 2, :], mul_mod(twiddle, twiddle, q), q)
        omega = pow(root, branch_length, q)
        omega2 = omega * omega % q
        output = (
            ((first_branch + second_branch).remainder(q) + third_branch).remainder(q),
            (
                (first_branch + mul_mod(second_branch, omega, q)).remainder(q)
                + mul_mod(third_branch, omega2, q)
            ).remainder(q),
            (
                (first_branch + mul_mod(second_branch, omega2, q)).remainder(q)
                + mul_mod(third_branch, omega, q)
            ).remainder(q),
        )
    return torch.stack(output, dim=-2).reshape(*values.shape[:-1], length)


@dataclass(frozen=True)
class NTTPlan:
    """Mixed-radix transform of length 2**a * 3**b in natural order.

    A[k] = sum_j a[j] * root**(j*k) mod q. The optional root must
    have exact order length. Other tensor dimensions are batch axes.
    """

    q: int
    length: int
    root: int | None = None

    def __post_init__(self) -> None:
        default_root = root_of_unity(self.q, self.length)
        root = default_root if self.root is None else self.root
        if type(root) is not int:
            raise TypeError("root must be an integer")
        root %= self.q
        if pow(root, self.length, self.q) != 1 or any(
            pow(root, self.length // factor, self.q) == 1
            for factor in set(smooth_factors(self.length))
        ):
            raise ValueError("root must have exact order length modulo q")
        object.__setattr__(self, "root", root)

    def transform(
        self, values: torch.Tensor, *, dim: int = -1, inverse: bool = False
    ) -> torch.Tensor:
        if values.ndim == 0 or not -values.ndim <= dim < values.ndim:
            raise ValueError("dim must name an existing tensor axis")
        if values.shape[dim] != self.length:
            raise ValueError(f"expected axis length {self.length}, got {values.shape[dim]}")
        values = residues(values, self.q).movedim(dim, -1)
        root = pow(self.root, -1, self.q) if inverse else self.root
        values = _dft(values, self.q, root)
        if inverse:
            values = mul_mod(values, pow(self.length, -1, self.q), self.q)
        return values.movedim(-1, dim)

    def forward(self, values: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        return self.transform(values, dim=dim)

    def inverse(self, values: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        return self.transform(values, dim=dim, inverse=True)


def ntt(values: torch.Tensor, q: int, *, dim: int = -1) -> torch.Tensor:
    """Transform an integer tensor along dim without mutating the input."""
    if values.ndim == 0 or not -values.ndim <= dim < values.ndim:
        raise ValueError("dim must name an existing tensor axis")
    return NTTPlan(q, values.shape[dim]).forward(values, dim=dim)


def intt(values: torch.Tensor, q: int, *, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`ntt`, normalized by the transform length."""
    if values.ndim == 0 or not -values.ndim <= dim < values.ndim:
        raise ValueError("dim must name an existing tensor axis")
    return NTTPlan(q, values.shape[dim]).inverse(values, dim=dim)
