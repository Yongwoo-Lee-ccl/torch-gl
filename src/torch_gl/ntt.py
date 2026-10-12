"""Exact mixed-radix (2 and 3) number-theoretic transforms."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch

from ._number_theory import root_of_unity, smooth_factors
from ._transform_tables import resolve_device, stage_tables
from .modular import ResidueArithmetic, arithmetic_context, residues


@lru_cache(maxsize=256)
def _butterfly_constants(q: int, root: int, length: int) -> tuple[tuple[int, int, int], ...]:
    arithmetic = arithmetic_context(q)
    stages = []
    while length > 1:
        radix = 2 if length % 2 == 0 else 3
        branch_length = length // radix
        omega = pow(root, branch_length, q)
        stages.append((radix, arithmetic.constant(omega), arithmetic.constant(omega * omega % q)))
        root = pow(root, radix, q)
        length = branch_length
    return tuple(stages)


def _dft(
    values: torch.Tensor,
    arithmetic: ResidueArithmetic,
    tables: tuple[torch.Tensor, ...],
    constants: tuple[tuple[int, int, int], ...],
    depth: int = 0,
) -> torch.Tensor:
    """Decimation in time with vectorized radix-2 and radix-3 butterflies."""
    length = values.shape[-1]
    if length == 1:
        return values
    radix, omega, omega_squared = constants[depth]
    branch_length = length // radix
    branches = values.reshape(*values.shape[:-1], branch_length, radix).transpose(-2, -1)
    branches = _dft(branches, arithmetic, tables, constants, depth + 1)
    twiddles = tables[depth]
    first_branch = branches[..., 0, :]
    second_branch = arithmetic.multiply(branches[..., 1, :], twiddles[0])
    if radix == 2:
        output = (
            arithmetic.add(first_branch, second_branch),
            arithmetic.subtract(first_branch, second_branch),
        )
    else:
        third_branch = arithmetic.multiply(branches[..., 2, :], twiddles[1])
        output = (
            arithmetic.add(arithmetic.add(first_branch, second_branch), third_branch),
            arithmetic.add(
                arithmetic.add(first_branch, arithmetic.multiply(second_branch, omega)),
                arithmetic.multiply(third_branch, omega_squared),
            ),
            arithmetic.add(
                arithmetic.add(first_branch, arithmetic.multiply(second_branch, omega_squared)),
                arithmetic.multiply(third_branch, omega),
            ),
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
        values = residues(values, self.q)
        return self._transform_residues(values, dim=dim, inverse=inverse)

    def _transform_residues(
        self, values: torch.Tensor, *, dim: int = -1, inverse: bool = False
    ) -> torch.Tensor:
        return self._kernel(values.device, dim=dim, inverse=inverse)(values)

    @lru_cache(maxsize=128)  # noqa: B019 -- bounded immutable plans; cleared explicitly
    def _kernel(
        self, device: torch.device, *, dim: int = -1, inverse: bool = False
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        arithmetic = arithmetic_context(self.q)
        root = pow(self.root, -1, self.q) if inverse else self.root
        tables = stage_tables(self.q, root, self.length, str(device))
        constants = _butterfly_constants(self.q, root, self.length)
        normalization = pow(self.length, -1, self.q)

        def kernel(values: torch.Tensor) -> torch.Tensor:
            transformed = _dft(values.movedim(dim, -1), arithmetic, tables, constants)
            if inverse:
                transformed = arithmetic.multiply(transformed, normalization)
            return transformed.movedim(-1, dim)

        return kernel

    def prepare(self, device: torch.device | str = "cpu") -> "NTTPlan":
        """Warm bounded forward/inverse device caches before timed execution."""
        device = resolve_device(device)
        for root in (self.root, pow(self.root, -1, self.q)):
            stage_tables(self.q, root, self.length, str(device))
            _butterfly_constants(self.q, root, self.length)
        return self

    def compile(
        self,
        device: torch.device | str = "cpu",
        *,
        dim: int = -1,
        inverse: bool = False,
        backend: str = "inductor",
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Create an optional fused transform with tables bound outside compiler tracing.

        Compilation is lazy on the first call. Use warmup before benchmarking.
        This is an experimental path; no GPU performance is claimed here.
        """
        device = resolve_device(device)
        prepared_kernel = self._kernel(device, dim=dim, inverse=inverse)

        def kernel(values: torch.Tensor) -> torch.Tensor:
            return prepared_kernel(values.remainder(self.q))

        compiled_kernel = torch.compile(kernel, backend=backend, fullgraph=True)

        def transform(values: torch.Tensor) -> torch.Tensor:
            if values.ndim == 0 or not -values.ndim <= dim < values.ndim:
                raise ValueError("dim must name an existing tensor axis")
            if values.shape[dim] != self.length or values.device != device:
                raise ValueError("compiled transform input must match the plan length and device")
            if values.dtype != torch.int64:
                values = residues(values, self.q)
            return compiled_kernel(values)

        return transform

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
