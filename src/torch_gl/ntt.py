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


def _butterfly(branches, arithmetic, omega):
    first_branch, second_branch = branches[:2]
    if len(branches) == 2:
        output = (
            arithmetic.add(first_branch, second_branch),
            arithmetic.subtract(first_branch, second_branch),
        )
    else:
        third_branch = branches[2]
        # omega**2 = -1-omega: one multiplication replaces four.
        difference = arithmetic.multiply(arithmetic.subtract(second_branch, third_branch), omega)
        output = (
            arithmetic.add(arithmetic.add(first_branch, second_branch), third_branch),
            arithmetic.add(arithmetic.subtract(first_branch, third_branch), difference),
            arithmetic.subtract(arithmetic.subtract(first_branch, second_branch), difference),
        )
    return output


def _paired_dft(
    values: torch.Tensor,
    arithmetic: ResidueArithmetic,
    tables: tuple[torch.Tensor, ...],
    constants: tuple[tuple[int, int, int], ...],
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """GS DIF: natural -> digit order; CT DIT: digit order -> natural.

    Every stage batches contiguous groups. No digit-reversal gather occurs
    here; pointwise operations retain the forward transform's ordering.
    """
    length = values.shape[-1]
    stage_lengths = []
    stage_length = length
    for radix, _, _ in constants:
        stage_lengths.append(stage_length)
        stage_length //= radix
    depths = range(len(constants) - 1, -1, -1) if inverse else range(len(constants))
    for depth in depths:
        radix, omega, _ = constants[depth]
        branch_length = stage_lengths[depth] // radix
        branches = values.reshape(
            *values.shape[:-1], length // stage_lengths[depth], radix, branch_length
        ).unbind(-2)
        if not inverse:
            branches = _butterfly(branches, arithmetic, omega)
        # The innermost stage has only unity twiddles; leave its branches
        # untouched instead of allocating Montgomery temporaries for *1.
        if branch_length > 1:
            branches = (branches[0],) + tuple(
                arithmetic.multiply(branches[branch], tables[depth][branch - 1])
                for branch in range(1, radix)
            )
        if inverse:
            branches = _butterfly(branches, arithmetic, omega)
        values = torch.stack(branches, dim=-2).reshape(*values.shape[:-1], length)
    return values


@lru_cache(maxsize=128)
def _digit_order(length: int) -> tuple[int, ...]:
    """Natural frequency index at each GS output position (mixed radix)."""
    if length == 1:
        return (0,)
    radix = 2 if length % 2 == 0 else 3
    return tuple(
        branch + radix * frequency
        for branch in range(radix)
        for frequency in _digit_order(length // radix)
    )


@lru_cache(maxsize=128)
def _device_digit_order(length: int, device: str):
    order = torch.tensor(_digit_order(length), dtype=torch.int64, device=device)
    return order, torch.argsort(order)


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
        arithmetic = arithmetic_context(self.q)
        values = arithmetic.encode(residues(values, self.q))
        return arithmetic.decode(self._transform_residues(values, dim=dim, inverse=inverse))

    def _transform_residues(
        self, values: torch.Tensor, *, dim: int = -1, inverse: bool = False
    ) -> torch.Tensor:
        return self._kernel(values.device, dim=dim, inverse=inverse)(values)

    @lru_cache(maxsize=128)  # noqa: B019 -- bounded immutable plans; cleared explicitly
    def _kernel(
        self,
        device: torch.device,
        *,
        dim: int = -1,
        inverse: bool = False,
        permuted: bool = False,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        arithmetic = arithmetic_context(self.q)
        root = pow(self.root, -1, self.q) if inverse else self.root
        tables = stage_tables(self.q, root, self.length, str(device))
        constants = _butterfly_constants(self.q, root, self.length)
        normalization = arithmetic.constant(pow(self.length, -1, self.q))
        order = None if permuted else _device_digit_order(self.length, str(device))

        def kernel(values: torch.Tensor) -> torch.Tensor:
            transformed = values.movedim(dim, -1)
            if inverse and order is not None:
                transformed = transformed.index_select(-1, order[0])
            transformed = _paired_dft(transformed, arithmetic, tables, constants, inverse=inverse)
            if inverse:
                transformed = arithmetic.multiply(transformed, normalization)
            elif order is not None:
                transformed = transformed.index_select(-1, order[1])
            return transformed.movedim(-1, dim)

        return kernel

    def prepare(self, device: torch.device | str = "cpu") -> "NTTPlan":
        """Warm bounded forward/inverse device caches before timed execution."""
        device = resolve_device(device)
        _device_digit_order(self.length, str(device))
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
        arithmetic = arithmetic_context(self.q)
        prepared_kernel = self._kernel(device, dim=dim, inverse=inverse)

        def kernel(values: torch.Tensor) -> torch.Tensor:
            encoded = arithmetic.encode(values.remainder(self.q))
            return arithmetic.decode(prepared_kernel(encoded))

        compiled_kernel = torch.compile(kernel, backend=backend, fullgraph=True)

        def transform(values: torch.Tensor) -> torch.Tensor:
            if values.ndim == 0 or not -values.ndim <= dim < values.ndim:
                raise ValueError("dim must name an existing tensor axis")
            if values.shape[dim] != self.length or values.device != device:
                raise ValueError("compiled transform input must match the plan length and device")
            if values.dtype != torch.int32:
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
