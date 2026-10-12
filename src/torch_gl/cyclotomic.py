"""Compact transforms for Phi_{3**exponent}, batching both primitive-root classes."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import torch

from ._number_theory import odd_prime_power, root_of_unity
from ._transform_tables import device_powers, resolve_device
from .modular import arithmetic_context, residues
from .ntt import NTTPlan


@dataclass(frozen=True)
class CyclotomicNTTPlan:
    """Evaluation at primitive p-th roots, p=3**exponent, with phi(p) slots.

    Requires p | q-1. Slot order is 3*index+1 followed by 3*index+2,
    for index in range(p/3). A two-branch twisted butterfly and a single
    batched radix-3 transform implement both root classes without padding.
    """

    q: int
    p: int
    root: int = field(init=False)

    def __post_init__(self) -> None:
        prime, _ = odd_prime_power(self.p)
        if prime != 3:
            raise ValueError("native cyclotomic NTT currently supports powers of three only")
        object.__setattr__(self, "root", root_of_unity(self.q, self.p))

    @property
    def length(self) -> int:
        return 2 * (self.p // 3)

    def transform(
        self, values: torch.Tensor, *, dim: int = -1, inverse: bool = False
    ) -> torch.Tensor:
        if values.ndim == 0 or not -values.ndim <= dim < values.ndim:
            raise ValueError("dim must name an existing tensor axis")
        if values.shape[dim] != self.length:
            raise ValueError(f"expected phi(p) = {self.length} coefficients/slots")
        arithmetic = arithmetic_context(self.q)
        normalized = arithmetic.encode(residues(values, self.q))
        return arithmetic.decode(self._transform_residues(normalized, dim=dim, inverse=inverse))

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
        block_length = self.p // 3
        plan = NTTPlan(self.q, block_length, pow(self.root, 3, self.q))
        branch_kernel = plan._kernel(device, inverse=inverse, permuted=permuted)
        omega = pow(self.root, block_length, self.q)
        encoded_omega = arithmetic.constant(omega)
        inverse_difference = arithmetic.constant(pow((omega - omega * omega) % self.q, -1, self.q))
        roots = tuple(
            pow(self.root, -root_class if inverse else root_class, self.q) for root_class in (1, 2)
        )
        twists = device_powers(self.q, roots, block_length, str(device))
        cube_roots = device_powers(self.q, (omega,), 3, str(device))[0, 1:].unsqueeze(-1)

        def kernel(values: torch.Tensor) -> torch.Tensor:
            coefficients = values.movedim(dim, -1)
            if inverse:
                branches = coefficients.reshape(*coefficients.shape[:-1], 2, block_length)
                unfolded = arithmetic.multiply(branch_kernel(branches), twists)
                first_branch, second_branch = unfolded.unbind(-2)
                upper = arithmetic.multiply(
                    arithmetic.subtract(first_branch, second_branch), inverse_difference
                )
                lower = arithmetic.subtract(first_branch, arithmetic.multiply(upper, encoded_omega))
                output = torch.stack((lower, upper), dim=-2)
            else:
                lower, upper = coefficients[..., :block_length], coefficients[..., block_length:]
                branches = arithmetic.add(
                    lower.unsqueeze(-2), arithmetic.multiply(upper.unsqueeze(-2), cube_roots)
                )
                output = branch_kernel(arithmetic.multiply(branches, twists))
            return output.reshape(*coefficients.shape[:-1], self.length).movedim(-1, dim)

        return kernel

    def prepare(self, device: torch.device | str = "cpu") -> "CyclotomicNTTPlan":
        device = resolve_device(device)
        block_length = self.p // 3
        NTTPlan(self.q, block_length, pow(self.root, 3, self.q)).prepare(device)
        for sign in (1, -1):
            roots = tuple(pow(self.root, sign * root_class, self.q) for root_class in (1, 2))
            device_powers(self.q, roots, block_length, str(device))
        device_powers(self.q, (pow(self.root, block_length, self.q),), 3, str(device))
        return self

    def forward(self, values: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        return self.transform(values, dim=dim)

    def inverse(self, values: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        return self.transform(values, dim=dim, inverse=True)
