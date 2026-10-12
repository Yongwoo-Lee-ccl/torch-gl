"""Compact transforms for Phi_{3**a}, using radix-2/3 decomposition."""

from dataclasses import dataclass, field

import torch

from ._number_theory import odd_prime_power, root_of_unity
from .modular import mul_mod, residues
from .ntt import NTTPlan, _powers


@dataclass(frozen=True)
class CyclotomicNTTPlan:
    """Evaluation at primitive p-th roots, p = 3**a, with phi(p) slots.

    Requires p | q-1. With r=p/3, slot order is k=3t+1 (t=0..r-1),
    followed by k=3t+2. A two-branch twisted butterfly combines coefficient
    blocks of length r; each branch uses a radix-3 transform. No zero
    padding or full p-point transform is needed.
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

    def _input(self, values: torch.Tensor, dim: int) -> torch.Tensor:
        if values.ndim == 0 or not -values.ndim <= dim < values.ndim:
            raise ValueError("dim must name an existing tensor axis")
        if values.shape[dim] != self.length:
            raise ValueError(f"expected phi(p) = {self.length} coefficients/slots")
        return residues(values, self.q).movedim(dim, -1)

    def forward(self, values: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        coefficients = self._input(values, dim)
        block_length = self.p // 3
        lower_coefficients, upper_coefficients = (
            coefficients[..., :block_length],
            coefficients[..., block_length:],
        )
        plan = NTTPlan(self.q, block_length, pow(self.root, 3, self.q))
        output = []
        for root_class in (1, 2):
            twiddle = torch.tensor(
                _powers(self.q, pow(self.root, root_class, self.q), block_length),
                dtype=torch.int64,
                device=coefficients.device,
            )
            folded = (
                lower_coefficients
                + mul_mod(
                    upper_coefficients, pow(self.root, root_class * block_length, self.q), self.q
                )
            ).remainder(self.q)
            output.append(plan.forward(mul_mod(folded, twiddle, self.q)))
        return torch.cat(output, dim=-1).movedim(-1, dim)

    def inverse(self, values: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        coefficients = self._input(values, dim)
        block_length = self.p // 3
        plan = NTTPlan(self.q, block_length, pow(self.root, 3, self.q))
        unfolded = []
        for root_class in (1, 2):
            twiddle = torch.tensor(
                _powers(self.q, pow(self.root, -root_class, self.q), block_length),
                dtype=torch.int64,
                device=coefficients.device,
            )
            branch = plan.inverse(
                coefficients[..., (root_class - 1) * block_length : root_class * block_length]
            )
            unfolded.append(mul_mod(branch, twiddle, self.q))
        omega = pow(self.root, block_length, self.q)
        difference = (unfolded[0] - unfolded[1]).remainder(self.q)
        upper = mul_mod(difference, pow((omega - omega * omega) % self.q, -1, self.q), self.q)
        lower = (unfolded[0] - mul_mod(upper, omega, self.q)).remainder(self.q)
        return torch.cat((lower, upper), dim=-1).movedim(-1, dim)
