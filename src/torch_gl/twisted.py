"""Compact Gaussian-coefficient NTT for T**n-i."""

from dataclasses import dataclass, field

import torch

from ._number_theory import is_power_of_two, root_of_unity
from .modular import mul_mod, residues
from .ntt import NTTPlan, _powers


@dataclass(frozen=True)
class GaussianNTTPlan:
    """Length-n transform in Z_q[i][T]/(T**n-i).

    Requires 4n | q-1. The final tensor axis stores (real, imaginary).
    Split i into I and -I, twist each branch by rho**j and rho**(-j),
    and perform radix-2 NTTs. Merge back into formal Gaussian pairs.
    Pointwise Gaussian multiplication in this representation corresponds
    to quotient-ring multiplication; the inverse returns canonical coefficients.
    """

    q: int
    length: int
    root: int = field(init=False)

    def __post_init__(self) -> None:
        if not is_power_of_two(self.length):
            raise ValueError("n must be a positive power of two")
        object.__setattr__(self, "root", root_of_unity(self.q, 4 * self.length))

    def _merge(self, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        imaginary = pow(self.root, self.length, self.q)
        real = mul_mod((positive + negative).remainder(self.q), pow(2, -1, self.q), self.q)
        imag = mul_mod(
            (positive - negative).remainder(self.q), pow(2 * imaginary, -1, self.q), self.q
        )
        return torch.stack((real, imag), dim=-1)

    def transform(
        self, values: torch.Tensor, *, dim: int = -2, inverse: bool = False
    ) -> torch.Tensor:
        if values.ndim < 2 or not -values.ndim <= dim < values.ndim:
            raise ValueError("dim must name a variable axis")
        dim %= values.ndim
        if dim == values.ndim - 1 or values.shape[-1] != 2:
            raise ValueError("the final axis must be Gaussian channels, not the transform axis")
        if values.shape[dim] != self.length:
            raise ValueError(f"expected variable axis length {self.length}")
        coefficients = residues(values, self.q).movedim(dim, -2)
        real, imag = coefficients.unbind(-1)
        imaginary = pow(self.root, self.length, self.q)
        component = mul_mod(imag, imaginary, self.q)
        branches = ((real + component).remainder(self.q), (real - component).remainder(self.q))
        output = []
        for sign, branch in zip((1, -1), branches, strict=True):
            plan = NTTPlan(self.q, self.length, pow(self.root, 4 * sign, self.q))
            twist = torch.tensor(
                _powers(self.q, pow(self.root, -sign if inverse else sign, self.q), self.length),
                dtype=torch.int64,
                device=coefficients.device,
            )
            if inverse:
                output.append(mul_mod(plan.inverse(branch), twist, self.q))
            else:
                output.append(plan.forward(mul_mod(branch, twist, self.q)))
        return self._merge(*output).movedim(-2, dim)

    def forward(self, values: torch.Tensor, *, dim: int = -2) -> torch.Tensor:
        return self.transform(values, dim=dim)

    def inverse(self, values: torch.Tensor, *, dim: int = -2) -> torch.Tensor:
        return self.transform(values, dim=dim, inverse=True)
