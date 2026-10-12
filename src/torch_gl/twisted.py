"""Compact Gaussian-coefficient NTT for T**n-i, with batched split branches."""

from dataclasses import dataclass, field

import torch

from ._number_theory import is_power_of_two, root_of_unity
from ._transform_tables import device_powers, resolve_device
from .modular import arithmetic_context, residues
from .ntt import NTTPlan


def _reverse_frequencies(values: torch.Tensor) -> torch.Tensor:
    """Map k to -k modulo length, turning a forward root into its inverse."""
    if values.shape[-1] == 1:
        return values
    return torch.cat((values[..., :1], values[..., 1:].flip(-1)), dim=-1)


@dataclass(frozen=True)
class GaussianNTTPlan:
    """Length-n transform in Z_q[i][T]/(T**n-i), requiring 4n | q-1.

    The final tensor axis stores (real, imaginary). The two split branches
    share one batched radix-2 transform. The inverse-root branch is obtained
    through a frequency permutation. Both branches use exact normal-form
    arithmetic within a single 30-bit prime limb.
    """

    q: int
    length: int
    root: int = field(init=False)

    def __post_init__(self) -> None:
        if not is_power_of_two(self.length):
            raise ValueError("n must be a positive power of two")
        object.__setattr__(self, "root", root_of_unity(self.q, 4 * self.length))

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
        normalized = residues(values, self.q)
        return self._transform_residues(normalized, dim=dim, inverse=inverse)

    def _transform_residues(
        self, values: torch.Tensor, *, dim: int = -2, inverse: bool = False
    ) -> torch.Tensor:
        arithmetic = arithmetic_context(self.q)
        coefficients = values.movedim(dim, -2)
        real, imag = coefficients.unbind(-1)
        imaginary_root = pow(self.root, self.length, self.q)
        component = arithmetic.multiply(imag, arithmetic.constant(imaginary_root))
        positive = arithmetic.add(real, component)
        negative = arithmetic.subtract(real, component)
        plan = NTTPlan(self.q, self.length, pow(self.root, 4, self.q))
        roots = (
            (pow(self.root, -1, self.q), self.root)
            if inverse
            else (self.root, pow(self.root, -1, self.q))
        )
        twists = device_powers(self.q, roots, self.length, str(values.device))
        if inverse:
            branches = torch.stack((positive, _reverse_frequencies(negative)), dim=-2)
            transformed = plan._transform_residues(branches, inverse=True)
            transformed = arithmetic.multiply(transformed, twists)
            positive, negative = transformed.unbind(-2)
        else:
            branches = torch.stack((positive, negative), dim=-2)
            branches = arithmetic.multiply(branches, twists)
            transformed = plan._transform_residues(branches)
            positive = transformed[..., 0, :]
            negative = _reverse_frequencies(transformed[..., 1, :])
        merged_real = arithmetic.multiply(
            arithmetic.add(positive, negative), arithmetic.constant(pow(2, -1, self.q))
        )
        merged_imag = arithmetic.multiply(
            arithmetic.subtract(positive, negative),
            arithmetic.constant(pow(2 * imaginary_root, -1, self.q)),
        )
        return torch.stack((merged_real, merged_imag), dim=-1).movedim(-2, dim)

    def prepare(self, device: torch.device | str = "cpu") -> "GaussianNTTPlan":
        device = resolve_device(device)
        NTTPlan(self.q, self.length, pow(self.root, 4, self.q)).prepare(device)
        for roots in (
            (self.root, pow(self.root, -1, self.q)),
            (pow(self.root, -1, self.q), self.root),
        ):
            device_powers(self.q, roots, self.length, str(device))
        return self

    def forward(self, values: torch.Tensor, *, dim: int = -2) -> torch.Tensor:
        return self.transform(values, dim=dim)

    def inverse(self, values: torch.Tensor, *, dim: int = -2) -> torch.Tensor:
        return self.transform(values, dim=dim, inverse=True)
