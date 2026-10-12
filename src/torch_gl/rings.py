"""Configurable, separable monic quotient ideals over Z_q[i]."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from ._number_theory import (
    is_power_of_two,
    next_power_of_two,
    odd_prime_power,
    root_of_unity,
    validate_modulus,
)
from .cyclotomic import CyclotomicNTTPlan
from .ntt import NTTPlan
from .twisted import GaussianNTTPlan

if TYPE_CHECKING:
    from .multivariate_poly import MultivariatePoly

GaussianInteger = int | tuple[int, int]


def _pair(value: GaussianInteger) -> tuple[int, int]:
    if type(value) is int:
        return value, 0
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(type(component) is int for component in value)
    ):
        return value
    raise TypeError("a Gaussian coefficient must be an int or a pair of Python ints")


@dataclass(frozen=True)
class QuotientAxis:
    """One variable modulo a monic polynomial in that variable alone.

    ``modulus`` lists coefficients from constant to leading coefficient.
    Integers denote real coefficients; pairs denote (real, imaginary).
    Coupled relations involving multiple variables are outside this API.
    """

    name: str
    modulus: tuple[GaussianInteger, ...]
    cyclotomic_order: int | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier() or self.name == "i":
            raise ValueError("variable names must be identifiers other than 'i'")
        coefficients = tuple(_pair(coefficient) for coefficient in self.modulus)
        if len(coefficients) < 2 or coefficients[-1] != (1, 0):
            raise ValueError("modulus must have positive degree and leading coefficient 1")
        object.__setattr__(self, "modulus", coefficients)
        if self.cyclotomic_order is not None:
            prime, exponent = odd_prime_power(self.cyclotomic_order)
            stride = prime ** (exponent - 1)
            expected = tuple(
                (int(coefficient_index % stride == 0), 0)
                for coefficient_index in range((prime - 1) * stride + 1)
            )
            if coefficients != expected:
                raise ValueError("modulus does not match the specified cyclotomic order")

    @property
    def degree(self) -> int:
        return len(self.modulus) - 1

    @classmethod
    def power_of_i(cls, name: str, n: int) -> "QuotientAxis":
        """The relation name**n = i, with n a power of two."""
        if not is_power_of_two(n):
            raise ValueError("n must be a positive power of two")
        return cls(name, ((0, -1),) + (0,) * (n - 1) + (1,))

    @classmethod
    def cyclotomic(cls, name: str, p: int) -> "QuotientAxis":
        """Phi_p for an odd prime power p = ell**a.

        Phi_p(T) = sum(T**(j * ell**(a-1)), j=0..ell-1).
        """
        prime, exponent = odd_prime_power(p)
        stride = prime ** (exponent - 1)
        coefficients = [0] * ((prime - 1) * stride + 1)
        for coefficient_index in range(prime):
            coefficients[coefficient_index * stride] = 1
        return cls(name, tuple(coefficients), cyclotomic_order=p)


@dataclass(frozen=True)
class PolynomialRing:
    """Z_q[i][variables]/(i**2+1, one monic relation per variable).

    There are at most three variable axes, hence at most four coefficient
    tensor dimensions including the final length-two Gaussian axis.
    Native transforms are selected when the required roots exist in Z_q.
    Other axes use padded radix-2 convolution and monic ideal reduction.
    """

    q: int
    axes: tuple[QuotientAxis, ...]

    def __post_init__(self) -> None:
        validate_modulus(self.q)
        axes = tuple(self.axes)
        if len(axes) > 3 or any(not isinstance(axis, QuotientAxis) for axis in axes):
            raise ValueError("axes must contain at most three QuotientAxis objects")
        if len({axis.name for axis in axes}) != len(axes):
            raise ValueError("variable names must be distinct")
        object.__setattr__(self, "axes", axes)
        for axis, length in zip(self.axes, self.ntt_shape, strict=True):
            order = axis.cyclotomic_order if self._uses_cyclotomic_ntt(axis) else length
            if self._uses_twisted_ntt(axis):
                order = 4 * axis.degree
            root_of_unity(self.q, order)

    def _uses_cyclotomic_ntt(self, axis: QuotientAxis) -> bool:
        p = axis.cyclotomic_order
        return p is not None and odd_prime_power(p)[0] == 3 and (self.q - 1) % p == 0

    def _uses_twisted_ntt(self, axis: QuotientAxis) -> bool:
        return (
            is_power_of_two(axis.degree)
            and axis.modulus[0] == (0, -1)
            and all(coefficient == (0, 0) for coefficient in axis.modulus[1:-1])
            and (self.q - 1) % (4 * axis.degree) == 0
        )

    @classmethod
    def multivariate_poly(
        cls, q: int, n: int, p: int, *, variables: tuple[str, ...] = ("X", "Y", "W")
    ) -> "PolynomialRing":
        """Construct any subset of X**n-i, Y**n-i, Phi_p(W)."""
        if not is_power_of_two(n):
            raise ValueError("n must be a positive power of two")
        odd_prime_power(p)
        if any(name not in {"X", "Y", "W"} for name in variables):
            raise ValueError("Gentry–Lee variables must be selected from X, Y, W")
        return cls(
            q,
            tuple(
                QuotientAxis.cyclotomic(name, p)
                if name == "W"
                else QuotientAxis.power_of_i(name, n)
                for name in variables
            ),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Canonical coefficient shape, including the Gaussian channel."""
        return tuple(axis.degree for axis in self.axes) + (2,)

    @property
    def ntt_shape(self) -> tuple[int, ...]:
        return tuple(
            axis.degree
            if self._uses_cyclotomic_ntt(axis) or self._uses_twisted_ntt(axis)
            else next_power_of_two(2 * axis.degree - 1)
            for axis in self.axes
        )

    @property
    def transform_plans(self) -> tuple[NTTPlan | CyclotomicNTTPlan | GaussianNTTPlan, ...]:
        """Plans in variable order; apply to coefficient tensors along the matching axis."""
        plans = []
        for axis, length in zip(self.axes, self.ntt_shape, strict=True):
            if self._uses_cyclotomic_ntt(axis):
                plan = CyclotomicNTTPlan(self.q, axis.cyclotomic_order)
            elif self._uses_twisted_ntt(axis):
                plan = GaussianNTTPlan(self.q, axis.degree)
            else:
                plan = NTTPlan(self.q, length)
            plans.append(plan)
        return tuple(plans)

    @property
    def convolution_shape(self) -> tuple[int, ...]:
        """Support after inverse transforms, before general ideal reduction."""
        return tuple(
            axis.degree
            if self._uses_cyclotomic_ntt(axis) or self._uses_twisted_ntt(axis)
            else 2 * axis.degree - 1
            for axis in self.axes
        )

    def polynomial(self, coefficients: torch.Tensor) -> "MultivariatePoly":
        """Create from (..., 2) integer coefficients, reducing excess degrees."""
        from .multivariate_poly import MultivariatePoly

        return MultivariatePoly(self, coefficients)

    def from_real(self, coefficients: torch.Tensor) -> "MultivariatePoly":
        """Create from real coefficients without a Gaussian channel axis."""
        if coefficients.ndim != len(self.axes):
            raise ValueError("expected one coefficient dimension per variable")
        return self.polynomial(torch.stack((coefficients, torch.zeros_like(coefficients)), -1))

    def constant(
        self, value: GaussianInteger, *, device: torch.device | str = "cpu"
    ) -> "MultivariatePoly":
        real, imag = _pair(value)
        coefficients = torch.zeros(self.shape, dtype=torch.int64, device=device)
        coefficients[(0,) * len(self.axes)] = torch.tensor(
            [real % self.q, imag % self.q], dtype=torch.int64, device=device
        )
        return self.polynomial(coefficients)

    def zero(self, *, device: torch.device | str = "cpu") -> "MultivariatePoly":
        return self.constant(0, device=device)

    def one(self, *, device: torch.device | str = "cpu") -> "MultivariatePoly":
        return self.constant(1, device=device)

    def generator(self, name: str, *, device: torch.device | str = "cpu") -> "MultivariatePoly":
        if name == "i":
            return self.constant((0, 1), device=device)
        names = [axis.name for axis in self.axes]
        if name not in names:
            raise ValueError(f"unknown generator {name!r}")
        axis = names.index(name)
        shape = list(self.shape)
        shape[axis] = max(2, shape[axis])
        coefficients = torch.zeros(shape, dtype=torch.int64, device=device)
        index = [0] * len(shape)
        index[axis] = 1
        coefficients[tuple(index)] = 1
        return self.polynomial(coefficients)
