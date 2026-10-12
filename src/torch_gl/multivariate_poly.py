"""Tensor-backed quotient polynomial values and exact NTT multiplication."""

import torch

from .modular import arithmetic_context, gaussian_mul, gaussian_mul_residues, residues
from .rings import GaussianInteger, PolynomialRing


def _reduce(coefficients: torch.Tensor, ring: PolynomialRing) -> torch.Tensor:
    """Monic long division along each independent variable axis."""
    # The constructor passes an owned, normalized tensor. Canonical inputs
    # require no long division, axis permutations, or additional copy.
    if tuple(coefficients.shape) == ring.shape:
        return coefficients
    result = coefficients
    for dim, axis in enumerate(ring.axes):
        if result.shape[dim] == axis.degree:
            continue
        axis_coefficients = result.movedim(dim, 0).contiguous()
        degree = axis.degree
        terms = [
            (
                modulus_index,
                torch.tensor(
                    [real % ring.q, imag % ring.q],
                    dtype=torch.int64,
                    device=axis_coefficients.device,
                ),
            )
            for modulus_index, (real, imag) in enumerate(axis.modulus[:-1])
            if real % ring.q or imag % ring.q
        ]
        for coefficient_index in range(axis_coefficients.shape[0] - 1, degree - 1, -1):
            leading = axis_coefficients[coefficient_index].clone()
            for modulus_index, coefficient in terms:
                target = coefficient_index - degree + modulus_index
                axis_coefficients[target] = (
                    axis_coefficients[target] - gaussian_mul(leading, coefficient, ring.q)
                ).remainder(ring.q)
        if axis_coefficients.shape[0] < degree:
            padding = torch.zeros(
                (degree - axis_coefficients.shape[0], *axis_coefficients.shape[1:]),
                dtype=torch.int64,
                device=axis_coefficients.device,
            )
            axis_coefficients = torch.cat((axis_coefficients, padding), dim=0)
        result = axis_coefficients[:degree].movedim(0, dim)
    return result.contiguous()


class MultivariatePoly:
    """A canonical polynomial with coefficients in [0, q).

    Input shape is (degree_X, degree_Y, ..., 2) in ring axis order.
    The final channels are real and imaginary. Arbitrary positive lengths
    on variable axes are reduced/padded into the canonical basis.
    Construction and the coefficients property copy tensors so callers
    cannot mutate a polynomial through a shared input or returned tensor.
    """

    __slots__ = ("_ring", "_coefficients")
    __hash__ = None

    def __init__(self, ring: PolynomialRing, coefficients: torch.Tensor):
        if not isinstance(ring, PolynomialRing):
            raise TypeError("ring must be a PolynomialRing")
        if not isinstance(coefficients, torch.Tensor):
            raise TypeError("coefficients must be a torch.Tensor")
        if coefficients.ndim != len(ring.axes) + 1 or coefficients.shape[-1] != 2:
            raise ValueError("expected one dimension per variable and a final Gaussian axis of 2")
        if any(size == 0 for size in coefficients.shape):
            raise ValueError("coefficient axes must be nonempty")
        self._ring = ring
        self._coefficients = _reduce(residues(coefficients, ring.q), ring)

    @classmethod
    def _from_owned(cls, ring: PolynomialRing, coefficients: torch.Tensor) -> "MultivariatePoly":
        """Take ownership of a fresh canonical arithmetic result without copying."""
        result = object.__new__(cls)
        result._ring, result._coefficients = ring, coefficients
        return result

    @property
    def ring(self) -> PolynomialRing:
        return self._ring

    @property
    def coefficients(self) -> torch.Tensor:
        return self._coefficients.clone()

    @property
    def device(self) -> torch.device:
        return self._coefficients.device

    def to(self, device: torch.device | str) -> "MultivariatePoly":
        """Return a polynomial on a CPU or CUDA device."""
        return MultivariatePoly(self.ring, self._coefficients.to(device))

    def _coerce(self, other: object) -> "MultivariatePoly":
        if type(other) is int or isinstance(other, tuple):
            other = self.ring.constant(other, device=self.device)
        if not isinstance(other, MultivariatePoly):
            raise TypeError("operand must be a MultivariatePoly, int, or Gaussian integer pair")
        if self.ring != other.ring:
            raise ValueError("polynomials must belong to the same ring")
        if self.device != other.device:
            raise ValueError("polynomials must be on the same device; use .to(device)")
        return other

    def __add__(self, other: "MultivariatePoly | GaussianInteger") -> "MultivariatePoly":
        other = self._coerce(other)
        return self._from_owned(
            self.ring, (self._coefficients + other._coefficients).remainder(self.ring.q)
        )

    __radd__ = __add__

    def __neg__(self) -> "MultivariatePoly":
        return self._from_owned(self.ring, (-self._coefficients).remainder(self.ring.q))

    def __sub__(self, other: "MultivariatePoly | GaussianInteger") -> "MultivariatePoly":
        return self + (-self._coerce(other))

    def __rsub__(self, other: GaussianInteger) -> "MultivariatePoly":
        return self._coerce(other) - self

    def __mul__(self, other: "MultivariatePoly | GaussianInteger") -> "MultivariatePoly":
        return self.multiply(other)

    def multiply(
        self,
        other: "MultivariatePoly | GaussianInteger",
        *,
        compiled: bool = False,
        backend: str = "inductor",
    ) -> "MultivariatePoly":
        """Multiply with an optional full native-ring compiler kernel.

        The compiled path is lazy; warm one call before timing repeated products.
        General padded ideals use the eager convolution/reduction path.
        """
        other = self._coerce(other)
        if self.ring.convolution_shape == tuple(axis.degree for axis in self.ring.axes):
            from ._multiplication import native_multiplier

            kernel = native_multiplier(self.ring, self.device, backend if compiled else None)
            return self._from_owned(self.ring, kernel(self._coefficients, other._coefficients))
        if compiled:
            raise ValueError("compiled multiplication requires native transforms on every axis")
        # Native transforms use the canonical basis. General ideals need
        # padding for linear convolution before monic quotient reduction.
        operand_shape = (2, *self.ring.ntt_shape, 2)
        operands = torch.zeros(operand_shape, dtype=torch.int64, device=self.device)
        slices = tuple(slice(0, axis.degree) for axis in self.ring.axes) + (slice(None),)
        operands[(0, *slices)] = self._coefficients
        operands[(1, *slices)] = other._coefficients
        arithmetic = arithmetic_context(self.ring.q)
        plans = self.ring.transform_plans
        for dim, plan in enumerate(plans):
            operands = plan._transform_residues(operands, dim=dim + 1)
        left, right = operands.unbind(0)
        product = gaussian_mul_residues(left, right, arithmetic)
        for dim, plan in enumerate(plans):
            product = plan._transform_residues(product, dim=dim, inverse=True)
        support = tuple(slice(0, length) for length in self.ring.convolution_shape) + (slice(None),)
        return MultivariatePoly(self.ring, product[support])

    __rmul__ = __mul__

    def __pow__(self, exponent: int) -> "MultivariatePoly":
        if type(exponent) is not int or exponent < 0:
            raise ValueError("exponent must be a nonnegative integer")
        result, base = self.ring.one(device=self.device), self
        while exponent:
            if exponent & 1:
                result = result * base
            exponent >>= 1
            if exponent:
                base = base * base
        return result

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MultivariatePoly)
            and self.ring == other.ring
            and self.device == other.device
            and torch.equal(self._coefficients, other._coefficients)
        )

    def __repr__(self) -> str:
        variables = ",".join(axis.name for axis in self.ring.axes)
        return f"MultivariatePoly(q={self.ring.q}, variables=({variables}), device={self.device})"
