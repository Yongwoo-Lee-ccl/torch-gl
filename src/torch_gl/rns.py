"""Residue-number polynomials composed of independent 30-bit prime limbs."""

import math
from dataclasses import dataclass
from functools import cached_property

import torch

from ._number_theory import (
    MAX_MODULUS,
    is_power_of_two,
    is_prime,
    next_power_of_two,
    odd_prime_power,
)
from ._transform_tables import resolve_device
from .modular import mul_mod
from .multivariate_poly import MultivariatePoly
from .rings import GaussianInteger, PolynomialRing, QuotientAxis


def select_ntt_primes(n: int, p: int, limb_count: int, *, bits: int = 30) -> tuple[int, ...]:
    """Select distinct bits-bit primes q=1 mod 4np, in ascending order."""
    if not is_power_of_two(n):
        raise ValueError("n must be a positive power of two")
    prime, exponent = odd_prime_power(p)
    if type(limb_count) is not int or limb_count < 1:
        raise ValueError("limb_count must be a positive integer")
    if type(bits) is not int or not 3 <= bits <= 30:
        raise ValueError("bits must be between 3 and 30")
    step = 4 * n * p
    if prime != 3:
        degree = (prime - 1) * prime ** (exponent - 1)
        step = math.lcm(step, next_power_of_two(2 * degree - 1))
    lower_bound, upper_bound = 1 << (bits - 1), min((1 << bits) - 1, MAX_MODULUS)
    multiplier = (lower_bound - 1 + step - 1) // step
    primes = []
    while len(primes) < limb_count:
        candidate = multiplier * step + 1
        if candidate > upper_bound:
            raise ValueError("not enough compatible primes in the requested bit range")
        if is_prime(candidate):
            primes.append(candidate)
        multiplier += 1
    return tuple(primes)


@dataclass(frozen=True)
class RNSRing:
    """A common quotient basis modulo Q=product(moduli), stored limb by limb."""

    moduli: tuple[int, ...]
    axes: tuple[QuotientAxis, ...]

    def __post_init__(self) -> None:
        moduli, axes = tuple(self.moduli), tuple(self.axes)
        if not moduli or len(set(moduli)) != len(moduli):
            raise ValueError("RNS moduli must be nonempty and distinct")
        object.__setattr__(self, "moduli", moduli)
        object.__setattr__(self, "axes", axes)
        # Each limb validates primality, size, and transform compatibility.
        _ = self.rings

    @classmethod
    def multivariate_poly(
        cls,
        moduli: tuple[int, ...],
        n: int,
        p: int,
        *,
        variables: tuple[str, ...] = ("X", "Y", "W"),
    ) -> "RNSRing":
        moduli = tuple(moduli)
        if not moduli:
            raise ValueError("RNS moduli must be nonempty")
        base_ring = PolynomialRing.multivariate_poly(moduli[0], n, p, variables=variables)
        return cls(moduli, base_ring.axes)

    @classmethod
    def for_parameters(
        cls,
        n: int,
        p: int,
        *,
        limb_count: int = 3,
        bits: int = 30,
        variables: tuple[str, ...] = ("X", "Y", "W"),
    ) -> "RNSRing":
        return cls.multivariate_poly(
            select_ntt_primes(n, p, limb_count, bits=bits), n, p, variables=variables
        )

    @cached_property
    def rings(self) -> tuple[PolynomialRing, ...]:
        return tuple(PolynomialRing(modulus, self.axes) for modulus in self.moduli)

    @cached_property
    def modulus(self) -> int:
        """The aggregate modulus as a Python integer; it may exceed int64."""
        return math.prod(self.moduli)

    @cached_property
    def _crt_weights(self) -> tuple[int, ...]:
        return tuple(
            (self.modulus // modulus) * pow(self.modulus // modulus, -1, modulus)
            for modulus in self.moduli
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.rings[0].shape

    @property
    def limb_count(self) -> int:
        return len(self.moduli)

    def polynomial(self, coefficients: torch.Tensor) -> "RNSPoly":
        """Encode the same integer coefficients in every limb."""
        return RNSPoly(self, tuple(ring.polynomial(coefficients) for ring in self.rings))

    def from_real(self, coefficients: torch.Tensor) -> "RNSPoly":
        return RNSPoly(self, tuple(ring.from_real(coefficients) for ring in self.rings))

    def constant(self, value: GaussianInteger, *, device: torch.device | str = "cpu") -> "RNSPoly":
        return RNSPoly(self, tuple(ring.constant(value, device=device) for ring in self.rings))

    def zero(self, *, device: torch.device | str = "cpu") -> "RNSPoly":
        return self.constant(0, device=device)

    def one(self, *, device: torch.device | str = "cpu") -> "RNSPoly":
        return self.constant(1, device=device)

    def generator(self, name: str, *, device: torch.device | str = "cpu") -> "RNSPoly":
        return RNSPoly(self, tuple(ring.generator(name, device=device) for ring in self.rings))

    def uniform(
        self, *, device: torch.device | str = "cpu", generator: torch.Generator | None = None
    ) -> "RNSPoly":
        """Sample uniform residues independently in each prime limb."""
        device = resolve_device(device)
        limbs = tuple(
            MultivariatePoly._from_owned(
                ring,
                torch.randint(
                    ring.q, ring.shape, dtype=torch.int32, device=device, generator=generator
                ),
            )
            for ring in self.rings
        )
        return RNSPoly(self, limbs)


@dataclass(frozen=True, eq=False)
class RNSPoly:
    """A polynomial represented by a tuple of MultivariatePoly blocks."""

    ring: RNSRing
    limbs: tuple[MultivariatePoly, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ring, RNSRing):
            raise TypeError("ring must be an RNSRing")
        limbs = tuple(self.limbs)
        if len(limbs) != self.ring.limb_count:
            raise ValueError("expected exactly one polynomial per RNS limb")
        for limb, ring in zip(limbs, self.ring.rings, strict=True):
            if not isinstance(limb, MultivariatePoly) or limb.ring != ring:
                raise ValueError(
                    "limb polynomial does not match its RNS modulus and quotient basis"
                )
        if any(limb.device != limbs[0].device for limb in limbs):
            raise ValueError("all RNS limbs must be on the same device")
        object.__setattr__(self, "limbs", limbs)

    @property
    def device(self) -> torch.device:
        return self.limbs[0].device

    def to(self, device: torch.device | str) -> "RNSPoly":
        return RNSPoly(self.ring, tuple(limb.to(device) for limb in self.limbs))

    def _coerce(self, other: object) -> "RNSPoly":
        if type(other) is int or isinstance(other, tuple):
            other = self.ring.constant(other, device=self.device)
        if not isinstance(other, RNSPoly):
            raise TypeError("operand must be an RNSPoly, int, or Gaussian integer pair")
        if other.ring != self.ring or other.device != self.device:
            raise ValueError("RNS operands must have the same ring, limb order, and device")
        return other

    def __add__(self, other: "RNSPoly | GaussianInteger") -> "RNSPoly":
        other = self._coerce(other)
        return RNSPoly(
            self.ring,
            tuple(left + right for left, right in zip(self.limbs, other.limbs, strict=True)),
        )

    __radd__ = __add__

    def __neg__(self) -> "RNSPoly":
        return RNSPoly(self.ring, tuple(-limb for limb in self.limbs))

    def __sub__(self, other: "RNSPoly | GaussianInteger") -> "RNSPoly":
        other = self._coerce(other)
        return RNSPoly(
            self.ring,
            tuple(left - right for left, right in zip(self.limbs, other.limbs, strict=True)),
        )

    def __rsub__(self, other: GaussianInteger) -> "RNSPoly":
        return self._coerce(other) - self

    def multiply(
        self,
        other: "RNSPoly | GaussianInteger",
        *,
        compiled: bool = False,
        backend: str = "inductor",
    ) -> "RNSPoly":
        if type(other) is int:
            # Integer scaling does not need an NTT in each limb.
            return RNSPoly(
                self.ring,
                tuple(
                    MultivariatePoly._from_owned(
                        limb.ring,
                        mul_mod(limb._coefficients, other % limb.ring.q, limb.ring.q),
                    )
                    for limb in self.limbs
                ),
            )
        other = self._coerce(other)
        return RNSPoly(
            self.ring,
            tuple(
                left.multiply(right, compiled=compiled, backend=backend)
                for left, right in zip(self.limbs, other.limbs, strict=True)
            ),
        )

    def __mul__(self, other: "RNSPoly | GaussianInteger") -> "RNSPoly":
        return self.multiply(other)

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RNSPoly)
            and self.ring == other.ring
            and all(left == right for left, right in zip(self.limbs, other.limbs, strict=True))
        )

    def centered_coefficients(self) -> torch.Tensor:
        """Fast signed int32 lift for coefficients lying within half the first prime.

        Verify consistency against every limb, so the returned small integers
        are also the exact centered CRT representatives. For general large
        values use reconstruct(), which returns arbitrary-precision integers.
        """
        first = self.limbs[0]._coefficients
        modulus = self.ring.moduli[0]
        centered = torch.where(first > modulus // 2, first - modulus, first)
        for limb in self.limbs[1:]:
            if not torch.equal(centered.remainder(limb.ring.q), limb._coefficients):
                raise ValueError(
                    "coefficients exceed the first-limb signed range; use reconstruct()"
                )
        return centered

    def reconstruct(self, *, centered: bool = True) -> tuple[int, ...]:
        """Exact CPU CRT reconstruction as flat C-order Python integers.

        This debugging/export operation supports Q larger than int64; it is
        deliberately outside the tensor arithmetic hot path.
        """
        rows = [limb._coefficients.reshape(-1).cpu().tolist() for limb in self.limbs]
        result = []
        for residues in zip(*rows, strict=True):
            value = (
                sum(
                    residue * weight
                    for residue, weight in zip(residues, self.ring._crt_weights, strict=True)
                )
                % self.ring.modulus
            )
            if centered and value > self.ring.modulus // 2:
                value -= self.ring.modulus
            result.append(value)
        return tuple(result)
