"""GL's BGV-style integer variant (2025/1935, Section 4).

Plaintext t is a splitting prime. Ciphertext/key primes are additionally 1 mod t,
which makes the plaintext-preserving ModSwitch correction particularly simple.
"""

import math

import torch

from ._number_theory import MAX_MODULUS, is_prime
from ._transform_tables import resolve_device
from .ciphertext import Ciphertext, MatrixProductCiphertext, QuadraticCiphertext
from .engine import Engine
from .matrix import MatrixNTTPlan
from .modular import mul_mod, sub_mod
from .multivariate_poly import MultivariatePoly
from .rings import PolynomialRing
from .rns import RNSPoly, RNSRing


def integer_ring(n: int, p: int, plaintext_modulus: int, *, limb_count=4, bits=30) -> RNSRing:
    """Select a GL chain with every q_i = 1 mod lcm(4np,t)."""
    if type(plaintext_modulus) is not int or not is_prime(plaintext_modulus):
        raise ValueError("plaintext_modulus must be a splitting prime")
    reference = PolynomialRing.multivariate_poly(plaintext_modulus, n, p)
    MatrixNTTPlan(reference)
    if (
        type(limb_count) is not int
        or limb_count < 1
        or type(bits) is not int
        or not 3 <= bits <= 30
    ):
        raise ValueError("limb_count must be positive and bits must be between 3 and 30")
    step = math.lcm(4 * n * p, plaintext_modulus)
    candidate = ((1 << (bits - 1)) // step + 1) * step + 1
    primes = []
    while len(primes) < limb_count and candidate <= min(MAX_MODULUS, (1 << bits) - 1):
        if is_prime(candidate):
            primes.append(candidate)
        candidate += step
    if len(primes) != limb_count:
        raise ValueError("not enough integer-compatible primes in this bit range")
    return RNSRing(tuple(primes), reference.axes)


class IntegerEncoder:
    """Exact encoding of 2*phi(p) independent Z_t matrices.

    Slots have shape (phi(p),n,n,2). The last axis selects +I and -I branches;
    these are independent integer matrices, not real/imaginary channels.
    """

    def __init__(self, ring: RNSRing, plaintext_modulus: int, *, device="cpu"):
        if not isinstance(ring, RNSRing):
            raise TypeError("ring must be an RNSRing")
        self.ring, self.t, self.device = ring, plaintext_modulus, resolve_device(device)
        self.plaintext_ring = PolynomialRing(plaintext_modulus, ring.axes)
        self.plan = MatrixNTTPlan(self.plaintext_ring, device=self.device)
        self.slot_shape = self.plan.slot_shape

    def encode(self, matrices: torch.Tensor) -> RNSPoly:
        if not isinstance(matrices, torch.Tensor) or matrices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("integer matrices must be an int32 or int64 tensor")
        polynomial = self.plan.inverse(matrices)
        values = polynomial._coefficients
        centered = torch.where(values > self.t // 2, values - self.t, values)
        return self.ring.polynomial(centered)

    def decode(self, polynomial: RNSPoly) -> torch.Tensor:
        if not isinstance(polynomial, RNSPoly):
            raise TypeError("expected an RNSPoly")
        if polynomial.ring.axes != self.ring.axes or polynomial.device != self.device:
            raise ValueError("polynomial must have the encoder's axes and device")
        values = torch.tensor(
            [v % self.t for v in polynomial.reconstruct(centered=True)],
            dtype=torch.int32,
            device=self.device,
        ).reshape(self.ring.shape)
        return self.plan.forward(MultivariatePoly._from_owned(self.plaintext_ring, values))


def mod_switch_polynomial(polynomial: RNSPoly, plaintext_modulus: int) -> RNSPoly:
    """Plaintext-preserving BGV ModSwitch dropping the last prime p=1 mod t.

    Set r = centered(c_last/t mod p), then c'=(c-t*r)/p. Since p=1 mod t,
    the represented plaintext is unchanged and the rounding error is t times
    a small polynomial. This differs from CKKS rescale().
    """
    ring, t = polynomial.ring, plaintext_modulus
    if type(t) is not int or t < 2 or any(q % t != 1 for q in ring.moduli):
        raise ValueError("ModSwitch requires t >= 2 and ciphertext primes 1 modulo t")
    lower = ring._lower_ring
    p = ring.moduli[-1]
    residue = mul_mod(polynomial.limbs[-1]._coefficients, pow(t, -1, p), p)
    correction = torch.where(residue > p // 2, residue - p, residue)
    limbs = []
    for limb in polynomial.limbs[:-1]:
        prime = limb.ring.q
        delta = mul_mod(correction.remainder(prime), t, prime)
        values = mul_mod(sub_mod(limb._coefficients, delta, prime), pow(p, -1, prime), prime)
        limbs.append(MultivariatePoly._from_owned(limb.ring, values))
    return RNSPoly(lower, tuple(limbs))


class IntegerEngine(Engine):
    """Encrypt m+t*e; use IntegerEncoder to reduce decrypted coefficients mod t."""

    def __init__(self, ring, plaintext_modulus, **kwargs):
        super().__init__(ring, **kwargs)
        if type(plaintext_modulus) is not int or plaintext_modulus < 2:
            raise ValueError("plaintext_modulus must be an integer >= 2")
        if any(q % plaintext_modulus != 1 for q in ring.moduli):
            raise ValueError("integer engine requires ciphertext primes 1 modulo t")
        self.plaintext_modulus = plaintext_modulus

    def _sample_noise(self):
        return super()._sample_noise().to(torch.int64) * self.plaintext_modulus

    def mod_switch(self, ciphertext):
        if isinstance(ciphertext, Ciphertext):
            components = (ciphertext.b, ciphertext.a)
        elif isinstance(ciphertext, QuadraticCiphertext):
            components = (ciphertext.c0, ciphertext.c1, ciphertext.c2)
        elif isinstance(ciphertext, MatrixProductCiphertext):
            components = (ciphertext.c00, ciphertext.c10, ciphertext.c01, ciphertext.c11)
        else:
            raise TypeError("expected a ciphertext")
        for component in components:
            self._check_polynomial(component)
        return type(ciphertext)(
            *(mod_switch_polynomial(c, self.plaintext_modulus) for c in components)
        )
