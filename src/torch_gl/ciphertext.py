"""Research RLWE ciphertexts before and after unrelinearized multiplication."""

from dataclasses import dataclass

from .rns import RNSPoly


@dataclass(frozen=True)
class Ciphertext:
    """Ciphertext components in (b, a) order; contains no secret or sampled noise."""

    b: RNSPoly
    a: RNSPoly

    def __post_init__(self) -> None:
        if not isinstance(self.b, RNSPoly) or not isinstance(self.a, RNSPoly):
            raise TypeError("ciphertext components must be RNSPoly values")
        if self.b.ring != self.a.ring or self.b.device != self.a.device:
            raise ValueError("ciphertext components must have the same ring and device")

    def multiply(
        self, other: "Ciphertext", *, compiled: bool = False, backend: str = "inductor"
    ) -> "QuadraticCiphertext":
        """Multiply under (1,s,s^2); plaintext scales multiply as well."""
        if not isinstance(other, Ciphertext):
            raise TypeError("multiplication requires two Ciphertext values")
        if self.b.ring != other.b.ring or self.b.device != other.b.device:
            raise ValueError("ciphertexts must have the same ring and device")
        constant_component = self.b.multiply(other.b, compiled=compiled, backend=backend)
        linear_component = self.b.multiply(other.a, compiled=compiled, backend=backend)
        linear_component = linear_component + other.b.multiply(
            self.a, compiled=compiled, backend=backend
        )
        quadratic_component = self.a.multiply(other.a, compiled=compiled, backend=backend)
        return QuadraticCiphertext(constant_component, linear_component, quadratic_component)

    def __mul__(self, other: "Ciphertext") -> "QuadraticCiphertext":
        return self.multiply(other)


@dataclass(frozen=True)
class QuadraticCiphertext:
    """Components (c0,c1,c2) decrypted as c0+c1*s+c2*s^2.

    Multiplication does not relinearize or rescale. The original secret s is
    unchanged; the decryption basis gains s^2. Scale is managed by the caller.
    """

    c0: RNSPoly
    c1: RNSPoly
    c2: RNSPoly

    def __post_init__(self) -> None:
        components = (self.c0, self.c1, self.c2)
        if any(not isinstance(component, RNSPoly) for component in components):
            raise TypeError("ciphertext components must be RNSPoly values")
        if any(
            component.ring != self.c0.ring or component.device != self.c0.device
            for component in components[1:]
        ):
            raise ValueError("ciphertext components must have the same ring and device")


@dataclass(frozen=True)
class MatrixProductCiphertext:
    """Four components under (1,s(X,W),s(Y,W),s(X,W)*s(Y,W)).

    Ordered as (b0 mm b1^T, a0 mm b1^T, b0 mm a1^T, a0 mm a1^T).
    No key switching or rescaling is performed; scale multiplies.
    """

    c00: RNSPoly
    c10: RNSPoly
    c01: RNSPoly
    c11: RNSPoly

    def __post_init__(self) -> None:
        components = (self.c00, self.c10, self.c01, self.c11)
        if any(not isinstance(component, RNSPoly) for component in components):
            raise TypeError("ciphertext components must be RNSPoly values")
        if any(
            component.ring != self.c00.ring or component.device != self.c00.device
            for component in components[1:]
        ):
            raise ValueError("ciphertext components must have the same ring and device")
