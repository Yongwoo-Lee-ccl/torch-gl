"""Two-component research RLWE ciphertexts."""

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
