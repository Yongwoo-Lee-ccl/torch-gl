"""Research symmetric encryption engine; Torch sampling is not a secure RNG."""

import math

import torch

from ._transform_tables import resolve_device
from .ciphertext import Ciphertext
from .rings import GaussianInteger
from .rns import RNSPoly, RNSRing


class Engine:
    """Encrypt b=-a*s+e+m and decrypt b+a*s=m+e in an RNS ring.

    The secret's real and imaginary coefficient channels are independently
    binary. Secret and noise tensors are each sampled once, then reduced into
    all limbs. Uniform a is sampled independently in each limb. This is an
    AI-written arithmetic prototype, not a security-proven FHE implementation.
    """

    def __init__(
        self,
        ring: RNSRing,
        *,
        secret_key: RNSPoly | None = None,
        noise_std: float = 3.2,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
        compiled: bool = False,
        backend: str = "inductor",
    ):
        if not isinstance(ring, RNSRing):
            raise TypeError("ring must be an RNSRing")
        if (
            isinstance(noise_std, bool)
            or not isinstance(noise_std, (int, float))
            or (not math.isfinite(noise_std) or noise_std < 0)
        ):
            raise ValueError("noise_std must be finite and nonnegative")
        self.ring, self.device = ring, resolve_device(device)
        if generator is not None and resolve_device(generator.device) != self.device:
            raise ValueError("random generator and engine must be on the same device")
        self.noise_std, self.compiled, self.backend = float(noise_std), compiled, backend
        self._generator = generator
        if secret_key is None:
            coefficients = torch.randint(
                2, ring.shape, dtype=torch.int64, device=self.device, generator=generator
            )
            secret_key = ring.polynomial(coefficients)
        self._check_polynomial(secret_key)
        coefficients = secret_key.centered_coefficients()
        if not bool(((coefficients == 0) | (coefficients == 1)).all()):
            raise ValueError("secret key coefficients must be binary (0 or 1) across all limbs")
        self._secret_key = secret_key

    @property
    def secret_key(self) -> RNSPoly:
        return self._secret_key

    def _check_polynomial(self, polynomial: RNSPoly) -> None:
        if not isinstance(polynomial, RNSPoly):
            raise TypeError("expected an RNSPoly")
        if polynomial.ring != self.ring or polynomial.device != self.device:
            raise ValueError("polynomial must have the engine's ring and device")

    def _sample_noise(self) -> torch.Tensor:
        return (
            torch.empty(self.ring.shape, dtype=torch.float32, device=self.device)
            .normal_(mean=0, std=self.noise_std, generator=self._generator)
            .round_()
            .to(torch.int64)
        )

    def encrypt(self, message: RNSPoly | torch.Tensor | GaussianInteger) -> Ciphertext:
        if isinstance(message, torch.Tensor):
            if message.device != self.device:
                raise ValueError("message tensor must be on the engine's device")
            message = self.ring.polynomial(message)
        elif type(message) is int or isinstance(message, tuple):
            message = self.ring.constant(message, device=self.device)
        self._check_polynomial(message)
        uniform_component = self.ring.uniform(device=self.device, generator=self._generator)
        noise = self.ring.polynomial(self._sample_noise())
        secret_product = uniform_component.multiply(
            self.secret_key, compiled=self.compiled, backend=self.backend
        )
        masked_component = message + noise - secret_product
        return Ciphertext(b=masked_component, a=uniform_component)

    def decrypt(self, ciphertext: Ciphertext, secret_key: RNSPoly) -> RNSPoly:
        """Return the noisy message in RNS form, without rounding away its error."""
        if not isinstance(ciphertext, Ciphertext):
            raise TypeError("ciphertext must be a Ciphertext")
        self._check_polynomial(ciphertext.b)
        self._check_polynomial(ciphertext.a)
        self._check_polynomial(secret_key)
        secret_product = ciphertext.a.multiply(
            secret_key, compiled=self.compiled, backend=self.backend
        )
        return ciphertext.b + secret_product
