"""Research symmetric encryption engine; Torch sampling is not a secure RNG."""

import math

import torch

from ._transform_tables import resolve_device
from .ciphertext import Ciphertext, MatrixProductCiphertext, QuadraticCiphertext
from .matrix import _matrix_plan, matrix_transpose
from .multivariate_poly import MultivariatePoly
from .rings import GaussianInteger
from .rns import RNSPoly, RNSRing


def create_secret_key(
    ring: RNSRing,
    generator: torch.Generator,
    *,
    device: torch.device | str | None = None,
) -> RNSPoly:
    """Sample binary Gaussian coefficients using a caller-owned key generator.

    In an X/Y/W ring the returned key lies in the small X/W ring. Engines
    embed it when needed, without retaining it. Use a separate generator for
    encryption randomness so key generation does not consume that stream.
    """
    if not isinstance(ring, RNSRing):
        raise TypeError("ring must be an RNSRing")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    device = resolve_device(generator.device if device is None else device)
    if resolve_device(generator.device) != device:
        raise ValueError("random generator and secret key must be on the same device")
    if {axis.name for axis in ring.axes} == {"X", "Y", "W"}:
        ring = RNSRing(ring.moduli, tuple(axis for axis in ring.axes if axis.name != "Y"))
    coefficients = torch.randint(
        2, ring.shape, dtype=torch.int32, device=device, generator=generator
    )
    return ring.polynomial(coefficients)


class Engine:
    """Encrypt b=-a*s+e+m and decrypt b+a*s=m+e in an RNS ring.

    Small-secret real and imaginary coefficient channels are independently
    binary; X/Y/W engines embed caller-supplied s(X,W) as constant in Y.
    The engine retains no secret. Noise is sampled once per encryption and
    reduced into all limbs. Uniform a is sampled independently in each limb. This is an
    AI-written arithmetic prototype, not a security-proven FHE implementation.
    """

    def __init__(
        self,
        ring: RNSRing,
        *,
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
        names = tuple(axis.name for axis in ring.axes)
        self._Y_dim = names.index("Y") if set(names) == {"X", "Y", "W"} else None
        self._small_ring = (
            RNSRing(ring.moduli, tuple(axis for axis in ring.axes if axis.name != "Y"))
            if self._Y_dim is not None
            else ring
        )

    def _embed_secret_key(self, secret_key: RNSPoly) -> RNSPoly:
        if (
            not isinstance(secret_key, RNSPoly)
            or self._Y_dim is None
            or secret_key.ring != self._small_ring
        ):
            return secret_key
        if secret_key.device != self.device:
            raise ValueError("secret key must have the engine's device")
        limbs = []
        for small_limb, full_ring in zip(secret_key.limbs, self.ring.rings, strict=True):
            coefficients = torch.zeros(full_ring.shape, dtype=torch.int32, device=self.device)
            coefficients.select(self._Y_dim, 0).copy_(small_limb._coefficients)
            limbs.append(MultivariatePoly._from_owned(full_ring, coefficients))
        return RNSPoly(self.ring, tuple(limbs))

    def _check_matrix_secret(self, secret_key: RNSPoly) -> None:
        self._check_polynomial(secret_key)
        if self._Y_dim is None:
            raise ValueError("matrix encryption requires the X/Y/W ring")
        for limb in secret_key.limbs:
            if bool(
                limb._coefficients.narrow(self._Y_dim, 1, self.ring.shape[self._Y_dim] - 1).any()
            ):
                raise ValueError("matrix secret must lie in the small X/W ring (constant in Y)")

    def _check_matrix_layout(self) -> None:
        # The prescribed basis s(X),s(Y) requires X rows and Y columns.
        if tuple(axis.name for axis in self.ring.axes if axis.name != "W") != ("X", "Y"):
            raise ValueError("matrix encryption requires X rows and Y columns")

    def _check_polynomial(self, polynomial: RNSPoly) -> None:
        if not isinstance(polynomial, RNSPoly):
            raise TypeError("expected an RNSPoly")
        if polynomial.ring != self.ring or polynomial.device != self.device:
            raise ValueError("polynomial must have the engine's ring and device")

    def _check_secret_key(self, secret_key):
        if not isinstance(secret_key, RNSPoly):
            raise TypeError("expected an RNSPoly secret key")
        if secret_key.ring not in (self.ring, self._small_ring) or secret_key.device != self.device:
            raise ValueError("secret key must have the engine's ring or small ring and device")

    def _sample_noise(self) -> torch.Tensor:
        return (
            torch.empty(self.ring.shape, dtype=torch.float32, device=self.device)
            .normal_(mean=0, std=self.noise_std, generator=self._generator)
            .round_()
            .to(torch.int32)
        )

    def encrypt(
        self, message: RNSPoly | torch.Tensor | GaussianInteger, secret_key: RNSPoly
    ) -> Ciphertext:
        """Encrypt using an explicit binary key; never retain the supplied key."""
        self._check_secret_key(secret_key)
        coefficients = secret_key.centered_coefficients()
        if not bool(((coefficients == 0) | (coefficients == 1)).all()):
            raise ValueError("secret key coefficients must be binary (0 or 1) across all limbs")
        if isinstance(message, torch.Tensor):
            if message.device != self.device:
                raise ValueError("message tensor must be on the engine's device")
            message = self.ring.polynomial(message)
        elif type(message) is int or isinstance(message, tuple):
            message = self.ring.constant(message, device=self.device)
        self._check_polynomial(message)
        secret_key = self._embed_secret_key(secret_key)
        uniform_component = self.ring.uniform(device=self.device, generator=self._generator)
        noise = self.ring.polynomial(self._sample_noise())
        secret_product = uniform_component.multiply(
            secret_key, compiled=self.compiled, backend=self.backend
        )
        masked_component = message + noise - secret_product
        return Ciphertext(b=masked_component, a=uniform_component)

    def multiply(self, left: Ciphertext, right: Ciphertext) -> QuadraticCiphertext:
        """Return an unrelinearized product, using the engine's compiler settings."""
        for ciphertext in (left, right):
            if not isinstance(ciphertext, Ciphertext):
                raise TypeError("multiplication requires two Ciphertext values")
            self._check_polynomial(ciphertext.b)
            self._check_polynomial(ciphertext.a)
        return left.multiply(right, compiled=self.compiled, backend=self.backend)

    def matrix_multiply(self, left: Ciphertext, right: Ciphertext) -> MatrixProductCiphertext:
        """Encrypt the matrix product m0 @ m1.T without key switching."""
        self._check_matrix_layout()
        for ciphertext in (left, right):
            if not isinstance(ciphertext, Ciphertext):
                raise TypeError("matrix multiplication requires two Ciphertext values")
            self._check_polynomial(ciphertext.b)
            self._check_polynomial(ciphertext.a)
        component_limbs = ([], [], [], [])
        for left_b, left_a, right_b, right_a in zip(
            left.b.limbs, left.a.limbs, right.b.limbs, right.a.limbs, strict=True
        ):
            products = _matrix_plan(left_b.ring, self.device)._multiply_ciphertext_components(
                left_b, left_a, right_b, right_a
            )
            for limbs, product in zip(component_limbs, products, strict=True):
                limbs.append(product)
        return MatrixProductCiphertext(
            *(RNSPoly(self.ring, tuple(limbs)) for limbs in component_limbs)
        )

    def decrypt(
        self,
        ciphertext: Ciphertext | QuadraticCiphertext | MatrixProductCiphertext,
        secret_key: RNSPoly,
    ) -> RNSPoly:
        """Return the noisy message in RNS form, without rounding away its error."""
        if not isinstance(ciphertext, (Ciphertext, QuadraticCiphertext, MatrixProductCiphertext)):
            raise TypeError(
                "ciphertext must be a Ciphertext, QuadraticCiphertext, or MatrixProductCiphertext"
            )
        self._check_secret_key(secret_key)
        secret_key = self._embed_secret_key(secret_key)
        if isinstance(ciphertext, MatrixProductCiphertext):
            self._check_matrix_secret(secret_key)
            self._check_matrix_layout()
            for component in (ciphertext.c00, ciphertext.c10, ciphertext.c01, ciphertext.c11):
                self._check_polynomial(component)
            secret_Y = matrix_transpose(secret_key)
            # c00+c10*sX+(c01+c11*sX)*sY: three ordinary products, avoiding
            # separately materializing the mixed secret sX*sY.
            X_term = ciphertext.c10.multiply(
                secret_key, compiled=self.compiled, backend=self.backend
            )
            Y_term = ciphertext.c01 + ciphertext.c11.multiply(
                secret_key, compiled=self.compiled, backend=self.backend
            )
            return (
                ciphertext.c00
                + X_term
                + Y_term.multiply(secret_Y, compiled=self.compiled, backend=self.backend)
            )
        if isinstance(ciphertext, QuadraticCiphertext):
            for component in (ciphertext.c0, ciphertext.c1, ciphertext.c2):
                self._check_polynomial(component)
            # Horner evaluation equals c0+c1*s+c2*s^2 with two products,
            # avoiding an extra multiplication to materialize s^2.
            linear_term = (
                ciphertext.c2.multiply(secret_key, compiled=self.compiled, backend=self.backend)
                + ciphertext.c1
            )
            return ciphertext.c0 + linear_term.multiply(
                secret_key, compiled=self.compiled, backend=self.backend
            )
        self._check_polynomial(ciphertext.b)
        self._check_polynomial(ciphertext.a)
        secret_product = ciphertext.a.multiply(
            secret_key, compiled=self.compiled, backend=self.backend
        )
        return ciphertext.b + secret_product
