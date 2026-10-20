"""Leveled approximate GL evaluation without retaining the decryption key."""

import math
from dataclasses import dataclass, field

import torch

from ._transform_tables import resolve_device
from .automorphisms import automorphism, conjugate, hermitian_transpose, swap_axes
from .ciphertext import Ciphertext
from .encoding import MatrixEncoder
from .engine import Engine
from .key_switching import KeySwitcher, KeySwitchKey, MatrixKeySwitchKeys
from .matrix import matrix_multiply
from .rns import RNSRing


@dataclass(frozen=True)
class ScaledCiphertext:
    """A two-component ciphertext with its positive finite approximate scale."""

    ciphertext: Ciphertext
    scale: float

    def __post_init__(self):
        if not isinstance(self.ciphertext, Ciphertext):
            raise TypeError("ciphertext must be a Ciphertext")
        if isinstance(self.scale, bool) or not isinstance(self.scale, (int, float)):
            raise TypeError("scale must be a real number")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")

    @property
    def ring(self):
        return self.ciphertext.ring

    @property
    def device(self):
        return self.ciphertext.device

    @property
    def level(self):
        return self.ring.limb_count

    def at_level(self, limb_count):
        return ScaledCiphertext(self.ciphertext.at_level(limb_count), self.scale)


@dataclass(frozen=True)
class EvaluationKeys:
    """Public evaluation material, generated once at the maximum level.

    Row/W automorphisms need small keys, column automorphisms need no key,
    and transposition changes the secret to s(Y,W), requiring a big key.
    """

    hadamard: KeySwitchKey
    matrix: MatrixKeySwitchKeys | None = None
    transpose: KeySwitchKey | None = None
    conjugation: KeySwitchKey | None = None
    adjoint_matrix: MatrixKeySwitchKeys | None = None
    rotations: dict = field(default_factory=dict)


def generate_evaluation_keys(switcher: KeySwitcher, secret, *, matrices=True, rotations=()):
    """Generate evaluation keys explicitly; neither evaluator nor keys retain s."""
    small = switcher._small_secret(secret)
    hadamard = switcher.generate_hadamard_key(small)
    conjugation = switcher.generate_key(conjugate(small), small)
    matrix = transpose = adjoint_matrix = None
    if matrices:
        full = switcher._embed(small)
        transpose = switcher.generate_key(swap_axes(full, "X", "Y"), small, kind="big")
        matrix = switcher.generate_matrix_keys(small)
        adjoint = hermitian_transpose(full)
        adjoint_matrix = MatrixKeySwitchKeys(
            switcher.generate_key(adjoint, small, kind="big"),
            switcher.generate_key(full * adjoint, small, kind="big"),
        )
    rotation_keys = {}
    axes = {axis.name: axis for axis in switcher.parameters.ring.axes}
    for axis, steps in rotations:
        if axis not in axes or axis not in {"X", "Y", "W"} or type(steps) is not int:
            raise ValueError("rotations must be (X/Y/W, integer steps) pairs")
        degree = axes[axis].degree
        steps %= degree
        if steps and axis != "Y":
            exponent = pow(
                2 if axis == "W" else 5, steps, axes[axis].cyclotomic_order or 4 * degree
            )
            rotation_keys[axis, steps] = switcher.generate_key(
                automorphism(small, {axis: exponent}), small
            )
    return EvaluationKeys(hadamard, matrix, transpose, conjugation, adjoint_matrix, rotation_keys)


class Evaluator:
    """Track levels/scales for GL arithmetic using only public evaluation keys.

    Products relinearize and rescale by default. Operands are projected to a
    common prefix level; addition aligns unequal scales by an encoded scalar.
    The existing low-level Engine.matrix_multiply retains its A @ B.T contract.
    Here matmul returns A @ B, and matmul_adjoint implements the paper's A @ B^H.
    """

    def __init__(self, ring: RNSRing, keys: EvaluationKeys, *, device="cpu", rounding="floor"):
        if not isinstance(ring, RNSRing) or not isinstance(keys, EvaluationKeys):
            raise TypeError("expected an RNSRing and EvaluationKeys")
        if keys.hadamard.parameters.ring != ring:
            raise ValueError("evaluation keys must be generated at the evaluator's maximum level")
        if keys.hadamard.parameters.plaintext_modulus is not None:
            raise ValueError("Evaluator is approximate; integer ciphertexts use IntegerEngine")
        if rounding not in {"floor", "nearest"}:
            raise ValueError("rounding must be 'floor' or 'nearest'")
        self.rounding = rounding
        self.ring, self.keys, self.device = ring, keys, resolve_device(device)
        if keys.hadamard.device != self.device:
            raise ValueError("evaluation keys and evaluator must have the same device")
        self._switchers = {}

    def _check(self, value):
        if not isinstance(value, ScaledCiphertext):
            raise TypeError("expected a ScaledCiphertext")
        if (
            value.ring.axes != self.ring.axes
            or self.ring.moduli[: value.level] != value.ring.moduli
            or value.device != self.device
        ):
            raise ValueError("ciphertext must use this evaluator's axes, chain and device")

    def _switcher(self, ring):
        if ring not in self._switchers:
            parameters = self.keys.hadamard.parameters
            self._switchers[ring] = KeySwitcher(
                ring, k=parameters.k, key_moduli=parameters.key_moduli, device=self.device
            )
        return self._switchers[ring]

    def _pair(self, left, right):
        self._check(left)
        self._check(right)
        level = min(left.level, right.level)
        return left.at_level(level), right.at_level(level)

    def _encode(self, value, ring, scale):
        encoder = MatrixEncoder(ring, scale=scale, device=self.device)
        tensor = torch.as_tensor(value, device=self.device)
        try:
            tensor = torch.broadcast_to(tensor, encoder.slot_shape)
        except RuntimeError as error:
            raise ValueError("plaintext does not broadcast to the encoder's slots") from error
        return encoder.encode(tensor)

    def add(self, left, right):
        left, right = self._pair(left, right)
        if not math.isclose(left.scale, right.scale, rel_tol=1e-12):
            # Accurate scale alignment consumes one level. Do not silently
            # relabel metadata: that changes the represented plaintext.
            level = left.level
            if level < 2:
                raise ValueError("aligning unequal scales needs a rescaling level")
            target = max(left.scale, right.scale)
            left = self.multiply_plain(left, target / left.scale)
            right = self.multiply_plain(right, target / right.scale)
            left = ScaledCiphertext(left.ciphertext, target)
            right = ScaledCiphertext(right.ciphertext, target)
        return ScaledCiphertext(left.ciphertext + right.ciphertext, left.scale)

    def negate(self, value):
        self._check(value)
        return ScaledCiphertext(-value.ciphertext, value.scale)

    def subtract(self, left, right):
        return self.add(left, self.negate(right))

    def add_plain(self, value, plaintext):
        self._check(value)
        return ScaledCiphertext(
            value.ciphertext + self._encode(plaintext, value.ring, value.scale), value.scale
        )

    def rescale(self, value):
        self._check(value)
        if value.level < 2:
            raise ValueError("rescaling requires at least two primes")
        return ScaledCiphertext(
            Ciphertext(
                value.ciphertext.b.rescale(rounding=self.rounding),
                value.ciphertext.a.rescale(rounding=self.rounding),
            ),
            value.scale / value.ring.moduli[-1],
        )

    def multiply_plain(self, value, plaintext, *, rescale=True, scale=None):
        self._check(value)
        if rescale and value.level < 2:
            raise ValueError("plaintext multiplication needs a rescaling level")
        plaintext_scale = value.ring.moduli[-1] if scale is None else scale
        encoded = self._encode(plaintext, value.ring, plaintext_scale)
        ct = value.ciphertext
        result = ScaledCiphertext(
            Ciphertext(ct.b * encoded, ct.a * encoded), value.scale * plaintext_scale
        )
        return self.rescale(result) if rescale else result

    def multiply(self, left, right, *, rescale=True):
        left, right = self._pair(left, right)
        if rescale and left.level < 2:
            raise ValueError("ciphertext multiplication needs a rescaling level")
        result = self._switcher(left.ring).relinearize(
            left.ciphertext * right.ciphertext, self.keys.hadamard
        )
        result = ScaledCiphertext(result, left.scale * right.scale)
        return self.rescale(result) if rescale else result

    def square(self, value, *, rescale=True):
        return self.multiply(value, value, rescale=rescale)

    def conjugate(self, value):
        self._check(value)
        if self.keys.conjugation is None:
            raise ValueError("conjugation evaluation key is missing")
        ct = value.ciphertext
        transformed = Ciphertext(conjugate(ct.b), conjugate(ct.a))
        return ScaledCiphertext(
            self._switcher(value.ring).switch_ciphertext(transformed, self.keys.conjugation),
            value.scale,
        )

    def transpose(self, value, *, conjugate=False):
        self._check(value)
        if conjugate:
            return self.transpose(self.conjugate(value))
        if self.keys.transpose is None:
            raise ValueError("transpose evaluation key is missing")
        ct = value.ciphertext
        b, a = (swap_axes(component, "X", "Y") for component in (ct.b, ct.a))
        switched = self._switcher(value.ring).switch_big(a, self.keys.transpose)
        return ScaledCiphertext(Ciphertext(b + switched.b, switched.a), value.scale)

    def rotate(self, value, steps, *, axis="X"):
        self._check(value)
        axes = {a.name: a for a in value.ring.axes}
        if axis not in axes or axis not in {"X", "Y", "W"} or type(steps) is not int:
            raise ValueError("rotation requires an X/Y/W axis and integer steps")
        steps %= axes[axis].degree
        if not steps:
            return value
        exponent = pow(
            2 if axis == "W" else 5, steps, axes[axis].cyclotomic_order or 4 * axes[axis].degree
        )
        ct = value.ciphertext
        transformed = Ciphertext(
            automorphism(ct.b, {axis: exponent}), automorphism(ct.a, {axis: exponent})
        )
        if axis != "Y":
            key = self.keys.rotations.get((axis, steps))
            if key is None:
                raise ValueError("the requested rotation key was not generated")
            transformed = self._switcher(value.ring).switch_ciphertext(transformed, key)
        return ScaledCiphertext(transformed, value.scale)

    def matmul_plain(self, value, plaintext, *, rescale=True):
        """A @ P, using the secret-independent Y trace, with no key switch."""
        self._check(value)
        if rescale and value.level < 2:
            raise ValueError("matrix multiplication needs a rescaling level")
        scale = value.ring.moduli[-1]
        encoded = self._encode(plaintext, value.ring, scale)
        ct = value.ciphertext
        result = ScaledCiphertext(
            Ciphertext(matrix_multiply(ct.b, encoded), matrix_multiply(ct.a, encoded)),
            value.scale * scale,
        )
        return self.rescale(result) if rescale else result

    def matmul_transposed(self, left, right, *, rescale=True):
        left, right = self._pair(left, right)
        if self.keys.matrix is None:
            raise ValueError("matrix evaluation keys are missing")
        if rescale and left.level < 2:
            raise ValueError("matrix multiplication needs a rescaling level")
        product = Engine(left.ring, device=self.device).matrix_multiply(
            left.ciphertext, right.ciphertext
        )
        ct = self._switcher(left.ring).relinearize_matrix(product, self.keys.matrix)
        result = ScaledCiphertext(ct, left.scale * right.scale)
        return self.rescale(result) if rescale else result

    def matmul(self, left, right, *, rescale=True):
        return self.matmul_transposed(left, self.transpose(right), rescale=rescale)

    def matmul_adjoint(self, left, right, *, rescale=True):
        """The paper's circledast: A @ B^H with two big switches."""
        left, right = self._pair(left, right)
        keys = self.keys.adjoint_matrix
        if keys is None:
            raise ValueError("adjoint matrix evaluation keys are missing")
        if rescale and left.level < 2:
            raise ValueError("matrix multiplication needs a rescaling level")
        left_ct, right_ct = left.ciphertext, right.ciphertext
        rb, ra = hermitian_transpose(right_ct.b), hermitian_transpose(right_ct.a)
        switcher = self._switcher(left.ring)
        first = switcher.switch_big(matrix_multiply(left_ct.b, ra), keys.Y_key)
        second = switcher.switch_big(matrix_multiply(left_ct.a, ra), keys.mixed_key)
        ct = Ciphertext(
            matrix_multiply(left_ct.b, rb) + first.b + second.b,
            matrix_multiply(left_ct.a, rb) + first.a + second.a,
        )
        result = ScaledCiphertext(ct, left.scale * right.scale)
        return self.rescale(result) if rescale else result
