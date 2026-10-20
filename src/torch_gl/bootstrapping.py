"""GL bootstrapping using the coefficient ring changes of ePrint 2026/956.

This reference implementation follows Section 4: ModRaise, X/W inverse DFT,
homomorphic sine/arcsine EvalMod, then W/X DFT. Linear transforms use native
matrix traces with a constant number of key switches, not a diagonal rotation
method. No secret, decryptor or plaintext oracle is available to Bootstrapper.
"""

import math
from dataclasses import dataclass

import torch

from ._transform_tables import resolve_device
from .automorphisms import embed, swap_axes
from .ciphertext import Ciphertext
from .encoding import MatrixEncoder
from .evaluator import EvaluationKeys, Evaluator, ScaledCiphertext, generate_evaluation_keys
from .key_switching import KeySwitcher, KeySwitchKey
from .matrix import matrix_multiply
from .ring_changes import merge_ciphertexts, split_ciphertext
from .rns import RNSRing
from .trace import auxiliary_matrix_multiply


@dataclass(frozen=True)
class BootstrapParameters:
    """Full refresh chain and the approximation domain declared by the caller.

    Input is at one prime q0. Each real/imaginary coefficient must be within
    message_ratio*q0 of zero before ModRaise, and the carry I must satisfy the
    declared carry_bound. These encrypted bounds cannot be checked by the
    evaluator. The default sine approximation is accurate for |I| <= 8 and
    message_ratio <= 1/16. These are numerical, not security parameters.
    """

    ring: RNSRing
    k: int = 2
    double_angle: int = 6
    carry_bound: float = 8.0
    message_ratio: float = 1 / 16
    arcsine_degree: int = 7

    def __post_init__(self):
        if not isinstance(self.ring, RNSRing):
            raise TypeError("ring must be an RNSRing")
        if tuple(a.name for a in self.ring.axes) != ("X", "Y", "W"):
            raise ValueError("bootstrapping requires the standard X/Y/W axis order")
        x, y, w = self.ring.axes
        if x.modulus != y.modulus or w.cyclotomic_order is None:
            raise ValueError("bootstrapping requires matching X/Y and a cyclotomic W")
        if type(self.k) is not int or self.k < 1:
            raise ValueError("k must be positive")
        if type(self.double_angle) is not int or not 0 <= self.double_angle <= 20:
            raise ValueError("double_angle must be between 0 and 20")
        if type(self.arcsine_degree) is not int or self.arcsine_degree not in (1, 3, 5, 7, 9):
            raise ValueError("arcsine_degree must be one of 1,3,5,7,9")
        for value in (self.carry_bound, self.message_ratio):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("approximation bounds must be finite real numbers")
        if self.carry_bound < 1 or not 0 < self.message_ratio < 0.25:
            raise ValueError("carry_bound must be >= 1 and 0 < message_ratio < 1/4")
        # Taylor degrees 9/10 on this small interval, followed by doubling.
        if 2 * math.pi * (self.carry_bound + self.message_ratio) / 2**self.double_angle > 0.85:
            raise ValueError("increase double_angle for the declared carry bound")
        if self.ring.limb_count < self.consumed_levels + 2:
            raise ValueError(f"refresh needs at least {self.consumed_levels + 2} primes")
        # Validate the encoder/native-transform parameter family before keygen.
        MatrixEncoder(self.ring)

    @property
    def consumed_levels(self):
        # CtS 2, real/imag split 1, range scaling 1, Taylor 5,
        # double-angle r, arcsine ceil(log2(d))+1, final scaling 1, StC 2.
        arc = 0 if self.arcsine_degree == 1 else math.ceil(math.log2(self.arcsine_degree)) + 1
        return 12 + self.double_angle + arc

    @property
    def base_ring(self):
        return RNSRing(self.ring.moduli, (self.ring.axes[0], self.ring.axes[2]))

    @property
    def auxiliary_ring(self):
        return RNSRing.auxiliary(
            self.ring.moduli, self.ring.axes[0].degree, self.ring.axes[2].cyclotomic_order
        )


@dataclass(frozen=True)
class BootstrapKeys:
    """Public refresh material: small EvalMod keys and two transpose keys."""

    parameters: BootstrapParameters
    base: EvaluationKeys
    x_transpose: KeySwitchKey
    w_transpose: KeySwitchKey

    def __post_init__(self):
        if not isinstance(self.parameters, BootstrapParameters) or not isinstance(
            self.base, EvaluationKeys
        ):
            raise TypeError("expected bootstrap parameters and base evaluation keys")
        if (
            self.base.hadamard.parameters.ring != self.parameters.base_ring
            or self.base.conjugation is None
        ):
            raise ValueError(
                "base refresh keys must include matching small relinearization/conjugation keys"
            )
        for key, ring in (
            (self.x_transpose, self.parameters.ring),
            (self.w_transpose, self.parameters.auxiliary_ring),
        ):
            if (
                not isinstance(key, KeySwitchKey)
                or key.kind != "big"
                or key.parameters.ring != ring
            ):
                raise ValueError("transpose refresh keys must match their extension rings")
            if key.device != self.base.hadamard.device:
                raise ValueError("all refresh keys must have the same device")


def generate_bootstrap_keys(
    parameters: BootstrapParameters, secret, *, device="cpu", generator=None, noise_std=3.2
):
    """Generate refresh keys once. The returned material contains no secret."""
    if not isinstance(parameters, BootstrapParameters):
        raise TypeError("parameters must be BootstrapParameters")
    device = resolve_device(device)
    full_switcher = KeySwitcher(
        parameters.ring, k=parameters.k, device=device, generator=generator, noise_std=noise_std
    )
    small = full_switcher._small_secret(secret)
    full_secret = embed(small, parameters.ring)
    x_key = full_switcher.generate_key(swap_axes(full_secret, "X", "Y"), small, kind="big")
    auxiliary = parameters.auxiliary_ring
    aux_switcher = KeySwitcher(
        auxiliary, k=parameters.k, device=device, generator=generator, noise_std=noise_std
    )
    w_key = aux_switcher.generate_key(
        swap_axes(embed(small, auxiliary), "W", "V"), small, kind="big"
    )
    base_switcher = KeySwitcher(
        parameters.base_ring,
        k=parameters.k,
        device=device,
        generator=generator,
        noise_std=noise_std,
    )
    return BootstrapKeys(
        parameters, generate_evaluation_keys(base_switcher, small, matrices=False), x_key, w_key
    )


class Bootstrapper:
    """Refresh one or more GL ciphertexts using only public evaluation material.

    Section 4's unoptimized four-switch CtS/StC is used. Ordinary transpose
    and its correspondingly interpolated matrices give the same trace maps
    as the paper's Hermitian convention. X and W each consume one level.
    The optional stage callback receives ciphertexts for external diagnostics;
    it is never used to compute the result.
    """

    def __init__(self, keys: BootstrapKeys, *, device="cpu"):
        if not isinstance(keys, BootstrapKeys):
            raise TypeError("keys must be BootstrapKeys")
        self.keys, self.parameters, self.device = keys, keys.parameters, resolve_device(device)
        if keys.x_transpose.device != self.device or keys.w_transpose.device != self.device:
            raise ValueError("refresh keys and bootstrapper must have the same device")
        self.evaluator = Evaluator(
            self.parameters.base_ring, keys.base, device=self.device, rounding="nearest"
        )
        self._switchers, self._kernels = {}, {}
        self._fourier = {}
        for name, axis in (
            ("X", self.parameters.ring.axes[0]),
            ("W", self.parameters.ring.axes[2]),
        ):
            order = axis.cyclotomic_order or 4 * axis.degree
            generator = 2 if axis.cyclotomic_order else 5
            exponents = torch.tensor(
                [pow(generator, j, order) for j in range(axis.degree)],
                dtype=torch.int64,
                device=self.device,
            )
            phases = (
                exponents[:, None] * torch.arange(axis.degree, device=self.device)[None, :]
            ).remainder(order)
            matrix = torch.exp((2j * math.pi / order) * phases.to(torch.float64))
            # U_W^* U_W is not p*I for the compact cyclotomic power basis.
            self._fourier[name] = (matrix, torch.linalg.inv(matrix))

    def _switcher(self, ring, key):
        cache_key = ring, key.parameters
        if cache_key not in self._switchers:
            self._switchers[cache_key] = KeySwitcher(
                ring, k=key.parameters.k, key_moduli=key.parameters.key_moduli, device=self.device
            )
        return self._switchers[cache_key]

    def _transpose(self, ciphertext, first, second, key):
        b, a = (swap_axes(component, first, second) for component in (ciphertext.b, ciphertext.a))
        switched = self._switcher(ciphertext.ring, key).switch_big(a, key)
        return Ciphertext(b + switched.b, switched.a)

    def _kernel(self, ring, axis, inverse):
        cache_key = ring, axis, inverse
        if cache_key not in self._kernels:
            matrix = self._fourier[axis][int(inverse)].T
            encoder = MatrixEncoder(ring, scale=ring.moduli[-1], device=self.device)
            if axis == "X":
                slots = matrix.unsqueeze(0).expand(encoder.slot_shape)
            else:
                slots = matrix.unsqueeze(1).expand(encoder.slot_shape)
            self._kernels[cache_key] = encoder.encode(slots)
        return self._kernels[cache_key]

    def _axis_transform(self, values, axis, *, inverse):
        """Pack R -> R'/Raux, transpose, trace, transpose back, split R."""
        if not values:
            return ()
        for value in values:
            self.evaluator._check(value)
        level = values[0].level
        if level < 2 or any(v.level != level or v.scale != values[0].scale for v in values):
            raise ValueError("axis transform requires equal levels/scales and at least two primes")
        if axis == "X":
            top, extension, first, key = self.parameters.ring, "Y", "X", self.keys.x_transpose
            product = matrix_multiply
        elif axis == "W":
            top, extension, first, key = (
                self.parameters.auxiliary_ring,
                "V",
                "W",
                self.keys.w_transpose,
            )
            product = auxiliary_matrix_multiply
        else:
            raise ValueError("transform axis must be X or W")
        ring = RNSRing(top.moduli[:level], top.axes)
        width = next(a.degree for a in ring.axes if a.name == extension)
        kernel = self._kernel(ring, axis, inverse)
        results = []
        for start in range(0, len(values), width):
            group = values[start : start + width]
            packed = merge_ciphertexts((v.ciphertext for v in group), ring, extension)
            transposed = self._transpose(packed, first, extension, key)
            traced = Ciphertext(product(transposed.b, kernel), product(transposed.a, kernel))
            # Kernel scale is q_last; this rescale preserves the input scale.
            traced = Ciphertext(
                traced.b.rescale(rounding="nearest"), traced.a.rescale(rounding="nearest")
            )
            restored = self._transpose(traced, first, extension, key)
            results.extend(
                ScaledCiphertext(ct, group[0].scale)
                for ct in split_ciphertext(restored, extension)[: len(group)]
            )
        return tuple(results)

    def coeff_to_slot(self, values):
        """Inverse X then W DFT on R ciphertexts via extension-ring traces."""
        return self._axis_transform(
            self._axis_transform(tuple(values), "X", inverse=True), "W", inverse=True
        )

    def slot_to_coeff(self, values):
        """W then X DFT, inverse to coeff_to_slot up to encoding/rounding error."""
        return self._axis_transform(
            self._axis_transform(tuple(values), "W", inverse=False), "X", inverse=False
        )

    @staticmethod
    def _integer_scale(value, constant):
        ct = value.ciphertext
        return ScaledCiphertext(Ciphertext(ct.b * constant, ct.a * constant), value.scale)

    def _polynomial(self, value, coefficients):
        """Balanced powers, then one common-scale coefficient layer.

        Every product is relinearized with the same maximum-level small key.
        Unequal power scales are normalized by the encoded coefficients, not
        by changing metadata or consuming an extra level per addition.
        """
        nonzero = {degree: c for degree, c in enumerate(coefficients) if c and degree}
        if not nonzero:
            raise ValueError("EvalMod polynomial needs a nonconstant term")
        powers = {1: value}

        def power(degree):
            if degree not in powers:
                half = degree // 2
                powers[degree] = self.evaluator.multiply(power(half), power(degree - half))
            return powers[degree]

        for degree in nonzero:
            power(degree)
        common_level = min(powers[degree].level for degree in nonzero)
        result = None
        for degree, coefficient in nonzero.items():
            term = powers[degree].at_level(common_level)
            plaintext_scale = term.ring.moduli[-1] * value.scale / term.scale
            term = self.evaluator.multiply_plain(term, coefficient, scale=plaintext_scale)
            term = ScaledCiphertext(term.ciphertext, value.scale)
            result = term if result is None else self.evaluator.add(result, term)
        return self.evaluator.add_plain(result, coefficients[0]) if coefficients[0] else result

    def _eval_mod_real(self, value):
        """Approximate x -> x-round(x) on the declared union of intervals."""
        value = self.evaluator.multiply_plain(value, 2 * math.pi / 2**self.parameters.double_angle)
        sine = [0.0] * 10
        cosine = [0.0] * 11
        for j in range(5):
            sine[2 * j + 1] = (-1) ** j / math.factorial(2 * j + 1)
        for j in range(6):
            cosine[2 * j] = (-1) ** j / math.factorial(2 * j)
        sine_ct, cosine_ct = self._polynomial(value, sine), self._polynomial(value, cosine)
        for _ in range(self.parameters.double_angle):
            next_sine = self._integer_scale(self.evaluator.multiply(sine_ct, cosine_ct), 2)
            next_cosine = self.evaluator.add_plain(
                self._integer_scale(self.evaluator.square(cosine_ct), 2), -1
            )
            sine_ct, cosine_ct = next_sine, next_cosine
        if self.parameters.arcsine_degree > 1:
            coefficients = [0.0] * (self.parameters.arcsine_degree + 1)
            for j in range((self.parameters.arcsine_degree + 1) // 2):
                coefficients[2 * j + 1] = math.comb(2 * j, j) / (4**j * (2 * j + 1))
            sine_ct = self._polynomial(sine_ct, coefficients)
        return self.evaluator.multiply_plain(sine_ct, 1 / (2 * math.pi))

    def eval_mod(self, value):
        """Reduce real and imaginary coefficients independently, in R slots."""
        conjugated = self.evaluator.conjugate(value)
        real = self.evaluator.multiply_plain(self.evaluator.add(value, conjugated), 0.5)
        imag = self.evaluator.multiply_plain(self.evaluator.subtract(value, conjugated), -0.5j)
        real, imag = self._eval_mod_real(real), self._eval_mod_real(imag)
        return self.evaluator.add(real, self._integer_scale(imag, (0, 1)))

    def bootstrap(self, value: ScaledCiphertext, *, on_stage=None) -> ScaledCiphertext:
        return self.bootstrap_many((value,), on_stage=on_stage)[0]

    def bootstrap_many(self, values, *, on_stage=None):
        """Batch full R' inputs, repacking groups to phi(p) for the W transform."""
        values = tuple(values)
        if not values:
            raise ValueError("bootstrapping requires at least one ciphertext")
        top = self.parameters.ring
        for value in values:
            if not isinstance(value, ScaledCiphertext):
                raise TypeError("bootstrapping requires ScaledCiphertext inputs")
            if (
                value.ring.axes != top.axes
                or value.ring.moduli != top.moduli[:1]
                or value.device != self.device
            ):
                raise ValueError(
                    "input must be in R' at the single bottom prime and matching device"
                )
        if any(value.scale != values[0].scale for value in values):
            raise ValueError("batched bootstrapping inputs must have matching scales")
        q0 = top.moduli[0]
        small_top = self.parameters.base_ring
        parts = tuple(
            ScaledCiphertext(Ciphertext(ct.b.mod_raise(small_top), ct.a.mod_raise(small_top)), q0)
            for value in values
            for ct in split_ciphertext(value.ciphertext)
        )
        if on_stage is not None:
            on_stage("ModRaise", parts)
        parts = self.coeff_to_slot(parts)
        if on_stage is not None:
            on_stage("CtS", parts)
        parts = tuple(self.eval_mod(part) for part in parts)
        if on_stage is not None:
            on_stage("EvalMod", parts)
        parts = self.slot_to_coeff(parts)
        if on_stage is not None:
            on_stage("StC", parts)
        output_ring = RNSRing(top.moduli[: parts[0].level], top.axes)
        n = top.axes[0].degree
        result = []
        for start in range(0, len(parts), n):
            group = parts[start : start + n]
            ct = merge_ciphertexts((part.ciphertext for part in group), output_ring)
            result.append(ScaledCiphertext(ct, values[0].scale * group[0].scale / q0))
        return tuple(result)
