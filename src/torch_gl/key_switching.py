"""Gentry–Lee small/big switching with Han–Ki grouped RNS decomposition."""

import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import torch

from ._native_transform import NativeTransform
from ._number_theory import MAX_MODULUS, is_prime, odd_prime_power, supports_native_cyclotomic
from ._transform_tables import resolve_device
from .basis import RNSBasisConverter
from .ciphertext import Ciphertext, MatrixProductCiphertext, QuadraticCiphertext
from .matrix import matrix_transpose
from .modular import mul_mod, sub_mod
from .multivariate_poly import MultivariatePoly
from .rns import RNSPoly, RNSRing


def _special_primes(ring, count, plaintext_modulus=None):
    orders = [
        axis.cyclotomic_order if axis.cyclotomic_order else 4 * axis.degree for axis in ring.axes
    ]
    orders += [
        axis.cyclotomic_order - 1
        for axis in ring.axes
        if axis.cyclotomic_order
        and odd_prime_power(axis.cyclotomic_order)[0] != 3
        and supports_native_cyclotomic(axis.cyclotomic_order)
    ]
    step = math.lcm(*orders, plaintext_modulus or 1)
    candidate = (max(ring.moduli) // step + 1) * step + 1
    result = []
    while len(result) < count and candidate <= MAX_MODULUS:
        if is_prime(candidate):
            result.append(candidate)
        candidate += step
    if len(result) != count:
        raise ValueError("not enough larger NTT key primes; supply key_moduli explicitly")
    return tuple(result)


@dataclass(frozen=True)
class KeySwitchParameters:
    """k key primes and k ciphertext primes per digit; dnum=ceil(limb_count/k).

    The final digit may have fewer than k primes. Keys generated here can also
    be used at every prefix of this modulus chain, including a partial digit.
    """

    ring: RNSRing
    k: int = 4
    key_moduli: tuple[int, ...] | None = None
    plaintext_modulus: int | None = None

    def __post_init__(self):
        if not isinstance(self.ring, RNSRing):
            raise TypeError("ring must be an RNSRing")
        if type(self.k) is not int or self.k < 1:
            raise ValueError("k must be a positive integer")
        if not {"X", "W"}.issubset(axis.name for axis in self.ring.axes):
            raise ValueError("key switching requires a Gentry–Lee X/W or X/Y/W ring")
        t = self.plaintext_modulus
        if t is not None and (
            type(t) is not int or t < 2 or any(q % t != 1 for q in self.ring.moduli)
        ):
            raise ValueError("integer key switching needs t >= 2 and ciphertext primes 1 modulo t")
        moduli = (
            _special_primes(self.ring, self.k, t)
            if self.key_moduli is None
            else tuple(self.key_moduli)
        )
        if len(moduli) != self.k or len(set(moduli)) != self.k:
            raise ValueError("key_moduli must contain exactly k distinct primes")
        if set(moduli) & set(self.ring.moduli):
            raise ValueError("key primes and ciphertext primes must be disjoint")
        if t is not None and any(q % t != 1 for q in moduli):
            raise ValueError("integer key primes must be 1 modulo the plaintext modulus")
        object.__setattr__(self, "key_moduli", moduli)
        _ = self.extended_ring  # Validate all prime sizes and NTT requirements.
        if self.key_modulus < max(self.digit_moduli):
            raise ValueError("product of key primes must be at least every digit modulus")

    @cached_property
    def small_ring(self):
        return RNSRing(
            self.ring.moduli, tuple(axis for axis in self.ring.axes if axis.name not in {"Y", "V"})
        )

    @cached_property
    def extended_ring(self):
        # P first lets the switch stream its ModDown inputs before Q outputs.
        return RNSRing(self.key_moduli + self.ring.moduli, self.ring.axes)

    @cached_property
    def extended_small_ring(self):
        return RNSRing(self.extended_ring.moduli, self.small_ring.axes)

    @cached_property
    def groups(self):
        return tuple(
            tuple(range(start, min(start + self.k, self.ring.limb_count)))
            for start in range(0, self.ring.limb_count, self.k)
        )

    @property
    def dnum(self):
        return len(self.groups)

    @cached_property
    def digit_moduli(self):
        return tuple(math.prod(self.ring.moduli[index] for index in group) for group in self.groups)

    @cached_property
    def key_modulus(self):
        return math.prod(self.key_moduli)


@dataclass(frozen=True)
class KeySwitchKey:
    """Public evaluation key in int32 Montgomery NTT form, with no secret fields.

    Optional file backing stores one extended-basis limb per Torch file, mapped
    read-only by convention on CPU. Both storage modes execute the same algorithm.
    """

    parameters: KeySwitchParameters
    kind: str
    device: torch.device
    _limbs: tuple = field(default=(), repr=False, compare=False)
    _paths: tuple = field(default=(), repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.parameters, KeySwitchParameters) or self.kind not in (
            "small",
            "big",
        ):
            raise ValueError("invalid key-switch parameters or kind")
        expected = self.parameters.ring.limb_count + self.parameters.k
        if bool(self._limbs) == bool(self._paths):
            raise ValueError("provide exactly one evaluation-key storage mode")
        if len(self._paths if self._paths else self._limbs) != expected:
            raise ValueError("expected one evaluation-key entry per extended-basis limb")
        for digit_keys in self._limbs:
            if len(digit_keys) != self.parameters.dnum:
                raise ValueError("evaluation-key digit count does not match dnum")

    def _load(self, index):
        if self._paths:
            return torch.load(
                self._paths[index], map_location=self.device, weights_only=True, mmap=True
            )
        return self._limbs[index]


@dataclass(frozen=True)
class MatrixKeySwitchKeys:
    """The two big keys: s(Y,W)->s(X,W) and s(X,W)s(Y,W)->s(X,W)."""

    Y_key: KeySwitchKey
    mixed_key: KeySwitchKey

    def __post_init__(self):
        if not all(isinstance(key, KeySwitchKey) for key in (self.Y_key, self.mixed_key)):
            raise TypeError("matrix relinearization requires two KeySwitchKey values")
        if self.Y_key.kind != "big" or self.mixed_key.kind != "big":
            raise ValueError("matrix relinearization requires two big switching keys")
        if (
            self.Y_key.parameters != self.mixed_key.parameters
            or self.Y_key.device != self.mixed_key.device
        ):
            raise ValueError("matrix switching keys must have matching parameters and device")


class KeySwitcher:
    """RNS key switching with evaluation-key reuse at shorter prefix levels."""

    def __init__(
        self,
        ring: RNSRing,
        *,
        k: int = 4,
        key_moduli=None,
        noise_std: float = 3.2,
        device="cpu",
        generator=None,
        plaintext_modulus=None,
    ):
        self.parameters = KeySwitchParameters(ring, k, key_moduli, plaintext_modulus)
        if isinstance(noise_std, bool) or not isinstance(noise_std, (int, float)):
            raise ValueError("noise_std must be finite and nonnegative")
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        self.device = resolve_device(device)
        if generator is not None and resolve_device(generator.device) != self.device:
            raise ValueError("generator and key switcher must have the same device")
        self.noise_std, self.generator = float(noise_std), generator
        names = tuple(axis.name for axis in ring.axes)
        extension = "Y" if "Y" in names else "V" if "V" in names else None
        self._Y_dim = names.index(extension) if extension else None
        self._converters, self._digit_inverses = [], []
        for group, modulus in zip(
            self.parameters.groups, self.parameters.digit_moduli, strict=True
        ):
            source_moduli = tuple(ring.moduli[index] for index in group)
            converter = RNSBasisConverter(source_moduli, self.parameters.extended_ring.moduli)
            self._converters.append(converter)
            cofactor = ring.modulus // modulus
            # Fuse Han–Ki's decomposition inverse and Conv's inverse constants.
            self._digit_inverses.append(
                tuple(
                    pow(cofactor, -1, prime) * pow(modulus // prime, -1, prime) % prime
                    for prime in source_moduli
                )
            )
        self._down_converter = RNSBasisConverter(self.parameters.key_moduli, ring.moduli)
        self._transforms = {}
        # Reject unsupported/padded ideals now, before sampling evaluation keys.
        NativeTransform(ring.rings[0], self.device)

    @property
    def k(self):
        return self.parameters.k

    @property
    def dnum(self):
        return self.parameters.dnum

    def _check(self, polynomial, ring):
        if not isinstance(polynomial, RNSPoly):
            raise TypeError("expected an RNSPoly")
        if polynomial.ring != ring or polynomial.device != self.device:
            raise ValueError("polynomial has the wrong ring or device")

    def _small_secret(self, secret):
        if not isinstance(secret, RNSPoly):
            raise TypeError("secret must be an RNSPoly")
        if secret.ring == self.parameters.small_ring:
            self._check(secret, self.parameters.small_ring)
            return secret
        self._check(secret, self.parameters.ring)
        if self._Y_dim is None:
            return secret
        for limb in secret.limbs:
            if bool(
                limb._coefficients.narrow(self._Y_dim, 1, limb.ring.shape[self._Y_dim] - 1).any()
            ):
                raise ValueError("small switching requires a source/destination constant in Y")
        return RNSPoly(
            self.parameters.small_ring,
            tuple(
                MultivariatePoly._from_owned(
                    ring, limb._coefficients.select(self._Y_dim, 0).clone()
                )
                for ring, limb in zip(self.parameters.small_ring.rings, secret.limbs, strict=True)
            ),
        )

    def _embed(self, small):
        self._check(small, self.parameters.small_ring)
        if self._Y_dim is None:
            return small
        limbs = []
        for ring, limb in zip(self.parameters.ring.rings, small.limbs, strict=True):
            values = torch.zeros(ring.shape, dtype=torch.int32, device=self.device)
            values.select(self._Y_dim, 0).copy_(limb._coefficients)
            limbs.append(MultivariatePoly._from_owned(ring, values))
        return RNSPoly(self.parameters.ring, tuple(limbs))

    def _transform(self, kind, index, *, batch=False):
        cache_key = (kind, index, batch)
        if cache_key not in self._transforms:
            ring = (
                self.parameters.extended_small_ring
                if kind == "small"
                else self.parameters.extended_ring
            )
            self._transforms[cache_key] = NativeTransform(
                ring.rings[index], self.device, batch_ndim=int(batch and self._Y_dim is not None)
            )
        return self._transforms[cache_key]

    def generate_key(
        self, source_secret, destination_secret, *, kind="small", storage_directory=None
    ):
        """Generate b_j+a_j*s_target=P*hat_Q_j*s_source+e_j modulo P*Q.

        Secrets must admit a consistent small int32 coefficient lift. This
        includes binary keys, their squares, s(Y), and s(X)*s(Y).
        """
        if kind not in ("small", "big"):
            raise ValueError("kind must be 'small' or 'big'")
        source = self._small_secret(source_secret) if kind == "small" else source_secret
        key_ring = (
            self.parameters.extended_small_ring
            if kind == "small"
            else self.parameters.extended_ring
        )
        self._check(source, self.parameters.small_ring if kind == "small" else self.parameters.ring)
        destination = self._small_secret(destination_secret)
        source_values = source.centered_coefficients()
        destination_values = destination.centered_coefficients()
        noises = tuple(
            torch.empty(key_ring.shape, dtype=torch.float32, device=self.device)
            .normal_(0, self.noise_std, generator=self.generator)
            .round_()
            .to(torch.int64 if self.parameters.plaintext_modulus else torch.int32)
            * (self.parameters.plaintext_modulus or 1)
            for _ in self.parameters.groups
        )
        directory = Path(storage_directory) if storage_directory is not None else None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            if any(
                (directory / f"limb_{index}.pt").exists() for index in range(len(key_ring.moduli))
            ):
                raise FileExistsError("evaluation-key files already exist")
        limbs, paths = [], []
        for index, prime in enumerate(key_ring.moduli):
            transform = self._transform(kind, index)
            arithmetic = transform.arithmetic
            destination_slots = self._transform("small", index).forward(
                destination_values.remainder(prime)
            )
            if kind == "big" and self._Y_dim is not None:
                destination_slots = destination_slots.unsqueeze(self._Y_dim)
            source_slots = (
                transform.forward(source_values.remainder(prime))
                if prime in self.parameters.ring.moduli
                else None
            )
            digit_keys = []
            for noise, modulus in zip(noises, self.parameters.digit_moduli, strict=True):
                uniform_slots = arithmetic.encode(
                    torch.randint(
                        prime,
                        key_ring.shape,
                        dtype=torch.int32,
                        device=self.device,
                        generator=self.generator,
                    )
                )
                noise_slots = transform.forward(noise.remainder(prime).to(torch.int32))
                masked_slots = arithmetic.subtract(
                    noise_slots, arithmetic.multiply(uniform_slots, destination_slots)
                )
                factor = (
                    self.parameters.key_modulus * (self.parameters.ring.modulus // modulus)
                ) % prime
                if factor:
                    masked_slots = arithmetic.add(
                        masked_slots, arithmetic.multiply(source_slots, arithmetic.constant(factor))
                    )
                digit_keys.append((masked_slots, uniform_slots))
            if directory is None:
                limbs.append(tuple(digit_keys))
            else:
                path = directory / f"limb_{index}.pt"
                torch.save(tuple(digit_keys), path)
                paths.append(path)
        return KeySwitchKey(self.parameters, kind, self.device, tuple(limbs), tuple(paths))

    def generate_hadamard_key(self, secret, *, storage_directory=None):
        small = self._small_secret(secret)
        return self.generate_key(
            small * small, small, kind="small", storage_directory=storage_directory
        )

    def generate_matrix_keys(self, secret, *, storage_directory=None):
        small = self._small_secret(secret)
        full = self._embed(small)
        secret_Y = matrix_transpose(full)
        mixed = full * secret_Y
        directory = Path(storage_directory) if storage_directory is not None else None
        Y_key = self.generate_key(
            secret_Y, small, kind="big", storage_directory=directory / "Y" if directory else None
        )
        mixed_key = self.generate_key(
            mixed, small, kind="big", storage_directory=directory / "mixed" if directory else None
        )
        return MatrixKeySwitchKeys(Y_key, mixed_key)

    def _check_key(self, key, kind):
        if not isinstance(key, KeySwitchKey):
            raise TypeError("expected a KeySwitchKey")
        top = key.parameters
        if (
            top.ring.axes != self.parameters.ring.axes
            or top.ring.moduli[: self.parameters.ring.limb_count] != self.parameters.ring.moduli
            or top.k != self.k
            or top.key_moduli != self.parameters.key_moduli
            or top.plaintext_modulus != self.parameters.plaintext_modulus
            or key.device != self.device
            or key.kind != kind
        ):
            raise ValueError("switching key has the wrong parameters, device, or kind")

    def _apply(self, component, key, kind):
        self._check(component, self.parameters.ring)
        self._check_key(key, kind)
        # A maximum-level key encrypts P*(Q_top/B_top_j)*s_source. Retain
        # that cofactor in the decomposition inverse at a lower level. Merely
        # truncating a key and using Q_level/B_level_j silently changes the
        # plaintext, particularly when the last active digit is incomplete.
        digit_inverses = self._digit_inverses
        if key.parameters != self.parameters:
            digit_inverses = []
            for digit, (group, modulus) in enumerate(
                zip(self.parameters.groups, self.parameters.digit_moduli, strict=True)
            ):
                cofactor = key.parameters.ring.modulus // key.parameters.digit_moduli[digit]
                digit_inverses.append(
                    tuple(
                        pow(cofactor, -1, prime) * pow(modulus // prime, -1, prime) % prime
                        for prime in (self.parameters.ring.moduli[index] for index in group)
                    )
                )
        prepared_digits = tuple(
            converter.prepare(
                tuple(component.limbs[index]._coefficients for index in group), inverses=inverses
            )
            for group, inverses, converter in zip(
                self.parameters.groups, digit_inverses, self._converters, strict=True
            )
        )
        temporary_b, temporary_a, result_b, result_a = [], [], [], []
        prepared_b = prepared_a = None
        for index, prime in enumerate(self.parameters.extended_ring.moduli):
            transform = self._transform(kind, index, batch=kind == "small")
            arithmetic = transform.arithmetic
            digit_keys = key._load(index)[: self.dnum]
            accumulator_b = accumulator_a = None
            for prepared, converter, (key_b, key_a) in zip(
                prepared_digits, self._converters, digit_keys, strict=True
            ):
                coefficients = converter.convert_prepared(prepared, prime)
                if kind == "small" and self._Y_dim is not None:
                    coefficients = coefficients.movedim(self._Y_dim, 0)
                slots = transform.forward(coefficients)
                term_b, term_a = (
                    arithmetic.multiply(slots, key_b),
                    arithmetic.multiply(slots, key_a),
                )
                accumulator_b = (
                    term_b if accumulator_b is None else arithmetic.add(accumulator_b, term_b)
                )
                accumulator_a = (
                    term_a if accumulator_a is None else arithmetic.add(accumulator_a, term_a)
                )
            coefficients_b, coefficients_a = (
                transform.inverse(accumulator_b),
                transform.inverse(accumulator_a),
            )
            if kind == "small" and self._Y_dim is not None:
                coefficients_b = coefficients_b.movedim(0, self._Y_dim).contiguous()
                coefficients_a = coefficients_a.movedim(0, self._Y_dim).contiguous()
            if index < self.k:
                temporary_b.append(coefficients_b)
                temporary_a.append(coefficients_a)
                if index == self.k - 1:
                    if self.parameters.plaintext_modulus is not None:
                        # Integer ModDown subtracts t times a lift of c/t in P,
                        # so all rounding error remains a multiple of t.
                        t = self.parameters.plaintext_modulus
                        temporary_b = [
                            mul_mod(v, pow(t, -1, p), p)
                            for v, p in zip(temporary_b, self.parameters.key_moduli, strict=True)
                        ]
                        temporary_a = [
                            mul_mod(v, pow(t, -1, p), p)
                            for v, p in zip(temporary_a, self.parameters.key_moduli, strict=True)
                        ]
                    prepared_b = self._down_converter.prepare(temporary_b)
                    prepared_a = self._down_converter.prepare(temporary_a)
                    temporary_b.clear()
                    temporary_a.clear()
            else:
                # Han–Ki ModDown: (Q-part - Conv_P->Q(P-part)) / P modulo q.
                converted_b = self._down_converter.convert_prepared(prepared_b, prime)
                converted_a = self._down_converter.convert_prepared(prepared_a, prime)
                if self.parameters.plaintext_modulus is not None:
                    converted_b = mul_mod(converted_b, self.parameters.plaintext_modulus, prime)
                    converted_a = mul_mod(converted_a, self.parameters.plaintext_modulus, prime)
                inverse_P = pow(self.parameters.key_modulus, -1, prime)
                values_b = mul_mod(sub_mod(coefficients_b, converted_b, prime), inverse_P, prime)
                values_a = mul_mod(sub_mod(coefficients_a, converted_a, prime), inverse_P, prime)
                ring = self.parameters.ring.rings[index - self.k]
                result_b.append(MultivariatePoly._from_owned(ring, values_b))
                result_a.append(MultivariatePoly._from_owned(ring, values_a))
        return Ciphertext(
            RNSPoly(self.parameters.ring, tuple(result_b)),
            RNSPoly(self.parameters.ring, tuple(result_a)),
        )

    def switch_small(self, component, key):
        """Return a two-component encryption of component*s_source in R'."""
        return self._apply(component, key, "small")

    def switch_big(self, component, key):
        return self._apply(component, key, "big")

    def switch_ciphertext(self, ciphertext, key):
        if not isinstance(ciphertext, Ciphertext):
            raise TypeError("expected a Ciphertext")
        self._check(ciphertext.b, self.parameters.ring)
        switched = self.switch_small(ciphertext.a, key)
        return Ciphertext(ciphertext.b + switched.b, switched.a)

    def relinearize(self, ciphertext, key):
        if not isinstance(ciphertext, QuadraticCiphertext):
            raise TypeError("expected a QuadraticCiphertext")
        self._check(ciphertext.c0, self.parameters.ring)
        self._check(ciphertext.c1, self.parameters.ring)
        switched = self.switch_small(ciphertext.c2, key)
        return Ciphertext(ciphertext.c0 + switched.b, ciphertext.c1 + switched.a)

    def relinearize_matrix(self, ciphertext, keys):
        if not isinstance(ciphertext, MatrixProductCiphertext) or not isinstance(
            keys, MatrixKeySwitchKeys
        ):
            raise TypeError("expected a MatrixProductCiphertext and MatrixKeySwitchKeys")
        self._check(ciphertext.c00, self.parameters.ring)
        self._check(ciphertext.c10, self.parameters.ring)
        switched_Y = self.switch_big(ciphertext.c01, keys.Y_key)
        result_b, result_a = ciphertext.c00 + switched_Y.b, ciphertext.c10 + switched_Y.a
        del switched_Y
        switched_mixed = self.switch_big(ciphertext.c11, keys.mixed_key)
        return Ciphertext(result_b + switched_mixed.b, result_a + switched_mixed.a)
