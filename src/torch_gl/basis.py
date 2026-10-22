"""Han–Ki fast RNS conversion: CRT sums without large-integer composition."""

import math

from ._bounded import montgomery_linear_form
from .modular import mul_mod


class RNSBasisConverter:
    """Approximate Conv_C->B from Han–Ki §2.2, page 4.

    For source product M, lift as sum(centered(x_i*hat_M_i^-1) * hat_M_i).
    This is x+v*M, not the canonical CRT representative. Original source
    residues are preserved. Constants use Python ints; tensor data stays int32.
    """

    def __init__(self, source_moduli, target_moduli):
        self.source_moduli, self.target_moduli = tuple(source_moduli), tuple(target_moduli)
        if not self.source_moduli or len(set(self.source_moduli)) != len(self.source_moduli):
            raise ValueError("source moduli must be nonempty and distinct")
        self.modulus = math.prod(self.source_moduli)
        self._partial_products = tuple(self.modulus // modulus for modulus in self.source_moduli)
        self._inverses = tuple(
            pow(partial, -1, modulus)
            for partial, modulus in zip(self._partial_products, self.source_moduli, strict=True)
        )
        self._weights = {
            target: tuple(partial % target for partial in self._partial_products)
            for target in self.target_moduli
        }

    def prepare(self, source_values, *, inverses=None):
        """Compute the small CRT summands once, reusable across target primes."""
        if len(source_values) != len(self.source_moduli):
            raise ValueError("expected one tensor per source modulus")
        constants = self._inverses if inverses is None else inverses
        result = []
        for value, inverse, modulus in zip(
            source_values, constants, self.source_moduli, strict=True
        ):
            summand = mul_mod(value, inverse, modulus)
            result.append(summand.add_(summand > modulus // 2, alpha=-modulus))
        return tuple(result)

    def convert_prepared(self, prepared, target, *, source_values=None):
        if target not in self._weights:
            raise ValueError("target modulus is not in this converter")
        if source_values is not None and target in self.source_moduli:
            return source_values[self.source_moduli.index(target)]
        form = montgomery_linear_form(
            target, tuple(q // 2 for q in self.source_moduli), self._weights[target]
        )
        return form(prepared)

    def subtract_and_scale(self, base, prepared, target, scale, *, multiplier=1):
        """Fused (base - multiplier*Conv(prepared))*scale, preserving the lift."""
        if target not in self._weights:
            raise ValueError("target modulus is not in this converter")
        weights = (scale,) + tuple(-multiplier * w * scale for w in self._weights[target])
        bounds = (target - 1,) + tuple(q // 2 for q in self.source_moduli)
        return montgomery_linear_form(target, bounds, weights)((base,) + tuple(prepared))

    def convert(self, source_values):
        prepared = self.prepare(source_values)
        return tuple(
            self.convert_prepared(prepared, target, source_values=source_values)
            for target in self.target_moduli
        )
