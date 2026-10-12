"""Han–Ki fast RNS conversion: CRT sums without large-integer composition."""

import math

import torch

from .modular import add_mod, mul_mod


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
            result.append(torch.where(summand > modulus // 2, summand - modulus, summand))
        return tuple(result)

    def convert_prepared(self, prepared, target, *, source_values=None):
        if target not in self._weights:
            raise ValueError("target modulus is not in this converter")
        if source_values is not None and target in self.source_moduli:
            return source_values[self.source_moduli.index(target)]
        if target in self.source_moduli:
            index = self.source_moduli.index(target)
            return mul_mod(
                prepared[index].remainder(target), self._partial_products[index] % target, target
            )
        result = torch.zeros_like(prepared[0])
        for summand, weight in zip(prepared, self._weights[target], strict=True):
            result = add_mod(result, mul_mod(summand.remainder(target), weight, target), target)
        return result

    def convert(self, source_values):
        prepared = self.prepare(source_values)
        return tuple(
            self.convert_prepared(prepared, target, source_values=source_values)
            for target in self.target_moduli
        )
