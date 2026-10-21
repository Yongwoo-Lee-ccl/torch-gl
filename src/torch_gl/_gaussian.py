"""Gaussian representation boundaries with public folded Montgomery constants."""

from functools import lru_cache

import torch

from ._compilation import maybe_gaussian
from .modular import arithmetic_context


class GaussianBoundary:
    def __init__(self, q, imaginary_root):
        self.q = q
        self.arithmetic = arithmetic_context(q)
        context = self.arithmetic.context
        self.forward_weights = (context.radix_squared, imaginary_root * context.radix_squared % q)
        self.inverse_weights = (pow(2, -1, q), pow(2 * imaginary_root, -1, q))
        self.imaginary = self.arithmetic.constant(imaginary_root)
        self.legacy_inverse = tuple(self.arithmetic.constant(w) for w in self.inverse_weights)

    def split(self, coefficients):
        arithmetic = self.arithmetic
        compiled = maybe_gaussian(
            coefficients,
            self.forward_weights,
            self.q,
            arithmetic.context.negative_inverse,
            inverse=False,
        )
        if compiled is not None:
            return compiled
        if not coefficients.is_contiguous():
            # Preserve the measured eager layout path: a full-pair encode can
            # outperform two strided real/imag passes, despite an extra product.
            real, imag = arithmetic.encode(coefficients).unbind(-1)
            imaginary = arithmetic.multiply(imag, self.imaginary)
        else:
            real, imag = coefficients.unbind(-1)
            real = arithmetic.multiply(real, self.forward_weights[0])
            imaginary = arithmetic.multiply(imag, self.forward_weights[1])
        return torch.stack(
            (arithmetic.add(real, imaginary), arithmetic.subtract(real, imaginary)), -1
        )

    def join(self, values):
        arithmetic = self.arithmetic
        compiled = maybe_gaussian(
            values, self.inverse_weights, self.q, arithmetic.context.negative_inverse, inverse=True
        )
        if compiled is not None:
            return compiled
        positive, negative = values.unbind(-1)
        weights = self.inverse_weights if values.is_contiguous() else self.legacy_inverse
        real = arithmetic.multiply(arithmetic.add(positive, negative), weights[0])
        imag = arithmetic.multiply(arithmetic.subtract(positive, negative), weights[1])
        result = torch.stack((real, imag), -1)
        return result if values.is_contiguous() else arithmetic.decode(result)


@lru_cache(maxsize=128)
def gaussian_boundary(q, imaginary_root):
    return GaussianBoundary(q, imaginary_root)
