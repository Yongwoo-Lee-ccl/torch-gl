"""Run with: python examples/multivariate_poly.py"""

import torch

from torch_gl import MultivariatePoly, PolynomialRing

ring = PolynomialRing.multivariate_poly(q=433, n=4, p=27)
coefficients = torch.arange(4 * 4 * 18 * 2, dtype=torch.int32).reshape(ring.shape)
polynomial = MultivariatePoly(ring, coefficients)
product = polynomial * polynomial
print(f"Coefficient shape: {ring.shape}; transform shape: {ring.ntt_shape}")
print(f"Product shape: {tuple(product.coefficients.shape)}")

X = ring.generator("X")
W = ring.generator("W")
assert X**4 == ring.constant((0, 1))
assert W**18 + W**9 + 1 == ring.zero()

small_ring = PolynomialRing.multivariate_poly(q=433, n=4, p=27, variables=("X", "W"))
print(f"Small-ring coefficient shape: {small_ring.shape}")
