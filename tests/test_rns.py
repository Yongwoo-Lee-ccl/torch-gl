import math

import pytest
import torch

from torch_gl import PolynomialRing, RNSPoly, RNSRing, select_ntt_primes
from torch_gl._number_theory import is_prime


def test_prime_selection_and_modulus_product():
    moduli = select_ntt_primes(4, 9, 3)
    assert len(set(moduli)) == 3
    assert all(is_prime(modulus) and modulus.bit_length() == 30 for modulus in moduli)
    assert all((modulus - 1) % (4 * 4 * 9) == 0 for modulus in moduli)
    ring = RNSRing.for_parameters(4, 9)
    assert ring.modulus == math.prod(moduli)
    assert ring.modulus > (1 << 63)
    assert ring.shape == (4, 4, 6, 2)


@pytest.mark.parametrize("variables", [("X", "Y", "W"), ("X", "W"), ()])
def test_shared_integer_encoding_and_arithmetic(variables):
    ring = RNSRing.for_parameters(4, 9, variables=variables)
    coefficients = torch.arange(math.prod(ring.shape)).reshape(ring.shape) - 70
    polynomial = ring.polynomial(coefficients)
    assert torch.equal(polynomial.centered_coefficients(), coefficients)
    assert polynomial.reconstruct() == tuple(coefficients.reshape(-1).tolist())
    for limb in polynomial.limbs:
        assert torch.equal(limb.coefficients, coefficients.remainder(limb.ring.q))
    assert (polynomial + 8 - polynomial) == ring.constant(8)
    assert 3 - polynomial == -(polynomial - 3)
    assert polynomial * ring.one() == polynomial


def test_rns_product_matches_each_polynomial_limb():
    ring = RNSRing.for_parameters(2, 9)
    generator = torch.Generator().manual_seed(1935)
    left_values, right_values = [
        torch.randint(-50, 51, ring.shape, generator=generator) for _ in range(2)
    ]
    product = ring.polynomial(left_values) * ring.polynomial(right_values)
    for limb in product.limbs:
        expected = limb.ring.polynomial(left_values) * limb.ring.polynomial(right_values)
        assert limb == expected
    assert product.reconstruct() == tuple(product.centered_coefficients().reshape(-1).tolist())


def test_full_crt_handles_values_larger_than_int64():
    ring = RNSRing.for_parameters(1, 3, variables=())
    real, imag = ring.modulus // 3, -(ring.modulus // 4)
    polynomial = ring.constant((real, imag))
    assert real > (1 << 63)
    assert polynomial.reconstruct() == (real, imag)
    assert polynomial.reconstruct(centered=False) == (real, imag % ring.modulus)
    with pytest.raises(ValueError, match="reconstruct"):
        polynomial.centered_coefficients()


def test_uniform_residues_and_input_validation():
    ring = RNSRing.for_parameters(4, 9)
    polynomial = ring.uniform(generator=torch.Generator().manual_seed(2025))
    for limb in polynomial.limbs:
        assert bool(((limb.coefficients >= 0) & (limb.coefficients < limb.ring.q)).all())
    # Uniform residues are not formed by copying one small integer tensor.
    with pytest.raises(ValueError):
        polynomial.centered_coefficients()
    with pytest.raises(ValueError):
        RNSRing((97, 97), ring.axes)
    with pytest.raises(ValueError):
        RNSRing((), ring.axes)
    with pytest.raises(ValueError):
        RNSPoly(ring, polynomial.limbs[:-1])
    wrong_limb = PolynomialRing.multivariate_poly(97, 4, 9).one()
    with pytest.raises(ValueError):
        RNSPoly(ring, (wrong_limb, *polynomial.limbs[1:]))
    reversed_ring = RNSRing(ring.moduli[::-1], ring.axes)
    with pytest.raises(ValueError):
        ring.one() + reversed_ring.one()


@pytest.mark.parametrize(
    "arguments", [(3, 9, 3, 30), (4, 15, 3, 30), (4, 9, 0, 30), (4, 9, 3, 31), (128, 729, 3, 8)]
)
def test_invalid_prime_selection(arguments):
    n, p, count, bits = arguments
    with pytest.raises(ValueError):
        select_ntt_primes(n, p, count, bits=bits)
