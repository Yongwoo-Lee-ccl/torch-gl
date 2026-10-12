import pytest
import torch

from torch_gl import PolynomialRing, QuotientAxis


def test_defining_relations_and_gaussian_arithmetic():
    ring = PolynomialRing.multivariate_poly(433, 4, 27)
    X, Y, W, imaginary_unit = (ring.generator(name) for name in ("X", "Y", "W", "i"))
    assert imaginary_unit * imaginary_unit == ring.constant(-1)
    assert X**4 == imaginary_unit
    assert Y**4 == imaginary_unit
    assert W**18 + W**9 + 1 == ring.zero()
    assert W**27 == ring.one()
    assert (X + Y) * (X - Y) == X**2 - Y**2
    assert (X + 3) * (2, 1) == (2, 1) * (X + 3)
    assert 5 - X == -(X - 5)


@pytest.mark.parametrize("variables", [(), ("X",), ("W",), ("X", "W"), ("W", "Y", "X")])
def test_configurable_axis_subsets(variables):
    ring = PolynomialRing.multivariate_poly(433, 4, 27, variables=variables)
    assert len(ring.shape) == len(variables) + 1
    assert ring.one() * ring.constant((3, 4)) == ring.constant((3, 4))


def test_construct_reduce_normalize_pad_and_copy():
    ring = PolynomialRing.multivariate_poly(97, 2, 3, variables=("X",))
    coefficients = torch.tensor([[0, 0], [0, 0], [-1, 0], [0, 0], [1, 0]])
    original = coefficients.clone()
    polynomial = ring.polynomial(coefficients)
    assert polynomial == ring.constant((-1, -1))  # -X**2+X**4 = -i-1
    assert torch.equal(coefficients, original)
    coefficients.zero_()
    exported = polynomial.coefficients
    exported.zero_()
    assert polynomial == ring.constant((-1, -1))
    assert ring.from_real(torch.tensor([5])) == ring.constant(5)


def test_prime_power_cyclotomic_formula():
    phi = QuotientAxis.cyclotomic("W", 27)
    assert phi.degree == 18
    assert [
        coefficient_index for coefficient_index, third in enumerate(phi.modulus) if third != (0, 0)
    ] == [0, 9, 18]
    assert QuotientAxis.cyclotomic("W", 25).degree == 20


def test_arithmetic_ring_laws_with_deterministic_dense_inputs():
    ring = PolynomialRing.multivariate_poly(433, 2, 9)
    rng = torch.Generator().manual_seed(1935)
    first, second, third = [
        ring.polynomial(torch.randint(ring.q, ring.shape, generator=rng)) for _ in range(3)
    ]
    assert first * second == second * first
    assert (first * second) * third == first * (second * third)
    assert first * (second + third) == first * second + first * third
    assert first**0 == ring.one()
    assert first * ring.zero() == ring.zero()


def test_general_monic_ideal_and_high_degree_input():
    ring = PolynomialRing(97, (QuotientAxis("T", (2, -3, (0, 1), 1)),))
    custom_generator = ring.generator("T")
    assert (
        custom_generator**3 + (0, 1) * custom_generator**2 - 3 * custom_generator + 2 == ring.zero()
    )
    coefficients = torch.zeros(12, 2, dtype=torch.int64)
    coefficients[-1, 0] = 1
    assert ring.polynomial(coefficients) == custom_generator**11


def test_padded_binomial_path_without_splitting_roots():
    # 32 divides 96, but 64 does not: convolution is available even though
    # primitive 4n-th roots for the compact X transform do not exist.
    ring = PolynomialRing.multivariate_poly(97, 16, 3, variables=("X",))
    assert ring.ntt_shape == (32,)
    X = ring.generator("X")
    assert X**16 == ring.constant((0, 1))
    assert X**32 == ring.constant(-1)
    assert (X**15 + (2, 3)) * (X + 1) == X**15 + (2, 3) * X + (2, 4)


@pytest.mark.parametrize("p", [1, 2, 4, 15, 45])
def test_invalid_prime_powers(p):
    with pytest.raises(ValueError):
        QuotientAxis.cyclotomic("W", p)


def test_invalid_rings_shapes_and_operands():
    with pytest.raises(ValueError):
        PolynomialRing.multivariate_poly(97, 3, 3)
    with pytest.raises(ValueError):
        PolynomialRing.multivariate_poly(97, 4, 3, variables=("X", "X"))
    with pytest.raises(ValueError):
        PolynomialRing(97, tuple(QuotientAxis.power_of_i(str(third), 2) for third in "ABCD"))
    with pytest.raises(ValueError):
        QuotientAxis("T", (1, 2))
    ring = PolynomialRing.multivariate_poly(97, 2, 3)
    with pytest.raises(ValueError):
        ring.polynomial(torch.zeros(2, 2, 2, dtype=torch.int64))
    with pytest.raises(TypeError):
        ring.polynomial(torch.zeros(ring.shape))
    with pytest.raises(ValueError):
        ring.one() + PolynomialRing.multivariate_poly(193, 2, 3).one()
    with pytest.raises(ValueError):
        ring.one() ** -1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_device_mismatch():
    ring = PolynomialRing.multivariate_poly(97, 2, 3)
    with pytest.raises(ValueError):
        ring.one() + ring.one(device="cuda")
