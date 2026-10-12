import pytest
import torch

from torch_gl import Engine, NTTPlan, PolynomialRing, RNSRing
from torch_gl.modular import mul_mod, residues


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8])
def test_input_normalization_before_narrowing(dtype):
    values = torch.tensor([0, 1, 120], dtype=dtype)
    if dtype == torch.int64:
        values = torch.tensor([-(1 << 63), (1 << 63) - 1, (1 << 40) + 123], dtype=dtype)
    actual = residues(values, 754974721)
    assert actual.dtype == torch.int32
    assert actual.tolist() == [int(value) % 754974721 for value in values]


def test_large_products_do_not_overflow_int32():
    q = 754974721
    left = torch.tensor([q - 1, q - 2, q // 2], dtype=torch.int32)
    right = torch.tensor([q - 1, q - 2, q // 3], dtype=torch.int32)
    product = mul_mod(left, right, q)
    assert product.dtype == torch.int32
    assert product.tolist() == [
        int(first) * int(second) % q for first, second in zip(left, right, strict=True)
    ]


def test_polynomial_rns_key_ciphertext_noise_and_tables_use_int32():
    ring = PolynomialRing.multivariate_poly(754974721, 4, 9)
    polynomial = ring.polynomial(torch.arange(192).reshape(ring.shape))
    assert polynomial.coefficients.dtype == torch.int32
    assert polynomial.coefficients.element_size() == 4
    assert (polynomial * polynomial).coefficients.dtype == torch.int32
    assert (polynomial + polynomial).coefficients.dtype == torch.int32
    assert (-polynomial).coefficients.dtype == torch.int32
    plan = NTTPlan(754974721, 16)
    values = torch.arange(16, dtype=torch.int32)
    assert plan.forward(values).dtype == torch.int32
    from torch_gl._transform_tables import device_powers

    assert device_powers(754974721, (plan.root,), 16, "cpu").dtype == torch.int32
    rns_ring = RNSRing.for_parameters(4, 9)
    engine = Engine(rns_ring, generator=torch.Generator().manual_seed(32))
    ciphertext = engine.encrypt(10000)
    decrypted = engine.decrypt(ciphertext, engine.secret_key)
    for polynomial in (
        *engine.secret_key.limbs,
        *ciphertext.a.limbs,
        *ciphertext.b.limbs,
        *decrypted.limbs,
    ):
        assert polynomial.coefficients.dtype == torch.int32
    assert decrypted.centered_coefficients().dtype == torch.int32
    assert engine._sample_noise().dtype == torch.int32
