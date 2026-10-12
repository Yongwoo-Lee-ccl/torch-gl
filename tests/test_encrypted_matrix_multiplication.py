import math
from dataclasses import fields

import pytest
import torch

from torch_gl import (
    Engine,
    MatrixEncoder,
    MatrixProductCiphertext,
    RNSRing,
    create_secret_key,
    matrix_multiply,
    matrix_transpose,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("noise_std", [0, 3.2])
@pytest.mark.parametrize(
    "n,p,variables",
    [(2, 9, ("X", "Y", "W")), (4, 27, ("W", "X", "Y")), (2, 729, ("X", "Y", "W"))],
)
def test_encrypted_matrix_product_with_small_secret_and_transpose(
    device, noise_std, n, p, variables
):
    ring = RNSRing.for_parameters(n, p, limb_count=4, variables=variables)
    generator = torch.Generator(device=device).manual_seed(1935)
    engine = Engine(ring, noise_std=noise_std, device=device, generator=generator)
    secret_key = create_secret_key(
        ring, generator=torch.Generator(device=engine.device).manual_seed((1935) + 100000)
    )
    encoder = MatrixEncoder(ring, device=device)
    product_decoder = MatrixEncoder(ring, scale=encoder.scale**2, device=device)
    left = torch.randn(
        encoder.slot_shape, dtype=torch.complex128, device=device, generator=generator
    )
    right = torch.randn(
        encoder.slot_shape, dtype=torch.complex128, device=device, generator=generator
    )
    left_encoded, right_encoded = encoder.encode(left), encoder.encode(right)
    left_ct, right_ct = (
        engine.encrypt(left_encoded, secret_key),
        engine.encrypt(right_encoded, secret_key),
    )
    product = engine.matrix_multiply(left_ct, right_ct)
    assert isinstance(product, MatrixProductCiphertext)
    assert [field.name for field in fields(product)] == ["c00", "c10", "c01", "c11"]
    assert product.c00 == matrix_multiply(left_ct.b, matrix_transpose(right_ct.b))
    assert product.c10 == matrix_multiply(left_ct.a, matrix_transpose(right_ct.b))
    assert product.c01 == matrix_multiply(left_ct.b, matrix_transpose(right_ct.a))
    assert product.c11 == matrix_multiply(left_ct.a, matrix_transpose(right_ct.a))
    secret_X = engine._embed_secret_key(secret_key)
    secret_Y = matrix_transpose(secret_X)
    assert secret_X != secret_Y
    Y_dim = variables.index("Y")
    assert not bool(secret_X.centered_coefficients().narrow(Y_dim, 1, n - 1).any())
    assert tuple(axis.name for axis in secret_key.ring.axes) == tuple(
        variable for variable in variables if variable != "Y"
    )
    assert torch.equal(
        secret_X.centered_coefficients().select(Y_dim, 0),
        secret_key.centered_coefficients(),
    )
    explicit_decryption = (
        product.c00
        + product.c10 * secret_X
        + product.c01 * secret_Y
        + product.c11 * (secret_X * secret_Y)
    )
    decrypted = engine.decrypt(product, secret_key)
    assert decrypted == explicit_decryption
    left_decrypted = engine.decrypt(left_ct, secret_key)
    right_decrypted = engine.decrypt(right_ct, secret_key)
    assert decrypted == matrix_multiply(left_decrypted, matrix_transpose(right_decrypted))
    if noise_std == 0:
        assert decrypted == matrix_multiply(left_encoded, matrix_transpose(right_encoded))
    left_noise = left_decrypted - left_encoded
    right_noise = right_decrypted - right_encoded
    coefficient_count = math.prod(ring.shape[:-1])
    left_error_bound = (
        sum(abs(value) for value in left_noise.reconstruct()) + coefficient_count / math.sqrt(2)
    ) / encoder.scale
    right_error_bound = (
        sum(abs(value) for value in right_noise.reconstruct()) + coefficient_count / math.sqrt(2)
    ) / encoder.scale
    product_error_bound = n * (
        left.abs().max().item() * right_error_bound
        + right.abs().max().item() * left_error_bound
        + left_error_bound * right_error_bound
    )
    left_l1 = sum(abs(value) for value in left_decrypted.reconstruct())
    right_l1 = sum(abs(value) for value in right_decrypted.reconstruct())
    assert 2 * n * left_l1 * right_l1 < ring.modulus // 2
    recovered = product_decoder.decode(decrypted)
    expected = left @ right.transpose(-1, -2)
    assert (recovered - expected).abs().max().item() <= product_error_bound + 2e-11
    noisy_expected = encoder.decode(left_decrypted) @ encoder.decode(right_decrypted).transpose(
        -1, -2
    )
    torch.testing.assert_close(recovered, noisy_expected, atol=2e-11, rtol=2e-12)
    assert not torch.allclose(recovered, left @ right.mH, atol=1e-6, rtol=1e-6)
    assert all(
        limb.coefficients.dtype == torch.int32
        for component in (product.c00, product.c10, product.c01, product.c11)
        for limb in component.limbs
    )


def test_supplied_small_key_embedding_and_compiled_matrix_decryption():
    ring = RNSRing.for_parameters(2, 9, limb_count=2)
    small_ring = RNSRing(ring.moduli, tuple(axis for axis in ring.axes if axis.name != "Y"))
    coefficients = (
        torch.arange(math.prod(small_ring.shape), dtype=torch.int32).reshape(small_ring.shape) % 2
    )
    small_secret = small_ring.polynomial(coefficients)
    engine = Engine(ring, noise_std=0, compiled=True, backend="eager")
    secret_key = small_secret
    left = ring.generator("X") + 7 * ring.generator("W")
    right = 3 * ring.generator("Y") + 11
    product = engine.matrix_multiply(
        engine.encrypt(left, secret_key), engine.encrypt(right, secret_key)
    )
    assert engine.decrypt(product, small_secret) == matrix_multiply(left, matrix_transpose(right))


def test_transpose_swaps_variables_without_conjugating_or_changing_W():
    ring = RNSRing.for_parameters(4, 9)
    X, Y, W = (ring.generator(name) for name in ("X", "Y", "W"))
    polynomial = X * Y * Y * W * W * W
    polynomial = polynomial + ring.constant((2, -3))
    expected = Y * X * X * W * W * W
    expected = expected + ring.constant((2, -3))
    assert matrix_transpose(polynomial) == expected
    assert matrix_transpose(matrix_transpose(polynomial)) == polynomial
    encoder = MatrixEncoder(ring, scale=1)
    torch.testing.assert_close(
        encoder.decode(matrix_transpose(polynomial)),
        encoder.decode(polynomial).transpose(-1, -2),
        atol=1e-12,
        rtol=1e-12,
    )


def test_matrix_ciphertext_validation_and_rejection_of_Y_dependent_secret():
    ring = RNSRing.for_parameters(2, 9)
    other_ring = RNSRing.for_parameters(4, 9)
    engine = Engine(ring)
    secret_key = create_secret_key(
        ring, generator=torch.Generator(device=engine.device).manual_seed((1935) + 100000)
    )
    with pytest.raises(TypeError):
        MatrixProductCiphertext(ring.one(), ring.one(), ring.one(), 1)
    with pytest.raises(ValueError):
        MatrixProductCiphertext(ring.one(), other_ring.one(), ring.one(), ring.one())
    with pytest.raises(ValueError):
        engine.matrix_multiply(
            engine.encrypt(1, secret_key),
            Engine(other_ring).encrypt(
                1, create_secret_key(other_ring, generator=torch.Generator().manual_seed(99))
            ),
        )
    with pytest.raises(TypeError):
        engine.matrix_multiply(engine.encrypt(1, secret_key), ring.one())
    Y_secret = ring.generator("Y")
    # Evaluation no longer inspects a retained key; matrix decryption checks
    # that the caller's explicit key belongs to the required small ring.
    valid_product = engine.matrix_multiply(
        engine.encrypt(1, secret_key), engine.encrypt(1, secret_key)
    )
    with pytest.raises(ValueError, match="constant in Y"):
        engine.decrypt(valid_product, Y_secret)
