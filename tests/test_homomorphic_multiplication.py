import json
from pathlib import Path

import pytest
import torch

from torch_gl import Engine, MatrixEncoder, QuadraticCiphertext, RNSRing

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("compiled", [False, True])
def test_ciphertext_product_components_and_decryption_equation(compiled):
    ring = RNSRing.for_parameters(2, 9)
    engine = Engine(
        ring, compiled=compiled, backend="eager", generator=torch.Generator().manual_seed(1935)
    )
    message_left = 10000 * ring.generator("X")
    message_right = 10000 * ring.generator("W") + 3 * ring.generator("Y")
    left, right = engine.encrypt(message_left), engine.encrypt(message_right)
    product = engine.multiply(left, right)
    assert isinstance(product, QuadraticCiphertext)
    assert product.c0 == left.b * right.b
    assert product.c1 == left.b * right.a + right.b * left.a
    assert product.c2 == left.a * right.a
    assert left * right == product
    secret = engine.secret_key
    explicit_decryption = product.c0 + product.c1 * secret + product.c2 * (secret * secret)
    decrypted_left = engine.decrypt(left, secret)
    decrypted_right = engine.decrypt(right, secret)
    assert engine.decrypt(product, secret) == explicit_decryption
    assert explicit_decryption == decrypted_left * decrypted_right
    error_left = decrypted_left - message_left
    error_right = decrypted_right - message_right
    expected_error = (
        message_left * error_right + message_right * error_left + error_left * error_right
    )
    assert explicit_decryption - message_left * message_right == expected_error


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("noise_std", [0, 3.2])
def test_sage_encoded_homomorphic_square_decodes_at_squared_scale(device, noise_std):
    vectors = json.loads((Path(__file__).parent / "data/sage_encoding.json").read_text())
    case = next(case for case in vectors["cases"] if case["name"] == "full_n4_p9")
    # Four limbs give headroom for scale^2 and coefficient growth, with no rescaling.
    ring = RNSRing.for_parameters(case["n"], case["p"], limb_count=4)
    encoder = MatrixEncoder(ring, scale=case["scale"], device=device)
    product_decoder = MatrixEncoder(ring, scale=case["scale"] ** 2, device=device)
    pairs = torch.tensor(case["input_slots"], dtype=torch.float64, device=device)
    matrices = (
        torch.complex(pairs[:, 0], pairs[:, 1]).reshape(case["slot_shape"])
        / case["slot_denominator"]
    )
    encoded = encoder.encode(matrices)
    assert encoded.reconstruct() == tuple(case["encoded_coefficients"])
    engine = Engine(
        ring,
        device=device,
        noise_std=noise_std,
        generator=torch.Generator(device=device).manual_seed(1935),
    )
    left, right = engine.encrypt(encoded), engine.encrypt(encoded)
    decrypted_left = engine.decrypt(left, engine.secret_key)
    decrypted_right = engine.decrypt(right, engine.secret_key)
    product = engine.multiply(left, right)
    decrypted = engine.decrypt(product, engine.secret_key)
    assert decrypted == decrypted_left * decrypted_right
    reconstructed = decrypted.reconstruct()
    assert max(abs(value) for value in reconstructed) > 2**63
    # Bound coefficient magnitudes without modular arithmetic. Each quotient
    # monomial reduces to at most two W terms. This ensures Q cannot wrap the
    # integer product represented by these particular inputs.
    left_l1 = sum(abs(value) for value in decrypted_left.reconstruct())
    right_l1 = sum(abs(value) for value in decrypted_right.reconstruct())
    assert 2 * left_l1 * right_l1 < ring.modulus // 2
    recovered = product_decoder.decode(decrypted)
    expected = matrices * matrices  # Ordinary ring multiplication is Hadamard in slots.
    torch.testing.assert_close(recovered, expected, atol=2e-9, rtol=1e-11)
    # Also check the exact canonical-embedding multiplication property, with noise.
    noisy_expected = encoder.decode(decrypted_left) * encoder.decode(decrypted_right)
    torch.testing.assert_close(recovered, noisy_expected, atol=2e-12, rtol=1e-12)
    # Using the original scale would leave an extra factor of scale.
    torch.testing.assert_close(encoder.decode(decrypted) / case["scale"], recovered, atol=0, rtol=0)


def test_quadratic_ciphertext_validation_and_engine_context():
    ring = RNSRing.for_parameters(2, 9)
    other_ring = RNSRing.for_parameters(4, 9)
    engine = Engine(ring)
    ciphertext = engine.encrypt(10000)
    with pytest.raises(TypeError):
        QuadraticCiphertext(ring.one(), ring.one(), 1)
    with pytest.raises(ValueError):
        QuadraticCiphertext(ring.one(), other_ring.one(), ring.one())
    with pytest.raises(ValueError):
        engine.multiply(ciphertext, Engine(other_ring).encrypt(1))
    with pytest.raises(ValueError):
        engine.decrypt(
            QuadraticCiphertext(other_ring.one(), other_ring.one(), other_ring.one()),
            engine.secret_key,
        )
    with pytest.raises(TypeError):
        engine.multiply(ciphertext, ciphertext * ciphertext)
