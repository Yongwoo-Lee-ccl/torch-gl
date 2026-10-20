import math
from unittest.mock import patch

import pytest
import torch

from torch_gl import (
    BootstrapParameters,
    Bootstrapper,
    Engine,
    Evaluator,
    KeySwitcher,
    MatrixEncoder,
    RNSRing,
    ScaledCiphertext,
    create_secret_key,
    generate_bootstrap_keys,
    generate_evaluation_keys,
    split,
    split_ciphertext,
)


@pytest.fixture(scope="module")
def refresh_setup():
    ring = RNSRing.for_parameters(2, 3, limb_count=26)
    parameters = BootstrapParameters(ring)
    secret = create_secret_key(ring, torch.Generator().manual_seed(10))
    keys = generate_bootstrap_keys(parameters, secret, generator=torch.Generator().manual_seed(22))
    return parameters, secret, Bootstrapper(keys)


def decode(value, secret):
    return MatrixEncoder(value.ring, scale=value.scale).decode(
        Engine(value.ring).decrypt(value.ciphertext, secret.at_level(value.level))
    )


def test_full_noisy_complex_refresh_without_secret_or_decode_oracle(refresh_setup):
    parameters, secret, bootstrapper = refresh_setup
    bottom = RNSRing(parameters.ring.moduli[:1], parameters.ring.axes)
    encoder = MatrixEncoder(bottom, scale=2**25)
    generator = torch.Generator().manual_seed(41)
    values = torch.randn(encoder.slot_shape, dtype=torch.complex128, generator=generator) / 5
    engine = Engine(bottom, generator=generator)
    encrypted = ScaledCiphertext(
        engine.encrypt(encoder.encode(values), secret.at_level(1)), encoder.scale
    )
    original = engine.decrypt(encrypted.ciphertext, secret.at_level(1))
    stages = {}
    # Refresh has no access to any decryptor or slot decoder. Fail immediately
    # if either is accidentally introduced in place of homomorphic EvalMod.
    with (
        patch.object(
            Engine, "decrypt", side_effect=AssertionError("secret decryption during refresh")
        ),
        patch.object(
            MatrixEncoder, "decode", side_effect=AssertionError("slot decode during refresh")
        ),
    ):
        result = bootstrapper.bootstrap(
            encrypted, on_stage=lambda name, parts: stages.update({name: parts})
        )
    assert list(stages) == ["ModRaise", "CtS", "EvalMod", "StC"]
    assert result.level == parameters.ring.limb_count - parameters.consumed_levels
    assert result.level > encrypted.level
    assert result.ciphertext.a != encrypted.ciphertext.a.mod_raise(result.ring)
    q0 = bottom.modulus
    original_parts = split(original)
    observed_nonzero_carry = False
    for raised, before, cts in zip(stages["ModRaise"], original_parts, stages["CtS"], strict=True):
        plain = Engine(raised.ring).decrypt(raised.ciphertext, secret)
        lifted, base = plain.reconstruct(), before.reconstruct()
        carries = [(a - b) // q0 for a, b in zip(lifted, base, strict=True)]
        assert all((a - b) % q0 == 0 for a, b in zip(lifted, base, strict=True))
        observed_nonzero_carry |= any(carries)
        assert max(abs(i) for i in carries) <= parameters.carry_bound
        coefficients = torch.tensor(lifted, dtype=torch.float64).reshape(raised.ring.shape)
        target = torch.complex(coefficients[..., 0], coefficients[..., 1]).T / q0
        torch.testing.assert_close(decode(cts, secret), target, atol=2e-6, rtol=2e-6)
    assert observed_nonzero_carry
    torch.testing.assert_close(decode(result, secret), values, atol=7e-4, rtol=7e-4)
    # The same public refresh material can be reused on refreshed ciphertexts.
    twice = bootstrapper.bootstrap(result.at_level(1))
    assert twice.level == result.level
    torch.testing.assert_close(decode(twice, secret), values, atol=1e-3, rtol=1e-3)
    # A refresh must restore computation capacity, not just change the modulus
    # or rescale the metadata. Exercise a genuine encrypted product afterward.
    switcher = KeySwitcher(parameters.ring, k=parameters.k)
    evaluator = Evaluator(
        parameters.ring, generate_evaluation_keys(switcher, secret, matrices=False)
    )
    squared = evaluator.square(result)
    assert squared.level == result.level - 1
    torch.testing.assert_close(decode(squared, secret), values.square(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("n,p", [(4, 3), (2, 9)])
def test_linear_transforms_ring_padding_and_grouping_against_coefficients(n, p):
    ring = RNSRing.for_parameters(n, p, limb_count=24)
    parameters = BootstrapParameters(ring)
    secret = create_secret_key(ring, torch.Generator().manual_seed(43))
    bootstrapper = Bootstrapper(
        generate_bootstrap_keys(parameters, secret, generator=torch.Generator().manual_seed(51))
    )
    encoder = MatrixEncoder(ring, scale=2**32)
    values = (
        torch.randn(
            encoder.slot_shape, dtype=torch.complex128, generator=torch.Generator().manual_seed(6)
        )
        / 8
    )
    engine = Engine(ring, generator=torch.Generator().manual_seed(30))
    original = engine.encrypt(encoder.encode(values), secret)
    parts = tuple(ScaledCiphertext(ct, encoder.scale) for ct in split_ciphertext(original))
    transformed = bootstrapper.coeff_to_slot(parts)
    plaintext = split(engine.decrypt(original, secret))
    for value, polynomial in zip(transformed, plaintext, strict=True):
        coefficients = torch.tensor(polynomial.reconstruct(), dtype=torch.float64).reshape(
            polynomial.ring.shape
        )
        expected = torch.complex(coefficients[..., 0], coefficients[..., 1]).T / encoder.scale
        torch.testing.assert_close(decode(value, secret), expected, atol=2e-6, rtol=2e-6)
    restored = bootstrapper.slot_to_coeff(transformed)
    for actual, before in zip(restored, parts, strict=True):
        torch.testing.assert_close(
            decode(actual, secret), decode(before, secret), atol=2e-5, rtol=2e-5
        )


def test_bootstrap_parameter_and_input_validation(refresh_setup):
    parameters, secret, bootstrapper = refresh_setup
    with pytest.raises(ValueError, match="at least"):
        BootstrapParameters(RNSRing.for_parameters(2, 3, limb_count=4))
    with pytest.raises(ValueError, match="double_angle"):
        BootstrapParameters(parameters.ring, double_angle=1)
    with pytest.raises(ValueError):
        BootstrapParameters(parameters.ring, message_ratio=0.5)
    with pytest.raises(ValueError):
        BootstrapParameters(parameters.ring, carry_bound=math.inf)
    with pytest.raises(ValueError):
        bootstrapper.bootstrap_many([])
    with pytest.raises(TypeError):
        bootstrapper.bootstrap(parameters.ring.one())
    engine = Engine(parameters.ring)
    wrong_level = ScaledCiphertext(engine.encrypt(0, secret), 2**25)
    with pytest.raises(ValueError, match="single bottom prime"):
        bootstrapper.bootstrap(wrong_level)


def test_evalmod_at_positive_negative_carries_and_complex_domain_boundaries(refresh_setup):
    parameters, secret, bootstrapper = refresh_setup
    ring = parameters.base_ring
    encoder = MatrixEncoder(ring, scale=ring.moduli[0])
    values = torch.tensor(
        [[8.04 + 7.95j, -7.94 - 8.02j], [6.97 - 0.06j, -0.03 + 3.06j]], dtype=torch.complex128
    )
    expected = torch.complex(values.real - values.real.round(), values.imag - values.imag.round())
    engine = Engine(ring, generator=torch.Generator().manual_seed(5))
    encrypted = ScaledCiphertext(engine.encrypt(encoder.encode(values), secret), encoder.scale)
    result = bootstrapper.eval_mod(encrypted)
    torch.testing.assert_close(decode(result, secret), expected, atol=3e-5, rtol=3e-5)


def test_nearest_rescale_against_large_signed_integer_quotients():
    ring = RNSRing.for_parameters(2, 3, limb_count=3)
    p = ring.moduli[-1]
    coefficients = [
        -(1 << 69),
        -p - p // 2,
        -p // 2,
        -1,
        0,
        1,
        p // 2,
        p + p // 2,
        (1 << 69),
        12,
        15,
        19,
        31,
        44,
        100,
        -98,
    ]
    from torch_gl import MultivariatePoly, RNSPoly

    polynomial = RNSPoly(
        ring,
        tuple(
            MultivariatePoly(
                limb,
                torch.tensor([c % limb.q for c in coefficients], dtype=torch.int32).reshape(
                    ring.shape
                ),
            )
            for limb in ring.rings
        ),
    )
    actual = polynomial.rescale(rounding="nearest").reconstruct(centered=False)
    expected = tuple(((c + p // 2) // p) % (ring.modulus // p) for c in coefficients)
    assert actual == expected
    with pytest.raises(ValueError):
        polynomial.rescale(rounding="invalid")


def test_bootstrap_many_repackages_across_input_ciphertexts():
    ring = RNSRing.for_parameters(1, 3, limb_count=26)
    parameters = BootstrapParameters(ring)
    secret = create_secret_key(ring, torch.Generator().manual_seed(11))
    bootstrapper = Bootstrapper(
        generate_bootstrap_keys(parameters, secret, generator=torch.Generator().manual_seed(28))
    )
    bottom = RNSRing(ring.moduli[:1], ring.axes)
    encoder = MatrixEncoder(bottom, scale=2**25)
    engine = Engine(bottom, generator=torch.Generator().manual_seed(55))
    values = [
        torch.tensor([[[0.1 + 0.05j]], [[-0.15 + 0.02j]]], dtype=torch.complex128),
        torch.tensor([[[-0.07 - 0.1j]], [[0.03 + 0.12j]]], dtype=torch.complex128),
    ]
    ciphertexts = [
        ScaledCiphertext(engine.encrypt(encoder.encode(v), secret.at_level(1)), encoder.scale)
        for v in values
    ]
    # Each input contributes one R ciphertext; W packing merges the two
    # different inputs into the same Raux ciphertext and splits them back.
    results = bootstrapper.bootstrap_many(ciphertexts)
    assert len(results) == 2
    for result, expected in zip(results, values, strict=True):
        torch.testing.assert_close(decode(result, secret), expected, atol=1e-3, rtol=1e-3)
    with pytest.raises(ValueError, match="matching scales"):
        bootstrapper.bootstrap_many(
            [ciphertexts[0], ScaledCiphertext(ciphertexts[1].ciphertext, 12345)]
        )
