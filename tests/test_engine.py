from dataclasses import fields

import pytest
import torch

from torch_gl import Ciphertext, Engine, RNSPoly, RNSRing, create_secret_key

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("message_name", ["constant", "X", "W", "Y"])
def test_requested_messages_decrypt_to_message_plus_shared_noise(device, message_name, monkeypatch):
    ring = RNSRing.for_parameters(4, 9, limb_count=3)
    generator = torch.Generator(device=device).manual_seed(1935)
    engine = Engine(ring, device=device, generator=generator)
    secret_key = create_secret_key(
        ring, generator=torch.Generator(device=engine.device).manual_seed((1935) + 100000)
    )
    key_coefficients = secret_key.centered_coefficients()
    assert bool(((key_coefficients == 0) | (key_coefficients == 1)).all())
    assert bool((key_coefficients == 0).any()) and bool((key_coefficients == 1).any())
    original_sample = engine._sample_noise
    sampled_noise = []

    def capture_noise():
        coefficients = original_sample()
        sampled_noise.append(coefficients.clone())
        return coefficients

    monkeypatch.setattr(engine, "_sample_noise", capture_noise)
    message = (
        ring.constant(10000, device=device)
        if message_name == "constant"
        else (10000 * ring.generator(message_name, device=device))
    )
    encryption_input = 10000 if message_name == "constant" else message
    ciphertext = engine.encrypt(encryption_input, secret_key)
    assert [field.name for field in fields(ciphertext)] == ["b", "a"]
    decrypted = engine.decrypt(ciphertext, secret_key)
    error = decrypted.centered_coefficients() - message.centered_coefficients()
    # Exact contract, stronger than just checking an approximate tolerance.
    assert torch.equal(error, sampled_noise[0])
    assert len(sampled_noise) == 1
    assert int(error.abs().max()) <= 32
    for decrypted_limb, message_limb in zip(decrypted.limbs, message.limbs, strict=True):
        assert torch.equal(
            (decrypted_limb - message_limb).coefficients,
            sampled_noise[0].remainder(decrypted_limb.ring.q),
        )


def test_encryption_sign_and_component_order_with_known_secret(monkeypatch):
    ring = RNSRing.for_parameters(2, 9)
    engine = Engine(ring, generator=torch.Generator().manual_seed(3))
    secret_key = ring.one()
    noise = (
        torch.arange(ring.shape[0] * ring.shape[1] * ring.shape[2] * 2).reshape(ring.shape) % 9 - 4
    )
    monkeypatch.setattr(engine, "_sample_noise", lambda: noise)
    ciphertext = engine.encrypt(10000, secret_key)
    assert ciphertext.b == ring.constant(10000) + ring.polynomial(noise) - ciphertext.a
    assert engine.decrypt(ciphertext, secret_key) == ring.constant(10000) + ring.polynomial(noise)


def test_zero_noise_and_fresh_ciphertexts():
    ring = RNSRing.for_parameters(4, 9)
    engine = Engine(ring, noise_std=0, generator=torch.Generator().manual_seed(4))
    secret_key = create_secret_key(
        ring, generator=torch.Generator(device=engine.device).manual_seed((4) + 100000)
    )
    first, second = engine.encrypt(10000, secret_key), engine.encrypt(10000, secret_key)
    assert first.a != second.a
    assert first.b != second.b
    assert engine.decrypt(first, secret_key) == ring.constant(10000)
    assert engine.decrypt(second, secret_key) == ring.constant(10000)


def test_encryption_with_configurable_smaller_ring_and_padded_cyclotomic():
    ring = RNSRing.for_parameters(2, 25, limb_count=2, variables=("X", "W"))
    engine = Engine(ring, noise_std=0, generator=torch.Generator().manual_seed(25))
    secret_key = create_secret_key(
        ring, generator=torch.Generator(device=engine.device).manual_seed((25) + 100000)
    )
    message = 10000 * ring.generator("W")
    assert engine.decrypt(engine.encrypt(message, secret_key), secret_key) == message


def test_compiled_encryption_and_decryption():
    ring = RNSRing.for_parameters(4, 9)
    engine = Engine(
        ring,
        noise_std=0,
        compiled=True,
        backend="eager",
        generator=torch.Generator().manual_seed(5),
    )
    secret_key = create_secret_key(
        ring, generator=torch.Generator(device=engine.device).manual_seed((5) + 100000)
    )
    message = 10000 * ring.generator("Y")
    assert engine.decrypt(engine.encrypt(message, secret_key), secret_key) == message


def test_rounded_gaussian_statistics():
    ring = RNSRing.for_parameters(32, 9, limb_count=1)
    engine = Engine(ring, generator=torch.Generator().manual_seed(6))
    noise = engine._sample_noise()
    assert noise.dtype == torch.int32
    assert abs(float(noise.float().mean())) < 0.15
    assert 3.0 < float(noise.float().std()) < 3.4


def test_secret_and_noise_share_residues_and_seed_reproducibility():
    ring = RNSRing.for_parameters(4, 9)
    first = Engine(ring, generator=torch.Generator().manual_seed(7))
    first_secret_key = create_secret_key(
        ring, generator=torch.Generator(device=first.device).manual_seed((7) + 100000)
    )
    second = Engine(ring, generator=torch.Generator().manual_seed(7))
    second_secret_key = create_secret_key(
        ring, generator=torch.Generator(device=second.device).manual_seed((7) + 100000)
    )
    assert first_secret_key == second_secret_key
    assert first.encrypt(10000, first_secret_key) == second.encrypt(10000, second_secret_key)


def test_key_generation_is_separate_from_engine_randomness():
    ring = RNSRing.for_parameters(4, 9, limb_count=3)
    encryption_generator = torch.Generator().manual_seed(17)
    key_generator = torch.Generator().manual_seed(23)
    encryption_state = encryption_generator.get_state().clone()
    key_state = key_generator.get_state().clone()
    engine = Engine(ring, generator=encryption_generator, noise_std=0)
    assert torch.equal(encryption_generator.get_state(), encryption_state)
    secret_key = create_secret_key(ring, generator=key_generator)
    assert torch.equal(encryption_generator.get_state(), encryption_state)
    assert not torch.equal(key_generator.get_state(), key_state)
    assert tuple(axis.name for axis in secret_key.ring.axes) == ("X", "W")
    assert secret_key == create_secret_key(ring, generator=torch.Generator().manual_seed(23))
    ciphertext = engine.encrypt(10000, secret_key)
    assert engine.decrypt(ciphertext, secret_key) == ring.constant(10000)
    assert not any(isinstance(value, RNSPoly) for value in vars(engine).values())
    for name in ("secret_key", "small_secret_key", "_secret_key"):
        assert not hasattr(engine, name)


def test_one_engine_can_encrypt_under_independent_explicit_keys():
    ring = RNSRing.for_parameters(4, 9)
    engine = Engine(ring, noise_std=0, generator=torch.Generator().manual_seed(29))
    first_key = create_secret_key(ring, generator=torch.Generator().manual_seed(31))
    second_key = create_secret_key(ring, generator=torch.Generator().manual_seed(37))
    assert first_key != second_key
    for secret_key in (first_key, second_key, first_key):
        ciphertext = engine.encrypt(10000, secret_key)
        assert engine.decrypt(ciphertext, secret_key) == ring.constant(10000)
    with pytest.raises(TypeError):
        engine.encrypt(10000)
    with pytest.raises(TypeError):
        Engine(ring, secret_key=first_key)


def test_secret_key_generator_arguments_are_required_and_validated():
    ring = RNSRing.for_parameters(4, 9)
    with pytest.raises(TypeError):
        create_secret_key(ring)
    with pytest.raises(TypeError, match="generator"):
        create_secret_key(ring, generator=None)
    with pytest.raises(TypeError, match="ring"):
        create_secret_key(None, generator=torch.Generator())


def test_invalid_keys_messages_and_ciphertexts():
    ring = RNSRing.for_parameters(4, 9)
    with pytest.raises(ValueError, match="binary"):
        Engine(ring).encrypt(10000, ring.constant(2))
    for invalid_std in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Engine(ring, noise_std=invalid_std)
    engine = Engine(ring)
    secret_key = create_secret_key(
        ring, generator=torch.Generator(device=engine.device).manual_seed((1935) + 100000)
    )
    other_ring = RNSRing.for_parameters(2, 9)
    with pytest.raises(ValueError):
        engine.encrypt(other_ring.one(), secret_key)
    with pytest.raises(TypeError):
        engine.encrypt(3.5, secret_key)
    with pytest.raises(TypeError):
        engine.decrypt((ring.one(), ring.one()), secret_key)
    with pytest.raises(ValueError):
        Ciphertext(ring.one(), other_ring.one())
    with pytest.raises(ValueError):
        engine.decrypt(engine.encrypt(10000, secret_key), other_ring.one())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_engine_generator_device_mismatch():
    with pytest.raises(ValueError):
        Engine(RNSRing.for_parameters(2, 9), device="cuda", generator=torch.Generator())
