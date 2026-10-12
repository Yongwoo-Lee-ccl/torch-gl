from dataclasses import fields

import pytest
import torch

from torch_gl import Ciphertext, Engine, RNSRing

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("message_name", ["constant", "X", "W", "Y"])
def test_requested_messages_decrypt_to_message_plus_shared_noise(device, message_name, monkeypatch):
    ring = RNSRing.for_parameters(4, 9, limb_count=3)
    generator = torch.Generator(device=device).manual_seed(1935)
    engine = Engine(ring, device=device, generator=generator)
    key_coefficients = engine.secret_key.centered_coefficients()
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
    ciphertext = engine.encrypt(encryption_input)
    assert [field.name for field in fields(ciphertext)] == ["b", "a"]
    decrypted = engine.decrypt(ciphertext, engine.secret_key)
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
    engine = Engine(ring, secret_key=ring.one(), generator=torch.Generator().manual_seed(3))
    noise = (
        torch.arange(ring.shape[0] * ring.shape[1] * ring.shape[2] * 2).reshape(ring.shape) % 9 - 4
    )
    monkeypatch.setattr(engine, "_sample_noise", lambda: noise)
    ciphertext = engine.encrypt(10000)
    assert ciphertext.b == ring.constant(10000) + ring.polynomial(noise) - ciphertext.a
    assert engine.decrypt(ciphertext, engine.secret_key) == ring.constant(10000) + ring.polynomial(
        noise
    )


def test_zero_noise_and_fresh_ciphertexts():
    ring = RNSRing.for_parameters(4, 9)
    engine = Engine(ring, noise_std=0, generator=torch.Generator().manual_seed(4))
    first, second = engine.encrypt(10000), engine.encrypt(10000)
    assert first.a != second.a
    assert first.b != second.b
    assert engine.decrypt(first, engine.secret_key) == ring.constant(10000)
    assert engine.decrypt(second, engine.secret_key) == ring.constant(10000)


def test_encryption_with_configurable_smaller_ring_and_padded_cyclotomic():
    ring = RNSRing.for_parameters(2, 25, limb_count=2, variables=("X", "W"))
    engine = Engine(ring, noise_std=0, generator=torch.Generator().manual_seed(25))
    message = 10000 * ring.generator("W")
    assert engine.decrypt(engine.encrypt(message), engine.secret_key) == message


def test_compiled_encryption_and_decryption():
    ring = RNSRing.for_parameters(4, 9)
    engine = Engine(
        ring,
        noise_std=0,
        compiled=True,
        backend="eager",
        generator=torch.Generator().manual_seed(5),
    )
    message = 10000 * ring.generator("Y")
    assert engine.decrypt(engine.encrypt(message), engine.secret_key) == message


def test_rounded_gaussian_statistics():
    ring = RNSRing.for_parameters(32, 9, limb_count=1)
    engine = Engine(ring, generator=torch.Generator().manual_seed(6))
    noise = engine._sample_noise()
    assert noise.dtype == torch.int64
    assert abs(float(noise.float().mean())) < 0.15
    assert 3.0 < float(noise.float().std()) < 3.4


def test_secret_and_noise_share_residues_and_seed_reproducibility():
    ring = RNSRing.for_parameters(4, 9)
    first = Engine(ring, generator=torch.Generator().manual_seed(7))
    second = Engine(ring, generator=torch.Generator().manual_seed(7))
    assert first.secret_key == second.secret_key
    assert first.encrypt(10000) == second.encrypt(10000)


def test_invalid_keys_messages_and_ciphertexts():
    ring = RNSRing.for_parameters(4, 9)
    with pytest.raises(ValueError, match="binary"):
        Engine(ring, secret_key=ring.constant(2))
    for invalid_std in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Engine(ring, noise_std=invalid_std)
    engine = Engine(ring)
    other_ring = RNSRing.for_parameters(2, 9)
    with pytest.raises(ValueError):
        engine.encrypt(other_ring.one())
    with pytest.raises(TypeError):
        engine.encrypt(3.5)
    with pytest.raises(TypeError):
        engine.decrypt((ring.one(), ring.one()), engine.secret_key)
    with pytest.raises(ValueError):
        Ciphertext(ring.one(), other_ring.one())
    with pytest.raises(ValueError):
        engine.decrypt(engine.encrypt(10000), other_ring.one())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_engine_generator_device_mismatch():
    with pytest.raises(ValueError):
        Engine(RNSRing.for_parameters(2, 9), device="cuda", generator=torch.Generator())
