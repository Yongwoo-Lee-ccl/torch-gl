import math

import pytest
import torch

from torch_gl import (
    Ciphertext,
    Engine,
    KeySwitcher,
    KeySwitchParameters,
    MatrixEncoder,
    RNSRing,
    create_secret_key,
)
from torch_gl.basis import RNSBasisConverter
from torch_gl.modular import arithmetic_context, mul_mod


def test_han_ki_conversion_matches_integer_CRT_sum():
    source = (97, 193, 577)
    targets = source + (433, 1297)
    values = tuple(
        torch.tensor([0, 1, prime - 1, prime // 2], dtype=torch.int32) for prime in source
    )
    converter = RNSBasisConverter(source, targets)
    lifted = []
    product = math.prod(source)
    for index in range(values[0].numel()):
        total = 0
        for source_index, prime in enumerate(source):
            summand = int(values[source_index][index]) * pow(product // prime, -1, prime) % prime
            if summand > prime // 2:
                summand -= prime
            total += summand * (product // prime)
        lifted.append(total)
    converted = converter.convert(values)
    for target, actual in zip(targets, converted, strict=True):
        assert torch.equal(
            actual, torch.tensor([value % target for value in lifted], dtype=torch.int32)
        )
    assert max(abs(value) for value in lifted) < len(source) * product / 2


def test_han_ki_mod_down_matches_exact_integer_division_with_bounded_error():
    key_primes, ciphertext_primes = (577, 1297), (97, 193, 433)
    P, Q = math.prod(key_primes), math.prod(ciphertext_primes)
    integers = [-P * Q // 2 + 3, -P - 3, -P // 2, -1, 0, 1, P // 2, P + 3, P * Q // 2 - 3]
    P_values = tuple(
        torch.tensor([value % prime for value in integers], dtype=torch.int32)
        for prime in key_primes
    )
    converter = RNSBasisConverter(key_primes, ciphertext_primes)
    raised = converter.convert(P_values)
    for column, value in enumerate(integers):
        lifted_remainder = 0
        for prime in key_primes:
            summand = value * pow(P // prime, -1, prime) % prime
            if summand > prime // 2:
                summand -= prime
            lifted_remainder += summand * (P // prime)
        approximate_quotient = (value - lifted_remainder) // P
        nearest_quotient = (value + P // 2) // P
        assert abs(approximate_quotient - nearest_quotient) < len(key_primes)
        for prime, converted in zip(ciphertext_primes, raised, strict=True):
            result = (value % prime - int(converted[column])) * pow(P, -1, prime) % prime
            assert result == approximate_quotient % prime


def test_noiseless_evaluation_keys_have_only_bounded_mod_down_error():
    ring = RNSRing.for_parameters(4, 9, limb_count=3)
    source = Engine(ring, generator=torch.Generator().manual_seed(101))
    source_secret_key = create_secret_key(
        ring, generator=torch.Generator(device=source.device).manual_seed((101) + 100000)
    )
    target = Engine(ring, generator=torch.Generator().manual_seed(102))
    target_secret_key = create_secret_key(
        ring, generator=torch.Generator(device=target.device).manual_seed((102) + 100000)
    )
    switcher = KeySwitcher(ring, k=2, noise_std=0, generator=torch.Generator().manual_seed(103))
    key = switcher.generate_key(source_secret_key, target_secret_key)
    original = source.encrypt(10000 * ring.generator("Y"), source_secret_key)
    switched = switcher.switch_ciphertext(original, key)
    before = source.decrypt(original, source_secret_key)
    after = target.decrypt(switched, target_secret_key)
    secret_l1 = int(target_secret_key.centered_coefficients().abs().sum())
    # Conv gives each discarded remainder in (-k*P/2,k*P/2). Phi_p reduction can
    # duplicate a convolution term twice; Gaussian channels use the total l1.
    assert int((after - before).centered_coefficients().abs().max()) < switcher.k * (
        1 + 2 * secret_l1
    )


@pytest.mark.parametrize("limb_count,k", [(5, 2), (4, 1), (3, 3), (2, 4)])
def test_grouped_decomposition_reconstructs_original_residues(limb_count, k):
    ring = RNSRing.for_parameters(2, 9, limb_count=limb_count)
    switcher = KeySwitcher(ring, k=k)
    assert switcher.dnum == (limb_count + k - 1) // k
    assert len(switcher.parameters.key_moduli) == k
    assert switcher.parameters.key_modulus >= max(switcher.parameters.digit_moduli)
    values = ring.uniform(generator=torch.Generator().manual_seed(19))
    digits = [
        tuple(
            mul_mod(values.limbs[index]._coefficients, inverse, ring.moduli[index])
            for index, inverse in zip(group, inverses, strict=True)
        )
        for group, inverses in zip(
            switcher.parameters.groups, switcher._digit_inverses, strict=True
        )
    ]
    for limb in values.limbs:
        prime = limb.ring.q
        reconstructed = torch.zeros(ring.shape, dtype=torch.int32)
        for prepared, converter, modulus in zip(
            digits, switcher._converters, switcher.parameters.digit_moduli, strict=True
        ):
            digit = converter.convert_prepared(prepared, prime)
            term = mul_mod(digit, (ring.modulus // modulus) % prime, prime)
            reconstructed = (reconstructed + term).remainder(prime)
        assert torch.equal(reconstructed, limb.coefficients)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_switch_small_between_independent_secrets(k):
    ring = RNSRing.for_parameters(4, 9, limb_count=3)
    source_engine = Engine(ring, generator=torch.Generator().manual_seed(1))
    source_secret_key = create_secret_key(
        ring, generator=torch.Generator(device=source_engine.device).manual_seed((1) + 100000)
    )
    target_engine = Engine(ring, generator=torch.Generator().manual_seed(2))
    target_secret_key = create_secret_key(
        ring, generator=torch.Generator(device=target_engine.device).manual_seed((2) + 100000)
    )
    assert source_secret_key != target_secret_key
    switcher = KeySwitcher(ring, k=k, generator=torch.Generator().manual_seed(3))
    key = switcher.generate_key(source_secret_key, target_secret_key)
    message = (
        10000 * ring.generator("X") + 10000 * ring.generator("Y") + 10000 * ring.generator("W")
    )
    original = source_engine.encrypt(message, source_secret_key)
    switched = switcher.switch_ciphertext(original, key)
    assert isinstance(switched, Ciphertext)
    original_plaintext = source_engine.decrypt(original, source_secret_key)
    switched_plaintext = target_engine.decrypt(switched, target_secret_key)
    difference = (switched_plaintext - original_plaintext).centered_coefficients()
    assert int(difference.abs().max()) < 10000
    assert int((switched_plaintext - message).centered_coefficients().abs().max()) < 10000
    assert bool(switched.a.limbs[0].coefficients.any())
    assert all(
        "Y" not in [axis.name for axis in transform.ring.axes]
        for transform in switcher._transforms.values()
    )


@pytest.mark.parametrize("k", [1, 2, 3])
def test_hadamard_relinearization_switches_s_squared_with_small_key(k):
    ring = RNSRing.for_parameters(4, 9, limb_count=4)
    engine = Engine(ring, generator=torch.Generator().manual_seed(19))
    secret_key = create_secret_key(
        ring, generator=torch.Generator(device=engine.device).manual_seed((19) + 100000)
    )
    switcher = KeySwitcher(ring, k=k, generator=torch.Generator().manual_seed(20))
    key = switcher.generate_hadamard_key(secret_key)
    assert key.kind == "small"
    encoder = MatrixEncoder(ring)
    decoder = MatrixEncoder(ring, scale=encoder.scale**2)
    generator = torch.Generator().manual_seed(21)
    left = torch.randn(encoder.slot_shape, dtype=torch.complex128, generator=generator)
    right = torch.randn(encoder.slot_shape, dtype=torch.complex128, generator=generator)
    product = engine.multiply(
        engine.encrypt(encoder.encode(left), secret_key),
        engine.encrypt(encoder.encode(right), secret_key),
    )
    switched = switcher.relinearize(product, key)
    before = engine.decrypt(product, secret_key)
    after = engine.decrypt(switched, secret_key)
    assert int((after - before).centered_coefficients().abs().max()) < 10000
    torch.testing.assert_close(decoder.decode(after), left * right, atol=2e-9, rtol=1e-11)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("variables", [("X", "Y", "W"), ("W", "X", "Y")])
def test_matrix_relinearization_uses_two_big_switches(k, variables, monkeypatch):
    ring = RNSRing.for_parameters(4, 9, limb_count=4, variables=variables)
    engine = Engine(ring, generator=torch.Generator().manual_seed(31))
    secret_key = create_secret_key(
        ring, generator=torch.Generator(device=engine.device).manual_seed((31) + 100000)
    )
    switcher = KeySwitcher(ring, k=k, generator=torch.Generator().manual_seed(32))
    keys = switcher.generate_matrix_keys(secret_key)
    assert keys.Y_key.kind == keys.mixed_key.kind == "big"
    encoder = MatrixEncoder(ring)
    decoder = MatrixEncoder(ring, scale=encoder.scale**2)
    generator = torch.Generator().manual_seed(33)
    left = torch.randn(encoder.slot_shape, dtype=torch.complex128, generator=generator)
    right = torch.randn(encoder.slot_shape, dtype=torch.complex128, generator=generator)
    product = engine.matrix_multiply(
        engine.encrypt(encoder.encode(left), secret_key),
        engine.encrypt(encoder.encode(right), secret_key),
    )
    original_switch = switcher.switch_big
    calls = []

    def capture_switch(component, key):
        calls.append(key)
        return original_switch(component, key)

    monkeypatch.setattr(switcher, "switch_big", capture_switch)
    switched = switcher.relinearize_matrix(product, keys)
    assert len(calls) == 2 and calls[0] is keys.Y_key and calls[1] is keys.mixed_key
    before = engine.decrypt(product, secret_key)
    after = engine.decrypt(switched, secret_key)
    assert int((after - before).centered_coefficients().abs().max()) < 20000
    torch.testing.assert_close(
        decoder.decode(after), left @ right.transpose(-1, -2), atol=2e-9, rtol=1e-11
    )


def test_evaluation_key_relation_and_file_backed_keys(tmp_path):
    ring = RNSRing.for_parameters(2, 9, limb_count=3)
    source_engine = Engine(ring, generator=torch.Generator().manual_seed(41))
    source_secret_key = create_secret_key(
        ring, generator=torch.Generator(device=source_engine.device).manual_seed((41) + 100000)
    )
    target_engine = Engine(ring, generator=torch.Generator().manual_seed(42))
    target_secret_key = create_secret_key(
        ring, generator=torch.Generator(device=target_engine.device).manual_seed((42) + 100000)
    )
    switcher = KeySwitcher(ring, k=2, noise_std=0, generator=torch.Generator().manual_seed(43))
    key = switcher.generate_key(source_secret_key, target_secret_key)
    source_values = source_secret_key.centered_coefficients()
    target_values = target_secret_key.centered_coefficients()
    for index, prime in enumerate(switcher.parameters.extended_ring.moduli):
        transform = switcher._transform("small", index)
        arithmetic = arithmetic_context(prime)
        target_slots = transform.forward(target_values.remainder(prime))
        for (key_b, key_a), digit_modulus in zip(
            key._load(index), switcher.parameters.digit_moduli, strict=True
        ):
            decrypted = transform.inverse(
                arithmetic.add(key_b, arithmetic.multiply(key_a, target_slots))
            )
            factor = (switcher.parameters.key_modulus * (ring.modulus // digit_modulus)) % prime
            expected = mul_mod(source_values.remainder(prime), factor, prime)
            assert torch.equal(decrypted, expected)
    file_switcher = KeySwitcher(ring, k=2, noise_std=0, generator=torch.Generator().manual_seed(43))
    file_key = file_switcher.generate_key(
        source_secret_key, target_secret_key, storage_directory=tmp_path
    )
    assert file_key._paths and not file_key._limbs
    ciphertext = source_engine.encrypt(10000, source_secret_key)
    assert switcher.switch_ciphertext(ciphertext, key) == file_switcher.switch_ciphertext(
        ciphertext, file_key
    )


def test_invalid_switching_configuration_and_key_kinds():
    ring = RNSRing.for_parameters(2, 9, limb_count=3)
    for k in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            KeySwitchParameters(ring, k=k)
    with pytest.raises(ValueError, match="disjoint"):
        KeySwitchParameters(ring, k=1, key_moduli=ring.moduli[:1])
    engine = Engine(ring)
    secret_key = create_secret_key(
        ring, generator=torch.Generator(device=engine.device).manual_seed((1935) + 100000)
    )
    switcher = KeySwitcher(ring, k=2)
    with pytest.raises(ValueError, match="constant in Y"):
        switcher.generate_key(ring.generator("Y"), secret_key)
    key = switcher.generate_hadamard_key(secret_key)
    with pytest.raises(ValueError, match="kind"):
        switcher.switch_big(ring.one(), key)
    with pytest.raises(ValueError):
        KeySwitcher(ring, k=1).switch_small(ring.one(), key)
