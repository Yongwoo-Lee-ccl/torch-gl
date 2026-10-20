import math

import pytest
import torch

from torch_gl import (
    Engine,
    Evaluator,
    IntegerEncoder,
    IntegerEngine,
    KeySwitcher,
    MatrixEncoder,
    RNSRing,
    ScaledCiphertext,
    automorphism,
    auxiliary_matrix_multiply,
    conjugate,
    create_secret_key,
    embed,
    generate_evaluation_keys,
    hermitian_transpose,
    integer_ring,
    merge,
    merge_ciphertexts,
    split,
    split_ciphertext,
)


def key(ring):
    return create_secret_key(ring, torch.Generator().manual_seed(702))


@pytest.mark.parametrize("n,p", [(2, 3), (4, 3), (2, 9), (4, 27)])
def test_coefficient_ring_changes(n, p):
    ring = RNSRing.for_parameters(n, p, limb_count=2)
    secret = key(ring)
    engine = Engine(ring, generator=torch.Generator().manual_seed(44))
    message = ring.polynomial(torch.arange(math.prod(ring.shape)).reshape(ring.shape))
    ct = engine.encrypt(message, secret)
    parts = split(ct.b)
    assert len(parts) == n
    assert all(part.ring == secret.ring for part in parts)
    assert merge(parts, ring) == ct.b
    ciphertexts = split_ciphertext(ct)
    recovered = merge_ciphertexts(ciphertexts, ring)
    assert recovered.b == ct.b and recovered.a == ct.a
    plaintext_parts = split(engine.decrypt(ct, secret))
    for small_ct, plaintext in zip(ciphertexts, plaintext_parts, strict=True):
        assert Engine(secret.ring).decrypt(small_ct, secret) == plaintext
    auxiliary = RNSRing.auxiliary(ring.moduli, n, p)
    width = auxiliary.axes[-1].degree
    for start in range(0, n, width):
        group = ciphertexts[start : start + width]
        packed = merge_ciphertexts(group, auxiliary)
        auxiliary_plaintext = Engine(auxiliary).decrypt(packed, secret)
        expected = merge(plaintext_parts[start : start + width], auxiliary)
        assert auxiliary_plaintext == expected
        for actual, wanted in zip(split_ciphertext(packed, "V")[: len(group)], group, strict=True):
            assert actual.b == wanted.b and actual.a == wanted.a
    with pytest.raises(ValueError):
        merge(parts * (width + 1), auxiliary)


@pytest.mark.parametrize("p", [3, 9, 27])
def test_automorphisms_against_complex_embedding(p):
    ring = RNSRing.for_parameters(4, p, limb_count=3)
    encoder = MatrixEncoder(ring, scale=2**35)
    values = torch.randn(
        encoder.slot_shape, dtype=torch.complex128, generator=torch.Generator().manual_seed(8)
    )
    polynomial = encoder.encode(values)
    torch.testing.assert_close(
        encoder.decode(conjugate(polynomial)), values.conj(), atol=1e-8, rtol=1e-8
    )
    torch.testing.assert_close(
        encoder.decode(hermitian_transpose(polynomial)), values.mH, atol=1e-8, rtol=1e-8
    )
    assert conjugate(conjugate(polynomial)) == polynomial
    for variable, exponent, dim in [("X", 5, 1), ("Y", 5, 2), ("W", 2, 0)]:
        actual = encoder.decode(automorphism(polynomial, {variable: exponent}))
        torch.testing.assert_close(actual, torch.roll(values, -1, dim), atol=1e-8, rtol=1e-8)
    with pytest.raises(ValueError):
        automorphism(polynomial, {"X": 2})
    with pytest.raises(ValueError):
        automorphism(polynomial, {"W": 3})


@pytest.mark.parametrize("k", [1, 2, 3])
def test_evaluation_keys_reused_at_all_prefixes_including_partial_digits(k):
    ring = RNSRing.for_parameters(2, 3, limb_count=5)
    secret = key(ring)
    top = KeySwitcher(ring, k=k, generator=torch.Generator().manual_seed(61))
    relin_key = top.generate_hadamard_key(secret)
    for count in range(1, 6):
        active = RNSRing(ring.moduli[:count], ring.axes)
        switcher = KeySwitcher(active, k=k, key_moduli=top.parameters.key_moduli)
        engine = Engine(active, noise_std=0, generator=torch.Generator().manual_seed(count))
        small_secret = secret.at_level(count)
        message = active.generator("X") * 3 + 7
        ct = engine.encrypt(message, small_secret)
        result = switcher.relinearize(ct * ct, relin_key)
        error = engine.decrypt(result, small_secret) - message * message
        assert error.centered_coefficients().abs().max() < 300


@pytest.fixture(scope="module")
def approximate_setup():
    ring = RNSRing.for_parameters(2, 3, limb_count=6)
    secret = key(ring)
    switcher = KeySwitcher(ring, k=2, generator=torch.Generator().manual_seed(101))
    keys = generate_evaluation_keys(switcher, secret, rotations=[("X", 1), ("W", 1)])
    return ring, secret, Evaluator(ring, keys)


def test_approximate_evaluator_operations_and_levels(approximate_setup):
    ring, secret, evaluator = approximate_setup
    scale = ring.moduli[-1]
    encoder = MatrixEncoder(ring, scale=scale)
    generator = torch.Generator().manual_seed(71)
    left, right = [
        torch.randn(encoder.slot_shape, dtype=torch.complex128, generator=generator) / 3
        for _ in range(2)
    ]
    engine = Engine(ring, generator=generator)
    a, b = [
        ScaledCiphertext(engine.encrypt(encoder.encode(v), secret), scale) for v in (left, right)
    ]

    def check(result, expected, atol=2e-5):
        decoded = MatrixEncoder(result.ring, scale=result.scale).decode(
            Engine(result.ring).decrypt(result.ciphertext, secret.at_level(result.level))
        )
        torch.testing.assert_close(decoded, expected, atol=atol, rtol=atol)

    check(evaluator.add(a, b), left + right)
    check(evaluator.subtract(a, b), left - right)
    check(evaluator.add_plain(a, 0.25), left + 0.25)
    check(evaluator.multiply_plain(a, 0.125), left * 0.125)
    product = evaluator.multiply(a, b)
    assert product.level == 5
    check(product, left * right)
    check(evaluator.multiply(product, a), left * right * left)
    check(evaluator.transpose(a), left.transpose(-1, -2))
    check(evaluator.transpose(a, conjugate=True), left.mH)
    check(evaluator.conjugate(a), left.conj())
    for axis, dim in [("X", 1), ("Y", 2), ("W", 0)]:
        check(evaluator.rotate(a, 1, axis=axis), torch.roll(left, -1, dim))
    check(evaluator.matmul_plain(a, right), left @ right)
    check(evaluator.matmul(a, b), left @ right)
    check(evaluator.matmul_transposed(a, b), left @ right.transpose(-1, -2))
    check(evaluator.matmul_adjoint(a, b), left @ right.mH)
    check(evaluator.add(a, product), left + left * right)
    with pytest.raises(ValueError, match="rescaling level"):
        evaluator.multiply(a.at_level(1), b.at_level(1))
    with pytest.raises(ValueError):
        a.at_level(0)


def test_public_key_encryption():
    ring = RNSRing.for_parameters(2, 3, limb_count=3)
    engine = Engine(ring, generator=torch.Generator().manual_seed(12))
    secret = key(ring)
    public = engine.create_public_key(secret)
    encoder = MatrixEncoder(ring, scale=2**30)
    message = torch.ones(encoder.slot_shape, dtype=torch.complex128) * (0.2 + 0.3j)
    ct = engine.encrypt_public(encoder.encode(message), public)
    torch.testing.assert_close(
        encoder.decode(engine.decrypt(ct, secret)), message, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("kind", ["hadamard", "matrix"])
def test_integer_encoding_products_key_switching_and_modulus_switching(kind):
    ring = integer_ring(2, 3, 97, limb_count=4)
    encoder = IntegerEncoder(ring, 97)
    generator = torch.Generator().manual_seed(53)
    left, right = [
        torch.randint(5, encoder.slot_shape, dtype=torch.int32, generator=generator)
        for _ in range(2)
    ]
    torch.testing.assert_close(encoder.decode(encoder.encode(left)), left)
    secret = key(ring)
    engine = IntegerEngine(ring, 97, generator=generator)
    switcher = KeySwitcher(ring, k=2, plaintext_modulus=97, generator=generator)
    a, b = [engine.encrypt(encoder.encode(v), secret) for v in (left, right)]
    if kind == "hadamard":
        ct = switcher.relinearize(a * b, switcher.generate_hadamard_key(secret))
        expected = (left * right) % 97
    else:
        ct = switcher.relinearize_matrix(
            engine.matrix_multiply(a, b), switcher.generate_matrix_keys(secret)
        )
        expected = (
            torch.stack(
                [
                    left[..., branch].to(torch.int64)
                    @ right[..., branch].to(torch.int64).transpose(-1, -2)
                    for branch in range(2)
                ],
                -1,
            )
            .remainder(97)
            .to(torch.int32)
        )
    torch.testing.assert_close(encoder.decode(engine.decrypt(ct, secret)), expected)
    switched = engine.mod_switch(ct)
    active = switched.ring
    recovered = IntegerEngine(active, 97).decrypt(switched, secret.at_level(active.limb_count))
    torch.testing.assert_close(encoder.decode(recovered), expected)
    public = engine.create_public_key(secret)
    encrypted = engine.encrypt_public(encoder.encode(left), public)
    torch.testing.assert_close(encoder.decode(engine.decrypt(encrypted, secret)), left)


def test_mod_raise_centers_before_extending():
    ring = RNSRing.for_parameters(2, 3, limb_count=3)
    small = RNSRing(ring.moduli[:1], ring.axes)
    p = small.modulus
    original = small.constant(-p // 2) + 11 * small.generator("X")
    lifted = original.mod_raise(ring)
    assert lifted.at_level(1) == original
    assert lifted.reconstruct(centered=True) == original.reconstruct(centered=True)
    with pytest.raises(ValueError):
        original.mod_raise(RNSRing(ring.moduli[1:], ring.axes))
    with pytest.raises(ValueError):
        embed(original, RNSRing(ring.moduli, ring.axes))


@pytest.mark.parametrize("n,p", [(2, 3), (2, 9), (2, 27)])
def test_auxiliary_trace_product_and_encoding(n, p):
    full = RNSRing.for_parameters(n, p, limb_count=3)
    ring = RNSRing.auxiliary(full.moduli, n, p)
    encoder = MatrixEncoder(ring, scale=2**30)
    generator = torch.Generator().manual_seed(62)
    a, b = [
        torch.randn(encoder.slot_shape, dtype=torch.complex128, generator=generator) / 5
        for _ in range(2)
    ]
    left, right = encoder.encode(a), encoder.encode(b)
    # Encoder order is (W,X,V); W/V matrix batches have shape (X,W,V).
    expected = (a.permute(1, 0, 2) @ b.permute(1, 0, 2)).permute(1, 0, 2)
    result = auxiliary_matrix_multiply(left, right)
    decoded = MatrixEncoder(ring, scale=encoder.scale**2).decode(result)
    torch.testing.assert_close(decoded, expected, atol=2e-7, rtol=2e-7)
    # Independent direct polynomial evaluation, with all primitive W/V roots.
    x = torch.exp(
        2j * math.pi * torch.tensor([pow(5, j, 4 * n) for j in range(n)]).double() / (4 * n)
    )
    roots = torch.exp(
        2j * math.pi * torch.tensor([pow(2, j, p) for j in range(2 * p // 3)]).double() / p
    )
    coefficients = torch.tensor(result.reconstruct(), dtype=torch.float64).reshape(ring.shape)
    values = torch.complex(coefficients[..., 0], coefficients[..., 1]) / encoder.scale**2
    reference = torch.einsum(
        "ar,bs,ct,rst->bac",
        x[:, None] ** torch.arange(n),
        roots[:, None] ** torch.arange(len(roots)),
        roots[:, None] ** torch.arange(len(roots)),
        values,
    )
    torch.testing.assert_close(reference, expected, atol=2e-7, rtol=2e-7)


def test_auxiliary_big_key_and_plaintext_trace_commute_with_decryption():
    full = RNSRing.for_parameters(2, 9, limb_count=4)
    ring = RNSRing.auxiliary(full.moduli, 2, 9)
    secret = key(ring)
    encoder = MatrixEncoder(ring, scale=2**30)
    values = (
        torch.randn(
            encoder.slot_shape, dtype=torch.complex128, generator=torch.Generator().manual_seed(72)
        )
        / 5
    )
    engine = Engine(ring)
    ct = engine.encrypt(encoder.encode(values), secret)
    from torch_gl import swap_axes

    switcher = KeySwitcher(ring, k=2)
    eval_key = switcher.generate_key(swap_axes(embed(secret, ring), "W", "V"), secret, kind="big")
    a, b = swap_axes(ct.a, "W", "V"), swap_axes(ct.b, "W", "V")
    switched = switcher.switch_big(a, eval_key)
    from torch_gl import Ciphertext

    transformed = Ciphertext(b + switched.b, switched.a)
    multiplier = torch.randn(encoder.slot_shape, dtype=torch.complex128) / 5
    polynomial = encoder.encode(multiplier)
    product = Ciphertext(
        auxiliary_matrix_multiply(transformed.b, polynomial),
        auxiliary_matrix_multiply(transformed.a, polynomial),
    )
    decoded = MatrixEncoder(ring, scale=2**60).decode(engine.decrypt(product, secret))
    expected = (values.transpose(0, 2).permute(1, 0, 2) @ multiplier.permute(1, 0, 2)).permute(
        1, 0, 2
    )
    torch.testing.assert_close(decoded, expected, atol=2e-5, rtol=2e-5)
