import math
from dataclasses import fields

import pytest
import torch

from torch_gl import (
    Ciphertext,
    Engine,
    MatrixEncoder,
    MatrixProductCiphertext,
    QuadraticCiphertext,
    RNSPoly,
    RNSRing,
    create_secret_key,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def polynomial_from_python_integers(ring, coefficients, device):
    return RNSPoly(
        ring,
        tuple(
            limb_ring.polynomial(
                torch.tensor(
                    [value % limb_ring.q for value in coefficients],
                    dtype=torch.int32,
                    device=device,
                ).reshape(ring.shape)
            )
            for limb_ring in ring.rings
        ),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("variables", [("X", "Y", "W"), ("W", "X"), ()])
def test_rescale_matches_python_floor_division_for_large_signed_coefficients(device, variables):
    ring = RNSRing.for_parameters(2, 9, limb_count=6, variables=variables)
    last_modulus = ring.moduli[-1]
    boundary_values = [
        0,
        1,
        -1,
        last_modulus - 1,
        last_modulus,
        last_modulus + 1,
        -last_modulus - 1,
        -last_modulus,
        -last_modulus + 1,
        2**100 + last_modulus // 2,
        -(2**100) - last_modulus // 2,
        ring.modulus // 2,
        -(ring.modulus // 2),
        ring.modulus - 1,
        ring.modulus,
        -ring.modulus - 1,
    ]
    # Constants have two channels; exercise every boundary pair in that case.
    batches = (
        [boundary_values[index : index + 2] for index in range(0, len(boundary_values), 2)]
        if not variables
        else [
            [
                boundary_values[index % len(boundary_values)]
                for index in range(math.prod(ring.shape))
            ]
        ]
    )
    for coefficients in batches:
        polynomial = polynomial_from_python_integers(ring, coefficients, device)
        original = tuple(limb.coefficients for limb in polynomial.limbs)
        divided = polynomial.rescale()
        assert divided.ring.moduli == ring.moduli[:-1]
        assert divided.ring.axes == ring.axes
        assert divided.device == polynomial.device
        quotient = [value // last_modulus for value in coefficients]
        assert divided.reconstruct(centered=False) == tuple(
            value % divided.ring.modulus for value in quotient
        )
        for limb in divided.limbs:
            assert limb.coefficients.dtype == torch.int32
            expected = torch.tensor(
                [value % limb.ring.q for value in quotient], dtype=torch.int32, device=device
            ).reshape(ring.shape)
            assert torch.equal(limb.coefficients, expected)
        assert all(
            torch.equal(limb.coefficients, previous)
            for limb, previous in zip(polynomial.limbs, original, strict=True)
        )


@pytest.mark.parametrize("moduli", [(13, 37, 73), (73, 13, 37), (73, 37, 13)])
def test_rescale_uses_last_limb_in_supplied_order_and_can_repeat(moduli):
    ring = RNSRing.multivariate_poly(moduli, 1, 3, variables=())
    values = (-9876, 34567)
    polynomial = ring.constant(values)
    for divisor in reversed(moduli[1:]):
        polynomial = polynomial.rescale()
        values = tuple(value // divisor for value in values)
        assert polynomial.reconstruct(centered=False) == tuple(
            value % polynomial.ring.modulus for value in values
        )
    assert polynomial.ring.moduli == moduli[:1]
    with pytest.raises(ValueError, match="at least two"):
        polynomial.rescale()
    with pytest.raises(ValueError, match="at least two"):
        polynomial.drop_last_limb()


def test_key_projection_drops_modulus_without_dividing():
    ring = RNSRing.for_parameters(2, 9, limb_count=3)
    secret_key = create_secret_key(ring, torch.Generator().manual_seed(11))
    projected = secret_key.drop_last_limb()
    assert projected.ring.moduli == ring.moduli[:-1]
    assert projected.ring.axes == secret_key.ring.axes
    assert torch.equal(projected.centered_coefficients(), secret_key.centered_coefficients())
    assert projected != secret_key.rescale()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("noise_std", [0, 3.2])
def test_encoded_ciphertext_rescale_preserves_slots_with_reduced_scale(device, noise_std):
    ring = RNSRing.for_parameters(2, 9, limb_count=3)
    engine = Engine(
        ring,
        device=device,
        noise_std=noise_std,
        generator=torch.Generator(device=device).manual_seed(101),
    )
    # With s=1, coefficient rounding adds only the carry from floor(b)+floor(a).
    secret_key = ring.one(device=device)
    encoder = MatrixEncoder(ring, device=device)
    generator = torch.Generator(device=device).manual_seed(301)
    slots = torch.randn(
        encoder.slot_shape, dtype=torch.complex128, device=device, generator=generator
    )
    ciphertext = engine.encrypt(encoder.encode(slots), secret_key)
    original_decrypted = engine.decrypt(ciphertext, secret_key)
    scaled = engine.rescale(ciphertext)
    assert isinstance(scaled, Ciphertext)
    lower_engine = Engine(scaled.b.ring, device=device)
    lower_key = secret_key.drop_last_limb()
    decrypted = lower_engine.decrypt(scaled, lower_key)
    rounding_carry = (decrypted - original_decrypted.rescale()).centered_coefficients()
    assert bool(((rounding_carry == -1) | (rounding_carry == 0)).all())
    reduced_scale = encoder.scale / ring.moduli[-1]
    decoder = MatrixEncoder(scaled.b.ring, scale=reduced_scale, device=device)
    # Each Gaussian coefficient loses <2 per channel from the two floors.
    rounding_bound = 2 * math.sqrt(2) * math.prod(ring.shape[:-1]) / reduced_scale
    recovered = decoder.decode(decrypted)
    torch.testing.assert_close(
        recovered, encoder.decode(original_decrypted), atol=rounding_bound, rtol=0
    )
    torch.testing.assert_close(recovered, slots, atol=rounding_bound + 1e-8, rtol=0)
    assert engine.ring == ring
    assert ciphertext.b.ring == ring


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("product_kind", ["hadamard", "matrix"])
@pytest.mark.parametrize("noise_std", [0, 3.2])
def test_rescaled_encrypted_products_decode_at_product_scale_divided_by_last_prime(
    device, product_kind, noise_std
):
    ring = RNSRing.for_parameters(2, 9, limb_count=4)
    engine = Engine(
        ring,
        device=device,
        noise_std=noise_std,
        generator=torch.Generator(device=device).manual_seed(101),
    )
    secret_key = create_secret_key(ring, torch.Generator(device=device).manual_seed(11))
    encoder = MatrixEncoder(ring, device=device)
    generator = torch.Generator(device=device).manual_seed(301)
    left, right = [
        torch.randn(encoder.slot_shape, dtype=torch.complex128, device=device, generator=generator)
        for _ in range(2)
    ]
    left_ct, right_ct = (
        engine.encrypt(encoder.encode(slots), secret_key) for slots in (left, right)
    )
    if product_kind == "hadamard":
        product = engine.multiply(left_ct, right_ct)
        expected, expected_type = left * right, QuadraticCiphertext
    else:
        product = engine.matrix_multiply(left_ct, right_ct)
        expected, expected_type = left @ right.transpose(-1, -2), MatrixProductCiphertext
    scaled = engine.rescale(product)
    assert isinstance(scaled, expected_type)
    # Check every component against an independent integer quotient oracle.
    for field in fields(product):
        original, divided = getattr(product, field.name), getattr(scaled, field.name)
        assert divided.reconstruct(centered=False) == tuple(
            (value // ring.moduli[-1]) % divided.ring.modulus
            for value in original.reconstruct(centered=False)
        )
        assert all(limb.coefficients.dtype == torch.int32 for limb in divided.limbs)
    lower_ring = getattr(scaled, fields(scaled)[0].name).ring
    lower_engine = Engine(lower_ring, device=device, compiled=True, backend="eager")
    lower_key = secret_key.drop_last_limb()
    decoder = MatrixEncoder(lower_ring, scale=encoder.scale**2 / ring.moduli[-1], device=device)
    recovered = decoder.decode(lower_engine.decrypt(scaled, lower_key))
    torch.testing.assert_close(recovered, expected, atol=2e-8, rtol=2e-9)


def test_engine_rescale_validates_ciphertext_context_and_remaining_chain():
    ring = RNSRing.for_parameters(2, 9, limb_count=2)
    engine = Engine(ring)
    with pytest.raises(TypeError, match="ciphertext"):
        engine.rescale(ring.one())
    lower_ring = ring.one().drop_last_limb().ring
    lower_ct = Ciphertext(lower_ring.one(), lower_ring.one())
    with pytest.raises(ValueError, match="engine's ring"):
        engine.rescale(lower_ct)
    with pytest.raises(ValueError, match="at least two"):
        Engine(lower_ring).rescale(lower_ct)
