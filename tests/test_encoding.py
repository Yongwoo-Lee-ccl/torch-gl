import math

import pytest
import torch

from torch_gl import Engine, MatrixEncoder, RNSPoly, RNSRing, create_secret_key

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def canonical_embedding_matrix(n, p):
    """Full evaluation matrix, rows (ell,j,k), columns (r,s,t)."""
    X_powers = torch.tensor([pow(5, slot, 4 * n) for slot in range(n)])
    W_powers = torch.tensor([pow(2, slot, p) for slot in range(2 * p // 3)])
    X_phases = (X_powers[:, None] * torch.arange(n)[None, :]).remainder(4 * n)
    W_phases = (W_powers[:, None] * torch.arange(2 * p // 3)[None, :]).remainder(p)
    X_matrix = torch.exp(2j * math.pi * X_phases.double() / (4 * n))
    W_matrix = torch.exp(2j * math.pi * W_phases.double() / p)
    dimension = n * n * (2 * p // 3)
    return torch.einsum("jr,ks,lt->ljkrst", X_matrix, X_matrix, W_matrix).reshape(
        dimension, dimension
    )


@pytest.mark.parametrize("n,p", [(1, 3), (2, 9), (4, 27), (1, 729)])
def test_encoding_is_inverse_of_dense_canonical_embedding(n, p):
    ring = RNSRing.for_parameters(n, p)
    encoder = MatrixEncoder(ring)
    assert encoder.scale == 2**40
    generator = torch.Generator().manual_seed(1935)
    slots = torch.randn(encoder.slot_shape, dtype=torch.complex128, generator=generator)
    embedding = canonical_embedding_matrix(n, p)
    # Solve E*c=slots independently. No FFTs, ideal-reduction helpers, or
    # production interpolation functions enter this reference computation.
    reference = torch.linalg.solve(embedding, slots.reshape(-1)).reshape(ring.shape[:-1])
    interpolated = encoder._interpolate(slots)
    torch.testing.assert_close(interpolated, reference, atol=2e-14, rtol=2e-13)
    encoded = encoder.encode(slots)
    integer_coefficients = torch.tensor(encoded.reconstruct(), dtype=torch.float64).reshape(
        ring.shape
    )
    recovered_coefficients = torch.complex(
        integer_coefficients[..., 0], integer_coefficients[..., 1]
    )
    # Rounding contributes at most sqrt(1/2) per Gaussian coefficient, plus
    # the measured discrepancy between floating implementations of E^-1.
    discrepancy = (interpolated - reference).abs().max().item() * encoder.scale
    assert (recovered_coefficients - reference * encoder.scale).abs().max() <= (
        1 / math.sqrt(2) + discrepancy + 1e-3
    )
    independently_decoded = (embedding @ recovered_coefficients.reshape(-1)) / encoder.scale
    torch.testing.assert_close(independently_decoded.reshape(slots.shape), slots, atol=1e-9, rtol=0)


def direct_evaluation(coefficients, n, p):
    """Independent dense evaluation at the paper's roots, without FFTs."""
    X_exponents = torch.tensor([pow(5, slot, 4 * n) for slot in range(n)])
    W_exponents = torch.tensor([pow(2, slot, p) for slot in range(2 * p // 3)])
    X_matrix = torch.exp(
        2j * math.pi * X_exponents[:, None].double() * torch.arange(n)[None, :] / (4 * n)
    )
    W_matrix = torch.exp(
        2j * math.pi * W_exponents[:, None].double() * torch.arange(2 * p // 3)[None, :] / p
    )
    return torch.einsum("jr,ks,lt,rst->ljk", X_matrix, X_matrix, W_matrix, coefficients)


@pytest.mark.parametrize("n,p", [(1, 3), (4, 9), (8, 27)])
def test_encode_and_decode_match_independent_paper_evaluation(n, p):
    ring = RNSRing.for_parameters(n, p)
    generator = torch.Generator().manual_seed(31)
    coefficients = torch.randint(-100, 101, ring.shape, generator=generator, dtype=torch.int32)
    complex_coefficients = torch.complex(
        coefficients[..., 0].double(), coefficients[..., 1].double()
    )
    expected = direct_evaluation(complex_coefficients, n, p)
    encoder = MatrixEncoder(ring, scale=1)
    polynomial = ring.polynomial(coefficients)
    torch.testing.assert_close(encoder.decode(polynomial), expected, atol=1e-9, rtol=1e-12)
    assert encoder.encode(expected) == polynomial


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale", [2**20, 2**40, 2**70])
def test_encoded_matrices_encrypt_decrypt_decode(device, scale):
    ring = RNSRing.for_parameters(4, 9)
    encoder = MatrixEncoder(ring, scale=scale, device=device)
    generator = torch.Generator(device=device).manual_seed(1935)
    engine = Engine(ring, generator=generator, device=device)
    secret_key = create_secret_key(
        ring, generator=torch.Generator(device=engine.device).manual_seed((1935) + 100000)
    )
    matrices = torch.randn(
        encoder.slot_shape, dtype=torch.complex128, device=device, generator=generator
    )
    encoded = encoder.encode(matrices)
    assert isinstance(encoded, RNSPoly)
    assert all(limb.coefficients.dtype == torch.int32 for limb in encoded.limbs)
    if scale > 2**30:
        assert max(abs(value) for value in encoded.reconstruct()) > ring.moduli[0] // 2
    decrypted = engine.decrypt(engine.encrypt(encoded, secret_key), secret_key)
    # Worst-case l1 evaluation bound for rounding plus the actual sampled noise.
    noise = (decrypted - encoded).centered_coefficients()
    coefficient_count = math.prod(ring.shape[:-1])
    bound = (float(noise.abs().sum()) + coefficient_count / math.sqrt(2)) / scale
    error = (encoder.decode(decrypted) - matrices).abs().max().item()
    assert error <= bound + 1e-12


@pytest.mark.parametrize("value", [2**40, -(2**40), 2**70, -(2**70)])
def test_large_integer_coefficients_are_centered_modulo_full_Q(value):
    ring = RNSRing.for_parameters(2, 9)
    encoder = MatrixEncoder(ring, scale=1)
    matrices = torch.full(encoder.slot_shape, complex(value, -value), dtype=torch.complex128)
    encoded = encoder.encode(matrices)
    reconstructed = encoded.reconstruct()
    assert reconstructed[:2] == (value, -value)
    assert all(coefficient == 0 for coefficient in reconstructed[2:])
    with pytest.raises(ValueError, match="first-limb"):
        encoded.centered_coefficients()
    torch.testing.assert_close(encoder.decode(encoded), matrices, atol=0, rtol=0)
    engine = Engine(ring, generator=torch.Generator().manual_seed(40))
    secret_key = create_secret_key(
        ring, generator=torch.Generator(device=engine.device).manual_seed((40) + 100000)
    )
    decrypted = engine.decrypt(engine.encrypt(encoded, secret_key), secret_key)
    assert max(abs(error) for error in (decrypted - encoded).reconstruct()) <= 32
    # Beyond 2**53, double rounding can dominate the few-unit encryption error.
    torch.testing.assert_close(encoder.decode(decrypted), matrices, atol=128, rtol=1e-14)


@pytest.mark.parametrize("variables", [("X", "W"), ("W", "Y", "X"), ("W",)])
def test_configurable_variable_order_and_smaller_ring(variables):
    ring = RNSRing.for_parameters(4, 27, variables=variables)
    encoder = MatrixEncoder(ring)
    matrices = torch.randn(encoder.slot_shape, dtype=torch.complex128)
    torch.testing.assert_close(
        encoder.decode(encoder.encode(matrices)), matrices, atol=1e-4, rtol=0
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("left_scale,right_scale", [(2**40, 2**40), (2**30, 2**40)])
@pytest.mark.parametrize(
    "n,p,variables",
    [
        (4, 9, ("X", "Y", "W")),
        (8, 27, ("X", "W")),
        (4, 27, ("W", "Y", "X")),
        (2, 729, ("X", "Y", "W")),
    ],
)
def test_encoded_plaintext_product_decodes_to_elementwise_slot_product(
    device, left_scale, right_scale, n, p, variables
):
    ring = RNSRing.for_parameters(n, p, limb_count=4, variables=variables)
    left_encoder = MatrixEncoder(ring, scale=left_scale, device=device)
    right_encoder = MatrixEncoder(ring, scale=right_scale, device=device)
    product_decoder = MatrixEncoder(ring, scale=left_scale * right_scale, device=device)
    generator = torch.Generator(device=device).manual_seed(1935)
    left_slots = torch.randn(
        left_encoder.slot_shape, dtype=torch.complex128, device=device, generator=generator
    )
    right_slots = torch.randn(
        right_encoder.slot_shape, dtype=torch.complex128, device=device, generator=generator
    )
    left_encoded = left_encoder.encode(left_slots)
    right_encoded = right_encoder.encode(right_slots)
    # Pure plaintext polynomial multiplication: no Engine, ciphertext, or noise.
    product = left_encoded * right_encoded
    left_l1 = sum(abs(value) for value in left_encoded.reconstruct())
    right_l1 = sum(abs(value) for value in right_encoded.reconstruct())
    # Each reduced monomial contributes to at most two coefficients under Phi_p.
    # This sufficient bound rules out aggregate-modulus wraparound for the test.
    assert 2 * left_l1 * right_l1 < ring.modulus // 2
    decoded = product_decoder.decode(product)
    # Separate rounding from the embedding's algebraic multiplication property.
    rounded_slot_product = left_encoder.decode(left_encoded) * right_encoder.decode(right_encoded)
    torch.testing.assert_close(decoded, rounded_slot_product, atol=2e-12, rtol=2e-12)
    coefficient_count = math.prod(ring.shape[:-1])
    left_rounding_bound = coefficient_count / (math.sqrt(2) * left_scale)
    right_rounding_bound = coefficient_count / (math.sqrt(2) * right_scale)
    product_error_bound = (
        left_slots.abs().max().item() * right_rounding_bound
        + right_slots.abs().max().item() * left_rounding_bound
        + left_rounding_bound * right_rounding_bound
    )
    assert (decoded - left_slots * right_slots).abs().max().item() <= product_error_bound + 1e-12


def test_encoding_rejects_invalid_inputs_and_aggregate_overflow():
    ring = RNSRing.for_parameters(2, 9, limb_count=1)
    for scale in (0, -1, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="scale"):
            MatrixEncoder(ring, scale=scale)
    encoder = MatrixEncoder(ring)
    with pytest.raises(ValueError, match="shape"):
        encoder.encode(torch.ones(3))
    with pytest.raises(ValueError, match="finite"):
        encoder.encode(torch.full(encoder.slot_shape, float("nan")))
    with pytest.raises(ValueError, match="aggregate"):
        encoder.encode(torch.full(encoder.slot_shape, float(ring.modulus)))
    with pytest.raises(TypeError):
        encoder.encode([1, 2])
    with pytest.raises(TypeError):
        encoder.decode(torch.ones(encoder.slot_shape))
    with pytest.raises(ValueError, match="ring"):
        encoder.decode(RNSRing.for_parameters(4, 9).one())
    with pytest.raises(ValueError, match="power-of-three"):
        MatrixEncoder(RNSRing.for_parameters(2, 25))
