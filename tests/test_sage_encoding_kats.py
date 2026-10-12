"""Read frozen Sage vectors only; never import Sage or regenerate fixtures."""

import json
import math
from pathlib import Path

import pytest
import torch

from torch_gl import Engine, MatrixEncoder, RNSPoly, RNSRing

VECTORS = json.loads((Path(__file__).parent / "data/sage_encoding.json").read_text())
CASES = VECTORS["cases"]


def complex_tensor(pairs, shape):
    components = torch.tensor(
        [[float(real), float(imag)] for real, imag in pairs], dtype=torch.float64
    )
    return torch.complex(components[:, 0], components[:, 1]).reshape(shape)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_encoding_matches_frozen_sage_canonical_inverse(case):
    ring = RNSRing.for_parameters(case["n"], case["p"], variables=tuple(case["variables"]))
    encoder = MatrixEncoder(ring, scale=case["scale"])
    inputs = complex_tensor(case["input_slots"], case["slot_shape"]) / case["slot_denominator"]
    reference = complex_tensor(case["inverse_coefficients"], case["coefficient_shape"][:-1])
    torch.testing.assert_close(encoder._interpolate(inputs), reference, atol=2e-14, rtol=2e-13)
    encoded = encoder.encode(inputs)
    # Exact rounded Gaussian integers and every RNS limb must match Sage.
    assert encoded.reconstruct() == tuple(case["encoded_coefficients"])
    for limb in encoded.limbs:
        expected = torch.tensor(
            [value % limb.ring.q for value in case["encoded_coefficients"]], dtype=torch.int32
        ).reshape(case["coefficient_shape"])
        assert torch.equal(limb.coefficients, expected)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_decoding_matches_frozen_sage_primitive_root_evaluation(case):
    ring = RNSRing.for_parameters(case["n"], case["p"], variables=tuple(case["variables"]))
    encoder = MatrixEncoder(ring, scale=case["scale"])
    # Build the fixture's large signed coefficients through Python-int residues,
    # so this KAT does not rely on encode() or a first-limb lift.
    polynomial = RNSPoly(
        ring,
        tuple(
            limb_ring.polynomial(
                torch.tensor(
                    [value % limb_ring.q for value in case["decode_coefficients"]],
                    dtype=torch.int32,
                ).reshape(case["coefficient_shape"])
            )
            for limb_ring in ring.rings
        ),
    )
    assert polynomial.reconstruct() == tuple(case["decode_coefficients"])
    expected = complex_tensor(case["decoded_slots"], case["slot_shape"])
    torch.testing.assert_close(encoder.decode(polynomial), expected, atol=3e-11, rtol=1e-12)


def test_sage_encoded_message_encrypts_decrypts_and_decodes():
    case = next(case for case in CASES if case["name"] == "full_n4_p9")
    ring = RNSRing.for_parameters(case["n"], case["p"])
    encoder = MatrixEncoder(ring, scale=case["scale"])
    engine = Engine(ring, generator=torch.Generator().manual_seed(case["seed"]))
    inputs = complex_tensor(case["input_slots"], case["slot_shape"]) / case["slot_denominator"]
    encoded = encoder.encode(inputs)
    assert encoded.reconstruct() == tuple(case["encoded_coefficients"])
    decrypted = engine.decrypt(engine.encrypt(encoded), engine.secret_key)
    noise = (decrypted - encoded).reconstruct()
    bound = (sum(abs(value) for value in noise) + math.prod(ring.shape[:-1]) / math.sqrt(2)) / case[
        "scale"
    ]
    assert (encoder.decode(decrypted) - inputs).abs().max().item() <= bound + 1e-12
