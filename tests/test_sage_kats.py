import json
from pathlib import Path

import pytest
import torch

from torch_gl import PolynomialRing, QuotientAxis

CASES = json.loads((Path(__file__).parent / "data/sage_multiplication.json").read_text())["cases"]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_multiplication_against_saved_sage_vectors(case, device):
    if case["name"] == "custom_gaussian_ideal":
        axes = tuple(
            QuotientAxis(
                axis_info["name"], tuple(tuple(coefficient) for coefficient in axis_info["modulus"])
            )
            for axis_info in case["axes"]
        )
        ring = PolynomialRing(case["q"], axes)
    else:
        ring = PolynomialRing.multivariate_poly(
            case["q"], case["n"], case["p"], variables=tuple(case["variables"])
        )
    left, right, expected = (
        torch.tensor(case[key], dtype=torch.int64, device=device).reshape(case["shape"])
        for key in ("left", "right", "product")
    )
    product = ring.polynomial(left) * ring.polynomial(right)
    assert torch.equal(product.coefficients, expected)


def test_native_path_has_compact_dimensions():
    ring = PolynomialRing.multivariate_poly(1297, 4, 81)
    assert ring.ntt_shape == (4, 4, 54)
    assert ring.shape == (4, 4, 54, 2)


def test_native_and_padded_cyclotomic_paths_agree(sage_vectors):
    case = next(
        vector_case
        for vector_case in sage_vectors["cases"]
        if vector_case["name"] == "full_n4_p9_native"
    )
    native = PolynomialRing.multivariate_poly(case["q"], case["n"], case["p"])
    # Strip optimization metadata to exercise independent padded W convolution.
    padded = PolynomialRing(
        native.q,
        tuple(QuotientAxis(axis.name, axis.modulus) for axis in native.axes),
    )
    tensors = [torch.tensor(case[key]).reshape(case["shape"]) for key in ("left", "right")]
    native_product = native.polynomial(tensors[0]) * native.polynomial(tensors[1])
    padded_product = padded.polynomial(tensors[0]) * padded.polynomial(tensors[1])
    assert padded.ntt_shape[-1] == 16
    assert torch.equal(native_product.coefficients, padded_product.coefficients)
