import pytest
import torch

from torch_gl import PolynomialRing, QuotientAxis


@pytest.mark.parametrize("name", ["full_n2_p3", "full_n4_p9_native", "w_p27_native"])
def test_full_compiled_product_against_saved_sage(sage_vectors, name):
    case = next(vector for vector in sage_vectors["cases"] if vector["name"] == name)
    ring = PolynomialRing.multivariate_poly(
        case["q"], case["n"], case["p"], variables=tuple(case["variables"])
    )
    tensors = [
        torch.tensor(case[key]).reshape(case["shape"]) for key in ("left", "right", "product")
    ]
    left, right = ring.polynomial(tensors[0]), ring.polynomial(tensors[1])
    product = left.multiply(right, compiled=True, backend="eager")
    assert torch.equal(product.coefficients, tensors[2])


def test_owned_arithmetic_results_do_not_alias_inputs():
    ring = PolynomialRing.multivariate_poly(97, 2, 3)
    coefficients = torch.arange(16).reshape(ring.shape)
    left = ring.polynomial(coefficients)
    right = ring.one()
    product = left * right
    product._coefficients.zero_()
    assert torch.equal(left.coefficients, coefficients)
    assert right == ring.one()


def test_compile_rejects_general_padded_ideals():
    ring = PolynomialRing(97, (QuotientAxis("T", (1, 0, 0, 1)),))
    with pytest.raises(ValueError, match="native transforms"):
        ring.one().multiply(ring.one(), compiled=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_native_compiled_cuda_product(sage_vectors):
    case = next(vector for vector in sage_vectors["cases"] if vector["name"] == "full_n4_p9_native")
    ring = PolynomialRing.multivariate_poly(case["q"], case["n"], case["p"])
    tensors = [
        torch.tensor(case[key], device="cuda").reshape(case["shape"])
        for key in ("left", "right", "product")
    ]
    result = ring.polynomial(tensors[0]).multiply(ring.polynomial(tensors[1]), compiled=True)
    assert torch.equal(result.coefficients, tensors[2])
