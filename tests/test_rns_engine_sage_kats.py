import pytest
import torch

from torch_gl import RNSPoly, RNSRing


def test_rns_arithmetic_preserves_the_saved_sage_limb_product(sage_vectors):
    # Existing oracle coefficients can be lifted into other prime limbs;
    # the original prime's product must still match the frozen Sage result.
    case = next(vector for vector in sage_vectors["cases"] if vector["name"] == "full_n4_p9_native")
    ring = RNSRing.multivariate_poly((case["q"], 1297), case["n"], case["p"])
    left = ring.polynomial(torch.tensor(case["left"]).reshape(case["shape"]))
    right = ring.polynomial(torch.tensor(case["right"]).reshape(case["shape"]))
    product = left * right
    assert torch.equal(
        product.limbs[0].coefficients, torch.tensor(case["product"]).reshape(case["shape"])
    )


def test_centered_lift_rejects_inconsistent_noise_limbs():
    ring = RNSRing.for_parameters(2, 9)
    mismatched = RNSPoly(
        ring, tuple(base_ring.constant(index) for index, base_ring in enumerate(ring.rings))
    )
    with pytest.raises(ValueError):
        mismatched.centered_coefficients()
