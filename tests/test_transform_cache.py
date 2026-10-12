import pytest
import torch

from torch_gl import (
    CyclotomicNTTPlan,
    GaussianNTTPlan,
    NTTPlan,
    PolynomialRing,
    clear_transform_cache,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "plan,shape,dim",
    [
        (NTTPlan(1297, 108), (3, 108), 1),
        (CyclotomicNTTPlan(1297, 81), (4, 54, 2), 1),
        (GaussianNTTPlan(193, 16), (4, 16, 2), 1),
    ],
)
def test_warm_transforms_reuse_device_tables(plan, shape, dim, device, monkeypatch):
    clear_transform_cache()
    plan.prepare(device)
    coefficients = (
        torch.arange(torch.tensor(shape).prod().item(), device=device).reshape(shape) % plan.q
    )

    def fail_table_construction(*args, **kwargs):
        raise AssertionError("warm transform recreated a host/device table")

    monkeypatch.setattr(torch, "tensor", fail_table_construction)
    slots = plan.forward(coefficients, dim=dim)
    recovered = plan.inverse(slots, dim=dim)
    assert torch.equal(recovered, coefficients)


def test_ring_reuses_plan_objects():
    ring = PolynomialRing.multivariate_poly(433, 4, 27)
    assert ring.transform_plans is ring.transform_plans


def test_30bit_limb_parameter_limit():
    ring = PolynomialRing.multivariate_poly(754974721, 4, 9)
    assert ring.generator("W") ** 9 == ring.one()
    with pytest.raises(ValueError, match="q must be an odd prime"):
        NTTPlan(2013265921, 16)
