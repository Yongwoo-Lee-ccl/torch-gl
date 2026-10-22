import pytest
import torch

from torch_gl import PolynomialRing
from torch_gl._gaussian import gaussian_boundary
from torch_gl._number_theory import root_of_unity


@pytest.mark.parametrize("q", [97, 1297, 268118017, 1073655809])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_gaussian_boundaries_against_integer_oracle(q, compiled, contiguous, monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("TORCH_GL_COMPILE", "1" if compiled else "0")
    try:
        root = root_of_unity(q, 4)
        boundary = gaussian_boundary(q, root)
        values = torch.randint(
            q, (64, 64, 2), dtype=torch.int32, generator=torch.Generator().manual_seed(37)
        )
        values[0, 0] = q - 1
        if not contiguous:
            values = values.transpose(0, 1)
        original = values.clone()
        flat = values.reshape(-1, 2).tolist()
        expected = torch.tensor(
            [[(a + root * b) * (1 << 31) % q, (a - root * b) * (1 << 31) % q] for a, b in flat],
            dtype=torch.int32,
        ).reshape(values.shape)
        slots = boundary.split(values)
        assert torch.equal(slots, expected)
        assert torch.equal(boundary.join(slots), values)
        assert torch.equal(values, original)
        # An independent arbitrary Montgomery split input checks join scaling.
        inverse_R = pow(1 << 31, -1, q)
        arbitrary = values
        expected_join = torch.tensor(
            [
                [
                    (a + b) * pow(2, -1, q) * inverse_R % q,
                    (a - b) * pow(2 * root, -1, q) * inverse_R % q,
                ]
                for a, b in flat
            ],
            dtype=torch.int32,
        ).reshape(values.shape)
        assert torch.equal(boundary.join(arbitrary), expected_join)
        opposite = gaussian_boundary(q, q - root)
        assert torch.equal(opposite.split(values), slots.flip(-1))
    finally:
        torch.set_num_threads(previous)


def test_native_gaussian_product_without_imaginary_root():
    ring = PolynomialRing.multivariate_poly(7, 1, 3, variables=("W",))
    left = ring.polynomial(torch.tensor([[2, 3], [1, 0]], dtype=torch.int32))
    right = ring.polynomial(torch.tensor([[6, 2], [2, 0]], dtype=torch.int32))
    assert torch.equal(
        (left * right).coefficients, torch.tensor([[4, 1], [1, 1]], dtype=torch.int32)
    )


def test_gaussian_preparation_reuses_graphs_across_moduli(monkeypatch):
    from torch._dynamo.utils import counters

    from torch_gl._compilation import prepare_cpu_kernels

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("TORCH_GL_COMPILE", "1")
    try:
        prepare_cpu_kernels((97, 1297))
        before = counters["stats"]["unique_graphs"]
        for q in (433, 1073655809):
            boundary = gaussian_boundary(q, root_of_unity(q, 4))
            values = torch.full((7777, 2), q - 1, dtype=torch.int32)
            assert torch.equal(boundary.join(boundary.split(values)), values)
        assert counters["stats"]["unique_graphs"] == before
    finally:
        torch.set_num_threads(previous)
