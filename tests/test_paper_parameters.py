import pytest
import torch

from torch_gl import CyclotomicNTTPlan, KeySwitcher, MatrixEncoder, MatrixNTTPlan, RNSRing
from torch_gl._native_transform import NativeTransform
from torch_gl._number_theory import root_of_unity, unit_generator
from torch_gl.matrix import matrix_transpose


@pytest.mark.parametrize("p,q", [(5, 241), (7, 337), (13, 313), (17, 5441)])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_prime_cyclotomic_matches_python_dft(p, q, dim):
    plan = CyclotomicNTTPlan(q, p)
    axis = dim % 3
    shape = [2, 3, 4]
    shape[axis] = p - 1
    values = torch.randint(q, shape, dtype=torch.int32, generator=torch.Generator().manual_seed(17))
    moved = values.movedim(axis, -1)
    oracle = []
    for row in moved.reshape(-1, p - 1).tolist():
        oracle.append(
            [sum(a * pow(plan.root, j * k, q) for j, a in enumerate(row)) % q for k in range(1, p)]
        )
    expected = torch.tensor(oracle, dtype=torch.int32).reshape(moved.shape).movedim(-1, axis)
    assert torch.equal(plan.forward(values, dim=dim), expected)
    assert torch.equal(plan.inverse(expected, dim=dim), values)
    assert len({pow(unit_generator(p), j, p) for j in range(p - 1)}) == p - 1


def test_paper_roots_and_native_roundtrip():
    ring = RNSRing.for_parameters(256, 17, limb_count=1, bits=28)
    assert ring.shape == (256, 256, 16, 2)
    root_of_unity(ring.moduli[0], 17)
    root_of_unity(ring.moduli[0], 1024)
    # Exercise the same transforms cheaply, including a noncontiguous batch.
    small = RNSRing.for_parameters(4, 17, limb_count=1).rings[0]
    transform = NativeTransform(small, torch.device("cpu"))
    values = torch.randint(small.q, small.shape, dtype=torch.int32)
    assert torch.equal(transform.inverse(transform.forward(values)), values)


@pytest.mark.parametrize("variables", [("X", "Y", "W"), ("W", "Y", "X")])
def test_prime17_matrix_embedding_and_encoding(variables):
    ring = RNSRing.for_parameters(2, 17, limb_count=2, variables=variables)
    plan = MatrixNTTPlan(ring.rings[0])
    values = torch.randint(ring.moduli[0], ring.shape, dtype=torch.int32)
    poly = ring.rings[0].polynomial(values)
    assert plan.inverse(plan.forward(poly)) == poly
    # A transpose in coefficients must transpose the canonical full-NTT slots.
    transposed = matrix_transpose(ring.polynomial(values)).limbs[0]
    assert torch.equal(plan.forward(transposed), plan.forward(poly).transpose(1, 2))
    encoder = MatrixEncoder(ring, scale=2**30)
    slots = torch.randn(encoder.slot_shape, dtype=torch.complex128)
    torch.testing.assert_close(encoder.decode(encoder.encode(slots)), slots, atol=1e-7, rtol=0)


def test_prime17_small_and_big_switching():
    ring = RNSRing.for_parameters(2, 17, limb_count=3)
    switcher = KeySwitcher(ring, k=2, noise_std=3.2, generator=torch.Generator().manual_seed(57))
    small = switcher.parameters.small_ring
    source = small.polynomial(torch.randint(-1, 2, small.shape, dtype=torch.int32))
    destination = small.polynomial(torch.randint(-1, 2, small.shape, dtype=torch.int32))
    component = ring.uniform(generator=torch.Generator().manual_seed(58))
    for kind in ("small", "big"):
        source_full = switcher._embed(source)
        if kind == "big":
            source_full = matrix_transpose(source_full)
        key = switcher.generate_key(
            source if kind == "small" else source_full, destination, kind=kind
        )
        result = getattr(switcher, "switch_" + kind)(component, key)
        error = result.b + result.a * switcher._embed(destination) - component * source_full
        assert int(error.centered_coefficients().abs().max()) < 10000


def test_prime17_full_ntt_product_is_a_times_b_transpose_without_extra_scale():
    ring = RNSRing.for_parameters(2, 17, limb_count=1)
    plan = MatrixNTTPlan(ring.rings[0])
    q = ring.moduli[0]
    generator = torch.Generator().manual_seed(193517)
    left = torch.randint(q, plan.slot_shape, dtype=torch.int32, generator=generator)
    right = torch.randint(q, plan.slot_shape, dtype=torch.int32, generator=generator)
    left_poly, right_poly = plan.inverse(left), plan.inverse(right)
    transposed = matrix_transpose(ring.polynomial(right_poly.coefficients)).limbs[0]
    actual = plan.forward(plan.multiply(left_poly, transposed))
    expected = torch.empty_like(actual)
    for w in range(16):
        for row in range(2):
            for column in range(2):
                for channel in range(2):
                    expected[w, row, column, channel] = (
                        sum(
                            int(left[w, row, inner, channel])
                            * int(right[w, column, inner, channel])
                            for inner in range(2)
                        )
                        % q
                    )
    assert torch.equal(actual, expected)
