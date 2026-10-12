import pytest
import torch

from torch_gl import CyclotomicNTTPlan, GaussianNTTPlan, NTTPlan, intt, ntt
from torch_gl.modular import gaussian_mul


@pytest.mark.parametrize("length", [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 54, 81, 108])
def test_mixed_radix_against_direct_dft(length):
    plan = NTTPlan(1297, length)
    values = torch.tensor(
        [
            (17 * coefficient_index * coefficient_index - 11 * coefficient_index - 3) % plan.q
            for coefficient_index in range(length)
        ]
    )
    expected = torch.tensor(
        [
            sum(
                int(values[coefficient_index])
                * pow(plan.root, coefficient_index * frequency_index, plan.q)
                for coefficient_index in range(length)
            )
            % plan.q
            for frequency_index in range(length)
        ]
    )
    assert torch.equal(plan.forward(values), expected)
    assert torch.equal(plan.inverse(expected), values)


def test_axis_batch_and_noncontiguous_inputs():
    values = torch.arange(3 * 12 * 2).reshape(3, 12, 2).transpose(0, 2)
    original = values.clone()
    assert not values.is_contiguous()
    assert torch.equal(intt(ntt(values, 97, dim=1), 97, dim=1), values)
    assert torch.equal(values, original)


@pytest.mark.parametrize("p", [3, 9, 27, 81])
def test_cyclotomic_against_primitive_root_evaluation(p):
    plan = CyclotomicNTTPlan(1297, p)
    values = torch.arange(plan.length, dtype=torch.int64) * 7 + 1
    indices = [3 * slot_index + root_class for root_class in (1, 2) for slot_index in range(p // 3)]
    expected = torch.tensor(
        [
            sum(
                int(value) * pow(plan.root, coefficient_index * frequency_index, plan.q)
                for coefficient_index, value in enumerate(values)
            )
            % plan.q
            for frequency_index in indices
        ]
    )
    assert torch.equal(plan.forward(values), expected)
    assert torch.equal(plan.inverse(expected), values)
    batched = torch.stack((values, values + 5), dim=1)
    assert torch.equal(plan.inverse(plan.forward(batched, dim=0), dim=0), batched)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_twisted_ntt_against_split_evaluation(n):
    plan = GaussianNTTPlan(193, n)
    coefficients = torch.arange(2 * n).reshape(n, 2)
    slots = plan.forward(coefficients)
    imaginary = pow(plan.root, n, plan.q)
    for sign, channel in ((1, 0), (-1, 1)):
        expected = [
            sum(
                (int(real) + sign * imaginary * int(imag))
                * pow(plan.root, sign * (1 + 4 * frequency_index) * coefficient_index, plan.q)
                for coefficient_index, (real, imag) in enumerate(coefficients)
            )
            % plan.q
            for frequency_index in range(n)
        ]
        actual = (slots[:, 0] + sign * imaginary * slots[:, 1]).remainder(plan.q)
        assert actual.tolist() == expected, channel
    assert torch.equal(plan.inverse(slots), coefficients)


def test_native_cyclotomic_multiplication_identity():
    # W**6 = -1-W**3 in Phi_9. Compute W**3 * W**3 in the transform domain.
    plan = CyclotomicNTTPlan(577, 9)
    coefficients = torch.zeros(6, 2, dtype=torch.int64)
    coefficients[3, 0] = 1
    slots = plan.forward(coefficients, dim=0)
    result = plan.inverse(gaussian_mul(slots, slots, 577), dim=0)
    expected = torch.zeros_like(coefficients)
    expected[0, 0] = expected[3, 0] = 576
    assert torch.equal(result, expected)


@pytest.mark.parametrize("q,length", [(15, 2), (97, 5), (97, 64), (97, 0)])
def test_invalid_ntt_parameters(q, length):
    with pytest.raises(ValueError):
        NTTPlan(q, length)


def test_roots_and_invalid_inputs():
    with pytest.raises(ValueError):
        NTTPlan(97, 6, root=1)
    with pytest.raises(ValueError):
        CyclotomicNTTPlan(97, 9)
    with pytest.raises(ValueError):
        CyclotomicNTTPlan(101, 5)
    with pytest.raises(ValueError):
        GaussianNTTPlan(97, 16)
    with pytest.raises(ValueError):
        ntt(torch.tensor(1), 97)
    with pytest.raises(TypeError):
        ntt(torch.ones(4), 97)
