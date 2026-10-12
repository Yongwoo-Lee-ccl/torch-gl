import pytest
import torch

from torch_gl import (
    MONTGOMERY_RADIX,
    is_montgomery_form,
    montgomery_mul,
    to_montgomery_form,
    to_normal_form,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", [3, 97, 754974721, (1 << 30) - 1])
def test_conversion_and_product_against_python_integers(q, device):
    generator = torch.Generator().manual_seed(1935)
    left = torch.randint(q, (4096,), generator=generator)
    right = torch.randint(q, (4096,), generator=generator)
    left[:5] = torch.tensor([0, 1, q - 1, q - 2, q // 2])
    right[:5] = torch.tensor([q - 1, q - 1, q - 1, q - 2, q // 3])
    expected = [
        left_value * right_value % q
        for left_value, right_value in zip(left.tolist(), right.tolist(), strict=True)
    ]
    left, right = left.to(device), right.to(device)
    encoded_left = to_montgomery_form(left, q)
    encoded_right = to_montgomery_form(right, q)
    assert encoded_left.values.dtype == torch.int32
    assert to_normal_form(encoded_left).dtype == torch.int32
    assert MONTGOMERY_RADIX == 1 << 31
    assert encoded_left.values.cpu().tolist() == [
        value * MONTGOMERY_RADIX % q for value in left.cpu().tolist()
    ]
    assert is_montgomery_form(encoded_left)
    assert not is_montgomery_form(encoded_left.values)
    assert torch.equal(to_normal_form(encoded_left), left)
    product = montgomery_mul(encoded_left, encoded_right)
    assert product.q == q
    assert product.device == left.device
    assert to_normal_form(product).cpu().tolist() == expected


def test_broadcasting_normalization_and_defensive_copies():
    q = 754974721
    values = torch.tensor([[-1], [q], [q + 1], [-(1 << 63)]])
    encoded = to_montgomery_form(values, q)
    exported = encoded.values
    exported.zero_()
    assert torch.equal(to_normal_form(encoded), values.remainder(q))
    encoded_right = to_montgomery_form(torch.tensor([[2, q - 1]]), q)
    product = montgomery_mul(encoded, encoded_right)
    # Use Python integers as the independent oracle for all intermediates.
    expected = [[(int(value) % q) * factor % q for factor in (2, q - 1)] for value in values]
    assert to_normal_form(product).tolist() == expected


def test_representation_and_modulus_errors():
    values = torch.tensor([1, 2, 3])
    encoded = to_montgomery_form(values, 97)
    with pytest.raises(TypeError, match="already"):
        to_montgomery_form(encoded, 97)
    with pytest.raises(TypeError):
        to_normal_form(values)
    with pytest.raises(TypeError):
        montgomery_mul(encoded, values)
    with pytest.raises(ValueError):
        montgomery_mul(encoded, to_montgomery_form(values, 193))
    for invalid_q in (2, (1 << 30), 1152921504606847201):
        with pytest.raises(ValueError):
            to_montgomery_form(values, invalid_q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_device_mismatch():
    values = torch.tensor([1, 2, 3])
    with pytest.raises(ValueError):
        montgomery_mul(to_montgomery_form(values, 97), to_montgomery_form(values.cuda(), 97))
