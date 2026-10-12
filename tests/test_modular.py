import pytest
import torch

from torch_gl.modular import gaussian_mul, mul_mod


@pytest.mark.parametrize("q", [97, 65537, 469762049, 754974721, 1073741823])
def test_products_against_python_big_integers(q):
    left_values = [0, 1, q - 1, q - 2, q // 2, q // 3]
    right_values = [q - 1, q - 2, q - 1, q - 2, q // 3, q // 2]
    left, right = torch.tensor(left_values), torch.tensor(right_values)
    assert mul_mod(left, right, q).tolist() == [
        left_value * right_value % q
        for left_value, right_value in zip(left_values, right_values, strict=True)
    ]
    assert mul_mod(left, q - 1, q).tolist() == [
        left_value * (q - 1) % q for left_value in left_values
    ]
    pairs = torch.stack((left, right), -1)
    expected = [
        [
            (left_value * left_value - right_value * right_value) % q,
            2 * left_value * right_value % q,
        ]
        for left_value, right_value in zip(left_values, right_values, strict=True)
    ]
    assert gaussian_mul(pairs, pairs, q).tolist() == expected
