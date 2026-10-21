import pytest
import torch

from torch_gl.modular import arithmetic_context
from torch_gl.montgomery import montgomery_context


@pytest.mark.parametrize("q", [3, 97, 1297, 268118017, 1073655809, (1 << 30) - 1])
@pytest.mark.parametrize("layout", ["vector", "scalar", "broadcast", "noncontiguous"])
def test_canonical_arithmetic_integer_oracle(q, layout, monkeypatch):
    context = montgomery_context(q)
    arithmetic = arithmetic_context(q)
    boundary = [0, 1, q // 2, q - 2, q - 1]
    left = torch.tensor(boundary * 3, dtype=torch.int32).reshape(3, 5)
    right = left.flip(-1)
    if layout == "scalar":
        right = q - 1
    elif layout == "broadcast":
        right = right[:1]
    elif layout == "noncontiguous":
        left = left.T
        right = right.T
    original_left = left.clone()
    original_right = right.clone() if isinstance(right, torch.Tensor) else right
    expanded_left, expanded_right = torch.broadcast_tensors(
        left, torch.as_tensor(right, dtype=torch.int32)
    )
    expected = {
        "product": [
            (a * b * pow(1 << 31, -1, q)) % q
            for a, b in zip(
                expanded_left.reshape(-1).tolist(), expanded_right.reshape(-1).tolist(), strict=True
            )
        ],
        "add": [
            (a + b) % q
            for a, b in zip(
                expanded_left.reshape(-1).tolist(), expanded_right.reshape(-1).tolist(), strict=True
            )
        ],
        "sub": [
            (a - b) % q
            for a, b in zip(
                expanded_left.reshape(-1).tolist(), expanded_right.reshape(-1).tolist(), strict=True
            )
        ],
    }

    def reject(*args, **kwargs):
        raise AssertionError("canonical arithmetic must not call Tensor remainder")

    for name in ("remainder", "remainder_", "__mod__"):
        monkeypatch.setattr(torch.Tensor, name, reject)
    actual = {
        "product": context.multiply(left, right),
        "add": arithmetic.add(left, right),
        "sub": arithmetic.subtract(left, right),
    }
    for name, value in actual.items():
        assert value.dtype == torch.int32
        assert torch.equal(
            value, torch.tensor(expected[name], dtype=torch.int32).reshape(expanded_left.shape)
        )
    assert torch.equal(left, original_left)
    if isinstance(right, torch.Tensor):
        assert torch.equal(right, original_right)
