import os

import pytest
import torch

from torch_gl._compilation import prepare_cpu_kernels
from torch_gl.montgomery import montgomery_context


@pytest.fixture(scope="module", autouse=True)
def single_thread_compilation():
    previous = torch.get_num_threads()
    flag = os.environ.get("TORCH_GL_COMPILE")
    torch.set_num_threads(1)
    os.environ["TORCH_GL_COMPILE"] = "1"
    prepare_cpu_kernels((97, 1297, 268118017, 1073655809, (1 << 30) - 1))
    yield
    torch.set_num_threads(previous)
    if flag is None:
        os.environ.pop("TORCH_GL_COMPILE", None)
    else:
        os.environ["TORCH_GL_COMPILE"] = flag


@pytest.mark.parametrize("q", [97, 1297, 268118017, 1073655809, (1 << 30) - 1])
@pytest.mark.parametrize("layout", ["scalar", "vector", "broadcast", "noncontiguous"])
def test_compiled_montgomery_oracle_and_input_immutability(q, layout):
    boundary = torch.tensor([0, 1, q // 2, q - 2, q - 1, 0, 1, q - 1], dtype=torch.int32)
    left = boundary.repeat(1024).reshape(64, 128)
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
    expected = torch.tensor(
        [
            a * b * pow(1 << 31, -1, q) % q
            for a, b in zip(
                expanded_left.reshape(-1).tolist(), expanded_right.reshape(-1).tolist(), strict=True
            )
        ],
        dtype=torch.int32,
    ).reshape(expanded_left.shape)
    assert torch.equal(montgomery_context(q).multiply(left, right), expected)
    assert torch.equal(left, original_left)
    if isinstance(right, torch.Tensor):
        assert torch.equal(right, original_right)


def test_product_preparation_has_no_modulus_recompilation():
    from torch._dynamo.utils import counters

    before = counters["stats"]["unique_graphs"]
    for q in (433, 1073741789):
        left = torch.full((12345,), q - 1, dtype=torch.int32)
        for right in (q - 1, left.clone()):
            expected = pow(1 << 31, -1, q)
            assert torch.equal(
                montgomery_context(q).multiply(left, right), torch.full_like(left, expected)
            )
    assert counters["stats"]["unique_graphs"] == before


@pytest.mark.parametrize("reason", ["flag", "small", "threads", "dtype"])
def test_product_fallbacks(reason, monkeypatch):
    import torch_gl._compilation as compilation

    left = torch.arange(8192, dtype=torch.int32).remainder(97)
    original_threads = torch.get_num_threads()
    if reason == "flag":
        monkeypatch.setenv("TORCH_GL_COMPILE", "0")
    elif reason == "small":
        left = left[:12]
    elif reason == "threads":
        torch.set_num_threads(2)
    else:
        left = left.to(torch.int64)

    def reject(*args, **kwargs):
        raise AssertionError("fallback attempted product compilation")

    monkeypatch.setattr(compilation, "_product_kernel", reject)
    try:
        expected = (left.to(torch.int64) * 17 * pow(1 << 31, -1, 97)).remainder(97).to(torch.int32)
        assert torch.equal(montgomery_context(97).multiply(left, 17), expected)
    finally:
        torch.set_num_threads(original_threads)


def test_outer_compiler_does_not_dispatch_nested_compiler(monkeypatch):
    import torch_gl._compilation as compilation

    context = montgomery_context(97)

    def reject(*args, **kwargs):
        raise AssertionError("outer compiler started an inner compiler")

    monkeypatch.setattr(compilation, "_product_kernel", reject)
    outer = torch.compile(lambda x: context.multiply(x, 17), backend="eager", fullgraph=True)
    values = torch.arange(8192, dtype=torch.int32).remainder(97)
    expected = (values.to(torch.int64) * 17 * pow(1 << 31, -1, 97)).remainder(97).to(torch.int32)
    assert torch.equal(outer(values), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_preserves_eager_product_fallback(monkeypatch):
    import torch_gl._compilation as compilation

    values = torch.arange(8192, device="cuda", dtype=torch.int32).remainder(97)

    def reject(*args, **kwargs):
        raise AssertionError("CPU compiler dispatched on CUDA")

    monkeypatch.setattr(compilation, "_product_kernel", reject)
    expected = (values.to(torch.int64) * 17 * pow(1 << 31, -1, 97)).remainder(97).to(torch.int32)
    assert torch.equal(montgomery_context(97).multiply(values, 17), expected)
