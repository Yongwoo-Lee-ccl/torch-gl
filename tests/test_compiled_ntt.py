import pytest
import torch

from torch_gl import NTTPlan


@pytest.mark.parametrize("length", [16, 18, 27])
def test_compiler_full_graph_capture(length):
    # The eager backend checks graph capture and conventions on any CPU.
    # Native CUDA/Inductor performance needs the separate hardware benchmark.
    plan = NTTPlan(1297, length)
    coefficients = torch.arange(length * 6).reshape(2, length, 3).transpose(0, 2)
    compiled_forward = plan.compile("cpu", dim=1, backend="eager")
    compiled_inverse = plan.compile("cpu", dim=1, inverse=True, backend="eager")
    slots = compiled_forward(coefficients)
    assert torch.equal(slots, plan.forward(coefficients, dim=1))
    assert torch.equal(compiled_inverse(slots), coefficients)
    with pytest.raises(ValueError):
        compiled_forward(torch.zeros(2, length + 1, 3, dtype=torch.int64))
    with pytest.raises(TypeError):
        compiled_forward(torch.zeros(2, length, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_native_compiled_cuda_transform():
    plan = NTTPlan(754974721, 256).prepare("cuda")
    cpu_coefficients = torch.arange(32 * 256).reshape(32, 256)
    gpu_coefficients = cpu_coefficients.cuda()
    compiled_forward = plan.compile("cuda")
    compiled_inverse = plan.compile("cuda", inverse=True)
    gpu_slots = compiled_forward(gpu_coefficients)
    assert torch.equal(gpu_slots.cpu(), plan.forward(cpu_coefficients))
    assert torch.equal(compiled_inverse(gpu_slots), gpu_coefficients)
