import pytest
import torch

from torch_gl._bounded import montgomery_linear_form
from torch_gl._compilation import prepare_cpu_kernels
from torch_gl._number_theory import MAX_MODULUS


def test_compiled_linear_oracle_and_graph_reuse(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("TORCH_GL_COMPILE", "1")
    try:
        from torch._dynamo.utils import counters

        prepare_cpu_kernels((97, 1297))
        for q in (97, 1297, 1073655809, MAX_MODULUS):
            for count in (1, 2, 3):
                bounds = (q // 2,) * count
                weights = tuple(range(1, count + 1))
                values = tuple(
                    torch.tensor([-q // 2 + 1, q // 2, 0, 1], dtype=torch.int32).repeat(2048)
                    for _ in range(count)
                )
                form = montgomery_linear_form(q, bounds, weights)
                actual = form(values)
                expected = (
                    sum(v.to(torch.int64) * w for v, w in zip(values, weights, strict=True))
                    .remainder(q)
                    .to(torch.int32)
                )
                assert torch.equal(actual, expected)
        warmed = counters["stats"]["unique_graphs"]
        for q in (433, 268118017):
            values = (torch.full((12345,), q // 2, dtype=torch.int32),) * 3
            actual = montgomery_linear_form(q, (q // 2,) * 3, (1, 2, 3))(values)
            assert torch.equal(actual, torch.full_like(values[0], (q // 2 * 6) % q))
        assert counters["stats"]["unique_graphs"] == warmed
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("reason", ["flag", "small", "threads"])
def test_linear_fallback(reason, monkeypatch):
    import torch_gl._compilation as compilation

    previous = torch.get_num_threads()
    monkeypatch.setenv("TORCH_GL_COMPILE", "1")
    torch.set_num_threads(1)
    values = torch.arange(8192, dtype=torch.int32).remainder(97)
    if reason == "flag":
        monkeypatch.delenv("TORCH_GL_COMPILE")
    elif reason == "small":
        values = values[:8]
    else:
        torch.set_num_threads(2)

    def reject(*args, **kwargs):
        raise AssertionError("fallback attempted compilation")

    monkeypatch.setattr(compilation, "_linear_kernel", reject)
    try:
        actual = montgomery_linear_form(97, (96,), (3,))((values,))
        assert torch.equal(actual, (values * 3).remainder(97))
    finally:
        torch.set_num_threads(previous)
