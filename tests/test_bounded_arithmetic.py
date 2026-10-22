import math

import pytest
import torch

from torch_gl._bounded import bounded_sum_reducer, montgomery_linear_form
from torch_gl.basis import RNSBasisConverter


@pytest.mark.parametrize("q", [3, 97, 1297, 1073655809, (1 << 30) - 1])
@pytest.mark.parametrize("count", [1, 2, 3, 17, 65])
def test_signed_linear_form_worst_signs_and_group_bounds(q, count):
    bound = (1 << 29) - 1
    weights = tuple(((index + 1) * pow(1 << 31, -1, q)) % q for index in range(count))
    bounds = (bound,) * count
    form = montgomery_linear_form(q, bounds, weights)
    values = tuple(
        torch.tensor([-bound, bound, (-1) ** index * bound, 0], dtype=torch.int32)
        for index in range(count)
    )
    originals = tuple(v.clone() for v in values)
    expected = torch.tensor(
        [sum(int(values[i][j]) * weights[i] for i in range(count)) % q for j in range(4)],
        dtype=torch.int32,
    )
    assert torch.equal(form(values), expected)
    for group in form.groups:
        assert sum(bounds[i] * form.weights[i] for i in group) < q * (1 << 31)
    assert all(torch.equal(a, b) for a, b in zip(values, originals, strict=True))


def test_many_large_weights_force_safe_groups_and_broadcast():
    q = 97
    bound = (1 << 30) - 1
    weights = (96 * pow(1 << 31, -1, q) % q,) * 19
    form = montgomery_linear_form(q, (bound,) * 19, weights)
    assert len(form.groups) > 1
    values = tuple(
        torch.tensor([[bound], [-bound]], dtype=torch.int32)
        if i % 2
        else torch.tensor([0, bound, -bound], dtype=torch.int32)
        for i in range(19)
    )
    expanded = torch.broadcast_tensors(*values)
    expected = torch.tensor(
        [
            sum(int(v.reshape(-1)[j]) * w for v, w in zip(expanded, weights, strict=True)) % q
            for j in range(6)
        ],
        dtype=torch.int32,
    ).reshape(2, 3)
    assert torch.equal(form(values), expected)
    assert not torch.count_nonzero(montgomery_linear_form(q, (bound,), (0,))((values[0],)))
    with pytest.raises(ValueError, match="individual"):
        montgomery_linear_form(q, (1 << 62,), (1,))


@pytest.mark.parametrize("q", [3, 97, 1297, 1073655809])
@pytest.mark.parametrize("count", [1, 2, 16])
def test_bounded_sum_against_integer_oracle(q, count):
    values = torch.full((3, count, 2), q - 1, dtype=torch.int32)
    values[0].zero_()
    values[1, :, 0] = 1
    result = bounded_sum_reducer(q, count)(values, dim=1)
    expected = values.to(torch.int64).sum(1).remainder(q).to(torch.int32)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("t", [1, 17])
def test_fused_moddown_and_custom_inverses_preserve_centered_lift(t):
    sources = (1073655809, 268118017)
    targets = (97, 1297)
    converter = RNSBasisConverter(sources, targets + sources)
    values = tuple(torch.tensor([0, 1, q - 1, q // 2], dtype=torch.int32) for q in sources)
    custom = (7, 11)
    prepared = converter.prepare(values, inverses=custom)
    product = math.prod(sources)
    lifts = []
    for j in range(4):
        total = 0
        for value, inverse, q in zip(values, custom, sources, strict=True):
            term = int(value[j]) * inverse % q
            if term > q // 2:
                term -= q
            total += term * (product // q)
        lifts.append(total)
    for q in targets + sources:
        converted = converter.convert_prepared(prepared, q)
        assert converted.tolist() == [v % q for v in lifts]
        base = torch.tensor([0, 1, q - 1, q // 2], dtype=torch.int32)
        scale = pow(product, -1, q) if q not in sources else 13
        result = converter.subtract_and_scale(base, prepared, q, scale, multiplier=t)
        assert result.tolist() == [
            (int(b) - t * v) * scale % q for b, v in zip(base, lifts, strict=True)
        ]
    for q, value in zip(sources, values, strict=True):
        assert converter.convert_prepared(prepared, q, source_values=values) is value


def test_warmed_switching_has_no_tensor_remainder(monkeypatch):
    from torch_gl import KeySwitcher, RNSRing

    ring = RNSRing.for_parameters(2, 17, limb_count=3)
    switcher = KeySwitcher(ring, k=2, generator=torch.Generator().manual_seed(179))
    small = switcher.parameters.small_ring
    source = small.polynomial(torch.ones(small.shape, dtype=torch.int32))
    component = ring.uniform(generator=torch.Generator().manual_seed(180))
    keys = [
        switcher.generate_key(source, source),
        switcher.generate_key(switcher._embed(source), source, kind="big"),
    ]
    expected = [switcher.switch_small(component, keys[0]), switcher.switch_big(component, keys[1])]

    def reject(*args, **kwargs):
        raise AssertionError("warmed switching must not normalize canonical residues")

    for name in ("remainder", "remainder_", "__mod__"):
        monkeypatch.setattr(torch.Tensor, name, reject)
    assert switcher.switch_small(component, keys[0]) == expected[0]
    assert switcher.switch_big(component, keys[1]) == expected[1]


def test_linear_cache_rejects_tensor_weights():
    with pytest.raises(ValueError, match="public Python integers"):
        montgomery_linear_form(97, (48,), (torch.tensor(3),))
