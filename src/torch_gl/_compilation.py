"""Opt-in small CPU graphs; dispatch and caches stay outside compiler tracing."""

import os
from functools import lru_cache

import torch
from torch._dynamo import is_compiling

_THRESHOLD = 4096
_MASK = (1 << 31) - 1


def _redc(product, q, negative_inverse, *, signed=False):
    cancellation = (product.to(torch.int32) * negative_inverse) & _MASK
    reduced = (product + cancellation.to(torch.int64) * q) >> 31
    if signed:
        reduced = torch.where(reduced < 0, reduced + q, reduced)
    return torch.where(reduced >= q, reduced - q, reduced).to(torch.int32)


def _product(left, right, q, negative_inverse):
    return _redc(left.to(torch.int64) * right, q, negative_inverse)


def _enabled(values):
    # Nested compiler dispatch is never initiated during outer tracing.
    return (
        not is_compiling()
        and values.device.type == "cpu"
        and values.dtype == torch.int32
        and values.numel() >= _THRESHOLD
        and torch.get_num_threads() == 1
        and os.environ.get("TORCH_GL_COMPILE") == "1"
    )


@lru_cache(maxsize=512)
def _scalar(value, dtype, slot=0):
    return torch.tensor(value, dtype=dtype)


def _constants(q, negative_inverse):
    return _scalar(q, torch.int64), _scalar(negative_inverse, torch.int32)


@lru_cache(maxsize=16)
def _linear_kernel(arity):
    def kernel(values, weights, q, negative_inverse):
        total = values[0].to(torch.int64) * weights[0]
        for index in range(1, arity):
            total = total + values[index].to(torch.int64) * weights[index]
        return _redc(total, q, negative_inverse, signed=True)

    return torch.compile(kernel, backend="inductor", fullgraph=True, dynamic=True)


def maybe_linear(values, weights, q, negative_inverse):
    if len(values) > 3 or not _enabled(values[0]):
        return None
    if any(value.dtype != torch.int32 or value.device.type != "cpu" for value in values):
        return None
    shape = values[0].shape
    flat = tuple(value.contiguous().view(-1) for value in values)
    constants = tuple(
        _scalar(weight, torch.int32, index + 1) for index, weight in enumerate(weights)
    )
    return _linear_kernel(len(values))(flat, constants, *_constants(q, negative_inverse)).reshape(
        shape
    )


def prepare_cpu_kernels(moduli):
    """Execute every graph on real tensors before timed operation warmup."""
    if os.environ.get("TORCH_GL_COMPILE") != "1" or torch.get_num_threads() != 1:
        return cpu_optimization_info()
    for q in moduli:
        inverse = -pow(q, -1, 1 << 31) & _MASK
        values = torch.arange(8192, dtype=torch.int32).remainder(q)
        for arity in (1, 2, 3):
            _linear_kernel(arity)(
                tuple(values.clone() for _ in range(arity)),
                tuple(_scalar(index + 1, torch.int32, index + 1) for index in range(arity)),
                *_constants(q, inverse),
            )
        for inverse_direction in (False, True):
            _gaussian_kernel(inverse_direction)(
                values.reshape(-1, 2),
                (_scalar(1, torch.int32, 1), _scalar(2, torch.int32, 2)),
                *_constants(q, inverse),
            )
    return cpu_optimization_info()


def cpu_optimization_info():
    from torch._dynamo.utils import counters

    return {
        "enabled": os.environ.get("TORCH_GL_COMPILE") == "1",
        "threshold": _THRESHOLD,
        "linear_cache": _linear_kernel.cache_info()._asdict(),
        "scalar_cache": _scalar.cache_info()._asdict(),
        "gaussian_cache": _gaussian_kernel.cache_info()._asdict(),
        "compiler_stats": dict(counters["stats"]),
    }


def _add(left, right, q):
    total = left + right
    return torch.where(total >= q, total - q, total)


def _subtract(left, right, q):
    difference = left - right
    return torch.where(difference < 0, difference + q, difference)


@lru_cache(maxsize=2)
def _gaussian_kernel(inverse):
    def kernel(values, weights, q, negative_inverse):
        left, right = values.unbind(-1)
        if inverse:
            real = _product(_add(left, right, q), weights[0], q, negative_inverse)
            imag = _product(_subtract(left, right, q), weights[1], q, negative_inverse)
            return torch.stack((real, imag), -1)
        real = _product(left, weights[0], q, negative_inverse)
        imag = _product(right, weights[1], q, negative_inverse)
        return torch.stack((_add(real, imag, q), _subtract(real, imag, q)), -1)

    return torch.compile(kernel, backend="inductor", fullgraph=True, dynamic=True)


def maybe_gaussian(values, weights, q, negative_inverse, *, inverse):
    if not _enabled(values):
        return None
    flat = values.contiguous().view(-1, 2)
    weights = tuple(_scalar(weight, torch.int32, index + 1) for index, weight in enumerate(weights))
    return _gaussian_kernel(inverse)(flat, weights, *_constants(q, negative_inverse)).reshape(
        values.shape
    )
