"""Benchmark warm transforms and single-limb multiplication on CPU or CUDA.

Examples:
    python scripts/benchmark_ntt.py --output benchmarks/cpu.json
    python scripts/benchmark_ntt.py --device cuda --compile

CUDA uses events with explicit synchronization. Compilation and device table
construction are warmup costs, excluded from steady-state medians. Sage is never run.
"""

import argparse
import json
import platform
import statistics
import time
from functools import partial
from pathlib import Path

import torch

from torch_gl import (
    CyclotomicNTTPlan,
    GaussianNTTPlan,
    NTTPlan,
    montgomery_mul,
    to_montgomery_form,
    to_normal_form,
)
from torch_gl.modular import mul_mod
from torch_gl.montgomery import montgomery_context


def measure(operation, device, warmup, repeats):
    for _ in range(warmup):
        operation()
    durations = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        for _ in range(repeats):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            operation()
            finished.record()
            finished.synchronize()
            durations.append(started.elapsed_time(finished))
    else:
        for _ in range(repeats):
            started = time.perf_counter()
            operation()
            durations.append((time.perf_counter() - started) * 1000)
    return statistics.median(durations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--q", type=int, default=754974721)
    parser.add_argument("--batches", type=int, nargs="+", default=[64, 4096])
    parser.add_argument("--lengths", type=int, nargs="+", default=[16, 256])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--compile", action="store_true", help="experimental torch.compile fusion")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("exact arithmetic supports CPU and CUDA only")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA hardware is unavailable")
        if device.index is not None:
            torch.cuda.set_device(device.index)
        device = torch.device("cuda", torch.cuda.current_device())
    if args.repeats < 1 or args.warmup < 1 or any(size < 1 for size in args.batches):
        parser.error("repeats, warmup, and batch sizes must be positive")
    generator = torch.Generator().manual_seed(1935)
    rows = []
    for batch_size in args.batches:
        plans = [
            ("ntt", NTTPlan(args.q, length), (batch_size, length), -1) for length in args.lengths
        ]
        plans += [
            ("cyclotomic", CyclotomicNTTPlan(args.q, 9), (batch_size, 6, 2), -2),
            ("gaussian", GaussianNTTPlan(args.q, 256), (batch_size, 256, 2), -2),
        ]
        for kind, plan, shape, dim in plans:
            plan.prepare(device)
            coefficients = torch.randint(args.q, shape, dtype=torch.int32, generator=generator).to(
                device
            )
            assert torch.equal(
                plan.inverse(plan.forward(coefficients, dim=dim), dim=dim), coefficients
            )

            def forward(values=coefficients, transform=plan, axis=dim):
                return transform.forward(values, dim=axis)

            compiled_ntt = args.compile and isinstance(plan, NTTPlan)
            if compiled_ntt:
                compiled_transform = plan.compile(device, dim=dim)
                operation = partial(compiled_transform, coefficients)
            else:
                operation = forward
            assert torch.equal(operation(), plan.forward(coefficients, dim=dim))
            median = measure(operation, device, args.warmup, args.repeats)
            rows.append(
                {
                    "operation": kind,
                    "batch": batch_size,
                    "length": plan.length,
                    "shape": list(shape),
                    "median_ms": median,
                    "compiled": compiled_ntt,
                }
            )
        left = torch.randint(args.q, (batch_size, 256), dtype=torch.int32, generator=generator).to(
            device
        )
        right = torch.randint(args.q, left.shape, dtype=torch.int32, generator=generator).to(device)
        encoded_left, encoded_right = (
            to_montgomery_form(left, args.q),
            to_montgomery_form(right, args.q),
        )
        assert torch.equal(
            to_normal_form(montgomery_mul(encoded_left, encoded_right)),
            mul_mod(left, right, args.q),
        )
        operations = [
            ("normal_mul", partial(mul_mod, left, right, args.q)),
            ("montgomery_mul", partial(montgomery_mul, encoded_left, encoded_right)),
        ]
        if args.compile:
            context = montgomery_context(args.q)
            operations[1] = (
                "montgomery_mul",
                partial(context.multiply, encoded_left.values, encoded_right.values),
            )
        for kind, operation in operations:
            if args.compile:
                operation = torch.compile(operation, fullgraph=True)
                compiled_result = operation()
                expected_result = (
                    montgomery_mul(encoded_left, encoded_right).values
                    if kind == "montgomery_mul"
                    else mul_mod(left, right, args.q)
                )
                assert torch.equal(compiled_result, expected_result)
            median = measure(operation, device, args.warmup, args.repeats)
            rows.append(
                {
                    "operation": kind,
                    "batch": batch_size,
                    "length": 256,
                    "shape": list(left.shape),
                    "median_ms": median,
                    "compiled": args.compile,
                }
            )
    result = {
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "q": args.q,
        "q_bits": args.q.bit_length(),
        "storage_dtype": "torch.int32",
        "compiled": args.compile,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "measurements": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
