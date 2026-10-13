"""Compare exact batched GEMM and ring products.

One RNS limb, shared dense inputs, warm caches. The reference contraction is
the former Torch modular inner-index loop. No Sage or evaluation keys are used.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from torch_gl import MatrixNTTPlan, RNSRing
from torch_gl._transform_tables import resolve_device
from torch_gl.matrix import _matrix_product_residues


def reference_contraction(left, right, arithmetic):
    output = torch.zeros_like(left)
    for inner_index in range(left.shape[2]):
        product = arithmetic.multiply(
            left[:, :, inner_index, :].unsqueeze(2),
            right[:, inner_index, :, :].unsqueeze(1),
        )
        output = arithmetic.add(output, product)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--p", type=int, default=243)
    parser.add_argument("--q", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.runs < 1 or (args.threads is not None and args.threads < 1):
        parser.error("runs/threads must be positive; warmups must be nonnegative")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    ring = (
        RNSRing.for_parameters(args.n, args.p, limb_count=1)
        if args.q is None
        else RNSRing.multivariate_poly((args.q,), args.n, args.p)
    )
    generator = torch.Generator(device=device).manual_seed(1935)
    left = ring.uniform(device=device, generator=generator)
    right = ring.uniform(device=device, generator=generator)
    plan = MatrixNTTPlan(ring.rings[0], device=device)
    left_slots = plan._coefficient_slots(left.limbs[0]._coefficients)
    right_slots = plan._coefficient_slots(right.limbs[0]._coefficients)
    assert torch.equal(
        _matrix_product_residues(left_slots, right_slots, plan._arithmetic),
        reference_contraction(left_slots, right_slots, plan._arithmetic),
    )
    operations = {
        "contraction_modular_loop": lambda: reference_contraction(
            left_slots, right_slots, plan._arithmetic
        ),
        "contraction_split_gemm": lambda: _matrix_product_residues(
            left_slots, right_slots, plan._arithmetic
        ),
        "polynomial_multiply": lambda: left * right,
        "plaintext_matrix_multiply": lambda: plan.multiply(left.limbs[0], right.limbs[0]),
    }
    measurements = {}
    for name, operation in operations.items():
        for _ in range(args.warmups):
            operation()
        samples = []
        for _ in range(args.runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            operation()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - started)
        measurements[name] = {
            "seconds": samples,
            "median_seconds": statistics.median(samples),
            "average_seconds": statistics.mean(samples),
        }
        print(f"{name}: {statistics.median(samples):.6f} s median", flush=True)
    report = {
        "n": args.n,
        "p": args.p,
        "q": ring.moduli[0],
        "limbs": 1,
        "device": str(device),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "storage_dtype": "torch.int32",
        "gemm_workspace_dtype": "torch.float64",
        "warmups": args.warmups,
        "runs": args.runs,
        "correctness": "GEMM equals modular loop",
        "measurements": measurements,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
