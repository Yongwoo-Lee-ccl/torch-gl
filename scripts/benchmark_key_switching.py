"""Measure existing matrix multiplication and its two SwitchBig calls.

Defaults: n=128, p=3**5, a 600-bit product of twenty near-2**30 primes, k=4.
Keys are file-backed to bound resident memory. Setup and correctness checks are
outside operation timings. --warmups/--runs are explicit; defaults 0/1.
"""

import argparse
import datetime
import json
import math
import platform
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import torch

from torch_gl import Engine, KeySwitcher, RNSRing, create_secret_key
from torch_gl._number_theory import MAX_MODULUS, is_prime
from torch_gl.matrix import _matrix_plan


def benchmark_moduli(n, p, limb_count, k):
    """Reserve the highest primes for P, leaving twenty 30-bit primes for Q."""
    step = 4 * n * p
    candidate = ((MAX_MODULUS - 1) // step) * step + 1
    primes = []
    while len(primes) < limb_count + k:
        if candidate < 2**29:
            raise ValueError("not enough near-30-bit compatible primes")
        if is_prime(candidate):
            primes.append(candidate)
        candidate -= step
    ordered = tuple(sorted(primes))
    return ordered[:limb_count], ordered[limb_count:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--p", type=int, default=3**5)
    parser.add_argument("--modulus-bits", type=int, default=600)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/key_switching_600bit.json"))
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0 or args.modulus_bits < 1:
        parser.error("runs/bits must be positive and warmups nonnegative")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("exact arithmetic supports CPU and CUDA only")
    if device.type == "cuda":
        try:
            # Initialize before writing the report or allocating evaluation keys.
            torch.empty(0, device=device)
        except RuntimeError as error:
            parser.error(
                f"cannot initialize {device} with PyTorch {torch.__version__} "
                f"(CUDA {torch.version.cuda}): {error}\n"
                "Check nvidia-smi and install a PyTorch CUDA build compatible "
                "with the NVIDIA driver."
            )
    moduli, key_moduli = benchmark_moduli(args.n, args.p, math.ceil(args.modulus_bits / 30), args.k)
    ring = RNSRing.multivariate_poly(moduli, args.n, args.p)
    estimated_key_bytes = (
        4
        * math.ceil(ring.limb_count / args.k)
        * (ring.limb_count + args.k)
        * math.prod(ring.shape)
        * 4
    )
    if shutil.disk_usage(tempfile.gettempdir()).free < estimated_key_bytes + 2**30:
        raise RuntimeError("not enough temporary disk space for file-backed evaluation keys")
    report = {
        "status": "running",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n": args.n,
        "p": args.p,
        "ciphertext_moduli": moduli,
        "ciphertext_modulus_bits": ring.modulus.bit_length(),
        "ciphertext_limb_count": ring.limb_count,
        "key_moduli": key_moduli,
        "k": args.k,
        "dnum": math.ceil(ring.limb_count / args.k),
        "key_modulus_bits": math.prod(key_moduli).bit_length(),
        "device": str(device),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "storage_dtype": "torch.int32",
        "evaluation_key_storage": (
            "Torch file-backed mmap on CPU; one extended-basis limb at a time"
        ),
        "warmups": args.warmups,
        "runs": args.runs,
        "compiled": False,
        "polynomial_bytes": math.prod(ring.shape) * 4 * ring.limb_count,
        "plaintext": "10000 constant; encryption error std=3.2; dense uniform ciphertext masks",
        "operations": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed(name, operation):
        print(f"Starting {name}", flush=True)
        for _ in range(args.warmups):
            operation()
        durations = []
        result = None
        for run in range(args.runs):
            result = None
            synchronize()
            started = time.perf_counter()
            result = operation()
            synchronize()
            elapsed = time.perf_counter() - started
            durations.append(elapsed)
            print(f"{name} run {run + 1}: {elapsed:.6f} s", flush=True)
        report["operations"][name] = {
            "seconds": durations,
            "average_seconds": statistics.mean(durations),
        }
        save()
        return result

    save()
    with tempfile.TemporaryDirectory(prefix="torch-gl-switch-keys-") as key_directory:
        print("Generating secret key and input ciphertexts", flush=True)
        started = time.perf_counter()
        engine = Engine(
            ring, device=device, generator=torch.Generator(device=device).manual_seed(1935)
        )
        secret_key = create_secret_key(
            ring, generator=torch.Generator(device=engine.device).manual_seed((1935) + 100000)
        )
        left, right = engine.encrypt(10000, secret_key), engine.encrypt(10000, secret_key)
        report["input_setup_seconds"] = time.perf_counter() - started
        switcher = KeySwitcher(
            ring,
            k=args.k,
            key_moduli=key_moduli,
            device=device,
            generator=torch.Generator(device=device).manual_seed(688),
        )
        print("Generating the two big evaluation keys (excluded from operation timing)", flush=True)
        started = time.perf_counter()
        keys = switcher.generate_matrix_keys(secret_key, storage_directory=key_directory)
        report["evaluation_key_setup_seconds"] = time.perf_counter() - started
        report["evaluation_key_bytes"] = sum(
            path.stat().st_size for key in (keys.Y_key, keys.mixed_key) for path in key._paths
        )
        save()
        print(f"Evaluation-key setup: {report['evaluation_key_setup_seconds']:.6f} s", flush=True)
        for limb_ring in ring.rings:
            _matrix_plan(limb_ring, engine.device)
        product = timed(
            "homomorphic_matrix_multiplication_before_switching",
            lambda left=left, right=right: engine.matrix_multiply(left, right),
        )
        del left, right
        switched = timed(
            "matrix_relinearization_two_switch_big",
            lambda: switcher.relinearize_matrix(product, keys),
        )
        print(
            "Checking decryption before and after switching (excluded from operation timing)",
            flush=True,
        )
        before = engine.decrypt(product, secret_key)
        after = engine.decrypt(switched, secret_key)
        difference = (after - before).centered_coefficients()
        report["correctness"] = {
            "all_limb_consistent_small_CRT_error": True,
            "max_absolute_switching_error": int(difference.abs().max()),
            "checked_coefficients": difference.numel(),
        }
        report["status"] = "complete"
        save()
        print(json.dumps(report["operations"], indent=2), flush=True)
        print(json.dumps(report["correctness"], indent=2), flush=True)


if __name__ == "__main__":
    main()
