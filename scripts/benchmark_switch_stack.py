"""Frozen-fixture, single-thread key-switch-only stack benchmark.

Run --create-fixture once on the fixed baseline. Every branch loads the exact
same tensors; RNG seeds alone are not used as evidence of identical keys.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from torch_gl import KeySwitcher, RNSRing
from torch_gl.key_switching import KeySwitchKey
from torch_gl.matrix import matrix_transpose
from torch_gl.rns import select_ntt_primes


def tensor_hash(tensor):
    tensor = tensor.contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tree_hash(value):
    if isinstance(value, torch.Tensor):
        return tensor_hash(value)
    if isinstance(value, dict):
        payload = {key: tree_hash(item) for key, item in sorted(value.items())}
    elif isinstance(value, (tuple, list)):
        payload = [tree_hash(item) for item in value]
    else:
        payload = value
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def poly_values(poly):
    return tuple(limb._coefficients for limb in poly.limbs)


def load_poly(ring, values):
    from torch_gl import MultivariatePoly, RNSPoly

    return RNSPoly(
        ring,
        tuple(MultivariatePoly._from_owned(r, v) for r, v in zip(ring.rings, values, strict=True)),
    )


def create_fixture(path, n):
    q = select_ntt_primes(n, 17, 5, bits=28)
    p = select_ntt_primes(n, 17, 2, bits=30)
    ring = RNSRing.multivariate_poly(q, n, 17)
    switcher = KeySwitcher(
        ring, k=2, key_moduli=p, noise_std=3.2, generator=torch.Generator().manual_seed(193503)
    )
    small = switcher.parameters.small_ring
    source = small.polynomial(
        torch.randint(
            -1, 2, small.shape, dtype=torch.int32, generator=torch.Generator().manual_seed(193501)
        )
    )
    destination = small.polynomial(
        torch.randint(
            -1, 2, small.shape, dtype=torch.int32, generator=torch.Generator().manual_seed(193502)
        )
    )
    component = ring.uniform(generator=torch.Generator().manual_seed(193504))
    source_full = switcher._embed(source)
    source_y = matrix_transpose(source_full)
    source_mixed = source_full * source_y
    keys = {}
    for name, secret, kind in [
        ("small", source, "small"),
        ("big", source_y, "big"),
        ("mixed", source_mixed, "big"),
    ]:
        print("Generating frozen " + name + " evaluation key", flush=True)
        keys[name] = switcher.generate_key(secret, destination, kind=kind)._limbs
    fixture = {
        "spec": {
            "n": n,
            "p": 17,
            "Q": q,
            "P": p,
            "k": 2,
            "noise_std": 3.2,
            "seeds": {
                "source": 193501,
                "destination": 193502,
                "evaluation_keys": 193503,
                "component": 193504,
            },
            "secret_distribution": "independent ternary Gaussian channels",
            "storage": "in-memory evaluation keys; no timed disk IO",
        },
        "source": poly_values(source),
        "destination": poly_values(destination),
        "component": poly_values(component),
        "keys": keys,
    }
    print("Preparing exact source products outside timing", flush=True)
    fixture["expected"] = {
        name: poly_values(component * secret)
        for name, secret in [("small", source_full), ("big", source_y), ("mixed", source_mixed)]
    }
    fixture["hash"] = tree_hash(fixture)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "spec": fixture["spec"],
                "sha256": fixture["hash"],
                "serialization": "recursive sorted JSON of SHA256(dtype,shape,C-order raw bytes)",
            },
            indent=2,
        )
        + "\n"
    )
    print("Fixture saved: " + fixture["hash"], flush=True)


def graph_count():
    from torch._dynamo.utils import counters

    return dict(counters["stats"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--create-fixture", action="store_true")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--revision", help="commit of an archived source snapshot")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.create_fixture:
        create_fixture(args.fixture, args.n)
        return
    if args.runs < 5:
        raise ValueError("at least five measured samples required")
    fixture = torch.load(args.fixture, weights_only=True, map_location="cpu")
    digest = fixture.pop("hash")
    assert tree_hash(fixture) == digest, "fixture corruption"
    spec = fixture["spec"]
    ring = RNSRing.multivariate_poly(spec["Q"], spec["n"], spec["p"])
    switcher = KeySwitcher(ring, k=spec["k"], key_moduli=spec["P"])
    component = load_poly(ring, fixture["component"])
    destination = switcher._embed(load_poly(switcher.parameters.small_ring, fixture["destination"]))
    keys = {
        name: KeySwitchKey(
            switcher.parameters, "small" if name == "small" else "big", torch.device("cpu"), limbs
        )
        for name, limbs in fixture["keys"].items()
    }
    immutable_before = tree_hash({"component": fixture["component"], "keys": fixture["keys"]})
    try:
        from torch_gl._compilation import prepare_cpu_kernels

        print("Prewarming CPU graphs", flush=True)
        prepare_cpu_kernels(spec["P"] + spec["Q"])
    except ImportError:
        pass
    operations = {
        "SwitchSmall": lambda: switcher.switch_small(component, keys["small"]),
        "SwitchBig": lambda: switcher.switch_big(component, keys["big"]),
        "TwoSwitchBig": lambda: (
            switcher.switch_big(component, keys["big"]),
            switcher.switch_big(component, keys["mixed"]),
        ),
    }
    reference = json.loads(args.reference.read_text()) if args.reference else None
    report = {
        "spec": spec,
        "fixture_sha256": digest,
        "commit": args.revision
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": tree_hash(
            {str(p): p.read_bytes().hex() for p in sorted(Path("src").rglob("*.py"))}
        ),
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else "not supported by macOS",
            "compile": os.getenv("TORCH_GL_COMPILE", "0"),
        },
        "operations": {},
    }
    if reference:
        assert reference["fixture_sha256"] == digest
    for name, operation in operations.items():
        print("Warming " + name, flush=True)
        result = operation()
        del result
        graphs_before = graph_count()
        wall = []
        cpu = []
        for sample in range(args.runs):
            start_cpu = time.process_time()
            start = time.perf_counter()
            result = operation()
            wall.append(time.perf_counter() - start)
            cpu.append(time.process_time() - start_cpu)
            print(f"{name} sample {sample + 1}: {wall[-1]:.6f}s", flush=True)
            if sample + 1 < args.runs:
                del result
        graphs_after = graph_count()
        assert graphs_after.get("unique_graphs", 0) == graphs_before.get("unique_graphs", 0), (
            "timed recompilation"
        )
        outputs = result if isinstance(result, tuple) else (result,)
        labels = (
            ("big", "mixed")
            if name == "TwoSwitchBig"
            else ("small" if name == "SwitchSmall" else "big",)
        )
        hashes = []
        errors = []
        for output, label in zip(outputs, labels, strict=True):
            hashes.append(tree_hash((poly_values(output.b), poly_values(output.a))))
            expected = load_poly(ring, fixture["expected"][label])
            error = (output.b + output.a * destination - expected).centered_coefficients()
            maximum = int(error.abs().max())
            assert maximum < min(spec["Q"]) // 100, "switching error is not small"
            errors.append(
                {
                    "max_abs": maximum,
                    "coefficients": error.numel(),
                    "consistent_Q_limbs": len(spec["Q"]),
                }
            )
        if reference:
            assert hashes == reference["operations"][name]["output_sha256"], (
                "bitwise baseline mismatch"
            )
        report["operations"][name] = {
            "wall_seconds": wall,
            "process_cpu_seconds": cpu,
            "mean": statistics.mean(wall),
            "stddev": statistics.stdev(wall),
            "output_sha256": hashes,
            "errors": errors,
            "compiler_before": graphs_before,
            "compiler_after": graphs_after,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del result, outputs
    assert (
        tree_hash({"component": fixture["component"], "keys": fixture["keys"]}) == immutable_before
    )
    report["input_and_key_immutable"] = True
    report["status"] = "complete"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
