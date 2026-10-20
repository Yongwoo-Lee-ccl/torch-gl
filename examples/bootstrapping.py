"""Refresh noisy GL ciphertexts and perform an encrypted square afterward.

Run: python examples/bootstrapping.py [--device cuda] [--batch 2]
Uses tiny demonstration parameters and a moderate-precision sine/arcsine circuit.
"""

import argparse
import time

import torch

from torch_gl import (
    BootstrapParameters,
    Bootstrapper,
    Engine,
    Evaluator,
    KeySwitcher,
    MatrixEncoder,
    RNSRing,
    ScaledCiphertext,
    create_secret_key,
    generate_bootstrap_keys,
    generate_evaluation_keys,
)


def run(*, device="cpu", seed=1935, batch=1):
    if type(batch) is not int or batch < 1:
        raise ValueError("batch must be a positive integer")
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    ring = RNSRing.for_parameters(2, 3, limb_count=26)
    parameters = BootstrapParameters(ring)
    secret = create_secret_key(ring, torch.Generator(device=device).manual_seed(seed + 100000))
    started = time.perf_counter()
    keys = generate_bootstrap_keys(parameters, secret, device=device, generator=generator)
    bootstrapper = Bootstrapper(keys, device=device)
    switcher = KeySwitcher(ring, k=parameters.k, device=device, generator=generator)
    evaluator = Evaluator(
        ring,
        generate_evaluation_keys(switcher, secret, matrices=False),
        device=device,
        rounding="nearest",
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"Refresh key generation: {time.perf_counter() - started:.2f}s", flush=True)
    bottom = RNSRing(ring.moduli[:1], ring.axes)
    encoder = MatrixEncoder(bottom, scale=2**25, device=device)
    engine = Engine(bottom, device=device, generator=generator)
    messages = [
        torch.randn(encoder.slot_shape, dtype=torch.complex128, device=device, generator=generator)
        / 8
        for _ in range(batch)
    ]
    ciphertexts = [
        ScaledCiphertext(engine.encrypt(encoder.encode(m), secret.at_level(1)), encoder.scale)
        for m in messages
    ]

    def stage(name, parts):
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"{name}: {len(parts)} ciphertexts in R, {parts[0].level} limbs", flush=True)

    started = time.perf_counter()
    refreshed = bootstrapper.bootstrap_many(ciphertexts, on_stage=stage)
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime = time.perf_counter() - started
    errors, product_errors = [], []
    for result, message in zip(refreshed, messages, strict=True):
        decoded = MatrixEncoder(result.ring, scale=result.scale, device=device).decode(
            Engine(result.ring, device=device).decrypt(
                result.ciphertext, secret.at_level(result.level)
            )
        )
        errors.append((decoded - message).abs().max().item())
        torch.testing.assert_close(decoded, message, atol=2e-3, rtol=2e-3)
        squared = evaluator.square(result)
        squared_plain = MatrixEncoder(squared.ring, scale=squared.scale, device=device).decode(
            Engine(squared.ring, device=device).decrypt(
                squared.ciphertext, secret.at_level(squared.level)
            )
        )
        product_errors.append((squared_plain - message.square()).abs().max().item())
        torch.testing.assert_close(squared_plain, message.square(), atol=2e-3, rtol=2e-3)
    print(f"Bootstrapping: {runtime:.2f}s for {batch} ciphertext(s)")
    print(f"Restored modulus limbs: 1 -> {refreshed[0].level}")
    print(f"Maximum absolute refresh error: {max(errors):.3e}")
    print(f"Maximum absolute post-refresh square error: {max(product_errors):.3e}")
    return refreshed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", default=1935, type=int)
    parser.add_argument("--batch", default=1, type=int)
    args = parser.parse_args()
    run(device=args.device, seed=args.seed, batch=args.batch)
