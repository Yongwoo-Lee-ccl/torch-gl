"""Two-layer encrypted inference using a square polynomial activation.

Run: python examples/mlp.py [--device cuda] [--seed 1935]
The reference is (X @ W1 + b1)^2 @ W2 + b2, using an explicit square activation.
"""

import argparse

import torch

from torch_gl import (
    Engine,
    Evaluator,
    KeySwitcher,
    MatrixEncoder,
    RNSRing,
    ScaledCiphertext,
    create_secret_key,
    generate_evaluation_keys,
)


def run(*, device="cpu", seed=1935):
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    ring = RNSRing.for_parameters(4, 3, limb_count=6)
    engine = Engine(ring, device=device, generator=generator)
    secret = create_secret_key(ring, torch.Generator(device=device).manual_seed(seed + 100000))
    switcher = KeySwitcher(ring, k=2, device=device, generator=generator)
    evaluator = Evaluator(
        ring, generate_evaluation_keys(switcher, secret, matrices=False), device=device
    )
    encoder = MatrixEncoder(ring, scale=ring.moduli[-1], device=device)

    def random(shape):
        return torch.randn(shape, dtype=torch.float64, device=device, generator=generator) / 4

    features = random(encoder.slot_shape)
    w1, w2 = random((4, 4)), random((4, 4))
    b1, b2 = random((4,)), random((4,))
    encrypted = ScaledCiphertext(engine.encrypt(encoder.encode(features), secret), encoder.scale)
    # Server evaluation: reuse the maximum-level Hadamard key after the first
    # linear layer has already consumed one modulus prime.
    hidden = evaluator.add_plain(evaluator.matmul_plain(encrypted, w1), b1)
    hidden = evaluator.square(hidden)
    result = evaluator.add_plain(evaluator.matmul_plain(hidden, w2), b2)
    decoded = MatrixEncoder(result.ring, scale=result.scale, device=device).decode(
        Engine(result.ring, device=device).decrypt(result.ciphertext, secret.at_level(result.level))
    )
    expected = (features @ w1 + b1).square() @ w2 + b2
    error = (decoded - expected).abs().max().item()
    torch.testing.assert_close(decoded.real, expected, atol=2e-4, rtol=2e-4)
    assert decoded.imag.abs().max().item() < 2e-4
    return {"max_error": error, "remaining_levels": result.level}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", default=1935, type=int)
    args = parser.parse_args()
    report = run(device=args.device, seed=args.seed)
    print("Polynomial MLP: encrypted 4 -> 4 -> 4 inference over 8 feature rows")
    print(f"Maximum absolute error: {report['max_error']:.3e}")
    print(f"Remaining modulus limbs: {report['remaining_levels']}")
