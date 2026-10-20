"""Fit a plaintext regressor, then evaluate it on encrypted feature rows.

Run: python examples/linear_regression.py [--device cuda] [--seed 1935]
No datasets, downloads, or dependencies beyond torch-gl are required.
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
    # Three features plus the constant bias column fit in a 4x4 GL matrix.
    training = torch.randn(64, 4, dtype=torch.float64, device=device, generator=generator)
    training[:, -1] = 1
    true_weights = torch.tensor([0.75, -1.25, 0.5, 0.2], device=device, dtype=torch.float64)
    weights = torch.linalg.lstsq(training, training @ true_weights).solution
    ring = RNSRing.for_parameters(4, 3, limb_count=3)
    engine = Engine(ring, device=device, generator=generator)
    secret = create_secret_key(ring, torch.Generator(device=device).manual_seed(seed + 100000))
    switcher = KeySwitcher(ring, k=2, device=device, generator=generator)
    evaluator = Evaluator(
        ring, generate_evaluation_keys(switcher, secret, matrices=False), device=device
    )
    encoder = MatrixEncoder(ring, scale=ring.moduli[-1], device=device)
    features = torch.randn(
        encoder.slot_shape, dtype=torch.float64, device=device, generator=generator
    )
    features[..., -1] = 1
    weight_matrix = torch.zeros(4, 4, dtype=torch.float64, device=device)
    weight_matrix[:, 0] = weights
    # Client: encrypt private feature rows using the public encryption key.
    public_key = engine.create_public_key(secret)
    encrypted = ScaledCiphertext(
        engine.encrypt_public(encoder.encode(features), public_key), encoder.scale
    )
    # Server: one ciphertext-plaintext matrix multiply, with no decryption key.
    result = evaluator.matmul_plain(encrypted, weight_matrix)
    # Client: decrypt the result. The other three output columns are zero.
    decoded = MatrixEncoder(result.ring, scale=result.scale, device=device).decode(
        Engine(result.ring, device=device).decrypt(result.ciphertext, secret.at_level(result.level))
    )
    expected = features @ weight_matrix
    error = (decoded - expected).abs().max().item()
    torch.testing.assert_close(decoded.real, expected, atol=2e-4, rtol=2e-4)
    assert decoded.imag.abs().max().item() < 2e-4
    return {
        "max_error": error,
        "remaining_levels": result.level,
        "predictions": decoded[..., 0].real.flatten().tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", default=1935, type=int)
    args = parser.parse_args()
    report = run(device=args.device, seed=args.seed)
    print("Linear regression: encrypted inference over 8 feature rows")
    print(f"Maximum absolute error: {report['max_error']:.3e}")
    print(f"Remaining modulus limbs: {report['remaining_levels']}")
    print("Predictions:", [round(value, 5) for value in report["predictions"]])
