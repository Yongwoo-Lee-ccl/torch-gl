"""RNS plaintext and encrypted-product rescaling: python examples/rescaling.py."""

import torch

from torch_gl import Engine, MatrixEncoder, RNSRing, create_secret_key


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ring = RNSRing.for_parameters(n=4, p=9, limb_count=4)
    engine = Engine(ring, device=device, generator=torch.Generator(device=device).manual_seed(101))
    secret_key = create_secret_key(ring, torch.Generator(device=device).manual_seed(11))
    encoder = MatrixEncoder(ring, device=device)
    generator = torch.Generator(device=device).manual_seed(301)
    left, right = [
        torch.randn(encoder.slot_shape, dtype=torch.complex128, device=device, generator=generator)
        for _ in range(2)
    ]
    left_encoded, right_encoded = encoder.encode(left), encoder.encode(right)
    left_ct, right_ct = (
        engine.encrypt(left_encoded, secret_key),
        engine.encrypt(right_encoded, secret_key),
    )

    plaintext_product = (left_encoded * right_encoded).rescale()
    lower_ring = plaintext_product.ring
    # Project the same key. Dividing the key with rescale() would change it.
    lower_key = secret_key.drop_last_limb()
    lower_engine = Engine(lower_ring, device=device)
    product_scale = encoder.scale**2
    reduced_scale = product_scale / ring.moduli[-1]
    decoder = MatrixEncoder(lower_ring, scale=reduced_scale, device=device)
    print(f"Device: {device}; modulus chain: {ring.limb_count} -> {lower_ring.limb_count} limbs")
    print(f"Dropped prime: {ring.moduli[-1]}; product scale: {product_scale} -> {reduced_scale}")

    recovered = decoder.decode(plaintext_product)
    torch.testing.assert_close(recovered, left * right, atol=2e-8, rtol=2e-9)
    slot_error = (recovered - left * right).abs().max().item()
    print(f"Plaintext Hadamard maximum slot error: {slot_error:.3e}")

    hadamard_product = engine.rescale(engine.multiply(left_ct, right_ct))
    recovered = decoder.decode(lower_engine.decrypt(hadamard_product, lower_key))
    torch.testing.assert_close(recovered, left * right, atol=2e-8, rtol=2e-9)
    slot_error = (recovered - left * right).abs().max().item()
    print(f"Encrypted Hadamard maximum slot error: {slot_error:.3e}")

    matrix_product = engine.rescale(engine.matrix_multiply(left_ct, right_ct))
    expected = left @ right.transpose(-1, -2)
    recovered = decoder.decode(lower_engine.decrypt(matrix_product, lower_key))
    torch.testing.assert_close(recovered, expected, atol=2e-8, rtol=2e-9)
    print(f"Encrypted matrix maximum slot error: {(recovered - expected).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
