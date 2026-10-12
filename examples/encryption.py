"""Research encryption demo: python examples/encryption.py."""

import torch

from torch_gl import Engine, RNSRing, create_secret_key

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ring = RNSRing.for_parameters(n=4, p=9, limb_count=3)
engine = Engine(ring, device=device, generator=torch.Generator(device=device).manual_seed(1935))
secret_key = create_secret_key(
    ring, generator=torch.Generator(device=engine.device).manual_seed((1935) + 100000)
)
print(
    f"Device: {device}; RNS primes: {ring.moduli}; total modulus: {ring.modulus.bit_length()} bits"
)
for name in ("constant", "X", "W", "Y"):
    message = (
        ring.constant(10000, device=device)
        if name == "constant"
        else (10000 * ring.generator(name, device=device))
    )
    ciphertext = engine.encrypt(message, secret_key)
    decrypted = engine.decrypt(ciphertext, secret_key)
    error = decrypted.centered_coefficients() - message.centered_coefficients()
    print(
        f"10000{'*' + name if name != 'constant' else ''}: "
        f"maximum coefficient error = {int(error.abs().max())}"
    )
