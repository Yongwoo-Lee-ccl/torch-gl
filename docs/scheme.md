# Complete leveled GL arithmetic

The approximate variant uses rounded complex encoding. The integer variant
uses the two independent finite-field branches from Section 4 of
[Gentry–Lee](https://eprint.iacr.org/2025/1935). Both use explicit small-ring
secrets `s(X,W)`. The existing research-only randomness and security limitations
in the README apply to these APIs as well.

## Approximate evaluation

```python
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

ring = RNSRing.for_parameters(2, 3, limb_count=5)
secret = create_secret_key(ring, torch.Generator().manual_seed(7))
switcher = KeySwitcher(ring, k=2)
keys = generate_evaluation_keys(switcher, secret, rotations=[("X", 1), ("W", 1)])
evaluator = Evaluator(ring, keys)  # Public evaluation material only.
encoder = MatrixEncoder(ring, scale=ring.moduli[-1])
engine = Engine(ring)
values = torch.randn(encoder.slot_shape, dtype=torch.float64) / 4
ct = ScaledCiphertext(engine.encrypt(encoder.encode(values), secret), encoder.scale)
result = evaluator.add_plain(evaluator.square(ct), 0.1)
active_secret = secret.at_level(result.level)
recovered = MatrixEncoder(result.ring, scale=result.scale).decode(
    Engine(result.ring).decrypt(result.ciphertext, active_secret)
)
torch.testing.assert_close(recovered.real, values.square() + 0.1, atol=1e-4, rtol=1e-4)
```

`multiply` is the Hadamard product. `matmul`, `matmul_transposed` and
`matmul_adjoint` return `A @ B`, `A @ B.T` and `A @ B.mH`, respectively.
`matmul_adjoint` uses the paper's transformed right secret and two big keys.
`matmul_plain(ct, P)` returns `A @ P` without key switching.
`transpose(ct, conjugate=True)` is the Hermitian transpose; `conjugate(ct)`
conjugates the slots element by element.

`rotate(ct, steps, axis="X"/"Y"/"W")` is a cyclic shift by **minus steps** in
Torch's `roll` convention. X rotates rows, Y columns, W the matrix batch.
Column rotation needs no key because the secret is independent of Y. Other
requested rotation keys must be generated explicitly.

Products rescale by default and return the correct multiplied/divided scale.
The evaluator projects operands to their common level. Addition of unequal
scales aligns them using encoded scalar multiplication and consumes a level;
it never changes a scale label without adjusting the ciphertext. There is no
automatic bootstrapping. The low-level Engine APIs retain their original
explicit-scale behavior, including `Engine.matrix_multiply`'s `A @ B.T`.

Maximum-level evaluation keys work at every prefix, even if a Han–Ki digit is
only partly active. A lower `KeySwitcher` must use the original `k` and
`key_moduli`. The cofactor in decomposition is taken from the original key;
simply truncating a key with an active-level cofactor would corrupt the result.

## Integer matrices

```python
from torch_gl import IntegerEncoder, IntegerEngine, integer_ring

ring = integer_ring(2, 3, plaintext_modulus=97, limb_count=4)
encoder = IntegerEncoder(ring, 97)
engine = IntegerEngine(ring, 97)
secret = create_secret_key(ring, torch.Generator().manual_seed(8))
values = torch.randint(10, encoder.slot_shape, dtype=torch.int32)
ct = engine.encrypt(encoder.encode(values), secret)
switcher = KeySwitcher(ring, k=2, plaintext_modulus=97)
product = switcher.relinearize(ct * ct, switcher.generate_hadamard_key(secret))
product = engine.mod_switch(product)
recovered = encoder.decode(
    IntegerEngine(product.ring, 97).decrypt(product, secret.at_level(product.ring.limb_count))
)
assert torch.equal(recovered, values.square().remainder(97))
```

Integer slots have shape `(phi(p), n, n, 2)`. The last axis distinguishes the
independent +I and -I matrices; it is not the complex coefficient channel.
The plaintext modulus must be a fully splitting prime. This implementation
chooses ciphertext and key primes congruent to 1 modulo that plaintext modulus.
Encryption and key-switch errors are multiples of t. For the dropped prime p,
ModSwitch subtracts `t * centered(c_last/t mod p)` before dividing by p. The
rounding error remains a multiple of t, and p = 1 mod t preserves the plaintext.
Use `mod_switch`, not approximate `rescale`, for this variant. As with BGV,
the centered decryption must stay within the active ciphertext modulus.

## Public-key encryption

`public_key = engine.create_public_key(secret)` encrypts zero. Anyone with that
public key can call `engine.encrypt_public(encoded_message, public_key)`. The
method samples a binary ephemeral polynomial and two independent noise
polynomials. The same API works on `IntegerEngine`, with noise multiplied by t.

## Coefficient ring changes

`split(poly)` extracts the n coefficients of Y from R', yielding n polynomials
in R. `merge(parts, full_ring)` reconstructs R'.
`RNSRing.auxiliary(moduli, n, p)` constructs
`Z_Q[i][X,W,V]/(X^n-i, Phi_p(W), Phi_p(V))`, and
`merge(parts, auxiliary_ring)` packs the same R polynomials into V coefficients.
Missing coefficients are zero-padded; excess input is rejected. Use groups of
at most phi(p) polynomials when phi(p) < n.

`split_ciphertext` and `merge_ciphertexts` apply those maps to both public
components. Extraction commutes with decryption precisely because s(X,W) is
independent of Y/V. These coefficient maps are exact and require no key switch,
encoding, decoding or modulus change. `split(poly, "V")` extracts auxiliary
coefficients. Levels, devices, quotient axes and ordering must match.

`RNSPoly.mod_raise(target)` separately lifts **centered** coefficient
representatives into a longer modulus chain. For a ciphertext this introduces
the carry term Q*I; it is not a refresh on its own.
