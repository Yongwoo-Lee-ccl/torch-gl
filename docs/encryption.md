# RNS and research encryption

`RNSRing` defines a common polynomial basis over `Q=product(moduli)`.
`RNSPoly` stores a `MultivariatePoly` for each distinct prime modulus, with
identical axes and a common device. Int32 limbs stay below `2^30`; multiplication
by the aggregate modulus never occurs in Torch int64.

## Prime selection and representation

`RNSRing.for_parameters(n=4, p=9, limb_count=3)` selects ascending primes
`q=multiplier*(4*n*p)+1` in the requested bit range, defaulting to 30 bits.
Explicit moduli can be supplied with `RNSRing.multivariate_poly(moduli,n,p)`;
other separable ideals use `RNSRing(moduli,axes)`. Limb order is part of the context.

For cyclotomic prime bases other than three, the selector also includes the
required padded radix-2 transform length in its congruence step.

RNS construction uses the same integer coefficient tensor in every limb. Arithmetic
operates limb by limb; integer scaling avoids NTTs. Uniform sampling independently
samples each prime limb, corresponding to a uniform element modulo Q under CRT.

`centered_coefficients()` returns a signed int32 tensor for coefficients within
half the first prime. It checks agreement with all other limbs, so it does not
silently assume that range. For arbitrary values, `reconstruct()` performs exact
CPU CRT reconstruction with Python integers and returns a flat tuple in C order.
Full CRT export supports values above int64 and stays outside the arithmetic hot path.

## Engine and ciphertext

`create_secret_key(ring, generator=key_generator)` samples a binary coefficient
tensor once and encodes it in every limb. The generator is required and belongs
to the caller; use a separate generator for encryption randomness. Both Gaussian
channels have coefficients in `{0,1}`. In an X/Y/W context the function returns
a key in the smaller X/W ring, retaining the moduli and relative axis order.
Other contexts return a key in the supplied ring. Its device defaults to the
generator's device, and an explicit device must match the generator.

`Engine` never samples or stores a secret key. Encryption and decryption receive
the key explicitly, embed small-ring keys at Y degree zero temporarily, and use
ordinary polynomial multiplication with the engine's compiler settings.
Supplied keys must match the context and device; encryption requires binary
coefficients. Full-ring binary keys remain supported for ordinary encryption.
Matrix evaluation needs no key; matrix decryption rejects keys depending on Y.

Each encryption samples uniform `a` independently per limb and one shared error
tensor `e`. The latter is sampled from a mean-zero normal distribution with
standard deviation 3.2, rounded to integers, then encoded in every limb.
It computes `b=-a*s+e+m` and returns `Ciphertext(b,a)`, containing no key or error.

`engine.encrypt(message, secret_key)` accepts an RNS polynomial, an integer coefficient
tensor, an integer constant, or a Gaussian integer pair.
`engine.decrypt(ciphertext, secret_key)` returns an RNS polynomial representing
`b+a*s=m+e` modulo Q. It does not remove noise or apply plaintext scaling.

Sampling uses [`Tensor.normal_`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.normal_.html)
followed by rounding. This is a rounded continuous normal, not a security-proven
discrete Gaussian sampler. Torch generators are pseudorandom. The repository's
AI-written, briefly reviewed, never-use-in-a-product disclaimer applies here.

```python
import torch

from torch_gl import Engine, RNSRing, create_secret_key

ring = RNSRing.for_parameters(n=4, p=9, limb_count=3)
engine = Engine(ring, generator=torch.Generator().manual_seed(1935))
secret_key = create_secret_key(
    ring, generator=torch.Generator(device=engine.device).manual_seed(1935 + 100000)
)
message = 10000 * ring.generator("X")
ciphertext = engine.encrypt(message, secret_key)
decrypted = engine.decrypt(ciphertext, secret_key)
error = decrypted.centered_coefficients() - message.centered_coefficients()
print(error.abs().max())
```

Run `python examples/encryption.py` for all four requested messages. Set
`Engine(..., compiled=True)` to reuse the full native multiplication kernel
per limb. Compilation is lazy; its first call is a warmup cost. The engine does
not change process-wide thread settings. CPU/CUDA placement follows the polynomial API.

Tests verify the exact ciphertext equation, exact agreement with captured noise,
binary keys, shared residues, fresh ciphertexts, zero-noise recovery, compiled
graph capture, approximate noise statistics, and CRT integers above int64.
The saved SageMath KAT file remains unchanged.

For scaled complex matrix messages, use `MatrixEncoder.encode()` before
encryption and `MatrixEncoder.decode()` after decryption. It implements the
paper's Section 3.1 and uses Python integers at the RNS conversion boundary,
with centering modulo the full aggregate Q. See [encoding](encoding.md).

## Homomorphic multiplication before key switching

`engine.multiply(left, right)` or `left * right` multiplies two two-component
ciphertexts and returns `QuadraticCiphertext(c0,c1,c2)`:

```text
c0 = b0*b1
c1 = b0*a1 + b1*a0
c2 = a0*a1
```

`engine.decrypt(product, secret_key)` evaluates `c0+c1*s+c2*s^2`, using Horner
evaluation `(c2*s+c1)*s+c0` to avoid separately materializing `s^2`. The same
secret s is supplied; its decryption basis expands from `(1,s)` to `(1,s,s^2)`.
The engine method uses its configured compilation settings. Operator `*` is eager.
The inputs must use the same ring, device, and secret key; key identity is not
tracked in ciphertexts, so the caller must ensure the shared secret.

Scales multiply: inputs at scale Delta produce a product at scale Delta^2.
Decode with `MatrixEncoder(ring, scale=Delta**2)`. Multiplication does not
automatically rescale or relinearize, and ciphertexts carry no scale metadata. The modulus
product Q must accommodate coefficient growth. The decoded operation is the
elementwise/Hadamard product of the input matrices. The paper's matrix trace
product is a separate operation.

```python
import torch

from torch_gl import Engine, MatrixEncoder, RNSRing, create_secret_key

ring = RNSRing.for_parameters(n=4, p=9, limb_count=4)
engine = Engine(ring)
secret_key = create_secret_key(
    ring, generator=torch.Generator(device=engine.device).manual_seed(1935 + 100000)
)
encoder = MatrixEncoder(ring)  # Default Delta=2**40.
product_decoder = MatrixEncoder(ring, scale=encoder.scale**2)
left = torch.randn(encoder.slot_shape, dtype=torch.complex128)
right = torch.randn(encoder.slot_shape, dtype=torch.complex128)
left_ct = engine.encrypt(encoder.encode(left), secret_key)
right_ct = engine.encrypt(encoder.encode(right), secret_key)
product_ct = engine.multiply(left_ct, right_ct)
recovered = product_decoder.decode(engine.decrypt(product_ct, secret_key))
print((recovered - left * right).abs().max())
```

`pytest tests/test_homomorphic_multiplication.py` verifies the component formula,
the exact noisy decryption equation, eager/compiled paths, and encrypted
Sage-KAT messages decoded at squared scale. The tests use four limbs and verify
a coefficient-growth bound below Q/2, including product coefficients above int64.
Multiplication of an already quadratic ciphertext is outside this initial API.

## Encrypted matrix multiplication with a transposed right input

`engine.matrix_multiply(left_ct,right_ct)` computes the four-component result
`MatrixProductCiphertext(c00,c10,c01,c11)` in this order:

```text
c00 = b0 mm b1^T
c10 = a0 mm b1^T
c01 = b0 mm a1^T
c11 = a0 mm a1^T
```

Here `mm` is the existing NTT-domain matrix multiplication, and `^T` substitutes
`(X,Y,W)->(Y,X,W)`. `matrix_transpose()` performs this substitution without
conjugating coefficients or changing W. The resulting plaintext operation
is batch-wise `A @ B.T`, with W batches independent. It requires X rows and
Y columns in the encoder layout; W may occur anywhere in the coefficient order.

For a smaller-ring secret `s(X,W)`, decryption evaluates
`c00+c10*s(X,W)+c01*s(Y,W)+c11*s(X,W)*s(Y,W)`. All these key products are
ordinary polynomial products. The implementation factors the expression into
three products, avoiding a separately materialized mixed secret. The exact
noisy result is `(m0+e0) mm (m1+e1)^T`. Key switching is not performed.
Equal input scales Delta give output scale Delta^2; the caller supplies this
scale to the decoder. Matrix contraction currently uses the correctness-oriented
modular kernel described in [matrix notes](matrix.md), even when ordinary
polynomial multiplication is configured for compilation.

```python
import torch

from torch_gl import Engine, MatrixEncoder, RNSRing, create_secret_key

ring = RNSRing.for_parameters(n=4, p=9, limb_count=4)
engine = Engine(ring)
secret_key = create_secret_key(
    ring, generator=torch.Generator(device=engine.device).manual_seed(1935 + 100000)
)
encoder = MatrixEncoder(ring)
decoder = MatrixEncoder(ring, scale=encoder.scale**2)
left = torch.randn(encoder.slot_shape, dtype=torch.complex128)
right = torch.randn(encoder.slot_shape, dtype=torch.complex128)
left_ct = engine.encrypt(encoder.encode(left), secret_key)
right_ct = engine.encrypt(encoder.encode(right), secret_key)
product_ct = engine.matrix_multiply(left_ct, right_ct)
decrypted = engine.decrypt(product_ct, secret_key)
recovered = decoder.decode(decrypted)
print((recovered - left @ right.transpose(-1, -2)).abs().max())
```

`pytest tests/test_encrypted_matrix_multiplication.py` checks the four component
equations, the explicit key basis, smaller-ring sampling and supplied keys,
transpose semantics, zero/noisy recovery, compiled ordinary decryption products,
and p=729. It verifies that Q accommodates coefficient growth and rejects
Y-dependent secrets.

The unrelinearized APIs above remain available. To reduce a Hadamard or matrix
product back to `Ciphertext(b,a)`, use `KeySwitcher.relinearize()` or
`KeySwitcher.relinearize_matrix()`. See [key switching](key_switching.md) for
the small/big distinction, grouped RNS decomposition configured by k, and
switching between independent secrets.

## RNS-friendly rescaling

`polynomial.rescale()` drops the last modulus `q_last` and returns, for each
remaining prime `q_i`:

```text
new_coefficient_i = (coefficient_i - coefficient_last) * inverse(q_last, q_i) mod q_i
```

`coefficient_last` is the canonical residue in `[0,q_last)`, applied separately
to the real and imaginary channels. For any integer coefficient lift `value`,
the result is exactly `value // q_last` modulo the shorter chain's product.
This uses floor-style rounding, including negative coefficients, rather than
nearest rounding. It handles any supplied modulus order. Tensor subtraction and
reduction use int32; Montgomery products use local int64 intermediates. The
last-prime inverses are cached per ring, and no CRT export or coefficient
transfer to the CPU is required. Rescaling requires at least two input limbs.

`engine.rescale(ciphertext)` applies the same operation to every component of
`Ciphertext`, `QuadraticCiphertext`, or `MatrixProductCiphertext`. It preserves
the component order and decryption-key basis. Both APIs return new values;
the original polynomial, ciphertext, and engine stay at their existing level.
The message scale becomes `old_scale / q_last`. Component rounding contributes
additional decryption error, so decoded slots are approximately preserved.

Scales remain caller-managed. Use an engine and decoder for the shorter ring
and project the same secret with `secret_key.drop_last_limb()`. Projection keeps
its coefficients unchanged; calling `secret_key.rescale()` would divide them.
Evaluation keys still belong to their original modulus basis, so switching
after rescaling needs separately generated keys for that shorter basis.

```python
import torch
from torch_gl import Engine, MatrixEncoder, RNSRing, create_secret_key

ring = RNSRing.for_parameters(n=4, p=9, limb_count=4)
engine = Engine(ring, generator=torch.Generator().manual_seed(101))
secret_key = create_secret_key(ring, torch.Generator().manual_seed(11))
encoder = MatrixEncoder(ring)  # Delta = 2**40.
generator = torch.Generator().manual_seed(301)
left, right = [
    torch.randn(encoder.slot_shape, dtype=torch.complex128, generator=generator) for _ in range(2)
]
left_ct = engine.encrypt(encoder.encode(left), secret_key)
right_ct = engine.encrypt(encoder.encode(right), secret_key)
product = engine.multiply(left_ct, right_ct)  # Scale = Delta**2.
scaled = engine.rescale(product)

lower_ring = scaled.c0.ring
lower_engine = Engine(lower_ring)
lower_key = secret_key.drop_last_limb()
decoder = MatrixEncoder(lower_ring, scale=encoder.scale**2 / ring.moduli[-1])
recovered = decoder.decode(lower_engine.decrypt(scaled, lower_key))
torch.testing.assert_close(recovered, left * right, atol=2e-8, rtol=2e-9)
```

Run `python examples/rescaling.py` for plaintext and encrypted Hadamard/matrix
products. `pytest tests/test_rescaling.py` checks independent Python-integer
quotients above int64, signed boundaries, arbitrary limb order, repeated
rescaling, input preservation, same-key projection, all three ciphertext forms,
reduced-scale decoding, and the single-limb rejection.
