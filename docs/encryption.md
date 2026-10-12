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

`Engine` samples a binary coefficient tensor once and encodes it in every limb.
Both Gaussian channels have coefficients in `{0,1}`. A supplied `secret_key`
must be binary and match the context and device. In the full X/Y/W ring,
the default key is sampled in the smaller X/W ring and embedded at Y degree zero.
The constructor also accepts a key from that smaller ring with the same moduli
and axis order. `engine.secret_key` returns the full-ring embedding;
`engine.small_secret_key` returns the smaller-ring value. Decryption accepts either.
Supplied full-ring binary keys remain supported for ordinary encryption, but
encrypted matrix multiplication rejects any key depending on Y.

Each encryption samples uniform `a` independently per limb and one shared error
tensor `e`. The latter is sampled from a mean-zero normal distribution with
standard deviation 3.2, rounded to integers, then encoded in every limb.
It computes `b=-a*s+e+m` and returns `Ciphertext(b,a)`, containing no key or error.

`engine.encrypt(message)` accepts an RNS polynomial, an integer coefficient
tensor, an integer constant, or a Gaussian integer pair.
`engine.decrypt(ciphertext, secret_key)` returns an RNS polynomial representing
`b+a*s=m+e` modulo Q. It does not remove noise or apply plaintext scaling.

Sampling uses [`Tensor.normal_`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.normal_.html)
followed by rounding. This is a rounded continuous normal, not a security-proven
discrete Gaussian sampler. Torch generators are pseudorandom. The repository's
AI-written, briefly reviewed, never-use-in-a-product disclaimer applies here.

```python
import torch
from torch_gl import Engine, RNSRing

ring = RNSRing.for_parameters(n=4, p=9, limb_count=3)
engine = Engine(ring, generator=torch.Generator().manual_seed(1935))
message = 10000 * ring.generator("X")
ciphertext = engine.encrypt(message)
decrypted = engine.decrypt(ciphertext, engine.secret_key)
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
Decode with `MatrixEncoder(ring, scale=Delta**2)`. There is no rescaling or
relinearization here, and ciphertexts carry no scale metadata. The modulus
product Q must accommodate coefficient growth. The decoded operation is the
elementwise/Hadamard product of the input matrices. The paper's matrix trace
product is a separate operation.

```python
import torch
from torch_gl import Engine, MatrixEncoder, RNSRing

ring = RNSRing.for_parameters(n=4, p=9, limb_count=4)
engine = Engine(ring)
encoder = MatrixEncoder(ring)  # Default Delta=2**40.
product_decoder = MatrixEncoder(ring, scale=encoder.scale**2)
left = torch.randn(encoder.slot_shape, dtype=torch.complex128)
right = torch.randn(encoder.slot_shape, dtype=torch.complex128)
left_ct = engine.encrypt(encoder.encode(left))
right_ct = engine.encrypt(encoder.encode(right))
product_ct = engine.multiply(left_ct, right_ct)
recovered = product_decoder.decode(engine.decrypt(product_ct, engine.secret_key))
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
from torch_gl import Engine, MatrixEncoder, RNSRing

ring = RNSRing.for_parameters(n=4, p=9, limb_count=4)
engine = Engine(ring)
encoder = MatrixEncoder(ring)
decoder = MatrixEncoder(ring, scale=encoder.scale**2)
left = torch.randn(encoder.slot_shape, dtype=torch.complex128)
right = torch.randn(encoder.slot_shape, dtype=torch.complex128)
left_ct = engine.encrypt(encoder.encode(left))
right_ct = engine.encrypt(encoder.encode(right))
product_ct = engine.matrix_multiply(left_ct, right_ct)
decrypted = engine.decrypt(product_ct, engine.small_secret_key)
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
