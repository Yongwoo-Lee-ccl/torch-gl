# RNS and research encryption

`RNSRing` defines a common polynomial basis over `Q=product(moduli)`.
`RNSPoly` stores a `MultivariatePoly` for each distinct prime modulus, with
identical axes and a common device. Limbs stay below `2^30`; multiplication
by the aggregate modulus never occurs in Torch int64.

## Prime selection and representation

`RNSRing.for_parameters(n=4, p=9, limb_count=3)` selects ascending primes
`q=multiplier*(4*n*p)+1` in the requested bit range, defaulting to 30 bits.
Explicit moduli can be supplied with `RNSRing.multivariate_poly(moduli,n,p)`;
other separable ideals use `RNSRing(moduli,axes)`. Limb order is part of the context.

For cyclotomic prime bases other than three, the selector also includes the
required padded radix-2 transform length in its congruence step.

Encoding uses the same integer coefficient tensor in every limb. Arithmetic
operates limb by limb; integer scaling avoids NTTs. Uniform sampling independently
samples each prime limb, corresponding to a uniform element modulo Q under CRT.

`centered_coefficients()` returns a signed int64 tensor for coefficients within
half the first prime. It checks agreement with all other limbs, so it does not
silently assume that range. For arbitrary values, `reconstruct()` performs exact
CPU CRT reconstruction with Python integers and returns a flat tuple in C order.
Full CRT export supports values above int64 and stays outside the arithmetic hot path.

## Engine and ciphertext

`Engine` samples a binary coefficient tensor once and encodes it in every limb.
Both Gaussian channels have coefficients in `{0,1}`. A supplied `secret_key`
must be binary and match the context and device. The default key uses the full
engine basis; smaller-ring secret embeddings are a separate future feature.

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
