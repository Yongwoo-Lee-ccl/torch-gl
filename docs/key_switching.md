# Key switching with grouped RNS decomposition

`KeySwitcher` implements the Gentry–Lee SwitchSmall and SwitchBig distinction
using Han–Ki's full-RNS grouped decomposition, ModUp, evaluation-key inner
product, and ModDown. The implementation operates at a fixed ciphertext basis;
it does not implement dropping levels or plaintext rescaling.

The complete runnable examples are in
[examples/key_switching.py](../examples/key_switching.py). They cover independent
keys, raw SwitchSmall/SwitchBig, Hadamard and matrix relinearization, file-backed
matrix keys, and an X/W-only context. See [the example guide](examples.md) for
commands and expected behavior, and [the NTT walkthrough](ntt.md) for the
transform kernels used here.

## Shared setup for the snippets

Run this setup before the snippets below. The matrix examples use four prime
limbs to hold coefficients at scale `2**80` after multiplication. These are
small demonstration parameters, not a security parameter selection.

```python
import torch
from torch_gl import Engine, KeySwitcher, MatrixEncoder, RNSRing, create_secret_key

ring = RNSRing.for_parameters(n=4, p=9, limb_count=4)
engine = Engine(ring, generator=torch.Generator().manual_seed(101))
secret_key = create_secret_key(ring, generator=torch.Generator().manual_seed(11))
encoder = MatrixEncoder(ring)  # scale=2**40
product_decoder = MatrixEncoder(ring, scale=encoder.scale**2)
data_generator = torch.Generator().manual_seed(301)
left = torch.randn(encoder.slot_shape, dtype=torch.complex128, generator=data_generator)
right = torch.randn(encoder.slot_shape, dtype=torch.complex128, generator=data_generator)
message = left
left_ct = engine.encrypt(encoder.encode(left), secret_key)
right_ct = engine.encrypt(encoder.encode(right), secret_key)
```

## Configure by k

```python
switcher = KeySwitcher(ring, k=4)
print(switcher.k, switcher.dnum)
```

`k` is both the number of temporary key primes `p_i` and the maximum number of
ciphertext primes in each decomposition digit. With `L+1=ring.limb_count`,
`dnum=ceil((L+1)/k)`. The active primes are partitioned into consecutive groups;
the final group may contain fewer primes. Its actual product is used directly,
so no unused level primes need to be introduced. For a divisible full basis,
this is Han–Ki's `alpha=k`, `dnum=(L+1)/k` construction. Reusing keys at lower
levels through the paper's padding procedure is outside this fixed-level API.

The auxiliary product `P=product(key_moduli)` must be at least the largest
digit product. Automatic selection finds compatible primes above the largest
ciphertext prime, all below `2**30`. Alternatively provide exactly k disjoint
compatible primes with `KeySwitcher(ring,k=4,key_moduli=(...))`.
Larger k means fewer digits and more temporary primes; evaluation-key sizes
and switching cost depend on both. This configuration is not a security estimate.

## Han–Ki decomposition and conversion

For digit product `Q_j`, let `hat_Q_j=Q/Q_j`. Decomposition gives
`d_j=[component*hat_Q_j^-1]_(Q_j)` using per-prime modular products. ModUp
extends each digit to QP through the paper's fast conversion:

```text
Conv_M->B(x) = sum(centered(x_i * hat_M_i^-1 mod m_i) * hat_M_i) mod each b
hat_M_i = M / m_i
```

Centering applies to each small CRT summand, not to a first-prime lift of a
large coefficient. This approximate CRT sum can differ from a canonical lift
by a multiple of M; it preserves every source residue. The decomposition and
conversion inverse constants are fused. Every target-prime sum is reduced as
it is accumulated; no large tensor integers or floating conversions are used.
Metadata such as Q, P, and cofactors uses Python integers.

Evaluation keys satisfy, for every digit,
`key_b_j+key_a_j*s_target=P*hat_Q_j*s_source+e_j (mod QP)`.
The mask is sampled uniformly directly in the NTT domain, an invertible image
of the uniform coefficient distribution. Each e_j is sampled once in integer
Gaussian coefficient channels and shared across all primes. Stored keys are
int32 Montgomery NTT tensors; no source or destination secret is stored in them.
Evaluation keys now use GS mixed-radix digit order internally. Regenerate keys
created by older natural-order implementations before using this version.

The inner products are evaluated in the NTT domain and transformed back.
ModDown applies Han–Ki's coefficient-wise formula:

```text
output_q = (accumulator_q - Conv_P->q(accumulator_P)) * P^-1 mod q
```

Both components incur the approximate-division error, in addition to the
evaluation-key error divided by P. Thus switching is approximate even when
evaluation-key noise is zero. It preserves the plaintext scale and the Q basis.
It does not promise bit-exact decryption before and after switching.

## SwitchSmall

SwitchSmall keys live in `Z_QP[i][X,W]/(X^n-i,Phi_p(W))`. A full component is
interpreted as independent Y coefficient polynomials. X/W transforms are batched
over those Y coefficients, and the same smaller-ring key is broadcast over
the batch. No Y NTT is performed on this path.

```python
import torch

from torch_gl import create_secret_key

source = Engine(ring, generator=torch.Generator().manual_seed(1))
source_secret_key = create_secret_key(
    ring, generator=torch.Generator(device=source.device).manual_seed(1 + 100000)
)
target = Engine(ring, generator=torch.Generator().manual_seed(2))
target_secret_key = create_secret_key(
    ring, generator=torch.Generator(device=target.device).manual_seed(2 + 100000)
)
switcher = KeySwitcher(ring, k=4)
key = switcher.generate_key(source_secret_key, target_secret_key)
original = source.encrypt(encoder.encode(message), source_secret_key)
switched = switcher.switch_ciphertext(original, key)
recovered = encoder.decode(target.decrypt(switched, target_secret_key))
torch.testing.assert_close(recovered, message, atol=2e-8, rtol=2e-8)
```

`switch_small(component,key)` returns an encryption of `component*s_source`.
`switch_ciphertext()` adds the original b. For Hadamard multiplication:

```python
key = switcher.generate_hadamard_key(secret_key)
product = engine.multiply(left_ct, right_ct)
switched = switcher.relinearize(product, key)
# Decode at left_scale*right_scale; key switching does not rescale.
recovered = product_decoder.decode(engine.decrypt(switched, secret_key))
torch.testing.assert_close(recovered, left * right, atol=2e-8, rtol=2e-8)
```

This uses a single small key from s^2 to s and returns `Ciphertext(b,a)`.

## SwitchBig and matrix relinearization

SwitchBig keys live in the full X/Y/W ring and include Y transforms. Destinations
are still smaller-ring secrets embedded as constant in Y. For the implemented
ordinary-transpose matrix product, the two source secrets are `s(Y,W)` and
`s(X,W)*s(Y,W)`:

```python
keys = switcher.generate_matrix_keys(secret_key)
product = engine.matrix_multiply(left_ct, right_ct)
switched = switcher.relinearize_matrix(product, keys)
recovered = product_decoder.decode(engine.decrypt(switched, secret_key))
torch.testing.assert_close(recovered, left @ right.transpose(-1, -2), atol=2e-8, rtol=2e-8)
```

It computes `(c00,c10)+SwitchBig(c01,Y_key)+SwitchBig(c11,mixed_key)`.
This is exactly two big switches and preserves the decryption basis used by
the existing matrix product. The paper's normalized-adjoint version uses inverse
substitutions in the source secrets; this library's current transpose leaves W
unchanged, as required by its matrix multiplication API.

## Code flow in key_switching.py

`KeySwitchParameters` partitions the Q limbs into groups of up to k primes.
`KeySwitcher.__init__()` binds two kinds of coefficient conversion:

- One `RNSBasisConverter` per digit, with the decomposition inverse
  `(Q/Q_j)^-1` fused into its CRT-summand inverse constants.
- One converter from the P primes back to the Q primes for ModDown.

`generate_key()` then visits one extended-basis prime at a time. It transforms
the small destination secret, broadcasts it over Y slots for a big key, and
forms each digit's pair in Montgomery NTT form:

```text
key_a = uniform NTT-domain mask
key_b = NTT(error) - key_a*NTT(destination)
        + encoded(P*(Q/Q_j) mod prime)*NTT(source)
```

The source term is zero at P primes and at Q primes outside the digit's group.
Each error polynomial is sampled once and shared across all prime limbs.
Cached `NativeTransform` objects contain plans/twiddles, not secret coefficients.

The common `_apply(component,key,kind)` routine follows this sequence:

1. `converter.prepare(..., inverses=...)` computes each digit's signed small
   CRT summands once in coefficient form, before visiting target primes.
2. For one target prime, `convert_prepared()` performs ModUp for each digit.
3. SwitchSmall moves the Y coefficient axis to the front as a batch, then uses
   X/W NTTs; SwitchBig uses full X/Y/W NTTs.
4. For each digit, multiply its transformed coefficients by the matching
   `(key_b,key_a)` and add to two modular accumulators.
5. Apply inverse transforms once to each accumulator. Small switching restores
   the Y coefficient axis to its original position.
6. Retain coefficient outputs at the first k primes (P), and prepare their
   ModDown CRT summands. Subsequent Q outputs use
   `(accumulator_q-Conv_P->q(accumulator_P))*P^-1 mod q`.
7. Return the two resulting Q-basis polynomials as `Ciphertext(b,a)`.

The public methods add the components that do not need switching:

| Method | Result |
| --- | --- |
| `switch_small(component,key)` | Encrypts `component*source_secret` using a small switching key |
| `switch_big(component,key)` | Encrypts `component*source_secret` using a big switching key |
| `switch_ciphertext(ct,key)` | `(ct.b+switched.b, switched.a)` with SwitchSmall applied to ct.a |
| `relinearize(product,key)` | `(product.c0+switched.b, product.c1+switched.a)` with SwitchSmall applied to c2 |
| `relinearize_matrix(product,keys)` | `(c00,c10)` plus SwitchBig(c01,Y_key) plus SwitchBig(c11,mixed_key) |

Coefficient decomposition/ModDown must not be performed by treating arbitrary
NTT slots as coefficients. Evaluation keys and intermediate transforms retain
GS digit order consistently; inverse transforms restore coefficients before
ModDown. `switch_small` and `switch_big` are approximate because of conversion
and division errors, even with zero evaluation-key noise.

## Storage, constraints, and verification

Native X/Y twisted transforms and power-of-three W transforms are required.
Source and target secrets must admit a consistent signed-int32 coefficient
lift; binary secrets, their squares, and the mixed matrix secret satisfy this
for the tested parameters. Ciphertext components have no such small-lift
requirement: switching handles arbitrary Q residues without composing CRT integers.
Keys and ciphertexts must share parameters and device. Key identities are not
tracked; the caller must use keys generated for the actual source and target.

By default evaluation keys are resident tensors. Pass `storage_directory=...`
to generation to write one Torch file per extended-basis limb. Switching maps
one file at a time on CPU; this bounds resident key memory while preserving
the same computation. File-backed key reading is included in operation timings.
Existing files are not overwritten. Sampling uses the prototype's Torch RNG
and rounded Gaussian convention; the README disclaimer applies.

`tests/test_key_switching.py` checks independent s1->s2 switching, Hadamard
relinearization, and matrix relinearization with two big switches, with noisy
keys and several k values. Additional tests independently verify centered CRT
conversion, ModDown against integer division, decomposition recombination,
the exact evaluation-key relation at zero noise, the rounding-error bound,
and agreement of resident and file-backed keys. Saved Sage KATs are unchanged.

## Large benchmark

```sh
python scripts/benchmark_key_switching.py --n 128 --p 243 --modulus-bits 600 --k 4
```

The script selects twenty primes close to `2**30` for an actual 600-bit Q,
reserves four larger primes for a 120-bit P, and derives dnum=5. It records
matrix multiplication before switching and the two-big-switch relinearization
separately. Setup/key generation and decryption checks are excluded; full
operation times include NTTs, decomposition, conversion, and key reads.
The default is one measured run with no whole-operation warmup; tables and
evaluation keys are prepared beforehand. Use `--warmups` and `--runs` to request
repeated measurements. GPU times synchronize around each operation.

The benchmark uses dense ciphertext masks encrypting integer constants, checks
every coefficient's post-switching error against all Q limbs, and saves its
environment and timings to `benchmarks/key_switching_600bit.json`. Temporary
evaluation-key files are removed after completion. Matrix contraction uses the
exact split-15 batched GEMM described in the [matrix notes](matrix.md).
