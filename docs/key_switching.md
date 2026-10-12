# Key switching with grouped RNS decomposition

`KeySwitcher` implements the Gentry–Lee SwitchSmall and SwitchBig distinction
using Han–Ki's full-RNS grouped decomposition, ModUp, evaluation-key inner
product, and ModDown. The implementation operates at a fixed ciphertext basis;
it does not implement dropping levels or plaintext rescaling.

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
source = Engine(ring, generator=torch.Generator().manual_seed(1))
target = Engine(ring, generator=torch.Generator().manual_seed(2))
switcher = KeySwitcher(ring, k=4)
key = switcher.generate_key(source.small_secret_key, target.small_secret_key)
original = source.encrypt(encoder.encode(message))
switched = switcher.switch_ciphertext(original, key)
recovered = encoder.decode(target.decrypt(switched, target.small_secret_key))
```

`switch_small(component,key)` returns an encryption of `component*s_source`.
`switch_ciphertext()` adds the original b. For Hadamard multiplication:

```python
key = switcher.generate_hadamard_key(engine.small_secret_key)
product = engine.multiply(left_ct, right_ct)
switched = switcher.relinearize(product, key)
# Decode at left_scale*right_scale; key switching does not rescale.
```

This uses a single small key from s^2 to s and returns `Ciphertext(b,a)`.

## SwitchBig and matrix relinearization

SwitchBig keys live in the full X/Y/W ring and include Y transforms. Destinations
are still smaller-ring secrets embedded as constant in Y. For the implemented
ordinary-transpose matrix product, the two source secrets are `s(Y,W)` and
`s(X,W)*s(Y,W)`:

```python
keys = switcher.generate_matrix_keys(engine.small_secret_key)
product = engine.matrix_multiply(left_ct, right_ct)
switched = switcher.relinearize_matrix(product, keys)
recovered = product_decoder.decode(engine.decrypt(switched, engine.small_secret_key))
```

It computes `(c00,c10)+SwitchBig(c01,Y_key)+SwitchBig(c11,mixed_key)`.
This is exactly two big switches and preserves the decryption basis used by
the existing matrix product. The paper's normalized-adjoint version uses inverse
substitutions in the source secrets; this library's current transpose leaves W
unchanged, as required by its matrix multiplication API.

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
evaluation-key files are removed after completion. The existing modular matrix
kernel is a correctness baseline, not an optimized modular GEMM.
