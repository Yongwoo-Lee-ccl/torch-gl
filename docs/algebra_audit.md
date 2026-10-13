# Algebra audit and performance review

Reviewed the implementation against the local [Gentry–Lee paper](../2025-1935.pdf),
especially Sections 2.1, 3.1–3.6, and 5.1, and against the local
[Han–Ki paper](../2019-688.pdf), Sections 2.2 and 3.1. Relevant Gentry–Lee
equations were checked in rendered pages 12–16 and 22 to preserve the
coefficient-conjugation bars that disappear in extracted text.

The implementation is consistent with the currently requested ordinary-transpose
variant. It implements arithmetic components of the scheme, rather than every
operation of a complete FHE system. Passing these checks is not a security audit.

## Components checked

| Component | Implementation and algebra checked | Evidence |
| --- | --- | --- |
| Quotient rings | Formal `i²=-1`, `X^n=i`, `Y^n=i`; `Phi_(3^a)=1+W^(3^(a-1))+W^(2*3^(a-1))`; optional axis subsets and monic reduction | Frozen Sage multiplication KATs and quotient identities |
| Modular arithmetic | Stored canonical residues below `2^30`; Montgomery `R=2^31`; signed-int64 intermediates fit without overflow | Boundary-value modular/Montgomery tests |
| Native NTT | Twists enforce `X^n=i`; Gaussian branches evaluate at `+I/-I`; W evaluates only primitive roots of order `3^a` | Direct finite-field evaluation, inverse tests, Sage product KATs |
| GS/CT ordering | GS forward gives mixed-radix digit order; CT inverse consumes that same order; pointwise products and switching keys retain it | Independent digit-order evaluation and cyclic-convolution tests; gather-rejection tests |
| RNS | Independent prime limbs represent one polynomial modulo the product Q; signed export centers modulo Q | CRT reconstruction and cross-limb consistency tests |
| Complex encoding | Section 3.1 inverse canonical embedding, twisted X/Y roots and primitive W roots; nearest Gaussian-integer rounding after scaling | Frozen high-precision Sage encoding KATs, direct evaluation, encoded plaintext products |
| Encryption | `b=-a*s+e+m`; one integer error tensor shared across RNS limbs; s is sampled in X/W by `create_secret_key` for the standard X/Y/W context | Captured-noise ciphertext equations, independent RNG/key tests |
| Small-secret embedding | Explicit X/W keys are temporarily embedded at Y degree zero, then ordinary polynomial multiplication is used | Encryption/decryption equations and supplied-small-key compiled tests |
| Hadamard multiplication | `(b0*b1,b0*a1+b1*a0,a0*a1)` decrypts under `(1,s,s²)` | Component equations, plaintext product identity, encoded slotwise products |
| Ordinary matrix product | Coefficient contraction below decodes to `A@B`; encrypted transposition gives `A@B.T` | Independent integer coefficient contraction, canonical finite-field evaluation, complex batch products |
| Gaussian splitting | Section 5.1 map `real±I*imag` and its inverse; both split branches contract independently | Direct split evaluation and modular matrix boundary tests |
| SwitchSmall | Keys are in X/W; Y coefficient polynomials are independently switched using the same key, with Y as a batch axis | Independent-key switching and Hadamard relinearization tests |
| SwitchBig | Full X/Y/W transforms; source keys `s(Y,W)` and `s(X,W)*s(Y,W)` for this transpose convention; two switches produce two components | Four-component decryption identity and post-switching matrix decoding |
| Han–Ki decomposition | Groups of up to k primes; `dnum=ceil(limb_count/k)`; key relation `b_j+a_j*s_target=P*(Q/Q_j)*s_source+e_j` | Group recombination, independent evaluation-key relation at zero noise, several k/ragged groups |
| ModUp/ModDown | Centered small CRT summands preserve source residues; `(acc_q-Conv_P->q(acc_P))*P^-1` gives approximate division by P | Python CRT-sum oracle, division-error bounds, all-limb post-switching checks |

## Trace convention retained

For the user-requested ordinary slot product, write coefficient matrices as
`L[r,s](W)` and `R[t,u](W)`. The root sum is zero unless `s+t=0` or `s+t=n`,
where it is n or n*i respectively. Therefore

```text
C[r,u](W) = n * (L[r,0](W)*R[0,u](W)
                + i*sum(L[r,s](W)*R[n-s,u](W), s=1,...,n-1)).
```

Equivalently, this is n times the constant coefficient in Z after multiplying
`L(X,Z,W)*R(Z,Y,W)` and reducing `Z^n=i`. W can be transformed separately;
the remaining contraction is an ordinary matrix multiplication of X/Y coefficient
matrices, with the right rows prepared as above. No X/Y NTT is needed.

Applying `(X,Y)->(Y,X)` to the right ciphertext leaves W and its Gaussian
coefficients unchanged. The decryption basis is consequently
`(1,s(X,W),s(Y,W),s(X,W)*s(Y,W))`. Both secrets are independent of the contracted
Z variable, so they can be taken outside the trace. The implementation's ordering
`(c00,c10,c01,c11)` assigns the X-secret term to c10 and the Y-secret term to c01.

The paper instead defines `circledast` using conjugated coefficients and inverse
substitutions, decoding to `A@B.conj().T/n`; its transformed right secret is
`conjugate(s)(Y^-1,W^-1)`. This is an intentional difference requested by the
user, not a missing conjugation in the current API. Adjoint and inverse-W
substitutions have not been added.

There is also a cross-component ordering inconsistency in the paper's page-16
display: its initial tuple assigns d1 to `b_u circledast a_v`, but the displayed
relinearization then uses d1 with the X secret. Theorem 3.12 on page 15 associates
the X secret with `a_u circledast b_v`. The implementation follows the
decryption identity and the user's explicit ordering rather than copying that
display literally.

## Exact batched GEMM

The optimized contraction splits each canonical 30-bit Montgomery value into
two 15-bit parts. Packing left parts vertically and right parts horizontally
computes all four partial products in one `torch.bmm` using float64.
If `2*n*(2^15-1)^2 <= 2^53`, every product, partial sum, and combined cross sum
is an exactly representable integer. Full 30-bit values are never multiplied
directly in floating arithmetic.

The low, cross, and high partial sums are reduced modulo q before shifts by
0, 15, and 30 bits. Their recombined int64 value is below `2^61`. Finally, one
Montgomery reduction converts the product's `R²` representation back to R.
Tests include q near `2^30`, values on 15-bit boundaries, two independent split
channels, and a dot product whose unsplit int64 accumulation would overflow.

The ciphertext path reuses four input W transforms across all four product
components. The RNS limb loop is deliberately streamed to bound memory; there
is no Python loop over the matrix inner dimension.

## Performance regression addressed

The first GS/CT implementation stacked a full butterfly tensor and then built
another full tensor with `torch.cat` for the twiddle products at every stage.
That added memory traffic. The revised implementation weights individual branch
outputs and stacks once, and skips unity twiddles at the innermost stage.

On one CPU process with four Torch threads, n=64, p=243, and 30-bit residues,
the contraction-only loop measured 267.2 ms versus 14.44 ms for split GEMM.
This is about 18x for that kernel, not a claim about complete-operation speed
or parity with a separate C++ implementation.
The component timings and command metadata are saved in
[the arithmetic benchmark](../benchmarks/matrix_n64_p243.json), generated by
`scripts/benchmark_matrix.py`. Its older small-secret comparison entries are
historical; encryption/decryption now use ordinary polynomial multiplication.

The full benchmark with a 300-bit Q, k=4, one warmup, and three measured runs
averaged 3.509 s for encrypted matrix multiplication and 11.959 s for its two
big switches. The switching check covered 1,327,104 coefficients, with maximum
absolute error 24,873 and agreement across every Q limb. The report is saved in
[the optimized benchmark](../benchmarks/key_switching_n64_p243_300bit_fixed.json).
File-backed key reads are included; setup and decryption are excluded.

An intermediate W-only implementation still using the Python contraction loop
averaged 26.966 s for matrix multiplication and 12.830 s for switching, saved in
[the intermediate benchmark](../benchmarks/key_switching_n64_p243_300bit_w_ntt_loop.json).
Those intermediate matrix samples were variable (19.7–35.5 s), with overlapping
diagnostic CPU work, so they are not a controlled speedup ratio. The optimized
matrix samples were stable at 3.50–3.51 s. The separate existing benchmark report
was written by the user's run and was not overwritten by these measurements.

## Limits and deferred operations

- Native W transforms and the encoder currently support powers of three, as
  requested. The paper's example p=17/Rader implementation is not implemented.
- Binary Gaussian-channel key coefficients and rounded Gaussian noise with
  standard deviation 3.2 are the user's prototype choices; they do not establish
  the security of the paper's experimental parameter set.
- Encryption also accepts externally supplied full-ring binary keys for generic
  experimentation. Such Y-dependent keys are outside the paper's small-secret
  scheme and are rejected for matrix decryption. `create_secret_key` generates
  the proper small-ring key in the standard context.
- Key switching uses a fixed active Q basis. Its final short group uses the
  actual product, rather than Han–Ki's padded maximum-level key reuse procedure.
  Dropping levels and reusing keys across those levels are not implemented.
- Rescaling, BGV-style integer encryption/modulus switching, rotations,
  encrypted conjugation/transpose switching, public-key encryption, and
  bootstrapping are not implemented. Scale is managed explicitly by callers.
- CUDA paths are present, but the development machine has no CUDA device.
  GPU correctness, throughput, and comparison to C++ remain unmeasured.

The README's AI-written, briefly checked, never-use-in-a-product disclaimer
applies to every component reviewed here.
