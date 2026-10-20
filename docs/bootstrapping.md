# GL bootstrapping

This implementation follows the **Section 4 reference construction** of
[Efficient Bootstrapping in Fully Homomorphic Encryption for Matrix Arithmetic](https://eprint.iacr.org/2026/956).
It refreshes approximate complex GL ciphertexts, using public evaluation keys
and no secret/decryption callback. It does not bootstrap the integer variant.

```sh
python examples/bootstrapping.py
python examples/bootstrapping.py --batch 2
```

The example starts at one 30-bit prime, refreshes into four primes, decrypts to
check the plaintext reference, and performs an encrypted square afterward.
It uses tiny, insecure demonstration dimensions, as do the other examples.
`--device cuda` selects the same tensor algorithm on a CUDA machine; the
development machine has no CUDA hardware, so GPU execution is unverified.

## Ring changes and linear transforms

Write R = Z_Q[i][X,W]/(X^n-i,Phi_p(W)),
R' = R[Y]/(Y^n-i), and Raux = R[V]/Phi_p(V).
The secret s(X,W) is independent of both packing variables.

1. **Split:** extract the n coefficients of Y from each public R' ciphertext
   component. Each pair is a valid ciphertext over R, under the same s(X,W).
2. **ModRaise:** lift each component's centered representative from q0 to the
   full Q chain. Decryption becomes m+e+q0*I; this step does not remove I.
3. **X transform:** merge R ciphertexts into Y coefficients of R', swap X/Y,
   switch from s(Y,W) to s(X,W), and perform a ciphertext-plaintext matrix
   trace. Swap back, key-switch back to s(X,W), and split along Y.
4. **W transform:** merge R ciphertexts into V coefficients of Raux, swap W/V,
   switch from s(X,V) to s(X,W), and perform the corresponding auxiliary
   ciphertext-plaintext matrix trace. Swap back, key-switch, and split along V.
5. **Recombine:** after EvalMod and reverse transforms, merge R ciphertexts
   into the original Y coefficient positions in R'.

For X the merge width is n. For W it is phi(p): shorter groups are zero-padded,
and longer input sequences are divided into groups. `bootstrap_many` batches
multiple input ciphertexts through those same ring-change functions. It retains
the ordering of their Y coefficients even when n and phi(p) differ.

CtS applies U_X^-1 and U_W^-1; StC applies U_W and U_X in reverse order. Roots
are exp(2*pi*i*5^j/(4n)) and exp(2*pi*i*2^ell/p), consistent with MatrixEncoder.
The compact W power basis is not orthogonal: U_W^* U_W is **not** p*I.
The inverse is computed as a genuine inverse. Auxiliary matrix multiplication
uses the algebra trace of powers of a primitive p-th root (Ramanujan sums),
not a power-of-two cyclic convolution.

Each axis consumes one level. This implementation uses ordinary transpose and
the corresponding interpolated matrix instead of the equivalent Hermitian
convention. Each trace group uses two key switches: **four per CtS or StC**
when one X group and one W group suffice. If phi(p) < n, the W groups each
use their own pair, as prescribed in Section 4.
The optional three-switch optimization, sparse-secret encapsulation, and lazy
QP ModRaise of Section 5.1 are not implemented. Performance should not be
compared to the paper's optimized benchmark.

## Homomorphic EvalMod

Slot values after CtS are the Gaussian coefficients divided by q0, plus integer
carries. EvalMod separates real/imaginary parts using canonical conjugation and
a small switch, then applies the following real circuit independently:

- Range scaling by 2*pi/2^r.
- Degree-9 sine and degree-10 cosine Taylor approximations on an interval
  bounded by 0.85 in magnitude.
- r double-angle steps to recover sin(2*pi*x).
- A configurable odd-degree arcsine approximation, then division by 2*pi.

Balanced encrypted powers use small relinearization keys. A final encoded
coefficient layer aligns every term's actual scale; additions do not relabel
incompatible ciphertexts. Maximum-level evaluation keys are reused at all
shorter levels, including partially active Han–Ki digits. Nearest-integer
rescaling reduces rounding bias in this circuit; the existing default
`RNSPoly.rescale()` behavior remains floor division. The optional
`rounding="nearest"` is also available on `Evaluator`.

The default is r=6, arcsine degree 7, |I| <= 8, and real/imaginary message
coefficients (including their existing noise) bounded by q0/16. These are
**caller-declared numerical bounds**, not checks on encrypted data or a proof
of a cryptographic failure probability. Larger carry bounds need more
double-angle steps and more levels. Larger message intervals need an adequate
arcsine degree and accept larger approximation error. Inputs outside the
declared domain can fail. The example demonstrates moderate precision; it does
not reproduce the paper's degree-49 approximation or precision measurements.

The default consumes 22 levels: CtS 2, complex split 1, range scaling 1,
Taylor evaluation 5, doubling 6, arcsine 4, final scaling 1, and StC 2.
A chain of 26 primes therefore returns four. At least two output primes are
required. Input must be in the standard X/Y/W layout at the bottom single
prime; it may be projected there using `ScaledCiphertext.at_level(1)`.

The output scale accounts for all actual rescaling factors. It is generally
close to, but not exactly equal to, the input scale. Keep the returned scale.
The refresh changes the modulus capacity and removes the q0*I carry; it does
not eliminate preexisting encryption or arithmetic error.

## API and validation

`generate_bootstrap_keys(parameters, secret, ...)` is the explicit client-side
key-generation step. `Bootstrapper(keys)` retains only public evaluation
material. `bootstrap(value)` or `bootstrap_many(values)` performs refresh.
Optional `on_stage(name, parts)` reports ciphertext diagnostics; its return
value is ignored and it supplies no data to the computation.

Tests verify nonzero carry removal, real/imaginary periodic reduction near the
declared domain boundaries, coefficient-to-slot values, inverse transforms,
padding/grouping for phi(p) above/below n, repeated refresh with the same keys,
and a genuine encrypted product after refresh. During the end-to-end test,
`Engine.decrypt` and `MatrixEncoder.decode` are patched to fail if refresh
tries to call them. Nearest rescale is also checked against independent signed
Python-integer quotient calculations above int64.
