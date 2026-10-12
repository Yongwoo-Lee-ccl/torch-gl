# Arithmetic and correspondence with the paper

This implements the polynomial arithmetic layer in §2.1 of Gentry–Lee,
*Fully Homomorphic Encryption for Matrix Arithmetic*, ePrint 2025/1935.
The local reference reviewed is `2025-1935.pdf`, especially pp. 5–6, 14, 18–19,
and 21–23. The large ring is `R'_q`; the smaller `R_q` omits `Y`.

## Canonical basis

Coefficient `coefficients[x_index, y_index, w_index, channel]` multiplies
`X^x_index Y^y_index W^w_index`, with channel zero for the real part and channel
one for the formal imaginary part. X/Y exponents are below `n`; W exponents
are below `phi(p)`. The large ring has `2 n² phi(p)` scalar coefficients,
and the smaller ring has `2 n phi(p)`.

Keep formal `i` distinct from a chosen modular root `I`. Replacing `i` with `I`
alone would discard half the ring. The Gaussian split in §5.1 evaluates both
`i=I` and `i=-I`; its inverse recovers the formal coefficient pair.

General monic ideals are separable: each relation involves one variable with
coefficients in `Z_q[i]`. Mixed-variable relations are not supported. Arbitrary
positive input axis lengths are normalized modulo `q` and reduced by monic long
division. Floating and complex tensors are rejected to preserve exactness.

## Mixed-radix transform

For `length=2^a 3^b`, choose a primitive root of exact order `length` and compute

```text
slots[frequency] = sum(coefficients[index] * root^(index * frequency)) mod q.
```

Cooley–Tukey decimation in time splits into two or three branches. Radix-3
butterflies use a primitive cube root. Output is in natural frequency order.
The inverse uses the inverse root and normalization by the modular inverse of
the transform length.

## Power-of-three cyclotomic transform

Let `p=3^exponent`, `block_length=p/3`, and let `root` have order `p`.
The canonical polynomial has degree less than `2*block_length`. Split into
lower and upper coefficient blocks. For root class `s=1,2`, form

```text
branch_s[j] = root^(s*j) * (lower[j] + root^(s*block_length) * upper[j]).
```

Use a length-`block_length` radix-3 transform with root `root^3` on each branch.
These are evaluations at primitive roots `root^(3t+1)` and `root^(3t+2)`.
The two-branch stage plays the radix-2 role in this compact cyclotomic transform;
it is twisted rather than an ordinary two-point DFT because the third branch
is excluded. There are `phi(p)=2*3^(exponent-1)` slots, without padding.

To invert, undo branch transforms and twists. The resulting blocks are
`lower+omega*upper` and `lower+omega²*upper`, where `omega=root^block_length`.
Subtract and multiply by `(omega-omega²)^-1` to recover the upper coefficients;
then recover the lower coefficients. Gaussian channels are independent along W.

## Twisted X/Y transforms

Choose a primitive `4n`-th root `rho` and `I=rho^n`. Split coefficients into
`positive=real+I*imag` and `negative=real-I*imag`.
Twist positive coefficient `j` by `rho^j` and use an n-point NTT with root
`rho^4`. For the negative branch, use `rho^-j` and `rho^-4`.
Merge transformed branches using the inverse Gaussian split.

Positive slots evaluate at `rho^(1+4k)`; negative slots at `rho^-(1+4k)`.
Each branch evaluates the roots of its binomial relation. Pointwise formal
Gaussian multiplication corresponds to componentwise split-branch products,
and hence to multiplication modulo `T^n-i`.

## Slot ordering versus the paper

Our transforms use arithmetic frequency order. The encoding in §4.1 uses
`rho^(5^j)` for X/Y and `eta^(gamma^ell)` for W, with inverse roots on its negative
branch. These enumerate the same root sets with permutations. Future encoders,
automorphisms, and matrix operations must explicitly implement those permutations.
The current API does not implement the paper's matrix-slot encoding.

## Multiplication and parameter conditions

Transform each variable axis, multiply Gaussian pairs pointwise, and invert.
Native axes are already reduced. General axes use zero-padded linear convolution
followed by monic ideal reduction. Independent axis transforms commute.

The paper's completely split moduli in §5.1 satisfy `q ≡ 1 (mod 4np)`.
These enable native X/Y transforms and the native power-of-three W transform.
Other primes may permit padded arithmetic but not the paper's split encoding.
The experiments in §5.2 use `p=17` and Rader's algorithm; that native transform
is deferred in favor of powers of three as requested. Other odd prime-power
ideals are handled by padded multiplication. Multi-prime RNS is also deferred;
this stage handles a single prime limb up to `2^62-1`.

Ordinary ring multiplication supports the paper's Hadamard product. Its
`circledast` matrix product in §3.4 instead applies a trace with substitutions
and contractions, so it must be implemented as a separate future operation.

## Correctness evidence and limits

The frozen KAT oracle multiplies in `GF(q)[X,Y,W,i]` and reduces by an explicit
multivariate ideal independently of these transforms. Pytest reads the stored
file without SageMath. Direct-evaluation tests separately validate roots,
slot conventions, and normalization.

This is AI-written code for fun, reviewed by the author only very briefly.
It is not security-proven and must never be used in a product. Arithmetic is
not constant-time. The large-modulus path prioritizes exactness over optimized
GPU performance. No encryption or security claims follow from arithmetic tests.
