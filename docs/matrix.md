# Plaintext matrix multiplication through NTT slots

`matrix_multiply(left, right)` takes two encoded `RNSPoly` values. For each
prime q, it performs an NTT, multiplies the resulting matrices modulo q,
and performs the inverse NTT to recover a polynomial. W frequencies are
independent batches. Both Gaussian split branches are processed independently.
The multiplication kernel transforms W only and contracts X/Y coefficient
matrices. Public `MatrixNTTPlan.forward/inverse` still transform every variable
to expose canonical matrix slots.

```python
import torch
from torch_gl import MatrixEncoder, RNSRing, matrix_multiply

ring = RNSRing.for_parameters(n=4, p=9, limb_count=4)
encoder = MatrixEncoder(ring)  # scale=2**40
decoder = MatrixEncoder(ring, scale=encoder.scale**2)
left = torch.randn(encoder.slot_shape, dtype=torch.complex128)
right = torch.randn(encoder.slot_shape, dtype=torch.complex128)
product = matrix_multiply(encoder.encode(left), encoder.encode(right))
recovered = decoder.decode(product)
print((recovered - left @ right).abs().max())
```

The result represents ordinary batch-wise matrix multiplication, not the
Hadamard product performed by polynomial `*`. Inputs must share an RNS ring
and device, have equal X/Y degrees, and support native X/Y and power-of-three
W transforms. Slot layout matches `MatrixEncoder`: W first, with the remaining
axes in ring order. Reordering X/Y therefore also changes the row/column interpretation.

## Finite-field transforms

`MatrixNTTPlan(limb_ring, device=...)` exposes `forward(polynomial)` and
`inverse(slots)` for a single prime. Public slot tensors have shape
`(phi(p),n,n,2)` and contain normal-form int32 residues. The last dimension
stores the two Gaussian split evaluations, not real/imaginary components.

The positive branch evaluates at `(I, rho**(5**j), rho**(5**k), eta**(2**ell))`,
where `rho**n=I`, `I**2=-1`, and eta is primitive of order p. The negative branch
evaluates at `(-I, rho**(-5**j), rho**(-5**k), eta**(-2**ell))`.
Twisted X/Y NTTs enforce `X**n=i` and `Y**n=i`; the compact W NTT enforces
`Phi_p(W)`. Frequency permutations align the inner matrix indices and the
encoder's root ordering. The implementation splits once before transforms,
stays in Montgomery form through contraction and inverse transforms, and
merges once into Gaussian coefficients.

Matrix contraction sums only the shared row/column index. It splits canonical
30-bit Montgomery residues into low/high 15-bit parts, packs these into a single
batched float64 GEMM, and reconstructs the result modulo q with int64 arithmetic.
This avoids the Python inner-index loop and its per-term Montgomery temporaries.

This use of floating GEMM is exact under a checked bound, rather than an
approximate multiplication of full 30-bit values. Every partial product is an
integer at most `(2**15-1)**2`; the implementation requires
`2*n*(2**15-1)**2 <= 2**53`, including the sum of the two cross terms. Thus all
products and partial sums are representable exactly in float64. FP64 arithmetic
also avoids FP32/TF32 precision settings. Each integer partial sum is reduced
modulo q before shifting by 15 or 30 bits, so recombination remains below
`2**61` and cannot overflow int64. One Montgomery reduction then changes the
product's `R**2` factor back to `R`. Persisted polynomials remain int32.

Encrypted multiplication reuses four forward W transforms for its four product
components. Only the RNS-limb loop remains in Python, keeping workspace bounded
instead of stacking all large limbs together. CUDA execution is supported, but
GPU correctness and performance have not been measured on the development Mac.
Plans cache device tables and permutations; `clear_transform_cache()` also
clears the matrix-plan cache.

## Trace and normalization

For the default X/Y order, expand inputs as coefficients `L[r,s](W)` and
`R[t,u](W)`. The elementary root sum gives the output coefficients

```text
C[r,u](W) = n * (L[r,0](W)*R[0,u](W)
                + i * sum(L[r,s](W)*R[n-s,u](W), s=1,...,n-1)).
```

Indeed, summing `(rho**(5**k))**(s+t)` over k gives n when s+t=0,
n*i when s+t=n, and zero otherwise. This is the coefficient contraction
behind the matrix/trace identity. The finite-field slot multiplication computes
the full sum, so decoding at `left_scale*right_scale` yields `left @ right`.
The coefficient implementation prepares the right matrix with row zero unchanged
and row s replaced by `i*R[n-s,u](W)` for `s>0`, contracts, and multiplies by n.
It therefore implements this same identity without X/Y NTTs or slot permutations.

The paper's literal `circledast` uses adjoint substitutions and a normalized
trace, giving `A @ B* / n`. This API computes the ordinary `A @ B` requested
here. `engine.matrix_multiply()` extends this kernel to encrypted `A @ B.T`
using a smaller X/W secret and the four-component key basis described in the
[encryption notes](encryption.md#encrypted-matrix-multiplication-with-a-transposed-right-input).
The literal normalized-adjoint operation and rescaling remain separate features.
The four-component product can now be relinearized through two SwitchBig calls;
see [key switching](key_switching.md).

## Tests

`pytest tests/test_matrix_multiplication.py` verifies finite-field evaluation
against direct Python arithmetic, NTT/INTT inversion, an independent coefficient
contraction with cyclotomic reduction, and a modular dot product whose unreduced
sum exceeds int64. Encoding-only end-to-end tests compare with complex batched
matrix multiplication, including reordered axes and p=729. They verify a
coefficient-growth bound below Q/2 before interpreting the centered CRT result.
