# Plaintext matrix multiplication through NTT slots

`matrix_multiply(left, right)` takes two encoded `RNSPoly` values. For each
prime q, it performs an NTT, multiplies the resulting matrices modulo q,
and performs the inverse NTT to recover a polynomial. W frequencies are
independent batches. Both Gaussian split branches are processed independently.

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

Matrix contraction sums only the shared row/column index. Each product uses
local int64 Montgomery temporaries and returns int32; every addition is reduced
modulo q. An unreduced integer `torch.matmul` can overflow int64 for 30-bit
residues, and floating matmul would lose exactness. This initial implementation
uses a Python loop over the inner index with vectorized tensor operations over
all W batches, rows, columns, and branches. GPU performance has not been measured.
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

The paper's literal `circledast` uses adjoint substitutions and a normalized
trace, giving `A @ B* / n`. This API computes the ordinary `A @ B` requested
here. It does not implement encrypted trace multiplication, adjoint APIs,
key switching, or rescaling.

## Tests

`pytest tests/test_matrix_multiplication.py` verifies finite-field evaluation
against direct Python arithmetic, NTT/INTT inversion, an independent coefficient
contraction with cyclotomic reduction, and a modular dot product whose unreduced
sum exceeds int64. Encoding-only end-to-end tests compare with complex batched
matrix multiplication, including reordered axes and p=729. They verify a
coefficient-growth bound below Q/2 before interpreting the centered CRT result.
