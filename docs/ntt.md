# NTT implementation walkthrough

This describes the code currently used by the library. Runnable checks are in
[examples/ntt.py](../examples/ntt.py); key-switching examples are in
[examples/key_switching.py](../examples/key_switching.py).

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| [ntt.py](../src/torch_gl/ntt.py) | Standard mixed-radix plans, GS forward/CT inverse stages, public ordering, optional compilation |
| [_transform_tables.py](../src/torch_gl/_transform_tables.py) | Cached Montgomery twiddles on the requested CPU/CUDA device |
| [modular.py](../src/torch_gl/modular.py), [montgomery.py](../src/torch_gl/montgomery.py) | Exact modular add/subtract and Montgomery multiplication |
| [cyclotomic.py](../src/torch_gl/cyclotomic.py) | Compact evaluation modulo `Phi_(3^exponent)` |
| [twisted.py](../src/torch_gl/twisted.py) | Public Gaussian-coefficient transforms for `X^n-i` |
| [_multiplication.py](../src/torch_gl/_multiplication.py) | Native polynomial multiplication, keeping Gaussian branches split and frequency order permuted |
| [_native_transform.py](../src/torch_gl/_native_transform.py) | Prepared coefficient/slot transforms used by key switching, including leading batch axes |

There is no specialized secret-key multiplier. Engine encryption/decryption
temporarily embed a small X/W key in the full ring and use ordinary polynomial
multiplication. SwitchSmall still uses Y as a batch axis: that is part of the
switching algorithm and is independent of secret-key multiplication.

## What NTTPlan computes

`NTTPlan(q, length=L)` requires an odd prime `q<2^30`, a length whose prime
factors are only 2 and 3, and `L | q-1`. Its root has exact order L. The public
forward transform returns

```text
slots[k] = sum(coefficients[j] * root**(j*k), j=0,...,L-1) mod q.
```

The inverse uses `root^-1` and multiplies the output by `L^-1 mod q`.
An ordinary `NTTPlan` computes cyclic convolution modulo `T^L-1`; twists and
the compact cyclotomic layer supply the other quotient relations used here.

```python
import torch
from torch_gl import NTTPlan

plan = NTTPlan(q=1297, length=12)
coefficients = torch.arange(12, dtype=torch.int32)
slots = plan.forward(coefficients)
assert torch.equal(plan.inverse(slots), coefficients)
```

### Public boundary and representation

`NTTPlan.transform()` performs these steps:

1. Validate the requested tensor axis and its length.
2. Reduce input integers modulo q, then store residues as int32.
3. Encode the residues in Montgomery form.
4. Run the prepared kernel.
5. Decode its result to normal-form int32 residues.

Inside the kernel, arithmetic stays in Montgomery form with `R=2^31`.
`ResidueArithmetic.multiply()` delegates to Montgomery reduction, using
temporary int64 products and returning canonical int32 values. Twiddles and
butterfly constants are already Montgomery-encoded. Normalization is therefore
an ordinary call to the same multiplication function, with the encoded inverse L.
The public cyclic inverse performs this multiplication explicitly. Twisted
inverse transforms fold it into their cached inverse-twist tables instead.

An internal `_kernel(..., permuted=True)` takes and returns raw Montgomery
tensors. Passing ordinary coefficients directly to it is incorrect. Public
`forward()` and `inverse()` handle the conversions automatically.

## Stage construction

`stage_tables()` and `_butterfly_constants()` repeatedly choose radix 2 while
the current length is even, then radix 3. At a stage of length S:

```text
radix = 2 if S is even else 3
branch_length = S / radix
omega = stage_root**branch_length mod q
twiddle[branch,j] = stage_root**(branch*j) mod q, branch=1,...,radix-1
next stage_root = stage_root**radix mod q
```

The radix-2 omega is -1. Radix-3 omega has order 3. The metadata tuple stores
`(radix, encoded_omega, encoded_omega_squared)`; the optimized butterfly uses
only encoded_omega, so the last precomputed scalar is currently unused.

For L=12, the stage lengths are 12, 6, 3 and the radices are 2, 2, 3.
For L=128, there are seven radix-2 stages. For L=81, there are four radix-3 stages.
No roots or power sequences are rebuilt during a warm transform.

## Tensor layout in _paired_dft

The requested axis is moved to the last position by `values.movedim(dim,-1)`.
Every other dimension becomes a batch dimension. At each stage, the code uses:

```python
branches = values.reshape(*values.shape[:-1], length // stage_length, radix, branch_length).unbind(
    -2
)
```

The new dimensions mean `(independent groups, branches, positions in a branch)`.
For a tensor of shape `(batch,12)`, the stages operate on:

| Stage | Grouped shape |
| --- | --- |
| Length 12, radix 2 | `(batch,1,2,6)` |
| Length 6, radix 2 | `(batch,2,2,3)` |
| Length 3, radix 3 | `(batch,4,3,1)` |

`unbind` returns branch views. Tensor arithmetic processes every group and batch
simultaneously; there is no Python loop over coefficient positions. Python
controls the O(log L) stages and at most two nonzero twiddle branches per stage.

### Gentleman–Sande forward

Forward visits the stages from largest to smallest:

```text
reshape into groups and branches
compute the radix butterfly
multiply output branch r by twiddle[r]
stack the branches once and flatten the grouped dimensions
```

In a radix-2 stage, the implementation operates on whole branch tensors:

```python
first_branch, second_branch = branches
output_0 = arithmetic.add(first_branch, second_branch)
output_1 = arithmetic.subtract(first_branch, second_branch)
# GS applies the cached twiddle vector after the butterfly.
# Skip this multiplication when branch_length == 1.
if branch_length > 1:
    output_1 = arithmetic.multiply(output_1, twiddle_vector)
```

`arithmetic.add()` performs tensor `first_branch + second_branch` followed by
modular reduction; subtraction works the same way. The cached twiddle vector
contains `stage_root**j` at each position and broadcasts across every group and
batch. There is no `first_branch[j]` access or Python loop over `j` during the
transform. Selecting one of the two or three branch views is not indexing
individual coefficients.

All operations reduce modulo q. Output is in mixed-radix digit order, which is
bit-reversed frequency order for powers of two.

### Cooley–Tukey inverse

Inverse visits the same grouped shapes in reverse order, with inverse roots:

```text
reshape into groups and branches
multiply input branch r by inverse_twiddle[r]
compute the inverse-root radix butterfly
stack the branches once and flatten the grouped dimensions
normalize by L^-1 after all stages
```

This consumes the forward's digit order directly and returns natural coefficient
order. The branches are not reordered between a forward transform, pointwise
operation, and inverse transform.

### Radix-3 butterfly

`_butterfly()` uses `omega²=-1-omega`, valid for a primitive cube root. Its
radix-3 operations also process whole branch tensors:

```python
first_branch, second_branch, third_branch = branches
weighted_difference = arithmetic.multiply(arithmetic.subtract(second_branch, third_branch), omega)
output_0 = arithmetic.add(arithmetic.add(first_branch, second_branch), third_branch)
output_1 = arithmetic.add(arithmetic.subtract(first_branch, third_branch), weighted_difference)
output_2 = arithmetic.subtract(
    arithmetic.subtract(first_branch, second_branch), weighted_difference
)
```

`omega` is a Montgomery-encoded scalar broadcast over the tensors. The two
cached twiddle vectors likewise broadcast over groups and batches, without
Python coefficient indexing.

This needs one constant multiplication instead of evaluating four separate
omega/omega² products. GS applies the two branch twiddles afterwards; CT applies
them beforehand. At the innermost stage `branch_length=1`, all twiddles are unity
and their multiplications are skipped. The code allocates one `torch.stack`
output per stage, without the former additional full-tensor concatenation.

## Frequency order and public API

`_digit_order(L)` gives the natural frequency index at each internal storage
position. For example:

```text
L=8:  [0,4,2,6,1,5,3,7]
L=12: [0,4,8,2,6,10,1,5,9,3,7,11]
```

An internal forward returns this order, and an internal inverse accepts it.
Public `NTTPlan.forward()` gathers back to natural frequency order with a cached
inverse permutation; public `inverse()` gathers its natural-order input into
digit order before CT. Permutation indices are int64 metadata; residue storage
remains int32.

Thus public transform calls can reorder at their boundaries, while native
polynomial products and key switching request `permuted=True` and avoid those
passes. The private inspection example in `examples/ntt.py` demonstrates both
orders and verifies their exact correspondence.

## X/Y transforms modulo X^n-i

For a primitive `4n`-th root rho, define `I=rho^n`, so `I²=-1` in Z_q.
Formal Gaussian coefficients `(real,imag)` are split into:

```text
positive = real + I*imag
negative = real - I*imag
```

The positive coefficient at exponent j is twisted by `rho^j`, the negative by
`rho^-j`. Native multiplication uses a shared n-point NTT root `rho^4` for both
branches. Its evaluation points are `rho*(rho^4)^k` and
`rho^-1*(rho^4)^k`, respectively. Their n-th powers are I and -I, so these
transforms enforce the binomial ideal rather than `X^n-1`.

Both branches retain their internal ordering through multiplication. The inverse
uses an unnormalized CT kernel (`normalize=False`) followed by cached twists
`n^-1 * rho^-j` and `n^-1 * rho^j`. Combining normalization with untwisting
removes one full-tensor multiplication per X/Y inverse while preserving the
same result. Gaussian coefficients are recovered with:

```text
real = (positive + negative) / 2
imag = (positive - negative) / (2*I)
```

Division here means modular multiplication by a precomputed inverse.
The public `GaussianNTTPlan` additionally orders its negative evaluations as
`rho^-(1+4k)` by reversing negative frequencies and merges the Gaussian channels
around each public transform. Native whole-polynomial multiplication splits once
across all axes and does not need that negative-frequency permutation.

## W transform modulo Phi_(3^exponent)

Let `p=3^exponent`, `stride=p/3`, and xi have exact order p. A canonical
polynomial has `2*stride` coefficients:

```text
polynomial(W) = lower(W) + W**stride * upper(W)
Phi_p(W) = 1 + W**stride + W**(2*stride)
```

Set `omega=xi**stride`. `CyclotomicNTTPlan._kernel()` forms two branches,
one for each primitive-root class c=1,2:

```text
branch_c[j] = xi**(c*j) * (lower[j] + omega**c * upper[j])
```

One batched, length-stride radix-3 NTT with root `xi**3` transforms both branches.
Their slots are evaluations at `xi**(3*k+1)` and `xi**(3*k+2)`. No nonprimitive
roots are included and no length-p polynomial padding is needed. This compact
two-branch stage is a twisted linear map, not an ordinary length-two DFT.

The inverse applies CT to both branches, removes `xi**(c*j)`, and solves:

```text
first = lower + omega*upper
second = lower + omega**2*upper
upper = (first - second) / (omega - omega**2)
lower = first - omega*upper
```

Its cached inverse twists include `stride^-1`, so the two inverse branch NTTs
also skip a separate normalization pass. Scaling happens only when these
tables are constructed, not during a warm transform. Length-one cyclic
transforms skip multiplication by their unity normalization factor.

Public slots place root class 1 before class 2, each in natural k order.
Internal slots retain radix-3 digit order separately within each class. For
p=729, the transform has 486 slots, with two length-243 subtransforms.

## Native polynomial multiplication and key switching

`native_multiplier()` binds the kernels and twists outside tracing. Its callable:

1. Stacks the two operand tensors along a new leading batch axis and encodes once.
2. Splits the Gaussian channel once.
3. Applies each variable's forward kernel in digit order; X/Y use twists.
4. Removes the operand batch axis and multiplies the two split slot tensors.
5. Applies inverse kernels in reverse variable order and undoes twists.
6. Merges Gaussian coefficients and decodes once.

`NativeTransform` uses the same conventions for switching keys and decomposed
ciphertext components. SwitchBig transforms X, Y, W. SwitchSmall moves Y to a
leading batch axis and transforms only X/W. Its same small switching-key slots
are broadcast over those Y coefficient polynomials. Decomposition and ModDown
remain coefficient-wise operations; see [key-switching notes](key_switching.md).

Matrix multiplication's optimized contraction uses a W transform alone and
keeps X/Y as coefficient indices. Its public full matrix-slot transforms retain
the canonical encoder order; see [matrix notes](matrix.md).

## Cost and allocation behavior

For B independent transforms of length L, stage arithmetic is O(B*L*log L).
Each stage processes all values with tensor operations, and its output allocation
has O(B*L) elements. A moved or strided tensor may require a reshape copy.
Eager Montgomery arithmetic still allocates local int64 temporaries. Reduction
reuses its fresh product and cancellation buffers in place; it never modifies
input operands, broadcast views, or cached twiddles. CPU final correction uses
in-place remainder on the fresh int32 result; CUDA keeps conditional correction.
This is not an in-place or fully fused GPU butterfly kernel. See
[performance notes](performance.md#simple-allocation-and-normalization-savings)
for the measured CPU effect.

Twiddle tensors and prepared kernels have bounded device caches. `prepare(device)`
warms the public tables, while `clear_transform_cache()` releases tensor/kernel
cache references. Optional `NTTPlan.compile()` or native
`polynomial.multiply(..., compiled=True)` binds tables before full-graph tracing.
Compilation is lazy: warm the compiled call before timing it. The default
backend is Inductor; `backend="eager"` demonstrates graph capture without native
code generation. CUDA execution remains unverified on the development Mac.

Run the executable examples:

```sh
python examples/ntt.py
python examples/ntt.py --compile --backend eager
python examples/ntt.py --compile --backend inductor
python examples/ntt.py --device cuda:0
```
