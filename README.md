# torch-gl

PyTorch polynomial arithmetic for experimenting with the Gentry–Lee FHE scheme.

> **Disclaimer:** This is AI-written code made just for fun. The author has only
> checked the code very briefly. It does not have a proper security-proven
> implementation and has not undergone a cryptographic security audit.
> **Never use it in a product or production system.** Passing arithmetic tests
> does not establish cryptographic security.

## Install

Requires Python 3.10+ and PyTorch 2.2+. From this repository:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Polynomial rings

`MultivariatePoly` stores exact `torch.int64` coefficients in configurable
quotients of `Z_q[i]`, with formal `i² = -1`. The default ring is
`Z_q[i][X,Y,W] / (X^n-i, Y^n-i, Phi_p(W))`. Here `n` is a power of two and
`p` is an odd prime power. The canonical coefficient tensor has shape
`(n, n, phi(p), 2)`, with final channels for real and imaginary coefficients.
At most three variable axes plus the Gaussian channel axis are supported.

```python
import torch
from torch_gl import MultivariatePoly, PolynomialRing

ring = PolynomialRing.multivariate_poly(q=433, n=4, p=27)
coefficients = torch.arange(4 * 4 * 18 * 2).reshape(ring.shape)
polynomial = MultivariatePoly(ring, coefficients)
product = polynomial * polynomial

X = ring.generator("X")
W = ring.generator("W")
assert X**4 == ring.constant((0, 1))
assert W**18 + W**9 + 1 == ring.zero()

# The smaller R_q used for the paper's RLWE secret key:
small_ring = PolynomialRing.multivariate_poly(q=433, n=4, p=27, variables=("X", "W"))
assert small_ring.shape == (4, 18, 2)
```

Supports addition, subtraction, multiplication, nonnegative powers, Gaussian
integer constants, configurable variable order, high-degree input reduction,
and CPU/CUDA device transfer. Input tensors and exported coefficients are copied.
`ring.from_real(tensor)` supplies zero imaginary coefficients. Polynomial values
do not have batch axes; standalone transforms support batching.

Other separable monic ideals can be supplied explicitly:

```python
from torch_gl import PolynomialRing, QuotientAxis

custom_ring = PolynomialRing(
    q=97,
    axes=(
        QuotientAxis.power_of_i("X", n=4),
        QuotientAxis("T", modulus=(1, 0, 0, 1)),  # T^3 + 1
    ),
)
```

## NTTs and multiplication

- `NTTPlan(q, length)` supports mixed-radix lengths `2^a * 3^b`, requiring
  `length | q-1`. `ntt` and `intt` are convenience functions.
- `CyclotomicNTTPlan(q, p)` implements a compact `phi(p)`-slot transform for
  `p = 3^a`, requiring `p | q-1`. A two-branch twisted butterfly and radix-3
  subtransforms evaluate at primitive roots. Tested for `p = 3, 9, 27, 81`.
- `GaussianNTTPlan(q, length=n)` implements compact twisted transforms modulo
  `X^n-i` or `Y^n-i`, requiring `4n | q-1`.

The ring selects native transforms when their roots exist. Other ideals, other
prime-power cyclotomics, and nonsplitting moduli use padded radix-2 convolution
followed by ideal reduction. An axis of degree `degree` on this path requires
`next_power_of_two(2*degree-1) | q-1`. Construction rejects unsupported moduli.
`ring.transform_plans` and `ring.ntt_shape` expose the selected transforms.

For the paper's fully split setting, choose a prime `q ≡ 1 (mod 4np)`.
For example, `q=433, n=4, p=27` uses transform shape `(4, 4, 18)` without padding.
Native NTTs for other odd prime bases are deferred; generic arithmetic works.

Moduli are **odd primes below `2^30`**, representing a single RNS limb.
Ordinary products fit in signed int64 directly. Larger moduli belong in a future
higher-level RNS layer. CPU and CUDA are supported; MPS lacks the required int64
operations. The 60-bit implementation was removed after its optimized CPU NTT
remained over four times slower than a 30-bit limb.

Plans cache device-side twiddles and batch Gaussian/cyclotomic branches.
`plan.prepare(device)` warms forward/inverse caches before a benchmark.
Polynomial multiplication batches both operands and reuses the ring's plans.
`clear_transform_cache()` releases cached device tensor references.

For fully native rings, multiplication keeps Gaussian branches split across all
axes and merges them once. Canonical results are adopted without repeating
normalization, ideal reduction, or coefficient copies. To compile the complete
multiplication kernel, use:

```python
polynomial.multiply(polynomial, compiled=True)  # First call includes compilation.
product = polynomial.multiply(polynomial, compiled=True)  # Reuses the compiled kernel.
```

The compiled path currently requires native transforms on every variable axis.
Ordinary `*` remains eager to avoid an implicit compilation delay.

## Standalone Montgomery functions

The optional Montgomery API uses **`R = 2^31`**, with a single reduction limb.
The default NTT uses normal residues, which benchmark faster on this CPU.
Montgomery values carry their modulus and representation tag; raw integer tensors
cannot reveal whether they are encoded.

```python
from torch_gl import (
    to_montgomery_form,
    to_normal_form,
    is_montgomery_form,
    montgomery_mul,
)

coefficients = torch.tensor([1, 2, -3], dtype=torch.int64)
encoded = to_montgomery_form(coefficients, q=754974721)
assert is_montgomery_form(encoded)
product = montgomery_mul(encoded, encoded)
normal_product = to_normal_form(product)
```

`encoded.values` exports a defensive copy of the encoded integers. Keep the tagged
object for Montgomery arithmetic; ordinary polynomial constructors expect normal
coefficients. Conversion and multiplication preserve tensor shape and device.
Montgomery functions support odd moduli below `2^30`; ring moduli must also be prime.

## Performance measurement

```sh
python scripts/benchmark_ntt.py --output benchmarks/cpu.json
python scripts/benchmark_ntt.py --device cuda --compile
```

The CUDA benchmark uses synchronized GPU events. `--compile` optionally requests
Torch compiler fusion for standard NTTs and modular multiplication; specialized
Gaussian/cyclotomic transforms remain eager. `plan.compile(device, dim=...)` binds
tables before compiler tracing and returns a callable standard NTT.
Compilation is excluded from steady-state timings.
GPU speedups have **not** been measured on this machine. See
[performance notes](docs/performance.md) for the CPU results and limits.

## Paper correspondence and scope

The ring definitions match §2.1 of
[Fully Homomorphic Encryption for Matrix Arithmetic](https://eprint.iacr.org/2025/1935).
The Gaussian transform uses the split map from §5.1. See
[implementation notes](docs/arithmetic.md) for formulas and slot ordering.

`*` is ordinary quotient-ring multiplication, the arithmetic underlying the
paper's Hadamard product. The matrix product `circledast` is a distinct trace
operation for a later step. Ciphertexts, encoding, key switching, RNS chains,
rescaling, and bootstrapping are also future work.

## Tests

```sh
pytest
ruff check .
ruff format --check .
python -m build --no-isolation
```

Multiplication tests read **15 committed SageMath-generated known-answer vectors**
from [tests/data/sage_multiplication.json](tests/data/sage_multiplication.json).
Thirteen multiplication cases cover both requested rings, native/padded paths,
prime powers, and custom Gaussian ideals. The file remains unchanged; two historical
60/62-bit cases now test rejection outside the single-limb range. **Tests never run
SageMath or regenerate vectors.** SageMath is not a package dependency.

The independent [generator](scripts/generate_sage_kats.py) uses multivariate
polynomials over `GF(q)` and ideal reduction, with `i²+1` explicit. Run it only
when intentionally updating frozen vectors:

```sh
sage -python scripts/generate_sage_kats.py
```

Additional tests compare transforms with direct evaluation and check defining
relations, ring laws, input validation, and overflow boundaries. CUDA cases run
when hardware is available.

## Repository layout

```text
src/torch_gl/           Package, ring configuration, arithmetic, transforms
tests/                 Unit tests and frozen SageMath KATs
scripts/               Optional reference-vector generator
examples/              Runnable examples
docs/                  Mathematical and implementation notes
.github/workflows/     Tests, linting, and distribution builds
pyproject.toml         Packaging, dependencies, and tool configuration
```
