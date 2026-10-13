# Runnable NTT and key-switching examples

Run these from the repository root after activating the environment:

```sh
source .venv/bin/activate
python examples/ntt.py
python examples/key_switching.py --case all --k 2
```

Every example contains correctness assertions. These scripts use existing
arithmetic and never run SageMath or regenerate KAT fixtures. The repository's
[prototype disclaimer](../README.md) applies to the example parameters.

## NTT examples

[examples/ntt.py](../examples/ntt.py) covers the following operations:

| Example | What it demonstrates |
| --- | --- |
| Standard NTTs | Round trips for powers of two, powers of three, and mixed lengths, using q=1297 |
| Batched and noncontiguous tensors | Transform an interior axis with `dim=1`; other axes are independent batches; input is preserved |
| Cyclic convolution | Forward transforms, public Montgomery pointwise multiplication, inverse; comparison against a Python-integer oracle |
| Internal GS/CT pair | Raw Montgomery kernels retain bit-reversed order for length 8, avoiding the public reordering passes |
| Compact W transforms | Primitive-root evaluation and defining quotient identities for p=3,9,27,81 |
| Gaussian transforms | `X^n=i` twists and round trips for n=1,2,4,8,16 |
| Larger prime-power W axis | Select a compatible 30-bit q for n=128,p=729 and round-trip a batch of 486-slot W transforms |
| Native polynomial products | Full X/Y/W, smaller X/W, and reordered W/Y/X contexts |
| Optional compiler | Standard NTT and complete native polynomial multiplication using the requested backend |

The [NTT implementation walkthrough](ntt.md) explains the exact functions,
tensor shapes, stage order, butterflies, Montgomery representation, and quotient
twists behind these examples.

### Public mixed-radix transform

```python
import torch
from torch_gl import NTTPlan

plan = NTTPlan(q=1297, length=18).prepare("cpu")
coefficients = torch.arange(3 * 18, dtype=torch.int32).reshape(3, 18)
slots = plan.forward(coefficients)  # Natural frequency order.
assert torch.equal(plan.inverse(slots), coefficients)
```

q=1297 works because `q-1=2**4*3**4`. Length 18 uses one radix-2 stage and two
radix-3 stages. All three rows are transformed together.

### Noncontiguous input and another transform axis

```python
import torch
from torch_gl import NTTPlan

coefficients = torch.arange(72, dtype=torch.int32).reshape(2, 12, 3).transpose(0, 2)
assert not coefficients.is_contiguous()
plan = NTTPlan(q=1297, length=12)
slots = plan.forward(coefficients, dim=1)
assert torch.equal(plan.inverse(slots, dim=1), coefficients)
```

### Power-of-three cyclotomic axis at p=729

```python
import torch
from torch_gl import CyclotomicNTTPlan, select_ntt_primes

n, p = 128, 3**6
q = select_ntt_primes(n, p, limb_count=1)[0]
plan = CyclotomicNTTPlan(q, p)
assert plan.length == 486
coefficients = torch.arange(2 * 486 * 2, dtype=torch.int32).reshape(2, 486, 2)
slots = plan.forward(coefficients, dim=1)
assert torch.equal(plan.inverse(slots, dim=1), coefficients)
```

The leading size-2 dimension is a batch; the final size-2 dimension contains
Gaussian coefficient channels. W transforms the middle axis independently of
those dimensions. This example allocates only a W batch, rather than a full
`128*128*486*2` coefficient tensor.

### X^n=i transform

```python
import torch
from torch_gl import GaussianNTTPlan

plan = GaussianNTTPlan(q=193, length=16)
coefficients = torch.arange(64, dtype=torch.int32).reshape(2, 16, 2)
slots = plan.forward(coefficients, dim=1)
assert torch.equal(plan.inverse(slots, dim=1), coefficients)
```

The final channels remain formal real/imaginary pairs at the public Gaussian
boundary. The transform splits into `+I/-I` branches internally. Its roots
satisfy `X^16=i`, rather than `X^16=1`.

### Compilation and CUDA commands

```sh
# Check complete graph capture with the eager backend.
python examples/ntt.py --compile --backend eager

# Compile native kernels; the first call includes compilation.
python examples/ntt.py --compile --backend inductor

# Use a CUDA-enabled PyTorch installation on an NVIDIA machine.
python examples/ntt.py --device cuda:0
python examples/ntt.py --device cuda:0 --compile
```

CPU eager execution and eager-backend graph capture have been checked locally.
The commands include a hardware check; CUDA examples are not verified on the
development Mac. Use [the benchmark scripts](performance.md) for timed runs
with warmups instead of timing an entire demonstration script.

## Key-switching examples

[examples/key_switching.py](../examples/key_switching.py) uses n=4,p=9 and four
Q limbs. Key generation, encryption, data generation, and evaluation-key
generation use separate generators. Secret keys stay outside Engine.

```sh
python examples/key_switching.py --case all --k 2
python examples/key_switching.py --case all --k 1
python examples/key_switching.py --case all --k 3
python examples/key_switching.py --case all --k 4
```

### Select one operation

| `--case` | Inputs and operation | What is checked |
| --- | --- | --- |
| `parameters` | Inspect k=1,2,3,4 and the digit groups | `dnum=ceil(4/k)`; k=3 has a final one-prime digit |
| `independent` | Encrypt `10000*X` under s1, then `switch_ciphertext` with a small s1->s2 key | Decryption under s2 differs from the original decryption only by small switching error |
| `small` | Apply raw `switch_small` to the ciphertext a component | Its decryption is approximately `a*s1`; b is added by the higher-level method |
| `big` | Generate a big key from the full-ring polynomial `Y+W` to a small target secret | `switch_big(a,key)` decrypts approximately to `a*(Y+W)` |
| `hadamard` | Multiply two encoded ciphertexts and relinearize using s²->s | Small coefficient switching error and decoded slotwise multiplication at scale `2**80` |
| `matrix` | Produce a four-component ciphertext and use its two big keys | Small switching error and decoded W-batched `left@right.T` at scale `2**80` |
| `file-backed` | Generate resident and file-backed matrix keys with matching seeds | The resulting relinearized ciphertexts are exactly equal; decoded matrices match |
| `small-ring` | Use an X/W-only context with no Y variable | Independent-key switching also works in the smaller ring |

For example:

```sh
python examples/key_switching.py --case independent --k 2
python examples/key_switching.py --case hadamard --k 3
python examples/key_switching.py --case matrix --k 2
python examples/key_switching.py --case file-backed --k 2
python examples/key_switching.py --case small-ring --k 1
python examples/key_switching.py --case matrix --device cuda:0 --k 2
```

### Explicit auxiliary primes

The main script constructs `KeySwitchParameters` and passes its compatible
`key_moduli` to `KeySwitcher`. An alternative fully explicit basis is:

```python
import torch
from torch_gl import KeySwitcher, RNSRing, select_ntt_primes

prime_pool = select_ntt_primes(n=4, p=9, limb_count=6)
ring = RNSRing.multivariate_poly(prime_pool[:4], n=4, p=9)
switcher = KeySwitcher(
    ring,
    k=2,
    key_moduli=prime_pool[4:],
    generator=torch.Generator().manual_seed(201),
)
assert switcher.dnum == 2
assert len(switcher.parameters.key_moduli) == 2
```

The last two primes are disjoint from Q and larger than its primes, so their
product meets the digit-size condition. The library validates compatibility and
the requirement `P >= max(Q_j)`. It does not choose security parameters.

### Reading the example checks

Switching is approximate. The examples print the maximum coefficient change
between the decryption before and after switching; they do not demand exact
equality. `centered_coefficients()` is applied only to this small difference,
and verifies agreement across every Q limb. It is never used to infer the sign
of an encoded message near scale `2**40` or `2**80`.

Encoded values are decoded with full-Q CRT reconstruction through MatrixEncoder.
Hadamard and matrix products require the product scale: key switching preserves
that scale and does not rescale. The finite tolerances and coefficient-error
threshold in the script apply only to these fixed toy examples.

File-backed keys need their files while they are used. The demonstration uses
`TemporaryDirectory`, finishes switching and all checks inside it, then removes
the temporary keys. In a longer session, pass a directory whose lifetime covers
all switches; existing evaluation-key files are not overwritten.

The [key-switching implementation notes](key_switching.md) give the shared setup,
equations, and function-by-function decomposition/ModUp/NTT/ModDown flow.

## Larger runtime case

Use the existing benchmark, rather than changing the fixed toy examples:

```sh
python scripts/benchmark_key_switching.py \
  --n 64 --p 243 --modulus-bits 300 --k 4 \
  --device cpu --threads 4 --warmups 1 --runs 3 \
  --output benchmarks/key_switching_n64_p243_300bit.json
```

For n=128,p=243 and a 600-bit Q, change `--n 128 --modulus-bits 600`. CUDA uses
`--device cuda:0`. The benchmark records operation timings and checks switching
error independently; SageMath is not invoked.
