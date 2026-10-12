# Performance and the single-limb choice

The arithmetic layer deliberately handles prime moduli below `2^30`.
Larger ciphertext moduli should use a higher-level RNS representation.
This keeps ordinary scalar products below `2^60`, so signed int64 multiplication
is exact without multiword arithmetic.

## Why the 60-bit path was removed

Before restricting the range, an optimized two-limb Montgomery implementation
was compared with a 30-bit modulus on the same CPU using batched length-16 NTTs.
The experiment used four CPU threads, three warmups, and ten measured runs.

| Independent transforms | 30-bit prime | 61-bit prime | Large-limb slowdown |
| --- | ---: | ---: | ---: |
| 64 | 0.0955 ms | 0.4761 ms | 5.0× |
| 4096 | 1.2038 ms | 5.6308 ms | 4.7× |

The 61-bit prime was `1152921504606847201`; the 30-bit prime was `754974721`.
The two-limb path already improved substantially on binary add-and-double,
but still exceeded the requested 2× slowdown threshold. It was removed rather
than retained as unused complexity. This decision is based on CPU measurements;
it does not establish a corresponding GPU ratio.

## Current implementation

- Device-side twiddle tables have bounded caches keyed by modulus, root,
  transform length, and device. `prepare(device)` warms both directions.
- Gaussian split branches use one batched transform and a frequency permutation.
- Cyclotomic primitive-root classes use one batched radix-3 transform.
- Polynomial products batch both operands and reuse the ring's plan objects.
- Native products keep Gaussian branches split throughout all axes and pointwise
  multiplication, then merge once. This removes repeated conversions and reduces
  pointwise multiplication from four Gaussian products to two split-field products.
- Canonical input construction skips axis permutations. Fresh canonical arithmetic
  results are adopted directly, without normalization or quotient reduction copies.
- Warm transforms create no host/device twiddle tables. Tests enforce this.
- Optional `NTTPlan.compile(device, dim=...)` binds tables before compiler tracing
  and requests a full compiled graph. Standard NTTs and modular multiplication
  can be measured with the benchmark's `--compile` option. Specialized transforms
  are still eager; each JSON measurement records whether it was compiled.

`MultivariatePoly.multiply(other, compiled=True)` also compiles the complete
native-ring multiplication pipeline. Ordinary `*` remains eager. The first
compiled call may take seconds; timing that call measures compilation as well
as multiplication. Keep the same shapes and device, warm once, then measure.

For `n=128`, `p=243`, and `q=541582849` on the development CPU, the original eager
pipeline took 0.966 s. Profiling found 230 remainder calls, 46 stacks, and 50
concatenations. The full compiled pipeline measured about 0.308 s per product
after a 10.9 s first call. These are CPU measurements; GPU speed remains unverified.

Torch's compiler may fuse eligible operations, reducing intermediate memory
traffic and kernel launches. This is the motivation for the optional path,
not a claim of measured GPU speedup. See the official
[Torch compiler performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide).

## Reproducible measurements

Saved eager and compiled CPU results are in
[`benchmarks/cpu.json`](../benchmarks/cpu.json) and
[`benchmarks/cpu_compiled.json`](../benchmarks/cpu_compiled.json).
They record the Torch version, operating system, device, thread count, modulus,
shapes, warmup count, measured repetition count, and median timings.

```sh
python scripts/benchmark_ntt.py --output benchmarks/cpu.json
python scripts/benchmark_ntt.py --compile --output benchmarks/cpu_compiled.json
python scripts/benchmark_ntt.py --device cuda --compile --output benchmarks/cuda.json
```

CPU measurements use a monotonic wall clock. CUDA measurements use events on
the selected device and explicit synchronization. Table preparation and compilation
are outside steady-state timings. The script verifies transform round trips and
compiled outputs before recording performance. SageMath is never invoked.

Small batches can be dominated by Python and compiler dispatch overhead; a
compiled path is not necessarily faster for every shape. Eager Montgomery
multiplication is slower than normal multiplication in these CPU measurements,
so normal form remains the default. Fused Montgomery multiplication can be
competitive, but conversion costs and full transform performance must also be
measured before changing that default.

## GPU limits

There is no CUDA hardware on the development machine. CUDA KATs, conversion
checks, and cache tests are enabled when a CUDA device is present, but have not
been executed here. Native CUDA compiler behavior and throughput remain unverified.
Do not infer GPU efficiency from CPU timings or the use of Torch tensors alone.

The portable Montgomery API uses a single signed-safe limb with `R=2^31`.
Using `R=2^64` would require unsigned high-word multiplication and carries.
Eager [`torch.uint64`](https://docs.pytorch.org/docs/main/tensor_attributes.html)
has limited operation support, while CUDA provides
[`__umul64hi`](https://docs.nvidia.com/cuda/archive/13.3.1/libdevice-users-guide/__nv_umul64hi.html).
A specialized unsigned GPU backend is feasible, but is unnecessary for the
current small-limb component and would require separate validation and benchmarks.
