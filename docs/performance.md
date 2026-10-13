# Performance and the single-limb choice

The arithmetic layer deliberately handles prime moduli below `2^30`.
Larger ciphertext moduli should use a higher-level RNS representation.
Stored coefficients and tables are int32, halving their resident tensor memory
relative to the original int64 layout. Scalar products remain below `2^60`.
Montgomery multiplication uses local int64 intermediates and narrows its reduced
result back to int32. Transforms encode once, retain Montgomery form internally,
and decode at their public boundaries. Compiled kernels can keep wide temporaries
in registers; eager Torch materializes temporary tensors.

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
- Internal transforms pair GS DIF forward with CT DIT inverse and retain
  mixed-radix digit order through polynomial multiplication, matrix contraction,
  and key switching. There is no intermediate bit/digit-reversal pass. Public
  NTT methods still return natural order; public matrix slots use encoder order.
- Radix-3 butterflies use one constant multiplication plus two branch twiddles.
  Unity twiddles at the innermost stage are skipped. Each stage builds its output
  once; it no longer stacks a full butterfly tensor and then concatenates another
  full copy after multiplying twiddles. Eager stages still allocate temporaries;
  the paired ordering alone does not provide a fused or in-place GPU kernel.
- Matrix multiplication transforms only W and contracts X/Y coefficient matrices.
  Exact 15-bit partial products use one batched float64 GEMM, followed by integer
  modular recombination. See [matrix notes](matrix.md) for the exactness bound.
  Encrypted products reuse input transforms across all four components.
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

The current matrix and ring-product comparison can be run independently of
large evaluation-key generation:

```sh
python scripts/benchmark_matrix.py --n 64 --p 243 --q 1064627713 \
  --device cpu --threads 4 --warmups 1 --runs 5 \
  --output benchmarks/matrix_n64_p243.json
```

The saved CPU result measures 267.2 ms for the former contraction loop versus
14.44 ms for the exact split GEMM, about 18.5x for contraction alone. The
comparison verifies exact coefficient equality on shared inputs. Older saved
small-secret timings describe a specialized path that has since been removed.
The full n=64, p=243, 300-bit Q, k=4 benchmark averages 3.509 s for encrypted
matrix multiplication and 11.959 s for two SwitchBig calls; see
[`benchmarks/key_switching_n64_p243_300bit_fixed.json`](../benchmarks/key_switching_n64_p243_300bit_fixed.json).
GPU and separate C++ comparisons remain unmeasured. The algebra review is in
[the audit notes](algebra_audit.md).

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

These saved timing reports describe the earlier int64 storage implementation.
They remain historical measurements rather than benchmarks of the current int32
Montgomery pipeline. Small batches can be dominated by Python and compiler
dispatch overhead; a compiled path is not necessarily faster for every shape.
Fresh measurements should include encoding/decoding and full transform costs.

The current int32 pipeline was checked on a dense polynomial times `X*Y*W`,
using independent coefficient shifts and quotient reduction as the oracle.
At `n=128`, `p=243`, `q=541582849`, three warmups followed by ten compiled CPU
products averaged 0.187 s. Each polynomial occupied 21,233,664 bytes, compared
with 42,467,328 bytes for int64 storage. The earlier compiled int64 measurement
averaged 0.274 s. These are individual CPU measurements, not GPU performance
claims or a guarantee for other workloads. Metadata is saved in
[`benchmarks/native_multiply_int32.json`](../benchmarks/native_multiply_int32.json).

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
