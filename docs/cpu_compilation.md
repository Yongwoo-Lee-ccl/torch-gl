# Optional CPU arithmetic compilation

The arithmetic is authored entirely in Python/PyTorch. There are no handwritten
C++, CUDA, Triton kernels or Torch extensions. `torch.compile` uses Inductor's
normal generated code and a working compiler toolchain.

Install the development extra (`pip install -e ".[dev]"`) to run the benchmark;
NumPy is used only for raw-byte hashing outside timed arithmetic.

Eager execution remains the default. To enable the small shared CPU graphs:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TORCHINDUCTOR_COMPILE_THREADS=1 TORCH_GL_COMPILE=1 \
python your_program.py
```

Before creating workloads, set `torch.set_num_threads(1)` and
`torch.set_num_interop_threads(1)`. Before timing, execute:

```python
from torch_gl import cpu_optimization_info, prepare_cpu_kernels

prepare_cpu_kernels(P + Q)
print(cpu_optimization_info())
# Then warm up each actual operation, including transform/layout caches.
```

Preparation executes real kernels; it does not merely create compiled callables.
On the validated PyTorch 2.14.1 CPU build, the intended signatures are three
linear arities (1/2/3), two Gaussian directions, and scalar/vector pointwise
products. Compiler counters can vary by Torch version. The requirement is that
**no new graph appears within a timed operation**. The benchmark records the
counters before and after every measured group.

Only CPU int32 tensors with at least 4096 elements and one Torch thread qualify.
CUDA, smaller arrays, other dtypes, multiple threads and flag-off calls keep the
eager implementation. Linear forms of arity above three use eager execution.
During outer compiler tracing, inner compilation dispatch is skipped. The tracing
check uses `torch._dynamo.is_compiling`, available at the package's older minimum
Torch version; all execution results here were validated on 2.14.1.

Each compiled kernel contains pure functional tensor operations. Python checks,
bounded LRU caches and reshaping run outside tracing. Moduli, cancellation
inverses and weights are zero-dimensional Tensor data arguments, never `.item()`
values or Python specialization arguments. Cache slots distinguish scalar roles
to avoid accidental duplicate-input guards. Cached data consists only of public
constants, never input coefficients, secrets, outputs or evaluation keys.

The compiled linear form accumulates public-weighted signed CRT summands before
one bounded REDC. Groups satisfy sum(B_i*encoded_weight_i) < q*2^31; larger sums
split into safe groups. Integer ModDown keeps the original t-inverse centering and
absorbs t, subtraction and P-inverse into public weights. Han-Ki's centered lift,
custom digit inverses and maximum-level key cofactors are preserved.

Gaussian boundaries fold encoding into R^2 and I*R^2 forward constants. Inverse
weights are normal 1/2 and 1/(2I), so REDC directly produces normal coefficients.
The noncontiguous eager path keeps full-pair conversion; compiled paths include
their actual contiguous copies. Native products without a finite-field imaginary
root retain Gaussian pair multiplication. X/Y twists, inverse normalization,
full-NTT A@B.T semantics and coefficient-contraction trace factors are unchanged.

Compiled Montgomery multiplication flattens contiguous operands. A scalar right
operand stays zero-dimensional. Broadcast tensor operands are materialized when
needed; their copies and dispatch costs are included in switching benchmarks.
The low-word cancellation intentionally wraps in int32, then masks 31 bits.
All wide products and accumulators are bounded without signed int64 overflow.

The regression tests compare independent Python-integer oracles at small and
30-bit moduli, q-1 boundaries, signed worst cases, scalar/vector/broadcast and
noncontiguous layouts. They check input/table immutability, graph reuse, fallback
and nested compiler behavior. CUDA cases require CUDA hardware.

See `paper_baseline.md` for frozen-fixture generation and measurement commands.
The raw switching JSON, validation logs and fixture manifest are under
`benchmarks/stack/`; historical handoff numbers are not used as new evidence.
