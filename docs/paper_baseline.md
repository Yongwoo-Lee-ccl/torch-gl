# Fixed paper-parameter baseline

This baseline starts at remote `feat/gl-bootstrapping`, commit
`c9ce20df87d0a057cd03a4c8da37af8926bf0a7e`, fetched by a fresh clone.
No historical optimization object, ZIP, or benchmark result was restored.

Prime cyclotomics with 2/3-smooth p-1 use a Rader convolution. In particular,
p=17 uses a 16-point NTT and natural public evaluations rho^1,...,rho^16.
The previous power-of-three path and its slot order are preserved. Matrix
canonical permutations and the complex encoder use a genuine unit generator
(3 for 17; 2 for powers of three). X/Y normalization, coefficient contraction,
and matrix transpose semantics are unchanged.

`python scripts/benchmark_switch_stack.py --create-fixture --fixture fixture.pt`
creates the n=256,p=17 fixture once. Five Q primes have 28 bits; two P primes
have 30 bits; k=2,dnum=3. The script fixes four independent RNG streams and
stores all three evaluation keys, inputs, secrets and expected source products.
The adjacent JSON specifies primes and seeds. A recursive SHA-256 covers
metadata and each tensor's dtype, shape and C-order bytes. Every later run
loads and checks those tensors, then compares all output hashes to baseline.
Only key switching is timed; key generation, compilation, validation and hashes
are outside timing. Copies, dispatch and fresh buffers used by the operation
remain inside timing. One warmup and at least five raw wall/CPU samples are
recorded. Each decryption difference is verified to be a small consistent
centered value across every Q limb; it is not a theoretical security bound.

Set OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS and
TORCHINDUCTOR_COMPILE_THREADS to 1 before Python. The script sets both Torch
thread counts to 1. Linux can additionally use taskset; macOS has no supported
sched_setaffinity API and this limitation is recorded explicitly. Do not run
other tests or compiler setup concurrently with a measured benchmark.
