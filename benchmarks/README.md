# Saved performance measurements

`cpu.json` records eager execution. `cpu_compiled.json` records an optional
Torch compiler run; each row states whether that operation was compiled.
Specialized Gaussian/cyclotomic transforms remain eager in both files.

These are warm medians for one development machine, not universal performance
claims. GPU results are absent because no CUDA device was available.
These reports predate the int32-storage/Montgomery pipeline and are historical;
they do not measure the current implementation.

`native_multiply_int32.json` records the current compiled native-ring product
at n=128 and p=243, with three warmups and ten timed runs, validated against
independent monomial coefficient reduction.
See [performance notes](../docs/performance.md) for methodology and the decision
to restrict this component to 30-bit prime limbs.

Run `scripts/benchmark_ntt.py` explicitly when collecting a new measurement.
Ordinary pytest and CI never regenerate these results or the SageMath KATs.
