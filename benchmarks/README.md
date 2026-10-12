# Saved performance measurements

`cpu.json` records eager execution. `cpu_compiled.json` records an optional
Torch compiler run; each row states whether that operation was compiled.
Specialized Gaussian/cyclotomic transforms remain eager in both files.

These are warm medians for one development machine, not universal performance
claims. GPU results are absent because no CUDA device was available.
See [performance notes](../docs/performance.md) for methodology and the decision
to restrict this component to 30-bit prime limbs.

Run `scripts/benchmark_ntt.py` explicitly when collecting a new measurement.
Ordinary pytest and CI never regenerate these results or the SageMath KATs.
