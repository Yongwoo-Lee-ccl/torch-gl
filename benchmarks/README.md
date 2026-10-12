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

For `scripts/benchmark_key_switching.py --device cuda:0`, the PyTorch CUDA
runtime must be compatible with the installed NVIDIA driver. Check
`nvidia-smi` and `python -c 'import torch; print(torch.__version__, torch.version.cuda)'`.
For a driver supporting CUDA 12.4, install the matching build in the project
virtual environment:

```bash
python -m pip install 'torch==2.6.0+cu124' --extra-index-url https://download.pytorch.org/whl/cu124
```

See the [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/)
for other CUDA builds. The key-switching benchmark checks CUDA initialization
before writing its output, so a device initialization failure preserves an
existing report.
