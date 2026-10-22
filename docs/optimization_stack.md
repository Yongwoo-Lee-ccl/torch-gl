# Pure PyTorch switching optimization stack

Fresh clone of `Yongwoo-Lee-ccl/torch-gl`, starting from `c9ce20df87d0a057cd03a4c8da37af8926bf0a7e`.
The separate paper baseline adds prime-17 Rader support. Historical optimization objects and timings were not restored.

Primary comparison: Apple M2, 16 GB, macOS arm64, Python 3.13.2, PyTorch 2.14.1. All five stages use the same frozen n=256,p=17 fixture, Q/P primes, ternary secrets, evaluation keys and inputs; k=2,dnum=3, evaluation-key noise std=3.2. Q has 136 aggregate bits and P has 59 aggregate bits. One Torch intra/inter-op and BLAS thread; macOS USER_INITIATED QoS was explicitly requested and verified for every stage. CPU affinity/exclusive CPU access is not available.

Each cell is mean ± sample standard deviation in seconds from five wall-time samples after operation and compiler warmup. Raw CPU and wall samples, graph counters and output hashes are adjacent JSON files. Allocation, dispatch, broadcasting and actual contiguous copies remain inside timing; compilation, key generation, input loading, hashes and correctness checks are outside. The primary run order was PR3, baseline, PR4, PR1, PR2; no tests or compiler setup ran concurrently with measured operations.

| Stage | SwitchSmall | SwitchBig | Two SwitchBig |
|---|---:|---:|---:|
| baseline | 7.906378 ± 0.063976 | 10.463903 ± 0.313330 | 21.078693 ± 0.403988 |
| pr1 | 6.468009 ± 0.066505 | 8.071194 ± 0.045869 | 16.525834 ± 0.101072 |
| pr2 | 5.626840 ± 0.143898 | 8.000311 ± 0.045554 | 19.461464 ± 1.699340 |
| pr3 | 5.776519 ± 0.143964 | 7.598736 ± 0.091927 | 17.065674 ± 1.225446 |
| pr4 | 4.438550 ± 0.117806 | 6.786682 ± 0.517322 | 13.721039 ± 0.530053 |

| Comparison | SwitchSmall speedup | SwitchBig speedup | Two SwitchBig speedup |
|---|---:|---:|---:|
| pr1 / baseline | 1.222x | 1.296x | 1.275x |
| pr2 / pr1 | 1.149x | 1.009x | 0.849x |
| pr3 / pr2 | 0.974x | 1.053x | 1.140x |
| pr4 / pr3 | 1.301x | 1.120x | 1.244x |

Net PR4 / baseline: SwitchSmall 1.781x, SwitchBig 1.542x, TwoSwitchBig 1.536x.

PR2 TwoSwitchBig mean regressed by 17.8% against PR1 while its SwitchBig was essentially tied. PR3 SwitchSmall mean was 2.7% slower than PR2; its variability does not establish an improvement. These regressions are retained, not hidden by the final aggregate speedup.

Small adjacent differences and noisy samples are descriptive means, not a statistical significance claim. Initial progress measurements are also preserved as baseline/pr1/pr2 JSON; PR2 TwoSwitchBig was particularly variable in that first pass. The interrupted pre-compatibility PR2 run is not used. Primary PR bodies use only the common-condition controlled JSON. No metric was changed to match historical handoff numbers.

All b/a limb output hashes match the frozen baseline exactly. Inputs and evaluation keys are immutable. Decryption differences are small centered integers consistent across all five Q limbs (Small max 229, Big max 1160, mixed max 1107, each checking 2,097,152 entries). These observed errors are not theoretical security bounds. Every measured group created zero new compiler graphs; normal fresh-process counts were 0,0,3,5,7 for baseline through PR4.

Validation: baseline 276 passed/5 CUDA skipped; PR1 300/5; PR2 full 346/5 plus 42 focused bound/cache and 4 compatibility tests; PR3 365/5; PR4 391/6 with the flag both off and globally on. Ruff check/format and diff whitespace checks passed. CUDA execution could not be tested on this machine.

The source contains no handwritten C++/CUDA/Triton extension. Inductor-generated implementation code is allowed. Full-NTT A@B.T and normalization are covered by a prime-17 regression test; X/Y twists, inverse 1/n and the coefficient contraction trace factor were preserved.

Regenerate the fixture with `scripts/benchmark_switch_stack.py --create-fixture --fixture fixture.pt`. For each archived stage, use `scripts/run_controlled_benchmark.py CHECKOUT --fixture FIXTURE --reference BASELINE_JSON --revision CODE_SHA --output OUTPUT` under the thread/compile flags described in cpu_compilation.md. Use TORCH_GL_COMPILE=0 for baseline/PR1 and 1 for PR2/PR3/PR4. The supplied fixture file and manifest preserve exact original tensors.
