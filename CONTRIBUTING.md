# Contributing

Create a virtual environment and install `python -m pip install -e '.[dev]'`.
Use descriptive implementation variable names; retain mathematical parameters
such as `n`, `p`, and `q` where their meaning is explicit.

Before submitting changes:

```sh
ruff format .
ruff check .
pytest
python -m build --no-isolation
```

Normal tests must read committed SageMath KATs. Do not import SageMath or invoke
the generator from pytest, package imports, builds, or CI. Updating vectors is
an explicit maintenance task: run the generator once, review the file diff,
and commit the resulting JSON with the corresponding change.

Preserve exact arithmetic, overflow bounds, device placement, and the distinction
between formal `i` and a modular square root of minus one. Document transform
roots, normalization, and slot ordering when adding algorithms.
