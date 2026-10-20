import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("example", ["linear_regression.py", "mlp.py"])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_ml_examples_are_runnable_and_check_plaintext_reference(example, device):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "examples" / example),
            "--device",
            device,
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Maximum absolute error:" in result.stdout
    assert "Remaining modulus limbs:" in result.stdout
