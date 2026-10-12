"""Tests consume frozen Sage vectors; Sage is never imported by pytest."""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sage_vectors():
    return json.loads((Path(__file__).parent / "data/sage_multiplication.json").read_text())
