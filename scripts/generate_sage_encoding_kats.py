"""Generate frozen canonical-embedding KATs with Sage, never during pytest.

Oracle: direct root evaluation and dense linear solves over ComplexField(256).
No FFTs, torch, or torch_gl imports. Run explicitly with sage -python.
"""

import argparse
import itertools
import json
import math
import random
from pathlib import Path

from sage.all import QQ, ComplexField, cyclotomic_polynomial, matrix, vector
from sage.env import SAGE_VERSION


def complex_pair(value):
    return [str(value.real()), str(value.imag())]


def make_case(name, n, p, variables, seed, scale=2**40):
    field = ComplexField(256)
    imaginary = field.gen()
    phi = cyclotomic_polynomial(p)
    degrees = [int(phi.degree()) if variable == "W" else n for variable in variables]
    W_dim = variables.index("W")
    slot_dims = [W_dim] + [dim for dim in range(len(variables)) if dim != W_dim]
    slot_shape = [degrees[dim] for dim in slot_dims]
    coefficient_indices = list(itertools.product(*(range(degree) for degree in degrees)))
    slot_indices = list(itertools.product(*(range(degree) for degree in slot_shape)))
    zeta = (2 * field.pi() * imaginary / (4 * n)).exp()
    eta = (2 * field.pi() * imaginary / p).exp()
    # Explicitly verify these are roots of the intended quotient relations.
    assert abs(zeta**n - imaginary) < field(2) ** -240
    assert abs(phi(eta)) < field(2) ** -230
    rows = []
    for slot_index in slot_indices:
        variable_slots = [slot_index[slot_dims.index(dim)] for dim in range(len(variables))]
        roots = [
            eta ** pow(2, slot, p) if variable == "W" else zeta ** pow(5, slot, 4 * n)
            for variable, slot in zip(variables, variable_slots, strict=True)
        ]
        rows.append(
            [
                math.prod(root**degree for root, degree in zip(roots, index, strict=True))
                for index in coefficient_indices
            ]
        )
    embedding = matrix(field, rows)
    rng = random.Random(seed)
    input_pairs = [[rng.randrange(-64, 65), rng.randrange(-64, 65)] for _ in slot_indices]
    inputs = vector(
        field, [field(QQ(real) / 16) + imaginary * QQ(imag) / 16 for real, imag in input_pairs]
    )
    inverse = embedding.solve_right(inputs)
    assert max(abs(value) for value in embedding * inverse - inputs) < field(2) ** -220
    encoded = [
        int(component.round())
        for value in inverse
        for component in ((value * scale).real(), (value * scale).imag())
    ]
    # Independent decode input, including signed coefficients above a 30-bit limb.
    decode_coefficients = [
        rng.randrange(-32, 33) * 2**40 + rng.randrange(-20, 21)
        for _ in range(2 * len(coefficient_indices))
    ]
    decode_polynomial = vector(
        field,
        [
            field(real) + imaginary * imag
            for real, imag in zip(decode_coefficients[::2], decode_coefficients[1::2], strict=True)
        ],
    )
    decoded = embedding * decode_polynomial / scale
    return {
        "name": name,
        "n": n,
        "p": p,
        "variables": variables,
        "coefficient_shape": degrees + [2],
        "slot_shape": slot_shape,
        "scale": scale,
        "seed": seed,
        "slot_denominator": 16,
        "input_slots": input_pairs,
        "inverse_coefficients": [complex_pair(value) for value in inverse],
        "encoded_coefficients": encoded,
        "decode_coefficients": decode_coefficients,
        "decoded_slots": [complex_pair(value) for value in decoded],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/data/sage_encoding.json",
    )
    args = parser.parse_args()
    specs = [
        ("full_n1_p3", 1, 3, ["X", "Y", "W"]),
        ("full_n2_p9", 2, 9, ["X", "Y", "W"]),
        ("full_n4_p9", 4, 9, ["X", "Y", "W"]),
        ("xw_n8_p27", 8, 27, ["X", "W"]),
        ("reordered_n2_p27", 2, 27, ["W", "Y", "X"]),
        ("w_p81", 1, 81, ["W"]),
    ]
    cases = [make_case(*spec, seed=1935 + index) for index, spec in enumerate(specs)]
    payload = {
        "schema_version": 1,
        "generator": "scripts/generate_sage_encoding_kats.py",
        "sage_version": SAGE_VERSION,
        "precision_bits": 256,
        "oracle": "Direct root evaluation and dense inverse over Sage ComplexField(256); no FFT",
        "root_order": "X/Y: exp(2*pi*i*5^j/(4n)); W: exp(2*pi*i*2^ell/p)",
        "layout": "C order; coefficients in variables order with real/imag last; slots W first",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Generated {len(cases)} canonical embedding KATs in {args.output}")


if __name__ == "__main__":
    main()
