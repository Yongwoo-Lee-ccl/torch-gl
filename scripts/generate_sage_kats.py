"""Regenerate independent multiplication KATs: sage -python scripts/generate_sage_kats.py.

No torch_gl or torch imports: the oracle is Sage's multivariate ideal
reduction over GF(q), with explicit i**2+1. Flat arrays use C order.
"""

import argparse
import itertools
import json
import math
import random
from pathlib import Path

from sage.all import GF, PolynomialRing, cyclotomic_polynomial, is_prime
from sage.env import SAGE_VERSION


def large_prime(step, upper=False):
    q = (((1 << 62) - 2) // step * step + 1) if upper else ((1 << 60) // step * step + 1)
    while not is_prime(q):
        q += -step if upper else step
    return int(q)


def make_case(name, q, n, p, names, seed, custom=None):
    rng = random.Random(seed)
    ambient = PolynomialRing(GF(q), names=tuple(names) + ("i",), order="lex")
    generators = ambient.generators()
    imaginary = generators[-1]
    relations = [imaginary**2 + 1]
    axes = []
    for variable, symbol in zip(names, generators[:-1], strict=True):
        if custom is not None:
            modulus = custom
            relation = sum(
                (real + imaginary * imag) * symbol**coefficient_index
                for coefficient_index, (real, imag) in enumerate(modulus)
            )
        elif variable == "W":
            # Compute Phi_p via Sage, independently of the library formula.
            phi = cyclotomic_polynomial(p)
            modulus = [(int(coefficient), 0) for coefficient in phi.list()]
            relation = sum(
                coefficient * symbol**coefficient_index
                for coefficient_index, coefficient in enumerate(phi.list())
            )
        else:
            modulus = [(0, -1)] + [(0, 0)] * (n - 1) + [(1, 0)]
            relation = symbol**n - imaginary
        axes.append({"name": variable, "modulus": modulus})
        relations.append(relation)
    ideal = ambient.ideal(relations)
    shape = [len(left_polynomial["modulus"]) - 1 for left_polynomial in axes] + [2]
    indices = list(itertools.product(*(range(axis_length) for axis_length in shape)))

    def operand():
        # Include residues near q to expose signed int64 multiplication errors.
        choices = [0, 1, q - 1, q - 2]
        coefficients = [
            rng.choice(choices) if coefficient_index % 4 == 0 else rng.randrange(q)
            for coefficient_index in range(math.prod(shape))
        ]
        polynomial = ambient.zero()
        for index, coefficient in zip(indices, coefficients, strict=True):
            term = ambient(coefficient)
            for generator, exponent in zip(generators, index, strict=True):
                term *= generator**exponent
            polynomial += term
        return coefficients, polynomial

    left, left_polynomial = operand()
    right, right_polynomial = operand()
    reduced = ideal.reduce(left_polynomial * right_polynomial)
    expected = [
        int(
            reduced.monomial_coefficient(
                math.prod(
                    generator**exponent
                    for generator, exponent in zip(generators, index, strict=True)
                )
            )
        )
        for index in indices
    ]
    return {
        "name": name,
        "q": q,
        "n": n,
        "p": p,
        "variables": names,
        "axes": axes,
        "shape": shape,
        "seed": seed,
        "left": left,
        "right": right,
        "product": expected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed vectors")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/data/sage_multiplication.json",
    )
    args = parser.parse_args()
    specs = [
        ("full_n1_p3", 97, 1, 3, ["X", "Y", "W"]),
        ("full_n2_p3", 97, 2, 3, ["X", "Y", "W"]),
        ("full_n4_p9_native", 577, 4, 9, ["X", "Y", "W"]),
        ("full_n2_p27_native", 433, 2, 27, ["X", "Y", "W"]),
        ("full_n2_p81_native", 1297, 2, 81, ["X", "Y", "W"]),
        ("xw_n8_p9_native", 577, 8, 9, ["X", "W"]),
        ("xw_n4_p9_padded", 257, 4, 9, ["X", "W"]),
        ("full_n4_p5_padded", 193, 4, 5, ["X", "Y", "W"]),
        ("xw_n2_p25_padded", 257, 2, 25, ["X", "W"]),
        ("w_p27_native", 433, 1, 27, ["W"]),
        ("x_only", 97, 4, 3, ["X"]),
        ("gaussian_only", 97, 1, 3, []),
        ("large_60bit_native", large_prime(144), 2, 9, ["X", "W"]),
        ("large_near_62bit_native", large_prime(144, True), 2, 9, ["X", "W"]),
    ]
    cases = [
        make_case(*spec, seed=1935 + coefficient_index)
        for coefficient_index, spec in enumerate(specs)
    ]
    cases.append(
        make_case(
            "custom_gaussian_ideal",
            97,
            1,
            3,
            ["T"],
            seed=2025,
            custom=[(2, 1), (-3, 0), (0, 1), (1, 0)],
        )
    )
    payload = {
        "schema_version": 1,
        "generator": "scripts/generate_sage_kats.py",
        "sage_version": SAGE_VERSION,
        "oracle": "GF(q)[variables,i] ideal reduction with i^2+1",
        "coefficient_order": "C order; final axis (real, imaginary)",
        "cases": cases,
    }
    if args.check:
        committed = json.loads(args.output.read_text())
        # Metadata may differ across Sage versions; arithmetic must not.
        assert committed["cases"] == cases, "Sage KATs differ; regenerate and review"
        print(f"Verified {len(cases)} SageMath multiplication KATs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Generated {len(cases)} SageMath multiplication KATs in {args.output}")


if __name__ == "__main__":
    main()
