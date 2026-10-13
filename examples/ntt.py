"""NTT examples: python examples/ntt.py [--device cuda] [--compile --backend eager]."""

import argparse

import torch

from torch_gl import (
    CyclotomicNTTPlan,
    GaussianNTTPlan,
    NTTPlan,
    PolynomialRing,
    montgomery_mul,
    select_ntt_primes,
    to_montgomery_form,
    to_normal_form,
)
from torch_gl.modular import arithmetic_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--backend", default="inductor")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("integer transforms support CPU and CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA hardware is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    # 1297-1 = 2**4 * 3**4: one prime supports many mixed-radix lengths.
    q = 1297
    for length in (1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 27, 54, 81, 108):
        plan = NTTPlan(q, length).prepare(device)
        coefficients = (17 * torch.arange(length, dtype=torch.int32, device=device)) % q
        assert torch.equal(plan.inverse(plan.forward(coefficients)), coefficients)
    print("Mixed-radix round trips: 14 supported lengths up to 108 checked")

    # All dimensions except dim are batches, including strided dimensions.
    coefficients = torch.arange(72, dtype=torch.int32, device=device).reshape(2, 12, 3)
    coefficients = coefficients.transpose(0, 2)
    original = coefficients.clone()
    plan = NTTPlan(q, 12)
    assert not coefficients.is_contiguous()
    assert torch.equal(plan.inverse(plan.forward(coefficients, dim=1), dim=1), coefficients)
    assert torch.equal(coefficients, original)
    print("Batched, noncontiguous input: dim=1 checked")

    # A cyclic convolution modulo T**12-1 using only public functions.
    left = torch.arange(1, 13, dtype=torch.int32, device=device)
    right = 2 * left + 1
    left_slots = to_montgomery_form(plan.forward(left), q)
    right_slots = to_montgomery_form(plan.forward(right), q)
    product = plan.inverse(to_normal_form(montgomery_mul(left_slots, right_slots)))
    expected = torch.tensor(
        [
            sum(
                (input_index + 1) * (2 * ((output_index - input_index) % 12 + 1) + 1)
                for input_index in range(12)
            )
            % q
            for output_index in range(12)
        ],
        dtype=torch.int32,
        device=device,
    )
    assert torch.equal(product, expected)
    print("Cyclic convolution: checked against a Python-integer oracle")

    # Internal inspection only: GS keeps digit order, CT consumes it directly.
    # Private kernels expect raw Montgomery tensors, not normal coefficients.
    plan = NTTPlan(q, 8)
    coefficients = torch.arange(8, dtype=torch.int32, device=device)
    arithmetic = arithmetic_context(q)
    forward = plan._kernel(device, permuted=True)
    inverse = plan._kernel(device, inverse=True, permuted=True)
    digit_slots = forward(arithmetic.encode(coefficients))
    digit_frequencies = torch.tensor([0, 4, 2, 6, 1, 5, 3, 7], device=device)
    assert torch.equal(
        arithmetic.decode(digit_slots),
        plan.forward(coefficients).index_select(0, digit_frequencies),
    )
    assert torch.equal(arithmetic.decode(inverse(digit_slots)), coefficients)
    print("GS/CT internal order for length 8: [0,4,2,6,1,5,3,7]")

    for p in (3, 9, 27, 81):
        plan = CyclotomicNTTPlan(q, p).prepare(device)
        coefficients = torch.arange(plan.length * 2, dtype=torch.int32, device=device).reshape(
            plan.length, 2
        )
        assert torch.equal(plan.inverse(plan.forward(coefficients, dim=0), dim=0), coefficients)
        ring = PolynomialRing.multivariate_poly(q, n=1, p=p, variables=("W",))
        W = ring.generator("W", device=device)
        stride = p // 3
        assert W ** (2 * stride) + W**stride + 1 == ring.zero(device=device)
    print("Compact cyclotomics: Phi_3, Phi_9, Phi_27, Phi_81 checked")

    for n in (1, 2, 4, 8, 16):
        plan = GaussianNTTPlan(q=193, length=n).prepare(device)
        coefficients = torch.arange(4 * n, dtype=torch.int32, device=device).reshape(2, n, 2)
        assert torch.equal(plan.inverse(plan.forward(coefficients, dim=1), dim=1), coefficients)
    print("Twisted Gaussian X**n=i transforms: n=1,2,4,8,16 checked")

    # The user's larger parameters, without allocating a full n*n*phi(p) polynomial.
    n, p = 128, 3**6
    large_q = select_ntt_primes(n, p, limb_count=1)[0]
    plan = CyclotomicNTTPlan(large_q, p).prepare(device)
    coefficients = torch.arange(3 * plan.length * 2, dtype=torch.int32, device=device).reshape(
        3, plan.length, 2
    )
    assert torch.equal(plan.inverse(plan.forward(coefficients, dim=1), dim=1), coefficients)
    print(f"n=128, p=729: compatible q={large_q}; W has {plan.length} slots")

    for variables in (("X", "Y", "W"), ("X", "W"), ("W", "Y", "X")):
        ring = PolynomialRing.multivariate_poly(433, n=4, p=27, variables=variables)
        X, W = ring.generator("X", device=device), ring.generator("W", device=device)
        left, right = X + 2 * W + 3, X + W**3
        product = left.multiply(right, compiled=args.compile, backend=args.backend)
        expected = X**2 + X * W**3 + 2 * X * W + 2 * W**4 + 3 * X + 3 * W**3
        assert product == expected
    print("Native polynomial multiplication: full, small, and reordered rings checked")

    if args.compile:
        plan = NTTPlan(q, 18)
        coefficients = torch.arange(54, dtype=torch.int32, device=device).reshape(3, 18)
        forward = plan.compile(device, backend=args.backend)
        inverse = plan.compile(device, inverse=True, backend=args.backend)
        assert torch.equal(forward(coefficients), plan.forward(coefficients))
        assert torch.equal(inverse(forward(coefficients)), coefficients)
        print(f"Compiled standard NTT and native products: backend={args.backend} checked")


if __name__ == "__main__":
    main()
