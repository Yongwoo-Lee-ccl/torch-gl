import math

import pytest
import torch

from torch_gl import MatrixEncoder, MatrixNTTPlan, RNSRing, matrix_multiply
from torch_gl.matrix import _matrix_product_residues
from torch_gl.modular import arithmetic_context

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def direct_finite_embedding(coefficients, plan):
    n, p, q = plan.n, plan.p, plan.ring.q
    output = torch.empty(plan.slot_shape, dtype=torch.int32)
    for W_slot in range(2 * p // 3):
        for row in range(n):
            for column in range(n):
                for branch, sign in enumerate((1, -1)):
                    X_root = pow(plan.X_root, sign * pow(5, row, 4 * n), q)
                    Y_root = pow(plan.X_root, sign * pow(5, column, 4 * n), q)
                    W_root = pow(plan.W_root, sign * pow(2, W_slot, p), q)
                    imaginary = sign * plan.imaginary_root
                    total = 0
                    for exponent_X in range(n):
                        for exponent_Y in range(n):
                            for exponent_W in range(2 * p // 3):
                                real, imag = coefficients[
                                    exponent_X, exponent_Y, exponent_W
                                ].tolist()
                                total += (
                                    (real + imaginary * imag)
                                    * pow(X_root, exponent_X, q)
                                    * pow(Y_root, exponent_Y, q)
                                    * pow(W_root, exponent_W, q)
                                )
                    output[W_slot, row, column, branch] = total % q
    return output


def direct_trace_product(left, right, n, p, q):
    """Integer coefficient oracle: sum roots is n or n*i, otherwise zero."""
    degree = 2 * p // 3
    stride = p // 3
    output = torch.zeros_like(left)
    for row in range(n):
        for column in range(n):
            convolution = [[0, 0] for _ in range(2 * degree - 1)]
            for inner in range(n):
                paired_inner = (-inner) % n
                for left_W in range(degree):
                    left_real, left_imag = left[row, inner, left_W].tolist()
                    for right_W in range(degree):
                        right_real, right_imag = right[paired_inner, column, right_W].tolist()
                        real = left_real * right_real - left_imag * right_imag
                        imag = left_real * right_imag + left_imag * right_real
                        if inner != 0:
                            real, imag = -imag, real  # Sum of roots contributes n*i.
                        convolution[left_W + right_W][0] += n * real
                        convolution[left_W + right_W][1] += n * imag
            # Reduce with Phi_p=1+W^stride+W^(2*stride), using Python ints.
            for exponent in range(len(convolution) - 1, degree - 1, -1):
                for channel in range(2):
                    value = convolution[exponent][channel]
                    convolution[exponent - degree][channel] -= value
                    convolution[exponent - stride][channel] -= value
            output[row, column] = torch.tensor(
                [[real % q, imag % q] for real, imag in convolution[:degree]], dtype=torch.int32
            )
    return output


def test_matrix_ntt_matches_direct_evaluation_and_trace_coefficients():
    ring = RNSRing.for_parameters(4, 9, limb_count=1).rings[0]
    plan = MatrixNTTPlan(ring)
    generator = torch.Generator().manual_seed(1935)
    left_values = torch.randint(ring.q, ring.shape, dtype=torch.int32, generator=generator)
    right_values = torch.randint(ring.q, ring.shape, dtype=torch.int32, generator=generator)
    left, right = ring.polynomial(left_values), ring.polynomial(right_values)
    slots = plan.forward(left)
    assert torch.equal(slots, direct_finite_embedding(left_values, plan))
    assert plan.inverse(slots) == left
    expected = direct_trace_product(left_values, right_values, plan.n, plan.p, ring.q)
    assert torch.equal(plan.multiply(left, right).coefficients, expected)


def test_internal_matrix_product_does_not_permute_slots(monkeypatch):
    ring = RNSRing.for_parameters(4, 9, limb_count=1).rings[0]
    plan = MatrixNTTPlan(ring)
    generator = torch.Generator().manual_seed(1935)
    left_values = torch.randint(ring.q, ring.shape, dtype=torch.int32, generator=generator)
    right_values = torch.randint(ring.q, ring.shape, dtype=torch.int32, generator=generator)
    expected = direct_trace_product(left_values, right_values, plan.n, plan.p, ring.q)

    def reject_permutation(*args, **kwargs):
        raise AssertionError("internal matrix contraction must retain digit order")

    monkeypatch.setattr(torch.Tensor, "index_select", reject_permutation)
    result = plan.multiply(ring.polynomial(left_values), ring.polynomial(right_values))
    assert torch.equal(result.coefficients, expected)


def test_modular_matrix_sum_reduces_before_int64_overflow():
    ring = RNSRing.for_parameters(32, 3, limb_count=1).rings[0]
    arithmetic = arithmetic_context(ring.q)
    shape = MatrixNTTPlan(ring).slot_shape
    values = torch.full(shape, ring.q - 1, dtype=torch.int32)
    assert 32 * (ring.q - 1) ** 2 > 2**63 - 1
    encoded = arithmetic.encode(values)
    product = arithmetic.decode(_matrix_product_residues(encoded, encoded, arithmetic))
    assert torch.equal(product, torch.full(shape, 32, dtype=torch.int32))


@pytest.mark.parametrize("device", DEVICES)
def test_split_gemm_matches_python_integer_dot_products_at_30bit_boundaries(device):
    q, n = 1073741789, 8
    arithmetic = arithmetic_context(q)
    boundary = [0, 1, (1 << 15) - 1, 1 << 15, (1 << 29), q // 2, q - 2, q - 1]
    left_values = [[boundary[(row + 3 * column) % n] for column in range(n)] for row in range(n)]
    right_values = [[boundary[(5 * row + column) % n] for column in range(n)] for row in range(n)]
    left = torch.tensor(left_values, dtype=torch.int32, device=device)
    right = torch.tensor(right_values, dtype=torch.int32, device=device)
    left = torch.stack((left, left.flip(0)), -1).unsqueeze(0)
    right = torch.stack((right, right.flip(1)), -1).unsqueeze(0)
    product = arithmetic.decode(
        _matrix_product_residues(arithmetic.encode(left), arithmetic.encode(right), arithmetic)
    )
    left_python, right_python = left.cpu().tolist()[0], right.cpu().tolist()[0]
    expected = [
        [
            [
                sum(
                    left_python[row][inner][branch] * right_python[inner][column][branch]
                    for inner in range(n)
                )
                % q
                for branch in range(2)
            ]
            for column in range(n)
        ]
        for row in range(n)
    ]
    assert product.cpu().tolist() == [expected]


def test_internal_matrix_multiply_only_transforms_W(monkeypatch):
    ring = RNSRing.for_parameters(4, 9, limb_count=1).rings[0]
    plan = MatrixNTTPlan(ring)
    generator = torch.Generator().manual_seed(1935)
    left = ring.polynomial(
        torch.randint(ring.q, ring.shape, dtype=torch.int32, generator=generator)
    )
    right = ring.polynomial(
        torch.randint(ring.q, ring.shape, dtype=torch.int32, generator=generator)
    )
    expected = direct_trace_product(left.coefficients, right.coefficients, plan.n, plan.p, ring.q)

    def reject_full_transform(*args, **kwargs):
        raise AssertionError("coefficient contraction must not perform X/Y transforms")

    monkeypatch.setattr(plan, "_forward_residues", reject_full_transform)
    monkeypatch.setattr(plan, "_inverse_residues", reject_full_transform)
    assert torch.equal(plan.multiply(left, right).coefficients, expected)


def test_single_row_matrix_contraction_equals_polynomial_product():
    ring = RNSRing.for_parameters(1, 27, limb_count=1).rings[0]
    plan = MatrixNTTPlan(ring)
    generator = torch.Generator().manual_seed(1935)
    left = ring.polynomial(
        torch.randint(ring.q, ring.shape, dtype=torch.int32, generator=generator)
    )
    right = ring.polynomial(
        torch.randint(ring.q, ring.shape, dtype=torch.int32, generator=generator)
    )
    assert plan.multiply(left, right) == left * right


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "n,p,variables",
    [(4, 9, ("X", "Y", "W")), (4, 27, ("W", "Y", "X")), (2, 729, ("X", "Y", "W"))],
)
def test_encoded_plaintext_matrix_product_decodes_to_W_batched_matmul(device, n, p, variables):
    ring = RNSRing.for_parameters(n, p, limb_count=4, variables=variables)
    encoder = MatrixEncoder(ring, device=device)
    product_decoder = MatrixEncoder(ring, scale=encoder.scale**2, device=device)
    generator = torch.Generator(device=device).manual_seed(1935)
    left = torch.randn(
        encoder.slot_shape, dtype=torch.complex128, device=device, generator=generator
    )
    right = torch.randn(
        encoder.slot_shape, dtype=torch.complex128, device=device, generator=generator
    )
    left_encoded, right_encoded = encoder.encode(left), encoder.encode(right)
    product = matrix_multiply(left_encoded, right_encoded)
    # The W frequency and Gaussian branches are batch dimensions, never summed.
    for left_limb, right_limb, product_limb in zip(
        left_encoded.limbs, right_encoded.limbs, product.limbs, strict=True
    ):
        plan = MatrixNTTPlan(left_limb.ring, device=device)
        arithmetic = arithmetic_context(left_limb.ring.q)
        expected_slots = arithmetic.decode(
            _matrix_product_residues(
                arithmetic.encode(plan.forward(left_limb)),
                arithmetic.encode(plan.forward(right_limb)),
                arithmetic,
            )
        )
        assert torch.equal(plan.forward(product_limb), expected_slots)
    left_l1 = sum(abs(value) for value in left_encoded.reconstruct())
    right_l1 = sum(abs(value) for value in right_encoded.reconstruct())
    assert 2 * n * left_l1 * right_l1 < ring.modulus // 2
    recovered = product_decoder.decode(product)
    rounded_expected = encoder.decode(left_encoded) @ encoder.decode(right_encoded)
    torch.testing.assert_close(recovered, rounded_expected, atol=2e-11, rtol=2e-12)
    rounding_bound = math.prod(ring.shape[:-1]) / (math.sqrt(2) * encoder.scale)
    error_bound = n * (
        (left.abs().max().item() + right.abs().max().item()) * rounding_bound + rounding_bound**2
    )
    assert (recovered - left @ right).abs().max().item() <= error_bound + 2e-11
    # Non-symmetric independent inputs detect a transpose/conjugate error.
    assert not torch.allclose(recovered, left @ right.mH, atol=1e-6, rtol=1e-6)


def test_matrix_multiply_rejects_incompatible_contexts():
    ring = RNSRing.for_parameters(2, 9)
    with pytest.raises(TypeError):
        matrix_multiply(ring.one(), 1)
    with pytest.raises(ValueError):
        matrix_multiply(ring.one(), RNSRing.for_parameters(4, 9).one())
    with pytest.raises(ValueError, match="X, Y, W"):
        MatrixNTTPlan(RNSRing.for_parameters(2, 9, variables=("X", "W")).rings[0])
