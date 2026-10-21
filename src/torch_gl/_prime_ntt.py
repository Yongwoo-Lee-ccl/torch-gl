"""Prime cyclotomic evaluation via a length-(p-1) Rader convolution.

Public slots are rho**1,...,rho**(p-1). Private permuted calls use the
same order, so pointwise operations and matrix canonical gathers agree.
"""

import torch

from ._bounded import bounded_sum_reducer
from ._number_theory import unit_generator
from .modular import arithmetic_context
from .ntt import NTTPlan


def prime_kernel(q, p, root, device, dim, inverse):
    arithmetic = arithmetic_context(q)
    length = p - 1
    generator = unit_generator(p)
    powers = [pow(generator, index, p) for index in range(length)]
    inverse_powers = [pow(generator, -index, p) for index in range(length)]
    gather = torch.tensor([value - int(inverse) for value in inverse_powers], device=device)
    natural = torch.tensor([powers.index(index) for index in range(1, p)], device=device)
    plan = NTTPlan(q, length)
    forward = plan._kernel(device, permuted=True)
    backward = plan._kernel(device, inverse=True, permuted=True)
    kernel_root = pow(root, -1, q) if inverse else root
    weights = torch.tensor(
        [pow(kernel_root, value, q) for value in powers], dtype=torch.int32, device=device
    )
    spectrum = forward(arithmetic.encode(weights))
    inverse_p = arithmetic.constant(pow(p, -1, q))
    short_sum = bounded_sum_reducer(q, length) if length <= 16 else None

    def kernel(values):
        coefficients = values.movedim(dim, -1)
        if inverse:
            nonzero = coefficients.index_select(-1, gather)
        else:
            padded = torch.cat((coefficients, torch.zeros_like(coefficients[..., :1])), -1)
            nonzero = padded.index_select(-1, gather)
        convolution = backward(arithmetic.multiply(forward(nonzero), spectrum))
        if not inverse:
            result = arithmetic.add(convolution, coefficients[..., :1])
            return result.index_select(-1, natural).movedim(-1, dim)
        # DFT of [0,slots] at j=0 is the sum of its nonzero entries.
        zero = (
            short_sum(coefficients, keepdim=True)
            if short_sum is not None
            else coefficients.to(torch.int64).sum(-1, keepdim=True).remainder(q).to(torch.int32)
        )
        rest = convolution.index_select(-1, natural)
        joined = torch.cat((zero, rest[..., :-1]), -1)
        result = arithmetic.multiply(arithmetic.subtract(joined, rest[..., -1:]), inverse_p)
        return result.movedim(-1, dim)

    return kernel
