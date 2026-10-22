"""Public bounds for exact signed Montgomery linear forms and short sums."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from ._compilation import maybe_linear
from .modular import add_mod
from .montgomery import MONTGOMERY_RADIX, montgomery_context


@dataclass(frozen=True)
class MontgomeryLinearForm:
    q: int
    bounds: tuple[int, ...]
    weights: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    negative_inverse: int

    def __call__(self, values):
        if len(values) != len(self.bounds):
            raise ValueError("expected one tensor per public bound")
        # Broadcast metadata is cheap and no input is mutated or cached.
        values = torch.broadcast_tensors(*values)
        result = None
        for group in self.groups:
            inputs = tuple(values[index] for index in group)
            weights = tuple(self.weights[index] for index in group)
            reduced = maybe_linear(inputs, weights, self.q, self.negative_inverse)
            if reduced is None:
                total = inputs[0].to(torch.int64) * weights[0]
                for value, weight in zip(inputs[1:], weights[1:], strict=True):
                    total.add_(value, alpha=weight)
                cancellation = total.to(torch.int32)
                cancellation.mul_(self.negative_inverse).bitwise_and_(MONTGOMERY_RADIX - 1)
                total.add_(cancellation, alpha=self.q).bitwise_right_shift_(31)
                reduced = total.to(torch.int32)
                reduced.add_(reduced < 0, alpha=self.q)
                reduced.add_(reduced >= self.q, alpha=-self.q)
            result = reduced if result is None else add_mod(result, reduced, self.q)
        return torch.zeros_like(values[0]) if result is None else result


@lru_cache(maxsize=256)
def montgomery_linear_form(q, bounds, weights):
    """Prepare sum(v_i*w_i) mod q with |v_i|<=bounds_i, from public metadata.

    Each group has sum(bounds_i * encoded_weight_i) < q*R. Therefore
    -q*R < T < q*R and -q < REDC(T) < 2*q. T+m*q is below 2*q*R < 2**62.
    The intentional int32 cancellation wrap is the only overflowing operation.
    """
    context = montgomery_context(q)
    if (
        not bounds
        or len(bounds) != len(weights)
        or any(type(b) is not int or b < 0 for b in bounds)
    ):
        raise ValueError("nonnegative integer bounds must match a nonempty weight list")
    if any(type(weight) is not int for weight in weights):
        raise ValueError("weights must be public Python integers")
    encoded = tuple(context.encode_integer(weight) for weight in weights)
    groups = []
    current = []
    cost = 0
    for index, (bound, weight) in enumerate(zip(bounds, encoded, strict=True)):
        term_cost = bound * weight
        if term_cost >= q * MONTGOMERY_RADIX:
            raise ValueError("an individual term exceeds the signed REDC bound")
        if not weight:
            continue
        if current and cost + term_cost >= q * MONTGOMERY_RADIX:
            groups.append(tuple(current))
            current, cost = [], 0
        current.append(index)
        cost += term_cost
    if current:
        groups.append(tuple(current))
    return MontgomeryLinearForm(q, bounds, encoded, tuple(groups), context.negative_inverse)


@lru_cache(maxsize=128)
def bounded_sum_reducer(q, count):
    """Canonical sum of at most 16 canonical terms, using an integer reciprocal.

    S*floor(2**58/q) < 16*2**58 = 2**62. The quotient estimate is at
    most one low because S < 16*q and S/2**58 < 1 for q<2**30.
    """
    montgomery_context(q)
    if type(count) is not int or not 1 <= count <= 16:
        raise ValueError("bounded sum requires between 1 and 16 terms")
    reciprocal = (1 << 58) // q

    def reduce(values, *, dim=-1, keepdim=False):
        if values.shape[dim] != count:
            raise ValueError("sum axis does not match its public term count")
        total = values.to(torch.int64).sum(dim=dim, keepdim=keepdim)
        quotient = (total * reciprocal) >> 58
        reduced = (total - quotient * q).to(torch.int32)
        return reduced.add_(reduced >= q, alpha=-q)

    return reduce
