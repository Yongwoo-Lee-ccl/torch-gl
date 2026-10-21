"""Native-ring products with one Gaussian split for the entire calculation."""

from functools import lru_cache

import torch

from ._gaussian import gaussian_boundary
from ._number_theory import root_of_unity
from ._transform_tables import device_powers
from .modular import arithmetic_context, gaussian_mul_residues
from .ntt import NTTPlan
from .twisted import GaussianNTTPlan


@lru_cache(maxsize=32)
def native_multiplier(ring, device: torch.device, backend: str | None = None):
    """Bind immutable tables outside tracing; inputs are owned canonical residues."""
    arithmetic = arithmetic_context(ring.q)
    plans = ring.transform_plans
    gaussian_plans = [plan for plan in plans if isinstance(plan, GaussianNTTPlan)]
    if gaussian_plans:
        first_plan = gaussian_plans[0]
        imaginary_root = pow(first_plan.root, first_plan.length, ring.q)
    else:
        imaginary_root = root_of_unity(ring.q, 4) if (ring.q - 1) % 4 == 0 else None
    forward_steps, inverse_steps = [], []
    for axis_index, plan in enumerate(plans):
        forward_twist = inverse_twist = None
        if isinstance(plan, GaussianNTTPlan):
            root = plan.root
            if pow(root, plan.length, ring.q) != imaginary_root:
                root = pow(root, -1, ring.q)
            # Both split channels use the same n-th root. Their twists select
            # X^n=I and X^n=-I; no negative-frequency permutation is needed.
            shared_plan = NTTPlan(ring.q, plan.length, pow(root, 4, ring.q))
            forward_shape = [1] * (len(ring.axes) + 2)
            forward_shape[axis_index + 1], forward_shape[-1] = plan.length, 2
            inverse_shape = forward_shape[1:]
            inverse_shape[axis_index] = plan.length
            roots = (root, pow(root, -1, ring.q))
            forward_twist = device_powers(ring.q, roots, plan.length, str(device)).T.reshape(
                forward_shape
            )
            inverse_twist = device_powers(
                ring.q,
                roots[::-1],
                plan.length,
                str(device),
                scale=pow(plan.length, -1, ring.q),
            ).T.reshape(inverse_shape)
            inverse_kernel = shared_plan._kernel(
                device, dim=axis_index, inverse=True, permuted=True, normalize=False
            )
        else:
            shared_plan = plan
            inverse_kernel = shared_plan._kernel(
                device, dim=axis_index, inverse=True, permuted=True
            )
        forward_steps.append(
            (shared_plan._kernel(device, dim=axis_index + 1, permuted=True), forward_twist)
        )
        inverse_steps.append((inverse_kernel, inverse_twist))
    forward_steps, inverse_steps = tuple(forward_steps), tuple(reversed(inverse_steps))
    boundary = gaussian_boundary(ring.q, imaginary_root) if imaginary_root is not None else None

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        operands = torch.stack((left, right), dim=0)
        operands = boundary.split(operands) if boundary is not None else arithmetic.encode(operands)
        for transform, twist in forward_steps:
            if twist is not None:
                operands = arithmetic.multiply(operands, twist)
            operands = transform(operands)
        left_slots, right_slots = operands.unbind(0)
        product = (
            arithmetic.multiply(left_slots, right_slots)
            if imaginary_root is not None
            else (gaussian_mul_residues(left_slots, right_slots, arithmetic))
        )
        for transform, twist in inverse_steps:
            product = transform(product)
            if twist is not None:
                product = arithmetic.multiply(product, twist)
        return boundary.join(product) if boundary is not None else arithmetic.decode(product)

    return torch.compile(kernel, backend=backend, fullgraph=True) if backend else kernel
