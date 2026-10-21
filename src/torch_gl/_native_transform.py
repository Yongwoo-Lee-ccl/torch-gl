"""Prepared native Gaussian-split transforms, including independent batch axes."""

from ._gaussian import gaussian_boundary
from ._number_theory import root_of_unity
from ._transform_tables import device_powers
from .cyclotomic import CyclotomicNTTPlan
from .modular import arithmetic_context
from .ntt import NTTPlan
from .twisted import GaussianNTTPlan


class NativeTransform:
    """Private normal coefficients <-> Montgomery split-slot transform.

    Batch axes precede ring axes. Small switching uses Y as a batch, never as
    a transform dimension; big switching includes Y among the ring axes.
    """

    def __init__(self, ring, device, *, batch_ndim=0):
        self.ring, self.batch_ndim = ring, batch_ndim
        self.arithmetic = arithmetic_context(ring.q)
        plans = ring.transform_plans
        if any(not isinstance(plan, (GaussianNTTPlan, CyclotomicNTTPlan)) for plan in plans):
            raise ValueError("key switching requires native Gaussian/cyclotomic transforms")
        gaussian = [plan for plan in plans if isinstance(plan, GaussianNTTPlan)]
        imaginary_root = (
            pow(gaussian[0].root, gaussian[0].length, ring.q)
            if gaussian
            else root_of_unity(ring.q, 4)
        )
        self._gaussian = gaussian_boundary(ring.q, imaginary_root)
        self._forward_steps, self._inverse_steps = [], []
        for ring_dim, plan in enumerate(plans):
            dim = batch_ndim + ring_dim
            twist = inverse_twist = None
            if isinstance(plan, GaussianNTTPlan):
                root = plan.root
                if pow(root, plan.length, ring.q) != imaginary_root:
                    root = pow(root, -1, ring.q)
                shared = NTTPlan(ring.q, plan.length, pow(root, 4, ring.q))
                shape = [1] * (batch_ndim + len(plans) + 1)
                shape[dim], shape[-1] = plan.length, 2
                roots = (root, pow(root, -1, ring.q))
                twist = device_powers(ring.q, roots, plan.length, str(device)).T.reshape(shape)
                inverse_twist = device_powers(
                    ring.q,
                    roots[::-1],
                    plan.length,
                    str(device),
                    scale=pow(plan.length, -1, ring.q),
                ).T.reshape(shape)
                inverse_kernel = shared._kernel(
                    device, dim=dim, inverse=True, permuted=True, normalize=False
                )
            else:
                shared = plan
                inverse_kernel = shared._kernel(device, dim=dim, inverse=True, permuted=True)
            self._forward_steps.append((shared._kernel(device, dim=dim, permuted=True), twist))
            self._inverse_steps.append((inverse_kernel, inverse_twist))
        self._inverse_steps.reverse()

    def forward(self, coefficients):
        arithmetic = self.arithmetic
        values = self._gaussian.split(coefficients)
        for transform, twist in self._forward_steps:
            if twist is not None:
                values = arithmetic.multiply(values, twist)
            values = transform(values)
        return values

    def inverse(self, values):
        arithmetic = self.arithmetic
        for transform, twist in self._inverse_steps:
            values = transform(values)
            if twist is not None:
                values = arithmetic.multiply(values, twist)
        return self._gaussian.join(values)
