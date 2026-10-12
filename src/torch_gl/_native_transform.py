"""Prepared native Gaussian-split transforms, including independent batch axes."""

import torch

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
        self._imaginary = self.arithmetic.constant(imaginary_root)
        self._inverse_two = self.arithmetic.constant(pow(2, -1, ring.q))
        self._inverse_two_imaginary = self.arithmetic.constant(pow(2 * imaginary_root, -1, ring.q))
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
                    ring.q, roots[::-1], plan.length, str(device)
                ).T.reshape(shape)
            else:
                shared = plan
            self._forward_steps.append((shared._kernel(device, dim=dim, permuted=True), twist))
            self._inverse_steps.append(
                (shared._kernel(device, dim=dim, inverse=True, permuted=True), inverse_twist)
            )
        self._inverse_steps.reverse()

    def forward(self, coefficients):
        arithmetic = self.arithmetic
        real, imag = arithmetic.encode(coefficients).unbind(-1)
        imaginary = arithmetic.multiply(imag, self._imaginary)
        values = torch.stack(
            (arithmetic.add(real, imaginary), arithmetic.subtract(real, imaginary)), -1
        )
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
        positive, negative = values.unbind(-1)
        real = arithmetic.multiply(arithmetic.add(positive, negative), self._inverse_two)
        imag = arithmetic.multiply(
            arithmetic.subtract(positive, negative), self._inverse_two_imaginary
        )
        return arithmetic.decode(torch.stack((real, imag), -1))
