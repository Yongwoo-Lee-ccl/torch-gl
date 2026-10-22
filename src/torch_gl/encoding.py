"""Scaled complex canonical embedding from Section 3.1 of the paper."""

import math

import torch

from ._number_theory import odd_prime_power, supports_native_cyclotomic, unit_generator
from ._transform_tables import resolve_device
from .multivariate_poly import MultivariatePoly
from .rings import QuotientAxis
from .rns import RNSPoly, RNSRing


class MatrixEncoder:
    """Encode matrices as rounded Gaussian coefficients; decode noisy RNS values.

    Slots have W first, followed by the other variables in ring order. Thus the
    default ring accepts (phi(p), n, n) matrices. X/Y roots are exp(2*pi*i*5**j/(4n));
    W roots follow a deterministic generator of the units modulo p (2 for 3**k).
    Floating transforms use complex128. Stored polynomial limbs remain int32.
    """

    def __init__(self, ring: RNSRing, *, scale: float = 2**40, device: torch.device | str = "cpu"):
        if not isinstance(ring, RNSRing):
            raise TypeError("ring must be an RNSRing")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ValueError("scale must be finite and positive")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        if not ring.axes or len({axis.name for axis in ring.axes} - {"X", "Y", "W", "V"}):
            raise ValueError("encoding requires GL power-of-i and cyclotomic axes")
        self.ring, self.scale, self.device = ring, float(scale), resolve_device(device)
        self._W_dim = next((dim for dim, axis in enumerate(ring.axes) if axis.name == "W"), None)
        if self._W_dim is None:
            raise ValueError("encoding requires a W cyclotomic axis")
        W_axis = ring.axes[self._W_dim]
        self.p = W_axis.cyclotomic_order
        if self.p is None or not supports_native_cyclotomic(self.p):
            raise ValueError("encoding currently supports native cyclotomics")
        self._cyclotomics = {}
        self._twists = {}
        self._permutations = {}
        for dim, axis in enumerate(ring.axes):
            if axis.cyclotomic_order:
                p = axis.cyclotomic_order
                if not supports_native_cyclotomic(p):
                    raise ValueError("encoding currently supports native cyclotomics")
                indices = torch.tensor(
                    [pow(unit_generator(p), slot, p) for slot in range(axis.degree)],
                    device=self.device,
                )
                self._cyclotomics[dim] = (p, p // odd_prime_power(p)[0], indices)
                continue
            if axis != QuotientAxis.power_of_i(axis.name, axis.degree):
                raise ValueError("X/Y axes must satisfy X**n=i or Y**n=i")
            n = axis.degree
            powers = [pow(5, slot, 4 * n) for slot in range(n)]
            permutation = torch.tensor([(power - 1) // 4 for power in powers], device=self.device)
            self._permutations[dim] = (permutation, torch.argsort(permutation))
            shape = [1] * len(ring.axes)
            shape[dim] = n
            angles = torch.arange(n, dtype=torch.float64, device=self.device) * (math.pi / (2 * n))
            self._twists[dim] = torch.polar(torch.ones_like(angles), angles).reshape(shape)
        self._slot_dims = (self._W_dim,) + tuple(
            dim for dim in range(len(ring.axes)) if dim != self._W_dim
        )
        self._coefficient_dims = tuple(self._slot_dims.index(dim) for dim in range(len(ring.axes)))
        self.slot_shape = tuple(ring.axes[dim].degree for dim in self._slot_dims)

    def _evaluate(self, coefficients: torch.Tensor) -> torch.Tensor:
        values = coefficients
        for dim, twist in self._twists.items():
            values = torch.fft.ifft(values * twist, dim=dim) * self.ring.axes[dim].degree
            values = values.index_select(dim, self._permutations[dim][0])
        for dim, (p, _, indices) in self._cyclotomics.items():
            values = torch.fft.ifft(values, n=p, dim=dim) * p
            values = values.index_select(dim, indices)
        return values.permute(self._slot_dims)

    def _interpolate(self, slots: torch.Tensor) -> torch.Tensor:
        values = slots.permute(self._coefficient_dims)
        for dim, (p, stride, indices) in reversed(tuple(self._cyclotomics.items())):
            padded_shape = list(values.shape)
            padded_shape[dim] = p
            padded = torch.zeros(padded_shape, dtype=torch.complex128, device=self.device)
            padded.index_copy_(dim, indices, values)
            projected = torch.fft.fft(padded, dim=dim) / p
            # Reduce the length-p interpolant modulo Phi_p. Its tail is not
            # zero, even though values at nonprimitive roots were padded by 0.
            prime = odd_prime_power(p)[0]
            degree = (prime - 1) * stride
            tail = projected.narrow(dim, degree, stride)
            values = projected.narrow(dim, 0, degree) - torch.cat((tail,) * (prime - 1), dim=dim)
        for dim, twist in reversed(tuple(self._twists.items())):
            values = values.index_select(dim, self._permutations[dim][1])
            values = torch.fft.fft(values, dim=dim) / self.ring.axes[dim].degree
            values = values * twist.conj()
        return values

    def encode(self, matrices: torch.Tensor) -> RNSPoly:
        """Round floating coefficients to Python integers, then split into RNS.

        Integer size is bounded by Q/2, not by an individual limb. Numerical
        precision is bounded by the complex128 interpolation, not by Python ints.
        """
        if not isinstance(matrices, torch.Tensor):
            raise TypeError("matrices must be a Torch tensor")
        if matrices.device != self.device or tuple(matrices.shape) != self.slot_shape:
            raise ValueError("matrix shape or device does not match the encoder")
        if matrices.dtype == torch.bool:
            raise TypeError("boolean matrices are unsupported")
        if not bool(torch.isfinite(matrices).all()):
            raise ValueError("matrices must contain finite values")
        coefficients = torch.view_as_real(self._interpolate(matrices.to(torch.complex128)))
        scaled = coefficients * self.scale
        if not bool(torch.isfinite(scaled).all()):
            raise ValueError("scaled coefficients must be finite")
        integers = [round(value) for value in scaled.reshape(-1).cpu().tolist()]
        if any(abs(value) > self.ring.modulus // 2 for value in integers):
            raise ValueError("scaled coefficients exceed centered aggregate RNS range")
        return RNSPoly(
            self.ring,
            tuple(
                MultivariatePoly._from_owned(
                    limb_ring,
                    torch.tensor(
                        [value % limb_ring.q for value in integers],
                        dtype=torch.int32,
                        device=self.device,
                    ).reshape(self.ring.shape),
                )
                for limb_ring in self.ring.rings
            ),
        )

    def decode(self, message: RNSPoly) -> torch.Tensor:
        """Evaluate centered coefficients and divide by scale; retain complex slots."""
        if not isinstance(message, RNSPoly):
            raise TypeError("message must be an RNSPoly")
        if message.ring != self.ring or message.device != self.device:
            raise ValueError("message must have the encoder's ring and device")
        # Center modulo Q=product(moduli) in Python integers before any float
        # conversion. A single limb cannot determine the sign of a large value.
        coefficients = torch.tensor(
            message.reconstruct(centered=True), dtype=torch.float64, device=self.device
        ).reshape(self.ring.shape)
        complex_coefficients = torch.complex(coefficients[..., 0], coefficients[..., 1])
        return self._evaluate(complex_coefficients) / self.scale
