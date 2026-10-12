"""Scaled complex canonical embedding from Section 3.1 of the paper."""

import math

import torch

from ._number_theory import odd_prime_power
from ._transform_tables import resolve_device
from .multivariate_poly import MultivariatePoly
from .rings import QuotientAxis
from .rns import RNSPoly, RNSRing


class MatrixEncoder:
    """Encode matrices as rounded Gaussian coefficients; decode noisy RNS values.

    Slots have W first, followed by the other variables in ring order. Thus the
    default ring accepts (phi(p), n, n) matrices. X/Y roots are exp(2*pi*i*5**j/(4n));
    W roots are exp(2*pi*i*2**ell/p), with 2 generating the units modulo 3**k.
    Floating transforms use complex128. Stored polynomial limbs remain int32.
    """

    def __init__(self, ring: RNSRing, *, scale: float = 2**40, device: torch.device | str = "cpu"):
        if not isinstance(ring, RNSRing):
            raise TypeError("ring must be an RNSRing")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ValueError("scale must be finite and positive")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        if not ring.axes or len({axis.name for axis in ring.axes} - {"X", "Y", "W"}):
            raise ValueError("encoding requires X/Y power-of-i axes and a W cyclotomic axis")
        self.ring, self.scale, self.device = ring, float(scale), resolve_device(device)
        self._W_dim = next((dim for dim, axis in enumerate(ring.axes) if axis.name == "W"), None)
        if self._W_dim is None:
            raise ValueError("encoding requires a W cyclotomic axis")
        W_axis = ring.axes[self._W_dim]
        self.p = W_axis.cyclotomic_order
        if self.p is None or odd_prime_power(self.p)[0] != 3:
            raise ValueError("encoding currently supports only power-of-three cyclotomics")
        self._stride = self.p // 3
        self._W_indices = torch.tensor(
            [pow(2, slot, self.p) for slot in range(W_axis.degree)], device=self.device
        )
        self._twists = {}
        self._permutations = {}
        for dim, axis in enumerate(ring.axes):
            if dim == self._W_dim:
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
        values = torch.fft.ifft(values, n=self.p, dim=self._W_dim) * self.p
        return values.index_select(self._W_dim, self._W_indices).permute(self._slot_dims)

    def _interpolate(self, slots: torch.Tensor) -> torch.Tensor:
        values = slots.permute(self._coefficient_dims)
        padded_shape = list(self.ring.shape[:-1])
        padded_shape[self._W_dim] = self.p
        padded = torch.zeros(padded_shape, dtype=torch.complex128, device=self.device)
        padded.index_copy_(self._W_dim, self._W_indices, values)
        projected = torch.fft.fft(padded, dim=self._W_dim) / self.p
        # The length-p interpolation agrees at every primitive root. Its
        # degree-<p representative must still be reduced to degree <phi(p):
        # W**(2*stride+r) = -W**r-W**(stride+r), for 0 <= r < stride.
        tail = projected.narrow(self._W_dim, 2 * self._stride, self._stride)
        values = projected.narrow(self._W_dim, 0, 2 * self._stride) - torch.cat(
            (tail, tail), dim=self._W_dim
        )
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
