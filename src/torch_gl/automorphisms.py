"""Exact GL substitutions, conjugation, and extension-axis packing.

These act on coefficients, never floating-point slots. A transformed ciphertext
decrypts under the transformed secret; the Evaluator performs the required switch.
"""

import math
from functools import lru_cache

import torch

from .multivariate_poly import MultivariatePoly
from .rings import QuotientAxis
from .rns import RNSPoly, RNSRing


@lru_cache(maxsize=256)
def _monomial_map(axis, exponent):
    entries = []
    if axis.cyclotomic_order:
        p = axis.cyclotomic_order
        stride = p - axis.degree
        for source in range(axis.degree):
            power = source * exponent % p
            if power < axis.degree:
                entries.append((source, power, 1, 0))
            else:
                entries.extend(
                    (source, power - axis.degree + offset, -1, 0)
                    for offset in range(0, axis.degree, stride)
                )
    elif axis == QuotientAxis.power_of_i(axis.name, axis.degree):
        units = ((1, 0), (0, 1), (-1, 0), (0, -1))
        for source in range(axis.degree):
            quotient, remainder = divmod(source * exponent, axis.degree)
            entries.append((source, remainder, *units[quotient % 4]))
    else:
        raise ValueError("substitution requires GL power-of-i or cyclotomic axes")
    return tuple(entries)


def automorphism(polynomial: RNSPoly, powers=None, *, conjugate: bool = False) -> RNSPoly:
    """Substitute each variable T by T**powers[T], optionally i by -i.

    Unspecified powers are 1 (or -1 when conjugating). Gaussian-axis powers
    must be 1 modulo 4, or -1 modulo 4 with coefficient conjugation. W powers
    must be units modulo p. Thus conjugate=True alone is canonical conjugation
    and decodes to element-wise complex conjugation in the original slot order.
    """
    if not isinstance(polynomial, RNSPoly):
        raise TypeError("expected an RNSPoly")
    if type(conjugate) is not bool:
        raise TypeError("conjugate must be a bool")
    powers = {} if powers is None else dict(powers)
    names = {axis.name for axis in polynomial.ring.axes}
    if powers.keys() - names:
        raise ValueError("powers contains a variable absent from the ring")
    exponents = []
    for axis in polynomial.ring.axes:
        exponent = powers.get(axis.name, -1 if conjugate else 1)
        if type(exponent) is not int:
            raise TypeError("substitution exponents must be integers")
        if axis.cyclotomic_order:
            if math.gcd(exponent, axis.cyclotomic_order) != 1:
                raise ValueError("cyclotomic exponents must be units modulo their order")
        elif exponent % 4 != (-1 if conjugate else 1) % 4:
            raise ValueError("Gaussian exponents must respect i and the power-of-i relation")
        exponents.append(exponent)
    limbs = []
    for limb in polynomial.limbs:
        values = limb._coefficients.to(torch.int64)
        if conjugate:
            values[..., 1].neg_()
        for dim, (axis, exponent) in enumerate(zip(polynomial.ring.axes, exponents, strict=True)):
            if exponent == 1:
                continue
            entries = _monomial_map(axis, exponent)
            source, target, real, imag = (
                torch.tensor(column, dtype=torch.int64, device=polynomial.device)
                for column in zip(*entries, strict=True)
            )
            selected = values.index_select(dim, source)
            shape = [1] * (values.ndim - 1)
            shape[dim] = len(entries)
            real, imag = real.reshape(shape), imag.reshape(shape)
            mapped = torch.stack(
                (
                    selected[..., 0] * real - selected[..., 1] * imag,
                    selected[..., 0] * imag + selected[..., 1] * real,
                ),
                -1,
            )
            values = (
                torch.zeros_like(values).index_add_(dim, target, mapped).remainder_(limb.ring.q)
            )
        limbs.append(
            MultivariatePoly._from_owned(limb.ring, values.remainder(limb.ring.q).to(torch.int32))
        )
    return RNSPoly(polynomial.ring, tuple(limbs))


def swap_axes(polynomial: RNSPoly, first: str, second: str) -> RNSPoly:
    """Swap two identical quotient axes, e.g. X/Y or W/V."""
    if not isinstance(polynomial, RNSPoly):
        raise TypeError("expected an RNSPoly")
    names = tuple(axis.name for axis in polynomial.ring.axes)
    if first not in names or second not in names:
        raise ValueError("both swapped axes must occur in the ring")
    left, right = names.index(first), names.index(second)
    if polynomial.ring.axes[left].modulus != polynomial.ring.axes[right].modulus:
        raise ValueError("swapped axes must have identical quotient relations")
    return RNSPoly(
        polynomial.ring,
        tuple(
            MultivariatePoly._from_owned(
                limb.ring, limb._coefficients.transpose(left, right).contiguous()
            )
            for limb in polynomial.limbs
        ),
    )


def conjugate(polynomial: RNSPoly) -> RNSPoly:
    return automorphism(polynomial, conjugate=True)


def hermitian_transpose(polynomial: RNSPoly, first="X", second="Y") -> RNSPoly:
    """Decode to the Hermitian transpose of each matrix (or W/V matrix)."""
    return swap_axes(conjugate(polynomial), first, second)


def embed(polynomial: RNSPoly, target: RNSRing) -> RNSPoly:
    """Embed a polynomial as a constant in the target's additional axes."""
    if not isinstance(polynomial, RNSPoly) or not isinstance(target, RNSRing):
        raise TypeError("expected an RNSPoly and RNSRing")
    source_axes = {axis.name: axis for axis in polynomial.ring.axes}
    target_axes = {axis.name: axis for axis in target.axes}
    if (
        polynomial.ring.moduli != target.moduli
        or source_axes.keys() - target_axes.keys()
        or any(target_axes[name] != axis for name, axis in source_axes.items())
    ):
        raise ValueError("target must extend the source quotient axes at the same level")
    order = [
        next(i for i, axis in enumerate(polynomial.ring.axes) if axis.name == name)
        for name in target_axes
        if name in source_axes
    ]
    limbs = []
    for limb, ring in zip(polynomial.limbs, target.rings, strict=True):
        values = torch.zeros(ring.shape, dtype=torch.int32, device=polynomial.device)
        index = tuple(slice(None) if axis.name in source_axes else 0 for axis in target.axes)
        values[index] = limb._coefficients.permute(*order, len(order))
        limbs.append(MultivariatePoly._from_owned(ring, values))
    return RNSPoly(target, tuple(limbs))


def unpack(polynomial: RNSPoly, axis_name: str) -> tuple[RNSPoly, ...]:
    """Extract coefficients of an extension axis, retaining all other axes."""
    if not isinstance(polynomial, RNSPoly):
        raise TypeError("expected an RNSPoly")
    names = tuple(axis.name for axis in polynomial.ring.axes)
    if axis_name not in names:
        raise ValueError("axis is absent from the polynomial ring")
    dim = names.index(axis_name)
    ring = RNSRing(
        polynomial.ring.moduli,
        tuple(axis for axis in polynomial.ring.axes if axis.name != axis_name),
    )
    return tuple(
        RNSPoly(
            ring,
            tuple(
                MultivariatePoly._from_owned(small, limb._coefficients.select(dim, index).clone())
                for small, limb in zip(ring.rings, polynomial.limbs, strict=True)
            ),
        )
        for index in range(polynomial.ring.axes[dim].degree)
    )


def pack(polynomials, target: RNSRing, axis_name: str) -> RNSPoly:
    """Pack independent small-ring polynomials into extension coefficients."""
    polynomials = tuple(polynomials)
    if any(not isinstance(polynomial, RNSPoly) for polynomial in polynomials):
        raise TypeError("packing inputs must be RNSPoly values")
    names = tuple(axis.name for axis in target.axes)
    if axis_name not in names:
        raise ValueError("packing axis is absent from the target ring")
    dim = names.index(axis_name)
    if not polynomials or len(polynomials) > target.axes[dim].degree:
        raise ValueError("packing needs between one and the extension degree polynomials")
    expected = RNSRing(target.moduli, tuple(axis for axis in target.axes if axis.name != axis_name))
    if any(p.ring != expected or p.device != polynomials[0].device for p in polynomials):
        raise ValueError("all inputs must have the target's remaining axes, level and device")
    limbs = []
    for index, ring in enumerate(target.rings):
        values = torch.zeros(ring.shape, dtype=torch.int32, device=polynomials[0].device)
        for coefficient, polynomial in enumerate(polynomials):
            values.select(dim, coefficient).copy_(polynomial.limbs[index]._coefficients)
        limbs.append(MultivariatePoly._from_owned(ring, values))
    return RNSPoly(target, tuple(limbs))
