"""Coefficient ring changes used by GL bootstrapping, without key switching.

R' = R[Y]/(Y^n-i); R^aux = R[V]/Phi_p(V). The secret s(X,W)
does not depend on either packing variable, so extraction commutes with decryption.
"""

from .automorphisms import pack, unpack
from .ciphertext import Ciphertext
from .rns import RNSPoly, RNSRing


def split(polynomial: RNSPoly, axis: str = "Y") -> tuple[RNSPoly, ...]:
    """R' -> n R polynomials (Y coefficients); also supports V in R^aux."""
    if axis not in {"Y", "V"}:
        raise ValueError("split requires the secret-independent extension Y or V")
    return unpack(polynomial, axis)


def merge(polynomials, target: RNSRing, axis: str | None = None) -> RNSPoly:
    """R polynomials -> R' (Y) or R^aux (V), padding missing coefficients by 0."""
    if axis is None:
        names = {a.name for a in target.axes}
        axis = "Y" if "Y" in names else "V" if "V" in names else ""
    if axis not in {"Y", "V"}:
        raise ValueError("merge target must have an extension Y or V")
    return pack(polynomials, target, axis)


def split_ciphertext(ciphertext: Ciphertext, axis: str = "Y") -> tuple[Ciphertext, ...]:
    """Extract both public components; requires an s(X,W) encryption key."""
    if not isinstance(ciphertext, Ciphertext):
        raise TypeError("expected a Ciphertext")
    return tuple(
        Ciphertext(b, a)
        for b, a in zip(split(ciphertext.b, axis), split(ciphertext.a, axis), strict=True)
    )


def merge_ciphertexts(ciphertexts, target: RNSRing, axis: str | None = None) -> Ciphertext:
    """Pack ciphertexts under the same s(X,W), level and device into Y or V."""
    ciphertexts = tuple(ciphertexts)
    if not ciphertexts or any(not isinstance(ct, Ciphertext) for ct in ciphertexts):
        raise TypeError("expected a nonempty sequence of Ciphertext values")
    return Ciphertext(
        merge((ct.b for ct in ciphertexts), target, axis),
        merge((ct.a for ct in ciphertexts), target, axis),
    )
