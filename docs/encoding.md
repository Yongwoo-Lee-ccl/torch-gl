# Scaled complex matrix encoding

`MatrixEncoder` implements Section 3.1, Equation (3), of the Gentry–Lee paper:
`encode(T) = round(scale * sigma^-1(T))` and `decode(m) = sigma(m)/scale`.
Rounding acts separately on real and imaginary coefficient components, with
ties to even. There is no conjugate-symmetry constraint: the coefficient ring
is Gaussian, and every primitive W root carries an independent complex matrix.
This implements the complex encoding, not the integer/BGV encoding of Section 4.1.
The default scale is `2**40`.

## Layout and transforms

For the default X/Y/W ring, matrices have shape `(phi(p), n, n)`, with entries
`matrices[ell,j,k] = m(zeta_j,zeta_k,eta_ell)/scale`. The roots are
`zeta_j = exp(2*pi*i*5**j/(4*n))` and `eta_ell = exp(2*pi*i*2**ell/p)`.
The implementation currently supports `p=3**exponent`, for which 2 generates
the units modulo p. X and Y may have different power-of-two degrees. For a
subset or reordered ring, W is the first slot dimension and the other axes
follow their ring order. Coefficients retain the ring's usual axis order.

The complex transforms are Torch FFTs with cached twists and permutations on
the encoder's device. X/Y evaluation twists by `exp(2*pi*i*r/(4*n))`, performs
a positive-exponent DFT, then permutes into powers-of-five order. W evaluation
pads coefficients to length p and selects primitive frequencies in powers-of-two
order. W interpolation sets nonprimitive evaluations to zero, computes the
inverse DFT, then subtracts the high stride block from each lower stride block.
This reduces modulo `1+W**(p/3)+W**(2*p/3)` and recovers the canonical coefficients.
No dense Vandermonde inversion is used in the implementation.

The zero-filled length-p inverse DFT alone is not the canonical inverse. If its
coefficients are `d[0:p]`, the canonical coefficients are
`c[r]=d[r]-d[2*s+r]` and `c[s+r]=d[s+r]-d[2*s+r]`, for `s=p/3` and `0<=r<s`.
This follows from `W**(2*s+r)=-W**r-W**(s+r)` modulo Phi_p. The resulting
degree-<phi(p) polynomial has the supplied values at all primitive roots; uniqueness
of polynomial interpolation makes it the inverse canonical embedding. X/Y use
the inverse DFT with the powers-of-five permutation undone and the twist removed.

## Large coefficients and RNS centering

Interpolation and scaling use complex128/float64. Each real component is rounded
to a Python integer before reducing modulo each prime into an int32 limb.
Encoding rejects coefficients outside the centered aggregate interval
`[-Q//2,Q//2]`, where `Q=product(moduli)`. A single-limb bound is never used.

Decoding always calls exact Python-integer CRT reconstruction and centers modulo
Q before converting to float64 and evaluating. A negative coefficient and a
positive coefficient larger than `q0/2` therefore remain distinct. Coefficients
larger than int64 are supported. Double precision cannot retain individual units
at those magnitudes; Python integer storage prevents overflow, not floating
interpolation or evaluation error. The scale belongs to the encoder; ciphertexts
do not carry scale metadata or implement rescaling.

The Python-integer bridge runs on CPU. CUDA encoders transfer coefficients to
CPU for rounding/CRT and return the resulting limbs/slots to CUDA. This is a
correctness-oriented boundary, not a GPU performance claim.

The existing `RNSPoly.centered_coefficients()` is a separate fast helper for
small coefficients. It checks every limb and raises on inconsistent lifts;
encoding and decoding do not use it.

## Verification

`pytest tests/test_encoding.py` checks independent direct evaluation at the
paper's roots, interpolation of known coefficients, configurable axis orders,
aggregate-modulus bounds, coefficients near `2**40` and `2**70` of both signs,
and encode/encrypt/decrypt/decode with actual Gaussian noise at multiple scales.
It also solves the full dense canonical evaluation matrix for arbitrary complex
slots and compares the resulting coefficients to the FFT interpolation, including
`p=729`. This reference inversion is confined to the tests.
The frozen `tests/data/sage_encoding.json` fixture additionally checks the
canonical inverse and direct evaluation against SageMath's 256-bit dense
arithmetic, including exact rounded coefficients and a noisy encrypted message.
Its generator is standalone and is never invoked by tests or CI.
Plaintext-only tests multiply two independently encoded polynomials and check
that decoding at the product scale matches elementwise slot multiplication.
They cover equal and different input scales, reordered and smaller rings, and
`p=729`, with bounds for rounding error and aggregate-modulus wraparound.
The existing naive `10000*X` tests and saved Sage multiplication vectors are unchanged.
The engine's current full-ring binary key is unchanged; the paper's smaller-ring
secret embedding remains a separate feature, as documented in the encryption notes.
