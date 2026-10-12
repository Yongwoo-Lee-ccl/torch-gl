# Frozen SageMath known-answer vectors

`sage_multiplication.json` was generated once using SageMath 10.6.
It contains 15 multiplication cases with deterministic seeds and explicit
quotient relations. Coefficients are flattened in C order, with the final
axis holding `(real, imaginary)` pairs. `shape` reconstructs each tensor.

The oracle multiplies multivariate polynomials over `GF(q)` and reduces modulo
the ideal containing `i²+1` and the configured X/Y/W relations. It imports neither
PyTorch nor this package. Pytest only reads the stored JSON and compares results.
Neither tests nor CI invoke SageMath or regenerate this file.

The file is preserved unchanged. Thirteen cases test supported multiplication.
The two historical large-modulus cases now verify rejection of moduli above the
deliberate 30-bit single-limb limit; RNS composition belongs at a higher level.

Run the standalone generator only when explicitly maintaining the vectors,
then review and commit the resulting diff. Arithmetic tests are not evidence
of cryptographic security.

## Canonical embedding

`sage_encoding.json` contains six frozen canonical-embedding vectors, generated
once by `scripts/generate_sage_encoding_kats.py` with SageMath 10.6. The oracle
uses 256-bit complex arithmetic, directly constructs the evaluation matrix at
roots of `X^n-i`, `Y^n-i`, and Sage's `cyclotomic_polynomial(p)`, and solves the
dense linear system for its inverse. It uses no FFTs and imports neither Torch
nor this package. The generator checks root relations and inverse residuals.

Cases cover n=1,2,4,8 and p=3,9,27,81, the full ring, the smaller X/W ring,
a reordered ring, and W alone. Slot inputs are exact rational Gaussian values
with the recorded common denominator. Inverse coefficients and evaluated slots
are saved as high-precision decimal pairs. Rounded coefficients use scale 2^40
and are saved as exact JSON integers. Independent decode inputs contain positive
and negative coefficients above the first-prime range.

`tests/test_sage_encoding_kats.py` checks inverse coefficients, exact rounded
integers and every RNS limb, decoding against direct Sage evaluation, and noisy
encryption/decryption of a Sage-encoded message. Tests and CI never run Sage or
regenerate either fixture. To deliberately replace these vectors, run and review:

```sh
sage -python scripts/generate_sage_encoding_kats.py
```
