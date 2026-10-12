# Frozen SageMath known-answer vectors

`sage_multiplication.json` was generated once using SageMath 10.6.
It contains 15 multiplication cases with deterministic seeds and explicit
quotient relations. Coefficients are flattened in C order, with the final
axis holding `(real, imaginary)` pairs. `shape` reconstructs each tensor.

The oracle multiplies multivariate polynomials over `GF(q)` and reduces modulo
the ideal containing `i²+1` and the configured X/Y/W relations. It imports neither
PyTorch nor this package. Pytest only reads the stored JSON and compares results.
Neither tests nor CI invoke SageMath or regenerate this file.

Run the standalone generator only when explicitly maintaining the vectors,
then review and commit the resulting diff. Arithmetic tests are not evidence
of cryptographic security.
