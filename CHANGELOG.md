# Changelog

## 0.1.0

- Add floor-style RNS rescaling and component-wise engine rescaling, dropping the last prime.
- Add secret-key limb projection and reduced-scale encrypted-product tests/examples.

- Store polynomial/RNS coefficients, keys, ciphertexts, noise, and NTT tables in int32.
- Keep transforms in Montgomery form and widen only local products to int64.

- Add RNS polynomial blocks, compatible prime selection, and exact CRT export.
- Add a binary-secret engine with rounded Gaussian noise and `(b,a)` ciphertexts.
- Test encryption/decryption of `10000`, `10000*X`, `10000*W`, and `10000*Y`.

- Establish a src package with `MultivariatePoly` and configurable quotient rings.
- Implement mixed-radix 2/3 NTTs, compact power-of-three cyclotomic transforms,
  and twisted Gaussian-coefficient transforms for X/Y.
- Use simple single RNS limbs with prime moduli below `2^30`.
- Cache device-side transform tables, batch branches and operands, and reuse plans.
- Expose tagged single-limb Montgomery conversions and multiplication with `R=2^31`.
- Add frozen SageMath multiplication KATs, tests, examples, and paper correspondence notes.
