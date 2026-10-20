# Small machine learning examples

Install the repository with `python -m pip install -e '.[dev]'`, then run:

```sh
python examples/linear_regression.py
python examples/mlp.py
```

Both scripts accept `--seed 42` and `--device cuda`, use float64 plaintext
references, pack eight feature rows into two 4x4 matrices, and assert a maximum
absolute approximation error below 2e-4. These tiny parameters demonstrate
correctness; they are not security parameter recommendations.

`linear_regression.py` fits a public synthetic training set with `torch.linalg.lstsq`.
Three input features and a constant bias column are encrypted using a public
key. The server computes one matrix multiplication against public learned
weights, consuming one modulus limb and requiring no key switching. The client
decrypts predictions in the first output column.

`mlp.py` evaluates a 4 -> 4 -> 4 network with public toy weights and an explicit
square activation. Its ciphertext-plaintext matrix multiplications consume one
level each; the square uses Hadamard relinearization at a lower level with the
original maximum-level key. The reference computes the same square network.
The example demonstrates encrypted inference, not encrypted training.

Both scripts keep key generation and final decryption separate from the
server's `Evaluator`, which has public evaluation material and no secret field.
No external datasets, model downloads or additional packages are needed.
