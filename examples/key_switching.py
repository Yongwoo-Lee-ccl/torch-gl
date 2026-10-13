"""Toy key-switching examples: python examples/key_switching.py --case all --k 2."""

import argparse
import tempfile

import torch

from torch_gl import (
    Ciphertext,
    Engine,
    KeySwitcher,
    KeySwitchParameters,
    MatrixEncoder,
    RNSRing,
    create_secret_key,
)


def report_switching_error(label, before, after):
    # Only their small difference is lifted this way, never the encoded message.
    error = (after - before).centered_coefficients()
    maximum = int(error.abs().max())
    assert maximum < 20000  # A check for these fixed toy examples, not a general noise bound.
    print(f"{label}: maximum coefficient change = {maximum}")


def main():
    cases = (
        "all",
        "parameters",
        "independent",
        "small",
        "big",
        "hadamard",
        "matrix",
        "file-backed",
        "small-ring",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--case", choices=cases, default="all")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("integer arithmetic supports CPU and CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA hardware is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    ring = RNSRing.for_parameters(n=4, p=9, limb_count=4)
    selected = set(cases[1:]) if args.case == "all" else {args.case}
    if "parameters" in selected:
        for k in (1, 2, 3, 4):
            parameters = KeySwitchParameters(ring, k=k)
            print(f"k={k}: dnum={parameters.dnum}, digit groups={parameters.groups}")
        if args.case == "parameters":
            return

    # Each key has its own RNG, independent of ciphertext and evaluation-key sampling.
    source_secret = create_secret_key(
        ring, generator=torch.Generator(device=device).manual_seed(11)
    )
    target_secret = create_secret_key(
        ring, generator=torch.Generator(device=device).manual_seed(12)
    )
    source = Engine(ring, device=device, generator=torch.Generator(device=device).manual_seed(101))
    target = Engine(ring, device=device, generator=torch.Generator(device=device).manual_seed(102))
    parameters = KeySwitchParameters(ring, k=args.k)
    # Explicit key primes are optional; here they match the automatic choice above.
    switcher = KeySwitcher(
        ring,
        k=args.k,
        key_moduli=parameters.key_moduli,
        device=device,
        generator=torch.Generator(device=device).manual_seed(201),
    )
    message = 10000 * ring.generator("X", device=device)
    original = source.encrypt(message, source_secret)
    original_plaintext = source.decrypt(original, source_secret)

    if {"independent", "small"} & selected:
        key = switcher.generate_key(source_secret, target_secret, kind="small")
        if "independent" in selected:
            switched = switcher.switch_ciphertext(original, key)
            report_switching_error(
                "Independent s1->s2", original_plaintext, target.decrypt(switched, target_secret)
            )
        if "small" in selected:
            # switch_small encrypts a*s1; the original b is added by switch_ciphertext.
            switched_component = switcher.switch_small(original.a, key)
            expected = source.decrypt(
                Ciphertext(ring.zero(device=device), original.a), source_secret
            )
            report_switching_error(
                "Raw SwitchSmall", expected, target.decrypt(switched_component, target_secret)
            )

    if "big" in selected:
        # A full-ring source can depend on Y; the destination remains an X/W key.
        source_polynomial = ring.generator("Y", device=device) + ring.generator("W", device=device)
        key = switcher.generate_key(source_polynomial, target_secret, kind="big")
        switched_component = switcher.switch_big(original.a, key)
        report_switching_error(
            "Raw SwitchBig",
            original.a * source_polynomial,
            target.decrypt(switched_component, target_secret),
        )

    if {"hadamard", "matrix", "file-backed"} & selected:
        encoder = MatrixEncoder(ring, device=device)  # Default scale=2**40.
        product_decoder = MatrixEncoder(ring, scale=encoder.scale**2, device=device)
        data_generator = torch.Generator(device=device).manual_seed(301)
        left = torch.randn(
            encoder.slot_shape, dtype=torch.complex128, device=device, generator=data_generator
        )
        right = torch.randn(
            encoder.slot_shape, dtype=torch.complex128, device=device, generator=data_generator
        )
        left_ct = source.encrypt(encoder.encode(left), source_secret)
        right_ct = source.encrypt(encoder.encode(right), source_secret)

        if "hadamard" in selected:
            product = source.multiply(left_ct, right_ct)
            key = switcher.generate_hadamard_key(source_secret)
            switched = switcher.relinearize(product, key)
            recovered = source.decrypt(switched, source_secret)
            report_switching_error(
                "Hadamard SwitchSmall", source.decrypt(product, source_secret), recovered
            )
            torch.testing.assert_close(
                product_decoder.decode(recovered), left * right, atol=2e-8, rtol=2e-8
            )
            print("Hadamard decoding: slotwise left*right checked at scale=2**80")

        if {"matrix", "file-backed"} & selected:
            matrix_product = source.matrix_multiply(left_ct, right_ct)
            before = source.decrypt(matrix_product, source_secret)
            if "matrix" in selected:
                keys = switcher.generate_matrix_keys(source_secret)
                switched = switcher.relinearize_matrix(matrix_product, keys)
                recovered = source.decrypt(switched, source_secret)
                report_switching_error("Two SwitchBig calls", before, recovered)
                torch.testing.assert_close(
                    product_decoder.decode(recovered),
                    left @ right.transpose(-1, -2),
                    atol=2e-8,
                    rtol=2e-8,
                )
                print("Matrix decoding: W-batched left@right.T checked at scale=2**80")
            if "file-backed" in selected:
                resident_switcher = KeySwitcher(
                    ring,
                    k=args.k,
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(501),
                )
                file_switcher = KeySwitcher(
                    ring,
                    k=args.k,
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(501),
                )
                resident_keys = resident_switcher.generate_matrix_keys(source_secret)
                resident_result = resident_switcher.relinearize_matrix(
                    matrix_product, resident_keys
                )
                # Files must exist while the file-backed keys are being used.
                with tempfile.TemporaryDirectory(prefix="torch-gl-example-keys-") as directory:
                    file_keys = file_switcher.generate_matrix_keys(
                        source_secret, storage_directory=directory
                    )
                    file_result = file_switcher.relinearize_matrix(matrix_product, file_keys)
                    assert file_result == resident_result
                    recovered = source.decrypt(file_result, source_secret)
                    report_switching_error("File-backed matrix keys", before, recovered)
                    torch.testing.assert_close(
                        product_decoder.decode(recovered),
                        left @ right.transpose(-1, -2),
                        atol=2e-8,
                        rtol=2e-8,
                    )
                print("Resident/file-backed ciphertexts: exactly equal; temporary files removed")

    if "small-ring" in selected:
        small_ring = RNSRing.multivariate_poly(ring.moduli, n=4, p=9, variables=("X", "W"))
        small_engine = Engine(
            small_ring, device=device, generator=torch.Generator(device=device).manual_seed(601)
        )
        small_switcher = KeySwitcher(
            small_ring,
            k=args.k,
            device=device,
            generator=torch.Generator(device=device).manual_seed(602),
        )
        key = small_switcher.generate_key(source_secret, target_secret)
        ciphertext = small_engine.encrypt(10000, source_secret)
        switched = small_switcher.switch_ciphertext(ciphertext, key)
        report_switching_error(
            "X/W ring without Y",
            small_engine.decrypt(ciphertext, source_secret),
            small_engine.decrypt(switched, target_secret),
        )


if __name__ == "__main__":
    main()
