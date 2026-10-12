"""Small, deterministic integer utilities (no floating-point arithmetic)."""

from functools import lru_cache

MAX_MODULUS = (1 << 30) - 1


def is_prime(n: int) -> bool:
    """Deterministic Miller–Rabin for unsigned 64-bit integers."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    odd_part, two_adic_order = n - 1, 0
    while odd_part % 2 == 0:
        odd_part //= 2
        two_adic_order += 1
    for witness in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if witness % n == 0:
            continue
        residue = pow(witness, odd_part, n)
        if residue in (1, n - 1):
            continue
        for _ in range(two_adic_order - 1):
            residue = residue * residue % n
            if residue == n - 1:
                break
        else:
            return False
    return True


@lru_cache(maxsize=128)
def validate_modulus(q: int) -> None:
    if type(q) is not int or not 3 <= q <= MAX_MODULUS or not is_prime(q):
        raise ValueError(f"q must be an odd prime between 3 and {MAX_MODULUS}")


def is_power_of_two(n: int) -> bool:
    return type(n) is int and n > 0 and n & (n - 1) == 0


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def smooth_factors(n: int) -> tuple[int, ...]:
    """Factor a positive 2/3-smooth transform length."""
    if type(n) is not int or n < 1:
        raise ValueError("NTT length must be a positive integer")
    factors = []
    for prime in (2, 3):
        while n % prime == 0:
            factors.append(prime)
            n //= prime
    if n != 1:
        raise ValueError("NTT length must have only factors 2 and 3")
    return tuple(factors)


def odd_prime_power(p: int) -> tuple[int, int]:
    """Return (prime base, exponent), rejecting non odd prime powers."""
    if type(p) is not int or p < 3 or p % 2 == 0:
        raise ValueError("p must be an odd prime power")
    if is_prime(p):
        return p, 1
    # Test perfect powers before primality, avoiding trial division to sqrt(p).
    for exponent in range(2, p.bit_length()):
        lower_bound, upper_bound = 2, 1 << ((p.bit_length() + exponent - 1) // exponent)
        while lower_bound <= upper_bound:
            base = (lower_bound + upper_bound) // 2
            value = base**exponent
            if value == p:
                if is_prime(base):
                    return base, exponent
                break
            if value < p:
                lower_bound = base + 1
            else:
                upper_bound = base - 1
    raise ValueError("p must be an odd prime power")


@lru_cache(maxsize=128)
def root_of_unity(q: int, length: int) -> int:
    validate_modulus(q)
    factors = smooth_factors(length)
    if (q - 1) % length:
        raise ValueError(f"NTT length {length} must divide q - 1 ({q - 1})")
    if length == 1:
        return 1
    for candidate in range(2, q):
        root = pow(candidate, (q - 1) // length, q)
        if all(pow(root, length // factor, q) != 1 for factor in set(factors)):
            return root
    raise ValueError("could not find a root of unity")  # pragma: no cover
