"""Bivariate Lagrange interpolation over a prime field.

Used by the integer-DP + CRT pathway to recover the Tutte polynomial
coefficients from point-value evaluations modulo a prime, then later
CRT-combine across primes for exact integer coefficients.

Algorithm: two passes of 1D Lagrange.
1. For each fixed `y = y_j`, interpolate `(x_i, T(x_i, y_j))` over x
   → polynomial-in-x at fixed y_j, with coefficients mod p. Call this
   `Q_j(x) = Σ_a coef_aj · x^a`.
2. For each `x^a`, interpolate `(y_j, coef_aj)` over y → polynomial in
   y giving the coefficient of `x^a` in T(x, y) as a function of y_j.
   The final 2D coefficient `T_ab` is the coefficient of `y^b` in this
   polynomial.

All arithmetic is modular. No floats. Inputs must be distinct.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def _modinv(a: int, p: int) -> int:
    """Modular inverse of a mod p, using Fermat's little theorem (p prime)."""
    return pow(a % p, p - 2, p)


def lagrange_1d_mod(
    points: List[Tuple[int, int]],
    p: int,
) -> List[int]:
    """1D Lagrange interpolation modulo prime p.

    `points` = [(x_i, y_i)] with distinct x_i. Returns the coefficient
    list `[c_0, c_1, ..., c_d]` of the unique polynomial of degree
    ≤ `len(points) - 1` such that `P(x_i) = y_i mod p`.

    Implementation: Newton's divided differences are nicer but more
    code; we just expand `Σ y_i · L_i(x)` where `L_i(x) = ∏_{j≠i} (x − x_j) / (x_i − x_j)`.
    O(n^3) which is fine for our n ≤ 200.
    """
    n = len(points)
    if n == 0:
        return []
    # Result coefficients indexed by power; size n.
    result = [0] * n

    for i, (x_i, y_i) in enumerate(points):
        # Compute the numerator polynomial Π_{j≠i} (x − x_j), expanded
        # as a list of coefficients (low to high).
        num: List[int] = [1]  # constant polynomial 1
        denom = 1
        for j, (x_j, _) in enumerate(points):
            if j == i:
                continue
            # Multiply num by (x − x_j): new coeff k = -x_j * num[k] + num[k-1]
            new = [0] * (len(num) + 1)
            for k, c in enumerate(num):
                new[k] = (new[k] - x_j * c) % p
                new[k + 1] = (new[k + 1] + c) % p
            num = new
            denom = (denom * (x_i - x_j)) % p

        denom_inv = _modinv(denom, p)
        scale = (y_i * denom_inv) % p

        for k, c in enumerate(num):
            result[k] = (result[k] + scale * c) % p

    return result


def bivariate_lagrange_interpolate_mod(
    x_values: List[int],
    y_values: List[int],
    grid_evaluations: List[List[int]],
    p: int,
) -> Dict[Tuple[int, int], int]:
    """Bivariate Lagrange interpolation modulo prime p.

    Inputs:
    - `x_values`: list of `n_x` distinct x-coordinates.
    - `y_values`: list of `n_y` distinct y-coordinates.
    - `grid_evaluations[i][j]` = T(x_values[i], y_values[j]) mod p.
    - `p`: prime modulus.

    Returns: `{(a, b) → coef_ab mod p}` such that
    `T(x, y) mod p = Σ coef_ab · x^a · y^b`. Degrees are bounded by
    `len(x_values) − 1` and `len(y_values) − 1` respectively.

    Two-pass:
    1. For each y_j, 1D-interpolate (x_i, T_ij) → poly-in-x at y_j;
       collect coefficient-of-x^a for each j as a list.
    2. For each x^a row, 1D-interpolate (y_j, coef_aj) → final
       2D coefficients (a, b).
    """
    n_x = len(x_values)
    n_y = len(y_values)
    assert len(grid_evaluations) == n_x, "grid_evaluations must have n_x rows"
    assert all(len(row) == n_y for row in grid_evaluations), (
        "grid_evaluations rows must have n_y entries"
    )

    # Pass 1: for each y_j, get poly-in-x at y_j.
    # coefs_in_x[a][j] = coefficient of x^a in Q_j(x), where
    # Q_j(x) = T(x, y_j) mod p.
    coefs_in_x: List[List[int]] = [[0] * n_y for _ in range(n_x)]
    for j in range(n_y):
        pts_x = [(x_values[i], grid_evaluations[i][j]) for i in range(n_x)]
        poly_x_at_yj = lagrange_1d_mod(pts_x, p)
        for a, c in enumerate(poly_x_at_yj):
            coefs_in_x[a][j] = c

    # Pass 2: for each x^a, interpolate the y-axis to get final coefs.
    result: Dict[Tuple[int, int], int] = {}
    for a in range(n_x):
        pts_y = [(y_values[j], coefs_in_x[a][j]) for j in range(n_y)]
        poly_y_for_x_a = lagrange_1d_mod(pts_y, p)
        for b, c in enumerate(poly_y_for_x_a):
            if c != 0:
                result[(a, b)] = c
    return result


def crt_combine_coeff_dicts(
    coeff_dicts_per_prime: List[Dict[Tuple[int, int], int]],
    primes: List[int],
) -> Dict[Tuple[int, int], int]:
    """CRT-combine per-prime coefficient dicts into exact integer coefficients.

    Each `coeff_dicts_per_prime[k]` maps `(a, b) → coef_ab mod primes[k]`.
    Returns a dict `(a, b) → exact integer coef_ab` (signed; the convention
    matches `_crt_multi` in `tutte/validation.py`).

    All input dicts should cover the SAME set of `(a, b)` keys (we union
    them and assume `0` for any missing entry — which CRT will treat as
    "0 mod p", meaning the coefficient really is 0 in that prime field).
    """
    from ..validation import _crt_multi

    # Union of all keys across all dicts.
    all_keys = set()
    for d in coeff_dicts_per_prime:
        all_keys.update(d.keys())

    result: Dict[Tuple[int, int], int] = {}
    for key in all_keys:
        residues = [d.get(key, 0) for d in coeff_dicts_per_prime]
        val = _crt_multi(residues, primes)
        if val != 0:
            result[key] = val
    return result
