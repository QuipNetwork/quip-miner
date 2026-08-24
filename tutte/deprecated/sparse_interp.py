"""Sparse polynomial interpolation modulo a prime.

For polynomials with few nonzero coefficients (sparse), recovering them
from grid-of-evaluations via standard Lagrange wastes work. Sparse
interpolation methods recover an s-sparse polynomial from O(s) values.

This module provides:

- `bm_recover_recurrence(seq, p)` — Berlekamp-Massey: find shortest
  linear recurrence satisfied by `seq`. Returns the connection
  polynomial; the recurrence order = number of distinct exponents in a
  Prony-style sparse polynomial sampled at consecutive integer points.

- `prony_interpolate_geometric(values, base, p)` — given evaluations of
  univariate polynomial f at f(base^0), f(base^1), ... f(base^{2t-1})
  for any t exceeding the sparsity, recovers exponent–coefficient
  pairs `[(e_i, c_i)]` such that f(x) = Σ c_i x^{e_i}.

- `adaptive_lagrange_2d_mod(eval_fn, p, max_deg_x, max_deg_y, ...)` —
  start with small grid, grow until interpolation stabilizes. Useful
  when the true bidegree is unknown or the polynomial is sparse.

Use case: recovering Tutte polynomials from modular point evaluations
where the per-point cost is high (signed-DP, σ-orbit cover DP) and
the polynomial may have far fewer nonzero terms than the bidegree box.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple


def _modinv(a: int, p: int) -> int:
    return pow(a % p, p - 2, p)


def bm_recover_recurrence(seq: List[int], p: int) -> List[int]:
    """Berlekamp-Massey on a sequence over GF(p).

    Returns the minimal-length connection polynomial C(z) such that
    Σ_{i=0..L} C[i] · seq[k - i] = 0 for k ≥ L, with C[0] = 1.

    The recurrence order L = degree of returned polynomial = number of
    distinct exponents in a Prony-style polynomial.

    Standard algorithm (Massey 1969). O(|seq|^2) time.
    """
    n = len(seq)
    if n == 0:
        return [1]
    C = [1]      # current connection poly
    B = [1]      # last-update connection poly
    L = 0        # current recurrence length
    m = 1        # iteration delay since last update
    b = 1        # last discrepancy
    for k in range(n):
        # Discrepancy d_k = seq[k] + Σ_{i=1..L} C[i] * seq[k - i]
        d = seq[k] % p
        for i in range(1, L + 1):
            d = (d + C[i] * seq[k - i]) % p
        if d == 0:
            m += 1
            continue
        if 2 * L <= k:
            # Length expands.
            T = list(C)
            coef = (d * _modinv(b, p)) % p
            # C(z) := C(z) - coef * z^m * B(z)
            new_len = max(len(C), len(B) + m)
            new_C = [0] * new_len
            for i, c in enumerate(C):
                new_C[i] = c
            for i, c in enumerate(B):
                new_C[i + m] = (new_C[i + m] - coef * c) % p
            C = new_C
            L = k + 1 - L
            B = T
            b = d
            m = 1
        else:
            coef = (d * _modinv(b, p)) % p
            new_len = max(len(C), len(B) + m)
            if len(C) < new_len:
                C = C + [0] * (new_len - len(C))
            for i, c in enumerate(B):
                C[i + m] = (C[i + m] - coef * c) % p
            m += 1
    return C


def _poly_roots_mod(coeffs: List[int], p: int) -> List[int]:
    """Find all roots in GF(p) of a polynomial via brute scan.

    Coeffs are LOW-to-HIGH. Returns list of distinct roots.
    O(p · degree) — ONLY usable for small p (e.g., p < 10^4) or as fallback.
    """
    deg = len(coeffs) - 1
    while deg > 0 and coeffs[deg] % p == 0:
        deg -= 1
    if deg <= 0:
        return []
    roots = []
    for r in range(p):
        v = 0
        for i in range(deg + 1):
            v = (v + coeffs[i] * pow(r, i, p)) % p
        if v == 0:
            roots.append(r)
    return roots


def prony_interpolate_geometric(
    values: List[int],
    base: int,
    p: int,
) -> Optional[List[Tuple[int, int]]]:
    """Recover sparse polynomial f(x) = Σ c_i x^{e_i} from values
    at f(base^0), f(base^1), ... f(base^{n-1}).

    Args:
      values: [f(base^0), f(base^1), ..., f(base^{n-1})] mod p, n ≥ 2t+1
              where t = sparsity (number of distinct nonzero terms).
      base:   primitive-root-like value; exponents recovered as
              `e_i = discrete_log(base, root_i) mod (p-1)`.
      p:      prime.

    Returns: list of `(exponent, coef)` pairs, or `None` if recovery fails.

    Method: Berlekamp-Massey gives the connection polynomial whose roots
    are `base^{e_i}`. We find roots in GF(p), recover exponents by
    discrete log (small case via brute scan), then solve a Vandermonde
    system for coefficients.
    """
    seq = [v % p for v in values]
    C = bm_recover_recurrence(seq, p)
    L = len(C) - 1
    if L == 0:
        return [(0, seq[0])] if seq[0] != 0 else []

    # Roots of C(z) in GF(p). For sequence y_k = Σ_i c_i α_i^k with
    # α_i = base^{e_i}, BM's connection polynomial has roots 1/α_i.
    # So we invert each root, then DLP recovers e_i.
    roots = _poly_roots_mod(C, p)
    if len(roots) < L:
        return None
    alphas = [_modinv(r, p) for r in roots]

    # Discrete log of each α_i base `base` mod p.
    exponents = []
    base_pow = 1
    log_table: Dict[int, int] = {1: 0}
    target = set(alphas)
    for k in range(1, p):
        base_pow = (base_pow * base) % p
        if base_pow not in log_table:
            log_table[base_pow] = k
        if all(a in log_table for a in target):
            break
    for a in alphas:
        if a not in log_table:
            return None
        exponents.append(log_table[a])

    # Solve Vandermonde for coefficients using α_i (not roots).
    # values[k] = Σ_i c_i * α_i^k for k = 0..L-1.
    L_actual = len(exponents)
    V = [[pow(alphas[i], k, p) for i in range(L_actual)] for k in range(L_actual)]
    rhs = list(seq[:L_actual])
    # Gaussian elimination mod p.
    for col in range(L_actual):
        # Pivot.
        piv = -1
        for r in range(col, L_actual):
            if V[r][col] % p != 0:
                piv = r; break
        if piv < 0:
            return None
        if piv != col:
            V[col], V[piv] = V[piv], V[col]
            rhs[col], rhs[piv] = rhs[piv], rhs[col]
        inv = _modinv(V[col][col], p)
        for j in range(col, L_actual):
            V[col][j] = (V[col][j] * inv) % p
        rhs[col] = (rhs[col] * inv) % p
        for r in range(L_actual):
            if r == col: continue
            factor = V[r][col]
            if factor == 0: continue
            for j in range(col, L_actual):
                V[r][j] = (V[r][j] - factor * V[col][j]) % p
            rhs[r] = (rhs[r] - factor * rhs[col]) % p
    coeffs = rhs

    return list(zip(exponents, coeffs))


def adaptive_lagrange_2d_mod(
    eval_fn: Callable[[int, int], int],
    p: int,
    max_deg_x: int,
    max_deg_y: int,
    initial_grid: int = 4,
    growth_factor: int = 2,
    *,
    n_workers: int = 1,
) -> Tuple[Dict[Tuple[int, int], int], int]:
    """Adaptive bivariate Lagrange — start small grid, grow until stable.

    Repeatedly interpolate at increasing grid sizes; stop when a fresh
    grid produces the same polynomial as the previous one. Useful when
    the actual bidegree of the polynomial is much less than (max_deg_x,
    max_deg_y) — saves grid evaluations.

    Args:
      eval_fn: callable (x, y) → polynomial value mod p. Must be
               picklable when `n_workers > 1` (module-level function or
               instance method on a picklable class).
      p: prime.
      max_deg_x, max_deg_y: upper bounds on bidegree.
      initial_grid: starting grid side length.
      growth_factor: how much to grow per iteration.
      n_workers: parallel pool size for fresh evaluations. When > 1,
                 uncached `(x, y)` pairs in each grid-growth step are
                 evaluated concurrently via `multiprocessing.Pool`.
                 Defaults to 1 (serial, no Pool overhead).

    Returns:
      (poly_dict, n_evals_used) where poly_dict is {(a, b): coef mod p}.
    """
    from .interpolation import bivariate_lagrange_interpolate_mod

    cache: Dict[Tuple[int, int], int] = {}

    def _eval_uncached_parallel(pairs: List[Tuple[int, int]]) -> None:
        """Evaluate the given uncached (x, y) pairs in parallel and update cache."""
        if not pairs:
            return
        if n_workers <= 1 or len(pairs) <= 1:
            for x_v, y_v in pairs:
                cache[(x_v, y_v)] = eval_fn(x_v, y_v)
            return
        import multiprocessing
        with multiprocessing.Pool(n_workers) as pool:
            results = pool.starmap(eval_fn, pairs)
        for (x_v, y_v), val in zip(pairs, results):
            cache[(x_v, y_v)] = val

    def _grid(side_x: int, side_y: int) -> Dict[Tuple[int, int], int]:
        x_values = list(range(2, 2 + side_x))
        y_values = list(range(2, 2 + side_y))
        uncached = [
            (x_v, y_v) for x_v in x_values for y_v in y_values
            if (x_v, y_v) not in cache
        ]
        _eval_uncached_parallel(uncached)
        grid = [
            [cache[(x_v, y_v)] for y_v in y_values]
            for x_v in x_values
        ]
        return bivariate_lagrange_interpolate_mod(x_values, y_values, grid, p)

    side = initial_grid
    prev_poly = None
    while True:
        side_x = min(side, max_deg_x + 1)
        side_y = min(side, max_deg_y + 1)
        cur = _grid(side_x, side_y)
        if prev_poly is not None and cur == prev_poly:
            return cur, len(cache)
        prev_poly = cur
        if side_x == max_deg_x + 1 and side_y == max_deg_y + 1:
            # Full max-deg grid sampled; this is the true polynomial.
            return cur, len(cache)
        side *= growth_factor
