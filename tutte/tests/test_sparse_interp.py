"""Tests for sparse polynomial interpolation."""
from __future__ import annotations

import pytest

from tutte.deprecated.sparse_interp import (
    bm_recover_recurrence,
    prony_interpolate_geometric,
    adaptive_lagrange_2d_mod,
)


def test_bm_zero_sequence():
    """All-zero sequence should produce trivial recurrence."""
    C = bm_recover_recurrence([0, 0, 0, 0], 1009)
    assert C == [1]


def test_bm_constant_sequence():
    """Constant sequence c, c, c, ... has recurrence s_k = s_{k-1}."""
    C = bm_recover_recurrence([5, 5, 5, 5, 5], 1009)
    # Connection poly should be 1 - z (i.e., s_k = 1 * s_{k-1})
    assert len(C) == 2
    assert C[0] == 1
    assert C[1] == -1 % 1009


def test_bm_geometric():
    """Geometric sequence c · r^k satisfies s_k = r * s_{k-1}."""
    p = 1009
    r = 3
    seq = [pow(r, k, p) for k in range(6)]
    C = bm_recover_recurrence(seq, p)
    # Recurrence s_k = r * s_{k-1} → connection poly C(z) = 1 - r*z
    assert len(C) == 2
    assert C[0] == 1
    assert C[1] == (-r) % p


def test_prony_single_term():
    """Polynomial f(x) = 7 · x^3 sampled at x = 1, 2, 4, 8, 16."""
    p = 1009
    base = 2
    f = lambda x: (7 * pow(x, 3, p)) % p
    values = [f(pow(base, k, p)) for k in range(4)]
    result = prony_interpolate_geometric(values, base, p)
    assert result is not None
    # Should recover (3, 7).
    exponents = sorted(e for e, _ in result)
    assert exponents == [3]
    coef_dict = dict(result)
    assert coef_dict[3] == 7


def test_prony_two_terms():
    """f(x) = 5 + 8 · x^2."""
    p = 1009
    base = 2
    f = lambda x: (5 + 8 * pow(x, 2, p)) % p
    # Need 2t+1 = 5 samples.
    values = [f(pow(base, k, p)) for k in range(8)]
    result = prony_interpolate_geometric(values, base, p)
    assert result is not None
    coef_dict = dict(result)
    assert sorted(coef_dict.keys()) == [0, 2]
    assert coef_dict[0] == 5
    assert coef_dict[2] == 8


def test_adaptive_lagrange_dense_small():
    """Dense bivariate polynomial recovered correctly with adaptive grid."""
    p = 1009
    # f(x, y) = 2 + 3*y + 5*x + 7*x*y + 11*x^2 + y^2*4
    def f(x, y):
        return (2 + 3*y + 5*x + 7*x*y + 11*x*x + 4*y*y) % p
    poly, n_evals = adaptive_lagrange_2d_mod(
        f, p, max_deg_x=3, max_deg_y=3, initial_grid=2, growth_factor=2
    )
    expected = {(0, 0): 2, (0, 1): 3, (0, 2): 4,
                (1, 0): 5, (1, 1): 7, (2, 0): 11}
    # Filter zero coeffs.
    poly_nz = {k: v for k, v in poly.items() if v != 0}
    assert poly_nz == expected
    assert n_evals <= 16  # 4×4 grid suffices


def test_adaptive_lagrange_sparse_terminates_early():
    """Sparse polynomial: small grid sufficient."""
    p = 1009
    # f(x, y) = 1 + x^5  (sparse, but max_deg = 5)
    f = lambda x, y: (1 + pow(x, 5, p)) % p
    poly, n_evals = adaptive_lagrange_2d_mod(
        f, p, max_deg_x=5, max_deg_y=2, initial_grid=4, growth_factor=2
    )
    expected = {(0, 0): 1, (5, 0): 1}
    poly_nz = {k: v for k, v in poly.items() if v != 0}
    assert poly_nz == expected


def _picklable_test_poly(x: int, y: int) -> int:
    """Module-level eval fn (picklable for multiprocessing)."""
    p = 1009
    return (2 + 3*y + 5*x + 7*x*y + 11*x*x + 4*y*y) % p


def test_adaptive_lagrange_parallel_matches_serial():
    """n_workers=2 produces the same polynomial as n_workers=1."""
    p = 1009
    poly_serial, n_evals_serial = adaptive_lagrange_2d_mod(
        _picklable_test_poly, p, max_deg_x=3, max_deg_y=3,
        initial_grid=2, growth_factor=2, n_workers=1,
    )
    poly_parallel, n_evals_parallel = adaptive_lagrange_2d_mod(
        _picklable_test_poly, p, max_deg_x=3, max_deg_y=3,
        initial_grid=2, growth_factor=2, n_workers=2,
    )
    assert poly_serial == poly_parallel, (
        f"Serial vs parallel polynomial mismatch: "
        f"{poly_serial} vs {poly_parallel}"
    )
    # Eval counts should match too (deterministic).
    assert n_evals_serial == n_evals_parallel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
