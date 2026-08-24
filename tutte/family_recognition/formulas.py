"""Closed-form formulas and recurrences for known graph families.

Tier 1: Closed-form formulas — O(1) computation after detection.
Tier 2: Linear recurrences — O(n) polynomial multiplications after detection.

All base cases verified against SynthesisEngine + Kirchhoff T(1,1),
or derived from closed-form eigenvalue decompositions in [5] and [15].

References:
    [1] Tutte (1947) — foundational deletion-contraction, bridge/cut-vertex factorization
    [3] Brennan, Mansour, Mphako-Banda (2013) — wheel and fan formulas
    [4] Biggs, Damerell, Sands (1972) — recursive families, transfer matrices
    [5] Chang and Shrock (2004) — strip graph Tutte polynomials via transfer matrices
    [8] Shrock (2000) — Potts model partition functions on ladder graphs
    [15] Weisstein, MathWorld — gear graph closed-form formula
"""

from __future__ import annotations

from typing import Optional

from ..polynomial import TuttePolynomial, cycle_polynomial
from .constants import (BOOK_A, BOOK_B, BOOK_BASES, FAN_A, FAN_B, FAN_BASES,
                        GEAR_A, GEAR_B, GEAR_BASES, GEAR_C, LADDER_A, LADDER_B,
                        LADDER_BASES, MOBIUS_BASES, PRISM_BASES, WHEEL_A,
                        WHEEL_B, WHEEL_BASES, WHEEL_C, apply_order6_recurrence)

# =============================================================================
# TIER 1 — CLOSED-FORM FORMULAS
# =============================================================================

def tree_formula(n: int) -> TuttePolynomial:
    """T(tree on n vertices) = x^(n-1).

    Every edge is a bridge. Cut vertex factorization gives the product.
    Complexity: O(1).

    Source: Tutte [1].
    """
    if n <= 1:
        return TuttePolynomial.one()
    return TuttePolynomial.x(n - 1)


def cycle_formula(n: int) -> TuttePolynomial:
    """T(C_n) = x^(n-1) + x^(n-2) + ... + x + y.

    Complexity: O(n) to build coefficient dict.

    Source: Folklore, derivable from deletion-contraction [1].
    """
    return cycle_polynomial(n)


def pan_formula(cycle_size: int) -> TuttePolynomial:
    """T(Pan_n) = x · T(C_n).

    Pan = cycle C_n with one pendant edge. Cut vertex factorization:
    bridge contributes x, cycle contributes T(C_n).

    Complexity: O(n) to build cycle polynomial.

    Source: Cut-vertex factorization [1].
    """
    return TuttePolynomial.x(1) * cycle_formula(cycle_size)


def sunlet_formula(k: int) -> TuttePolynomial:
    """T(Sunlet_k) = x^k · T(C_k).

    Sunlet = cycle C_k with pendant at each vertex. Each pendant
    contributes a bridge factor of x.

    Complexity: O(k) to build cycle polynomial.

    Source: Iterated cut-vertex factorization [1].
    """
    return TuttePolynomial.x(k) * cycle_formula(k)


def helm_formula(k: int) -> Optional[TuttePolynomial]:
    """T(Helm_k) = x^k · T(W_k). Returns None if wheel bases unavailable."""
    w = wheel_recurrence(k)
    if w is None:
        return None
    return TuttePolynomial.x(k) * w


# =============================================================================
# TIER 1 — RECURRENCE-BASED (still O(k) after detection)
# =============================================================================

def wheel_recurrence(k: int) -> Optional[TuttePolynomial]:
    """T(W_k) for wheel with k rim vertices (k+1 total vertices, 2k edges).

    Returns None if base cases are not available in the rainbow table.
    """
    if k < 3:
        raise ValueError(f"Wheel requires k >= 3 rim vertices, got {k}")

    bases = WHEEL_BASES()
    if bases is None:
        return None

    if k <= 5:
        return bases[k - 3]

    prev3, prev2, prev1 = bases
    for _ in range(6, k + 1):
        curr = WHEEL_A * prev1 - WHEEL_B * prev2 + WHEEL_C * prev3
        prev3, prev2, prev1 = prev2, prev1, curr

    return prev1


def fan_recurrence(k: int) -> Optional[TuttePolynomial]:
    """T(F_k) for fan. Returns None if base cases unavailable."""
    if k < 1:
        raise ValueError(f"Fan requires k >= 1, got {k}")

    bases = FAN_BASES()
    if bases is None:
        return None

    if k <= 2:
        return bases[k - 1]

    prev2, prev1 = bases
    for _ in range(3, k + 1):
        curr = FAN_A * prev1 - FAN_B * prev2
        prev2, prev1 = prev1, curr

    return prev1


def ladder_recurrence(k: int) -> Optional[TuttePolynomial]:
    """T(L_k) for ladder P_k × P_2. Returns None if base cases unavailable."""
    if k < 2:
        raise ValueError(f"Ladder requires k >= 2, got {k}")

    bases = LADDER_BASES()
    if bases is None:
        return None

    if k <= 3:
        return bases[k - 2]

    prev2, prev1 = bases
    for _ in range(4, k + 1):
        curr = LADDER_A * prev1 - LADDER_B * prev2
        prev2, prev1 = prev1, curr

    return prev1


def book_recurrence(k: int) -> Optional[TuttePolynomial]:
    """T(Book_k) for k triangles sharing an edge. Returns None if base cases unavailable."""
    if k < 1:
        raise ValueError(f"Book requires k >= 1, got {k}")

    bases = BOOK_BASES()
    if bases is None:
        return None

    if k <= 2:
        return bases[k - 1]

    prev2, prev1 = bases
    for _ in range(3, k + 1):
        curr = BOOK_A * prev1 - BOOK_B * prev2
        prev2, prev1 = prev1, curr

    return prev1


def gear_recurrence(k: int) -> Optional[TuttePolynomial]:
    """T(Gear_k) for gear graph. Returns None if base cases unavailable."""
    if k < 3:
        raise ValueError(f"Gear requires k >= 3 rim vertices, got {k}")

    bases = GEAR_BASES()
    if bases is None:
        return None

    if k <= 5:
        return bases[k - 3]

    prev3, prev2, prev1 = bases
    for _ in range(6, k + 1):
        curr = GEAR_A * prev1 - GEAR_B * prev2 + GEAR_C * prev3
        prev3, prev2, prev1 = prev2, prev1, curr

    return prev1


def prism_recurrence(k: int) -> Optional[TuttePolynomial]:
    """T(CL_k) for prism (circular ladder). Returns None if base cases unavailable."""
    if k < 3:
        raise ValueError(f"Prism requires k >= 3, got {k}")

    bases = PRISM_BASES()
    if bases is None:
        return None

    if k <= 8:
        return bases[k - 3]

    cl3, cl4, cl5, cl6, cl7, cl8 = bases
    return apply_order6_recurrence(cl3, cl4, cl5, cl6, cl7, cl8, 9, k)


def mobius_recurrence(k: int) -> Optional[TuttePolynomial]:
    """T(M_k) for Möbius ladder. Returns None if base cases unavailable."""
    if k < 3:
        raise ValueError(f"Möbius ladder requires k >= 3, got {k}")

    bases = MOBIUS_BASES()
    if bases is None:
        return None

    if k <= 8:
        return bases[k - 3]

    m3, m4, m5, m6, m7, m8 = bases
    return apply_order6_recurrence(m3, m4, m5, m6, m7, m8, 9, k)


# =============================================================================
# TIER 2 — GRID RECURRENCE
# =============================================================================

def grid_recurrence(m: int, n: int) -> Optional[TuttePolynomial]:
    """T(P_m × P_n) for grid graph with m rows, n columns.

    For m=1, this is a path. For m=2, this is the ladder.

    Complexity: O(n) polynomial multiplications for fixed small m.

    Source: Transfer matrix framework [4]; Chang and Shrock [5].
    """
    if m > n:
        m, n = n, m

    if m == 1:
        return tree_formula(n)

    if m == 2:
        return ladder_recurrence(n)

    # m >= 3: transfer matrix has Bell(m) states per column (Bell(3)=5,
    # Bell(6)=203, Bell(10)=115975). A pure-Python implementation would be
    # slower than the engine's C-extension treewidth_dp for the same Bell(m)
    # state space. The k-sum handles m >= 7 grids via
    # chord-rule paths once treewidth_dp's heuristic decomposition fails.
    # See `tutte/tests/test_benchmark_family_recognition.py`'s
    # @pytest.mark.slow marker for the test that exercises this gap.
    return None