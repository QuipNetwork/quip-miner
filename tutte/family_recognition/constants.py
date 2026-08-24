"""Recurrence coefficients and base-case loaders for graph family Tutte polynomial formulas.

Recurrence coefficients are mathematical constants from published papers — they
are hardcoded here because they do not correspond to any graph's polynomial.

Base cases (seed polynomials for each recurrence) are loaded from the rainbow
table at first use. If the rainbow table is empty (fresh clone, no benchmarks
run yet), the base cases are computed on demand via the synthesis engine and
cached for subsequent calls.

To populate the rainbow table with all seed graphs, run:
    make benchmark

The benchmark includes all seed graphs (Gear_4/5, Prism_4-8, Mobius_4-8, etc.)
in NAMED_GRAPHS, so they are computed and stored automatically.

References:
    [1] Tutte (1947) — foundational deletion-contraction, bridge/cut-vertex factorization
    [3] Brennan, Mansour, Mphako-Banda (2013) — wheel and fan recurrences
    [4] Biggs, Damerell, Sands (1972) — ladder recurrence
    [5] Chang and Shrock (2004) — prism/Möbius transfer matrix eigenvalues
    [8] Shrock (2000) — Potts model partition functions on ladder graphs
    [15] Weisstein, MathWorld — gear graph recurrence
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..polynomial import TuttePolynomial


# =============================================================================
# RECURRENCE COEFFICIENTS (mathematical constants, NOT graph polynomials)
# =============================================================================

# WHEEL — order-3 recurrence [3]
# T(W_n) = (x+y+2)·T(W_{n-1}) - (x+1)(y+1)·T(W_{n-2}) + xy·T(W_{n-3})
WHEEL_A = TuttePolynomial.from_coefficients({(1, 0): 1, (0, 1): 1, (0, 0): 2})
WHEEL_B = TuttePolynomial.from_coefficients({
    (1, 1): 1, (1, 0): 1, (0, 1): 1, (0, 0): 1,
})
WHEEL_C = TuttePolynomial.from_coefficients({(1, 1): 1})

# FAN — order-2 recurrence [3]
# T(F_n) = (x+y+1)·T(F_{n-1}) - xy·T(F_{n-2})
FAN_A = TuttePolynomial.from_coefficients({(1, 0): 1, (0, 1): 1, (0, 0): 1})
FAN_B = TuttePolynomial.from_coefficients({(1, 1): 1})

# LADDER — order-2 recurrence [4], [5]
# T(L_n) = (x²+x+y+1)·T(L_{n-1}) - x²y·T(L_{n-2})
LADDER_A = TuttePolynomial.from_coefficients({
    (2, 0): 1, (1, 0): 1, (0, 1): 1, (0, 0): 1,
})
LADDER_B = TuttePolynomial.from_coefficients({(2, 1): 1})

# BOOK — order-2 recurrence (folklore)
# T(B_k) = (2x+y+1)·T(B_{k-1}) - (x+1)(x+y)·T(B_{k-2})
BOOK_A = TuttePolynomial.from_coefficients({(1, 0): 2, (0, 1): 1, (0, 0): 1})
BOOK_B = TuttePolynomial.from_coefficients({
    (2, 0): 1, (1, 1): 1, (1, 0): 1, (0, 1): 1,
})

# GEAR — order-3 recurrence [15]
# T(G_n) = (x²+x+y+2)·T(G_{n-1}) - (x²y+x²+x+y+1)·T(G_{n-2}) + x²y·T(G_{n-3})
GEAR_A = TuttePolynomial.from_coefficients({
    (2, 0): 1, (1, 0): 1, (0, 1): 1, (0, 0): 2,
})
GEAR_B = TuttePolynomial.from_coefficients({
    (2, 1): 1, (2, 0): 1, (1, 0): 1, (0, 1): 1, (0, 0): 1,
})
GEAR_C = TuttePolynomial.from_coefficients({(2, 1): 1})

# PRISM / MÖBIUS — order-6 recurrence [5], [8]
# Characteristic polynomial:
#   (z-1)(z-x)(z²-(x+y+2)z+xy)(z²-(x²+x+y+1)z+x²y)
ORDER6_A1 = TuttePolynomial.from_coefficients({
    (2, 0): 1, (1, 0): 3, (0, 1): 2, (0, 0): 4,
})
ORDER6_A2 = TuttePolynomial.from_coefficients({
    (3, 0): -2, (2, 1): -2, (2, 0): -6, (1, 1): -5, (1, 0): -9,
    (0, 2): -1, (0, 1): -5, (0, 0): -5,
})
ORDER6_A3 = TuttePolynomial.from_coefficients({
    (4, 0): 1, (3, 1): 4, (3, 0): 5, (2, 2): 1, (2, 1): 8, (2, 0): 8,
    (1, 2): 2, (1, 1): 9, (1, 0): 8, (0, 2): 1, (0, 1): 3, (0, 0): 2,
})
ORDER6_A4 = TuttePolynomial.from_coefficients({
    (4, 1): -2, (4, 0): -1, (3, 2): -2, (3, 1): -7, (3, 0): -3,
    (2, 2): -2, (2, 1): -7, (2, 0): -3, (1, 2): -2, (1, 1): -4, (1, 0): -2,
})
ORDER6_A5 = TuttePolynomial.from_coefficients({
    (4, 2): 1, (4, 1): 2, (3, 2): 2, (3, 1): 3, (2, 2): 1, (2, 1): 1,
})
ORDER6_A6 = TuttePolynomial.from_coefficients({(4, 2): -1})


# =============================================================================
# BASE CASE LOADING (from rainbow table, with compute-on-demand fallback)
# =============================================================================

# Cache: populated on first access per family
_base_cache: dict = {}

# Per-seed polynomial cache populated by `_load_or_compute`. Hits avoid the
# multi-MB disk read inside `load_default_table` and the engine round-trip
# in `_compute_seed`.
_seed_polys: dict = {}

# Singleton rainbow table reference. `load_default_table` re-reads the binary
# table file on every call (~200ms for a 1k-entry table). Cached here so
# multiple `_load_or_compute` calls in a single process share one load.
_cached_table = None

# Reentrancy guard. While True, `recognize_family` short-circuits to None so
# the synthesis engine that we are calling to compute a missing seed doesn't
# turn around and recurse into us. Module-global is fine because seed
# computation is single-threaded.
_computing_seeds: bool = False


def _get_cached_table():
    """Lazily load the default rainbow table once per process."""
    global _cached_table
    if _cached_table is None:
        from ..lookup.core import load_default_table
        _cached_table = load_default_table()
    return _cached_table


def _load_by_name(name: str) -> Optional[TuttePolynomial]:
    """Load base case from the rainbow table by name. Returns None if not found.

    Uses name-based lookup (not canonical key) because WL hashing is not
    a complete graph invariant — the same graph built two different ways
    can produce different canonical keys. Name-based lookup is reliable
    as long as the benchmark stores seeds under consistent names.
    """
    return _get_cached_table().lookup_by_name(name)


def _compute_seed(name: str) -> Optional[TuttePolynomial]:
    """Compute a missing seed via the synthesis engine. Returns None if no
    builder is registered for `name` or seed computation is already in progress
    (to avoid infinite recursion through `recognize_family`)."""
    global _computing_seeds
    if _computing_seeds:
        return None
    from ._seed_builders import SEED_BUILDERS
    builder = SEED_BUILDERS.get(name)
    if builder is None:
        return None
    # Lazy import — synthesis.engine imports family_recognition, so we must
    # break the cycle by deferring this until first call.
    from ..synthesis.engine import SynthesisEngine
    _computing_seeds = True
    try:
        engine = SynthesisEngine(table=_get_cached_table(), verbose=False)
        engine.skip_target_lookup = True
        result = engine.synthesize(builder())
        return result.polynomial
    finally:
        _computing_seeds = False


def _load_or_compute(name: str) -> Optional[TuttePolynomial]:
    """Look up `name` in the rainbow table, or compute via the engine. Result
    is cached in-process (`_seed_polys`); not written back to disk. Run
    `python -m tutte.scripts.warmup_lookup_table` to persist seeds."""
    if name in _seed_polys:
        return _seed_polys[name]
    poly = _load_by_name(name)
    if poly is None:
        poly = _compute_seed(name)
    if poly is not None:
        _seed_polys[name] = poly
    return poly



def _get_wheel_bases() -> Optional[Tuple[TuttePolynomial, ...]]:
    """Load T(W_3), T(W_4), T(W_5) — seeds for the wheel recurrence.
    Returns None if any base case is missing from the rainbow table."""
    if 'wheel' not in _base_cache:
        polys = (
            _load_or_compute("K_4"),       # W_3 = K_4
            _load_or_compute("W_4"),
            _load_or_compute("W_5"),
        )
        if any(p is None for p in polys):
            return None
        _base_cache['wheel'] = polys
    return _base_cache['wheel']


def _get_fan_bases() -> Optional[Tuple[TuttePolynomial, ...]]:
    """Load T(F_1), T(F_2) — seeds for the fan recurrence.
    Returns None if any base case is missing from the rainbow table."""
    if 'fan' not in _base_cache:
        polys = (
            _load_or_compute("K_2"),       # F_1 = single edge
            _load_or_compute("K_3"),       # F_2 = triangle
        )
        if any(p is None for p in polys):
            return None
        _base_cache['fan'] = polys
    return _base_cache['fan']


def _get_ladder_bases() -> Optional[Tuple[TuttePolynomial, ...]]:
    """Load T(L_2), T(L_3) — seeds for the ladder recurrence.
    Returns None if any base case is missing from the rainbow table."""
    if 'ladder' not in _base_cache:
        polys = (
            _load_or_compute("C_4"),         # L_2 = C_4
            _load_or_compute("Grid_2x3"),    # L_3
        )
        if any(p is None for p in polys):
            return None
        _base_cache['ladder'] = polys
    return _base_cache['ladder']


def _get_book_bases() -> Optional[Tuple[TuttePolynomial, ...]]:
    """Load T(B_1), T(B_2) — seeds for the book recurrence.
    Returns None if any base case is missing from the rainbow table."""
    if 'book' not in _base_cache:
        polys = (
            _load_or_compute("K_3"),       # B_1 = K_3
            _load_or_compute("B_2"),
        )
        if any(p is None for p in polys):
            return None
        _base_cache['book'] = polys
    return _base_cache['book']


def _get_gear_bases() -> Optional[Tuple[TuttePolynomial, ...]]:
    """Load T(G_3), T(G_4), T(G_5) — seeds for the gear recurrence.
    Returns None if any base case is missing from the rainbow table."""
    if 'gear' not in _base_cache:
        polys = (
            _load_or_compute("Gear_3"),
            _load_or_compute("Gear_4"),
            _load_or_compute("Gear_5"),
        )
        if any(p is None for p in polys):
            return None
        _base_cache['gear'] = polys
    return _base_cache['gear']


def _get_prism_bases() -> Optional[Tuple[TuttePolynomial, ...]]:
    """Load T(CL_3)..T(CL_8) — seeds for the prism recurrence.
    Returns None if any base case is missing from the rainbow table."""
    if 'prism' not in _base_cache:
        polys = tuple(
            _load_or_compute(f"Prism_{k}")
            for k in range(3, 9)
        )
        if any(p is None for p in polys):
            return None
        _base_cache['prism'] = polys
    return _base_cache['prism']


def _get_mobius_bases() -> Optional[Tuple[TuttePolynomial, ...]]:
    """Load T(M_3)..T(M_8) — seeds for the Möbius recurrence.
    Returns None if any base case is missing from the rainbow table."""
    if 'mobius' not in _base_cache:
        polys = tuple(
            _load_or_compute(f"Mobius_{k}")
            for k in range(3, 9)
        )
        if any(p is None for p in polys):
            return None
        _base_cache['mobius'] = polys
    return _base_cache['mobius']


# =============================================================================
# PUBLIC API — lazy properties that formulas.py imports
# =============================================================================

class _LazyBases:
    """Loader that fetches base cases from the rainbow table on first access.

    Returns None if the rainbow table doesn't have the required seed graphs.
    This allows recognize_family() to gracefully fall through to the engine's
    other synthesis paths (SP, treewidth DP, chord addition).
    """
    _NOT_LOADED = object()

    def __init__(self, loader):
        self._loader = loader
        self._value = self._NOT_LOADED

    def __call__(self) -> Optional[Tuple[TuttePolynomial, ...]]:
        if self._value is self._NOT_LOADED:
            self._value = self._loader()  # Returns None if table missing seeds
        return self._value


# These are called as functions in formulas.py: WHEEL_BASES(), not WHEEL_BASES
WHEEL_BASES = _LazyBases(_get_wheel_bases)
FAN_BASES = _LazyBases(_get_fan_bases)
LADDER_BASES = _LazyBases(_get_ladder_bases)
BOOK_BASES = _LazyBases(_get_book_bases)
GEAR_BASES = _LazyBases(_get_gear_bases)
PRISM_BASES = _LazyBases(_get_prism_bases)
MOBIUS_BASES = _LazyBases(_get_mobius_bases)


def apply_order6_recurrence(
    b6: TuttePolynomial, b5: TuttePolynomial, b4: TuttePolynomial,
    b3: TuttePolynomial, b2: TuttePolynomial, b1: TuttePolynomial,
    start_k: int, target_k: int,
) -> TuttePolynomial:
    """Apply order-6 recurrence with sliding window of 6 base values.

    b6..b1 are T(start_k-6)..T(start_k-1). Returns T(target_k).

    Complexity: O(target_k - start_k) polynomial multiplications.
    """
    p6, p5, p4, p3, p2, p1 = b6, b5, b4, b3, b2, b1
    for _ in range(start_k, target_k + 1):
        curr = (ORDER6_A1 * p1 + ORDER6_A2 * p2 + ORDER6_A3 * p3
                + ORDER6_A4 * p4 + ORDER6_A5 * p5 + ORDER6_A6 * p6)
        p6, p5, p4, p3, p2, p1 = p5, p4, p3, p2, p1, curr
    return p1
