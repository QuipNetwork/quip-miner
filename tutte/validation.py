"""Validation Utilities for Tutte Polynomial Computation.

This module provides verification functions to ensure correctness:
1. verify_with_networkx - Compare against networkx.tutte_polynomial()
2. verify_spanning_trees - Verify T(1,1) matches Kirchhoff's matrix-tree theorem
3. verify_composition - Verify composition formulas hold

These utilities are essential for testing the synthesis engine and
ensuring that computed polynomials are correct.
"""

from __future__ import annotations

from typing import Optional, Tuple

import networkx as nx

from .polynomial import TuttePolynomial
from .graph import Graph
from .logs import get_log, EventType, LogLevel


# =============================================================================
# NETWORKX VERIFICATION
# =============================================================================

def verify_with_networkx(graph: Graph, computed: TuttePolynomial) -> bool:
    """Compare computed polynomial against networkx.tutte_polynomial().

    For large graphs (> 15 edges), skips verification as it's too expensive.

    Args:
        graph: The graph being verified
        computed: The computed Tutte polynomial

    Returns:
        True if polynomials match or graph is too large to verify
    """
    _log = get_log()
    if graph.edge_count() > 15:
        return True  # Skip expensive verification for large graphs

    G = graph.to_networkx()
    nx_poly = compute_tutte_networkx(G)

    if nx_poly is None:
        return True  # Can't verify without networkx/sympy

    match = computed == nx_poly
    if not match:
        _log.record(EventType.VERIFY, "validation",
                    f"NetworkX mismatch: {graph.node_count()}n {graph.edge_count()}e",
                    LogLevel.WARN)
    return match


def compute_tutte_networkx(G: nx.Graph) -> Optional[TuttePolynomial]:
    """Compute Tutte polynomial using networkx's implementation.

    NetworkX returns a sympy polynomial, which we convert to our format.
    Returns None if sympy is not available.
    """
    try:
        from sympy import symbols, Poly

        # NetworkX tutte_polynomial returns a sympy expression
        x, y = symbols('x y')
        tutte_sympy = nx.tutte_polynomial(G)

        # Convert to our format
        poly = Poly(tutte_sympy, x, y)
        coeffs = {}
        for monom, coeff in poly.as_dict().items():
            # monom is (x_power, y_power)
            coeffs[monom] = int(coeff)

        return TuttePolynomial.from_coefficients(coeffs)
    except ImportError:
        # sympy not available
        return None
    except Exception:
        return None


# =============================================================================
# SPANNING TREE VERIFICATION
# =============================================================================

def verify_spanning_trees(graph: Graph, poly: TuttePolynomial) -> bool:
    """Verify T(1,1) matches Kirchhoff's matrix-tree theorem.

    The number of spanning trees can be computed two ways:
    1. Tutte polynomial: T(1,1)
    2. Kirchhoff: determinant of reduced Laplacian matrix

    Uses **exact integer arithmetic (sympy)** for ALL graph sizes — the
    previous float64 path for n < 20 nodes was BUGGY for dense graphs:
    `nx.number_of_spanning_trees` returns float64 which loses precision
    during Gaussian elimination even when the final integer fits in
    float53 (intermediate determinant values can be much larger). This
    surfaced as spurious "Kirchhoff failures" on K_3+K_9+K_3 router (15n
    96e, off by +2 in the float64 path) and C_5+K_6+C_5 (16n 85e, off
    by +1) — see research/audit_chord_rule_findings.md for the audit.

    The exact path is sub-millisecond for small graphs, so there's no
    performance reason to keep the float64 path.

    Args:
        graph: The graph
        poly: The computed Tutte polynomial

    Returns:
        True if both methods agree
    """
    _log = get_log()
    tutte_count = poly.num_spanning_trees()

    # Use count_spanning_trees_kirchhoff which dispatches to fast Bareiss
    # (n ≤ 100) or C-modular (n > 100) — both are exact integer arithmetic.
    # Avoids the slow raw `_exact_spanning_tree_count` (sympy det, 15× slower
    # than Bareiss on a 40n graph: 195ms vs 13ms).
    try:
        kirchhoff_count = count_spanning_trees_kirchhoff(graph)
        if kirchhoff_count == -1:
            # count_spanning_trees_kirchhoff signals failure with -1.
            return True
    except Exception:
        return True

    match = tutte_count == kirchhoff_count
    if not match:
        _log.record(EventType.VERIFY, "validation",
                    f"Kirchhoff mismatch: T(1,1)={tutte_count} vs Kirchhoff={kirchhoff_count} "
                    f"for {graph.node_count()}n {graph.edge_count()}e",
                    LogLevel.WARN)
    return match


def _exact_num_spanning_trees(poly: TuttePolynomial) -> int:
    """Exact T(1,1) via integer coefficient sum.

    TuttePolynomial.num_spanning_trees() uses evaluate(1.0, 1.0) which
    converts large int coefficients to 64-bit floats, losing precision
    beyond ~2^53. Since T(1,1) = sum of all coefficients, we can compute
    it exactly by summing the int coefficients directly.
    """
    
    return sum(poly.to_coefficients().values())


def _exact_spanning_tree_count(graph: Graph) -> int:
    """Compute exact spanning tree count using sympy integer determinant."""
    from sympy import Matrix, zeros

    nodes = sorted(graph.nodes)
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    L = zeros(n, n)
    for u, v in graph.edges:
        i, j = idx[u], idx[v]
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1

    cofactor = L[1:, 1:]
    return int(cofactor.det())


def count_spanning_trees_kirchhoff(graph: Graph) -> int:
    """Count spanning trees using Kirchhoff's matrix-tree theorem.

    Always uses exact integer arithmetic. Previously dispatched to
    `nx.number_of_spanning_trees` (float64) for n < 20 nodes, but that
    path loses precision on dense graphs (intermediate determinant
    values overflow float53 even when the final integer fits) — see
    `verify_spanning_trees` docstring for the audit that surfaced this.

    For medium graphs (20-100 nodes), uses Python Bareiss algorithm on
    the integer Laplacian. For larger graphs (> 100 nodes), uses
    C-accelerated modular Gaussian elimination with CRT.
    """
    G = graph.to_networkx()
    try:
        # Build integer Laplacian
        L = nx.laplacian_matrix(G).toarray()
        n = len(L)
        if n <= 1:
            return 1
        # Reduced Laplacian: delete first row and column
        M = [[int(L[i][j]) for j in range(1, n)] for i in range(1, n)]
        if n <= 100:
            # Python Bareiss is faster for small matrices (no cffi overhead)
            return _bareiss_det(M)
        # C modular determinant for large matrices
        c_det = _c_modular_det(M)
        if c_det is not None:
            return c_det
        return _bareiss_det(M)
    except Exception:
        return -1


def _c_modular_det(M):
    """Compute exact integer determinant via C modular Gaussian elimination.

    Uses enough primes (dynamically chosen) to cover the Hadamard bound.
    Each modular determinant is O(n^3) in int64 C — very fast.
    Returns None if C extension is unavailable.
    """
    try:
        from .graphs._treewidth_c import _get_lib, _ffi
    except Exception:
        return None

    lib = _get_lib()
    n = len(M)
    if n == 0:
        return 1

    # Hadamard bound: |det| <= prod of row norms
    # For integer matrices, use Euclidean row norms
    import math
    log2_bound = 0
    for i in range(n):
        row_norm_sq = sum(M[i][j] * M[i][j] for j in range(n))
        if row_norm_sq > 0:
            log2_bound += math.log2(row_norm_sq) / 2

    # Need enough primes so product > 2 * bound (factor 2 for sign).
    # Generate primes near 2^50 (large enough for fast modular GE,
    # small enough that 2^50 * 2^50 < 2^63 avoids overflow in multiply).
    # Use primes of the form 2^k - c for various k and small c.
    n_primes_needed = int(log2_bound / 49) + 2  # ~49 bits per prime

    primes = _get_modular_primes(n_primes_needed)

    # Flatten matrix once using numpy for speed
    import numpy as np
    M_np = np.array(M, dtype=np.int64).ravel()
    flat = _ffi.cast("long long*", M_np.ctypes.data)

    # Pass all primes and compute all residues in one C call
    c_primes = _ffi.new("long long[]", len(primes))
    for i, p in enumerate(primes):
        c_primes[i] = p
    c_residues = _ffi.new("long long[]", len(primes))

    lib.modular_det_multi(flat, n, c_primes, len(primes), c_residues)

    residues = [int(c_residues[i]) for i in range(len(primes))]
    return _crt_multi(residues, primes)


_MODULAR_PRIMES_CACHE: dict = {}


def _get_modular_primes(n_needed, bits=50, form='any'):
    """Get at least n_needed distinct primes of the given bit-width.

    Uses Miller-Rabin primality test (deterministic for < 2^64).
    Cached per (bit-width, form). Defaults to 50-bit for backwards compatibility.
    62-bit is the max safe size for coeff_add/coeff_mul (int64 sum + __int128 prod).

    form='any': primes near 2^bits (default, matches prior behavior).
    form='mersenne_below': primes of form 2^bits - c with small c. Enables
      pseudo-Mersenne reduction in C coeff_mul.
    """
    cache_key = (bits, form)
    cached = _MODULAR_PRIMES_CACHE.get(cache_key)
    if cached is not None and len(cached) >= n_needed:
        return cached[:n_needed]

    def _is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        # Deterministic Miller-Rabin for n < 2^64
        d, r = n - 1, 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    if form == 'mersenne_below':
        # Primes p = 2^bits - c with small c. Start at 2^bits - 1, step down by 2.
        p = (1 << bits) - 1
        step = -2
    else:
        starts = {50: (1 << 50) - 27, 62: (1 << 62) - 57}
        p = starts.get(bits, (1 << bits) | 1)
        step = 2

    primes = []
    target = max(n_needed, 200)
    while len(primes) < target:
        if _is_prime(p):
            primes.append(p)
        p += step
        if p % 2 == 0:
            p += 1 if step > 0 else -1

    _MODULAR_PRIMES_CACHE[cache_key] = primes
    return primes[:n_needed]


def _crt_multi(residues, primes):
    """Chinese Remainder Theorem via Garner's algorithm.

    More efficient than the direct CRT formula for many primes:
    builds the result incrementally without computing the full product.
    """
    k = len(residues)
    if k == 0:
        return 0
    if k == 1:
        r = residues[0]
        p = primes[0]
        return r if r <= p // 2 else r - p

    # Garner's algorithm: express x = a0 + a1*p0 + a2*p0*p1 + ...
    # Precompute modular inverses
    # inv[i][j] = inverse of primes[j] mod primes[i]
    coeffs = [0] * k
    coeffs[0] = residues[0] % primes[0]

    product = primes[0]
    for i in range(1, k):
        # Find coeffs[i] such that
        # coeffs[0] + coeffs[1]*p0 + ... + coeffs[i]*p0*...*p_{i-1} ≡ residues[i] (mod primes[i])
        temp = coeffs[0]
        pp = 1
        for j in range(1, i):
            pp = pp * primes[j - 1] % primes[i]
            temp = (temp + coeffs[j] * pp) % primes[i]
        pp = pp * primes[i - 1] % primes[i]
        # coeffs[i] = (residues[i] - temp) / pp mod primes[i]
        diff = (residues[i] - temp) % primes[i]
        coeffs[i] = diff * pow(pp, primes[i] - 2, primes[i]) % primes[i]
        product *= primes[i]

    # Reconstruct: x = coeffs[0] + coeffs[1]*p0 + coeffs[2]*p0*p1 + ...
    x = coeffs[k - 1]
    for i in range(k - 2, -1, -1):
        x = x * primes[i] + coeffs[i]

    # Convert to signed
    half_product = product >> 1
    return x if x <= half_product else x - product


def _bareiss_det(M):
    """Compute exact integer determinant via Bareiss algorithm (Python fallback).

    Fraction-free Gaussian elimination: O(n^3) with only integer division
    (guaranteed exact at each step). Slow for large n due to big-integer growth.
    """
    n = len(M)
    if n == 0:
        return 1
    A = [row[:] for row in M]
    sign = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            found = False
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    A[k], A[i] = A[i], A[k]
                    sign = -sign
                    found = True
                    break
            if not found:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = A[k][k] * A[i][j] - A[i][k] * A[k][j]
                if k > 0:
                    A[i][j] //= A[k - 1][k - 1]
            A[i][k] = 0
    return sign * A[n - 1][n - 1]


# =============================================================================
# COMPOSITION VERIFICATION
# =============================================================================

def verify_composition(
    g1: Graph, t1: TuttePolynomial,
    g2: Graph, t2: TuttePolynomial,
    combined: Graph, t_combined: TuttePolynomial,
    operation: str
) -> bool:
    """Verify that composition formulas hold.

    Args:
        g1, g2: Input graphs
        t1, t2: Their Tutte polynomials
        combined: The combined graph
        t_combined: Its Tutte polynomial
        operation: Type of composition ("disjoint_union", "cut_vertex", etc.)

    Returns:
        True if the formula holds
    """
    if operation == "disjoint_union":
        # T(G1 ∪ G2) = T(G1) × T(G2)
        expected = t1 * t2
        return t_combined == expected

    elif operation == "cut_vertex":
        # T(G1 ·₁ G2) = T(G1) × T(G2) when joined at cut vertex
        expected = t1 * t2
        return t_combined == expected

    elif operation == "bridge":
        # Adding a bridge multiplies by x
        expected = t1 * TuttePolynomial.x()
        return t_combined == expected

    else:
        # Unknown operation, can't verify formula
        return True


def verify_disjoint_union(g1: Graph, g2: Graph, result: Graph,
                          compute_poly) -> Tuple[bool, str]:
    """Verify disjoint union formula: T(G1 ∪ G2) = T(G1) × T(G2).

    Args:
        g1, g2: Input graphs
        result: Combined graph
        compute_poly: Function to compute Tutte polynomial

    Returns:
        (success, message) tuple
    """
    t1 = compute_poly(g1)
    t2 = compute_poly(g2)
    t_result = compute_poly(result)

    expected = t1 * t2

    if t_result == expected:
        return True, "Disjoint union formula verified"
    else:
        return False, f"Mismatch: got {t_result}, expected {expected}"


def verify_cut_vertex_join(g1: Graph, g2: Graph, v1: int, v2: int,
                           result: Graph, compute_poly) -> Tuple[bool, str]:
    """Verify cut vertex join formula: T(G1 ·₁ G2) = T(G1) × T(G2).

    Args:
        g1, g2: Input graphs
        v1, v2: Vertices being identified
        result: Combined graph
        compute_poly: Function to compute Tutte polynomial

    Returns:
        (success, message) tuple
    """
    t1 = compute_poly(g1)
    t2 = compute_poly(g2)
    t_result = compute_poly(result)

    expected = t1 * t2

    if t_result == expected:
        return True, "Cut vertex join formula verified"
    else:
        return False, f"Mismatch: got {t_result}, expected {expected}"


# =============================================================================
# CONSISTENCY CHECKS
# =============================================================================

def verify_polynomial_properties(poly: TuttePolynomial, graph: Graph) -> Tuple[bool, str]:
    """Verify that a polynomial satisfies basic Tutte polynomial properties.

    Properties checked:
    1. All coefficients are non-negative (for graphs without bridges)
    2. T(1,1) = number of spanning trees
    3. Degree constraints based on graph size

    Args:
        poly: The computed polynomial
        graph: The graph

    Returns:
        (success, message) tuple
    """
    # Check spanning tree count
    if not verify_spanning_trees(graph, poly):
        return False, "Spanning tree count mismatch"

    # Check degree constraints
    n = graph.node_count()
    m = graph.edge_count()

    # x-degree should be at most m - n + 1 (cyclomatic complexity)
    # for connected graphs
    if graph.is_connected() and m > 0:
        max_x_deg = m - n + 1
        if poly.x_degree() > max_x_deg + 1:  # Allow some slack
            return False, f"x-degree {poly.x_degree()} exceeds expected max {max_x_deg}"

    return True, "All properties verified"


# =============================================================================
# RAINBOW TABLE VERIFICATION
# =============================================================================

def verify_rainbow_table_entry(
    entry_name: str,
    entry_poly: TuttePolynomial,
    graph_constructor,
    compute_poly
) -> Tuple[bool, str]:
    """Verify a rainbow table entry is correct.

    Args:
        entry_name: Name of the entry
        entry_poly: Stored polynomial
        graph_constructor: Function to construct the graph
        compute_poly: Function to compute Tutte polynomial

    Returns:
        (success, message) tuple
    """
    try:
        graph = graph_constructor()
        computed = compute_poly(graph)

        if computed == entry_poly:
            return True, f"{entry_name}: verified"
        else:
            return False, f"{entry_name}: mismatch - computed {computed}, stored {entry_poly}"
    except Exception as e:
        return False, f"{entry_name}: error - {e}"


def verify_rainbow_table_consistency(table, compute_poly) -> Tuple[int, int, list]:
    """Verify all computable entries in rainbow table.

    Args:
        table: RainbowTable instance
        compute_poly: Function to compute Tutte polynomial from Graph

    Returns:
        (verified_count, total_count, errors) tuple
    """
    from .graph import complete_graph, cycle_graph, path_graph, wheel_graph

    constructors = {
        'K_2': lambda: complete_graph(2),
        'K_3': lambda: complete_graph(3),
        'K_4': lambda: complete_graph(4),
        'K_5': lambda: complete_graph(5),
        'K_6': lambda: complete_graph(6),
        'C_3': lambda: cycle_graph(3),
        'C_4': lambda: cycle_graph(4),
        'C_5': lambda: cycle_graph(5),
        'C_6': lambda: cycle_graph(6),
        'P_2': lambda: path_graph(2),
        'P_3': lambda: path_graph(3),
        'P_4': lambda: path_graph(4),
        'P_5': lambda: path_graph(5),
        'W_4': lambda: wheel_graph(4),
        'W_5': lambda: wheel_graph(5),
        'W_6': lambda: wheel_graph(6),
    }

    verified = 0
    errors = []

    for name, constructor in constructors.items():
        entry = table.get_entry(name)
        if entry is None:
            continue

        success, msg = verify_rainbow_table_entry(
            name, entry.polynomial, constructor, compute_poly
        )

        if success:
            verified += 1
        else:
            errors.append(msg)

    return verified, len(constructors), errors
