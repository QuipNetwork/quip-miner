"""Graph Family Recognition — O(n+m) heuristics for known graph families.

Recognizes trees, cycles, wheels, fans, ladders, pans, sunlets, books, helms,
and grids, returning their Tutte polynomials via closed-form formulas or linear
recurrences. Bypasses the expensive canonical_key computation (O(n² log n)).

Public API:
    recognize_family(graph) -> Optional[TuttePolynomial]
    compute_structural_fingerprint(graph) -> StructuralFingerprint
    StructuralFingerprint  (frozen dataclass)
"""

from __future__ import annotations

from typing import Optional

from ..graph import Graph
from ..polynomial import TuttePolynomial
from .fingerprint import StructuralFingerprint, compute_structural_fingerprint
from .formulas import (
    book_recurrence,
    cycle_formula,
    fan_recurrence,
    gear_recurrence,
    grid_recurrence,
    helm_formula,
    ladder_recurrence,
    mobius_recurrence,
    pan_formula,
    prism_recurrence,
    sunlet_formula,
    tree_formula,
    wheel_recurrence,
)
from .verification import (
    detect_grid_dims,
    verify_book,
    verify_gear,
    verify_helm,
    verify_ladder,
    verify_mobius,
    verify_prism,
    verify_sunlet,
)

__all__ = [
    'recognize_family',
    'compute_structural_fingerprint',
    'StructuralFingerprint',
]


def recognize_family(graph: Graph) -> Optional[TuttePolynomial]:
    """Recognize known graph families and return Tutte polynomial.

    Runs a cascade of O(n+m) structural checks. Each check either identifies
    the family and returns its polynomial, or falls through to the next.

    Pipeline position: after base cases and cut vertex split, before
    series-parallel check and canonical_key computation.

    Complexity: O(n + m) — dominated by fingerprint computation (BFS for
    bipartiteness) and structural verification passes.

    Returns:
        TuttePolynomial if the family is recognized, None otherwise.
    """
    # Reentry guard: if the constants module is currently computing a missing
    # seed via the engine, the engine will call us back during step 1 of its
    # pipeline. Returning None lets the engine fall through to cotree_dp /
    # treewidth_dp, which compute the seed without needing family recognition.
    from .constants import _computing_seeds
    if _computing_seeds:
        return None

    n = graph.node_count()
    m = graph.edge_count()

    # Quick connectivity check — required for all family formulas.
    # Uses DFS from first node; O(n+m) but exits early if disconnected.
    if n > 1:
        start = next(iter(graph.nodes))
        reached = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in reached:
                continue
            reached.add(node)
            for nb in graph.neighbors(node):
                if nb not in reached:
                    stack.append(nb)
        if len(reached) != n:
            return None  # Disconnected — no single family formula applies

    # Precompute derived values used by multiple checks below.
    # Avoids redundant division/modulus on large vertex counts.
    n_is_even = n % 2 == 0
    n_is_odd = not n_is_even
    n_half = n // 2           # valid when n_is_even
    n_half_odd = (n - 1) // 2 # valid when n_is_odd (k for helm/gear)
    twice_n_minus_1 = 2 * (n - 1)

    # --- O(1) + O(n+m) checks based on (n, m) ---

    # Tree (covers paths, stars, all trees)
    if m == n - 1:
        return tree_formula(n)

    # Compute fingerprint once — O(n+m)
    fp = compute_structural_fingerprint(graph)
    dc = fp.degree_counts  # shorthand for degree count dict

    # --- O(1) checks against fingerprint ---

    # Cycle: connected, |E| = |V|, all degree 2
    if m == n and dc == {2: n}:
        return cycle_formula(n)

    # Wheel: one hub with degree n-1, all others degree 3, 2(n-1) edges
    # W_k has k+1 vertices. For n=4 (W_3=K_4), hub degree = 3 = rim degree.
    # Verify: hub connected to all rim vertices, rim forms a cycle.
    if n >= 4 and m == twice_n_minus_1:
        if n == 4:
            if dc == {3: 4}:
                return wheel_recurrence(3)
        elif (dc.get(n - 1, 0) == 1
                and dc.get(3, 0) == n - 1
                and len(dc) == 2):
            # Find hub and verify rim is a cycle
            hub = None
            for v in graph.nodes:
                if graph.degree(v) == n - 1:
                    hub = v
                    break
            if hub is not None:
                hub_nbs = set(graph.neighbors(hub))
                rim = [v for v in graph.nodes if v != hub]
                # Hub must be connected to all rim vertices
                if hub_nbs == set(rim):
                    # Rim vertices must form a single cycle: each has exactly
                    # 2 rim neighbors AND the rim is connected (one component).
                    rim_set = set(rim)
                    all_deg2 = all(
                        sum(1 for nb in graph.neighbors(v) if nb in rim_set) == 2
                        for v in rim
                    )
                    if all_deg2:
                        # Check rim connectivity via DFS
                        rim_visited = set()
                        rim_stack = [rim[0]]
                        while rim_stack:
                            rv = rim_stack.pop()
                            if rv in rim_visited:
                                continue
                            rim_visited.add(rv)
                            for nb in graph.neighbors(rv):
                                if nb in rim_set and nb not in rim_visited:
                                    rim_stack.append(nb)
                        if len(rim_visited) == len(rim):
                            return wheel_recurrence(n - 1)

    # Fan: one apex with degree n-1, two degree-2 endpoints, rest degree 3
    # F_k has k+1 vertices, 2k-1 edges. k = n-1.
    # Verify: apex connected to all others, non-apex vertices form a path.
    if (n >= 4 and m == twice_n_minus_1 - 1
            and dc.get(n - 1, 0) == 1
            and dc.get(2, 0) == 2
            and dc.get(3, 0) == n - 3
            and len(dc) == 3):
        # Find apex (degree n-1) and verify rim is a path
        apex = None
        for v in graph.nodes:
            if graph.degree(v) == n - 1:
                apex = v
                break
        if apex is not None:
            rim = [v for v in graph.nodes if v != apex]
            # rim should form a path: 2 endpoints (degree 2 in full graph = degree 1 in rim)
            rim_set = set(rim)
            rim_adj = {v: [] for v in rim}
            for v in rim:
                for nb in graph.neighbors(v):
                    if nb in rim_set:
                        rim_adj[v].append(nb)
            endpoints = [v for v in rim if len(rim_adj[v]) == 1]
            if len(endpoints) == 2:
                # Trace path from endpoint to endpoint
                path = [endpoints[0]]
                visited = {endpoints[0]}
                while len(path) < len(rim):
                    cur = path[-1]
                    found_next = False
                    for nb in rim_adj[cur]:
                        if nb not in visited:
                            path.append(nb)
                            visited.add(nb)
                            found_next = True
                            break
                    if not found_next:
                        break
                if len(path) == len(rim):
                    return fan_recurrence(n - 1)

    # Pan: cycle C_{n-1} with one pendant vertex
    # One pendant (deg 1), one deg-3 vertex, rest deg-2, |E|=|V|
    # Verify: removing pendant + its neighbor's extra edge leaves a cycle.
    if (n >= 4 and m == n
            and dc.get(1, 0) == 1
            and dc.get(3, 0) == 1
            and dc.get(2, 0) == n - 2):
        # Find pendant and its neighbor (the deg-3 vertex)
        pendant = None
        for v in graph.nodes:
            if graph.degree(v) == 1:
                pendant = v
                break
        if pendant is not None:
            hub = next(iter(graph.neighbors(pendant)))
            # Remaining graph (without pendant) should be a cycle C_{n-1}
            rim = [v for v in graph.nodes if v != pendant]
            if all(graph.degree(v) == 2 for v in rim if v != hub):
                # hub has degree 3 in full graph, degree 2 in rim → cycle check
                rim_set = set(rim)
                hub_rim_deg = sum(1 for nb in graph.neighbors(hub) if nb in rim_set)
                if hub_rim_deg == 2:
                    return pan_formula(n - 1)

    # --- O(1) + O(n+m) verification checks ---

    # Ladder: P_k × P_2 — 2k vertices, 3k-2 edges, 4 degree-2 corners
    if n >= 4 and n_is_even:
        k = n_half
        if (m == 3 * k - 2
                and dc.get(2, 0) == 4
                and dc.get(3, 0) == n - 4
                and fp.is_bipartite
                and len(dc) == 2):
            if verify_ladder(graph, k):
                return ladder_recurrence(k)

    # Helm: hub degree k, rim degree 4, pendants degree 1
    # 2k+1 vertices, 3k edges
    # Special case k=4: hub and rim both degree 4 → degree_counts = {4: k+1, 1: k}
    if n >= 7 and n_is_odd:
        k = n_half_odd
        if m == 3 * k and dc.get(1, 0) == k:
            if k == 4:
                if dc.get(4, 0) == k + 1:
                    if verify_helm(graph, k):
                        return helm_formula(k)
            elif (dc.get(k, 0) == 1
                    and dc.get(4, 0) == k):
                if verify_helm(graph, k):
                    return helm_formula(k)

    # Gear: hub degree k, k rim vertices degree 3, k subdivision vertices degree 2
    # 2k+1 vertices, 3k edges (same as helm but different degree pattern)
    # Special case k=3: hub and rim both degree 3 → degree_counts = {3: k+1, 2: k}
    if n >= 7 and n_is_odd:
        k = n_half_odd
        if m == 3 * k and dc.get(2, 0) == k:
            if k == 3:
                if dc.get(3, 0) == k + 1:
                    if verify_gear(graph, k):
                        return gear_recurrence(k)
            elif (dc.get(k, 0) == 1
                    and dc.get(3, 0) == k):
                if verify_gear(graph, k):
                    return gear_recurrence(k)

    # Book: k triangles sharing one edge — (k+2) vertices, (2k+1) edges
    if n >= 4:
        k = n - 2
        if k >= 1:
            hub_deg = k + 1
            if (m == 2 * k + 1
                    and dc.get(hub_deg, 0) == 2
                    and dc.get(2, 0) == k):
                if verify_book(graph, k):
                    return book_recurrence(k)

    # Sunlet: half degree 1, half degree 3, |E| = |V|
    if n >= 6 and n_is_even and m == n:
        k = n_half
        if (dc.get(1, 0) == k
                and dc.get(3, 0) == k):
            if verify_sunlet(graph, k):
                return sunlet_formula(k)

    # Prism / Möbius: 3-regular, 2k vertices, 3k edges
    # Prism C_k×K_2 is bipartite iff k is even; Möbius M_k is bipartite iff k is odd.
    # Try both verifiers — bipartiteness alone cannot distinguish them.
    if (n >= 6 and n_is_even
            and fp.is_regular and fp.regularity == 3
            and m == 3 * n_half):
        k = n_half
        if verify_prism(graph, k):
            return prism_recurrence(k)
        if verify_mobius(graph, k):
            return mobius_recurrence(k)

    # Grid: bipartite, specific degree pattern, m <= 5 rows
    if fp.is_bipartite and dc.get(2, 0) == 4:
        dims = detect_grid_dims(graph, fp)
        if dims is not None:
            m_dim, n_dim = dims
            result = grid_recurrence(m_dim, n_dim)
            if result is not None:
                return result

    return None  # Not recognized — fall through to expensive path