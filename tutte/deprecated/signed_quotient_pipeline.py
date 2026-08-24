"""Signed-quotient / σ-equivariant Tutte pipeline — DEPRECATED (test-only).

Brute-force evaluation + 2D-Lagrange-interpolation machinery for computing the
Tutte polynomial via σ-quotients (signed graphs). None of this is on a live
engine path; it is retained for the test suite (test_signed_quotient.py)
and as a research reference. The live σ-finder
`find_best_sigma` stays in `tutte/roots/signed_quotient.py`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx

from .signed_elim_dp import (
    compute_signed_tutte_elim_mod,
    compute_t_fix_sigma_mod,
)
from tutte.deprecated.interpolation import bivariate_lagrange_interpolate_mod
from tutte.roots.signed_quotient import find_best_sigma


def build_quotient_with_monodromy(
    g: nx.Graph, perm: Dict[int, int]
) -> Tuple[List[int], List[Tuple[Tuple[int, int], int]]]:
    """Build the quotient graph G/⟨σ⟩ with monodromy assignments.

    Given graph G and a permutation σ (as a vertex map), build:
      - quotient nodes: σ-orbit representatives (smallest-vertex per orbit).
      - quotient edges (as multi-edges with signs from monodromy χ).

    For a free 2-fold cover (σ has order 2, no fixed points), each
    quotient edge gets a sign χ = (sheet[u] + sheet[v]) mod 2, where
    sheets are assigned canonically (smallest vertex in each orbit = sheet 0).

    Returns:
      quotient_nodes: list of orbit IDs (consecutive integers from 0).
      quotient_edges: list of ((u, v), sign) tuples. May have multi-edges.
    """
    nodes = list(g.nodes())
    seen_v = set()
    orbit_of_v = {}
    next_oid = 0
    for v in nodes:
        if v in seen_v:
            continue
        oid = next_oid
        next_oid += 1
        cur = v
        for _ in range(20):
            if cur in seen_v:
                break
            seen_v.add(cur)
            orbit_of_v[cur] = oid
            cur = perm[cur]
    quotient_nodes = sorted(set(orbit_of_v.values()))

    orbit_min = {}
    for v, oid in orbit_of_v.items():
        if oid not in orbit_min or v < orbit_min[oid]:
            orbit_min[oid] = v
    sheet_of = {}
    for v, oid in orbit_of_v.items():
        sheet_of[v] = 0 if v == orbit_min[oid] else 1

    edges = sorted(g.edges())
    seen_e = set()
    quotient_edges = []
    for e in edges:
        ekey = tuple(sorted(e))
        if ekey in seen_e:
            continue
        u, v = e
        cur = e
        for _ in range(10):
            ckey = tuple(sorted(cur))
            seen_e.add(ckey)
            cur = (perm[cur[0]], perm[cur[1]])
            if tuple(sorted(cur)) == ekey:
                break
        u_q, v_q = orbit_of_v[u], orbit_of_v[v]
        sign = (sheet_of[u] + sheet_of[v]) & 1
        quotient_edges.append(((min(u_q, v_q), max(u_q, v_q)), sign))

    return quotient_nodes, quotient_edges


def evaluate_t_signed_mod(
    nodes: List[int],
    edges_with_signs: List[Tuple[Tuple[int, int], int]],
    x_val: int,
    y_val: int,
    p: int,
    engine: str = "c",
) -> int:
    """Evaluate T_signed at a single (x, y) point mod p.

    Wraps `compute_signed_tutte_elim_mod` and returns only the value.
    Default engine="c" (full-DP-in-C, 5-7× faster than pure Python; falls
    back to Python automatically if C-ext unavailable).
    """
    value, _ = compute_signed_tutte_elim_mod(
        nodes, edges_with_signs, x_val, y_val, p, engine=engine
    )
    return value


def _eval_one_point(args):
    """Worker fn for multiprocessing.Pool. Module-level for pickling."""
    nodes, edges_with_signs, x_v, y_v, p, engine = args
    return evaluate_t_signed_mod(nodes, edges_with_signs, x_v, y_v, p, engine=engine)


def _eval_one_point_xy(nodes, edges_with_signs, p, x_v, y_v):
    """Worker fn for adaptive_lagrange_2d_mod (positional (x, y) signature).

    Module-level so `functools.partial(_eval_one_point_xy, nodes, edges, p)`
    is picklable for multiprocessing.Pool.starmap.
    """
    return evaluate_t_signed_mod(nodes, edges_with_signs, x_v, y_v, p)


def interpolate_t_signed_mod(
    nodes: List[int],
    edges_with_signs: List[Tuple[Tuple[int, int], int]],
    x_values: List[int],
    y_values: List[int],
    p: int,
    engine: str = "c",
    n_workers: int = 1,
) -> Dict[Tuple[int, int], int]:
    """Compute T_signed as a polynomial dict {(a, b): coef mod p} via interpolation.

    Evaluates at the cartesian product of `x_values × y_values` (must each be
    distinct mod p) and then runs 2D Lagrange interpolation.

    Args:
      n_workers: if > 1, parallelize point evaluations via multiprocessing.Pool.
                 Default 1 (sequential).

    Caller must ensure `len(x_values) > deg_x(T_signed)` and likewise for y;
    otherwise the recovered polynomial is incorrect.

    Returns: {(a, b) -> coef mod p} such that T_signed(x, y) = Σ coef · x^a y^b.
    """
    n_x = len(x_values)
    n_y = len(y_values)
    tasks = [
        (nodes, edges_with_signs, x_v, y_v, p, engine)
        for x_v in x_values
        for y_v in y_values
    ]
    if n_workers > 1:
        import multiprocessing
        with multiprocessing.Pool(n_workers) as pool:
            results = pool.map(_eval_one_point, tasks)
    else:
        results = [_eval_one_point(t) for t in tasks]
    grid = [[0] * n_y for _ in range(n_x)]
    k = 0
    for i in range(n_x):
        for j in range(n_y):
            grid[i][j] = results[k]
            k += 1
    return bivariate_lagrange_interpolate_mod(x_values, y_values, grid, p)


def _eval_one_t_fix(args):
    """Worker fn for multiprocessing — module-level for picklability."""
    g, perm, x_v, y_v, p, engine_arg = args
    return compute_t_fix_sigma_quotient_mod(g, perm, x_v, y_v, p, engine=engine_arg)


def interpolate_t_fix_sigma_polynomial_mod(
    g: nx.Graph,
    perm: Dict[int, int],
    x_values: List[int],
    y_values: List[int],
    p: int,
    engine: str = "c",
    n_workers: int = 1,
) -> Dict[Tuple[int, int], int]:
    """Compute T_fix^σ as polynomial dict {(a, b): coef mod p} via interpolation.

    Evaluates `compute_t_fix_sigma_quotient_mod` at the cartesian product
    `x_values × y_values` and runs 2D Lagrange interpolation modulo p.

    Args:
      g, perm: graph + free order-2 σ.
      x_values, y_values: distinct mod-p values; must satisfy
        len(x_values) > deg_x(T_fix^σ), len(y_values) > deg_y.
      n_workers: parallelize via multiprocessing.Pool when > 1.

    Returns: {(a, b) → coef mod p} such that T_fix^σ(x, y) = Σ coef · x^a y^b mod p.
    """
    n_x = len(x_values)
    n_y = len(y_values)
    tasks = [
        (g, perm, x_v, y_v, p, engine)
        for x_v in x_values
        for y_v in y_values
    ]
    if n_workers > 1:
        import multiprocessing
        with multiprocessing.Pool(n_workers) as pool:
            results = pool.map(_eval_one_t_fix, tasks)
    else:
        results = [_eval_one_t_fix(t) for t in tasks]
    grid = [[0] * n_y for _ in range(n_x)]
    k = 0
    for i in range(n_x):
        for j in range(n_y):
            grid[i][j] = results[k]
            k += 1
    return bivariate_lagrange_interpolate_mod(x_values, y_values, grid, p)


def compute_t_signed_quotient_mod(
    g: nx.Graph,
    perm: Dict[int, int],
    x_val: int,
    y_val: int,
    p: int,
) -> int:
    """Compute T_signed(G/⟨σ⟩, χ; x_val, y_val) mod p from (G, σ).

    Convenience wrapper that builds the quotient + monodromy and evaluates
    at one point.
    """
    nodes, edges = build_quotient_with_monodromy(g, perm)
    return evaluate_t_signed_mod(nodes, edges, x_val, y_val, p)


def compute_t_fix_sigma_quotient_mod(
    g: nx.Graph,
    perm: Dict[int, int],
    x_val: int,
    y_val: int,
    p: int,
    engine: str = "c",
) -> int:
    """Compute T_fix^σ(G; x_val, y_val) mod p from (G, σ) for FREE 2-fold cover.

    T_fix^σ(G) := Σ_{A ⊆ E(G) : σ(A)=A} (x-1)^{r(E_G)-r_G(A)} (y-1)^{|A|-r_G(A)}.

    Uses the lift identity r_G(A_L) = r_quot(L) + r_signed(L, χ) (valid only
    for FREE covers — i.e., σ has no fixed edges). For covers with fixed
    edges (e.g., K_4 + (01)(23) where edges {01} and {23} are σ-fixed),
    this formula is invalid and a different DP is needed.

    Internally: builds the quotient + monodromy, computes r(E_G) via direct
    rank, and invokes compute_t_fix_sigma_mod.
    """
    nodes, edges = build_quotient_with_monodromy(g, perm)
    # r(E_G): rank of full edge set in G (= |V(G)| - num connected components).
    g_nodes = list(g.nodes())
    g_edges = list(g.edges())
    # Standard rank via union-find on g.
    parent = {v: v for v in g_nodes}
    def find_(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in g_edges:
        ru, rv = find_(u), find_(v)
        if ru != rv:
            parent[max(ru, rv)] = min(ru, rv)
    n_components = len({find_(v) for v in g_nodes})
    r_E_G = len(g_nodes) - n_components
    value, _ = compute_t_fix_sigma_mod(
        nodes, edges, r_E_G, x_val, y_val, p, engine=engine
    )
    return value


def derive_t_free_sigma_mod_via_cover(
    g: nx.Graph,
    perm: Dict[int, int],
    x_val: int,
    y_val: int,
    p: int,
) -> int:
    """Derive T_free^σ(G; x, y) mod p WITHOUT engine dependency.

    Computes T(G) directly via the σ-equivariant cover DP (sigma_orbit_dp_full)
    AND T_fix^σ via signed-DP on quotient. Returns the difference:
        T_free^σ = T(G) - T_fix^σ

    Both primitives are independent of the engine. This is the from-scratch
    σ-equivariant decomposition.

    Requires FREE σ (no σ-fixed vertices or edges).

    Args:
      g: networkx graph.
      perm: dict v → σ(v), free order-2 automorphism.
      x_val, y_val, p: evaluation point.

    Returns: T_free^σ(G; x_val, y_val) mod p.

    Performance note: cover DP runs on full G — for Z(1,2) cover (24v 76e)
    this takes >60s per modular point. Quotient signed-DP is much faster
    but only gives T_fix^σ. Use this only when you need T_free^σ from-scratch.
    """
    from ._signed_elim_c import sigma_orbit_dp_full

    # Build int-relabeled graph + σ.
    nm = {v: i for i, v in enumerate(sorted(g.nodes()))}
    n_v = len(nm)
    edges_int = [(nm[u], nm[v]) for u, v in g.edges()]
    perm_int = {nm[v]: nm[perm[v]] for v in g.nodes()}

    # r_E (unsigned rank of full edge set in G).
    parent = list(range(n_v))
    def find_(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in edges_int:
        ru, rv = find_(u), find_(v)
        if ru != rv:
            parent[max(ru, rv)] = min(ru, rv)
    n_comp = len({find_(v) for v in range(n_v)})
    r_E = n_v - n_comp

    # T(G) via cover DP.
    result = sigma_orbit_dp_full(n_v, edges_int, perm_int, r_E, x_val, y_val, p)
    if result is None:
        raise RuntimeError(
            "sigma_orbit_dp_full failed (likely σ has fixed edges → non-free cover)"
        )
    t_g = result[0]

    # T_fix^σ via quotient signed-DP.
    t_fix = compute_t_fix_sigma_quotient_mod(g, perm, x_val, y_val, p)
    return (t_g - t_fix) % p


def derive_t_free_sigma_mod(
    g: nx.Graph,
    perm: Dict[int, int],
    x_val: int,
    y_val: int,
    p: int,
    engine=None,
) -> int:
    """Derive T_free^σ(G; x, y) mod p as T(G) - T_fix^σ(G).

    Uses the σ-equivariant decomposition T(G) = T_fix^σ + T_free^σ.

    Auto-dispatches:
      - If T(G) is in the engine's rainbow-table lookup → use it (fast).
      - Otherwise → falls back to `derive_t_free_sigma_mod_via_cover`
        (from-scratch via σ-orbit cover DP, no engine dependency).

    Args:
      g: networkx graph.
      perm: dict v → σ(v), free order-2 automorphism.
      x_val, y_val: evaluation point.
      p: prime modulus.
      engine: optional SynthesisEngine (creates default if None). Only the
              lookup table is consulted — the engine never runs from-scratch
              synthesis here. Pass `engine=False` to skip the lookup
              entirely and force the cover-DP path.

    Returns:
      T_free^σ(G; x_val, y_val) mod p.
    """
    from tutte.graph import Graph as TutteGraph
    # Use Graph.from_networkx for proper int relabeling — dnx tuple labels
    # would otherwise produce wrong canonical_key and miss lookup.
    g_obj = TutteGraph.from_networkx(g)

    poly = None
    if engine is not False:
        if engine is None:
            from tutte.synthesis.engine import SynthesisEngine
            engine = SynthesisEngine()
        poly = engine.table.lookup(g_obj)

    if poly is not None:
        t_g = poly.evaluate(x_val, y_val) % p
        t_fix = compute_t_fix_sigma_quotient_mod(g, perm, x_val, y_val, p)
        return (t_g - t_fix) % p

    # Lookup miss → fall through to from-scratch cover DP.
    return derive_t_free_sigma_mod_via_cover(g, perm, x_val, y_val, p)


def decompose_t_polynomial_via_sigma(
    g: nx.Graph,
    perm: Dict[int, int],
    x_values: List[int],
    y_values: List[int],
    p: int,
    n_workers: int = 1,
    engine=None,
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
    """Decompose T(G) into (T_fix^σ, T_free^σ, T(G)) polynomials mod p.

    For a 2-fold cover G → G/⟨σ⟩, returns the full σ-equivariant
    polynomial decomposition. Auto-dispatch:
      - If T(G) is in the engine's lookup → use it (fastest).
      - Otherwise → from-scratch via σ-orbit cover DP.
    Then computes T_fix^σ via signed-DP on quotient with interpolation +
    multiprocessing, and derives T_free^σ = T(G) - T_fix^σ.

    Args:
      g, perm: graph + free order-2 σ.
      x_values, y_values: distinct mod-p values for Lagrange grid.
      p: prime modulus.
      n_workers: parallelize T_fix^σ point evaluations via multiprocessing.
      engine: optional SynthesisEngine; pass `engine=False` to skip lookup
              and force from-scratch path.

    Returns:
      (t_fix_poly, t_free_poly, t_total_poly) — each is a dict
      {(a, b) → coef mod p}.

    Performance (Z(1,2)):
      - With engine lookup hit + 8-way multiproc: ~50s for full polynomial
        decomposition (T_fix grid is the bottleneck).
      - From-scratch (no lookup): bounded by cover DP per-point cost.
    """
    from tutte.graph import Graph as TutteGraph
    # Use Graph.from_networkx for proper int relabeling — dnx tuple labels
    # would otherwise produce wrong canonical_key and miss lookup.
    g_obj = TutteGraph.from_networkx(g)

    # Auto-dispatch T(G) source.
    poly = None
    if engine is not False:
        if engine is None:
            from tutte.synthesis.engine import SynthesisEngine
            engine = SynthesisEngine()
        poly = engine.table.lookup(g_obj)

    # T_fix^σ via interpolation + multiprocessing.
    t_fix_poly = interpolate_t_fix_sigma_polynomial_mod(
        g, perm, x_values, y_values, p, engine="c", n_workers=n_workers,
    )

    if poly is not None:
        # Engine lookup hit — evaluate engine polynomial on grid.
        n_x = len(x_values)
        n_y = len(y_values)
        t_total_grid = [[0] * n_y for _ in range(n_x)]
        for i, x_v in enumerate(x_values):
            for j, y_v in enumerate(y_values):
                t_total_grid[i][j] = poly.evaluate(x_v, y_v) % p
        t_total_poly = bivariate_lagrange_interpolate_mod(
            x_values, y_values, t_total_grid, p
        )
    else:
        # From-scratch: T(G) via σ-orbit cover DP grid + interpolation.
        from ._signed_elim_c import sigma_orbit_dp_full

        # Map nodes + perm to ints.
        nm = {v: i for i, v in enumerate(sorted(g.nodes()))}
        n_v = len(nm)
        edges_int = [(nm[u], nm[v]) for u, v in g.edges()]
        perm_int = {nm[v]: nm[perm[v]] for v in g.nodes()}
        # r_E (unsigned rank).
        parent = list(range(n_v))
        def find_(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in edges_int:
            ru, rv = find_(u), find_(v)
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
        n_comp = len({find_(v) for v in range(n_v)})
        r_E = n_v - n_comp

        n_x = len(x_values)
        n_y = len(y_values)
        t_total_grid = [[0] * n_y for _ in range(n_x)]
        for i, x_v in enumerate(x_values):
            for j, y_v in enumerate(y_values):
                result = sigma_orbit_dp_full(
                    n_v, edges_int, perm_int, r_E, x_v, y_v, p
                )
                if result is None:
                    raise RuntimeError(
                        "sigma_orbit_dp_full failed on cover (likely non-free σ)"
                    )
                t_total_grid[i][j] = result[0]
        t_total_poly = bivariate_lagrange_interpolate_mod(
            x_values, y_values, t_total_grid, p
        )

    # T_free^σ = T(G) - T_fix^σ as polynomial dict.
    all_keys = set(t_total_poly.keys()) | set(t_fix_poly.keys())
    t_free_poly = {}
    for k in all_keys:
        diff = (t_total_poly.get(k, 0) - t_fix_poly.get(k, 0)) % p
        if diff != 0:
            t_free_poly[k] = diff

    return t_fix_poly, t_free_poly, t_total_poly


def compute_t_via_sigma_auto(
    g: nx.Graph,
    x_val: int,
    y_val: int,
    p: int,
    engine=None,
) -> int:
    """Single-point T(G; x, y) mod p with auto-dispatch.

    Dispatch priority:
      1. Engine lookup hit (canonical_key match) → use it (0s)
      2. find_best_sigma succeeds → T_fix^σ via signed-DP + T_free^σ
         derivation. Requires either engine lookup hit on T(G) (for the
         derivation) OR FREE σ (for sigma_orbit_dp_full direct path).
      3. Falls back to engine.synthesize from-scratch (slow)

    For from-scratch on graphs without lookup, this combines the σ-DP
    framework (compute_t_fix_sigma_quotient_mod) with sigma_orbit_dp_full
    on cover when σ is free.

    Args:
      g: networkx graph (any vertex labels — dnx tuple labels are OK,
         relabeled to ints internally).
      x_val, y_val, p: evaluation point.
      engine: optional SynthesisEngine. Pass `engine=False` to skip
              engine entirely (forces from-scratch path).

    Returns: T(G; x, y) mod p.
    """
    from tutte.graph import Graph as TutteGraph

    # Use Graph.from_networkx for proper int relabeling (handles dnx tuple
    # vertex labels like (i, j, u, k)). Manual frozenset construction
    # produces wrong canonical_keys for such graphs.
    g_obj = TutteGraph.from_networkx(g)

    # Path 1: engine lookup hit → instant.
    if engine is not False:
        if engine is None:
            from tutte.synthesis.engine import SynthesisEngine
            engine = SynthesisEngine()
        poly = engine.table.lookup(g_obj)
        if poly is not None:
            return poly.evaluate(x_val, y_val) % p

    # Path 2: σ-DP via free σ on cover → direct T(G).
    perm = find_best_sigma(g, require_free=True)
    if perm is not None:
        from ._signed_elim_c import sigma_orbit_dp_full
        nm = {v: i for i, v in enumerate(sorted(g.nodes()))}
        n_v = len(nm)
        edges_int = [(nm[u], nm[v]) for u, v in g.edges()]
        perm_int = {nm[v]: nm[perm[v]] for v in g.nodes()}
        # r_E
        parent = list(range(n_v))
        def find_(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for u, v in edges_int:
            ru, rv = find_(u), find_(v)
            if ru != rv: parent[max(ru, rv)] = min(ru, rv)
        n_comp = len({find_(v) for v in range(n_v)})
        r_E = n_v - n_comp
        result = sigma_orbit_dp_full(n_v, edges_int, perm_int, r_E, x_val, y_val, p)
        if result is not None:
            return result[0]

    # Path 3: engine.synthesize from scratch (no σ help; possibly slow).
    if engine is not False:
        result = engine.synthesize(g_obj)
        return result.polynomial.evaluate(x_val, y_val) % p

    raise RuntimeError(
        "compute_t_via_sigma_auto: engine=False and no free σ found — "
        "no from-scratch path available"
    )


def decompose_t_via_sigma_auto(
    g: nx.Graph,
    x_values: List[int],
    y_values: List[int],
    p: int,
    n_workers: int = 1,
    engine=None,
) -> Optional[Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]]:
    """Auto-detect σ and run the full σ-equivariant polynomial decomposition.

    Searches for a valid free order-2 σ in `g.Aut()` via `find_best_sigma`,
    then calls `decompose_t_polynomial_via_sigma`. Returns
    `(t_fix_poly, t_free_poly, t_total_poly)` or None if no σ found.

    For Z(1,2): uses cell-swap ±2 (free σ).
    For Z(1,3) / Cm_2: falls back to non-free σ; framework still works.
    """
    perm = find_best_sigma(g, require_free=False)
    if perm is None:
        return None
    return decompose_t_polynomial_via_sigma(
        g, perm, x_values, y_values, p, n_workers=n_workers, engine=engine
    )


_COMMON_CELL_TEMPLATES_SEEDED = set()  # (id(table), canonical_key) pairs


def _seed_common_cell_templates(table) -> None:
    """Augment existing K_n / K_{a,b} entries in `table` with their graph.

    The cell-tree detector (`try_hierarchical_partition`) computes cell
    signatures from `entry.graph`. Entries loaded from the rainbow-table
    JSON typically have `graph=None`, so detection misses them. This helper
    re-attaches `Graph` objects to canonical small cell entries in-place,
    enabling cell-tree DP path 1 of `compute_t_via_pipeline`.

    Idempotent: caches `(id(table), canonical_key)` pairs in
    `_COMMON_CELL_TEMPLATES_SEEDED` so multi-table runs (e.g., empty engine
    + production engine in the same process) each get seeded independently.
    """
    from tutte.graph import Graph as TutteGraph

    table_id = id(table)
    # Build canonical small cell templates: K_n for n=2..6, K_{a,b} for a,b in 2..4.
    # Plus Z(1,1) and Cm_1 (the dnx D-Wave family base cells) so larger Zephyr /
    # Chimera graphs in the rainbow table can be recognized as multi-cell instances.
    templates = []
    for n in range(2, 7):
        templates.append(TutteGraph.from_networkx(nx.complete_graph(n)))
    for a in range(2, 5):
        for b in range(a, 5):
            templates.append(TutteGraph.from_networkx(nx.complete_bipartite_graph(a, b)))

    # D-Wave family base cells: Z(1,1), Cm_1, Pm_1 — only attached if dnx is
    # available (it's an optional dependency for this codebase).
    try:
        import dwave_networkx as dnx
        templates.append(TutteGraph.from_networkx(dnx.zephyr_graph(1, 1)))
        templates.append(TutteGraph.from_networkx(dnx.chimera_graph(1)))
        try:
            templates.append(TutteGraph.from_networkx(dnx.pegasus_graph(1)))
        except Exception:
            pass  # Pm_1 sometimes unavailable
    except ImportError:
        pass

    for tmpl in templates:
        ck = tmpl.canonical_key()
        key = (table_id, ck)
        if key in _COMMON_CELL_TEMPLATES_SEEDED:
            continue
        entry = table.entries.get(ck)
        if entry is not None and entry.graph is None:
            entry.graph = tmpl
        _COMMON_CELL_TEMPLATES_SEEDED.add(key)


def compute_t_via_pipeline(
    g: nx.Graph,
    x_val: int,
    y_val: int,
    p: int,
    engine=None,
    verbose: bool = False,
) -> Tuple[int, str]:
    """Unified Z(m,t) / Cm_m / Pm_m pipeline — returns (T mod p, framework_used).

    Wires Z(m, t) and related D-Wave families into all three frameworks:
      0. Lookup     — rainbow-table canonical_key match (0s, all known)
      1. Cell-tree  — engine cell_quotient_tree_dp (chain-recurrence fast path,
                      sub-second on warm cache)
      2. σ-DP free  — sigma_orbit_dp_full on cover (Z(1,2) free σ, ~7s)
      3. Engine     — engine.synthesize from-scratch (chord-rule, treewidth_dp)

    Each path is tried in order until one returns a result. Returns a tuple
    `(T_value_mod_p, framework_name)` where framework_name is one of
    ``"lookup" | "sigma_dp_free" | "cell_quotient_tree" | "engine_synth"``.

    This is the explicit cross-framework wiring entry point. For Z(1,2):
    framework_name == "lookup" (sub-second). For Cm_2: same. For Z(1,3)+ /
    Cm_3+ / Pm_2+ without lookup: falls through to slower paths.

    Args:
      g: networkx graph (any vertex labels — relabeled internally).
      x_val, y_val, p: evaluation point modulo prime.
      engine: optional SynthesisEngine; created lazily if None.
      verbose: when True, prints which path was attempted/used.

    Returns: (T(g; x, y) mod p, framework name string)
    """
    from tutte.graph import Graph as TutteGraph

    g_obj = TutteGraph.from_networkx(g)

    # Path 0: Lookup.
    if engine is None:
        from tutte.synthesis.engine import SynthesisEngine
        engine = SynthesisEngine()
    poly = engine.table.lookup(g_obj)
    if poly is not None:
        if verbose:
            print(f"  pipeline: lookup hit")
        return poly.evaluate(x_val, y_val) % p, "lookup"

    # Path 1: Cell-tree (delegates to chain-recurrence fast path when applicable).
    # Tried BEFORE σ-DP because chain_recurrence is sub-second on warm cache
    # (~ms after first template extract), whereas σ-DP is multi-second.
    # The cell-tree detector requires rainbow-table entries with `graph` attached
    # so cell signatures can be computed. Production lookup entries often have
    # graph=None (loaded from JSON without graph reconstruction). Seed canonical
    # small cell templates with graphs so try_hierarchical_partition can match.
    try:
        from tutte.roots import compute_cell_quotient_tree_dp
        _seed_common_cell_templates(engine.table)
        cq_poly = compute_cell_quotient_tree_dp(g_obj, engine.table)
        if cq_poly is not None:
            if verbose:
                print(f"  pipeline: cell-quotient tree DP")
            return cq_poly.evaluate(x_val, y_val) % p, "cell_quotient_tree"
    except Exception:
        pass

    # Path 2: σ-DP free σ on cover.
    perm = find_best_sigma(g, require_free=True)
    if perm is not None:
        from ._signed_elim_c import sigma_orbit_dp_full
        nm = {v: i for i, v in enumerate(sorted(g.nodes()))}
        n_v = len(nm)
        edges_int = [(nm[u], nm[v]) for u, v in g.edges()]
        perm_int = {nm[v]: nm[perm[v]] for v in g.nodes()}
        parent = list(range(n_v))
        def find_(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for u, v in edges_int:
            ru, rv = find_(u), find_(v)
            if ru != rv: parent[max(ru, rv)] = min(ru, rv)
        n_comp = len({find_(v) for v in range(n_v)})
        r_E = n_v - n_comp
        result = sigma_orbit_dp_full(n_v, edges_int, perm_int, r_E, x_val, y_val, p)
        if result is not None:
            if verbose:
                print(f"  pipeline: σ-DP free σ on cover")
            return result[0], "sigma_dp_free"

    # Path 3: Engine synthesize from-scratch.
    if verbose:
        print(f"  pipeline: engine.synthesize fallback")
    result = engine.synthesize(g_obj)
    return result.polynomial.evaluate(x_val, y_val) % p, "engine_synth"


def compute_t_decomposition_via_pipeline(
    g: nx.Graph,
    x_values: List[int],
    y_values: List[int],
    p: int,
    n_workers: int = 1,
    engine=None,
    verbose: bool = False,
) -> Tuple[
    Optional[Dict[Tuple[int, int], int]],
    Optional[Dict[Tuple[int, int], int]],
    Optional[Dict[Tuple[int, int], int]],
    str,
]:
    """Full σ-equivariant decomposition T(G) = T_fix^σ + T_free^σ via pipeline.

    Auto-detects σ (free preferred, falls back to non-free), then returns
    the polynomial decomposition over the supplied `x_values × y_values`
    grid. Returns `(T_fix_poly, T_free_poly, T_total_poly, sigma_class)`
    where each polynomial is a `{(a, b) → coef mod p}` dict, or
    `(None, None, None, "none")` if no σ is found.

    Wrapper around `decompose_t_via_sigma_auto` with the pipeline-name
    signature for consistency with the other 4 entry points.
    """
    perm = find_best_sigma(g, require_free=True)
    sigma_class = "free"
    if perm is None:
        perm = find_best_sigma(g, require_free=False)
        sigma_class = "non-free"
    if perm is None:
        if verbose:
            print(f"  decomposition pipeline: no σ found")
        return None, None, None, "none"

    if verbose:
        print(f"  decomposition pipeline: T_fix^σ + T_free^σ via {sigma_class} σ "
              f"on {len(x_values)}×{len(y_values)} grid")

    result = decompose_t_polynomial_via_sigma(
        g, perm, x_values, y_values, p, n_workers=n_workers, engine=engine
    )
    if result is None:
        return None, None, None, sigma_class
    t_fix, t_free, t_total = result
    return t_fix, t_free, t_total, sigma_class


def compute_t_signed_via_pipeline(
    g: nx.Graph,
    x_val: int,
    y_val: int,
    p: int,
    verbose: bool = False,
) -> Tuple[Optional[int], str]:
    """Compute Zaslavsky's signed-graph Tutte polynomial T_signed(G/⟨σ⟩) mod p.

    This is the "signed-treewidth DP" path. Unlike `compute_t_via_pipeline`
    which returns the standard unsigned Tutte T(G), this returns the
    σ-equivariant signed-Tutte invariant of the quotient G/⟨σ⟩, χ.

    Pipeline:
      1. Find best σ via `find_best_sigma` (free σ preferred but not required).
      2. Build quotient + monodromy χ via `build_quotient_with_monodromy`.
      3. Evaluate T_signed on quotient via `evaluate_t_signed_mod`.

    Returns `(T_signed_mod_p, sigma_class)` where sigma_class is one of
    ``"free" | "non-free" | "none"``. If no σ is found, returns
    ``(None, "none")``.

    For Z(1, 2): finds free cell-swap σ, returns T_signed in ~8s.
    For Cm_2: finds free cell-swap σ, similar.
    """
    perm = find_best_sigma(g, require_free=True)
    sigma_class = "free"
    if perm is None:
        perm = find_best_sigma(g, require_free=False)
        sigma_class = "non-free"
    if perm is None:
        if verbose:
            print(f"  pipeline: no σ found")
        return None, "none"

    if verbose:
        print(f"  pipeline: T_signed via {sigma_class} σ on quotient")
    val = compute_t_signed_quotient_mod(g, perm, x_val, y_val, p)
    return val, sigma_class


def compute_t_signed_polynomial_adaptive_via_pipeline(
    g: nx.Graph,
    p: int,
    max_deg_x: int = 30,
    max_deg_y: int = 30,
    initial_grid: int = 4,
    growth_factor: int = 2,
    n_workers: int = 1,
    verbose: bool = False,
) -> Tuple[Optional[Dict[Tuple[int, int], int]], str, int]:
    """T_signed polynomial via ADAPTIVE Lagrange — early-stops when stable.

    Faster variant of `compute_t_signed_polynomial_via_pipeline` for cases
    where the polynomial's effective bidegree is much less than the
    theoretical maximum. Calls `adaptive_lagrange_2d_mod` from
    `tutte/roots/sparse_interp.py` which doubles the grid until two
    successive sizes recover the same polynomial.

    For graphs where T_signed has dense max bidegree (e.g., Z(1,2)
    quotient with degrees ~11×27), this offers no advantage. For graphs
    where the polynomial is sparse, this can be 2-10× faster.

    Args:
      g: networkx graph.
      p: prime modulus.
      max_deg_x, max_deg_y: upper bounds (defaults 30 cover most quotients).
      initial_grid: starting grid side (default 4).
      growth_factor: grid multiplier per iteration (default 2).
      n_workers: parallelism for evaluations.

    Returns: ({(a, b) → coef mod p}, sigma_class, n_evals_used).
    """
    from .sparse_interp import adaptive_lagrange_2d_mod

    perm = find_best_sigma(g, require_free=True)
    sigma_class = "free"
    if perm is None:
        perm = find_best_sigma(g, require_free=False)
        sigma_class = "non-free"
    if perm is None:
        if verbose:
            print(f"  adaptive polynomial pipeline: no σ found")
        return None, "none", 0

    if verbose:
        print(f"  adaptive polynomial pipeline: T_signed via {sigma_class} σ")

    import functools
    nodes, edges = build_quotient_with_monodromy(g, perm)

    # Picklable bound eval fn — module-level `_eval_one_point_xy` partial-applied
    # with (nodes, edges, p) so multiprocessing.Pool.starmap can serialize it.
    eval_fn = functools.partial(_eval_one_point_xy, nodes, edges, p)

    poly, n_evals = adaptive_lagrange_2d_mod(
        eval_fn, p, max_deg_x=max_deg_x, max_deg_y=max_deg_y,
        initial_grid=initial_grid, growth_factor=growth_factor,
        n_workers=n_workers,
    )
    if verbose:
        print(f"  adaptive: {n_evals} evals, {len(poly)} terms")
    return poly, sigma_class, n_evals


def compute_t_polynomial_via_pipeline(
    g: nx.Graph,
    engine=None,
    verbose: bool = False,
) -> Tuple[Optional["TuttePolynomial"], str]:
    """Return the full Tutte polynomial T(G) as a TuttePolynomial via pipeline routing.

    Polynomial form of `compute_t_via_pipeline`. Dispatch:

      0. Lookup → return stored polynomial directly (instant — Z(1,2),
         Cm_2 etc. hit this).
      1. Cell-tree → return `compute_cell_quotient_tree_dp(g)` poly.
      2. Engine.synthesize → return `engine.synthesize(g).polynomial`.

    Returns `(TuttePolynomial, framework_name)` where framework is one of
    ``"lookup" | "cell_quotient_tree" | "engine_synth"``. For graphs
    without lookup, the engine path's internal dispatch (treewidth_dp,
    chord_rule, etc.) handles the heavy lifting.

    Unlike `compute_t_via_pipeline`, this returns the FULL polynomial dict
    (callable as `poly.evaluate(x, y)`), not a single-point evaluation.
    """
    from tutte.graph import Graph as TutteGraph
    from tutte.polynomial import TuttePolynomial  # noqa: F401

    g_obj = TutteGraph.from_networkx(g)

    if engine is None:
        from tutte.synthesis.engine import SynthesisEngine
        engine = SynthesisEngine()

    # Path 0: Lookup.
    poly = engine.table.lookup(g_obj)
    if poly is not None:
        if verbose:
            print(f"  poly pipeline: lookup hit")
        return poly, "lookup"

    # Path 1: Cell-tree.
    try:
        from tutte.roots import compute_cell_quotient_tree_dp
        _seed_common_cell_templates(engine.table)
        cq_poly = compute_cell_quotient_tree_dp(g_obj, engine.table)
        if cq_poly is not None:
            if verbose:
                print(f"  poly pipeline: cell-quotient tree DP")
            return cq_poly, "cell_quotient_tree"
    except Exception:
        pass

    # Path 2: Engine.synthesize from-scratch.
    if verbose:
        print(f"  poly pipeline: engine.synthesize fallback")
    result = engine.synthesize(g_obj)
    return result.polynomial, "engine_synth"


def compute_t_signed_polynomial_via_pipeline(
    g: nx.Graph,
    x_values: List[int],
    y_values: List[int],
    p: int,
    n_workers: int = 1,
    verbose: bool = False,
) -> Tuple[Optional[Dict[Tuple[int, int], int]], str]:
    """Compute T_signed(G/⟨σ⟩) as a full polynomial dict mod p via interpolation.

    Polynomial form of `compute_t_signed_via_pipeline`. Evaluates T_signed
    at the cartesian product `x_values × y_values`, then runs 2D Lagrange
    interpolation. Returns `({(a, b) → coef}, sigma_class)` or
    `(None, "none")` if no σ is found.

    For polynomial recovery the caller must supply enough points:
    `len(x_values) > deg_x(T_signed)` and likewise for y. For Z(1, 2)'s
    quotient (12 verts, 38 edges), deg ≤ 11 in each variable so 12+12
    points suffice.

    Args:
      g: networkx graph.
      x_values, y_values: evaluation points (distinct mod p).
      p: prime modulus.
      n_workers: parallel pool size for point evaluations.

    Returns: ({(a, b) → coef mod p}, sigma_class).
    """
    perm = find_best_sigma(g, require_free=True)
    sigma_class = "free"
    if perm is None:
        perm = find_best_sigma(g, require_free=False)
        sigma_class = "non-free"
    if perm is None:
        if verbose:
            print(f"  polynomial pipeline: no σ found")
        return None, "none"

    if verbose:
        print(f"  polynomial pipeline: T_signed via {sigma_class} σ "
              f"on {len(x_values)}×{len(y_values)} grid")

    nodes, edges = build_quotient_with_monodromy(g, perm)
    poly = interpolate_t_signed_mod(
        nodes, edges, x_values, y_values, p, n_workers=n_workers
    )
    return poly, sigma_class


def zephyr_cell_swap_perm(m: int) -> Dict[int, int]:
    """Build the Zephyr cell-swap permutation on Z(m, t=2).

    Z(m, 2) has 4m × 4m × 2 = 32m^2 vertices total. With t=2, each cell
    contributes 8 vertices in dnx labeling. The "cell-swap" symmetry
    swaps adjacent cell-pairs within a column.

    For dnx labeling: vertices are integers 0..32m^2-1. Cells indexed
    by `(i // 2)` modulo 2 — i.e., pairs (i, i+2) swap under σ.

    This matches the perm used in research/scripts/signed_dp_modular.py:
        perm[i] = i + 2 if (i // 2) % 2 == 0 else i - 2

    For m=1 (Z(1,2)): yields the 24-vertex permutation used to validate
    the signed DP in 33s on the 12-vertex quotient.
    """
    n_v = 32 * m * m  # heuristic; caller should pass actual node count
    # Be lenient: just generate the perm for max possible vertex index 24m^2
    # since some graphs use different numbering.
    perm = {}
    for i in range(max(n_v, 24)):
        if (i // 2) % 2 == 0:
            perm[i] = i + 2
        else:
            perm[i] = i - 2
    return perm
