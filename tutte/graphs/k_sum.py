"""K-Sum Tutte polynomial computation via the chord rule.

This module is the single entry point for k-sum-flavored Tutte polynomial
computations. It contains:

1. Polynomial division utilities — `polynomial_divmod`, `polynomial_divide`.
2. The chord rule itself — `_iterative_chord_rule`, the universal building
   block.
3. Two public algorithms that replace the deprecated matroid-theoretic
   Theorem 6 / Theorem 10 machinery:

   - `boundary_quotient_tutte(target, partition, inter_edges, engine)` — for
     hierarchically-tiled graphs (disjoint cells with inter-cell edges).
     Applies the boundary-quotient identity to the chord-free residual and
     iterative chord recursion on inter-cell chord edges.

   - `clique_chord_k_sum(target, separator, k, engine, missing_edges=None)` —
     for true k-sums (overlapping cells sharing a k-clique whose edges are
     deleted). Builds the parallel connection (clique edges added back) then
     applies chord recursion to those clique edges.

Both algorithms cost O(chord_count) full syntheses and avoid all matroid
infrastructure (no flat lattices, no Möbius function, no inclusion-exclusion
bookkeeping).

See `tutte/docs/08_2_chord_rule_formalization.md` for the formalization,
empirical validation, and the constructive replacement for Bonin-de Mier
Theorem 6 / Theorem 10.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from ..graph import Graph, MultiGraph
from ..logs import EventType, LogLevel, get_log
from ..polynomial import TuttePolynomial

if TYPE_CHECKING:
    from ..synthesis.base import BaseMultigraphSynthesizer


# =============================================================================
# POLYNOMIAL DIVISION
# =============================================================================

def polynomial_divmod(
    numerator: TuttePolynomial, denominator: TuttePolynomial,
) -> Tuple[TuttePolynomial, TuttePolynomial]:
    """Divide two Tutte polynomials with remainder.

    Returns `(quotient, remainder)` such that
    `numerator = quotient * denominator + remainder`. Long-division-style;
    stops when the leading term of the remainder is no longer divisible.
    """
    num_coeffs = numerator.to_coefficients()
    den_coeffs = denominator.to_coefficients()
    if not den_coeffs:
        raise ValueError("Cannot divide by zero polynomial")
    if not num_coeffs:
        return TuttePolynomial.zero(), TuttePolynomial.zero()

    den_leading = max(den_coeffs.keys(), key=lambda t: (t[0] + t[1], t[0]))
    den_leading_coeff = den_coeffs[den_leading]
    result_coeffs: Dict[Tuple[int, int], int] = {}
    remainder = dict(num_coeffs)

    while remainder:
        rem_leading = max(remainder.keys(), key=lambda t: (t[0] + t[1], t[0]))
        rem_leading_coeff = remainder[rem_leading]
        quot_exp = (rem_leading[0] - den_leading[0], rem_leading[1] - den_leading[1])
        if quot_exp[0] < 0 or quot_exp[1] < 0:
            break
        if rem_leading_coeff % den_leading_coeff != 0:
            break
        quot_coeff = rem_leading_coeff // den_leading_coeff
        result_coeffs[quot_exp] = result_coeffs.get(quot_exp, 0) + quot_coeff
        for (dx, dy), dc in den_coeffs.items():
            rx, ry = quot_exp[0] + dx, quot_exp[1] + dy
            remainder[(rx, ry)] = remainder.get((rx, ry), 0) - quot_coeff * dc
            if remainder[(rx, ry)] == 0:
                del remainder[(rx, ry)]

    result_coeffs = {k: v for k, v in result_coeffs.items() if v != 0}
    remainder = {k: v for k, v in remainder.items() if v != 0}
    quotient = TuttePolynomial.from_coefficients(result_coeffs) if result_coeffs else TuttePolynomial.zero()
    remainder_poly = TuttePolynomial.from_coefficients(remainder) if remainder else TuttePolynomial.zero()
    return quotient, remainder_poly


def polynomial_divide(
    numerator: TuttePolynomial, denominator: TuttePolynomial,
) -> TuttePolynomial:
    """Exact polynomial division. Raises if the division has a remainder."""
    quotient, remainder = polynomial_divmod(numerator, denominator)
    if not remainder.is_zero():
        raise ValueError(
            f"Polynomial division has non-zero remainder. "
            f"Numerator: {numerator}, Denominator: {denominator}, "
            f"Remainder: {remainder}"
        )
    if quotient.is_zero() and not numerator.is_zero():
        raise ValueError(
            f"Unexpected zero quotient for non-zero numerator. "
            f"Numerator: {numerator}, Denominator: {denominator}"
        )
    return quotient


def _try_polynomial_divide(
    num: TuttePolynomial, denom: TuttePolynomial,
) -> Optional[TuttePolynomial]:
    """Return num / denom if division is exact, else None."""
    if denom.is_zero():
        return None
    q, r = polynomial_divmod(num, denom)
    if not r.is_zero():
        return None
    return q


# =============================================================================
# LOW-LEVEL GRAPH HELPERS
# =============================================================================

def _delete_edge(g: Graph, u: int, v: int) -> Graph:
    """Return g with edge (u, v) removed. Both endpoints stay in the node set."""
    e = (min(u, v), max(u, v))
    return Graph(nodes=g.nodes, edges=frozenset(g.edges - {e}))


def _to_multigraph(g: Graph) -> MultiGraph:
    """Convert a simple Graph to MultiGraph (multiplicity 1, no loops)."""
    return MultiGraph(
        nodes=g.nodes,
        edge_counts={(min(u, v), max(u, v)): 1 for u, v in g.edges},
        loop_counts={},
    )


def _contract_edge_multi(g: Graph, u: int, v: int) -> MultiGraph:
    """Contract edge (u, v), preserving parallel-edge multiplicities and loops.

    Critical: Tutte polynomial is sensitive to parallel-edge multiplicity, so
    contracting via simple Graph (which deduplicates edges) gives wrong answers.
    """
    mg = _to_multigraph(g)
    e = (min(u, v), max(u, v))
    new_counts = dict(mg.edge_counts)
    if e in new_counts:
        if new_counts[e] == 1:
            del new_counts[e]
        else:
            new_counts[e] -= 1
    mg2 = MultiGraph(nodes=mg.nodes, edge_counts=new_counts, loop_counts=mg.loop_counts)
    return mg2.merge_nodes(u, v)


def _recognize_family_safe(g: Graph) -> Optional[TuttePolynomial]:
    """Try to recognize `g` as a known parametric family (tree, cycle, wheel,
    fan, ladder, prism, book, gear, Möbius ladder, grid, etc.) and return its
    closed-form polynomial in O(n + m).

    Returns None if `g` is not a recognized family. Used as a fast
    path inside `boundary_quotient_tutte` to skip the more expensive boundary-
    quotient division when the chord-free residual happens to be a family.

    Reuses `tutte/family_recognition/recognize_family` — no new logic here.
    """
    # Deferred import: family_recognition imports from polynomial which is
    # already loaded; keep at function scope to make the dependency intentional.
    from ..family_recognition import recognize_family
    try:
        return recognize_family(g)
    except Exception:
        return None


def _is_bridge(g: Graph, u: int, v: int) -> bool:
    """Return True iff edge (u, v) is a bridge in g — i.e. its removal would
    disconnect u from v.

    Used to dispatch the correct deletion-contraction formula at each step of
    the chord-rule iteration. Standard chord rule `T(G) = T(G − e) + T(G / e)`
    is only valid for non-bridge non-loop edges; for bridges we use
    `T(G) = x · T(G − e)` instead.

    Cost: O(n + m) BFS from u, restricted to g − {(u, v)}.
    """
    e = (min(u, v), max(u, v))
    if e not in g.edges:
        return False  # not an edge at all
    # BFS from u in g − e, see if we reach v.
    visited = {u}
    stack = [u]
    while stack:
        node = stack.pop()
        for nb in g.neighbors(node):
            # Skip the edge being tested.
            if (min(node, nb), max(node, nb)) == e:
                continue
            if nb not in visited:
                if nb == v:
                    return False  # found alternate path
                visited.add(nb)
                stack.append(nb)
    return True  # no alternate path → bridge


def _classify_bridges_chords(
    partition: List[Set[int]],
    inter_edges: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Split inter-cell edges into bridges (no cycle in inter-cell graph) and
    chords (closes a cycle). Uses UnionFind on cell super-nodes."""
    # Deferred import: avoids tutte.graphs ↔ tutte.synthesis circular import.
    from ..synthesis.base import UnionFind
    all_nodes: Set[int] = set()
    for cell_nodes in partition:
        all_nodes |= cell_nodes
    uf = UnionFind(all_nodes)
    for cell_nodes in partition:
        nodes_list = list(cell_nodes)
        for i in range(1, len(nodes_list)):
            uf.union(nodes_list[0], nodes_list[i])
    bridges: List[Tuple[int, int]] = []
    chords: List[Tuple[int, int]] = []
    for u, v in inter_edges:
        if uf.find(u) != uf.find(v):
            bridges.append((u, v))
            uf.union(u, v)
        else:
            chords.append((u, v))
    return bridges, chords


# =============================================================================
# RULE C* — BOUNDARY-QUOTIENT FORMULA (chord-free inter-cell case)
# =============================================================================

def _rule_c_star_predict(
    target: Graph,
    partition: List[Set[int]],
    inter_edges: List[Tuple[int, int]],
    cells_polys: List[TuttePolynomial],
    engine: 'BaseMultigraphSynthesizer',
) -> Optional[TuttePolynomial]:
    """boundary quotient\\*: T(target) = [∏ T(C_i)] · T(B) / [∏ T(B_i)].

    `B` is the boundary subgraph (induced on the union of all cell-boundary
    nodes — those touched by inter-cell edges; includes both inter-cell edges
    and the intra-cell edges among boundary nodes). `B_i` is the boundary
    subgraph within cell i (intra-cell only).

    Returns the predicted polynomial when polynomial division is exact, else
    None (signaling that boundary quotient\\* doesn't apply — caller should use chord recursion).
    """
    _log = get_log()
    boundary_nodes: Set[int] = set()
    for u, v in inter_edges:
        boundary_nodes.add(u); boundary_nodes.add(v)
    if not boundary_nodes:
        # No inter-cell edges → cells are disjoint → product of cell polys.
        _log.record(
            EventType.CHORD_RULE, "k_sum",
            f"boundary quotient: no inter-cell edges, returning ∏ T(C_i) over {len(cells_polys)} cells",
            LogLevel.INFO, graph=target,
        )
        product = TuttePolynomial.one()
        for tc in cells_polys:
            product = product * tc
        return product

    # Note: a tempting "trivial-boundary short-circuit" — if boundary_nodes ==
    # target.nodes, the formula tautologically reduces to T(target) = T(B).
    # We intentionally do NOT short-circuit here: recursing through
    # engine.synthesize(target) would re-enter the decomposition path and loop. The
    # full formula still computes T(target) correctly via the per-cell + B
    # synthesis (which itself benefits from family recognition / lookup /
    # treewidth_dp on B, which is the full target in this case).

    boundary_subgraph = target.subgraph(boundary_nodes)
    _log.record(
        EventType.CHORD_RULE, "k_sum",
        f"boundary quotient: boundary B has {boundary_subgraph.node_count()}n {boundary_subgraph.edge_count()}e",
        LogLevel.DEBUG, graph=boundary_subgraph,
    )
    t_boundary = engine.synthesize(boundary_subgraph).polynomial
    denom = TuttePolynomial.one()
    for cell_idx, cell_nodes in enumerate(partition):
        cb = cell_nodes & boundary_nodes
        if not cb:
            continue
        bi = target.subgraph(cb)
        _log.record(
            EventType.CHORD_RULE, "k_sum",
            f"boundary quotient: cell {cell_idx} boundary B_i has {bi.node_count()}n {bi.edge_count()}e",
            LogLevel.DEBUG, graph=bi,
        )
        denom = denom * engine.synthesize(bi).polynomial
    numer = TuttePolynomial.one()
    for tc in cells_polys:
        numer = numer * tc
    numer = numer * t_boundary
    quotient = _try_polynomial_divide(numer, denom)
    _log.record(
        EventType.CHORD_RULE, "k_sum",
        (
            "boundary quotient: polynomial division exact — formula applies"
            if quotient is not None else
            "boundary quotient: polynomial division has remainder — caller will fall back"
        ),
        LogLevel.INFO, graph=target,
    )
    return quotient


# =============================================================================
# RULE F — ITERATIVE CHORD RULE (the universal building block)
# =============================================================================

def _sigma_orbit_chord_order(
    chord_edges: List[Tuple[int, int]],
    sigma: Dict[int, int],
) -> List[Tuple[int, int]]:
    """Reorder chord_edges so each σ-orbit is contiguous.

    For σ ∈ Aut(target), σ permutes chord_edges into orbits. Processing an
    orbit's chords back-to-back maximizes engine cache hits on intermediate
    contractions: contraction polys T(g_i / c_k) for σ-related c_k are
    isomorphic to T(g_i / c_0) (the orbit's first chord), so the cache fires.

    The ordering preserves the input chord set; only the sequence changes.
    Chords whose σ-image isn't in the set are placed in their own singleton
    orbits.

    Per `tutte/research/plans/sigma_chord_rule_design.md` for the
    σ-equivariant chord rule. Helps Z(1, 3), Pm_2 family targets where the
    chord rule fires; no effect for graphs that hit tw_dp (e.g., Z(1, 2)).
    """
    edge_set = {(min(u, v), max(u, v)) for (u, v) in chord_edges}
    visited: set = set()
    ordered: List[Tuple[int, int]] = []
    for chord in chord_edges:
        canon = (min(chord), max(chord))
        if canon in visited:
            continue
        # Build orbit starting from canon.
        orbit: List[Tuple[int, int]] = [chord]
        visited.add(canon)
        u, v = chord
        cur = (sigma.get(u, u), sigma.get(v, v))
        cur_canon = (min(cur), max(cur))
        while cur_canon != canon and cur_canon not in visited and cur_canon in edge_set:
            # Find the original chord (preserve orientation from chord_edges).
            for c in chord_edges:
                if (min(c), max(c)) == cur_canon:
                    orbit.append(c)
                    break
            visited.add(cur_canon)
            u2, v2 = cur
            cur = (sigma.get(u2, u2), sigma.get(v2, v2))
            cur_canon = (min(cur), max(cur))
        ordered.extend(orbit)
    return ordered


def _iterative_chord_rule(
    target: Graph,
    chord_edges: List[Tuple[int, int]],
    engine: 'BaseMultigraphSynthesizer',
    *,
    smart_order: bool = False,
    sigma: Optional[Dict[int, int]] = None,
) -> Tuple[Graph, List[TuttePolynomial], List[TuttePolynomial]]:
    """Bridge-aware iterative deletion-contraction over `chord_edges`.

    Standard chord rule `T(G) = T(G − e) + T(G / e)` only holds for
    non-bridge, non-loop edges. For bridges it's `T(G) = x · T(G − e)`; for
    loops `T(G) = y · T(G − e)`. This function classifies each edge in the
    *current* graph at iteration step i and emits the right (factor, add) pair.

    Returns `(g_chord_free, factors, adds)` such that

        T(target) = (∏_i factors[i]) · T(g_chord_free)
                    + Σ_i (∏_{j<i} factors[j]) · adds[i]

    For non-degenerate graphs (no bridges/loops appear during iteration) every
    factor is 1, every add is the contraction polynomial, and the formula
    reduces to the simple `T(target) = T(g_chord_free) + Σ adds`.

    Chord processing order: processes chords in the order the caller provides.
    Set ``smart_order=True`` to sort chords by descending
    ``|common_neighbors(u, v)|`` in the *original* graph — chords with more
    shared neighbors create denser parallel-edge multigraphs on contraction,
    which the engine's `_synthesize_multigraph` can simplify via the
    parallel-edge / loop fast paths.

    Set ``sigma`` to a vertex permutation that's an automorphism of ``target``
    to enable σ-equivariant ordering — chords in the same σ-orbit are processed
    back-to-back so the engine's canonical_key cache catches isomorphic
    intermediate contractions. Per `tutte/research/plans/sigma_chord_rule_design.md`.
    """
    _log = get_log()
    if sigma is not None and len(chord_edges) > 1:
        chord_edges = _sigma_orbit_chord_order(chord_edges, sigma)
    if smart_order and len(chord_edges) > 1:
        target_neighbors = {
            n: set(target.neighbors(n)) for n in target.nodes
        }

        def _common_count(edge):
            u, v = edge
            return -len(target_neighbors.get(u, set()) & target_neighbors.get(v, set()))

        chord_edges = sorted(chord_edges, key=_common_count)
    g_i = target
    factors: List[TuttePolynomial] = []
    adds: List[TuttePolynomial] = []
    n_chords = len(chord_edges)
    if n_chords > 0:
        _log.record(
            EventType.CHORD_RULE, "k_sum",
            f"chord recursion: bridge-aware iteration over {n_chords} chord(s)",
            LogLevel.INFO, graph=target,
        )
    x_poly = TuttePolynomial.x()  # T-poly for a bridge factor

    for i, chord in enumerate(chord_edges):
        u, v = chord
        # Skip if the edge isn't actually present (e.g. caller passed a stale
        # edge after some external mutation).
        if (min(u, v), max(u, v)) not in g_i.edges:
            factors.append(TuttePolynomial.one())
            adds.append(TuttePolynomial.zero())
            continue

        if _is_bridge(g_i, u, v):
            # Bridge in g_i: T(g_i) = x · T(g_i − e). No contraction term.
            factors.append(x_poly)
            adds.append(TuttePolynomial.zero())
            _log.record(
                EventType.CHORD_RULE, "k_sum",
                f"chord recursion chord {i + 1}/{n_chords} ({u},{v}): BRIDGE — factor x, no contract term",
                LogLevel.DEBUG, graph=g_i,
            )
        else:
            # Non-bridge non-loop: T(g_i) = T(g_i − e) + T(g_i / e).
            # Compute the contraction polynomial as a multigraph (may have
            # parallel edges + loops created by the contraction itself).
            mg = _contract_edge_multi(g_i, u, v)
            if not mg.edge_counts and not mg.loop_counts:
                contract_poly = TuttePolynomial.one()
                _log.record(
                    EventType.CHORD_RULE, "k_sum",
                    f"chord recursion chord {i + 1}/{n_chords} ({u},{v}): contraction is empty → T = 1",
                    LogLevel.DEBUG,
                )
            else:
                _log.record(
                    EventType.CHORD_RULE, "k_sum",
                    f"chord recursion chord {i + 1}/{n_chords} ({u},{v}): contraction "
                    f"{sum(mg.edge_counts.values())}e {sum(mg.loop_counts.values())}loops",
                    LogLevel.DEBUG, graph=mg,
                )
                contract_poly = engine._synthesize_multigraph(mg)
            factors.append(TuttePolynomial.one())
            adds.append(contract_poly)
        g_i = _delete_edge(g_i, u, v)
        _log.record(
            EventType.CHORD_RULE, "k_sum",
            f"chord recursion chord {i + 1}/{n_chords}: deleted from working graph "
            f"({g_i.node_count()}n {g_i.edge_count()}e remain)",
            LogLevel.DEBUG, graph=g_i,
        )
    return g_i, factors, adds


def _combine_chord_iteration(
    t_chord_free: TuttePolynomial,
    factors: List[TuttePolynomial],
    adds: List[TuttePolynomial],
) -> TuttePolynomial:
    """Apply the bridge-aware chord-rule formula to combine intermediate results.

        T(target) = (∏_i factors[i]) · T(chord_free)
                    + Σ_i (∏_{j<i} factors[j]) · adds[i]

    For non-degenerate cases (all factors are 1) this collapses to the simple
    T(target) = T(chord_free) + Σ adds.
    """
    # Total factor for the chord-free term.
    total_factor = TuttePolynomial.one()
    for f in factors:
        total_factor = total_factor * f
    total = total_factor * t_chord_free

    # Each add term gets the product of preceding factors as its prefix factor.
    prefix = TuttePolynomial.one()
    for f, a in zip(factors, adds):
        if not a.is_zero():
            total = total + prefix * a
        prefix = prefix * f
    return total


def _solve_for_chord_free(
    t_target: TuttePolynomial,
    factors: List[TuttePolynomial],
    adds: List[TuttePolynomial],
) -> TuttePolynomial:
    """Inverse of `_combine_chord_iteration`: given T(target), solve for T(chord_free).

        T(chord_free) = [T(target) − Σ_i (∏_{j<i} factors[j]) · adds[i]] / (∏_i factors[i])

    Used by `clique_chord_k_sum` where T(PC) is known and we want T(target = PC − all chords).
    Polynomial division by the factor product is always exact (the chord rule
    guarantees it for valid inputs).
    """
    numerator = t_target
    prefix = TuttePolynomial.one()
    for f, a in zip(factors, adds):
        if not a.is_zero():
            numerator = numerator - prefix * a
        prefix = prefix * f
    # prefix is now ∏_i factors[i] — divide out to recover T(chord_free).
    if prefix == TuttePolynomial.one():
        return numerator
    return polynomial_divide(numerator, prefix)


# =============================================================================
# PUBLIC API — HIERARCHICAL CASE
# =============================================================================

def boundary_quotient_tutte(
    target: Graph,
    partition: List[Set[int]],
    inter_edges: List[Tuple[int, int]],
    engine: 'BaseMultigraphSynthesizer',
) -> TuttePolynomial:
    """Compute T(target) for a hierarchically-tiled graph using boundary quotient\\* + chord recursion.

    Disjoint cells `partition` with inter-cell edges. Steps:
      1. Classify inter-cell edges into bridges (no cycle in inter-cell graph)
         and chords (closes a cycle).
      2. Apply iterative chord rule to peel chords; each contraction leaf is
         synthesized via the engine (chord contraction breaks the disjoint
         partition, so chord leaves use direct synthesis).
      3. The chord-free leaf has only bridges in the inter-cell graph; apply
         boundary quotient\\* to it. If boundary quotient\\* returns None (division not exact), fall
         back to direct synthesis of the chord-free leaf.

    Cost: 1 + (# chord inter-cell edges) full syntheses.
    """
    _log = get_log()
    _log.record(
        EventType.CHORD_RULE, "k_sum",
        f"boundary_quotient_tutte: {len(partition)} cells, "
        f"{len(inter_edges)} inter-cell edges, target "
        f"{target.node_count()}n {target.edge_count()}e",
        LogLevel.INFO, graph=target,
    )
    bridges, chords = _classify_bridges_chords(partition, inter_edges)
    _log.record(
        EventType.CHORD_RULE, "k_sum",
        f"classified inter-cell edges: {len(bridges)} bridge(s), {len(chords)} chord(s)",
        LogLevel.DEBUG, graph=target,
    )

    # σ-equivariant chord ordering: if target has a free automorphism σ
    # discoverable by `find_best_sigma`, reorder chords so σ-orbits are
    # processed contiguously — the engine's canonical_key cache will
    # catch isomorphic intermediate contractions. On by default
    # (`engine.chord_sigma_order`, default True); Petersen measured a ~1.44×
    # speedup with bit-for-bit identical results (see test_sigma.py).
    sigma_perm = None
    if getattr(engine, "chord_sigma_order", True) and chords:
        try:
            import networkx as nx
            from tutte.roots.signed_quotient import find_best_sigma
            g_nx = nx.Graph()
            g_nx.add_nodes_from(target.nodes)
            g_nx.add_edges_from(target.edges)
            sigma_perm = find_best_sigma(g_nx, require_free=True)
            if sigma_perm is None:
                sigma_perm = find_best_sigma(g_nx, require_free=False)
        except Exception:
            sigma_perm = None

    # Strip chords first (bridge-aware chord recursion). Returns
    # (g_chord_free, factors, adds) for the universal combine formula.
    smart_order = getattr(engine, "chord_smart_order", False)
    g_chord_free, factors, adds = _iterative_chord_rule(
        target, chords, engine, smart_order=smart_order, sigma=sigma_perm,
    )
    chord_free_inter = [e for e in inter_edges if e not in set(chords)]

    # Family recognition on the chord-free leaf.
    # If g_chord_free is a known parametric family (tree, cycle, wheel,
    # ladder, etc.), `recognize_family` returns its polynomial in O(n + m) —
    # much cheaper than the boundary-quotient computation below.
    chord_free_poly = _recognize_family_safe(g_chord_free)

    if chord_free_poly is None:
        # No family hit; try boundary quotient on the chord-free leaf.
        cells_polys = [
            engine.synthesize(g_chord_free.subgraph(cell_nodes)).polynomial
            for cell_nodes in partition
        ]
        chord_free_poly = _rule_c_star_predict(
            g_chord_free, partition, chord_free_inter, cells_polys, engine,
        )
    if chord_free_poly is None:
        # Final fallback: direct synthesis of the chord-free leaf. (Only happens
        # for exotic boundary structures boundary quotient cannot divide.)
        _log.record(
            EventType.CHORD_RULE, "k_sum",
            "boundary quotient did not apply — falling back to direct synthesis of chord-free leaf",
            LogLevel.INFO, graph=g_chord_free,
        )
        chord_free_poly = engine.synthesize(g_chord_free).polynomial

    # Combine: T(target) = (∏ factors) · T(chord_free) + Σ (prefix factors) · adds[i].
    # For non-degenerate cases (every factor=1) this is just T(chord_free) + Σ adds.
    total = _combine_chord_iteration(chord_free_poly, factors, adds)
    _log.record(
        EventType.CHORD_RULE, "k_sum",
        f"boundary_quotient_tutte: done — polynomial has {total.num_terms()} term(s)",
        LogLevel.INFO, graph=target,
    )
    return total


# =============================================================================
# PUBLIC API — TRUE k-SUM CASE
# =============================================================================

def clique_chord_k_sum(
    target: Graph,
    separator: Tuple[int, ...],
    k: int,
    engine: 'BaseMultigraphSynthesizer',
    missing_edges: Optional[List[Tuple[int, int]]] = None,
) -> TuttePolynomial:
    """Compute T(target) for a true k-sum via iterative chord rule on the
    shared K_k clique edges.

    `target` is the k-sum graph (typically with the K_k clique edges deleted —
    we rebuild the parallel connection by adding them back, then peel them off
    one at a time via the chord rule).

    `separator` is the tuple of k shared vertices.

    `missing_edges` (optional): if only a subset of the K_k clique edges are
    deleted in `target`, pass the list of deleted edges here. Defaults to all
    C(k, 2) clique edges.

    Algorithm:
      PC = target + missing_edges                     (parallel connection)
      For each missing edge e_i in some order:
        contract leaf = (PC − e_1 − ... − e_{i-1}) / e_i
        Synthesize contract leaf.
      Final: T(target) = T(PC) − Σ contraction polynomials.

    Cost: 1 + |missing_edges| full syntheses. For a classic k-sum,
    |missing_edges| = C(k, 2), so cost is O(k²).
    """
    _log = get_log()
    sv = sorted(separator)
    if missing_edges is None:
        missing_edges = [(sv[i], sv[j]) for i in range(k) for j in range(i + 1, k)]

    _log.record(
        EventType.CHORD_RULE, "k_sum",
        f"clique_chord_k_sum: k={k} separator={tuple(sv)}, "
        f"{len(missing_edges)} clique edge(s) to peel, target "
        f"{target.node_count()}n {target.edge_count()}e",
        LogLevel.INFO, graph=target,
    )

    if not missing_edges:
        # All clique edges already present in target → target IS PC →
        # synthesize directly.
        _log.record(
            EventType.CHORD_RULE, "k_sum",
            "clique_chord_k_sum: no missing clique edges — target IS PC, synthesizing directly",
            LogLevel.INFO, graph=target,
        )
        return engine.synthesize(target).polynomial

    # Build PC = target + missing clique edges.
    pc = Graph(nodes=target.nodes, edges=target.edges | frozenset(missing_edges))
    _log.record(
        EventType.CHORD_RULE, "k_sum",
        f"built parallel connection PC: {pc.node_count()}n {pc.edge_count()}e",
        LogLevel.DEBUG, graph=pc,
    )

    # Synthesize T(PC) — the engine handles the parallel-connection structure
    # internally (no special handling required here).
    t_pc = engine.synthesize(pc).polynomial

    # Bridge-aware iterative chord rule on the clique edges.
    smart_order = getattr(engine, "chord_smart_order", False)
    g_chord_free, factors, adds = _iterative_chord_rule(
        pc, missing_edges, engine, smart_order=smart_order,
    )

    # Solve T(PC) = (∏ factors) · T(target) + Σ (prefix factors) · adds[i]
    # for T(target). Polynomial division by the factor product is always exact
    # because the chord rule guarantees it.
    total = _solve_for_chord_free(t_pc, factors, adds)
    _log.record(
        EventType.CHORD_RULE, "k_sum",
        f"clique_chord_k_sum: done — polynomial has {total.num_terms()} term(s)",
        LogLevel.INFO, graph=target,
    )
    return total
