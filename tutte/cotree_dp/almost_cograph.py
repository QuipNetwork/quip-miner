"""Almost-cograph Tutte computation.

A graph is "almost a cograph" if removing a small set of *anomaly edges* —
edges that participate in induced P_4s — leaves a cograph (P_4-free graph).

Algorithm: iterated greedy P_4 elimination + bridge-aware chord rule.

1. Find a small set A ⊆ E(G) whose removal yields a cograph (greedy P_4
   detection: pick any induced P_4, designate its middle edge as anomaly,
   delete, repeat until P_4-free).
2. Apply `_iterative_chord_rule` over A. The "chord-free" leaf is the
   cograph G − A, which routes through cotree_dp via the engine.
3. The k contraction leaves are recursively synthesized via the engine.
   They may or may not be cographs themselves.

Cost: 1 cotree_dp call (the cograph skeleton) + |A| recursive engine
calls (the contraction leaves). Compared to the naive 2^|A| subset
enumeration, the iterated chord rule gives O(|A|) leaves.

Best for graphs that are mostly cographs: D-Wave Cm/Pm cells (which ARE
cographs) joined by a sparse set of inter-cell edges (which are the
anomalies). Per cmtw analysis, Cm2 has 4 inter-cell
edge bundles → expected anomaly count ~16 (4 bundles × 4 edges); Cm3
has 12 → ~48. Capping at max_anomalies=20 covers Cm2 cleanly; Cm3
falls through to the existing pipeline.

Public API:
    find_anomaly_edges(graph, max_anomalies=20) -> Optional[List[Tuple[int,int]]]
    compute_tutte_almost_cograph(graph, engine, max_anomalies=20) -> Optional[TuttePolynomial]
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, List, Optional, Tuple

from ..graph import Graph
from ..polynomial import TuttePolynomial
from .dp import compute_tutte_cotree_dp
from .recognition import _build_cotree

if TYPE_CHECKING:
    from ..synthesis.base import BaseMultigraphSynthesizer


def _find_induced_p4_middle_edge(graph: Graph) -> Optional[Tuple[int, int]]:
    """Find one induced P_4 in the graph and return its middle edge.

    P_4 = a-b-c-d (path on 4 vertices, 3 edges). The middle edge is b-c.
    Removing b-c (alone) breaks this specific induced P_4.

    Returns None if the graph contains no induced P_4 (i.e., is a cograph).

    Complexity: O(n^4) brute-force enumeration. Adequate for n ≤ ~100;
    matches cotree_dp's own O(n^4) recognition complexity.
    """
    nodes = sorted(graph.nodes)
    if len(nodes) < 4:
        return None

    # Pre-build a dense neighbor lookup; graph.neighbors() returns a frozenset.
    neighbor_map = {n: graph.neighbors(n) for n in nodes}

    for combo in combinations(nodes, 4):
        # Count edges within the 4-vertex induced subgraph.
        a, b, c, d = combo
        edges_in_combo = []
        for u, v in [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]:
            if v in neighbor_map[u]:
                edges_in_combo.append((u, v))

        if len(edges_in_combo) != 3:
            continue

        # Compute degrees within the induced subgraph.
        induced_deg = {a: 0, b: 0, c: 0, d: 0}
        for u, v in edges_in_combo:
            induced_deg[u] += 1
            induced_deg[v] += 1

        # P_4 has degree sequence (1, 1, 2, 2).
        deg2_nodes = [n for n, d in induced_deg.items() if d == 2]
        deg1_nodes = [n for n, d in induced_deg.items() if d == 1]
        if len(deg2_nodes) == 2 and len(deg1_nodes) == 2:
            # Middle edge connects the two degree-2 nodes (in the induced
            # subgraph). They must be adjacent in the graph since induced P_4
            # has its middle vertices connected.
            x, y = sorted(deg2_nodes)
            if y in neighbor_map[x]:
                return (x, y)

    return None


def _delete_edge(graph: Graph, u: int, v: int) -> Graph:
    """Return a copy of `graph` with edge (u, v) removed."""
    a, b = (u, v) if u < v else (v, u)
    new_edges = graph.edges - frozenset({(a, b)})
    return Graph(nodes=graph.nodes, edges=new_edges)


def find_anomaly_edges(
    graph: Graph,
    max_anomalies: int = 20,
    max_nodes: int = 60,
) -> Optional[List[Tuple[int, int]]]:
    """Find edges whose removal makes `graph` a cograph (greedy P_4 elimination).

    Returns:
        - [] if `graph` is already a cograph.
        - [edges...] (length ≤ max_anomalies) if greedy elimination succeeds.
        - None if more than max_anomalies eliminations are required.
        - None immediately if `graph` has more than `max_nodes` nodes
          (default 60) — `_find_induced_p4_middle_edge` is O(n^4) per
          iter (≤max_anomalies+1 iters). C(72,4)=1.06M × 17 iters ≈ 9
          min on Cm3 alone; Pm3 (128n) is >>1hr. Empirically Cm2 (32n)
          uses grid_dp_streamed not almost_cograph, so cap 60 keeps
          Cm2 working while skipping Cm3+ and Pm3+.

    The greedy heuristic finds A locally-optimally per step but does NOT
    guarantee minimum |A|. For the structured graphs we target (cells +
    sparse inter-cell edges), the greedy result is typically tight.

    Complexity: O(n^4) per iteration × at most `max_anomalies` iterations.
    """
    if graph.node_count() > max_nodes:
        return None
    g = graph
    anomalies: List[Tuple[int, int]] = []

    for _ in range(max_anomalies + 1):  # +1 to detect overflow
        cotree = _build_cotree(g)
        if cotree is not None:
            return anomalies
        edge = _find_induced_p4_middle_edge(g)
        if edge is None:
            # Cotree said not-cograph but we can't find a P_4 — defensive.
            return None
        anomalies.append(edge)
        g = _delete_edge(g, *edge)
        if len(anomalies) > max_anomalies:
            return None

    return None


def compute_tutte_almost_cograph(
    graph: Graph,
    engine: 'BaseMultigraphSynthesizer',
    max_anomalies: int = 20,
) -> Optional[TuttePolynomial]:
    """Compute T(graph) by reducing to a cograph + chord rule on anomalies.

    Args:
        graph: Simple input graph.
        engine: Synthesis engine used to recursively synthesize contraction
                leaves (the chord rule produces |anomalies| leaves; the
                "chord-free" leaf is a cograph and uses cotree_dp directly).
        max_anomalies: Cap on anomaly-edge count. If the greedy elimination
                       finds more than this, returns None (caller falls
                       through to the next pipeline step).

    Returns:
        The Tutte polynomial, or None if the graph isn't tractable as
        an almost-cograph within `max_anomalies`.
    """
    if not isinstance(graph, Graph):
        return None  # Multigraphs not supported in cograph recognition

    anomalies = find_anomaly_edges(graph, max_anomalies=max_anomalies)
    if anomalies is None:
        return None  # too many anomalies

    if len(anomalies) == 0:
        # Already a cograph — compute directly.
        try:
            return compute_tutte_cotree_dp(graph)
        except (ValueError, TypeError):
            return None

    # Apply iterated chord rule on anomalies.
    # Skeleton (graph minus all anomalies) is a cograph by construction.
    from ..graphs.k_sum import _combine_chord_iteration, _iterative_chord_rule

    g_skeleton, factors, adds = _iterative_chord_rule(
        graph, anomalies, engine, smart_order=False,
    )

    # The skeleton is a cograph; compute via cotree_dp directly.
    try:
        t_skeleton = compute_tutte_cotree_dp(g_skeleton)
    except (ValueError, TypeError):
        # Defensive: if the skeleton somehow isn't a cograph (greedy
        # elimination edge case), fall back to the engine.
        t_skeleton = engine.synthesize(g_skeleton).polynomial

    return _combine_chord_iteration(t_skeleton, factors, adds)
