"""Structural fingerprint computation for graph family recognition.

The fingerprint is computed once in O(n+m) and shared across all family detectors,
avoiding redundant graph traversals.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, Optional

from ..graph import Graph


@dataclass(frozen=True)
class StructuralFingerprint:
    """O(n+m) structural fingerprint shared across all family detectors.

    Computed once, then each detector reads fields in O(1).
    The only O(n+m) cost is the bipartiteness BFS.

    Attributes:
        node_count: |V|
        edge_count: |E|
        degree_counts: {degree: count} histogram
        min_degree: minimum vertex degree
        max_degree: maximum vertex degree
        is_bipartite: whether the graph admits a 2-coloring
        is_regular: whether all vertices have the same degree
        regularity: the common degree if regular, None otherwise
    """
    node_count: int
    edge_count: int
    degree_counts: Dict[int, int]
    min_degree: int
    max_degree: int
    is_bipartite: bool
    is_regular: bool
    regularity: Optional[int]


def _check_bipartite(graph: Graph) -> bool:
    """Check if graph is bipartite via BFS 2-coloring.

    Complexity: O(n + m)
    """
    if graph.node_count() == 0:
        return True

    color: Dict[int, int] = {}
    for start in graph.nodes:
        if start in color:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in graph.neighbors(u):
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def compute_structural_fingerprint(graph: Graph) -> StructuralFingerprint:
    """Compute structural fingerprint of a graph.

    Single traversal extracts degree distribution and bipartiteness.
    All per-family detectors use this fingerprint — no redundant traversals.

    Complexity: O(n + m)
        - Degree computation: O(n) — one O(1) lookup per node via cached adjacency
        - Bipartiteness: O(n + m) — single BFS with 2-coloring
        - Aggregation: O(n) — counting degree frequencies

    Args:
        graph: Input graph (simple, undirected).

    Returns:
        StructuralFingerprint with degree distribution and bipartiteness.
    """
    n = graph.node_count()
    m = graph.edge_count()

    if n == 0:
        return StructuralFingerprint(
            node_count=0, edge_count=0, degree_counts={},
            min_degree=0, max_degree=0, is_bipartite=True,
            is_regular=True, regularity=0,
        )

    # O(n): compute degree of each node
    degrees = [graph.degree(v) for v in graph.nodes]
    degree_counts = dict(Counter(degrees))
    min_deg = min(degrees)
    max_deg = max(degrees)

    # O(n + m): BFS 2-coloring for bipartiteness
    bipartite = _check_bipartite(graph)

    is_reg = (min_deg == max_deg)
    reg = min_deg if is_reg else None

    return StructuralFingerprint(
        node_count=n, edge_count=m,
        degree_counts=degree_counts,
        min_degree=min_deg, max_degree=max_deg,
        is_bipartite=bipartite,
        is_regular=is_reg, regularity=reg,
    )