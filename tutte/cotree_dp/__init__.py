"""Cotree DP for computing Tutte polynomials of cographs.

Implements the algorithm from Giménez, Hliněný & Noy (2006):
"Computing the Tutte Polynomial on Graphs of Bounded Clique-Width"
(Theorem 1.1, Sections 2-3).

A cograph is a P₄-free graph (no induced path on 4 vertices). Every cograph
has a unique cotree — a binary tree of disjoint union (∪) and complete union
(⊗) operations. The algorithm runs DP on this cotree using signature tables
that track component-size distributions of spanning subgraphs.

Complexity: exp(O(vertices^{2/3})) — subexponential in the number of vertices.

Target graph families:
  - Complete graphs K_n (n ≥ 11 where the engine times out)
  - Dense threshold graphs (alternating dominating/isolated vertex additions)
  - Complete bipartite K_{a,b} (for large a, b)
  - Any P₄-free graph

Public API:
    compute_tutte_cotree_dp(graph) -> TuttePolynomial
        Computes the Tutte polynomial if the graph is a cograph.
        Raises ValueError if not a cograph (contains induced P₄).
        Raises TypeError if graph is a MultiGraph.
"""

from .recognition import CotreeNode, CotreeNodeType
from .dp import compute_tutte_cotree_dp
from .almost_cograph import (
    compute_tutte_almost_cograph,
    find_anomaly_edges,
)

__all__ = [
    'CotreeNode',
    'CotreeNodeType',
    'compute_tutte_cotree_dp',
    'compute_tutte_almost_cograph',
    'find_anomaly_edges',
]
