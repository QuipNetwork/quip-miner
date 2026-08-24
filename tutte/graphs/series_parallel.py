"""Series-Parallel Graph Recognition and Decomposition.

A 2-terminal series-parallel graph can be reduced to a single edge using:
- Series reduction: Remove degree-2 vertex, connect its neighbors
- Parallel reduction: Merge multiple edges between same endpoints

Series-parallel graphs have treewidth ≤ 2.

This module also provides:
- SPNode: Decomposition tree for series-parallel graphs
- decompose_series_parallel: Build the decomposition tree during reduction
- compute_sp_tutte: Compute Tutte polynomial from decomposition tree in O(n) time
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple, Set
from ..graph import Graph, MultiGraph
from ..polynomial import TuttePolynomial
from ..logs import get_log, EventType, LogLevel


# =============================================================================
# SERIES-PARALLEL DECOMPOSITION TREE
# =============================================================================

@dataclass
class SPNode:
    """Node in series-parallel decomposition tree.

    The tree structure represents how the graph was built:
    - EDGE: A single edge (leaf node)
    - SERIES: Series composition of children (cut vertex)
    - PARALLEL: Parallel composition of children (multi-edges)

    The decomposition is computed during the reduction process by tracking
    which operations were performed.
    """
    type: str  # "EDGE", "SERIES", or "PARALLEL"
    children: List['SPNode'] = field(default_factory=list)
    edge: Optional[Tuple[int, int]] = None  # For EDGE nodes: original edge

    def __repr__(self) -> str:
        if self.type == "EDGE":
            return f"E{self.edge}"
        elif self.type == "SERIES":
            return f"S({', '.join(repr(c) for c in self.children)})"
        else:  # PARALLEL
            return f"P({', '.join(repr(c) for c in self.children)})"

    def edge_count(self) -> int:
        """Count total edges in this subtree."""
        if self.type == "EDGE":
            return 1
        return sum(c.edge_count() for c in self.children)

    def depth(self) -> int:
        """Maximum depth of the tree."""
        if self.type == "EDGE":
            return 0
        return 1 + max(c.depth() for c in self.children)


def is_series_parallel(graph: Graph) -> bool:
    """Check if graph is series-parallel via reduction algorithm. O(n+m)."""
    if graph.edge_count() <= 1:
        return True

    # Adjacency with edge multiplicities
    adj: Dict[int, Dict[int, int]] = {n: {} for n in graph.nodes}
    for u, v in graph.edges:
        adj[u][v] = adj[u].get(v, 0) + 1
        adj[v][u] = adj[v].get(u, 0) + 1

    n_edges = graph.edge_count()

    while n_edges > 1:
        reduced = False

        # Parallel reduction: multiple edges between same pair
        for u in adj:
            for v, count in adj[u].items():
                if count > 1:
                    adj[u][v] = adj[v][u] = 1
                    n_edges -= count - 1
                    reduced = True
                    break
            if reduced:
                break

        if not reduced:
            # Series reduction: degree-2 vertex
            for u in list(adj.keys()):
                neighbors = list(adj[u].keys())
                if len(neighbors) == 2:
                    v, w = neighbors
                    del adj[u]
                    del adj[v][u]
                    del adj[w][u]
                    adj[v][w] = adj[v].get(w, 0) + 1
                    adj[w][v] = adj[w].get(v, 0) + 1
                    n_edges -= 1
                    reduced = True
                    break

        if not reduced:
            return False

    return True


# =============================================================================
# DECOMPOSITION ALGORITHM
# =============================================================================

def decompose_series_parallel(graph: Graph) -> Optional[SPNode]:
    """Build SP decomposition tree during reduction, or None if not SP.

    The algorithm tracks reduction operations and builds the tree bottom-up:
    1. Start with EDGE nodes for each original edge
    2. When we reduce parallel edges, create PARALLEL node
    3. When we reduce a series vertex, create SERIES node
    4. Return the final tree root

    Args:
        graph: Graph to decompose

    Returns:
        SPNode root of decomposition tree, or None if graph is not SP
    """
    if graph.edge_count() == 0:
        # Empty graph - no decomposition
        return None

    if graph.edge_count() == 1:
        # Single edge
        edge = next(iter(graph.edges))
        return SPNode(type="EDGE", edge=edge)

    # Track decomposition nodes for each edge pair
    # edge_trees[(u,v)] = list of SPNode for edges between u and v
    edge_trees: Dict[Tuple[int, int], List[SPNode]] = {}

    # Initialize with EDGE nodes for each original edge
    for u, v in graph.edges:
        key = (min(u, v), max(u, v))
        if key not in edge_trees:
            edge_trees[key] = []
        edge_trees[key].append(SPNode(type="EDGE", edge=(u, v)))

    # Build adjacency with multiplicities
    adj: Dict[int, Dict[int, int]] = {n: {} for n in graph.nodes}
    for u, v in graph.edges:
        adj[u][v] = adj[u].get(v, 0) + 1
        adj[v][u] = adj[v].get(u, 0) + 1

    n_edges = graph.edge_count()

    while n_edges > 1:
        reduced = False

        # Parallel reduction: multiple edges between same pair
        for u in list(adj.keys()):
            if u not in adj:
                continue
            for v, count in list(adj[u].items()):
                if count > 1:
                    # Create PARALLEL node from all edges between u,v
                    key = (min(u, v), max(u, v))
                    children = edge_trees.pop(key)
                    parallel_node = SPNode(type="PARALLEL", children=children)
                    edge_trees[key] = [parallel_node]

                    adj[u][v] = adj[v][u] = 1
                    n_edges -= count - 1
                    reduced = True
                    break
            if reduced:
                break

        if not reduced:
            # Series reduction: degree-2 vertex
            for u in list(adj.keys()):
                if u not in adj:
                    continue
                neighbors = list(adj[u].keys())
                if len(neighbors) == 2:
                    v, w = neighbors

                    # Get decomposition trees for edges (u,v) and (u,w)
                    key_uv = (min(u, v), max(u, v))
                    key_uw = (min(u, w), max(u, w))

                    trees_uv = edge_trees.pop(key_uv)
                    trees_uw = edge_trees.pop(key_uw)

                    # Create SERIES node (composition at cut vertex u)
                    # Each edge contributes one tree
                    series_children = trees_uv + trees_uw
                    series_node = SPNode(type="SERIES", children=series_children)

                    # Remove u from adjacency
                    del adj[u]
                    del adj[v][u]
                    del adj[w][u]

                    # Add new edge (v,w) or increment if exists
                    adj[v][w] = adj[v].get(w, 0) + 1
                    adj[w][v] = adj[w].get(v, 0) + 1

                    # Add the series node as a tree for the new edge
                    key_vw = (min(v, w), max(v, w))
                    if key_vw not in edge_trees:
                        edge_trees[key_vw] = []
                    edge_trees[key_vw].append(series_node)

                    n_edges -= 1
                    reduced = True
                    break

        if not reduced:
            return None  # Not series-parallel

    # Should have exactly one edge with one tree
    if len(edge_trees) != 1:
        return None

    final_trees = list(edge_trees.values())[0]

    if len(final_trees) == 1:
        return final_trees[0]
    else:
        # Multiple trees for final edge -> parallel composition
        return SPNode(type="PARALLEL", children=final_trees)


# =============================================================================
# TUTTE POLYNOMIAL COMPUTATION FROM DECOMPOSITION
# =============================================================================

def compute_sp_tutte(tree: SPNode) -> TuttePolynomial:
    """Compute Tutte polynomial from SP decomposition tree in O(n) time.

    Uses (T, D) pair recursion where:
    - T = Tutte polynomial of the 2-terminal graph
    - D = Tutte polynomial after merging the two terminals

    Args:
        tree: SPNode decomposition tree

    Returns:
        TuttePolynomial for the graph represented by this tree
    """
    T, _ = _sp_TD(tree)
    return T


def _sp_TD(node: SPNode) -> Tuple[TuttePolynomial, TuttePolynomial]:
    """Compute (T, D) for an SP node.

    T = Tutte polynomial of the 2-terminal graph
    D = Tutte polynomial after merging the two terminals

    EDGE: T = x, D = y
    SERIES(A₁,...,Aₖ): T = ∏T(Aᵢ), D = T(ring formed by merging outer terminals)
    PARALLEL(A₁,...,Aₖ): D = ∏D(Aᵢ), T computed by sequential parallel addition
    """
    if node.type == "EDGE":
        return (TuttePolynomial.x(), TuttePolynomial.y())

    elif node.type == "SERIES":
        children = node.children
        # T = product of T(child) via cut vertex factorization
        T = TuttePolynomial.one()
        child_TDs = [_sp_TD(c) for c in children]
        for T_c, _ in child_TDs:
            T = T * T_c

        # D = T(ring) where ring = merging outer terminals of the series
        # Ring = PARALLEL(Aₖ, SERIES(A₁,...,Aₖ₋₁)) at the merged vertex
        # Computed as: start with S(A₁,...,Aₖ₋₁), add Aₖ in parallel
        D = _series_D(children, child_TDs)

        return (T, D)

    elif node.type == "PARALLEL":
        children = node.children
        child_TDs = [_sp_TD(c) for c in children]

        # D = product of D(child) — after merging terminals, each child has
        # its terminals merged independently, sharing only the merged vertex
        D = TuttePolynomial.one()
        for _, D_c in child_TDs:
            D = D * D_c

        # T = add children in parallel sequentially
        T, D_running = child_TDs[0]
        for i, child in enumerate(children[1:], 1):
            T, D_running = _add_parallel(child, T, D_running, child_TDs[i])

        return (T, D)

    else:
        raise ValueError(f"Unknown node type: {node.type}")


def _series_D(
    children: List[SPNode],
    child_TDs: List[Tuple[TuttePolynomial, TuttePolynomial]],
) -> TuttePolynomial:
    """Compute D for a SERIES node = T(ring formed by merging outer terminals).

    Ring = SERIES(A₁,...,Aₖ) with outer terminals merged
         = PARALLEL(Aₖ, SERIES(A₁,...,Aₖ₋₁)) at the merged terminals

    Recursion: D(S(A₁,...,Aₖ)) = add_parallel(Aₖ, T(S(A₁,...,Aₖ₋₁)), D(S(A₁,...,Aₖ₋₁)))[0]
    Base: D(S(A₁)) = D(A₁)
    """
    k = len(children)
    if k == 1:
        return child_TDs[0][1]  # D of the single child

    # T_base = T(S(A₁,...,Aₖ₋₁)) = product of T(Aᵢ) for i < k
    T_base = TuttePolynomial.one()
    for i in range(k - 1):
        T_base = T_base * child_TDs[i][0]

    # D_base = D(S(A₁,...,Aₖ₋₁)) — recursive
    D_base = _series_D(children[:k-1], child_TDs[:k-1])

    # Add Aₖ in parallel to get the ring
    T_ring, _ = _add_parallel(children[k-1], T_base, D_base, child_TDs[k-1])
    return T_ring


def _add_parallel(
    child: SPNode,
    T: TuttePolynomial,
    D: TuttePolynomial,
    child_TD: Tuple[TuttePolynomial, TuttePolynomial],
) -> Tuple[TuttePolynomial, TuttePolynomial]:
    """Add a 2-terminal SP subgraph in parallel to current graph.

    Current graph has (T, D) where D = T(G/{s,t}).
    Child shares terminals s, t with current graph.

    Returns (T_new, D_new) for the combined graph.

    Key formulas:
    - D_new = D × D(child) always (terminals merge independently, cut vertex)
    - EDGE: T_new = T + D (chord formula)
    - SERIES(C₁,...,Cₖ): bridge phase (multiply by T(Cᵢ) for i<k),
      then chord phase (add Cₖ between internal vertex and terminal)
    - PARALLEL(C₁,...,Cⱼ): add each sub-child sequentially
    """
    D_child = child_TD[1]
    D_new = D * D_child

    if child.type == "EDGE":
        T_new = T + D
        return (T_new, D_new)

    elif child.type == "PARALLEL":
        sub_TDs = [_sp_TD(c) for c in child.children]
        T_running = T
        D_running = D
        for i, subchild in enumerate(child.children):
            T_running, D_running = _add_parallel(subchild, T_running, D_running, sub_TDs[i])
        return (T_running, D_new)

    elif child.type == "SERIES":
        children = child.children
        k = len(children)
        sub_TDs = [_sp_TD(c) for c in children]

        # Bridge phase: C₁ through Cₖ₋₁ introduce new internal vertices
        # Each attaches at a cut vertex, so T and D both multiply by T(Cᵢ)
        T_partial = T
        for i in range(k - 1):
            T_partial = T_partial * sub_TDs[i][0]

        # Chord phase: add Cₖ between wₖ₋₁ (internal) and t (terminal)
        # Need D' = T(G_partial/{wₖ₋₁,t})
        # = T(PARALLEL(base, S(C₁,...,Cₖ₋₁)) at merged terminals)
        if k == 1:
            D_prime = D  # No bridge phase, just use current D
        else:
            # Recursively add S(C₁,...,Cₖ₋₁) in parallel to base
            sub_series = SPNode(type="SERIES", children=list(children[:k-1]))
            sub_series_T = TuttePolynomial.one()
            for i in range(k - 1):
                sub_series_T = sub_series_T * sub_TDs[i][0]
            sub_series_D = _series_D(children[:k-1], sub_TDs[:k-1])
            sub_series_TD = (sub_series_T, sub_series_D)

            D_prime, _ = _add_parallel(sub_series, T, D, sub_series_TD)

        # Add Cₖ in parallel using (T_partial, D_prime)
        T_new, _ = _add_parallel(children[k-1], T_partial, D_prime, sub_TDs[k-1])

        return (T_new, D_new)

    else:
        raise ValueError(f"Unknown child type: {child.type}")


# =============================================================================
# CONVENIENCE FUNCTION FOR SYNTHESIS
# =============================================================================

def compute_sp_tutte_if_applicable(graph: Graph) -> Optional[TuttePolynomial]:
    """Try to compute Tutte polynomial using SP decomposition.

    This is the main entry point for synthesis integration.
    Returns None if the graph is not series-parallel.

    Args:
        graph: Graph to compute polynomial for

    Returns:
        TuttePolynomial or None if graph is not SP
    """
    if graph.edge_count() == 0:
        return TuttePolynomial.one()

    tree = decompose_series_parallel(graph)
    if tree is None:
        return None

    _log = get_log()
    _log.record(EventType.SERIES_PARALLEL, "series_parallel",
                f"SP graph: {graph.node_count()}n {graph.edge_count()}e, "
                f"tree depth {tree.depth()}",
                LogLevel.DEBUG, graph=graph)
    return compute_sp_tutte(tree)


# =============================================================================
# MULTIGRAPH SERIES-PARALLEL RECOGNITION AND TUTTE POLYNOMIAL
# =============================================================================

@dataclass
class SPMGNode:
    """Node in series-parallel decomposition tree for multigraphs.

    Like SPNode but EDGE nodes carry a multiplicity.
    """
    type: str  # "EDGE", "SERIES", or "PARALLEL"
    children: List['SPMGNode'] = field(default_factory=list)
    edge: Optional[Tuple[int, int]] = None
    multiplicity: int = 1  # For EDGE nodes: number of parallel copies


def decompose_sp_multigraph(mg: MultiGraph) -> Optional[SPMGNode]:
    """Build SP decomposition tree for a multigraph, or None if not SP.

    Works on the underlying topology (with multiplicities tracked).
    Loops are NOT handled here — they must be stripped before calling.
    """
    if mg.total_loop_count() > 0:
        return None  # Caller should handle loops first

    if not mg.edge_counts:
        return None

    if len(mg.edge_counts) == 1:
        edge, mult = next(iter(mg.edge_counts.items()))
        if mult == 1:
            return SPMGNode(type="EDGE", edge=edge, multiplicity=1)
        # Multiple parallel edges between 2 nodes
        children = [SPMGNode(type="EDGE", edge=edge, multiplicity=1) for _ in range(mult)]
        return SPMGNode(type="PARALLEL", children=children)

    # Build adjacency with multiplicities
    adj: Dict[int, Dict[int, int]] = {n: {} for n in mg.nodes}
    for (u, v), count in mg.edge_counts.items():
        adj[u][v] = count
        adj[v][u] = count

    # Track decomposition trees for each edge pair
    edge_trees: Dict[Tuple[int, int], List[SPMGNode]] = {}
    for (u, v), mult in mg.edge_counts.items():
        key = (min(u, v), max(u, v))
        # Each multiplicity-k edge becomes k EDGE leaves
        edge_trees[key] = [SPMGNode(type="EDGE", edge=(u, v), multiplicity=1) for _ in range(mult)]

    n_edges = sum(mg.edge_counts.values())

    while n_edges > 1:
        reduced = False

        # Parallel reduction: multiple edges/trees between same pair
        for u in list(adj.keys()):
            if u not in adj:
                continue
            for v, count in list(adj[u].items()):
                key = (min(u, v), max(u, v))
                trees = edge_trees.get(key, [])
                if len(trees) > 1:
                    parallel_node = SPMGNode(type="PARALLEL", children=trees)
                    edge_trees[key] = [parallel_node]
                    adj[u][v] = adj[v][u] = 1
                    n_edges -= count - 1
                    reduced = True
                    break
            if reduced:
                break

        if not reduced:
            # Series reduction: degree-2 vertex
            for u in list(adj.keys()):
                if u not in adj:
                    continue
                neighbors = list(adj[u].keys())
                if len(neighbors) == 2:
                    v, w = neighbors

                    key_uv = (min(u, v), max(u, v))
                    key_uw = (min(u, w), max(u, w))

                    trees_uv = edge_trees.pop(key_uv)
                    trees_uw = edge_trees.pop(key_uw)

                    series_children = trees_uv + trees_uw
                    series_node = SPMGNode(type="SERIES", children=series_children)

                    del adj[u]
                    del adj[v][u]
                    del adj[w][u]

                    adj[v][w] = adj[v].get(w, 0) + 1
                    adj[w][v] = adj[w].get(v, 0) + 1

                    key_vw = (min(v, w), max(v, w))
                    if key_vw not in edge_trees:
                        edge_trees[key_vw] = []
                    edge_trees[key_vw].append(series_node)

                    n_edges -= 1
                    reduced = True
                    break

        if not reduced:
            return None  # Not series-parallel

    if len(edge_trees) != 1:
        return None

    final_trees = list(edge_trees.values())[0]

    if len(final_trees) == 1:
        return final_trees[0]
    else:
        return SPMGNode(type="PARALLEL", children=final_trees)


def compute_sp_tutte_multigraph(tree: SPMGNode) -> TuttePolynomial:
    """Compute Tutte polynomial from SP multigraph decomposition tree."""
    T, _ = _sp_mg_TD(tree)
    return T


def _sp_mg_TD(node: SPMGNode) -> Tuple[TuttePolynomial, TuttePolynomial]:
    """Compute (T, D) for an SP multigraph node.

    Same recursion as simple SP, but EDGE base case is always (x, y)
    since multiplicities are handled structurally via PARALLEL nodes.
    """
    if node.type == "EDGE":
        return (TuttePolynomial.x(), TuttePolynomial.y())

    elif node.type == "SERIES":
        children = node.children
        child_TDs = [_sp_mg_TD(c) for c in children]

        T = TuttePolynomial.one()
        for T_c, _ in child_TDs:
            T = T * T_c

        D = _series_mg_D(children, child_TDs)
        return (T, D)

    elif node.type == "PARALLEL":
        children = node.children
        child_TDs = [_sp_mg_TD(c) for c in children]

        D = TuttePolynomial.one()
        for _, D_c in child_TDs:
            D = D * D_c

        T, D_running = child_TDs[0]
        for i in range(1, len(children)):
            T, D_running = _add_parallel_mg(children[i], T, D_running, child_TDs[i])

        return (T, D)

    else:
        raise ValueError(f"Unknown node type: {node.type}")


def _series_mg_D(
    children: List[SPMGNode],
    child_TDs: List[Tuple[TuttePolynomial, TuttePolynomial]],
) -> TuttePolynomial:
    """Compute D for a SERIES multigraph node."""
    k = len(children)
    if k == 1:
        return child_TDs[0][1]

    T_base = TuttePolynomial.one()
    for i in range(k - 1):
        T_base = T_base * child_TDs[i][0]

    D_base = _series_mg_D(children[:k-1], child_TDs[:k-1])

    T_ring, _ = _add_parallel_mg(children[k-1], T_base, D_base, child_TDs[k-1])
    return T_ring


def _add_parallel_mg(
    child: SPMGNode,
    T: TuttePolynomial,
    D: TuttePolynomial,
    child_TD: Tuple[TuttePolynomial, TuttePolynomial],
) -> Tuple[TuttePolynomial, TuttePolynomial]:
    """Add an SP multigraph subgraph in parallel to current graph."""
    D_child = child_TD[1]
    D_new = D * D_child

    if child.type == "EDGE":
        T_new = T + D
        return (T_new, D_new)

    elif child.type == "PARALLEL":
        sub_TDs = [_sp_mg_TD(c) for c in child.children]
        T_running = T
        D_running = D
        for i, subchild in enumerate(child.children):
            T_running, D_running = _add_parallel_mg(subchild, T_running, D_running, sub_TDs[i])
        return (T_running, D_new)

    elif child.type == "SERIES":
        children = child.children
        k = len(children)
        sub_TDs = [_sp_mg_TD(c) for c in children]

        T_partial = T
        for i in range(k - 1):
            T_partial = T_partial * sub_TDs[i][0]

        if k == 1:
            D_prime = D
        else:
            sub_series = SPMGNode(type="SERIES", children=list(children[:k-1]))
            sub_series_T = TuttePolynomial.one()
            for i in range(k - 1):
                sub_series_T = sub_series_T * sub_TDs[i][0]
            sub_series_D = _series_mg_D(children[:k-1], sub_TDs[:k-1])
            sub_series_TD = (sub_series_T, sub_series_D)

            D_prime, _ = _add_parallel_mg(sub_series, T, D, sub_series_TD)

        T_new, _ = _add_parallel_mg(children[k-1], T_partial, D_prime, sub_TDs[k-1])
        return (T_new, D_new)

    else:
        raise ValueError(f"Unknown child type: {child.type}")


def compute_sp_tutte_multigraph_if_applicable(mg: MultiGraph) -> Optional[TuttePolynomial]:
    """Try to compute Tutte polynomial for a multigraph using SP decomposition.

    Handles loops separately, then tries SP decomposition on the loop-free part.
    Returns None if the underlying graph is not series-parallel.
    """
    if not mg.edge_counts and not mg.loop_counts:
        return TuttePolynomial.one()

    # Handle loops: T(G with k loops) = y^k × T(G without loops)
    loop_count = mg.total_loop_count()
    if loop_count > 0:
        mg_no_loops = mg.remove_loops()
        base = compute_sp_tutte_multigraph_if_applicable(mg_no_loops)
        if base is None:
            return None
        return TuttePolynomial.y(loop_count) * base

    if not mg.edge_counts:
        return TuttePolynomial.one()

    tree = decompose_sp_multigraph(mg)
    if tree is None:
        return None

    _log = get_log()
    _log.record(EventType.SERIES_PARALLEL, "series_parallel",
                f"SP multigraph: {mg.node_count()}n {mg.edge_count()}e",
                LogLevel.DEBUG, graph=mg)
    return compute_sp_tutte_multigraph(tree)


# =============================================================================
# CHARACTERISTIC POLYNOMIAL VIA SP DECOMPOSITION
# =============================================================================

def compute_sp_chi_coeffs(graph: Graph) -> Optional[Dict[int, int]]:
    """Compute chi(M(G); q) = (-1)^r * T(1-q, 0) for SP graphs.

    Returns {power: coefficient} or None if graph is not SP.
    Uses compute_sp_tutte() for T, then evaluates at (1-q, 0).
    O(n) time via SP decomposition.
    """
    tutte = compute_sp_tutte_if_applicable(graph)
    if tutte is None:
        return None

    return _chi_from_tutte(tutte, graph)


def _chi_from_tutte(tutte: TuttePolynomial, graph: Graph) -> Dict[int, int]:
    """Compute chi(M; q) from T(M; x, y) via chi(q) = (-1)^r * T(1-q, 0).

    T(1-q, 0) means substitute x = 1-q, y = 0 in T(x, y).
    Only terms with y_power == 0 survive.

    For a term c * x^i * y^j:
      If j > 0: contributes 0 (since y=0)
      If j == 0: contributes c * (1-q)^i

    Then multiply by (-1)^r.
    """
    # Compute rank
    n = graph.node_count()
    # For connected graph, rank = n - 1
    # For general graph, rank = n - components
    # We compute via edge count and y-degree: rank = x_degree of T
    # Actually, for Tutte polynomial T(x,y), the x-degree = n - k where k = components
    # Simpler: rank = number of edges in spanning forest
    # For connected: rank = n-1, for disconnected: rank = n - num_components
    # We can read it from T: rank = max x-power in T
    rank = tutte.x_degree()

    # Collect terms with y^0 (only x terms survive y=0)
    # T(1-q, 0) = sum_{i} c_i * (1-q)^i  where c_i = coefficient(x^i, y^0)
    x_coeffs: Dict[int, int] = {}
    for (i, j), c in tutte.to_coefficients().items():
        if j == 0:
            x_coeffs[i] = c

    if not x_coeffs:
        return {0: 0}

    # Expand sum c_i * (1-q)^i using binomial theorem
    # (1-q)^i = sum_{k=0}^{i} C(i,k) * (-q)^k = sum_{k} C(i,k) * (-1)^k * q^k
    from math import comb

    chi_coeffs: Dict[int, int] = {}
    for i, c_i in x_coeffs.items():
        for k in range(i + 1):
            binom = comb(i, k)
            sign = (-1) ** k
            contribution = c_i * binom * sign
            chi_coeffs[k] = chi_coeffs.get(k, 0) + contribution

    # Multiply by (-1)^r
    sign_r = (-1) ** rank
    result = {k: sign_r * v for k, v in chi_coeffs.items() if v != 0}

    return result


def compute_contraction_chi(
    graph: Graph, flat: FrozenSet[Tuple[int, int]]
) -> Optional[Dict[int, int]]:
    """Compute chi(M(G)/W; q) by contracting flat W in graph G.

    1. Contract flat edges (merge endpoints)
    2. Try SP decomposition on contracted graph -> O(n) chi
    3. If not SP, return None (caller should fall back to lattice chi)
    """
    if not flat:
        return compute_sp_chi_coeffs(graph)

    # Contract flat edges: merge endpoints
    # Build union-find for contraction
    all_nodes = set(graph.nodes)
    parent = {v: v for v in all_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry

    for u, v in flat:
        union(u, v)

    # Build contracted graph
    node_map = {n: find(n) for n in all_nodes}
    new_nodes = frozenset(node_map.values())

    new_edges = set()
    for u, v in graph.edges:
        if (u, v) not in flat:
            nu, nv = node_map[u], node_map[v]
            if nu != nv:
                edge = (min(nu, nv), max(nu, nv))
                new_edges.add(edge)

    if not new_edges:
        # Contracted to a single vertex or no remaining edges
        # chi(point; q) = 1
        return {0: 1}

    contracted_graph = Graph(nodes=new_nodes, edges=frozenset(new_edges))

    # Try SP decomposition first (O(n))
    return compute_sp_chi_coeffs(contracted_graph)
