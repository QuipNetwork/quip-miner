"""Graph Minor Detection via BFS Edge Contraction.

This module provides algorithms for checking whether one graph is a minor
of another using subgraph monomorphism and BFS edge contraction.

Key functions:
- is_graph_minor: Check if minor is a graph minor of major
- _find_tree_model: Specialized tree minor detection via branch-set search
"""

from __future__ import annotations

from typing import Optional

from ..graph import Graph


def _find_tree_model(G, T):
    """Check if tree T is a minor of connected graph G via branch-set search.

    Finds vertex-disjoint connected subgraphs of G (branch sets), one per
    node of T, with edges between branch sets of adjacent tree nodes.

    Works by rooting T and assigning branch sets top-down. For each tree node,
    we grow a connected component in G that reaches all children's branch sets.

    Args:
        G: NetworkX graph (the major, must be connected).
        T: NetworkX graph (the minor, must be a tree).

    Returns:
        True/False, or None if search exceeds budget.
    """
    import networkx as nx
    from collections import deque

    t_nodes = list(T.nodes())
    if len(t_nodes) <= 1:
        return True
    if len(t_nodes) > G.number_of_nodes():
        return False

    # Root tree at highest-degree node for best pruning
    root = max(t_nodes, key=lambda v: T.degree(v))

    # Build parent/children structure via BFS from root
    parent = {}
    children = {v: [] for v in t_nodes}
    bfs_order = [root]
    visited_t = {root}
    queue = deque([root])
    while queue:
        v = queue.popleft()
        for w in T.neighbors(v):
            if w not in visited_t:
                visited_t.add(w)
                parent[w] = v
                children[v].append(w)
                bfs_order.append(w)
                queue.append(w)

    # Leaf-to-root order for bottom-up processing
    leaves = [v for v in t_nodes if not children[v]]

    # Use backtracking: assign each tree node a single G-vertex first
    # (checks topological minor), then try expanding branch sets.
    g_nodes = list(G.nodes())
    assignment = {}  # tree_node -> g_vertex
    used = set()
    call_count = [0]
    max_calls = 50000  # budget to avoid combinatorial explosion

    def _backtrack(idx):
        call_count[0] += 1
        if call_count[0] > max_calls:
            return None  # budget exceeded

        if idx == len(bfs_order):
            return True

        t_node = bfs_order[idx]

        if idx == 0:
            # Root: try each G-vertex
            for gv in g_nodes:
                assignment[t_node] = gv
                used.add(gv)
                result = _backtrack(idx + 1)
                if result is True:
                    return True
                if result is None:
                    return None
                used.remove(gv)
                del assignment[t_node]
        else:
            # Non-root: must be adjacent in G to parent's assigned vertex
            p_vertex = assignment[parent[t_node]]
            for gv in G.neighbors(p_vertex):
                if gv not in used:
                    assignment[t_node] = gv
                    used.add(gv)
                    result = _backtrack(idx + 1)
                    if result is True:
                        return True
                    if result is None:
                        return None
                    used.remove(gv)
                    del assignment[t_node]

        return False

    result = _backtrack(0)
    if result is True:
        return True
    if result is None:
        return None  # budget exceeded, inconclusive

    # Single-vertex assignment failed. For trees with max_degree <= 3,
    # topological minor == minor (Mader's theorem), so False is definitive.
    t_max_deg = max(T.degree(v) for v in T.nodes())
    if t_max_deg <= 3:
        return False

    # For higher-degree trees, single-vertex search may miss valid models
    # where branch sets contain multiple vertices. Try shallow contraction:
    # contract each edge of G, then retry topological search on the result.
    # This catches cases like K_{1,4} in Petersen (one contraction creates
    # a degree-4 vertex, making K_{1,4} a subgraph).
    from networkx.algorithms.isomorphism import GraphMatcher as _GM

    max_depth = min(3, G.number_of_nodes() - len(t_nodes))
    if max_depth <= 0:
        return False

    def _graph_key(g):
        return tuple(sorted(tuple(sorted(e)) for e in g.edges()))

    current_level = {_graph_key(G): G}
    for _depth in range(max_depth):
        next_level = {}
        for gkey, g in current_level.items():
            for u, v in list(g.edges()):
                contracted = g.copy()
                for w in list(contracted.neighbors(v)):
                    if w != u:
                        contracted.add_edge(u, w)
                contracted.remove_node(v)
                ckey = _graph_key(contracted)
                if ckey in next_level:
                    continue
                if _GM(contracted, T).subgraph_is_monomorphic():
                    return True
                next_level[ckey] = contracted
        current_level = next_level
        if not current_level:
            break

    return False


def is_graph_minor(major: Graph, minor: Graph, max_contractions: int = 5) -> Optional[bool]:
    """Check if `minor` is a graph minor of `major` via BFS edge contraction.

    Uses subgraph monomorphism (nx.algorithms.isomorphism.GraphMatcher) to
    check whether `minor` can be found as a subgraph of some contraction of
    `major`.

    Args:
        major: The larger graph (potential major).
        minor: The smaller graph (potential minor).
        max_contractions: Maximum number of edge contractions to try.
            If more contractions are needed, returns None (inconclusive).

    Returns:
        True if minor IS a graph minor of major.
        False if minor is NOT a graph minor (exhaustive search within budget).
        None if inconclusive (exceeded max_contractions budget).
    """
    from networkx.algorithms.isomorphism import GraphMatcher

    # Quick reject on node/edge counts
    if minor.node_count() > major.node_count():
        return False
    if minor.edge_count() > major.edge_count():
        return False

    # Structural rules for tree minors
    minor_is_tree = (
        minor.edge_count() == minor.node_count() - 1
        and minor.node_count() >= 1
    )
    if minor_is_tree:
        import networkx as nx
        G_nx = major.to_networkx()
        if nx.is_connected(G_nx):
            minor_max_deg = max(
                (len(minor.neighbors(v)) for v in range(minor.node_count())),
                default=0,
            )
            if minor_max_deg <= 2:
                # Path minor: any connected graph with enough nodes contains it
                return True
            major_max_deg = max(dict(G_nx.degree()).values())
            if major_max_deg <= 2:
                # 2-regular major (cycle/path): contractions never exceed
                # degree 2, so trees with higher max degree can't be minors
                return False
            # Connected major with high enough max degree — use tree model search
            H_nx = minor.to_networkx()
            result = _find_tree_model(G_nx, H_nx)
            if result is not None:
                return result

    G = major.to_networkx() if not minor_is_tree else G_nx
    H = minor.to_networkx() if not minor_is_tree else H_nx

    # Check zero contractions: is H a subgraph of G?
    if GraphMatcher(G, H).subgraph_is_monomorphic():
        return True

    # How many contractions might we need?
    needed = major.node_count() - minor.node_count()
    if needed > max_contractions:
        return None  # Inconclusive — too many contractions to explore

    # BFS contraction: try contracting each edge, deduplicate, check monomorphism
    # Each level = one contraction step
    def _graph_key(g):
        """Canonical key for deduplication: sorted edge tuple."""
        return tuple(sorted(tuple(sorted(e)) for e in g.edges()))

    current_level = {_graph_key(G): G}

    for _depth in range(needed):
        next_level = {}
        for gkey, g in current_level.items():
            for u, v in list(g.edges()):
                # Contract edge (u, v): merge v into u
                contracted = g.copy()
                # Transfer v's neighbors to u
                for w in list(contracted.neighbors(v)):
                    if w != u:
                        contracted.add_edge(u, w)
                contracted.remove_node(v)

                ckey = _graph_key(contracted)
                if ckey in next_level:
                    continue

                # Check monomorphism
                if GraphMatcher(contracted, H).subgraph_is_monomorphic():
                    return True

                next_level[ckey] = contracted

        current_level = next_level
        if not current_level:
            break

    return False
