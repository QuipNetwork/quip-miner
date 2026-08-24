"""Cograph recognition and cotree construction.

A cograph is a graph with no induced P₄ (path on 4 vertices). Equivalently,
it can be built from single vertices using two operations:
  - Disjoint union (∪): two graphs side by side, no edges between them
  - Complete union (⊗): two graphs side by side, ALL edges between them

The cotree is the unique binary tree representing this construction:
  - Leaves are single vertices
  - Internal nodes are ∪ or ⊗

Recognition algorithm: recursive modular decomposition.
  - If G is disconnected: root is ∪, children are connected components
  - If complement(G) is disconnected: root is ⊗, children are co-components
  - If neither G nor complement(G) is disconnected: G is NOT a cograph

Complexity: O(vertices² + vertices × edges) for this recursive implementation.
  (Linear-time O(vertices + edges) algorithms exist via partition refinement,
   but the quadratic version is simpler and sufficient for vertices ≤ 500.)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, List, Optional, Set

from ..graph import Graph


# Maximum vertex count for cograph recognition. The recursive decomposition
# has O(vertices) depth in the worst case (threshold graphs). Python's default
# recursion limit is 1000, so we cap at 500 to leave room for the caller.
_MAX_RECOGNITION_VERTICES = 500


# =============================================================================
# NODE TYPE ENUM
# =============================================================================

class CotreeNodeType(Enum):
    """Type of a node in the cotree decomposition."""
    LEAF = "leaf"
    DISJOINT_UNION_OP = "disjoint_union_op"
    COMPLETE_UNION_OP = "complete_union_op"


# =============================================================================
# COTREE NODE
# =============================================================================

@dataclass
class CotreeNode:
    """Node in the cotree decomposition of a cograph.

    Attributes:
        node_type: LEAF, DISJOINT_UNION_OP (∪), or COMPLETE_UNION_OP (⊗)
        vertex: The vertex label (only for leaf nodes, None for internal nodes)
        children: Child cotree nodes (only for internal nodes, None for leaves)
        vertices: All vertices in the subtree rooted at this node
    """
    node_type: CotreeNodeType
    vertex: Optional[int] = None
    children: Optional[List['CotreeNode']] = None
    vertices: FrozenSet[int] = frozenset()

    def __post_init__(self):
        if not isinstance(self.node_type, CotreeNodeType):
            raise ValueError(
                f"Invalid node_type: {self.node_type!r}. "
                f"Must be a CotreeNodeType enum value "
                f"(LEAF, DISJOINT_UNION_OP, or COMPLETE_UNION_OP)."
            )

    def __repr__(self) -> str:
        if self.node_type == CotreeNodeType.LEAF:
            return f'Leaf({self.vertex})'
        symbol = '∪' if self.node_type == CotreeNodeType.DISJOINT_UNION_OP else '⊗'
        children = self.children or []
        return f'{symbol}({", ".join(repr(child) for child in children)})'

    def size(self) -> int:
        """Number of vertices in this subtree."""
        return len(self.vertices)

    def depth(self) -> int:
        """Maximum depth of the cotree."""
        if self.node_type == CotreeNodeType.LEAF or not self.children:
            return 0
        return 1 + max(child.depth() for child in self.children)


# =============================================================================
# COTREE CONSTRUCTION
# =============================================================================

def _build_cotree(graph: Graph) -> Optional[CotreeNode]:
    """Build the cotree decomposition of a cograph.

    Returns None if the graph is NOT a cograph (contains induced P₄).
    For non-cographs, exits early at the first subgraph where neither
    the graph nor its complement is disconnected.

    Args:
        graph: A simple undirected graph.

    Returns:
        Root CotreeNode of the cotree, or None if not a cograph.

    Raises:
        ValueError: if graph has more than _MAX_RECOGNITION_VERTICES vertices.

    Complexity: O(vertices² + vertices × edges) where vertices = |V|, edges = |E|.
    """
    all_vertices = sorted(graph.nodes)
    num_vertices = len(all_vertices)

    if num_vertices > _MAX_RECOGNITION_VERTICES:
        raise ValueError(
            f"Graph has {num_vertices} vertices; cograph recognition uses "
            f"O(vertices)-depth recursion. "
            f"vertices > {_MAX_RECOGNITION_VERTICES} risks stack overflow."
        )

    if num_vertices == 0:
        return CotreeNode(node_type=CotreeNodeType.DISJOINT_UNION_OP, vertices=frozenset())

    if num_vertices == 1:
        vertex = all_vertices[0]
        return CotreeNode(node_type=CotreeNodeType.LEAF, vertex=vertex, vertices=frozenset({vertex}))

    # Build adjacency dict from the Graph's neighbors() API.
    # This references the same sets already cached in Graph._adj,
    # so no edge data is duplicated.
    adjacency = {vertex: graph.neighbors(vertex) for vertex in all_vertices}

    def _recurse(vertices: List[int]) -> Optional[CotreeNode]:
        """Recursive modular decomposition on a vertex subset.

        At each level:
        1. If disconnected → ∪ node, recurse on each component
        2. If complement is disconnected → ⊗ node, recurse on each co-component
        3. Neither → not a cograph (return None)
        """
        count = len(vertices)

        if count == 0:
            return CotreeNode(node_type=CotreeNodeType.DISJOINT_UNION_OP, vertices=frozenset())

        if count == 1:
            vertex = vertices[0]
            return CotreeNode(
                node_type=CotreeNodeType.LEAF, vertex=vertex, vertices=frozenset({vertex}),
            )

        if count == 2:
            first, second = vertices[0], vertices[1]
            nt = CotreeNodeType.COMPLETE_UNION_OP if second in adjacency[first] else CotreeNodeType.DISJOINT_UNION_OP
            return CotreeNode(
                node_type=nt,
                children=[
                    CotreeNode(
                        node_type=CotreeNodeType.LEAF, vertex=first,
                        vertices=frozenset({first}),
                    ),
                    CotreeNode(
                        node_type=CotreeNodeType.LEAF, vertex=second,
                        vertices=frozenset({second}),
                    ),
                ],
                vertices=frozenset({first, second}),
            )

        vertex_set = set(vertices)

        # Check if induced subgraph is disconnected → ∪ node
        components = _find_components(vertices, vertex_set)
        if len(components) > 1:
            children = []
            collected_vertices: Set[int] = set()
            for component in components:
                child = _recurse(component)
                if child is None:
                    return None
                children.append(child)
                collected_vertices.update(component)
            return CotreeNode(
                node_type=CotreeNodeType.DISJOINT_UNION_OP,
                children=children,
                vertices=frozenset(collected_vertices),
            )

        # Check if complement is disconnected → ⊗ node
        co_components = _find_co_components(vertices)
        if len(co_components) > 1:
            children = []
            collected_vertices = set()
            for component in co_components:
                child = _recurse(component)
                if child is None:
                    return None
                children.append(child)
                collected_vertices.update(component)
            return CotreeNode(
                node_type=CotreeNodeType.COMPLETE_UNION_OP,
                children=children,
                vertices=frozenset(collected_vertices),
            )

        # Neither disconnected nor complement-disconnected → not a cograph
        return None

    # ------------------------------------------------------------------
    # Component-finding helpers (close over adjacency from _build_cotree)
    # ------------------------------------------------------------------

    def _find_components(
        vertices: List[int],
        vertex_set: set,
    ) -> List[List[int]]:
        """Find connected components of the induced subgraph.

        Only considers edges between vertices in vertex_set.
        """
        visited: Set[int] = set()
        components: List[List[int]] = []

        for start_vertex in vertices:
            if start_vertex in visited:
                continue
            component: List[int] = []
            stack = [start_vertex]
            visited.add(start_vertex)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor in vertex_set and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            components.append(component)

        return components

    def _find_co_components(vertices: List[int]) -> List[List[int]]:
        """Find connected components of the complement graph.

        Two vertices are complement-adjacent if they are NOT adjacent
        in the original graph.

        Uses an unvisited-set optimization: each vertex is removed from
        the unvisited set exactly once across all components, giving
        O(vertices + complement_edges) total work.
        """
        unvisited = set(vertices)
        components: List[List[int]] = []

        while unvisited:
            start_vertex = next(iter(unvisited))
            unvisited.remove(start_vertex)
            component = [start_vertex]
            stack = [start_vertex]
            while stack:
                current = stack.pop()
                current_neighbors = adjacency[current]
                still_unvisited: Set[int] = set()
                for candidate in unvisited:
                    if candidate not in current_neighbors:
                        # candidate is a complement-neighbor → same component
                        component.append(candidate)
                        stack.append(candidate)
                    else:
                        # candidate is adjacent in original → different component
                        still_unvisited.add(candidate)
                unvisited = still_unvisited
            components.append(component)

        return components

    return _recurse(all_vertices)
