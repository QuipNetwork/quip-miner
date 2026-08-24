"""Graph Representation for Tutte Polynomial Computation.

This module provides:
1. Graph - Immutable graph for hashing and storage
2. MutableGraph - Mutable graph for incremental construction
3. NetworkX conversion utilities
4. Canonical key generation for isomorphism-invariant identification
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, FrozenSet, Optional, Iterator

import networkx as nx


# =============================================================================
# CANONICAL KEY COMPUTATION (isomorphism-invariant graph hashing)
# =============================================================================

def canonical_node_mapping(G: nx.Graph) -> Dict[int, int]:
    """Return a deterministic mapping ``original_label → canonical_label``
    (0..n-1) for the graph's vertices.

    Uses the same WL refinement as `_compute_canonical_key`. Two
    isomorphic graphs may have DIFFERENT mappings (the canonical labels
    are the same set 0..n-1 but the original→canonical correspondence
    differs because vertex labels differ). The KEY guarantee: applying
    this mapping makes the graph's edge list canonical (same as what
    `_compute_canonical_key` hashes).

    Used by the persistent T_rooted cache (`tutte/roots/rooted_tutte.py`)
    to translate partition keys between graphs with the same
    canonical_key but different original vertex labels.
    """
    return _wl_canonical_mapping(G)


def _wl_canonical_mapping(G: nx.Graph) -> Dict[int, int]:
    """Internal: WL canonical labeling computation. Returns
    {original_label → canonical_label (0..n-1)}.
    """
    if len(G) == 0:
        return {}
    n = len(G)
    colors: Dict[int, int] = {node: hash((G.degree(node),)) for node in G.nodes()}
    for _ in range(n):
        new_colors = {}
        for node in G.nodes():
            neighbor_colors = tuple(sorted(colors[nb] for nb in G.neighbors(node)))
            new_colors[node] = hash((colors[node], neighbor_colors))
        old_partitions = len(set(colors.values()))
        new_partitions = len(set(new_colors.values()))
        colors = new_colors
        if new_partitions == old_partitions:
            break
    color_to_int = {c: i for i, c in enumerate(sorted(set(colors.values())))}
    int_colors = {node: color_to_int[colors[node]] for node in G.nodes()}
    sorted_nodes = sorted(G.nodes(), key=lambda node: int_colors[node])
    node_to_index = {node: i for i, node in enumerate(sorted_nodes)}

    def canonical_sort_key(idx: int) -> tuple:
        node = sorted_nodes[idx]
        color = int_colors[node]
        neighbor_indices = tuple(sorted(
            node_to_index[nb] for nb in G.neighbors(node)
        ))
        return (color, neighbor_indices)

    indices = list(range(len(sorted_nodes)))
    indices.sort(key=canonical_sort_key)
    sorted_nodes = [sorted_nodes[i] for i in indices]
    return {old: new for new, old in enumerate(sorted_nodes)}


def _compute_canonical_key(G: nx.Graph) -> str:
    """Compute a truly canonical key for a NetworkX graph.

    Uses Weisfeiler-Lehman refinement to compute a canonical node ordering,
    then hashes the resulting edge list. This is isomorphism-invariant:
    two isomorphic graphs will always produce the same key.

    This is faster than graph6 encoding and more reliable for cache hits.
    """
    if len(G) == 0:
        return hashlib.sha256(b'empty').hexdigest()

    mapping = _wl_canonical_mapping(G)

    # Build canonical edge list
    edges = []
    for u, v in G.edges():
        new_u, new_v = mapping[u], mapping[v]
        edges.append((min(new_u, new_v), max(new_u, new_v)))
    edges.sort()

    # Hash the canonical form
    canonical_str = f'{len(G)}:{edges}'
    return hashlib.sha256(canonical_str.encode()).hexdigest()


# =============================================================================
# CELL SIGNATURE (for fast structural matching without VF2)
# =============================================================================

@dataclass(frozen=True)
class CellSignature:
    """Structural fingerprint for fast cell matching without VF2.

    This signature captures key graph invariants that must match
    for two graphs to be isomorphic. Checking signatures is O(n log n)
    compared to VF2's exponential worst case.
    """
    node_count: int
    edge_count: int
    degree_sequence: Tuple[int, ...]  # sorted
    triangle_count: int

    def could_match(self, other: 'CellSignature') -> bool:
        """Quick check if two signatures could represent isomorphic graphs."""
        return (self.node_count == other.node_count and
                self.edge_count == other.edge_count and
                self.degree_sequence == other.degree_sequence and
                self.triangle_count == other.triangle_count)


@dataclass(frozen=True)
class NodeSignature:
    """Local structural fingerprint for a single node.

    Used for partitioning nodes into cell groups by matching
    their local structure.
    """
    degree: int
    neighbor_degree_sum: int  # sum of degrees of neighbors
    triangles: int  # number of triangles containing this node

    def distance_to(self, other: 'NodeSignature') -> int:
        """Compute similarity distance between node signatures."""
        return (abs(self.degree - other.degree) +
                abs(self.neighbor_degree_sum - other.neighbor_degree_sum) +
                abs(self.triangles - other.triangles))


def compute_signature(graph: 'Graph') -> CellSignature:
    """Compute structural signature for a graph.

    This is used for fast filtering before expensive VF2 checks.
    """
    degrees = sorted(graph.degree(n) for n in graph.nodes)

    # Count triangles
    triangle_count = 0
    for u, v in graph.edges:
        # Count common neighbors of u and v
        u_neighbors = graph.neighbors(u)
        v_neighbors = graph.neighbors(v)
        triangle_count += len(u_neighbors & v_neighbors)
    triangle_count //= 3  # Each triangle counted 3 times

    return CellSignature(
        node_count=graph.node_count(),
        edge_count=graph.edge_count(),
        degree_sequence=tuple(degrees),
        triangle_count=triangle_count
    )


def compute_node_signature(graph: 'Graph', node: int) -> NodeSignature:
    """Compute local structural signature for a node."""
    degree = graph.degree(node)
    neighbors = graph.neighbors(node)
    neighbor_degree_sum = sum(graph.degree(n) for n in neighbors)

    # Count triangles containing this node
    triangles = 0
    neighbor_list = list(neighbors)
    for i, n1 in enumerate(neighbor_list):
        for n2 in neighbor_list[i+1:]:
            # Check if n1-n2 edge exists
            edge = (min(n1, n2), max(n1, n2))
            if edge in graph.edges:
                triangles += 1

    return NodeSignature(
        degree=degree,
        neighbor_degree_sum=neighbor_degree_sum,
        triangles=triangles
    )


def compute_all_node_signatures(graph: 'Graph') -> Dict[int, NodeSignature]:
    """Compute node signatures for all nodes in the graph."""
    return {node: compute_node_signature(graph, node) for node in graph.nodes}


# =============================================================================
# IMMUTABLE GRAPH CLASS
# =============================================================================

@dataclass(frozen=True)
class Graph:
    """Immutable graph representation for hashing and storage.

    Edges are stored as frozenset of (min, max) tuples to ensure
    canonical representation regardless of edge direction.
    """
    nodes: FrozenSet[int]
    edges: FrozenSet[Tuple[int, int]]

    def __post_init__(self):
        # Validate edges reference valid nodes
        for u, v in self.edges:
            if u not in self.nodes or v not in self.nodes:
                raise ValueError(f"Edge ({u}, {v}) references invalid node")

        # Build adjacency cache (avoids O(m) scan on every neighbors()/degree() call)
        adj: Dict[int, Set[int]] = {n: set() for n in self.nodes}
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
        object.__setattr__(self, '_adj', adj)

    @classmethod
    def from_networkx(cls, G: nx.Graph) -> 'Graph':
        """Convert NetworkX graph to immutable Graph.

        Node labels are converted to sequential integers.
        """
        node_map = {n: i for i, n in enumerate(G.nodes())}
        nodes = frozenset(node_map.values())
        edges = frozenset(
            (min(node_map[u], node_map[v]), max(node_map[u], node_map[v]))
            for u, v in G.edges()
            if u != v  # Skip self-loops in edge representation
        )
        return cls(nodes=nodes, edges=edges)

    @classmethod
    def from_edge_list(cls, edge_list: List[Tuple[int, int]]) -> 'Graph':
        """Create graph from edge list, inferring nodes."""
        nodes = set()
        edges = set()
        for u, v in edge_list:
            nodes.add(u)
            nodes.add(v)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return cls(nodes=frozenset(nodes), edges=frozenset(edges))

    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX graph."""
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        return G

    def fast_hash(self) -> str:
        """Fast structural hash (isomorphism-invariant but not collision-free).

        Uses sorted degree sequence — O(n log n), under 1ms.
        Use as a cheap pre-filter before expensive canonical_key().
        """
        degrees = sorted(self.degree(n) for n in self.nodes)
        return f"{len(self.nodes)}:{self.edge_count()}:{degrees}"

    def canonical_key(self) -> str:
        """Create a canonical string key for the graph (isomorphism-invariant).

        Uses Weisfeiler-Lehman refinement to compute a canonical node ordering,
        then hashes the resulting edge list. This is truly isomorphism-invariant
        and faster than the previous graph6-based approach.

        Result is memoized on the object (frozen dataclass, key never changes).
        """
        cached = getattr(self, '_canonical_key_cache', None)
        if cached is not None:
            return cached
        G = self.to_networkx()
        key = _compute_canonical_key(G)
        object.__setattr__(self, '_canonical_key_cache', key)
        return key

    def node_count(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Number of edges."""
        return len(self.edges)

    def degree(self, node: int) -> int:
        """Get degree of a node."""
        return len(self._adj.get(node, set()))

    def neighbors(self, node: int) -> Set[int]:
        """Get neighbors of a node."""
        return self._adj.get(node, set())

    def subgraph(self, nodes: Set[int]) -> 'Graph':
        """Return induced subgraph on given nodes."""
        new_nodes = frozenset(nodes & self.nodes)
        new_edges = frozenset(
            (u, v) for u, v in self.edges
            if u in new_nodes and v in new_nodes
        )
        return Graph(nodes=new_nodes, edges=new_edges)

    def induced_relabeled(self, relabel: Dict[int, int]) -> 'Graph':
        """Induced subgraph on ``relabel.keys()`` with vertices renamed.

        Keeps only edges whose endpoints are both in ``relabel`` and renames
        each kept edge ``(u, v)`` to ``(relabel[u], relabel[v])``; the node set
        is ``relabel.values()``. Unlike :meth:`subgraph` (which preserves the
        original labels), this relabels — typically to ``0..k-1`` for a cell.
        """
        new_edges: Set[Tuple[int, int]] = set()
        for (u, v) in self.edges:
            if u in relabel and v in relabel:
                a, b = relabel[u], relabel[v]
                new_edges.add((min(a, b), max(a, b)))
        return Graph(
            nodes=frozenset(relabel.values()),
            edges=frozenset(new_edges),
        )

    def edge_induced_subgraph(self, edges: Set[Tuple[int, int]]) -> 'Graph':
        """Return edge-induced subgraph."""
        normalized_edges = set()
        for u, v in edges:
            normalized_edges.add((min(u, v), max(u, v)))

        new_edges = frozenset(self.edges & normalized_edges)
        new_nodes = set()
        for u, v in new_edges:
            new_nodes.add(u)
            new_nodes.add(v)
        return Graph(nodes=frozenset(new_nodes), edges=new_edges)

    def is_connected(self) -> bool:
        """Check if graph is connected."""
        if not self.nodes:
            return True
        if not self.edges:
            return len(self.nodes) <= 1

        visited = set()
        start = next(iter(self.nodes))
        stack = [start]
        visited.add(start)

        while stack:
            node = stack.pop()
            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        return len(visited) == len(self.nodes)

    def connected_components(self) -> List['Graph']:
        """Return list of connected component subgraphs."""
        if not self.nodes:
            return []

        visited = set()
        components = []

        for start in self.nodes:
            if start in visited:
                continue

            component_nodes = set()
            stack = [start]

            while stack:
                node = stack.pop()
                if node in component_nodes:
                    continue
                component_nodes.add(node)
                visited.add(node)
                for neighbor in self.neighbors(node):
                    if neighbor not in component_nodes:
                        stack.append(neighbor)

            components.append(self.subgraph(component_nodes))

        return components

    def is_simple(self) -> bool:
        """True if no parallel edges or loops (always True for Graph)."""
        return True

    def add_edge(self, u: int, v: int) -> 'Graph':
        """Return new graph with an additional edge."""
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Nodes {u} and {v} must exist")
        if u == v:
            raise ValueError("Graph does not support self-loops")
        edge = (min(u, v), max(u, v))
        if edge in self.edges:
            raise ValueError(f"Edge {edge} already exists (use MultiGraph for parallel edges)")
        return Graph(nodes=self.nodes, edges=self.edges | {edge})

    def remove_edge(self, u: int, v: int) -> 'Graph':
        """Return new graph with an edge removed."""
        edge = (min(u, v), max(u, v))
        if edge not in self.edges:
            raise ValueError(f"Edge {edge} not found")
        return Graph(nodes=self.nodes, edges=self.edges - {edge})

    def has_cut_vertex(self) -> Optional[int]:
        """Find a cut vertex (articulation point) if one exists.

        A cut vertex is a vertex whose removal disconnects the graph.
        Returns the first cut vertex found, or None if none exist.
        """
        if len(self.nodes) <= 2:
            return None

        # Use DFS-based articulation point algorithm
        visited = set()
        disc = {}  # Discovery time
        low = {}   # Lowest reachable discovery time
        parent = {}
        ap = set()  # Articulation points
        time = [0]

        def dfs(u: int):
            children = 0
            visited.add(u)
            disc[u] = low[u] = time[0]
            time[0] += 1

            for v in self.neighbors(u):
                if v not in visited:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    # u is articulation point if:
                    # 1) u is root of DFS tree and has 2+ children
                    # 2) u is not root and low[v] >= disc[u]
                    if parent.get(u) is None and children > 1:
                        ap.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])

        # Start DFS from first node
        start = next(iter(self.nodes))
        dfs(start)

        # Return first articulation point, or None
        return next(iter(ap)) if ap else None

    def find_all_cut_vertices(self) -> Set[int]:
        """Find all cut vertices (articulation points).

        Returns set of all cut vertices in the graph.
        """
        if len(self.nodes) <= 2:
            return set()

        visited = set()
        disc = {}
        low = {}
        parent = {}
        ap = set()
        time = [0]

        def dfs(u: int):
            children = 0
            visited.add(u)
            disc[u] = low[u] = time[0]
            time[0] += 1

            for v in self.neighbors(u):
                if v not in visited:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    if parent.get(u) is None and children > 1:
                        ap.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])

        start = next(iter(self.nodes))
        dfs(start)
        return ap

    def split_at_cut_vertex(self, v: int) -> List['Graph']:
        """Split graph into components at cut vertex.

        Returns list of subgraphs, each containing the cut vertex.
        The cut vertex appears in all returned subgraphs.
        """
        if v not in self.nodes:
            raise ValueError(f"Node {v} not in graph")

        # Find components when v is removed
        other_nodes = self.nodes - {v}
        if not other_nodes:
            return [self]

        # Build adjacency for nodes other than v
        visited = set()
        components = []

        for start in other_nodes:
            if start in visited:
                continue

            component_nodes = {v}  # Include cut vertex in each component
            stack = [start]

            while stack:
                node = stack.pop()
                if node in visited or node == v:
                    continue
                visited.add(node)
                component_nodes.add(node)
                for neighbor in self.neighbors(node):
                    if neighbor not in visited and neighbor != v:
                        stack.append(neighbor)

            # Get induced subgraph
            component_edges = frozenset(
                (u, w) for u, w in self.edges
                if u in component_nodes and w in component_nodes
            )
            components.append(Graph(
                nodes=frozenset(component_nodes),
                edges=component_edges
            ))

        return components if len(components) > 1 else [self]

    def merge_nodes(self, u: int, v: int) -> 'MultiGraph':
        """Merge nodes u and v, creating a MultiGraph.

        This operation identifies u and v:
        - Edges from both u and v to the same node w become parallel edges
        - The edge (u, v) if it exists becomes a loop
        - All nodes connected to v become connected to u

        Returns a MultiGraph (which may have parallel edges or loops).
        """
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Nodes {u} and {v} must exist")
        if u == v:
            return MultiGraph.from_graph(self)

        # The surviving node is the smaller one (for consistency)
        survivor = min(u, v)
        removed = max(u, v)

        # Build new node set
        new_nodes = frozenset(n if n != removed else survivor for n in self.nodes) - {removed}

        # Build edge counts (tracking parallel edges and loops)
        edge_counts: Dict[Tuple[int, int], int] = {}
        loop_counts: Dict[int, int] = {}

        for a, b in self.edges:
            # Remap removed node to survivor
            new_a = survivor if a == removed else a
            new_b = survivor if b == removed else b

            if new_a == new_b:
                # This edge becomes a loop
                loop_counts[new_a] = loop_counts.get(new_a, 0) + 1
            else:
                edge = (min(new_a, new_b), max(new_a, new_b))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        return MultiGraph(
            nodes=new_nodes,
            edge_counts=edge_counts,
            loop_counts=loop_counts
        )


# =============================================================================
# MULTIGRAPH CLASS (supports parallel edges and loops)
# =============================================================================

@dataclass(frozen=True)
class MultiGraph:
    """Immutable multigraph that supports parallel edges and loops.

    This class is used when merging nodes creates parallel edges.
    For simple graphs (no parallel edges or loops), use Graph instead.
    """
    nodes: FrozenSet[int]
    edge_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)  # edge -> count
    loop_counts: Dict[int, int] = field(default_factory=dict)  # node -> loop count

    def __post_init__(self):
        # Validate edges reference valid nodes
        for (u, v), count in self.edge_counts.items():
            if u not in self.nodes or v not in self.nodes:
                raise ValueError(f"Edge ({u}, {v}) references invalid node")
            if count <= 0:
                raise ValueError(f"Edge count must be positive")
        for node, count in self.loop_counts.items():
            if node not in self.nodes:
                raise ValueError(f"Loop at node {node} references invalid node")
            if count <= 0:
                raise ValueError(f"Loop count must be positive")

        # Build adjacency cache (avoids O(m) scan on every neighbors() call)
        adj: Dict[int, Set[int]] = {n: set() for n in self.nodes}
        for u, v in self.edge_counts:
            adj[u].add(v)
            adj[v].add(u)
        object.__setattr__(self, '_adj', adj)

    @classmethod
    def from_graph(cls, g: Graph) -> 'MultiGraph':
        """Convert simple Graph to MultiGraph."""
        edge_counts = {e: 1 for e in g.edges}
        return cls(nodes=g.nodes, edge_counts=edge_counts, loop_counts={})

    def is_simple(self) -> bool:
        """True if no parallel edges or loops."""
        if self.loop_counts:
            return False
        return all(c == 1 for c in self.edge_counts.values())

    def to_simple_graph(self) -> Optional[Graph]:
        """Convert to simple Graph if possible (no parallel edges or loops)."""
        if not self.is_simple():
            return None
        return Graph(nodes=self.nodes, edges=frozenset(self.edge_counts.keys()))

    def node_count(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Total number of edges (counting multiplicities)."""
        return sum(self.edge_counts.values()) + sum(self.loop_counts.values())

    def unique_edge_count(self) -> int:
        """Number of unique edges (not counting multiplicities)."""
        return len(self.edge_counts) + len(self.loop_counts)

    def total_loop_count(self) -> int:
        """Total number of loops."""
        return sum(self.loop_counts.values())

    def degree(self, node: int) -> int:
        """Get degree of a node (loops count as 2 each)."""
        deg = 0
        for nb in self._adj.get(node, set()):
            edge = (min(node, nb), max(node, nb))
            deg += self.edge_counts.get(edge, 0)
        deg += 2 * self.loop_counts.get(node, 0)
        return deg

    def neighbors(self, node: int) -> Set[int]:
        """Get neighbors of a node (not counting multiplicities)."""
        return self._adj.get(node, set())

    def edge_multiplicity(self, u: int, v: int) -> int:
        """Get number of edges between u and v."""
        if u == v:
            return self.loop_counts.get(u, 0)
        edge = (min(u, v), max(u, v))
        return self.edge_counts.get(edge, 0)

    def has_cut_vertex(self) -> Optional[int]:
        """Find a cut vertex if one exists (ignoring multiplicities)."""
        if len(self.nodes) <= 2:
            return None

        # Convert to simple adjacency for cut vertex detection
        visited = set()
        disc = {}
        low = {}
        parent = {}
        ap = set()
        time = [0]

        def dfs(u: int):
            children = 0
            visited.add(u)
            disc[u] = low[u] = time[0]
            time[0] += 1

            for v in self.neighbors(u):
                if v not in visited:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    if parent.get(u) is None and children > 1:
                        ap.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])

        if self.nodes:
            start = next(iter(self.nodes))
            dfs(start)

        return next(iter(ap)) if ap else None

    def split_at_cut_vertex(self, v: int) -> List['MultiGraph']:
        """Split multigraph into components at cut vertex."""
        if v not in self.nodes:
            raise ValueError(f"Node {v} not in graph")

        other_nodes = self.nodes - {v}
        if not other_nodes:
            return [self]

        visited = set()
        components = []

        for start in other_nodes:
            if start in visited:
                continue

            component_nodes = {v}
            stack = [start]

            while stack:
                node = stack.pop()
                if node in visited or node == v:
                    continue
                visited.add(node)
                component_nodes.add(node)
                for neighbor in self.neighbors(node):
                    if neighbor not in visited and neighbor != v:
                        stack.append(neighbor)

            # Get induced subgraph edges
            component_edge_counts = {
                (u, w): c for (u, w), c in self.edge_counts.items()
                if u in component_nodes and w in component_nodes
            }
            component_loop_counts = {
                n: c for n, c in self.loop_counts.items()
                if n in component_nodes
            }
            components.append(MultiGraph(
                nodes=frozenset(component_nodes),
                edge_counts=component_edge_counts,
                loop_counts=component_loop_counts
            ))

        return components if len(components) > 1 else [self]

    def remove_loops(self) -> 'MultiGraph':
        """Return new multigraph with all loops removed."""
        return MultiGraph(
            nodes=self.nodes,
            edge_counts=dict(self.edge_counts),
            loop_counts={}
        )

    def is_connected(self) -> bool:
        """Check if the multigraph is connected."""
        if not self.nodes:
            return True
        if not self.edge_counts and not self.loop_counts:
            return len(self.nodes) <= 1

        visited = set()
        start = next(iter(self.nodes))
        stack = [start]
        visited.add(start)

        while stack:
            node = stack.pop()
            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        return len(visited) == len(self.nodes)

    def connected_components(self) -> List['MultiGraph']:
        """Return list of connected component subgraphs as MultiGraphs."""
        if not self.nodes:
            return []

        visited = set()
        components: List['MultiGraph'] = []

        for start in self.nodes:
            if start in visited:
                continue

            component_nodes = set()
            stack = [start]

            while stack:
                node = stack.pop()
                if node in component_nodes:
                    continue
                component_nodes.add(node)
                visited.add(node)
                for neighbor in self.neighbors(node):
                    if neighbor not in component_nodes:
                        stack.append(neighbor)

            comp_edge_counts = {
                (u, w): c for (u, w), c in self.edge_counts.items()
                if u in component_nodes and w in component_nodes
            }
            comp_loop_counts = {
                n: c for n, c in self.loop_counts.items()
                if n in component_nodes
            }
            components.append(MultiGraph(
                nodes=frozenset(component_nodes),
                edge_counts=comp_edge_counts,
                loop_counts=comp_loop_counts,
            ))

        return components

    def in_same_component(self, u: int, v: int) -> bool:
        """Check if nodes u and v are in the same connected component."""
        if u == v:
            return True
        visited = {u}
        stack = [u]
        while stack:
            node = stack.pop()
            for neighbor in self.neighbors(node):
                if neighbor == v:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return False

    def merge_nodes(self, u: int, v: int) -> 'MultiGraph':
        """Merge nodes u and v."""
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Nodes {u} and {v} must exist")
        if u == v:
            return self

        survivor = min(u, v)
        removed = max(u, v)

        new_nodes = frozenset(n if n != removed else survivor for n in self.nodes) - {removed}

        new_edge_counts: Dict[Tuple[int, int], int] = {}
        new_loop_counts: Dict[int, int] = dict(self.loop_counts)

        # Handle loops at removed node
        if removed in new_loop_counts:
            new_loop_counts[survivor] = new_loop_counts.get(survivor, 0) + new_loop_counts.pop(removed)

        for (a, b), count in self.edge_counts.items():
            new_a = survivor if a == removed else a
            new_b = survivor if b == removed else b

            if new_a == new_b:
                new_loop_counts[new_a] = new_loop_counts.get(new_a, 0) + count
            else:
                edge = (min(new_a, new_b), max(new_a, new_b))
                new_edge_counts[edge] = new_edge_counts.get(edge, 0) + count

        return MultiGraph(
            nodes=new_nodes,
            edge_counts=new_edge_counts,
            loop_counts=new_loop_counts
        )

    def is_just_parallel_edges(self) -> bool:
        """True if graph is just parallel edges between 2 nodes (no loops)."""
        if self.loop_counts:
            return False
        if len(self.nodes) != 2:
            return False
        if len(self.edge_counts) != 1:
            return False
        return True

    def parallel_edge_count(self) -> int:
        """For a graph that is just parallel edges, return the count."""
        if not self.is_just_parallel_edges():
            return 0
        return next(iter(self.edge_counts.values()))

    def fast_hash(self) -> str:
        """Fast structural hash (isomorphism-invariant but not collision-free).

        Uses sorted degree sequence with edge multiplicities and loops.
        O(n log n), under 1ms. Use as a cheap pre-filter before canonical_key().
        """
        degrees = sorted(self.degree(n) for n in self.nodes)
        mults = sorted(self.edge_counts.values())
        loops = self.total_loop_count()
        return f"MG:{len(self.nodes)}:{self.edge_count()}:{degrees}:{mults}:{loops}"

    def canonical_key(self) -> str:
        """Create a canonical key for the multigraph (isomorphism-invariant).

        Uses WL refinement for node ordering, then encodes edge multiplicities
        and loops in canonical form.

        When the multigraph is actually simple (no parallel edges, no loops),
        delegates to `Graph.canonical_key()` so that ``MultiGraph.from_graph(g)``
        and ``g`` produce the same key. Keeps the rainbow table and visualizer
        Contributing-Graphs dedupe consistent across representations.

        Result is memoized on the object (frozen dataclass, key never changes).
        """
        cached = getattr(self, '_canonical_key_cache', None)
        if cached is not None:
            return cached

        # Fast path: if the multigraph is structurally simple, use the
        # simple-Graph canonical key so both representations match.
        if not self.loop_counts and all(
            mult == 1 for mult in self.edge_counts.values()
        ):
            simple = Graph(
                nodes=self.nodes,
                edges=frozenset(self.edge_counts.keys()),
            )
            key = simple.canonical_key()
            object.__setattr__(self, '_canonical_key_cache', key)
            return key

        if not self.nodes:
            key = hashlib.sha256(b'empty').hexdigest()
            object.__setattr__(self, '_canonical_key_cache', key)
            return key

        n = len(self.nodes)

        # Build adjacency info including multiplicities and loops
        def node_signature(node: int) -> tuple:
            """Compute initial signature for a node."""
            degree = sum(self.edge_counts.get((min(node, nb), max(node, nb)), 0)
                        for nb in self.neighbors(node))
            loops = self.loop_counts.get(node, 0)
            return (degree, loops)

        # Initialize colors
        colors: Dict[int, tuple] = {node: node_signature(node) for node in self.nodes}

        # WL refinement
        for _ in range(n):
            new_colors = {}
            for node in self.nodes:
                neighbor_info = []
                for nb in self.neighbors(node):
                    edge = (min(node, nb), max(node, nb))
                    mult = self.edge_counts.get(edge, 0)
                    neighbor_info.append((colors[nb], mult))
                neighbor_info = tuple(sorted(neighbor_info))
                new_colors[node] = (colors[node], neighbor_info)

            old_parts = len(set(colors.values()))
            new_parts = len(set(new_colors.values()))
            colors = new_colors
            if new_parts == old_parts:
                break

        # Sort nodes by color
        color_to_int = {c: i for i, c in enumerate(sorted(set(colors.values())))}
        int_colors = {node: color_to_int[colors[node]] for node in self.nodes}
        sorted_nodes = sorted(self.nodes, key=lambda n: int_colors[n])

        # Tie-breaking using neighbor structure
        node_to_index = {node: i for i, node in enumerate(sorted_nodes)}

        def canonical_sort_key(idx: int) -> tuple:
            node = sorted_nodes[idx]
            color = int_colors[node]
            neighbor_indices = []
            for nb in self.neighbors(node):
                edge = (min(node, nb), max(node, nb))
                mult = self.edge_counts.get(edge, 0)
                neighbor_indices.append((node_to_index[nb], mult))
            return (color, tuple(sorted(neighbor_indices)))

        indices = list(range(len(sorted_nodes)))
        indices.sort(key=canonical_sort_key)
        sorted_nodes = [sorted_nodes[i] for i in indices]

        # Create canonical mapping
        mapping = {old: new for new, old in enumerate(sorted_nodes)}

        # Build canonical edge list with multiplicities
        edges = []
        for (u, v), mult in self.edge_counts.items():
            new_u, new_v = mapping[u], mapping[v]
            edges.append((min(new_u, new_v), max(new_u, new_v), mult))
        edges.sort()

        # Build canonical loop list
        loops = []
        for node, count in self.loop_counts.items():
            loops.append((mapping[node], count))
        loops.sort()

        # Hash canonical form
        canonical_str = f'MG:{n}:e{edges}:l{loops}'
        key = hashlib.sha256(canonical_str.encode()).hexdigest()
        object.__setattr__(self, '_canonical_key_cache', key)
        return key


# =============================================================================
# GRAPH BUILDERS FOR COMMON STRUCTURES
# =============================================================================

def complete_graph(n: int) -> Graph:
    """Create complete graph K_n."""
    if n <= 0:
        return Graph(nodes=frozenset(), edges=frozenset())
    nodes = frozenset(range(n))
    edges = frozenset(
        (i, j) for i in range(n) for j in range(i + 1, n)
    )
    return Graph(nodes=nodes, edges=edges)


def cycle_graph(n: int) -> Graph:
    """Create cycle graph C_n."""
    if n <= 0:
        return Graph(nodes=frozenset(), edges=frozenset())
    if n == 1:
        return Graph(nodes=frozenset([0]), edges=frozenset())
    if n == 2:
        return Graph(nodes=frozenset([0, 1]), edges=frozenset([(0, 1)]))
    nodes = frozenset(range(n))
    edges = frozenset(
        (i, (i + 1) % n) if i < (i + 1) % n else ((i + 1) % n, i)
        for i in range(n)
    )
    return Graph(nodes=nodes, edges=edges)


def path_graph(n: int) -> Graph:
    """Create path graph P_n with n nodes."""
    if n <= 0:
        return Graph(nodes=frozenset(), edges=frozenset())
    if n == 1:
        return Graph(nodes=frozenset([0]), edges=frozenset())
    nodes = frozenset(range(n))
    edges = frozenset((i, i + 1) for i in range(n - 1))
    return Graph(nodes=nodes, edges=edges)


def star_graph(n: int) -> Graph:
    """Create star graph S_n (center + n leaves)."""
    if n <= 0:
        return Graph(nodes=frozenset([0]), edges=frozenset())
    nodes = frozenset(range(n + 1))
    edges = frozenset((0, i) for i in range(1, n + 1))
    return Graph(nodes=nodes, edges=edges)


def wheel_graph(n: int) -> Graph:
    """Create wheel graph W_n (center + n-cycle rim)."""
    if n < 3:
        raise ValueError("Wheel graph requires at least 3 rim nodes")
    nodes = frozenset(range(n + 1))  # 0 is center, 1..n are rim
    # Rim edges
    rim_edges = frozenset(
        (i, i + 1) if i < n else (1, n)
        for i in range(1, n + 1)
    )
    # Spoke edges
    spoke_edges = frozenset((0, i) for i in range(1, n + 1))
    return Graph(nodes=nodes, edges=rim_edges | spoke_edges)


def grid_graph(m: int, n: int) -> Graph:
    """Create m x n grid graph."""
    if m <= 0 or n <= 0:
        return Graph(nodes=frozenset(), edges=frozenset())

    def node_id(i, j):
        return i * n + j

    nodes = frozenset(node_id(i, j) for i in range(m) for j in range(n))
    edges = set()

    # Horizontal edges
    for i in range(m):
        for j in range(n - 1):
            edges.add((node_id(i, j), node_id(i, j + 1)))

    # Vertical edges
    for i in range(m - 1):
        for j in range(n):
            edges.add((node_id(i, j), node_id(i + 1, j)))

    return Graph(nodes=nodes, edges=frozenset(edges))


def petersen_graph() -> Graph:
    """Create the Petersen graph."""
    G = nx.petersen_graph()
    return Graph.from_networkx(G)


def disjoint_union(g1: Graph, g2: Graph) -> Graph:
    """Create disjoint union of two graphs.

    Nodes in g2 are relabeled to avoid collision.
    T(G1 ∪ G2) = T(G1) × T(G2)
    """
    # Relabel g2 nodes
    offset = max(g1.nodes) + 1 if g1.nodes else 0
    g2_nodes = frozenset(n + offset for n in g2.nodes)
    g2_edges = frozenset((u + offset, v + offset) for u, v in g2.edges)

    return Graph(
        nodes=g1.nodes | g2_nodes,
        edges=g1.edges | g2_edges
    )


def parallel_connection_graph(g1: Graph, g2: Graph, shared_edge: Tuple[int, int]) -> Graph:
    """Build P_N(G1, G2) by identifying a shared edge.

    Relabels g2 nodes to avoid collision with g1,
    identifies the shared edge endpoints, keeps the shared edge.

    Args:
        g1: First graph
        g2: Second graph
        shared_edge: Edge (u, v) present in both graphs. The endpoints of
            this edge in g2 are mapped to the corresponding endpoints in g1.

    Returns:
        The parallel connection graph P_N(G1, G2).
    """
    u, v = shared_edge
    if u not in g1.nodes or v not in g1.nodes:
        raise ValueError(f"Shared edge ({u}, {v}) not in g1")

    # Find the shared edge in g2 (try both orientations)
    e_g2 = (min(u, v), max(u, v))
    if e_g2 not in g2.edges:
        raise ValueError(f"Shared edge {e_g2} not in g2")

    # Relabel g2 nodes, mapping shared endpoints to g1's
    offset = max(g1.nodes) + 1 if g1.nodes else 0
    g2_node_map = {u: u, v: v}  # Keep shared endpoints
    for n in g2.nodes:
        if n not in g2_node_map:
            g2_node_map[n] = n + offset

    g2_nodes = frozenset(g2_node_map[n] for n in g2.nodes)
    g2_edges = set()
    for a, b in g2.edges:
        na, nb = g2_node_map[a], g2_node_map[b]
        g2_edges.add((min(na, nb), max(na, nb)))

    return Graph(
        nodes=g1.nodes | g2_nodes,
        edges=g1.edges | frozenset(g2_edges),
    )


def k_sum_graph(g1: Graph, g2: Graph, k: int, shared_vertices: List[int]) -> Graph:
    """Build G1 ⊕_k G2 by identifying k vertices, deleting shared clique edges.

    Args:
        g1: First graph (must contain all shared_vertices)
        g2: Second graph (must contain all shared_vertices)
        k: Number of shared vertices
        shared_vertices: List of k vertex labels present in both graphs.
            These vertices are identified, and all edges of the K_k clique
            among them are deleted from the result.

    Returns:
        The k-sum graph G1 ⊕_k G2.
    """
    if len(shared_vertices) != k:
        raise ValueError(f"Expected {k} shared vertices, got {len(shared_vertices)}")

    shared_set = set(shared_vertices)
    for sv in shared_vertices:
        if sv not in g1.nodes:
            raise ValueError(f"Shared vertex {sv} not in g1")
        if sv not in g2.nodes:
            raise ValueError(f"Shared vertex {sv} not in g2")

    # Relabel g2 nodes, keeping shared vertices
    offset = max(g1.nodes) + 1 if g1.nodes else 0
    g2_node_map = {sv: sv for sv in shared_vertices}
    for n in g2.nodes:
        if n not in g2_node_map:
            g2_node_map[n] = n + offset

    g2_nodes = frozenset(g2_node_map[n] for n in g2.nodes)
    g2_edges = set()
    for a, b in g2.edges:
        na, nb = g2_node_map[a], g2_node_map[b]
        g2_edges.add((min(na, nb), max(na, nb)))

    all_nodes = g1.nodes | g2_nodes
    all_edges = g1.edges | frozenset(g2_edges)

    # Delete shared clique edges (K_k among shared vertices)
    clique_edges = set()
    sv_list = sorted(shared_vertices)
    for i in range(len(sv_list)):
        for j in range(i + 1, len(sv_list)):
            clique_edges.add((sv_list[i], sv_list[j]))

    all_edges = all_edges - frozenset(clique_edges)

    return Graph(nodes=all_nodes, edges=all_edges)


def cut_vertex_join(g1: Graph, v1: int, g2: Graph, v2: int) -> Graph:
    """Join two graphs at a cut vertex (1-sum).

    Identifies vertex v1 in g1 with vertex v2 in g2.
    T(G) = T(G1) × T(G2) when the shared vertex is a cut vertex.
    """
    if v1 not in g1.nodes or v2 not in g2.nodes:
        raise ValueError("Vertices must exist in their respective graphs")

    # Relabel g2 nodes, mapping v2 to v1
    offset = max(g1.nodes) + 1 if g1.nodes else 0
    g2_node_map = {v2: v1}
    for n in g2.nodes:
        if n != v2:
            g2_node_map[n] = n + offset

    g2_nodes = frozenset(g2_node_map[n] for n in g2.nodes)
    g2_edges = frozenset(
        (min(g2_node_map[u], g2_node_map[v]), max(g2_node_map[u], g2_node_map[v]))
        for u, v in g2.edges
    )

    return Graph(
        nodes=g1.nodes | g2_nodes,
        edges=g1.edges | g2_edges
    )
