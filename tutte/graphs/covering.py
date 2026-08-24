"""Disjoint Cover Algorithms for Graph Synthesis.

This module provides algorithms for covering a graph with disjoint
copies of known minors from the rainbow table. This is a key component
of the creation-expansion-join algorithm.

Key concepts:
- Tile: A mapping of a minor graph onto a subgraph
- Cover: A collection of non-overlapping tiles
- Fringe: Edges in the cover that don't exist in the input (over-coverage)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms import isomorphism

from ..graph import (CellSignature, Graph, MultiGraph, NodeSignature,
                     compute_all_node_signatures, compute_node_signature,
                     compute_signature)
from ..logs import EventType, LogLevel, get_log
from ..lookup.core import MinorEntry, RainbowTable
from ..polynomial import TuttePolynomial

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Tile:
    """A mapping of a minor graph onto a portion of the target graph.

    The tile represents placing a copy of the minor at specific nodes
    and edges in the target graph.
    """
    minor: MinorEntry
    node_mapping: Dict[int, int]  # minor_node -> target_node
    edge_mapping: Dict[Tuple[int, int], Tuple[int, int]]  # minor_edge -> target_edge

    @property
    def covered_nodes(self) -> Set[int]:
        """Nodes in target graph covered by this tile."""
        return set(self.node_mapping.values())

    @property
    def covered_edges(self) -> Set[Tuple[int, int]]:
        """Edges in target graph covered by this tile."""
        return set(self.edge_mapping.values())

    @property
    def minor_nodes(self) -> Set[int]:
        """Nodes in the minor graph."""
        return set(self.node_mapping.keys())

    @property
    def minor_edges(self) -> Set[Tuple[int, int]]:
        """Edges in the minor graph."""
        return set(self.edge_mapping.keys())


@dataclass
class Cover:
    """A collection of non-overlapping tiles covering a graph."""
    tiles: List[Tile] = field(default_factory=list)
    covered_nodes: Set[int] = field(default_factory=set)
    covered_edges: Set[Tuple[int, int]] = field(default_factory=set)
    uncovered_edges: Set[Tuple[int, int]] = field(default_factory=set)

    def is_complete(self) -> bool:
        """Check if all edges are covered."""
        return len(self.uncovered_edges) == 0

    def add_tile(self, tile: Tile) -> bool:
        """Add a tile if it doesn't overlap with existing tiles.

        Returns True if tile was added, False if it overlaps.
        """
        # Check for node overlap
        if tile.covered_nodes & self.covered_nodes:
            return False

        # Check for edge overlap
        if tile.covered_edges & self.covered_edges:
            return False

        # Add tile
        self.tiles.append(tile)
        self.covered_nodes.update(tile.covered_nodes)
        self.covered_edges.update(tile.covered_edges)
        self.uncovered_edges -= tile.covered_edges

        return True

    def total_tiles(self) -> int:
        """Number of tiles in cover."""
        return len(self.tiles)


@dataclass
class Fringe:
    """Edges in the cover that don't exist in the input (over-coverage).

    When we tile a graph with minors, sometimes the tiling includes
    edges that aren't in the original graph. These are "fringe" edges
    that need to be handled in the synthesis.
    """
    edges: Set[Tuple[int, int]] = field(default_factory=set)
    nodes: Set[int] = field(default_factory=set)

    def is_empty(self) -> bool:
        """Check if there are no fringe edges."""
        return len(self.edges) == 0

    def as_graph(self) -> Graph:
        """Convert fringe edges to a Graph."""
        if not self.edges:
            return Graph(nodes=frozenset(), edges=frozenset())
        return Graph(nodes=frozenset(self.nodes), edges=frozenset(self.edges))

    def edge_count(self) -> int:
        """Number of fringe edges."""
        return len(self.edges)


# =============================================================================
# SUBGRAPH ISOMORPHISM
# =============================================================================

def _wl_color_multiset(g: Graph, rounds: int = 2) -> "Counter":
    """Compute multiset of WL colors after `rounds` of 1-WL refinement.

    Round 0: color = degree.
    Round k: color = (old_color, sorted multiset of neighbor old_colors).

    Used by `can_pattern_fit_by_wl` as a stronger structural filter than
    pure degree multiset. WL colors are STANDALONE — computed inside the
    graph itself, not relative to a containing target.
    """
    from collections import Counter

    nx_g = g.to_networkx()
    nodes = list(nx_g.nodes())
    colors = {v: nx_g.degree(v) for v in nodes}
    for _ in range(max(0, rounds - 1)):
        new = {}
        for v in nodes:
            new[v] = (colors[v], tuple(sorted(colors[u] for u in nx_g.neighbors(v))))
        if Counter(new.values()) == Counter(colors.values()):
            break  # fixed point
        colors = new
    return Counter(colors.values())


def can_pattern_fit_by_wl(
    target: Graph, pattern: Graph, rounds: int = 2
) -> bool:
    """WL-based structural pre-filter — is `pattern` POSSIBLY a subgraph of `target`?

    Necessary condition for subgraph isomorphism:
      - count(P-nodes with WL color c) ≤ count(T-nodes with WL color compatible with c)

    Standalone WL is not sufficient (P's WL colors are computed inside P,
    not inside T), so we use the **degree-monotone variant**: a P-node of
    degree d maps to a T-node of degree ≥ d. We apply this round-by-round
    using WL multisets, comparing cumulative counts at each degree
    threshold.

    Returns True if pattern *could* fit (caller still runs VF2 to confirm).
    Returns False if pattern *definitely* cannot fit. O((n+m)·rounds).
    """
    from collections import Counter

    if pattern.node_count() > target.node_count():
        return False
    if pattern.edge_count() > target.edge_count():
        return False

    # Round 1 (degree): cheapest, catches gross mismatches.
    t_degs = Counter()
    for v in target.nodes:
        d = sum(1 for e in target.edges if v in e)
        t_degs[d] += 1
    p_degs = Counter()
    for v in pattern.nodes:
        d = sum(1 for e in pattern.edges if v in e)
        p_degs[d] += 1

    all_thresholds = sorted(set(t_degs) | set(p_degs), reverse=True)
    t_cum = 0
    p_cum = 0
    for d in all_thresholds:
        t_cum += t_degs.get(d, 0)
        p_cum += p_degs.get(d, 0)
        if p_cum > t_cum:
            return False

    # Round 2+ (degree + sorted neighbor degrees): stronger, catches
    # "wrong local structure" cases the degree filter alone passes.
    if rounds < 2:
        return True
    t_wl = _wl_color_multiset(target, rounds=rounds)
    p_wl = _wl_color_multiset(pattern, rounds=rounds)

    # Each P-color must map to a "compatible" T-color. A WL color is
    # (deg, sorted_nbr_degs). T's compatible colors have deg' ≥ deg AND
    # for each pattern neighbor-degree d, T-nbr-multiset has a sufficient
    # count of degrees ≥ d (Hall-condition style on neighbor-degree).
    def t_compat_count(p_color) -> int:
        if isinstance(p_color, int):
            p_deg = p_color
            p_nbr_degs = ()
        else:
            p_deg, p_nbr_degs = p_color
        p_nbr_sorted = sorted(p_nbr_degs, reverse=True)
        count = 0
        for t_color, t_n in t_wl.items():
            if isinstance(t_color, int):
                t_deg = t_color
                t_nbr_degs = ()
            else:
                t_deg, t_nbr_degs = t_color
            if t_deg < p_deg:
                continue
            t_nbr_sorted = sorted(t_nbr_degs, reverse=True)
            if len(t_nbr_sorted) < len(p_nbr_sorted):
                continue
            ok = True
            for i, pd in enumerate(p_nbr_sorted):
                if t_nbr_sorted[i] < pd:
                    ok = False
                    break
            if ok:
                count += t_n
        return count

    # For each P-color, total T-capacity must accommodate it. Strong
    # version: ordered Hall — sort P-colors by required capacity desc,
    # greedily assign. Approximation: just check per-color capacity
    # against single P-class size.
    for p_color, p_n in p_wl.items():
        if t_compat_count(p_color) < p_n:
            return False
    return True


# Back-compat alias for callers using the older degree-only name.
def can_pattern_fit_by_degree_multiset(target: Graph, pattern: Graph) -> bool:
    return can_pattern_fit_by_wl(target, pattern, rounds=1)


def find_subgraph_isomorphisms(
    target: Graph,
    pattern: Graph,
    max_matches: int = 100,
    max_search_time_s: Optional[float] = None,
) -> List[Dict[int, int]]:
    """Find all subgraph isomorphisms of pattern in target.

    Uses NetworkX's VF2 algorithm for subgraph isomorphism.

    Args:
        target: Graph to search in
        pattern: Pattern graph to find
        max_matches: Maximum number of matches to return
        max_search_time_s: Optional wall-clock budget. The deadline is
            checked BETWEEN yields from VF2's iterator. VF2 cannot be
            interrupted mid-yield, so this only bounds total search if
            VF2 yields periodically — fine for dense symmetric targets
            (Z(1,3) yields ~900 K_{4,4} subgraphs quickly) but a hard
            hang between yields will not be interrupted. Default None
            preserves legacy behavior.

    Returns:
        List of node mappings {pattern_node: target_node}
    """
    G_target = target.to_networkx()
    G_pattern = pattern.to_networkx()

    matcher = isomorphism.GraphMatcher(G_target, G_pattern)

    deadline = (time.time() + max_search_time_s) if max_search_time_s else None
    matches = []
    for mapping in matcher.subgraph_isomorphisms_iter():
        # mapping is {target_node: pattern_node}, we want the inverse
        inverse_mapping = {v: k for k, v in mapping.items()}
        matches.append(inverse_mapping)
        if len(matches) >= max_matches:
            break
        if deadline is not None and time.time() >= deadline:
            break

    return matches


def find_edge_mapping(
    target: Graph,
    pattern: Graph,
    node_mapping: Dict[int, int]
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Given a node mapping, compute the corresponding edge mapping.

    Args:
        target: Target graph
        pattern: Pattern graph
        node_mapping: {pattern_node: target_node}

    Returns:
        {pattern_edge: target_edge} mapping
    """
    edge_mapping = {}

    for p_u, p_v in pattern.edges:
        t_u = node_mapping[p_u]
        t_v = node_mapping[p_v]
        target_edge = (min(t_u, t_v), max(t_u, t_v))

        if target_edge in target.edges:
            pattern_edge = (min(p_u, p_v), max(p_u, p_v))
            edge_mapping[pattern_edge] = target_edge

    return edge_mapping


# =============================================================================
# DISJOINT COVER ALGORITHM
# =============================================================================

def find_disjoint_cover(
    graph: Graph,
    minor: MinorEntry,
    table: RainbowTable,
    max_depth: int = 5,
    max_search_time_s: Optional[float] = None,
) -> Cover:
    """Greedily tile graph with disjoint copies of minor.

    Algorithm:
    1. Find all occurrences of minor in graph (VF2 subgraph isomorphism)
    2. Greedily select non-overlapping occurrences (largest coverage first)
    3. For uncovered edges, recursively tile with smaller minors
    4. Base case: tile remaining edges with K_2

    Args:
        graph: Target graph to cover
        minor: Minor to use as primary tile
        table: Rainbow table for finding smaller minors
        max_depth: Maximum recursion depth
        max_search_time_s: Optional per-VF2-call budget; propagates to
            the recursive smaller-minor search.

    Returns:
        Cover with all tiles and any uncovered edges
    """
    _log = get_log()
    cover = Cover()
    cover.uncovered_edges = set(graph.edges)

    # Base case: no edges or no recursion budget
    if not graph.edges or max_depth <= 0:
        return cover

    # Build pattern graph from minor
    pattern = _minor_to_graph(minor)
    if pattern is None:
        return cover

    # Skip if pattern is larger than graph
    if pattern.edge_count() > graph.edge_count():
        return cover

    # WL-based structural pre-filter (O((n+m)·k) for k WL rounds): cheap
    # necessary check before launching VF2. Uses degree + sorted neighbor
    # degrees (2-round WL). Filters out candidates where pattern's local
    # structure is incompatible with target's — VF2 would otherwise hang
    # exploring impossible mappings.
    if not can_pattern_fit_by_wl(graph, pattern, rounds=2):
        return cover

    # Find all occurrences
    matches = find_subgraph_isomorphisms(
        graph, pattern,
        max_matches=50,
        max_search_time_s=max_search_time_s,
    )
    _log.record(EventType.VF2_MATCH, "covering",
                f"Disjoint cover: {len(matches)} placements of {minor.name}",
                LogLevel.DEBUG, graph=graph)

    # Sort by coverage (prefer matches that cover more uncovered edges)
    def coverage_score(mapping):
        edge_mapping = find_edge_mapping(graph, pattern, mapping)
        return len(set(edge_mapping.values()) & cover.uncovered_edges)

    matches.sort(key=coverage_score, reverse=True)

    # Greedily add non-overlapping tiles
    for node_mapping in matches:
        edge_mapping = find_edge_mapping(graph, pattern, node_mapping)

        tile = Tile(
            minor=minor,
            node_mapping=node_mapping,
            edge_mapping=edge_mapping
        )

        # Only add if covers at least one uncovered edge
        if tile.covered_edges & cover.uncovered_edges:
            if cover.add_tile(tile):
                # Update uncovered edges
                cover.uncovered_edges -= tile.covered_edges

    # If still have uncovered edges, try smaller minors (with reduced depth)
    if cover.uncovered_edges and table is not None and max_depth > 1:
        remaining_graph = graph.edge_induced_subgraph(cover.uncovered_edges)

        # Only try if remaining graph is smaller
        if remaining_graph.edge_count() < graph.edge_count():
            smaller_minors = table.find_minors_of(remaining_graph)

            tried_keys = {minor.canonical_key}
            for smaller in smaller_minors:
                if smaller.canonical_key in tried_keys:
                    continue
                tried_keys.add(smaller.canonical_key)

                # Skip if smaller is same size or larger than remaining
                if smaller.edge_count >= remaining_graph.edge_count():
                    continue

                # Recursively cover remaining edges
                sub_cover = find_disjoint_cover(
                    remaining_graph, smaller, table, max_depth - 1,
                    max_search_time_s=max_search_time_s,
                )
                for tile in sub_cover.tiles:
                    if cover.add_tile(tile):
                        cover.uncovered_edges -= tile.covered_edges

                if cover.is_complete():
                    break

    _log.record(EventType.COVER_RESULT, "covering",
                f"Cover: {cover.total_tiles()} tiles, "
                f"{len(cover.uncovered_edges)} uncovered edges",
                LogLevel.DEBUG, graph=graph)
    return cover


_ATLAS_CACHE: Optional[List[Optional[Graph]]] = None


def _graph_atlas_cached(n: int) -> Optional[Graph]:
    """Return atlas index `n` from a one-shot bulk-loaded atlas cache.

    Each `nx.graph_atlas(n)` call walks the gzip-compressed atlas DB
    (~37ms per call). With ~1000 atlas entries in the default rainbow
    table, repeated lookups dominate `find_cell_candidates` /
    `try_heterogeneous_partition` (~36s wall on a cold table). Bulk-load
    via `nx.graph_atlas_g()` once and serve from memory thereafter.
    """
    global _ATLAS_CACHE
    if _ATLAS_CACHE is None:
        import networkx as nx
        atlas_list = list(nx.graph_atlas_g())
        _ATLAS_CACHE = [
            Graph.from_networkx(g) if g is not None and g.number_of_nodes() > 0 else None
            for g in atlas_list
        ]
    if 0 <= n < len(_ATLAS_CACHE):
        return _ATLAS_CACHE[n]
    return None


def _minor_to_graph(minor: MinorEntry) -> Optional[Graph]:
    """Reconstruct a graph from a minor entry.

    First checks if the entry has a stored graph. Otherwise falls back
    to name-based reconstruction for common graph types.
    """
    # Use stored graph if available
    if minor.graph is not None:
        return minor.graph

    from ..graph import (complete_graph, cycle_graph, path_graph, star_graph,
                         wheel_graph)

    def _mobius_ladder_nx(n: int):
        """Möbius ladder ML_n: 2n vertices, cycle plus n diametric rungs."""
        import networkx as nx
        g = nx.cycle_graph(2 * n)
        for i in range(n):
            g.add_edge(i, i + n)
        return g

    name = minor.name

    # Complete bipartite graphs: K_{a,b}
    # IMPORTANT: check K_{a,b} BEFORE K_n since K_{4,4} would otherwise
    # match the K_n prefix and fail int(name[2:]) on "{4,4}".
    if name.startswith('K_{') and ',' in name and name.endswith('}'):
        try:
            import networkx as nx
            inner = name[3:-1]  # strip "K_{" and "}"
            a_s, b_s = inner.split(',')
            a, b = int(a_s.strip()), int(b_s.strip())
            return Graph.from_networkx(nx.complete_bipartite_graph(a, b))
        except (ValueError, ImportError):
            pass

    # Complete graphs
    if name.startswith('K_'):
        try:
            n = int(name[2:])
            return complete_graph(n)
        except ValueError:
            pass

    # Cycle graphs
    if name.startswith('C_'):
        try:
            n = int(name[2:])
            return cycle_graph(n)
        except ValueError:
            pass

    # Path graphs
    if name.startswith('P_'):
        try:
            n = int(name[2:])
            return path_graph(n)
        except ValueError:
            pass

    # Wheel graphs
    if name.startswith('W_'):
        try:
            n = int(name[2:])
            return wheel_graph(n)
        except ValueError:
            pass

    # Star graphs
    if name.startswith('S_'):
        try:
            n = int(name[2:])
            return star_graph(n)
        except ValueError:
            pass

    # Zephyr graphs: Z(m,t) or Zm_t format
    if name.startswith('Z(') and ',' in name:
        try:
            import dwave_networkx as dnx

            # Parse "Z(m,t)" format
            inner = name[2:-1]  # Remove "Z(" and ")"
            m, t = inner.split(',')
            m, t = int(m.strip()), int(t.strip())
            G = dnx.zephyr_graph(m, t)
            return Graph.from_networkx(G)
        except (ValueError, ImportError):
            pass

    if name.startswith('Z') and '_' in name and not name.startswith('Z('):
        try:
            import dwave_networkx as dnx
            parts = name[1:].split('_')
            if len(parts) == 2:
                m, t = int(parts[0]), int(parts[1])
                G = dnx.zephyr_graph(m, t)
                return Graph.from_networkx(G)
        except (ValueError, ImportError):
            pass

    # Chimera graphs: Cmm format
    if name.startswith('Cm') and name[2:].isdigit():
        try:
            import dwave_networkx as dnx
            m = int(name[2:])
            G = dnx.chimera_graph(m)
            return Graph.from_networkx(G)
        except (ValueError, ImportError):
            pass

    # NetworkX graph-atlas entries: atlas_N → nx.graph_atlas(N).
    # The atlas covers all connected simple graphs up to 7 vertices
    # (~1200 entries) — these are frequent rainbow-table candidates
    # for small subgraph matching. Reconstruction is O(1) lookup via
    # `_graph_atlas_cached`.
    #
    # NOTE: atlas reconstruction is GATED behind opt-in because it
    # surfaces ~1000 extra "anonymous" candidates that are valid
    # induced subgraphs of many target graphs but rarely the cell
    # structure the downstream cell-quotient / k-matching consumers
    # need. Without gating, K_4 path / K_3 cycle tests pick
    # atlas_142-style cells over the intended K_n cells, then
    # downstream specialization fails. Specific paths that benefit
    # from atlas cells (small-graph tilings, family recognition) call
    # `_graph_atlas_cached` directly.
    if name.startswith('atlas_'):
        return None

    # Pan graphs: Pan_N → cycle C_N with a pendant edge (NetworkX has no
    # builder; construct directly).
    if name.startswith('Pan_'):
        try:
            import networkx as nx
            n = int(name[4:])
            if n < 3:
                return None
            g_nx = nx.cycle_graph(n)
            g_nx.add_edge(0, n)  # pendant
            return Graph.from_networkx(g_nx)
        except ValueError:
            pass

    # Fan graphs: Fan_N → join of single vertex with path P_N.
    if name.startswith('Fan_'):
        try:
            import networkx as nx
            n = int(name[4:])
            if n < 1:
                return None
            # F_n is K_1 + P_n; for n=1 it's just edge.
            path = nx.path_graph(n)
            g_nx = nx.Graph()
            g_nx.add_nodes_from(path.nodes())
            g_nx.add_edges_from(path.edges())
            apex = n
            g_nx.add_node(apex)
            for v in range(n):
                g_nx.add_edge(apex, v)
            return Graph.from_networkx(g_nx)
        except ValueError:
            pass

    # Helm graphs: Helm_N → wheel W_N with a pendant on each rim.
    if name.startswith('Helm_'):
        try:
            import networkx as nx
            n = int(name[5:])
            if n < 3:
                return None
            g_nx = nx.wheel_graph(n + 1)
            for v in range(1, n + 1):
                g_nx.add_edge(v, n + 1 + v - 1)
            return Graph.from_networkx(g_nx)
        except ValueError:
            pass

    # Ladder graphs: Ladder_N → P_N × K_2.
    if name.startswith('Ladder_'):
        try:
            import networkx as nx
            n = int(name[7:])
            if n < 2:
                return None
            return Graph.from_networkx(nx.ladder_graph(n))
        except ValueError:
            pass

    # Möbius–Kantor / Möbius ladders: Mobius_N or MoebiusLadder_N.
    if name.startswith('MoebiusLadder_') or name.startswith('MobiusLadder_'):
        try:
            import networkx as nx
            n = int(name.split('_')[-1])
            return Graph.from_networkx(nx.mobius_kantor_graph()) if n == 8 \
                else None
        except (ValueError, AttributeError):
            pass

    # Prism graphs: Prism_N → circular ladder CL_N.
    if name.startswith('Prism_'):
        try:
            import networkx as nx
            n = int(name[6:])
            if n < 3:
                return None
            return Graph.from_networkx(nx.circular_ladder_graph(n))
        except ValueError:
            pass

    # Book graphs: Book_N → N triangles sharing an edge.
    if name.startswith('Book_'):
        try:
            import networkx as nx
            n = int(name[5:])
            if n < 1:
                return None
            # B_n = K_1 ∨ S_n (or star with each leaf joined to apex2).
            g_nx = nx.Graph()
            g_nx.add_edge(0, 1)  # shared edge
            for k in range(n):
                v = 2 + k
                g_nx.add_edge(0, v)
                g_nx.add_edge(1, v)
            return Graph.from_networkx(g_nx)
        except ValueError:
            pass

    # Gear graphs: Gear_N → wheel W_N with each rim edge subdivided.
    # NetworkX has no built-in `gear_graph`; build explicitly.
    if name.startswith('Gear_'):
        try:
            import networkx as nx
            n = int(name[5:])
            if n < 3:
                return None
            # Gear G_n: hub h, rim vertices r_0..r_{n-1}, subdivision
            # vertices s_0..s_{n-1} between consecutive rim vertices.
            g_nx = nx.Graph()
            hub = 2 * n
            for i in range(n):
                g_nx.add_edge(hub, i)              # spokes
                g_nx.add_edge(i, n + i)             # rim → subdiv
                g_nx.add_edge(n + i, (i + 1) % n)   # subdiv → next rim
            return Graph.from_networkx(g_nx)
        except (ValueError, AttributeError):
            pass

    # Wheel_N alias for W_N.
    if name.startswith('Wheel_'):
        try:
            n = int(name[6:])
            return wheel_graph(n)
        except ValueError:
            pass

    # Möbius ladder: Mobius_N → ML_N (cycle on 2N vertices + N diametrically
    # opposite "rung" edges). All values of N use the same ladder construction
    # — Mobius_8 is the Möbius ladder ML_8, NOT the (similarly-sized)
    # Möbius–Kantor graph GP(8,3).
    if name.startswith('Mobius_'):
        try:
            n = int(name[7:])
            if n < 3:
                return None
            return Graph.from_networkx(_mobius_ladder_nx(n))
        except ValueError:
            pass

    # Grid_AxB → nx.grid_2d_graph(A, B) with integer relabel.
    if name.startswith('Grid_') and 'x' in name[5:]:
        try:
            import networkx as nx
            inner = name[5:]
            a_s, b_s = inner.split('x')
            a, b = int(a_s), int(b_s)
            g_nx = nx.grid_2d_graph(a, b)
            return Graph.from_networkx(nx.convert_node_labels_to_integers(g_nx))
        except ValueError:
            pass

    # Petersen
    if name == 'Petersen':
        try:
            import networkx as nx
            return Graph.from_networkx(nx.petersen_graph())
        except (ImportError, AttributeError):
            pass

    # Heawood
    if name == 'Heawood':
        try:
            import networkx as nx
            return Graph.from_networkx(nx.heawood_graph())
        except (ImportError, AttributeError):
            pass

    # Desargues
    if name == 'Desargues':
        try:
            import networkx as nx
            return Graph.from_networkx(nx.desargues_graph())
        except (ImportError, AttributeError):
            pass

    # Dodecahedral
    if name == 'Dodecahedral':
        try:
            import networkx as nx
            return Graph.from_networkx(nx.dodecahedral_graph())
        except (ImportError, AttributeError):
            pass

    # Sunlet graphs: Sunlet_N → cycle C_N with a pendant at each vertex.
    if name.startswith('Sunlet_'):
        try:
            import networkx as nx
            n = int(name[7:])
            if n < 3:
                return None
            g_nx = nx.cycle_graph(n)
            for v in range(n):
                g_nx.add_edge(v, n + v)
            return Graph.from_networkx(g_nx)
        except ValueError:
            pass

    return None


# =============================================================================
# FRINGE COMPUTATION
# =============================================================================

def compute_fringe(cover: Cover, input_graph: Graph) -> Fringe:
    """Compute over-coverage: edges in tiled cover not in input.

    For each tile, check which of its minor's edges don't exist
    in the input graph at the mapped positions.

    Args:
        cover: The computed cover
        input_graph: The original input graph

    Returns:
        Fringe containing edges in cover but not in input
    """
    fringe = Fringe()

    input_edges = set(input_graph.edges)

    for tile in cover.tiles:
        # Get the minor's graph structure
        pattern = _minor_to_graph(tile.minor)
        if pattern is None:
            continue

        # Check each edge in the minor
        for p_u, p_v in pattern.edges:
            # Map to target graph positions
            if p_u not in tile.node_mapping or p_v not in tile.node_mapping:
                continue

            t_u = tile.node_mapping[p_u]
            t_v = tile.node_mapping[p_v]
            target_edge = (min(t_u, t_v), max(t_u, t_v))

            # If this edge doesn't exist in input, it's fringe
            if target_edge not in input_edges:
                fringe.edges.add(target_edge)
                fringe.nodes.add(t_u)
                fringe.nodes.add(t_v)

    return fringe


def compute_inter_tile_edges(cover: Cover, input_graph: Graph) -> Set[Tuple[int, int]]:
    """Find edges in input graph that connect nodes from different tiles.

    These are the edges where we need to apply k-join formulas.

    Args:
        cover: The computed cover
        input_graph: The original input graph

    Returns:
        Set of edges connecting different tiles
    """
    # Build node -> tile index mapping
    node_to_tile = {}
    for i, tile in enumerate(cover.tiles):
        for node in tile.covered_nodes:
            node_to_tile[node] = i

    inter_edges = set()

    for u, v in input_graph.edges:
        tile_u = node_to_tile.get(u, -1)
        tile_v = node_to_tile.get(v, -1)

        # Edge connects different tiles (or uncovered node)
        if tile_u != tile_v:
            inter_edges.add((u, v))

    return inter_edges


def analyze_tile_connections(
    cover: Cover,
    input_graph: Graph
) -> Dict[Tuple[int, int], Dict]:
    """Analyze how tiles are connected in the input graph.

    For each pair of tiles, determine:
    - Number of edges connecting them
    - Shared vertices (if any)
    - Type of connection (k-join type)

    Args:
        cover: The computed cover
        input_graph: The original input graph

    Returns:
        Dict mapping (tile_i, tile_j) to connection info
    """
    # Build node -> tile index mapping
    node_to_tile = {}
    for i, tile in enumerate(cover.tiles):
        for node in tile.covered_nodes:
            node_to_tile[node] = i

    connections = {}
    n_tiles = len(cover.tiles)

    for i in range(n_tiles):
        for j in range(i + 1, n_tiles):
            nodes_i = cover.tiles[i].covered_nodes
            nodes_j = cover.tiles[j].covered_nodes

            # Find shared vertices (shouldn't exist for disjoint cover)
            shared = nodes_i & nodes_j

            # Find edges connecting the tiles
            connecting_edges = []
            for u, v in input_graph.edges:
                u_tile = node_to_tile.get(u, -1)
                v_tile = node_to_tile.get(v, -1)
                if (u_tile == i and v_tile == j) or (u_tile == j and v_tile == i):
                    connecting_edges.append((u, v))

            if connecting_edges or shared:
                connections[(i, j)] = {
                    'shared_vertices': len(shared),
                    'connecting_edges': len(connecting_edges),
                    'edges': connecting_edges,
                    'k_join_type': _determine_k_join_type(len(shared), len(connecting_edges))
                }

    return connections


def _determine_k_join_type(shared_vertices: int, connecting_edges: int) -> str:
    """Determine the type of k-join based on connection structure."""
    if shared_vertices == 0:
        if connecting_edges == 0:
            return "disjoint"
        elif connecting_edges == 1:
            return "bridge"
        else:
            return "multi_edge"
    elif shared_vertices == 1:
        return "1_join"  # Cut vertex
    elif shared_vertices == 2:
        return "2_join"  # Edge identification
    else:
        return f"{shared_vertices}_join"  # k-clique join


# =============================================================================
# HIERARCHICAL TILING (for graphs with repeating cell structure)
# =============================================================================

@dataclass
class InterCellInfo:
    """Information about edges connecting different cells."""
    edges: List[Tuple[int, int]]
    is_regular: bool  # True if same # edges between each cell pair
    edges_per_pair: int  # Number of edges between each adjacent cell pair
    cell_adjacencies: List[Tuple[int, int]]  # Which cells are adjacent


def find_cell_candidates(
    graph: Graph,
    table: RainbowTable,
    min_cells: int = 2
) -> List[MinorEntry]:
    """Find rainbow table entries that could tile the graph.

    Uses arithmetic filters to quickly eliminate candidates without
    running expensive VF2 subgraph isomorphism.

    Args:
        graph: Target graph to tile
        table: Rainbow table with potential tiles
        min_cells: Minimum number of cells required

    Returns:
        List of candidate entries sorted by edge count (descending)
    """
    target_nodes = graph.node_count()
    target_edges = graph.edge_count()
    target_sig = compute_signature(graph)

    candidates = []

    for entry in table.entries.values():
        cell_nodes = entry.node_count
        cell_edges = entry.edge_count

        # Filter 1 (cheap): node count must divide evenly. Do this BEFORE
        # any reconstruction or signature work to avoid touching entries
        # that can't possibly tile.
        if cell_nodes <= 0 or target_nodes % cell_nodes != 0:
            continue

        k = target_nodes // cell_nodes
        if k < min_cells:
            continue

        # Filter 2 (cheap): edge count consistency.
        cell_total_edges = k * cell_edges
        inter_cell_edges = target_edges - cell_total_edges

        if inter_cell_edges < 0:
            continue  # Would need negative inter-cell edges

        # Filter 3: degree sequence compatibility. Needs the cell graph
        # (for its degree sequence); reconstruct on demand and CACHE
        # back onto the entry so future calls skip the work.
        if entry.graph is None:
            pattern = _minor_to_graph(entry)
            if pattern is None:
                continue
            entry.graph = pattern  # cache the reconstruction
        else:
            pattern = entry.graph

        if entry.signature is None:
            entry.signature = compute_signature(pattern)
        cell_sig = entry.signature
        cell_degree_sum = sum(cell_sig.degree_sequence)
        target_degree_sum = sum(target_sig.degree_sequence)

        # With k cells and inter_cell_edges between them:
        # target_degree_sum = k * cell_degree_sum + 2 * inter_cell_edges
        expected_degree_sum = k * cell_degree_sum + 2 * inter_cell_edges
        if target_degree_sum != expected_degree_sum:
            continue

        candidates.append(entry)

    # Sort by edge count descending (prefer larger tiles). This is the
    # engine's long-standing default; the partition trial loop in
    # `try_hierarchical_partition` filters candidates that can't
    # actually tile the graph, so we don't need a smarter heuristic.
    return sorted(candidates, key=lambda e: e.edge_count, reverse=True)


def partition_into_cells(
    graph: Graph,
    cell: MinorEntry,
    k: int
) -> Optional[List[Set[int]]]:
    """Partition graph nodes into k groups that look like the cell.

    This is the main entry point for partitioning. It tries multiple
    strategies in order of sophistication:
    1. Disconnected components (trivial case)
    2. Node signature matching (works when cells are disjoint)
    3. Structural clustering (works when cells share edges)

    Args:
        graph: Target graph to partition
        cell: Cell pattern from rainbow table
        k: Expected number of cells

    Returns:
        List of node sets (one per cell) or None if partitioning fails
    """
    cell_graph = cell.graph if cell.graph is not None else _minor_to_graph(cell)
    if cell_graph is None:
        return None

    cell_size = cell.node_count

    if graph.node_count() != k * cell_size:
        return None

    # Strategy 1: Disconnected components
    if not graph.is_connected():
        components = graph.connected_components()
        if len(components) == k:
            if all(comp.node_count() == cell_size for comp in components):
                return [set(comp.nodes) for comp in components]

    # Strategy 2: Node signature matching (for disjoint cells)
    target_sigs = compute_all_node_signatures(graph)
    cell_sigs = compute_all_node_signatures(cell_graph)

    partitions = _greedy_partition(graph, cell_graph, k, target_sigs, cell_sigs)
    if partitions is not None:
        # Verify the partition is actually valid before returning
        if _verify_partition_structure(graph, partitions, cell_graph):
            return partitions

    # Strategy 3: VF2-based structural matching for connected cells
    # This handles cases where inter-cell edges change node signatures
    partitions = _partition_by_structure(graph, cell_graph, k)

    return partitions


def _verify_partition_structure(
    graph: Graph,
    partition: List[Set[int]],
    cell_graph: Graph
) -> bool:
    """Quick verification that partition cells are isomorphic to pattern."""
    cell_edges = cell_graph.edge_count()

    for cell_nodes in partition:
        subgraph = graph.subgraph(cell_nodes)
        # Check edge count first (fast)
        if subgraph.edge_count() != cell_edges:
            return False

    # All cells have correct edge count - check one for isomorphism
    if partition:
        subgraph = graph.subgraph(partition[0])
        G1 = subgraph.to_networkx()
        G2 = cell_graph.to_networkx()
        if not nx.is_isomorphic(G1, G2):
            return False

    return True


def _partition_by_structure(
    graph: Graph,
    cell_graph: Graph,
    k: int,
    max_matches: int = 200,
    time_budget_seconds: Optional[float] = None,
) -> Optional[List[Set[int]]]:
    """Partition using VF2 to find disjoint isomorphic copies.

    When cells share edges, node signatures change. We need to find
    groups where the *induced subgraph* is isomorphic to the cell.

    Strategy:
    1. Find subgraph isomorphisms via VF2, capped by `max_matches` and
       a per-call wall-clock budget. When `time_budget_seconds` is
       None (default), the budget is computed from graph size:
       `0.05s × max(20, n)`, giving ~1s for n=20 graphs, ~3.6s for
       Pm3-sized n=72, etc. Pm2 cProfile (May 21 2026) showed VF2
       in this routine consumed ~90% of cold-cache time iterating
       candidate cells that all failed to tile Pegasus.
    2. Find k disjoint copies that cover all nodes via
       `_find_disjoint_partition` (separately budgeted).
    3. Return the partition.
    """
    cell_size = cell_graph.node_count()
    total_nodes = graph.node_count()

    if total_nodes != k * cell_size:
        return None

    # Dynamic budget: ~0.01s per node, floored at 0.3s. Pm2 (40n) → 0.4s;
    # Cm3 (72n) → 0.72s; Pm3 (128n) → 1.28s. Small graphs (n<30) still
    # get 0.3s minimum so the K_{4,4} success path on small inputs isn't
    # interrupted.
    # NOTE (May 26 2026): on Cm_3 (72n), K_{4,4} VF2 succeeds in ~0.7-0.8s
    # which races the 0.72s budget — ~60% of trials timed out, falling through
    # to Fan_1 (size-2 cells) which breaks the cell-quotient grid DP path.
    # Smarter budget: scale by cell_size × node_count to give larger budgets
    # for the cases where VF2 takes longer (large cells embedded in large
    # targets). For K_{4,4} (8n) in Cm_3 (72n): 0.02 × 72 + 0.05 × 8 = 1.84s.
    # For Z(1,2) (24n) with K_{3,3}-class cells (6n): 0.01 × 24 + 0.05 × 6 = 0.54s.
    # Keep 0.3s floor for tiny graphs.
    if time_budget_seconds is None:
        time_budget_seconds = max(0.3, 0.01 * total_nodes + 0.05 * cell_size)

    # Use VF2 to find all isomorphic copies of cell in graph
    G_target = graph.to_networkx()
    G_pattern = cell_graph.to_networkx()

    matcher = isomorphism.GraphMatcher(G_target, G_pattern)

    # Collect all matches as sets of nodes (budget VF2 by both count and time).
    all_matches: List[Set[int]] = []
    seen_node_sets: Set[FrozenSet[int]] = set()
    import time as _time
    deadline = _time.monotonic() + time_budget_seconds

    # Hard wall-clock budget: networkx VF2 is pure-Python but spends most
    # time backtracking inside `subgraph_isomorphisms_iter`. If VF2 yields
    # zero matches (Cm_3 worst case), the loop body never runs and the
    # in-loop deadline check can't fire. Run the iter in a daemon thread
    # so the consumer side can enforce the deadline regardless of yield
    # frequency. The thread continues spinning briefly after exit but is
    # bounded by the same deadline (and is killed at process shutdown).
    import threading, queue
    q: queue.Queue = queue.Queue(maxsize=64)
    sentinel = object()

    def _producer() -> None:
        try:
            for m in matcher.subgraph_isomorphisms_iter():
                if _time.monotonic() >= deadline:
                    break
                q.put(m)
        except Exception:
            pass
        finally:
            q.put(sentinel)

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    while True:
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            break
        try:
            mapping = q.get(timeout=min(remaining, 0.25))
        except queue.Empty:
            continue  # re-check deadline
        if mapping is sentinel:
            break
        nodes = frozenset(mapping.keys())
        if nodes not in seen_node_sets:
            seen_node_sets.add(nodes)
            all_matches.append(set(nodes))
        if len(all_matches) >= max_matches:
            break

    if len(all_matches) < k:
        return None

    # Find k disjoint matches that cover all nodes
    partition = _find_disjoint_partition(all_matches, k, total_nodes)

    return partition


def _find_disjoint_partition(
    matches: List[Set[int]],
    k: int,
    total_nodes: int,
    max_iterations: int = 100_000,
) -> Optional[List[Set[int]]]:
    """Find k disjoint matches that cover all nodes.

    Uses backtracking search to find a valid partition.

    Bounded by ``max_iterations`` (default 100k) backtrack frame entries —
    graphs that admit a homogeneous tiling usually find one in the first
    few hundred frames; graphs that DON'T admit one (e.g. Pm_m, Z(1, t)
    when no clean tiling exists) used to burn unbounded time exhausting
    the search. cProfile of Pm2 showed 44.7M backtrack calls over 605s
    of CPU; capping at 100k drops the worst-case to sub-second per attempt.
    Returns None on budget exhaustion (false negatives are acceptable —
    upstream falls back to other partitioners).
    """
    if k == 0:
        return []

    if not matches:
        return None

    # Sort matches by node indices for deterministic behavior
    matches = sorted(matches, key=lambda m: tuple(sorted(m)))

    iterations = [0]
    budget_exhausted = [False]

    def backtrack(
        index: int,
        used: Set[int],
        partition: List[Set[int]]
    ) -> Optional[List[Set[int]]]:
        iterations[0] += 1
        if iterations[0] > max_iterations:
            budget_exhausted[0] = True
            return None
        if len(partition) == k:
            if len(used) == total_nodes:
                return partition.copy()
            return None

        for i in range(index, len(matches)):
            if budget_exhausted[0]:
                return None
            match = matches[i]

            # Check if this match is disjoint from already used nodes
            if not (match & used):
                partition.append(match)
                new_used = used | match

                result = backtrack(i + 1, new_used, partition)
                if result is not None:
                    return result

                partition.pop()

        return None

    return backtrack(0, set(), [])


def _grow_by_edge_density(
    graph: Graph,
    start: int,
    cell_size: int,
    cell_edges: int,
    used_nodes: Set[int]
) -> Optional[Set[int]]:
    """Grow a cell by maximizing internal edge density.

    Greedily add nodes that maximize edges within the growing cell.
    """
    cell_nodes = {start}
    all_edges = graph.edges

    while len(cell_nodes) < cell_size:
        best_node = None
        best_score = -1

        # Consider all neighbors of current cell
        frontier = set()
        for node in cell_nodes:
            for neighbor in graph.neighbors(node):
                if neighbor not in cell_nodes and neighbor not in used_nodes:
                    frontier.add(neighbor)

        if not frontier:
            return None

        for candidate in frontier:
            # Count edges from candidate to current cell
            edges_to_cell = sum(1 for n in cell_nodes
                               if (min(candidate, n), max(candidate, n)) in all_edges)

            if edges_to_cell > best_score:
                best_score = edges_to_cell
                best_node = candidate

        if best_node is None:
            return None

        cell_nodes.add(best_node)

    return cell_nodes


def _greedy_partition(
    graph: Graph,
    cell_graph: Graph,
    k: int,
    target_sigs: Dict[int, NodeSignature],
    cell_sigs: Dict[int, NodeSignature]
) -> Optional[List[Set[int]]]:
    """Greedily partition graph into k cell-shaped groups.

    Strategy:
    1. Find anchor nodes (distinct signature in cell)
    2. Grow cells from anchors by signature matching
    3. Verify each cell is isomorphic to pattern

    For disconnected graphs, use connected components as natural partitions.
    """
    cell_size = len(cell_sigs)

    # Special case: disconnected graph
    # Use connected components as natural cell boundaries
    if not graph.is_connected():
        components = graph.connected_components()
        if len(components) == k:
            # Check if each component has the right size
            valid = all(comp.node_count() == cell_size for comp in components)
            if valid:
                return [set(comp.nodes) for comp in components]

    # Find nodes with unique signatures in cell (good anchors)
    sig_counts: Dict[Tuple[int, int, int], int] = {}
    for sig in cell_sigs.values():
        key = (sig.degree, sig.neighbor_degree_sum, sig.triangles)
        sig_counts[key] = sig_counts.get(key, 0) + 1

    unique_sig_keys = [k for k, v in sig_counts.items() if v == 1]

    if not unique_sig_keys:
        # No unique signatures - all nodes look the same
        # For regular graphs, use connectivity-based partitioning
        # Try to find k disjoint subgraphs of the right size
        return _partition_by_connectivity(graph, cell_size, k)

    anchor_sig_key = unique_sig_keys[0]

    # Find all nodes in target with this signature
    anchor_candidates = [
        n for n, sig in target_sigs.items()
        if (sig.degree, sig.neighbor_degree_sum, sig.triangles) == anchor_sig_key
    ]

    if len(anchor_candidates) != k:
        return None  # Wrong number of anchors

    partitions: List[Set[int]] = []
    used_nodes: Set[int] = set()

    for anchor in anchor_candidates:
        if anchor in used_nodes:
            continue

        # Grow a cell from this anchor using BFS with signature matching
        cell_nodes = _grow_cell(graph, anchor, cell_size, target_sigs, cell_sigs, used_nodes)

        if cell_nodes is None or len(cell_nodes) != cell_size:
            # Try alternative approach: just grow connected component of size cell_size
            cell_nodes = _grow_connected_cell(graph, anchor, cell_size, used_nodes)

        if cell_nodes is None or len(cell_nodes) != cell_size:
            return None

        partitions.append(cell_nodes)
        used_nodes.update(cell_nodes)

    if len(partitions) != k:
        return None

    return partitions


def _partition_by_connectivity(
    graph: Graph,
    cell_size: int,
    k: int
) -> Optional[List[Set[int]]]:
    """Partition graph into k groups of cell_size using connectivity.

    For graphs where all nodes have identical signatures, use
    connected components or greedy growth.
    """
    # First check if graph has natural connected components
    if not graph.is_connected():
        components = graph.connected_components()
        if len(components) == k:
            valid = all(comp.node_count() == cell_size for comp in components)
            if valid:
                return [set(comp.nodes) for comp in components]

    # For connected graphs, use greedy growth from distributed starting points
    # Pick k starting nodes that are maximally spread out
    nodes_list = list(graph.nodes)
    if len(nodes_list) != k * cell_size:
        return None

    partitions: List[Set[int]] = []
    used_nodes: Set[int] = set()

    # Start from first unused node and grow cells
    for start_candidate in nodes_list:
        if start_candidate in used_nodes:
            continue

        cell_nodes = _grow_connected_cell(graph, start_candidate, cell_size, used_nodes)
        if cell_nodes is None or len(cell_nodes) != cell_size:
            return None

        partitions.append(cell_nodes)
        used_nodes.update(cell_nodes)

        if len(partitions) == k:
            break

    if len(partitions) != k or len(used_nodes) != k * cell_size:
        return None

    return partitions


def _grow_cell(
    graph: Graph,
    anchor: int,
    cell_size: int,
    target_sigs: Dict[int, NodeSignature],
    cell_sigs: Dict[int, NodeSignature],
    used_nodes: Set[int]
) -> Optional[Set[int]]:
    """Grow a cell from anchor by matching node signatures."""
    # Get required signature distribution
    sig_needed: Dict[Tuple[int, int, int], int] = {}
    for sig in cell_sigs.values():
        key = (sig.degree, sig.neighbor_degree_sum, sig.triangles)
        sig_needed[key] = sig_needed.get(key, 0) + 1

    cell_nodes = {anchor}
    anchor_sig = target_sigs[anchor]
    anchor_key = (anchor_sig.degree, anchor_sig.neighbor_degree_sum, anchor_sig.triangles)
    sig_needed[anchor_key] -= 1

    # BFS from anchor
    frontier = list(graph.neighbors(anchor) - used_nodes)

    while len(cell_nodes) < cell_size and frontier:
        best_node = None
        best_score = float('inf')

        for node in frontier:
            if node in cell_nodes or node in used_nodes:
                continue

            sig = target_sigs[node]
            sig_key = (sig.degree, sig.neighbor_degree_sum, sig.triangles)

            if sig_needed.get(sig_key, 0) > 0:
                # This signature is still needed
                # Score by how many cell neighbors it has
                cell_neighbors = len(graph.neighbors(node) & cell_nodes)
                score = -cell_neighbors  # More neighbors = better
                if score < best_score:
                    best_score = score
                    best_node = node

        if best_node is None:
            # No matching node in frontier, expand frontier to all
            # un-used cell-reachable nodes. If expansion adds nothing
            # beyond what's already in `frontier`, growth is wedged —
            # break to avoid infinite loop (no anchor exists that
            # matches the remaining signature multiset).
            new_frontier = set()
            for node in cell_nodes:
                for neighbor in graph.neighbors(node):
                    if neighbor not in cell_nodes and neighbor not in used_nodes:
                        new_frontier.add(neighbor)
            if not new_frontier or new_frontier <= set(frontier):
                break
            frontier = list(new_frontier)
        else:
            cell_nodes.add(best_node)
            sig = target_sigs[best_node]
            sig_key = (sig.degree, sig.neighbor_degree_sum, sig.triangles)
            sig_needed[sig_key] -= 1

            # Expand frontier
            for neighbor in graph.neighbors(best_node):
                if neighbor not in cell_nodes and neighbor not in used_nodes:
                    if neighbor not in frontier:
                        frontier.append(neighbor)

    if len(cell_nodes) == cell_size:
        return cell_nodes
    return None


def _grow_connected_cell(
    graph: Graph,
    start: int,
    cell_size: int,
    used_nodes: Set[int]
) -> Optional[Set[int]]:
    """Grow a connected component of exactly cell_size nodes from start."""
    cell_nodes = {start}
    frontier = [n for n in graph.neighbors(start) if n not in used_nodes]

    while len(cell_nodes) < cell_size and frontier:
        # Pick node with most connections to current cell
        best = max(frontier, key=lambda n: len(graph.neighbors(n) & cell_nodes))
        cell_nodes.add(best)
        frontier.remove(best)

        for neighbor in graph.neighbors(best):
            if neighbor not in cell_nodes and neighbor not in used_nodes and neighbor not in frontier:
                frontier.append(neighbor)

    if len(cell_nodes) == cell_size:
        return cell_nodes
    return None


def verify_cell_partition(
    graph: Graph,
    partition: List[Set[int]],
    cell: MinorEntry
) -> bool:
    """Verify each partition element is isomorphic to the cell.

    Uses signature pre-checks before VF2 for speed.
    VF2 on small cell-sized graphs is fast!

    Args:
        graph: The full graph
        partition: List of node sets (one per cell)
        cell: Cell pattern from rainbow table

    Returns:
        True if all partitions are isomorphic to cell
    """
    cell_graph = cell.graph if cell.graph is not None else _minor_to_graph(cell)
    if cell_graph is None:
        return False

    cell_sig = compute_signature(cell_graph)

    for cell_nodes in partition:
        subgraph = graph.subgraph(cell_nodes)

        # Quick signature check first
        sub_sig = compute_signature(subgraph)
        if not cell_sig.could_match(sub_sig):
            return False

        # Edge count check (faster than VF2)
        if subgraph.edge_count() != cell.edge_count:
            return False

        # VF2 isomorphism check on small graph (fast!)
        G1 = subgraph.to_networkx()
        G2 = cell_graph.to_networkx()

        if not nx.is_isomorphic(G1, G2):
            return False

    return True


def analyze_inter_cell_edges(
    graph: Graph,
    partition: List[Set[int]]
) -> InterCellInfo:
    """Analyze edges between cells in a partition.

    Determines:
    - Which edges connect different cells
    - Whether the connection pattern is regular
    - How many edges exist between each cell pair

    Args:
        graph: The full graph
        partition: List of node sets (one per cell)

    Returns:
        InterCellInfo with edge analysis
    """
    k = len(partition)

    # Build node-to-cell mapping
    node_to_cell: Dict[int, int] = {}
    for i, cell_nodes in enumerate(partition):
        for node in cell_nodes:
            node_to_cell[node] = i

    # Find all inter-cell edges
    inter_edges: List[Tuple[int, int]] = []
    cell_pair_edges: Dict[Tuple[int, int], int] = {}

    for u, v in graph.edges:
        cell_u = node_to_cell.get(u, -1)
        cell_v = node_to_cell.get(v, -1)

        if cell_u != cell_v and cell_u >= 0 and cell_v >= 0:
            inter_edges.append((u, v))
            pair = (min(cell_u, cell_v), max(cell_u, cell_v))
            cell_pair_edges[pair] = cell_pair_edges.get(pair, 0) + 1

    # Check if pattern is regular
    edge_counts = list(cell_pair_edges.values())
    is_regular = len(set(edge_counts)) <= 1 if edge_counts else True
    edges_per_pair = edge_counts[0] if edge_counts else 0

    # Find adjacent cell pairs
    cell_adjacencies = list(cell_pair_edges.keys())

    return InterCellInfo(
        edges=inter_edges,
        is_regular=is_regular,
        edges_per_pair=edges_per_pair,
        cell_adjacencies=cell_adjacencies
    )


def extract_cell_topology(
    partition: List[Set[int]],
    inter_edges: List[Tuple[int, int]],
) -> Optional[MultiGraph]:
    """Return the cell-topology multigraph H if the unified formula applies.

    The unified formula

        T(G) = (∏_i T(cell_i)) × T(H)

    is valid when **every (cell_i, cell_j) pair's inter-cell edges share a
    single (vertex-in-cell-i, vertex-in-cell-j) endpoint pair**. That is, all
    inter-cell edges between two given cells must connect the same vertex
    pair (parallel-edge structure between two specific cell vertices).

    H has one node per cell; for each inter-cell edge in G between cells
    (i, j) we add an edge (i, j) in H, so multiple G-edges between the same
    cell-vertex-pair become parallel edges in H.

    Returns None when the precondition fails (the genuine "chord case" — at
    least one cell-pair has inter-cell edges connecting distinct vertex
    pairs). Callers should fall through to the existing chord-rule pipeline.

    Note: this function is cell-agnostic. The precondition is a property of
    the inter-cell edge set only, so heterogeneous partitions (cells of
    different shapes) are handled identically to homogeneous ones.
    """
    k = len(partition)

    # Empty partition or no inter-cell edges → trivially unified
    # (caller handles the empty-edge case before invoking us, but be safe)
    if k == 0:
        return MultiGraph(nodes=frozenset(), edge_counts={}, loop_counts={})

    node_to_cell: Dict[int, int] = {}
    for i, cell_nodes in enumerate(partition):
        for node in cell_nodes:
            node_to_cell[node] = i

    # Group inter-cell edges by (cell_i, cell_j) and track the
    # vertex-pair(s) used for each cell-pair. The precondition requires
    # exactly one vertex-pair per cell-pair.
    pair_to_vertex_pairs: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    pair_to_count: Dict[Tuple[int, int], int] = {}

    for u, v in inter_edges:
        cu = node_to_cell.get(u)
        cv = node_to_cell.get(v)
        if cu is None or cv is None or cu == cv:
            # Edge isn't actually inter-cell or references an unknown node;
            # be conservative and bail.
            return None
        cell_pair = (min(cu, cv), max(cu, cv))
        vertex_pair = (min(u, v), max(u, v))
        pair_to_vertex_pairs.setdefault(cell_pair, set()).add(vertex_pair)
        pair_to_count[cell_pair] = pair_to_count.get(cell_pair, 0) + 1

    for cell_pair, vps in pair_to_vertex_pairs.items():
        if len(vps) > 1:
            # Distinct vertex pairs between same cell-pair → genuine chord
            # case. Unified formula does not apply.
            return None

    # Build the cell-topology multigraph: nodes are cell indices
    # 0..k-1; edges are (cell_i, cell_j) with multiplicity = number of
    # inter-cell edges between that pair.
    edge_counts: Dict[Tuple[int, int], int] = {
        cell_pair: count for cell_pair, count in pair_to_count.items()
    }
    return MultiGraph(
        nodes=frozenset(range(k)),
        edge_counts=edge_counts,
        loop_counts={},
    )


# =============================================================================
# k-matching topology detection 
# =============================================================================
#
# When inter-cell edges form k-matching junctions (k > 1 distinct vertex
# pairs between two cells, with the k anchors on each side belonging to
# a single vertex-transitive class), the closed-form formula
#
#     T(G_1 + M_k + G_2) = (x + k - 1) · T(G_1)·T(G_2)
#                          + Σ_{j=2..k} C(k, j) · T(G_1 ⊕_j G_2)
#
# applies. For multi-cell topologies (cell-tree, cell-cycle, etc.) the
# formula extends recursively.
# =============================================================================


@dataclass
class KMatchingJunction:
    """One inter-cell junction with a k-matching coupler structure.

    edges[i] = (anchors_i[i], anchors_j[i]) for each i in 0..k-1.
    All anchor vertices on each side must be in a single
    vertex-transitive class within the cell (typically: same
    bipartition side of a bipartite cell).
    """
    cell_i: int
    cell_j: int
    edges: List[Tuple[int, int]]
    anchors_i: List[int]
    anchors_j: List[int]

    @property
    def k(self) -> int:
        return len(self.edges)


@dataclass
class BipartiteJunction:
    """Generalized inter-cell junction with arbitrary bipartite structure.

    The strict k-matching contract (each anchor appears in exactly one
    inter-cell edge) is relaxed to:

    - The set of inter-cell edges between cell_i and cell_j forms a
      bipartite graph (trivially true since edges cross cells).
    - Each anchor on either side MAY have degree > 1 (multi-edge anchor).
    - The junction subgraph may be disconnected (multiple components).

    Unlike `KMatchingJunction`, the `edges` list is the actual edge
    multiset (no requirement that each anchor appears exactly once);
    `anchors_i` and `anchors_j` are deduplicated vertex lists.

    The chain / cycle DP machinery downstream uses ``junction_template``
    (the actual junction graph as a Graph object) instead of building
    an M_k matching template. T_rooted on the junction template is
    computed via `t_rooted_smart` which decomposes disconnected
    junctions into per-component pieces (much faster than naïve
    2^|E| brute force when the junction splits).

    Example: Z(1, 2) inter-cell structure between its two Z(1, 1)
    cells: 32 edges split into 2 disconnected bipartite components
    of 16 edges each. Anchors have degree sequence [2,2,2,2,2,2,4,4,4,4]
    on each side. Cannot be expressed as a `KMatchingJunction`; this
    class captures it directly.
    """
    cell_i: int
    cell_j: int
    edges: List[Tuple[int, int]]
    anchors_i: List[int]  # deduplicated, sorted
    anchors_j: List[int]  # deduplicated, sorted

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_junction_graph(self) -> "Graph":
        """Return the junction subgraph as a Graph (anchors_i first, then anchors_j).

        Vertex ordering: ``anchors_i + anchors_j`` (cell-i side, then cell-j side).
        Edges are translated to use vertex IDs from this combined list.
        """
        from ..graph import Graph as TutteGraph
        # Build vertex list in fixed order: cell-i side first, then cell-j side.
        # Renumber locally so the Graph nodes start at 0 (anchors_i_local =
        # 0..len(anchors_i)-1, anchors_j_local = len(anchors_i)..len(both)).
        n_i = len(self.anchors_i)
        ai_map = {v: i for i, v in enumerate(self.anchors_i)}
        aj_map = {v: n_i + i for i, v in enumerate(self.anchors_j)}
        nodes = list(range(n_i + len(self.anchors_j)))
        edges_local = []
        for (u, v) in self.edges:
            if u in ai_map and v in aj_map:
                edges_local.append((ai_map[u], aj_map[v]))
            elif u in aj_map and v in ai_map:
                edges_local.append((ai_map[v], aj_map[u]))
            else:
                raise ValueError(f"Junction edge {(u, v)} doesn't match anchors")
        return TutteGraph(nodes, edges_local)


def _anchors_single_class(
    graph: Graph, cell_nodes: Set[int], anchors: Set[int],
) -> bool:
    """Check if all anchors lie in a single vertex-transitive class
    within the cell's induced subgraph.

    Heuristic: induce subgraph on cell_nodes; check if bipartite; if
    so, all anchors must be in the same bipartition side. If the cell
    is not bipartite, fall back to assuming all vertices are equivalent
    (true for complete graphs K_n; conservative for others).
    """
    if not anchors:
        return True
    cell_nx = nx.Graph()
    cell_nx.add_nodes_from(cell_nodes)
    for u, v in graph.edges:
        if u in cell_nodes and v in cell_nodes:
            cell_nx.add_edge(u, v)
    if cell_nx.number_of_edges() == 0:
        # Empty cell — anchors trivially equivalent
        return True
    try:
        if nx.is_bipartite(cell_nx):
            sides = nx.bipartite.sets(cell_nx)
            for side in sides:
                if anchors.issubset(side):
                    return True
            # Anchors span both sides → not single-class
            return False
    except nx.NetworkXError:
        # Not connected etc. — fall back
        pass
    # Non-bipartite cell: the formula's proof needs k-SET transitivity
    # of the anchor class, not just vertex-transitivity. K_n (any n)
    # satisfies this trivially: every k-subset induces K_k. Other
    # vertex-transitive cells (Petersen, C_n, Möbius–Kantor) DO NOT —
    # empirical test shows the formula gives wrong results at k ≥ 3
    # for those. Reference: tutte/research/data/kmatching_non_kn_findings.md
    # (May 2026). Restrict to K_n by checking edge count equals n(n-1)/2.
    #
    # Soundness only — no perf impact on D-Wave production graphs:
    # Cm_m, Pm_m, Z(m, t) cells are all K_{a,b} (bipartite), so they hit
    # the bipartite branch above and never reach this check. The
    # restriction is defensive correctness for hypothetical inputs with
    # non-K_n vertex-transitive cells.
    n_cell = len(cell_nodes)
    expected_complete = n_cell * (n_cell - 1) // 2
    return cell_nx.number_of_edges() == expected_complete


def detect_kmatching_topology(
    graph: Graph,
    partition: List[Set[int]],
    inter_edges: List[Tuple[int, int]],
) -> Optional[List[KMatchingJunction]]:
    """Detect if inter-cell edges form k-matching junctions where the
    cell-cycle formula applies.

    Returns a list of `KMatchingJunction` (one per cell-pair with
    edges) if **every junction** satisfies the single-class anchor
    precondition (anchors on each side lie in a single
    vertex-transitive class of the cell). Returns None otherwise.

    For Chimera Cm_m: each junction has k=4 with all 4 anchors on a
    single bipartition side of K_{4,4}; this function returns the
    junction list and the recursive cell-cycle formula applies.
    """
    if not inter_edges:
        return []

    node_to_cell: Dict[int, int] = {}
    for i, cell_nodes in enumerate(partition):
        for node in cell_nodes:
            node_to_cell[node] = i

    # Group edges by cell-pair, tracking anchors per side
    pair_data: Dict[Tuple[int, int], Tuple[List[Tuple[int, int]], Set[int], Set[int]]] = {}
    for u, v in inter_edges:
        cu = node_to_cell.get(u)
        cv = node_to_cell.get(v)
        if cu is None or cv is None or cu == cv:
            return None
        if cu < cv:
            ci, cj, ai, aj = cu, cv, u, v
        else:
            ci, cj, ai, aj = cv, cu, v, u
        pair = (ci, cj)
        if pair not in pair_data:
            pair_data[pair] = ([], set(), set())
        edges_list, anchors_i, anchors_j = pair_data[pair]
        edges_list.append((ai, aj))
        anchors_i.add(ai)
        anchors_j.add(aj)

    junctions: List[KMatchingJunction] = []
    for (ci, cj), (edges_list, anchors_i, anchors_j) in pair_data.items():
        # Each anchor used at most once per side (true matching, no
        # repeat). If some anchor is reused, the structure is not a
        # simple k-matching.
        if len(anchors_i) != len(edges_list) or len(anchors_j) != len(edges_list):
            return None
        # Single-class precondition per side
        if not _anchors_single_class(graph, partition[ci], anchors_i):
            return None
        if not _anchors_single_class(graph, partition[cj], anchors_j):
            return None
        junctions.append(KMatchingJunction(
            cell_i=ci, cell_j=cj,
            edges=edges_list,
            anchors_i=[a for a, _ in edges_list],
            anchors_j=[b for _, b in edges_list],
        ))

    # Stronger precondition: shared anchors across junctions at the
    # same cell cause parallel edges when contractions close a cycle
    # in the cell-topology graph. For a tree cell-topology,
    # contractions chain linearly and shared anchors are safe. For a
    # cyclic cell-topology, shared anchors break the formula.
    #
    # Rule: if the cell-topology has any cycle, reject any cell where
    # two junctions share an anchor vertex.
    cell_topology = nx.Graph()
    cell_topology.add_nodes_from(range(len(partition)))
    for j in junctions:
        cell_topology.add_edge(j.cell_i, j.cell_j)
    # Tree iff cycle_basis is empty (on the undirected graph).
    has_cycle = bool(nx.cycle_basis(cell_topology))

    if has_cycle:
        cell_used_anchors: Dict[int, Set[int]] = {}
        for j in junctions:
            for anchor in j.anchors_i:
                if anchor in cell_used_anchors.setdefault(j.cell_i, set()):
                    return None
                cell_used_anchors[j.cell_i].add(anchor)
            for anchor in j.anchors_j:
                if anchor in cell_used_anchors.setdefault(j.cell_j, set()):
                    return None
                cell_used_anchors[j.cell_j].add(anchor)

    return junctions


def detect_bipartite_junction_topology(
    graph: Graph,
    partition: List[Set[int]],
    inter_edges: List[Tuple[int, int]],
) -> Optional[List[BipartiteJunction]]:
    """Generalized junction detection — accepts non-matching bipartite junctions.

    Drops two restrictions from `detect_kmatching_topology`:

    1. **No matching restriction**: an anchor on either side may participate
       in MULTIPLE inter-cell edges (degree > 1 allowed).
    2. **No single-class restriction**: anchors may span multiple
       vertex-transitive classes within the cell — the consumer handles
       this via the actual junction graph (not an M_k template), so cell
       symmetries that the formula relies on aren't required.

    Returns one `BipartiteJunction` per cell-pair with edges. Always
    succeeds if `inter_edges` are non-empty and partition cell IDs are
    valid (returns `None` only on malformed input).

    The trade-off vs k-matching detection: the chain DP consumer must
    use the actual junction graph for T_rooted (via `t_rooted_smart`)
    rather than the optimized M_k template path. Worth it for graphs
    like Z(1, 2) where the inter-cell structure is genuinely non-matching.
    """
    if not inter_edges:
        return []

    node_to_cell: Dict[int, int] = {}
    for i, cell_nodes in enumerate(partition):
        for node in cell_nodes:
            node_to_cell[node] = i

    pair_data: Dict[Tuple[int, int], Tuple[List[Tuple[int, int]], Set[int], Set[int]]] = {}
    for u, v in inter_edges:
        cu = node_to_cell.get(u)
        cv = node_to_cell.get(v)
        if cu is None or cv is None or cu == cv:
            return None
        if cu < cv:
            ci, cj, ai, aj = cu, cv, u, v
        else:
            ci, cj, ai, aj = cv, cu, v, u
        pair = (ci, cj)
        if pair not in pair_data:
            pair_data[pair] = ([], set(), set())
        edges_list, anchors_i, anchors_j = pair_data[pair]
        edges_list.append((ai, aj))
        anchors_i.add(ai)
        anchors_j.add(aj)

    junctions: List[BipartiteJunction] = []
    for (ci, cj), (edges_list, anchors_i, anchors_j) in pair_data.items():
        junctions.append(BipartiteJunction(
            cell_i=ci, cell_j=cj,
            edges=edges_list,
            anchors_i=sorted(anchors_i),
            anchors_j=sorted(anchors_j),
        ))
    return junctions


def _apply_junction_merge(
    g: nx.MultiGraph,
    junction_edges: List[Tuple[int, int]],
    j: int,
) -> nx.MultiGraph:
    """Return a new nx.MultiGraph: contract first j edges of junction,
    delete the remaining (k - j). Contraction merges endpoints; any
    resulting parallel edges become multiplicities."""
    new = g.copy()
    for i, (u, v) in enumerate(junction_edges):
        if i < j:
            # Contract: remove all edges at v, rewire to u.
            if v in new:
                v_edges = list(new.edges(v, keys=True))
                new.remove_node(v)
                for a, b, _key in v_edges:
                    a2 = u if a == v else a
                    b2 = u if b == v else b
                    if a2 != b2:
                        new.add_edge(a2, b2)
        else:
            if new.has_edge(u, v):
                new.remove_edge(u, v)
    return new


def _is_bridge_junction(
    g: nx.MultiGraph, junction_edges: List[Tuple[int, int]],
) -> bool:
    """Return True iff removing all junction edges disconnects g."""
    test = g.copy()
    for u, v in junction_edges:
        if test.has_edge(u, v):
            test.remove_edge(u, v)
    return not nx.is_connected(test)


def _nx_mg_to_mg(g: nx.MultiGraph) -> MultiGraph:
    edge_counts: Dict[Tuple[int, int], int] = {}
    for u, v in g.edges():
        e = (min(u, v), max(u, v))
        edge_counts[e] = edge_counts.get(e, 0) + 1
    return MultiGraph(
        nodes=frozenset(g.nodes()),
        edge_counts=edge_counts,
        loop_counts={},
    )


def apply_kmatching_formula(
    graph: Graph,
    junctions: List[KMatchingJunction],
    synth_multigraph,
):
    """Apply the recursive cell-cycle/cell-tree formula.

    Iterates the 2-cell formula at each junction with state-caching.
    At each junction:
      - If removing the k junction edges disconnects the graph (tree
        case): coefficients are (x, k-1, C(k,2), ..., C(k,k)) for
        j = 0, 1, 2, ..., k.
      - Otherwise (cycle case): coefficients are C(k, j) for all j.

    `synth_multigraph(mg)` is a callable that returns T(mg) for a
    tutte.graph.MultiGraph (typically engine._synthesize_multigraph).
    """
    # Build nx.MultiGraph from the simple Graph
    g_nx = nx.MultiGraph()
    g_nx.add_nodes_from(graph.nodes)
    for u, v in graph.edges:
        g_nx.add_edge(u, v)

    # Translate each KMatchingJunction into an nx-edge list
    junction_edge_lists: List[List[Tuple[int, int]]] = [
        [(a, b) for a, b in junc.edges] for junc in junctions
    ]

    # Caches scoped to this call (do NOT leak across calls)
    leaf_cache: Dict[str, TuttePolynomial] = {}
    state_cache: Dict[tuple, TuttePolynomial] = {}

    def _t_leaf(mg: MultiGraph) -> TuttePolynomial:
        try:
            key = mg.canonical_key()
        except Exception:
            return synth_multigraph(mg)
        cached = leaf_cache.get(key)
        if cached is not None:
            return cached
        T = synth_multigraph(mg)
        leaf_cache[key] = T
        return T

    def _state_key(g: nx.MultiGraph, remaining_idx: Tuple[int, ...]) -> Optional[tuple]:
        try:
            mg_key = _nx_mg_to_mg(g).canonical_key()
        except Exception:
            return None
        return (mg_key, remaining_idx)

    x_poly = TuttePolynomial.x()

    def _resolve_edge(
        relabel: Dict[int, int], u: int, v: int,
    ) -> Tuple[int, int]:
        """Walk the relabel dict to get the current canonical
        representatives of u, v."""
        def _root(x: int) -> int:
            while x in relabel:
                x = relabel[x]
            return x
        return (_root(u), _root(v))

    def recurse(
        g: nx.MultiGraph,
        remaining_idx: Tuple[int, ...],
        relabel: Dict[int, int],
    ) -> TuttePolynomial:
        if not remaining_idx:
            return _t_leaf(_nx_mg_to_mg(g))
        key = _state_key(g, remaining_idx)
        if key is not None:
            cached = state_cache.get(key)
            if cached is not None:
                return cached

        # Resolve each remaining junction's current edges via relabel.
        # A junction's edges may have moved to merged vertices; resolve
        # each (u, v) through the relabel dict to get the current
        # physical endpoints, then verify the edge exists in g.
        def _present_for(idx: int) -> List[Tuple[int, int]]:
            resolved: List[Tuple[int, int]] = []
            for u, v in junction_edge_lists[idx]:
                ru, rv = _resolve_edge(relabel, u, v)
                if ru == rv:
                    # Endpoints merged → edge became a self-loop and
                    # was eliminated during contraction; not part of
                    # this junction anymore.
                    continue
                if g.has_edge(ru, rv):
                    resolved.append((ru, rv))
            return resolved

        # Pick next junction: prefer bridge-junctions for cache reuse.
        chosen = 0
        for i, idx in enumerate(remaining_idx):
            present = _present_for(idx)
            if present and _is_bridge_junction(g, present):
                chosen = i
                break
        junc_idx = remaining_idx[chosen]
        other_idx = remaining_idx[:chosen] + remaining_idx[chosen + 1:]

        present = _present_for(junc_idx)
        if not present:
            result = recurse(g, other_idx, relabel)
            if key is not None:
                state_cache[key] = result
            return result

        k_eff = len(present)
        is_bridge = _is_bridge_junction(g, present)

        total = TuttePolynomial.zero()
        for j in range(0, k_eff + 1):
            g_j = _apply_junction_merge(g, present, j)
            # Extend relabel: for each contracted edge (u, v) with i < j,
            # v merged into u. Build child relabel.
            child_relabel = dict(relabel)
            for i, (u, v) in enumerate(present):
                if i < j:
                    child_relabel[v] = u

            T_j = recurse(g_j, other_idx, child_relabel)
            if is_bridge:
                if j == 0:
                    coeff_poly = x_poly
                elif j == 1:
                    coeff = k_eff - 1
                    if coeff == 0:
                        continue
                    coeff_poly = coeff * TuttePolynomial.one()
                else:
                    coeff_poly = math.comb(k_eff, j) * TuttePolynomial.one()
            else:
                coeff_poly = math.comb(k_eff, j) * TuttePolynomial.one()
            total = total + coeff_poly * T_j

        if key is not None:
            state_cache[key] = total
        return total

    all_idx = tuple(range(len(junctions)))
    return recurse(g_nx, all_idx, {})


# Cache: (graph_canonical_key, table_id) → partition result.
# `engine.synthesize` calls this 6+ times per graph (cell_quotient_grid,
# cycle, tree, bipartite_junction, per_component, hybrid all retry). Without
# caching, each call repeats find_cell_candidates + VF2 partitioning. For
# Pegasus-like graphs with non-tree cell topology this is the bottleneck
# (>500s wasted on Pm_2 after the May 17 K_{a,b} reconstruction fix
# unblocked atlas_49 / K_4 as candidates). Keyed by (canon, id(table)) so
# different tables don't collide.
_HIER_PARTITION_CACHE: Dict[Tuple[str, int], Optional[Tuple]] = {}
# Multi-partition cache for `iter_hierarchical_partitions`: stores the
# full priority-ordered list of valid tilings (not just the first match).
_HIER_PARTITIONS_ALL_CACHE: Dict[Tuple[str, int], List[Tuple]] = {}
_HET_PARTITION_CACHE: Dict[Tuple[str, int, int, int, int], Optional[Tuple]] = {}


def _hierarchical_candidate_priority(name: str) -> int:
    """Priority for ordering rainbow-table candidates in VF2 partition search.

    Lower number = higher priority. D-Wave native cells (Cm, Pm, Z)
    first because when they DO tile a target, they give the FEWEST-cell
    partition, which is what the unified chord-junction theorem needs
    (e.g., Z(1, 2) prefers 2× Z1_1 over 4× K_{3,3} — both valid, but
    only the former lets the engine use the Z(1, 1) merger cache).
    K_n/K_{a,b} are universal fallbacks. Asymmetric Book/Pan/Sunlet
    before vertex-transitive Ladder/Prism/Mobius/Fan — empirically
    (May 22 2026) Z(2,1) Mobius_10/Ladder_10/Prism_10/Fan_9 each
    burned 9-25s on no-match exhaustion before Book_3 succeeded in
    0.01s. Cm3 with Ladder_12 candidate took 1249s before the K-first
    gate landed; D-Wave cells are still gated above Ladder so that
    regression doesn't recur.
    """
    if name.startswith(('Cm', 'Pm', 'Z')):
        return 0  # D-Wave native cells — prefer fewest-cell partition
    if name.startswith('K_'):
        return 1  # K_n, K_{a,b} — universal fallback
    if name.startswith('Grid_'):
        return 2  # Grid_n×m
    if name.startswith(('C_', 'W_', 'P_', 'S_')):
        return 3  # Cycle/Wheel/Path/Star
    if name.startswith(('Book_', 'Pan_', 'Sunlet_')):
        return 5  # Asymmetric, fast VF2 reject
    # Ladder_, Prism_, Mobius_, Fan_ — vertex-transitive/symmetric.
    return 9


def try_hierarchical_partition(
    graph: Graph,
    table: RainbowTable
) -> Optional[Tuple[MinorEntry, List[Set[int]], InterCellInfo]]:
    """Try to partition graph into cells from the rainbow table.

    This is the main entry point for hierarchical tiling. It:
    1. Finds candidate cells that could tile the graph
    2. Tries to partition nodes into cell groups
    3. Verifies each group is isomorphic to the cell
    4. Analyzes inter-cell edges

    Args:
        graph: Graph to partition
        table: Rainbow table with potential tiles

    Returns:
        (cell_entry, partition, inter_cell_info) or None if no tiling found

    Cached by `(graph.canonical_key(), id(table))`. Cache is module-scoped
    so the same engine invocation's cell-quotient cascade reuses one
    partition lookup rather than re-running VF2 6+ times.

    For callers that need MULTIPLE valid tilings (e.g., to try a
    closed-form formula on each), use `iter_hierarchical_partitions`.
    """
    cache_key: Optional[Tuple[str, int]]
    try:
        cache_key = (graph.canonical_key(), id(table))
    except Exception:
        cache_key = None
    if cache_key is not None and cache_key in _HIER_PARTITION_CACHE:
        return _HIER_PARTITION_CACHE[cache_key]

    _log = get_log()
    # Find candidate tiles
    candidates = find_cell_candidates(graph, table)
    _log.record(EventType.CANDIDATE_FILTER, "covering",
                f"Hierarchical: {len(candidates)} cell candidates for {graph.node_count()}n {graph.edge_count()}e",
                LogLevel.DEBUG, graph=graph)

    candidates = sorted(
        candidates,
        key=lambda c: (_hierarchical_candidate_priority(c.name), -c.node_count),
    )

    for cell in candidates:
        cell_size = cell.node_count
        k = graph.node_count() // cell_size

        # Try to partition
        partition = partition_into_cells(graph, cell, k)
        if partition is None:
            continue

        # Verify partition
        if not verify_cell_partition(graph, partition, cell):
            continue

        # Analyze inter-cell edges
        inter_info = analyze_inter_cell_edges(graph, partition)
        _log.record(EventType.HIERARCHICAL, "covering",
                    f"Tiled with {cell.name}: {len(partition)} cells, "
                    f"{len(inter_info.edges)} inter-cell edges")

        result = (cell, partition, inter_info)
        if cache_key is not None:
            _HIER_PARTITION_CACHE[cache_key] = result
        return result

    if cache_key is not None:
        _HIER_PARTITION_CACHE[cache_key] = None
    return None


def iter_hierarchical_partitions(
    graph: Graph,
    table: RainbowTable,
    max_partitions: int = 4,
) -> List[Tuple[MinorEntry, List[Set[int]], InterCellInfo]]:
    """Return up to `max_partitions` valid homogeneous tilings in priority order.

    Same candidate iteration as `try_hierarchical_partition` but collects
    multiple successful partitions instead of stopping at the first. The
    merged decomposition-chord-peel dispatcher uses this to try closed-form
    formulas against several partitions before falling back to chord-rule —
    Z(1,2) tiles as both 4× K_{3,3} and 2× Z1_1; only the latter may admit
    the product formula on a given inter-cell topology.

    Bounded at 4 partitions to cap VF2 cost on graphs admitting many
    distinct tilings.

    Cached separately from the single-result cache so repeat calls with
    different `max_partitions` still work (the cache stores the full
    priority-ordered list, sliced on read).
    """
    cache_key: Optional[Tuple[str, int]]
    try:
        cache_key = (graph.canonical_key(), id(table))
    except Exception:
        cache_key = None
    if cache_key is not None and cache_key in _HIER_PARTITIONS_ALL_CACHE:
        return _HIER_PARTITIONS_ALL_CACHE[cache_key][:max_partitions]

    _log = get_log()
    candidates = find_cell_candidates(graph, table)
    candidates = sorted(
        candidates,
        key=lambda c: (_hierarchical_candidate_priority(c.name), -c.node_count),
    )

    results: List[Tuple[MinorEntry, List[Set[int]], InterCellInfo]] = []
    for cell in candidates:
        if len(results) >= max_partitions:
            break
        cell_size = cell.node_count
        k = graph.node_count() // cell_size
        partition = partition_into_cells(graph, cell, k)
        if partition is None:
            continue
        if not verify_cell_partition(graph, partition, cell):
            continue
        inter_info = analyze_inter_cell_edges(graph, partition)
        _log.record(EventType.HIERARCHICAL, "covering",
                    f"Tiled with {cell.name}: {len(partition)} cells, "
                    f"{len(inter_info.edges)} inter-cell edges")
        results.append((cell, partition, inter_info))

    if cache_key is not None:
        _HIER_PARTITIONS_ALL_CACHE[cache_key] = results
    return results[:max_partitions]


def clear_hierarchical_partition_cache() -> None:
    """Clear the hierarchical / heterogeneous partition result caches.

    Useful for tests that mutate the rainbow table mid-run.
    """
    _HIER_PARTITION_CACHE.clear()
    _HIER_PARTITIONS_ALL_CACHE.clear()
    _HET_PARTITION_CACHE.clear()


def _find_induced_match(
    target: Graph,
    pattern: Graph,
    available: Set[int],
) -> Optional[Set[int]]:
    """Find one induced-subgraph copy of `pattern` whose node set lies in `available`.

    Uses VF2 induced-subgraph isomorphism on the subgraph of `target` induced
    by `available`. Returns the matching node set, or None if no copy fits.

    Pre-filters via degree-sequence inclusion (fast) before invoking VF2 — if
    the pattern's degree sequence is not a multiset-subsequence of the
    available subgraph's degree sequence, no induced copy can exist.
    """
    if len(available) < pattern.node_count():
        return None
    if pattern.edge_count() == 0:
        return set(list(available)[:pattern.node_count()])

    induced = target.subgraph(available)
    if induced.edge_count() < pattern.edge_count():
        return None

    # Degree-sequence pre-filter: every degree value in the pattern must be
    # achievable by some node in the available subgraph (modulo the
    # cardinality of nodes with that degree).
    pat_degrees = sorted(pattern.degree(n) for n in pattern.nodes)
    avail_degrees = sorted(induced.degree(n) for n in induced.nodes)
    # Multiset-inclusion check: walk both sorted lists; advance avail
    # pointer until each pattern degree is met.
    j = 0
    for d in pat_degrees:
        while j < len(avail_degrees) and avail_degrees[j] < d:
            j += 1
        if j >= len(avail_degrees):
            return None
        j += 1

    G_avail = induced.to_networkx()
    G_pat = pattern.to_networkx()

    matcher = isomorphism.GraphMatcher(G_avail, G_pat)
    for mapping in matcher.subgraph_isomorphisms_iter():
        return set(mapping.keys())
    return None


def try_heterogeneous_partition(
    graph: Graph,
    table: RainbowTable,
    *,
    min_cell_nodes: int = 3,
    min_cells: int = 2,
    min_graph_nodes: int = 10,
    max_graph_nodes: int = 100,
) -> Optional[Tuple[List[MinorEntry], List[Set[int]], InterCellInfo]]:
    """Greedy largest-first heterogeneous partitioner.

    Walks through rainbow-table cells in descending node-count order. For each
    cell, repeatedly finds induced copies in the still-unmatched portion of
    the graph and consumes them. Falls through to smaller cells until either
    the graph is fully covered or no further cells fit.

    Returns ``(cells, partition, inter_info)`` where ``cells[i]`` is the
    rainbow-table entry for the i-th part and ``partition[i]`` is its node
    set, or ``None`` if no full cover is found.

    Each cell must have ``node_count >= min_cell_nodes`` (default 3) so we
    don't tile with edges or single vertices, and the final partition must
    have at least ``min_cells`` parts (default 2). The hierarchical pipeline
    relies on the boundary quotient + chord recursion downstream, which only
    pays off when there's more than one cell.

    `min_graph_nodes` (default 12) gates the whole attempt — small graphs
    route through faster paths anyway, and VF2 induced-subgraph search is
    expensive, so we skip the whole thing on graphs that are too small to
    benefit.

    `max_graph_nodes` (default 100) gates large graphs: VF2 induced-subgraph
    search scales poorly with graph size. Empirically Pm3 (128n/704e) gets
    stuck for >6 min in a single dispatch call here on a ~1000-entry table
    — much longer than the engine's other fallbacks. Skip to let chord rule
    or treewidth_dp take over.

    Cached by `(graph.canonical_key(), id(table), min_cell_nodes,
    min_cells, min_graph_nodes)`. Chord-rule contractions on D-Wave
    decompositions repeatedly call this on intermediates that mostly
    return None; the cache turns repeated VF2 sweeps into O(1) misses.
    """
    target_nodes = graph.node_count()
    if target_nodes < min_graph_nodes:
        return None
    if target_nodes > max_graph_nodes:
        return None
    target_edges = graph.edge_count()

    cache_key: Optional[Tuple[str, int, int, int, int]]
    try:
        cache_key = (
            graph.canonical_key(), id(table),
            min_cell_nodes, min_cells, min_graph_nodes,
        )
    except Exception:
        cache_key = None
    if cache_key is not None and cache_key in _HET_PARTITION_CACHE:
        return _HET_PARTITION_CACHE[cache_key]

    # Reconstruct cell graphs once and sort by descending node count.
    # Restrict to "canonical" cell families (K_n, K_{a,b}, cycles, D-Wave
    # families). Named families like Pan_, Fan_, Helm_, Mobius_ are
    # reconstructible but they're noise as cells: their induced-subgraph
    # copies often overlap K_n / K_{a,b} matches at the same node count,
    # and the greedy partitioner picks them first, leaving incomplete
    # remainder.
    #
    # D-Wave family cells (Cm_, Pm_, Z*) are explicit structural
    # candidates — Z(1,3) decomposes as Z(1,2)+Z(1,1) (24+12=36n) and
    # similar heterogeneous patterns matter for the engine's downstream
    # cell-quotient DPs. Cm1 = K_{4,4} aliases would collide but that
    # specific entry isn't currently in the rainbow table.
    canonical_prefix = ('K_', 'C_', 'W_', 'P_', 'S_', 'Cm', 'Pm')
    canonical_names = {
        'Petersen', 'Heawood', 'MoebiusKantor', 'Desargues', 'Dodecahedral',
        # D-Wave Zephyr cells (no consistent prefix — Z1_1, Z1_2, ...)
        'Z1_1', 'Z1_2', 'Z1_3', 'Z2_1', 'Z2_2',
    }
    candidates: List[Tuple[MinorEntry, Graph]] = []
    for entry in table.entries.values():
        if entry.node_count < min_cell_nodes:
            continue
        if entry.node_count > target_nodes:
            continue
        # Cell edge density must not exceed the target's local density —
        # if every cell-sized region has fewer edges than the cell, no
        # induced match can exist anywhere.
        if entry.edge_count > target_edges:
            continue
        # A "trivial" tree/forest cell defeats the point of hierarchical
        # decomposition (T(tree) is closed-form via family recognition).
        if entry.edge_count < entry.node_count:
            continue
        # Restrict to canonical cell families (see above).
        if not (
            entry.name.startswith(canonical_prefix)
            or entry.name in canonical_names
        ):
            continue
        if entry.graph is None:
            cell_graph = _minor_to_graph(entry)
            if cell_graph is None:
                continue
            entry.graph = cell_graph  # cache reconstruction
        else:
            cell_graph = entry.graph
        candidates.append((entry, cell_graph))

    # Sort by (node_count DESC, edge_count DESC, prefer_canonical_name):
    # largest cells first, then densest among same-size cells, with
    # K_/C_/W_/P_/S_-prefixed names preferred over D-Wave aliases
    # (Cm_, Pm_, Z*) when both refer to the same canonical structure.
    # This is the implicit alias selection — e.g. K_{4,4} over Cm1.
    _classical_prefixes = ('K_', 'C_', 'W_', 'P_', 'S_')

    def _name_priority(name: str) -> int:
        # 0 = classical (K_/C_/...), 1 = named families, 2 = D-Wave (Cm/Pm/Z*).
        if name.startswith(_classical_prefixes):
            return 0
        if name.startswith(('Cm', 'Pm', 'Z')):
            return 2
        return 1

    candidates.sort(key=lambda pair: (
        -pair[0].node_count,
        -pair[0].edge_count,
        _name_priority(pair[0].name),
    ))

    # Deduplicate by canonical_key — aliased entries (e.g., Cm1 vs
    # K_{4,4} for the same 8n 16e bipartite graph) would otherwise
    # both appear and the greedy partitioner would double-match the
    # same shape. After the sort, the first occurrence of each
    # canonical key wins, preferring classical names.
    deduped: List[Tuple[MinorEntry, Graph]] = []
    seen_canon: Set[str] = set()
    for entry, cell_graph in candidates:
        try:
            canon = entry.canonical_key
        except Exception:
            canon = None
        if canon is not None:
            if canon in seen_canon:
                continue
            seen_canon.add(canon)
        deduped.append((entry, cell_graph))
    candidates = deduped

    unmatched: Set[int] = set(graph.nodes)
    partition: List[Set[int]] = []
    cells: List[MinorEntry] = []

    # Dynamic wall-clock budget for the entire candidate loop. Each
    # _find_induced_match call is a VF2 search that proves no-match by
    # exhaustion when the pattern doesn't fit. Budget scales with graph
    # size + candidate count: `2s + 0.5 × n_candidates + 0.1 × n_nodes`.
    # For Pm2 (40n, ~39 candidates) → ~25.5s; for Cm3 (72n) → ~37s.
    # Budget-exhausted → return None (cached negative; engine falls
    # through to chord rule / treewidth_dp).
    n_cands_het = max(1, len(candidates))
    n_nodes_het = graph.node_count()
    budget_seconds_het = 2.0 + 0.5 * n_cands_het + 0.1 * n_nodes_het
    import time as _time
    deadline = _time.monotonic() + budget_seconds_het

    for entry, cell_graph in candidates:
        cell_size = entry.node_count
        while len(unmatched) >= cell_size:
            if _time.monotonic() >= deadline:
                # Bail out; partial partition is incomplete by definition.
                if cache_key is not None:
                    _HET_PARTITION_CACHE[cache_key] = None
                return None
            tile = _find_induced_match(graph, cell_graph, unmatched)
            if tile is None:
                break
            partition.append(tile)
            cells.append(entry)
            unmatched -= tile

        if not unmatched:
            break

    if unmatched:
        if cache_key is not None:
            _HET_PARTITION_CACHE[cache_key] = None
        return None
    if len(partition) < min_cells:
        if cache_key is not None:
            _HET_PARTITION_CACHE[cache_key] = None
        return None
    # All cells identical → caller's homogeneous path already handles this;
    # don't shadow it here.
    if len({c.canonical_key for c in cells}) == 1:
        if cache_key is not None:
            _HET_PARTITION_CACHE[cache_key] = None
        return None

    inter_info = analyze_inter_cell_edges(graph, partition)
    get_log().record(
        EventType.HIERARCHICAL, "covering",
        f"Heterogeneous tiling: {len(partition)} cells "
        f"({', '.join(c.name for c in cells)}), "
        f"{len(inter_info.edges)} inter-cell edges",
    )
    result = (cells, partition, inter_info)
    if cache_key is not None:
        _HET_PARTITION_CACHE[cache_key] = result
    return result
