"""Base classes for Tutte polynomial synthesis.

Provides shared infrastructure used by all synthesis engines:
- UnionFind for bridge/chord classification
- BaseMultigraphSynthesizer with pattern recognition for multigraphs
- SynthesisResult dataclass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from ..graph import Graph, MultiGraph
from ..graphs.series_parallel import compute_sp_tutte_multigraph_if_applicable
from ..graphs.treewidth import compute_treewidth_tutte_if_applicable
from ..logs import EventType, LogLevel, get_log
from ..polynomial import TuttePolynomial

# =============================================================================
# UNION-FIND (for bridge/chord classification)
# =============================================================================

class UnionFind:
    """O(alpha(n)) union-find for bridge/chord classification."""

    def __init__(self, elements):
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}

    def find(self, x):
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union by rank. Returns True if x,y were in different sets."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


# =============================================================================
# BASE MULTIGRAPH SYNTHESIZER
# =============================================================================

class BaseMultigraphSynthesizer:
    """Base class for multigraph synthesis with pattern recognition.

    This class provides the shared multigraph synthesis logic used by
    SynthesisEngine. Subclasses must implement:
    - synthesize(graph) -> result with polynomial attribute
    - _log(msg) -> logging function
    - _multigraph_cache -> dict for caching multigraph polynomials
    """

    def _record_synth(self, graph_or_mg: Any, key: Optional[str] = None) -> None:
        """Record a graph that's being genuinely synthesized (not a lookup hit).

        Populates `self._synth_accum_graphs` — a Dict[canonical_key, Graph/MultiGraph]
        that the public `synthesize()` wrapper attaches to the top-level result
        so the visualizer can separate lookup-hits from real decomposition work.

        Always safe to call; ignores duplicates and handles missing attrs lazily.
        """
        accum = getattr(self, '_synth_accum_graphs', None)
        if accum is None:
            accum = {}
            self._synth_accum_graphs = accum
        try:
            k = key if key is not None else graph_or_mg.canonical_key()
        except Exception:
            return
        accum.setdefault(k, graph_or_mg)

    def _synthesize_multigraph(
        self,
        mg: MultiGraph,
        max_depth: int = 10,
        skip_minor_search: bool = False
    ) -> TuttePolynomial:
        """Synthesize polynomial for a multigraph with pattern recognition.

        Cheap structural checks (O(n+m)) are done BEFORE canonical_key (O(n²))
        to avoid expensive hashing on graphs that can be factored directly.

        Pattern recognition order:
        1. Handle loops: T(G with loop) = y × T(G without loop)
        2. Parallel edges formula (for simple 2-node multi-edge graphs)
        3. Disconnected: T(G1 ∪ G2) = T(G1) × T(G2)
        4. Cut vertex factorization: T(G1 · G2) = T(G1) × T(G2)
        4.5 Cache lookup (requires canonical_key, but cheaper than SP/treewidth)
        4.6 Series-parallel O(n) computation
        4.7 Treewidth-based O(n · B(w+1)²) computation
        5. Simple graph -> use regular synthesis
        6. Reduce parallel edges: T(G) = T(G\\e) + T(G/e)

        Args:
            mg: MultiGraph to synthesize
            max_depth: Maximum recursion depth (for subclass methods)
            skip_minor_search: If True, skip expensive minor search for simple graphs

        Returns:
            TuttePolynomial for the multigraph
        """
        _log = get_log()

        # 1. Handle loops first: T(G with loop) = y × T(G without loop)
        if mg.total_loop_count() > 0:
            loop_count = mg.total_loop_count()
            mg_no_loops = mg.remove_loops()
            return TuttePolynomial.y(loop_count) * self._synthesize_multigraph(mg_no_loops, max_depth, skip_minor_search)

        # 2. Parallel edges formula (very cheap check)
        if mg.is_just_parallel_edges():
            return self._parallel_edges_formula(mg.parallel_edge_count())

        # 3. Disconnected: T(G1 ∪ G2) = T(G1) × T(G2)
        if not mg.is_connected():
            _log.record(EventType.FACTORIZE, "base",
                        f"Disconnected MG: {mg.node_count()}n {mg.edge_count()}e",
                        LogLevel.DEBUG, graph=mg)
            return self._handle_disconnected_multigraph(mg, max_depth, skip_minor_search)

        # 4. Cut vertex factorization: T(G1 · G2) = T(G1) × T(G2)
        #    O(n+m) check, avoids expensive canonical_key for factorable graphs
        cut = mg.has_cut_vertex()
        if cut is not None:
            components = mg.split_at_cut_vertex(cut)
            if len(components) > 1:
                _log.record(EventType.FACTORIZE, "base",
                            f"Cut vertex in MG: {len(components)} components",
                            LogLevel.DEBUG, graph=mg)
                self._log(f"Cut vertex {cut} splits into {len(components)} components")
                poly = TuttePolynomial.one()
                for comp in components:
                    poly = poly * self._synthesize_multigraph(comp, max_depth, skip_minor_search)
                return poly

        # 4.5 Two-level cache lookup: fast_hash filter before expensive canonical_key
        #    _fast_hash_set contains hashes of all entries cached this session.
        #    If fast_hash not in set AND _fast_hash_set_complete is True, skip
        #    canonical_key entirely (guaranteed miss). Otherwise, fall through.
        #    This runs BEFORE SP/treewidth DP since cache hits are O(n²) vs O(n·B(w+1)²).
        fh = mg.fast_hash()
        if not hasattr(self, '_fast_hash_set'):
            self._fast_hash_set = set()
        if not hasattr(self, '_fast_hash_set_complete'):
            self._fast_hash_set_complete = not bool(self._multigraph_cache)

        if fh in self._fast_hash_set:
            cache_key = mg.canonical_key()
            if cache_key in self._multigraph_cache:
                _log.record(EventType.CACHE_HIT, "base",
                            f"MG cache hit: {mg.node_count()}n {mg.edge_count()}e",
                            LogLevel.DEBUG, graph=mg)
                # Record cache-hit multigraphs for the visualizer's
                # Contributing Graphs panel.
                self._record_synth(mg, cache_key)
                return self._multigraph_cache[cache_key]
        elif not self._fast_hash_set_complete:
            cache_key = mg.canonical_key()
            if cache_key in self._multigraph_cache:
                _log.record(EventType.CACHE_HIT, "base",
                            f"MG cache hit (unindexed): {mg.node_count()}n {mg.edge_count()}e",
                            LogLevel.DEBUG, graph=mg)
                self._fast_hash_set.add(fh)
                self._record_synth(mg, cache_key)
                return self._multigraph_cache[cache_key]
        else:
            cache_key = None  # Guaranteed miss — skip canonical_key

        # Past all cache gates: record this MG as synthesized from scratch.
        if cache_key is not None:
            self._record_synth(mg, cache_key)

        # 4.6 Series-parallel multigraph O(n) computation
        sp_poly = compute_sp_tutte_multigraph_if_applicable(mg)
        if sp_poly is not None:
            self._log(f"SP multigraph: {mg.node_count()} nodes, {mg.edge_count()} edges")
            if cache_key is None:
                cache_key = mg.canonical_key()
            self._multigraph_cache[cache_key] = sp_poly
            self._fast_hash_set.add(fh)
            return sp_poly

        # Cell-decomposition-only chord-peel for simple multigraph
        # contractions. Chord-rule LEAVES occasionally have cell structure
        # (Z(1,1) inside a Z(1,2)-leaf, Cm1 inside a Cm2-leaf) that
        # decomposes via the merged dispatcher faster than treewidth_dp.
        # We pass `skip_atoms=True` so only the gated cell discovery
        # fires — atom discovery would succeed on almost every contraction
        # and cascade chord-rule indefinitely (each contraction →
        # _synthesize_multigraph → here → another chord-peel → more
        # contractions...). The earlier hierarchical dispatch got this gate
        # for free because cell partition discovery rarely matched a
        # contracted intermediate; the merged dispatcher's atom path is
        # too eager, so we explicitly suppress it here. Only the
        # SynthesisEngine subclass has this method; other multigraph
        # synthesizers (e.g. test stubs) skip via the hasattr check.
        if (
            mg.is_simple()
            and mg.edge_count() >= 20
            and hasattr(self, '_try_decomposition_chord_peel')
        ):
            simple = mg.to_simple_graph()
            if simple is not None:
                hier_result = self._try_decomposition_chord_peel(
                    simple, max_depth,
                    recurse_residue=False,
                    skip_atoms=True,
                )
                if hier_result is not None:
                    self._log(
                        f"Decomposition+peel (multigraph leaf): "
                        f"{mg.node_count()}n {mg.edge_count()}e"
                    )
                    if cache_key is None:
                        cache_key = mg.canonical_key()
                    self._multigraph_cache[cache_key] = hier_result.polynomial
                    self._fast_hash_set.add(fh)
                    if hasattr(self, '_mg_minors_accum'):
                        self._mg_minors_accum |= hier_result.minors_used
                    return hier_result.polynomial

        # 4.7 Treewidth-based O(n · B(w+1)²) computation
        # max_width=10: matches engine.py:999 + hybrid.py:640 gate.
        # Python tw_dp at tw=11 takes 3-10+ min on n=40 graphs; the
        # C-ext is gated 5 <= tw <= 10. For multigraph contractions
        # produced during chord-rule, falling through to chord-rule
        # recursion is faster than Python tw_dp at tw=11.
        tw_poly = compute_treewidth_tutte_if_applicable(mg, max_width=10)
        if tw_poly is not None:
            self._log(f"Treewidth-based: {mg.node_count()} nodes, {mg.edge_count()} edges")
            if cache_key is None:
                cache_key = mg.canonical_key()
            self._multigraph_cache[cache_key] = tw_poly
            self._fast_hash_set.add(fh)
            return tw_poly

        self._log(f"Synthesizing multigraph: {mg.node_count()} nodes, {mg.edge_count()} edges")

        # 6. Simple graph -> use regular synthesis (or fast path)
        if mg.is_simple():
            simple = mg.to_simple_graph()
            if simple is not None:
                if skip_minor_search:
                    result = self._synthesize_fast(simple, max_depth)
                else:
                    result = self.synthesize(simple, max_depth)
                # Track minors: both from sub-synthesis AND table entry identity
                simple_key = simple.canonical_key()
                if hasattr(self, '_mg_minors_accum'):
                    self._mg_minors_accum |= result.minors_used
                    # If this graph is a known table entry, track it as a used minor
                    if simple_key in self.table.entries:
                        self._mg_minors_accum.add(simple_key)
                # Record the simple graph as synthesized-from-scratch for the visualizer.
                self._record_synth(simple, simple_key)
                if cache_key is None:
                    cache_key = mg.canonical_key()
                self._multigraph_cache[cache_key] = result.polynomial
                self._fast_hash_set.add(fh)
                return result.polynomial

        # 7. Batch reduce parallel edges using closed-form formula
        # For k parallel edges between u,v:
        #   Connected case: T(G) = T(G_0) + T(G_c) * (1 + y + ... + y^{k-1})
        #   Disconnected case: T(G) = T(G_c) * (x + y + ... + y^{k-1})
        # where G_0 = G without u-v edges, G_c = G_0 with u,v merged
        max_mult_edge = max(mg.edge_counts.keys(), key=lambda e: mg.edge_counts[e])
        if mg.edge_counts[max_mult_edge] > 1:
            k = mg.edge_counts[max_mult_edge]
            _log.record(EventType.MULTIGRAPH_OP, "base",
                        f"Batch reduce {k} parallel edges: {mg.node_count()}n {mg.edge_count()}e",
                        LogLevel.DEBUG, graph=mg)
            poly = self._batch_reduce_parallel(mg, max_mult_edge, max_depth, skip_minor_search)
            if cache_key is None:
                cache_key = mg.canonical_key()
            self._multigraph_cache[cache_key] = poly
            self._fast_hash_set.add(fh)
            return poly

        # 8. Fall back (should not be reached)
        raise RuntimeError(
            f"D-C fallback reached for multigraph with "
            f"{mg.node_count()} nodes, {mg.edge_count()} edges"
        )

    def _parallel_edges_formula(self, k: int) -> TuttePolynomial:
        """Compute Tutte polynomial for k parallel edges between 2 nodes.

        The formula is derived from deletion-contraction:
        - T(1 edge) = x
        - T(2 parallel) = T(1 edge) + T(loop) = x + y
        - T(3 parallel) = T(2 parallel) + T(2 loops) = x + y + y^2
        - T(k parallel) = x + y + y^2 + ... + y^{k-1}

        General formula:
        T(k parallel) = x + sum_{i=1}^{k-1} y^i
        """
        if k <= 0:
            return TuttePolynomial.one()
        if k == 1:
            return TuttePolynomial.x()
        if k == 2:
            return TuttePolynomial.from_coefficients({(1, 0): 1, (0, 1): 1})  # x + y

        # T(k parallel) = x + y + y^2 + ... + y^{k-1}
        coeffs = {(1, 0): 1}  # x
        for i in range(1, k):
            coeffs[(0, i)] = 1  # + y^i
        return TuttePolynomial.from_coefficients(coeffs)

    def _should_parallelize(self, mg1: MultiGraph, mg2: MultiGraph) -> bool:
        """Check if two multigraphs are large enough to benefit from parallel synthesis."""
        if getattr(self, '_in_worker', False):
            return False
        MIN_NODES, MIN_EDGES = 12, 40
        return (mg1.node_count() >= MIN_NODES and sum(mg1.edge_counts.values()) >= MIN_EDGES and
                mg2.node_count() >= MIN_NODES and sum(mg2.edge_counts.values()) >= MIN_EDGES)

    def _merge_worker_cache(self, worker_cache: Dict[str, TuttePolynomial]) -> None:
        """Merge cache entries discovered by a worker process."""
        for key, poly in worker_cache.items():
            if key not in self._multigraph_cache:
                self._multigraph_cache[key] = poly
        self._fast_hash_set_complete = False

    def _handle_disconnected_multigraph(
        self,
        mg: MultiGraph,
        max_depth: int,
        skip_minor_search: bool = False
    ) -> TuttePolynomial:
        """Handle disconnected multigraph: T(G1 ∪ G2) = T(G1) × T(G2)."""
        start = next(iter(mg.nodes))
        visited = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in mg.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        comp1_edges = {e: c for e, c in mg.edge_counts.items() if e[0] in visited}
        comp1_loops = {n: c for n, c in mg.loop_counts.items() if n in visited}
        comp1 = MultiGraph(nodes=frozenset(visited), edge_counts=comp1_edges, loop_counts=comp1_loops)

        rest_nodes = mg.nodes - visited
        rest_edges = {e: c for e, c in mg.edge_counts.items() if e[0] in rest_nodes}
        rest_loops = {n: c for n, c in mg.loop_counts.items() if n in rest_nodes}
        rest = MultiGraph(nodes=frozenset(rest_nodes), edge_counts=rest_edges, loop_counts=rest_loops)

        return (self._synthesize_multigraph(comp1, max_depth, skip_minor_search) *
                self._synthesize_multigraph(rest, max_depth, skip_minor_search))

    def _batch_reduce_parallel(
        self,
        mg: MultiGraph,
        edge: Tuple[int, int],
        max_depth: int,
        skip_minor_search: bool = False
    ) -> TuttePolynomial:
        """Batch reduce all k parallel edges between u,v using closed-form formula.

        For k parallel edges between u,v:
          If u,v connected in G_0 (G without u-v edges):
            T(G) = T(G_0) + T(G_c) * (1 + y + y^2 + ... + y^{k-1})
          If u,v disconnected in G_0:
            T(G) = T(G_c) * (x + y + y^2 + ... + y^{k-1})

        where G_c = G_0 with u,v merged (no u-v edges, no loops from them).
        This replaces k sequential delete-contract steps with 2 recursive calls.
        """
        u, v = edge
        k = mg.edge_counts[edge]

        # Build G_0: remove ALL edges between u,v
        new_edge_counts = dict(mg.edge_counts)
        del new_edge_counts[edge]
        mg_0 = MultiGraph(
            nodes=mg.nodes,
            edge_counts=new_edge_counts,
            loop_counts=mg.loop_counts
        )

        # Build G_c: merge u,v in G_0 (no u-v edges to create loops)
        mg_c = mg_0.merge_nodes(u, v)

        # Compute y-geometric sum: 1 + y + y^2 + ... + y^{k-1}
        y_sum_coeffs = {}
        for i in range(k):
            y_sum_coeffs[(0, i)] = 1
        y_sum = TuttePolynomial.from_coefficients(y_sum_coeffs)

        # Check connectivity of u,v in G_0
        if mg_0.in_same_component(u, v):
            # Connected case: T(G) = T(G_0) + T(G_c) * y_sum
            if self._should_parallelize(mg_0, mg_c):
                from .parallel import parallel_synthesize_pair
                t_g0, t_gc = parallel_synthesize_pair(
                    self, mg_0, mg_c, max_depth, skip_minor_search)
            else:
                t_g0 = self._synthesize_multigraph(mg_0, max_depth, skip_minor_search)
                t_gc = self._synthesize_multigraph(mg_c, max_depth, skip_minor_search)
            return t_g0 + t_gc * y_sum
        else:
            # Disconnected case: T(G) = T(G_c) * (x - 1 + y_sum)
            # = T(G_c) * (x + y + y^2 + ... + y^{k-1})
            t_gc = self._synthesize_multigraph(mg_c, max_depth, skip_minor_search)
            x_plus_y_sum_coeffs = {(1, 0): 1}  # x
            for i in range(1, k):
                x_plus_y_sum_coeffs[(0, i)] = 1  # + y^i
            x_plus_y_sum = TuttePolynomial.from_coefficients(x_plus_y_sum_coeffs)
            return t_gc * x_plus_y_sum



# =============================================================================
# SYNTHESIS RESULT
# =============================================================================

@dataclass
class SynthesisResult:
    """Result of a graph synthesis attempt."""
    polynomial: TuttePolynomial
    recipe: List[str] = field(default_factory=list)  # Human-readable steps
    verified: bool = False
    method: str = "unknown"
    tiles_used: int = 0
    fringe_edges: int = 0
    minors_used: Set[str] = field(default_factory=set)  # Canonical keys of table entries used
    # Canonical keys of simple graphs the engine synthesized from scratch
    # (successful sub-problems, regardless of whether they appear in the table).
    synthesized_minors: Set[str] = field(default_factory=set)
    # Snapshot of the Graph object keyed by canonical_key, for visualizer
    # reconstruction of synthesized (non-table) graphs.
    synthesized_graphs: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "verified" if self.verified else "unverified"
        return f"SynthesisResult({self.polynomial}, method={self.method}, {status})"
