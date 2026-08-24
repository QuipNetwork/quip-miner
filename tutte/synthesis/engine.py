"""Creation-Expansion-Join Synthesis Engine.

This module implements the main synthesis algorithm for computing
Tutte polynomials using algebraic composition of known minors.

The algorithm:
1. Find all minors of input_graph from rainbow table
2. Select largest minor M by polynomial complexity
3. Tile input_graph with disjoint copies of M
4. Compute base polynomial: T = T(M₁) × T(M₂) × ... (multiplication)
5. For each edge connecting different tiles, apply k-join formula
6. Compute fringe = edges_in_cover - edges_in_input (over-coverage)
7. If fringe is empty: return polynomial
8. If fringe is small: adjust polynomial directly
9. Else: recurse on fringe, combine results

Base case: K₂ with T(x) = x

Alternative approach (algebraic):
Use polynomial division and GCD to decompose target polynomials into
known factors, without requiring graph-level tiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ..cotree_dp import compute_tutte_almost_cograph, compute_tutte_cotree_dp
from ..family_recognition import recognize_family
from ..graph import Graph, MultiGraph, compute_signature
from ..graphs.covering import (Cover, Fringe, InterCellInfo, KMatchingJunction,
                               Tile, analyze_tile_connections,
                               apply_kmatching_formula, compute_fringe,
                               compute_inter_tile_edges,
                               detect_kmatching_topology,
                               extract_cell_topology, find_disjoint_cover,
                               iter_hierarchical_partitions,
                               try_heterogeneous_partition,
                               try_hierarchical_partition)
from ..graphs.series_parallel import compute_sp_tutte_if_applicable
from ..logs import EventType, LogLevel, get_log
from ..lookup.core import MinorEntry, RainbowTable, load_default_table
from ..lookup.merger import MergerTable, load_default_merger_table
from ..polynomial import TuttePolynomial
from ..roots import compute_cell_quotient_grid_dp_streamed
from ..transfer_matrix import compute_tutte_via_transfer_matrix
from ..validation import verify_spanning_trees
from .base import BaseMultigraphSynthesizer, SynthesisResult, UnionFind


def _is_complete_graph(g: Graph) -> bool:
    """Check if g is a complete graph K_n."""
    n = len(g.nodes)
    return len(g.edges) == n * (n - 1) // 2


# =============================================================================
# DECOMPOSITION RECORD (used by _try_decomposition_chord_peel)
# =============================================================================

@dataclass
class Decomposition:
    """A candidate decomposition of a target graph for chord-peel dispatch.

    Unifies the two granularities the engine previously handled separately:
      - ATOMS  (lightweight family shapes: K_n, K_{a,b}, B_n, W_n, L_n, Y_n)
               discovered via `tutte/graphs/atom_detection.py`. Cheap to
               find (ms) but carry no precomputed polynomial.
      - CELLS  (rainbow-table `MinorEntry` partitions) discovered via VF2
               in `tutte/graphs/covering.py`. Slow but cached, and each
               component has a precomputed Tutte polynomial — required by
               the cell-only closed-form formulas (unified_formula,
               kmatching_formula, product_formula).

    A Decomposition is consumed by `_try_decomposition_chord_peel` in two
    phases: cell-only closed-form trial (Phase B), then cost-gated
    chord-rule peel (Phase C).
    """
    kind: str                                    # "atom" | "cell"
    label: str                                   # e.g. "inter_legacy",
                                                 #      "intra_legacy",
                                                 #      "homo_K_{4,4}"
    components: List[Set[int]]                   # vertex sets, one per atom/cell
    families: List[str]                          # per-component family name
    cell_entries: Optional[List[MinorEntry]]     # only for kind=="cell";
                                                 # carries precomputed polys
                                                 # for closed-form paths
    inter_edges: List[Tuple[int, int]]           # all inter-component edges
    chord_edges: List[Tuple[int, int]]           # the edges chord-rule will peel
                                                 # (for atom "intra" peel: the
                                                 # internal K_n edges)
    predicted_chord_cost: float                  # edges × per_edge × tw_ratio
    peel_type: str                               # "inter" | "intra"
    inter_info: Optional[InterCellInfo]          # cell adjacency metadata for
                                                 # closed-form formulas;
                                                 # None for atom decompositions


# =============================================================================
# SYNTHESIS ENGINE
# =============================================================================

class SynthesisEngine(BaseMultigraphSynthesizer):
    """Main synthesis engine using creation-expansion-join algorithm."""

    def __init__(
        self,
        table: Optional[RainbowTable] = None,
        verbose: bool = False,
        auto_promote: bool = False,
        promote_cache_on_finish: bool = False,
        k_max: int = 12,
    ):
        """Initialize synthesis engine.

        Args:
            table: Rainbow table for lookups (loads default if None)
            verbose: Print progress information
            auto_promote: If True, auto-promote synthesized simple graphs to the rainbow table
            promote_cache_on_finish: If True, at the end of each top-level
                `synthesize()` call, promote every simple-graph entry in
                `self._cache` to the rainbow table. Lets cache_hits from
                the current run become `lookup_hit`s in the next run.
            k_max: Maximum k for k-sum vertex-separator search (default 12).
                Bounded by the engine's top-20-degree candidate filter; values
                above 20 are clamped. The chord rule is `1 + C(k,2)` syntheses
                per attempt, so larger k is uniformly more expensive.
        """
        self.table = table if table is not None else load_default_table()
        # Auto-load the rooted-Tutte lookup table from `tutte/data/`.
        # Tries `rooted_lookup.bin` first, falls back to JSON. Partitions
        # are stored in canonical labels (per WL refinement) and translated
        # to the runtime graph's actual labels on cache hit, so any
        # isomorphic graph reuses the same entry regardless of labeling.
        try:
            from ..roots.rooted_tutte import load_default_rooted_lookup
            load_default_rooted_lookup()
        except Exception:
            pass  # best-effort; engine still works via on-demand compute
        self.verbose = verbose
        self.auto_promote = auto_promote
        self.promote_cache_on_finish = promote_cache_on_finish
        self.k_max = max(2, min(k_max, 20))
        # Smart ordering sorts chord edges by descending
        # |common_neighbors(u, v)| in the original graph, so high-impact
        # contractions happen early and the engine's parallel-edge / loop fast
        # paths fire sooner. Set to False to revert if a regression surfaces
        # for a chord-rule-heavy target (Pm3+, Cm3+, Z(2,t)+).
        self.chord_smart_order: bool = True
        # σ-equivariant chord ordering: when set, `_iterative_chord_rule`
        # reorders chords so σ-orbits are contiguous, maximizing
        # engine.canonical_key cache hits on isomorphic intermediate
        # contractions. Petersen measured ~1.44× speedup with results
        # matching bit-for-bit; larger chord-rule targets (Pm_2, Z(1,3))
        # expected to see larger gains because |Aut| × orbit-pair savings
        # scale with cell count.
        self.chord_sigma_order: bool = True
        self._cache: Dict[str, SynthesisResult] = {}
        self._multigraph_cache: Dict[str, TuttePolynomial] = {}  # For multigraph polynomials
        self._fast_hash_set: Set[str] = set()  # Fast hashes of all cached multigraphs
        self._fast_hash_set_complete: bool = True  # True when _fast_hash_set covers all cache entries
        self._fast_simple_hash_set: Set[str] = set()  # Fast hashes of all cached simple graphs
        self._table_nm_set: Set[Tuple[int, int]] = {
            (e.node_count, e.edge_count) for e in self.table.entries.values()
        }
        self._inter_cell_cache: Dict[str, TuttePolynomial] = {}  # For inter-cell graph polynomials
        self._mg_minors_accum: Set[str] = set()  # Accumulates minors found during multigraph synthesis
        # Accumulates (canonical_key -> Graph|MultiGraph) for every sub-problem
        # the engine actually synthesized (not a cache/lookup hit). Attached to
        # the top-level SynthesisResult so the visualizer can split "contributing
        # graphs from the lookup table" vs "graphs synthesized along the way".
        self._synth_accum_graphs: Dict[str, object] = {}
        self._synth_depth: int = 0
        # Always auto-load the multigraph lookup table at init —
        # chord-rule contractions reuse cached intermediates whether or
        # not the caller plans to write back. This trades one disk read
        # (~50ms) for cache hits on chord-rule sub-syntheses across all
        # engine instances (visualizer reruns, successive targets,
        # heterogeneous chord-peel residues that match prior runs).
        try:
            self.load_multigraph_cache()
        except Exception:
            pass
        # Session-scoped chord-junction merger cache. Loaded from the
        # disk-backed `merger_lookup_table` (populated by the warmup
        # script). The engine grows this in-memory only during
        # synthesis — entries computed on cache miss are NOT written
        # back to disk. The warmup script is the only writer of the
        # persistent table; production engine instances are pure
        # readers + augmenters. See `tutte/roots/chord_junction_closed_form.py`.
        try:
            self._merger_session_cache: MergerTable = load_default_merger_table()
        except Exception:
            self._merger_session_cache = MergerTable()
        # When True, the top-level synthesize() call skips the rainbow-table
        # lookup for the input graph (but sub-problems may still be looked up).
        # Useful for visualizer runs where we want to see what the engine
        # would do without a direct hit on the target.
        self.skip_target_lookup: bool = False

    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[Synth] {msg}", flush=True)

    def _promote_to_table(self, graph: Graph, cache_key: str, result: 'SynthesisResult') -> None:
        """Auto-promote a synthesized simple graph to the rainbow table.

        Only promotes if auto_promote is enabled and the key is not already in the table.
        """
        if not self.auto_promote or cache_key in self.table.entries:
            return
        entry = MinorEntry(
            name=f"auto_{graph.node_count()}n{graph.edge_count()}e_{cache_key[:8]}",
            polynomial=result.polynomial,
            node_count=graph.node_count(),
            edge_count=graph.edge_count(),
            canonical_key=cache_key,
            spanning_trees=result.polynomial.num_spanning_trees(),
            num_terms=result.polynomial.num_terms(),
            graph=graph,
            signature=compute_signature(graph),
        )
        self.table.add_entry(entry)

    def save_rainbow_table(self, bin_path: str = None) -> None:
        """Save the rainbow table (with any auto-promoted entries) to disk.

        Args:
            bin_path: Path for binary format (default: tutte/data/lookup_table.bin)
        """
        import os

        from ..lookup.binary import save_binary_rainbow_table

        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        if bin_path is None:
            bin_path = os.path.join(base_dir, 'lookup_table.bin')

        self.table.resort()
        save_binary_rainbow_table(self.table, bin_path)

    def save_multigraph_cache(self) -> None:
        """Save the multigraph polynomial cache to the default location (binary)."""
        from ..lookup.core import save_default_multigraph_table
        save_default_multigraph_table(self._multigraph_cache)

    def load_multigraph_cache(self) -> int:
        """Load multigraph polynomial cache from default location.

        Tries binary format first, falls back to JSON.

        Returns the number of entries loaded.
        """
        from ..lookup.core import load_default_multigraph_table
        loaded = load_default_multigraph_table()
        count = 0
        for key, poly in loaded.items():
            if key not in self._multigraph_cache:
                self._multigraph_cache[key] = poly
                count += 1
        if count > 0:
            self._fast_hash_set_complete = False
        return count

    def synthesize(
        self,
        graph: Graph,
        max_depth: int = 10
    ) -> SynthesisResult:
        """Main entry point: compute Tutte polynomial via creation-expansion-join.

        Thin wrapper around `_synthesize_inner` that tracks recursion depth
        so the top-level call can stamp the final SynthesisResult with
        `synthesized_minors` and `synthesized_graphs` — the per-run snapshot
        of every sub-graph the engine actually synthesized (as opposed to
        retrieved from the rainbow table).
        """
        is_top = self._synth_depth == 0
        self._synth_depth += 1
        try:
            result = self._synthesize_inner(graph, max_depth)
        finally:
            self._synth_depth -= 1
        if is_top:
            result.synthesized_graphs = dict(self._synth_accum_graphs)
            result.synthesized_minors = set(self._synth_accum_graphs.keys())
            # Promote cache entries to the rainbow table so that
            # cache_hits become lookup_hits on subsequent runs.
            if self.promote_cache_on_finish:
                self._flush_cache_to_table()
        return result

    def _flush_cache_to_table(self) -> None:
        """Promote cache entries to the persistent lookup tables.

        - Simple-graph `self._cache` entries with matching snapshots in
          `self._synth_accum_graphs` → added to `self.table` (rainbow
          table) in memory, then saved to disk.
        - Multigraph `self._multigraph_cache` entries → saved to the
          default multigraph lookup table on disk.

        Skips rainbow-table entries that already exist. Disk I/O is
        guarded inside try/except so a save failure doesn't abort the
        synthesis result.

        Called at the end of each top-level `synthesize()` when
        `self.promote_cache_on_finish` is enabled.
        """
        # In-memory promotion: simple graph cache → rainbow table.
        new_simple_entries = 0
        for cache_key, result in list(self._cache.items()):
            if cache_key in self.table.entries:
                continue
            g = self._synth_accum_graphs.get(cache_key)
            if g is None or not hasattr(g, 'edges'):
                continue
            try:
                entry = MinorEntry(
                    name=(
                        f"auto_{g.node_count()}n{g.edge_count()}e_"
                        f"{cache_key[:8]}"
                    ),
                    polynomial=result.polynomial,
                    node_count=g.node_count(),
                    edge_count=g.edge_count(),
                    canonical_key=cache_key,
                    spanning_trees=result.polynomial.num_spanning_trees(),
                    num_terms=result.polynomial.num_terms(),
                    graph=g,
                    signature=compute_signature(g),
                )
                self.table.add_entry(entry)
                new_simple_entries += 1
            except Exception:
                continue

        # Persist both caches to disk so the NEXT run (new engine
        # instance) sees these entries as lookup_hit rather than
        # recomputing them. Merge with the on-disk tables to avoid
        # overwriting entries written by previous runs.
        try:
            if new_simple_entries > 0:
                self.save_rainbow_table()
        except Exception:
            pass
        try:
            if self._multigraph_cache:
                from ..lookup.core import (load_default_multigraph_table,
                                           save_default_multigraph_table)
                existing = load_default_multigraph_table()
                # Merge in-memory cache into the on-disk table. Only
                # add entries not already present; do not overwrite.
                for k, p in self._multigraph_cache.items():
                    existing.setdefault(k, p)
                save_default_multigraph_table(existing)
        except Exception:
            pass

    def _synthesize_inner(
        self,
        graph: Graph,
        max_depth: int = 10
    ) -> SynthesisResult:
        """Core synthesis logic — see `synthesize()` for the public entrypoint."""
        _log = get_log()
        n, m = graph.node_count(), graph.edge_count()
        _log.record(EventType.SYNTHESIS_START, "engine",
                     f"{n}n {m}e", LogLevel.INFO, graph=graph)

        # 1. Family recognition fast path — O(n+m)
        # Runs before canonical_key() to avoid the O(n² log n) cost for
        # known families. canonical_key is only needed for cache/table ops.
        family_poly = recognize_family(graph)
        if family_poly is not None:
            _log.record(EventType.FAMILY_RECOGNITION, "engine",
                        f"Family recognized: {n}n {m}e", LogLevel.INFO,
                        graph=graph)
            self._log(f"Family recognition: O(n+m) fast path")
            result = SynthesisResult(
                polynomial=family_poly,
                recipe=["Family recognition"],
                verified=True,
                method="family_recognition",
            )
            # Family recognition is synthesis-from-formula; count the input.
            self._record_synth(graph)
            return result

        # 1.5 Transfer matrix for periodic lattice strips — O(V+E) detection
        # Handles grid (m > 2), triangular, honeycomb, square-octagon and
        # elongated-triangular strips that family recognition doesn't cover.
        # Runs before canonical_key to avoid the O(n² log n) cost for lattice
        # graphs. The transfer-matrix module owns its own C-accelerated sweep
        # over non-crossing partition states.
        tm_poly = compute_tutte_via_transfer_matrix(graph)
        if tm_poly is not None:
            _log.record(EventType.SYNTHESIS_START, "engine",
                        f"Transfer matrix: {n}n {m}e", LogLevel.INFO,
                        graph=graph)
            self._log(f"Transfer matrix: O(V+E) detection + sweep")
            result = SynthesisResult(
                polynomial=tm_poly,
                recipe=["Transfer matrix"],
                verified=True,
                method="transfer_matrix",
            )
            self._record_synth(graph)
            return result

        # 2. Compute canonical key (expensive — O(n² log n))
        cache_key = graph.canonical_key()
        if cache_key in self._cache:
            _log.record(EventType.CACHE_HIT, "engine",
                        f"Cache hit: {cache_key[:12]}", LogLevel.DEBUG,
                        graph=graph)
            self._log(f"Cache hit: {cache_key[:16]}...")
            # Record the cache-hit graph so the visualizer surfaces it
            # under Contributing Graphs alongside from-scratch syntheses.
            self._record_synth(graph, cache_key)
            return self._cache[cache_key]

        self._log(f"Synthesizing graph with {n} nodes, {m} edges")

        # 3. Check rainbow table (optionally skipped for the top-level call
        # when skip_target_lookup is enabled — lets the visualizer show the
        # engine's decomposition path for a graph that happens to be in the table).
        is_top_call = self._synth_depth == 1
        if not (is_top_call and self.skip_target_lookup):
            cached = self.table.lookup(graph)
            if cached is not None:
                _log.record(EventType.LOOKUP_HIT, "engine",
                            f"Rainbow table hit: {n}n {m}e", graph=graph)
                self._log("Direct rainbow table lookup")
                result = SynthesisResult(
                    polynomial=cached,
                    recipe=["Rainbow table lookup"],
                    verified=True,
                    method="lookup",
                    minors_used={cache_key} if cache_key in self.table.entries else set(),
                )
                self._cache[cache_key] = result
                return result

        # Past the lookup gate: every remaining path is real synthesis work.
        # Record the input graph so the visualizer can show it in the
        # "synthesized" panel.
        self._record_synth(graph, cache_key)

        # 4. Handle base cases
        if graph.edge_count() == 0:
            _log.record(EventType.BASE_CASE, "engine", "Empty graph: T = 1",
                        graph=graph)
            result = SynthesisResult(
                polynomial=TuttePolynomial.one(),
                recipe=["Empty graph: T = 1"],
                verified=True,
                method="base_case"
            )
            self._cache[cache_key] = result
            return result

        if graph.edge_count() == 1:
            _log.record(EventType.BASE_CASE, "engine", "Single edge: T = x",
                        graph=graph)
            result = SynthesisResult(
                polynomial=TuttePolynomial.x(),
                recipe=["Single edge: T = x"],
                verified=True,
                method="base_case"
            )
            self._cache[cache_key] = result
            return result

        # 5. Check if graph is disconnected
        components = graph.connected_components()
        if len(components) > 1:
            _log.record(EventType.FACTORIZE, "engine",
                        f"Disconnected: {len(components)} components",
                        graph=graph)
            # Per-component snapshot + provenance for the visualizer.
            self._emit_subgraph_provenance(
                graph, [c.nodes for c in components], EventType.FACTORIZE,
                lambda i, vs: f"Connected component {i + 1}: {len(vs)}v",
            )
            result = self._synthesize_disconnected(components, max_depth)
            self._cache[cache_key] = result
            self._promote_to_table(graph, cache_key, result)
            return result

        # 6. Block-cut / cut-vertex factorization.
        # Structure gate (mirrors hybrid §3.5):
        #   - 0 articulation points  → biconnected, skip to §6.5 (2-sum)
        #   - 1 articulation point   → cut_vertex (single split, recursive)
        #   - 2+ articulation points → block-cut (one pass for all blocks;
        #                              subsumes recursive cut_vertex)
        try:
            import networkx as _nx
            nxg_for_arts = graph.to_networkx()
            arts = list(_nx.articulation_points(nxg_for_arts))
        except Exception:
            arts = []
        if len(arts) >= 2:
            blocks = list(_nx.biconnected_components(nxg_for_arts))
            _log.record(EventType.FACTORIZE, "engine",
                        f"Block-cut: {len(blocks)} blocks, "
                        f"{len(arts)} articulation points", graph=graph)
            # Per-block snapshot + provenance for the visualizer.
            self._emit_subgraph_provenance(
                graph, blocks, EventType.FACTORIZE,
                lambda i, vs: f"Block {i + 1}: {len(vs)}v",
            )
            self._log(f"Block-cut decomposition: {len(blocks)} blocks")
            poly = TuttePolynomial.one()
            recipe = [
                f"Block-cut: {len(blocks)} biconnected blocks, "
                f"{len(arts)} articulation points"
            ]
            all_minors: Set[str] = set()
            for i, block_vs in enumerate(blocks):
                block_subgraph = Graph(
                    nodes=frozenset(block_vs),
                    edges=frozenset(
                        (min(u, v), max(u, v))
                        for (u, v) in graph.edges
                        if u in block_vs and v in block_vs
                    ),
                )
                comp_result = self.synthesize(block_subgraph, max_depth)
                poly = poly * comp_result.polynomial
                recipe.append(
                    f"  Block {i + 1} ({len(block_vs)} verts): "
                    f"{comp_result.polynomial}"
                )
                all_minors |= comp_result.minors_used
            result = SynthesisResult(
                polynomial=poly,
                recipe=recipe,
                verified=True,
                method="block_cut",
                minors_used=all_minors,
            )
            self._cache[cache_key] = result
            self._promote_to_table(graph, cache_key, result)
            return result
        if len(arts) == 1:
            cut = arts[0]
            _log.record(EventType.FACTORIZE, "engine",
                        f"Cut vertex at {cut}", graph=graph)
            # Per-component snapshot + provenance for the visualizer.
            cut_components = graph.split_at_cut_vertex(cut)
            self._emit_subgraph_provenance(
                graph, [c.nodes for c in cut_components], EventType.FACTORIZE,
                lambda i, vs: f"Cut-vertex component {i + 1}: {len(vs)}v",
            )
            result = self._synthesize_via_cut_vertex(graph, cut, max_depth)
            self._cache[cache_key] = result
            self._promote_to_table(graph, cache_key, result)
            return result

        # 6.5. Early 2-sum / SPQR-style decomposition for biconnected graphs.
        # Mirrors hybrid §3.7. Gate (cheap-first):
        #   1. m ≥ 80 — small graphs are handled cheaply by downstream
        #      cascade; the gate probes alone cost ~10-50ms per graph
        #      which regressed small low-tw graphs.
        #   2. kappa == 2 AND treewidth upper bound > 8.
        if graph.edge_count() >= 80 and graph.node_count() >= 6:
            try:
                kappa = _nx.node_connectivity(nxg_for_arts)
            except Exception:
                kappa = 0
            invoke_ksum = False
            if kappa == 2:
                try:
                    from networkx.algorithms.approximation import (
                        treewidth_min_degree,
                    )
                    tw_upper, _ = treewidth_min_degree(nxg_for_arts)
                    invoke_ksum = tw_upper > 8
                except Exception:
                    invoke_ksum = False
            if invoke_ksum:
                try:
                    ksum_result = self._try_ksum_decomposition(graph)
                except Exception:
                    ksum_result = None
                if ksum_result is not None:
                    _log.record(EventType.FACTORIZE, "engine",
                                f"Early 2-sum: {n}n {m}e", graph=graph)
                    # Per-side snapshot + provenance. The 2-sum splits the
                    # graph at a 2-vertex separator; each side is itself a
                    # subgraph worth highlighting in the visualizer. The
                    # separator vertices appear in BOTH sides.
                    try:
                        sep = set(_nx.minimum_node_cut(nxg_for_arts))
                        residual = nxg_for_arts.copy()
                        residual.remove_nodes_from(sep)
                        sides = [
                            set(c) | sep
                            for c in _nx.connected_components(residual)
                        ]
                        self._emit_subgraph_provenance(
                            graph, sides, EventType.FACTORIZE,
                            lambda i, vs: (
                                f"2-sum side {i + 1}: {len(vs)}v "
                                f"(incl. 2-separator)"
                            ),
                        )
                    except Exception:
                        pass  # provenance is best-effort; non-fatal
                    self._log(f"Early 2-sum: {ksum_result.method}")
                    self._cache[cache_key] = ksum_result
                    self._promote_to_table(graph, cache_key, ksum_result)
                    return ksum_result

        # 7. Try series-parallel O(n) computation
        sp_poly = compute_sp_tutte_if_applicable(graph)
        if sp_poly is not None:
            _log.record(EventType.SERIES_PARALLEL, "engine",
                        f"SP decomposition: {n}n {m}e", graph=graph)
            self._log("Series-parallel: O(n) computation")
            result = SynthesisResult(
                polynomial=sp_poly,
                recipe=["Series-parallel decomposition"],
                verified=True,
                method="series_parallel",
            )
            self._cache[cache_key] = result
            self._promote_to_table(graph, cache_key, result)
            return result

        # 7.4 Chain recurrence — for cell-decomposable graphs whose
        # cell-quotient is a LINEAR PATH (n >= 3 cells). Uses
        # `compute_chain_full_poly_from_spec` which extracts a transfer
        # matrix from the (cell, junction) template, then iterates it n-1
        # times. For Chimera Cm(1, n) (n=K_{4,4} cells joined by M_4
        # matchings with shared-anchor interior cells), the transfer
        # matrix has order r=5, so cost is O(n·r) per evaluation point.
        # Cm(1, 6) solves in ~1s vs 60s+ timeout via the engine's
        # alternative paths.
        #
        # Gate edge_count >= 80 to skip small graphs (Cm(1, 3) at 56e
        # and Cm(1, 4) at 76e) where treewidth_dp at tw=4 is faster
        # (~0.1s vs ~5s for spec build + chain extraction).
        if graph.edge_count() >= 80:
            try:
                from ..roots.cell_quotient_bipartite_junction import (
                    build_bipartite_junction_spec,
                )
                from ..roots.chain_recurrence import (
                    compute_chain_full_poly_from_spec,
                    is_chain_topology,
                )
                _spec_built = build_bipartite_junction_spec(graph, self.table)
                if (_spec_built is not None
                        and is_chain_topology(_spec_built[0].cell_tree)
                        and _spec_built[0].cell_tree.number_of_nodes() >= 3):
                    chain_poly = compute_chain_full_poly_from_spec(_spec_built[0])
                    if (chain_poly is not None
                            and verify_spanning_trees(graph, chain_poly)):
                        _log.record(EventType.HIERARCHICAL, "engine",
                                    f"Chain recurrence: {n}n {m}e, "
                                    f"{_spec_built[0].cell_tree.number_of_nodes()} cells",
                                    graph=graph)
                        self._log(f"Chain recurrence: {n}n, {m}e")
                        result = SynthesisResult(
                            polynomial=chain_poly,
                            recipe=[
                                f"Chain recurrence: "
                                f"{_spec_built[0].cell_tree.number_of_nodes()} cells"
                            ],
                            verified=True,
                            method="chain_recurrence",
                            tiles_used=_spec_built[0].cell_tree.number_of_nodes(),
                            fringe_edges=0,
                        )
                        self._cache[cache_key] = result
                        self._promote_to_table(graph, cache_key, result)
                        return result
            except Exception:
                pass  # any failure — fall through

        # 7.45 Cell-quotient grid DP. For
        # cell-decomposable graphs whose cell-quotient is a 2D grid of
        # K_{a,b}-style cells connected by M_k matchings with disjoint
        # per-direction anchors (Cm2-style — Cm3 has shared anchors and
        # is rejected by precondition). Beats the §7.5 formula
        # short-circuit on Cm2 (~36 s vs ~55 s, 1.5×). Returns None
        # on any precondition mismatch so the engine falls through.
        # Same edge_count >= 60 gate as the other hierarchical paths.
        if graph.edge_count() >= 60:
            try:
                grid_streamed_poly = compute_cell_quotient_grid_dp_streamed(
                    graph, self.table,
                )
                if grid_streamed_poly is not None:
                    return self._emit_cell_quotient_result(
                        graph, cache_key, grid_streamed_poly,
                        method='cell_quotient_grid_dp_streamed', recipe='Cell-quotient grid DP (streamed)',
                        label='Cell-quotient grid DP (streamed)', record_label='Grid DP (streamed)',
                    )
            except Exception:
                pass  # any failure — fall through

        # 7.5 Hierarchical-formula short-circuit 
        # When the graph has a detectable cell
        # decomposition AND its inter-cell structure satisfies the
        # preconditions of the unified or k-matching formula, these
        # closed-form paths can beat `treewidth_dp` (Cm2: ~4× speedup
        # vs tw_dp). The formula-only shortcut avoids the internal
        # tw_dp fall-through inside the decomposition path, so we
        # only commit to that path when the formula
        # actually applies.
        #
        # Gate: edge_count ≥ 60. This filters out small structured
        # graphs (Petersen, small cycles, atlas graphs) where the
        # partition/detection overhead outweighs the tw_dp cost. Cm2
        # (80e) and Cm3 (192e) clear the gate; Z(1,1) (22e), Cm1
        # (16e), Petersen (15e) skip the shortcut and go straight to
        # tw_dp.
        if graph.edge_count() >= 60:
            formula_result = self._try_formula_shortcircuit(graph, max_depth)
            if formula_result is not None:
                _method_event = {
                    "unified_formula": EventType.UNIFIED_FORMULA,
                    "kmatching_formula": EventType.KMATCHING_FORMULA,
                }.get(formula_result.method, EventType.HIERARCHICAL)
                _log.record(_method_event, "engine",
                            f"Formula shortcut via {formula_result.method}: "
                            f"{formula_result.tiles_used} tiles",
                            graph=graph)
                self._cache[cache_key] = formula_result
                self._promote_to_table(graph, cache_key, formula_result)
                return formula_result

        # 7.5. Cotree DP — subexponential exp(O(n^{2/3})) for cographs
        # (P_4-free graphs). Wins where treewidth_dp can't fit (K_12+) and
        # the graph is a cograph. Fast no-op (early reject) on non-cographs.
        try:
            cotree_poly = compute_tutte_cotree_dp(graph)
            if cotree_poly is not None:
                _log.record(EventType.COTREE_DP, "engine",
                            f"Cotree DP: {n}n {m}e", graph=graph)
                self._log(f"Cotree DP: {n}n, {m}e")
                result = SynthesisResult(
                    polynomial=cotree_poly,
                    recipe=["Cotree-based DP (subexponential cograph)"],
                    verified=True,
                    method="cotree_dp",
                )
                self._cache[cache_key] = result
                self._promote_to_table(graph, cache_key, result)
                return result
        except (ValueError, TypeError):
            pass  # not a cograph, or input rejected — fall through

        # 7.6. Almost-cograph DP — for graphs that become cographs after
        # removing a small set of anomaly edges (e.g., D-Wave cells joined by
        # sparse inter-cell edges). Greedy P_4 elimination + bridge-aware
        # iterated chord rule on anomalies; the cograph skeleton routes
        # through cotree_dp. Returns None if anomaly count > max_anomalies.
        # Gate: cap at 16 anomalies (covers Cm2's 16 inter-cell edges; Cm3's
        # 48 falls through to existing chord-rule paths).
        try:
            almost_poly = compute_tutte_almost_cograph(
                graph, self, max_anomalies=16,
            )
            if almost_poly is not None:
                _log.record(EventType.COTREE_DP, "engine",
                            f"Almost-cograph DP: {n}n {m}e", graph=graph)
                self._log(f"Almost-cograph DP: {n}n, {m}e")
                result = SynthesisResult(
                    polynomial=almost_poly,
                    recipe=["Almost-cograph DP (greedy P_4 elim + chord rule)"],
                    verified=True,
                    method="almost_cograph",
                )
                self._cache[cache_key] = result
                self._promote_to_table(graph, cache_key, result)
                return result
        except Exception:
            pass  # any failure — fall through

        # 7.88 Unified decomposition + chord-peel — discovers atom AND cell
        # decompositions in one pass, tries cell-only closed-form formulas
        # (unified_formula / kmatching_formula / product_formula), then
        # applies cost-gated chord-rule on whichever decomposition has the
        # cheapest predicted peel. Recursive residue peel exposes a SECOND
        # decomposition that contraction may have revealed (the user's
        # "2-chord" framing). Replaces legacy steps 7.88a (unified atom),
        # 7.88 (cross-cell), 7.9 (clique-atom), and step 10 (hierarchical).
        # See `tutte/docs/08_5_decomposition_chord_peel.md`.
        #
        # Gates:
        #   - edge_count >= 20 (smallest graph admitting cell discovery)
        #   - node_count <= 30 (chord-rule's per-step sub-synth cost
        #     scales as 2^tw·m; larger graphs (e.g. Pm(2) 40n/164m) are
        #     reliably faster via step 8 treewidth_dp(max_width=10)
        #     fallback than chord-rule on the same shape)
        #   - synth_depth <= 2: at depth 1 (top) full discovery; at
        #     depth 2 (recursive intermediate) the dispatcher auto-skips
        #     cell discovery + closed-form trials (`skip_cells=True` in
        #     `_discover_decompositions`), matching legacy step 7.9
        #     `_try_clique_atom_chord_peel`'s lighter behavior on
        #     contracted intermediates. Deeper recursion (depth >= 3)
        #     was tested but causes Z(1,2) to timeout: chord-rule fans
        #     out by branching factor `chord_edges`, so depth 3 of 4-
        #     edge peeling gives ~64 leaves each requiring sub-synth.
        #     The engine cache helps when intermediates collide on
        #     canonical_key but in practice they often don't.
        # Edge-count lower bound: need enough structure to bother with
        # discovery (skip trivially small inputs that other steps
        # handle in microseconds). The upper-bound has been DROPPED:
        # the cache-aware probe inside `_try_decomposition_chord_peel`
        # (Phase C) now decides per-graph whether the chord-peel
        # contractions will hit `self._multigraph_cache`. Graphs
        # without warmed intermediates (Pm(2), Z(2,1) cold) get
        # rejected by the probe and fall through to step 8; graphs
        # whose contractions match cached entries (Z(1,2) post-warmup)
        # proceed. This makes the gate self-tuning: warming a target
        # via `warmup_chord_peel_cache.py` automatically unlocks
        # chord-peel for it on subsequent runs.
        if (graph.edge_count() >= 20 and self._synth_depth <= 2):
            try:
                dp_result = self._try_decomposition_chord_peel(
                    graph, max_depth,
                )
                if dp_result is not None:
                    _method_event = {
                        "unified_formula": EventType.UNIFIED_FORMULA,
                        "kmatching_formula": EventType.KMATCHING_FORMULA,
                        "product_formula": EventType.HIERARCHICAL,
                    }.get(dp_result.method, EventType.CHORD_RULE)
                    _log.record(_method_event, "engine",
                                f"Decomposition+peel via {dp_result.method}: "
                                f"{n}n {m}e", graph=graph)
                    self._log(
                        f"Decomposition+peel: {n}n, {m}e "
                        f"({dp_result.method})"
                    )
                    self._cache[cache_key] = dp_result
                    self._promote_to_table(graph, cache_key, dp_result)
                    return dp_result
            except Exception:
                pass  # any failure — fall through to step 8 treewidth_dp

        # 8. Treewidth DP — fast for graphs with treewidth ≤ 10. When this
        # succeeds it's usually the best path for graphs that fit. For graphs
        # whose treewidth exceeds the cap, returns None and we fall through to
        # the chord-rule paths below.
        # Gate at tw ≤ 10: the C-extension fast path is gated 5 ≤ tw ≤ 10
        # (`tutte/graphs/treewidth.py:1260`). Python at tw=11 takes 3-10 min
        # on n=40 graphs (e.g. Z(2,1), tw=11) — measured stuck >10 min on
        # May 24 profile. Lowering this cap forces such graphs into chord-
        # rule / spanning-tree fallbacks which converge faster. Cm(2) is
        # also tw=11 but reaches step 7.45 cell_quotient_grid_dp_streamed
        # first, so unaffected.
        if graph.edge_count() >= 10:
            from ..graphs.treewidth import \
                compute_treewidth_tutte_if_applicable
            full_mg = MultiGraph.from_graph(graph)
            tw_poly = compute_treewidth_tutte_if_applicable(full_mg, max_width=10)
            if tw_poly is not None:
                _log.record(EventType.TREEWIDTH_DP, "engine",
                            f"Treewidth DP: {n}n {m}e", graph=graph)
                self._log(f"Treewidth DP: {graph.node_count()}n, {graph.edge_count()}e")
                result = SynthesisResult(
                    polynomial=tw_poly,
                    recipe=["Treewidth-based DP (full graph)"],
                    verified=True,
                    method="treewidth_dp",
                )
                self._cache[cache_key] = result
                self._promote_to_table(graph, cache_key, result)
                return result

        # 9. k-sum decomposition (k=2..7, vertex separators) via the chord
        # rule (clique_chord_k_sum). Triggers when the graph has a vertex
        # separator that disconnects it cleanly.
        if graph.edge_count() >= 6:
            result = self._try_ksum_decomposition(graph)
            if result is not None:
                _log.record(EventType.KSUM, "engine",
                            f"k-sum: {result.method}", graph=graph)
                self._cache[cache_key] = result
                self._promote_to_table(graph, cache_key, result)
                return result

        # 12. Try creation-expansion-join
        result = self._synthesize_connected(graph, max_depth)
        self._cache[cache_key] = result
        self._promote_to_table(graph, cache_key, result)
        return result

    def _synthesize_disconnected(
        self,
        components: List[Graph],
        max_depth: int
    ) -> SynthesisResult:
        """Synthesize polynomial for disconnected graph.

        For disconnected graphs: T(G₁ ∪ G₂ ∪ ...) = T(G₁) × T(G₂) × ...
        """
        self._log(f"Disconnected graph with {len(components)} components")

        poly = TuttePolynomial.one()
        recipe = [f"Disconnected: {len(components)} components"]
        all_minors = set()

        for i, comp in enumerate(components):
            comp_result = self.synthesize(comp, max_depth)
            poly = poly * comp_result.polynomial
            recipe.append(f"  Component {i+1}: {comp_result.polynomial}")
            all_minors |= comp_result.minors_used

        recipe.append(f"Product: {poly}")

        return SynthesisResult(
            polynomial=poly,
            recipe=recipe,
            verified=True,  # Product formula is exact
            method="disjoint_union",
            minors_used=all_minors,
        )

    def _try_ksum_decomposition(
        self,
        graph: Graph,
    ) -> Optional[SynthesisResult]:
        """Try to decompose graph via k-vertex separator (k=2..7).

        For each k, looks for a set S of k vertices whose removal disconnects
        the graph. Separator vertices may be adjacent — only the *missing*
        clique edges need to be deleted via inclusion-exclusion.

        Searches all k values and picks the separator with the lowest cost,
        preferring full k-sums (0 missing edges → flat-grouped Theorem 6)
        over partial separators that require expensive brute-force.

        Returns SynthesisResult if successful, None otherwise.
        """
        # Depth gate: kmatching-formula leaves (e.g. Cm(2,3) at depth 3-5)
        # recurse here repeatedly, each call running `_find_vertex_separator`
        # for up to 50k combinations. Mirror `_try_decomposition_chord_peel`'s
        # depth-2 cap to bound recursion: at depth > 2, defer to treewidth_dp
        # (step 8) / CEJ (step 12), both of which terminate without
        # combinatorial separator search.
        if self._synth_depth > 2:
            return None

        # Collect all separators across k values
        candidates = []
        # Approximate min-edge gates: roughly C(k, 2) + a small constant for
        # internal cycles. Below this, no useful separator can exist.
        min_edges = {
            2: 3, 3: 6, 4: 12, 5: 15, 6: 20, 7: 28,
            8: 36, 9: 45, 10: 55, 11: 66, 12: 78,
            13: 91, 14: 105, 15: 120, 16: 136, 17: 153,
            18: 171, 19: 190, 20: 210,
        }
        # Bound the high-k search by the actual node connectivity. If the
        # graph's minimum vertex cut is `kappa`, no k < kappa can produce a
        # separator and there's no point searching for k > kappa + 4 (separators
        # of size kappa + 5+ have very different structural properties and
        # almost never beat smaller separators in practice).
        try:
            import networkx as _nx
            kappa = _nx.node_connectivity(graph.to_networkx())
        except Exception:
            kappa = 0
        # Search range: [max(2, kappa), min(k_max, kappa + 5)] when kappa is
        # known. When kappa is unknown (failure), fall back to the configured
        # range.
        k_lo = max(2, kappa) if kappa else 2
        k_hi = min(self.k_max, kappa + 5) if kappa else self.k_max
        for k in range(k_lo, k_hi + 1):
            if graph.edge_count() < min_edges.get(k, 3 * k):
                continue

            separator = self._find_vertex_separator(graph, k)
            if separator is not None:
                sv = sorted(separator)
                total_clique = k * (k - 1) // 2
                missing = sum(1 for i in range(k) for j in range(i+1, k)
                             if (min(sv[i], sv[j]), max(sv[i], sv[j])) not in graph.edges)
                candidates.append((k, separator, missing, total_clique))

        # For graphs with 15-30 nodes, also search for full k-sums using
        # NetworkX minimum vertex cut expanded to nearby k values.
        if 15 <= graph.node_count() <= 30:
            self._search_full_ksum_separators(graph, candidates)

        if not candidates:
            return None

        # Score candidates: full k-sums (0 missing) are vastly cheaper than partial
        # Full k-sum cost ∝ #flats(K_k) (polynomial).
        # Partial cost ∝ 2^missing (exponential in missing edges).
        # Prefer: (1) full k-sums by k (higher k = more flats but still polynomial),
        #         (2) partial with fewest missing edges
        def score(candidate):
            k, sep, missing, total = candidate
            if missing == total:
                # Full k-sum: score by -k (prefer higher k = bigger decomposition)
                return (0, -k)
            elif missing <= 10:
                # Partial: score by 2^missing (exponential cost)
                return (1, 2 ** missing)
            else:
                # Too expensive
                return (2, missing)

        candidates.sort(key=score)

        for k, separator, missing, total in candidates:
            if missing > 10 and missing != total:
                continue  # skip expensive partial separators
            result = self._apply_ksum(graph, separator, k)
            if result is not None:
                return result

        return None

    def _find_vertex_separator(
        self,
        graph: Graph,
        k: int,
    ) -> Optional[Tuple[int, ...]]:
        """Find k vertices whose removal disconnects the graph.

        Unlike the old _find_independent_vertex_separator, this allows
        separator vertices to be adjacent. Prefers separators with fewer
        existing edges (more missing clique edges = more work), but accepts
        any disconnecting k-set.

        Returns tuple of k vertices if found, None otherwise. Only checks a bounded
        number of candidates to avoid combinatorial explosion on large graphs.
        """
        from itertools import combinations

        nodes = sorted(graph.nodes, key=lambda n: graph.degree(n), reverse=True)

        candidates = nodes[:min(len(nodes), 20)]

        if len(candidates) < k:
            return None

        best = None
        best_missing = None  # prefer fewer missing edges (cheaper inclusion-exclusion)
        checked = 0
        # Scale max_checks down for large graphs — cProfile showed 5.4s per
        # call on 36-node Z(1,3) at the old 200k cap (32s wasted searching
        # for separators that don't exist at high k). For n >= 30, drop to
        # 50k (~1.3s/call); under 30n the original 200k cap stays.
        max_checks = 50_000 if graph.node_count() >= 30 else 200_000

        for combo in combinations(candidates, k):
            checked += 1
            if checked > max_checks:
                break
            # Check if removing these k vertices disconnects the graph
            sep_set = set(combo)
            remaining = graph.nodes - sep_set
            if not remaining:
                continue

            start = next(iter(remaining))
            reached = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node in reached:
                    continue
                reached.add(node)
                for nb in graph.neighbors(node):
                    if nb not in reached and nb not in sep_set:
                        stack.append(nb)

            if len(reached) < len(remaining):
                # Count missing clique edges (fewer = cheaper)
                missing = 0
                for i in range(k):
                    for j in range(i + 1, k):
                        edge = (min(combo[i], combo[j]), max(combo[i], combo[j]))
                        if edge not in graph.edges:
                            missing += 1

                # Accept any separator, including full-clique (missing == 0).
                # `_apply_ksum` handles missing == 0 by peeling the EXISTING
                # K_k clique edges via the chord rule (the symmetric case of
                # adding back missing edges) — same C(k, 2) cost. Without
                # this, the chord rule is silently dropped on graphs with
                # full clique separators for k ≥ 2 (e.g., router-style
                # synthetics). See research/audit_chord_rule_findings.md.
                if best is None or missing < best_missing:
                    best = combo
                    best_missing = missing

        return best

    def _search_full_ksum_separators(
        self,
        graph: Graph,
        candidates: list,
    ) -> None:
        """Search for full k-sum separators (all clique edges missing) using
        NetworkX minimum vertex cut as a starting point, then expanding.

        Much faster than exhaustive search for finding FULL k-sums where
        the flat-grouped Theorem 6 path applies.
        """
        from itertools import combinations

        import networkx as nx

        # Build NX graph
        nxg = nx.Graph()
        nxg.add_nodes_from(graph.nodes)
        nxg.add_edges_from(graph.edges)

        # Get all minimum vertex cuts (typically very fast)
        try:
            kappa = nx.node_connectivity(nxg)
        except Exception:
            return

        # Search for independent separators of size kappa, kappa+1, kappa+2
        # "Independent" = no edges between separator vertices = full k-sum
        all_nodes = sorted(graph.nodes)
        for k in range(max(kappa, 5), min(kappa + 3, self.k_max + 1)):
            total_clique = k * (k - 1) // 2
            # Skip if we already have a full k-sum at this k
            if any(c[0] == k and c[2] == c[3] for c in candidates):
                continue

            # Try minimum node cuts first (fast)
            try:
                all_cuts = list(nx.all_node_cuts(nxg, k=kappa))
            except Exception:
                all_cuts = []

            for cut in all_cuts:
                if len(cut) > k:
                    continue
                # Expand cut to size k by adding neighboring nodes
                if len(cut) == k:
                    sep = tuple(sorted(cut))
                    missing = sum(1 for i in range(k) for j in range(i+1, k)
                                 if (min(sep[i], sep[j]), max(sep[i], sep[j])) not in graph.edges)
                    if missing == total_clique:
                        candidates.append((k, sep, missing, total_clique))
                        self._log(f"Found full {k}-sum separator via min-cut: {sep}")
                        break

            # Also try: independent sets of high-degree nodes
            # For Zephyr-like graphs, the separator is often even-numbered nodes
            checked = 0
            for combo in combinations(all_nodes, k):
                checked += 1
                if checked > 50_000:
                    break
                # Quick check: all pairs must be non-adjacent (full k-sum)
                all_independent = True
                for i in range(k):
                    for j in range(i + 1, k):
                        edge = (min(combo[i], combo[j]), max(combo[i], combo[j]))
                        if edge in graph.edges:
                            all_independent = False
                            break
                    if not all_independent:
                        break
                if not all_independent:
                    continue

                # Check if removing these vertices disconnects
                sep_set = set(combo)
                remaining = graph.nodes - sep_set
                if not remaining:
                    continue
                start = next(iter(remaining))
                reached = set()
                stack = [start]
                while stack:
                    node = stack.pop()
                    if node in reached:
                        continue
                    reached.add(node)
                    for nb in graph.neighbors(node):
                        if nb not in reached and nb not in sep_set:
                            stack.append(nb)

                if len(reached) < len(remaining):
                    candidates.append((k, combo, total_clique, total_clique))
                    self._log(f"Found full {k}-sum separator: {combo}")
                    return  # Found one, no need to continue

    def _apply_ksum(
        self,
        graph: Graph,
        separator: Tuple[int, ...],
        k: int,
    ) -> Optional[SynthesisResult]:
        """Apply Theorem 10 to compute Tutte polynomial via vertex separator.

        Reconstructs the parallel connection by adding back missing K_k clique
        edges, then uses inclusion-exclusion to delete them.

        Supports both fully-independent separators (classic k-sum, all C(k,2)
        clique edges missing) and partially-adjacent separators (only missing
        edges are deleted).

        Args:
            graph: The graph with k-vertex separator
            separator: Tuple of k separator vertices
            k: Number of shared vertices

        Returns:
            SynthesisResult if successful, None otherwise
        """
        try:
            sv = sorted(separator)
            all_clique_edges = [(sv[i], sv[j]) for i in range(k) for j in range(i + 1, k)]
            missing_edges = [e for e in all_clique_edges if e not in graph.edges]
            num_missing = len(missing_edges)

            self._log(f"Found {k}-vertex separator: {separator} "
                      f"({num_missing}/{len(all_clique_edges)} clique edges missing)")

            if num_missing > 0:
                # Standard "true k-sum" path: graph has SOME clique edges
                # missing. Build PC by adding missing edges back, then peel
                # them off via chord rule.
                from ..graphs.k_sum import clique_chord_k_sum
                poly = clique_chord_k_sum(
                    graph, separator, k, self,
                    missing_edges=missing_edges,
                )
                method_label = f"{k}sum_chord_rule"
            else:
                # Full clique separator: graph has ALL K_k clique edges. Symmetric
                # case — use the chord rule to PEEL the existing clique edges off
                # `graph`, getting a chord-free leaf with no shared edges between
                # cells, then combine via _combine_chord_iteration. Cost: same
                # 1 + C(k, 2) syntheses as the chord rule for missing > 0.
                #
                # T(graph) = (∏ factors) · T(graph − all clique) + Σ prefix·adds
                #
                # The chord-free leaf (graph − all clique edges) IS the true
                # k-sum target where the cells share only the k separator
                # vertices (no shared edges). Synthesizing it terminates via the
                # standard pipeline (it has no full-clique separator anymore;
                # the separator's clique edges are gone).
                from ..graphs.k_sum import (_combine_chord_iteration,
                                            _iterative_chord_rule)
                smart_order = getattr(self, "chord_smart_order", False)
                g_chord_free, factors, adds = _iterative_chord_rule(
                    graph, all_clique_edges, self, smart_order=smart_order,
                )
                t_chord_free = self.synthesize(g_chord_free).polynomial
                poly = _combine_chord_iteration(t_chord_free, factors, adds)
                method_label = f"{k}sum_full_clique_chord_peel"

            if poly is None:
                return None

            # Verify
            if verify_spanning_trees(graph, poly):
                self._log(f"{k}-vertex separator decomposition verified! ({method_label})")
                return SynthesisResult(
                    polynomial=poly,
                    recipe=[f"{k}-vertex separator at {separator} "
                            f"({num_missing} missing)", f"T = {poly}"],
                    verified=True,
                    method=method_label,
                )
            else:
                self._log(f"{k}-vertex separator decomposition failed Kirchhoff ({method_label})")
                return None

        except (KeyError, IndexError) as e:
            self._log(f"{k}-vertex separator decomposition failed: {e}")
            return None

    def _synthesize_via_cut_vertex(
        self,
        graph: Graph,
        cut: int,
        max_depth: int
    ) -> SynthesisResult:
        """Synthesize using cut vertex factorization.

        For graphs with cut vertices:
        T(G1 · G2 at v) = T(G1) × T(G2)

        where G1 and G2 are the components obtained by splitting at the cut vertex.
        This is much faster than general synthesis.
        """
        self._log(f"Cut vertex factorization at node {cut}")

        components = graph.split_at_cut_vertex(cut)
        recipe = [f"Cut vertex factorization at node {cut}: {len(components)} components"]

        poly = TuttePolynomial.one()
        all_minors = set()
        for i, comp in enumerate(components):
            comp_result = self.synthesize(comp, max_depth)
            poly = poly * comp_result.polynomial
            recipe.append(f"  Component {i+1}: {comp_result.polynomial}")
            all_minors |= comp_result.minors_used

        return SynthesisResult(
            polynomial=poly,
            recipe=recipe,
            verified=True,  # Cut vertex formula is exact
            method="cut_vertex",
            minors_used=all_minors,
        )

    def _maybe_emit_cell_partition(self, graph: Graph) -> None:
        """Visualizer-only: emit per-cell partition snapshot+provenance.

        Gated on ``capture_graphs`` so headless benchmarks don't pay the
        partition-discovery cost. Used by cell-quotient DP paths that
        otherwise return only a polynomial — they don't have the cell
        partition in their return value, so we re-discover it here for
        the visualizer overlay.
        """
        log = get_log()
        if not log.capture_graphs:
            return
        try:
            from ..graphs.covering import (
                try_heterogeneous_partition, try_hierarchical_partition,
            )
            homo = try_hierarchical_partition(graph, self.table)
            if homo is not None:
                cell, partition, inter_info = homo
                self._emit_partition_provenance(
                    graph, [cell] * len(partition), partition, inter_info,
                )
                return
            # Fall back to heterogeneous if homogeneous didn't match.
            het = try_heterogeneous_partition(graph, self.table)
            if het is not None:
                cells, partition, inter_info = het
                self._emit_partition_provenance(
                    graph, cells, partition, inter_info,
                )
        except Exception:
            pass  # best-effort overlay; never block synth

    def _emit_subgraph_provenance(
        self,
        graph: Graph,
        components: List[Set[int]],
        event_type: 'EventType',
        label_fn,
    ) -> None:
        """Record a snapshot+provenance event per component.

        Used by hierarchical-tiling paths that decompose ``graph`` into
        vertex-set ``components`` (blocks of a block-cut tree, components
        of a cut-vertex split, parts of a 2-sum, components of a
        disconnected graph). Each component becomes a snapshot in the
        EventLog with provenance pointing back to the target-graph
        vertices/edges — the visualizer renders these as highlightable
        sub-graph cards.

        Args:
            graph: The parent graph being decomposed.
            components: List of vertex sets, one per sub-graph.
            event_type: EventType to tag the per-component records with
                (typically ``EventType.FACTORIZE``).
            label_fn: Callable ``(idx, vertex_set) -> str`` producing the
                event message (e.g. ``f"Block {idx + 1}: {n}v {e}e"``).
        """
        log = get_log()
        if not log.capture_graphs:
            return
        for idx, vertices in enumerate(components):
            try:
                sub = graph.subgraph(vertices)
            except Exception:
                continue
            node_list = sorted(vertices)
            edge_list = [[u, v] for (u, v) in sub.edges]
            log.record(
                event_type, "engine",
                label_fn(idx, vertices),
                LogLevel.DEBUG, graph=sub,
                provenance={
                    "target_nodes": node_list,
                    "target_edges": edge_list,
                },
            )

    def _emit_decomposition_provenance(
        self, graph: Graph, decomp: 'Decomposition',
    ) -> None:
        """Emit visualizer snapshots for any `Decomposition` (atom or cell).

        Cell decompositions route through `_emit_partition_provenance`
        (which uses MinorEntry names for the per-cell label). Atom
        decompositions don't carry MinorEntries — synthesize a thin
        provenance record from `families` + `components` so the
        visualizer can still highlight where each atom lives.
        """
        if decomp.kind == "cell" and decomp.cell_entries:
            self._emit_partition_provenance(
                graph, decomp.cell_entries, decomp.components,
                decomp.inter_info,
            )
            return
        log = get_log()
        if not log.capture_graphs:
            return
        for atom_idx, (family, vertices) in enumerate(
            zip(decomp.families, decomp.components)
        ):
            try:
                sub = graph.subgraph(vertices)
            except Exception:
                continue
            node_list = sorted(vertices)
            edge_list = [[u, v] for (u, v) in sub.edges]
            log.record(
                EventType.CHORD_RULE, "engine",
                f"Atom {atom_idx} ({family}): "
                f"{len(node_list)}n {len(edge_list)}e",
                LogLevel.DEBUG, graph=sub,
                provenance={
                    "target_nodes": node_list,
                    "target_edges": edge_list,
                },
            )
        if decomp.inter_edges:
            inter_nodes = set()
            inter_edge_list = []
            for u, v in decomp.inter_edges:
                inter_nodes.add(u)
                inter_nodes.add(v)
                a, b = (u, v) if u < v else (v, u)
                inter_edge_list.append([a, b])
            try:
                inter_sub = graph.subgraph(inter_nodes)
            except Exception:
                inter_sub = None
            if inter_sub is not None:
                log.record(
                    EventType.CHORD_RULE, "engine",
                    f"Inter-atom edges: {len(inter_nodes)}n "
                    f"{len(inter_edge_list)}e",
                    LogLevel.DEBUG, graph=inter_sub,
                    provenance={
                        "target_nodes": sorted(inter_nodes),
                        "target_edges": inter_edge_list,
                    },
                )

    def _emit_partition_provenance(
        self, graph: Graph, cells, partition, inter_info,
    ) -> None:
        """Record a snapshot+provenance event for each cell of a
        hierarchical decomposition + the inter-cell edge set, so the
        visualizer can highlight where each sub-graph lives in the
        input graph.
        """
        log = get_log()
        if not log.capture_graphs:
            return
        # Per-cell records.
        for cell_idx, (cell_entry, cell_nodes) in enumerate(zip(cells, partition)):
            try:
                cell_subgraph = graph.subgraph(cell_nodes)
            except Exception:
                continue
            cell_node_list = sorted(cell_nodes)
            cell_edge_list = [
                [u, v] for (u, v) in cell_subgraph.edges
            ]
            log.record(
                EventType.HIERARCHICAL, "engine",
                f"Cell {cell_idx} ({cell_entry.name}): "
                f"{len(cell_node_list)}n {len(cell_edge_list)}e",
                LogLevel.DEBUG, graph=cell_subgraph,
                provenance={
                    "target_nodes": cell_node_list,
                    "target_edges": cell_edge_list,
                },
            )
        # Inter-cell edge "graph" snapshot — just the inter-cell
        # endpoints + their edges; provenance points back to the
        # relevant target nodes/edges.
        if inter_info is not None and inter_info.edges:
            inter_nodes = set()
            inter_edge_list = []
            for u, v in inter_info.edges:
                inter_nodes.add(u)
                inter_nodes.add(v)
                a, b = (u, v) if u < v else (v, u)
                inter_edge_list.append([a, b])
            try:
                inter_subgraph = graph.subgraph(inter_nodes)
            except Exception:
                inter_subgraph = None
            if inter_subgraph is not None:
                log.record(
                    EventType.HIERARCHICAL, "engine",
                    f"Inter-cell edges: {len(inter_nodes)}n "
                    f"{len(inter_edge_list)}e",
                    LogLevel.DEBUG, graph=inter_subgraph,
                    provenance={
                        "target_nodes": sorted(inter_nodes),
                        "target_edges": inter_edge_list,
                    },
                )

    def _try_formula_shortcircuit(
        self,
        graph: Graph,
        max_depth: int
    ) -> Optional[SynthesisResult]:
        """Closed-form shortcut for hierarchical graphs BEFORE treewidth_dp.

        Attempts the unified formula and the k-matching
        cell-cycle formula *without* falling through to treewidth_dp or the
        chord rule. Returns None if neither formula applies — the caller
        should continue to the standard pipeline.

        Gated to graphs where a D-Wave-style cell candidate (Cm1 = K_{4,4},
        Z1_1, or similar structured tile) exists. This avoids the VF2
        slow path on dense graphs (like router/clique hybrids) where
        K_n is the only candidate and tiling search is expensive without
        any payoff.

        Used to prefer the formula paths for visualization and for the
        Cm2-like case where the k-matching formula beats direct
        treewidth_dp (4x speedup on Cm2).
        """
        from ..graphs.covering import (apply_kmatching_formula,
                                       detect_kmatching_topology,
                                       extract_cell_topology,
                                       find_cell_candidates,
                                       try_heterogeneous_partition,
                                       try_hierarchical_partition)
        from ..graphs.k_sum import _classify_bridges_chords

        # Fast gate: skip the shortcut if no D-Wave-style cell candidate
        # exists. This leaves dense clique-hybrid graphs (router+clique)
        # unaffected since K_n-only candidates are not in the gate set.
        candidates = find_cell_candidates(graph, self.table)
        dwave_cell_names = {"Cm1", "Cm2", "Z1_1", "Z1_2"}
        if not any(c.name in dwave_cell_names for c in candidates):
            return None

        homo = try_hierarchical_partition(graph, self.table)
        # Heterogeneous partition is VF2-heavy (~60-180s/call on 72n+ graphs
        # because it walks the full ~1000-entry rainbow table). Skip in
        # recursive sub-syntheses (depth > 1): they produce derived graphs
        # (e.g., k-sum's PC = target + extra clique edges) that disrupt
        # cell structure; heterogeneous almost always fails after burning
        # significant VF2 time. Top-level calls still run both partitioners
        # so we don't miss legitimate heterogeneous covers on user inputs.
        if self._synth_depth <= 1:
            het = try_heterogeneous_partition(graph, self.table)
        else:
            het = None

        candidates = []
        if homo is not None:
            cell, partition, inter_info = homo
            candidates.append(([cell] * len(partition), partition, inter_info))
        if het is not None:
            cells, partition, inter_info = het
            candidates.append((cells, partition, inter_info))

        for cells, partition, inter_info in candidates:
            if not inter_info.edges:
                continue
            k_cells = len(partition)
            base_poly = TuttePolynomial.one()
            for c in cells:
                base_poly = base_poly * c.polynomial
            all_minors = {c.canonical_key for c in cells}
            recipe = [f"Formula shortcut: {k_cells} cells"]

            # Emit a snapshot+provenance event for each cell so the
            # visualizer can highlight where each cell lives in the
            # target graph. Also one for the inter-cell edge set.
            self._emit_partition_provenance(graph, cells, partition, inter_info)

            # Try unified formula
            H = extract_cell_topology(partition, list(inter_info.edges))
            if H is not None:
                T_H = self._synthesize_multigraph(H)
                unified_poly = base_poly * T_H
                if verify_spanning_trees(graph, unified_poly):
                    recipe.append(
                        f"Unified formula: H has {len(H.nodes)} nodes, "
                        f"{sum(H.edge_counts.values())} edges"
                    )
                    return SynthesisResult(
                        polynomial=unified_poly,
                        recipe=recipe,
                        verified=True,
                        method="unified_formula",
                        tiles_used=k_cells,
                        fringe_edges=0,
                        minors_used=all_minors,
                    )

            # Try k-matching formula
            junctions = detect_kmatching_topology(
                graph, partition, list(inter_info.edges)
            )
            if junctions is not None and any(j.k > 1 for j in junctions):
                # Fast path: unified chord-junction theorem via session
                # merger cache. Equivalent polynomial to
                # apply_kmatching_formula but O(1) per merger lookup
                # when the warmup populated the relevant V_T entries.
                km_poly = self._try_unified_chord_junction(
                    graph, junctions, partition,
                )
                # apply_kmatching_formula recurses one junction at a time,
                # producing k_eff^|junctions| leaf evaluations. For chains
                # with > 3 junctions (Cm(1, n) for n >= 5, multi-cell
                # Zephyr) this explodes: Cm(1, 6) has 5 K_{4,4}+M_4
                # junctions = 5^5 = 3125 leaves where each leaf re-enters
                # the engine and hits expensive ksum/treewidth_dp paths.
                # Defer to step 7.82 (bipartite_junction_dp) which solves
                # these in O(n) via the per-cell tree DP.
                if km_poly is None and len(junctions) <= 3:
                    try:
                        km_poly = apply_kmatching_formula(
                            graph, junctions, self._synthesize_multigraph
                        )
                    except Exception:
                        km_poly = None
                if km_poly is not None and verify_spanning_trees(graph, km_poly):
                    k_values = sorted({j.k for j in junctions})
                    recipe.append(
                        f"k-matching formula: {len(junctions)} junctions, "
                        f"k={k_values}"
                    )
                    return SynthesisResult(
                        polynomial=km_poly,
                        recipe=recipe,
                        verified=True,
                        method="kmatching_formula",
                        tiles_used=k_cells,
                        fringe_edges=0,
                        minors_used=all_minors,
                    )

            # Sokal-Z generalized chord-junction (fallback for 2-cell
            # partitions with non-matching / multi-edge / dense E_J that
            # the unified theorem can't handle). See
            # `tutte/roots/sokal_z_chord_junction.py`. Gated on small
            # H_J components — true tree-DP for large components is
            # task #438 follow-up.
            if k_cells == 2:
                sokal_z_poly = self._try_sokal_z_chord_junction(
                    graph, cells, partition, inter_info,
                )
                if (sokal_z_poly is not None
                        and verify_spanning_trees(graph, sokal_z_poly)):
                    recipe.append(
                        f"Sokal-Z chord-junction: {len(inter_info.edges)} "
                        f"chord edges"
                    )
                    return SynthesisResult(
                        polynomial=sokal_z_poly,
                        recipe=recipe,
                        verified=True,
                        method="sokal_z_chord_junction",
                        tiles_used=k_cells,
                        fringe_edges=0,
                        minors_used=all_minors,
                    )

        return None

    def _try_sokal_z_chord_junction(
        self,
        graph: 'Graph',
        cells: List,
        partition: List[Set[int]],
        inter_info,
    ) -> Optional[TuttePolynomial]:
        """Dispatch the Sokal-Z generalized chord-junction theorem.

        Handles 2-cell partitions with **arbitrary** chord junctions
        (matching, multi-edge, non-matching) that the unified theorem
        rejects. Gated to small H_J components — large components
        (Z(1, 2)-class, |E_J| > 16 with dense connectivity) fall
        through and rely on tree-DP follow-up (task #438).
        """
        if len(partition) != 2:
            return None
        cell_left_verts = sorted(partition[0])
        cell_right_verts = sorted(partition[1])
        relabel_left = {v: idx for idx, v in enumerate(cell_left_verts)}
        relabel_right = {v: idx for idx, v in enumerate(cell_right_verts)}
        left_set = set(cell_left_verts)
        right_set = set(cell_right_verts)

        # Build cell graphs
        cell_left = graph.induced_relabeled(relabel_left)
        cell_right = graph.induced_relabeled(relabel_right)

        # Build chord-edge list in (left_relabel, right_relabel) form
        chord_pairs: List[Tuple[int, int]] = []
        for (a, b) in inter_info.edges:
            if a in left_set and b in right_set:
                chord_pairs.append((relabel_left[a], relabel_right[b]))
            elif b in left_set and a in right_set:
                chord_pairs.append((relabel_left[b], relabel_right[a]))
            else:
                return None
        if not chord_pairs:
            return None
        try:
            from ..roots.sokal_z_chord_junction import (
                compute_sokal_z_chord_junction,
            )
            return compute_sokal_z_chord_junction(
                cell_left, cell_right, chord_pairs,
                self._synthesize_multigraph,
            )
        except Exception:
            return None

    # Fast-path threshold for the unified chord-junction theorem. When
    # the chord junction has |V_k| ≤ this many anchors, the I-E sum
    # over 2^|V_k| subsets is small enough to beat
    # `apply_kmatching_formula` even on a cold cache (and is essentially
    # free when the merger persistent cache is warm). At |V_k| = 12 the
    # sum has 4096 terms; with the Z(1,1) merger cache populated
    # (per `project_vf2_thread_budget_and_z11_warmup.md`, 4095 entries
    # covering all V_T subsets of Z(1,1)) each term is an O(1) lookup,
    # so Z(1, 2) decomposed as 2 Z(1,1) cells solves via this path in
    # << 1s. Without the cache, |V_k| ≥ 8 is impractically slow due
    # to cold merger synthesis cost. Cap at 12 to match Z(1,1)'s
    # 12-vertex boundary.
    _UNIFIED_CHORD_JUNCTION_MAX_VK = 12

    def _try_unified_chord_junction(
        self,
        graph: 'Graph',
        junctions: List['KMatchingJunction'],
        partition: List[Set[int]],
    ) -> Optional[TuttePolynomial]:
        """Fast-path for chord-junction cell-pairs (symmetric + asymmetric).

        Applies the unified bivariate I-E theorem (see
        ``tutte/roots/chord_junction_closed_form.py``) when the partition
        has exactly two cells joined by a single chord junction.

        Two dispatch tiers:

        1. **Symmetric** — cells isomorphic AND chord pairs align in
           canonical form (same anchor position on both sides). Lookup
           via ``(base_canonical_key, V_T)``; the warmup script populates
           this index per cell template.
        2. **Asymmetric** — anything else (mixed-bipartition anchors on
           ``Aut``-rich cells, non-isomorphic cells, …). Lookup via the
           merger multigraph's canonical key, so any asymmetric chord
           pattern whose merger is isomorphic to a cached symmetric
           merger still hits the cache. Misses fall back to
           ``synth_multigraph`` and are inserted into the session cache.

        Returns ``None`` when:
          * more than one junction (multi-cell chain/cycle — outside this
            theorem's scope, falls through to ``apply_kmatching_formula``)
          * ``|V_k|`` exceeds the threshold (chord-rule path is cheaper)
          * a chord edge has both endpoints in the same cell (junction
            mis-detection — defensive bail)
        """
        if len(junctions) != 1:
            return None
        junc = junctions[0]
        if junc.k < 1 or junc.k > self._UNIFIED_CHORD_JUNCTION_MAX_VK:
            return None
        cell_i_verts = sorted(partition[junc.cell_i])
        cell_j_verts = sorted(partition[junc.cell_j])
        relabel_i = {v: idx for idx, v in enumerate(cell_i_verts)}
        relabel_j = {v: idx for idx, v in enumerate(cell_j_verts)}
        relabel_i_set = set(cell_i_verts)
        relabel_j_set = set(cell_j_verts)

        cell_i = graph.induced_relabeled(relabel_i)
        cell_j = graph.induced_relabeled(relabel_j)

        try:
            V_k_i = sorted(relabel_i[v] for v in junc.anchors_i)
            V_k_j = sorted(relabel_j[v] for v in junc.anchors_j)
        except KeyError:
            return None

        # Tier 1: symmetric. Cells isomorphic, V_k anchors aligned, chord
        # pairs map matching positions.
        can_use_symmetric = False
        if len(cell_i_verts) == len(cell_j_verts) and V_k_i == V_k_j:
            try:
                if cell_i.canonical_key() == cell_j.canonical_key():
                    can_use_symmetric = True
                    for (a, b) in junc.edges:
                        if a in relabel_i_set and b in relabel_j_set:
                            if relabel_i[a] != relabel_j[b]:
                                can_use_symmetric = False
                                break
                        elif b in relabel_i_set and a in relabel_j_set:
                            if relabel_i[b] != relabel_j[a]:
                                can_use_symmetric = False
                                break
                        else:
                            # Chord edge with both endpoints in one cell —
                            # not a chord junction; bail both tiers.
                            return None
            except Exception:
                can_use_symmetric = False

        if can_use_symmetric:
            try:
                from ..roots.chord_junction_closed_form import unified_chord_junction
                return unified_chord_junction(
                    cell_i, V_k_i, self._synthesize_multigraph,
                    merger_table=self._merger_session_cache,
                    update_merger_table=True,
                    family_tag="session",
                )
            except Exception:
                return None

        # Tier 2: asymmetric. Build explicit chord pairs (left, right) in
        # the relabeled cell vertex space. Cells need not be isomorphic
        # and V_k anchors need not align.
        chord_pairs: List[Tuple[int, int]] = []
        for (a, b) in junc.edges:
            if a in relabel_i_set and b in relabel_j_set:
                chord_pairs.append((relabel_i[a], relabel_j[b]))
            elif b in relabel_i_set and a in relabel_j_set:
                chord_pairs.append((relabel_i[b], relabel_j[a]))
            else:
                return None
        if not chord_pairs:
            return None

        try:
            from ..roots.chord_junction_closed_form import (
                unified_chord_junction_asymmetric,
            )
            return unified_chord_junction_asymmetric(
                cell_i, cell_j, chord_pairs, self._synthesize_multigraph,
                merger_table=self._merger_session_cache,
                update_merger_table=True,
                family_tag="session",
            )
        except Exception:
            return None

    # Empirical per-edge cost constants — keep in sync with the docstring
    # of `_try_decomposition_chord_peel`. Calibrated from engine.synthesize
    # timings on Z(1,2) (May 22-23 2026). The cell-per-edge constant is set
    # to the legacy-K_n value because a homogeneous K_n CELL partition and
    # the corresponding ATOM decomposition produce identical chord_edges
    # and identical residues at this granularity — they are the same peel.
    _INTER_LEGACY_PER_EDGE = 1.3   # legacy K_n atoms: ~38s / 4 edges
    _INTER_HET_PER_EDGE    = 9.8   # heterogeneous atoms (cache miss penalty)
    _INTER_WHOLE_PER_EDGE  = 1.0   # whole-junction: each edge contributes
                                   # less because residue is structurally
                                   # simpler (cycle→chain). Tuned post-
                                   # validation when chain_recurrence is
                                   # wired to residue. Conservative now.
    _WHOLE_JUNCTION_MAX    = 16    # full inter-atom junction; K_4×K_4 has
                                   # at most 16 edges (full K_{4,4}), so
                                   # this caps to one cell-pair.
    _INTRA_PER_EDGE        = 1.1   # clique_atom intra-K_n: ~95s / 12 edges
    _CELL_PER_EDGE         = 1.3   # cell-partition chord edges
    _PREDICTED_COST_REJECT = 0.85  # reject candidates predicted slower
                                   # than treewidth_dp baseline

    def _build_atom_decomposition(
        self,
        graph: Graph,
        atoms: List,                       # List[Atom]; avoid import cycle
        label: str,
        per_edge_const: float,
        tw_ratio: float,
        max_junction_size: int,
        nxg=None,
        strategy: str = "smallest_component",
    ) -> Optional[Decomposition]:
        """Inter-atom Decomposition from a list of atoms (or None if junction is bad).

        `strategy`:
          - `smallest_component` (default): peel the smallest CONNECTED
            COMPONENT of inter-atom edges. Original behavior.
          - `whole_junction`: peel ALL edges between the smallest cell
            pair. This converts cycle-of-atoms → chain-of-atoms
            (validated on Cm(2,2) and Z(1,3) in probes May 23 2026),
            enabling chain_recurrence on the residue in O(r²·n).
            See `project_cycle_to_chain_chord_peel.md`.
        """
        from ..graphs.atom_detection import (find_smallest_full_junction,
                                             find_smallest_junction)
        if len(atoms) < 2:
            return None
        if strategy == "whole_junction":
            j = find_smallest_full_junction(graph, atoms, nxg=nxg)
        else:
            j = find_smallest_junction(graph, atoms, nxg=nxg)
        if j is None or len(j) > max_junction_size:
            return None
        chord_edges = list(j)
        predicted = len(chord_edges) * per_edge_const * tw_ratio
        # Aggregate inter-component edges via the existing helper would
        # require nx; the smallest junction already gives us a peelable
        # connected component, so use it as both inter_edges and chord_edges.
        components = [set(a.vertices) for a in atoms]
        return Decomposition(
            kind="atom",
            label=label,
            components=components,
            families=[a.family for a in atoms],
            cell_entries=None,
            inter_edges=chord_edges,
            chord_edges=chord_edges,
            predicted_chord_cost=predicted,
            peel_type="inter",
            inter_info=None,
        )

    def _build_intra_atom_decomposition(
        self,
        graph: Graph,
        atoms: List,
        label: str,
        per_edge_const: float,
        tw_ratio: float,
    ) -> Optional[Decomposition]:
        """Intra-atom Decomposition (peel all internal K_n edges). K_n only."""
        if len(atoms) < 2:
            return None
        # K_n family only (clique structure has clean internals)
        for a in atoms:
            if not a.family.startswith("K_") or a.family.startswith("K_{"):
                return None
        graph_edges = {(min(u, v), max(u, v)) for u, v in graph.edges}
        internal_edges: List[Tuple[int, int]] = []
        for atom in atoms:
            vs = sorted(atom.vertices)
            for i in range(len(vs)):
                for k in range(i + 1, len(vs)):
                    e = (vs[i], vs[k])
                    if e in graph_edges:
                        internal_edges.append(e)
        if not internal_edges:
            return None
        predicted = len(internal_edges) * per_edge_const * tw_ratio
        components = [set(a.vertices) for a in atoms]
        return Decomposition(
            kind="atom",
            label=label,
            components=components,
            families=[a.family for a in atoms],
            cell_entries=None,
            inter_edges=internal_edges,
            chord_edges=internal_edges,
            predicted_chord_cost=predicted,
            peel_type="intra",
            inter_info=None,
        )

    def _build_cell_decomposition(
        self,
        cells: List[MinorEntry],
        partition: List[Set[int]],
        inter_info: InterCellInfo,
        label: str,
        per_edge_const: float,
        tw_ratio: float,
    ) -> Optional[Decomposition]:
        """Cell Decomposition with bridge/chord classification."""
        from ..graphs.k_sum import _classify_bridges_chords
        if len(partition) < 2:
            return None
        # Apply the cell "is it worth it" gate:
        # cells must have at least one cycle (edge_count >= node_count) so
        # the chord-rule has non-trivial atomic Tutte polynomials to consume.
        # Heterogeneous partitions are vetted by the partitioner itself; for
        # homogeneous, all cells share the same minor entry so one check.
        if all(c.edge_count >= c.node_count for c in cells) is False:
            return None
        _, chords = _classify_bridges_chords(partition, list(inter_info.edges))
        predicted = len(chords) * per_edge_const * tw_ratio
        return Decomposition(
            kind="cell",
            label=label,
            components=partition,
            families=[c.name for c in cells],
            cell_entries=list(cells),
            inter_edges=list(inter_info.edges),
            chord_edges=list(chords),
            predicted_chord_cost=predicted,
            peel_type="inter",
            inter_info=inter_info,
        )

    def _discover_decompositions(
        self,
        graph: Graph,
        max_junction_size: int,
        tw_ratio: float,
        *,
        skip_cells: bool = False,
        force_cells: bool = False,
        skip_atoms: bool = False,
    ) -> List[Decomposition]:
        """Discover atom + cell decompositions; return priority-ordered list.

        `skip_cells=True` (used at recursive depth) bypasses the
        expensive cell partition VF2 — atom detection is ms-fast and
        gives the dispatcher enough to fall through to chord-rule
        without paying the cell-quotient cost on intermediates.
        """
        from ..graphs.atom_detection import (_to_nx, find_atoms_heterogeneous,
                                             find_disjoint_atoms)

        nxg = _to_nx(graph)
        decomps: List[Decomposition] = []

        if skip_atoms:
            atoms_legacy = []
            atoms_het = []
        else:
            atoms_legacy = find_disjoint_atoms(graph)
            atoms_het = find_atoms_heterogeneous(
                graph, max_junction_size=max_junction_size,
            )
        # `max_junction_size` (default 6) bounds the smallest-component
        # peel. The whole-junction strategy is bounded separately by
        # `_WHOLE_JUNCTION_MAX` because a full inter-atom junction can
        # be much larger (e.g., 8 edges between K_4 atoms in Z(1,3)).
        # The benefit: peeling a whole junction severs cycle-of-atoms
        # topology → chain-of-atoms residue, which chain_recurrence
        # can evaluate in O(r²·n). Validation Cm(2,2)+Z(1,3) confirms
        # cycle→chain conversion. See
        # project_cycle_to_chain_chord_peel.md.
        for d in (
            self._build_atom_decomposition(
                graph, atoms_legacy, "atom_inter_legacy",
                self._INTER_LEGACY_PER_EDGE, tw_ratio,
                max_junction_size, nxg=nxg),
            self._build_atom_decomposition(
                graph, atoms_het, "atom_inter_het",
                self._INTER_HET_PER_EDGE, tw_ratio,
                max_junction_size, nxg=nxg),
            self._build_atom_decomposition(
                graph, atoms_legacy, "atom_inter_whole_legacy",
                self._INTER_WHOLE_PER_EDGE, tw_ratio,
                self._WHOLE_JUNCTION_MAX, nxg=nxg,
                strategy="whole_junction"),
            self._build_intra_atom_decomposition(
                graph, atoms_legacy, "atom_intra_legacy",
                self._INTRA_PER_EDGE, tw_ratio),
        ):
            if d is not None:
                decomps.append(d)

        # A2. Cells (gated; expensive but cached). Use the SINGLE-result
        # `try_hierarchical_partition` — it shares the cache populated by
        # earlier engine dispatch steps (cell_quotient_grid_streamed,
        # _try_formula_shortcircuit, etc.), so this call is typically a
        # cache hit (~ms). Falling back to `iter_hierarchical_partitions`
        # added ~16s of cold-cache VF2 work on Z(1,2). The iter primitive
        # remains exposed for callers that genuinely need multiple
        # tilings — but Phase B's product_formula only ever benefits from
        # the FIRST partition (its inter-component structure determines
        # whether the formula applies regardless of cell shape).
        n, m = graph.node_count(), graph.edge_count()
        if (not skip_cells
                and (force_cells
                     or (10 <= n <= 100 and m >= 20 and self._synth_depth <= 1))):
            homo = try_hierarchical_partition(graph, self.table)
            if homo is not None:
                cell, partition, inter_info = homo
                d = self._build_cell_decomposition(
                    [cell] * len(partition), partition, inter_info,
                    f"cell_homo_{cell.name}",
                    self._CELL_PER_EDGE, tw_ratio,
                )
                if d is not None:
                    decomps.append(d)
            # Heterogeneous partition VF2 (`_find_induced_match`) is
            # super-linear in n; on Pm(2) (n=40, m=164) it exhausts
            # alternatives without finding a match. Gate by graph size
            # — the homogeneous partition above is the typical winner
            # for D-Wave targets anyway, and heterogeneous is most
            # valuable for small mixed-family graphs (Z(1,3) =
            # Z(1,2)+Z(1,1), n=36 but m=162 — borderline). Threshold
            # n <= 32 matches the structural break between Cm(2,2)
            # (32n, tractable) and Pm(2)/Z(2,1) (40n, blows up).
            het = (try_heterogeneous_partition(graph, self.table)
                   if (force_cells or n <= 32) else None)
            if het is not None:
                cells, partition, inter_info = het
                names = "+".join(c.name for c in cells)
                d = self._build_cell_decomposition(
                    list(cells), partition, inter_info,
                    f"cell_het_{names}",
                    self._CELL_PER_EDGE, tw_ratio,
                )
                if d is not None:
                    decomps.append(d)

        # A3. Deduplicate. When an atom decomposition has the same vertex
        # partition as a cell decomposition (e.g. K_4 atoms == K_4 cells),
        # keep the cell version — its MinorEntry polynomials unlock the
        # closed-form formulas in Phase B.
        seen_sig: Set[FrozenSet[FrozenSet[int]]] = set()
        kept: List[Decomposition] = []
        # Iterate cell decomps first so they win on ties.
        ordered = sorted(decomps, key=lambda d: 0 if d.kind == "cell" else 1)
        for d in ordered:
            sig = frozenset(frozenset(c) for c in d.components)
            if d.peel_type == "inter" and sig in seen_sig:
                continue
            kept.append(d)
            if d.peel_type == "inter":
                seen_sig.add(sig)
        return kept

    def _try_cell_closed_forms(
        self,
        graph: Graph,
        decomp: Decomposition,
        max_depth: int,
    ) -> Optional[SynthesisResult]:
        """Cell-only closed-form trial: unified_formula → kmatching → product."""
        if decomp.kind != "cell" or not decomp.inter_edges:
            return None
        cells = decomp.cell_entries
        partition = decomp.components
        inter_info = decomp.inter_info
        k_cells = len(partition)
        all_minors = {c.canonical_key for c in cells}

        # Provenance for the visualizer.
        self._emit_partition_provenance(graph, cells, partition, inter_info)

        base_poly = TuttePolynomial.one()
        for c in cells:
            base_poly = base_poly * c.polynomial
        cell_names = [c.name for c in cells]
        homogeneous = len(set(cell_names)) == 1
        recipe_prefix = (
            f"Decomposition (cell, homo): {k_cells} × {cell_names[0]}"
            if homogeneous else
            f"Decomposition (cell, het): {' + '.join(cell_names)}"
        )

        # 1. unified_formula
        H = extract_cell_topology(partition, list(inter_info.edges))
        if H is not None:
            T_H = self._synthesize_multigraph(H)
            unified_poly = base_poly * T_H
            if verify_spanning_trees(graph, unified_poly):
                return SynthesisResult(
                    polynomial=unified_poly,
                    recipe=[
                        recipe_prefix,
                        f"Unified formula: H has {len(H.nodes)} nodes, "
                        f"{sum(H.edge_counts.values())} edges",
                    ],
                    verified=True,
                    method="unified_formula",
                    tiles_used=k_cells,
                    fringe_edges=0,
                    minors_used=all_minors,
                )

        # 2. kmatching_formula
        junctions = detect_kmatching_topology(graph, partition,
                                              list(inter_info.edges))
        if junctions is not None and any(j.k > 1 for j in junctions):
            # Fast path (see _try_unified_chord_junction docstring).
            km_poly = self._try_unified_chord_junction(
                graph, junctions, partition,
            )
            if km_poly is None:
                try:
                    km_poly = apply_kmatching_formula(
                        graph, junctions, self._synthesize_multigraph,
                    )
                except Exception:
                    km_poly = None
            if km_poly is not None and verify_spanning_trees(graph, km_poly):
                k_values = sorted({j.k for j in junctions})
                return SynthesisResult(
                    polynomial=km_poly,
                    recipe=[
                        recipe_prefix,
                        f"k-matching formula: {len(junctions)} junctions, "
                        f"k={k_values}",
                    ],
                    verified=True,
                    method="kmatching_formula",
                    tiles_used=k_cells,
                    fringe_edges=0,
                    minors_used=all_minors,
                )

        # No cell-only closed form applies here; fall through to the cascade.
        # (A product_formula T(G)=∏T(cells)·∏T(inter) was tried historically but
        # only helped partitions whose inter-cell graph splits into independent
        # components — a class the chord-rule path already covers — so it was removed.)
        return None

    def _chord_peel_decomposition(
        self,
        graph: Graph,
        decomp: Decomposition,
        max_depth: int,
        recurse_residue: bool,
        min_recursion_size: int,
    ) -> Optional[SynthesisResult]:
        """Apply chord-rule to the chosen decomposition.

        Recursive residue peel: after `_iterative_chord_rule` returns
        `g_chord_free`, the residue is synthesized via `self.synthesize`,
        which re-dispatches through the engine (and back into THIS method
        when depth-gate is loose). This lets the contracted residue's
        SECOND decomposition surface (atoms created by contraction
        triangles, etc.) — the user's "2-chord" framing.
        """
        from ..graphs.k_sum import (_combine_chord_iteration,
                                    _iterative_chord_rule)
        from ..graphs.atom_detection import _to_nx

        # Visualizer provenance for the chosen decomposition. Cell
        # decompositions emit per-cell snapshots via
        # `_emit_partition_provenance` in `_try_cell_closed_forms`; atom
        # decompositions (which never go through Phase B) need to emit
        # their own per-atom snapshots so the visualizer can highlight
        # which vertices each atom occupies before chord-peel begins.
        self._emit_decomposition_provenance(graph, decomp)

        try:
            from ..roots.signed_quotient import find_best_sigma
            sigma = find_best_sigma(_to_nx(graph), require_free=True)
        except Exception:
            sigma = None

        try:
            g_chord_free, factors, adds = _iterative_chord_rule(
                graph, decomp.chord_edges, self,
                smart_order=getattr(self, "chord_smart_order", False),
                sigma=sigma,
            )
        except Exception:
            return None

        # Recursive residue: synthesize via the standard engine pipeline,
        # which re-enters this dispatcher if the residue admits another
        # decomposition (the user's "2-chord" framing). The engine's
        # per-canonical-key cache and the outer max_depth bound terminate
        # recursion. max_depth is preserved across the residue call —
        # reducing it triggers earlier creation-expansion-join fallback
        # which is much slower on dense intermediates.
        # `recurse_residue` and `min_recursion_size` are kept as opt-out
        # knobs for tests/profiling; default behavior matches legacy.
        t_chord_free = self.synthesize(g_chord_free, max_depth).polynomial

        poly = _combine_chord_iteration(t_chord_free, factors, adds)

        family_str = ", ".join(decomp.families[:3])
        if len(decomp.families) > 3:
            family_str += f", … (+{len(decomp.families) - 3})"
        method = (
            f"decomposition_chord_peel_{decomp.kind}_{decomp.peel_type}"
        )
        return SynthesisResult(
            polynomial=poly,
            recipe=[
                f"Decomposition ({decomp.kind}, {decomp.peel_type}): "
                f"{decomp.label} — {len(decomp.components)} components "
                f"[{family_str}], {len(decomp.chord_edges)} chord edges, "
                f"σ={sigma is not None}"
            ],
            verified=True,
            method=method,
            tiles_used=len(decomp.components),
            fringe_edges=0,
        )

    def _try_decomposition_chord_peel(
        self,
        graph: Graph,
        max_depth: int,
        *,
        max_junction_size: int = 6,
        recurse_residue: bool = True,
        min_recursion_size: int = 12,
        force: bool = False,
        skip_atoms: bool = False,
    ) -> Optional[SynthesisResult]:
        """Unified decomposition + chord-peel dispatcher.

        See `tutte/docs/08_5_decomposition_chord_peel.md`.

        `force=True` bypasses the tw<=8 early-return gate so test fixtures
        that exercise the cell-closed-form formula paths on small graphs
        (n<=10, tw<=3) can still trigger discovery + Phase B.

        Phases:
          A. DISCOVERY — collect candidate `Decomposition` records from
             both atom detectors (`find_disjoint_atoms`,
             `find_atoms_heterogeneous`) and cell partitioners
             (`iter_hierarchical_partitions`, `try_heterogeneous_partition`).
             Atoms run unconditionally; cells run only at top-level
             (depth<=1) and within size gate (10 <= n <= 100, edges >= 20).
          B. CLOSED-FORM TRIAL — for cell decompositions only, try
             unified_formula → kmatching_formula → product_formula. First
             hit returns immediately with the formula's method label
             preserved (`unified_formula` / `kmatching_formula` /
             `product_formula`).
          C. COST-GATED CHORD-RULE — rank surviving decompositions by
             predicted cost; apply `_iterative_chord_rule` on the
             cheapest accepted. Reject all if min predicted cost
             >= `_PREDICTED_COST_REJECT` (0.85).
          D. RECURSIVE RESIDUE PEEL (if `recurse_residue` and residue
             has >= `min_recursion_size` nodes) — call
             `self.synthesize(g_chord_free, max_depth-1)` so the residue
             re-enters the dispatcher. Contracted residues often expose
             a SECOND decomposition the original graph hid.

        Method labels emitted:
          - `unified_formula`, `kmatching_formula`, `product_formula`
            (cell-level closed forms)
          - `decomposition_chord_peel_cell_inter` (cell chord-rule)
          - `decomposition_chord_peel_atom_inter` (inter-atom chord-rule)
          - `decomposition_chord_peel_atom_intra` (intra-atom chord-rule
            on K_n internal edges)
        """
        from ..graphs.treewidth import compute_best_tree_decomposition

        # Treewidth probe runs only at TOP level (depth == 1) to gate
        # against low-tw graphs where step 8 treewidth_dp(max_width=10)
        # is faster than chord-rule. At recursive depth, the contracted
        # intermediate already came from a chord-peel that chose this
        # path — skip the probe (saves ~0.3s/call, matches legacy
        # `_try_clique_atom_chord_peel` which never probed treewidth).
        recursive_call = self._synth_depth > 1
        if recursive_call:
            tw_ratio = 0.05  # neutral; chord-rule already picked
        else:
            try:
                full_mg = MultiGraph.from_graph(graph)
                td = compute_best_tree_decomposition(full_mg, max_width=20)
                tw = td.width if td is not None else 12
            except Exception:
                tw = 12
            tw_ratio = 0.05 + 0.02 * max(0, tw - 10)

            # Defer to step 8 treewidth_dp (max_width=10) when treewidth is
            # low enough that tw_dp is definitively faster than chord-rule.
            # Cm(1,3) (tw=4): tw_dp 0.07s vs chord-rule 11s. Cm(1,2) (tw=3):
            # similar. Threshold tw=8 leaves chord-rule for borderline cases
            # like Z(1,2) (tw=10) where chord-rule's 36s beats tw_dp's 111s.
            if tw <= 8 and not force:
                return None

            # Defer to step 8/9/12 for high-treewidth graphs. The cost
            # predictor is tuned for tw ∈ [9, 11] where chord-rule is
            # the engine's last competitive option; at tw >= 12 (Pm(2)
            # tw=14, Z(2,1) tw=12) atom discovery / cell partition VF2
            # are themselves the slow part — even before any chord-rule
            # cost — and the predicted cost estimator is wrong (one
            # "cheap" chord edge can produce a residue requiring 60s of
            # sub-synth). This is the structural replacement for the
            # old `node_count <= 30` gate. Graphs that need a denser
            # decomposition should be promoted to step 9 (k-sum) or
            # step 12 (CEJ) where the engine has explicit large-graph
            # paths. `force=True` (test fixtures) bypasses.
            if tw >= 12 and not force:
                return None

        # Phase A — discovery. At recursive depth > 1 (called from a
        # chord-rule sub-synthesis on an intermediate graph), we behave
        # like the legacy `_try_clique_atom_chord_peel` (step 7.9): skip
        # the expensive cell partition VF2 and Phase B closed-form
        # trials, just enumerate atom decompositions. Profiling
        # (cProfile of Z(1,2)) showed legacy ran clique_atom_chord_peel
        # for 14s on one intermediate; the merged dispatcher running its
        # FULL discovery there added ~8s of unnecessary work.
        decompositions = self._discover_decompositions(
            graph, max_junction_size=max_junction_size, tw_ratio=tw_ratio,
            skip_cells=recursive_call,
            force_cells=force,
            skip_atoms=skip_atoms,
        )
        if not decompositions:
            return None

        # Phase B — cell-only closed-form trial. Skipped at recursive
        # depth (where no cell decompositions exist). Only the FIRST
        # (highest-priority) cell decomposition is tried at top level —
        # Phase B's product_formula path recursively synthesizes each
        # inter-cell component (~seconds per attempt). Trying formulas
        # across all 4 candidate cell partitions cost ~16s on Z(1,2).
        # For graphs that satisfy a closed-form (Cm(2,2): kmatching,
        # K_4×K_4 chain: unified), the first cell partition is the one
        # with the smallest chord count — already the natural winner
        # for closed-form success.
        if not recursive_call:
            for d in decompositions:
                if d.kind != "cell":
                    continue
                result = self._try_cell_closed_forms(graph, d, max_depth)
                if result is not None:
                    self._cache_and_promote(graph, result)
                    return result
                break  # only first cell decomposition

        # Phase C: rank by predicted cost.
        #
        # At recursive depth > 1, prefer INTRA-atom peel over inter-atom
        # peel, matching legacy step 7.9 `_try_clique_atom_chord_peel`'s
        # behavior on contracted intermediates. Empirically (Z(1,2)
        # cProfile, May 23 2026): inter-atom peel on a depth-2
        # intermediate keeps atoms intact → recursive structurally-
        # similar sub-problems; intra-atom peel breaks atoms → sparser
        # residue solvable cheaply by other engine dispatch (treewidth_dp,
        # cell_quotient_*). The per-edge cost predictor doesn't model
        # this "residue quality" — it sees inter (4 edges) as cheaper
        # than intra (12 edges) and picks the wrong one. Legacy 7.9 only
        # offered intra, so this bias is hidden there but emerges in the
        # merged dispatcher.
        if recursive_call:
            # Sort intra first, then by predicted cost
            decompositions.sort(
                key=lambda d: (
                    0 if d.peel_type == "intra" else 1,
                    d.predicted_chord_cost,
                    len(d.components),
                ),
            )
        else:
            decompositions.sort(
                key=lambda d: (d.predicted_chord_cost, len(d.components)),
            )
        best = decompositions[0]

        # Cache-aware accept: at TOP level, compute the canonical_key
        # of one trial contraction (best decomposition, first chord
        # edge) and check `self._multigraph_cache`. If hit → the
        # chord-rule's recursive sub-syntheses will amortize through
        # the cache (Z(1,2) post-warmup case). If miss → the
        # contractions will each pay full sub-synth cost (Pm(2) case).
        # This is the structural replacement for the old
        # `node_count <= 30` gate and the brittle `m <= 100` gate
        # that preceded this rule. The two-way classification —
        # Z(1,2)/Cm cases empirically split from Pm(2)/Z(2,1) — drove
        # the validation: chord-peel wins iff intermediates are cached.
        # The probe is ~1ms (just one contraction + canonical_key).
        # Falls back to the predicted-cost threshold for cases where
        # the trial probe can't be computed cleanly (no chord edges,
        # exception, etc.).
        if not recursive_call and not force and best.chord_edges:
            try:
                from ..graphs.k_sum import _contract_edge_multi
                trial_chord = best.chord_edges[0]
                trial_mg = _contract_edge_multi(
                    graph, trial_chord[0], trial_chord[1],
                )
                trial_key = trial_mg.canonical_key()
                cache_hit = trial_key in self._multigraph_cache
            except Exception:
                cache_hit = None  # probe failed — fall back to predictor
            if cache_hit is False:
                # Confirmed miss → chord-rule has no cache amortization.
                # Defer to step 8 treewidth_dp.
                return None
            # On hit (True) or probe failure (None), fall through to
            # the predictor as a secondary gate.

        # Predicted-cost reject (fallback when cache probe inconclusive
        # or trial contraction not feasible). Calibrated against
        # `_INTER_LEGACY_PER_EDGE = 1.3` (Z(1,2) 4 chord edges
        # ×1.3×0.05=0.26 — well under the 0.85 threshold).
        if (not recursive_call
                and best.predicted_chord_cost >= self._PREDICTED_COST_REJECT):
            return None

        result = self._chord_peel_decomposition(
            graph, best, max_depth,
            recurse_residue=recurse_residue,
            min_recursion_size=min_recursion_size,
        )
        if result is not None:
            self._cache_and_promote(graph, result)
        return result

    def _cache_and_promote(self, graph: Graph, result: SynthesisResult) -> None:
        """Cache result under graph's canonical_key and promote to table."""
        try:
            cache_key = graph.canonical_key()
        except Exception:
            return
        self._cache[cache_key] = result
        self._promote_to_table(graph, cache_key, result)

    def _emit_cell_quotient_result(self, graph, cache_key, poly, *,
                                   method, recipe, label, record_label=None):
        """Finalize a cell-quotient dispatch hit: log, build result, cache, promote.

        Shared tail of the cell-quotient DP dispatch arms (grid-streamed / cycle /
        tree / bipartite-junction / per-component / hybrid) — each differs only in
        the dispatch fn plus these labels.
        """
        _log = get_log()
        n, m = graph.node_count(), graph.edge_count()
        _log.record(EventType.CELL_QUOTIENT_DP, "engine",
                    f"{record_label or label}: {n}n {m}e", graph=graph)
        self._maybe_emit_cell_partition(graph)
        self._log(f"{label}: {n}n, {m}e")
        result = SynthesisResult(
            polynomial=poly, recipe=[recipe], verified=True, method=method,
        )
        self._cache[cache_key] = result
        self._promote_to_table(graph, cache_key, result)
        return result

    def _synthesize_connected(
        self,
        graph: Graph,
        max_depth: int
    ) -> SynthesisResult:
        """Synthesize polynomial for connected graph using creation-expansion-join."""
        _log = get_log()
        target_edges = graph.edge_count()

        # For small graphs, spanning tree expansion is faster than VF2 search.
        if target_edges <= 15:
            _log.record(EventType.EDGE_ADD, "engine",
                        f"Small graph ({target_edges}e), spanning tree expansion")
            return self._synthesize_from_k2(graph, max_depth)

        # For graphs too large for VF2 cover-search to converge, skip
        # straight to spanning-tree expansion. The WL pre-filter rejects
        # impossible candidates but for dense symmetric graphs (Z(1,3):
        # 36n/162m) the necessary-condition is satisfied by every
        # plausible candidate — VF2 then hangs exploring valid placements
        # that can't form a complete cover. The cover-based fallback was
        # designed for n ≤ 25-ish; beyond that it's wasted work.
        if graph.node_count() > 30:
            _log.record(EventType.EDGE_ADD, "engine",
                        f"Large graph ({graph.node_count()}n {target_edges}e), "
                        f"skip cover-search and use spanning tree expansion")
            self._log(f"Large graph fallback ({graph.node_count()}n), "
                      f"spanning tree expansion")
            return self._synthesize_from_k2(graph, max_depth)

        # Only use tiles that cover a meaningful portion of the graph
        min_tile_edges = max(target_edges // 3, 4)

        # Only use explicitly-named entries as tile candidates
        candidates = [
            c for c in self.table.find_minors_of(graph)
            if not c.name.startswith("synth_") and not c.name.startswith("hybrid_")
               and c.edge_count >= min_tile_edges
               and c.canonical_key != graph.canonical_key()
        ]

        cover = None
        minor = None

        # Per-candidate VF2 time budget for graphs in the (20, 30] range
        # where the n>30 short-circuit doesn't fire but VF2 may still
        # explore many no-cover placements. Keeps cover-search bounded.
        per_candidate_budget = 30.0 if graph.node_count() > 20 else None

        for candidate in candidates:
            self._log(f"Trying minor: {candidate.name} ({candidate.edge_count} edges)")
            trial_cover = find_disjoint_cover(
                graph, candidate, self.table,
                max_search_time_s=per_candidate_budget,
            )

            if not trial_cover.tiles:
                continue  # Not a real subgraph, try next

            if trial_cover.covered_nodes != graph.nodes:
                self._log(f"  Cover incomplete ({len(trial_cover.covered_nodes)}/{len(graph.nodes)} nodes)")
                continue  # Doesn't cover all nodes, try next

            # Found a usable cover
            cover = trial_cover
            minor = candidate
            break

        if cover is None:
            _log.record(EventType.COVER_RESULT, "engine",
                        f"No cover found, falling back to spanning tree expansion")
            return self._synthesize_from_k2(graph, max_depth)

        _log.record(EventType.COVER_RESULT, "engine",
                    f"{len(cover.tiles)} tiles of {minor.name}, "
                    f"{len(cover.uncovered_edges)} uncovered edges")
        self._log(f"Cover: {len(cover.tiles)} tiles, {len(cover.uncovered_edges)} uncovered edges")

        # Compute base polynomial from disjoint tiles (product formula)
        poly = TuttePolynomial.one()
        recipe = [f"Tiling with {len(cover.tiles)} copies of {minor.name}"]
        all_minors = {minor.canonical_key}

        # Snapshot accumulator before edge addition
        pre_minors = set(self._mg_minors_accum)

        for tile in cover.tiles:
            poly = poly * tile.minor.polynomial
            recipe.append(f"  Tile {tile.minor.name}: T = {tile.minor.polynomial}")

        # Build the covered subgraph as a MultiGraph for edge addition
        covered_edge_counts = {}
        for tile in cover.tiles:
            for edge in tile.covered_edges:
                covered_edge_counts[edge] = covered_edge_counts.get(edge, 0) + 1

        # Handle uncovered edges using the correct formula:
        # - Bridge (connects different components): T(G+e) = x · T(G)
        # - Chord (within same component): T(G+e) = T(G) + T(G/{u,v})
        if cover.uncovered_edges:
            uncovered_list = sorted(cover.uncovered_edges)
            _log.record(EventType.EDGE_ADD, "engine",
                        f"Adding {len(uncovered_list)} uncovered edges")
            self._log(f"Adding {len(uncovered_list)} uncovered edges via edge addition")
            recipe.append(f"Edge addition for {len(uncovered_list)} uncovered edges")

            current_mg = MultiGraph(
                nodes=graph.nodes,
                edge_counts=covered_edge_counts,
                loop_counts={}
            )

            for u, v in uncovered_list:
                if current_mg.in_same_component(u, v):
                    # Chord: T(G+e) = T(G) + T(G/{u,v})
                    merged = current_mg.merge_nodes(u, v)
                    merged_poly = self._synthesize_multigraph(merged, skip_minor_search=True)
                    poly = poly + merged_poly
                else:
                    # Bridge: T(G+e) = x · T(G)
                    poly = TuttePolynomial.x() * poly

                edge = (min(u, v), max(u, v))
                new_edge_counts = dict(current_mg.edge_counts)
                new_edge_counts[edge] = new_edge_counts.get(edge, 0) + 1
                current_mg = MultiGraph(
                    nodes=current_mg.nodes,
                    edge_counts=new_edge_counts,
                    loop_counts=current_mg.loop_counts
                )

        # Harvest minors from edge addition
        all_minors |= (self._mg_minors_accum - pre_minors)

        # Verify
        verified = verify_spanning_trees(graph, poly)
        _log.record(EventType.VERIFY, "engine",
                    f"CEJ {'passed' if verified else 'FAILED'} Kirchhoff check")

        return SynthesisResult(
            polynomial=poly,
            recipe=recipe,
            verified=verified,
            method="creation_expansion_join",
            tiles_used=len(cover.tiles),
            fringe_edges=0,
            minors_used=all_minors,
        )

    def _synthesize_from_k2(
        self,
        graph: Graph,
        max_depth: int,
        *,
        sort_chords: bool = False,
        fast: bool = False,
    ) -> SynthesisResult:
        """Build polynomial from spanning tree + edge addition (CEJ fallback).

        Find a spanning tree (T = x^(n-1)), then add each non-tree edge (chord)
        via T(G + e) = T(G) + T(G/{u,v}).

        ``fast=True`` is the multigraph fast path: it sorts chords by a
        contraction-priority heuristic and propagates ``max_depth`` into the
        recursive multigraph synthesis. Chord order does NOT change the result
        (Tutte is order-invariant — see NOTE), so the two modes differ only in
        performance, recipe text, and the emitted method tag.
        """
        self._log("Building via spanning tree + edge addition"
                  + (" (fast path)" if fast else ""))

        n = graph.node_count()
        if n == 0:
            return SynthesisResult(
                polynomial=TuttePolynomial.one(),
                recipe=["Empty graph"],
                verified=True,
                method="base_case",
            )

        # Snapshot accumulator to diff later
        pre_minors = set(self._mg_minors_accum)
        recipe = ["Spanning tree + edge addition (fast)" if fast
                  else "Spanning tree + edge addition"]

        # Find a spanning tree using BFS
        tree_edges = set()
        visited = set()
        start = next(iter(graph.nodes))
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    tree_edges.add((min(node, neighbor), max(node, neighbor)))

        chords = [e for e in graph.edges if e not in tree_edges]
        if sort_chords:
            # Prefer chords whose contraction is more likely to create cut
            # vertices (fewer shared neighbours between endpoints).
            def chord_priority(e):
                u, v = e
                nu, nv = graph.neighbors(u), graph.neighbors(v)
                return (len(nu & nv), min(len(nu), len(nv)))
            chords.sort(key=chord_priority)
        # NOTE: chord-ordering optimizations from `_iterative_chord_rule` do NOT
        # transfer to this additive loop. Both sigma-orbit and smart-order
        # REGRESSED Z(1,3) empirically (May 2026): this loop ADDS each chord to
        # current_mg, so the graph evolves and sigma-orbit isomorphism / cache
        # hits fail. The `sort_chords` heuristic above is the only ordering that
        # helped (fast path). See [[project_wl_filter_and_largegraph_gate]].

        self._log(f"Spanning tree: {len(tree_edges)} edges, chords: {len(chords)}")
        recipe.append(f"Spanning tree: {len(tree_edges)} edges"
                      + ("" if fast else f", T = x^{len(tree_edges)}"))
        recipe.append(f"Chords: {len(chords)}" if fast
                      else f"Chords to add: {len(chords)}")

        # Start with spanning tree polynomial: x^(n-1)
        poly = TuttePolynomial.x(len(tree_edges))
        current_mg = MultiGraph(
            nodes=graph.nodes,
            edge_counts={e: 1 for e in tree_edges},
            loop_counts={},
        )

        # Add each chord via the edge-addition formula.
        for i, (u, v) in enumerate(chords):
            merged = current_mg.merge_nodes(u, v)
            if fast:
                merged_poly = self._synthesize_multigraph(
                    merged, max_depth, skip_minor_search=True)
            else:
                merged_poly = self._synthesize_multigraph(
                    merged, skip_minor_search=True)
            poly = poly + merged_poly
            edge = (min(u, v), max(u, v))
            new_edge_counts = dict(current_mg.edge_counts)
            new_edge_counts[edge] = new_edge_counts.get(edge, 0) + 1
            current_mg = MultiGraph(
                nodes=current_mg.nodes,
                edge_counts=new_edge_counts,
                loop_counts=current_mg.loop_counts,
            )
            if not fast:
                self._log(f"Added chord {i+1}/{len(chords)}: ({u},{v})")

        recipe.append(f"Final: {poly.num_terms()} terms" if fast
                      else f"Final polynomial has {poly.num_terms()} terms")
        new_minors = self._mg_minors_accum - pre_minors
        return SynthesisResult(
            polynomial=poly,
            recipe=recipe,
            verified=True,
            method="spanning_tree_expansion_fast" if fast else "spanning_tree_expansion",
            minors_used=new_minors,
        )

    def _synthesize_fast(
        self,
        graph: Graph,
        max_depth: int
    ) -> SynthesisResult:
        """Fast synthesis path that skips minor search.

        This method handles basic cases and optimizations but skips the
        expensive minor search in _synthesize_connected. It's used for
        intermediate merged graphs that are unlikely to match known minors.

        The order of checks:
        1. Rainbow table lookup (still fast)
        2. Base cases (empty, single edge)
        3. Disconnected graphs
        4. Cut vertex factorization
        5. Direct spanning tree expansion (skip minor search)

        Args:
            graph: Graph to compute polynomial for
            max_depth: Maximum recursion depth

        Returns:
            SynthesisResult with computed polynomial
        """
        # Base cases (0-1 edges) — checked first to avoid hashing trivial graphs
        if graph.edge_count() <= 1:
            if graph.edge_count() == 0:
                return SynthesisResult(
                    polynomial=TuttePolynomial.one(),
                    recipe=["Empty graph: T = 1"],
                    verified=True,
                    method="base_case"
                )
            return SynthesisResult(
                polynomial=TuttePolynomial.x(),
                recipe=["Single edge: T = x"],
                verified=True,
                method="base_case"
            )

        # Two-level cache: fast_hash filter before expensive canonical_key
        fh = graph.fast_hash()
        if not hasattr(self, '_fast_simple_hash_set'):
            self._fast_simple_hash_set = set()
        if not hasattr(self, '_table_nm_set'):
            self._table_nm_set = {
                (e.node_count, e.edge_count) for e in self.table.entries.values()
            }

        if fh in self._fast_simple_hash_set:
            # Potential cache/table hit — compute canonical_key
            cache_key = graph.canonical_key()
        elif (graph.node_count(), graph.edge_count()) in self._table_nm_set:
            # Could match a rainbow table entry — compute canonical_key
            cache_key = graph.canonical_key()
        else:
            # No cache entry and no table entry with this n,m — skip canonical_key
            cache_key = None

        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Rainbow table lookup by key (avoids recomputing canonical_key)
        if cache_key is not None:
            entry = self.table.get_entry_by_key(cache_key)
        else:
            entry = None
        if entry is not None:
            result = SynthesisResult(
                polynomial=entry.polynomial,
                recipe=["Rainbow table lookup"],
                verified=True,
                method="lookup",
                minors_used={cache_key},
            )
            self._cache[cache_key] = result
            return result

        # (Base cases already handled above — edge_count <= 1 returns early)

        # Helper to ensure cache_key is computed before caching
        def _ensure_cache_key():
            nonlocal cache_key
            if cache_key is None:
                cache_key = graph.canonical_key()
            return cache_key

        # 3. Disconnected graphs (recurse through _synthesize_fast, not full synthesize)
        components = graph.connected_components()
        if len(components) > 1:
            poly = TuttePolynomial.one()
            all_minors = set()
            for comp in components:
                comp_result = self._synthesize_fast(comp, max_depth)
                poly = poly * comp_result.polynomial
                all_minors |= comp_result.minors_used
            result = SynthesisResult(
                polynomial=poly,
                recipe=[f"Disconnected: {len(components)} components (fast)"],
                verified=True,
                method="disjoint_union",
                minors_used=all_minors,
            )
            ck = _ensure_cache_key()
            self._cache[ck] = result
            self._fast_simple_hash_set.add(fh)
            self._promote_to_table(graph, ck, result)
            return result

        # 4. Cut vertex factorization (recurse through _synthesize_fast, not full synthesize)
        cut = graph.has_cut_vertex()
        if cut is not None:
            sub_components = graph.split_at_cut_vertex(cut)
            poly = TuttePolynomial.one()
            all_minors = set()
            for comp in sub_components:
                comp_result = self._synthesize_fast(comp, max_depth)
                poly = poly * comp_result.polynomial
                all_minors |= comp_result.minors_used
            result = SynthesisResult(
                polynomial=poly,
                recipe=[f"Cut vertex at {cut}: {len(sub_components)} components (fast)"],
                verified=True,
                method="cut_vertex",
                minors_used=all_minors,
            )
            ck = _ensure_cache_key()
            self._cache[ck] = result
            self._fast_simple_hash_set.add(fh)
            self._promote_to_table(graph, ck, result)
            return result

        # 4.5 Try series-parallel O(n) computation
        sp_poly = compute_sp_tutte_if_applicable(graph)
        if sp_poly is not None:
            result = SynthesisResult(
                polynomial=sp_poly,
                recipe=["Series-parallel decomposition (fast)"],
                verified=True,
                method="series_parallel",
            )
            ck = _ensure_cache_key()
            self._cache[ck] = result
            self._fast_simple_hash_set.add(fh)
            self._promote_to_table(graph, ck, result)
            return result

        # 4.6 Try k-sum decomposition (useful for intermediate merged graphs)
        if graph.edge_count() >= 6:
            result = self._try_ksum_decomposition(graph)
            if result is not None:
                ck = _ensure_cache_key()
                self._cache[ck] = result
                self._fast_simple_hash_set.add(fh)
                self._promote_to_table(graph, ck, result)
                return result

        # 5. Direct spanning tree expansion (skip minor search)
        result = self._synthesize_from_k2(graph, max_depth, sort_chords=True, fast=True)
        ck = _ensure_cache_key()
        self._cache[ck] = result
        self._fast_simple_hash_set.add(fh)
        self._promote_to_table(graph, ck, result)
        return result

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def synthesize(graph: Graph, verbose: bool = False, method: str = "auto") -> SynthesisResult:
    """Convenience function to synthesize polynomial for a graph.

    Args:
        graph: Graph to compute polynomial for
        verbose: Print progress information
        method: Retained for backward compatibility; all values now route
            through the single SynthesisEngine.

    Returns:
        SynthesisResult with computed polynomial
    """
    engine = SynthesisEngine(verbose=verbose)
    return engine.synthesize(graph)


def compute_tutte_polynomial(graph: Graph, method: str = "auto") -> TuttePolynomial:
    """Compute Tutte polynomial for a graph.

    This is the main entry point for polynomial computation.

    Args:
        graph: Graph to compute polynomial for
        method: Synthesis method - "auto", "tiling", or "algebraic"

    Returns:
        TuttePolynomial for the graph
    """
    result = synthesize(graph, method=method)
    return result.polynomial
