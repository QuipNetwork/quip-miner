# Decomposition + Chord-Peel — Unified Dispatcher

`SynthesisEngine._try_decomposition_chord_peel`
(`tutte/synthesis/engine.py`, engine step 7.88) — a single dispatcher
that consumes both atom decompositions and rainbow-table cell
partitions, tries cell-only closed-form formulas, then falls through
to a cost-gated chord-rule peel. Replaces four separate legacy
methods:

- `_try_unified_atom_chord_peel` (step 7.88a, atom-based chord-peel)
- `_try_cross_cell_chord_peel` (step 7.88, smallest-junction peel)
- `_try_clique_atom_chord_peel` (step 7.9, internal-K_n peel)
- `_try_hierarchical` → `_synthesize_hierarchical` (step 10, cell
  partition + 5 sub-paths)

All four shared the same underlying primitive: discover a
decomposition of the graph into structurally simpler components, then
peel the connecting edges via `_iterative_chord_rule`
(`tutte/graphs/k_sum.py:368`). The merge consolidates discovery,
closed-form trial, and chord-peel into one pipeline.

## Phases

### Phase A — Discovery

Collect candidate `Decomposition` records from two granularities:

- **Atoms** — lightweight structural shapes (`K_n`, `K_{a,b}`, `B_n`,
  `W_n`, `L_n`, `Y_n`) detected via fast pattern matching in
  `tutte/graphs/atom_detection.py`. Produces three candidates:
  - `atom_inter_legacy` (K_n-first, smallest inter-atom junction)
  - `atom_inter_het` (heterogeneous family mix)
  - `atom_intra_legacy` (intra-K_n edges, clique_atom style)
- **Cells** — rainbow-table `MinorEntry` partitions discovered via VF2
  in `tutte/graphs/covering.py`. Single-result entry points
  (`try_hierarchical_partition`, `try_heterogeneous_partition`) are
  preferred — the multi-result `iter_hierarchical_partitions` is
  exposed but not used by default because Phase B's product_formula
  only ever needs one partition (its precondition is structural, not
  cell-specific).

Each `Decomposition` carries:

- `kind`: `"atom"` or `"cell"`
- `components`: vertex-set partition
- `families`: per-component family name
- `cell_entries`: `MinorEntry` list (cell only — required for Phase B)
- `chord_edges`: the edges chord-rule will peel
- `predicted_chord_cost`: `edges × per_edge × tw_ratio`

Per-edge cost constants (calibrated against engine.synthesize timings
on Z(1,2), May 22-23 2026):

```
INTER_LEGACY = 1.3   # K_n atoms
INTRA        = 1.1   # internal K_n edges
INTER_HET    = 9.8   # heterogeneous (cache miss penalty)
CELL         = 1.3   # cell-partition chord edges
```

`tw_ratio = 0.05 + 0.02 × max(0, tw − 10)`.

A shared treewidth probe runs once for cost scaling. When `tw ≤ 8`,
the dispatcher returns `None` so step 8 `treewidth_dp(max_width=11)`
handles the graph trivially — this avoids regressing low-treewidth
graphs like Cm(1,3) (tw=4, tw_dp 0.07s vs chord-rule 11s).

### Phase B — Cell-only closed-form trial

For the FIRST (highest-priority) cell decomposition, try:

1. **`unified_formula`** (`extract_cell_topology` →
   `T(G) = (∏ T(cells)) × T(H)`) — works when every cell-pair shares
   a single vertex-pair connection. Returns method
   `"unified_formula"`.
2. **`kmatching_formula`** (`detect_kmatching_topology` →
   `apply_kmatching_formula`) — works when inter-cell edges form
   k-matchings between vertex-transitive cells. 3.2× speedup vs
   `treewidth_dp` on Cm2. Returns method `"kmatching_formula"`.

Only the first cell decomposition is tried because the legacy
`_try_formula_shortcircuit` (step 7.5 in older engine versions) had
the same behavior — trying multiple cell candidates each at multiple
formulas added ~16s of recursive sub-synth on Z(1,2) for no benefit
(Z(1,2)'s structure provably defeats both formulas regardless of
cell choice).

The legacy `product_formula` (`T(G) = (∏ T(cells)) × ∏ T(inter_components)`)
was REMOVED from Phase B. It worked only for partitions whose
inter-cell graph splits into truly independent components — rare in
D-Wave topologies. The chord-rule path (Phase C) produces the same
result correctly. Available behind `TUTTE_USE_LEGACY_DISPATCH=1` env
flag if a regression analysis ever needs it.

### Phase C — Cost-gated chord-rule

1. Sort all decompositions by `(predicted_chord_cost, len(components))`.
2. Pick the cheapest. We DO NOT reject high-predicted candidates
   outright (legacy `_try_unified_atom_chord_peel` had a `predicted
   >= 0.85` reject, but its caller chain had a different fallback
   structure; in the merge, an expensive chord-rule still beats the
   creation-expansion-join fallback for large dense graphs).
3. Compute σ via `find_best_sigma(nxg, require_free=True)` for
   σ-equivariant chord ordering.
4. Call `_iterative_chord_rule(graph, decomp.chord_edges, self,
   smart_order=False, sigma=sigma)`.

Method label depends on the winning decomposition:
- `decomposition_chord_peel_cell_inter`
- `decomposition_chord_peel_atom_inter`
- `decomposition_chord_peel_atom_intra`

### Phase D — Recursive residue peel

After `_iterative_chord_rule` returns `(g_chord_free, factors, adds)`,
the chord-free residue is synthesized via `self.synthesize(g_chord_free,
max_depth)`. The residue often admits a SECOND decomposition that the
original graph hid — atoms created by contraction triangles, cells
revealed by simplified inter-cell topology, etc.

The dispatcher's gate is `synth_depth ≤ 2`, so the residue (at
depth 2) can re-enter THIS dispatcher and peel its own decomposition.
The engine's per-canonical-key cache (`self._cache`) prevents
re-peeling identical residues. Termination is guaranteed by edge
monotonicity (`_iterative_chord_rule` strictly removes
`len(chord_edges)` edges per call) plus the outer `max_depth` budget.

`recurse_residue=True` and `min_recursion_size=12` are exposed as
opt-out knobs for tests and profiling; default behavior matches the
legacy chord-rule.

## Gates (in dispatch order)

```
if not _use_legacy_dispatch
   and graph.edge_count() >= 20         # smallest graph admitting cells
   and graph.node_count() <= 30         # chord-rule per-step cost
                                        # scales as 2^tw·m; Pm(2)-class
                                        # (n=40) reliably faster via
                                        # step 8 treewidth_dp(max_width=11)
   and self._synth_depth <= 2           # recursive residue peel once
```

`TUTTE_USE_LEGACY_DISPATCH=1` env flag routes through the legacy
7.88a / 7.88 / 7.9 / 10 sequence instead. Used for debug/regression
analysis; will be removed in a follow-up commit.

## Per-graph behavior (cold cache, post-merge)

| Graph    | n   | m   | Time   | Method                              |
| -------- | --- | --- | ------ | ----------------------------------- |
| Z(1,1)   | 12  | 22  | 0.01s  | `treewidth_dp` (step 8 short-circuit) |
| Cm(1,2)  | 16  | 36  | 0.02s  | `treewidth_dp`                      |
| Cm(1,3)  | 24  | 56  | 0.08s  | `treewidth_dp`                      |
| Z(1,2)   | 24  | 76  | ~50s   | `decomposition_chord_peel_atom_inter` |
| Cm(2,2)  | 32  | 80  | ~26s   | `cell_quotient_grid_dp_streamed` (step 7.45) |
| Cm(1,4)  | 32  | 76  | ~9s    | `cell_quotient_tree_dp` (step 7.8)  |
| Pm(2)    | 40  | 164 | ~58s   | `treewidth_dp` (step 8)             |

Z(1,2) regression vs legacy (~36s → ~50s, +14s) is the cost of unified
discovery + closed-form trials. Acceptable tradeoff — still well under
60s target, and the merge reduces four dispatch paths to one with
better failure modes for graphs that need multiple decomposition tiers.

## Reused infrastructure

- `find_disjoint_atoms`, `find_atoms_heterogeneous`,
  `find_smallest_junction` — `tutte/graphs/atom_detection.py`
- `try_hierarchical_partition`, `try_heterogeneous_partition`,
  `iter_hierarchical_partitions`, `analyze_inter_cell_edges`,
  `extract_cell_topology`, `detect_kmatching_topology`,
  `apply_kmatching_formula` — `tutte/graphs/covering.py`
- `_iterative_chord_rule`, `_combine_chord_iteration`,
  `_classify_bridges_chords` — `tutte/graphs/k_sum.py`
- `compute_best_tree_decomposition` — `tutte/graphs/treewidth.py`
- `find_best_sigma` — `tutte/roots/signed_quotient.py`
