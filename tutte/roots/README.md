# `tutte/roots/` — rooted-Tutte composition for cell-decomposable graphs

Dynamic-programming methods that decompose a graph into **cells** plus
inter-cell **junctions**, compute each cell's **rooted Tutte
polynomial** (boundary-partition-indexed), and compose them via
vertex-sum convolution.

> **Why "roots"?** Every file here computes or composes _rooted Tutte
> polynomials_ `T_rooted(G, S)[P]` indexed by partitions `P` of a
> boundary set `S ⊆ V(G)`. The boundary vertices are the "roots" that
> anchor the polynomial to the rest of the graph during composition.
> The standard Tutte polynomial recovers via
> `T(G) = Σ_P T_rooted(G, S)[P]`.

## When the engine uses this package

`tutte/synthesis/engine.py::_synthesize_inner` dispatches to two live
entry points in this package:

| Entry point                                          | Engine step | Trigger                                                                                       |
| ---------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------- |
| `compute_chain_full_poly_from_spec`                  | 7.4 (`edge_count ≥ 80`) | Cell-quotient is a linear chain of ≥ 3 cells; extracts a transfer matrix and iterates it (Chimera Cm(1, n)) |
| `compute_cell_quotient_grid_dp_streamed`             | 7.45 (`edge_count ≥ 60`) | 2D-grid cell-quotient of `K_{a,b}` cells with disjoint per-direction anchors (Cm_2 fits)      |

Both dispatchers share the precondition: `try_hierarchical_partition`
(in `tutte/graphs/covering.py`) must return a cell decomposition. If
no decomposition is found or the cell-quotient topology doesn't match,
the entry returns `None` and the engine falls through.

The remaining cell-quotient routines in this package
(`compute_cell_quotient_tree_dp`,
`compute_cell_quotient_bipartite_junction_dp` and its per-component
variant) are **chain-recurrence infrastructure**: the chain path
(7.4) builds its `(cell, junction)` spec via
`build_bipartite_junction_spec` and carries it as a `CellTreeSpec`.
They are no longer dispatched directly from the engine cascade. The
former cycle and hybrid cell-quotient DPs moved to
[`../deprecated/`](../deprecated/README.md).

## Module map

| File                                                                       | Purpose                                                                                                                                                                                       |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`__init__.py`](__init__.py)                                               | Public engine entries (`compute_cell_quotient_grid_dp_streamed`, `compute_cell_quotient_tree_dp`) plus re-exports.                                                                            |
| [`rooted_tutte.py`](rooted_tutte.py)                                       | Brute-force `T_rooted` (cells of ≤ 16 edges); boundary primitives (`join_partitions`, `delta`, `restrict_partition`, `divide_by_x_minus_1_power`); persistent rooted-lookup load/save.        |
| [`aut_orbit.py`](aut_orbit.py)                                             | Aut-based orbit canonicaliser: `compute_cell_aut` (VF2), `canonical_partition` (lex-min over aut), `aut_compress_t_rooted`, `build_relabel_aut`. Generic over any cell with non-trivial aut.    |
| [`cell_anchor_adapter.py`](cell_anchor_adapter.py)                         | Graph-agnostic cycle / grid / tree topology detection plus per-cell anchor alignment (`normalize_cell_anchors_for_cycle`, `detect_cell_anchor_groups`, `extract_grid_specs`, `extract_path_specs`). |
| [`cell_quotient_helpers.py`](cell_quotient_helpers.py)                     | Hot-path helpers: `precompute_M_table` (orbit-level), `orbit_convolve` (raw-dict arithmetic + `_polynomial_c.poly_mul`), `enumerate_partitions_cached`, junction component count `c_J`.        |
| [`cell_quotient_path.py`](cell_quotient_path.py)                           | `compute_path_dp(...)` — path-of-cells DP (no closing step). Returns the boundary-partition-indexed dict; consumed by `cell_quotient_grid.py`.                                                |
| [`cell_quotient_grid.py`](cell_quotient_grid.py)                           | `compute_grid_dp_streamed_kab(...)` — composes a 2D grid via row path DPs convolved through vertical junctions; streaming version avoids OOM on Cm_3-scale targets.                            |
| [`cell_quotient_tree.py`](cell_quotient_tree.py)                           | `compute_tree_dp_recursive` — branching cell-tree topology DP via post-order recursion (orbit-compressed). `CellTreeSpec` carries the topology + per-cell anchor groups; the chain-recurrence path builds on it. Live chain-recurrence infrastructure. |
| [`cell_quotient_bipartite_junction.py`](cell_quotient_bipartite_junction.py) | `build_bipartite_junction_spec` (consumed by the chain-recurrence path) + `compute_cell_quotient_bipartite_junction_dp` and its per-component variant. Generalises the k-matching path to non-matching bipartite junctions; per-component variant bounds joint Bell counts. Live chain-recurrence infrastructure. |
| [`chain_recurrence.py`](chain_recurrence.py)                               | Constructive re-derivation of Noy & Ribò (2007) for chain-of-cells families. Faddeev-LeVerrier mod p extracts the order-`r` recurrence in ms per modular point.                              |
| [`chord_junction_closed_form.py`](chord_junction_closed_form.py)            | Unified bivariate chord-junction theorem (matching `E_J`). Symmetric + asymmetric closed forms over `2^|V_k|` merger terms. Backed by persistent merger lookup table.                          |
| [`sokal_z_chord_junction.py`](sokal_z_chord_junction.py)                    | Sokal-Z generalization for **arbitrary** `E_J` (non-matching / multi-edge). Brute-force + per-H_J-component + edge-by-edge tree-DP enumeration; multi-point eval + bivariate Lagrange interpolation. Handles Z(1,2)-class junctions where the matching-only theorem fails. |
| [`signed_quotient.py`](signed_quotient.py)                                  | Live σ-finder `find_best_sigma` for the σ-equivariant chord-ordering path (used by the engine + `graphs/k_sum.py`). The test-only signed-DP-via-interpolation pipeline moved to [`../deprecated/`](../deprecated/README.md). |
| [`_partition_c.py`](_partition_c.py)                                       | cffi C extension wrappers for partition / canonicalisation hot paths (`apply_perm_canonical_c`, `h_canonicalize_c_batched`, `precompute_M_batched_inner_c_mod`).                              |

## Algorithm overview

### Tree DP — post-order recursion

Composes `T(graph)` by post-order recursion over the cell tree: each
cell absorbs each child subtree via a junction step (vertex-sum
convolution at the junction's shared boundary) followed by a
cell-merge step (vertex-sum at the child's parent-facing boundary).

`CellTreeSpec` carries the cell template, junction template,
per-cell-anchor groups, and root choice. Per-cell orbit compression is
enabled by default in the recursive entry point.

For chain-shaped cell trees, `compute_chain_full_poly_from_spec` (in
`chain_recurrence.py`) can replace the entire DP with a forward
recurrence in the number of cells.

### Grid DP — row composition

`compute_grid_dp_streamed_kab` composes a `rows × cols` grid by
computing per-row `T_rooted` via path DP and convolving rows through
vertical junctions. Streaming `precompute_M_table` chunks state orbits
to bound peak memory; the modular variant
(`precompute_M_table_mod`) builds the M-table directly in `int mod p`,
no polynomial allocation per chunk.

### Disconnected-junction divisor

The standard vertex-sum convolution divisor is `(x − 1)^{|S| − 1}` per
junction step, which assumes the junction is a connected graph. For
matching junctions (`M_k` = `k` disjoint edges), the correct divisor
is `(x − 1)^{|S| − c_J(S)}` where `c_J(S)` is the number of junction
components touching the shared boundary. `components_touching` in
`cell_quotient_helpers.py` detects this automatically; all DPs use the
corrected formula.

### Performance optimisations

- **Aut orbit compression** — the cell template's automorphism group
  acts on partitions; storing one polynomial per orbit (not per
  partition) cuts state space by `|Aut|`. K_{4,4} compresses 4140
  partitions to ~43 orbits.
- **Orbit-level `precompute_M_table`** — pick one representative per
  state orbit, multiply contributions by orbit size.
- **Position-invariant orbit caching** —
  `enumerate_partitions_cached` computes orbits once for canonical
  positions `[0..n−1]` and relabels per call.
- **Raw-dict polynomial arithmetic** — inner loops accumulate
  `dict[(xpow, ypow)] → coeff` directly, skipping the
  `TuttePolynomial.encode/decode` cycle. Combined with the
  `_polynomial_c` C extension this gives a meaningful per-mul
  speedup.
- **Streaming junction enumeration** — `precompute_M_table` accepts
  compressed junction orbits and expands their per-cell orbit members
  internally, one orbit at a time. Combined with `out_cell_anchor_groups`
  on the 2a vertical-junction step (which lifts the state's per-cell
  aut through `M_k` edges to the output boundary) keeps the 2a output
  orbit-compressed.
- **Chunked row composition** (`precompute_M_and_convolve_streaming`)
  wraps `precompute_M_table` in a state-orbit chunk loop with a
  raw-dict accumulator, bounding peak memory.
- **Modular point-value DP** — `precompute_M_table_mod` and
  `precompute_and_convolve_c_mod` evaluate the DP in modular
  arithmetic; the full polynomial is recovered by bivariate Lagrange
  interpolation per prime and CRT combine across primes. The
  precision-safe path for graphs whose coefficients overflow `int64`.

## Generalising for shared anchors

The current cycle / grid DPs assume each cell has **disjoint** anchor
sets per junction. D-Wave Cm_3's interior cells break this assumption:
the same anchor group serves both junctions on a cell's side.

`cell_anchor_adapter.detect_cell_anchor_groups` returns a
`CellAnchorGroups` that carries per-cell anchor IDs, per-junction
group assignments, and `has_shared_anchors()`. Consuming this in
`compute_path_dp` and `compute_grid_dp_with_layout` to allow shared
boundary positions is the remaining work to unlock Cm_3 via this path.

## Testing

`tutte/tests/test_roots.py` covers the live grid / tree DPs against
the engine + Kirchhoff oracle (it also exercises the deprecated cycle
and interleaved DPs from their new home in
[`../deprecated/`](../deprecated/README.md) so they don't bit-rot).
`tutte/tests/test_cell_quotient_bipartite_junction.py` covers the
bipartite-junction DP including the per-component decomposition.
`tutte/tests/test_chain_recurrence.py` covers chain + cycle
recurrences (the Noy-Ribò re-derivation).

## Related docs

- [`tutte/docs/06_3_rooted_tutte_framework.md`](../docs/06_3_rooted_tutte_framework.md)
  — algebraic foundation for rooted Tutte composition
- [`tutte/docs/06_4_cell_quotient_cycle_dp.md`](../docs/06_4_cell_quotient_cycle_dp.md)
  — cycle DP deep dive (module now in [`../deprecated/`](../deprecated/README.md))
- [`tutte/docs/06_5_cell_quotient_grid_dp.md`](../docs/06_5_cell_quotient_grid_dp.md)
  — grid DP and streaming variants
- [`tutte/docs/06_6_cell_quotient_tree_dp.md`](../docs/06_6_cell_quotient_tree_dp.md)
  — tree DP and per-cell compression
- [`tutte/docs/06_7_chain_recurrence_algebra.md`](../docs/06_7_chain_recurrence_algebra.md)
  — chain recurrence framework
- [`tutte/docs/06_8_modular_arithmetic_pathways.md`](../docs/06_8_modular_arithmetic_pathways.md)
  — modular point-value paths and interpolation
- [`tutte/research/engine_workflow_primer.md`](../research/engine_workflow_primer.md)
  — first-principles entry into where the cell-quotient DPs (including
  the chain recurrence) sit in the engine cascade
