# 6.6 — Cell-Quotient Tree Dynamic Programming

## Summary

Generalizes the [Cell-Quotient Cycle DP](06_4_cell_quotient_cycle_dp.md)
and the [Cell-Quotient Grid DP](06_5_cell_quotient_grid_dp.md) from
fixed cycle/grid topology to **arbitrary tree topology** in the
cell-quotient graph. For graphs whose hierarchical decomposition has
cell-quotient that is a TREE (n cells, n−1 junctions, no cycles), this
DP composes T(graph) by post-order recursion over the cell-tree,
absorbing each child subtree into its parent via junction + cell-merge
convolutions.

**Status**: prototype with optional per-cell orbit compression behind
`enable_per_cell_compression=False` flag. Default uncompressed mode
validated on 9 path + 4 branching test cases including 5-cell K*{4,4}
M_2 Cm₃ interior pattern (full polynomial match against engine).
Per-cell mode validated on 7 cases including K_5 claw shared M_3 and
K*{4,4} mixed-direction; **50× speedup** on 5-cell K\_{4,4} M_2 Cm₃
pattern (12.7 s → 0.3 s) with full polynomial match.

The tree DP is a building block: combined with cycle-closing chord
rule on closing junctions, it provides the leaves for hybrid
graphical-algebraic decomposition of cyclic cell-quotient
topologies (e.g., D-Wave Cm₃ grid).

## When it is used

`tutte/roots/cell_quotient_tree.py:compute_tree_dp_recursive(spec)`.
Not yet wired into the engine pipeline; consumed directly by hybrid
research scripts. Future engine integration would dispatch when:

1. `try_hierarchical_partition(graph, table)` returns a valid cell
   decomposition.
2. The cell-quotient graph is a TREE (all cells have a single
   spanning-tree role; no cycles in the quotient).
3. `cell_anchor_groups` is provided per `(cell_idx, neighbor_idx)`
   pair.

If the cell-quotient is a grid, dispatch to the specialized grid DP
instead (engine step 7.45); the cycle DP is deprecated (now in
`tutte/deprecated/cell_quotient_cycle.py`). If the cell-quotient has
cycles but admits a chord-rule peeling (cycle close, see
[`08_3_kmatching_formula.md`](08_3_kmatching_formula.md)), the tree
DP becomes the per-leaf computation in a chord-rule-peeled path.

## Algorithm — post-order recursion over the cell-tree

```
def compute_tree_dp_recursive(spec):
    pos = allocate_per_cell_positions(spec)  # one position per (cell, neighbor)
                                             # deduped when shared anchors

    def dp_subtree(cell_idx, parent_cell_idx):
        # Initial state: T_rooted(cell_template, all anchors used by
        # this cell) over per-cell allocated positions.
        state = aut_compress(T_rooted(cell_template, anchors), aut_group)

        children = sorted(neighbors(cell_idx) - {parent_cell_idx})
        for child in children:
            # Recurse on child subtree.
            child_state = dp_subtree(child, cell_idx)

            # Junction step: convolve state with junction T_rooted at
            # the boundary (cell_outward, child_inward).
            state = orbit_convolve(state, junction_T_rooted, M_junction)
            total_div += k - c_J(junction)

            # Cell-merge step: convolve state with child's returned
            # state at the shared boundary child_inward.
            state = orbit_convolve(state, child_state, M_cell_merge)
            total_div += k - c_J(cell)

            # Marginalize positions no longer needed (dead positions
            # after this child is fully absorbed).
            state = marginalize(state, future_live_positions)

        return state  # over parent_facing positions

    final_state = dp_subtree(spec.root, None)
    return divide_by_x_minus_1_power(sum(final_state.values()), total_div)
```

### Per-cell position allocation with shared-anchor support

`_allocate_tree_positions` assigns positions per (cell, neighbor)
pair. When the SAME underlying cell-template vertex appears in
multiple neighbor groups (shared anchors), the SAME position label is
allocated. This ensures the partition state correctly tracks shared
anchors across multiple junctions.

For example, K\_{4,4} cell with H neighbors using A-side anchors
[0, 1, 2, 3] and V neighbors using B-side anchors [4, 5, 6, 7]: each
H neighbor gets the same 4 positions for the A-side; each V neighbor
gets the same 4 positions for the B-side; the H and V anchor sets
are disjoint position sets.

### Junction step

Convolves state's T_rooted (over `state_open_pos`) with junction's
T_rooted (over `cell_outward_pos + child_inward_pos`) at the shared
boundary `cell_outward_pos`. Output state is over a new boundary that:

- Includes state's other open positions ("state_extra_pos").
- Includes child_inward_pos (junction's B side; the "new" boundary).
- Optionally includes cell_outward_pos (when it must remain "live"
  because future children share these anchors → keep_shared=True).

### Cell-merge step

Convolves state with the child subtree's returned T_rooted at the
shared boundary `child_inward_pos`. Output state has child_inward
consumed. Total divisor accumulates `(x − 1)^{|child_inward| − c_cell}`
where c_cell = number of cell-template components touching the
inward anchors.

### Marginalization

After processing each child, positions no longer needed (no remaining
sibling or parent uses them) are marginalized out — sum the state
polynomial over partitions consistent with each restricted partition,
yielding a state over the live boundary only.

## Per-cell orbit compression — the four-fix breakthrough

Per-cell compression for tree DP required FOUR coupled correctness
conditions. Each was discovered when a "simpler" partial fix exposed
the next bug.

### Fix 1 — Disjoint per-cell groups via Aut orbits

`_compute_aut_orbits_on_positions` with `preserve_anchor_sets`:

- Compute `Aut(cell_template)` via VF2.
- Filter to auts that map the anchor SET to itself, AND preserve
  each per-neighbor anchor set as a SET.
- Compute orbits of valid auts on positions.
- Each orbit is one per-cell group.

Without `preserve_anchor_sets`: K\_{4,4} cell with disjoint A/B side
anchors gets ONE orbit (the bipartite-side-swap aut maps A → B). The
filter keeps S_4 × S_4 acting independently on each side.

For K_n cells with overlapping anchor sets across neighbors (e.g.,
K_4 with neighbors using `[0,1]`, `[1,2]`, `[2,3]`): orbits MERGE
into one group of 4 (full S_4 acts on all 4), avoiding the per_cell
canonical-key precondition violation that would arise from naively
listing one group per (cell, neighbor) pair.

### Fix 2 — `keep_shared` fallback to uncompressed

When `cell_outward_pos` overlaps `future_live_set` (anchor-sharing
case), the junction's diagonal S_k aut on (cell_outward,
child_inward) doesn't realize the independent S_k × S_k that
`per_cell_canonical_key` assumes on output `[[cell_outward],
[child_inward]]`. This OVER-COMPRESSES output orbits by factor k!,
causing non-divisible coefficients in `orbit_convolve`.

Fix: detect `cell_outward_is_shared` and fall back to uncompressed
state for that step (`_expand_per_cell_state`).

### Fix 3 — Fully-consumed state group fallback

When a state per-cell group is ENTIRELY in `cell_outward` (typical
at the root cell with no parent), state's per-cell aut on consumed
positions doesn't lift to output via fixed `P_junc` iteration. The
M-table iterates `state_rep × all P_junc`; off-diagonal `(σ(state_rep)
× P_junc)` pairs land in different output orbits and get missed by
the `n_state` factor.

Detection: any state per-cell group is a subset of
`cell_outward_set` AND has non-trivial per-cell orbit size (≥ 2).
Expand state when detected. Junction aut compression deferred until
AFTER the fallback decision.

### Fix 4 — Per-cell child expansion at cell-merge

When child returns per-cell, the M-table iterates only `[rep]` per
child orbit (because `precompute_M_table` doesn't support
`junction_cell_anchor_groups` — only `state_cell_anchor_groups`).
Other child orbit members are missed → wrong convolution result.

Fix: when child is per-cell at cell-merge, expand child to
uncompressed. Also expand state if it's per-cell (state's aut on
fully-consumed groups in cell-merge has same lift issue as junction
step's fully-consumed fallback).

## Test results

All 7 per-cell test cases pass with full polynomial match
(`tutte/research/scripts/tree_dp_per_cell_compression_test.py`):

| Case                                | uncompressed | compressed | speedup |
| ----------------------------------- | ------------ | ---------- | ------- |
| K_3 2-cell path M_2                 | 2.0 s        | 0.001 s    | 1527×   |
| K_3 3-cell path M_2                 | 0.002 s      | 0.002 s    | 1×      |
| K_4 claw M_2 shared                 | 0.008 s      | 0.009 s    | 1×      |
| K_4 claw M_2 disjoint               | 0.022 s      | 0.026 s    | 1×      |
| K_5 claw M_3 shared                 | 0.16 s       | 0.45 s     | 0.35×   |
| K\_{4,4} 3-cell path M_2            | 7.1 s        | 0.07 s     | 100×    |
| K\_{4,4} 3-cell mixed-direction     | 12.5 s       | 4.5 s      | 2.8×    |
| **5-cell K\_{4,4} Cm₃ pattern M_2** | 12.7 s       | 0.3 s      | **50×** |

The K_5 claw 0.35× regression reflects the cost of repeated state
expansion at every junction step (no persistent per-cell positions
after fallback). For larger boundary cells where compression
genuinely helps, the speedup is dramatic (50–1500×).

## Limitations / future work

### Current limitations

- **No engine integration** — used by research scripts only.
- **No `junction_cell_anchor_groups` support** in
  `precompute_M_table` — child must be expanded to uncompressed at
  cell-merge step (loses some compression).
- **5-cell K\_{4,4} M_4 (Cm₃ interior wall) does not complete in 30
  minutes** with per-cell alone. The interior cell triggers
  `keep_shared=True` at the first H child junction step (since other
  H children share the same A-side anchors), expanding state to
  Bell(8) = 4140 partitions. Subsequent steps iterate ~4140 × ~4140
  per junction × 4 children — too slow without further compression
  or hybrid decomposition.

### Future work

1. **`junction_cell_anchor_groups`** — extend
   `precompute_M_table` so child per-cell groups give n_junc factor
   analytically, avoiding child expansion.
2. **Hybrid via cycle close + tree DP** — apply
   `apply_kmatching_formula` to closing junctions in a cyclic
   cell-quotient (e.g., Cm₃ grid), with each leaf computed via this
   tree DP. Tree DP works correctly on tree-topology leaves; the
   chord rule handles cycle closing with bridge-aware coefficients.
3. **Engine integration** — add a dispatch step (e.g., near the
   cell-quotient grid DP at step 7.45, ahead of treewidth DP) that
   dispatches to tree DP when the cell-quotient is a tree. (Note:
   the original cell-quotient cycle DP dispatch step 7.7 was removed
   in the Round 3 cleanup; cycle DP now lives in `tutte/deprecated/`.)

## Implementation

`tutte/roots/cell_quotient_tree.py`:

| Symbol                                                               | Purpose                                                                                                                                                                 |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CellTreeSpec`                                                       | Dataclass: cell template, junction template, cell-tree, anchor groups, junction A/B anchors, root cell.                                                                 |
| `compute_tree_dp_simple(spec)`                                       | Linear-path-only path DP.                                                                                                                                               |
| `compute_tree_dp_recursive(spec, enable_per_cell_compression=False)` | Branching tree DP via post-order recursion. Per-cell compression behind opt-in flag.                                                                                    |
| `_allocate_tree_positions(spec)`                                     | Per-(cell, neighbor) position allocation with shared-anchor dedup.                                                                                                      |
| `_compute_aut_orbits_on_positions(...)`                              | Cell-aut orbit computation with `preserve_anchor_sets`.                                                                                                                 |
| `_initial_cell_groups(...)`                                          | Wraps the orbit computation for initial state setup.                                                                                                                    |
| `_state_groups_after_*`                                              | Per-cell group evolution through junction / cell-merge / marginalize steps with subset removal.                                                                         |
| `_expand_per_cell_state(...)`                                        | Orbit-member enumeration via S_n^N permutations.                                                                                                                        |
| `_marginalize_state(...)`                                            | Standard (uncompressed) marginalization.                                                                                                                                |
| `_marginalize_state_per_cell(...)`                                   | Per-cell-aware marginalization with re-canonicalization.                                                                                                                |
| `compute_corrected_leaf_dp(spec)`                                    | Brute-force leaf DP for chord-rule leaves with cross-cell vertex identifications. Uses corrected `(y-1)^{k-m}/(x-1)^{m-c_comp}` convolution rule (RESOLVED 2026-05-07). |
| `_merge_cells_corrected(...)`                                        | Per-pair convolution applying the corrected formula for shared positions.                                                                                               |
| `build_leaf_graph_from_spec(spec)`                                   | Constructs the implicit leaf graph for ground-truth validation.                                                                                                         |
| `_allocate_positions_with_ids(spec)`                                 | Position allocator that includes `cross_cell_identifications` endpoints (the standard `_allocate_tree_positions` skips them).                                           |

## Cross-cell vertex-identification convolution (Step 3.B.3 — RESOLVED)

For chord-rule leaves of the cycle-close hybrid (`compute_cell_quotient_hybrid`,
now in `tutte/deprecated/cell_quotient_hybrid.py`; its engine dispatch was
removed in the Round 3 cleanup), cells may share vertices via
`cross_cell_identifications` (from the closing junction's contracted edges).
The convolution rule for these merges:

```
T_combined[P_comb] = Σ_{(P_1, P_2) → P_comb}
    T_rooted_1[P_1] · T_rooted_2[P_2] · (y-1)^{k-m} / (x-1)^{m - c_comp}
```

where:

- `k` = number of shared positions (vertex identifications),
- `m` = merge events on shared positions = `|P_1| + |P_2| - |P_comb|`,
- `c_comp` = #components in the full graph being merged (1 for connected
  cells; `k` for an M_k matching template).

When `m - c_comp < 0`, the divisor becomes a `(x-1)^{c_comp - m}` multiplier.

`compute_corrected_leaf_dp` is invoked from
`tutte/deprecated/cell_quotient_hybrid.py:_try_spec_leaf_dispatch` when the
closing junction's removal yields a tree-quotient cell-topology. See
`tutte/research/data/step3_milestone_b_design.md` for derivation
and validation table.

## References

- [6.4 — Cell-Quotient Cycle DP](06_4_cell_quotient_cycle_dp.md) — sister technique for cycle topology.
- [6.5 — Cell-Quotient Grid DP](06_5_cell_quotient_grid_dp.md) — sister technique for grid topology.
- [6.3 — Rooted Tutte Framework](06_3_rooted_tutte_framework.md) — math foundation (rooted Tutte polynomials, vertex-sum convolution).
- [8.3 — k-matching formula](08_3_kmatching_formula.md) — closed-form / chord-rule cycle close primitive that pairs with tree DP for hybrid cyclic decomposition.
- `tutte/research/data/tree_dp_per_cell_compression_findings.md` — full diagnosis of the four-fix breakthrough.
- `tutte/research/data/step3_milestone_b_design.md` — corrected vertex-identification convolution derivation (RESOLVED 2026-05-07).
- `tutte/research/data/hybrid_cycle_close_findings.md` — hybrid cycle-close findings + orbit-aware breakthrough.
