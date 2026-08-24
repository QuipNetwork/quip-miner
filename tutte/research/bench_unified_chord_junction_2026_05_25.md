# Unified Chord-Junction Benchmark — 2026-05-25

Single-process sequential timing of the engine on D-Wave-class targets
after Step 7 (Zephyr + asymmetric chord patterns) shipped. Engine init
loads 333 merger entries from disk (255 K_{4, 4} + 78 Z(1, 1)).

Script:
```
PYTHONUNBUFFERED=1 tutte/.venv/bin/python -m \
    tutte.research.scripts.bench_unified_chord_junction
```

## Results

| Target  | \|V\| | \|E\| | Method                              | Wall time | Δ merger cache |
| ------- | ---- | ---- | ----------------------------------- | --------- | -------------- |
| Cm(1,2) |  16  |  36  | `treewidth_dp`                       | 1.19 s    | +0             |
| Cm(1,3) |  24  |  56  | `treewidth_dp`                       | 0.08 s    | +0             |
| Cm(2)   |  32  |  80  | `cell_quotient_grid_dp_streamed`     | 26.30 s   | +0             |
| Pm(2)   |  40  | 164  | `treewidth_dp`                       | 61.90 s   | +0             |
| Z(1,1)  |  12  |  22  | `treewidth_dp`                       | 0.01 s    | +0             |
| Z(1,2)  |  24  |  76  | `decomposition_chord_peel_atom_inter`| 1.17 s    | +0             |

## Interpretation

**The unified chord-junction fast path did NOT fire on any production
D-Wave target.** Every target hit a more specialized dispatch path
first:

- **Cm(1, 2), Cm(1, 3), Z(1, 1)**: small-graph treewidth short-circuit
  (step 7.55) wins for `|V| ≤ ~24`.
- **Cm(2)**: cell-quotient grid DP (`cell_quotient_grid_dp_streamed`)
  wins for `(rows × cols)` grid topologies.
- **Pm(2)**: direct `treewidth_dp` — Pegasus 40-vertex / 164-edge with
  treewidth ≤ 11 fits the C-ext DP envelope.
- **Z(1, 2)**: decomposition + chord-peel via the atom-inter path
  (`decomposition_chord_peel_atom_inter`) — selected ahead of
  k-matching by the cost predictor.

Cache delta is +0 for all targets — the merger cache wasn't consulted
because none of these dispatch paths route through
`_try_unified_chord_junction`. The chord-junction fast path is wired
into the call sites that would have invoked
`apply_kmatching_formula`, and for current in-scope targets the engine
picks something cheaper before reaching kmatching dispatch.

## Where the fast path DOES fire

`tutte/tests/test_engine_unified_chord_dispatch.py` exercises the dispatch
on synthetic K_{4, 4} cell-pair fixtures:

- `test_fast_path_fires_on_clean_side_a_cell_pair`: symmetric tier wins,
  1 synth call + 15 cache lookups (vs ~16 sub-syntheses in legacy
  k-matching). Polynomial: 226 492 416 spanning trees.
- `test_asymmetric_path_wins_on_mixed_bipartition_anchors`: asymmetric
  tier wins via merger-canonical-key lookup. Same polynomial; no fall-
  back to k-matching.

So the **path is correct and fast on the fixtures it's wired for** —
just not selected by the cost predictor for the current production
D-Wave targets.

## Implications

1. **Step 7 shipped without regressing the existing D-Wave envelope.**
   The Pm(2) 61.9 s is consistent with the pre-Step-7 timings
   (`Pm2 cold 164.6 s → 57.3 s` per `project_pm2_under_60s_shipped`);
   no fast-path overhead is added to dispatches that don't fire.

2. **Real value lands on out-of-scope targets** — Z(m ≥ 2, t) graphs
   that put Z(1, 1) into a boundary-cell role, Pegasus boundary cells,
   user-supplied graphs whose decomposition the engine routes through
   k-matching. None of those are in the current production benchmark
   set; they're the directional next step.

3. **To validate the speedup empirically**, the next benchmark should
   target a graph the engine would route through `apply_kmatching_formula`
   without the unified fast path firing. Easiest is to construct a
   chain `K_{4, 4} + M_4 + K_{4, 4} + M_4 + … + K_{4, 4}` of length
   ≥ 3 (sidesteps the cell-quotient grid DP, sidesteps the small-graph
   short-circuit, hits the k-matching path). Alternative: temporarily
   disable the cost predictor's grid-DP preference and re-run Cm(2).

4. **Cache hit-rate observability**: the engine currently doesn't log
   `lookup_by_source` / `lookup_by_merger` hit counts. Adding a counter
   on `MergerTable` (parallel to the rainbow table's existing counters)
   would make the next benchmark self-documenting.

## Sanity check

All 44 tests in `test_chord_junction_closed_form.py`,
`test_engine_unified_chord_dispatch.py`, `test_merger_lookup.py`, and
`test_unified_chord_junction.py` pass in 6.7 s. Polynomial outputs
match the existing test-fixture oracles exactly.
