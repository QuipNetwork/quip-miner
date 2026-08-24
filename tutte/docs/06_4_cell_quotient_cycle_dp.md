# 6.4 — Cell-Quotient Cycle Dynamic Programming (deprecated)

## Summary

For graphs whose hierarchical decomposition has **cell-quotient
topology of a simple cycle** (e.g., D-Wave Cm₂'s 4-cycle of K\_{4,4}
cells), this technique computes T(graph) by composing per-cell
**rooted Tutte polynomials** through vertex-sum convolution and a final
identification step.

The mathematical foundation is the rooted Tutte polynomial framework
documented in [6.3 — Rooted Tutte Polynomial: Algebraic
Framework](06_3_rooted_tutte_framework.md). This document focuses on
the productionized engine path.

**Complexity per junction step:** `O(Bell(|S|)² × poly_size²)` before
optimization; with Aut-orbit compression and the C-extension
polynomial multiply, real Cm₂ (4-cell K\_{4,4} cycle, M_4 junctions)
takes ~50 s cold.

## When it is used

**Deprecated** — the cycle-DP code now lives in
`tutte/deprecated/cell_quotient_cycle.py` and its engine dispatch
(formerly step 7.7) was removed in the Round 3 cleanup. The
description below documents the original dispatch contract for
historical reference.

The original engine dispatch fired in `SynthesisEngine._synthesize_inner`
when:

1. `graph.edge_count() >= 60` (matches the formula-shortcut gate).
2. `try_hierarchical_partition(graph, table)` returns a valid cell
   decomposition.
3. The cell-quotient graph (cells as nodes, junctions as edges) is a
   **simple cycle** — every cell has degree 2 in the quotient.
4. `cell_anchor_adapter.normalize_cell_anchors_for_cycle` succeeds at
   aligning per-cell anchors to a single canonical template.

If any precondition failed, the entry returned `None` and the engine
fell through to treewidth DP.

The technique was positioned _after_ the formula shortcut (step 7.5)
and almost-cograph (step 7.6) — both of those handle Cm₂-shape
graphs faster when applicable. Cell-quotient cycle DP caught the
cases they don't.

## Algorithm — three phases

```
def compute_cycle_dp(cell_template, cell_left, cell_right,
                    junction_template, junction_A, junction_B, n_cells):
    # 1. Cache per-template T_rooted (cell + junction).
    T_cell = t_rooted_cached(cell_template, cell_left + cell_right)
    T_junction = t_rooted_cached(junction_template, junction_A + junction_B)

    # 2. Path DP through n−1 junction-cell pairs.
    state_orbit_T = aut_compress(T_cell)
    total_div = 0
    for k in range(n_cells - 1):
        state_orbit_T = orbit_convolve(state_orbit_T, T_junction, M_junction)
        total_div += b - c_J(junction_template, junction_A)
        state_orbit_T = orbit_convolve(state_orbit_T, T_cell, M_cell)
        total_div += a - c_J(cell_template, cell_left)

    # 3a. Cycle close, step 1 — convolve state with closing junction
    #     to a fresh boundary pos_cB_FRESH.
    state = orbit_convolve(state_orbit_T, T_closing_junction, M_close)
    total_div += b - c_J(junction_template, junction_A)

    # 3b. Cycle close, step 2 — identification formula.
    #     T(cycle) = (x-1)^{-a} · Σ_P ((x-1)(y-1))^{actually_same(P)} · state[P]
    #     where actually_same(P) = a − n_merges_via_unionfind(P, identifications)
    T_total = TuttePolynomial.zero()
    for P, val in state.items():
        actually_same = a - count_merges(P, [(state_left[i], pos_cB_FRESH[i]) for i in range(a)])
        T_total += (xy_minus_1)^actually_same * val
    total_div += a

    return divide_by_x_minus_1_power(T_total, total_div)
```

## Key correctness notes

### Disconnected-junction divisor `c_J`

The standard vertex-sum divisor is `(x − 1)^{|S| − 1}` per junction
step, which assumes the junction is **connected**. For matching
junctions (M_k = k disjoint edges), the correct divisor is
`(x − 1)^{|S| − c_J(S)}` where `c_J(S)` is the number of junction
components touching the shared boundary. `components_touching` (in
`tutte/roots/cell_quotient_helpers.py`) detects this automatically;
the cycle DP uses the corrected formula.

### Identification formula `actually_same(P)`

Closing the cycle identifies `state_left[i] ≡ pos_cB_FRESH[i]` for
each `i = 0, ..., a − 1`. The naive formula uses
`count_same(P)` (number of identifications where both endpoints lie
in the same block of P) — but this **over-counts when identifications
chain through blocks** for `a > 1`. The correct quantity is
`actually_same(P) = a − n_merges` computed via union-find: initialize
parent map from P's blocks, sequentially apply identifications,
count how many actually unioned distinct blocks.

## Performance optimizations

| Optimization                   | Mechanism                                                           | Speedup                   |
| ------------------------------ | ------------------------------------------------------------------- | ------------------------- |
| Aut orbit compression          | `aut_orbit.canonical_partition` collapses Aut-equivalent partitions | K\_{4,4}: 4140 → 43 (96×) |
| Orbit-level `M_precompute`     | Pick rep_state ∈ O_state, multiply by orbit size                    | ~143× on K\_{4,4} cycles  |
| Position-invariant cache       | `enumerate_partitions_cached` caches orbits for canonical positions | 7×                        |
| Raw-dict polynomial arithmetic | Inner loops accumulate `dict[(xpow, ypow)] -> coeff` directly       | 3-5×                      |
| C-extension polynomial mul     | `tutte/_polynomial_c.poly_mul` via cffi                             | 1.3-12.8× per mul         |

Combined effect on Cm₂: ~496 s → post ~50 s
(~10× total). Now at parity with the kmatching formula closed-form
shortcut on Cm₂ (~49 s).

## Where it wins (and doesn't)

- **Wins on**: cycle-topology cell-decomposable graphs that the
  formula shortcut can't handle (because the inter-cell structure
  doesn't match unified or k-matching preconditions). Sparse
  research targets where chord-rule recursion would be too deep.
- **Ties on**: Cm₂ — the formula shortcut already gives a closed
  form. Cell-quotient DP runs as the fallback if formula fires
  return `None`.
- **Doesn't apply to**: Cm₃ and larger — the cell-quotient is a
  _grid_, not a cycle. Grid topology is handled by the sibling
  technique [6.5 — Cell-Quotient Grid DP](06_5_cell_quotient_grid_dp.md).

## Files

| File                                                                        | Purpose                                                                                      |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [`tutte/deprecated/cell_quotient_cycle.py`](../deprecated/cell_quotient_cycle.py) | `compute_cycle_dp(...)` — the three-phase DP (now in `tutte/deprecated/`; engine dispatch removed) |
| [`tutte/roots/cell_anchor_adapter.py`](../roots/cell_anchor_adapter.py)     | `normalize_cell_anchors_for_cycle` — graph-agnostic cycle detection + alignment              |
| [`tutte/roots/cell_quotient_helpers.py`](../roots/cell_quotient_helpers.py) | `precompute_M_table`, `orbit_convolve`, `enumerate_partitions_cached`, `components_touching` |
| [`tutte/roots/aut_orbit.py`](../roots/aut_orbit.py)                         | Aut-based orbit canonicalizer                                                                |
| [`tutte/roots/rooted_tutte.py`](../roots/rooted_tutte.py)                   | Brute-force `T_rooted` + boundary primitives                                                 |
| [`tutte/_polynomial_c.py`](../_polynomial_c.py)                             | C-extension polynomial multiply (cffi)                                                       |

## References

- [6.3 — Rooted Tutte Polynomial: Algebraic Framework](06_3_rooted_tutte_framework.md)
  for the math.
- [6.5 — Cell-Quotient Grid DP](06_5_cell_quotient_grid_dp.md) for
  the grid generalization.
- [`tutte/roots/README.md`](../roots/README.md) for the package
  overview.
