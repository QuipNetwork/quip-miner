# 6.8 — Modular Arithmetic Pathways

## Summary

For cell-decomposable graphs where the polynomial Tutte coefficients
would overflow `int64` mid-DP (e.g., D-Wave Cm₂'s coefficients reach
~10¹⁸), we use **modular point-value DPs**: evaluate the Tutte
polynomial at integer `(x_0, y_0)` modulo a prime `p`, then recover
the full polynomial via **Lagrange interpolation across the grid +
Chinese Remainder Theorem across multiple primes**.

It is the
**primary path for graphs that exceed the polynomial-DP coefficient
budget** — currently `Cm₂` (validated) and `Cm₃` (modular point
evaluation feasible; full polynomial pending C-side aggregation
and pair-orbit decomposition).

## When it is used

Currently dispatched via:

- `chimera_modular_dp` (`tutte/research/scripts/cm3_via_modular_dp.py`)
  — Cm₂ and (in principle) Cm₃
- `compute_tree_dp_simple_mod` (`tutte/roots/cell_quotient_tree.py`) —
  generic linear-path cell-tree-quotient graphs with cells + M_k
  junctions

Dispatch criteria:

- `graph.edge_count() ≥ 60` (matches other modern engine paths)
- Cell-decomposition succeeds (`try_hierarchical_partition`)
- Coefficient overflow risk (heuristic: cell count × cell vertex
  count > some threshold; not yet automated)

## Why modular?

The Tutte polynomial `T(G; x, y)` has integer coefficients, but for
moderately sized D-Wave graphs the **maximum coefficient magnitude
grows exponentially in the chord count**. For Cm₂ (32 vertices, 96
edges):

```
max coefficient(T(Cm₂)) ≈ 10¹⁸
```

This exceeds `int64` (max ~9 × 10¹⁸ unsigned). Intermediate state
polynomials during DP can be larger still (the final answer is the
sum/difference of bigger intermediates). Python `int` is arbitrary-
precision but slow; the C extensions use `int64`.

**Workaround**: compute `T(G; x_0, y_0) mod p` for many primes `p`
and many integer points `(x_0, y_0)`, then reconstruct.

## The pathway

### Step 1 — Point evaluation modulo a prime

For each `(x_0, y_0, p)`:

```python
T_mod = compute_modular_dp(graph, x_0, y_0, p)
```

The DP runs in `Z/pZ` arithmetic throughout. No polynomial allocation;
intermediate state is `Dict[partition_key, int]`. C extension
(`precompute_M_table_mod`, `precompute_and_convolve_c_mod`) handles
the inner loop in single-pass mod-p hashmap aggregation.

Chimera ships the modular path end-to-end. The C-extension dispatcher
trims per-chunk overhead via orbit-size early-termination and a
bytes-keyed state map.

### Step 2 — Lagrange interpolation across the grid

For each prime `p`, evaluate `T_mod` at a grid of `(x_0, y_0)` integer
points sufficient to determine the polynomial `T(G; x, y) mod p`.
The grid needs at least `(d_x + 1)(d_y + 1)` points where
`(d_x, d_y)` is the polynomial's bidegree. For Cm₂: bidegree (31, 18) →
need ~600 grid points minimum (with verification: 700+).

Implementation: `tutte/deprecated/interpolation.py`

### Step 3 — CRT reconstruction across primes

Each prime gives `T(G; x, y) mod p_i`. CRT combines them into the
true integer-coefficient polynomial:

```
T(G; x, y) mod (p_1 · p_2 · ... · p_k) = T(G; x, y)  (once product exceeds max coeff)
```

For Cm₂: 3-4 50-bit primes suffice. For Cm₃: 6-8 50-bit primes
projected.

## Per-target cost

### Cm₂ (validated)

- Per `(x_0, y_0, p)` point: ~14 s on a current laptop
- Grid size: ~700 points
- Number of primes: 3
- Total: ~700 × 3 × 14 s ≈ **8 hours** with multiprocessing
  (`cm_modular_interp.py` parallelizes across primes)

### Cm₃ (feasibility-only)

- Per `(x_0, y_0, p)` point: ≥75 min on the pure-Python path
- Projected after pair-orbit + multiprocessing optimisations: ~2-5 min per point
- Grid: ~1500-2000 points × 4-6 primes ≈ ~10 000 points total
- Total projected: **~3-4 days** wall-clock with full optimization

This is **practical for one-off computations** but not interactive.

## When NOT to use modular

For graphs whose Tutte coefficients fit `int64` (most graphs ≤ 50
edges), the standard polynomial DP is faster:

- No per-point repetition
- Single pass produces full polynomial
- C extensions in `tutte/graphs/_treewidth_c.py` are well-optimized

Roughly: if `edge_count ≤ 50`, prefer polynomial DP. Above 60-80
edges (D-Wave regime), modular is the only viable path for the cell-
quotient cycle / grid DPs.

## Open direction — aut-compressed junction T_rooted

The current `compute_tree_dp_simple_mod` builds the junction T_rooted
via `t_rooted_cached(junction_template, anchor_list)` which enumerates
**Bell(boundary_size) partitions**. For Z(1,2)'s 32-edge bipartite
junction with 8 anchors per side: Bell(16) ≈ 10 billion partitions —
infeasible.

An `S_a × S_b` aut compression on the junction's boundary partitions
would cut the orbit count to roughly 2000 (a ~5×10⁶ reduction).
Design notes lived in
`tutte/research/plans/aut_compressed_junction_dp_design.md` (**LOST** —
never committed, died with the authoring machine; see bead riff-ggi).

## Files

- `tutte/polynomial.py:evaluate_mod` — single-point modular evaluation of a TuttePolynomial
- `tutte/deprecated/interpolation.py` — 1D and 2D Lagrange + CRT combine
- `tutte/roots/cell_quotient_helpers.py:precompute_M_and_convolve_streaming_mod` — modular streaming convolve
- `tutte/roots/_partition_c.py:precompute_M_batched_inner_c_mod` — C-ext dispatch
- `tutte/research/scripts/cm_modular_interp.py` — multiprocessing wrapper for Cm₂/Cm₃ (**LOST**, see riff-ggi)

## See also

- [Engine workflow primer Part IV — modular point-value pathways](../research/engine_workflow_primer.md)
  — first-principles entry into where this pathway sits in the engine
  cascade.
- [6.5 Cell-Quotient Grid DP](06_5_cell_quotient_grid_dp.md) — the
  narrative for the grid-DP modular variant.
- [6.7 Chain & Cycle Recurrence Algebra](06_7_chain_recurrence_algebra.md)
  — modular evaluation extended to algebraic recurrences (chain
  transfer matrix + Faddeev-LeVerrier mod p).
- [6.9 Signed-graph DP and σ-equivariant decomposition](06_9_signed_equivariant_dp.md)
  — uses `interpolate_t_signed_mod` (in `signed_quotient.py`) for
  bivariate Lagrange recovery on signed graphs.
