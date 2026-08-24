# 6.3 — Rooted Tutte Polynomial: Algebraic Framework

The **rooted Tutte polynomial** is the algebraic foundation underneath
the cell-quotient cycle DP (technique 6.4; now deprecated, was engine
step 7.7) and the cell-quotient grid DP (technique 6.5, engine step
7.45). This document is a _shared theory reference_, not a separate
pipeline step — the engine never dispatches to "rooted Tutte"
directly; it dispatches to the grid DP (and the chain recurrence at
step 7.4) which are built on this framework.

For the productionized algorithms and engine placement see:

- [6.4 — Cell-Quotient Cycle DP](06_4_cell_quotient_cycle_dp.md)
- [6.5 — Cell-Quotient Grid DP](06_5_cell_quotient_grid_dp.md)
- [`tutte/roots/README.md`](../roots/README.md) — module layout

## Definition

For a graph G with marked **boundary set** S ⊆ V(G), the rooted Tutte
polynomial is a function from set-partitions of S to polynomials in
(x, y):

```
T_rooted(G, S)[P] = Σ over spanning subgraphs A of G of:
                    (x − 1)^{r(E) − r(A)} · (y − 1)^{|A| − r(A)}
                   where A's component-partition restricted to S equals P
```

The standard Tutte polynomial collapses the partition index by
summation:

```
T(G; x, y) = Σ_P T_rooted(G, S)[P]
```

So `T_rooted` carries strictly more information than `T`: it knows
_which boundary vertices end up in the same connected component_ in
each spanning subgraph. That extra information is exactly what makes
**vertex-sum composition** possible without the matroid bookkeeping
that the (now retired) Bonin-de Mier framework required.

## Vertex-sum convolution

Given two graphs G_1, G_2 sharing a boundary S (no shared cell-edges
between identified vertices), the rooted Tutte of the vertex-sum
factors via convolution over partition pairs:

```
T(G_1 ⊕_S G_2) = (x − 1)^{−(|S| − c_J(S))} · Σ_{(P_1, P_2)} of:
                  T_rooted(G_1, S)[P_1] · T_rooted(G_2, S)[P_2] · ((x − 1)(y − 1))^{Δ(P_1, P_2)}
```

where:

- `Δ(P_1, P_2) = |JOIN(P_1, P_2)| + |S| − |P_1| − |P_2|` measures the
  number of components that merge across the shared boundary,
- `c_J(S)` is the number of components of G_2 (the "junction") that
  touch the shared boundary S — for connected junctions this is 1
  (the standard divisor `(x − 1)^{|S| − 1}`); for disconnected
  matching junctions M_k, `c_J = k` (the corrected divisor
  `(x − 1)^{|S| − k}`).

The accumulated sum is divisible by `(x − 1)^{|S| − c_J(S)}` —
`tutte/roots/rooted_tutte.py:divide_by_x_minus_1_power` performs the
synthetic division.

## Path DP — composing N cells in a chain

For G = `cell_0 ⊕_J0 cell_1 ⊕_J1 ... ⊕_J_{n-2} cell_{n-1}`, the
**partial vertex-sum** convolves cells one junction at a time, keeping
the open boundary on the _unprocessed_ side:

```
T_rooted(cell_0 ⊕ ... ⊕ cell_k, C_k)[P_C] =
    Σ_{(P_state, P_cell consistent with P_C)} of:
        T_rooted_state[P_state] · T_rooted_cell[P_cell] · ((x − 1)(y − 1))^{Δ(P_state|S, P_cell|S)}
    / (x − 1)^{|S| − c_J(S)}    (deferred to chain end)
```

The accumulated `(x − 1)` divisor is tracked through the chain and
applied as a single `divide_by_x_minus_1_power` at the end.

## Cycle close — identification formula

Closing a cycle requires identifying boundary vertices across the
last junction. The **chain-aware** identification formula:

```
T(cycle) = (x − 1)^{−a} · Σ_P ((x − 1)(y − 1))^{actually_same(P)} · T_rooted_intermediate[P]
```

where `actually_same(P) = a − n_merges(P)` is computed by union-find:
initialize parent map from P's blocks, then sequentially apply
identifications `state_left[i] ≡ pos_cB_FRESH[i]` for `i = 0, ..., a − 1`.
`n_merges` counts how many of those identifications actually unioned
two distinct blocks. The naive `count_same(P)` (blocks with both
endpoints already in the same block) over-counts when identifications
chain through blocks; the union-find captures the correct merges.

## Boundary-label discipline

For a chain of n cells and n − 1 junctions, there are 2(n − 1) distinct
boundary positions — junction j has two sides (A and B) at positions
2j and 2j+1. **Each position must use a disjoint integer label set**
(e.g., `[10000 * (i + 1) + k]` for position i, anchor k). Reusing
labels between positions silently collapses adjacent boundaries and
corrupts the result — this was a class of bugs encountered during
prototyping.

## Cost model

For one junction step at shared boundary S with cell template C:

- `Bell(|S|)` partitions per side; `Bell(|S|)²` partition-pair
  iterations in `M_precompute`.
- After `Aut(C)`-orbit compression: `|Aut(C)|`-fold reduction. K\_{4,4}
  has 1152 automorphisms, compressing 4140 partitions of an 8-vertex
  boundary to 43 orbits.
- `orbit_convolve` does one polynomial multiply per `(O_state, O_junc,
O_out)` triple. The C extension (`tutte/_polynomial_c.py`) accelerates
  these multiplies by 1.3–12.8× depending on polynomial size.

For an n-cell cycle: `n` such steps, plus the final close. Wall-clock
on real Cm2 (n = 4, K\_{4,4} cells, M_4 junctions): ~50 s optimization
(vs ~49 s for the kmatching closed-form
shortcut — parity, with the cycle DP positioned to win when the
formula doesn't apply).

## Status and limits

- **Cycle topology with disjoint per-cell anchors** — implemented
  in `tutte/deprecated/cell_quotient_cycle.py` (deprecated; its engine
  dispatch step 7.7 was removed in the Round 3 cleanup). Validated on
  real Cm2 against `kmatching_formula` and Kirchhoff.
- **Grid topology with disjoint per-cell anchors** — productionized in
  `tutte/roots/cell_quotient_grid.py`. Validated on synthetic K_n
  grids; engine integration pending the anchor-sharing-aware adapter
  (next item).
- **Anchor sharing across junctions** (Cm₃ interior cells) — algorithm
  is sound, the missing piece is the cell-anchor adapter that
  recognizes when the same vertex set serves multiple junctions on a
  cell.
- **Scaling wall** — a Bell(W)² × poly*size² cost per junction step.
  Even with C-extension polynomial mul, real Pm₃ / Z(1,3) targets
  remain out of reach without further work (generic Aut
  compression beyond K*{4,4}, balanced-tree composition,
  or multivariate Tutte representation).

## Files

| File                                   | Purpose                                                                                                  |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `tutte/roots/rooted_tutte.py`          | Brute-force `T_rooted` + boundary primitives (`join`, `delta`, `restrict`, `divide_by_x_minus_1_power`)  |
| `tutte/roots/aut_orbit.py`             | Aut-based orbit canonicalizer (generic over any cell with non-trivial automorphism)                      |
| `tutte/roots/cell_quotient_helpers.py` | Hot-path `precompute_M_table` + `orbit_convolve` + `enumerate_partitions_cached` + `components_touching` |
| `tutte/deprecated/cell_quotient_cycle.py` | `compute_cycle_dp` (technique 6.4; deprecated, engine dispatch step 7.7 removed)                      |
| `tutte/roots/cell_quotient_path.py`    | `compute_path_dp` — cycle DP minus the close step (consumed by grid DP)                                  |
| `tutte/roots/cell_quotient_grid.py`    | `compute_grid_dp_with_layout` — row-by-row composition via path DP + vertical convolution                |
| `tutte/roots/cell_anchor_adapter.py`   | `normalize_cell_anchors_for_cycle` — graph-agnostic cycle detection + per-cell anchor alignment          |

## References

- Brylawski, T. (1971). "A combinatorial model for series-parallel
  networks." Original 2-clique-sum Tutte formula.
- Welsh, D. (1976). _Matroid Theory_. Boundary-partition / rank
  polynomial framework.
- Bollobás, B. (1998). _Modern Graph Theory_. Chapter on graph
  polynomials.
- Sokal, A. D. (2005). "The multivariate Tutte polynomial..."
  arXiv:math/0503607. Series/parallel reduction in the multivariate
  setting.
