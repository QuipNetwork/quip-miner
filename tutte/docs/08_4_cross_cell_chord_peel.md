# 8.4 Cross-Cell Chord-Peel

> **MERGED** (May 23, 2026): the standalone `_try_cross_cell_chord_peel`
> and `_try_clique_atom_chord_peel` dispatchers have been folded into
> `_try_decomposition_chord_peel`
> ([8.5 Decomposition + Chord-Peel — Unified Dispatcher](08_5_decomposition_chord_peel.md)).
> This document remains as the conceptual reference for why peeling
> small inter-atom junctions is faster than peeling internal-clique
> edges. The legacy methods are still on disk behind
> `TUTTE_USE_LEGACY_DISPATCH=1` for debugging.

Peel the **smallest connected inter-atom junction** edge set, not internal
clique edges, when a graph decomposes into disjoint named-family atoms
(K_n, K_{a,b}, …) connected by small bipartite junctions.

## Motivation

When a graph G has two disjoint dense atoms A_1, A_2 connected by a small
bipartite junction J, applying the chord rule to J is much faster than
applying it to internal-atom edges:

* The chord rule's wall is dominated by its per-step sub-synth on a
  contraction multigraph. That cost is **roughly constant** for
  same-density intermediates (~6–8 s for a 23-vertex graph at
  treewidth ~10).
* **Fewer chord edges → fewer sub-syntheses.** A 4-edge junction costs
  ~30 s; a 12-edge internal clique-peel costs ~95 s on the same graph.
* `g_chord_free` (G minus J) still contains the dense atoms, so the
  recursive synth can find more structure (cell-quotient, recursive
  cross-cell, treewidth_dp on the smaller residual).

## Empirical headline

**Z(1,2)**: 2 K_4 atoms with two K_{2,2} junctions of 4 edges each.

| chord set                 | wall   | ratio vs baseline |
| ------------------------- | ------ | ----------------- |
| 12 internal K_4 edges     | ~95 s  | 1.45×             |
| 4 inter-atom edges (K_{2,2}) | **~47 s**  | **~2.94×**        |
| treewidth_dp baseline     | ~138 s | 1.00×             |

The 4-edge cross-cell peel reliably puts Z(1,2) under 60 s on clean
state (cron goal achieved May 19, 2026).

## Algorithm

```
def cross_cell_chord_peel(G):
    atoms = find_disjoint_atoms(G)               # K_n preferred, K_{a,b} fallback
    if len(atoms) < 2: return None
    J = find_smallest_junction(G, atoms)         # smallest connected bipartite block
    if J is None or len(J) > max_junction_size: return None
    sigma = find_best_sigma(G, require_free=True)
    g_cf, factors, adds = iterative_chord_rule(G, J, sigma=sigma)
    t_cf = engine.synthesize(g_cf)               # recursion picks up cell-quotient,
                                                 # treewidth_dp, etc. on the residual
    return combine_chord_iteration(t_cf, factors, adds)
```

### Atom families supported

Implemented in `tutte/graphs/atom_detection.py`. Preference order
(unified entry `find_disjoint_atoms`):

1. **K_n cliques** (`K_3` … `K_6`). Enumerated via `nx.find_cliques`,
   then greedy-disjoint selection. For a parent maximal clique of size
   m > k, emits a single canonical sub-K_k.

2. **K_{a,b} complete bipartite** (`K_{2,2}` … `K_{4,4}`). Enumerated
   via degree-pruned neighborhood intersection — for each candidate
   a-vertex set, intersect their neighborhoods and check for a b-set
   in the intersection. Faster than full VF2 for small (a, b).

3. **B_n books** (`B_2` … `B_5`). B_n = n triangles sharing a common
   edge: vertex set `{u, v, p_1, …, p_n}` with shared edge (u, v) and
   each p_i adjacent to both u and v. Detection: for each edge (u, v),
   take common neighbors of u and v; if ≥n exist, pick lex-smallest n.
   Useful for triangulated graphs with fan-like substructure.

4. **W_n wheels** (`W_5` … `W_8`). W_n = hub vertex u + cycle C_n on
   n neighbors of u. Detection: for each high-degree vertex, check if
   a rim-sized neighbor subset induces a single cycle (2-regular and
   exactly rim_size edges). `min_rim = 5` avoids overlap with K_4 (W_3)
   and other lower-tier matches.

The dispatch returns the FIRST tier that yields ≥2 disjoint atoms.
This ordering keeps the densest available atom for chord-peel
(maximises the "atom intact in g_chord_free" benefit) without losing
applicability to sparser graphs (books, wheels are fallbacks).

### Junction analysis

For each atom pair (A_i, A_j), gather inter-atom edges (one endpoint
in A_i, other in A_j). Split by **connected component** of the
bipartite-edge subgraph — picks the smallest connected block, since
two atoms may be joined by multiple independent bipartite components
(e.g. Z(1,2)'s 8-edge junction is two K_{2,2} blocks of 4 edges each).

## Generality

The principle applies to any graph with detectable disjoint dense
atoms connected by a small bipartite junction:

* **D-Wave**: Z(m, t) (K_4 atoms), Cm (K_{4,4} cells), Pegasus (K_4
  + K_{4,4} hybrid).
* **Triangulated planar**: K_3 atoms with shared-edge junctions.
* **Bipartite data graphs**: K_{a,b} atoms in recommendation, knowledge,
  and citation graphs.
* **Random graphs**: only when atom-detection finds ≥2 disjoint
  occurrences with a small junction (rare for sparse random graphs,
  common for dense or block-structured ones).

## Engine wiring

The cross-cell peel runs at engine step 7.88, between cell-quotient
hybrid (7.85) and clique-atom chord-peel (7.9). As noted in the banner
above, the standalone `_try_cross_cell_chord_peel` method has been folded
into `SynthesisEngine._try_decomposition_chord_peel`; the legacy method
is still on disk behind `TUTTE_USE_LEGACY_DISPATCH=1`.

### Gating

* `edge_count >= 60` — small graphs are handled faster by
  treewidth_dp / cell-quotient.
* `node_count <= 30` — chord rule's per-step cost grows with graph
  size; above ~30 nodes the residual synth dominates. Larger graphs
  fall through to clique-atom-chord-peel (which has its own cost-aware
  gate) and treewidth_dp.
* `self._synth_depth == 1` — top-level synthesis only. Recursive
  invocations on `g_chord_free` would otherwise cascade, peeling
  the *next* small junction on each layer (Z(1,2) measured 83 s
  cascaded vs 47 s gated).
* `max_junction_size = 6` — junctions larger than 6 edges are
  unlikely to beat treewidth_dp's per-step cost.

### Hybrid depth tracking (historical)

> The separate `HybridSynthesisEngine` was removed in the Round 3 cleanup
> (there is now one `SynthesisEngine`); this note is retained for the
> depth-tracking lesson it records.

`HybridSynthesisEngine` called `engine._try_cross_cell_chord_peel`
directly rather than going through `engine.synthesize`. The engine's
own `_synth_depth` therefore didn't auto-increment. Hybrid had to
**manually bump** `engine._synth_depth += 1` around the call so the
engine's recursive-cascade gate (`_synth_depth == 1`) fired. Without
this fix, hybrid measured 77 s vs engine 47 s on Z(1,2).

## When it doesn't fire

* **No K_n / K_{a,b} / B_n / W_n atoms**: planar series-parallel
  graphs, expanders, random graphs of low density, isolated trees
  with no clique/book/wheel substructure. Returns `None`, falls
  through to other dispatches.
* **Junction too large**: when the smallest connected bipartite
  block between any atom pair exceeds 6 edges. Returns `None`,
  falls through to clique-atom-chord-peel.
* **Recursion depth > 1**: prevents cascading; recursive g_chord_free
  synthesis routes through the rest of the dispatch pipeline.
* **n > 30 nodes**: chord rule's per-step cost grows with graph
  size; above the cap, treewidth_dp / cell_quotient handle better.

## Why not other named families?

| Family | Why not added |
|--------|---------------|
| Stars K_{1,n} | Covered by K_{a,b} with a=1; star edges are bridges, so chord-peeling trivially reduces (factor x) — no junction structure to exploit. |
| Cycles C_n | Handled by series-parallel recognition (`is_series_parallel`), not by chord-peel. |
| Paths P_n | SP-handled; would be useless as atoms (only one bridge edge). |
| Prisms Y_n / Möbius ladders M_n | Rare in practice; chord-rule cost on a single prism is small enough that direct treewidth_dp suffices. |
| Hypercube Q_n | Niche (3D lattice graphs); detection cost rises sharply with n. Could be added on demand. |
| Friendship F_n / Windmill | Specialised; rare matches in practice. |
| Petersen, Heawood, K_{3,3,3} | Better handled as direct rainbow-table lookups, not chord-peel atoms. |

## Related techniques

* [Chord-rule formalization](08_2_chord_rule_formalization.md) —
  underlying chord rule and its correctness.
* [Clique-atom chord-peel](07_k_sum_decomposition.md) — older path
  that peels internal-clique edges; remains as fallback (step 7.9).
* [Cell-quotient bipartite-junction DP](06_3_rooted_tutte_framework.md) —
  alternative bipartite-junction approach using per-cell T_rooted
  caching.

## Empirical scope (May 2026)

* **Z(1,2)** (24 n, 76 e): SHIPPED — reliably <60 s cold.
* **Z(1,1)** (12 n, 22 e): doesn't fire (n=12 below 60-edge gate);
  treewidth_dp handles in 1.8 s.
* **Cm_2** (32 n, 80 e): doesn't fire (n=32 above 30-node cap);
  cell_quotient_grid_dp_streamed handles in 36 s.
* **Z(1,3)** (36 n, 162 e): doesn't fire (n=36 above 30-node cap);
  open. Atom detection finds 3 K_4 or 3 K_{4,4} with 4-edge
  junctions, but the per-step cost on a 35 n × 161 e contraction
  multigraph is high enough that even a 4-edge chord set ≫ 60 s.
* **Pm_2** (40 n, 164 e): doesn't fire (n=40 above 30-node cap);
  treewidth_dp handles cold in ~173 s (variable with system load).

The 30-node cap is a measured bound, not a fundamental limit —
relaxing it requires faster per-step contraction synthesis (the
research-grade work is in chain-recurrence / signed-equivariant
paths, see `06_7_chain_recurrence_algebra.md` and `06_9_signed_equivariant_dp.md`).
