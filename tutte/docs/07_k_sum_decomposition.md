# 7. k-Sum Decomposition (Chord Rule)

## Summary

When a graph has a **k-vertex separator** `S` whose removal disconnects the graph, we can decompose it as a true k-sum and compute the Tutte polynomial via **chord recursion** on the shared K_k clique edges. Implemented in `tutte/graphs/k_sum.py:clique_chord_k_sum`. This replaces the retired matroid-theoretic Theorem 10 (Brylawski / Bonin-de Mier).

## When It's Used

**Step 7** in the synthesis pipeline. Runs after treewidth_dp (step 6) returns `None` (treewidth too high) or doesn't apply. Gates:

- Graph has **≥ 6 edges**
- A k-vertex separator exists for some `k ∈ {2, ..., 7}` (found via `_try_ksum_decomposition` in `engine.py`)
- The K_k clique edges among the separator are either present or deleted in the target — both cases work; partial-deletion subsets are also handled

If no separator is found, the engine falls through to hierarchical tiling (step 8).

## Algorithm

For a target graph with separator `S = {s_1, ..., s_k}` and `missing_edges` = the K_k clique edges deleted from the target:

1. Build the **parallel connection** `PC = target ∪ missing_edges` — restoring all K_k edges.
2. Apply the **iterative chord rule** (`_iterative_chord_rule`) to the missing clique edges:
   ```
   T(PC) = T(PC − e_1) + T(PC / e_1)
   T(PC − e_1) = T(PC − e_1 − e_2) + T((PC − e_1) / e_2)
   ...
   ```
3. After peeling all `|missing_edges|` clique edges:
   ```
   T(target) = T(PC) − Σ_{i=1}^{|missing_edges|} T((PC − e_1 − ... − e_{i-1}) / e_i)
   ```

The chord-free residual `PC − all missing_edges` equals the original target (by construction). Each contraction leaf is synthesized as a `MultiGraph` (parallel edges and loops are preserved by `MultiGraph.merge_nodes()`).

## Cost

`1 + |missing_edges|` full syntheses. For a **classic k-sum** (all `C(k, 2)` clique edges deleted):

| k | syntheses | brute-force Theorem 10 | flat-grouped Theorem 6 |
|---:|---:|---:|---:|
| 2 | 2 | 2 | 2 |
| 3 | 4 | 8 | 5 |
| 4 | 7 | 64 | 12 |
| 5 | 11 | 1024 | 52 |

Chord recursion is uniformly the fewest syntheses, with no matroid infrastructure (no flat lattices, no Möbius function, no inclusion-exclusion bookkeeping).

## Implementation

- `tutte/graphs/k_sum.py:clique_chord_k_sum` — public API.
- `tutte/synthesis/engine.py:_try_ksum_decomposition` — pipeline integration; finds vertex separators and dispatches.
- `tutte/synthesis/engine.py:_apply_ksum` — calls `clique_chord_k_sum` with the right separator + missing-edges arguments.

## Why This Is Step 7 (After Treewidth DP)

When treewidth_dp succeeds in C-extension time (typically under a second for `tw ≤ 11`), it's the fastest path. For graphs whose treewidth exceeds the cap, chord recursion is essential — but for most of the test corpus, treewidth_dp wins outright. Putting k-sum after treewidth_dp lets the cheap C path handle most cases and reserves the chord rule for genuinely large graphs.

## Limitations

- **Separator search cost**: finding a k-vertex separator takes O(C(n, k)) in the worst case. The engine bounds the search and gives up when no separator is found within reasonable time.
- **Degenerate K_k ⊕_k K_k case**: when both cells exactly equal the shared K_k clique (so the target is the empty graph on k vertices), both chord recursion and the historical Theorem 10 return wrong polynomials. Easy fix: short-circuit when the chord-recursion residual has zero edges. Not yet implemented.

## See Also

- [08_2_chord_rule_formalization.md](08_2_chord_rule_formalization.md) — formalization, empirical validation, comparison to Bonin-de Mier.
- [08_hierarchical_tiling.md](08_hierarchical_tiling.md) — sister technique for disjoint-cell decompositions.
