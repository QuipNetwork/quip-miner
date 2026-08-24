# 5. Hierarchical Tiling — Chord Rule

> **MERGED** (May 23, 2026): the standalone `_try_hierarchical` /
> `_synthesize_hierarchical` dispatcher (legacy engine step 10) has
> been folded into `_try_decomposition_chord_peel`
> ([8.5 Decomposition + Chord-Peel — Unified Dispatcher](08_5_decomposition_chord_peel.md))
> at engine step 7.88. Cell-only closed-form formulas
> (`unified_formula`, `kmatching_formula`) are preserved verbatim in
> Phase B of the new dispatcher. The internal `treewidth_dp` and
> `product_formula` fallbacks were removed (the chord-rule path
> subsumes them correctly). Legacy method still on disk behind
> `TUTTE_USE_LEGACY_DISPATCH=1` for debugging.

## Summary

For graphs with a **repeating cell structure**, the engine partitions the input into k disjoint copies of a known cell `C` (from the rainbow table) plus a multigraph of inter-cell edges. The Tutte polynomial is then computed via the **chord rule** (deletion-contraction applied iteratively to the inter-cell chord edges) plus a **boundary-quotient formula** for the chord-free residual. Both pieces are implemented in `tutte/graphs/k_sum.py`.

## When It's Used

This is **step 5** in the synthesis pipeline (now **before** treewidth_dp; see [README.md](README.md) for the full ordering). By the time a graph reaches step 5 it has already failed:
- Rainbow table lookup
- Base cases
- Disconnected factorization
- Cut vertex factorization

Additional gates for hierarchical tiling:
- Graph has **≥ 20 edges** (smaller graphs go to treewidth_dp instead)
- A cell partition exists (`try_hierarchical_partition` from `graphs/covering.py` returns non-None)
- The cell has **non-trivial structure** (not a tree/forest — `cell_edges ≥ cell_nodes`)

If no valid tiling is found, the engine falls through to **k-sum decomposition** (step 6) → **treewidth_dp** (step 7) → **CEJ** (step 8).

## Algorithm

```mermaid
flowchart TD
    A[Graph instance ≥ 20 edges] --> B[Find cell + partition + inter-cell edges]
    B --> C{Partition found?}
    C -- no --> FAIL[Return None — fall through to step 6]
    C -- yes --> D[Try product formula:\nT(cell)^k × ∏ T(inter_components)]
    D --> E{Kirchhoff verified?}
    E -- yes --> R1[Return product result]
    E -- no --> F[Apply boundary quotient + chord recursion\nvia boundary_quotient_tutte]
    F --> R2[Return chord-rule result]
```

The product formula is kept as a fast path for the special case where it works (basically: tree-of-bridges with no chords). The boundary-quotient + chord-rule path handles everything else and is the workhorse.

## Implementation

The full algorithm is in `tutte/graphs/k_sum.py:boundary_quotient_tutte`. The pseudocode:

```python
def boundary_quotient_tutte(target, partition, inter_edges, engine):
    bridges, chords = classify(inter_edges)              # UnionFind on cell super-nodes

    # chord recursion: peel each chord one at a time via the chord rule.
    g_chord_free, contractions = iterative_chord_rule(target, chords, engine)
    # Result: T(target) = T(g_chord_free) + Σ contractions

    # boundary quotient: closed-form for the chord-free residual.
    cells_polys = [T(g_chord_free.subgraph(cell)) for cell in partition]
    result = boundary_quotient(g_chord_free, partition, bridges, cells_polys, engine)
    if result is None:
        result = engine.synthesize(g_chord_free)         # rare fallback

    return result + sum(contractions)
```

### Boundary Quotient Formula

For a chord-free decomposition (only bridges connect cells):

```
T(target) = [∏_i T(C_i)] · T(B) / [∏_i T(B_i)]
```

where:
- `C_i` is the i-th cell (induced subgraph on cell i's nodes)
- `B` is the **boundary subgraph**: induced on the union of all boundary nodes (cell vertices touched by any inter-cell edge), including both inter-cell edges and intra-cell edges among boundary nodes
- `B_i` is each cell's **boundary-induced subgraph**: intra-cell edges only

This is a strict generalization of the classical `k_join` formula `T(G_1 ⊕_k G_2) = T(G_1) · T(G_2) / T(K_k)` — when the boundary is a clique and per-cell boundaries match, this reduces to the standard k-join.

### Chord Recursion (Iterative Chord Rule)

For each chord inter-cell edge (an inter-cell edge that closes a cycle through cell-components):

```
T(G) = T(G − e) + T(G / e)
```

Iterated, this expresses `T(target)` as `T(chord-free residual) + Σ T(contraction leaf)`. Each contraction leaf is synthesized as a `MultiGraph` (preserving parallel edges and loops created by the contraction).

**Critical implementation detail**: chord contraction must use `MultiGraph.merge_nodes()` to preserve edge multiplicities. Using `Graph` (which deduplicates edges) silently produces wrong polynomials.

## Cost

| Phase | Cost |
|-------|------|
| Find cell candidates (`graphs/covering.py:find_cell_candidates`) | O(table_size × n) — arithmetic filters |
| Partition (signature matching or VF2) | O(n × k) typical, exponential VF2 fallback |
| Bridge/chord classification | O(α(n)) per edge via UnionFind |
| **chord recursion leaves** | **O(chord_count)** full multigraph syntheses |
| **boundary quotient on chord-free leaf** | 1 boundary synthesis + per-cell boundary syntheses + 1 polynomial division |
| **Total** | **O(chord_count + 1) syntheses** + polynomial division |

Compare to the retired Theorem 6 path:
- Theorem 6 cost: O(F²) where F is the number of flats of the inter-cell matroid (up to 50,000-flat cap)
- Old edge-by-edge fallback: O(2^chord_count) worst case

The chord rule is **uniformly fewer or equal syntheses**, with no matroid infrastructure (no flat lattices, no Möbius function, no inclusion-exclusion bookkeeping).

## Limitations

- Only works when the graph has a **repeating cell structure** with the cell already in the rainbow table.
- **Node count must divide evenly** by the cell size — off-by-one means no tiling found.
- The chord-free residual's **boundary quotient formula** assumes exact polynomial division; for very exotic boundary structures it may have a remainder, in which case the engine falls back to direct synthesis of the chord-free residual.
- The **K_k ⊕_k K_k degenerate case** (cells exactly equal the shared K_k clique → empty target) is currently mishandled by both the chord rule and the historical Theorem 10. Easy fix: short-circuit when chord recursion bottoms out at zero edges. Not yet implemented; affects only K_3 ⊕_3 K_3 and K_4 ⊕_4 K_4 type cases that don't arise in practice.

## See Also

- [05_1 Find and Partition Cells](08_1_find_and_partition_cells.md) — how the partition is discovered
- [06 — Chord-Rule Formalization](08_2_chord_rule_formalization.md) — formalization of boundary quotient + chord recursion, empirical validation, replacement for Theorems 6 & 10
