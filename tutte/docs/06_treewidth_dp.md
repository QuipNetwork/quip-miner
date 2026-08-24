# 6. Treewidth Dynamic Programming

## Summary

For graphs that don't match a known family, hit the rainbow table, or decompose by cut vertex, the engine builds a **tree decomposition** of the graph and runs a **bag-by-bag dynamic program** to compute the Tutte polynomial. The DP is implemented in C via cffi for speed (`tutte/graphs/_treewidth_c.py`), with Python wrappers in `tutte/graphs/treewidth.py`.

Treewidth DP is **step 6** in the synthesis pipeline — the first heavy computation tried after all the cheap structural fast paths (family recognition, lookup, base cases, disconnected, cut vertex) have failed.

## When It's Used

- Graph has **≥ 10 edges** (smaller graphs use the cheaper paths above).
- Graph has **treewidth ≤ 11** (`max_width` cap; configurable). When the treewidth exceeds this cap, the function returns `None` and the engine falls through to k-sum decomposition (step 7) or hierarchical tiling (step 8).

The 11-edge cap covers most graphs of interest up to about 50–100 edges depending on density. D-Wave Z(1,1), Z(1,2), Cm₂ all fit. Larger Zephyr/Chimera/Pegasus topologies (Cm₃+, Pm₃+, Z(2,t)+) generally exceed it and require the chord-rule paths.

## Algorithm

```mermaid
flowchart TD
    A[Input MultiGraph] --> B[Build tree decomposition\nupper_bound search ≤ 11]
    B --> C{Treewidth ≤ 11?}
    C -- no --> FAIL[Return None — fall through to step 7]
    C -- yes --> D[Bulk DP via cffi C extension]
    D --> E{Coefficient bound choice}
    E -- ≤62 edges --> E1[int64 (a,b)-basis DP\n+ int64 conversion]
    E -- 63–120 edges --> E2[__int128 (a,b)-basis DP\n+ Python basis conversion]
    E -- > 120 edges --> E3[full modular DP + CRT\nover multiple primes]
    E1 --> R[Tutte polynomial]
    E2 --> R
    E3 --> R
```

The DP itself is a standard bag-by-bag computation indexed on partial-rank-and-connectivity profiles. Each bag's profile contributes coefficients to the running polynomial in an `(a, b) = (x-1, y-1)` basis; a final basis conversion produces the standard `(x, y)` polynomial.

## Implementation

- `tutte/graphs/treewidth.py` — Python wrappers, tree-decomposition computation.
- `tutte/graphs/_treewidth_c.py` — cffi build script; defines the C source for `treewidth_tutte_dp`, `treewidth_tutte_dp_ab`, `treewidth_tutte_dp_ab128`, and the modular CRT variants.
- Engine entry point: `compute_treewidth_tutte_if_applicable(mg, max_width=11)` in `tutte/graphs/treewidth.py`. Returns `Optional[TuttePolynomial]`.

### Coefficient-bound tiers

The (a, b)-basis Tutte polynomial coefficients can grow as large as `2^E / sqrt(E)` where E is the edge count (a consequence of the rank-generating-function representation). Different DP variants are used for different size regimes:

| Edge count | DP variant | Basis-conversion variant |
|---|---|---|
| ≤ 62 | int64 | int64 |
| 63 – 120 | `__int128` | Python bignums |
| > 120 | full modular CRT | Python bignums |

The previous "63–76 edges: int64 DP + modular conversion" variant was retired in April 2026 — it silently overflowed `2^63` for E ≥ 63, producing polynomials with correct `T(1, 1)` (so Kirchhoff verification passed) but wrong other coefficients. Symptom showed up only when the engine tried to compose two such polynomials downstream. Fix: lowered the int128-DP threshold to `> 62` so the entire 63–120 regime uses `__int128`.

## Cost

| Phase | Cost |
|-------|------|
| Tree decomposition (upper-bound search) | O(2^tw × n) typical |
| Bag DP (per bag, per profile) | O(profile_count × bag_size) |
| Basis conversion `(a, b) → (x, y)` | O(degree²) Python big-int ops |
| **Total** | **O(2^tw × n)** with low constant in C |

For graphs with `tw ≤ 11`, the constant is small enough that this is typically the fastest path in absolute wall-clock time among heavy synthesis routes (e.g. Cm₂ in ~10 minutes vs ~5 minutes for chord-rule, but treewidth_dp wins on Petersen, Z(1,1), and most non-D-Wave graphs in the test corpus).

## Limitations

- **Treewidth cap**: graphs with `tw > 11` are not attempted. Raising the cap costs exponentially more memory and time.
- **Single-graph only**: doesn't decompose disconnected inputs (caller should split first — handled by step 4 of the pipeline).
- **Multigraph required**: input must be a `MultiGraph` (loops and parallel edges allowed). The engine wraps `Graph` inputs via `MultiGraph.from_graph(g)`.

## See Also

- [08_hierarchical_tiling.md](08_hierarchical_tiling.md) — chord-rule fallback for larger graphs (treewidth > 11) with hierarchical decomposition.
- [08_2_chord_rule_formalization.md](08_2_chord_rule_formalization.md) — replacement for the matroid-theoretic Theorem 6 / Theorem 10 paths.
