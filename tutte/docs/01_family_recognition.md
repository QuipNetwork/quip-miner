# 2. Family Recognition

## Summary

Before any structural decomposition runs, the engine checks whether the input graph belongs to a **known parametric family** with a closed-form Tutte polynomial. If so, the polynomial is computed in O(n + m) time from a formula or constant-coefficient recurrence — orders of magnitude faster than going through any of the decomposition paths.

This is the **first non-trivial check** in the synthesis pipeline (after rainbow-table lookup). Implemented in `tutte/family_recognition/`.

## When It's Used

Step 2 in the synthesis pipeline. Runs before:
- Base cases (since family recognition handles base-case-equivalent graphs more cheaply)
- Disconnected / cut-vertex factorization (a known family doesn't need to be split)
- Treewidth DP and the chord-rule paths

If the input graph isn't recognized as any family, this step returns `None` and synthesis falls through to the next pipeline stage.

## Recognized Families

| Family | Formula source | T(G) closed form |
|---|---|---|
| **Tree / forest** on n nodes | Tutte (1947) | `x^(n-1)` |
| **Cycle** C_n | Tutte (1947) | `x^(n-1) + x^(n-2) + … + x + y` |
| **Pan** (cycle + tail edge) | Closed form | derived |
| **Sunlet** (cycle with k pendant edges per node) | Closed form | derived |
| **Helm** (wheel + pendant per rim node) | Closed form | derived |
| **Wheel** W_n | Brennan–Mansour–Mphako-Banda (2013) | order-3 recurrence in n |
| **Fan** F_n | Brennan–Mansour–Mphako-Banda (2013) | order-3 recurrence |
| **Ladder** P_2 × P_n | Shrock (2000) | order-3 recurrence |
| **Book** B_n (n triangles sharing one edge) | Closed form | order-3 recurrence |
| **Gear** | Weisstein, MathWorld | order-6 recurrence |
| **Prism** C_n × K_2 | Biggs–Damerell–Sands (1972) | order-4 recurrence |
| **Möbius ladder** ML_n | Chang–Shrock (2004) | order-5 recurrence |
| **Grid** P_m × P_n (small m) | Chang–Shrock (2004) | transfer matrix |

These cover most of the "named graph" entries in the rainbow table (and many graphs that aren't in the table) without any synthesis work.

## Algorithm

1. Compute a cheap **structural fingerprint** of the input graph (degree sequence, triangle count, cycle/girth indicators) in O(n + m).
2. Match the fingerprint against the known-family signatures.
3. If a unique family matches, dispatch to its closed-form formula or recurrence.
4. Verify against Kirchhoff (`T(1, 1) == ST count`); if it doesn't match, the family was a false positive and we fall through.

The fingerprint phase rejects most graphs immediately (a random 30-node graph almost never matches a family signature). The dispatch phase has zero overhead for matched families — the formulas are arithmetic on `n`.

## Cost

| Phase | Cost |
|-------|------|
| Structural fingerprint | O(n + m) |
| Family match check | O(1) — hash table lookup on fingerprint |
| Formula evaluation (single closed form) | O(n) integer ops |
| Recurrence evaluation (k-step recurrence in n) | O(n) integer ops |

Total: **O(n + m)** for any recognized family. This is the fastest path in the synthesis pipeline by a wide margin — the next-cheapest path (rainbow table lookup) costs O(n² × d) for canonical-key computation.

## Implementation

- `tutte/family_recognition/fingerprint.py` — `compute_structural_fingerprint(graph)` returns the lookup key used to identify a family.
- `tutte/family_recognition/formulas.py` — closed-form formulas and recurrences per family.
- `tutte/family_recognition/verification.py` — Kirchhoff cross-check after dispatch.
- Engine entry point: `recognize_family(graph)` in `tutte/family_recognition/__init__.py`. Returns `Optional[TuttePolynomial]`.

## Why This Comes Before Rainbow-Table Lookup

Family recognition is O(n + m). The rainbow-table lookup involves canonical-key computation, which is O(n² × d) for the WL-based hash. For a 100-node graph, family recognition costs ~milliseconds; canonical-key costs ~seconds. When the graph is a known family, we save the canonical-key work entirely.

When family recognition fails (most graphs in the wild), we fall through to canonical-key + rainbow-table lookup as the next step. The fingerprint cost is small enough that the false-negative path is essentially free.

## References

- Tutte, W. T. (1947). *A ring in graph theory.*
- Brennan, C., Mansour, T., Mphako-Banda, E. (2013). *Tutte polynomial of wheel graphs.*
- Biggs, N., Damerell, R., Sands, D. (1972). *Recursive families of graphs.*
- Shrock, R. (2000). *Potts model partition functions on ladder graphs.*
- Chang, S.-C., Shrock, R. (2004). *Strip graph Tutte polynomials via transfer matrices.*
- Weisstein, E. (n.d.). *Gear graph.* MathWorld.
