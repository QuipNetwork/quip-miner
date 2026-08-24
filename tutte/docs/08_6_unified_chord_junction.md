# 8.6. Unified Bivariate Chord-Junction Theorem

A closed-form expression for `T(G ⊕ G; x, y)` — the Tutte polynomial of
a graph formed by two copies of a cell `G` joined by an arbitrary set
of chord edges between corresponding `V_k` vertices. Generalises both
the y=0 chromatic-line cyclotomic theorem and the x=0 flow-line
length-invariance result; reduces to `apply_kmatching_formula` (§8.3)
when the chord pattern is a perfect matching, and to the chain transfer
matrix (§6.7) when stacked into a chain.

> **Status**: proved May 25 2026. Symmetric and asymmetric forms shipped
> in `tutte/roots/chord_junction_closed_form.py`; engine dispatch wired
> at `tutte/synthesis/engine.py::_try_unified_chord_junction`; persistent
> merger cache in `tutte/data/merger_lookup_table.bin` (the `.bin` is
> authoritative; the JSON mirror is no longer written).

## 1. Statement

Let `G` be a graph with `V_k ⊆ V(G)` a chord-position subset. Form the
**chord junction** `G ⊕_{V_k} G` by taking two disjoint copies of `G`
(at vertex IDs `V(G)` and `V(G) + n` with `n = |V(G)|`) and adding a
chord edge between each pair `(v, v + n)` for `v ∈ V_k`. Then:

```
T(G ⊕_{V_k} G; x, y) = (x − 1) · T(G; x, y)²
                     + Σ_{∅ ≠ S ⊆ V_k} T(G ∪_{V_S} G; x, y)
```

where `G ∪_{V_S} G` is the **merger** multigraph obtained from two
copies of `G` by identifying each vertex pair `(v, v + n)` for `v ∈ S`.
Parallel edges arising from the identification are preserved.

### 1.1 Asymmetric variant

When the two cells are not isomorphic — e.g. boundary cells in Pegasus
or Zephyr graphs — the formula extends:

```
T(G_1 ⊕_{chord_pairs} G_2; x, y) = (x − 1) · T(G_1) · T(G_2)
                                  + Σ_{∅ ≠ S ⊆ {1, …, k}}
                                        T(G_1 ∪_{S} G_2)
```

where `chord_pairs = [(u_1, w_1), …, (u_k, w_k)]` names the chord edges
between `G_1`'s vertex `u_i` and `G_2`'s vertex `w_i`, and the merger
`G_1 ∪_S G_2` identifies copy-1's `u_i` with copy-2's `w_i + n_1` for
each `i ∈ S`.

## 2. Derivation sketch

Apply multivariate Sokal-Z deletion-contraction to each chord edge
`e_v = (v, v + n)` for `v ∈ V_k`. Each `e_v` contributes a binary choice:

- **Delete** `e_v` → factor of `1`.
- **Contract** `e_v` → identifies copy-1's `v` with copy-2's `v + n`,
  producing the multigraph term in the `S` sum (with `v ∈ S`).

The all-delete branch gives `T(G ⊔ G; x, y) = T(G)²` on disjoint
copies. The chord-junction's `T` already includes the inter-cell edges,
so we subtract the disjoint-union baseline (encoded as `(x − 1) · T(G)²`
via inclusion-exclusion on the chord edges' "phantom" contribution).
Re-summing over the `2^|V_k|` choice patterns and absorbing the
all-empty term into the `(x − 1)` coefficient yields the formula above.
A complete proof is in
[`tutte/research/cyclotomic_chord_junction_theorem.md`](../research/cyclotomic_chord_junction_theorem.md);
empirical validation across `K_2`, `K_3`, `P_3`, `C_4`, diamond, `K_4`,
`K_{4,4}` (Chimera cell), and a `1×3` Chimera chain is locked in.

## 3. Equivalence with `apply_kmatching_formula`

For the symmetric matching case (each `v ∈ V_k` carries exactly one
chord), the unified formula is mathematically identical to the
k-matching formula (§8.3). The convention bridge is the cut-vertex
identity:

```
T(G ∪_v G) = T(G)²   when v is a single shared vertex (cut vertex)
```

which converts our `(x − 1, C(k, 1), C(k, 2), …, C(k, k))` coefficients
into the k-matching formula's `(x, k − 1, C(k, 2), …, C(k, k))`
coefficients:

```
(x − 1) · T(G)² + Σ_{j=1..k} C(k, j) · T(M_j)        (unified)
  = (x − 1) · T(G)² + k · T(M_1) + Σ_{j=2..k} C(k, j) · T(M_j)
  = (x − 1) · T(G)² + k · T(G)² + Σ_{j=2..k} …       (cut-vertex)
  = (x + k − 1) · T(G)² + Σ_{j=2..k} C(k, j) · T(M_j)  (k-matching)
```

So `apply_kmatching_formula` (`tutte/graphs/covering.py::apply_kmatching_formula`)
is the **special case** of the unified theorem when the chord pattern
is a perfect matching on vertex-transitive anchors. The unified theorem
**generalises** it to:

- **Non-matching chord patterns**: anchors can be reused (e.g. two chord
  edges incident to the same `V_k` vertex).
- **Asymmetric cells**: `G_1 ≠ G_2` (boundary cells in Pegasus / Zephyr).
- **Arbitrary `V_k` orbits**: anchors need not lie in a single
  vertex-transitive class (k-matching formula precondition P2).

The engine's `_try_unified_chord_junction` (engine.py) dispatches to the
unified theorem *before* falling through to `apply_kmatching_formula`,
so symmetric matching cases hit the same `kmatching_formula` method
label but benefit from O(1) merger cache lookups instead of recomputing
each `M_j` term.

## 4. Equivalence with chain transfer matrix

For a chain of `n` cells joined by identical chord junctions (e.g.
`K_{4,4} + M_4 + K_{4,4} + M_4 + … + K_{4,4}`), the chain transfer
matrix (§6.7) extracts an order-`r` polynomial recurrence with `r` =
number of distinct merger types under the cell's automorphism group.
For `K_{4,4} + M_4`, the 5 mergers
`(T(K_{4,4})², 4·M_1, 6·M_2, 4·M_3, M_4)` correspond exactly to the
5-dimensional transfer matrix state space.

The unified theorem and the chain framework are two dual descriptions
of the same algebra:

- **Unified theorem**: explicit closed form per chord-junction, easiest
  to cache (one entry per `(base_canonical_key, V_T)`).
- **Chain recurrence**: order-`r` matrix recurrence in `n`, easiest to
  evaluate for very long chains (modular Faddeev-LeVerrier extracts the
  characteristic polynomial in `O(r³)` mod p work).

The merger cache (§5) operationalises the unified theorem; the chain
framework (`tutte/roots/chain_recurrence.py`) operationalises the
recurrence form. They share the same merger values internally.

## 5. Merger lookup table

The persistent cache lives at
`tutte/data/merger_lookup_table.bin` (the authoritative format; the JSON
mirror is no longer written, though `load_default_merger_table` still
reads a `.json` if one is present), parallel to the existing
`lookup_table`, `multigraph_lookup_table`, and `rooted_lookup_table`.
Each entry is a `MergerEntry` (`tutte/lookup/merger.py`):

| Field                   | Purpose                                                |
| ----------------------- | ------------------------------------------------------ |
| `base_canonical_key`    | SHA-256 canonical key of the base cell `G`.            |
| `v_t`                   | Sorted tuple of base-graph vertex IDs in `V_T`.        |
| `polynomial`            | `T(G ∪_{V_T} G; x, y)` as a `TuttePolynomial`.         |
| `merger_canonical_key`  | SHA-256 canonical key of the merger multigraph itself. |
| `base_name`, `family_tag` | Human-readable labels (`K_{4,4}`, `chimera`, …).     |

Two indices: `by_source[(base_key, v_t)]` for fast cell-pair dispatch,
and `by_merger[merger_key]` so asymmetric chord patterns whose merger
is isomorphic to a cached symmetric one still hit the cache. Loaded at
engine init via `load_default_merger_table()`; promoted to disk by the
warmup script via `save_default_merger_table()`.

### 5.1 Session-vs-disk hybrid

Engine init loads the on-disk table into `self._merger_session_cache`.
Runtime misses on the asymmetric path **populate the session cache
in-memory only** — they are NOT written back to disk automatically.
The warmup script
(`tutte/scripts/warmup_merger_lookup.py`) is the only writer; this
keeps production runs side-effect-free and lets a curator audit which
mergers get promoted to the shared cache.

Current contents (post Step 6/7):

| Cell template        | Family tag    | Entries | V_T enumeration                              |
| -------------------- | ------------- | ------- | -------------------------------------------- |
| `K_{4, 4}`           | `chimera`     | 255     | All non-empty subsets of `{0, …, 7}` (`2^8 − 1`). |
| `K_{4, 4}` (Pegasus)  | `pegasus`     | (same)  | Pegasus identifies `K_{4, 4}` sub-cells via the rainbow table → re-uses the Chimera entries. |
| `Z(1, 1)`            | `zephyr`      | 78      | All non-empty subsets of sizes 1 and 2 (`C(12,1) + C(12,2)`). |

### 5.2 Engine dispatch (§7.88)

`_try_unified_chord_junction(graph, junctions, partition)` runs inside
`_try_decomposition_chord_peel` *before* the existing
`apply_kmatching_formula` call sites. Two tiers:

1. **Symmetric tier** — cells isomorphic AND chord pattern aligns
   canonically (same anchor position on both sides). Lookup via
   `MergerTable.lookup_by_source(base_key, v_t)`. Returns the closed-form
   polynomial in `O(2^|V_k|)` cached lookups + one base `T(G)` synth.
2. **Asymmetric tier** — anything else (mixed-bipartition anchors,
   non-isomorphic cells, asymmetric chord patterns). Builds the explicit
   chord pairs, dispatches to `unified_chord_junction_asymmetric`, which
   looks up each merger by its multigraph canonical key
   (`MergerTable.lookup_by_merger(merger_key)`). Symmetric mergers
   cached for `K_{4, 4}` cover most Pegasus / Zephyr boundary cases
   transparently because the merger graph's canonical key is the same
   regardless of which super-cell it came from.

Both tiers bound `|V_k| ≤ 6` (`_UNIFIED_CHORD_JUNCTION_MAX_VK`) — `2^6
= 64` sub-syntheses is the break-even against the legacy chord rule.

The Round 3 cleanup collapsed the former `HybridSynthesisEngine` /
`AlgebraicSynthesisEngine` into the single `SynthesisEngine` (a.k.a.
CEJ), so `_try_decomposition_chord_peel` — and with it the
unified-chord-junction fast path — now lives in one place with no
separate hybrid wiring to keep in sync.

## 6. Specialisations

### 6.1 y = 0: chromatic chord-junction (cyclotomic)

At `y = 0`, `T(G; x, 0)` reduces (up to a sign) to the chromatic
polynomial. The unified theorem becomes the **cyclotomic chord-junction
theorem**:

```
R_H(x, 0) = -(1 − x) · P(G ⊕_chord G; 1 − x) / P(G; 1 − x)²
```

with a closed form for the cycle-shadow correction
`c(x) = 2x(x² − 1) / Φ_3(x)²` (where `Φ_3` is the 3rd cyclotomic
polynomial). The chord-junction ratio `R_H` depends only on the chord-
position-induced subgraph `H` (tree-fluff-irrelevant), tabulated in
[`tutte/research/cyclotomic_chord_junction_theorem.md`](../research/cyclotomic_chord_junction_theorem.md).

### 6.2 x = 0: flow-line length-invariance

At `x = 0`, `T` reduces to the flow polynomial. The unified theorem
gives the **flow-line length-invariance** identity: for a base graph
`G` containing a cycle of length `m`, `T(G ⊕_chord G; 0, y)` is
invariant under elongating the base cycle when the chord-position-
induced subgraph `N` is **b-invariant**:

```
T = (−1)^(m−1) · P(C_m; 1 − y) / (1 − y) · N(G; 1 − y)²
```

### 6.3 Matroid self-duality

A chord-junction Tutte polynomial is matroid-self-dual
(`T(x, y) = T(y, x)`) iff `|E| = 2|V| − 2`. The smallest example is
the diamond graph (`|V| = 4`, `|E| = 5`, chord junction `|E| = 12`,
`|V| = 8`, `|E| − 2|V| + 2 = 12 − 16 + 2 = -2` — not self-dual; the
8-cycle chord junction with two crossings is the first that hits
`|E| = 2|V| − 2`). See research note §"Hunt 6-vertex 8-edge
self-dual chord-junctions" for the catalogue.

## 7. Cost model

| Path                                    | Cost                                              |
| --------------------------------------- | ------------------------------------------------- |
| Symmetric warm cache hit                | 1 base `T(G)` synth + `O(2^|V_k|)` cache lookups  |
| Symmetric cold cache (first call)       | 1 base + `2^|V_k| − 1` merger synths              |
| Asymmetric warm via canonical-key index | 2 base synths + `O(2^k)` cache lookups            |
| Asymmetric cold                         | 2 base + `2^k − 1` merger synths                  |
| Fallback to `apply_kmatching_formula`   | `O(k)` syntheses per junction (§8.3 cost)         |

The base synth dominates when the cell is large (e.g. `T(K_{4,4})` =
108 terms, ~0.1s via treewidth DP). Mergers for small cells (≤ 12
vertices) take well under 1s each. For `K_{4,4}` mergers, the warmed
255 entries cover all 256 chord-pair subsets; cold cost is ~0.5s for
the full warmup-by-on-demand cycle.

## 8. Worked example: `K_{4, 4}` Chimera cell pair

Two `K_{4, 4}` cells joined by a side-A matching (chord edges
`(i, i + 8)` for `i ∈ {0, 1, 2, 3}`), giving `|V| = 16`, `|E| = 36`.

Symmetric path:

1. Base: `T(K_{4, 4})` — cached in the rainbow table (108 terms).
2. `V_k = (0, 1, 2, 3)`. Iterate over the `2^4 − 1 = 15` non-empty
   subsets `S ⊆ V_k`.
3. For each `S`, `lookup_by_source(base_key, tuple(S))` returns the
   cached `T(K_{4, 4} ∪_S K_{4, 4})` — all 15 hits when the warmup has
   run (`_chimera_cells()` warms all 255 subsets of `{0, …, 7}`).
4. Sum: `T(target) = (x − 1) · T(K_{4, 4})² + Σ_S T(merger_S)`.

Result: 1 synth call (the base `K_{4, 4}`, itself a cache hit), 15
cache lookups, sub-second polynomial assembly. Without the unified
theorem the legacy chord rule does 16 sub-syntheses (each a full chord-
recursion leaf, several seconds each).

## 9. Sokal-Z generalization for non-matching junctions

The unified theorem above requires the chord pattern to be a **matching**:
each anchor in `V_k^A` joins exactly one anchor in `V_k^B`. D-Wave graphs
violate this — Z(1, 2)'s cell-pair has 32 chord edges over 12 anchors
per side with degree distribution `{2: 16, 4: 8}`, not a matching.

The **Sokal-Z generalization** (`tutte/roots/sokal_z_chord_junction.py`)
covers arbitrary `E_J ⊆ V_k^A × V_k^B`:

```
Z(G_1 ⊕_{E_J} G_2; q, v) = Σ_{A_J ⊆ E_J} v^|A_J| · Z(merger(φ(A_J)); q, v)
```

where `φ(A_J)` is the partition of `V_k^A ∪ V_k^B` into connected
components of `(V_k^A ∪ V_k^B, A_J)`. The result converts back to
`T(x, y)` via multi-point evaluation + bivariate Lagrange interpolation
(see research doc).

Two enumeration paths:

1. **Brute force** (`compute_sokal_z_chord_junction`): iterates all
   `2^|E_J|` subsets directly. Gated at `|E_J| ≤ 16` (`max_subsets = 65536`).
2. **Per-H_J-component + tree-DP** (`compute_sokal_z_chord_junction_per_component`):
   decomposes `H_J = (V_k^A ∪ V_k^B, E_J)` into connected components,
   runs edge-by-edge DP per component, then cross-products φ tuples.
   Tree-DP state space ≤ `Bell(|V_c|)` per component; auto-engages when
   `|component_edges| ≥ tree_dp_edge_threshold` (default 13).

Empirical tree-DP speedups (`probe_sokal_z_tree_dp_perf.py`):
K_{4,4} 56×, K_{4,5} 140×, K_{4,8} (Z(1,2)-scale 32-edge component) days → 12.6s.

Dispatch is wired at `engine._try_sokal_z_chord_junction`, called for
2-cell partitions whose junction the matching-based unified theorem
rejects. See [§9 in the research doc](../research/cyclotomic_chord_junction_theorem.md#extension-sokal-z-generalized-chord-junction-theorem-may-26-27-2026)
for the full theorem statement, proof sketch, and per-D-Wave-target
H_J structure tables.

## 10. References

- **Theorem & empirical validation**: `tutte/research/cyclotomic_chord_junction_theorem.md`
  (research note with probes, intermediate hypotheses, and proof
  sketches). The note is research-grade — this doc is the polished
  production reference.
- **Module**: `tutte/roots/chord_junction_closed_form.py` —
  `unified_chord_junction`, `unified_chord_junction_asymmetric`,
  `build_symmetric_merger`.
- **Cache**: `tutte/lookup/merger.py` — `MergerEntry`, `MergerTable`,
  `load_default_merger_table`, `save_default_merger_table`.
- **Warmup script**: `tutte/scripts/warmup_merger_lookup.py` —
  `--family {chimera, pegasus, zephyr, all}`.
- **Engine wiring**: `tutte/synthesis/engine.py` —
  `_try_unified_chord_junction`, `_UNIFIED_CHORD_JUNCTION_MAX_VK`.
- **Related**:
  - §8.3 [k-Matching Formula](08_3_kmatching_formula.md) — the symmetric
    matching special case.
  - §6.7 [Chain & Cycle Recurrence Algebra](06_7_chain_recurrence_algebra.md)
    — the dual chain-form description.
  - §8.2 [Chord-Rule Formalization](08_2_chord_rule_formalization.md) —
    the general chord-rule background.
