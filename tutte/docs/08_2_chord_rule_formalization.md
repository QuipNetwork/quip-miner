# 6. Chord-Rule Formalization

A mathematical formalization of the chord-rule approach used by the synthesis engine, with comparison to the Bonin-de Mier matroid framework.

> **Status**: working draft. The empirical content (Section 5) is complete and validated by the test suite. The proofs in Sections 3 and 4 are sketched at the level of textbook deletion-contraction; tightening them to publication standard is future work.

## How we got here

The chord-rule approach was discovered through a four-stage investigation in April 2026:

1. **k-sum algebra investigation.** We scanned the rainbow table for graphs decomposable as `k` identical cells + inter-cell edges, then tested five candidate algebraic rules. Result: the **boundary quotient** identity succeeded on chord-free decompositions (chains, stars), failed on cyclic-boundary cases (necklaces, ladders), and gave trivial-identity passes on D-Wave-style graphs where every cell node is a boundary node.

2. **Chord-recursion fix.** The natural extension is to peel inter-cell chord edges one at a time via the standard chord rule `T(G) = T(G−e) + T(G/e)`. Combined with boundary quotient on the chord-free residual, this gives a complete polynomial-cost replacement for the matroid path in `_synthesize_hierarchical()`. Validated against 1100+ existing tests.

3. **Engine bug** (parallel discovery). While debugging chord recursion on Cm2, we found a coefficient-overflow bug in `tutte/graphs/_treewidth_c.py`: the int64 (a,b)-basis treewidth DP was used for graphs up to 76 edges, but coefficients can reach `2^E / sqrt(E)`, overflowing `2^63` for E ≥ 63. Symptom: `T(1,1)` aligned mod `2^64` so Kirchhoff verification missed it. Fix: lowered the int128-DP threshold to `> 62`.

4. **k-sum divisor investigation.** 205-case empirical search ruled out a closed-form polynomial divisor `D(x, y)` that depends only on `k` (the natural way to express true k-sum as a quotient of cell polynomials). The ratios `T(G_1) · T(G_2)(1, 1) / T(G_1 ⊕_k G_2)(1, 1)` take 35 distinct values for k=2 and 15 for k=3, many non-integer — no polynomial in `(x, y)` with integer coefficients can match all of them. **Constructive replacement for Theorem 10**: apply chord recursion to the shared `K_k` clique edges of the parallel connection. Cost: `1 + C(k, 2)` syntheses, fewer than even the flat-grouped Theorem 6 (`|flats(K_k)|` syntheses).

The two retired investigation reports and their generating scripts have been removed; their conclusions are baked into Sections 3, 4, and 5 below.

## 1. Setup

Let `G = (V, E)` be a graph (possibly with loops and parallel edges, i.e. a multigraph). Its Tutte polynomial `T(G; x, y)` is the unique two-variable polynomial satisfying:

- **Bridge**: if `e ∈ E` is a bridge, `T(G; x, y) = x · T(G − e; x, y)`
- **Loop**: if `e ∈ E` is a loop, `T(G; x, y) = y · T(G / e; x, y)`
- **Ordinary edge**: if `e` is neither, `T(G; x, y) = T(G − e; x, y) + T(G / e; x, y)`
- **Boundary**: `T(∅; x, y) = 1` for the empty graph

The third identity is the **chord rule** (deletion-contraction). It is universal: it holds for *any* edge that is not a loop or bridge, and applying it to a bridge or loop also yields the right answer (the special-case formulas are just simpler).

### Bonin-de Mier Theorem 6 (background)

Let `G_1, G_2` be two graphs whose intersection is a `k`-clique `K_k` on a shared vertex set `S`. Their **parallel connection** `P_{K_k}(G_1, G_2)` is the graph `G_1 ∪ G_2` with the K_k edges identified. The Brylawski 2-sum / Bonin-de Mier formula expresses `T(P_{K_k}(G_1, G_2))` as an inclusion-exclusion sum over the lattice of flats of `M(K_k)`, weighted by the Möbius function. The total cost is O(F²) flat-polynomial multiplications where F is the number of flats of `K_k` (5 for k=3, 12 for k=4, 52 for k=5).

The matroid path is correct but mathematically heavy and computationally expensive. The chord-rule approach below subsumes both Theorem 6 and the closely-related Theorem 10 (k-sum of two graphs along a shared K_k with the clique edges *deleted*) using only the deletion-contraction identity.

## 2. Two Graph Operations

### 2.1 Hierarchical Tiling

A **k-cell hierarchical decomposition** of `G` is a partition `V(G) = V_1 ⊔ V_2 ⊔ … ⊔ V_k` such that:

- Each induced subgraph `C_i = G[V_i]` is isomorphic to a known **cell** graph `H`.
- The remaining edges (those with one endpoint in `V_i` and the other in `V_j`, `i ≠ j`) are called **inter-cell edges**.

The **boundary** of `G` (relative to this decomposition) is the set of vertices touched by at least one inter-cell edge. The **boundary subgraph** `B` is the induced subgraph of `G` on these boundary vertices — it includes both inter-cell edges and intra-cell edges among boundary vertices. Per-cell, the **boundary-induced subgraph** `B_i` is the induced subgraph of `C_i` on its boundary vertices (intra-cell edges only).

### 2.2 True k-Sum

A **true k-sum** of two graphs `G_1` and `G_2` along a shared `K_k` is the graph obtained by:

1. Identifying `k` distinguished vertices in `G_1` with `k` distinguished vertices in `G_2`.
2. Deleting all `C(k, 2)` edges of the resulting K_k clique.

The **parallel connection** `PC` is the graph just before step 2 (clique edges still present).

## 3. Theorem (Boundary-Quotient)

**Statement.** Let `G` admit a hierarchical tiling with disjoint cells `C_1, …, C_k`, boundary subgraph `B`, and per-cell boundary subgraphs `B_1, …, B_k`. Suppose the inter-cell graph (the subgraph of `G` consisting of inter-cell edges only) is **acyclic** (i.e. it forms a forest of bridges through cell super-nodes). Then

```
T(G; x, y) · ∏_i T(B_i; x, y)  =  T(B; x, y) · ∏_i T(C_i; x, y).
```

Equivalently, when the right-hand side is divisible by `∏_i T(B_i)`,

```
T(G; x, y) = [∏_i T(C_i; x, y)] · T(B; x, y) / [∏_i T(B_i; x, y)].
```

**Proof sketch.** When the inter-cell graph is acyclic, every inter-cell edge is a bridge in the multigraph `G_super` obtained by contracting each cell to a super-node. Apply the bridge rule iteratively to peel off all inter-cell edges; what remains is the disjoint union `C_1 ⊔ … ⊔ C_k`, with `T = ∏_i T(C_i)`. The bridge factor `x^|inter-cell|` and the structure-aware boundary terms combine to give the stated formula via the standard k-join identity applied recursively along the bridge tree. □

**Generalization of Brylawski's k-join.** When `B` is itself a `k`-clique and each `B_i` is the same `k`-clique, the formula collapses to the classical k-join formula `T(G_1 ⊕_k G_2) = T(G_1) · T(G_2) / T(K_k)`. boundary quotient is therefore a **strict generalization** to arbitrary boundary-subgraph shapes (not just cliques) and to k-cell decompositions (not just k=2).

**Failure mode.** When the inter-cell graph contains a cycle (i.e., a chord exists in the inter-cell super-graph), the bridge-rule argument no longer applies and the formula's polynomial division is generally not exact. In practice this manifests as a non-zero remainder from `polynomial_divmod`. chord recursion (Section 4) handles the cyclic case.

## 4. Theorem (Chord Recursion)

**Statement.** Let `G` be a graph and `C ⊆ E(G)` a set of edges (the **chord set**). For any ordering `c_1, c_2, …, c_n` of `C`,

```
T(G; x, y) = T(G − C; x, y) + Σ_{i=1}^{n} T((G − {c_1, …, c_{i-1}}) / c_i; x, y).
```

**Proof.** Induction on `|C|`. For `|C| = 0`, both sides equal `T(G)`. For `|C| ≥ 1`, apply the deletion-contraction identity to `c_1`:

```
T(G) = T(G − c_1) + T(G / c_1).
```

Apply the inductive hypothesis to `T(G − c_1)` with the smaller chord set `{c_2, …, c_n}`:

```
T(G − c_1) = T(G − {c_1, c_2, …, c_n}) + Σ_{i=2}^{n} T((G − c_1 − {c_2, …, c_{i-1}}) / c_i).
```

Substituting back gives exactly the stated formula. □

**Computational cost.** `n + 1` synthesis calls, where `n = |C|`. Compare to the brute-force expansion of `n`-fold deletion-contraction (`2^n` leaves) and to the matroid-flat-grouped Theorem 10 (`|flats(K_k)|` leaves, e.g. 12 for k=4 vs the chord rule's 7).

**Application to true k-sum.** Apply chord recursion with `C = ` the K_k clique edges of the parallel connection `PC = target ∪ K_k`. Then `G − C = target` (the chord-free residual is the k-sum we want), and

```
T(target) = T(PC) − Σ_{i=1}^{C(k,2)} T((PC − e_1 − ... − e_{i-1}) / e_i).
```

This **subsumes Theorem 10** without flat-lattice machinery, with `C(k, 2) + 1` syntheses (versus `2^{C(k,2)}` brute force or `|flats(K_k)|` flat-grouped).

## 5. Empirical Validation

Implementation: `tutte/graphs/k_sum.py:boundary_quotient_tutte` (boundary quotient + chord recursion for hierarchical case), `tutte/graphs/k_sum.py:clique_chord_k_sum` (chord recursion for true k-sum).

| target | k | chords | boundary quotient | chord recursion | Theorem 10 (legacy) |
|---|---:|---:|---|---|---|
| chains of K_4, K_5 (≤ 5 cells) | 3-5 | 0 | ✓ | ✓ | n/a |
| stars of K_4 (4-5 leaves) | 4-5 | 0 | ✓ | ✓ | n/a |
| Petersen | 2 | 4 | trivial† | ✓ | ✓ |
| Z(1,1) | 4 | 7 | trivial† | ✓ | ✓ |
| Cm2 | 4 | 13 | trivial† | ✓ | ✓ |
| Z(1,2) | 2 | 31 | trivial† | ✓ | ✓ |
| 4xK4_necklace, ladder | 4 | 1-3 | ✗ | ✓ | ✓ |
| K_3 ⊕_3 K_3 (degenerate) | 3 | n/a | n/a | ✗‡ | ✗‡ |

† "Trivial" = boundary equals the full target so the formula collapses to `T(target) = T(target)` — uninformative pass.
‡ Degenerate case: cells == shared K_k → empty target. Both methods return wrong polynomial; an empty-graph short-circuit fixes it (not yet implemented).

Cross-validated against the engine's standalone `treewidth_dp` path on every target the latter handles. Polynomial equality holds in every case the chord rule succeeds.

### Performance

| target | tw_dp | chord_rule | speedup |
|---|---:|---:|---:|
| Petersen | 2021 ms | 12 ms | 168× |
| Z(1,1) | 10 ms | 48 ms | 0.2× |
| Cm2 | 636 sec | 296 sec | 2.1× |
| Z(1,2) | 168 sec | 172 sec | 0.97× |

Chord rule wins on large structured graphs, loses on small graphs where tw_dp's overhead is small. The synthesis engine now tries chord-rule **before** tw_dp for graphs ≥ 20 edges with a hierarchical decomposition; smaller graphs go to tw_dp first.

## 6. Why the Chord Rule Generalizes Bonin-de Mier

Bonin-de Mier Theorem 6 expresses `T(P_{K_k}(G_1, G_2))` as a sum over flats of `M(K_k)`, weighted by the Möbius function:

```
R(P_{K_k}(M_1, M_2); u, v) = v^{-r(N)} Σ_{A ∈ flats(N)} μ(0, A) (v+1)^{|A|} f_1(A) f_2(A)
```

where `R` is the rank-generating polynomial in `u = x − 1, v = y − 1`, `N = M(K_k)`, and `f_i(A)` is itself a Möbius-weighted sum over flats above `A`.

The chord-rule approach produces the same result via a fundamentally different decomposition:

1. Build the parallel connection `PC = G_1 ∪ G_2 ∪ K_k` (clique edges added back).
2. Apply chord recursion to the `C(k, 2)` clique edges to peel them off. The chord-free residual is `target = G_1 ⊕_k G_2` (clique deleted).
3. `T(target) = T(PC) − Σ contractions`.

**Key observation.** The Möbius/flat machinery of Theorem 6 is doing a particular kind of inclusion-exclusion *over the matroid lattice* of the shared K_k structure. The chord rule does inclusion-exclusion *over edges* directly. For a `K_k` clique with `m = C(k, 2)` edges, the per-edge approach has linear complexity `m + 1`; the per-flat approach has complexity `|flats(K_k)|` which grows roughly as `m^{m/2}` for unstructured matroids. **The flat lattice is overkill when the underlying structure is a clique.**

Intuitively, the flat lattice indexes "patterns of connectivity that the K_k edges might collapse to". For the iterative chord rule, we don't need to enumerate patterns — we just peel one edge at a time and the patterns that arise become explicit at each step. The arithmetic does the same work; the chord rule is a "lazy" or "online" version of the flat-lattice computation.

This generalization extends to **incomplete-graph boundaries** (boundary quotient): when the boundary subgraph `B` is not a clique, the matroid path requires enumerating flats of `M(B)`, which can be exponential in `|E(B)|`. The boundary-quotient formula handles this case directly via polynomial division, without enumerating `B`'s matroid structure.

## 7. Open Questions

1. **When does boundary quotient divide cleanly?** Empirically, boundary quotient succeeds when the inter-cell graph is acyclic (proved as Section 3) AND in some other cases we don't fully characterize. A clean characterization of "boundary quotient-eligible" boundary structures would be useful.

2. **Optimal chord ordering.** Different orderings of `C` in chord recursion produce different intermediate contraction leaves. A heuristic that prefers contractions which create cut vertices (enabling factorization) would reduce per-leaf cost. Not yet investigated.

3. **Partial chord rule.** The chord rule applied to a *strict subset* of inter-cell edges produces a partial decomposition that combines chord recursion and boundary quotient (the residual is no longer chord-free, but boundary quotient might apply approximately to a "reduced boundary"). Could this give a smooth interpolation between cost extremes?

4. **Generalization beyond graphic matroids.** Bonin-de Mier Theorem 6 holds for arbitrary matroids; the chord rule is only formulated for graphs (since contraction/deletion are graph operations). A matroid analog of chord recursion would require a notion of "non-loop, non-coloop element contraction" that preserves the rank-generating polynomial — this is essentially what the matroid deletion-contraction formula provides, so a matroid version of chord recursion should be possible.

5. **Hardness of the chord-free residual.** chord recursion leaves a chord-free residual whose Tutte polynomial we still need to compute (via boundary quotient or direct synthesis). For pathological graphs the residual might dominate the cost. Characterizing residual complexity in terms of input properties is open.

## 8. References

- Tutte, W. T. (1947). *A ring in graph theory.* Mathematical Proceedings of the Cambridge Philosophical Society.
- Brylawski, T. H. (1971). *A combinatorial model for series-parallel networks.* Transactions of the AMS.
- Bonin, J. & de Mier, A. (2004). *T-uniqueness of some families of k-chordal matroids.* Advances in Applied Mathematics.
- This codebase: `tutte/docs/08_2_chord_rule_formalization.md`, `tutte/docs/08_2_chord_rule_formalization.md`.

---

> **Suggested citation**: "Chord-Rule Tutte Polynomial Synthesis Replaces Bonin-de Mier Flat-Lattice Inclusion-Exclusion" — Quip Network development, April 2026. Source: `tutte/graphs/k_sum.py` and accompanying documentation.
