# 8.3. The k-Matching Cell-Topology Formula

A closed-form expression for the Tutte polynomial of a multi-cell graph whose inter-cell edges form k-edge matchings on vertex-transitive anchor sets. Extends the chord rule (§8.2) from a linear recursion to a polynomial-coefficient formula that short-circuits deletion-contraction when the cell-topology satisfies specific structural preconditions. Directly applies to D-Wave Chimera targets.

> **Status**: working draft. Single-cell-pair and cell-tree forms have proof sketches; the cell-cycle extension is empirical-only. Preconditions are tight in the sense that violations have been verified to break the formula (Sections 5, 7).
>
> **Relation to the unified theorem**: §8.6 (the Unified Bivariate Chord-Junction Theorem, May 25 2026) is a strict generalisation. The k-matching formula is the special case where the chord pattern is a perfect matching on vertex-transitive anchors. The two are mathematically equivalent on that case via the cut-vertex identity `T(G ∪_v G) = T(G)²` (see §8.6 §3). At runtime, `_try_unified_chord_junction` runs *before* `apply_kmatching_formula` so the matching case enjoys O(1) merger-cache lookups instead of `O(k)` recursive syntheses — the engine reports the same `kmatching_formula` method label either way.

## How we got here

The k-matching formula was discovered through an algebraic-pattern hunt in April 2026, motivated by the observation that the unified formula (§8.unified) fails on D-Wave Chimera targets because their inter-cell couplers are the _distinct-vertex-pair_ chord case, not the _shared-vertex-pair_ parallel case.

1. **2-cell polynomial-divmod scan.** We computed `T(G_1 + M_k + G_2)` for K*3, K_4, K*{4,4}=Cm1 cells with `k = 1..4` matching edges. Divmod against `T(G_1)·T(G_2)` gave misleading near-matches for small k (the divmod quotient coincidentally equaled `T(P_k)` = k parallel edges for k ≤ 3), but the remainder was non-zero and grew with cell size. No clean factorization exists in this basis.

2. **Bridge-aware iterated chord rule.** Formally unrolling the chord rule on the k matching edges — with a bridge factor `x` at the last step because the final remaining inter-cell edge becomes a bridge — produced a 2-dimensional recurrence `P(p, q)` on (anchor-identifications, remaining-chords). Solving the recurrence with a hockey-stick identity gave the single-cell-pair closed form (Section 3).

3. **Multi-cell extension.** For a cell-PATH of `M_k` matchings, applying the 2-cell formula at the last junction gives a valid recurrence in `n` (cell count). For a cell-CYCLE, the closing edges are all chords (no bridge factor) and a direct binomial closure works.

4. **Precondition discovery.** On K*3 cells in cycle topology the formula gave wrong answers; on K*{4,4} cells the formula gave correct answers. The distinguishing property is whether junctions at the same cell share anchor vertices: when a contraction at one junction can create parallel edges with another junction's edge, the closed-form formula breaks. K\_{4,4} cells sidestep this because horizontal and vertical couplers use disjoint bipartition sides.

## 1. Setup

Let `G` admit a hierarchical tiling (§8.1) with cells `C_1, …, C_n` and inter-cell edge set `E_inter`. Define the **cell-topology graph** `H` by:

- `V(H) = {1, 2, …, n}` (one vertex per cell).
- For each inter-cell edge `e ∈ E_inter` between cells `C_i` and `C_j`, add an edge `(i, j)` to `H`. Multiple inter-cell edges between the same cell-pair become parallel edges in `H`.

We say the inter-cell structure is a **k-matching topology** when:

**(P1)** Every pair `(i, j)` of cells connected in `H` has exactly `k_{ij}` inter-cell edges (the **junction size**), and those `k_{ij}` edges form a matching — distinct vertex pairs with no vertex used twice on either side.

**(P2)** For each junction `(i, j)`, the `k_{ij}` anchor vertices on the cell-`i` side all lie in a single vertex-transitive class of `C_i`'s automorphism group (likewise for cell-`j`). For bipartite cells, this reduces to "all anchors on the same bipartition side."

**(P3)** If `H` has at least one cycle, no two junctions at the same cell `C_i` may share an anchor vertex.

### 1.1 Why the preconditions

Preconditions (P1) and (P2) establish that the cell-pair inter-cell structure is (up to cell automorphism) a true k-matching on equivalent anchors. Precondition (P3) rules out _feedback_ in the contraction dynamics: when `H` has a cycle AND two junctions at the same cell share an anchor vertex, contracting a chord edge at one junction relabels that anchor and creates a parallel edge with the other junction's edge — violating the bookkeeping that the closed form relies on.

K\_{4,4} Chimera cells satisfy (P3) automatically because each cell's horizontal-coupler anchors are its `A`-side vertices and its vertical-coupler anchors are its `B`-side vertices, which are disjoint. Small complete-graph cells (`K_3`, `K_4`) with cycle-topology decompositions generally violate (P3).

## 2. Theorem (Single-Cell-Pair Closed Form)

**Statement.** Let `G_1` and `G_2` be graphs with distinguished anchor sets `A_1 ⊂ V(G_1)` and `A_2 ⊂ V(G_2)` each of size `k ≥ 1`, where the anchors within each cell lie in a single vertex-transitive class of the respective automorphism group. Let `G = G_1 + M_k + G_2` denote the graph obtained by disjoint-uniting `G_1, G_2` and adding `k` matching edges `{(a_i, b_i) : 1 ≤ i ≤ k}` with `a_i ∈ A_1, b_i ∈ A_2`. Then

```
T(G; x, y) = (x + k − 1) · T(G_1; x, y) · T(G_2; x, y)
             + Σ_{j = 2}^{k} C(k, j) · T(G_1 ⊕_j G_2; x, y)
```

where `G_1 ⊕_j G_2` denotes the multigraph obtained by identifying `j` of the `k` anchor pairs (any choice of `j` pairs gives an isomorphic result by the vertex-transitivity assumption), keeping all intra-cell edges of both `G_1` and `G_2`, and dropping the remaining matching edges.

**Proof sketch.** Process the `k` matching edges by iterated chord rule. In every intermediate state except the final step, at least two matching edges remain between `G_1` and `G_2`, so the edge being processed is not a bridge (the other matching edges provide an alternative path). Applying the chord rule `T(H) = T(H − e) + T(H / e)` at each step gives a 2^k expansion over subsets `S ⊆ {1, …, k}` of contracted-edge indices, each term contributing `T(H_S)` where `H_S` is `G_1 ∪ G_2` with the anchors in `S` identified and the matching edges outside `S` deleted.

Two observations collapse the expansion:

1. **Isomorphism equivalence.** By vertex-transitivity of anchors, `H_S` depends only on `|S| = j`, so all `C(k, j)` subsets of size `j` contribute the same polynomial `T(G_1 ⊕_j G_2)`.

2. **Bridge factor at step `j = 1`.** The only step where an intermediate edge _is_ a bridge is when `j = 1` — specifically, when the contraction sub-tree reaches the state "one matching edge remains, no contractions yet." There, `T(H) = x · T(H − e)`, where `T(H − e) = T(G_1 ∪ G_2) = T(G_1) · T(G_2)`. Summing this `x · P(0, 0)` contribution with the binomial `(k − 1) · P(1, 0) = (k − 1) · T(G_1) · T(G_2)` (from the `j = 1` subsets that contract one edge after other non-first chord steps) yields the `(x + k − 1) · T(G_1) · T(G_2)` leading term.

The remaining `j ≥ 2` subsets give the binomial sum `Σ C(k, j) · T(G_1 ⊕_j G_2)`. □

**Special cases.**

| k                  | Formula                                    | Justification                                                                                                   |
| ------------------ | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `k = 0` (disjoint) | `T(G_1) · T(G_2)`                          | Disjoint factorization; formula gives `(x − 1) · T(G_1) T(G_2)` which is incorrect — handle `k = 0` separately. |
| `k = 1` (bridge)   | `x · T(G_1) · T(G_2)`                      | Formula yields `(x + 0) · T(G_1) T(G_2)` ✓.                                                                     |
| `k = 2`            | `(x + 1) · T(G_1) T(G_2) + T(G_1 ⊕_2 G_2)` | Validated on K*3, K_4, K*{4,4}.                                                                                 |

## 3. Theorem (Cell-Path Recurrence)

**Statement.** Let `G` be a cell-path of `n` identical cells `c_1, c_2, …, c_n` connected by `M_k` matchings at each junction, with all junctions using the same vertex-transitive anchor class. Let `T_n = T(G; x, y)` and let `P_{n-1} ⊕_j c_n` denote the `(n−1)`-cell path with `c_n` attached sharing `j` anchors with `c_{n-1}` (no new matching edges; `j`-vertex identification only). Then

```
T_n = (x + k − 1) · T_{n-1} · T(c)
      + Σ_{j = 2}^{k} C(k, j) · T(P_{n-1} ⊕_j c_n)
```

with base case `T_1 = T(c)`.

**Proof.** Apply the single-cell-pair closed form (§2) at the last junction. The `G_1`-role is taken by the `(n−1)`-cell path `P_{n-1}`, and `G_2`-role by `c_n`. The anchor vertices on the `c_n` side are a vertex-transitive set in `c_n` by assumption; the anchor vertices on the `P_{n-1}` side are a vertex-transitive subset of `c_{n-1}` because the previous junction — if it used the same vertex class of `c_{n-1}` — shares anchors with this one, but on the _other_ side of the cell, so the two anchor sets are distinct vertex-transitive classes within `c_{n-1}`. When that's not the case (e.g. `K_3` cells with only three vertices), (P3) fails and the formula does not apply. □

**Note on precondition for the cell-path case.** Path topology has no cycles, so (P3) is automatically satisfied — shared anchors across junctions of a single cell do not cause parallel-edge feedback because contractions chain linearly along the path. Empirically verified on K_3 and K_4 cell paths where anchor sharing does occur.

## 4. Theorem (Cell-Cycle Closing-Edges Formula)

**Statement.** Let `G_cyc` be a cell-cycle of `n` cells `c_1, …, c_n` connected by `M_k` matchings at each of `n` junctions, with all junctions satisfying (P1)–(P3). Let `P_n` be the cell-path obtained by removing one junction (the "closing junction") from `G_cyc`. Then

```
T(G_cyc; x, y) = Σ_{j = 0}^{k} C(k, j) · T(P_n with c_1, c_n j-glued; x, y)
```

where `P_n with c_1, c_n j-glued` is the cell-path `P_n` with `j` anchor pairs of `c_1` identified with `j` anchor pairs of `c_n`.

**Proof sketch.** Apply the chord rule to the `k` closing-junction matching edges. None of these is a bridge (the cell-path already connects `c_1` to `c_n` transitively via `c_2, …, c_{n-1}`), so the `j = 1` bridge factor of §2 is absent. Each subset `S ⊆ {1, …, k}` of contracted edges gives `T(P_n` with `c_1, c_n` sharing `|S|` anchors`)`. By vertex-transitivity (P2) of the anchors and disjoint-anchors (P3) within each cell, all `C(k, |S|)` subsets of size `|S|` give isomorphic multigraphs, and the sum collapses to the stated binomial. □

**Corollary (Chimera Cm2).** `T(Cm2)` factors via the cell-cycle formula applied to the 2×2 cell-grid of K\_{4,4} tiles. Empirically: 675-term polynomial matches direct synthesis (`verify_spanning_trees` ✓), with ~4× speedup versus treewidth_dp (51 s vs 196 s cold cache).

> **Update Cell-Quotients.** The cell-quotient grid DP
> with streaming junction enumeration (see
> [§6.5 — Cell-Quotient Grid DP](06_5_cell_quotient_grid_dp.md), now
> computes T(Cm2) in ~36 seconds — 1.5× faster
> than the k-matching formula's 55 s baseline. The k-matching formula
> still wins on cell topologies that don't admit a 2D-grid quotient
> (single cell-pairs, cell-trees, cell-cycles where rows/cols ≥ 2 is
> not detected by `is_grid_topology`), so it remains the appropriate
> dispatch for those cases. The two paths are complementary: this
> §8.3 formula is closed-form (no DP state); §6.5 is DP over orbit
> states, and gains from grid-specific orbit compressions.

## 5. Recursive Multi-Junction Formula

For cell-topologies with multiple junctions (cell-paths, cell-cycles, general cell-trees, or grids), iterate the formulas of §3 and §4. Each recursion level processes one junction of the cell-topology graph `H`; after `n − 1` reductions of a cell-tree, the sub-problems are single multigraphs that the standalone synthesis pipeline can handle.

**Bridge/cycle classification at each recursion level.** At each level, classify the current junction as a _bridge junction_ (removing all its edges disconnects the current graph) or a _cycle junction_ (alternative paths remain). Use the §2 formula with bridge factor for bridge junctions, and the §4 binomial-only formula for cycle junctions. A `verify_spanning_trees` check on the final result catches precondition violations.

**Cache structure.** Two caches defeat the combinatorial growth:

1. **Leaf cache.** `canonical_key(M) → T(M)` for each unique multigraph `M` reached at a recursion base case.
2. **State cache.** `(canonical_key(g_current), remaining_junction_indices) → T_subproblem` for each unique intermediate recursion state.

On Chimera Cm2 (4 junctions), the state cache provides a 3.2× speedup versus a cacheless recursion, with 124 unique leaves for a state-space of `5^4 = 625` nominal paths.

**Cost.** For a cell-topology graph `H` with `|V(H)| = n` cells and `|E(H)| = m` junctions each of size `k`, the naive recursion is `(k+1)^m` leaves; with the two caches, empirical leaf counts are orders of magnitude smaller. A tight bound is open.

## 6. Empirical Validation

Implementation: `tutte/graphs/covering.py:detect_kmatching_topology` and `apply_kmatching_formula`, wired into `tutte/synthesis/engine.py:_try_formula_shortcircuit` (engine step 7.5), which tries the unified formula then the k-matching formula before falling through to the cell-quotient / chord-rule paths.

| target                      |   cells | junctions |   k | method fires        | T match        |
| --------------------------- | ------: | --------: | --: | ------------------- | -------------- |
| 2 K_3 + M_2                 | 2 × K_3 |         1 |   2 | ✓                   | ✓              |
| K_3 path M_2 (3 cells)      | 3 × K_3 |         2 |   2 | ✓                   | ✓              |
| K_4 path M_2 (3 cells)      | 3 × K_4 |         2 |   2 | ✓                   | ✓              |
| K_3 cycle M_2 (3 cells)     | 3 × K_3 |         3 |   2 | ✗ (precondition P3) | fall-through ✓ |
| Cm1 + M*4 + Cm1 (2 K*{4,4}) | 2 × Cm1 |         1 |   4 | ✓                   | ✓ (143 terms)  |
| Cm2 = dnx.chimera_graph(2)  | 4 × Cm1 |         4 |   4 | ✓                   | ✓ (675 terms)  |

The K_3 cycle row is informative: the small complete-graph cell has only 3 vertices, forcing any two junctions to share anchor vertices. Combined with the cycle topology, this violates (P3). The detector correctly identifies the violation and returns `None`, letting the synthesis pipeline fall through to the chord rule, which handles the case correctly (at higher cost).

### Performance on Cm2

| method                             | wall clock | method tag          |
| ---------------------------------- | ---------: | ------------------- |
| treewidth_dp (direct, C-extension) |      196 s | `treewidth_dp`      |
| k-matching formula                 |       51 s | `kmatching_formula` |

**~4× speedup on Cm2.** The k-matching formula's advantage grows with graph size: direct synthesis is dominated by the width-11 tree-decomposition DP (per-node state ∝ `Bell(12)² ≈ 1.8 × 10¹³`), while the k-matching recursion reduces to 124 multigraph syntheses on graphs of size ≤ 32 nodes, each handled quickly by the standard pipeline.

For Chimera Cm3 (72n 192e, tree-decomposition width 15), direct synthesis is infeasible; the k-matching recursion theoretically applies but the naive implementation hit an exponential state-cache wall due to insufficient symmetry exploitation. Opening a path on Cm3 is the subject of future work.

## 7. Failure Modes and Detector Design

The detector `detect_kmatching_topology` (`covering.py`) enforces the preconditions:

- **(P1 violation).** If any cell-pair junction has repeated anchors on one side (two matching edges sharing a vertex), return `None`. This rules out parallel-edge junctions that aren't true matchings.

- **(P2 violation).** For each cell, induce the cell subgraph; if bipartite, check whether each junction's anchor set lies entirely in one bipartition side. If not, return `None`. Non-bipartite cells are assumed vertex-transitive (true for `K_n`, conservative for others).

- **(P3 violation).** Compute `cycle_basis(H)` on the cell-topology graph. If non-empty (cycle present), check for every cell that its junction anchors are disjoint across junctions. If any cell has two junctions sharing an anchor vertex, return `None`.

Every path where the detector returns `None` falls through to the product formula / chord rule pipeline, which computes the correct polynomial at higher cost.

**Verification.** The engine always applies `verify_spanning_trees(graph, km_poly)` after the formula returns a polynomial. A verification failure triggers a loud log entry and falls through — providing a safety net for any preconditions we haven't captured correctly.

## 8. Why the k-Matching Formula Specializes the Chord Rule

Chord-rule iteration on `k` matching edges (per §8.2, Theorem 4) gives:

```
T(G) = T(G − M_k) + Σ_{i = 1}^{k} T((G − {m_1, …, m_{i-1}}) / m_i).
```

This is `k + 1` synthesis calls — linear in `k`. The k-matching formula of §2 is:

```
T(G) = (x + k − 1) · T(G_1) · T(G_2) + Σ_{j = 2}^{k} C(k, j) · T(G_1 ⊕_j G_2).
```

This is `1 + (k − 1) = k` distinct multigraph syntheses (the first term reuses `T(G_1) · T(G_2)`). Comparing:

- **Chord rule** produces `k` intermediate contraction graphs `(G − {m_1, …, m_{i-1}}) / m_i` that grow in size as `i` decreases (because we delete more edges first, then contract). The `i = 1` graph has the most edges; the `i = k` graph is `(G_1 + M_0 + G_2) / m_k` — essentially `G_1 ⊕_1 G_2`.

- **k-matching formula** produces `k − 1` intermediate graphs `G_1 ⊕_j G_2` for `j = 2, …, k`, which are strictly smaller than the chord-rule intermediates (fewer vertices and edges). Combined via binomial coefficients rather than explicit per-step computation.

The formulas are **mathematically equivalent** — both compute the same polynomial. The k-matching version is preferable when:

1. The `(x + k − 1) · T(G_1) · T(G_2)` leading term can be computed cheaply (especially when `T(G_1)` and `T(G_2)` are in the rainbow table).
2. The `T(G_1 ⊕_j G_2)` multigraphs are smaller than the chord-rule leaves, so each synthesis call is cheaper.
3. For multi-cell topologies, the recursive version exposes symmetry (via vertex-transitivity) that the chord rule does not: a single junction with `k` matching edges collapses from `2^k` chord-rule subsets to `k + 1` binomial classes.

The chord rule is the universal fallback when preconditions fail. The k-matching formula is a specialized dispatch path whose preconditions hold on D-Wave Chimera targets and structurally similar graphs.

## 9. Open Questions

1. **Multi-junction cache compression.** Empirically, the state cache on Cm2 compresses `5^4 = 625` recursion paths into 124 unique leaves. For Cm3 (`5^12 = 2.4 × 10^8`) the naive state cache doesn't fit in RAM in a 1-hour budget. A _symmetry-aware_ state key (quotienting by the `D_4` symmetry of the 3×3 cell-topology grid) could reduce the state space by ~8×; additional savings from leaf-cell isomorphism detection are plausible but not yet characterized.

2. **Pegasus and Zephyr cell decomposition.** structural analysis showed that Pegasus `Pm2, Pm3` and Zephyr `Z(1,2), Z(1,3)` have non-trivial cell structures (cells are not K\_{4,4}; some tiles are oriented in multiple "types") that our standard cell detector misses. If proper cells can be identified and the preconditions verified, the formula may extend to these families.

3. **Beyond matching junctions.** Chimera has `M_k` junctions where all `k` edges connect anchors in a single bipartition class. Pegasus has "mixed" junctions with edges connecting multiple classes. A generalization that tracks _per-edge_ anchor classes (rather than per-junction) would extend the applicability.

4. **Precondition (P2) tightening.** For bipartite cells, "single bipartition side" is a clean precondition. For non-bipartite cells we assume vertex-transitivity of anchors; this is a conservative simplification. A precise characterization of "sufficient cell symmetry" in terms of automorphism-group orbits would clarify what cells the formula supports.

5. **Closed form for `T(G_1 ⊕_j G_2)` on Chimera cells.** The `T(G_1 ⊕_j G_2)` terms for K\_{4,4} cells are bipartite graphs with boundary identifications. Knowing these in closed form (via table lookup or analytic formula) would reduce the k-matching dispatch to `O(k)` polynomial arithmetic for Chimera — potentially making even Cm3-sized targets tractable.

## 10. References

- §8.2 _Chord-Rule Formalization_ — the universal deletion-contraction-based decomposition that the k-matching formula specializes.
- §8.unified _Unified Cell-Topology Formula_ — the parallel-case companion to the k-matching chord case.
- Tutte, W. T. (1947). _A ring in graph theory._ Mathematical Proceedings of the Cambridge Philosophical Society.
- Brylawski, T. H. (1971). _A combinatorial model for series-parallel networks._ Transactions of the AMS.
- Bonin, J. & de Mier, A. (2004). _T-uniqueness of some families of k-chordal matroids._ Advances in Applied Mathematics.
- This codebase: `tutte/graphs/covering.py`, `tutte/synthesis/engine.py`.

---

> **Suggested citation**: "Closed-Form Tutte Polynomial for k-Matching Cell Decompositions with Applications to D-Wave Chimera Graphs" — Quip Network development, April 2026. Source: `tutte/graphs/covering.py:apply_kmatching_formula` and accompanying documentation.
