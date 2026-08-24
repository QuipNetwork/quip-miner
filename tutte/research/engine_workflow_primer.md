# Engine Workflow Primer

A self-contained walkthrough of the Tutte synthesis engine for readers
who know high school algebra but not necessarily graph theory. We
build up vocabulary, define the Tutte polynomial, then walk the engine
pipeline (`tutte/synthesis/engine.py`) in the order the engine actually
tries techniques.

Each section ends with pointers into `tutte/docs/` for the technique
deep-dive and into the source for the implementation.

---

## Part I — Vocabulary

### 1. What is a graph?

A **graph** is a pair `G = (V, E)`:

- **`V`** is a finite set of _vertices_ (also called _nodes_). For us,
  vertices are usually integers `0, 1, 2, …`.
- **`E`** is a set of _edges_. An edge is an unordered pair `{u, v}` of
  two distinct vertices. We write it `(u, v)` with the convention that
  `u < v`.

Pictorially, a graph is dots (vertices) connected by lines (edges). The
graph `G = ({0, 1, 2}, {(0, 1), (1, 2), (0, 2)})` is a triangle.

In the codebase the type is `Graph(nodes, edges)` in `tutte/graph.py`.

A **MultiGraph** allows the same edge to appear more than once (a
_parallel edge_) and allows _loops_ — edges from a vertex to itself.
The `MultiGraph` class in `tutte/graph.py` stores edges as a multiset
`edge_counts: Dict[(u, v), int]`.

### 2. Adjacency, degree, neighbors

Two vertices are **adjacent** if they share an edge. The **degree** of
a vertex `v`, written `deg(v)`, is the number of edges incident to `v`.
The **neighbors** of `v` are the set `N(v) = { u : (u, v) ∈ E }`.

### 3. Walks, paths, cycles

A **walk** is a sequence of vertices `v_0, v_1, …, v_k` where each
consecutive pair `(v_i, v_{i+1})` is an edge.

- A walk with no repeated vertices is a **path**.
- A walk where `v_0 = v_k` and no other vertex repeats is a **cycle**.

The **path graph on `n` vertices**, `P_n`, is a single path. The
**cycle graph on `n` vertices**, `C_n`, is a single cycle.

### 4. Connectedness, components, disconnected graphs

A graph is **connected** if you can reach any vertex from any other by
walking along edges. If a graph is not connected, it splits into
maximal connected pieces called **connected components**. A
disconnected graph factors trivially under the Tutte polynomial — see
Stage 4 of the pipeline.

### 5. Subgraphs

A **subgraph** of `G = (V, E)` is a graph `G' = (V', E')` with
`V' ⊆ V` and `E' ⊆ E` (and every edge of `G'` only uses vertices in
`V'`).

- **Spanning subgraph**: `V' = V`. Same vertices, possibly fewer edges.
- **Induced subgraph on `V'`**: take exactly the edges of `G` whose
  endpoints both lie in `V'`. Notation: `G[V']`.

### 6. Trees, spanning trees, forests

A **tree** is a connected graph with no cycles. A graph with no cycles
(but possibly disconnected) is a **forest** — every connected
component is a tree.

A **spanning tree** of `G` is a spanning subgraph that is also a tree.
Every connected graph has at least one spanning tree.

**Kirchhoff's matrix-tree theorem** (Kirchhoff, 1847) counts spanning
trees as a determinant of the graph Laplacian. The engine uses this as
a sanity check: `T(1, 1) =` number of spanning trees, computable in
`O(n³)` via determinant. See
`tutte/validation.py::count_spanning_trees_kirchhoff`.

### 7. Cut vertices and bridges

- A **cut vertex** (or _articulation point_) is a vertex whose removal
  disconnects the graph.
- A **bridge** is an edge whose removal disconnects the graph.

In a tree, every edge is a bridge and every non-leaf vertex is a cut
vertex. In a cycle, none are. Cut vertices and bridges are the
canonical "weak points" of a graph; the engine factors at every cut
vertex (Stage 5).

### 8. Series, parallel, deletion, contraction

Two edges are in **series** if they share a vertex of degree 2 with no
other neighbors — like two resistors in series. Replacing a degree-2
vertex `v` and its two edges `(u, v), (v, w)` by a single edge `(u, w)`
is a **series reduction**.

Two edges between the same pair of vertices are **parallel**. Replacing
them by a single edge is a **parallel reduction**.

A graph that can be reduced to a single edge by repeated series and
parallel reductions is **series-parallel** (SP).

Other operations:

- **Edge deletion** `G − e`: remove edge `e`, keep its endpoints.
- **Edge contraction** `G / e`: shrink edge `e` to a point — merge its
  two endpoints, re-route their other edges to the merged vertex,
  delete `e` itself. Parallel edges and loops created by the merge are
  kept (they appear in the multigraph view).
- **Disjoint union** of `G_1` and `G_2`: both graphs side by side with
  no edges between them. Notation: `G_1 ∪ G_2`.

### 9. Cliques and complete graphs

A **clique** is a set of vertices that are pairwise adjacent. The
**complete graph on `n` vertices**, `K_n`, has every possible edge:
`|E(K_n)| = n(n − 1) / 2`.

The **complete bipartite graph** `K_{a,b}` has `a + b` vertices split
into two sides (size `a` and size `b`) with every left vertex adjacent
to every right vertex (and no edges within a side).

### 10. Isomorphism and automorphism

Two graphs `G_1 = (V_1, E_1)` and `G_2 = (V_2, E_2)` are
**isomorphic** if there is a bijection `σ: V_1 → V_2` that maps edges
to edges. Informally: same shape, just relabeled vertices.

A bijection `σ: V → V` from a graph to itself that preserves edges is
an **automorphism**. The set of all automorphisms forms the
**automorphism group** `Aut(G)`. For `K_n`, `Aut(K_n)` is the full
symmetric group `S_n`. Automorphism groups matter to the engine
because they let us identify equivalent sub-problems and cache once —
the cell-quotient DPs (Stages 6.8-6.10) rely on this.

---

## Part II — The Tutte Polynomial

### 11. Definition by deletion-contraction

The **Tutte polynomial** `T(G; x, y)` is a two-variable polynomial
defined recursively. Introduced by W. T. Tutte (1947) building on
earlier work by Hassler Whitney (1932).

| Case                                       | Rule                         |
| ------------------------------------------ | ---------------------------- |
| `E` is empty                               | `T(G) = 1`                   |
| `e` is a bridge                            | `T(G) = x · T(G − e)`        |
| `e` is a loop                              | `T(G) = y · T(G − e)`        |
| `e` is neither a bridge nor a loop         | `T(G) = T(G − e) + T(G / e)` |

A nontrivial theorem says the definition does not depend on the order
in which edges are processed. The same polynomial comes out either way.

### 12. Evaluations

The Tutte polynomial encodes a remarkable amount of structural
information:

| Evaluation | Counts                          |
| ---------- | ------------------------------- |
| `T(1, 1)`  | spanning trees                  |
| `T(2, 1)`  | spanning forests                |
| `T(1, 2)`  | connected spanning subgraphs    |
| `T(2, 0)`  | acyclic orientations            |
| `T(0, 2)`  | strongly connected orientations |

For example, `T(C_3; x, y) = x² + x + y`, so `T(C_3; 1, 1) = 3`, which
is the number of spanning trees of a triangle. The chromatic and
reliability polynomials are also evaluations after a change of
variable.

### 13. Why it matters here

The Quip Protocol uses the Tutte polynomial as a structural difficulty
measure for Ising-model proof-of-work — see
`tutte/docs/00_motivation.md`. Computing `T(G)` is `#P`-hard in
general (Jaeger, Vertigan & Welsh, 1990), but verifying many of its
properties is cheap, which is what we want from a proof-of-work
mechanism.

The engine's job is to compute `T(G)` as fast as possible for the
kinds of graphs that come out of quantum hardware topologies (D-Wave
Chimera, Pegasus, Zephyr). The pipeline is a sequence of fast paths,
each cheaper than the next, each catching a structural shape the
engine can handle without falling all the way to brute-force
deletion-contraction.

---

## Part III — The Pipeline

The engine in `tutte/synthesis/engine.py::_synthesize_inner` tries
techniques in roughly increasing order of cost. Each technique either
returns a polynomial and stops, or returns `None` and the next
technique gets a turn. The full per-stage flowchart lives in
`tutte/docs/README.md`.

### Stage 1 — Family Recognition

**Doc**: `tutte/docs/01_family_recognition.md`. **Cost**: `O(n + m)`.

Some graph families have a **closed-form** Tutte polynomial — a single
formula in `n` or `(m, n)`:

- A tree on `n` vertices: `T = x^{n−1}`.
- A cycle on `n` vertices: `T = x^{n−1} + x^{n−2} + … + x + y`.
- Wheels, ladders, fans, prisms, books, gears, Möbius ladders, grids:
  all have known closed forms or constant-coefficient recurrences.

Recognition checks "is this graph one of these named shapes?" by
degree sequence + small structural fingerprint. If yes, the polynomial
is returned without ever computing the canonical key. This is the
fastest path in the engine and runs before the expensive
canonical-key computation.

### Stage 1.5 — Transfer Matrix (periodic lattice strips)

**Doc**: `tutte/docs/01_1_transfer_matrix.md`. **Cost**: `O(V + E)`
detection + `O(length × Catalan(width)²)` sweep.

Handles grids (`m ≥ 3`), triangular, honeycomb, square-octagon, and
elongated-triangular strips that family recognition doesn't cover.
Uses the Fortuin-Kasteleyn random-cluster transfer matrix on
non-crossing partition states; a C extension accelerates the column
sweep. Runs before canonical-key so lattice strips never pay that cost.
Implemented in `tutte/transfer_matrix/`.

### Stage 2 — Rainbow Table Lookup

**Doc**: `tutte/docs/02_rainbow_table_lookup.md`. **Cost**: dominated by
canonical-key computation (roughly `O(n² × d)`).

Given two isomorphic graphs they have the same Tutte polynomial. So if
we have already computed `T(G_1)` and stored it, we can look up
`T(G_2)` instantly — _if_ we have a way to recognize that two graphs
are isomorphic.

The engine uses a **canonical key**: a function that maps any graph to
a string such that isomorphic graphs always get the same string. The
key is built from the **Weisfeiler-Leman algorithm** (Weisfeiler &
Leman, 1968), which iteratively refines vertex colors based on
neighbor colors.

The rainbow table itself (`tutte/data/lookup_table.bin`) is a
precomputed dictionary from canonical keys to Tutte polynomials. It
includes the small named graphs and grows over time by saving every
successfully synthesized polynomial.

### Stage 3 — Base Cases

**Doc**: `tutte/docs/03_base_cases.md`. **Cost**: `O(1)`.

- Empty graph (no edges): `T = 1`.
- Single-edge graph: `T = x`.

Self-loops contribute a factor of `y` per loop; parallel edges fold
into the parallel-edge formula at the multigraph level.

### Stage 4 — Disconnected Factorization

**Doc**: `tutte/docs/04_disconnected_factorization.md`. **Cost**:
`O(n + m)` plus recursive synthesis on each component.

A disjoint union factors:

```
T(G_1 ∪ G_2) = T(G_1) · T(G_2)
```

### Stage 5 — Cut Vertex Factorization

**Doc**: `tutte/docs/05_cut_vertex_factorization.md`. **Cost**:
`O(n + m)` plus recursive synthesis per **block**.

If `v` is a cut vertex, the graph splits into "blocks" — maximal
subgraphs with no cut vertex of their own — that share `v` at the
seams. The Tutte polynomial multiplies across blocks:

```
T(G_1 · G_2) = T(G_1) · T(G_2)        (G_1, G_2 share exactly one vertex)
```

This is a special case of the general 1-sum identity (Brylawski,
1971). The engine finds all cut vertices in `O(n + m)` (Tarjan, 1972)
and recursively synthesizes each block.

### Stage 6 — Series-Parallel Fast Path

**Doc**: covered alongside Stage 8 in
`tutte/docs/08_2_chord_rule_formalization.md`. **Cost**: `O(n + m)`.

If the graph is series-parallel (treewidth ≤ 2), an SP-tree
decomposition (`tutte/graphs/series_parallel.py`) gives the Tutte
polynomial directly without deletion-contraction.

### Stage 7 — Cell-Quotient and Closed-Form Paths (high-volume D-Wave gate)

When the graph has at least 60 edges, the engine tries several
cell-decomposition-aware paths in order. They share a precondition:
`try_hierarchical_partition` (in `tutte/graphs/covering.py`) returns a
**cell decomposition** — a partition of vertices into pieces, each
isomorphic to a known small graph (a "cell"), connected by inter-cell
**junctions** (e.g., `M_k` perfect matchings).

The order matters: closed-form formulas are tried before iterative
DPs.

#### 7.45 Cell-Quotient Grid DP (streamed)

**Doc**: `tutte/docs/06_5_cell_quotient_grid_dp.md`. Implemented in
`tutte/roots/cell_quotient_grid.py`.

For cell-decomposable graphs whose cell-quotient is a 2D grid of
`K_{a,b}`-style cells connected by `M_k` matchings with **disjoint
per-direction anchors**. Cm_2 fits cleanly. The streaming row
composition keeps the per-chunk memory bounded.

#### 7.5 Formula short-circuit

The engine first looks for a closed-form match against the **unified
formula** or the **k-matching formula** before falling through to any
DP path. See `_try_formula_shortcircuit` in
`tutte/synthesis/engine.py`.

- **Unified formula**: when every cell-pair shares a single
  vertex-pair connection, `T(G) = (∏ T(cell_i)) · T(H)` where `H` is
  the cell-topology graph (one node per cell, one edge per inter-cell
  edge). Generalizes cut-vertex, bridge, and parallel-edge
  factorizations. Doc: `tutte/docs/08_hierarchical_tiling.md`.

- **k-matching formula**: when each junction is a `k`-edge matching
  between vertex-transitive cells,
  `T(G_1 + M_k + G_2) = (x + k − 1) · T(G_1) · T(G_2) +
  Σ_{j=2}^{k} C(k, j) · T(G_1 ⊕_j G_2)`. Validated on K_3, K_4, K_5,
  and `K_{4,4}` (D-Wave Chimera cell), so Cm_2-class targets land
  here. Doc: `tutte/docs/08_3_kmatching_formula.md`.

- **Sokal-Z generalized chord-junction** (2-cell partitions only):
  when the chord junction has non-matching / multi-edge / dense `E_J`
  that the unified theorem can't accept, fall through to the Sokal Z
  basis formula `Z(G_1 ⊕_{E_J} G_2; q, v) = Σ_{A_J ⊆ E_J} v^|A_J|
  Z(merger(φ(A_J)); q, v)`. Enumeration over `A_J` is by
  per-H_J-component edge-by-edge tree DP (O(|E_c| · Bell(|V_c|))) +
  Aut-orbit compression of φ partitions. Result converts back to
  T(x, y) via multi-point evaluation + bivariate Lagrange. Handles
  Z(1,2)-class junctions (32 chord edges over 24 anchors) once
  per-component gates are tuned for the size; the algorithm itself
  scales to component sizes ≥ 32 edges in seconds. Doc:
  `tutte/docs/08_6_unified_chord_junction.md` §9 and research note
  `tutte/research/cyclotomic_chord_junction_theorem.md` § EXTENSION.

#### 7.5b Cotree DP

**Doc**: `tutte/docs/06_1_cotree_dp.md`. **Cost**:
`exp(O(n^{2/3}))` — subexponential.

A **cograph** is built from single vertices using only disjoint union
and complete join. The Tutte polynomial of a cograph on `n` vertices
can be computed subexponentially (Giménez, Hliněný, Noy 2006), which
beats treewidth_dp when the graph is dense and `tw > 11`. Implemented
in `tutte/cotree_dp/`.

#### 7.55 Small-graph treewidth_dp short-circuit

For small graphs (`n ≤ 20`, `m ≥ 10`), the engine tries
`treewidth_dp(max_width=8)` before almost-cograph. Many small dense
graphs (Petersen, Heawood) finish in milliseconds via treewidth DP and
would otherwise pay multi-second chord-rule overhead in
almost-cograph.

#### 7.6 Almost-cograph DP

**Doc**: `tutte/docs/06_2_almost_cograph_dp.md`. **Cost**: 1 cotree DP
+ `O(|A|)` recursive syntheses, where `|A|` is the anomaly-edge count.

An **anomaly edge set** `A ⊆ E(G)` is one whose removal leaves a
cograph. The engine uses a greedy `P_4`-elimination procedure to find
a small `A` (capped at 16 by `compute_tutte_almost_cograph`), then
applies the chord rule to peel off one anomaly at a time, calling
cotree DP on each residual.

#### 7.7 Cell-Quotient Cycle DP

**Doc**: `tutte/docs/06_4_cell_quotient_cycle_dp.md`. Implemented in
`tutte/roots/cell_quotient_cycle.py`.

For cell-decomposable graphs whose cell-quotient is a **simple cycle**
(e.g., D-Wave Cm_2). Combines `T_rooted` of cells via vertex-sum
convolution at each junction, then closes the cycle with an
identification step. Generic over junction connectivity — handles
`M_k` matchings and `K_{a,b}` bipartite junctions through the
auto-detected component count `c_J`.

#### 7.8 Cell-Quotient Tree DP

**Doc**: `tutte/docs/06_6_cell_quotient_tree_dp.md`. Implemented in
`tutte/roots/cell_quotient_tree.py`.

Generalizes the cycle DP to **arbitrary tree topology** in the
cell-quotient graph (`n` cells, `n − 1` junctions, no cycles). Composes
`T(graph)` by post-order recursion over the cell tree, with per-cell
orbit compression for K_{a,b}-style cells.

##### 7.81 Special case: cell-tree is a path (chain recurrence)

When the cell-quotient tree degenerates to a **linear path**
(`cell ⊕_M cell ⊕_M … ⊕_M cell` for `n` identical cells joined by a
fixed connector), the tree DP exposes more algebraic structure than
the general case. From first principles:

1. The path DP carries a per-cell **state vector** indexed by
   boundary-partition orbits under the cell's automorphism group
   acting on its connector anchors. Call the orbit count `r`.
2. Composing one more cell+junction step is a **linear map** on this
   `r`-dimensional space whose entries are themselves polynomials in
   `(x, y)`. Call the map `M(x, y)`.
3. Cayley-Hamilton on `M` over the polynomial ring `Z[x, y]` produces
   a characteristic polynomial `char(λ) = λ^r + c_1 λ^{r-1} + … + c_r`
   with `c_i ∈ Z[x, y]`. Multiplying through by the chain state vector
   gives an **exact linear recurrence in `n`** for the raw state-sum
   `S_n = T(chain_n) · (x − 1)^{total_div(n)}`:

   ```
   S_{n+r} + c_1(x, y) · S_{n+r−1} + … + c_r(x, y) · S_n = 0
   ```

This is an explicit, constructive re-derivation of Noy & Ribò (2007).
The recurrence-order theorem `r = n_orbits` is empirical (validated on
five templates: `K_{2,2}+M_2`, `K_{3,3}+M_3`, `K_{4,4}+M_4`,
`K_4+M_2`, `K_5+M_2`).

**Practical impact**: once `M` is extracted from one cell-quotient
step, evaluating `T(chain_n; x_0, y_0) mod p` at any point costs
`O(r³)` setup (Faddeev-LeVerrier mod p for the char poly) plus
`O(n)` modular multiplications. For `K_{4,4}+M_4` chains: ~3-9 ms per
modular point at `n = 100`, ~4000× faster than direct DP at 10× the
length.

**Implementation**: `tutte/roots/chain_recurrence.py` provides
`extract_chain_transfer_matrix`, `compute_chain_recurrence_mod`, and
`is_chain_topology(spec)`. Engine dispatch in
`compute_cell_quotient_tree_dp` short-circuits to the chain path when
the cell-tree is detected as linear. Deep dive:
`tutte/docs/06_7_chain_recurrence_algebra.md`.

The same construction extends to **cell-quotient cycles** (closing
the chain) with a higher recurrence order — empirically
`order_cycle ≈ 2.5 · order_chain + const`. Validated symbolically on
the `K_{2,2}+M_2` cycle (order 5). The 2D-grid generalisation is
open; see "Ongoing research directions" in
`tutte/research/literature_search.md`.

#### 7.82 Cell-Quotient Bipartite-Junction DP

Implemented in `tutte/roots/cell_quotient_bipartite_junction.py`.

Generalization of the k-matching path that accepts non-matching
bipartite junctions (asymmetric anchor degrees, disconnected junction
subgraphs). Unblocks Z(m, t) families whose inter-cell graph has
multi-degree anchors.

#### 7.83 Per-component bipartite-junction DP

When the standard bipartite-junction DP would face a Bell-number wall
on the joint boundary partition, this variant factors a disconnected
junction into its connected components and processes each as a
separate convolution step.

#### 7.85 Cell-Quotient Hybrid DP

Implemented in `tutte/roots/cell_quotient_hybrid.py`.

Chord-rule cycle-close + per-leaf synthesis for cyclic cell-quotients
(D-Wave Cm_3's 3×3 grid). Recursively peels closing junctions; each
leaf is synthesized via the engine's standard pipeline.

### Stage 8 — Treewidth Dynamic Programming

**Doc**: `tutte/docs/06_treewidth_dp.md`. **Cost**: `O(2^tw × n)` via
a C extension.

A **tree decomposition** of `G` is a tree `T` plus a _bag_ `B_t ⊆ V(G)`
for each node `t` of `T` such that:

1. Every vertex of `G` appears in at least one bag.
2. Every edge of `G` has both endpoints in at least one common bag.
3. For any vertex `v ∈ V(G)`, the bags containing `v` form a connected
   subtree of `T`.

The **width** of a tree decomposition is `max_t |B_t| − 1`. The
**treewidth** `tw(G)` is the minimum width over all tree decompositions.
Treewidth is a structural parameter introduced by Robertson & Seymour
(1984). Many `NP`-hard problems become tractable on graphs of bounded
treewidth, including the Tutte polynomial (Andrzejak, 1998; Noble,
1998).

Given a tree decomposition of width `w`, the algorithm walks the tree
from leaves to root. For each bag it maintains a table indexed by
_partial states_ — partitions of the bag's vertices encoding which
subgraph components are connected — and contributes coefficients to
the running polynomial. Table size is bounded by `Bell(w + 1)`.

The engine uses a C extension (`tutte/graphs/_treewidth_c.py`), gated
to `5 ≤ tw ≤ 10` (where head-to-head measurements show the C path
wins over the pure-Python wrapper). Python fallback covers
`tw ≤ 11`. For graphs whose coefficients exceed `int64`, a
modular-CRT path activates above 62 edges.

#### 8.5 σ-equivariant decomposition via signed-graph DP

When the input graph admits a known **order-2 automorphism** `σ ∈ Aut(G)`
(common in D-Wave topologies — `Z(m, t)` cells often have a cell-swap σ
that's free of fixed edges), the Tutte polynomial decomposes by σ-action
on edge subsets:

```
T(G; x, y) = T_fix^σ(G; x, y) + T_free^σ(G; x, y)
```

where `T_fix^σ` sums over σ-invariant subsets (orbits of size 1) and
`T_free^σ` sums over σ-paired subsets (orbits of size 2). For graphs
with large `Aut(G)` and intractable treewidth, this can compute the
**fixed half on the quotient graph** at a fraction of the cover-graph
cost.

**First-principles core (20 lines).** Form the quotient
`G_base = G / ⟨σ⟩` whose vertices are σ-orbits and whose edges are
σ-orbits of edges. When σ acts freely on edges, this is a **2-fold
cover**: each base edge lifts to two cover edges. A monodromy character
`χ : E(G_base) → Z_2` records, for each base edge, whether the lift
preserves or swaps the two sheets. Then for any σ-invariant subset
`A_L ⊆ E(G)` (lifted uniquely from a base subset `L ⊆ E(G_base)`):

```
|A_L|        = 2 |L|                                     (free case)
r_G(A_L)     = r_quot(L) + r_signed(L, χ)                (lift identity)
```

Here `r_signed` is **Zaslavsky's frame-matroid rank function** on the
signed graph `(G_base, χ)` — viewing χ as edge signs and treating
unbalanced cycles (sign-sum ≠ 0 mod 2) as contributing one extra unit
of rank (Zaslavsky 1982). Substituting into the Whitney form gives

```
T_fix^σ = Σ_L (x − 1)^{r(E_G) − r_quot(L) − r_signed(L)}
              · (y − 1)^{2|L| − r_quot(L) − r_signed(L)}
```

— a polynomial summable by a single **elimination-order DP on the
quotient** that tracks the standard partition state plus per-block sign
balance. For `Z(1, 2)`, the cover has 24 vertices and 76 edges; the
quotient under cell-swap σ has 12 vertices and 38 edges. The quotient
DP fits comfortably where the cover doesn't.

**Non-free covers.** When σ fixes some individual edges of `G`, those
edges become **loops** in the quotient. The same lift identity holds
with `|A_L| = 2|L| − |L_loop|`, and the per-edge DP multipliers gain a
1-power adjustment on loop edges. Validated on `K_4 + (01)(23)` and
`K_{3,3}` part-swap.

**Implementation.** The high-level API lives in
`tutte/roots/signed_quotient.py`:

- `build_quotient_with_monodromy(g, perm)` — extracts
  `(V_quot, E_quot, χ)` from `(G, σ)`. Handles free and non-free σ.
- `evaluate_t_signed_mod(nodes, edges_with_signs, x, y, p)` —
  Zaslavsky's signed-graph Tutte at one `(x, y, p)` point. Wraps
  `signed_elim_dp.compute_signed_tutte_elim_mod`.
- `compute_t_fix_sigma_quotient_mod(g, perm, x, y, p)` — σ-invariant
  Tutte of the cover `G` at one point, computed on the quotient via
  the lift identity.
- `derive_t_free_sigma_mod(g, perm, x, y, p, engine=None)` —
  σ-paired Tutte half, derived as `T(G) − T_fix^σ` when `T(G)` is in
  the engine's lookup table. Useful for cross-validation.
- `interpolate_t_signed_mod(...)` — multi-point evaluation + bivariate
  Lagrange interpolation to recover the polynomial.
- `zephyr_cell_swap_perm(m)` — convenience helper for `Z(m, 2)`
  cell-swap σ.

Low-level point-evaluation DPs live in `tutte/graphs/signed_elim_dp.py`
and `tutte/graphs/sigma_equivariant_dp.py`.

**Status.** `T_fix^σ` is computable end-to-end. `T_free^σ` —
the σ-paired half — has four candidate computation approaches
documented in `tutte/docs/06_9_signed_equivariant_dp.md`; none have
beaten the direct engine path for `Z(1, 2)` yet. The framework is the
candidate path for adding `Z(1, 3)`, `Cm_3`, and larger D-Wave graphs
to the lookup table once a `T_free^σ` recovery formula or a sufficiently
fast direct cover-side DP lands.

Deep dive: `tutte/docs/06_9_signed_equivariant_dp.md`. Related lit:
the "Signed graphs and equivariant matroids" section of
`tutte/research/literature_search.md`.

### Stage 9 — k-Sum Decomposition (chord rule)

**Doc**: `tutte/docs/07_k_sum_decomposition.md`. **Cost**: `1 + C(k, 2)`
full syntheses.

A **k-vertex separator** is a set of `k` vertices whose removal
disconnects the graph. Treewidth is essentially the minimum size of
the worst separator over a recursive decomposition.

For a graph with a `k`-vertex separator `S`, add any missing `K_k`
clique edges among `S` (call the result `PC` for "parallel
completion"), then peel them off one at a time via the chord rule:

```
T(G) = T(PC) − Σ_i T((PC − e_1 − … − e_{i−1}) / e_i)
```

If `S` already contains the full `K_k` clique (no edges missing), the
engine peels the existing clique edges instead — same cost.

For `k = 7`, that's `1 + 21 = 22` syntheses on smaller graphs.
Worthwhile when the smaller recursive calls hit the rainbow table or
treewidth_dp. Implemented in `tutte/graphs/k_sum.py::clique_chord_k_sum`.

### Stage 10 — Hierarchical Tiling (chord rule)

**Doc**: `tutte/docs/08_hierarchical_tiling.md`. **Cost**:
`O(chord_count)` full syntheses.

When a graph is cell-decomposable but neither the cell-quotient DPs
nor the closed-form formulas apply, the engine falls back to the
**boundary quotient + chord recursion** approach. The boundary
subgraph `B` is the induced subgraph on every cell's boundary
vertices plus all inter-cell edges plus intra-cell edges between
boundary vertices. When the inter-cell graph is a tree:

```
T(G) = (∏_i T(C_i)) · T(B) / (∏_i T(B_i))
```

When the inter-cell graph has cycles, the chord rule peels off
cycle-creating chords one at a time. Implemented in
`tutte/graphs/k_sum.py::boundary_quotient_tutte`.

### Stage 11 — Creation-Expansion-Join (CEJ)

**Doc**: `tutte/docs/09_creation_expansion_join.md`. **Cost**:
`O(chords × synthesis_cost)`.

The final fallback. Pick a spanning tree of `G`, compute its Tutte
polynomial trivially (`x^{n − 1}`), then add the non-tree edges one at
a time:

```
T(G + e) = T(G − e) + T(G / e)
```

Slow but always terminates and always gives the right answer.

---

## Part IV — Algebraic shortcuts and modular evaluation

Several closed-form formulas fire above the pipeline when applicable,
replacing entire DP recursions with a single polynomial
multiplication.

### Unified formula

```
T(G) = (∏_i T(cell_i)) · T(H)
```

holds whenever `G` decomposes into cells joined by single-vertex-pair
connections. `H` is the cell-topology graph. Generalizes cut-vertex
(Stage 5), bridge, and parallel-edge factorizations into one
statement. See `tutte/docs/08_hierarchical_tiling.md`.

### k-matching formula

```
T(G_1 + M_k + G_2) = (x + k − 1) · T(G_1) · T(G_2)
                   + Σ_{j=2}^{k} C(k, j) · T(G_1 ⊕_j G_2)
```

holds for vertex-transitive cells joined by a `k`-edge perfect
matching `M_k`. Validated on K_3, K_4, K_5, and `K_{4,4}` (D-Wave
Chimera cell). See `tutte/docs/08_3_kmatching_formula.md`.

### Chain recurrence

See §7.81 above for the first-principles derivation. Practical impact:
modular `T(chain_n; x_0, y_0) mod p` evaluation in `O(n)` modular
multiplications after `O(r³)` setup, regardless of `n`. Implementation
in `tutte/roots/chain_recurrence.py`; deep dive in
`tutte/docs/06_7_chain_recurrence_algebra.md`.

### Modular point-value pathways

For graphs whose coefficients overflow `int64` mid-DP, the engine
evaluates `T(G; x_0, y_0) mod p` for many integer points and primes,
recovers `T(G; x, y) mod p` via Lagrange interpolation per prime, then
combines across primes via CRT. Doc:
`tutte/docs/06_8_modular_arithmetic_pathways.md`; implementation in
`tutte/roots/interpolation.py` and the modular variants in
`tutte/roots/cell_quotient_grid.py`.

---

## Part V — Things to read next

In approximate order from least to most technical:

1. `tutte/docs/00_motivation.md` — why we care about all this.
2. `tutte/docs/README.md` — pipeline overview with the flowchart and
   per-stage links.
3. `tutte/docs/01_family_recognition.md` through
   `09_creation_expansion_join.md` — one per pipeline stage.
4. `tutte/roots/README.md` — module overview for the cell-quotient DPs.
5. `tutte/research/literature_search.md` — catalog of Tutte-polynomial
   papers organized by direction (includes open research directions:
   D-Wave family symmetry, Möbius inversion / Burnside framework,
   cycle/2D recurrence extensions).
6. `tutte/docs/06_7_chain_recurrence_algebra.md` — chain & cycle
   recurrence algebra, validation matrix, Faddeev-LeVerrier modular
   pathway.
7. `tutte/docs/06_9_signed_equivariant_dp.md` — signed-graph DP and
   σ-equivariant decomposition deep dive (Zaslavsky frame matroid,
   `T_fix^σ` derivation, open `T_free^σ` directions).

---

## Bibliography

References listed in roughly the order they appear above.

### Foundational

- Whitney, H. (1932). "The coloring of graphs." _Annals of Mathematics_.
  _(rank polynomial; precursor to the Tutte polynomial)_
- Tutte, W. T. (1947). "A ring in graph theory." _Math. Proc. Cambridge
  Philos. Soc._ **43**(1), 26–40.
- Tutte, W. T. (1954). "A contribution to the theory of chromatic
  polynomials." _Canadian J. Math._ **6**, 80–91.

### Computational complexity

- Jaeger, F.; Vertigan, D. L.; Welsh, D. J. A. (1990). "On the
  computational complexity of the Jones and Tutte polynomials."
  _Math. Proc. Cambridge Philos. Soc._ **108**(1), 35–53.
- Andrzejak, A. (1998). "An algorithm for the Tutte polynomials of
  graphs of bounded treewidth." _Discrete Mathematics_ **190**(1-3),
  39–54.
- Noble, S. D. (1998). "Evaluating the Tutte polynomial for graphs of
  bounded tree-width." _Combin. Probab. Comput._ **7**(3), 307–321.

### Treewidth and tree decompositions

- Robertson, N.; Seymour, P. D. (1984). "Graph minors III: Planar
  tree-width." _J. Combin. Theory Ser. B_ **36**(1), 49–64.
- Bodlaender, H. L. (1996). "A linear-time algorithm for finding
  tree-decompositions of small treewidth." _SIAM J. Comput._ **25**(6),
  1305–1317.

### Cographs and clique-width

- Corneil, D. G.; Lerchs, H.; Stewart Burlingham, L. (1981).
  "Complement reducible graphs." _Discrete Applied Mathematics_ **3**(3),
  163–174.
- Courcelle, B.; Olariu, S. (2000). "Upper bounds to the clique-width
  of graphs." _Discrete Applied Mathematics_ **101**(1–3), 77–114.
- Giménez, O.; Hliněný, P.; Noy, M. (2006). "Computing the Tutte
  polynomial on graphs of bounded clique-width." _SIAM J. Discrete Math._
  **20**(4), 932–946.

### k-vertex sums and matroid theory

- Brylawski, T. H. (1971). "A combinatorial model for series-parallel
  networks." _Trans. Amer. Math. Soc._ **154**, 1–22.
- Oxley, J. G. (2011). _Matroid Theory_ (2nd ed.). Oxford University
  Press.

### Multivariate Tutte / Potts model

- Sokal, A. D. (2005). "The multivariate Tutte polynomial (alias Potts
  model) for graphs and matroids." In: _Surveys in Combinatorics, 2005_.
  `arXiv:math/0503607`.

### Linear recurrences for Tutte polynomials

- Noy, M.; Ribò, A. (2007). "Linear Recurrence Relations for Graph
  Polynomials." In: _Algebraic Combinatorics and Computer Science: A
  Tribute to Gian-Carlo Rota_, Springer LNCS.
- Fischer, E.; Makowsky, J. A. (2008). "Linear recurrence relations
  for graph polynomials." (Extension to all MSOL-definable graph
  polynomials.)
- Kotek, T.; Makowsky, J. A.; Ravve, E. R. (2013). "Recurrence
  relations for graph polynomials on bi-iterative families of graphs."
  _European J. Combin._; `arXiv:1309.4020`.

### Transfer matrices for Tutte/Potts on lattice strips

- Chang, S.-C.; Shrock, R. (2005). "Transfer Matrices for the
  Partition Function of the Potts Model on Cyclic and Möbius Lattice
  Strips." _Physica A_ **347**, 314–352; `arXiv:cond-mat/0404524`.
- Beaudin, L.; Ellis-Monaghan, J.; Pangborn, G.; Shrock, R. (2010).
  "Tutte polynomials of bracelets." _J. Algebraic Combin._ **32**,
  393–408.

### Foundational algorithms

- Sun Tzu (~3rd century CE); Gauss, C. F. (1801). _Disquisitiones
  Arithmeticae_. _(Chinese Remainder Theorem)_
- Kirchhoff, G. (1847). "Über die Auflösung der Gleichungen…"
  _Annalen der Physik_. _(matrix-tree theorem)_
- Tarjan, R. (1972). "Depth-first search and linear graph algorithms."
  _SIAM J. Comput._ **1**(2), 146–160.
- Weisfeiler, B.; Leman, A. A. (1968). "A reduction of a graph to a
  canonical form and an algebra arising during this reduction."
  _Nauchno-Technicheskaya Informatsia_.

### D-Wave topologies

- Boothby, K.; Bunyk, P.; Raymond, J.; Roy, A. (2020).
  "Next-Generation Topology of D-Wave Quantum Processors."
  `arXiv:2003.00133`.
- D-Wave Systems Inc. _Zephyr Topology of D-Wave Quantum Processors_.
  Technical report.

### Signed / equivariant matroids

- Zaslavsky, T. (1982). "Signed graphs." _Discrete Applied Mathematics_
  **4**, 47–74.
- Zaslavsky, T. (1991). "Biased graphs II." _J. Combin. Theory Ser. B_
  **51**, 46–72.
- Kamiya, H.; Miyamoto, K.; Yoshinaga, M. (2017). "G-Tutte Polynomials
  and Abelian Lie Group Arrangements." _Int. Math. Res. Notices_.

### Recent (2024–2025)

- Yardim, B.; Türker, T. (2025). "Linear-time exact Potts on
  series-parallel graphs." `arXiv:2507.22579`.
- Blažej, V.; Jana, S.; Ramanujan, M. S. (2025). "Cograph-modular
  treewidth." IPEC 2025.
