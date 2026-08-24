# Literature catalog

Catalog of papers relevant to the Tutte synthesis engine: techniques
the engine implements, techniques that could replace or accelerate
existing paths, and structural results that underpin the cell-quotient
DPs.

Each entry gives the citation, a one-line summary, and whether/where
the result is wired into the codebase.

---

## Foundational

### Tutte (1947, 1954) — The polynomial itself

> Tutte, W. T. "A ring in graph theory." _Math. Proc. Cambridge Philos.
> Soc._ **43**(1), 26–40 (1947).
>
> Tutte, W. T. "A contribution to the theory of chromatic polynomials."
> _Canadian J. Math._ **6**, 80–91 (1954).

Defines the two-variable polynomial `T(G; x, y)` via deletion-
contraction. Foundation for the entire engine.

### Whitney (1932) — Rank polynomial

> Whitney, H. "The coloring of graphs." _Annals of Mathematics_ (1932).

Precursor to the Tutte polynomial via the rank function.

### Jaeger, Vertigan & Welsh (1990) — Complexity

> Jaeger, F.; Vertigan, D. L.; Welsh, D. J. A. "On the computational
> complexity of the Jones and Tutte polynomials." _Math. Proc. Cambridge
> Philos. Soc._ **108**(1), 35–53 (1990).

`#P`-hardness in general. Motivates the engine's structural fast paths.

### Sokal (2005) — Multivariate Tutte polynomial

> Sokal, A. D. "The multivariate Tutte polynomial (alias Potts model)
> for graphs and matroids." In: _Surveys in Combinatorics, 2005_.
> `arXiv:math/0503607`.

Defines `Z(G; q, v) = Σ_{A ⊆ E} q^{k(A)} ∏_{e ∈ A} v_e` (one variable
per edge plus a global `q`). Recovers ordinary Tutte via
`T(G; x, y) = (x − 1)^{−r(E)} (y − 1)^{−|V|+1} Z(G; (x−1)(y−1), y−1)`.

The two key algebraic tools are:

- **Parallel composition** (two parallel edges, weights w_1, w_2):
  `w* = (1 + w_1)(1 + w_2) − 1`.
- **Series composition** (length-2 path, weights w_1, w_2):
  `w* = w_1 · w_2 / (q + w_1 + w_2)`.

Both are exact, reduce the graph by one edge per step, and apply to
any SP sub-structure within a larger graph. The engine's
series-parallel fast path (`tutte/graphs/series_parallel.py`) uses the
single-variable specialization; the multivariate version is a
potential future optimization for chord-rule leaves that are SP.

---

## Treewidth and tree decompositions

### Robertson & Seymour (1984) — Treewidth

> Robertson, N.; Seymour, P. D. "Graph minors III: Planar tree-width."
> _J. Combin. Theory Ser. B_ **36**(1), 49–64 (1984).

Defines treewidth as a structural parameter.

### Andrzejak (1998) / Noble (1998) — Tutte DP on bounded treewidth

> Andrzejak, A. "An algorithm for the Tutte polynomials of graphs of
> bounded treewidth." _Discrete Math._ **190**(1-3), 39–54 (1998).
>
> Noble, S. D. "Evaluating the Tutte polynomial for graphs of bounded
> tree-width." _Combin. Probab. Comput._ **7**(3), 307–321 (1998).

`O(2^tw × n)` bag DP for the Tutte polynomial. Implemented in
`tutte/graphs/treewidth.py` (Python) and `tutte/graphs/_treewidth_c.py`
(cffi C extension). The C path is gated to `5 ≤ tw ≤ 10`.

### Bodlaender (1996) — Linear-time tree decomposition

> Bodlaender, H. L. "A linear-time algorithm for finding
> tree-decompositions of small treewidth." _SIAM J. Comput._ **25**(6),
> 1305–1317 (1996).

Foundational. The engine uses a simpler mindegree heuristic plus an
early-exit time budget in `compute_best_tree_decomposition`.

---

## Cographs and clique-width

### Corneil, Lerchs & Stewart Burlingham (1981) — P_4-free graphs

> Corneil, D. G.; Lerchs, H.; Stewart Burlingham, L. "Complement
> reducible graphs." _Discrete Applied Mathematics_ **3**(3), 163–174
> (1981).

Characterizes cographs as `P_4`-free graphs. Implemented in
`tutte/cotree_dp/recognition.py`.

### Giménez, Hliněný & Noy (2006) — Tutte on bounded clique-width

> Giménez, O.; Hliněný, P.; Noy, M. "Computing the Tutte polynomial on
> graphs of bounded clique-width." _SIAM J. Discrete Math._ **20**(4),
> 932–946 (2006).

Subexponential `exp(O(n^{1−ε}))` for clique-width-`k` graphs. The
cograph (clique-width-2) special case is implemented in
`tutte/cotree_dp/dp.py`. General clique-width-`k` extension remains an
open direction.

### Courcelle & Olariu (2000) — Clique-width

> Courcelle, B.; Olariu, S. "Upper bounds to the clique-width of
> graphs." _Discrete Applied Mathematics_ **101**(1–3), 77–114 (2000).

Defines clique-width via vertex-labeled construction operations.

---

## k-vertex sums and matroid theory

### Brylawski (1971) — 2-sum

> Brylawski, T. H. "A combinatorial model for series-parallel networks."
> _Trans. Amer. Math. Soc._ **154**, 1–22 (1971).

The classical 2-sum identity:
`T(G_1 ⊕_2 G_2) = T(G_1) · T(G_2) / T(K_2)`. Generalizes to the
boundary-quotient formula used by `tutte/graphs/k_sum.py`.

### Oxley (2011) — Matroid theory reference

> Oxley, J. G. _Matroid Theory_ (2nd ed.). Oxford University Press
> (2011).

Chapter 11 covers k-sum identities. Reference for the theoretical
background of the chord-rule k-sum.

---

## Linear recurrences for Tutte polynomials

### Noy & Ribò (2007) — The classical theorem

> Noy, M.; Ribò, A. "Linear Recurrence Relations for Graph Polynomials."
> In: _Algebraic Combinatorics and Computer Science: A Tribute to
> Gian-Carlo Rota_, Springer LNCS (2007).

For any recursively constructible graph family `{G_n}`, the Tutte
polynomial satisfies a linear recurrence

```
T(G_{n+r}) = p_1(x, y) T(G_{n+r−1}) + … + p_r(x, y) T(G_n)
```

with `p_i ∈ Z[x, y]`. Proof uses a transfer-matrix construction on the
rank polynomial.

`tutte/roots/chain_recurrence.py` is an explicit constructive
re-derivation for cell-decomposable chain families with a fixed
connector. Order equals the cell-aut orbit count on boundary
partitions. Modular evaluation via Faddeev-LeVerrier mod p gives ms-
scale per-point cost.

### Fischer & Makowsky (2008) — MSOL polynomials

> Fischer, E.; Makowsky, J. A. "Linear recurrence relations for graph
> polynomials" (2008).

Extends Noy-Ribò to all MSOL-definable graph polynomials. General but
non-constructive.

### Kotek, Makowsky & Ravve (2013) — Bi-iterative families

> Kotek, T.; Makowsky, J. A.; Ravve, E. R. "Recurrence relations for
> graph polynomials on bi-iterative families of graphs." _European J.
> Combin._; `arXiv:1309.4020`.

Extends Noy-Ribò to bi-iterative families (constructed by two repeated
operations). Recurrence coefficients themselves satisfy linear
recurrences. Relevant for 2D Cm_m grids; not currently implemented.

---

## Transfer matrices for Tutte/Potts on lattice strips

### Chang & Shrock (2005) — Cyclic and Möbius strips

> Chang, S.-C.; Shrock, R. "Transfer Matrices for the Partition
> Function of the Potts Model on Cyclic and Möbius Lattice Strips."
> _Physica A_ **347**, 314–352 (2005); `arXiv:cond-mat/0404524`.

Explicit transfer matrices decomposed by `S_q` color-permutation
orbits. Structurally parallel to the cell-quotient DPs in
`tutte/roots/` (which decompose by cell-aut orbits on boundary
partitions). The trace formula `Z(cycle_n) = Tr[T^n]` is the
classical analog of the cycle-close step.

### Beaudin et al. (2010) — Bracelets

> Beaudin, L.; Ellis-Monaghan, J.; Pangborn, G.; Shrock, R. "Tutte
> polynomials of bracelets." _J. Algebraic Combin._ **32**, 393–408
> (2010).

Closed-form Tutte polynomials for bracelet graphs (cycles of `n` copies
of a fixed cell joined by a fixed connector). The exact algebraic
structure that the cycle DP and chain-recurrence frameworks compute.

### Shrock & Chang (2009) — Strips in a magnetic field

> Shrock, R.; Chang, S.-C. "Structure of the Partition Function and
> Transfer Matrices for the Potts Model in a Magnetic Field on Lattice
> Strips." _J. Stat. Phys._ **137**, 1037–1101 (2009).

Extension of the cyclic-strip framework. Reference for periodic-strip
transfer matrix construction.

---

## Series-parallel and recent algorithmic results

### Yardim & Türker (2025) — Linear-time Potts on SP graphs

> Yardim, B.; Türker, T. "Linear-time exact Potts on series-parallel
> graphs." `arXiv:2507.22579` (July 2025).

Linear-time exact computation of the Potts partition function on
series-parallel graphs with arbitrary edge weights. Builds on Sokal's
series+parallel reductions.

The engine's SP fast path (`tutte/graphs/series_parallel.py`) uses
single-variable SP-tree decomposition; pairing with this multivariate
reduction would extend the SP path to chord-rule leaves with retained
edge weights.

### Blažej, Jana & Ramanujan (2025) — Cograph-modular treewidth

> Blažej, V.; Jana, S.; Ramanujan, M. S. "Cograph-modular treewidth."
> IPEC 2025.

A structural parameter strictly between treewidth and clique-width.
Take the modular decomposition of `G`; restrict each module to be a
cograph; cmtw is the treewidth of the resulting decomposition tree.

The paper does not extend to the Tutte polynomial directly. The
structural parameter alone, however, unlocks generic
MSOL-expressible-problem algorithms via Courcelle's theorem — and the
Tutte polynomial is MSOL-expressible.

Worth investigating as a structural parameterization for the D-Wave
Cm/Pm/Z family, since their cells are themselves cographs.

### Patel & Regts (2017) — Deterministic poly-time approximation

> Patel, V.; Regts, G. "Deterministic poly-time approximation
> algorithms for partition functions." `arXiv:1607.01167` (2017).

Computes single-point evaluations, not the full polynomial. Out of
scope; the engine returns the exact polynomial.

---

## D-Wave topologies

### Boothby et al. (2020) — Pegasus

> Boothby, K.; Bunyk, P.; Raymond, J.; Roy, A. "Next-Generation
> Topology of D-Wave Quantum Processors." `arXiv:2003.00133` (2020).

Pegasus specification. Each `Pm` has `~8m(3m − 1)` vertices, `K_{4,4}`
cells, three coupler types (internal, external, odd/diagonal).
Treewidth grows as `12m − O(1)`. The odd couplers break a clean grid
topology and route through chord-rule paths instead of
`compute_cell_quotient_grid_dp_streamed`.

### Zephyr (D-Wave technical report)

> D-Wave Systems Inc. _Zephyr Topology of D-Wave Quantum Processors_.
> Technical report.

Third-generation D-Wave topology after Chimera and Pegasus. Cells are
smaller than Chimera/Pegasus `K_{4,4}` cells. Z(1, t) and Z(2, t) are
the current benchmark targets for the cell-quotient bipartite-junction
paths.

### D-Wave family symmetry — `Z(m, t)` σ-finder results

A probe over the candidate σ patterns (`i + n/2`, `reverse`,
`cell-swap ±2`, `cell-swap ±4`) identifies which Zephyr graphs admit
a **free** order-2 automorphism (no σ-fixed edges) — the simpler case
of the lift identity used by Stage 8.5.

| Graph    |  V |  E  | Free σ found via | Quotient (free)  |
| -------- | -- | --- | ---------------- | ---------------- |
| Z(1, 1)  | 12 |  22 | none             | —                |
| Z(1, 2)  | 24 |  76 | cell-swap ±2     | 12v 38e FREE     |
| Z(1, 3)  | 36 | 162 | none             | —                |
| Z(2, 1)  | 40 | 114 | none             | —                |
| Z(2, 2)  | 80 | 356 | cell-swap ±4     | 40v 178e FREE    |

`Z(1, 1)`, `Z(1, 3)`, `Z(2, 1)` admit only non-free σ via `i + n/2`
and `reverse`; the non-free DP supports them (`L_loop` adjustment in
`compute_t_fix_sigma_mod`) but the quotient gains loops.

The probe script is at
`tutte/research/scripts/probe_z_family_sigma.py`. A richer σ-search
(e.g. via the `nx.GraphMatcher` isomorphism iterator) is open work
and might surface free σ for the currently non-free entries, or
smaller quotients than `i + n/2` / `reverse` for graphs with rich
`Aut` groups. `Z(m, t)` automorphism groups are typically much larger
than `Z_2`; larger-than-`Z_2` quotients could give even smaller
quotient graphs.

---

## Signed graphs and equivariant matroids

### Zaslavsky (1982, 1991) — Signed and biased graphs

> Zaslavsky, T. "Signed graphs." _Discrete Applied Mathematics_ **4**,
> 47–74 (1982).
>
> Zaslavsky, T. "Biased graphs II." _J. Combin. Theory Ser. B_ **51**,
> 46–72 (1991).

Foundational. Defines signed graphs (edges with ± labels) and their
**frame matroid** with rank function

```
r_signed(L) = |V| − (# balanced components of (V, L))
```

where a component is balanced iff every cycle sums to 0 mod 2 in the
edge signs. Unbalanced components contribute one extra unit of rank.

This is the rank function consumed by Stage 8.5 of the engine
(σ-equivariant decomposition; see
`tutte/docs/06_9_signed_equivariant_dp.md`). For a 2-fold cover
`G → G/⟨σ⟩` with monodromy `χ`, the cover-side Whitney rank of any
σ-invariant subset satisfies the **lift identity**

```
r_G(A_L) = r_quot(L) + r_signed(L, χ)
```

which makes `T_fix^σ(G)` computable by a single elimination-order DP
on the (smaller) quotient `G_base` with signed-frame-matroid rank
tracking. Implementations live in `tutte/graphs/signed_elim_dp.py`,
`tutte/graphs/signed_treewidth.py`, and `tutte/roots/signed_quotient.py`.

### Reiner — Equivariant matroid theory

> Reiner, V. "Equivariant fiber polytopes." _Doc. Math._ **7**,
> 113–132 (2002).

When a finite group `H` acts on a matroid `M`, the `H`-equivariant
Tutte polynomial decomposes via character theory. Reference for the
σ-equivariant work in `tutte/graphs/sigma_equivariant_dp.py` and
`tutte/docs/06_9_signed_equivariant_dp.md`.

### Kamiya, Miyamoto & Yoshinaga (2017) — G-Tutte polynomial

> Kamiya, H.; Miyamoto, K.; Yoshinaga, M. "G-Tutte Polynomials and
> Abelian Lie Group Arrangements." _Int. Math. Res. Notices_ (2017).

Introduces `T_G(M; x, y)` for matroid `M` with abelian group `G`
acting. Specializes to chromatic, Tutte, and other polynomials.
Applicable for abelian aut groups; generalizations needed for
non-abelian cases.

### Burnside (1911) / Pfeiffer (1997) — Table of marks

> Burnside, W. _Theory of Groups of Finite Order, 2nd ed._ Cambridge
> University Press (1911). § 180.
>
> Pfeiffer, G. "The subgroups of M_24, or how to compute the table of
> marks of a finite group." _Experimental Mathematics_ **6**, 247–270
> (1997).

The table of marks gives the recovery framework for
`T(G) = Σ_j (|H| / |K_j|) × n_j(G)` from `T_fix^K(G)` over conjugacy
classes of subgroups `K ⊆ Aut(G)`. Verified on K_4, K_{3,3}, and
cubical prism in `tutte/research/scripts/burnside_marks_tutte.py`.

### Solomon (1967) — Burnside ring

> Solomon, L. "The Burnside algebra of a finite group." _J. Combin.
> Theory_ **2**, 603–615 (1967).

Foundation for the Burnside-ring recovery framework above.

### Open `T_free^σ` directions (orbit-2 half)

`T_fix^σ` is in hand (`tutte/roots/signed_quotient.py`); the σ-paired
half `T_free^σ` is open. Four candidate approaches have been explored,
each catalogued in `tutte/docs/06_9_signed_equivariant_dp.md`:

1. **Direct cover-side per-orbit DP** — branch by σ-edge orbit;
   blocked on `Z(1, 2)` by cover state-space size, needs a C extension.
2. **Character-theoretic identity** `T(G) = 2 T_fix^σ − T_sign` —
   requires distinguishing σ-invariant from σ-paired subsets in a DP,
   same fundamental challenge as Approach 1.
3. **Burnside marking matrix** over `Aut(G)` subgroups — verified
   exact for `K_4`, `K_{3,3}`, cubical prism in
   `tutte/research/scripts/burnside_marks_tutte.py`; for `Z(1, 2)` the
   per-class evaluation cost exceeds the direct engine path.
4. **Algebraic substitution** `T(G; x, y) = F(T_signed, T_quot)` —
   no closed form found empirically (Cube test point).

Related open directions on the representation-theory side:

- **Möbius inversion on the subgroup lattice** of `Aut(G)`. The
  Möbius function `μ(K, H)` on the subgroup lattice gives an inversion
  formula complementary to the Burnside marking matrix; whether it
  yields a sharper `T(G)` recovery from `{T_fix^K}` than Burnside is
  open.
- **Higher-order `H`-Tutte polynomials** (Kamiya-Miyamoto-Yoshinaga
  2017) for non-abelian `H ⊆ Aut(G)`. The current `G_H`-Tutte
  literature handles abelian `H`; the natural extension to symmetric
  groups acting on `K_{4,4}`-cell D-Wave graphs is open.

---

## Foundational algorithms

### Kirchhoff (1847) — Matrix-tree theorem

> Kirchhoff, G. "Über die Auflösung der Gleichungen, auf welche man bei
> der Untersuchung der linearen Verteilung galvanischer Ströme geführt
> wird." _Annalen der Physik_ (1847).

`T(G; 1, 1)` = `det(L̃)` where `L̃` is the reduced Laplacian. The
engine uses this as a cross-check on every synthesis result
(`tutte/validation.py::count_spanning_trees_kirchhoff`).

### Tarjan (1972) — Cut vertices in linear time

> Tarjan, R. "Depth-first search and linear graph algorithms." _SIAM J.
> Comput._ **1**(2), 146–160 (1972).

Linear-time cut vertex detection. Backing for `Graph.has_cut_vertex()`
and the cut-vertex factorization stage.

### Weisfeiler & Leman (1968) — Color refinement

> Weisfeiler, B.; Leman, A. A. "A reduction of a graph to a canonical
> form and an algebra arising during this reduction" (Russian).
> _Nauchno-Technicheskaya Informatsia_ (1968).

The canonical-key foundation. `Graph.canonical_key()` and
`MultiGraph.canonical_key()` use WL refinement to produce
isomorphism-invariant keys for rainbow-table lookup.

### Gauss (1801) — Chinese Remainder Theorem

> Gauss, C. F. _Disquisitiones Arithmeticae_ (1801).

If `m_1, …, m_k` are pairwise coprime, the system `x ≡ a_i (mod m_i)`
has a unique solution modulo `M = ∏ m_i`. Used for modular polynomial
arithmetic in `tutte/graphs/_treewidth_c.py` (treewidth DP with
coefficients exceeding `int64`) and in the modular point-value
pathways in `tutte/roots/interpolation.py`.

---

## Sources

- [The multivariate Tutte polynomial (Sokal 2005)](https://arxiv.org/abs/math/0503607)
- [Linear-time Potts on SP graphs (Yardim & Türker 2025)](https://arxiv.org/abs/2507.22579)
- [Cograph-Modular-Treewidth (Blažej et al. IPEC 2025)](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.IPEC.2025.18)
- [Tutte on Bounded Clique-Width (Giménez-Hliněný-Noy 2006)](https://web.mat.upc.edu/marc.noy/uploads/2013/05/Tutte-Clique-width.pdf)
- [Deterministic Polynomial-Time Approximation (Patel-Regts 2017)](https://arxiv.org/abs/1607.01167)
- [Noy & Ribò (2007), Linear Recurrence Relations](https://link.springer.com/chapter/10.1007/978-3-540-78127-1_15)
- [Pegasus Topology (Boothby et al. 2020)](https://arxiv.org/abs/2003.00133)
- [Zephyr Topology technical report](https://www.dwavequantum.com/media/2uznec4s/14-1056a-a_zephyr_topology_of_d-wave_quantum_processors.pdf)

---

## Ongoing research directions

The engine's main exponential walls live in two regimes:

1. **Higher-treewidth D-Wave graphs** (`Cm_3+`, `Pm_3+`, `Z(2, t)+`)
   where `tw > 11` and the cell-quotient DPs hit Bell-number walls on
   the joint boundary partition. The chain-recurrence framework caches
   the per-row transfer matrix once; extending it to 2D grid
   composition via Kotek-Makowsky-Ravve bi-iterative families is the
   natural next step.

2. **Full-polynomial recovery via modular interpolation**. `Cm_2` is
   bit-exact in seconds; `Cm_3` needs the modular DP to absorb the
   H-canonicalization optimization currently wired only into the
   pair-orbit consumer path.

Other open directions worth tracking:

- **Cycle-recurrence symbolic extraction.** The chain recurrence
  generalizes to cycles with higher order — empirically
  `order_cycle ≈ 2.5 × order_chain + const`. Validated symbolically
  on `K_{2,2}+M_2` (order 5). Open: `K_{3,3}+M_3` cycle (empirical
  order > 8, needs N≈25–30 cycle values), `K_4+M_2` cycle, and the
  2D-grid generalisation that would unlock `Cm_m` rows × cols at
  scale.
- **2D-grid recurrence (Kotek-Makowsky-Ravve bi-iterative families).**
  D-Wave `Cm_m` is row × column composition. Each row satisfies the
  1D chain recurrence; the open question is whether the row-by-row
  Tutte polynomials satisfy a second recurrence in the row index.
  Bi-iterative families (Kotek-Makowsky-Ravve 2013) prove such
  recurrences exist; the constructive extraction is open.
- **General clique-width-`k` DP** extending the current cograph-only
  path (Giménez-Hliněný-Noy 2006).
- **Cograph-modular-treewidth** (IPEC 2025) as a structural
  parameterization for the D-Wave cell families.
- **Multivariate-Z SP reductions** (Yardim-Türker 2025) as a fast
  path for chord-rule leaves with retained edge weights.
- **σ-equivariant `T_free^σ` computation.** Four candidate approaches
  catalogued in `tutte/docs/06_9_signed_equivariant_dp.md`; none
  currently beat the direct engine path on the `Z(1, 2)` benchmark.
  A C-extension for the signed-DP inner loop is the most-pressing
  enabler.
- **Burnside-table-of-marks recovery for σ-equivariant Tutte** —
  verified in `tutte/research/scripts/burnside_marks_tutte.py`; not
  currently on a critical path.
- **Möbius inversion on the `Aut(G)` subgroup lattice** as an
  alternative to the Burnside marking matrix for recovering `T(G)`
  from `{T_fix^K(G)}`.
