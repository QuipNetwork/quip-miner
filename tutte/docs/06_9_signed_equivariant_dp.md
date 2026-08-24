# 6.9 — Signed-Graph DP and σ-Equivariant Decomposition

> **Status (2026-05):** The σ-equivariant cover DP (`sigma_equivariant_dp.py`),
> the treewidth-based signed DP (`signed_treewidth.py`), and the
> elimination-order signed DP (`signed_elim_dp.py` + its `_signed_elim_c.py`
> C extension) were validated but never beat the general path, so they were
> moved to [`tutte/deprecated/`](../deprecated/README.md). What remains **live**
> is the σ-finder `find_best_sigma` (`roots/signed_quotient.py`, used by the
> engine's chord-ordering). This document is retained as the theory reference
> for that work; the DP module paths below now point into `tutte/deprecated/`.

## Summary

For a graph `G` that admits an order-2 automorphism `σ ∈ Aut(G)`, the
Tutte polynomial decomposes by σ-action on edge subsets:

```
T(G; x, y) = T_fix^σ(G; x, y) + T_free^σ(G; x, y)
```

where

- **`T_fix^σ`** sums over σ-invariant subsets `A ⊆ E(G)` (orbits of
  size 1, i.e. `σ(A) = A` as a set);
- **`T_free^σ`** sums over σ-paired subsets (orbits of size 2 under σ).

`T_fix^σ` is computable end-to-end via a single elimination-order DP
on the σ-quotient graph that tracks signed-graph (Zaslavsky 1982) rank
information. `T_free^σ` is open research: four candidate approaches
are catalogued at the bottom of this page; none has beaten the direct
engine path on the current benchmark `Z(1, 2)`.

The signed-DP framework is the candidate path to add `Z(1, 3)`,
`Cm_3`, and larger D-Wave graphs to the rainbow-table lookup once a
`T_free^σ` recovery formula or sufficiently fast direct cover-side DP
lands.

## When it is used

Stage 8.5 of the engine cascade
(`tutte/synthesis/engine.py::_synthesize_inner`), triggered when:

- The input graph has a known order-2 σ-automorphism (commonly the
  cell-swap σ for D-Wave `Z(m, 2)`).
- The cover graph's treewidth is too large for the standard
  treewidth DP (Stage 8) to handle, but the **quotient** under σ has
  manageable treewidth.

## The lift identity (free 2-fold covers)

Form the quotient `G_base = G / ⟨σ⟩`:

- **Vertices** of `G_base` = σ-orbits on `V(G)` (pairs `{v, σ(v)}`
  become one super-vertex; σ-fixed vertices stay singletons).
- **Edges** of `G_base` = σ-orbits on `E(G)`.

When σ has no σ-fixed individual edges (the **free** case), `G` is a
**2-fold cover** of `G_base`: each base edge lifts to exactly two
cover edges. The cover is specified by a **monodromy character**
`χ : E(G_base) → Z_2` that records, for each base edge, whether the
lift preserves or swaps the two sheets of the cover.

**The lift identity.** For any σ-invariant cover subset `A_L ⊆ E(G)`
lifted from a base subset `L ⊆ E(G_base)`:

```
|A_L|        = 2 |L|                                (free case)
r_G(A_L)     = r_quot(L) + r_signed(L, χ)           (rank identity)
```

where `r_signed` is Zaslavsky's **frame-matroid rank function** on the
signed graph `(G_base, χ)`. Concretely, viewing `χ(e) ∈ {0, 1}` as a
sign on edge `e`, a cycle in `(V_quot, L)` is **balanced** iff the sum
of signs around it is 0 mod 2; a connected component is balanced iff
all its cycles are balanced. Then:

```
r_signed(L) = |V_quot| − (# balanced components of (V_quot, L))
```

Unbalanced components contribute one extra unit of rank vs the
unsigned graph rank.

Substituting the lift identity into the Whitney form of the Tutte
polynomial gives the closed expression

```
T_fix^σ(G; x, y) =
   Σ_L (x − 1)^{r(E_G) − r_quot(L) − r_signed(L, χ)}
       · (y − 1)^{2|L| − r_quot(L) − r_signed(L, χ)}
```

which is summable by a single **elimination-order DP on `G_base`**
that tracks the standard partition state plus per-block sign-balance
information.

## Non-free covers

When σ fixes some individual edges of `G`, those edges become **loops**
in the quotient. The same lift identity holds with

```
|A_L|        = 2 |L| − |L_loop|             (loops contribute 1 cover edge each)
r_G(A_L)     = r_quot(L) + r_signed(L, χ)   (unchanged)
```

The per-edge DP multipliers gain a 1-power adjustment on loop edges:

- balanced cycle in balanced component: `(y − 1)` for loop,
  `(y − 1)²` for non-loop
- unbalanced cycle in balanced component: `(x − 1)⁻¹` for loop,
  `(x − 1)⁻¹ (y − 1)` for non-loop
- cycle in unbalanced component: `(y − 1)` for loop, `(y − 1)²` for
  non-loop

Validated on `K_4 + (01)(23)` (2 σ-fixed edges) and `K_{3,3}` with
part-swap σ (3 σ-fixed edges).

## Implementation

### High-level API (`tutte/roots/signed_quotient.py`)

- `build_quotient_with_monodromy(g, perm)` — extracts
  `(V_quot, E_quot, χ)` from `(G, σ)` where `σ` is given as a vertex
  permutation. Handles both free and non-free cases.
- `evaluate_t_signed_mod(nodes, edges_with_signs, x, y, p)` —
  Zaslavsky's signed-graph Tutte polynomial of the quotient signed
  graph at one `(x, y, p)` point. Wraps
  `signed_elim_dp.compute_signed_tutte_elim_mod` (now in
  `tutte/deprecated/`).
- `compute_t_fix_sigma_quotient_mod(g, perm, x, y, p)` — σ-invariant
  Tutte polynomial of the **cover** `G` at one point, computed via
  the lift identity by running DP on the **quotient**.
- `derive_t_free_sigma_mod(g, perm, x, y, p, engine=None)` —
  σ-paired Tutte half, derived as `T(G) − T_fix^σ` when `T(G)` is
  already in the engine's rainbow-table lookup (e.g., `Z(1, 2)`). The
  two computations are independent (engine vs signed-DP), so this
  provides a cross-validation check.
- `interpolate_t_signed_mod(...)` — multi-point evaluation +
  bivariate Lagrange interpolation to recover the polynomial.
- `zephyr_cell_swap_perm(m)` — convenience helper for the `Z(m, 2)`
  cell-swap permutation.

### Low-level DPs

- `tutte/deprecated/signed_elim_dp.py` (moved from `graphs/` — see status note above):
    - `compute_signed_tutte_elim_mod(nodes, edges_with_signs, x, y, p)`
      — signed-graph (Zaslavsky frame-matroid) Tutte via
      vertex-elimination DP.
    - `compute_t_fix_sigma_mod(nodes, edges_with_signs, r_E_G, x, y, p)`
      — σ-invariant Tutte of cover; per-edge multipliers handle free
      and non-free cases.
- `tutte/deprecated/sigma_equivariant_dp.py` (moved from `graphs/` — see status note above):
    - `compute_tutte_per_orbit_mod(...)` — per-orbit batched DP that
      correctly handles σ-canonicalization (processes σ-paired edges
      together so σ-action commutes with edge order).
      Removed May 20, 2026: the older wrapper
      `compute_tutte_sigma_equivariant_mod` had a σ-asymmetric local
      canonicalization bug; its `use_sigma=True` path was correctly
      delegated to `compute_tutte_per_orbit_mod` and its
      `use_sigma=False` path was just a redundant min-fill DP.

### Validation

- `tutte/tests/test_signed_elim.py` — 10 tests, signed-DP small
  cases (imports the DPs from `tutte/deprecated/`).
- `tutte/tests/test_signed_quotient.py` — 10 tests, high-level API.
- `tutte/tests/test_sigma.py` — 9 tests, σ-equivariant DP
  (chord ordering + per-orbit batched DP).

## Reference performance

| Graph         | Quotient size | Operation                | Time          |
| ------------- | ------------- | ------------------------ | ------------- |
| Cube (8v 12e) | K_4 (4v 6e)   | `T_fix^σ` at one `(x,y,p)` | < 0.01 s      |
| Z(1, 2)       | 12v 38e       | `T_fix^σ` at one `(x,y,p)` | ~ 4 s         |
| Z(1, 2)       | 12v 38e       | `T_signed` at one `(x,y,p)`| ~ 4 s         |
| Z(1, 2)       | 12v 38e       | `T_signed` full poly via 2D Lagrange (507 pts, 8 workers) | ~ 308 s |
| Z(1, 3)       | 18v 87e       | `T_signed` at one `(x,y,p)`| > 2 min (killed) |

The single-point cost for `Z(1, 2)` is now `~4 s` (down from `~33 s`
after the C-ext shipped). Full-polynomial recovery via 2D Lagrange
interpolation at 507 evaluation points runs `~308 s` with 8 worker
processes — slower than direct treewidth_dp on the COVER (`~110 s`),
and only recovers `T_fix^σ` (half of `T(G)`), so this path is NOT a
competitive way to compute `T(G)` even with the recent speedup. The
single-point cost is competitive for verification / cross-validation.

## Open problem: `T_free^σ`

`T_free^σ` requires summing over **σ-paired** subsets. Each paired
orbit contributes twice the orbit-rep weight, where the rep is an
asymmetric choice across edge orbits (3 choices per orbit: both /
smaller / neither, excluding the all-symmetric patterns which form
`T_fix^σ`).

Four candidate approaches have been explored:

### Approach A — direct cover-side DP (per-orbit branching)

Process σ-edge orbits one at a time; for each orbit, 4 branches:
`(del e₁, del e₂)`, `(del e₁, keep e₂)`, `(keep e₁, del e₂)`,
`(keep e₁, keep e₂)`. Maintain partition state on `V(G)` (cover) with
σ-canonicalization. Implemented as `compute_tutte_per_orbit_mod`;
correct on Cube, `C_4`, `C_6`. **Blocker**: `Z(1, 2)` cover state
space is too large for pure Python (> 1.5 min with hybrid
min-fill σ-pair elim order). Would need a C extension or smarter
state representation.

### Approach B — character-theoretic identity

For `Z_2 = ⟨σ⟩`, define

```
T_sign(G) = Σ_A weight(A) × χ_sign(σ-orbit of A)
```

Then `T_sign = T_fix − T_free = 2 T_fix − T(G)`, giving

```
T(G) = 2 T_fix^σ − T_sign
```

`T_fix^σ` is in hand; `T_sign` requires distinguishing invariant
versus paired subsets in a DP — the same fundamental challenge as
Approach A.

### Approach C — Burnside marking matrix

For graphs with `|Aut(G)|` large (e.g. `Aut(Z(1, 2))` has order 512),
compute `T_fix^K` for all conjugacy classes of subgroups `K ⊆ Aut(G)`
and solve the **Burnside marking matrix** equation `M × n = T_fix` to
recover orbit counts `n_K`, then

```
T(G) = Σ_K (|H| / |K|) · n_K(G)
```

Verified empirically on `K_4`, `K_{3,3}`, and the cubical prism
(exact polynomial match in all three cases) in
`tutte/research/scripts/burnside_marks_tutte.py`. For `Z(1, 2)`, the
estimated Burnside cost is `~30-100 × ~10 s = 5-15 min` per `(x, y)`
point — slower than the direct engine path (~178 s).

### Approach D — algebraic substitution

Is there a polynomial substitution `(x, y) → (x', y')` such that

```
T(G; x, y) = F(T_signed(G/σ; x', y'), T_quot(G/σ; x', y'))
```

Empirically checked on the Cube
(`T(Cube) = 746`, `T_signed(K_4 + χ) = 162`,
`T_quot(K_4 unsigned) = 108` at `(2, 3) mod 1009`) — no obvious linear
combination matches.

## D-Wave family applicability

A probe over the candidate σ patterns
(`i + n/2`, `reverse`, `cell-swap ±2`, `cell-swap ±4`) finds:

| Graph    |  V |  E  | Free σ via       | Quotient (free)  |
| -------- | -- | --- | ---------------- | ---------------- |
| Z(1, 1)  | 12 |  22 | none             | —                |
| Z(1, 2)  | 24 |  76 | cell-swap ±2     | 12v 38e FREE     |
| Z(1, 3)  | 36 | 162 | none             | —                |
| Z(2, 1)  | 40 | 114 | none             | —                |
| Z(2, 2)  | 80 | 356 | cell-swap ±4     | 40v 178e FREE    |

`Z(1, 1)`, `Z(1, 3)`, `Z(2, 1)` admit only non-free σ via `i + n/2`
and `reverse`; the non-free DP supports them but the quotient gains
loops. `Z(2, 2)` cleanly fits the free framework. Richer σ-search
(e.g. via the `nx.GraphMatcher` isomorphism iterator) might surface
free σ for the currently-non-free entries — that is open work.

## See also

- [Engine workflow primer §8.5](../research/engine_workflow_primer.md)
  — first-principles introduction for non-specialists.
- [Literature catalog: Signed graphs and equivariant matroids](../research/literature_search.md)
  — Zaslavsky, Reiner, Kamiya-Miyamoto-Yoshinaga, Burnside, Solomon.
- [6.7 Chain & Cycle Recurrence Algebra](06_7_chain_recurrence_algebra.md)
  — sibling algebraic framework for chain-structured cell graphs.
- [6.8 Modular Arithmetic Pathways](06_8_modular_arithmetic_pathways.md)
  — point-value evaluation + Lagrange + CRT reconstruction (used by
  `interpolate_t_signed_mod`).
