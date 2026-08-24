# 6.7 — Chain & Cycle Recurrence Algebra

## Summary

For any **(cell, connector) chain** — a graph built by attaching `n`
copies of a fixed cell template via a fixed connector template — the
Tutte polynomial of the chain satisfies an **exact linear recurrence
in `n` with polynomial coefficients in `(x, y)`**:

```
T(chain_{n+r}) = c_1(x,y) T(chain_{n+r-1}) + c_2(x,y) T(chain_{n+r-2})
                 + ... + c_r(x,y) T(chain_n)
```

where the `c_i(x, y) ∈ Z[x, y]` are derived from the characteristic
polynomial of an `r × r` polynomial-valued transfer matrix `M(x, y)`,
and `r = n_orbits` is the number of aut-orbit classes of the
boundary-partition state space under the cell's automorphism group.

> **Relation to the unified theorem (§8.6)**: the chain recurrence and the unified bivariate chord-junction theorem are two dual descriptions of the same algebra. The order `r = n_orbits` of the recurrence equals the number of distinct merger types under the cell's automorphism group; e.g. `K_{4, 4} + M_4` has `r = 5` matching the 5 mergers `(T(K_{4, 4})², 4·M_1, 6·M_2, 4·M_3, M_4)`. The unified theorem is easiest to cache per chord-junction; the recurrence is easiest to evaluate for very long chains. See §8.6 §4.

**This is a re-derivation of a classical result** (Noy & Ribò 2007,
[Linear Recurrence Relations for Graph Polynomials][noy-ribo]). Our
contribution is:

- A concrete algorithm for **extracting `M(x, y)` and the
  characteristic polynomial** from existing cell-quotient DP
  infrastructure (`compute_path_dp_grouped` observer hook).
- A **modular evaluation pathway** (Faddeev-LeVerrier mod p) that
  bypasses symbolic char poly extraction for fast per-point
  evaluation — `T(K_{4,4} chain_100)` in ~3-9 ms per (x, y, p) point.
- A **cycle extension** that fits a higher-order recurrence to the
  cycle Tutte polynomial (validated symbolically on K\_{2,2}+M_2
  cycle, order 5 with integer-polynomial coefficients of bidegrees
  ≤ (6, 6)).

**Status:** five chain templates validated bit-for-bit at the
polynomial level; one cycle template (K\_{2,2}+M_2) symbolically
extracted; modular scaling demonstrated for both chain and cycle.
**Wired into the production engine** at the chain dispatch in
`compute_cell_quotient_tree_dp` (see `tutte/roots/chain_recurrence.py`
and `tutte/roots/__init__.py`).

[noy-ribo]: https://link.springer.com/chapter/10.1007/978-3-540-78127-1_15

## When it is used

The framework is dispatched from `compute_cell_quotient_tree_dp` (in
`tutte/roots/__init__.py`) whenever `is_chain_topology(spec.cell_tree)`
returns true and a modular point-value pathway is requested. The
chain-DP, modular evaluators, AND the transfer-matrix extractor all
live in `tutte/roots/chain_recurrence.py` (the extractor was lifted
from the lost `tutte/research/scripts/extract_chain_transfer_matrix.py`
after the original was lost uncommitted — see bead riff-u6q).

The framework applies whenever the input graph decomposes as
`cell_template ⊕_M cell_template ⊕_M ... ⊕_M cell_template` where:

- `cell_template` is a fixed graph (e.g., K*{4,4}, K*{3,3}, K_4)
- `⊕_M` is attachment via a fixed `connector` template (e.g., M*k
  matching, K*{a,b} bipartite)
- The cell's automorphism group acts non-trivially on the connector
  anchor vertices

D-Wave Chimera, Pegasus, and Zephyr ROWS satisfy this. (The full 2D
Cm_m / Pm_m / Z(m, t) lattice does not — see §"2D limitations" below.)

## Algorithm — three phases

### Phase 1: extract the polynomial transfer matrix `M(x, y)`

The cell-quotient path DP (`compute_path_dp_grouped` in
`tutte/roots/cell_quotient_path.py`) produces an internal state of the
form `Dict[orbit_key, TuttePolynomial]` where `orbit_key` ranges over
the boundary-partition aut orbits. An observer hook lets us read this
state after each cell+junction step.

Build `M` by feeding **unit-vector inputs** into one `apply_step`:

```python
for j, orbit_key in enumerate(orbit_keys):
    unit_state = {k: zero for k in orbit_keys}
    unit_state[orbit_key] = TuttePolynomial.from_coefficients({(0, 0): 1})
    out_state = apply_step(unit_state)
    for i, out_key in enumerate(orbit_keys):
        M[i][j] = out_state.get(out_key, zero)
```

`M` is an `r × r` matrix of `TuttePolynomial`s. The Cayley-Hamilton
theorem on `M(x, y)` over the polynomial ring `Z[x, y]` gives a
characteristic polynomial `char(λ) = λ^r + c_1 λ^{r-1} + ... + c_r`
with `c_i ∈ Z[x, y]`. By Cayley-Hamilton:

```
M^r + c_1 M^{r-1} + ... + c_r I = 0
```

Multiplying by the chain state vector and projecting to scalar
output:

```
S_{n+r} + c_1 S_{n+r-1} + ... + c_r S_n = 0
```

where `S_n = T(chain_n) × (x-1)^total_div(n)` is the raw state-sum
(the divisor `(x-1)^total_div` accumulates `div_per_step` per cell
step).

### Phase 2: extract the symbolic characteristic polynomial

```python
import sympy
M_sym = sympy.Matrix([[entry_as_sympy(M[i][j]) for j in range(r)] for i in range(r)])
char_sym = sympy.expand((lam * sympy.eye(r) - M_sym).det())
char_coeffs = sympy.Poly(char_sym, lam).all_coeffs()  # [1, c_1, c_2, ..., c_r]
```

Cost: dominated by `sympy.det` on the `r × r` polynomial matrix.
For K*{4,4}+M_4 (r=5): ~23 minutes one-time setup. For smaller
templates (K*{2,2}+M_2, r=2): seconds.

### Phase 3: evaluate via recurrence

Once `c_i(x, y)` are known and `S_2, ..., S_{r+1}` are computed via
direct DP, all higher `S_n` follow by the recurrence:

```python
for n in range(r + 2, target):
    S_n = sum(-c_k * S[n-k] for k in range(1, r+1))
T_n = divide_by_x_minus_1_power(S_n, total_div(n))
```

For modular evaluation at a specific `(x_0, y_0, p)` point:

```python
M_mod = [[poly.evaluate_mod(x_0, y_0, p) for poly in row] for row in M_poly]
char_mod = faddeev_leverrier_mod(M_mod, p)  # O(r^3) integer ops
# Then iterate recurrence on integer values mod p — O(n) per evaluation
```

This **bypasses symbolic char poly extraction entirely** — char poly
mod p extracted numerically per evaluation point in O(r^3) integer
ops. Scaling demonstration: `T(K_{4,4} chain_100)` at any
`(x_0, y_0, p)` in **2.9–8.8 ms per point**.

## Validated templates

Five chain templates with bit-for-bit polynomial-level validation
(regression tests in `tutte/tests/test_chain_recurrence.py`):

| Template | Cell                | Connector | r = n_orbits | Char poly bidegrees |
| -------- | ------------------- | --------- | ------------ | ------------------- |
| `k22_m2` | K\_{2,2}            | M_2       | 2            | (5,2), (6,2)        |
| `k33_m3` | K\_{3,3}            | M_3       | 3            | up to (12, 8)       |
| `k44_m4` | K\_{4,4}            | M_4       | 5            | up to (~29, ~24)    |
| `k4_m2`  | K_4 (non-bipartite) | M_2       | 2            | (5,4), (6,4)        |
| `k5_m2`  | K_5 (non-bipartite) | M_2       | 2            | similar             |

**Order = n_orbits theorem** (empirical, all five templates): the
recurrence order equals the number of aut-orbit classes of the
boundary partition state space. This generalizes the Noy-Ribò
result (which proved recurrence existence) to a concrete `r` value
derivable from the cell's automorphism group.

## Cycle extension

For a **cycle** of `n` cells (e.g., Cm*2's 4-cycle of K*{4,4} cells),
the Tutte polynomial also satisfies a linear recurrence, but at a
**higher order** than the chain. The closing operator `C` (the
identification of cell n's right boundary with cell 0's left
boundary) expands the effective state space.

### K\_{2,2}+M_2 cycle — fully extracted

Empirically order 5 (vs chain order 2). Symbolic char poly extracted
via Lagrange interpolation on a wide integer grid:

| c_i | Bidegree | Terms |
| --- | -------- | ----- |
| c_1 | (4, 4)   | 8     |
| c_2 | (6, 4)   | 20    |
| c_3 | (6, 4)   | 29    |
| c_4 | (6, 4)   | 27    |
| c_5 | (6, 6)   | 16    |

`c_1(x, y) = x⁴ + 2x³ + 5x² + xy + 7x + 2y² + 7y + 9`

Full coefficients saved at
`/tmp/chain_recurrence_cycle_charpoly_k22m2.txt` (regenerable via
`tutte/research/scripts/chain_recurrence_cycle_symbolic_v2.py`).

Validated bit-for-bit: `T(K_{2,2}+M_2 cycle_8)` from the recurrence
matches direct engine computation.

### The (2, 3) order-3 anomaly

At the integer point `(x, y) = (2, 3)`, the cycle T values fit an
order-3 recurrence (not order 5). This is **char-poly factoring at a
special point**, not rational coefficients:

```
char(λ) at (2, 3)  =  λ² · (λ³ + c_1' λ² + c_2' λ + c_3')
```

because `c_4(2, 3) = c_5(2, 3) = 0`. The two zero roots contribute
trivially to the recurrence; the non-trivial sequence satisfies the
reduced order-3 recurrence.

This is the classical phenomenon of **eigenvalue collisions at
special parameter values** (see Shrock-Chang on Potts transfer
matrices for the periodic-lattice analog).

### Other cycle templates

- K\_{3,3}+M_3 cycle: empirical order > 8 (verified with N=17 cycle
  values). Symbolic extraction deferred (needs N≈25-30).
- K_4+M_2 cycle: engine.synthesize too slow at n ≥ 10 to survey.
- K\_{4,4}+M_4 cycle (Cm_2!): chain order 5 ⇒ cycle order likely
  10-15. Symbolic extraction would require cycle_n at n≈25-30 each
  taking minutes via engine.synthesize. Modular evaluation is more
  tractable.

**Pattern**: cycle order grows faster than chain order — empirically
order_cycle ≈ 2.5 × order_chain + constant. Theoretical explanation
open.

## Modular evaluation

The chain framework's primary practical value is **modular point
evaluation**. For any (cell, connector) template with known transfer
matrix `M`:

1. **Setup cost** (one-time per template):

   - Extract `M` via cell-quotient observer (~18 s for K\_{4,4}+M_4)
   - Compute initial `S_2, ..., S_{r+1}` via direct DP (~3 s for
     K\_{4,4}+M_4)

2. **Per-point cost** (at each `(x_0, y_0, p)`):
   - Evaluate `M` at `(x_0, y_0) mod p` (5×5 = 25 polynomial evals)
   - Faddeev-LeVerrier mod p on `M_int` (~r³ = 125 integer ops)
   - Iterate recurrence to `n_target` — O(n) modular muls

For `T(K_{4,4} chain_100)`: per-point cost **~3-9 ms**. For comparison:
direct DP at n=10 takes 13.3 s; the recurrence at n=100 is
~4000× faster on a sequence 10× longer.

The same approach works for cycles once char poly is known.
`T(K_{2,2}+M_2 cycle_100)` evaluated in 0.4-21 ms per modular point.

Combined with **CRT reconstruction** and **2D Lagrange interpolation**
(see [6.8 — Modular Arithmetic Pathways](06_8_modular_arithmetic_pathways.md)),
the full polynomial can be recovered from a grid of modular point
values.

## 2D limitations

The chain framework gives fast scalar `T(row)` for a row of m
K\_{4,4}+M_4 cells in Cm_m. But for **2D composition** across rows,
we need the row's bottom-boundary partition state — Bell(4m)
partitions regardless of how fast `T(row)` is.

Concretely: `Cm_3` unlock requires reducing the `Bell(12) ≈ 4 M`
partition-state explosion. The chain framework doesn't address this.
The 2D `Cm_m` unlock is primarily an **engineering problem** (C-side
hash-map aggregation + pair-orbit decomposition + multiprocessing),
not an algebraic one. The Kotek-Makowsky-Ravve "bi-iterative
families" result (2013) sketches a path to nested 2D recurrences;
see the literature catalog for the open direction.

## Relation to existing literature

### Noy & Ribò (2007)

`[1]` M. Noy and A. Ribò, "Linear Recurrence Relations for Graph
Polynomials," in _Algebraic Combinatorics and Computer Science: A
Tribute to Gian-Carlo Rota_ (Springer, 2007).

Proved that **for any recursively constructible graph family**, the
Tutte polynomial satisfies a linear recurrence with coefficients
that are polynomials in `(x, y)`. The proof uses transfer-matrix
techniques on the rank polynomial.

Our framework is a **concrete implementation** of this classical
result, with:

- Explicit construction of the transfer matrix from cell-quotient DP
- Order = n_orbits theorem (Noy-Ribò didn't give an explicit `r`)
- Modular evaluation pathway

### Fischer & Makowsky

Extended Noy-Ribò to **all MSOL-definable graph polynomials**
(matching, independence, interlace, domination). The result is general
but doesn't give a constructive algorithm.

### Chang & Shrock — Transfer matrices on lattice strips

`[2]` Chang, S.-C., Shrock, R., "Transfer Matrices for the Partition
Function of the Potts Model on Cyclic and Möbius Lattice Strips,"
_Physica A_ **347** (2005). [arxiv:cond-mat/0404524][chang-shrock]

`[3]` Beaudin, L., Ellis-Monaghan, J., Pangborn, G., Shrock, R.,
"Tutte polynomials of bracelets," _J. Algebraic Combin._ **32**
(2010). [Springer][beaudin]

These compute Tutte/Potts transfer matrices for **lattice strips**
(square, triangular, honeycomb cyclic/Möbius). Use degree-by-color
subspace decomposition. Our framework is structurally similar but
uses aut-orbit decomposition (S_n action on boundary partitions)
rather than color-symmetry decomposition (S_q action on Potts
colors).

[chang-shrock]: https://arxiv.org/abs/cond-mat/0404524
[beaudin]: https://link.springer.com/article/10.1007/s10801-010-0220-1

## Files

### Production-ready

- `tutte/roots/chain_recurrence.py` — public API
  (`compute_chain_recurrence_mod`, `compute_chain_full_poly_from_spec`,
  `is_chain_topology`, `faddeev_leverrier_charpoly_mod`).
- `tutte/roots/__init__.py:compute_cell_quotient_tree_dp` — dispatch wiring.

### Research scripts (LOST — never committed)

The original `tutte/research/scripts/` chain modules were lost with the
machine that authored them. Rebuilt equivalents:

- `extract_chain_transfer_matrix` + `chain_S_at_n` — now production code
  in `tutte/roots/chain_recurrence.py`.
- template builders, sympy converters, cycle probe + recurrence fitter —
  `tutte/tests/chain_recurrence_helpers.py`.
- NOT rebuilt: the scaling demos (`chain_recurrence_modular_scaling.py`,
  `chain_recurrence_cycle_modular_scaling.py`) and the cycle symbolic
  extraction script (`chain_recurrence_cycle_symbolic_v2.py`).

### Regression tests

- `tutte/tests/test_chain_recurrence.py` — 5 chain templates (K*{2,2}, K*{3,3}, K*4+M_2, K_5+M_2, K*{2,2} modular)

## See also

- [Engine workflow primer §7.81](../research/engine_workflow_primer.md)
  — first-principles introduction at the pipeline stage where the
  framework dispatches.
- [6.6 Cell-Quotient Tree DP](06_6_cell_quotient_tree_dp.md) — the
  general tree-quotient DP that the chain framework specializes.
- [6.8 Modular Arithmetic Pathways](06_8_modular_arithmetic_pathways.md)
  — CRT + Lagrange recovery of the full polynomial from a grid of
  modular points (consumes the chain recurrence's per-point evaluator).
- [Literature catalog: Linear recurrences for Tutte polynomials](../research/literature_search.md)
  — Noy-Ribò 2007, Fischer-Makowsky 2008, Kotek-Makowsky-Ravve 2013
  (bi-iterative families — the 2D extension open direction).
