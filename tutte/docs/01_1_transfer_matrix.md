# 1.5 Transfer Matrix for Periodic Lattice Strips

## Summary

This technique computes the Tutte polynomial of a graph that is a **periodic
lattice strip** — a finite "row × column" piece of an infinite Archimedean
tiling — by running the Fortuin–Kasteleyn (FK) random-cluster transfer matrix
across the columns. The matrix entries are polynomials in the cluster
parameters `(a, b)`; after the sweep, a binomial change of variables maps the
final state vector to the Tutte polynomial in `(x, y)`.

**Complexity:** O(V + E) detection plus O(`length × C_width²`) sweep, where
`C_width = Catalan(width)` is the dimension of the state space.

**Supported families:** `P_width × P_length` grid (period 1), triangular
(period 1), honeycomb / hexagonal brick (period 2), square-octagon /
truncated-square (period 4), elongated-triangular (period 1).

## When It Is Used

Pipeline **step 1.5** in `tutte/synthesis/engine.py` — runs after family
recognition (step 1) and **before** the O(n² log n) canonical-key
computation (step 2). The entry point returns `None` when the graph is not a
periodic-strip family, in which case the engine continues to canonical-key
lookup as usual.

**Entry point:** `compute_tutte_via_transfer_matrix(graph) → Optional[TuttePolynomial]`

**Pipeline placement rationale:** family recognition explicitly returns
`None` for grids with `m ≥ 3` (`grid_recurrence` in
`tutte/family_recognition/formulas.py`). Without this technique those graphs
fell through to cell-quotient DP / treewidth DP / k-sum decomposition. For a
4×4 grid, that meant minutes of synthesis where transfer matrix finishes in
~1 second.

## Notation

| Symbol | Definition |
|---|---|
| `w` | strip width (number of vertices in a column) |
| `n` | strip length (number of columns) |
| `C_w` | the `w`-th Catalan number, `binom(2w, w) / (w + 1)` — the number of non-crossing partitions of `w` boundary vertices |
| `M` | the per-column transfer matrix, size `C_w × C_w`, entries are polynomials in `(a, b)` |
| `s_i` | FK state vector after sweeping `i` columns |
| `e_0` | initial state — the singleton partition `{{0}, {1}, …, {w − 1}}` |
| `a`, `b` | random-cluster variables; `a = y − 1` and `b = q / (y − 1)` in the FK parameterization |
| `T(G; x, y)` | the standard Tutte polynomial |

## Why Periodic Lattices

A *periodic strip* is a graph whose adjacency is determined by:

1. A finite **unit cell** of `w` vertices (a single column).
2. A repeating **cell-pair edge pattern** that tiles `n` times horizontally
   plus a fixed **boundary** terminating both ends.

Every interior column has the same pattern of intra-column edges, and every
adjacent column pair has the same pattern of inter-column edges. The transfer
matrix's columns are indexed by **non-crossing partitions** of the current
boundary (the right-hand column), and the matrix is identical at every step.

The Tutte polynomial of `G` then satisfies:

```
T(G; x, y) = extract(M^{n−1} · e_0)
```

where `extract` is the FK-to-Tutte conversion described below.

The five supported families all admit a periodic unit-cell representation:

| Family                          | Period | Unit-cell edges (3-tuples `(row_a, row_b, is_cross_column)`) |
| ------------------------------- | -----: | ------------------------------------------------------------ |
| Grid `P_w × P_n`                |      1 | `w − 1` within-column edges, `w` cross-column edges          |
| Triangular                      |      1 | grid pattern + one diagonal per cell                         |
| Honeycomb (brick wall)          |      2 | period-2 pattern (alternating brick offset)                  |
| Square-octagon (4.8.8)          |      4 | period-4 pattern                                             |
| Elongated triangular (3.3.3.4.4) | 1     | grid pattern + alternating diagonals                          |

## Algorithm Overview

```
1. Recognition (O(V + E)):
   - Compute structural fingerprint (degree sequence, bipartiteness, etc.)
   - For each supported family in turn, attempt BFS-based isomorphism
     against the canonical builder.
   - On success, return `StripProperties(width, length, transition_patterns,
     num_vertices, first_col_edges)`.
2. State enumeration:
   - Enumerate the `C_w` non-crossing partitions of {0, 1, …, w − 1}.
   - Build `partition_index_map: partition → int`.
3. Transfer matrix construction:
   - For each pair `(P_in, P_out)` of non-crossing partitions:
     - Compute the FK polynomial-in-`(a, b)` weight of the
       partition-merging transition induced by the unit-cell edges.
   - Store as `M[i][j] = polynomial[(a, b)]`.
4. Sweep:
   - `s_0 = e_0` (the initial singleton partition).
   - For `i = 1 … length − 1`:
     - `s_i = M × s_{i−1}` (matrix-vector multiply with polynomial entries).
5. Extraction:
   - Project the final state vector onto the all-blocks-joined partition.
   - Apply the binomial conversion `(a, b) → (x, y)` to obtain
     `T(G; x, y)`.
```

The C extension (`_transfer_matrix_c.py`) fuses steps 4 and 5 into a single
`sweep_in_c` call. For widths where polynomial coefficients exceed int64, a
Chinese-Remainder-Theorem (CRT) path multiplexes the sweep over several primes
and reconstructs the integer coefficients at the end.

## Why FK Rather Than Direct Tutte

The Tutte polynomial in `(x, y)` is *not* multiplicative under composition of
strips because `T(G_1 ∪ G_2)` depends on the connectivity of `G_1 ∪ G_2`, not
just on the connectivity of `G_1` and `G_2` separately. The FK random-cluster
polynomial `Z(G; q, v)` is multiplicative under disjoint union and obeys a
*transfer* relation under column-pair joining: when a new column is added,
each non-crossing partition of the new boundary can be reached from each
non-crossing partition of the old boundary by a definite weight in `(a, b)`.
That weight is what populates `M`.

The conversion at the end is the well-known FK ↔ Tutte change of variables:

```
(x − 1)(y − 1) = q,    y − 1 = v.
```

`extract_tutte_polynomial` collects the FK state into a polynomial in `(a, b)`
and substitutes back to `(x, y)` via binomial expansion.

## When This Technique Fails

`compute_tutte_via_transfer_matrix(graph)` returns `None` when:

- The graph fails every family's BFS-isomorphism check (most graphs).
- The strip width exceeds `MAX_TRANSFER_MATRIX_WIDTH` (state space too big
  for the pure-Python path; the C extension can lift this).

In both cases the engine continues past step 1.5 to canonical-key lookup.

## Files

| File                       | Responsibility                                                              |
| -------------------------- | --------------------------------------------------------------------------- |
| `tutte/transfer_matrix/__init__.py`              | Public API; orchestrates detect → build → sweep → extract                   |
| `tutte/transfer_matrix/lattice_recognition.py`   | Family-specific O(V + E) recognition; returns `StripProperties`             |
| `tutte/transfer_matrix/core.py`                  | Non-crossing partition enumeration; transfer matrix construction            |
| `tutte/transfer_matrix/sweep.py`                 | Pure-Python matrix-vector sweep + initial vector builder                    |
| `tutte/transfer_matrix/_transfer_matrix_c.py`          | cffi-built C extension: full sweep + CRT combine for large coefficients      |
| `tutte/transfer_matrix/extraction.py`            | FK-state-vector → Tutte polynomial via binomial conversion                  |

## Related Techniques

- **Family Recognition** ([01_family_recognition.md](01_family_recognition.md))
  — Owns trees, cycles, wheels, fans, ladders (= 2 × n grids), books, gears,
  prisms, Möbius ladders. Transfer matrix takes over for `m ≥ 3` grids and
  the other Archimedean strips.
- **Cell-Quotient Tree / Cycle / Grid DPs**
  ([06_4](06_4_cell_quotient_cycle_dp.md),
  [06_5](06_5_cell_quotient_grid_dp.md),
  [06_6](06_6_cell_quotient_tree_dp.md))
  — Different transfer-matrix idea: per-cell rooted-Tutte operators for
  D-Wave cell shapes (K_{4,4} chains, Pegasus, Zephyr). State is a compressed
  orbit dict of `TuttePolynomial`s; target families and state representation
  are disjoint from this module's lattice-strip path.

## References

- C. M. Fortuin & P. W. Kasteleyn (1972), *On the random-cluster model: I.
  Introduction and relation to other models*, Physica 57(4), 536–564.
- D. J. A. Welsh (1993), *Complexity: Knots, Colourings and Counting*,
  Cambridge University Press — Chapter 3, transfer matrix for the Potts /
  random-cluster model.
- A. D. Sokal (2005), *The multivariate Tutte polynomial (alias Potts model)
  for graphs and matroids*, in Surveys in Combinatorics — Section 4.2 on
  transfer matrices for strips.
- R. Shrock (2000–2010 series), papers on transfer-matrix Tutte
  polynomials for square-octagon and elongated-triangular lattices.
