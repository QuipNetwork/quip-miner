# `tutte/transfer_matrix/` — Tutte polynomial for periodic lattice strips

This package computes the Tutte polynomial of *periodic lattice strips* — grids
(`P_w × P_n`), triangular, honeycomb, square-octagon, and elongated-triangular
strips — via the **Fortuin-Kasteleyn random-cluster transfer matrix** acting on
non-crossing partition states. A C extension (cffi) accelerates the column
sweep; a pure-Python CRT fallback handles overflow for wide strips.

> **Why a separate module?** Family recognition handles trees, cycles, wheels,
> fans, ladders (= 2 × n grids), books, gears, prisms and Möbius ladders in
> O(n + m). It explicitly returns `None` for grids with `m ≥ 3` (`grid_recurrence`
> in `tutte/family_recognition/formulas.py`). This module fills that gap, plus
> the other Archimedean periodic strips, without giving up the O(V + E)
> detection cost or the O(strip-length × Catalan(width)²) sweep cost.

## When the engine uses this package

`tutte/synthesis/engine.py` step **1.5** dispatches to
`compute_tutte_via_transfer_matrix(graph)`. It fires after family recognition
(step 1) and **before** the O(n² log n) canonical-key computation (step 2), so
lattice strips never pay the canonical-key cost.

It returns `None` and the engine continues if the graph is not one of the
supported periodic-strip families.

## Supported families

| Family                          | Period | Detection            |
| ------------------------------- | -----: | -------------------- |
| Grid `P_width × P_length`       |      1 | BFS-based isomorphism |
| Triangular strip                |      1 | BFS + diagonal check |
| Honeycomb / hexagonal (brick)   |      2 | BFS + 2-cell pattern |
| Square-octagon (4.8.8)          |      4 | BFS + 4-cell pattern |
| Elongated triangular (3.3.3.4.4) | 1     | BFS + adjacency match |

All recognition is O(V + E). Each family has its own `_recognize_*` function in
`lattice_recognition.py` that returns `StripProperties` (width, length, unit-cell
edge pattern) or `None`.

## Public API

```python
from tutte.transfer_matrix import (
    compute_tutte_via_transfer_matrix,   # high-level entry point
    detect_periodic_strip,               # recognition only — returns StripProperties
    build_transfer_matrix,               # FK transfer matrix from unit-cell edges
    enumerate_noncrossing_partitions,    # state-space enumeration
    partition_index_map,                 # state ↔ index map
    direct_multiply,                     # pure-Python matrix-vector sweep
    build_initial_vector,                # initial FK state (singleton partition)
    extract_tutte_polynomial,            # FK state → Tutte polynomial in (a, b)
    CATALAN_NUMBERS,                     # state-count table
    MAX_TRANSFER_MATRIX_WIDTH,           # safety cap on the strip width
)
```

## Pipeline overview

```
detect_periodic_strip → unit-cell edges → build_transfer_matrix
                                              ↓
   ┌─── C-extension `sweep_in_c` (preferred when available) ─────┐
   │                                                              │
   │  build_initial_vector → (length-1) × (matrix × vector)        │
   │                                                              │
   └─── pure-Python `direct_multiply` (CRT fallback for big coeffs) ┘
                                              ↓
                              extract_tutte_polynomial(state_vec)
                                              ↓
                              TuttePolynomial(x, y)
```

## File map

| File                       | Responsibility                                                              |
| -------------------------- | --------------------------------------------------------------------------- |
| `__init__.py`              | Public API; orchestrates detect → build → sweep → extract                   |
| `lattice_recognition.py`   | Family-specific O(V + E) recognition; returns `StripProperties`             |
| `core.py`                  | Non-crossing partition enumeration; transfer matrix construction            |
| `sweep.py`                 | Pure-Python matrix-vector sweep + initial vector builder                    |
| `_transfer_matrix_c.py`          | cffi-built C extension: full sweep + CRT combine for large coefficients      |
| `extraction.py`            | FK-state-vector → Tutte polynomial via binomial conversion `(a, b) → (x, y)` |

## Complexity

- Detection: **O(V + E)**, dominated by BFS-based isomorphism.
- Transfer matrix build: **O(C_w²)** where `C_w = Catalan(w)` (e.g., `C_3 = 5`,
  `C_6 = 132`, `C_10 = 16796`).
- Sweep: **O((length - 1) × nnz)** where `nnz` is the number of nonzero
  matrix-vector products per column. Multiplied by polynomial-coefficient
  cost in `(a, b)`.

`MAX_TRANSFER_MATRIX_WIDTH` caps `w` so the state space stays tractable in
pure Python; the C extension lifts this for wider strips on supported builds.

## Tests

`tutte/tests/test_lattice_graphs.py` (445 fast + 5 slow params). Coverage:

- Recognition: each family on its canonical builder and on negative cases
  (non-lattice graphs of the same fingerprint must NOT be recognized).
- Sweep correctness: `T(2, 2) = 2^|E|` and `T(1, 1) = #ST` (vs `Kirchhoff`).
- Engine integration: `SynthesisEngine.synthesize` on grids 3×3 … 6×8.
- C-extension parity: `direct_multiply` matches `sweep_in_c` bit-for-bit.

## Related techniques

- Family recognition's `grid_recurrence(m, n)` (in
  `tutte/family_recognition/formulas.py`) handles `m = 1` (path) and `m = 2`
  (ladder via order-2 recurrence). This module owns `m ≥ 3`.
- The `roots/cell_quotient_*` family computes rooted Tutte transfer operators
  for D-Wave cell shapes (K_{4,4} chains, Pegasus, Zephyr) — different state
  representation (compressed orbit dicts of `TuttePolynomial`s), different
  target families. See `tutte/roots/README.md`.
