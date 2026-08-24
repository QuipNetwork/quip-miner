# tutte/deprecated

Modules that are **not on any live `SynthesisEngine`
path** but are kept (rather than deleted) because each is a *validated-but-not-winning*
algorithm or a recorded negative result that may inform future work. The test suite still
exercises them from here, so they don't bit-rot.

Moved here during the 2026-05-28 dead-code audit. Nothing in `tutte/` (outside this
directory and the tests) imports these.

| Module | What it is | Why deprecated |
|---|---|---|
| `cell_quotient_interleaved.py` | Hamiltonian-path cell-quotient DP for 2D grids with **shared-anchor / closing-edge** ("interleaved chord") structure that `cell_quotient_grid_dp_streamed` rejects. | A corrected cell-template × grid-size sweep showed it has **no graph it uniquely solves in practical time**: where the engine is fast (low-tw cells) it isn't needed, and in its exclusive shared-anchor niche (K₃,₃/K₄,₄ at 2×3+) it walls >200s like the engine. The Cm3 wall is *structural* (S₄ᵏ per-cell orbit explosion 109→167K; see `research/data/cm3_interleaved_attempt.md`), and it raises `NotImplementedError` on K₃,₃-cell grids. Could be revived only with the documented σ_idr C-extension **and** an algorithmic break of the orbit wall. |
| `sigma_equivariant_dp.py` | σ-equivariant per-orbit Tutte DP on the 2-fold cover (`compute_tutte_per_orbit_mod`). | Correct (validated vs. brute force), but **0.01–0.43× the speed** of the general treewidth DP while computing only a single modular point, with severe state blow-up (107K states on Q4). Never wins, never wired into the engine. |
| `signed_treewidth.py` | Treewidth-based DP for the *signed* Tutte polynomial. | Superseded by the faster elimination-order signed DP (`signed_elim_dp.py` + `_signed_elim_c.py`, now also here). No live importer; the prior README claim that it "backs σ-equivariant evaluations" was stale. |
| `signed_elim_dp.py` | Vertex-elimination DP for the *signed*/twisted Tutte polynomial (modular point evaluation for full-polynomial recovery via interpolation). | No live engine consumer: the σ-equivariant chord-ordering path it once fed is gone; the live signed piece is `find_best_sigma` in `roots/signed_quotient.py`. Test-only via `tests/test_signed_elim.py`. |
| `_signed_elim_c.py` | cffi C extension for the `signed_elim_dp.py` inner loop. | Moved alongside its only caller, `signed_elim_dp.py`. No live importer. |
| `multivariate.py` | Multivariate / Sokal-`Z` form of the Tutte polynomial (`MultivariateTutte`, `UniformZ`). | Test-only: no live engine path computes T from the multivariate form. Exercised from `tests/test_tutte.py`. |
| `cell_quotient_cycle.py` | Cycle-topology cell-quotient DP (`compute_cycle_dp`) for rings of isomorphic cells. | Shadowed by the grid cell-quotient DP (`roots/cell_quotient_grid.py`), which the engine reaches first; never the winning path. Test-only via `tests/test_roots.py`. |
| `cell_quotient_hybrid.py` | Chord-rule cycle-close hybrid cell-quotient DP. | Negative result: never beat the grid / chain cell-quotient paths on any target. No live importer. |
| `potts_tensor_network.py` | Potts / `opt_einsum` tensor-network evaluation of `Z(G; q, v)`. | **Negative result**: same `q^treewidth` asymptotic wall as `treewidth_dp` (see memory `project_tensor_network_negative`). Never beat the existing path on the hard targets. |
| `signed_quotient_pipeline.py` | Brute-force + 2D-Lagrange-interpolation machinery for computing T via σ-quotients (the bulk of the old `roots/signed_quotient.py`). | Test-only: no live engine path computes T from modular point-values. The **live** σ-finder `find_best_sigma` stays in `roots/signed_quotient.py`; only this pipeline moved. |
| `interpolation.py` | Dense 1D/2D Lagrange interpolation + CRT helpers (mod p). | Only ever *executed* by the signed pipeline above (and research modular-interp scripts). The live engine operates on `TuttePolynomial` objects directly and does no modular interpolation. |
| `sparse_interp.py` | Sparse polynomial interpolation (Prony / Berlekamp–Massey, adaptive Lagrange grid). | Reachable only through the (now-deprecated) signed pipeline; its genuinely-sparse functions had zero callers even there. |

## Reviving something here

These modules use absolute/`..`-relative imports back into the live tree, so they import and
run as-is. If a module earns its place on a live path again, move it back to its home package
(`roots/`, `graphs/`) and re-point the tests. The git history and the `research/` findings docs
record the experiments that led to each decision.
