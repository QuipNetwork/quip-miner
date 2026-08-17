# Zero-field (h=0) rerun of the reads × diversity × min-valid frontier

## Context

The existing `reads_diversity_count` study maps the CUDA-SA winnability frontier on
random Ising instances with **ternary fields** (`h∈{−1,0,+1}`, `J∈{−1,+1}`). We want
to rerun the same experiment on the **zero-field / J-only** problem class (all
`h_i=0`) to isolate how the coupling structure alone drives the count (`k`),
diversity (`D`), and `num_reads` gates — and to compare the two classes.

Exploration surfaced two things that would silently corrupt a naive rerun, plus a
gate inconsistency:

1. **The energy band shifts.** With `h=0` the field term vanishes, moving the
   ground-state energy by **~+474** (≈3.2%, `shared/energy_utils.py`
   `expected_solution_energy`/`calc_energy_range`). The `−14635…−15020` ladder is the
   wrong band for `h=0` and must be re-derived.
2. **Z2 (spin-inversion) symmetry.** `E(s)=E(−s)` exactly when `h=0`, so every
   solution has an equal-energy twin. The **diversity** metric already uses a
   flip-invariant distance (`min(hamming, N−hamming)`,
   `shared/quantum_proof_of_work.py:403`), but the **count gate**
   (`n_unique_below_threshold`) dedups *raw* spin rows (`_unique_rows`,
   `compute_solution_meta` at `:330`) and would **double-count twins**.

**Decisions locked (with user):**
- **Count gate → flip-invariant** (gauge-canonicalize twins; consistent with diversity).
- **Target ladder → matched difficulty** (re-derive `h=0` energies at the same
  difficulty rungs; reuse each rung's original adapt params).
- **Sequencing → calibrate + pilot first** (1–2 SA rungs) before the full sweep.
- **Cleanups to fold in:** multi-seed floor error bars; pool/dedupe boundary re-runs;
  standardize `m` + fill the `(target, reads)` grid.

The new `h=0` dataset is built to the *clean spec from the start* (uniform `m`, full
grid, ≥3 seeds at the floor, flip-invariant count). The existing `h≠0` dataset gets
the free post-processing cleanup (pooling) now; the expensive `h≠0` backfills are
listed as optional operator GPU tasks.

All GPU runs execute on the **CUDA box (operator)** — this Mac can't run CUDA. My
deliverable is the code + commands; consolidation/analysis run on CPU afterward.

## Part 1 — Code changes (main repo; new branch `feat/h0-field-sweep`)

All changes are **additive / opt-in — no consensus or default-behavior change.**
`generate_ising_model_from_nonce` *already* accepts `allowed_h`, so this is plumbing.

**1a. Thread an `h`-spec through the feeder** (zero field is `allowed_h=AllowedValueSet((0,))`):
- `shared/ising_feeder.py` — add `allowed_h=None` to `RandomIsingFeeder.__init__` and
  to `_generate_one_model` (`:60`); pass it to the pool `submit(...)` and into
  `generate_ising_model_from_nonce(nonce, nodes, edges, allowed_h=allowed_h)` (`:76`).
  `AllowedValueSet` is picklable, so it crosses the `ProcessPoolExecutor` cleanly.
- `test_results/cuda_tts_test/tools/cuda_tts_canary.py` — add `--h-spec` (default
  `"-1,0,1"`; `"0"` ⇒ zero field), parse to `AllowedValueSet`, pass to the feeder
  (`_run_target`, `:524`). **Record `h_spec` in each cell summary JSON** so the
  dataset is self-describing.

**1b. Flip-invariant count gate (opt-in)** — `shared/quantum_proof_of_work.py`:
- Add a gauge-canonicalization helper (flip each spin row to a fixed convention, e.g.
  anchor qubit `= +1`, deterministic tie-break) and a `gauge_fix: bool = False` param
  to `compute_solution_meta` (`:330`). When `True`, canonicalize `samples` rows
  *before* `_unique_rows`, so twins collapse → `n_unique_*` becomes flip-invariant.
  Default `False` ⇒ **zero change to chain/production callers**. Diversity is
  unaffected (already flip-invariant).
- Canary threads `gauge_fix=True` into its `compute_solution_meta` call when
  `--h-spec 0` (auto) or `--gauge-fix-count`.
- Verify on existing `h≠0` data that raw ≈ flip-invariant (twins negligible at
  `h≠0`), so the cross-class comparison is valid even though the `h≠0` set used raw.

## Part 2 — Matched-difficulty ladder (new `derive_h0_ladder.py`)

New script under `test_results/reads_diversity_count_h0/`. Reuses
`shared/energy_utils` (`calc_energy_range`, `energy_to_difficulty`) and the existing
`dataset.json`:
- For each original rung `E_old`, compute its difficulty `d` on the `h≠0` curve.
- Map `d → E_new` on the `h=0` band (`calc_energy_range(h_values=(0,))`); expect
  `E_new ≈ E_old + 474`, curve-accurate.
- Pull each rung's `num_sweeps` and reads-grid from `dataset.json` (so the `h=0` run
  reuses the original per-rung compute — **no `adapt_parameters` change needed**).
- Emit `h0_ladder.json`: `[{E_h0, difficulty, num_sweeps, reads_grid, seeds}]`.

## Part 3 — Calibrate + pilot (operator, CUDA box)

1. **Pilot run** at 1–2 rungs (one mid-band, one near-frontier), SA, `--h-spec 0`,
   `m=10000`, pinned params from `h0_ladder.json`.
2. **Validate before committing the full sweep:** (a) achieved `best_energy`
   distribution lands where the matched-difficulty ladder predicts (else recalibrate
   the band empirically); (b) flip-invariant vs raw `n_unique` gap confirms the twin
   effect; (c) yields are in a sane range. Adjust `h0_ladder.json` if the analytic
   band is off.

## Part 4 — Full h=0 sweep (operator, CUDA box) — clean spec

Drive via the existing runner/shell scripts pointed at a **new root**
`test_results/reads_diversity_count_h0/data/`, with `--h-spec 0` and pinned params:
- **Reads sweeps** (`design=reads_sweep`) at every rung — `m=10000` uniform, full
  `1,2,4,…,adapt` grid, **including the fine-region targets** that were adapt-only
  before (grid-fill cleanup).
- **Adapt frontier** (`design=adapt_only`) for any rung not reads-swept.
- **Multi-seed at the floor:** ≥3 seeds at the deep targets, output namespaced by
  seed (`…/e<E>/r<reads>/s<seed>/`), for tail error bars.

Runner change: add `--params-config h0_ladder.json` (skip `adapt_parameters`, use the
table) and `--seeds` to `reads_sweep_run.py`. Reuse `run_reads_sweep.sh` /
`run_cutoff_sweep.sh` via `OUT_ROOT`, `TARGETS`, `M` env overrides + the new flags.

Rough cost: uniform `m=10k` (band was 5k) + multi-seed floor + grid-fill push this
**above the ~155 GPU-h original**; the pilot refines the estimate.

## Part 5 — Consolidate, analyze, compare

**`consolidate.py`** (extend; serves both datasets):
- Add `--data-root` / `--out` so it builds `dataset_h0.json` + `dataset_adapt_h0.json`.
- Add **pooling**: merge cells with identical `(E, num_reads, num_sweeps, mode, h_spec)`
  across regions *and seeds* into combined stats (sum models, recompute `C`/`qbar`/
  `best_energy_min` over the union; provenance becomes a list). This delivers the
  multi-seed aggregation (h=0) *and* the boundary-re-run pooling (h≠0:
  −15000 ×3, −15010 ×2, −14900 ×2) in one mechanism.
- Record `h_spec` per cell.
- **Note:** pooling re-fits the model at `−14900` (band 5k + anchor 10k → 15k), so
  `dataset_adapt.json` no longer byte-matches `measured_yield.json` — expected. Re-run
  `yield_model.py`; constants should move only within noise (flag any delta).

**Analysis + Notion:** run the analysis scripts on the `h=0` dataset, then produce an
**h=0 vs h≠0 comparison** — do the 8 results hold? how far does the frontier shift?
how does flip-invariant counting change the `k`-gate? New Notion page (sibling in the
D-Wave data source) cross-linked to the existing one, same `[IMAGE:]`-placeholder
convention (Notion MCP can't upload binaries).

## Cleanups — disposition

| cleanup | h=0 (new) | h≠0 (existing) |
|---|---|---|
| flip-invariant count | built in (Part 1b) | raw kept; verify raw≈flip-inv |
| multi-seed floor | built in (Part 4) | optional operator GPU backfill |
| pool/dedupe re-runs | via pooling (Part 5) | **done now** (post-processing) |
| standardize m=10k + fill grid | built in (Part 4) | optional operator GPU backfill |

## Extra cleanliness (cheap — recommend including)

These cost ~no GPU and make the dataset more auditable; not in the four you picked,
flagged for a quick yes/no:
- **Record per-cell provenance**: the `--seed`, `h_spec`, `git` commit, and topology
  id in each cell summary + carried into `dataset.json`. Makes every number
  reproducible from the file alone.
- **Keep raw `n_unique` as a diagnostic column** *alongside* the flip-invariant gate
  value (gate stays flip-invariant per your decision). The `raw/flip-inv` ratio then
  directly quantifies the Z2-twin effect per cell — a free, interesting measurement
  rather than a discarded one.
- **Determinism check**: re-run one cell twice at the same seed and assert bit-identical
  attempts, confirming the canary's reproducibility claim before the big spend.
- **One high-m validation rung**: a single near-frontier target at `m≈50k` to pin the
  1-in-10k tail (frontier / deepest-reach) tighter than the multi-seed 10k cells.

## Verification

- **Unit:** `h=0` feeder produces `all(h_i==0)` and unchanged `J`; default path still
  ternary (a quick `generate_ising_model_from_nonce` check). `compute_solution_meta`
  with `gauge_fix=True` collapses a hand-built `s/−s` pair to `n_unique=1`; default
  `False` unchanged.
- **No-regression:** existing tests for `quantum_proof_of_work`/`ising_feeder` pass
  (`pytest shared/ -k "ising or proof or feeder"`).
- **Pilot gate:** achieved energies match the matched-difficulty ladder (Part 3).
- **Consolidate:** `dataset_h0.json` cell counts match the launched grid; pooled
  `h≠0` `−15000` shows summed `m` (25k); `yield_model.py` re-fit constants logged.
- **Comparison sanity:** at matched difficulty, `h=0` flip-invariant `k`-yield and
  `D`-survival are reported beside the `h≠0` numbers.

## Constraints

- **No consensus/default change** — `allowed_h` already exists; `gauge_fix` defaults
  `False`; feeder/canary changes are additive. Chain validation path untouched.
- **Operator runs all CUDA** (pilot + full); provide commands, don't execute GPU here.
- `shared/` + canary edits land on a feature branch (`feat/h0-field-sweep`) → MR; the
  `test_results/reads_diversity_count_h0/` tree is local/gitignored like its sibling.

## Phase C — full (E × R × S) characterization (deferred; after the TTS decision)

The Phase A/B work optimizes TTS: it finds the (num_reads, num_sweeps) policy and
the frontier map, stopping compute as soon as a sweeps-doubling stops paying
(early-stop rule: yield ratio < 2 with ≥20 wins in the prior rung). It deliberately
does NOT articulate the full improve-then-degrade response per target. That fuller
study is this phase, run only after the TTS policy ships:

- **Goal:** the full response surface `yield(E, R, S)` and TTS iso-surfaces over
  E ∈ [−14430, −14560], R ∈ {32…256}, S ∈ {1024 … knee+1 octave}, h=0; one ternary
  anchor slice (e.g. −14900/−14950 × {1×, 2×}) for cross-regime shape comparison.
- **Reuse:** every Phase A/B cell conforms to the Phase C conventions and slots into
  the grid as-is — pinned (R, S) per cell, recorded seed, per-cell out-dirs
  (`phaseA_e<E>_r<R>_s<S>_seed<seed>/`), self-describing summaries
  (h_spec/gauge_fix/c_range/sweep_mult fields), new 4577-node topology only.
  Phase C is gap-filling, not a rerun: design the grid, subtract cells already on
  disk (including early-stopped ladder rungs left unmeasured), run the difference.
- **Allocation:** fill remaining cells by information gain per GPU-hour (the
  knee/frontier neighborhoods first; deep 0-win regions get Wilson upper bounds at
  modest m, not precision).
- **Analysis:** joint surface fit (the analyze_h0_opt.py machinery generalizes:
  per-read reach q(E,S) × count-gate mixture link), published with per-cell raw data
  so the fit is reproducible; documents where and why TTS degrades past the knee.
