# tutte/data

Pre-computed lookup tables used by the library, plus the latest benchmark
timings. The lookup tables ship in a compact **binary** format; the loaders
(`load_default_table`, `load_default_multigraph_table`, `load_default_merger_table`,
…) read `.bin`, which is the authoritative on-disk form.

## Files

| File | Format | Description |
|------|--------|-------------|
| `lookup_table.bin` | Binary v2 | Rainbow table (named cells + minor relationships). Loaded by default. |
| `multigraph_lookup_table.bin` | Binary | Cached multigraph intermediate polynomials. |
| `merger_lookup_table.bin` | Binary | Cell-merger lookup. |
| `rooted_lookup_table.bin` | Binary | Rooted-Tutte (boundary-keyed) lookup. |
| `benchmark_results.json` | JSON | Timing data from the most recent benchmark run. |

## Reading the binary tables

The `.bin` files are not human-readable. The **preferred way to inspect them is
the visualizer in `tutte/scripts/`**, which loads the rainbow table and serves a
browser UI:

```bash
python tutte/scripts/visualize_tutte.py   # then open the printed localhost URL
```

JSON mirrors of these tables are no longer written — `.bin` is authoritative,
and the loaders only fall back to a `.json` if the binary is missing.

## Required setup — the tables are NOT in git

Every `.bin` here is gitignored, so a **fresh clone ships none of them**. That is
deliberate (they are multi-hour measurement exports, not source), but it is not
harmless: with no table on disk every lookup MISSES and callers fall through to
full DP, which turns seconds into hours. `load_default_table()` and
`load_default_multigraph_table()` therefore emit a loud warning naming this file
rather than silently returning an empty table.

**Run the warmup before running the suite**, from the repo root:

```bash
# Seeds the D-Wave headers (Cm2, Z(1,2), ...) + cograph atoms. Idempotent.
PYTHONPATH=. python -m tutte.scripts.warmup_lookup_table

# Single target, e.g. after adding one cell:
PYTHONPATH=. python -m tutte.scripts.warmup_lookup_table --target Z1_2
```

Without it, the lookup-hit regression tests (e.g.
`test_compute_t_via_sigma_auto_dnx_lookup_hit`) fall into the full σ-DP and take
many minutes each instead of milliseconds.

> **`PYTHONPATH` is required.** Both the warmup and the benchmark are run as
> `-m` modules and will die with `ModuleNotFoundError: No module named 'tutte'`
> unless `PYTHONPATH` points at the repo root.

> **The warmup saves only after ALL targets finish.** Interrupting a long run
> (e.g. during the Cm3 target, which can exceed 30 min) discards every entry
> computed up to that point. Use `--target` for anything you cannot babysit.

## Regenerating

```bash
# Update table with new polynomials discovered during testing
PYTHONPATH=. python -m pytest tutte/tests/ -v --update-rainbow-table

# Full rebuild via standalone benchmark
PYTHONPATH=. python -m tutte.benchmarks.benchmark --timeout 300
```

> **The standalone benchmark is DESTRUCTIVE to `lookup_table.bin`.** It measures
> synthesis *from an empty rainbow table* and then saves what it built, so it
> replaces an accumulated table with a much smaller one (observed: 18,774
> entries → 1,084). Back the file up first if you care about it.
