# tutte/benchmarks

Standalone benchmarking suite comparing the CEJ synthesis engine against NetworkX.

## Usage

```bash
# Run benchmark (synthesizes all named graphs + atlas from empty table)
python -m tutte.benchmarks.benchmark --timeout 300 --nx-timeout 300

# Compare two benchmark runs (e.g., across branches)
python -m tutte.benchmarks.benchmark --compare run_a.json run_b.json
```

## What Gets Benchmarked

- **Named graphs**: Complete, cycle, path, wheel, grid, Petersen, D-Wave topologies
- **Graph atlas**: All connected graphs from `nx.graph_atlas_g()` up to 7 nodes
- **Engines**: CEJ (`SynthesisEngine`) vs NetworkX reference (`nx.tutte_polynomial`)
- **Metrics**: Wall-clock time per graph, speedup ratios, minor relationship discovery

The CEJ engine builds up a rainbow table as it goes; NetworkX runs with
no table as the reference oracle.

### NetworkX per-family give-up

Graphs are processed in ascending edge-count order, so the members of a
scaling family (e.g. `C_*`, `Cm*`, `Pm*`, `Grid_*`, `K_{a,b}`, Zephyr)
appear from easiest to hardest. Once NetworkX times out on one member of
a family, the benchmark gives up on every later (strictly harder) member
of that same family rather than burning another `--nx-timeout` proving it
can't finish. Atlas graphs and unparametrised one-offs (Petersen,
Heawood, …) are each their own family, so a single timeout never
suppresses unrelated graphs.

Results are written to `tutte/data/benchmark_results.json`.
