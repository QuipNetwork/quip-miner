"""Warm up the multigraph cache with chord-peel intermediates.

The default `warmup_lookup_table.py` populates the rainbow table with
TARGET polynomials only. Chord-rule sub-syntheses produce per-edge
contraction multigraphs that are NOT auto-persisted by the existing
warmup. This script runs the dispatcher on D-Wave-class targets with
`promote_cache_on_finish=True`, so every intermediate multigraph
(canonical_key → polynomial) gets merged into
`tutte/data/multigraph_lookup_table.{bin,json}`.

Why this matters for cost calibration:
  The merged `_try_decomposition_chord_peel` ranks decompositions by
  `predicted_chord_cost = len(chord_edges) × per_edge × tw_ratio`.
  Heterogeneous atoms produce FEWER chord edges (1 vs 4 on Z(1,2)) but
  each one's contraction sub-synthesis dominates the cost because the
  produced graph is densely connected and doesn't hit any cached
  intermediates from prior dispatch steps. After this warmup, the het
  contractions become cache hits, dropping the per-edge cost from
  ~148s/edge cold to a much smaller number — finally making the
  cost predictor `_INTER_HET_PER_EDGE` empirically meaningful.

Idempotent: existing cache entries are preserved (`setdefault`).

Usage:
    python -m tutte.scripts.warmup_chord_peel_cache
    python -m tutte.scripts.warmup_chord_peel_cache --target Z1_2
    python -m tutte.scripts.warmup_chord_peel_cache --mode het
"""
from __future__ import annotations

import argparse
import sys
import time

import dwave_networkx as dnx

from tutte.graph import Graph
from tutte.synthesis.engine import SynthesisEngine

TARGETS = [
    ("Z1_1", lambda: dnx.zephyr_graph(1, 1)),
    ("Cm1_2", lambda: dnx.chimera_graph(1, 2)),
    ("Cm1_3", lambda: dnx.chimera_graph(1, 3)),
    ("Z1_2", lambda: dnx.zephyr_graph(1, 2)),
    # Cm2 and Pm2 are above the n<=30 chord-peel gate so they hit
    # `cell_quotient_grid_dp_streamed` / `treewidth_dp` directly, but
    # both still produce multigraph contractions deep in their dispatch
    # paths that benefit from caching. Cm2 ~40s cold, Pm2 ~63s cold.
    ("Cm2", lambda: dnx.chimera_graph(2)),
    ("Pm2", lambda: dnx.pegasus_graph(2)),
    # Z(1,3) / Z(2,1) / Z(1,4) — m=3-class targets that hit
    # `_synthesize_connected → _synthesize_from_k2` (n>30 gate). Each
    # cold synth is 25+ min; populates intermediate multigraph cache
    # for future runs. Per Z(1,2) precedent (40s → 1.2s, 35-130×) the
    # warm-cache speedup amortizes the one-time cost.
    # Run individually via `--target Z1_3 --timeout 2400`.
    ("Z1_3", lambda: dnx.zephyr_graph(1, 3)),
    ("Z2_1", lambda: dnx.zephyr_graph(2, 1)),
    ("Z1_4", lambda: dnx.zephyr_graph(1, 4)),
]


def _run_one(name: str, g: Graph, mode: str, timeout_s: int) -> None:
    """Run one target through the dispatcher; flush multigraph cache on finish.

    `mode` selects atom-detector bias:
      - `default`: dispatcher picks cheapest predicted (currently legacy).
      - `legacy`: monkey-patch heterogeneous → [].
      - `het`: monkey-patch legacy → []; force het atoms.
      - `both`: run twice, legacy then het, into the SAME engine cache.
    """
    from tutte.graphs import atom_detection as ad

    def run_mode(submode: str) -> None:
        orig_legacy = ad.find_disjoint_atoms
        orig_het = ad.find_atoms_heterogeneous
        if submode == "legacy":
            ad.find_atoms_heterogeneous = lambda g, **kw: []
        elif submode == "het":
            ad.find_disjoint_atoms = lambda g: []
        try:
            # Verbose=True for long-running targets so we can see chord
            # progression. Output to stderr to keep separable from the
            # summary line.
            engine = SynthesisEngine(
                verbose=True,
                promote_cache_on_finish=True,
            )
            engine.skip_target_lookup = True

            # Periodic checkpointer: flush multigraph cache to disk every
            # CHECKPOINT_SEC seconds via threading.Timer so partial progress
            # on long synth (e.g. Z(1,3) cold ~1-3hr) survives interruption.
            # Without this, only `promote_cache_on_finish` on successful
            # completion writes the cache — a timeout kill wastes all work.
            import threading
            CHECKPOINT_SEC = 60.0
            stop_event = threading.Event()
            def checkpointer():
                # Snapshot the multigraph cache and flush directly to
                # disk WITHOUT calling _flush_cache_to_table (which iterates
                # over self._cache + self._synth_accum_graphs that are not
                # thread-safe). Only the multigraph_lookup_table merge is
                # idempotent (setdefault). Snapshot via dict(...) avoids
                # concurrent-modification errors.
                try:
                    from tutte.lookup.core import (
                        load_default_multigraph_table,
                        save_default_multigraph_table,
                    )
                    while not stop_event.wait(CHECKPOINT_SEC):
                        snapshot = dict(engine._multigraph_cache)
                        if not snapshot:
                            continue
                        try:
                            existing = load_default_multigraph_table()
                            new_entries = 0
                            for k, p in snapshot.items():
                                if k not in existing:
                                    existing[k] = p
                                    new_entries += 1
                            if new_entries > 0:
                                save_default_multigraph_table(existing)
                            print(
                                f"  [checkpoint] {name} [{submode}]: "
                                f"mg_cache_snapshot={len(snapshot)}, "
                                f"new_to_disk={new_entries}",
                                file=sys.stderr, flush=True,
                            )
                        except Exception as ex:
                            print(
                                f"  [checkpoint ERR] {ex}",
                                file=sys.stderr, flush=True,
                            )
                except Exception:
                    pass
            t_chkpt = threading.Thread(target=checkpointer, daemon=True)
            t_chkpt.start()

            t0 = time.perf_counter()
            print(f"  {name} [{submode}] starting (verbose, "
                  f"checkpointing every {CHECKPOINT_SEC:.0f}s)...",
                  file=sys.stderr, flush=True)
            try:
                r = engine.synthesize(g, max_depth=20)
                wall = time.perf_counter() - t0
                mg_size = len(engine._multigraph_cache)
                print(
                    f"  {name} [{submode}]: {wall:.1f}s, method={r.method}, "
                    f"mg_cache={mg_size}",
                    file=sys.stderr, flush=True,
                )
            finally:
                stop_event.set()
                t_chkpt.join(timeout=5)
        finally:
            ad.find_disjoint_atoms = orig_legacy
            ad.find_atoms_heterogeneous = orig_het

    if mode == "both":
        run_mode("legacy")
        run_mode("het")
    else:
        run_mode(mode)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target", default="all",
        choices=["all"] + [n for n, _ in TARGETS],
        help="Which target(s) to populate cache for.",
    )
    p.add_argument(
        "--mode", default="both",
        choices=["default", "legacy", "het", "both"],
        help="Atom-detector bias: 'both' runs legacy + het (default).",
    )
    p.add_argument(
        "--timeout", type=int, default=600,
        help="Per-(target,mode) wall budget (default 600s).",
    )
    args = p.parse_args()

    targets = TARGETS if args.target == "all" else [
        (n, b) for n, b in TARGETS if n == args.target
    ]

    for name, builder in targets:
        try:
            gnx = builder()
        except Exception as e:
            print(f"  {name}: build failed: {e}", file=sys.stderr)
            continue
        g = Graph.from_networkx(gnx)
        print(f"=== {name}: n={g.node_count()}, m={g.edge_count()}, "
              f"mode={args.mode} ===", file=sys.stderr, flush=True)
        _run_one(name, g, args.mode, args.timeout)

    return 0


if __name__ == "__main__":
    sys.exit(main())
