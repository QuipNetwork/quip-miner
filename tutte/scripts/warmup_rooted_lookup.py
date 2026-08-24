"""Pre-compute T_rooted dicts for common cells and save to the rooted lookup table.

Writes both binary (`tutte/data/rooted_lookup_table.bin`, fast load) and
JSON (`tutte/data/rooted_lookup_table.json`, human-readable) forms. The
engine auto-loads the binary first via `load_default_rooted_lookup()`.
Filenames match the `*lookup_table*` `.gitignore` pattern.

Partitions are stored in CANONICAL vertex labels (0..n-1 per WL refinement)
so cache hits work across any isomorphic runtime graph regardless of its
actual label scheme.

Run as:
    PYTHONPATH=. python scripts/warmup_rooted_lookup.py [--small] [--slow]

`--small` skips Z(1,1) (~5 min cold compute).
`--slow` adds optimistic targets that may take 30+ min each: K_{5,5},
K_{5,6}, K_{6,6}.  Default keeps things under ~10 min total.
Targets unblock heterogeneous decomposition of Pegasus + Zephyr graphs by
caching T_rooted on common cell shapes.
"""
from __future__ import annotations

import os
import sys
import time

import networkx as nx
from tutte.graph import Graph
from tutte.roots.rooted_tutte import (clear_t_rooted_cache,
                                      save_rooted_lookup_default,
                                      t_rooted_cached)


def main():
    small_only = "--small" in sys.argv
    slow = "--slow" in sys.argv
    clear_t_rooted_cache()

    targets = []
    for n in range(2, 7):
        g = Graph.from_networkx(nx.complete_graph(n))
        targets.append((f"K_{n}", g, list(g.nodes)))
    # K_{a,b} cells appearing in Chimera (4,4) and Pegasus (varies). Sizes
    # up to K_{4,5} are seconds-cold (~50s); K_{5,5}+ is ≥10 min and gated
    # behind --slow. Boundary = all vertices so any consumer can derive a
    # sub-boundary T_rooted.
    bipartite_targets = [(2, 2), (2, 3), (3, 3), (2, 4), (3, 4), (4, 4)]
    if not small_only:
        bipartite_targets.append((4, 5))
    if slow:
        bipartite_targets.extend([(5, 5), (5, 6), (6, 6)])
    for a, b in bipartite_targets:
        g = Graph.from_networkx(nx.complete_bipartite_graph(a, b))
        targets.append((f"K_{{{a},{b}}}", g, list(g.nodes)))
    if not small_only:
        try:
            import dwave_networkx as dnx
            z11 = Graph.from_networkx(dnx.zephyr_graph(1, 1))
            targets.append(("Z(1,1)", z11, list(z11.nodes)))
        except ImportError:
            print("[skip] dwave_networkx unavailable; not adding Z(1,1)")

    total_t = 0.0
    for label, g, boundary in targets:
        t0 = time.time()
        result = t_rooted_cached(g, boundary)
        elapsed = time.time() - t0
        total_t += elapsed
        n_parts = len(result)
        print(f"  {label}: {n_parts} partitions, {elapsed:.2f}s")

    # `t_rooted_cached` populates `_T_ROOTED_GRAPHS` with the (3-tuple)
    # cache key → originating Graph mapping. `save_rooted_lookup_default`
    # uses that mapping internally so we don't have to thread it manually
    # (and stay in sync with cache-key format changes).
    n_bin = save_rooted_lookup_default()
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
    )
    bin_path = os.path.join(base_dir, "rooted_lookup_table.bin")
    print(f"\nSaved {n_bin} entries to {bin_path}")
    bin_size = os.path.getsize(bin_path)
    print(f"  BIN: {bin_size:,} bytes")
    print(f"Total computation: {total_t:.2f}s")


if __name__ == "__main__":
    main()
