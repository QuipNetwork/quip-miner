"""Warm up the rainbow table with atom polynomials + selected D-Wave headers.

Two value props:

1. **Cell lookup for D-Wave decompositions**: Cm2,
   Z(1,2) precomputed so Z(1,3), Cm3, Pm3 syntheses get instant cell hits
   via the cost-aware partitioner.

2. **Cograph atom polynomials** (May 2026): K_8..K_15, K_{a,b} for small
   (a, b), and other simple cograph atoms, seeded as reusable table
   cells/minors. Cheap to compute (cotree_dp synthesizes them in <1s
   each) and lets later syntheses recognise them as known minors.

Idempotent: skips targets already in the table.

Usage:
    python -m tutte.scripts.warmup_lookup_table
    python -m tutte.scripts.warmup_lookup_table --target K_10
    python -m tutte.scripts.warmup_lookup_table --timeout 1800
"""

import argparse
import os
import signal
import sys
import time

import dwave_networkx as dnx
import networkx as nx
from tutte.graph import Graph, grid_graph, path_graph, wheel_graph
from tutte.lookup.binary import save_binary_rainbow_table
from tutte.lookup.core import load_default_table
from tutte.synthesis.engine import SynthesisEngine
from tutte.validation import verify_spanning_trees

# --- Family-recognition seed builders ---------------------------------------
# Seeds for the `_LazyBases` loaders in tutte/family_recognition/constants.py.
# `recognize_family()` returns None for wheel/fan/ladder/book/gear/prism/möbius
# graphs unless these seeds are in the rainbow table.

def _build_book(k):
    """Book graph: k triangles sharing edge (0, 1)."""
    G = nx.Graph()
    G.add_edge(0, 1)
    for i in range(k):
        v = i + 2
        G.add_edge(0, v)
        G.add_edge(1, v)
    return G


def _gear(k):
    """Gear graph: hub + k rim vertices + k subdivision vertices."""
    G = nx.Graph()
    for i in range(k):
        G.add_edge(0, i + 1)
        G.add_edge(i + 1, k + 1 + i)
        G.add_edge(k + 1 + i, (i + 1) % k + 1)
    return G


def _prism(k):
    """Prism graph C_k × K_2 (circular ladder)."""
    return nx.circular_ladder_graph(k)


def _mobius(k):
    """Möbius ladder: 2k-cycle with k rungs connecting v_i to v_{i+k}."""
    G = nx.cycle_graph(2 * k)
    for i in range(k):
        G.add_edge(i, i + k)
    return G


TARGETS = [
    # D-Wave headers — cell lookups for larger D-Wave decompositions.
    # Cm1 is K_{4,4} structurally; included as an alias so `find_by_name("Cm1")`
    # works once added. Pm1 (and similarly Z(0,*)) are empty graphs in dwave_networkx
    # and intentionally omitted.
    ("Cm1", lambda: dnx.chimera_graph(1)),
    ("Z1_1", lambda: dnx.zephyr_graph(1, 1)),
    ("Z1_2", lambda: dnx.zephyr_graph(1, 2)),
    ("Cm2", lambda: dnx.chimera_graph(2)),
    ("Cm3", lambda: dnx.chimera_graph(3)),       # 72n 192e — runs only with --target Cm3 + a large --timeout
    ("Pm2", lambda: dnx.pegasus_graph(2)),

    # Cm(m, n) rectangular variants — share intermediate canonical_keys
    # with square Cm_m via the engine's recursive synthesize() cache.
    # Warming populates lookup so Cm(2,2)/Cm(3,3) syntheses can hit
    # cached substructures from Cm(1,2)/Cm(2,3) syntheses.
    ("Cm1_2", lambda: dnx.chimera_graph(1, 2)),  # 16n 36e
    ("Cm1_3", lambda: dnx.chimera_graph(1, 3)),  # 24n 56e
    ("Cm1_4", lambda: dnx.chimera_graph(1, 4)),  # 32n 76e
    ("Cm2_3", lambda: dnx.chimera_graph(2, 3)),  # 48n 124e — slow cold; consider --target gating

    # Cograph atom polynomials (May 2026) — fast via cotree_dp (<1s each),
    # seeded as reusable table cells/minors for later syntheses.
    ("K_8",  lambda: nx.complete_graph(8)),
    ("K_9",  lambda: nx.complete_graph(9)),
    ("K_10", lambda: nx.complete_graph(10)),
    ("K_11", lambda: nx.complete_graph(11)),
    ("K_12", lambda: nx.complete_graph(12)),
    ("K_13", lambda: nx.complete_graph(13)),
    ("K_14", lambda: nx.complete_graph(14)),
    ("K_15", lambda: nx.complete_graph(15)),
    # Bipartite cograph atoms — common as D-Wave Chimera cells.
    ("K_2_3", lambda: nx.complete_bipartite_graph(2, 3)),
    ("K_2_4", lambda: nx.complete_bipartite_graph(2, 4)),
    ("K_3_3", lambda: nx.complete_bipartite_graph(3, 3)),
    ("K_3_4", lambda: nx.complete_bipartite_graph(3, 4)),
    ("K_4_4", lambda: nx.complete_bipartite_graph(4, 4)),  # = Cm1
    ("K_4_5", lambda: nx.complete_bipartite_graph(4, 5)),
    ("K_5_5", lambda: nx.complete_bipartite_graph(5, 5)),
    ("K_5_6", lambda: nx.complete_bipartite_graph(5, 6)),
    ("K_6_6", lambda: nx.complete_bipartite_graph(6, 6)),

    # Family-recognition recurrence seeds — feed _LazyBases loaders in
    # tutte/family_recognition/constants.py. Without these, recognize_family
    # returns None for wheel/fan/ladder/book/gear/prism/möbius graphs.
    ("K_2", lambda: path_graph(2)),               # F_1 = single edge
    ("C_4", lambda: nx.cycle_graph(4)),           # L_2
    ("W_4", lambda: wheel_graph(4)),
    ("B_2", lambda: _build_book(2)),              # Book k=2
    ("Gear_3", lambda: _gear(3)),
    ("Grid_2x3", lambda: grid_graph(2, 3)),       # L_3
    ("Gear_4", lambda: _gear(4)),
    ("Gear_5", lambda: _gear(5)),
    # Prism seeds CL_3..CL_8
    ("Prism_3", lambda: _prism(3)),
    ("Prism_4", lambda: _prism(4)),
    ("Prism_5", lambda: _prism(5)),
    ("Prism_6", lambda: _prism(6)),
    ("Prism_7", lambda: _prism(7)),
    ("Prism_8", lambda: _prism(8)),
    # Möbius seeds M_3..M_8
    ("Mobius_3", lambda: _mobius(3)),
    ("Mobius_4", lambda: _mobius(4)),
    ("Mobius_5", lambda: _mobius(5)),
    ("Mobius_6", lambda: _mobius(6)),
    ("Mobius_7", lambda: _mobius(7)),
    ("Mobius_8", lambda: _mobius(8)),

    # Still NOT default (intractable / too slow):
    #   ("Z2_1", lambda: dnx.zephyr_graph(2, 1)), # 40n 114e — saw 3.5 GB / >5 min before kill
    #   ("Z1_3", lambda: dnx.zephyr_graph(1, 3)), # 36n 162e
]


def _data_paths() -> tuple[str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.normpath(os.path.join(here, "..", "data"))
    return (
        os.path.join(base, "lookup_table.json"),
        os.path.join(base, "lookup_table.bin"),
    )


def _existing_names(table) -> set[str]:
    return {e.name for e in table.entries.values()}


class _Timeout(BaseException):
    """Raised when a per-target synthesis exceeds its time budget.

    Inherits from BaseException so internal `except Exception` handlers in
    networkx / numpy / sympy don't swallow it.
    """


def _alarm_handler(signum, frame):
    raise _Timeout()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target", default="all",
        choices=["all"] + [name for name, _ in TARGETS],
        help="Which target(s) to add to the table.",
    )
    p.add_argument(
        "--timeout", type=int, default=1800,
        help="Per-target wall-clock budget in seconds (default 1800 = 30 min).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Synthesize but don't write the updated table to disk.",
    )
    args = p.parse_args()

    table = load_default_table()
    json_path, bin_path = _data_paths()
    print(f"Loaded {len(table.entries)} entries from {json_path}", file=sys.stderr)

    targets = TARGETS if args.target == "all" else [
        (n, b) for n, b in TARGETS if n == args.target
    ]

    existing = _existing_names(table)
    added = 0
    for name, builder in targets:
        if name in existing:
            print(f"  {name}: already in table — skip", file=sys.stderr)
            continue

        # Check canonical_key collision BEFORE synthesis (e.g., K_4_4 has
        # the same canonical_key as the existing Cm1 entry — adding it
        # would overwrite Cm1, breaking gates that key on cell name).
        try:
            built = builder()
        except Exception as e:
            print(f"  {name}: builder failed: {e}", file=sys.stderr)
            continue
        g_check = built if isinstance(built, Graph) else Graph.from_networkx(built)
        check_key = g_check.canonical_key()
        if check_key in table.entries:
            existing_entry = table.entries[check_key]
            if name not in table.name_index:
                # Alias only — preserves the existing entry's primary name
                # while making lookup_by_name(name) succeed (e.g. Mobius_3
                # aliased to K_3_3 so family_recognition can find the seed).
                table.name_index[name] = check_key
                added += 1
                print(
                    f"  {name}: aliased to existing entry '{existing_entry.name}' "
                    f"(same canonical_key)",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  {name}: canonical_key collision with existing entry "
                    f"'{existing_entry.name}' and name already aliased — skip",
                    file=sys.stderr,
                )
            continue

        print(f"  {name}: synthesizing (budget {args.timeout}s)...", file=sys.stderr,
              flush=True)
        g = g_check
        engine = SynthesisEngine(table=table, verbose=False)
        engine.skip_target_lookup = True

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(args.timeout)
        t0 = time.perf_counter()
        try:
            result = engine.synthesize(g)
        except _Timeout:
            wall = time.perf_counter() - t0
            print(
                f"  {name}: TIMEOUT after {wall:.0f}s (budget {args.timeout}s) — skip",
                file=sys.stderr,
            )
            continue
        except Exception as e:
            print(f"  {name}: synthesis FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        wall = time.perf_counter() - t0

        if not verify_spanning_trees(g, result.polynomial):
            print(
                f"  {name}: synthesized in {wall:.0f}s ({result.method}) but "
                f"FAILED Kirchhoff verification — NOT adding to table",
                file=sys.stderr,
            )
            continue

        table.add(g, name, result.polynomial, result.minors_used)
        added += 1
        print(
            f"  {name}: added in {wall:.0f}s, method={result.method}, "
            f"ST={int(result.polynomial.num_spanning_trees())}",
            file=sys.stderr,
        )

    if added == 0:
        print("\nNo new entries added.", file=sys.stderr)
        return 0

    if args.dry_run:
        print(
            f"\n--dry-run: would write {len(table.entries)} entries to "
            f"{bin_path}",
            file=sys.stderr,
        )
        return 0

    table.resort()
    bin_size = save_binary_rainbow_table(table, bin_path)
    print(
        f"\nSaved {len(table.entries)} entries — "
        f"{bin_path} ({bin_size} bytes).",
        file=sys.stderr,
    )

    # Also populate the multigraph cache with chord-peel intermediates.
    # `warmup_chord_peel_cache` runs Z(1,2)-class targets through the
    # merged dispatcher with `promote_cache_on_finish=True`, so every
    # contraction multigraph produced by chord-rule gets merged into
    # `tutte/data/multigraph_lookup_table.{bin,json}`. Without this step
    # the heterogeneous chord-peel path on Z(1,2) takes 150s cold; with
    # cached intermediates, both legacy and het paths drop to ~1.1s.
    if not args.dry_run:
        try:
            from .warmup_chord_peel_cache import main as _chord_peel_main
            print("\n=== Populating chord-peel multigraph cache ===",
                  file=sys.stderr)
            _orig_argv = sys.argv
            sys.argv = ["warmup_chord_peel_cache", "--mode", "both"]
            try:
                _chord_peel_main()
            finally:
                sys.argv = _orig_argv
        except Exception as e:
            print(f"  chord-peel cache warmup failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
