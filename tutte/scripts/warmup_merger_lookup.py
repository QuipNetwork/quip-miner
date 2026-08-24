"""Warm up the persistent ``merger_lookup_table`` with chord-junction
mergers for D-Wave cell-pair structures.

The merger table backs the unified bivariate chord-junction theorem
(``tutte/roots/chord_junction_closed_form.py``):

    T(G ⊕_{V_k} G; x, y) = (x − 1) · T(G)² + Σ_{∅ ≠ S ⊆ V_k} T(G ∪_{V_S} G)

When the per-subset ``T(G ∪_{V_S} G)`` values are precomputed and cached
on disk, the engine can evaluate any chord-junction in O(2^|V_k|) lookups
+ one base ``T(G)`` evaluation, sidestepping the chord-rule cost on
cell-pair-heavy D-Wave graphs.

This script writes both
``tutte/data/merger_lookup_table.bin`` (fast binary; engine prefers it)
and ``tutte/data/merger_lookup_table.json`` (human-readable, for diff
review). Filenames match the ``*lookup_table*`` ``.gitignore`` pattern;
generated artifacts stay out of source control.

The script is **idempotent**: entries already present (matching
``(base_canonical_key, V_T)``) are skipped on re-runs. Pass
``--force`` to recompute existing entries.

Usage:
    python -m tutte.scripts.warmup_merger_lookup
    python -m tutte.scripts.warmup_merger_lookup --family chimera --verbose
    python -m tutte.scripts.warmup_merger_lookup --family all --force
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from typing import Callable, List, Sequence, Tuple

import networkx as nx

from tutte.graph import Graph
from tutte.lookup.core import load_default_table
from tutte.lookup.merger import (
    MergerEntry, MergerTable,
    load_default_merger_table, save_default_merger_table,
)
from tutte.roots.chord_junction_closed_form import build_symmetric_merger
from tutte.synthesis.engine import SynthesisEngine


# ---------------------------------------------------------------------------
# Family registry — each family contributes a list of (base, V_T orbit reps)
# ---------------------------------------------------------------------------


CellSpec = Tuple[str, Graph, List[Tuple[int, ...]]]
"""(human_name, base_graph, list_of_V_T_orbit_representatives)."""

FamilyBuilder = Callable[[], List[CellSpec]]


def _all_nonempty_subsets(vertices: Sequence[int]) -> List[Tuple[int, ...]]:
    """All non-empty sorted subsets of ``vertices`` (size 1..|vertices|).

    The unified theorem sums over every non-empty subset; we enumerate
    them all so the cache key matches whatever ``V_T`` tuple the engine
    passes to ``MergerTable.lookup_by_source``. Subsets that are orbit-
    equivalent under the cell's automorphism group store the same
    polynomial under different keys — small redundancy in exchange for
    O(1) lookups without runtime aut canonicalization.
    """
    out: List[Tuple[int, ...]] = []
    sorted_v = sorted(vertices)
    for r in range(1, len(sorted_v) + 1):
        for combo in combinations(sorted_v, r):
            out.append(combo)
    return out


def _k44_cells() -> List[CellSpec]:
    """The K_{4,4} cell template, with V_T enumerated over ALL non-empty
    subsets of its 8 vertices.

    Yields 2^8 − 1 = 255 entries. Under ``Aut(K_{4,4})`` (acting as
    ``S_4 × S_4 ⋊ Z_2``) these fall into a small number of orbits, but
    the lookup cache keys on the exact ``V_T`` tuple so we materialize
    every subset. The underlying polynomial values repeat across orbits;
    disk overhead is negligible (~150 KB binary, ~750 KB JSON) and
    runtime lookups stay O(1) without aut canonicalization.

    Covers (at minimum):
      - Chimera matching on side A — bipartition {0,1,2,3} (cached in
        Step 4 already).
      - Chimera matching on side B — bipartition {4,5,6,7}.
      - Mixed-bipartition Chimera anchors (the fixture from
        ``test_engine_k44_cycle_uses_kmatching_formula``, which the
        Step 5 fast path correctly bailed on for lack of a cached
        entry — fills now).
      - Pegasus K_{4,4} sub-cell chord junctions (the engine identifies
        K_{4,4} sub-cells via the rainbow table regardless of which
        20-vertex super-cell they sit inside; chord patterns differ
        from Chimera but the V_T still lives in K_{4,4}'s 8-vertex
        ground set).
    """
    K44 = Graph.from_networkx(nx.complete_bipartite_graph(4, 4))
    V_k = list(K44.nodes)  # {0, 1, …, 7}
    return [("K_{4,4}", K44, _all_nonempty_subsets(V_k))]


def _chimera_cells() -> List[CellSpec]:
    """Chimera dispatches to the shared K_{4,4} cell template."""
    return _k44_cells()


def _pegasus_cells() -> List[CellSpec]:
    """Pegasus also uses K_{4,4} sub-cells (cell-detection routes through
    the rainbow table's K_{4,4} minor entry regardless of which
    Pegasus super-cell the K_{4,4} sits inside).

    Per the ``probe_pegasus_cell_topology.py`` empirical probe, Pegasus
    Pm(m) decomposes into 20-vertex super-cells with ~96-edge junctions,
    but the synthesis engine identifies K_{4,4} sub-cells within those
    super-cells and dispatches chord-rule there. So the K_{4,4} entries
    cover Pegasus chord junctions transparently.

    Returns the same cell spec as Chimera so re-running the warmup with
    ``--family pegasus`` re-tags entries (idempotent — same
    ``(base_canonical_key, V_T)`` keys).
    """
    return _k44_cells()


def _zephyr_cells() -> List[CellSpec]:
    """Zephyr cells: Z(1,1) atomic cell.

    Z(1,1) has 12 boundary vertices; the unified chord-junction theorem
    on Z(1, 2) and similar uses V_T subsets of ALL sizes 1..12 (per
    ``project_z1t_chain_framework_infeasible.md`` — Z(1, 2) decomposes
    as 2 Z(1, 1) cells with a 32-edge junction over all 12 anchors).

    Generates all ``2^12 - 1 = 4095`` non-empty V_T subsets. Aut(Z(1,1))
    has order 8, so the natural orbit count is ≈ 512 — the warmup loop
    detects duplicate merger canonical keys and reuses the cached
    polynomial, so the actual cold compute is ≈ 1.5 minutes
    (vs ~11 minutes without dedup).
    """
    Z11 = _z11_graph()
    if Z11 is None:
        return []
    V = sorted(Z11.nodes)
    v_t_orbits: List[Tuple[int, ...]] = []
    for r in range(1, len(V) + 1):
        for combo in combinations(V, r):
            v_t_orbits.append(combo)
    return [("Z(1,1)", Z11, v_t_orbits)]


def _z11_graph():
    """Build ``Z(1,1)`` (zephyr_graph(m=1, t=1)) via ``dwave_networkx``.

    Returns ``None`` if ``dwave_networkx`` is unavailable so the warmup
    can still proceed for other families.

    Note the ``t=1`` parameter: ``dwave_networkx.zephyr_graph(1)`` defaults
    to ``t=4`` (48 vertices). We use ``t=1`` (12 vertices) to match the
    project's Z(m, t) convention.
    """
    try:
        import dwave_networkx as dnx
    except ImportError:
        print("[zephyr] dwave_networkx not installed; skipping Z(1,1) warmup")
        return None
    nx_graph = dnx.zephyr_graph(1, t=1)
    mapping = {old: new for new, old in enumerate(sorted(nx_graph.nodes))}
    nx_graph = nx.relabel_nodes(nx_graph, mapping)
    return Graph.from_networkx(nx_graph)


FAMILIES: dict[str, FamilyBuilder] = {
    "chimera": _chimera_cells,
    "pegasus": _pegasus_cells,
    "zephyr":  _zephyr_cells,
}


# ---------------------------------------------------------------------------
# Warmup driver
# ---------------------------------------------------------------------------


def _resolve_families(arg: str) -> List[str]:
    if arg == "all":
        return list(FAMILIES.keys())
    if arg not in FAMILIES:
        raise SystemExit(f"unknown family {arg!r} (valid: all, {', '.join(FAMILIES)})")
    return [arg]


def _warmup_one(
    engine: SynthesisEngine,
    table: MergerTable,
    family: str,
    spec: CellSpec,
    force: bool,
    verbose: bool,
) -> Tuple[int, int, float]:
    """Compute mergers for one cell spec; return (added, skipped, seconds)."""
    name, base, v_t_orbits = spec
    base_key = base.canonical_key()
    added = 0
    skipped = 0
    start_family = time.time()
    for v_t in v_t_orbits:
        v_t = tuple(sorted(v_t))
        if not force and table.lookup_by_source(base_key, v_t) is not None:
            skipped += 1
            if verbose:
                print(f"  [skip] {family}/{name} V_T={v_t} (already cached)")
            continue
        merger = build_symmetric_merger(base, v_t)
        try:
            merger_key = merger.canonical_key()
        except Exception:
            merger_key = None
        # Aut-equivalent V_T subsets produce mergers with the same
        # canonical_key. Reuse the cached polynomial instead of resynth.
        # For Z(1,1) (Aut order 8) this saves ~7× compute.
        cached = (
            table.lookup_by_merger(merger_key) if merger_key is not None else None
        )
        if cached is not None:
            polynomial = cached.polynomial
            elapsed = 0.0
        else:
            t0 = time.time()
            polynomial = engine._synthesize_multigraph(merger)
            elapsed = time.time() - t0
        entry = MergerEntry(
            base_canonical_key=base_key,
            v_t=v_t,
            polynomial=polynomial,
            merger_canonical_key=merger_key,
            base_name=name,
            family_tag=family,
            base_node_count=base.node_count(),
            base_edge_count=base.edge_count(),
            merger_node_count=merger.node_count(),
            merger_edge_count=merger.edge_count(),
        )
        table.add_entry(entry)
        added += 1
        if verbose:
            print(f"  [add ] {family}/{name} V_T={v_t}: |V|={merger.node_count()}, "
                  f"|E|={merger.edge_count()}, terms={polynomial.num_terms()}, "
                  f"{elapsed:.2f}s")
    return added, skipped, time.time() - start_family


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Populate the persistent merger lookup table for chord junctions.",
    )
    parser.add_argument(
        "--family",
        default="all",
        help="Which family to warm up: 'chimera', 'pegasus', 'zephyr', or 'all'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute entries already present in the table.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per merger.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute mergers but do not save the table to disk.",
    )
    args = parser.parse_args(argv)

    families = _resolve_families(args.family)

    print(f"[warmup_merger_lookup] families={families}, force={args.force}")
    engine = SynthesisEngine(table=load_default_table(), verbose=False)
    engine.skip_target_lookup = True

    table = load_default_merger_table()
    initial_size = len(table)
    print(f"[warmup_merger_lookup] loaded {initial_size} existing entries")

    total_added = 0
    total_skipped = 0
    start = time.time()
    for family in families:
        specs = FAMILIES[family]()
        if not specs:
            print(f"[{family}] no cell specs yet (deferred to a later step)")
            continue
        for spec in specs:
            added, skipped, elapsed = _warmup_one(
                engine, table, family, spec, args.force, args.verbose,
            )
            total_added += added
            total_skipped += skipped
            print(f"[{family}] {spec[0]}: +{added} added, {skipped} skipped, "
                  f"{elapsed:.2f}s")

    print(f"[warmup_merger_lookup] +{total_added} added, {total_skipped} skipped "
          f"in {time.time() - start:.2f}s; table size {initial_size} → "
          f"{len(table)}")

    if args.dry_run:
        print("[warmup_merger_lookup] --dry-run: skipping disk write")
        return 0

    if total_added == 0:
        print("[warmup_merger_lookup] no new entries; not rewriting disk files")
        return 0

    save_default_merger_table(table)
    print(f"[warmup_merger_lookup] saved {len(table)} entries to "
          f"tutte/data/merger_lookup_table.{{bin,json}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
