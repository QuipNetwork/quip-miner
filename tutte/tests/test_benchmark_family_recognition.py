"""Benchmark for family recognition integrated into the engine pipeline.

Benchmarks:
  1. test_benchmark_family_pipeline — all 13 families, random sampling,
     100–200 edges, 5 min timeout
  2. test_benchmark_family — single family selected via FAMILY env var,
     runs ALL graphs in the edge range (no sampling)

Usage:
    make benchmark-family-pipeline                          # all families
    FAMILY=Wheel make benchmark-family                      # single family
    FAMILY=Wheel MIN_E=50 MAX_E=150 make benchmark-family   # custom range
"""

import os
import random
import signal
import time

import pytest

import networkx as nx
import pytest
from tutte.graph import Graph
from tutte.logs import LogLevel, get_log, reset_log
from tutte.lookup.core import load_default_table
from tutte.synthesis.engine import SynthesisEngine
from tutte.validation import (_exact_num_spanning_trees,
                              _exact_spanning_tree_count)


class _Timeout(BaseException):
    pass


def _with_timeout(fn, timeout_s):
    """Run fn() with SIGALRM timeout. Returns (elapsed_ms, result) or raises _Timeout."""
    def _handler(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        t0 = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        return elapsed, result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _fmt_time(ms):
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


# ===========================================================================
# Graph builders (for families not in networkx)
# ===========================================================================

def _build_fan(k: int) -> nx.Graph:
    """Fan F_k: apex 0 connected to path 1..k."""
    G = nx.Graph()
    for i in range(1, k + 1):
        G.add_edge(0, i)
    for i in range(1, k):
        G.add_edge(i, i + 1)
    return G


def _build_gear(k: int) -> nx.Graph:
    """Gear graph: hub + k rim vertices + k subdivision vertices."""
    G = nx.Graph()
    hub = 0
    rim = list(range(1, k + 1))
    sub = list(range(k + 1, 2 * k + 1))
    for i in range(k):
        G.add_edge(hub, rim[i])
        G.add_edge(rim[i], sub[i])
        G.add_edge(sub[i], rim[(i + 1) % k])
    return G


def _build_helm(k: int) -> nx.Graph:
    """Helm: wheel W_k + pendant at each rim vertex."""
    G = nx.wheel_graph(k + 1)
    for i in range(1, k + 1):
        G.add_edge(i, k + i)
    return G


def _build_book(k: int) -> nx.Graph:
    """Book: k triangles sharing edge (0, 1)."""
    G = nx.Graph()
    G.add_edge(0, 1)
    for i in range(k):
        v = i + 2
        G.add_edge(0, v)
        G.add_edge(1, v)
    return G


def _build_pan(cycle_size: int) -> nx.Graph:
    """Pan: cycle of size cycle_size + one pendant edge."""
    G = nx.cycle_graph(cycle_size)
    G.add_edge(0, cycle_size)
    return G


def _build_sunlet(k: int) -> nx.Graph:
    """Sunlet: C_k with pendant at each vertex."""
    G = nx.cycle_graph(k)
    for i in range(k):
        G.add_edge(i, k + i)
    return G


def _build_mobius(k: int) -> nx.Graph:
    """Mobius ladder: 2k-cycle with k rungs connecting v_i to v_{i+k}."""
    G = nx.cycle_graph(2 * k)
    for i in range(k):
        G.add_edge(i, i + k)
    return G


# ===========================================================================
# Family generators — parameterized by (min_e, max_e)
# ===========================================================================

def _path_graphs(min_e, max_e):
    for n in range(max(3, min_e + 1), max_e + 2):
        e = n - 1
        if min_e <= e <= max_e:
            yield f"P_{n}", nx.path_graph(n)


def _cycle_graphs(min_e, max_e):
    for n in range(max(3, min_e), max_e + 1):
        yield f"C_{n}", nx.cycle_graph(n)


def _wheel_graphs(min_e, max_e):
    for k in range(3, 200):
        e = 2 * k
        if e < min_e:
            continue
        if e > max_e:
            break
        yield f"W_{k}", nx.wheel_graph(k + 1)


def _fan_graphs(min_e, max_e):
    for k in range(3, 200):
        e = 2 * k - 1
        if e < min_e:
            continue
        if e > max_e:
            break
        yield f"F_{k}", _build_fan(k)


def _pan_graphs(min_e, max_e):
    for cycle_size in range(3, 200):
        e = cycle_size + 1
        if e < min_e:
            continue
        if e > max_e:
            break
        yield f"Pan_{cycle_size}", _build_pan(cycle_size)


def _sunlet_graphs(min_e, max_e):
    for k in range(3, 200):
        e = 2 * k
        if e < min_e:
            continue
        if e > max_e:
            break
        yield f"Sun_{k}", _build_sunlet(k)


def _book_graphs(min_e, max_e):
    for k in range(1, 200):
        e = 2 * k + 1
        if e < min_e:
            continue
        if e > max_e:
            break
        yield f"B_{k}", _build_book(k)


def _ladder_graphs(min_e, max_e):
    for k in range(2, 200):
        e = 3 * k - 2
        if e < min_e:
            continue
        if e > max_e:
            break
        yield f"L_{k}", nx.ladder_graph(k)


def _gear_graphs(min_e, max_e):
    for k in range(3, 200):
        e = 3 * k
        if e < min_e:
            continue
        if e > max_e:
            break
        yield f"G_{k}", _build_gear(k)


def _helm_graphs(min_e, max_e):
    for k in range(3, 200):
        e = 3 * k
        if e < min_e:
            continue
        if e > max_e:
            break
        yield f"H_{k}", _build_helm(k)


def _prism_graphs(min_e, max_e):
    for k in range(3, 200):
        e = 3 * k
        if e < min_e:
            continue
        if e > max_e:
            break
        yield f"CL_{k}", nx.circular_ladder_graph(k)


def _mobius_graphs(min_e, max_e):
    for k in range(3, 200):
        e = 3 * k
        if e < min_e:
            continue
        if e > max_e:
            break
        yield f"M_{k}", _build_mobius(k)


def _grid_graphs(min_e, max_e):
    # TODO: grid m>=3 not yet implemented in grid_recurrence() — re-enable when
    # transfer matrix approach is added
    for r in range(2, 3):
        for c in range(r, 200):
            e = (r - 1) * c + r * (c - 1)
            if e < min_e:
                continue
            if e > max_e:
                break
            G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(r, c))
            yield f"Grid_{r}x{c}", G


FAMILY_GENERATORS = [
    ("Path", _path_graphs),
    ("Cycle", _cycle_graphs),
    ("Wheel", _wheel_graphs),
    ("Fan", _fan_graphs),
    ("Pan", _pan_graphs),
    ("Sunlet", _sunlet_graphs),
    ("Book", _book_graphs),
    ("Ladder", _ladder_graphs),
    ("Gear", _gear_graphs),
    ("Helm", _helm_graphs),
    ("Prism", _prism_graphs),
    ("Mobius", _mobius_graphs),
    ("Grid", _grid_graphs),
]

# Lookup by case-insensitive name for FAMILY env var selection
_FAMILY_BY_NAME = {name.lower(): (name, gen) for name, gen in FAMILY_GENERATORS}


# ===========================================================================
# Shared pipeline runner
# ===========================================================================

def _run_pipeline(families, min_e, max_e, timeout_per_graph, samples_per_family=None,
                  seed=None, label="BENCHMARK"):
    """Run engine pipeline benchmark on the given families.

    Args:
        families: list of (name, generator_fn) pairs
        min_e, max_e: edge range passed to generators
        timeout_per_graph: SIGALRM timeout in seconds
        samples_per_family: if set, randomly sample this many per family
        seed: RNG seed (required if samples_per_family is set)
        label: header label for output
    """
    rng = random.Random(seed) if samples_per_family else None

    table = load_default_table()
    engine = SynthesisEngine(table)

    total_graphs = 0
    total_fast = 0
    total_slow = 0
    total_fail = 0
    total_timeout = 0
    family_summaries = []
    problems = []

    sample_info = f", {samples_per_family}/family, seed={seed}" if samples_per_family else ""
    print(f"\n{'=' * 100}")
    print(f"  {label} ({min_e}–{max_e} edges{sample_info})")
    print(f"{'=' * 100}")

    for family_name, generator in families:
        all_graphs = list(generator(min_e, max_e))
        if not all_graphs:
            continue

        # Sample or use all
        if samples_per_family and rng and len(all_graphs) > samples_per_family:
            sampled = rng.sample(all_graphs, samples_per_family)
            sampled.sort(key=lambda g: g[1].number_of_edges())
        else:
            sampled = all_graphs

        count_str = (f"{len(sampled)}/{len(all_graphs)} sampled"
                     if samples_per_family else f"{len(sampled)} graphs")
        print(f"\n  --- {family_name} ({count_str}) ---")
        print(f"  {'Name':<14s} {'N':>4} {'E':>4} "
              f"{'Engine':>14} {'Method':>20} {'Kirchhoff':>14} {'T(2,2)':>10}")

        fam_fast = 0
        fam_fail = 0

        for name, G_nx in sampled:
            n = G_nx.number_of_nodes()
            e = G_nx.number_of_edges()
            total_graphs += 1

            graph = Graph.from_networkx(G_nx)

            # Fresh log per graph
            reset_log()
            log = get_log()
            log.min_level = LogLevel.DEBUG

            # Run through full engine pipeline with timeout
            try:
                engine_ms, result = _with_timeout(
                    lambda: engine.synthesize(graph), timeout_per_graph
                )
            except _Timeout:
                total_timeout += 1
                problems.append((name, n, e, "TIMEOUT",
                                 f"exceeded {timeout_per_graph}s"))
                print(f"  {name:<14s} {n:>4} {e:>4} "
                      f"{'TIMEOUT':>14} {'-':>20} {'-':>14} {'-':>10}")
                continue

            engine_str = _fmt_time(engine_ms)
            method = result.method
            poly = result.polynomial

            # --- Validation: Kirchhoff T(1,1) ---
            kirchhoff = _exact_spanning_tree_count(graph)
            poly_trees = _exact_num_spanning_trees(poly)
            kirch_ok = poly_trees == kirchhoff

            # --- Validation: T(2,2) = 2^|E| ---
            t22 = poly.evaluate(2, 2)
            t22_ok = t22 == 2 ** e

            if kirch_ok and t22_ok:
                kirch_str = f"{kirchhoff}"
                t22_str = "OK"
            else:
                total_fail += 1
                fam_fail += 1
                kirch_str = f"FAIL({kirchhoff})" if not kirch_ok else f"{kirchhoff}"
                t22_str = "FAIL" if not t22_ok else "OK"
                problems.append((name, n, e, method,
                                 f"T(1,1)={poly_trees} vs kirchhoff={kirchhoff}, "
                                 f"T(2,2)={'FAIL' if not t22_ok else 'OK'}"))

            # --- Track fast vs slow path ---
            if method == "family_recognition":
                total_fast += 1
                fam_fast += 1
            else:
                total_slow += 1
                problems.append((name, n, e, method,
                                 f"fell through to {method} ({engine_str})"))

            print(f"  {name:<14s} {n:>4} {e:>4} "
                  f"{engine_str:>14} {method:>20} {kirch_str:>14} {t22_str:>10}")

        family_summaries.append((family_name, len(sampled), fam_fast, fam_fail))

    # --- Summary ---
    print(f"\n{'=' * 100}")
    print(f"  SUMMARY")
    print(f"{'=' * 100}")
    print(f"\n  {'Family':<14s} {'Sampled':>7} {'Fast':>7} {'Fail':>7}")
    print(f"  {'-' * 14} {'-' * 7} {'-' * 7} {'-' * 7}")
    for fam, count, fast, fail in family_summaries:
        print(f"  {fam:<14s} {count:>7} {fast:>7} {fail:>7}")

    print(f"\n  Total:          {total_graphs} graphs")
    print(f"  Fast path:      {total_fast}")
    print(f"  Slow path:      {total_slow}")
    print(f"  Timeout:        {total_timeout}")
    print(f"  Validation fail: {total_fail}")

    if problems:
        print(f"\n  PROBLEMS:")
        for name, n, e, method, reason in problems:
            print(f"    {name} ({n}n {e}e) [{method}]: {reason}")

    assert total_fail == 0, f"{total_fail} validation failures"
    assert total_timeout == 0, f"{total_timeout} graphs timed out"


# ===========================================================================
# BENCHMARK 1: All families, random sampling (100–200 edges)
# ===========================================================================

@pytest.mark.slow
def test_benchmark_family_pipeline():
    """Benchmark engine pipeline with family recognition integrated.

    Randomly samples 5 graphs per family (100-200 edges), runs them
    through engine.synthesize(), and verifies correctness + fast path.
    Seed is fixed for reproducibility.

    Marked @pytest.mark.slow because grids with m≥3 currently fall through
    to k-sum / hierarchical / CEJ (see `grid_recurrence` gap in
    `tutte/family_recognition/formulas.py`); each grid graph in the 100–200
    edge range can take many seconds to minutes.
    """
    _run_pipeline(
        families=FAMILY_GENERATORS,
        min_e=100, max_e=200,
        timeout_per_graph=300,
        samples_per_family=5,
        seed=42,
        label="ENGINE PIPELINE — RANDOM FAMILY BENCHMARK",
    )


# ===========================================================================
# BENCHMARK 2: Single family selected via FAMILY env var
# ===========================================================================

@pytest.mark.slow
def test_benchmark_family():
    """Benchmark a single family selected via the FAMILY env var.

    Runs ALL graphs in the edge range through the engine pipeline (no
    sampling). Edge range defaults to 100–200 but can be overridden
    with MIN_E and MAX_E env vars.

    Available families: Path, Cycle, Wheel, Fan, Pan, Sunlet, Book,
    Ladder, Gear, Helm, Prism, Mobius, Grid

    Usage:
        FAMILY=Wheel make benchmark-family
        FAMILY=Prism MIN_E=50 MAX_E=300 make benchmark-family
    """
    family_key = os.environ.get("FAMILY", "").strip().lower()
    if not family_key:
        available = ", ".join(name for name, _ in FAMILY_GENERATORS)
        pytest.skip(
            f"FAMILY env var not set. Choose one of: {available}\n"
            f"  Example: FAMILY=Wheel make benchmark-family"
        )

    if family_key not in _FAMILY_BY_NAME:
        available = ", ".join(name for name, _ in FAMILY_GENERATORS)
        raise ValueError(
            f"Unknown family '{os.environ['FAMILY']}'. Choose one of: {available}"
        )

    family_name, generator = _FAMILY_BY_NAME[family_key]
    min_e = int(os.environ.get("MIN_E", 100))
    max_e = int(os.environ.get("MAX_E", 200))

    _run_pipeline(
        families=[(family_name, generator)],
        min_e=min_e, max_e=max_e,
        timeout_per_graph=300,
        label=f"ENGINE PIPELINE — {family_name.upper()} FAMILY",
    )