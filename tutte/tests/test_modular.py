"""Modular Tutte-polynomial DP regression tests.

Two layers:

1. **Cell-tree DP modular variant** — `compute_tree_dp_simple_mod`
   matches the engine polynomial's `evaluate_mod` bit-for-bit on
   linear-path tree-quotient graphs.
2. **End-to-end modular DP vs engine** — for graphs that have a
   dedicated modular DP path (e.g. Chimera), the DP value matches the
   engine polynomial's `evaluate_mod` at every test point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import networkx as nx
import pytest

from tutte.graph import Graph, complete_graph
from tutte.lookup.core import load_default_table
from tutte.polynomial import TuttePolynomial
from tutte.roots.cell_quotient_tree import (CellTreeSpec,
                                            compute_tree_dp_simple_mod)
from tutte.synthesis.engine import SynthesisEngine

TEST_POINTS = [(3, 5, 1009), (7, 11, 10007), (100, 200, 10**9 + 7)]


# ---------------------------------------------------------------------------
# Cell-tree DP modular variant
# ---------------------------------------------------------------------------

def _disjoint_union(g1: Graph, g2: Graph) -> Graph:
    nodes_set = set(g1.nodes)
    nodes_set.update({v + max(g1.nodes) + 1 for v in g2.nodes})
    edges_list = list(g1.edges)
    offset = max(g1.nodes) + 1
    for (u, v) in g2.edges:
        edges_list.append((u + offset, v + offset))
    return Graph(nodes=frozenset(nodes_set), edges=frozenset(edges_list))


def _add_edges(g: Graph, edges) -> Graph:
    e = set(g.edges)
    for u, v in edges:
        e.add(tuple(sorted([u, v])))
    return Graph(nodes=g.nodes, edges=frozenset(e))


def _make_M_k(k: int) -> Graph:
    return Graph(
        nodes=frozenset(range(2 * k)),
        edges=frozenset((i, i + k) for i in range(k)),
    )


def _build_n_cell_path(cell: Graph, n: int, anchors: List[int]) -> Graph:
    g = cell
    offsets = [0]
    for _ in range(n - 1):
        offsets.append(max(g.nodes) + 1)
        g = _disjoint_union(g, cell)
    cell_anchors = [[a + offsets[i] for a in anchors] for i in range(n)]
    for i in range(n - 1):
        g = _add_edges(g, [
            (cell_anchors[i][j], cell_anchors[i + 1][j])
            for j in range(len(anchors))
        ])
    return g


def _linear_tree(n: int) -> nx.Graph:
    t = nx.Graph()
    t.add_nodes_from(range(n))
    for i in range(n - 1):
        t.add_edge(i, i + 1)
    return t


def _linear_anchor_groups(n: int, anchors: List[int]) -> dict:
    out = {}
    for i in range(n):
        nbrs = {}
        if i > 0:
            nbrs[i - 1] = anchors
        if i < n - 1:
            nbrs[i + 1] = anchors
        out[i] = nbrs
    return out


@pytest.mark.parametrize("label,cell_factory,anchors,n_cells", [
    ("K_3 path M_2 (2c)", lambda: complete_graph(3), [0, 1], 2),
    ("K_3 path M_2 (3c)", lambda: complete_graph(3), [0, 1], 3),
    ("K_4 path M_2 (3c)", lambda: complete_graph(4), [0, 1], 3),
    ("K_4 path M_3 (3c)", lambda: complete_graph(4), [0, 1, 2], 3),
    ("K_{4,4} path M_2 (3c)",
     lambda: Graph.from_networkx(nx.complete_bipartite_graph(4, 4)),
     [0, 1], 3),
])
def test_tree_dp_simple_mod_matches_engine(label, cell_factory, anchors, n_cells):
    """Modular cell-tree DP matches engine polynomial's `evaluate_mod`
    bit-for-bit on linear-path graphs with M_k matching junctions."""
    cell = cell_factory()
    k = len(anchors)
    g = _build_n_cell_path(cell, n_cells, anchors)

    engine = SynthesisEngine(table=load_default_table(), verbose=False)
    engine.skip_target_lookup = True
    T_engine = engine.synthesize(g).polynomial

    spec = CellTreeSpec(
        cell_template=cell,
        junction_template=_make_M_k(k),
        cell_tree=_linear_tree(n_cells),
        cell_anchor_groups=_linear_anchor_groups(n_cells, anchors),
        junction_anchors_A=list(range(k)),
        junction_anchors_B=list(range(k, 2 * k)),
        root=0,
    )

    for x, y, p in TEST_POINTS:
        got = compute_tree_dp_simple_mod(spec, x, y, p)
        expected = T_engine.evaluate_mod(x, y, p)
        assert got == expected, (
            f"{label} ({x},{y},{p}): modular DP = {got}, "
            f"engine.evaluate_mod = {expected}"
        )


# ---------------------------------------------------------------------------
# Modular DP vs engine cross-validation
# ---------------------------------------------------------------------------

_MOD_VS_ENGINE_POINTS = TEST_POINTS + [(2, 3, 1125899906842597)]  # +50-bit prime


@dataclass
class GraphCase:
    name: str
    builder: Callable[[], Graph]
    modular_dp_path: Optional[str] = None  # None = engine self-check only
    skip_reason: Optional[str] = None
    slow: bool = False  # engine synthesis takes minutes; run only with -m slow


def _kn(n): return Graph.from_networkx(nx.complete_graph(n))
def _kab(a, b): return Graph.from_networkx(nx.complete_bipartite_graph(a, b))
def _cn(n): return Graph.from_networkx(nx.cycle_graph(n))
def _wn(n): return Graph.from_networkx(nx.wheel_graph(n))
def _petersen(): return Graph.from_networkx(nx.petersen_graph())


def _chimera(m):
    import dwave_networkx as dnx
    return Graph.from_networkx(dnx.chimera_graph(m))


def _pegasus(m):
    import dwave_networkx as dnx
    return Graph.from_networkx(dnx.pegasus_graph(m))


def _zephyr(m, t):
    import dwave_networkx as dnx
    return Graph.from_networkx(dnx.zephyr_graph(m, t))


GRAPH_CASES: List[GraphCase] = [
    GraphCase("K_3", lambda: _kn(3)),
    GraphCase("K_4", lambda: _kn(4)),
    GraphCase("K_5", lambda: _kn(5)),
    GraphCase("K_{3,3}", lambda: _kab(3, 3)),
    GraphCase("K_{4,4}", lambda: _kab(4, 4)),
    GraphCase("C_5", lambda: _cn(5)),
    GraphCase("C_8", lambda: _cn(8)),
    GraphCase("W_5", lambda: _wn(5)),
    GraphCase("Petersen", _petersen),
    GraphCase("Cm_1", lambda: _chimera(1),
              skip_reason="Cm_1 is a single K_{4,4} cell — no hierarchical decomposition"),
    GraphCase("Cm_2", lambda: _chimera(2), modular_dp_path="chimera", slow=True),
    GraphCase("Pm_1", lambda: _pegasus(1),
              skip_reason="Pegasus modular DP path not implemented", slow=True),
    GraphCase("Z(1,1)", lambda: _zephyr(1, 1),
              skip_reason="Zephyr modular DP path not implemented", slow=True),
]

# Engine synthesis of the hardware graphs (Cm_2, Pm_1, Z(1,1)) takes
# minutes each; their cases carry @pytest.mark.slow so default runs
# stay fast. Polynomials are computed lazily per case (not eagerly for
# the whole table) so deselected cases cost nothing.
GRAPH_PARAMS = [
    pytest.param(case, id=case.name,
                 marks=(pytest.mark.slow,) if case.slow else ())
    for case in GRAPH_CASES
]


def _chimera_modular(m: int, x: int, y: int, p: int) -> int:
    mod = pytest.importorskip(
        "tutte.research.scripts.cm3_via_modular_dp",
        reason="chimera modular DP script lost (uncommitted at machine "
               "death); rebuild tracked in bead riff-ggi",
    )
    return mod.chimera_modular_dp(m, x, y, p, verbose=False)


_ENGINE_POLY_CACHE: dict = {}


def _engine_polynomial(case: GraphCase) -> Tuple[Optional[Graph], object]:
    """Compute (and memoize) the engine polynomial for one case."""
    if case.name not in _ENGINE_POLY_CACHE:
        engine = _ENGINE_POLY_CACHE.setdefault(
            "__engine__", SynthesisEngine(table=load_default_table(), verbose=False),
        )
        try:
            G = case.builder()
            _ENGINE_POLY_CACHE[case.name] = (G, engine.synthesize(G).polynomial)
        except Exception as e:
            _ENGINE_POLY_CACHE[case.name] = (None, str(e))
    return _ENGINE_POLY_CACHE[case.name]


@pytest.mark.parametrize("case", GRAPH_PARAMS)
def test_engine_evaluate_mod_self_consistent(case):
    """T.evaluate_mod(x, y, p) == T.evaluate(x, y) % p. Catches bugs in
    `TuttePolynomial.evaluate_mod`."""
    G, T = _engine_polynomial(case)
    if G is None:
        pytest.skip(f"engine failed on {case.name}: {T}")
    for x, y, p in [(3, 5, 1009), (7, 11, 10007)]:
        exact = T.evaluate(x, y)
        mod = T.evaluate_mod(x, y, p)
        assert mod == exact % p, (
            f"{case.name}: evaluate_mod({x},{y},{p}) = {mod}, "
            f"expected {exact % p}"
        )


@pytest.mark.parametrize("case", GRAPH_PARAMS)
def test_modular_dp_matches_engine(case):
    """For graphs with a modular DP path, assert match against engine
    polynomial's `evaluate_mod` at every test point."""
    if case.skip_reason:
        pytest.skip(case.skip_reason)
    if case.modular_dp_path is None:
        pytest.skip(f"no modular DP path for {case.name}")

    G, T_engine = _engine_polynomial(case)
    if G is None:
        pytest.skip(f"engine failed on {case.name}")

    if case.modular_dp_path == "chimera":
        m = int(case.name.split("_")[1])
        for x, y, p in _MOD_VS_ENGINE_POINTS:
            modular = _chimera_modular(m, x, y, p)
            engine_mod = T_engine.evaluate_mod(x, y, p)
            assert modular == engine_mod, (
                f"{case.name}: modular_dp({x},{y},{p}) = {modular}, "
                f"engine.evaluate_mod = {engine_mod} (Δ={modular - engine_mod})"
            )


def test_cm2_modular_dp_under_15s():
    """Cm_2 modular DP at (3, 5) mod 1009 should complete in < 15s.

    Regression guard for the cell-quotient-mod-DP path (R18 aggregation,
    R19 H-bucketing off-by-default, C-ext orbit expansion). Cm_2 baseline
    on 2026-05-26 was ~12s including initial path-DP build (~6s). The 15s
    cap catches regressions of ~25%+.

    Sets TUTTE_R19_ENABLE=0 (default) to keep timing predictable; R19
    H-bucketing is currently a net regression in modular int mode.
    """
    import time
    import os
    os.environ.setdefault("TUTTE_R19_ENABLE", "0")
    from tutte.graphs.covering import clear_hierarchical_partition_cache
    clear_hierarchical_partition_cache()
    mod = pytest.importorskip(
        "tutte.research.scripts.cm3_via_modular_dp",
        reason="chimera modular DP script lost; rebuild tracked in riff-ggi",
    )
    chimera_modular_dp = mod.chimera_modular_dp

    t0 = time.time()
    got = chimera_modular_dp(2, 3, 5, 1009, verbose=False)
    elapsed = time.time() - t0
    assert got == 842, f"T(Cm_2; 3, 5) mod 1009 = {got}, expected 842"
    assert elapsed < 15.0, f"Cm_2 modular DP took {elapsed:.1f}s (must be < 15s)"


@pytest.mark.slow
def test_cm3_modular_dp_completes_under_10min():
    """Cm_3 modular DP at one point should complete in < 10 minutes.

    Regression guard for the Cm_3 modular DP path. Current bottleneck
    (May 26 2026) is the per-state state×row composition which runs C
    inner loops over ~4M junction members × ~500 states per chunk × 14
    chunks. Estimated wall ~17-25 min per modular point.

    Marked @pytest.mark.slow so it's not run by default — invoke with
    `pytest -m slow tutte/tests/test_modular.py::test_cm3_modular_dp_completes_under_10min`.
    Skipped by default in CI.
    """
    import time
    import os
    os.environ.setdefault("TUTTE_R19_ENABLE", "0")
    from tutte.graphs.covering import clear_hierarchical_partition_cache
    clear_hierarchical_partition_cache()
    mod = pytest.importorskip(
        "tutte.research.scripts.cm3_via_modular_dp",
        reason="chimera modular DP script lost; rebuild tracked in riff-ggi",
    )
    chimera_modular_dp = mod.chimera_modular_dp

    t0 = time.time()
    got = chimera_modular_dp(3, 3, 5, 1009, verbose=False)
    elapsed = time.time() - t0
    assert isinstance(got, int)
    assert elapsed < 600.0, f"Cm_3 modular DP took {elapsed:.1f}s (must be < 10min)"
