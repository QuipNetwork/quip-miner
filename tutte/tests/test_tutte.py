"""Tutte polynomial test suite — core engine behavior.

Sections:
    A. Spanning tree verification (Kirchhoff)
    B. Cross-validation against NetworkX
    C. Rainbow table minor finding
    D. Graph atlas coverage
    E. D-Wave hardware topologies
    F. Composition formulas
    G. Performance regression

    --- Algebraic formulas (formerly test_formulas.py) ---
    H. Unified formula — T(G) = (∏ T(cells)) · T(H)
    I. k-matching formula — closed-form for k-edge matchings
    J. Multivariate Z (Sokal) — UniformZ + MultivariateTutte
    K. Bridge-aware chord rule on K_k ⊕_k K_k

    --- Engine pipeline dispatch (formerly test_engine_pipeline.py) ---
    L. Heterogeneous tiling 
    M. Raised k-sum cap 
    N. Forced-hierarchical regression

    --- Parallel synthesis (formerly test_parallel.py) ---
    O. Pickling round-trip
    P. Parallel synthesis correctness
    Q. Cache merging
    R. Symmetric chord pairing
"""

import json
import os
import time

import networkx as nx
import pytest
from tutte.graph import (Graph, complete_graph, cut_vertex_join, cycle_graph,
                         disjoint_union, grid_graph, path_graph,
                         petersen_graph, wheel_graph)
from tutte.polynomial import TuttePolynomial
from tutte.synthesis import SynthesisEngine
from tutte.validation import (compute_tutte_networkx,
                              count_spanning_trees_kirchhoff,
                              verify_spanning_trees)

# =============================================================================
# A. SPANNING TREE VERIFICATION (Kirchhoff)
# =============================================================================

STANDARD_GRAPHS = [
    ("K_3", lambda: complete_graph(3)),
    ("K_4", lambda: complete_graph(4)),
    ("K_5", lambda: complete_graph(5)),
    ("K_6", lambda: complete_graph(6)),
    ("K_7", lambda: complete_graph(7)),
    ("C_5", lambda: cycle_graph(5)),
    ("C_8", lambda: cycle_graph(8)),
    ("C_12", lambda: cycle_graph(12)),
    ("P_4", lambda: path_graph(4)),
    ("P_8", lambda: path_graph(8)),
    ("W_5", lambda: wheel_graph(5)),
    ("W_7", lambda: wheel_graph(7)),
    ("Petersen", lambda: petersen_graph()),
    ("K_3,3", lambda: Graph.from_networkx(nx.complete_bipartite_graph(3, 3))),
    ("Grid_3x3", lambda: grid_graph(3, 3)),
]


@pytest.mark.parametrize("name,builder", STANDARD_GRAPHS, ids=[g[0] for g in STANDARD_GRAPHS])
def test_spanning_trees(name, builder, engine, benchmark_collector):
    """T(1,1) must equal Kirchhoff spanning tree count."""
    import time

    graph = builder()
    kirchhoff = count_spanning_trees_kirchhoff(graph)
    assert kirchhoff > 0, f"Kirchhoff failed for {name}"

    t0 = time.perf_counter()
    result = engine.synthesize(graph)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    tutte_trees = result.polynomial.num_spanning_trees()
    assert tutte_trees == kirchhoff, (
        f"{name}: T(1,1)={tutte_trees} != Kirchhoff={kirchhoff}"
    )

    benchmark_collector.record(
        name=name,
        nodes=graph.node_count(),
        edges=graph.edge_count(),
        spanning_trees=kirchhoff,
        timings_ms={"synthesis_cej": round(elapsed_ms, 2)},
    )


# =============================================================================
# B. CROSS-VALIDATION AGAINST NETWORKX
# =============================================================================

SMALL_GRAPHS = [
    ("K_3", lambda: complete_graph(3)),
    ("K_4", lambda: complete_graph(4)),
    ("K_5", lambda: complete_graph(5)),
    ("C_4", lambda: cycle_graph(4)),
    ("C_5", lambda: cycle_graph(5)),
    ("C_6", lambda: cycle_graph(6)),
    ("P_3", lambda: path_graph(3)),
    ("P_5", lambda: path_graph(5)),
    ("W_5", lambda: wheel_graph(5)),
    ("Petersen", lambda: petersen_graph()),
]


@pytest.mark.parametrize("name,builder", SMALL_GRAPHS, ids=[g[0] for g in SMALL_GRAPHS])
def test_tutte_matches_networkx(name, builder, engine):
    """Our polynomial must match NetworkX for graphs with <=15 edges."""
    graph = builder()
    if graph.edge_count() > 15:
        pytest.skip(f"{name} has {graph.edge_count()} edges, too slow for NetworkX")

    result = engine.synthesize(graph)
    nx_poly = compute_tutte_networkx(graph.to_networkx())
    if nx_poly is None:
        pytest.skip("sympy not available")

    assert result.polynomial == nx_poly, (
        f"{name}: our polynomial != NetworkX polynomial"
    )


# =============================================================================
# C. RAINBOW TABLE MINOR FINDING
# =============================================================================


def test_minor_k4_contains_k3(default_table):
    """K_4 should contain K_3 as a minor."""
    g = complete_graph(4)
    minors = default_table.find_minors_of(g)
    minor_names = {m.name for m in minors}
    assert "K_3" in minor_names


def test_minor_k5_contains_k4(default_table):
    """K_5 should contain K_4 as a minor."""
    g = complete_graph(5)
    minors = default_table.find_minors_of(g)
    minor_names = {m.name for m in minors}
    assert "K_4" in minor_names
    assert "K_3" in minor_names


def test_minor_petersen_contains_c5(default_table):
    """Petersen graph should contain C_5 as a minor."""
    g = petersen_graph()
    minors = default_table.find_minors_of(g)
    minor_names = {m.name for m in minors}
    assert "C_5" in minor_names


# =============================================================================
# D. GRAPH ATLAS COVERAGE
# =============================================================================


def _atlas_graphs():
    """Return cached list of (index, graph) for connected atlas graphs.

    Cached on first call so the parametrize decorator and ids list don't
    pay a second ~11s pass through `nx.graph_atlas` (collection-time cost
    that pytest still pays even when the test is deselected via -m).
    """
    cache = getattr(_atlas_graphs, "_cache", None)
    if cache is not None:
        return cache
    out = []
    for i in range(1, 1253):
        try:
            G = nx.graph_atlas(i)
        except Exception:
            continue
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            continue
        if not nx.is_connected(G):
            continue
        out.append((i, G))
    _atlas_graphs._cache = out  # type: ignore[attr-defined]
    return out


@pytest.mark.slow
@pytest.mark.parametrize(
    "atlas_idx,nx_graph",
    _atlas_graphs(),
    ids=[f"atlas_{i}" for i, _ in _atlas_graphs()],
)
def test_graph_atlas_spanning_trees(atlas_idx, nx_graph, engine):
    """Every connected atlas graph must have T(1,1) == Kirchhoff count."""
    graph = Graph.from_networkx(nx_graph)
    result = engine.synthesize(graph)
    kirchhoff = round(nx.number_of_spanning_trees(nx_graph))
    tutte_trees = result.polynomial.num_spanning_trees()
    assert tutte_trees == kirchhoff, (
        f"Atlas #{atlas_idx}: T(1,1)={tutte_trees} != Kirchhoff={kirchhoff}"
    )

    # Cross-check with NetworkX for small graphs
    if nx_graph.number_of_edges() <= 15:
        nx_poly = compute_tutte_networkx(nx_graph)
        if nx_poly is not None:
            assert result.polynomial == nx_poly, (
                f"Atlas #{atlas_idx}: polynomial mismatch with NetworkX"
            )


# =============================================================================
# E. D-WAVE HARDWARE TOPOLOGIES
# =============================================================================

# Graphs whose Tutte polynomials have been solved.  Parametrized tests below
# will skip any graph not in this set (unsolved — too many edges).
_SOLVED_DWAVE = {
    "Cm1",   # Chimera(1):  8 nodes,  16 edges
    "Z1_1",  # Zephyr(1,1): 12 nodes, 22 edges
}


def _dwave_graph(kind, *args):
    """Build a D-Wave graph, skipping if dwave-networkx is unavailable."""
    dnx = pytest.importorskip("dwave_networkx")
    builders = {
        "chimera": dnx.chimera_graph,
        "pegasus": dnx.pegasus_graph,
        "zephyr": dnx.zephyr_graph,
    }
    G = builders[kind](*args)
    if G.number_of_nodes() == 0:
        pytest.skip(f"{kind}({','.join(str(a) for a in args)}) is degenerate")
    return Graph.from_networkx(G), G


# --- Chimera C1–C16 ---------------------------------------------------------

CHIMERA_PARAMS = list(range(1, 17))


@pytest.mark.parametrize("m", CHIMERA_PARAMS, ids=[f"Cm{m}" for m in CHIMERA_PARAMS])
def test_chimera(m, engine):
    """Chimera(m) synthesis — skips unsolved topologies."""
    graph, G = _dwave_graph("chimera", m)
    tag = f"Cm{m}"

    if tag not in _SOLVED_DWAVE:
        pytest.skip(f"Cm{m} unsolved ({graph.node_count()}n, {graph.edge_count()}e)")

    result = engine.synthesize(graph)
    kirchhoff = round(nx.number_of_spanning_trees(G))
    assert result.polynomial.num_spanning_trees() == kirchhoff, (
        f"Cm{m}: T(1,1)={result.polynomial.num_spanning_trees()} != Kirchhoff={kirchhoff}"
    )


# --- Pegasus P1–P16 ---------------------------------------------------------

PEGASUS_PARAMS = list(range(1, 17))


@pytest.mark.parametrize("m", PEGASUS_PARAMS, ids=[f"Pm{m}" for m in PEGASUS_PARAMS])
def test_pegasus(m, engine):
    """Pegasus(m) synthesis — skips unsolved topologies."""
    graph, G = _dwave_graph("pegasus", m)
    tag = f"Pm{m}"

    if tag not in _SOLVED_DWAVE:
        pytest.skip(f"Pm{m} unsolved ({graph.node_count()}n, {graph.edge_count()}e)")

    result = engine.synthesize(graph)
    kirchhoff = round(nx.number_of_spanning_trees(G))
    assert result.polynomial.num_spanning_trees() == kirchhoff, (
        f"Pm{m}: T(1,1)={result.polynomial.num_spanning_trees()} != Kirchhoff={kirchhoff}"
    )


# --- Zephyr Z(m,t) ----------------------------------------------------------

_MAX_ZEPHYR_M = int(os.environ.get("TUTTE_MAX_ZEPHYR_M", "12"))
_MAX_ZEPHYR_T = int(os.environ.get("TUTTE_MAX_ZEPHYR_T", "4"))

ZEPHYR_PARAMS = [
    (m, t)
    for m in range(1, _MAX_ZEPHYR_M + 1)
    for t in range(1, _MAX_ZEPHYR_T + 1)
]


@pytest.mark.parametrize("m,t", ZEPHYR_PARAMS, ids=[f"Z{m}_{t}" for m, t in ZEPHYR_PARAMS])
def test_zephyr(m, t, engine):
    """Zephyr Z(m,t) synthesis — skips unsolved topologies."""
    graph, G = _dwave_graph("zephyr", m, t)
    tag = f"Z{m}_{t}"

    if tag not in _SOLVED_DWAVE:
        pytest.skip(f"Z({m},{t}) unsolved ({graph.node_count()}n, {graph.edge_count()}e)")

    result = engine.synthesize(graph)
    kirchhoff = round(nx.number_of_spanning_trees(G))
    assert result.polynomial.num_spanning_trees() == kirchhoff, (
        f"Z({m},{t}): T(1,1)={result.polynomial.num_spanning_trees()} != Kirchhoff={kirchhoff}"
    )


# =============================================================================
# F. COMPOSITION FORMULAS
# =============================================================================


def test_disjoint_union_formula(engine):
    """T(G1 ∪ G2) = T(G1) × T(G2)."""
    g1 = complete_graph(3)
    g2 = cycle_graph(4)
    g_union = disjoint_union(g1, g2)

    t1 = engine.synthesize(g1).polynomial
    t2 = engine.synthesize(g2).polynomial
    t_union = engine.synthesize(g_union).polynomial

    assert t_union == t1 * t2


def test_cut_vertex_k3_c4(engine):
    """T(K3 · C4) = T(K3) × T(C4) at cut vertex."""
    g1 = complete_graph(3)
    g2 = cycle_graph(4)
    joined = cut_vertex_join(g1, 0, g2, 0)

    t1 = engine.synthesize(g1).polynomial
    t2 = engine.synthesize(g2).polynomial
    t_joined = engine.synthesize(joined).polynomial

    assert t_joined == t1 * t2


def test_cut_vertex_k3_k4(engine):
    """T(K3 · K4) = T(K3) × T(K4) at cut vertex."""
    g1 = complete_graph(3)
    g2 = complete_graph(4)
    joined = cut_vertex_join(g1, 0, g2, 0)

    t1 = engine.synthesize(g1).polynomial
    t2 = engine.synthesize(g2).polynomial
    t_joined = engine.synthesize(joined).polynomial

    assert t_joined == t1 * t2


def test_cut_vertex_c4_c5(engine):
    """T(C4 · C5) = T(C4) × T(C5) at cut vertex."""
    g1 = cycle_graph(4)
    g2 = cycle_graph(5)
    joined = cut_vertex_join(g1, 0, g2, 0)

    t1 = engine.synthesize(g1).polynomial
    t2 = engine.synthesize(g2).polynomial
    t_joined = engine.synthesize(joined).polynomial

    assert t_joined == t1 * t2


# =============================================================================
# G. PERFORMANCE REGRESSION
# =============================================================================


def _load_benchmark_baseline():
    """Load baseline timings from benchmark_results.json.

    Returns dict of {name: {"synthesis_cej": ms}}.
    Returns empty dict if file is missing.
    """
    path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    baseline = {}
    for r in data.get("results", []):
        timings = r.get("timings_ms", {})
        baseline[r["name"]] = {
            "synthesis_cej": timings.get("synthesis_cej"),
        }
    return baseline


def _build_dwave_graph(kind, *args):
    """Build a D-Wave graph, skipping if dwave-networkx unavailable."""
    dnx = pytest.importorskip("dwave_networkx")
    builders = {"chimera": dnx.chimera_graph, "zephyr": dnx.zephyr_graph}
    return Graph.from_networkx(builders[kind](*args))


PERF_GRAPHS = [
    ("Petersen", lambda: petersen_graph()),
    ("K_6", lambda: complete_graph(6)),
    ("Grid_3x3", lambda: grid_graph(3, 3)),
    ("Cm1", lambda: _build_dwave_graph("chimera", 1)),
    ("Z1_1", lambda: _build_dwave_graph("zephyr", 1, 1)),
]


@pytest.mark.perf
@pytest.mark.parametrize(
    "name,builder",
    PERF_GRAPHS,
    ids=[g[0] for g in PERF_GRAPHS],
)
def test_performance_regression(name, builder, engine):
    """Synthesis must not regress >10% vs benchmark_results.json baseline."""
    baseline = _load_benchmark_baseline()
    if not baseline:
        pytest.skip("no benchmark_results.json baseline file")

    if name not in baseline:
        pytest.skip(f"no baseline for {name} in benchmark_results.json")

    baseline_ms = baseline[name].get("synthesis_cej")
    if baseline_ms is None:
        pytest.skip(f"no CEJ timing for {name} in baseline")

    graph = builder()
    kirchhoff = count_spanning_trees_kirchhoff(graph)

    t0 = time.perf_counter()
    result = engine.synthesize(graph)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Correctness check
    assert result.polynomial.num_spanning_trees() == kirchhoff, (
        f"{name}: T(1,1)={result.polynomial.num_spanning_trees()} != Kirchhoff={kirchhoff}"
    )

    # Performance check: no more than 10% regression
    threshold_ms = baseline_ms * 1.10
    assert elapsed_ms <= threshold_ms, (
        f"{name}: {elapsed_ms:.1f}ms > {threshold_ms:.1f}ms "
        f"(baseline {baseline_ms:.1f}ms + 10%)"
    )


# =============================================================================
# H. MINOR VERIFICATION
# =============================================================================


def test_star_not_minor_of_cycle():
    """S_3 (star with 3 leaves) is NOT a graph minor of C_5.

    Stars need a degree-3+ node, but contracting a cycle can only produce
    degree-2 nodes or (at most) degree-2 after merging — never degree-3+
    without adding edges.
    """
    from tutte.graph import star_graph
    from tutte.lookup import is_graph_minor

    s3 = star_graph(3)   # 4 nodes, 3 edges; center has degree 3
    c5 = cycle_graph(5)  # 5 nodes, 5 edges; all degree 2

    result = is_graph_minor(c5, s3)
    assert result is False, "S_3 should NOT be a minor of C_5"


def test_path_is_minor_of_cycle():
    """P_4 IS a graph minor of C_5 (delete one edge from cycle)."""
    from tutte.lookup import is_graph_minor

    p4 = path_graph(4)   # 4 nodes, 3 edges
    c5 = cycle_graph(5)  # 5 nodes, 5 edges

    result = is_graph_minor(c5, p4)
    assert result is True, "P_4 should be a minor of C_5"


def test_k3_is_minor_of_k4():
    """K_3 IS a graph minor of K_4 (delete one vertex)."""
    from tutte.lookup import is_graph_minor

    k3 = complete_graph(3)
    k4 = complete_graph(4)

    result = is_graph_minor(k4, k3)
    assert result is True, "K_3 should be a minor of K_4"


def test_high_degree_tree_minor():
    """K_{1,4} (star with 4 leaves) IS a minor of Petersen graph.

    Requires contraction: Petersen is 3-regular, but contracting one edge
    creates a degree-4 vertex that hosts the star center.
    """
    from tutte.graph import star_graph
    from tutte.lookup import is_graph_minor

    s4 = star_graph(4)          # 5 nodes, 4 edges; center has degree 4
    petersen = petersen_graph()  # 10 nodes, 15 edges; 3-regular

    result = is_graph_minor(petersen, s4)
    assert result is True, "K_{1,4} should be a minor of Petersen"


# =============================================================================
# I. BINARY ROUNDTRIP
# =============================================================================


def test_binary_roundtrip():
    """Encode→decode preserves all entries and polynomials."""
    from tutte.lookup import (RainbowTable, decode_rainbow_table_binary,
                              encode_rainbow_table_binary)

    table = RainbowTable()
    k3 = complete_graph(3)
    c5 = cycle_graph(5)

    k3_poly = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
    c5_coeffs = {(i, 0): 1 for i in range(1, 5)}
    c5_coeffs[(0, 1)] = 1
    c5_poly = TuttePolynomial.from_coefficients(c5_coeffs)

    table.add(k3, "K_3", k3_poly)
    table.add(c5, "C_5", c5_poly)

    data = encode_rainbow_table_binary(table)
    decoded = decode_rainbow_table_binary(data)

    assert len(decoded) == 2
    assert decoded.lookup_by_name("K_3") == k3_poly
    assert decoded.lookup_by_name("C_5") == c5_poly

    # Verify metadata
    k3_entry = decoded.get_entry("K_3")
    assert k3_entry.node_count == 3
    assert k3_entry.edge_count == 3
    assert k3_entry.spanning_trees == 3


def test_binary_roundtrip_with_minors():
    """Minor relationships survive binary encode→decode roundtrip."""
    from tutte.lookup import (RainbowTable, decode_rainbow_table_binary,
                              encode_rainbow_table_binary)

    table = RainbowTable()
    k3 = complete_graph(3)
    k4 = complete_graph(4)
    p2 = path_graph(2)

    k3_poly = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
    k4_poly = TuttePolynomial.from_coefficients(
        {(3, 0): 1, (2, 0): 3, (1, 1): 4, (1, 0): 2, (0, 1): 2, (0, 2): 3, (0, 3): 1}
    )
    p2_poly = TuttePolynomial.x(1)

    table.add(p2, "P_2", p2_poly)
    table.add(k3, "K_3", k3_poly)
    table.add(k4, "K_4", k4_poly)

    # Manually set minor relationships
    k3_key = k3.canonical_key()
    k4_key = k4.canonical_key()
    p2_key = p2.canonical_key()
    table.minor_relationships[k4_key] = [k3_key, p2_key]
    table.minor_relationships[k3_key] = [p2_key]
    table._structural_minors_computed = True

    data = encode_rainbow_table_binary(table)
    decoded = decode_rainbow_table_binary(data)

    assert decoded._structural_minors_computed is True
    assert k4_key in decoded.minor_relationships
    assert set(decoded.minor_relationships[k4_key]) == {k3_key, p2_key}
    assert k3_key in decoded.minor_relationships
    assert decoded.minor_relationships[k3_key] == [p2_key]


# =============================================================================
# H-K. ALGEBRAIC FORMULAS
# =============================================================================

# Imports needed for sections H-K
import dwave_networkx as dnx  # noqa: E402
from tutte.graph import MultiGraph, k_sum_graph  # noqa: E402
from tutte.graphs.covering import (KMatchingJunction,  # noqa: E402
                                   apply_kmatching_formula,
                                   detect_kmatching_topology,
                                   extract_cell_topology)
from tutte.graphs.k_sum import clique_chord_k_sum  # noqa: E402
from tutte.lookup.core import load_default_table  # noqa: E402
from tutte.deprecated.multivariate import MultivariateTutte, UniformZ  # noqa: E402


@pytest.fixture(scope="module")
def table():
    return load_default_table()


@pytest.fixture(scope="module")
def formulas_engine(table):
    e = SynthesisEngine(table=table, verbose=False)
    e.skip_target_lookup = True
    return e


@pytest.fixture(scope="module")
def hybrid_engine(table):
    return SynthesisEngine(table=table, verbose=False)


def _add_edges_set(g: Graph, new_edges) -> Graph:
    edges = set(g.edges)
    for u, v in new_edges:
        edges.add((min(u, v), max(u, v)))
    return Graph(nodes=g.nodes, edges=frozenset(edges))


def _try_hierarchical(engine, graph, max_depth=10):
    """Test helper — invoke merged dispatcher with `force=True` so small
    test graphs (tw<=8) still trigger discovery + Phase B formula trial.

    Replaces the legacy `engine._try_hierarchical` entry point that was
    removed when the production dispatch path was merged into
    `_try_decomposition_chord_peel`. Tests should treat this as a black
    box that returns a `SynthesisResult` (or None) for the same input
    space the deleted `_try_hierarchical` accepted.
    """
    return engine._try_decomposition_chord_peel(
        graph, max_depth, force=True,
    )


def _synthesize_hierarchical(engine, graph, cells, partition, inter_info,
                             max_depth=10):
    """Test helper — invoke `_try_cell_closed_forms` with a caller-built
    `Decomposition`, bypassing discovery so tests can feed synthetic cell
    entries (e.g. C_4) that the rainbow table wouldn't propose. Falls
    back to `_chord_peel_decomposition` when the closed-form formulas
    don't apply, mirroring the legacy `_synthesize_hierarchical`'s
    chord-rule fallthrough.
    """
    from tutte.graphs.k_sum import _classify_bridges_chords
    from tutte.synthesis.engine import Decomposition
    _, chords = _classify_bridges_chords(partition, list(inter_info.edges))
    decomp = Decomposition(
        kind="cell",
        label="test_forced",
        components=partition,
        families=[c.name for c in cells],
        cell_entries=list(cells),
        inter_edges=list(inter_info.edges),
        chord_edges=list(chords),
        predicted_chord_cost=0.0,
        peel_type="inter",
        inter_info=inter_info,
    )
    res = engine._try_cell_closed_forms(graph, decomp, max_depth)
    if res is not None:
        return res
    return engine._chord_peel_decomposition(
        graph, decomp, max_depth,
        recurse_residue=False, min_recursion_size=12,
    )


# =============================================================================
# H. UNIFIED FORMULA  (T(G) = (∏ T(cells)) · T(H))
# =============================================================================


def test_extract_topology_disjoint_cells_returns_empty_h():
    partition = [{0, 1, 2}, {3, 4, 5}]
    H = extract_cell_topology(partition, [])
    assert H is not None
    assert len(H.nodes) == 2
    assert sum(H.edge_counts.values()) == 0


def test_extract_topology_one_bridge_returns_single_edge():
    partition = [{0, 1, 2}, {3, 4, 5}]
    H = extract_cell_topology(partition, [(0, 3)])
    assert H is not None
    assert len(H.nodes) == 2
    assert H.edge_counts == {(0, 1): 1}


def test_extract_topology_two_parallel_returns_multiplicity_two():
    partition = [{0, 1, 2}, {3, 4, 5}]
    H = extract_cell_topology(partition, [(0, 3), (0, 3)])
    assert H is not None
    assert H.edge_counts == {(0, 1): 2}


def test_extract_topology_distinct_pairs_returns_none():
    """Two inter-cell edges between same cell-pair but DIFFERENT vertex pairs
    → unified formula breaks → returns None."""
    partition = [{0, 1, 2}, {3, 4, 5}]
    H = extract_cell_topology(partition, [(0, 3), (1, 4)])
    assert H is None


def test_extract_topology_three_cells_triangle_of_bridges():
    partition = [{0, 1, 2}, {3, 4, 5}, {6, 7, 8}]
    inter = [(0, 3), (3, 6), (0, 6)]
    H = extract_cell_topology(partition, inter)
    assert H is not None
    assert len(H.nodes) == 3
    assert sorted(H.edge_counts.items()) == [((0, 1), 1), ((0, 2), 1), ((1, 2), 1)]


def test_extract_topology_three_cells_one_pair_distinct_returns_none():
    partition = [{0, 1, 2}, {3, 4, 5}, {6, 7, 8}]
    inter = [(0, 3), (1, 4), (3, 6)]  # cells (0,1) has TWO distinct pairs
    H = extract_cell_topology(partition, inter)
    assert H is None


def _two_k3_disjoint() -> Graph:
    return disjoint_union(complete_graph(3), complete_graph(3))


def _two_k3_one_bridge() -> Graph:
    return _add_edges_set(_two_k3_disjoint(), [(0, 3)])


def _three_k3_chain_of_bridges() -> Graph:
    g = disjoint_union(_two_k3_disjoint(), complete_graph(3))
    return _add_edges_set(g, [(0, 3), (3, 6)])


def _three_k3_triangle_of_bridges() -> Graph:
    g = disjoint_union(_two_k3_disjoint(), complete_graph(3))
    return _add_edges_set(g, [(0, 3), (3, 6), (0, 6)])


def _two_k3_distinct_pair_chords() -> Graph:
    return _add_edges_set(_two_k3_disjoint(), [(0, 3), (1, 4)])


def test_engine_two_k3_one_bridge_uses_unified_formula(formulas_engine):
    g = _two_k3_one_bridge()
    res = _try_hierarchical(formulas_engine, g, max_depth=10)
    assert res is not None
    assert res.method == "unified_formula"
    T_k3 = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
    expected = TuttePolynomial.x() * T_k3 * T_k3
    assert res.polynomial == expected
    assert verify_spanning_trees(g, res.polynomial)


def test_engine_three_k3_chain_of_bridges_uses_unified_formula(formulas_engine):
    g = _three_k3_chain_of_bridges()
    res = _try_hierarchical(formulas_engine, g, max_depth=10)
    assert res is not None
    assert res.method == "unified_formula"
    T_k3 = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
    expected = TuttePolynomial.x(2) * T_k3 * T_k3 * T_k3
    assert res.polynomial == expected


def test_engine_three_k3_triangle_of_bridges_uses_unified_formula(formulas_engine):
    g = _three_k3_triangle_of_bridges()
    res = _try_hierarchical(formulas_engine, g, max_depth=10)
    assert res is not None
    assert res.method == "unified_formula"
    T_k3 = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
    expected = T_k3 * T_k3 * T_k3 * T_k3
    assert res.polynomial == expected


def test_engine_chord_case_falls_through(formulas_engine):
    g = _two_k3_distinct_pair_chords()
    res = _try_hierarchical(formulas_engine, g, max_depth=10)
    assert res is not None
    assert res.method != "unified_formula"
    assert verify_spanning_trees(g, res.polynomial)


def _table_entry(table_, name: str):
    matches = [e for e in table_.entries.values() if e.name == name]
    assert matches, f"rainbow table missing entry {name!r}"
    return matches[0]


def _analyze_inter(graph: Graph, partition):
    from tutte.graphs.covering import analyze_inter_cell_edges
    return analyze_inter_cell_edges(graph, partition)


def test_engine_heterogeneous_k3_plus_k4_one_bridge_uses_unified_formula(
    formulas_engine, table
):
    """K₃ + K₄ + 1 bridge: heterogeneous decomposition fed directly."""
    g = _add_edges_set(
        disjoint_union(complete_graph(3), complete_graph(4)), [(0, 3)]
    )
    partition = [{0, 1, 2}, {3, 4, 5, 6}]
    cells = [_table_entry(table, "K_3"), _table_entry(table, "K_4")]
    inter = _analyze_inter(g, partition)

    res = _synthesize_hierarchical(formulas_engine, g, cells, partition, inter, max_depth=10)
    assert res.method == "unified_formula"
    T_k3 = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
    T_k4 = formulas_engine.synthesize(complete_graph(4)).polynomial
    expected = TuttePolynomial.x() * T_k3 * T_k4
    assert res.polynomial == expected
    assert verify_spanning_trees(g, res.polynomial)


def test_engine_heterogeneous_k3_plus_c4_one_bridge_uses_unified_formula(
    formulas_engine, table
):
    """K₃ + C₄ + 1 bridge with a synthetic MinorEntry for C₄."""
    from tutte.lookup.core import MinorEntry

    c4 = cycle_graph(4)
    T_c4 = formulas_engine.synthesize(c4).polynomial
    c4_entry = MinorEntry(
        name="C_4",
        polynomial=T_c4,
        node_count=c4.node_count(),
        edge_count=c4.edge_count(),
        canonical_key=c4.canonical_key(),
        spanning_trees=int(T_c4.num_spanning_trees()),
        num_terms=T_c4.num_terms(),
        graph=c4,
    )

    g = _add_edges_set(disjoint_union(complete_graph(3), c4), [(0, 3)])
    partition = [{0, 1, 2}, {3, 4, 5, 6}]
    cells = [_table_entry(table, "K_3"), c4_entry]
    inter = _analyze_inter(g, partition)

    res = _synthesize_hierarchical(formulas_engine, g, cells, partition, inter, max_depth=10)
    assert res.method == "unified_formula"
    T_k3 = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
    expected = TuttePolynomial.x() * T_k3 * T_c4
    assert res.polynomial == expected


@pytest.mark.slow
def test_cm2_chord_case_falls_through_to_treewidth_dp(table):
    """Cm2 chord case must NOT fire unified formula. ~200s."""
    cm2 = Graph.from_networkx(dnx.chimera_graph(2))

    e_hier = SynthesisEngine(table=table, verbose=False)
    e_hier.skip_target_lookup = True
    forced = _try_hierarchical(e_hier, cm2, max_depth=10)
    assert forced is not None
    assert forced.method != "unified_formula"
    assert verify_spanning_trees(cm2, forced.polynomial)


# =============================================================================
# I. K-MATCHING FORMULA 
# =============================================================================


def _build_2cell_k_matching(cell: Graph, k: int) -> Graph:
    """Two disjoint cells joined by a k-edge matching on the first k anchors."""
    g = disjoint_union(cell, cell)
    offset = max(cell.nodes) + 1
    anchors = sorted(cell.nodes)[:k]
    edges = [(a, a + offset) for a in anchors]
    return _add_edges_set(g, edges)


def _build_cell_path_k_matching(cell: Graph, n: int, k: int) -> Graph:
    g = cell
    offsets = [0]
    for _ in range(n - 1):
        offsets.append(max(g.nodes) + 1)
        g = disjoint_union(g, cell)
    anchors = sorted(cell.nodes)[:k]
    for i in range(n - 1):
        edges = [(a + offsets[i], a + offsets[i + 1]) for a in anchors]
        g = _add_edges_set(g, edges)
    return g


def _build_cell_cycle_k_matching(cell: Graph, n: int, k: int) -> Graph:
    g = _build_cell_path_k_matching(cell, n, k)
    offset_first = 0
    offset_last = (n - 1) * (max(cell.nodes) + 1)
    anchors = sorted(cell.nodes)[:k]
    edges = [(a + offset_first, a + offset_last) for a in anchors]
    return _add_edges_set(g, edges)


def test_kmatching_detector_two_k3_m2_returns_junction():
    K3 = complete_graph(3)
    g = _build_2cell_k_matching(K3, k=2)
    partition = [{0, 1, 2}, {3, 4, 5}]
    inter = [(0, 3), (1, 4)]
    result = detect_kmatching_topology(g, partition, inter)
    assert result is not None
    assert len(result) == 1
    j = result[0]
    assert j.k == 2
    assert j.cell_i == 0 and j.cell_j == 1
    assert set(j.anchors_i) == {0, 1}
    assert set(j.anchors_j) == {3, 4}


def test_kmatching_detector_no_inter_edges_returns_empty():
    partition = [{0, 1, 2}, {3, 4, 5}]
    result = detect_kmatching_topology(complete_graph(3), partition, [])
    assert result == []


def test_kmatching_detector_cm1_bipartite_mixed_returns_none():
    """K_{4,4} cells with mixed-side anchors should fail precondition."""
    cm1 = Graph.from_networkx(dnx.chimera_graph(1))
    g = disjoint_union(cm1, cm1)
    offset = max(cm1.nodes) + 1
    g = _add_edges_set(g, [(0, 0 + offset), (1, 1 + offset)])
    partition = [set(cm1.nodes), {n + offset for n in cm1.nodes}]
    inter = [(0, 0 + offset), (1, 1 + offset)]
    result = detect_kmatching_topology(g, partition, inter)
    assert result is None


def test_engine_k3_path_m2_uses_kmatching_formula(formulas_engine):
    g = _build_cell_path_k_matching(complete_graph(3), n=3, k=2)
    res = _try_hierarchical(formulas_engine, g, max_depth=10)
    assert res is not None
    assert res.method == "kmatching_formula"
    direct = formulas_engine.synthesize(g).polynomial
    assert res.polynomial == direct


def test_engine_k4_path_m2_uses_kmatching_formula(formulas_engine):
    g = _build_cell_path_k_matching(complete_graph(4), n=3, k=2)
    res = _try_hierarchical(formulas_engine, g, max_depth=10)
    assert res is not None
    assert res.method == "kmatching_formula"
    direct = formulas_engine.synthesize(g).polynomial
    assert res.polynomial == direct


def test_engine_small_cell_cycle_falls_through(formulas_engine):
    """K_3 cycle topology shares anchors → formula must reject + fall through.

    Note (May 2026): the SINGLE-junction §4 formula collapses correctly
    for K_n cells even with shared anchors, but the engine's MULTI-junction
    RECURSIVE form does NOT — recursion's parallel-edge bookkeeping
    breaks. Detector P3 must remain enforced for engine path.
    Reference: relaxed_shared_anchors_findings.md."""
    g = _build_cell_cycle_k_matching(complete_graph(3), n=3, k=2)
    res = _try_hierarchical(formulas_engine, g, max_depth=10)
    assert res is not None
    assert res.method != "kmatching_formula"
    direct = formulas_engine.synthesize(g).polynomial
    assert res.polynomial == direct


def test_engine_k44_cycle_uses_kmatching_formula(formulas_engine):
    """Cm1 = K_{4,4} cells joined via M_4 coupler (the actual D-Wave Cm2 case)."""
    cm1 = Graph.from_networkx(dnx.chimera_graph(1))
    g = disjoint_union(cm1, cm1)
    offset = max(cm1.nodes) + 1
    A_side = [0, 5, 6, 7]
    B_side = [1, 2, 3, 4]
    edges = [(A_side[i], B_side[i] + offset) for i in range(4)]
    g = _add_edges_set(g, edges)
    res = _try_hierarchical(formulas_engine, g, max_depth=10)
    assert res is not None
    assert res.method == "kmatching_formula"
    direct = formulas_engine.synthesize(g).polynomial
    assert res.polynomial == direct


def test_engine_mixed_side_k44_falls_through(formulas_engine):
    """Mixed-bipartition anchors → formula must NOT fire."""
    cm1 = Graph.from_networkx(dnx.chimera_graph(1))
    g = disjoint_union(cm1, cm1)
    offset = max(cm1.nodes) + 1
    g = _add_edges_set(g, [(0, 0 + offset), (1, 1 + offset)])
    res = _try_hierarchical(formulas_engine, g, max_depth=10)
    if res is not None:
        assert res.method != "kmatching_formula"


def test_engine_single_parallel_edge_still_uses_unified_formula(formulas_engine):
    """Two K_3 + one bridge: unified formula (k=1), not k-matching."""
    K3 = complete_graph(3)
    g = _add_edges_set(disjoint_union(K3, K3), [(0, 3)])
    res = _try_hierarchical(formulas_engine, g, max_depth=10)
    assert res is not None
    assert res.method == "unified_formula"


@pytest.mark.slow
def test_cm2_uses_kmatching_formula(table):
    """Cm2 routes through k-matching formula. ~50s wall-clock."""
    cm2 = Graph.from_networkx(dnx.chimera_graph(2))
    e_hier = SynthesisEngine(table=table, verbose=False)
    e_hier.skip_target_lookup = True
    res = _try_hierarchical(e_hier, cm2, max_depth=10)
    assert res is not None
    assert res.method == "kmatching_formula"
    assert res.polynomial.num_terms() == 675


# --- Full-clique-separator chord-rule fix (May 2026) ---


def _cycle_through_router(left_cycle: int, right_cycle: int, router_size: int) -> Graph:
    """Two cycles bridged by a complete K_router clique. The router is a
    full K_router_size separator (all clique edges present)."""
    G = nx.Graph()
    router = list(range(router_size))
    left = list(range(router_size, router_size + left_cycle))
    right = list(range(
        router_size + left_cycle,
        router_size + left_cycle + right_cycle,
    ))
    G.add_nodes_from(router + left + right)
    for i in range(router_size):
        for j in range(i + 1, router_size):
            G.add_edge(router[i], router[j])
    for i in range(left_cycle):
        G.add_edge(left[i], left[(i + 1) % left_cycle])
    for i in range(right_cycle):
        G.add_edge(right[i], right[(i + 1) % right_cycle])
    for u in left:
        for v in router:
            G.add_edge(u, v)
    for u in right:
        for v in router:
            G.add_edge(u, v)
    return Graph.from_networkx(G)


@pytest.mark.parametrize("k", [3, 4, 5])
def test_full_clique_separator_chord_peel(k, formulas_engine):
    """Regression: graphs with a full K_k clique separator (k ≥ 3) must be
    handled by the chord-peel-on-existing-clique-edges path.

    Pre-May 2026, `_find_vertex_separator` skipped `missing == 0` separators,
    silently dropping these from the k-sum search. The fix dispatches to
    `_apply_ksum` which now peels the existing K_k clique edges via the
    chord rule (symmetric to peeling missing edges in the standard case).

    Test verifies polynomial correctness via Kirchhoff matrix-tree theorem
    on synthetic C_5 + K_k + C_5 router graphs.
    """
    g = _cycle_through_router(left_cycle=5, right_cycle=5, router_size=k)
    separator = tuple(range(k))
    result = formulas_engine._apply_ksum(g, separator, k)
    assert result is not None, f"k={k}: _apply_ksum returned None"
    assert result.method == f"{k}sum_full_clique_chord_peel"
    assert result.verified
    expected = count_spanning_trees_kirchhoff(g)
    assert result.polynomial.num_spanning_trees() == expected, (
        f"k={k}: T(1,1) = {result.polynomial.num_spanning_trees()} "
        f"!= Kirchhoff = {expected}"
    )


def test_full_clique_separator_found_by_search(formulas_engine):
    """`_find_vertex_separator` must NOT skip full-clique separators
    (missing == 0). Pre-May 2026 it returned None for these; post-fix
    it returns the separator tuple.
    """
    g = _cycle_through_router(left_cycle=5, right_cycle=5, router_size=4)
    sep = formulas_engine._find_vertex_separator(g, 4)
    assert sep is not None, "_find_vertex_separator should find K_4 router separator"
    assert len(sep) == 4
    # Confirm all C(4, 2) = 6 clique edges are present (i.e., missing == 0)
    sv = sorted(sep)
    missing = sum(
        1 for i in range(4) for j in range(i + 1, 4)
        if (min(sv[i], sv[j]), max(sv[i], sv[j])) not in g.edges
    )
    assert missing == 0, f"Expected full clique separator (missing=0), got {missing}"


# --- Verification precision fix (May 2026) ---


def test_verify_spanning_trees_uses_exact_arithmetic_on_dense_graphs():
    """Regression: verify_spanning_trees and count_spanning_trees_kirchhoff
    must NOT use float64 `nx.number_of_spanning_trees` on small dense
    graphs. Pre-fix, K_3+K_9+K_3 (15n 96e, 4.78e14 spanning trees)
    triggered Gaussian-elimination float overflow in nx — the float
    result was 478296900000002.06, off by 2 from the exact integer
    478296900000000. This caused spurious Kirchhoff mismatches on every
    correct synthesis of dense small graphs and made cotree_dp /
    almost_cograph appear to produce wrong polynomials when they were
    actually correct.

    The fix (May 2026): use exact sympy determinant for ALL graph sizes.
    Sub-millisecond cost on small graphs.
    """
    from tutte.polynomial import TuttePolynomial
    from tutte.validation import (count_spanning_trees_kirchhoff,
                                  verify_spanning_trees)

    # Build K_3+K_9+K_3 router (15n 96e, cograph)
    G = nx.Graph()
    router = list(range(9))
    left = list(range(9, 12))
    right = list(range(12, 15))
    G.add_nodes_from(router + left + right)
    for i in range(9):
        for j in range(i + 1, 9):
            G.add_edge(router[i], router[j])
    for i in range(3):
        for j in range(i + 1, 3):
            G.add_edge(left[i], left[j])
            G.add_edge(right[i], right[j])
    for u in left + right:
        for v in router:
            G.add_edge(u, v)
    g = Graph.from_networkx(G)
    assert g.node_count() == 15
    assert g.edge_count() == 96

    expected = 478296900000000  # exact spanning tree count

    # Exact path: must hit the integer determinant, not float nx.
    assert count_spanning_trees_kirchhoff(g) == expected, (
        "count_spanning_trees_kirchhoff must use exact sympy det, not float64 nx, "
        f"on this dense 15n case. Pre-fix returned 478296900000002 (off by 2)."
    )

    # Construct a polynomial with T(1,1) = expected (any coefficient layout
    # summing to `expected` works) and verify it passes verify_spanning_trees.
    fake_poly = TuttePolynomial.from_coefficients({(0, 0): expected})
    assert verify_spanning_trees(g, fake_poly), (
        "verify_spanning_trees must use exact sympy det, not float64 nx, "
        "on this dense 15n case."
    )

    # And a polynomial off by 1 must fail verification.
    wrong_poly = TuttePolynomial.from_coefficients({(0, 0): expected + 1})
    assert not verify_spanning_trees(g, wrong_poly), (
        "verify_spanning_trees should reject an off-by-1 polynomial, "
        "not silently accept it via float-precision rounding."
    )


# =============================================================================
# J. MULTIVARIATE Z (Sokal)
# =============================================================================


def _components(graph: Graph) -> int:
    parent = {v: v for v in graph.nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in graph.edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[max(ru, rv)] = min(ru, rv)
    return len({find(v) for v in graph.nodes})


def _verify_z_via_sokal(graph: Graph) -> None:
    n_v = graph.node_count()
    n_e = graph.edge_count()
    n_k = _components(graph)

    eng = SynthesisEngine(table=load_default_table(), verbose=False)
    T = eng.synthesize(graph).polynomial

    Z = UniformZ.from_subgraph_sum(graph)
    assert Z.evaluate(1, 1) == 2 ** n_e

    test_points = [(2, 2), (3, 2), (2, 3), (3, 3), (4, 2), (-1, 2), (2, -1)]
    for x_val, y_val in test_points:
        if x_val == 1 or y_val == 1:
            continue
        q_val = (x_val - 1) * (y_val - 1)
        v_val = y_val - 1
        t_val = T.evaluate(x_val, y_val)
        z_val = Z.evaluate(q_val, v_val)
        # Sokal: Z((x-1)(y-1), y-1) = T(x, y) · (x-1)^{k(G)} · (y-1)^{|V|}
        rhs = t_val * ((x_val - 1) ** n_k) * ((y_val - 1) ** n_v)
        assert z_val == rhs


def test_uniform_z_zero_one_arithmetic():
    z = UniformZ.zero()
    assert z.coeff_count() == 0
    one = UniformZ.one()
    assert one.coeff_count() == 1
    assert one.evaluate(5, 7) == 1
    assert (one + one).evaluate(5, 7) == 2
    assert (one * one).evaluate(5, 7) == 1
    assert (3 * one).evaluate(5, 7) == 3
    assert (-one).evaluate(5, 7) == -1


def test_uniform_z_subgraph_sum_K3():
    K3 = Graph.from_networkx(nx.complete_graph(3))
    Z = UniformZ.from_subgraph_sum(K3)
    assert Z.evaluate(1, 1) == 8
    assert Z.evaluate(2, 0) == 8


def test_uniform_z_path_graph_3():
    """Path P_3: Z = q^3 + 2q^2 v + q v^2."""
    P3 = Graph(list(range(3)), [(0, 1), (1, 2)])
    Z = UniformZ.from_subgraph_sum(P3)
    expected = {(3, 0): 1, (2, 1): 2, (1, 2): 1}
    assert Z.to_dict() == expected


def test_sokal_identity_K3():
    _verify_z_via_sokal(Graph.from_networkx(nx.complete_graph(3)))


def test_sokal_identity_K4():
    _verify_z_via_sokal(Graph.from_networkx(nx.complete_graph(4)))


def test_sokal_identity_K_2_2():
    _verify_z_via_sokal(Graph.from_networkx(nx.complete_bipartite_graph(2, 2)))


def test_sokal_identity_C5():
    _verify_z_via_sokal(Graph.from_networkx(nx.cycle_graph(5)))


@pytest.mark.skip(
    reason="engine.synthesize errors on disconnected Graph (set & list bug "
    "in graph.subgraph); k=2 case validated via UniformZ alone in "
    "test_uniform_z_disconnected_components"
)
def test_sokal_identity_disconnected():
    g = Graph(list(range(4)), [(0, 1), (2, 3)])
    _verify_z_via_sokal(g)


def test_uniform_z_disconnected_components():
    """k=2 disconnected: Z = (q^2 + qv)^2 = q^4 + 2q^3 v + q^2 v^2."""
    g = Graph(list(range(4)), [(0, 1), (2, 3)])
    Z = UniformZ.from_subgraph_sum(g)
    expected = {(4, 0): 1, (3, 1): 2, (2, 2): 1}
    assert Z.to_dict() == expected
    assert Z.evaluate(1, 1) == 4


def test_uniform_z_arithmetic_linear():
    a = UniformZ.from_dict({(1, 0): 2})
    b = UniformZ.from_dict({(0, 1): 3})
    c = UniformZ.from_dict({(1, 1): 5})
    d = UniformZ.one()
    lhs = (a + b) * (c + d)
    rhs = a * c + a * d + b * c + b * d
    assert lhs == rhs


def test_multivariate_tutte_specialize():
    """MultivariateTutte → UniformZ via specialize_uniform aggregates v powers."""
    mt = MultivariateTutte.from_dict({
        (3, frozenset()): 1,
        (2, frozenset([(0, 1)])): 1,
        (2, frozenset([(0, 1), (1, 1)])): 1,
        (1, frozenset([(0, 2)])): 1,
    })
    z = mt.specialize_uniform()
    expected = {
        (3, 0): 1,
        (2, 1): 1,
        (2, 2): 1,
        (1, 2): 1,
    }
    assert z.to_dict() == expected


# =============================================================================
# K. BRIDGE-AWARE CHORD RULE — K_k ⊕_k K_k DEGENERATE CASES
# =============================================================================


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_kk_ksum_kk_returns_one(k, hybrid_engine):
    """K_k ⊕_k K_k is the empty graph on k vertices, T = 1."""
    target = k_sum_graph(complete_graph(k), complete_graph(k), k, list(range(k)))
    assert target.edge_count() == 0
    assert target.node_count() == k

    result = clique_chord_k_sum(target, tuple(range(k)), k, hybrid_engine)
    assert result == TuttePolynomial.one()
    assert result.num_spanning_trees() == 1


@pytest.mark.parametrize("k,n_extra", [(2, 2), (3, 1), (3, 2), (4, 1)])
def test_kk_ksum_knextra_correctness(k, n_extra, hybrid_engine):
    """K_k ⊕_k K_(k+n_extra) — non-degenerate true k-sum."""
    g1 = complete_graph(k)
    g2 = complete_graph(k + n_extra)
    target = k_sum_graph(g1, g2, k, list(range(k)))
    result = clique_chord_k_sum(target, tuple(range(k)), k, hybrid_engine)
    direct = hybrid_engine.synthesize(target).polynomial
    assert result == direct


def test_chord_rule_does_not_regress_petersen(hybrid_engine):
    """Bridge-aware fix doesn't break Petersen."""
    from tutte.graphs.covering import try_hierarchical_partition
    from tutte.graphs.k_sum import boundary_quotient_tutte

    petersen = Graph.from_networkx(nx.petersen_graph())
    table_local = load_default_table()
    decomp = try_hierarchical_partition(petersen, table_local)
    assert decomp is not None
    cell, partition, inter_info = decomp

    chord_result = boundary_quotient_tutte(
        petersen, partition, list(inter_info.edges), hybrid_engine,
    )
    direct = hybrid_engine.synthesize(petersen).polynomial
    assert chord_result == direct
    assert chord_result.num_spanning_trees() == 2000


# =============================================================================
# L-N. ENGINE PIPELINE DISPATCH
# =============================================================================

from tutte.graphs.covering import (try_heterogeneous_partition,  # noqa: E402
                                   try_hierarchical_partition)


@pytest.fixture(scope="module")
def pipeline_engine(table):
    return SynthesisEngine(table=table, verbose=False)


# =============================================================================
# L. HETEROGENEOUS TILING 
# =============================================================================


def _disjoint_blocks_with_bridges(block_sizes, cross_pairs):
    G = nx.Graph()
    offsets = []
    cursor = 0
    for size in block_sizes:
        offsets.append(cursor)
        block = nx.complete_graph(size)
        G = nx.disjoint_union(G, block)
        cursor += size
    for (bi, ni), (bj, nj) in cross_pairs:
        G.add_edge(offsets[bi] + ni, offsets[bj] + nj)
    return Graph.from_networkx(G)


def test_partitioner_finds_k4_plus_2k3(table):
    """K_4 + K_3 + K_3 disjoint union: heterogeneous picks K_4 first."""
    g = _disjoint_blocks_with_bridges([4, 3, 3], cross_pairs=[])
    assert try_hierarchical_partition(g, table) is None
    het = try_heterogeneous_partition(g, table)
    assert het is not None
    cells, partition, inter_info = het
    sizes = sorted(len(p) for p in partition)
    assert sizes == [3, 3, 4]
    names = sorted(c.name for c in cells)
    assert names == ["K_3", "K_3", "K_4"]
    assert len(inter_info.edges) == 0


def test_partitioner_rejects_pure_homogeneous(table):
    g = _disjoint_blocks_with_bridges([3, 3, 3], cross_pairs=[])
    het = try_heterogeneous_partition(g, table)
    assert het is None


def test_partitioner_returns_none_when_no_cover(table):
    g = Graph.from_networkx(nx.disjoint_union_all([nx.path_graph(2)] * 5))
    het = try_heterogeneous_partition(g, table)
    assert het is None


def test_engine_synthesizes_heterogeneous_with_inter_edges(pipeline_engine):
    g = _disjoint_blocks_with_bridges(
        [4, 3, 3],
        cross_pairs=[
            ((0, 0), (1, 0)),
            ((0, 1), (1, 1)),
            ((0, 2), (2, 0)),
            ((0, 3), (2, 1)),
            ((1, 2), (2, 2)),
        ],
    )
    result = pipeline_engine.synthesize(g)
    assert verify_spanning_trees(g, result.polynomial)


def test_engine_heterogeneous_matches_direct_synthesis_path(pipeline_engine, table):
    g = _disjoint_blocks_with_bridges(
        [4, 3, 3],
        cross_pairs=[
            ((0, 0), (1, 0)),
            ((0, 1), (2, 0)),
            ((1, 1), (2, 1)),
        ],
    )
    het = try_heterogeneous_partition(g, table)
    assert het is not None
    cells, partition, inter_info = het

    pipeline_poly = pipeline_engine.synthesize(g).polynomial
    forced_poly = _synthesize_hierarchical(pipeline_engine, 
        g, cells, partition, inter_info, max_depth=10,
    ).polynomial
    assert pipeline_poly == forced_poly


def test_petersen_homogeneous_still_wins(pipeline_engine, table):
    g = Graph.from_networkx(nx.petersen_graph())
    result = pipeline_engine.synthesize(g)
    assert verify_spanning_trees(g, result.polynomial)
    assert result.polynomial.num_spanning_trees() == 2000


# =============================================================================
# M. RAISED K-SUM CAP 
# =============================================================================


def test_k_max_default_and_clamping(table):
    eng_default = SynthesisEngine(table=table, verbose=False)
    assert eng_default.k_max == 12

    eng_low = SynthesisEngine(table=table, verbose=False, k_max=5)
    assert eng_low.k_max == 5

    eng_clamped_high = SynthesisEngine(table=table, verbose=False, k_max=99)
    assert eng_clamped_high.k_max == 20

    eng_clamped_low = SynthesisEngine(table=table, verbose=False, k_max=1)
    assert eng_clamped_low.k_max == 2


def _bipartite_through_router(left_size: int, right_size: int, router_size: int) -> Graph:
    G = nx.Graph()
    router = list(range(router_size))
    left = list(range(router_size, router_size + left_size))
    right = list(range(router_size + left_size, router_size + left_size + right_size))
    G.add_nodes_from(router + left + right)
    for i in range(router_size):
        for j in range(i + 1, router_size):
            G.add_edge(router[i], router[j])
    for i in range(left_size):
        for j in range(i + 1, left_size):
            G.add_edge(left[i], left[j])
    for i in range(right_size):
        for j in range(i + 1, right_size):
            G.add_edge(right[i], right[j])
    for u in left:
        for v in router:
            G.add_edge(u, v)
    for u in right:
        for v in router:
            G.add_edge(u, v)
    return Graph.from_networkx(G)


@pytest.mark.parametrize(
    "router_size",
    [
        5,  # fast smoke (~1.4s)
        pytest.param(6, marks=pytest.mark.slow),  # ~5s
        pytest.param(7, marks=pytest.mark.slow),  # ~18s
        pytest.param(8, marks=pytest.mark.slow),  # ~53s
    ],
)
def test_router_separator_synthesis_correct(table, router_size):
    """Two K_3 cliques bridged by a K_router_size router."""
    g = _bipartite_through_router(left_size=3, right_size=3, router_size=router_size)
    eng = SynthesisEngine(table=table, verbose=False)
    result = eng.synthesize(g)
    assert verify_spanning_trees(g, result.polynomial)


def test_below_min_edges_skipped(table):
    """C_6 still synthesizes correctly."""
    g = Graph.from_networkx(nx.cycle_graph(6))
    eng = SynthesisEngine(table=table, verbose=False)
    result = eng.synthesize(g)
    assert result.polynomial.num_spanning_trees() == 6


# =============================================================================
# N. FORCED-HIERARCHICAL REGRESSION
# =============================================================================


def _build_two_k5_with_connector() -> Graph:
    G = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    G.add_edge(0, 5)
    G.add_edge(1, 6)
    G.add_edge(2, 7)
    G.add_edge(3, 8)
    return Graph.from_networkx(G)


def test_synthetic_hierarchical_matches_default(table):
    """Synthetic 2×K_5 with 4 chords. Forced hierarchical agrees with default."""
    g = _build_two_k5_with_connector()
    assert g.edge_count() >= 20

    e_hier = SynthesisEngine(table=table, verbose=False)
    e_hier.skip_target_lookup = True
    forced = _try_hierarchical(e_hier, g, max_depth=10)
    assert forced is not None
    assert verify_spanning_trees(g, forced.polynomial)

    e_default = SynthesisEngine(table=table, verbose=False)
    e_default.skip_target_lookup = True
    default = e_default.synthesize(g)
    assert forced.polynomial == default.polynomial


def test_synthetic_hierarchical_timing_within_10x(table):
    g = _build_two_k5_with_connector()

    e_default = SynthesisEngine(table=table, verbose=False)
    e_default.skip_target_lookup = True
    t0 = time.perf_counter()
    e_default.synthesize(g)
    t_default = time.perf_counter() - t0

    e_hier = SynthesisEngine(table=table, verbose=False)
    e_hier.skip_target_lookup = True
    t0 = time.perf_counter()
    res = _try_hierarchical(e_hier, g, max_depth=10)
    t_hier = time.perf_counter() - t0

    assert res is not None
    assert t_hier < t_default * 10 + 0.5


@pytest.mark.slow
def test_pm2_routes_through_chord_rule(table):
    """Pm2 (40n 164e, 95 chords): routes through chord-rule paths."""
    g = Graph.from_networkx(dnx.pegasus_graph(2))
    eng = SynthesisEngine(table=table, verbose=False)
    eng.skip_target_lookup = True
    result = eng.synthesize(g)
    assert verify_spanning_trees(g, result.polynomial)
    assert result.method != "lookup"


def test_cell_quotient_tree_dp_path_topology(table):
    """compute_cell_quotient_tree_dp returns correct polynomial on a
    K_{4,4} 3-cell M_2 path tree topology graph."""
    from tutte.tests.tree_dp_branching import \
        build_tree_graph
    from tutte.roots import compute_cell_quotient_tree_dp

    K44 = Graph.from_networkx(nx.complete_bipartite_graph(4, 4))
    edges = [(0, 1), (1, 2)]
    anchors = {0: {1: [0, 1]}, 1: {0: [0, 1], 2: [0, 1]}, 2: {1: [0, 1]}}
    t = nx.Graph()
    t.add_nodes_from({n for e in edges for n in e})
    t.add_edges_from(edges)
    g = build_tree_graph(K44, t, anchors)
    poly = compute_cell_quotient_tree_dp(g, table)
    assert poly is not None, "tree DP should fire on K_{4,4} M_2 path"
    assert verify_spanning_trees(g, poly)


def test_cell_quotient_tree_dp_branching_topology(table):
    """compute_cell_quotient_tree_dp returns correct polynomial on a
    K_{4,4} star (1 center + 4 leaves) M_2 graph."""
    from tutte.tests.tree_dp_branching import \
        build_tree_graph
    from tutte.roots import compute_cell_quotient_tree_dp

    K44 = Graph.from_networkx(nx.complete_bipartite_graph(4, 4))
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    anchors = {
        0: {1: [0, 1], 2: [0, 1], 3: [4, 5], 4: [4, 5]},
        1: {0: [0, 1]}, 2: {0: [0, 1]},
        3: {0: [0, 1]}, 4: {0: [0, 1]},
    }
    t = nx.Graph()
    t.add_nodes_from({n for e in edges for n in e})
    t.add_edges_from(edges)
    g = build_tree_graph(K44, t, anchors)
    poly = compute_cell_quotient_tree_dp(g, table)
    assert poly is not None, "tree DP should fire on K_{4,4} star"
    assert verify_spanning_trees(g, poly)


def test_cell_quotient_tree_dp_returns_none_on_cycle(table):
    """compute_cell_quotient_tree_dp returns None when cell-quotient
    has a cycle (cycle DP handles those)."""
    import dwave_networkx as _dnx
    from tutte.roots import compute_cell_quotient_tree_dp

    g = Graph.from_networkx(_dnx.chimera_graph(2, 2, 4))  # Cm2: 4-cycle
    result = compute_cell_quotient_tree_dp(g, table)
    assert result is None, "tree DP must reject cycle-quotient graphs"


def test_cell_quotient_hybrid_3_K3_3_cycle(table):
    """compute_cell_quotient_hybrid produces correct full polynomial
    for the smallest non-trivial cyclic cell-quotient: 3 K_3 cells in
    a 3-cycle joined by single edges."""
    from tutte.deprecated.cell_quotient_hybrid import compute_cell_quotient_hybrid
    from tutte.synthesis.engine import SynthesisEngine

    g_nx = nx.Graph()
    for c in range(3):
        base = 3 * c
        for i in range(base, base + 3):
            for j in range(i + 1, base + 3):
                g_nx.add_edge(i, j)
    for c in range(3):
        g_nx.add_edge(3 * c, 3 * ((c + 1) % 3))
    g = Graph.from_networkx(g_nx)
    poly = compute_cell_quotient_hybrid(g, table)
    assert poly is not None, "hybrid should fire on cyclic cell-quotient"
    assert verify_spanning_trees(g, poly)
    eng = SynthesisEngine(table=table)
    T_eng = eng.synthesize(g).polynomial
    assert poly == T_eng, "hybrid polynomial must match engine"


def test_cell_quotient_hybrid_returns_none_on_tree(table):
    """compute_cell_quotient_hybrid returns None when cell-quotient is
    a tree (tree DP handles those)."""
    from tutte.tests.tree_dp_branching import \
        build_tree_graph
    from tutte.deprecated.cell_quotient_hybrid import compute_cell_quotient_hybrid

    K44 = Graph.from_networkx(nx.complete_bipartite_graph(4, 4))
    edges = [(0, 1), (1, 2)]
    anchors = {0: {1: [0, 1]}, 1: {0: [0, 1], 2: [0, 1]}, 2: {1: [0, 1]}}
    t = nx.Graph()
    t.add_nodes_from({n for e in edges for n in e})
    t.add_edges_from(edges)
    g = build_tree_graph(K44, t, anchors)
    result = compute_cell_quotient_hybrid(g, table)
    assert result is None, "hybrid must reject tree-quotient graphs"


def test_cell_quotient_hybrid_K3_M2_3_cycle(table):
    """k=2 junction in 3-cycle (originally broken until self-loop +
    contraction-guard fixes)."""
    from tutte.deprecated.cell_quotient_hybrid import compute_cell_quotient_hybrid
    from tutte.synthesis.engine import SynthesisEngine

    g_nx = nx.Graph()
    for c in range(3):
        base = 3 * c
        for i in range(base, base + 3):
            for j in range(i + 1, base + 3):
                g_nx.add_edge(i, j)
    for c in range(3):
        base_a = 3 * c
        base_b = 3 * ((c + 1) % 3)
        g_nx.add_edge(base_a, base_b)
        g_nx.add_edge(base_a + 1, base_b + 1)
    g = Graph.from_networkx(g_nx)
    poly = compute_cell_quotient_hybrid(g, table)
    assert poly is not None
    assert verify_spanning_trees(g, poly)
    eng = SynthesisEngine(table=table)
    T_eng = eng.synthesize(g).polynomial
    assert poly == T_eng, "hybrid k=2 polynomial must match engine"


def test_cell_quotient_hybrid_4_K3_M2_cycle(table):
    """4-cycle of K_3 cells joined by M_2 matchings — validates the
    chord rule on a longer symmetric cycle (cells share NO anchors
    across junctions, so Path A's C(k, j) shortcut applies). This is
    the anti-regression for the orbit-aware refactor that ensures
    symmetric cases retain their fast path."""
    from tutte.deprecated.cell_quotient_hybrid import compute_cell_quotient_hybrid
    from tutte.synthesis.engine import SynthesisEngine

    g_nx = nx.Graph()
    n_cells = 4
    for c in range(n_cells):
        base = 3 * c
        for i in range(base, base + 3):
            for j in range(i + 1, base + 3):
                g_nx.add_edge(i, j)
    for c in range(n_cells):
        nxt = (c + 1) % n_cells
        g_nx.add_edge(3 * c, 3 * nxt)
        g_nx.add_edge(3 * c + 1, 3 * nxt + 1)
    g = Graph.from_networkx(g_nx)
    poly = compute_cell_quotient_hybrid(g, table)
    assert poly is not None
    assert verify_spanning_trees(g, poly)
    eng = SynthesisEngine(table=table)
    T_eng = eng.synthesize(g).polynomial
    assert poly == T_eng, "hybrid 4-K_3 M_2 cycle polynomial must match engine"


def test_cell_quotient_hybrid_3x3_K3_grid_multicycle(table):
    """3x3 grid of K_3 cells with M_1 junctions: cell-quotient is a
    3x3 grid (4 fundamental cycles). Validates the bridge-formula fix
    for k=1 multi-cycle: T(G) = x*T(G/e) for bridge e (NOT x*T(G-e),
    which gave wrong answers because the recursive sums on
    `delete bridge` vs `contract bridge` diverge for chord-rule
    leaves even though their leaf polynomials agree)."""
    from tutte.deprecated.cell_quotient_hybrid import compute_cell_quotient_hybrid
    from tutte.synthesis.engine import SynthesisEngine

    g_nx = nx.Graph()
    cells = {}
    nid = 0
    for r in range(3):
        for c in range(3):
            cn = list(range(nid, nid + 3))
            nid += 3
            cells[(r, c)] = cn
            for i in range(3):
                for j in range(i + 1, 3):
                    g_nx.add_edge(cn[i], cn[j])
    for r in range(3):
        for c in range(3):
            if c + 1 < 3:
                g_nx.add_edge(cells[(r, c)][0], cells[(r, c + 1)][0])
            if r + 1 < 3:
                g_nx.add_edge(cells[(r, c)][1], cells[(r + 1, c)][1])
    g = Graph.from_networkx(g_nx)
    poly = compute_cell_quotient_hybrid(g, table)
    assert poly is not None
    assert verify_spanning_trees(g, poly)
    eng = SynthesisEngine(table=table)
    T_eng = eng.synthesize(g).polynomial
    assert poly == T_eng, "hybrid multi-cycle polynomial must match engine"


def test_cell_quotient_hybrid_2x2_K3_grid_M2_cross_anchors(table):
    """2x2 grid of K_3 cells with M_2 junctions where horizontal uses
    cell anchors {0, 1} and vertical uses {1, 2} — sharing anchor 1
    across junction directions. The chord rule's C(k, j) shortcut
    fails here because the two j=1 sub-cases (contract first vs
    contract second matching edge) produce non-isomorphic leaves.
    Orbit-aware enumeration (Path B) correctly handles this."""
    from tutte.deprecated.cell_quotient_hybrid import compute_cell_quotient_hybrid
    from tutte.synthesis.engine import SynthesisEngine

    g_nx = nx.Graph()
    cells = {}
    nid = 0
    for r in range(2):
        for c in range(2):
            cn = list(range(nid, nid + 3))
            nid += 3
            cells[(r, c)] = cn
            for i in range(3):
                for j in range(i + 1, 3):
                    g_nx.add_edge(cn[i], cn[j])
    # Horizontal junctions use anchors {0, 1}; vertical uses {1, 2}.
    for r in range(2):
        for c in range(2):
            if c + 1 < 2:
                for i in range(2):
                    g_nx.add_edge(cells[(r, c)][i], cells[(r, c + 1)][i])
            if r + 1 < 2:
                for i in range(2):
                    a = i + (2 if i + 2 < 3 else 0)
                    g_nx.add_edge(cells[(r, c)][a], cells[(r + 1, c)][a])
    g = Graph.from_networkx(g_nx)
    poly = compute_cell_quotient_hybrid(g, table)
    assert poly is not None
    assert verify_spanning_trees(g, poly)
    eng = SynthesisEngine(table=table)
    T_eng = eng.synthesize(g).polynomial
    assert poly == T_eng, (
        "2x2 K_3 M_2 grid (cross-junction shared anchors) must match engine"
    )


# =============================================================================
# O-R. PARALLEL SYNTHESIS
# =============================================================================

import pickle  # noqa: E402

from tutte.synthesis.parallel import (parallel_synthesize_pair,  # noqa: E402
                                      shutdown_pool)

# =============================================================================
# O. PICKLING ROUND-TRIP
# =============================================================================


class TestPickling:
    """Synthesis types must survive pickle round-trip."""

    def test_multigraph_pickle(self):
        mg = MultiGraph(
            nodes=frozenset({0, 1, 2}),
            edge_counts={(0, 1): 2, (1, 2): 1},
            loop_counts={0: 1},
        )
        mg2 = pickle.loads(pickle.dumps(mg))
        assert mg == mg2
        assert mg.nodes == mg2.nodes
        assert mg.edge_counts == mg2.edge_counts
        assert mg.loop_counts == mg2.loop_counts

    def test_tutte_polynomial_pickle(self):
        poly = TuttePolynomial.from_coefficients({(1, 0): 1, (0, 1): 1})
        poly2 = pickle.loads(pickle.dumps(poly))
        assert poly == poly2

    def test_complex_polynomial_pickle(self):
        engine = SynthesisEngine()
        result = engine.synthesize(complete_graph(5))
        poly = result.polynomial
        poly2 = pickle.loads(pickle.dumps(poly))
        assert poly == poly2
        assert poly.num_spanning_trees() == poly2.num_spanning_trees()


# =============================================================================
# P. PARALLEL CORRECTNESS
# =============================================================================


class TestParallelCorrectness:
    """Verify parallel results match sequential."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        shutdown_pool()

    def test_parallel_k5(self):
        engine = SynthesisEngine()
        g = complete_graph(5)
        mg = MultiGraph(
            nodes=g.nodes,
            edge_counts={e: 1 for e in g.edges},
            loop_counts={},
        )
        edge = next(iter(mg.edge_counts))
        u, v = edge
        new_edges = dict(mg.edge_counts)
        del new_edges[edge]
        mg_0 = MultiGraph(nodes=mg.nodes, edge_counts=new_edges, loop_counts={})
        mg_c = mg_0.merge_nodes(u, v)

        seq_poly0 = engine._synthesize_multigraph(mg_0, 10, False)
        seq_polyc = engine._synthesize_multigraph(mg_c, 10, False)
        par_poly0, par_polyc = parallel_synthesize_pair(engine, mg_0, mg_c, 10, False)

        assert par_poly0 == seq_poly0
        assert par_polyc == seq_polyc

    def test_parallel_petersen(self):
        engine = SynthesisEngine()
        g = petersen_graph()
        mg = MultiGraph(
            nodes=g.nodes,
            edge_counts={e: 1 for e in g.edges},
            loop_counts={},
        )
        edge = next(iter(mg.edge_counts))
        u, v = edge
        new_edges = dict(mg.edge_counts)
        del new_edges[edge]
        mg_0 = MultiGraph(nodes=mg.nodes, edge_counts=new_edges, loop_counts={})
        mg_c = mg_0.merge_nodes(u, v)

        seq_poly0 = engine._synthesize_multigraph(mg_0, 10, False)
        seq_polyc = engine._synthesize_multigraph(mg_c, 10, False)
        par_poly0, par_polyc = parallel_synthesize_pair(engine, mg_0, mg_c, 10, False)

        assert par_poly0 == seq_poly0
        assert par_polyc == seq_polyc


# =============================================================================
# Q. CACHE MERGING
# =============================================================================


class TestCacheMerging:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        shutdown_pool()

    def test_cache_grows_after_parallel(self):
        engine = SynthesisEngine()
        g = complete_graph(5)
        mg = MultiGraph(
            nodes=g.nodes,
            edge_counts={e: 1 for e in g.edges},
            loop_counts={},
        )
        edge = next(iter(mg.edge_counts))
        u, v = edge
        new_edges = dict(mg.edge_counts)
        del new_edges[edge]
        mg_0 = MultiGraph(nodes=mg.nodes, edge_counts=new_edges, loop_counts={})
        mg_c = mg_0.merge_nodes(u, v)

        poly0, polyc = parallel_synthesize_pair(engine, mg_0, mg_c, 10, False)
        assert poly0.num_spanning_trees() > 0
        assert polyc.num_spanning_trees() > 0

    def test_merge_worker_cache(self):
        engine = SynthesisEngine()
        sentinel_poly = TuttePolynomial.x()
        engine._multigraph_cache["sentinel_key"] = sentinel_poly

        worker_cache = {
            "new_key": TuttePolynomial.from_coefficients({(1, 0): 1, (0, 1): 1}),
            "sentinel_key": TuttePolynomial.one(),
        }
        engine._merge_worker_cache(worker_cache)

        assert engine._multigraph_cache["new_key"] == worker_cache["new_key"]
        assert engine._multigraph_cache["sentinel_key"] == sentinel_poly


class TestNestedPrevention:
    """_in_worker flag prevents nested parallel calls."""

    def test_should_parallelize_blocked_in_worker(self):
        engine = SynthesisEngine()
        engine._in_worker = True

        g = complete_graph(5)
        mg = MultiGraph(
            nodes=g.nodes,
            edge_counts={e: 1 for e in g.edges},
            loop_counts={},
        )
        mg2 = mg.merge_nodes(0, 1)

        assert not engine._should_parallelize(mg, mg2)

    def test_should_parallelize_allowed_normally(self):
        engine = SynthesisEngine()
        g = complete_graph(10)
        mg = MultiGraph(
            nodes=g.nodes,
            edge_counts={e: 1 for e in g.edges},
            loop_counts={},
        )
        mg2 = MultiGraph(
            nodes=g.nodes,
            edge_counts={e: 1 for e in g.edges},
            loop_counts={},
        )
        # 10 nodes < 12 → False
        assert not engine._should_parallelize(mg, mg2)

        g13 = complete_graph(13)
        mg_big = MultiGraph(
            nodes=g13.nodes,
            edge_counts={e: 1 for e in g13.edges},
            loop_counts={},
        )
        assert engine._should_parallelize(mg_big, mg_big)
