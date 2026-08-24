"""Tests for the Sokal-Z generalized chord-junction theorem dispatch.

See `tutte/research/cyclotomic_chord_junction_theorem.md` for the
theorem; see `tutte/roots/sokal_z_chord_junction.py` for the
implementation.

The prototype enumerates A_J ⊆ E_J directly (2^|E_J| terms) and is
gated at max_subsets = 65536, so these tests only cover |E_J| ≤ 16.
Tree-DP enumeration over H_J is future work for tractability on
larger junctions (e.g., Z(1, 2) where |E_J| = 32).
"""
from __future__ import annotations

import pytest

from tutte.graph import Graph, MultiGraph
from tutte.roots.sokal_z_chord_junction import (
    _enumerate_component_phi_terms,
    _tree_dp_component_phi_terms,
    compute_sokal_z_chord_junction,
    compute_sokal_z_chord_junction_per_component,
)
from tutte.synthesis.engine import SynthesisEngine


@pytest.fixture(scope="module")
def engine_and_synth():
    eng = SynthesisEngine()

    def synth(mg):
        return eng._synthesize_multigraph(mg)
    return eng, synth


def _direct_tutte(cell_A, cell_B, chord_edges, engine):
    """Build G_1 ⊕ G_2 + chord_edges as a MultiGraph and synthesize."""
    n_A = cell_A.node_count()
    edge_counts = {}
    loop_counts = {}

    def _add(u, v):
        if u == v:
            loop_counts[u] = loop_counts.get(u, 0) + 1
        else:
            e = (min(u, v), max(u, v))
            edge_counts[e] = edge_counts.get(e, 0) + 1

    for (u, v) in cell_A.edges:
        _add(u, v)
    for (u, v) in cell_B.edges:
        _add(u + n_A, v + n_A)
    for (a, b) in chord_edges:
        _add(a, b + n_A)
    nodes = set(cell_A.nodes) | set(v + n_A for v in cell_B.nodes)
    mg = MultiGraph(
        nodes=frozenset(nodes), edge_counts=edge_counts, loop_counts=loop_counts,
    )
    return engine._synthesize_multigraph(mg)


K2 = Graph(nodes=frozenset({0, 1}), edges=frozenset({(0, 1)}))
K3 = Graph(nodes=frozenset({0, 1, 2}),
           edges=frozenset({(0, 1), (1, 2), (0, 2)}))
K4 = Graph(nodes=frozenset({0, 1, 2, 3}),
           edges=frozenset({(0, 1), (0, 2), (0, 3),
                            (1, 2), (1, 3), (2, 3)}))
C4 = Graph(nodes=frozenset({0, 1, 2, 3}),
           edges=frozenset({(0, 1), (1, 2), (2, 3), (0, 3)}))


@pytest.mark.parametrize("name,cell_A,cell_B,chord", [
    ("K_2 ⊕ K_2, 1 chord", K2, K2, [(0, 0)]),
    ("K_2 ⊕ K_2, M_2 chord (matching)", K2, K2, [(0, 0), (1, 1)]),
    ("K_2 ⊕ K_2, 2 parallel chord", K2, K2, [(0, 0), (0, 0)]),
    ("K_2 ⊕ K_2, 3 non-matching", K2, K2, [(0, 0), (0, 1), (1, 1)]),
    ("K_2 ⊕ K_2, K_{2,2} chord", K2, K2,
        [(0, 0), (0, 1), (1, 0), (1, 1)]),
    ("K_3 ⊕ K_3, M_2 chord", K3, K3, [(0, 0), (1, 1)]),
    ("K_3 ⊕ K_3, M_3 chord", K3, K3, [(0, 0), (1, 1), (2, 2)]),
    ("K_4 ⊕ K_4, M_3 chord", K4, K4, [(0, 0), (1, 1), (2, 2)]),
    ("K_4 ⊕ K_4, M_4 chord", K4, K4, [(0, 0), (1, 1), (2, 2), (3, 3)]),
    ("K_4 ⊕ K_4, 2 parallel + 2 matching", K4, K4,
        [(0, 0), (0, 0), (1, 1), (2, 2)]),
    ("C_4 ⊕ C_4, K_{2,2} chord", C4, C4,
        [(0, 0), (0, 2), (2, 0), (2, 2)]),
])
def test_sokal_z_matches_direct(engine_and_synth, name, cell_A, cell_B, chord):
    engine, synth = engine_and_synth
    t_via_z = compute_sokal_z_chord_junction(cell_A, cell_B, chord, synth)
    t_direct = _direct_tutte(cell_A, cell_B, chord, engine)
    assert t_via_z == t_direct, (
        f"{name}: Sokal-Z {t_via_z} != direct {t_direct}"
    )


def test_sokal_z_aut_compression_preserves_polynomial(engine_and_synth):
    """Aut compression must produce identical results to brute-force.

    Tests on K_4 ⊕ K_4 with a K_{2,2} junction (Aut(H_J)>1).
    """
    engine, synth = engine_and_synth
    chord = [(0, 0), (0, 2), (2, 0), (2, 2)]
    t_aut_on = compute_sokal_z_chord_junction_per_component(
        K4, K4, chord, synth, use_aut_compression=True,
        max_phi_per_component=10000, max_phi_cross_product=10_000_000,
    )
    t_aut_off = compute_sokal_z_chord_junction_per_component(
        K4, K4, chord, synth, use_aut_compression=False,
        max_phi_per_component=10000, max_phi_cross_product=10_000_000,
    )
    assert t_aut_on is not None and t_aut_off is not None
    assert t_aut_on == t_aut_off


def test_sokal_z_per_component_handles_large_e_j(engine_and_synth):
    """Per-component path handles |E_J| > 16 when H_J components are small."""
    engine, synth = engine_and_synth
    # 20 chord edges grouped as 10 parallel at (0, 0) + 10 parallel at (1, 1).
    # H_J has 2 components, each 2 vertices / 10 multi-edges → only
    # 2 compatible φ per component (merged or split). Cross-product = 4.
    big_chord = [(i % 2, i % 2) for i in range(20)]
    t_via_z = compute_sokal_z_chord_junction(K2, K2, big_chord, synth)
    t_direct = _direct_tutte(K2, K2, big_chord, engine)
    assert t_via_z is not None
    assert t_via_z == t_direct


def test_engine_sokal_z_dispatch_helper(engine_and_synth):
    """Verify engine._try_sokal_z_chord_junction returns the right polynomial
    on a 2-cell graph with a non-matching K_{2,2} junction.
    """
    from tutte.graphs.covering import InterCellInfo
    engine, _ = engine_and_synth
    edges = []
    for i in range(4):
        edges.append((i, (i + 1) % 4))
    for i in range(4):
        edges.append((4 + i, 4 + (i + 1) % 4))
    chord = [(0, 4), (0, 6), (2, 4), (2, 6)]
    edges.extend(chord)
    g = Graph(nodes=frozenset(range(8)), edges=frozenset(edges))
    partition = [set(range(4)), set(range(4, 8))]
    inter = InterCellInfo(
        edges=chord, is_regular=True, edges_per_pair=4,
        cell_adjacencies=[(0, 1)],
    )
    cells = []
    cell_C4 = Graph(nodes=frozenset(range(4)),
                    edges=frozenset({(0, 1), (1, 2), (2, 3), (0, 3)}))
    chord_pairs = [(0, 0), (0, 2), (2, 0), (2, 2)]
    expected = compute_sokal_z_chord_junction(
        cell_C4, cell_C4, chord_pairs, engine._synthesize_multigraph,
    )
    result = engine._try_sokal_z_chord_junction(g, cells, partition, inter)
    assert result is not None
    assert result == expected


@pytest.mark.parametrize("name,nodes,edges", [
    ("3-edge triangle K_3", [0, 1, 2], [(0, 1), (1, 2), (0, 2)]),
    ("K_{2,3} bipartite", [0, 1, 2, 3, 4],
        [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]),
    ("K_{2,4} bipartite", [0, 1, 2, 3, 4, 5],
        [(0, 2), (0, 3), (0, 4), (0, 5),
         (1, 2), (1, 3), (1, 4), (1, 5)]),
    ("K_4 dense", [0, 1, 2, 3],
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    ("P_5 path", [0, 1, 2, 3, 4],
        [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ("multi-edge component", [0, 1, 2],
        [(0, 1), (0, 1), (1, 2), (1, 2), (0, 2)]),
    ("isolated vertex + edge", [0, 1, 2], [(0, 1)]),
    ("two parallel only", [0, 1], [(0, 1), (0, 1), (0, 1)]),
])
def test_tree_dp_matches_brute_force(name, nodes, edges):
    """Tree-DP and brute-force enumeration produce identical {φ → coef} dicts."""
    brute = _enumerate_component_phi_terms(edges, nodes)
    tree = _tree_dp_component_phi_terms(edges, nodes)
    assert brute == tree, (
        f"{name}: tree_dp={tree} != brute={brute}"
    )


def test_tree_dp_handles_dense_junction(engine_and_synth):
    """End-to-end: C_4 ⊕ C_4 with 14-chord junction matches direct synthesis.

    The 14-edge component routes through tree-DP (above threshold=13);
    output must equal direct construction.
    """
    engine, synth = engine_and_synth
    # 14 chord edges all in one H_J component: K_{2,7}-shaped
    chord = [(a, b) for a in range(2) for b in range(4)]  # K_{2,4} = 8 edges
    chord += [(0, 0), (0, 1), (1, 2), (1, 3), (0, 2), (1, 0)]  # +6 more = 14
    t_via_z = compute_sokal_z_chord_junction(C4, C4, chord, synth)
    t_direct = _direct_tutte(C4, C4, chord, engine)
    assert t_via_z is not None
    assert t_via_z == t_direct


def test_per_component_with_aut_matches_direct_on_k4_k44_k4(engine_and_synth):
    """Per-component path with Aut compression must match direct synthesis.

    Regression: K_4 cells with full K_{4,4} chord junction = K_8. Aut(K_8)
    has order 40320, but only the 1152-element cell-preserving subgroup
    is a valid Aut for the chord-junction Z. Without cell-coloring,
    over-aggregation produces wrong polynomials. This test guards both
    brute-force and tree-DP per-component paths against that regression.
    """
    engine, synth = engine_and_synth
    chord = [(a, b) for a in range(4) for b in range(4)]  # K_{4,4}, 16 edges
    direct = _direct_tutte(K4, K4, chord, engine)
    for tree_dp_thresh, label in [(999, "brute"), (13, "tree-DP")]:
        result = compute_sokal_z_chord_junction_per_component(
            K4, K4, chord, synth,
            max_phi_per_component=5000,
            max_phi_cross_product=10_000_000,
            use_aut_compression=True,
            tree_dp_edge_threshold=tree_dp_thresh,
        )
        assert result is not None, f"{label} returned None"
        assert result == direct, (
            f"{label} polynomial differs from direct: "
            f"#ST direct={direct.evaluate(1, 1)}, "
            f"#ST {label}={result.evaluate(1, 1)}"
        )


def test_sokal_z_returns_none_when_per_component_intractable(engine_and_synth):
    """Per-component gate triggers when a single H_J component is too dense."""
    _, synth = engine_and_synth
    # K_{2, 2} cell with very dense bipartite junction: each anchor on
    # each side gets ~5 chord edges, giving a large connected H_J
    # component well over max_phi_per_component=200.
    # K_{2, 2} cell (= C_4) has 4 nodes per cell, all anchors used.
    cell = Graph(nodes=frozenset({0, 1, 2, 3}),
                 edges=frozenset({(0, 2), (0, 3), (1, 2), (1, 3)}))
    # K_{4, 4} bipartite junction (every A-anchor connected to every B-anchor)
    big_chord = [(a, b) for a in range(4) for b in range(4)]
    # 16-edge junction in one component — far more than 200 compatible φ
    result = compute_sokal_z_chord_junction(
        cell, cell, big_chord, synth,
        max_subsets=4,  # disable direct path
        max_phi_per_component=10,  # block per-component path too
    )
    assert result is None
