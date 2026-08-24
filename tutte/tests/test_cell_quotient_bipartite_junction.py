"""Regression tests for cell_quotient_bipartite_junction DP.

Verifies the generalized junction-detection + chain-DP path:

1. Detection: hierarchical decomposition with non-matching bipartite
   inter-cell structure (asymmetric anchor degrees, possibly disconnected
   junction subgraph).
2. Correctness: result matches engine.synthesize bit-for-bit.
3. Backward compat: k-matching graphs (the previously-supported case)
   still work via the new path.
4. Guard: per-cell anchor union > max_cell_boundary returns None
   rather than hanging.
"""
from __future__ import annotations

import networkx as nx
import pytest

from tutte.graph import Graph
from tutte.polynomial import TuttePolynomial
from tutte.graphs.covering import (BipartiteJunction,
                                    detect_bipartite_junction_topology,
                                    try_hierarchical_partition)
from tutte.roots.cell_quotient_bipartite_junction import (
    build_bipartite_junction_spec,
    compute_bipartite_junction_per_component_dp,
    compute_cell_quotient_bipartite_junction_dp,
)
from tutte.roots.aut_orbit import (aut_compress_t_rooted, canonical_partition,
                                   clear_canonical_cache, compute_cell_aut)
from tutte.roots.rooted_tutte import (_T_ROOTED_ORBIT_CACHE, aut_orbit_size,
                                       clear_t_rooted_cache,
                                       load_default_rooted_lookup,
                                       t_rooted_bruteforce, t_rooted_orbit_compressed,
                                       t_rooted_outer_product, t_rooted_smart)


@pytest.fixture(autouse=True)
def _isolate_t_rooted_cache():
    """Reset T_rooted cache to default after each test to avoid leaking
    synthetic partition dicts into later tests (test_partition_c_r18 etc.)."""
    yield
    clear_t_rooted_cache()
    load_default_rooted_lookup()
from tutte.synthesis.engine import SynthesisEngine


# ---------------------------------------------------------------
# t_rooted_smart / outer_product primitives
# ---------------------------------------------------------------

def test_t_rooted_smart_matches_bruteforce_on_disjoint_k3_k3():
    """t_rooted_smart on K_3 ⊔ K_3 with empty boundary equals bruteforce."""
    g = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    G = Graph.from_networkx(g)
    smart = t_rooted_smart(G, [])
    brute = t_rooted_bruteforce(G, [])
    assert smart == brute


def test_t_rooted_smart_matches_bruteforce_on_disjoint_k3_k3_with_boundary():
    """6-vertex boundary on K_3 ⊔ K_3."""
    g = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    G = Graph.from_networkx(g)
    smart = t_rooted_smart(G, list(range(6)))
    brute = t_rooted_bruteforce(G, list(range(6)))
    assert smart == brute


def test_t_rooted_smart_falls_back_for_connected_k4():
    """Connected K_4: smart returns the bruteforce result (no decomposition)."""
    G = Graph.from_networkx(nx.complete_graph(4))
    smart = t_rooted_smart(G, [0, 1])
    brute = t_rooted_bruteforce(G, [0, 1])
    assert smart == brute


def test_t_rooted_orbit_compressed_matches_post_hoc_on_k4():
    """Inline orbit compression matches post-hoc aut_compress_t_rooted (K_4)."""
    _T_ROOTED_ORBIT_CACHE.clear(); clear_canonical_cache()
    G = Graph.from_networkx(nx.complete_graph(4))
    boundary = [0, 1, 2, 3]
    T_brute = t_rooted_bruteforce(G, boundary)
    aut = compute_cell_aut(G)
    orbit_post, _ = aut_compress_t_rooted(T_brute, aut)
    orbit_inline, aut_inline = t_rooted_orbit_compressed(G, boundary)
    # Same canonical keys + same per-orbit values.
    assert set(orbit_post.keys()) == set(orbit_inline.keys())
    for canon in orbit_post:
        assert orbit_post[canon] == orbit_inline[canon]


def test_t_rooted_orbit_compressed_sum_equals_brute_total_on_k_bipartite():
    """Sum over orbits (weighted by orbit_size) equals brute total (K_{2,3})."""
    _T_ROOTED_ORBIT_CACHE.clear(); clear_canonical_cache()
    G = Graph.from_networkx(nx.complete_bipartite_graph(2, 3))
    boundary = list(range(5))
    T_brute = t_rooted_bruteforce(G, boundary)
    total_brute = sum((v for v in T_brute.values()), TuttePolynomial.zero())
    orbit_T, aut = t_rooted_orbit_compressed(G, boundary)
    total_orbit = TuttePolynomial.zero()
    for canon, val in orbit_T.items():
        size = aut_orbit_size(canon, aut)
        # Sum val + val + ... (size times) — explicit because TuttePolynomial
        # doesn't define scalar mul.
        contrib = TuttePolynomial.zero()
        for _ in range(size):
            contrib = contrib + val
        total_orbit = total_orbit + contrib
    assert total_brute == total_orbit


def test_t_rooted_orbit_compressed_cache_hits_second_call():
    """Cache prevents re-computation on second call with same (graph, boundary)."""
    _T_ROOTED_ORBIT_CACHE.clear()
    G = Graph.from_networkx(nx.complete_graph(4))
    orbit1, aut1 = t_rooted_orbit_compressed(G, [0, 1, 2, 3])
    orbit2, aut2 = t_rooted_orbit_compressed(G, [0, 1, 2, 3])
    assert orbit1 is orbit2
    assert aut1 is aut2


def test_t_rooted_outer_product_associative_on_3_components():
    """Reduce A ⊗ B ⊗ C by associative outer-product."""
    g = nx.disjoint_union(nx.complete_graph(3),
                          nx.disjoint_union(nx.cycle_graph(3),
                                            nx.path_graph(3)))
    G = Graph.from_networkx(g)
    smart = t_rooted_smart(G, list(range(g.number_of_nodes())))
    brute = t_rooted_bruteforce(G, list(range(g.number_of_nodes())))
    assert smart == brute


# ---------------------------------------------------------------
# Generalized bipartite-junction detection
# ---------------------------------------------------------------

def test_detect_bipartite_junction_on_2_k4_non_matching():
    """Two K_4 cells with a 3-edge non-matching inter-cell junction."""
    g = nx.complete_graph(4)
    g.add_edges_from([(u + 4, v + 4) for u, v in nx.complete_graph(4).edges()])
    # Anchor 4 has degree 2 in inter-cell edges: NOT a k-matching.
    g.add_edges_from([(0, 4), (1, 4), (2, 5)])
    G = Graph.from_networkx(g)
    engine = SynthesisEngine()
    res = try_hierarchical_partition(G, engine.table)
    assert res is not None
    _cell_entry, partition, inter_info = res
    bjs = detect_bipartite_junction_topology(G, partition, list(inter_info.edges))
    assert bjs is not None and len(bjs) == 1
    assert bjs[0].edge_count == 3


# ---------------------------------------------------------------
# End-to-end DP correctness — bipartite_junction matches engine
# ---------------------------------------------------------------

def test_bipartite_junction_dp_2_k4_non_matching_matches_engine():
    g = nx.complete_graph(4)
    g.add_edges_from([(u + 4, v + 4) for u, v in nx.complete_graph(4).edges()])
    g.add_edges_from([(0, 4), (1, 4), (2, 5)])
    G = Graph.from_networkx(g)
    engine = SynthesisEngine()
    expected = engine.synthesize(G).polynomial
    actual = compute_cell_quotient_bipartite_junction_dp(G, engine.table)
    assert actual is not None
    assert actual == expected


def test_bipartite_junction_dp_3_k3_chain_matches_engine():
    """Backward compat: k-matching graphs (special case of bipartite junction)."""
    g = nx.Graph()
    for cell in range(3):
        base = 3 * cell
        for u in range(3):
            for v in range(u + 1, 3):
                g.add_edge(base + u, base + v)
    g.add_edge(0, 3)
    g.add_edge(3, 6)
    G = Graph.from_networkx(g)
    engine = SynthesisEngine()
    expected = engine.synthesize(G).polynomial
    actual = compute_cell_quotient_bipartite_junction_dp(G, engine.table)
    assert actual is not None
    assert actual == expected


def test_bipartite_junction_guard_skips_large_cell_boundary():
    """Z(1, 2) has 12-vert per-cell anchor boundary; default guard returns None."""
    try:
        import dwave_networkx as dnx
    except ImportError:
        pytest.skip("dwave_networkx unavailable")
    z12 = Graph.from_networkx(dnx.zephyr_graph(1, 2))
    engine = SynthesisEngine()
    # Default guard (max_cell_boundary=8) should reject Z(1, 2)'s 12-anchor cell.
    res = compute_cell_quotient_bipartite_junction_dp(z12, engine.table)
    assert res is None


# ---------------------------------------------------------------
# Per-component bipartite-junction DP (Round B, May 17 2026)
# ---------------------------------------------------------------

def test_per_component_dp_2_k3_with_m2():
    """Two K_3 cells with a 2-matching junction (2 disjoint edges, 2 components)."""
    g = nx.Graph()
    for u, v in [(0, 1), (1, 2), (0, 2)]: g.add_edge(u, v)
    for u, v in [(3, 4), (4, 5), (3, 5)]: g.add_edge(u, v)
    g.add_edge(0, 3)
    g.add_edge(2, 5)
    G = Graph.from_networkx(g)
    engine = SynthesisEngine()
    expected = engine.synthesize(G).polynomial
    actual = compute_bipartite_junction_per_component_dp(
        G, engine.table, max_cell_boundary=8,
    )
    assert actual is not None
    assert actual == expected


def test_per_component_dp_2_k3_with_m3():
    """Two K_3 cells with M_3 matching (3 disjoint edges, 3 components)."""
    g = nx.Graph()
    for u, v in [(0, 1), (1, 2), (0, 2)]: g.add_edge(u, v)
    for u, v in [(3, 4), (4, 5), (3, 5)]: g.add_edge(u, v)
    g.add_edges_from([(0, 3), (1, 4), (2, 5)])
    G = Graph.from_networkx(g)
    engine = SynthesisEngine()
    expected = engine.synthesize(G).polynomial
    actual = compute_bipartite_junction_per_component_dp(
        G, engine.table, max_cell_boundary=8,
    )
    assert actual is not None
    assert actual == expected


def test_per_component_dp_2_k4_with_m2():
    """Two K_4 cells with 2-matching junction (delegates to single-component path)."""
    g = nx.complete_graph(4)
    g.add_edges_from([(u + 4, v + 4) for u, v in nx.complete_graph(4).edges()])
    g.add_edges_from([(0, 4), (1, 5)])
    G = Graph.from_networkx(g)
    engine = SynthesisEngine()
    expected = engine.synthesize(G).polynomial
    actual = compute_bipartite_junction_per_component_dp(
        G, engine.table, max_cell_boundary=8,
    )
    assert actual is not None
    assert actual == expected


def test_per_component_dp_single_component_delegates():
    """Single-component junction (Cm_2-style M_4) should delegate to the
    standard `compute_tree_dp_simple` path and match `compute_cell_quotient_bipartite_junction_dp`."""
    g = nx.complete_graph(4)
    g.add_edges_from([(u + 4, v + 4) for u, v in nx.complete_graph(4).edges()])
    g.add_edges_from([(0, 4), (1, 4), (2, 5)])  # 1 connected component
    G = Graph.from_networkx(g)
    engine = SynthesisEngine()
    expected = compute_cell_quotient_bipartite_junction_dp(G, engine.table)
    actual = compute_bipartite_junction_per_component_dp(
        G, engine.table, max_cell_boundary=8,
    )
    assert actual is not None
    assert expected is not None
    assert actual == expected
