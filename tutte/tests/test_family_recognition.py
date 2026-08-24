"""Test suite for the family recognition module.

Tests recognize_family() on all supported graph families, verifying:
  1. Recognition succeeds (returns a polynomial, not None)
  2. Correctness via Kirchhoff: T(1,1) = spanning tree count
  3. Correctness via T(2,2) = 2^|E|

Families tested:
  - Tier 1 (closed-form): tree/path, cycle, wheel, fan, pan, sunlet, helm, book
  - Tier 2 (recurrence): ladder, gear, prism, Möbius, grid

Usage:
    pytest tests/tutte/test_family_recognition.py -v
"""

import networkx as nx
import pytest

from tutte.graph import Graph
from tutte.family_recognition import recognize_family
from tutte.validation import _exact_spanning_tree_count, _exact_num_spanning_trees
from tutte.tests.test_benchmark_family_recognition import (
    _build_gear, _build_helm, _build_book, _build_pan, _build_sunlet, _build_mobius,
)


# ===========================================================================
# Verification helper
# ===========================================================================

def _verify_recognition(G_nx: nx.Graph, family_name: str, expected_recognized: bool = True):
    """Recognize the graph and verify the polynomial is correct."""
    graph = Graph.from_networkx(G_nx)
    poly = recognize_family(graph)

    if not expected_recognized:
        assert poly is None, f"{family_name}: should NOT be recognized but got polynomial"
        return

    assert poly is not None, f"{family_name}: not recognized (returned None)"

    # Kirchhoff: T(1,1) = spanning tree count (exact integer arithmetic)
    trees = _exact_num_spanning_trees(poly)
    kirchhoff = _exact_spanning_tree_count(graph)
    assert trees == kirchhoff, (
        f"{family_name}: T(1,1)={trees} != Kirchhoff={kirchhoff}"
    )

    # T(2,2) = 2^|E|
    e = G_nx.number_of_edges()
    t22 = poly.evaluate(2, 2)
    assert t22 == 2 ** e, (
        f"{family_name}: T(2,2)={t22} != 2^{e}={2 ** e}"
    )


# ===========================================================================
# Tier 1 — Closed-form families
# ===========================================================================

class TestTier1ClosedForm:
    """Families with O(1) polynomial computation after detection."""

    @pytest.mark.parametrize("n", [3, 4, 8, 39])
    def test_path(self, n):
        """Path P_n (tree): T = x^(n-1)."""
        _verify_recognition(nx.path_graph(n), f"P_{n}")

    @pytest.mark.parametrize("n", [3, 4, 8, 39])
    def test_cycle(self, n):
        """Cycle C_n: T = x^(n-1) + ... + x + y."""
        _verify_recognition(nx.cycle_graph(n), f"C_{n}")

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7, 19])
    def test_wheel(self, k):
        """Wheel W_k (k+1 vertices)."""
        _verify_recognition(nx.wheel_graph(k + 1), f"W_{k}")

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 19])
    def test_fan(self, k):
        """Fan F_k (k+1 vertices)."""
        G = nx.Graph()
        # Apex 0 connected to path 1..k
        for i in range(1, k + 1):
            G.add_edge(0, i)
        for i in range(1, k):
            G.add_edge(i, i + 1)
        _verify_recognition(G, f"F_{k}")

    @pytest.mark.parametrize("cycle_size", [4, 5, 15, 29])
    def test_pan(self, cycle_size):
        """Pan: C_n with one pendant."""
        _verify_recognition(_build_pan(cycle_size), f"Pan_{cycle_size}")

    @pytest.mark.parametrize("k", [3, 4, 10, 19])
    def test_sunlet(self, k):
        """Sunlet: C_k with pendant at each vertex."""
        _verify_recognition(_build_sunlet(k), f"Sunlet_{k}")

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7, 14])
    def test_helm(self, k):
        """Helm: W_k with pendant at each rim vertex."""
        _verify_recognition(_build_helm(k), f"Helm_{k}")

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 19])
    def test_book(self, k):
        """Book: k triangles sharing one edge."""
        _verify_recognition(_build_book(k), f"Book_{k}")


# ===========================================================================
# Tier 2 — Recurrence-based families
# ===========================================================================

class TestTier2Recurrence:
    """Families with O(k) polynomial computation after detection."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5, 19])
    def test_ladder(self, k):
        """Ladder P_k x P_2."""
        _verify_recognition(nx.ladder_graph(k), f"Ladder_{k}")

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7, 14])
    def test_gear(self, k):
        """Gear: wheel with subdivided rim edges."""
        _verify_recognition(_build_gear(k), f"Gear_{k}")

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7, 8, 9, 14])
    def test_prism(self, k):
        """Prism (circular ladder) C_k x K_2."""
        _verify_recognition(nx.circular_ladder_graph(k), f"Prism_{k}")

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7, 8, 9, 14])
    def test_mobius(self, k):
        """Möbius ladder: 2k-cycle + k rungs connecting v_i to v_{i+k}."""
        _verify_recognition(_build_mobius(k), f"Mobius_{k}")

    @pytest.mark.parametrize("m,n", [
        (1, 2), (1, 3), (1, 9), (2, 2), (2, 3), (2, 4), (2, 9),
    ])
    def test_grid(self, m, n):
        """Grid P_m x P_n."""
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(m, n))
        _verify_recognition(G, f"Grid_{m}x{n}")


# ===========================================================================
# Negative cases — should NOT be recognized
# ===========================================================================

class TestNotRecognized:
    """Graphs that should NOT be recognized as any family."""

    def test_petersen(self):
        """Petersen graph: 3-regular but not prism/Möbius."""
        _verify_recognition(nx.petersen_graph(), "Petersen", expected_recognized=False)

    def test_complete_k5(self):
        """K_5: not a recognized family."""
        _verify_recognition(nx.complete_graph(5), "K_5", expected_recognized=False)

    def test_complete_bipartite_k33(self):
        """K_{3,3}: this is actually M_3 (Möbius ladder), so it SHOULD be recognized."""
        # K_{3,3} is the Möbius ladder M_3 — verify it IS recognized
        _verify_recognition(
            nx.complete_bipartite_graph(3, 3), "K_{3,3}", expected_recognized=True
        )

    def test_k22_m2_chain_not_misidentified_as_grid(self):
        """K_{2,2}+M_2 chain of 2 cells (8v, 10e, deg counts {2:4, 3:4},
        bipartite) shares (n, m, degrees, bipartiteness) with the 2×4
        grid (= ladder L_4) but is structurally distinct.

        Regression for the May 14, 2026 detect_grid_dims false-positive
        bug: pre-fix, recognize_family returned T(L_4) instead of None,
        causing engine.synthesize to silently produce wrong polynomials
        for chain-of-K_{2,2}-cells inputs.
        """
        # Build the chain explicitly
        nxG = nx.Graph()
        edges = [
            (0, 2), (0, 3), (1, 2), (1, 3),  # cell 0: K_{2,2}
            (4, 6), (4, 7), (5, 6), (5, 7),  # cell 1: K_{2,2}
            (2, 4), (3, 5),                   # M_2 junction
        ]
        nxG.add_edges_from(edges)
        # Should NOT be recognized as a grid/ladder/etc.
        _verify_recognition(nxG, "K22_M2_chain_2cells", expected_recognized=False)

    def test_complete_bipartite_k44(self):
        """K_{4,4}: 4-regular, not a recognized family."""
        _verify_recognition(
            nx.complete_bipartite_graph(4, 4), "K_{4,4}", expected_recognized=False
        )
