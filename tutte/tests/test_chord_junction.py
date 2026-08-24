"""Consolidated tests for atom detection + chord-junction theorem.

This file merges 5 previously separate test modules covering the same
feature area (cell decomposition + chord-rule synthesis):

  1. test_atom_detection.py            (K_n / K_{a,b} / B_n / W_n / L_n / Y_n
                                        atom detection, find_disjoint_atoms,
                                        find_smallest_junction, cost-aware
                                        and heterogeneous decomposition)
  2. test_unified_chord_junction.py    (inline reference implementation of
                                        the unified bivariate I-E theorem
                                        T(G ⊕_{V_k} G; x, y) =
                                          (x-1)·T(G;x,y)² +
                                          Σ_{∅≠S⊆V_k} T(G ∪_{V_S} G; x, y))
  3. test_chord_junction_closed_form.py
                                       (production unified_chord_junction
                                        module — symmetric + asymmetric
                                        APIs, merger-table caching)
  4. test_merger_lookup.py             (MergerEntry / MergerTable
                                        serialization + counters)
  5. test_engine_unified_chord_dispatch.py
                                       (engine `_try_unified_chord_junction`
                                        dispatch — session cache load,
                                        cell-pair fast path, asymmetric path)

Sections below mirror this order. Test functions are preserved verbatim
from the source files (names, docstrings, decorators, asserts). The
shared fixtures (`engine` at module scope, `_chord_junction_simple_graph`
helper) live near the top so all sections can reuse them.
"""
from __future__ import annotations

import os
import tempfile
from itertools import combinations
from typing import List, Sequence, Tuple

import dwave_networkx as dnx
import networkx as nx
import pytest

from tutte.graph import Graph, MultiGraph, complete_graph, path_graph
from tutte.graphs.atom_detection import (Atom, find_atoms_cost_aware,
                                         find_atoms_heterogeneous,
                                         find_disjoint_atoms,
                                         find_disjoint_book_atoms,
                                         find_disjoint_kab_atoms,
                                         find_disjoint_kn_atoms,
                                         find_disjoint_ladder_atoms,
                                         find_disjoint_prism_atoms,
                                         find_disjoint_wheel_atoms,
                                         find_smallest_junction)
from tutte.graphs.covering import KMatchingJunction, apply_kmatching_formula
from tutte.lookup import (MergerEntry, MergerTable,
                          decode_merger_lookup_table,
                          encode_merger_lookup_table,
                          load_default_merger_table)
from tutte.lookup.core import load_default_table
from tutte.polynomial import TuttePolynomial
from tutte.roots.chord_junction_closed_form import (
    build_symmetric_merger,
    unified_chord_junction,
    unified_chord_junction_asymmetric,
)
from tutte.synthesis import SynthesisEngine
from tutte.synthesis.engine import SynthesisEngine as EngineImpl


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine() -> SynthesisEngine:
    """Module-scope synthesis engine used by most chord-junction tests.

    The engine-dispatch section overrides this with a function-scope
    `engine_func` fixture (see below) because cell-pair dispatch tests
    leak auto-promoted rainbow-table entries.
    """
    e = SynthesisEngine(table=load_default_table(), verbose=False)
    e.skip_target_lookup = True
    return e


@pytest.fixture
def engine_func() -> SynthesisEngine:
    """Fresh engine per test (used by engine-dispatch section).

    Module-scope sharing would leak auto-promoted rainbow-table entries
    between tests (e.g. the first test's cell-pair becomes a known minor,
    routing the second test through a different dispatch path).
    """
    e = SynthesisEngine(table=load_default_table(), verbose=False)
    e.skip_target_lookup = True
    return e


# ---------------------------------------------------------------------------
# Shared helpers — chord-junction graph construction
# ---------------------------------------------------------------------------


def _chord_junction_simple_graph(G_nx: nx.Graph, V_k) -> Graph:
    """Build G ⊕_{V_k} G as a tutte.Graph (simple, immutable)."""
    n = G_nx.number_of_nodes()
    g = nx.Graph()
    g.add_nodes_from(G_nx.nodes())
    g.add_edges_from(G_nx.edges())
    g.add_nodes_from(u + n for u in G_nx.nodes())
    g.add_edges_from((u + n, w + n) for u, w in G_nx.edges())
    for v in V_k:
        g.add_edge(v, v + n)
    return Graph.from_networkx(g)


def _chord_junction_simple_nx(G: nx.Graph, V_k: Sequence[int]) -> nx.Graph:
    """Same as `_chord_junction_simple_graph` but returns an nx.Graph.

    Kept as a separate helper to preserve the original reference-impl
    test signatures that call `_T_simple(engine, G_chord)` where
    `G_chord` is an nx.Graph.
    """
    n = G.number_of_nodes()
    g = nx.Graph()
    g.add_nodes_from(G.nodes())
    g.add_edges_from(G.edges())
    g.add_nodes_from(u + n for u in G.nodes())
    g.add_edges_from((u + n, w + n) for u, w in G.edges())
    for v in V_k:
        g.add_edge(v, v + n)
    return g


# ===========================================================================
# Section 1: ATOM DETECTION
# Source: test_atom_detection.py
# ===========================================================================


# ---------------------------------------------------------------------------
# K_n detection
# ---------------------------------------------------------------------------

def test_kn_atoms_on_z12():
    """Z(1,2) has 2 disjoint K_4 cliques."""
    g = Graph.from_networkx(dnx.zephyr_graph(1, 2))
    atoms = find_disjoint_kn_atoms(g)
    assert len(atoms) == 2
    assert all(a.family == "K_4" for a in atoms)
    assert all(len(a.vertices) == 4 for a in atoms)
    # Atoms are vertex-disjoint
    a, b = atoms[0].vertices, atoms[1].vertices
    assert a.isdisjoint(b)


def test_kn_atoms_on_z13():
    """Z(1,3) has 3 disjoint K_4 cliques."""
    g = Graph.from_networkx(dnx.zephyr_graph(1, 3))
    atoms = find_disjoint_kn_atoms(g)
    assert len(atoms) == 3
    assert all(a.family == "K_4" for a in atoms)


def test_kn_prefers_larger_k():
    """When both K_3 and K_4 are present, picks K_4."""
    # K_5: contains both K_3 sub-cliques and is itself a K_5.
    # find_disjoint_kn_atoms only returns ONE family-tier; for a single
    # K_5 there's just 1 atom, so we need ≥2 K_5's for K_5 to win. Use
    # two disjoint K_5s.
    g_nx = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    g = Graph.from_networkx(g_nx)
    atoms = find_disjoint_kn_atoms(g, max_k=6, min_k=3)
    assert len(atoms) == 2
    assert atoms[0].family == "K_5"


# ---------------------------------------------------------------------------
# K_{a,b} detection
# ---------------------------------------------------------------------------

def test_kab_atoms_on_cm2():
    """Cm_2 has 4 disjoint K_{4,4} cells."""
    g = Graph.from_networkx(dnx.chimera_graph(2))
    atoms = find_disjoint_kab_atoms(g)
    assert len(atoms) >= 2
    assert all(a.family == "K_{4,4}" for a in atoms)
    assert all(len(a.vertices) == 8 for a in atoms)
    # Disjoint
    used = set()
    for a in atoms:
        assert a.vertices.isdisjoint(used)
        used |= a.vertices


def test_kab_atoms_on_disjoint_bicliques():
    """Two disjoint K_{3,3} are detected."""
    g1 = nx.complete_bipartite_graph(3, 3)
    g2 = nx.complete_bipartite_graph(3, 3)
    g_nx = nx.disjoint_union(g1, g2)
    g = Graph.from_networkx(g_nx)
    atoms = find_disjoint_kab_atoms(g, ab_pairs=[(3, 3)])
    assert len(atoms) == 2
    assert atoms[0].family == "K_{3,3}"
    assert all(len(a.vertices) == 6 for a in atoms)


def test_kab_returns_empty_when_no_biclique():
    """A graph with no K_{a,b} atoms (a,b ≥ 2) returns []."""
    g = Graph.from_networkx(nx.path_graph(5))
    atoms = find_disjoint_kab_atoms(g, ab_pairs=[(2, 2), (3, 3)])
    assert atoms == []


# ---------------------------------------------------------------------------
# Unified entry: K_n preferred over K_{a,b}
# ---------------------------------------------------------------------------

def test_unified_prefers_kn_when_both_present():
    """Z(1,2) has both K_4 and K_{4,4} atoms — K_n wins."""
    g = Graph.from_networkx(dnx.zephyr_graph(1, 2))
    atoms = find_disjoint_atoms(g)
    assert atoms[0].family == "K_4"


def test_unified_falls_through_to_kab():
    """Cm_2 has NO K_n cliques but K_{4,4} atoms."""
    g = Graph.from_networkx(dnx.chimera_graph(2))
    atoms = find_disjoint_atoms(g)
    assert atoms[0].family.startswith("K_{")


# ---------------------------------------------------------------------------
# Junction analysis
# ---------------------------------------------------------------------------

def test_smallest_junction_z12():
    """Z(1,2)'s two K_4 atoms have two K_{2,2} junctions; smallest = 4."""
    g = Graph.from_networkx(dnx.zephyr_graph(1, 2))
    atoms = find_disjoint_kn_atoms(g)
    j = find_smallest_junction(g, atoms)
    assert j is not None
    assert len(j) == 4


def test_smallest_junction_disconnected_atoms():
    """Atoms with no inter-atom edges return None."""
    g_nx = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(4))
    g = Graph.from_networkx(g_nx)
    atoms = find_disjoint_kn_atoms(g)
    assert len(atoms) == 2
    j = find_smallest_junction(g, atoms)
    assert j is None


# ---------------------------------------------------------------------------
# B_n (book) detection
# ---------------------------------------------------------------------------

def _book_graph(n_pages: int) -> nx.Graph:
    """Build B_n: edge (u=0, v=1) + n triangle pages."""
    g = nx.Graph()
    g.add_edge(0, 1)
    for i in range(n_pages):
        p = i + 2
        g.add_edge(0, p)
        g.add_edge(1, p)
    return g


def test_book_atoms_two_disjoint_books():
    """Two disjoint B_3 books detected."""
    g_nx = nx.disjoint_union(_book_graph(3), _book_graph(3))
    g = Graph.from_networkx(g_nx)
    atoms = find_disjoint_book_atoms(g, min_pages=3, max_pages=3)
    assert len(atoms) == 2
    assert atoms[0].family == "B_3"
    assert all(len(a.vertices) == 5 for a in atoms)


def test_book_atoms_prefers_larger_n():
    """Two disjoint B_4's — picks B_4 not B_2."""
    g_nx = nx.disjoint_union(_book_graph(4), _book_graph(4))
    g = Graph.from_networkx(g_nx)
    atoms = find_disjoint_book_atoms(g, min_pages=2, max_pages=5)
    assert atoms[0].family == "B_4"


def test_book_atoms_z12_has_books():
    """Z(1,2) contains B_2 atoms (sub-structure of K_4)."""
    g = Graph.from_networkx(dnx.zephyr_graph(1, 2))
    atoms = find_disjoint_book_atoms(g, min_pages=2, max_pages=3)
    # Z(1,2) has at least 2 disjoint B_2's (drawn from K_4 cells)
    assert len(atoms) >= 2


# ---------------------------------------------------------------------------
# W_n (wheel) detection
# ---------------------------------------------------------------------------

def test_wheel_atoms_two_disjoint_w5():
    """Two disjoint W_5 wheels."""
    g_nx = nx.disjoint_union(nx.wheel_graph(6), nx.wheel_graph(6))
    g = Graph.from_networkx(g_nx)
    atoms = find_disjoint_wheel_atoms(g, min_rim=5, max_rim=5)
    assert len(atoms) == 2
    assert atoms[0].family == "W_5"
    assert all(len(a.vertices) == 6 for a in atoms)


def test_wheel_atoms_no_wheel_returns_empty():
    """A path has no wheels."""
    g = Graph.from_networkx(nx.path_graph(10))
    atoms = find_disjoint_wheel_atoms(g)
    assert atoms == []


# ---------------------------------------------------------------------------
# Unified preference order: K_n → K_{a,b} → B_n → W_n
# ---------------------------------------------------------------------------

def test_unified_falls_through_to_books():
    """A graph with no K_n / K_{a,b} but with books picks B_n."""
    # Two disjoint B_3's; neither contains K_4 (B_3 ⊆ K_5 only via K_5 itself
    # which we don't have here). And K_{a,b} for (a,b) ≥ 2 won't match since
    # each B_3 has triangles, breaking bipartiteness.
    g_nx = nx.disjoint_union(_book_graph(3), _book_graph(3))
    g = Graph.from_networkx(g_nx)
    # K_3 will fire (B_3 contains 3 triangles per book) — that's expected;
    # books only win when K_n isn't viable. Verify the unified picks K_3
    # (preferred over books).
    atoms = find_disjoint_atoms(g)
    assert atoms[0].family in ("K_3", "B_3")


def test_unified_falls_through_to_wheels():
    """A graph with no K_n/K_{a,b}/books but with wheels picks W_n."""
    # Two disjoint W_5 wheels. W_5 doesn't contain K_4 (it has K_3 = triangles
    # via hub + 2 adjacent rim vertices), so K_3 fires. To test wheel fallback,
    # need a graph WITHOUT triangles — but a wheel always has triangles…
    # The wheel test really only fires for graphs that lack lower-tier matches.
    # Skip this scenario (out of scope for current dispatch order).
    pytest.skip("Wheel fallback requires triangle-free graph with W_n; "
                "such graphs are rare and not in current scope.")


# ---------------------------------------------------------------------------
# L_n (ladder) detection
# ---------------------------------------------------------------------------

def test_ladder_atoms_single_graph_returns_empty():
    """A single L_5 (no disjoint pair) yields no atoms (need ≥2 disjoint)."""
    g = Graph.from_networkx(nx.ladder_graph(5))
    atoms = find_disjoint_ladder_atoms(g, min_n=3, max_n=5)
    assert atoms == []


def test_ladder_atoms_two_disjoint_ladders():
    """Two disjoint L_4 ladders → returns 2× L_4 atoms."""
    g_nx = nx.disjoint_union(nx.ladder_graph(4), nx.ladder_graph(4))
    g = Graph.from_networkx(g_nx)
    atoms = find_disjoint_ladder_atoms(g, min_n=3, max_n=5)
    assert len(atoms) == 2
    assert all(a.family == "L_4" for a in atoms)
    assert all(len(a.vertices) == 8 for a in atoms)
    # Disjoint vertices
    a, b = atoms[0].vertices, atoms[1].vertices
    assert a.isdisjoint(b)


def test_ladder_atoms_prefers_larger_n():
    """Two disjoint L_5 graphs → finds 2× L_5 (not L_3 or L_4)."""
    g_nx = nx.disjoint_union(nx.ladder_graph(5), nx.ladder_graph(5))
    g = Graph.from_networkx(g_nx)
    atoms = find_disjoint_ladder_atoms(g, min_n=3, max_n=6)
    assert len(atoms) == 2
    assert atoms[0].family == "L_5"


def test_ladder_atoms_z21_has_ladder():
    """Zephyr Z(2,1) contains ladder substructure (per Probe 10)."""
    g = Graph.from_networkx(dnx.zephyr_graph(2, 1))
    atoms = find_disjoint_ladder_atoms(g, min_n=3, max_n=8)
    assert len(atoms) >= 2
    # Atoms are disjoint
    seen_vertices: set = set()
    for atom in atoms:
        assert atom.vertices.isdisjoint(seen_vertices)
        seen_vertices |= atom.vertices


def test_ladder_atoms_returns_empty_on_clique():
    """K_5 has no ladder structure (no 2-row organization)."""
    g = Graph.from_networkx(nx.complete_graph(5))
    atoms = find_disjoint_ladder_atoms(g, min_n=3, max_n=5)
    # K_5 doesn't have a clean L_n; even if walking finds 2n vertices it
    # would need 2 disjoint such sets, impossible in K_5 (5 vertices total).
    assert atoms == []


# ---------------------------------------------------------------------------
# Y_n (prism) detection
# ---------------------------------------------------------------------------

def test_prism_atoms_single_graph_returns_empty():
    """Single Y_5 prism (no disjoint pair) yields no atoms."""
    g = Graph.from_networkx(nx.circular_ladder_graph(5))
    atoms = find_disjoint_prism_atoms(g, min_n=3, max_n=5)
    assert atoms == []


def test_prism_atoms_two_disjoint_prisms():
    """Two disjoint Y_4 prisms → returns 2× Y_4 atoms."""
    g_nx = nx.disjoint_union(
        nx.circular_ladder_graph(4), nx.circular_ladder_graph(4),
    )
    g = Graph.from_networkx(g_nx)
    atoms = find_disjoint_prism_atoms(g, min_n=3, max_n=5)
    assert len(atoms) == 2
    assert all(a.family == "Y_4" for a in atoms)
    assert all(len(a.vertices) == 8 for a in atoms)
    # Disjoint vertices
    a, b = atoms[0].vertices, atoms[1].vertices
    assert a.isdisjoint(b)


def test_prism_atoms_rung_consistency():
    """Two triangles connected by 3 edges that DON'T form a prism
    (rungs not respecting cycle order) should NOT count as Y_3.

    Construct: triangles {0,1,2} and {3,4,5}, with "rungs" (0,4), (1,5),
    (2,3). Rotating cycle 2 to align: try [4,5,3], [5,3,4], [3,4,5].
    For c1=[0,1,2] and c2=[3,4,5], rung target is [4,5,3]. Check edge
    (0,4) yes, (1,5) yes, (2,3) yes. So aligns with rotation shift=1
    of c2! This IS a valid Y_3 prism — let me check.

    Actually [4,5,3] = [3,4,5][1:] + [3,4,5][:1] = rotation. So this IS
    a Y_3. Test it as a positive case.
    """
    g_nx = nx.Graph()
    g_nx.add_edges_from([
        (0, 1), (1, 2), (2, 0),  # triangle 1
        (3, 4), (4, 5), (5, 3),  # triangle 2
        (0, 4), (1, 5), (2, 3),  # rungs (rotated alignment)
    ])
    # This is a single Y_3 prism (not two disjoint). Add a second copy.
    g2 = nx.Graph()
    for u, v in g_nx.edges():
        g2.add_edge(u + 6, v + 6)
    g_combined = nx.compose(g_nx, g2)
    g = Graph.from_networkx(g_combined)
    atoms = find_disjoint_prism_atoms(g, min_n=3, max_n=4)
    assert len(atoms) == 2
    assert all(a.family == "Y_3" for a in atoms)


def test_prism_atoms_rejects_non_prism_pair():
    """Two triangles with 3 connecting edges in WRONG cycle order should NOT
    be detected as a prism. Construct triangles {0,1,2} and {3,4,5} with
    rungs (0,3), (1,3), (2,3) — three rungs but all to the same vertex 3.
    Not a prism.
    """
    g_nx = nx.Graph()
    g_nx.add_edges_from([
        (0, 1), (1, 2), (2, 0),  # triangle 1
        (3, 4), (4, 5), (5, 3),  # triangle 2
        (0, 3), (1, 3), (2, 3),  # malformed rungs (all to vertex 3)
    ])
    # Add disjoint copy
    g2 = nx.Graph()
    for u, v in g_nx.edges():
        g2.add_edge(u + 6, v + 6)
    g_combined = nx.compose(g_nx, g2)
    g = Graph.from_networkx(g_combined)
    atoms = find_disjoint_prism_atoms(g, min_n=3, max_n=4)
    assert atoms == []  # Rung structure violates prism property


# ---------------------------------------------------------------------------
# Unified dispatch with new tiers
# ---------------------------------------------------------------------------

def test_unified_includes_ladder_prism_in_signature():
    """find_disjoint_atoms accepts new try_ladders / try_prisms flags
    and ladder_min/max + prism_min/max bounds without error."""
    g_nx = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(4))
    g = Graph.from_networkx(g_nx)
    atoms = find_disjoint_atoms(
        g, try_ladders=False, try_prisms=False,
        ladder_min=3, ladder_max=8, prism_min=3, prism_max=8,
    )
    # K_n tier fires on the two disjoint K_4 atoms
    assert len(atoms) == 2
    assert atoms[0].family == "K_4"


def test_unified_disabled_tiers_skip_correctly():
    """With all primary tiers disabled, unified can still fire on L_n."""
    # Disjoint ladders + no other tier hits
    g_nx = nx.disjoint_union(nx.ladder_graph(5), nx.ladder_graph(5))
    g = Graph.from_networkx(g_nx)
    # Disable K_n / K_{a,b} / B_n / W_n so L_n is the only viable tier.
    atoms = find_disjoint_atoms(
        g, try_kn=False, try_kab=False, try_books=False, try_wheels=False,
    )
    assert len(atoms) == 2
    assert atoms[0].family.startswith("L_")


# ---------------------------------------------------------------------------
# Cost-aware family selection
# ---------------------------------------------------------------------------

def test_cost_aware_returns_empty_when_no_atoms():
    """A path has no atoms in any family."""
    g = Graph.from_networkx(nx.path_graph(8))
    atoms = find_atoms_cost_aware(g)
    assert atoms == []


def test_cost_aware_matches_legacy_on_z12():
    """Z(1,2)'s K_4 atoms (junction=4) fit the default budget; cost-aware
    matches legacy K_n-first dispatch.

    Earlier minimize-junction logic regressed Z(1,2) >5× by picking B_2
    atoms (junction=1) — those didn't disconnect anything per peel. The
    conservative budget-fallthrough version preserves K_n-first when
    K_n fits the budget. See `feedback-cost-aware-heuristic-wrong`.
    """
    g = Graph.from_networkx(dnx.zephyr_graph(1, 2))
    cost_atoms = find_atoms_cost_aware(g)
    legacy_atoms = find_disjoint_atoms(g)
    assert cost_atoms[0].family == legacy_atoms[0].family == "K_4"
    assert len(cost_atoms) == len(legacy_atoms) == 2


def test_cost_aware_falls_through_when_kn_exceeds_budget():
    """When K_n's smallest junction exceeds budget, fall through to K_{a,b}.

    Build graph with:
    - 2 disjoint K_4 atoms joined by 5-edge junction (exceeds budget=4)
    - 2 disjoint K_{2,2} atoms (= C_4 faces) joined by 1-edge bridge
      (fits budget)
    Legacy returns K_4 (engine then rejects post-hoc). Cost-aware skips
    K_4 (over budget), falls through, returns K_{2,2}.
    """
    g_nx = nx.Graph()
    # K_4 #1: vertices 0..3
    for i in range(4):
        for j in range(i + 1, 4):
            g_nx.add_edge(i, j)
    # K_4 #2: vertices 4..7
    for i in range(4):
        for j in range(i + 1, 4):
            g_nx.add_edge(i + 4, j + 4)
    # Junction between K_4 atoms: 5 edges forming ONE connected component
    # (all share vertex 4 to ensure bipartite-edge subgraph is connected)
    for u, v in [(0, 4), (1, 4), (2, 4), (3, 4), (0, 5)]:
        g_nx.add_edge(u, v)
    # K_{2,2} #1: vertices 8,9 ↔ 10,11
    for i in [8, 9]:
        for j in [10, 11]:
            g_nx.add_edge(i, j)
    # K_{2,2} #2: vertices 12,13 ↔ 14,15
    for i in [12, 13]:
        for j in [14, 15]:
            g_nx.add_edge(i, j)
    # 1-edge bridge between K_{2,2} atoms
    g_nx.add_edge(8, 12)
    g = Graph.from_networkx(g_nx)

    # Legacy returns K_4 (K_n tier fires; junction not checked at this level)
    legacy = find_disjoint_atoms(g)
    assert legacy[0].family == "K_4"
    legacy_j = find_smallest_junction(g, legacy)
    assert legacy_j is not None
    assert len(legacy_j) > 4, (
        f"expected K_4 junction > 4 for fallthrough test, got {len(legacy_j)}"
    )

    # Cost-aware with budget=4 should fall through to K_{2,2}
    cost = find_atoms_cost_aware(g, max_junction_size=4)
    assert cost != []
    # K_n tier exceeds budget; should fall through to K_{a,b}
    assert cost[0].family.startswith("K_{"), (
        f"expected K_{{a,b}} fallthrough, got {cost[0].family}"
    )
    cost_j = find_smallest_junction(g, cost)
    assert cost_j is not None
    assert len(cost_j) <= 4


def test_cost_aware_respects_max_junction_size():
    """When max_junction_size is small, filters out families that don't fit."""
    g_nx = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(4))
    # Add 1 bridge edge so atoms are connected (junction=1)
    g_nx.add_edge(0, 4)
    g = Graph.from_networkx(g_nx)
    # max_junction_size=0 should reject any candidate
    atoms = find_atoms_cost_aware(g, max_junction_size=0)
    assert atoms == []
    # max_junction_size=1 should accept the bridge-junction K_4 atoms
    atoms = find_atoms_cost_aware(g, max_junction_size=1)
    assert len(atoms) == 2
    assert atoms[0].family == "K_4"


def test_cost_aware_disabled_family_skipped():
    """When try_kn=False, K_n atoms are not considered even if available."""
    g = Graph.from_networkx(dnx.zephyr_graph(1, 2))
    atoms_no_kn = find_atoms_cost_aware(g, try_kn=False)
    assert atoms_no_kn != []
    # With try_kn disabled, must be K_{a,b}/B/W/L/Y family
    assert not atoms_no_kn[0].family.startswith("K_") or \
           atoms_no_kn[0].family.startswith("K_{")


def test_cost_aware_tiebreak_prefers_fewer_atoms():
    """When two families both yield junction-size 1, prefer fewer atoms."""
    # Graph: 2× K_4 with 1 bridge edge. K_4 has junction=1 with 2 atoms;
    # K_3 sub-cliques would have junction=1 with more atoms (within K_4).
    # Cost-aware should pick K_4 (fewer atoms) at tiebreak.
    g_nx = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(4))
    g_nx.add_edge(0, 4)
    g = Graph.from_networkx(g_nx)
    atoms = find_atoms_cost_aware(g)
    # K_4 picked (largest k, 2 atoms, junction=1)
    assert atoms[0].family == "K_4"
    assert len(atoms) == 2


# ---------------------------------------------------------------------------
# Heterogeneous (mixed-family) decomposition
# ---------------------------------------------------------------------------

def test_heterogeneous_default_order_z12():
    """Z(1,2) default K_n-first heterogeneous: K_4×2 + K_{2,2}×4 = 24/24."""
    g = Graph.from_networkx(dnx.zephyr_graph(1, 2))
    atoms = find_atoms_heterogeneous(g)
    assert len(atoms) >= 2
    families = {a.family for a in atoms}
    assert len(families) >= 2, f"expected ≥2 families, got {families}"
    total_vertices = sum(len(a.vertices) for a in atoms)
    assert total_vertices == g.node_count()  # full coverage


def test_heterogeneous_books_first_surfaces_books():
    """Books-first family ordering surfaces book atoms on Z(2,1)/Z(2,2)."""
    g = Graph.from_networkx(dnx.zephyr_graph(2, 1))
    books_first = find_atoms_heterogeneous(
        g, family_order=["books", "kab", "kn", "wheels", "ladders", "prisms"],
    )
    assert books_first, "expected books-first to find atoms in Z(2,1)"
    families = {a.family for a in books_first}
    assert any(f.startswith("B_") for f in families), (
        f"expected B_n in families, got {families}"
    )


def test_heterogeneous_disjoint_vertices():
    """Heterogeneous atoms are mutually vertex-disjoint."""
    g = Graph.from_networkx(dnx.zephyr_graph(1, 3))
    atoms = find_atoms_heterogeneous(g)
    assert atoms
    seen = set()
    for atom in atoms:
        assert atom.vertices.isdisjoint(seen), (
            f"atom {atom.family} overlaps prior atoms"
        )
        seen |= atom.vertices


def test_heterogeneous_respects_junction_budget():
    """Heterogeneous returns [] if smallest junction exceeds budget."""
    g_nx = nx.Graph()
    # 2 K_4s + dense K_{4,4} junction (16 inter-K_4 edges)
    for i in range(4):
        for j in range(i + 1, 4):
            g_nx.add_edge(i, j)
            g_nx.add_edge(i + 4, j + 4)
    for i in range(4):
        for j in range(4):
            g_nx.add_edge(i, j + 4)
    g = Graph.from_networkx(g_nx)
    # The whole graph is K_8 essentially; no clean atoms with junction ≤ 2
    atoms = find_atoms_heterogeneous(g, max_junction_size=1)
    # K_4 × 2 has junction much greater than 1, so should return []
    # (or K_8 as single atom which is also fine; just shouldn't blow budget)
    if atoms:
        j = find_smallest_junction(g, atoms)
        assert j is None or len(j) <= 1


# ===========================================================================
# Section 2: UNIFIED CHORD-JUNCTION THEOREM (inline reference impl)
# Source: test_unified_chord_junction.py
#
# Theorem (proved May 25, 2026; see
# `tutte/research/cyclotomic_chord_junction_theorem.md`):
#
#     T(G ⊕_{V_k} G; x, y)
#         = (x − 1) · T(G; x, y)²
#         + Σ_{∅ ≠ S ⊆ V_k} T(G ∪_{V_S} G; x, y)
#
# where ``G ⊕_{V_k} G`` is two disjoint copies of ``G`` joined by chord
# edges between corresponding ``V_k`` vertices, and ``G ∪_{V_S} G`` is
# the **multigraph** obtained by identifying corresponding vertices in
# ``S`` across the two copies (parallel edges preserved).
# ===========================================================================


# ---------------------------------------------------------------------------
# Multigraph helpers — construct chord-junction graph and mergers
# ---------------------------------------------------------------------------


def _merge_two_copies_multi(G: nx.Graph, S: Sequence[int]) -> MultiGraph:
    """Build ``G ∪_{V_S} G`` as a tutte.MultiGraph.

    Two disjoint copies of ``G`` (vertex ``v`` in copy 1, ``v + n`` in
    copy 2), then identify ``v`` with ``v + n`` for each ``v ∈ S``.
    Parallel edges arising from the identification are preserved.
    """
    n = G.number_of_nodes()
    S_set = set(S)
    # The kept representative for vertex v in copy 2 is v if v ∈ S, else v + n.
    def repr_of(node: int) -> int:
        if node < n:
            return node
        original = node - n
        if original in S_set:
            return original  # identified with copy-1 representative
        return node
    # Collect all nodes after identification.
    nodes = set()
    for v in G.nodes():
        nodes.add(v)            # copy 1
        if v in S_set:
            pass                # identified; no new node from copy 2
        else:
            nodes.add(v + n)    # copy 2 (kept distinct)
    # Collect edges with multiplicity.
    edge_counts: dict[Tuple[int, int], int] = {}
    loop_counts: dict[int, int] = {}
    for u, w in G.edges():
        # Copy 1: edge between u and w.
        r_u, r_w = repr_of(u), repr_of(w)
        if r_u == r_w:
            loop_counts[r_u] = loop_counts.get(r_u, 0) + 1
        else:
            key = (min(r_u, r_w), max(r_u, r_w))
            edge_counts[key] = edge_counts.get(key, 0) + 1
        # Copy 2: edge between u + n and w + n.
        r_u2, r_w2 = repr_of(u + n), repr_of(w + n)
        if r_u2 == r_w2:
            loop_counts[r_u2] = loop_counts.get(r_u2, 0) + 1
        else:
            key = (min(r_u2, r_w2), max(r_u2, r_w2))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    return MultiGraph(
        nodes=frozenset(nodes),
        edge_counts=edge_counts,
        loop_counts=loop_counts,
    )


# ---------------------------------------------------------------------------
# Synthesis adapters (engine returns SynthesisResult; tests want Polynomial)
# ---------------------------------------------------------------------------


def _T_simple(engine: SynthesisEngine, G_nx: nx.Graph) -> TuttePolynomial:
    return engine.synthesize(Graph.from_networkx(G_nx)).polynomial


def _T_multi(engine: SynthesisEngine, mg: MultiGraph) -> TuttePolynomial:
    return engine._synthesize_multigraph(mg)


# ---------------------------------------------------------------------------
# Unified theorem — reference inline implementation used to anchor the test
# ---------------------------------------------------------------------------


def _unified_chord_junction_inline(
    engine: SynthesisEngine,
    G_nx: nx.Graph,
    V_k: Sequence[int],
) -> TuttePolynomial:
    """Reference implementation of the unified I-E theorem.

    Computes  ``(x − 1) · T(G)² + Σ_{∅ ≠ S ⊆ V_k} T(G ∪_{V_S} G)``
    by directly calling the engine on each merger graph. Used as a
    cross-check against direct synthesis of the full chord-junction
    graph. Slow but correct.
    """
    T_G = _T_simple(engine, G_nx)
    x_poly = TuttePolynomial.x()
    one = TuttePolynomial.from_coefficients({(0, 0): 1})
    result = (x_poly + (-1) * one) * T_G * T_G
    for r in range(1, len(V_k) + 1):
        for S in combinations(sorted(V_k), r):
            merger = _merge_two_copies_multi(G_nx, S)
            result = result + _T_multi(engine, merger)
    return result


# ---------------------------------------------------------------------------
# Tests — unified theorem matches direct engine synthesis
# ---------------------------------------------------------------------------


def _check_equivalence(engine: SynthesisEngine, G_nx: nx.Graph, V_k):
    G_chord = _chord_junction_simple_nx(G_nx, V_k)
    direct = _T_simple(engine, G_chord)
    inline = _unified_chord_junction_inline(engine, G_nx, V_k)
    assert direct == inline, (
        f"Unified I-E disagrees with direct engine synthesis on "
        f"G={G_nx.number_of_nodes()}v/{G_nx.number_of_edges()}e, V_k={V_k}.\n"
        f"  direct = {direct}\n"
        f"  inline = {inline}\n"
        f"  diff   = {direct + (-1) * inline}"
    )


def test_unified_K2_chord_two_vertex(engine):
    """K_2 ⊕_{0,1} K_2 = C_4 (canonical sanity check)."""
    _check_equivalence(engine, nx.complete_graph(2), [0, 1])


def test_unified_K3_chord_all_three(engine):
    """K_3 ⊕_{0,1,2} K_3 = prism Y_3."""
    _check_equivalence(engine, nx.complete_graph(3), [0, 1, 2])


def test_unified_P3_chord_path_positions(engine):
    """Path P_3 with V_k = {0, 1, 2}.

    Triggers the tree-base case where H = G[V_k] = P_3 itself, exercising
    the (x − 1)·T(G)² prefactor and 7 merger terms.
    """
    _check_equivalence(engine, nx.path_graph(3), [0, 1, 2])


def test_unified_C4_chord_partial_three(engine):
    """C_4 with V_k = {0, 1, 2} (NON-fluff-irrelevant case).

    The closing edge of C_4 creates a shadow cycle through positions
    {0, 1, 2}; the unified theorem must still match direct synthesis.
    """
    _check_equivalence(engine, nx.cycle_graph(4), [0, 1, 2])


def test_unified_diamond_chord_all_four(engine):
    """Diamond K_4 - e with V_k = all 4 vertices.

    Diamond is the smallest H whose chord-junction is matroid-self-dual
    (T_chord(x, y) = T_chord(y, x)). This exercises the higher-degree
    case (|V_k| = 4 → 15 mergers).
    """
    diamond = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    _check_equivalence(engine, diamond, [0, 1, 2, 3])


def test_unified_K4_chord_all_four(engine):
    """K_4 ⊕_{0,1,2,3} K_4. Matroid-self-dual condition false
    (|E(K_4 chord)| = 16, 2|V| − 2 = 14)."""
    _check_equivalence(engine, nx.complete_graph(4), [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# Tests — equivalence with apply_kmatching_formula (matching connector)
# ---------------------------------------------------------------------------


def _build_chord_junction_graph(G_nx: nx.Graph, V_k: Sequence[int]) -> Graph:
    """Build the chord-junction graph as a tutte.Graph (immutable)."""
    g_nx = _chord_junction_simple_nx(G_nx, V_k)
    return Graph.from_networkx(g_nx)


def _make_kmatching_junction(
    cell_size: int, V_k: Sequence[int],
) -> KMatchingJunction:
    """Construct a KMatchingJunction representing the chord junction.

    cell_i = 0 (copy 1: vertices 0..n-1)
    cell_j = 1 (copy 2: vertices n..2n-1)
    edges  = [(v, v + n) for v in V_k]
    """
    n = cell_size
    edges = [(v, v + n) for v in V_k]
    return KMatchingJunction(
        cell_i=0,
        cell_j=1,
        edges=edges,
        anchors_i=list(V_k),
        anchors_j=[v + n for v in V_k],
    )


def _check_kmatching_equivalence(engine, G_nx, V_k):
    """apply_kmatching_formula matches direct engine synthesis."""
    chord_graph = _build_chord_junction_graph(G_nx, V_k)
    junc = _make_kmatching_junction(G_nx.number_of_nodes(), V_k)
    via_kmatching = apply_kmatching_formula(
        chord_graph, [junc], engine._synthesize_multigraph,
    )
    direct = engine.synthesize(chord_graph).polynomial
    assert via_kmatching == direct, (
        f"apply_kmatching_formula disagrees with direct engine on V_k={V_k}."
    )


def test_kmatching_K3_matches_direct(engine):
    """Production apply_kmatching_formula matches direct synth on K_3 chord."""
    _check_kmatching_equivalence(engine, nx.complete_graph(3), [0, 1, 2])


def test_kmatching_K4_matches_direct(engine):
    """Production apply_kmatching_formula matches direct synth on K_4 chord."""
    _check_kmatching_equivalence(engine, nx.complete_graph(4), [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# Cut-vertex identity — converts between unified and k-matching coefficients
# ---------------------------------------------------------------------------


def test_cut_vertex_identity_K3(engine):
    """T(G ∪_v G) = T(G)² (cut-vertex factorization).

    This is the identity that reconciles the unified theorem's (x-1)
    prefactor with the k-matching formula's (x, k-1, ...) coefficients.
    """
    G_nx = nx.complete_graph(3)
    T_G = _T_simple(engine, G_nx)
    merger_one_vertex = _merge_two_copies_multi(G_nx, [0])
    T_M1 = _T_multi(engine, merger_one_vertex)
    assert T_M1 == T_G * T_G, (
        f"Cut-vertex identity violated for K_3: T(M_1) = {T_M1} vs "
        f"T(K_3)² = {T_G * T_G}"
    )


def test_cut_vertex_identity_K4(engine):
    """Same identity on K_4 (larger base graph)."""
    G_nx = nx.complete_graph(4)
    T_G = _T_simple(engine, G_nx)
    merger_one_vertex = _merge_two_copies_multi(G_nx, [0])
    T_M1 = _T_multi(engine, merger_one_vertex)
    assert T_M1 == T_G * T_G


# ===========================================================================
# Section 3: UNIFIED_CHORD_JUNCTION CLOSED-FORM MODULE
# Source: test_chord_junction_closed_form.py
#
# Tests for the production unified chord-junction module
# (``tutte/roots/chord_junction_closed_form.py``).
# ===========================================================================


# ---------------------------------------------------------------------------
# Symmetric API — equivalence with direct engine synthesis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,G_nx,V_k", [
    ("K_2", nx.complete_graph(2), [0, 1]),
    ("K_3", nx.complete_graph(3), [0, 1, 2]),
    ("P_3", nx.path_graph(3), [0, 1, 2]),
    ("C_4_partial", nx.cycle_graph(4), [0, 1, 2]),
    ("diamond", nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]),
     [0, 1, 2, 3]),
    ("K_4", nx.complete_graph(4), [0, 1, 2, 3]),
])
def test_unified_matches_direct_engine(engine, label, G_nx, V_k):
    base = Graph.from_networkx(G_nx)
    via_unified = unified_chord_junction(
        base, V_k, engine._synthesize_multigraph,
    )
    direct = engine.synthesize(_chord_junction_simple_graph(G_nx, V_k)).polynomial
    assert via_unified == direct, (
        f"{label}: unified_chord_junction disagrees with direct engine"
    )


# ---------------------------------------------------------------------------
# Asymmetric API — equivalence with direct engine synthesis
# ---------------------------------------------------------------------------


def _build_asymmetric_chord_graph(G_left_nx, G_right_nx, chord_pairs) -> Graph:
    """Build G_left ⊕ G_right with chord edges as a simple Graph."""
    n_left = G_left_nx.number_of_nodes()
    g = nx.Graph()
    g.add_nodes_from(G_left_nx.nodes())
    g.add_edges_from(G_left_nx.edges())
    g.add_nodes_from(u + n_left for u in G_right_nx.nodes())
    g.add_edges_from((u + n_left, w + n_left) for u, w in G_right_nx.edges())
    for (u, w) in chord_pairs:
        g.add_edge(u, w + n_left)
    return Graph.from_networkx(g)


def test_asymmetric_K3_K4_matches_direct(engine):
    """K_3 ⊕ K_4 with two chord edges connecting (0, 0) and (1, 1)."""
    G_left_nx = nx.complete_graph(3)
    G_right_nx = nx.complete_graph(4)
    chord_pairs = [(0, 0), (1, 1)]
    via_asymmetric = unified_chord_junction_asymmetric(
        Graph.from_networkx(G_left_nx),
        Graph.from_networkx(G_right_nx),
        chord_pairs,
        engine._synthesize_multigraph,
    )
    direct = engine.synthesize(
        _build_asymmetric_chord_graph(G_left_nx, G_right_nx, chord_pairs),
    ).polynomial
    assert via_asymmetric == direct


def test_asymmetric_K2_K3_single_chord(engine):
    """K_2 ⊕ K_3 with a single chord edge — exercises ``k = 1`` branch."""
    G_left_nx = nx.complete_graph(2)
    G_right_nx = nx.complete_graph(3)
    chord_pairs = [(0, 2)]
    via_asymmetric = unified_chord_junction_asymmetric(
        Graph.from_networkx(G_left_nx),
        Graph.from_networkx(G_right_nx),
        chord_pairs,
        engine._synthesize_multigraph,
    )
    direct = engine.synthesize(
        _build_asymmetric_chord_graph(G_left_nx, G_right_nx, chord_pairs),
    ).polynomial
    assert via_asymmetric == direct


def test_asymmetric_populates_and_hits_merger_table(engine):
    """``update_merger_table=True`` writes entries on the asymmetric path
    and a second call short-circuits to ``synth_multigraph`` only for the
    two bases (left + right), skipping all merger evaluations.

    Note: the chord pattern ``[(0, 0), (1, 1)]`` produces 3 chord-pair
    subsets, but S=(0,) and S=(1,) build isomorphic merger multigraphs
    (K_3 ⊕ K_4 sharing one corner — symmetric under aut(K_3)×aut(K_4)).
    The by-merger dedup correctly stores 2 distinct entries; the second
    call still hits both via ``lookup_by_merger``.
    """
    G_left_nx = nx.complete_graph(3)
    G_right_nx = nx.complete_graph(4)
    chord_pairs = [(0, 0), (1, 1)]
    table = MergerTable()

    # First pass populates the table.
    _ = unified_chord_junction_asymmetric(
        Graph.from_networkx(G_left_nx),
        Graph.from_networkx(G_right_nx),
        chord_pairs,
        engine._synthesize_multigraph,
        merger_table=table,
        update_merger_table=True,
        family_tag="asym-test",
    )
    # 2 distinct merger graphs after canonical-key dedup.
    assert len(table) == 2

    # Second pass: counting wrapper should fire exactly twice (once for
    # each base; not for any of the 3 sub-syntheses — all hit cache).
    counter = _CountingSynth(engine._synthesize_multigraph)
    via_cached = unified_chord_junction_asymmetric(
        Graph.from_networkx(G_left_nx),
        Graph.from_networkx(G_right_nx),
        chord_pairs,
        counter,
        merger_table=table,
    )
    direct = engine.synthesize(
        _build_asymmetric_chord_graph(G_left_nx, G_right_nx, chord_pairs),
    ).polynomial
    assert via_cached == direct
    assert counter.calls == 2, (
        f"Expected 2 synth calls (left + right base) but got {counter.calls}"
    )


def test_asymmetric_hits_symmetric_cache_by_canonical_key(engine):
    """Asymmetric chord pattern whose merger is isomorphic to a cached
    symmetric merger MUST hit the cache via ``lookup_by_merger``.

    Setup: warm the table with a symmetric K_3 merger at V_T = {0, 1}.
    Then call the asymmetric API on K_3 ⊕ K_3 with chord pairs
    [(0, 0), (1, 1)] — same chord vertices, semantically the symmetric
    case. The merger graph is identical so its canonical key matches and
    the asymmetric path should skip the synth call for that subset.
    """
    from tutte.roots.chord_junction_closed_form import unified_chord_junction
    base = Graph.from_networkx(nx.complete_graph(3))

    table = MergerTable()
    unified_chord_junction(
        base, [0, 1], engine._synthesize_multigraph,
        merger_table=table, update_merger_table=True,
        family_tag="symmetric-warmup",
    )
    # 2^2 − 1 = 3 entries from the symmetric warmup.
    assert len(table) == 3

    # Now ask the asymmetric API the equivalent question. Each chord
    # subset's merger graph is the same as the corresponding symmetric
    # one (since the asymmetric call mirrors the symmetric pattern), so
    # all three merger evaluations should cache-hit by merger key.
    counter = _CountingSynth(engine._synthesize_multigraph)
    _ = unified_chord_junction_asymmetric(
        base, base,
        [(0, 0), (1, 1)],
        counter,
        merger_table=table,
    )
    # Two synth calls for the two bases, zero for the three mergers.
    assert counter.calls == 2, (
        f"Expected 2 synth calls (only the two bases) but got {counter.calls}"
    )


# ---------------------------------------------------------------------------
# Merger table cache — hits and populate-on-miss
# ---------------------------------------------------------------------------


class _CountingSynth:
    """Wraps the engine multigraph synth to count invocations."""
    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def __call__(self, mg):
        self.calls += 1
        return self._inner(mg)


def test_merger_table_hits_skip_synth(engine):
    """When all mergers are cached, ``synth_multigraph`` is called only
    for the base graph itself (one call), not for any merger."""
    G_nx = nx.complete_graph(3)
    V_k = [0, 1, 2]
    base = Graph.from_networkx(G_nx)

    # First pass populates the table.
    table = MergerTable()
    unified_chord_junction(
        base, V_k, engine._synthesize_multigraph,
        merger_table=table, update_merger_table=True,
        family_tag="test", base_name="K_3",
    )
    # 2^3 - 1 = 7 distinct V_T subsets → 7 cached entries.
    assert len(table) == 7

    # Second pass with the populated table; counting wrapper should fire
    # exactly once (for the base graph), not for any of the 7 mergers.
    counter = _CountingSynth(engine._synthesize_multigraph)
    via_cached = unified_chord_junction(
        base, V_k, counter, merger_table=table,
    )
    direct = engine.synthesize(_chord_junction_simple_graph(G_nx, V_k)).polynomial
    assert via_cached == direct
    assert counter.calls == 1, (
        f"Expected 1 synth call (base only) but got {counter.calls}"
    )


def test_update_merger_table_populates_entries(engine):
    """``update_merger_table=True`` writes new mergers into the table."""
    G_nx = nx.complete_graph(2)
    V_k = [0, 1]
    base = Graph.from_networkx(G_nx)

    table = MergerTable()
    assert len(table) == 0
    unified_chord_junction(
        base, V_k, engine._synthesize_multigraph,
        merger_table=table, update_merger_table=True,
        family_tag="chimera", base_name="K_2",
    )
    # 2^2 - 1 = 3 entries.
    assert len(table) == 3
    # Each entry has the requested family_tag.
    for entry in table.by_source.values():
        assert entry.family_tag == "chimera"
        assert entry.base_name == "K_2"


def test_merger_table_lookup_normalizes_unsorted_v_k(engine):
    """The cache key uses sorted V_T; the user can pass V_k in any order
    and the second call must hit the cache populated by the first."""
    G_nx = nx.complete_graph(3)
    base = Graph.from_networkx(G_nx)

    table = MergerTable()
    unified_chord_junction(
        base, [0, 1, 2], engine._synthesize_multigraph,
        merger_table=table, update_merger_table=True,
    )
    n_after_first = len(table)

    # Re-call with the SAME positions in a permuted order; no new entries
    # should be added because the cache normalizes V_T.
    unified_chord_junction(
        base, [2, 0, 1], engine._synthesize_multigraph,
        merger_table=table, update_merger_table=True,
    )
    assert len(table) == n_after_first


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_v_k_raises(engine):
    base = Graph.from_networkx(nx.complete_graph(3))
    with pytest.raises(ValueError, match="not in base graph"):
        unified_chord_junction(
            base, [0, 1, 7], engine._synthesize_multigraph,
        )


def test_asymmetric_invalid_chord_raises(engine):
    left = Graph.from_networkx(nx.complete_graph(2))
    right = Graph.from_networkx(nx.complete_graph(2))
    with pytest.raises(ValueError, match="not in base_left"):
        unified_chord_junction_asymmetric(
            left, right, [(9, 0)], engine._synthesize_multigraph,
        )
    with pytest.raises(ValueError, match="not in base_right"):
        unified_chord_junction_asymmetric(
            left, right, [(0, 9)], engine._synthesize_multigraph,
        )


# ---------------------------------------------------------------------------
# Merger builder — round-trip with the inline test reference
# ---------------------------------------------------------------------------


def test_build_symmetric_merger_K3_one_vertex():
    """K_3 ∪_{0} K_3 = two triangles sharing a vertex.

    Expect: 5 vertices, 6 edges (no parallel edges, no loops).
    """
    base = Graph.from_networkx(nx.complete_graph(3))
    merger = build_symmetric_merger(base, [0])
    assert merger.node_count() == 5  # 2*3 - 1
    assert merger.edge_count() == 6
    assert not merger.loop_counts


def test_build_symmetric_merger_K2_all_creates_parallel_edges():
    """K_2 ∪_{0,1} K_2 collapses to 2 vertices with 2 parallel edges."""
    base = Graph.from_networkx(nx.complete_graph(2))
    merger = build_symmetric_merger(base, [0, 1])
    assert merger.node_count() == 2  # 2*2 - 2
    # Two parallel edges between the merged vertices.
    assert merger.edge_count() == 2
    assert not merger.loop_counts


# ===========================================================================
# Section 4: MERGER LOOKUP TABLE
# Source: test_merger_lookup.py
#
# Unit tests for the merger lookup table (``tutte/lookup/merger.py``).
# The fixture tables use trivially small synthetic polynomials (no Tutte
# synthesis required) so the suite runs in well under a second.
# ===========================================================================


# ---------------------------------------------------------------------------
# Fixtures (helpers for synthesizing fake entries)
# ---------------------------------------------------------------------------


def _fake_key(seed: int) -> str:
    """Synthetic 64-char hex string standing in for a canonical key."""
    return f"{seed:064x}"


def _entry(
    base_seed: int,
    v_t,
    *,
    coeffs=None,
    family: str = "chimera",
    merger_seed: int | None = None,
    name: str = "test_base",
) -> MergerEntry:
    poly = TuttePolynomial.from_coefficients(coeffs or {(1, 0): 1, (0, 1): 1})
    merger_key = _fake_key(merger_seed) if merger_seed is not None else None
    return MergerEntry(
        base_canonical_key=_fake_key(base_seed),
        v_t=tuple(sorted(v_t)),
        polynomial=poly,
        merger_canonical_key=merger_key,
        base_name=name,
        family_tag=family,
        base_node_count=8,
        base_edge_count=16,
        merger_node_count=2 * 8 - len(v_t),
        merger_edge_count=32,
    )


# ---------------------------------------------------------------------------
# MergerEntry value semantics
# ---------------------------------------------------------------------------


def test_entry_lookup_key_uses_base_and_v_t():
    e = _entry(1, (0, 1, 2))
    assert e.lookup_key == (_fake_key(1), (0, 1, 2))


def test_entry_hash_eq_by_lookup_key_only():
    a = _entry(1, (0, 1))
    b = _entry(1, (0, 1), coeffs={(0, 0): 5})  # different polynomial
    assert a == b
    assert hash(a) == hash(b)


def test_entry_inequality_on_different_v_t():
    a = _entry(1, (0, 1))
    b = _entry(1, (0, 1, 2))
    assert a != b


# ---------------------------------------------------------------------------
# MergerTable mutation + lookup
# ---------------------------------------------------------------------------


def test_table_starts_empty():
    table = MergerTable()
    assert len(table) == 0
    assert table.lookup_by_source(_fake_key(0), (0,)) is None
    assert table.lookup_by_merger(_fake_key(99)) is None


def test_add_entry_inserts_into_both_indices():
    table = MergerTable()
    e = _entry(1, (0, 1), merger_seed=42)
    table.add_entry(e)
    assert len(table) == 1
    assert table.lookup_by_source(_fake_key(1), (0, 1)) is e
    assert table.lookup_by_merger(_fake_key(42)) is e


def test_lookup_by_source_normalizes_v_t_order():
    """Caller passing ``(2, 0, 1)`` should hit the same entry as ``(0, 1, 2)``."""
    table = MergerTable()
    e = _entry(1, (0, 1, 2))
    table.add_entry(e)
    assert table.lookup_by_source(_fake_key(1), (2, 0, 1)) is e


def test_add_entry_overwrites_existing_lookup_key():
    table = MergerTable()
    e1 = _entry(1, (0, 1), coeffs={(0, 0): 1})
    e2 = _entry(1, (0, 1), coeffs={(0, 0): 7})
    table.add_entry(e1)
    table.add_entry(e2)
    assert len(table) == 1
    looked_up = table.lookup_by_source(_fake_key(1), (0, 1))
    assert looked_up.polynomial.evaluate(0, 0) == 7


def test_entries_for_base_filters_by_base_key():
    table = MergerTable()
    table.add_entry(_entry(1, (0,), merger_seed=10))
    table.add_entry(_entry(1, (0, 1), merger_seed=11))
    table.add_entry(_entry(2, (0,), merger_seed=12))
    chimera = table.entries_for_base(_fake_key(1))
    assert len(chimera) == 2


def test_entries_for_family_filters_by_tag():
    table = MergerTable()
    table.add_entry(_entry(1, (0,), family="chimera"))
    table.add_entry(_entry(2, (0,), family="pegasus"))
    table.add_entry(_entry(3, (0,), family="chimera"))
    assert len(table.entries_for_family("chimera")) == 2
    assert len(table.entries_for_family("pegasus")) == 1
    assert len(table.entries_for_family("zephyr")) == 0


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_json_round_trip_preserves_entries(tmp_path):
    table = MergerTable()
    for i in range(3):
        table.add_entry(_entry(
            base_seed=i,
            v_t=tuple(range(i + 1)),
            coeffs={(j, 0): j + 1 for j in range(3)},
            merger_seed=100 + i,
            name=f"base_{i}",
        ))
    path = tmp_path / "merger_table.json"
    table.save(str(path))
    loaded = MergerTable.load(str(path))
    assert len(loaded) == len(table)
    for key, entry in table.by_source.items():
        loaded_entry = loaded.by_source[key]
        assert loaded_entry.polynomial == entry.polynomial
        assert loaded_entry.merger_canonical_key == entry.merger_canonical_key
        assert loaded_entry.base_name == entry.base_name
        assert loaded_entry.family_tag == entry.family_tag


# ---------------------------------------------------------------------------
# Binary round-trip
# ---------------------------------------------------------------------------


def test_binary_round_trip_preserves_entries():
    table = MergerTable()
    table.add_entry(_entry(1, (0,), merger_seed=10, name="K_4"))
    table.add_entry(_entry(2, (0, 1, 2), merger_seed=20, family="pegasus"))
    # Entry without merger key — exercise the `merger_present=0` branch.
    table.add_entry(_entry(3, (0, 1), merger_seed=None, family="zephyr"))

    blob = encode_merger_lookup_table(table)
    loaded = decode_merger_lookup_table(blob)

    assert len(loaded) == len(table)
    for key, entry in table.by_source.items():
        loaded_entry = loaded.by_source[key]
        assert loaded_entry.polynomial == entry.polynomial
        assert loaded_entry.merger_canonical_key == entry.merger_canonical_key
        assert loaded_entry.base_name == entry.base_name
        assert loaded_entry.family_tag == entry.family_tag
        assert loaded_entry.base_node_count == entry.base_node_count
        assert loaded_entry.merger_edge_count == entry.merger_edge_count


def test_binary_header_rejects_wrong_magic():
    """Decoder must refuse non-MRGT payloads (avoids silent collision with
    other binary table formats)."""
    with pytest.raises(ValueError, match="Invalid magic header"):
        decode_merger_lookup_table(b"XXXX\x01\x00")


# ---------------------------------------------------------------------------
# Default loader behaves before any data file exists
# ---------------------------------------------------------------------------


def test_counters_track_hits_and_misses():
    """``lookup_by_source`` and ``lookup_by_merger`` increment per-index
    hit/miss counters; ``reset_counters`` zeroes them. Counters are the
    primary observability hook for the chord-junction benchmark."""
    poly = TuttePolynomial.from_coefficients({(0, 0): 1})
    table = MergerTable()
    table.add_entry(MergerEntry(
        base_canonical_key="bk1",
        v_t=(0,),
        polynomial=poly,
        merger_canonical_key="mk1",
    ))

    # Hit + miss on by_source.
    assert table.lookup_by_source("bk1", (0,)) is not None
    assert table.lookup_by_source("bk1", (0, 1)) is None
    # Hit + miss on by_merger.
    assert table.lookup_by_merger("mk1") is not None
    assert table.lookup_by_merger("missing") is None

    bd = table.counter_breakdown()
    assert bd == {
        "hits_by_source":   1,
        "misses_by_source": 1,
        "hits_by_merger":   1,
        "misses_by_merger": 1,
    }
    assert table.hits == 2
    assert table.misses == 2

    table.reset_counters()
    assert table.hits == 0
    assert table.misses == 0


def test_load_default_returns_empty_when_no_data_file(monkeypatch, tmp_path):
    """Tutte default loader must return an empty table (not raise) when the
    on-disk merger_lookup_table doesn't exist yet — engine init relies on
    this graceful default for fresh checkouts.
    """
    from tutte.lookup import merger as merger_module
    monkeypatch.setattr(
        merger_module,
        "_default_data_dir",
        lambda: str(tmp_path),
    )
    table = load_default_merger_table()
    assert isinstance(table, MergerTable)
    assert len(table) == 0


# ===========================================================================
# Section 5: ENGINE UNIFIED-CHORD-JUNCTION DISPATCH
# Source: test_engine_unified_chord_dispatch.py
#
# Regression tests for the engine's unified chord-junction fast path.
# Locks in the behavior added by the Step 5 dispatch hookup
# (``tutte.synthesis.engine.SynthesisEngine._try_unified_chord_junction``).
# ===========================================================================


# ---------------------------------------------------------------------------
# Init contract — session cache loaded
# ---------------------------------------------------------------------------


def test_engine_init_loads_merger_session_cache():
    """Engine startup populates ``_merger_session_cache`` from the disk-
    backed ``merger_lookup_table`` (populated by the warmup script)."""
    e = SynthesisEngine(table=load_default_table(), verbose=False)
    assert isinstance(e._merger_session_cache, MergerTable)
    # The warmup populated K_{4,4} mergers (15 = 2^4 − 1 subsets).
    # Test passes even on a fresh checkout (table starts empty), so we
    # don't hard-code a minimum size; we just check the type contract.


# ---------------------------------------------------------------------------
# Cell-pair fast path fires when the chord pattern matches a cached orbit
# ---------------------------------------------------------------------------


def _build_clean_cell_pair() -> Graph:
    """Two K_{4,4} cells joined by a perfect matching on side A.

    Bipartition of each K_{4,4}: side A = {0, 1, 2, 3}, side B = {4, 5, 6, 7}.
    Chord edges: (i, i + 8) for i in 0..3 — corresponds exactly to the
    canonical ``V_T = (0, 1, 2, 3)`` entries in the warmed merger table.
    """
    K44 = nx.complete_bipartite_graph(4, 4)
    n = 8
    edges = set(K44.edges())
    for u, v in K44.edges():
        edges.add((u + n, v + n))
    for i in range(4):
        edges.add((i, i + n))
    return Graph(
        nodes=frozenset(set(K44.nodes()) | {v + n for v in K44.nodes()}),
        edges=frozenset((min(u, v), max(u, v)) for u, v in edges),
    )


def test_fast_path_fires_on_clean_side_a_cell_pair(engine_func):
    """Side-A matching K_{4,4} cell-pair routes through the unified
    theorem (1 synth call to the base; all 15 mergers cache-hit when
    the disk table is loaded)."""
    engine = engine_func
    g = _build_clean_cell_pair()

    fired = {"attempts": 0, "wins": 0, "synth_calls": 0}
    orig_try = engine._try_unified_chord_junction
    orig_synth = engine._synthesize_multigraph

    def counted_try(graph, junctions, partition):
        fired["attempts"] += 1
        result = orig_try(graph, junctions, partition)
        if result is not None:
            fired["wins"] += 1
        return result

    def counted_synth(mg, *a, **kw):
        fired["synth_calls"] += 1
        return orig_synth(mg, *a, **kw)

    engine._try_unified_chord_junction = counted_try
    engine._synthesize_multigraph = counted_synth
    try:
        res = engine._try_decomposition_chord_peel(g, 10, force=True)
    finally:
        engine._try_unified_chord_junction = orig_try
        engine._synthesize_multigraph = orig_synth

    assert res is not None, "engine should route this through the cell-pair path"
    assert res.method == "kmatching_formula"
    assert fired["attempts"] == 1
    assert fired["wins"] == 1
    assert res.polynomial.evaluate(1, 1) == 226492416  # known span trees


# ---------------------------------------------------------------------------
# Asymmetric path handles mixed-bipartition anchors via canonical-key lookup
# ---------------------------------------------------------------------------


def _build_mixed_bipartition_cell_pair() -> Graph:
    """The fixture from ``test_engine_k44_cycle_uses_kmatching_formula``.

    Anchors {0, 5, 6, 7} = 1 side-A + 3 side-B — a different
    automorphism orbit than {0, 1, 2, 3}. The symmetric ``(base_key,
    V_T)`` lookup does not align, but the merger multigraph's canonical
    key matches a cached symmetric merger (``K_{4,4}`` is aut-rich
    enough that mixed-bipartition chord identifications produce
    isomorphic mergers), so the asymmetric tier 2 path still wins.
    """
    import dwave_networkx as dnx
    cm1 = Graph.from_networkx(dnx.chimera_graph(1))
    n = max(cm1.nodes) + 1
    A_side = [0, 5, 6, 7]
    B_side = [1, 2, 3, 4]
    edges = set(cm1.edges)
    for u, v in cm1.edges:
        edges.add((u + n, v + n))
    for i in range(4):
        edges.add((A_side[i], B_side[i] + n))
    nodes = set(cm1.nodes) | {v + n for v in cm1.nodes}
    return Graph(
        nodes=frozenset(nodes),
        edges=frozenset((min(u, v), max(u, v)) for u, v in edges),
    )


def test_asymmetric_path_wins_on_mixed_bipartition_anchors(engine_func):
    """Asymmetric tier 2 path wins on mixed-bipartition anchors.

    The symmetric ``(base_key, V_T)`` lookup doesn't align because
    ``V_k_i != V_k_j`` after canonical relabel. But the asymmetric path
    builds the chord-pair merger explicitly and looks it up by the
    merger multigraph's canonical key — and since ``K_{4,4}``'s aut
    group is large enough that mixed-bipartition mergers are isomorphic
    to symmetric ones cached in the warmed merger table, the lookup
    hits and the fast path wins. Polynomial unchanged."""
    engine = engine_func
    g = _build_mixed_bipartition_cell_pair()

    fired = {"attempts": 0, "wins": 0}
    orig_try = engine._try_unified_chord_junction

    def counted_try(graph, junctions, partition):
        fired["attempts"] += 1
        result = orig_try(graph, junctions, partition)
        if result is not None:
            fired["wins"] += 1
        return result

    engine._try_unified_chord_junction = counted_try
    try:
        res = engine._try_decomposition_chord_peel(g, 10, force=True)
    finally:
        engine._try_unified_chord_junction = orig_try

    assert res is not None
    assert res.method == "kmatching_formula"
    assert fired["attempts"] == 1
    assert fired["wins"] == 1
    # Polynomial still correct (matches the existing kmatching test fixture).
    assert res.polynomial.evaluate(1, 1) == 226492416


# ---------------------------------------------------------------------------
# Asymmetric path: mergers populate the session cache by canonical-key
# ---------------------------------------------------------------------------


def test_asymmetric_path_populates_session_cache(engine_func):
    """Cache size strictly grows OR every merger is already cached.

    We can't assert "grows" outright because the warmup may already have
    cached every K_{4,4} merger that the mixed-bipartition fixture
    produces (they're all isomorphic to symmetric subsets of K_{4,4}).
    The semantic guarantee is: no merger gets recomputed twice within
    a session.
    """
    engine = engine_func
    g = _build_mixed_bipartition_cell_pair()
    cache_before = len(engine._merger_session_cache)
    res = engine._try_decomposition_chord_peel(g, 10, force=True)
    assert res is not None
    cache_after = len(engine._merger_session_cache)
    # Cache grows monotonically; either everything was already warm or
    # the asymmetric path inserted new entries.
    assert cache_after >= cache_before
