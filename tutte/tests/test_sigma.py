"""σ-equivariant Tutte machinery regressions.

Covers:

1. **σ-orbit chord ordering** (`_sigma_orbit_chord_order`,
   `_iterative_chord_rule`): when σ ∈ Aut(target) is a graph
   automorphism, reordering chords so σ-orbits are contiguous lets the
   engine's canonical_key cache catch isomorphic intermediate
   contractions.
2. **σ-equivariant treewidth DP** (`compute_tutte_per_orbit_mod`):
   elimination-style DP on the cover G that processes σ-paired edges
   together as 4-branch steps, keeping the active set σ-invariant.
"""
from __future__ import annotations

import networkx as nx
import pytest
import sympy

from tutte.graph import Graph
from tutte.graphs.k_sum import (
    _combine_chord_iteration, _iterative_chord_rule, _sigma_orbit_chord_order,
)
# sigma_equivariant_dp is deprecated (correct but no speed advantage, never wired
# into the engine). Moved to tutte/deprecated/; these per-orbit tests still run.
from tutte.deprecated.sigma_equivariant_dp import compute_tutte_per_orbit_mod
from tutte.synthesis.engine import SynthesisEngine


# ---------------------------------------------------------------------------
# σ-orbit chord ordering primitive
# ---------------------------------------------------------------------------

def test_sigma_orbit_pairs_grouped():
    """σ that swaps (0↔1, 2↔3) on 4 chords groups them into 2 orbits."""
    chords = [(0, 2), (1, 3), (0, 3), (1, 2)]
    sigma = {0: 1, 1: 0, 2: 3, 3: 2}
    ordered = _sigma_orbit_chord_order(chords, sigma)
    assert ordered.index((0, 2)) + 1 == ordered.index((1, 3))
    assert ordered.index((0, 3)) + 1 == ordered.index((1, 2))


def test_sigma_identity_preserves_order():
    """Identity σ leaves chord order unchanged."""
    chords = [(0, 1), (2, 3), (4, 5)]
    sigma = {i: i for i in range(6)}
    assert _sigma_orbit_chord_order(chords, sigma) == chords


def test_sigma_singleton_chord_kept():
    """A chord whose σ-image isn't in the set becomes a singleton orbit."""
    chords = [(0, 1), (2, 3)]
    sigma = {0: 10, 1: 11, 2: 2, 3: 3, 10: 0, 11: 1}
    assert set(_sigma_orbit_chord_order(chords, sigma)) == {(0, 1), (2, 3)}


def test_iterative_chord_rule_sigma_matches_baseline():
    """`_iterative_chord_rule` with σ-ordering matches the σ=None baseline
    (and both match the engine oracle) on 2×K_4 + 4-chord."""
    g = nx.Graph()
    for u in range(4):
        for v in range(u + 1, 4):
            g.add_edge(u, v)
            g.add_edge(4 + u, 4 + v)
    chords = [(0, 4), (1, 5), (2, 6), (3, 7)]
    g.add_edges_from(chords)
    G = Graph.from_networkx(g)

    engine = SynthesisEngine()
    oracle = engine.synthesize(G).polynomial

    engine._cache.clear()
    g_free, factors_a, adds_a = _iterative_chord_rule(
        G, chords, engine, smart_order=False, sigma=None,
    )
    no_sigma = _combine_chord_iteration(
        engine.synthesize(g_free).polynomial, factors_a, adds_a,
    )

    sigma = {i: i + 4 for i in range(4)}
    sigma.update({i + 4: i for i in range(4)})
    engine._cache.clear()
    g_free_s, factors_b, adds_b = _iterative_chord_rule(
        G, chords, engine, smart_order=False, sigma=sigma,
    )
    with_sigma = _combine_chord_iteration(
        engine.synthesize(g_free_s).polynomial, factors_b, adds_b,
    )

    assert no_sigma == oracle == with_sigma


def test_engine_chord_sigma_order_flag_on_by_default():
    """`engine.chord_sigma_order` defaults to True. Petersen sees a
    measured 1.44× speedup with bit-for-bit identical results."""
    engine = SynthesisEngine()
    assert engine.chord_sigma_order is True
    assert engine.chord_smart_order is True


# ---------------------------------------------------------------------------
# σ-equivariant treewidth DP on the cover G
# ---------------------------------------------------------------------------

def _eval_tutte_mod(g, x_v, y_v, p):
    """Evaluate networkx Tutte polynomial at (x_v, y_v) mod p."""
    poly = nx.tutte_polynomial(g)
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    return int(poly.subs({x_sym: x_v, y_sym: y_v})) % p


def _relabel_cube():
    """Return cube graph with integer-relabeled nodes 0..7."""
    g = nx.hypercube_graph(3)
    nm = {n: i for i, n in enumerate(sorted(g.nodes()))}
    return nx.relabel_nodes(g, nm)


def test_per_orbit_cube_antipodal():
    """Per-orbit DP processes σ-paired edges together → σ-canon works."""
    g = _relabel_cube()
    perm = {i: 7 - i for i in range(8)}
    nodes, edges = list(g.nodes()), list(g.edges())
    for x_v, y_v, p in [(2, 3, 1009), (5, 7, 2017), (-1, 4, 4019)]:
        expected = _eval_tutte_mod(g, x_v, y_v, p)
        actual, _ = compute_tutte_per_orbit_mod(nodes, edges, perm, x_v, y_v, p)
        assert actual == expected, f"({x_v},{y_v},mod {p}): {actual} != {expected}"


def test_per_orbit_c4():
    """Per-orbit DP on C_4 with σ=(02)(13)."""
    g = nx.cycle_graph(4)
    perm = {0: 2, 1: 3, 2: 0, 3: 1}
    nodes, edges = list(g.nodes()), list(g.edges())
    expected = _eval_tutte_mod(g, 2, 3, 1009)
    actual, _ = compute_tutte_per_orbit_mod(nodes, edges, perm, 2, 3, 1009)
    assert actual == expected


def test_per_orbit_c6():
    """Per-orbit DP on C_6 with rotation-by-3 σ."""
    g = nx.cycle_graph(6)
    perm = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}
    nodes, edges = list(g.nodes()), list(g.edges())
    expected = _eval_tutte_mod(g, 2, 3, 1009)
    actual, _ = compute_tutte_per_orbit_mod(nodes, edges, perm, 2, 3, 1009)
    assert actual == expected


