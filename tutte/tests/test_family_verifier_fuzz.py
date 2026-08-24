"""Phase I fuzz framework for family-recognition verifiers.

For each known family, generate random graphs by edge-swap perturbation
of the canonical family graph. The perturbed graph has the SAME degree
sequence (and bipartiteness, and other O(n+m) fingerprint properties)
but may have different topology. Then verify:

  engine.synthesize(perturbed) == nx.tutte_polynomial(perturbed)

This catches silent false-positive bugs where a verifier accepts a
non-family graph and returns the wrong polynomial — exactly the kind
of bug that hit `detect_grid_dims` (May 14 2026).

The seed is fixed per-test so failures are reproducible.
"""
from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

import networkx as nx
import pytest
import sympy

from tutte.graph import Graph
from tutte.lookup.core import load_default_table
from tutte.polynomial import TuttePolynomial
from tutte.synthesis.engine import SynthesisEngine

_engine = SynthesisEngine(table=load_default_table(), verbose=False)


def _t_via_nx(nxG: nx.Graph) -> sympy.Expr:
    return sympy.expand(nx.tutte_polynomial(nxG))


def _t_via_engine(nxG: nx.Graph) -> TuttePolynomial:
    G = Graph.from_networkx(nxG)
    return _engine.synthesize(G).polynomial


def _polys_match(eng: TuttePolynomial, nx_poly: sympy.Expr) -> bool:
    """Compare T at three points to detect engine-vs-nx divergence."""
    for x_val, y_val in [(2, 3), (-1, 5), (4, -2)]:
        e = int(eng.evaluate(x_val, y_val))
        n = int(nx_poly.subs({"x": x_val, "y": y_val}))
        if e != n:
            return False
    return True


def _double_edge_swap(g: nx.Graph, n_swaps: int, rng: random.Random,
                      max_attempts: int = 100) -> nx.Graph:
    """Apply n_swaps double-edge swaps to g, preserving degree sequence
    and connectivity. Returns a perturbed (possibly non-isomorphic) copy.
    """
    g = g.copy()
    edges = list(g.edges())
    n_done = 0
    attempts = 0
    while n_done < n_swaps and attempts < max_attempts * n_swaps:
        attempts += 1
        if len(edges) < 2:
            break
        e1, e2 = rng.sample(edges, 2)
        u1, v1 = e1
        u2, v2 = e2
        # Try swap: (u1, v1), (u2, v2) -> (u1, u2), (v1, v2)
        if len({u1, v1, u2, v2}) < 4:
            continue
        if g.has_edge(u1, u2) or g.has_edge(v1, v2):
            continue
        g.remove_edge(u1, v1)
        g.remove_edge(u2, v2)
        g.add_edge(u1, u2)
        g.add_edge(v1, v2)
        # Must remain connected
        if not nx.is_connected(g):
            g.remove_edge(u1, u2)
            g.remove_edge(v1, v2)
            g.add_edge(u1, v1)
            g.add_edge(u2, v2)
            continue
        edges = list(g.edges())
        n_done += 1
    return g


def _fuzz_family(canonical_builder: Callable[[], nx.Graph],
                  family_name: str, n_random: int = 10,
                  n_swaps: int = 3, seed: int = 12345) -> List[Tuple[nx.Graph, str]]:
    """Generate n_random perturbed graphs from the canonical family graph.

    Returns list of (graph, mismatch_reason or None) tuples — only those
    where engine ≠ nx are returned (i.e., bugs detected).
    """
    rng = random.Random(seed)
    failures = []
    base = canonical_builder()
    for i in range(n_random):
        perturbed = _double_edge_swap(base, n_swaps, rng)
        # Ensure graph isn't trivially isomorphic to base by checking
        # at least one edge changed
        if set(perturbed.edges()) == set(base.edges()):
            continue
        try:
            eng = _t_via_engine(perturbed)
            nx_poly = _t_via_nx(perturbed)
            if not _polys_match(eng, nx_poly):
                failures.append((
                    perturbed,
                    f"{family_name} fuzz #{i}: engine != nx.tutte_polynomial"
                ))
        except Exception as ex:
            failures.append((perturbed, f"{family_name} fuzz #{i}: exception {ex!r}"))
    return failures


# ===========================================================================
# Per-family canonical builders
# ===========================================================================

def _build_ladder_4() -> nx.Graph:
    """L_4 = P_4 × P_2 = 8 vertices, 10 edges (also 2×4 grid)."""
    return nx.grid_2d_graph(2, 4)


def _build_wheel_5() -> nx.Graph:
    return nx.wheel_graph(5)


def _build_prism_4() -> nx.Graph:
    """Prism C_4 × K_2 = cube Q_3."""
    return nx.cubical_graph()


def _build_helm_4() -> nx.Graph:
    """Helm H_4: hub + 4 rim + 4 pendants = 9 vertices, 12 edges."""
    g = nx.wheel_graph(4)  # hub + 4 rim, hub=0
    nodes = list(g.nodes())
    n = len(nodes)
    for i, rim in enumerate([1, 2, 3, 4]):
        g.add_edge(rim, n + i)
    return g


def _build_book_3() -> nx.Graph:
    """B_3 = 3 triangles sharing one edge."""
    g = nx.Graph()
    g.add_edge(0, 1)  # spine
    for i in range(3):
        v = 2 + i
        g.add_edge(0, v)
        g.add_edge(1, v)
    return g


def _build_pan_5() -> nx.Graph:
    """Pan P_5,1: cycle C_5 with one pendant."""
    g = nx.cycle_graph(5)
    g.add_edge(0, 5)
    return g


def _build_sunlet_4() -> nx.Graph:
    """Sunlet S_4: cycle C_4 with one pendant per cycle vertex."""
    g = nx.cycle_graph(4)
    for i in range(4):
        g.add_edge(i, 4 + i)
    return g


# ===========================================================================
# Fuzz tests — one per family
# ===========================================================================

# Smaller graphs for fuzz: nx.tutte_polynomial is slow on large graphs.
# Ladder/grid is the highest-priority test (the bug we just fixed).

def test_ladder_fuzz():
    failures = _fuzz_family(_build_ladder_4, "ladder L_4", n_random=15,
                            n_swaps=3, seed=42)
    assert not failures, "\n".join(f"{r}: {set(g.edges())}" for g, r in failures)


def test_wheel_fuzz():
    failures = _fuzz_family(_build_wheel_5, "wheel W_5", n_random=10,
                            n_swaps=3, seed=43)
    assert not failures, "\n".join(f"{r}: {set(g.edges())}" for g, r in failures)


def test_prism_fuzz():
    failures = _fuzz_family(_build_prism_4, "prism C_4xK_2", n_random=10,
                            n_swaps=3, seed=44)
    assert not failures, "\n".join(f"{r}: {set(g.edges())}" for g, r in failures)


def test_helm_fuzz():
    failures = _fuzz_family(_build_helm_4, "helm H_4", n_random=8,
                            n_swaps=2, seed=45)
    assert not failures, "\n".join(f"{r}: {set(g.edges())}" for g, r in failures)


def test_book_fuzz():
    failures = _fuzz_family(_build_book_3, "book B_3", n_random=8,
                            n_swaps=2, seed=46)
    assert not failures, "\n".join(f"{r}: {set(g.edges())}" for g, r in failures)


def test_pan_fuzz():
    failures = _fuzz_family(_build_pan_5, "pan P_5_1", n_random=8,
                            n_swaps=2, seed=47)
    assert not failures, "\n".join(f"{r}: {set(g.edges())}" for g, r in failures)


def test_sunlet_fuzz():
    failures = _fuzz_family(_build_sunlet_4, "sunlet S_4", n_random=8,
                            n_swaps=2, seed=48)
    assert not failures, "\n".join(f"{r}: {set(g.edges())}" for g, r in failures)


# ===========================================================================
# Specific regressions
# ===========================================================================

def test_k22_m2_chain_grid_regression():
    """The original bug: K_{2,2}+M_2 chain of 2 cells (8v, 10e, fingerprint
    matches L_4) was misidentified as 2x4 grid before May 14 2026 fix.
    This is the exact graph that triggered the discovery — keep it as
    a fuzz-anchor sample to ensure the fix doesn't regress.
    """
    nxG = nx.Graph()
    nxG.add_edges_from([
        (0, 2), (0, 3), (1, 2), (1, 3),
        (4, 6), (4, 7), (5, 6), (5, 7),
        (2, 4), (3, 5),
    ])
    eng = _t_via_engine(nxG)
    nx_poly = _t_via_nx(nxG)
    assert _polys_match(eng, nx_poly), \
        f"K_{{2,2}}+M_2 chain: engine != nx oracle"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
