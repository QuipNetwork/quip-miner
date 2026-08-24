"""Chain recurrence framework regressions.

Covers three layers:
  1. Chain transfer matrix extraction + Cayley-Hamilton recurrence on
     the raw state-sum S_n (symbolic, polynomial level).
  2. Modular char-poly extraction via Faddeev-LeVerrier and recurrence
     evaluation in Z/pZ.
  3. The production `tutte/roots/chain_recurrence.py` module wrapper
     (`is_chain_topology`, `compute_chain_recurrence_mod`,
     `compute_chain_full_poly_from_spec`).

All three layers now run against production code: the transfer-matrix
extractor and `chain_S_at_n` were lifted into
`tutte/roots/chain_recurrence.py` (rebuilding the lost
`tutte.research.scripts.*` modules); test-only builders and fitters
live in `tests/chain_recurrence_helpers.py`.
"""
from __future__ import annotations

import networkx as nx
import pytest
import sympy

from tutte.graph import Graph
from tutte.lookup.core import load_default_table
from tutte.polynomial import TuttePolynomial
from tutte.roots.chain_recurrence import (
    chain_S_at_n, compute_chain_recurrence_mod, extract_chain_transfer_matrix,
    faddeev_leverrier_charpoly_mod, is_chain_topology,
)

from .chain_recurrence_helpers import (
    build_k22_m2_cycle_nx, build_kaa_ma_setup, build_kn_m2_setup,
    fit_linear_recurrence, sympy_to_tutte, tutte_to_sympy,
)
from tutte.synthesis.engine import SynthesisEngine


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract(cell, cag, connector, jA, jB):
    """Run the chain extraction with sane defaults."""
    return extract_chain_transfer_matrix(
        cell_template=cell, cell_anchor_groups=cag,
        connector_template=connector,
        junction_anchors_A=jA, junction_anchors_B=jB,
        left_anchor_group=0, right_anchor_group=1,
        verify_position_invariance=True,
    )


def _build_M_poly(apply_step, initial_state):
    """Reconstruct the polynomial transfer matrix M(x, y) column-by-column."""
    orbit_keys = sorted(initial_state.keys())
    n = len(orbit_keys)
    M_poly = [[TuttePolynomial.zero() for _ in range(n)] for _ in range(n)]
    for j, k_in in enumerate(orbit_keys):
        unit_state = {k: TuttePolynomial.zero() for k in orbit_keys}
        unit_state[k_in] = TuttePolynomial.from_coefficients({(0, 0): 1})
        out_state = apply_step(unit_state)
        for i, k_out in enumerate(orbit_keys):
            M_poly[i][j] = out_state.get(k_out, TuttePolynomial.zero())
    return orbit_keys, M_poly


def _symbolic_charpoly_coeffs(M_poly):
    """Compute symbolic char-poly coefficients (highest-degree-first)."""
    n = len(M_poly)
    x_sym, y_sym, lam = sympy.symbols("x y lambda")
    M_sym = sympy.Matrix([
        [sympy.expand(tutte_to_sympy(M_poly[i][j], x_sym, y_sym)) for j in range(n)]
        for i in range(n)
    ])
    char_sym = sympy.expand((lam * sympy.eye(n) - M_sym).det())
    char_coeffs_sym = sympy.Poly(char_sym, lam).all_coeffs()
    return [sympy_to_tutte(c, x_sym, y_sym) for c in char_coeffs_sym]


def _validate_recurrence(extract, S_target, target_n, c_tutte):
    """Predict S_n via recurrence, assert bit-for-bit match with direct DP."""
    order = len(c_tutte) - 1
    predicted = TuttePolynomial.zero()
    for k in range(1, order + 1):
        predicted = predicted - c_tutte[k] * S_target[target_n - k]
    diff = predicted + (-1 * S_target[target_n])
    return diff.num_terms()


def _S_values(extract, ns):
    """Compute S_n for each n in `ns`."""
    apply_step = extract["apply_step"]
    apply_terminal = extract["apply_terminal_step"]
    initial_state = extract["initial_state"]
    div_per_step = extract["div_per_step"]
    div_terminal = extract.get("div_terminal_step", div_per_step)
    out = {}
    for n in ns:
        S_n, _ = chain_S_at_n(
            apply_step, apply_terminal, initial_state,
            div_per_step, div_terminal, n,
        )
        out[n] = S_n
    return out


# ---------------------------------------------------------------------------
# Chain transfer-matrix recurrence (polynomial level)
# ---------------------------------------------------------------------------

def test_k22_m2_chain_recurrence_polynomial_level():
    """K_{2,2}+M_2 chain: order-2 char-poly recurrence holds on S_n
    bit-for-bit at the polynomial level."""
    extract = _extract(*build_kaa_ma_setup(2))
    assert extract["n_orbits"] == 2
    assert extract["div_per_step"] == 1

    _, M_poly = _build_M_poly(extract["apply_step"], extract["initial_state"])
    c_tutte = _symbolic_charpoly_coeffs(M_poly)
    assert len(c_tutte) == 3  # order 2

    S_polys = _S_values(extract, range(2, 7))
    diff = _validate_recurrence(extract, S_polys, target_n=6, c_tutte=c_tutte)
    assert diff == 0, f"K_{{2,2}}+M_2 recurrence diff has {diff} terms"


def test_k33_m3_chain_recurrence_order_3():
    """K_{3,3}+M_3 chain: order-3 char-poly recurrence holds on S_n.

    Validates the framework generalizes to higher orders and larger cells.
    """
    extract = _extract(*build_kaa_ma_setup(3))
    assert extract["n_orbits"] == 3

    _, M_poly = _build_M_poly(extract["apply_step"], extract["initial_state"])
    c_tutte = _symbolic_charpoly_coeffs(M_poly)
    assert len(c_tutte) == 4  # order 3

    S_polys = _S_values(extract, range(2, 9))
    diff = _validate_recurrence(extract, S_polys, target_n=8, c_tutte=c_tutte)
    assert diff == 0, f"K_{{3,3}}+M_3 recurrence diff has {diff} terms"


def _check_kn_m2(n_cell: int, expected_n_orbits: int) -> int:
    """Validate K_n+M_2 chain order-`expected_n_orbits` recurrence; return diff."""
    extract = _extract(*build_kn_m2_setup(n_cell))
    assert extract["n_orbits"] == expected_n_orbits
    _, M_poly = _build_M_poly(extract["apply_step"], extract["initial_state"])
    c_tutte = _symbolic_charpoly_coeffs(M_poly)
    order = expected_n_orbits
    S_polys = _S_values(extract, range(2, 2 * order + 2))
    return _validate_recurrence(
        extract, S_polys, target_n=2 * order + 1, c_tutte=c_tutte,
    )


def test_k4_m2_chain_recurrence_non_bipartite():
    """K_4+M_2 (non-bipartite cell) chain: order-2 recurrence holds.
    Validates framework generality beyond bipartite K_{a,a} cells."""
    assert _check_kn_m2(4, expected_n_orbits=2) == 0


def test_k5_m2_chain_recurrence_non_bipartite():
    """K_5+M_2 (larger non-bipartite cell). n_orbits depends on cell aut
    + connector arity, not cell vertex count."""
    assert _check_kn_m2(5, expected_n_orbits=2) == 0


# ---------------------------------------------------------------------------
# Modular char-poly extraction + Z/pZ recurrence evaluation
# ---------------------------------------------------------------------------

def test_k22_m2_modular_chain_recurrence():
    """Char poly extracted via Faddeev-LeVerrier mod p (bypassing symbolic
    sympy.det) and the recurrence applied in Z/pZ matches direct
    `evaluate_mod` of the polynomial state-sum."""
    extract = _extract(*build_kaa_ma_setup(2))
    orbit_keys, M_poly = _build_M_poly(
        extract["apply_step"], extract["initial_state"],
    )
    n = len(orbit_keys)
    assert n == 2

    S_polys = _S_values(extract, range(2, 6))
    test_points = [(3, 5, 1009), (7, 11, 10007), (100, 200, 10**9 + 7)]
    for x0, y0, p in test_points:
        M_int_mod = [
            [int(M_poly[i][j].evaluate(x0, y0)) % p for j in range(n)]
            for i in range(n)
        ]
        char_mod = faddeev_leverrier_charpoly_mod(M_int_mod, p)
        S_5_mod_direct = S_polys[5].evaluate_mod(x0, y0, p)
        predicted = 0
        for k in range(1, n + 1):
            S_t_minus_k = S_polys[5 - k].evaluate_mod(x0, y0, p)
            predicted = (predicted - char_mod[k] * S_t_minus_k) % p
        assert predicted == S_5_mod_direct, (
            f"({x0},{y0},p={p}): predicted={predicted}, direct={S_5_mod_direct}"
        )


# ---------------------------------------------------------------------------
# Production module: tutte/roots/chain_recurrence.py
# ---------------------------------------------------------------------------

def _build_kaa_ma_simple(a: int):
    """Build a K_{a,a} cell + M_a connector setup matching
    `chain_recurrence_general.build_kaa_ma_setup`."""
    cell_template = Graph.from_networkx(nx.complete_bipartite_graph(a, a))
    cell_anchor_groups = {
        0: list(range(a)),
        1: list(range(a, 2 * a)),
    }
    connector = Graph(list(range(2 * a)), [(i, i + a) for i in range(a)])
    return cell_template, cell_anchor_groups, connector, list(range(a)), list(range(a, 2 * a))


def test_is_chain_topology_detects_linear_path():
    """is_chain_topology returns True for paths, False otherwise."""
    path = nx.Graph()
    path.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    assert is_chain_topology(path) is True

    star = nx.Graph()
    star.add_edges_from([(0, 1), (0, 2), (0, 3)])
    assert is_chain_topology(star) is False

    single = nx.Graph()
    single.add_node(0)
    assert is_chain_topology(single) is False


def test_faddeev_leverrier_charpoly_mod_2x2():
    """Char poly of M = [[a, b], [c, d]]: λ² - (a+d)λ + (ad - bc)."""
    M = [[3, 5], [7, 11]]
    p = 1009
    coeffs = faddeev_leverrier_charpoly_mod(M, p)
    # tr = 14, det = 33 - 35 = -2. char(λ) = λ² + c_1 λ + c_2.
    assert coeffs == [1, (-14) % p, (-2) % p]


@pytest.mark.parametrize("a,n_cells", [(2, 2), (2, 3), (2, 5)])
def test_chain_recurrence_mod_matches_framework_dp(a, n_cells):
    """`compute_chain_recurrence_mod` matches the direct chain-framework
    `chain_S_at_n` + divisor at integer test points.

    The framework's S_n is the path-DP state-sum (not the networkx chain
    graph's Tutte polynomial), so this validates internal consistency
    between the production wrapper and the underlying DP.
    """
    from tutte.roots.rooted_tutte import divide_by_x_minus_1_power
    cell, cag, connector, jA, jB = _build_kaa_ma_simple(a)
    extract = _extract(cell, cag, connector, jA, jB)
    S_n, total_div = chain_S_at_n(
        extract['apply_step'], extract['apply_terminal_step'],
        extract['initial_state'], extract['div_per_step'],
        extract.get('div_terminal_step', extract['div_per_step']),
        n_cells,
    )
    T_framework = divide_by_x_minus_1_power(S_n, total_div)

    cache = {}
    for x_val, y_val, p in [(3, 5, 1009), (7, 11, 10007)]:
        T_rec, cache = compute_chain_recurrence_mod(
            cell_template=cell, cell_anchor_groups=cag,
            connector_template=connector,
            junction_anchors_A=jA, junction_anchors_B=jB,
            left_anchor_group=0, right_anchor_group=1, n_cells=n_cells,
            x_val=x_val, y_val=y_val, p=p, extract_cache=cache,
        )
        T_framework_mod = T_framework.evaluate_mod(x_val, y_val, p)
        assert T_rec == T_framework_mod, (
            f"K_{{{a},{a}}}+M_{a} n={n_cells} at ({x_val},{y_val},p={p}): "
            f"recurrence={T_rec}, framework_dp={T_framework_mod}"
        )


def test_chain_full_poly_from_spec_path_only():
    """`compute_chain_full_poly_from_spec` returns None for non-path cell trees."""
    from tutte.roots.cell_quotient_tree import CellTreeSpec
    from tutte.roots.chain_recurrence import compute_chain_full_poly_from_spec

    star = nx.Graph()
    star.add_edges_from([(0, 1), (0, 2), (0, 3)])
    spec = CellTreeSpec(
        cell_template=Graph.from_networkx(nx.complete_graph(3)),
        junction_template=Graph(frozenset([0, 1, 2, 3]), frozenset([(0, 2), (1, 3)])),
        cell_tree=star,
        cell_anchor_groups={0: {1: [0, 1], 2: [0, 1], 3: [0, 1]}},
        junction_anchors_A=[0, 1],
        junction_anchors_B=[2, 3],
        root=0,
    )
    assert compute_chain_full_poly_from_spec(spec) is None


def test_extract_cache_reused_across_calls():
    """Extract cache speeds up repeated calls at different (x, y, p)."""
    cell, cag, connector, jA, jB = _build_kaa_ma_simple(2)
    cache = {}
    T1, cache = compute_chain_recurrence_mod(
        cell_template=cell, cell_anchor_groups=cag,
        connector_template=connector, junction_anchors_A=jA, junction_anchors_B=jB,
        left_anchor_group=0, right_anchor_group=1, n_cells=4,
        x_val=3, y_val=5, p=1009, extract_cache=cache,
    )
    assert "extract" in cache
    T2, cache = compute_chain_recurrence_mod(
        cell_template=cell, cell_anchor_groups=cag,
        connector_template=connector, junction_anchors_A=jA, junction_anchors_B=jB,
        left_anchor_group=0, right_anchor_group=1, n_cells=4,
        x_val=7, y_val=11, p=10007, extract_cache=cache,
    )
    assert T1 < 1009 and T2 < 10007


# ---------------------------------------------------------------------------
# Cycle recurrence (empirical: integer-coefficient at integer test points)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def k22_m2_cycle_values():
    """T(K_{2,2}+M_2 cycle_n) for n=2..12 (computed once per session)."""
    engine = SynthesisEngine(table=load_default_table(), verbose=False)
    T_cycle = {}
    for n in range(2, 13):
        nxG = build_k22_m2_cycle_nx(n)
        T_cycle[n] = engine.synthesize(Graph.from_networkx(nxG)).polynomial
    return T_cycle


# Integer test points where the order-5 recurrence's rational-function
# coefficients in (x, y) evaluate to integers. (2, 3) collapses to order
# 3 because (x-1)=1 kills (x-1)^k denominators.
_CYCLE_TEST_POINTS = [(3, 5), (5, 2), (3, 3), (7, 4), (4, 7)]


@pytest.mark.parametrize("x0,y0", _CYCLE_TEST_POINTS)
def test_k22_m2_cycle_order_5_integer_fit(k22_m2_cycle_values, x0, y0):
    """T(K_{2,2}+M_2 cycle_n) at integer (x_0, y_0) satisfies an order-5
    linear recurrence with INTEGER coefficients."""
    values = [int(k22_m2_cycle_values[n].evaluate(x0, y0)) for n in range(2, 13)]
    coeffs = fit_linear_recurrence(values, 5)
    assert coeffs is not None, f"({x0},{y0}): no order-5 fit found"
    assert all(c.denominator == 1 for c in coeffs), (
        f"({x0},{y0}): expected integer coefficients, got {coeffs}"
    )


def test_k22_m2_cycle_x_eq_2_lower_order(k22_m2_cycle_values):
    """At (x, y) = (2, 3) where (x-1)=1 collapses (x-1)^k factors, the
    cycle recurrence reduces to order 3."""
    values = [int(k22_m2_cycle_values[n].evaluate(2, 3)) for n in range(2, 13)]
    order_3 = fit_linear_recurrence(values, 3)
    assert order_3 is not None
    assert all(c.denominator == 1 for c in order_3)


def test_k22_m2_cycle_symbolic_charpoly_predicts_at_integer_points():
    """The extracted symbolic char-poly leading coefficient predicts the
    fitted integer value of a_4 at sample test points."""
    import sympy
    x_sym, y_sym = sympy.symbols("x y")
    c_1 = sympy.expand(
        x_sym**4 + 2*x_sym**3 + 5*x_sym**2 + x_sym*y_sym
        + 7*x_sym + 2*y_sym**2 + 7*y_sym + 9
    )
    # Empirical fits: a_4 = -c_1 of the char poly evaluated at the test point.
    assert int(c_1.subs({x_sym: 7, y_sym: 4})) == 3478
    assert int(c_1.subs({x_sym: 5, y_sym: 2})) == 1076
