"""Test helpers for the chain-recurrence framework.

Rebuilt replacements for the lost `tutte.research.scripts.*` utilities
(chain_recurrence_general, chain_recurrence_cycle_probe,
chain_recurrence_cycle_fit) that `test_chain_recurrence.py` depends on.
The transfer-matrix extractor itself now lives in production at
`tutte.roots.chain_recurrence.extract_chain_transfer_matrix`.
"""
from __future__ import annotations

from typing import List, Optional

import networkx as nx
import sympy

from tutte.graph import Graph
from tutte.polynomial import TuttePolynomial


# ---------------------------------------------------------------------------
# Chain template builders
# ---------------------------------------------------------------------------

def build_kaa_ma_setup(a: int):
    """K_{a,a} cell + M_a connector chain template.

    Returns (cell_template, cell_anchor_groups, connector_template,
    junction_anchors_A, junction_anchors_B); the left/right anchor groups
    are 0 and 1 (shore A / shore B of the bipartite cell).
    """
    cell_template = Graph.from_networkx(nx.complete_bipartite_graph(a, a))
    cell_anchor_groups = {
        0: list(range(a)),
        1: list(range(a, 2 * a)),
    }
    connector = Graph(list(range(2 * a)), [(i, i + a) for i in range(a)])
    return (
        cell_template, cell_anchor_groups, connector,
        list(range(a)), list(range(a, 2 * a)),
    )


def build_kn_m2_setup(n_cell: int):
    """K_n (non-bipartite) cell + M_2 connector chain template.

    Anchor group 0 = vertices [0, 1] (left side), group 1 = [2, 3]
    (right side); remaining vertices are interior.
    """
    if n_cell < 4:
        raise ValueError(f"K_n+M_2 template needs n >= 4, got {n_cell}")
    cell_template = Graph.from_networkx(nx.complete_graph(n_cell))
    cell_anchor_groups = {0: [0, 1], 1: [2, 3]}
    connector = Graph([0, 1, 2, 3], [(0, 2), (1, 3)])
    return cell_template, cell_anchor_groups, connector, [0, 1], [2, 3]


# ---------------------------------------------------------------------------
# TuttePolynomial <-> sympy conversion
# ---------------------------------------------------------------------------

def tutte_to_sympy(poly: TuttePolynomial, x_sym, y_sym):
    """Convert a TuttePolynomial to a sympy expression in (x_sym, y_sym)."""
    expr = sympy.Integer(0)
    for (i, j), c in poly.to_coefficients().items():
        expr += c * x_sym ** i * y_sym ** j
    return expr


def sympy_to_tutte(expr, x_sym, y_sym) -> TuttePolynomial:
    """Convert a sympy polynomial in (x_sym, y_sym) with integer
    coefficients to a TuttePolynomial."""
    expr = sympy.expand(expr)
    if expr == 0:
        return TuttePolynomial.zero()
    poly = sympy.Poly(expr, x_sym, y_sym)
    coeffs = {}
    for (i, j), c in zip(poly.monoms(), poly.coeffs()):
        coeffs[(int(i), int(j))] = int(c)
    return TuttePolynomial.from_coefficients(coeffs)


# ---------------------------------------------------------------------------
# Cycle probe: K_{2,2} + M_2 cycle graphs
# ---------------------------------------------------------------------------

def build_k22_m2_cycle_nx(n: int) -> nx.Graph:
    """Cycle of n K_{2,2} cells joined by M_2 connectors (closed chain).

    Cell k occupies vertices 4k..4k+3 with shores {4k, 4k+1} and
    {4k+2, 4k+3}; the M_2 connector matches cell k's right shore to
    cell (k+1) mod n's left shore.
    """
    if n < 2:
        raise ValueError(f"cycle needs >= 2 cells, got {n}")
    G = nx.Graph()
    for k in range(n):
        base = 4 * k
        for i in range(2):
            for j in range(2):
                G.add_edge(base + i, base + 2 + j)
    for k in range(n):
        nxt = 4 * ((k + 1) % n)
        G.add_edge(4 * k + 2, nxt)
        G.add_edge(4 * k + 3, nxt + 1)
    return G


# ---------------------------------------------------------------------------
# Linear recurrence fitting (exact, rational)
# ---------------------------------------------------------------------------

def fit_linear_recurrence(
    values: List[int], order: int,
) -> Optional[List[sympy.Rational]]:
    """Fit v_t = a_1 v_{t-1} + ... + a_order v_{t-order} exactly.

    Solves the linear system over the rationals using the given value
    window and verifies the fit on EVERY available equation. Returns
    [a_1, ..., a_order] as sympy Rationals, or None if no exact
    order-`order` recurrence fits.
    """
    if len(values) < 2 * order:
        raise ValueError(
            f"need >= {2 * order} values to fit + verify order {order}, "
            f"got {len(values)}"
        )
    rows = []
    rhs = []
    for t in range(order, len(values)):
        rows.append([sympy.Integer(values[t - k]) for k in range(1, order + 1)])
        rhs.append(sympy.Integer(values[t]))

    A_sq = sympy.Matrix(rows[:order])
    b_sq = sympy.Matrix(rhs[:order])
    if A_sq.det() != 0:
        sol = A_sq.solve(b_sq)
    else:
        try:
            sol, params = sympy.Matrix(rows).gauss_jordan_solve(sympy.Matrix(rhs))
        except ValueError:
            return None
        if params:
            sol = sol.subs({p: 0 for p in params})

    coeffs = [sympy.Rational(sol[i]) for i in range(order)]
    for row, target in zip(rows, rhs):
        if sum(c * v for c, v in zip(coeffs, row)) != target:
            return None
    return coeffs
