"""Bootstrap utilities for building the rainbow table.

Provides sympy conversion and basic table builder that seeds
known polynomials and synthesizes additional entries.
"""

from __future__ import annotations

from ..polynomial import TuttePolynomial
from .core import RainbowTable


# =============================================================================
# SYMPY CONVERSION
# =============================================================================

def sympy_to_tutte(nx_poly) -> TuttePolynomial:
    """Convert networkx/sympy polynomial to TuttePolynomial."""
    coeffs = {}

    # Handle different sympy types
    if hasattr(nx_poly, 'as_dict'):
        poly_dict = nx_poly.as_dict()
    elif hasattr(nx_poly, 'as_poly'):
        poly_dict = nx_poly.as_poly().as_dict()
    else:
        # Try to convert to Poly first
        from sympy import Poly, symbols
        x, y = symbols('x y')
        try:
            poly = Poly(nx_poly, x, y)
            poly_dict = poly.as_dict()
        except Exception:
            # Single term or constant
            poly_dict = {(0, 0): int(nx_poly)}

    for monom, coeff in poly_dict.items():
        if len(monom) >= 2:
            coeffs[(monom[0], monom[1])] = int(coeff)
        elif len(monom) == 1:
            coeffs[(monom[0], 0)] = int(coeff)
        else:
            coeffs[(0, 0)] = int(coeff)

    return TuttePolynomial.from_coefficients(coeffs)


# =============================================================================
# BASIC TABLE BUILDER
# =============================================================================

def build_basic_table() -> RainbowTable:
    """Build rainbow table using closed-form formulas and self-synthesis.

    Phase 1: Seed with closed-form polynomials (no computation needed)
      - Complete graphs K_2 through K_5
      - Cycle graphs C_3 through C_12
      - Path graphs P_2 through P_10
      - Star graphs S_3 through S_8

    Phase 2: Use the synthesis engine with the seeded table to compute
      - Complete graphs K_6 through K_8
      - Wheel graphs W_4 through W_8
      - Petersen graph
      - Small grid graphs
    """
    import networkx as nx

    from ..graph import (Graph, complete_graph, cycle_graph, path_graph, star_graph,
                         wheel_graph)

    table = RainbowTable()

    # === Phase 1: Closed-form polynomials ===

    # Complete graphs K_n: well-known polynomials
    k_polys = {
        2: {(1, 0): 1},  # x
        3: {(2, 0): 1, (1, 0): 1, (0, 1): 1},  # x^2 + x + y
        4: {(3, 0): 1, (2, 0): 3, (1, 1): 4, (1, 0): 2, (0, 1): 2, (0, 2): 3, (0, 3): 1},
        5: {(4, 0): 1, (3, 0): 6, (2, 1): 10, (2, 0): 11, (1, 1): 20, (1, 2): 15,
            (1, 3): 5, (1, 0): 6, (0, 1): 6, (0, 2): 15, (0, 3): 15, (0, 4): 10,
            (0, 5): 4, (0, 6): 1},
    }

    for n in range(2, 6):
        g = complete_graph(n)
        poly = TuttePolynomial.from_coefficients(k_polys[n])
        table.add(g, f"K_{n}", poly)

    # Cycle graphs C_n: T(C_n) = x^{n-1} + x^{n-2} + ... + x + y
    for n in range(3, 13):
        g = cycle_graph(n)
        coeffs = {(i, 0): 1 for i in range(1, n)}
        coeffs[(0, 1)] = 1
        poly = TuttePolynomial.from_coefficients(coeffs)
        table.add(g, f"C_{n}", poly)

    # Path graphs P_n: T(P_n) = x^{n-1}
    for n in range(2, 11):
        g = path_graph(n)
        poly = TuttePolynomial.x(n - 1)
        table.add(g, f"P_{n}", poly)

    # Star graphs S_n: T(S_n) = x^n (n bridges)
    for n in range(3, 9):
        g = star_graph(n)
        poly = TuttePolynomial.x(n)
        table.add(g, f"S_{n}", poly)

    # === Phase 2: Synthesize remaining graphs using the seeded table ===

    from ..synthesis.engine import SynthesisEngine
    engine = SynthesisEngine(table=table, verbose=False)

    # Complete graphs K_6, K_7, K_8
    for n in range(6, 9):
        g = complete_graph(n)
        result = engine.synthesize(g)
        table.add(g, f"K_{n}", result.polynomial)

    # Wheel graphs W_n (n spokes + hub)
    for n in range(4, 9):
        G_nx = nx.wheel_graph(n)
        g = Graph.from_networkx(G_nx)
        result = engine.synthesize(g)
        table.add(g, f"W_{n}", result.polynomial)

    # Petersen graph
    g_pet = Graph.from_networkx(nx.petersen_graph())
    result = engine.synthesize(g_pet)
    table.add(g_pet, "Petersen", result.polynomial)

    # Small grid graphs
    for rows in range(2, 4):
        for cols in range(2, 5):
            G_grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, cols))
            g_grid = Graph.from_networkx(G_grid)
            if G_grid.number_of_edges() <= 12:
                result = engine.synthesize(g_grid)
                table.add(g_grid, f"Grid_{rows}x{cols}", result.polynomial)

    return table
