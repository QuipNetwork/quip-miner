"""Builders for graph-family recurrence seeds.

Shared between:
  - `tutte/scripts/warmup_lookup_table.py` (bulk seed population)
  - `tutte/family_recognition/constants.py` (compute-on-demand fallback when
    the rainbow table is missing a seed)

All builders return a `tutte.graph.Graph` instance.
"""

from __future__ import annotations

import networkx as nx

from ..graph import Graph, grid_graph, path_graph, wheel_graph


def book_graph(k: int) -> Graph:
    """Book graph B_k: k triangles sharing edge (0, 1)."""
    G = nx.Graph()
    G.add_edge(0, 1)
    for i in range(k):
        v = i + 2
        G.add_edge(0, v)
        G.add_edge(1, v)
    return Graph.from_networkx(G)


def gear_graph(k: int) -> Graph:
    """Gear graph G_k: hub + k rim vertices + k subdivision vertices."""
    G = nx.Graph()
    for i in range(k):
        G.add_edge(0, i + 1)
        G.add_edge(i + 1, k + 1 + i)
        G.add_edge(k + 1 + i, (i + 1) % k + 1)
    return Graph.from_networkx(G)


def prism_graph(k: int) -> Graph:
    """Prism graph CL_k = C_k × K_2 (circular ladder)."""
    return Graph.from_networkx(nx.circular_ladder_graph(k))


def mobius_graph(k: int) -> Graph:
    """Möbius ladder M_k: 2k-cycle with k rungs connecting v_i to v_{i+k}."""
    G = nx.cycle_graph(2 * k)
    for i in range(k):
        G.add_edge(i, i + k)
    return Graph.from_networkx(G)


# Mapping from rainbow-table name to a no-arg builder. Used by the
# compute-on-demand fallback in `constants.py`.
SEED_BUILDERS: dict = {
    "K_2": lambda: path_graph(2),                                 # F_1 single edge
    "K_3": lambda: Graph.from_networkx(nx.complete_graph(3)),     # F_2 = B_1 = triangle
    "K_4": lambda: Graph.from_networkx(nx.complete_graph(4)),     # W_3 = K_4
    "C_4": lambda: Graph.from_networkx(nx.cycle_graph(4)),        # L_2
    "W_4": lambda: wheel_graph(4),
    "W_5": lambda: wheel_graph(5),
    "B_2": lambda: book_graph(2),
    "Gear_3": lambda: gear_graph(3),
    "Gear_4": lambda: gear_graph(4),
    "Gear_5": lambda: gear_graph(5),
    "Grid_2x3": lambda: grid_graph(2, 3),                          # L_3
    "Prism_3": lambda: prism_graph(3),
    "Prism_4": lambda: prism_graph(4),
    "Prism_5": lambda: prism_graph(5),
    "Prism_6": lambda: prism_graph(6),
    "Prism_7": lambda: prism_graph(7),
    "Prism_8": lambda: prism_graph(8),
    "Mobius_3": lambda: mobius_graph(3),
    "Mobius_4": lambda: mobius_graph(4),
    "Mobius_5": lambda: mobius_graph(5),
    "Mobius_6": lambda: mobius_graph(6),
    "Mobius_7": lambda: mobius_graph(7),
    "Mobius_8": lambda: mobius_graph(8),
}
