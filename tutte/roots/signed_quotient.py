"""Live σ-finder for the signed-quotient / σ-equivariant chord-ordering path.

`find_best_sigma` is the only live export of this module: it is used by the
engine's chord-peel decomposition (`synthesis/engine.py`) and by the
σ-equivariant chord ordering in `graphs/k_sum.py`.

The (test-only) signed-DP-via-interpolation pipeline that used to live here was
moved to `tutte/deprecated/signed_quotient_pipeline.py` during the 2026-05
dead-code cleanup (it was never on a live engine path).
"""
from __future__ import annotations

from typing import Dict, Optional

import networkx as nx


def find_best_sigma(
    g: nx.Graph,
    require_free: bool = False,
) -> Optional[Dict[int, int]]:
    """Search for a free order-2 graph automorphism σ of `g`.

    Tries a small set of structural candidates (i+n/2, reverse, cell-swap
    pairings) and returns the FIRST one that's a valid order-2 automorphism.
    Returns the σ with the FEWEST σ-fixed edges (free σ preferred — gives
    cleaner DP without per-edge loop factors).

    Args:
      g: networkx graph with INT vertex labels 0..n-1.
      require_free: if True, only accept σ with zero σ-fixed edges. Returns
                    None if no free σ found.

    Returns:
      Dict v → σ(v), or None if no valid σ candidate matches.
    """
    nodes = list(g.nodes())
    if any(not isinstance(v, int) for v in nodes):
        return None
    n_v = max(nodes) + 1
    if n_v < 2:
        return None

    edges_set = set(tuple(sorted((int(u), int(v)))) for u, v in g.edges())

    candidates = [
        ('i+n/2', lambda i: (i + n_v // 2) % n_v),
        ('cell-swap ±2', lambda i: i + 2 if (i // 2) % 2 == 0 else i - 2),
        ('cell-swap ±4', lambda i: i + 4 if (i // 4) % 2 == 0 else i - 4),
        ('cell-swap ±8', lambda i: i + 8 if (i // 8) % 2 == 0 else i - 8),
        ('reverse',     lambda i: n_v - 1 - i),
    ]
    best_perm = None
    best_fixed = float('inf')
    for name, fn in candidates:
        perm = {}
        valid = True
        for i in range(n_v):
            j = fn(i)
            if not (0 <= j < n_v):
                valid = False; break
            perm[i] = j
        if not valid:
            continue
        # Check order-2, free on vertices, is_aut.
        if not all(perm[v] != v for v in range(n_v)):
            continue
        if not all(perm[perm[v]] == v for v in range(n_v)):
            continue
        if not all(tuple(sorted((perm[u], perm[v]))) in edges_set for u, v in g.edges()):
            continue
        fixed_edges = sum(
            1 for u, v in g.edges() if sorted([perm[u], perm[v]]) == sorted([u, v])
        )
        if require_free and fixed_edges > 0:
            continue
        if fixed_edges < best_fixed:
            best_fixed = fixed_edges
            best_perm = perm
            if fixed_edges == 0:
                # Free σ found; prefer it — short-circuit.
                return best_perm
    return best_perm

