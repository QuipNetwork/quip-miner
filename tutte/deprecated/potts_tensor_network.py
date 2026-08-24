"""Potts-model tensor network evaluation of Z(G; q, v) at integer points.

Uses the Fortuin-Kasteleyn / random-cluster identity to express the
multivariate Tutte (Sokal Z) polynomial as a q-Potts partition function:

  Z(G; q, v) = Σ_{σ: V → [q]} Π_{e=(u,v) ∈ E} [1 + v · δ(σ_u, σ_v)]

This is a tensor network with bond dimension `q`:
- For each edge e = (u, v), an edge tensor `M_e[σ_u, σ_v] = 1 + v·δ`.
- Vertex `v` corresponds to an index variable shared across all
  incident edge tensors.

opt_einsum finds a good contraction order; cost is ~O(n · q^tw_eff).

Use case: modular evaluation of T(G; x, y) at many (q, v) points, then
Lagrange-interpolate to recover the bivariate polynomial. The exact
polynomial path uses integer arithmetic with q = (x-1)(y-1), v = y-1
for sampled integer (x, y).
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from ..graph import Graph


def _edge_tensor(q: int, v_val: int, dtype=np.int64) -> np.ndarray:
    """M[a, b] = 1 + v if a == b else 1, shape (q, q)."""
    M = np.ones((q, q), dtype=dtype)
    for k in range(q):
        M[k, k] = 1 + v_val
    return M


def potts_partition_function_einsum(
    graph: Graph,
    q: int,
    v_val: int,
    dtype=np.int64,
    optimize: str = "auto",
) -> int:
    """Compute Z(graph; q, v_val) via opt_einsum contraction.

    Args:
        graph: input graph (simple, no loops, no multi-edges).
        q: integer Potts state count (= (x-1)(y-1) at evaluation point).
        v_val: integer FK weight (= y-1).
        dtype: numpy dtype for intermediate tensors. int64 is safe for
            small q and small graphs; switch to object dtype for large
            results that overflow int64.
        optimize: opt_einsum optimization strategy. "auto" picks based
            on tensor count; "optimal" runs DP search (slow but best).

    Returns:
        Z(graph; q, v_val) as a Python int.
    """
    import opt_einsum as oe

    edges = sorted(graph.edges)
    n_edges = len(edges)
    if n_edges == 0:
        # Z(empty graph; q, v) = q^|V|
        return q ** graph.node_count()

    # Each edge tensor M_e has 2 indices labeled by the two endpoint vertex
    # IDs. opt_einsum uses arbitrary hashable index labels.
    tensors: List[np.ndarray] = []
    index_lists: List[Tuple[int, int]] = []
    for (u, v) in edges:
        tensors.append(_edge_tensor(q, v_val, dtype=dtype))
        index_lists.append((u, v))

    # Build the opt_einsum args list: tensor, indices, tensor, indices, ...
    args = []
    for T, idx in zip(tensors, index_lists):
        args.append(T)
        args.append(list(idx))
    args.append([])  # output: scalar

    Z = oe.contract(*args, optimize=optimize)
    if isinstance(Z, np.ndarray):
        Z = Z.item()
    return int(Z)


def potts_z_evaluations(
    graph: Graph,
    points: Iterable[Tuple[int, int]],
    dtype=np.int64,
    optimize: str = "auto",
) -> List[Tuple[int, int, int]]:
    """Evaluate Z(graph; q, v) at a list of (q, v_val) points.

    Returns list of (q, v_val, Z_value) triples.
    """
    out = []
    for (q, v_val) in points:
        z = potts_partition_function_einsum(
            graph, q, v_val, dtype=dtype, optimize=optimize,
        )
        out.append((q, v_val, z))
    return out
