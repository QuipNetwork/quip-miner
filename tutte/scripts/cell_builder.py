"""Construct cell-structured graphs for the visualizer's "Cell builder".

Builds graphs by composing a CELL template (K_n, K_{a,b}, C_n, named D-Wave
cells) with a JUNCTION template (matching, single edge, shared vertex)
under a FAMILY topology (path, cycle, grid, interleaved).

Public API:
    build_cell_graph(cell_type, cell_params, junction_type, junction_params,
                     family_type, family_params) -> (networkx.Graph, label)

Used by:
- `tutte/scripts/visualize_tutte.py`: the "Cell builder" form in the graph
  selector panel.
- Manual research probes: import directly and pass kwargs.

The constructed graphs are intended for use as Tutte-polynomial synthesis
inputs that strictly dominate a baseline graph in terms of evaluation
values at (0, 1), (1, 0), (1, 1), (2, 2), etc. — i.e., the builder
produces graphs whose spanning-tree count, acyclic-orientation count,
etc. are larger than the baseline's. See ``compare_to_baseline``.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Cell templates
# ---------------------------------------------------------------------------


def _make_cell(cell_type: str, params: Dict[str, Any]) -> Tuple[nx.Graph, str]:
    """Instantiate one cell as a NetworkX graph and return (graph, label).

    Anchors are taken to be vertices 0..k-1 in the cell unless otherwise
    noted; ``_get_anchors`` can refine this per cell type.
    """
    t = cell_type.lower()
    if t in ("k_n", "kn", "complete"):
        n = int(params.get("n", 4))
        if n < 1:
            raise ValueError("K_n requires n >= 1")
        return nx.complete_graph(n), f"K_{n}"
    if t in ("k_a_b", "kab", "complete_bipartite"):
        a = int(params.get("a", 4))
        b = int(params.get("b", 4))
        if a < 1 or b < 1:
            raise ValueError("K_{a,b} requires a, b >= 1")
        return nx.complete_bipartite_graph(a, b), f"K_{{{a},{b}}}"
    if t in ("c_n", "cn", "cycle"):
        n = int(params.get("n", 4))
        if n < 3:
            raise ValueError("C_n requires n >= 3")
        return nx.cycle_graph(n), f"C_{n}"
    if t in ("p_n", "pn", "path"):
        n = int(params.get("n", 4))
        if n < 2:
            raise ValueError("P_n requires n >= 2")
        return nx.path_graph(n), f"P_{n}"
    if t in ("z11", "z_1_1", "zephyr11"):
        # Legacy shortcut for Z(1, 1).
        return _make_cell("zephyr", {"m": 1, "t": 1})
    if t in ("cm1", "chimera1"):
        # Legacy shortcut for Chimera C(1) = K_{4,4}.
        return _make_cell("chimera", {"m": 1})
    if t in ("pm2", "pegasus2"):
        # Legacy shortcut for Pegasus P(2).
        return _make_cell("pegasus", {"m": 2})
    if t in ("zephyr", "z"):
        import dwave_networkx as dnx
        m = int(params.get("m", 1))
        zt = int(params.get("t", 1))
        if m < 1 or zt < 1:
            raise ValueError("Zephyr requires m >= 1 and t >= 1")
        G = dnx.zephyr_graph(m, t=zt)
        mapping = {old: new for new, old in enumerate(sorted(G.nodes))}
        return nx.relabel_nodes(G, mapping), f"Z({m},{zt})"
    if t in ("chimera", "cm"):
        import dwave_networkx as dnx
        m = int(params.get("m", 1))
        # n defaults to m for the square form; explicit n makes it rectangular.
        n = int(params.get("n", m))
        if m < 1 or n < 1:
            raise ValueError("Chimera requires m >= 1 and n >= 1")
        G = dnx.chimera_graph(m, n=n)
        mapping = {old: new for new, old in enumerate(sorted(G.nodes))}
        label = f"Cm({m})" if m == n else f"Cm({m},{n})"
        return nx.relabel_nodes(G, mapping), label
    if t in ("pegasus", "pm"):
        import dwave_networkx as dnx
        m = int(params.get("m", 2))
        if m < 2:
            raise ValueError("Pegasus requires m >= 2 (P(1) is empty)")
        G = dnx.pegasus_graph(m)
        mapping = {old: new for new, old in enumerate(sorted(G.nodes))}
        return nx.relabel_nodes(G, mapping), f"Pm({m})"
    raise ValueError(f"Unknown cell_type {cell_type!r}")


def _default_anchors(
    cell_graph: nx.Graph, cell_type: str, k: int,
) -> List[int]:
    """Pick ``k`` anchor vertices on a cell for junctions.

    Defaults:
      * K_n: any k vertices (vertex-transitive).
      * K_{a,b}: first k vertices of side A (i.e., 0..min(k, a)-1).
      * C_n / P_n: 0..k-1 (first k consecutive).
      * Z(1,1): the 4 K_4 vertices for k ≤ 4, else extend into C_8.
      * Cm1 (K_{4,4}): side A = vertices 0..3.
      * Pm2: first k vertices of side A.
    """
    n = cell_graph.number_of_nodes()
    if k > n:
        raise ValueError(
            f"junction k={k} exceeds cell |V|={n}"
        )
    t = cell_type.lower()
    if t in ("z11", "z_1_1", "zephyr11", "zephyr", "z"):
        # Zephyr cells: prefer higher-degree vertices first. For Z(1,1) this
        # surfaces the K_4 vertices; for larger Zephyr it surfaces the
        # densely-connected internal nodes.
        deg_sorted = sorted(
            cell_graph.nodes,
            key=lambda v: (-cell_graph.degree(v), v),
        )
        return deg_sorted[:k]
    if t in ("cm1", "chimera1", "chimera", "cm",
             "k_a_b", "kab", "complete_bipartite",
             "pegasus", "pm", "pm2", "pegasus2"):
        # K_{a,b} / Chimera / Pegasus: prefer one side of the bipartition.
        # The relabeling puts side A as 0..a-1.
        return list(range(min(k, n)))
    # Default: first k.
    return list(range(k))


# ---------------------------------------------------------------------------
# Junction templates
# ---------------------------------------------------------------------------


def _junction_anchor_count(junction_type: str, junction_params: Dict[str, Any]) -> int:
    """How many anchors does each cell side contribute to one junction?"""
    j = junction_type.lower()
    if j in ("matching", "m_k", "mk"):
        return int(junction_params.get("k", 4))
    if j in ("single_edge", "single", "edge"):
        return 1
    if j in ("shared_vertex", "cut_vertex", "vertex"):
        return 1
    if j in ("k_a_b_junction", "kab_junction", "complete_bipartite_junction"):
        # K_{a,b} bipartite junction: each side contributes a anchors,
        # joined to the other side's b anchors (assume a==b for symmetry).
        return int(junction_params.get("a", 4))
    raise ValueError(f"Unknown junction_type {junction_type!r}")


def _add_junction(
    G: nx.Graph,
    cell_a_anchors: List[int],
    cell_b_anchors: List[int],
    junction_type: str,
    junction_params: Dict[str, Any],
) -> None:
    """Mutate ``G`` to add junction edges between two cell anchor sets.

    Anchors are vertices ALREADY in ``G`` (after appropriate relabeling
    of cell B). For ``shared_vertex`` we delete cell-B's anchor and remap
    its edges to cell-A's anchor (caller is responsible for relabeling
    before calling).
    """
    j = junction_type.lower()
    if j in ("matching", "m_k", "mk"):
        k = int(junction_params.get("k", 4))
        if k > len(cell_a_anchors) or k > len(cell_b_anchors):
            raise ValueError(
                f"matching k={k} exceeds available anchors "
                f"({len(cell_a_anchors)}, {len(cell_b_anchors)})"
            )
        for i in range(k):
            G.add_edge(cell_a_anchors[i], cell_b_anchors[i])
        return
    if j in ("single_edge", "single", "edge"):
        G.add_edge(cell_a_anchors[0], cell_b_anchors[0])
        return
    if j in ("shared_vertex", "cut_vertex", "vertex"):
        # Caller is responsible for the relabeling that identifies the
        # two anchor vertices; nothing to do here.
        return
    if j in ("k_a_b_junction", "kab_junction", "complete_bipartite_junction"):
        a = int(junction_params.get("a", len(cell_a_anchors)))
        b = int(junction_params.get("b", len(cell_b_anchors)))
        a = min(a, len(cell_a_anchors))
        b = min(b, len(cell_b_anchors))
        for u in cell_a_anchors[:a]:
            for v in cell_b_anchors[:b]:
                G.add_edge(u, v)
        return
    raise ValueError(f"Unknown junction_type {junction_type!r}")


# ---------------------------------------------------------------------------
# Family topology
# ---------------------------------------------------------------------------


def _instantiate_cells(
    cell_specs: List[Tuple[str, Dict[str, Any]]],
) -> Tuple[List[nx.Graph], List[str], List[int]]:
    """Instantiate a list of cell specs. Returns (graphs, labels, sizes)."""
    graphs = []
    labels = []
    sizes = []
    for ct, cp in cell_specs:
        g, lbl = _make_cell(ct, cp)
        graphs.append(g)
        labels.append(lbl)
        sizes.append(g.number_of_nodes())
    return graphs, labels, sizes


def _disjoint_offset_union(
    cell_graphs: List[nx.Graph],
) -> Tuple[nx.Graph, List[int]]:
    """Disjoint union with vertex offsets. Returns (union, offsets)."""
    G = nx.Graph()
    offsets = [0]
    for i, cell in enumerate(cell_graphs):
        offset = offsets[i]
        for v in cell.nodes:
            G.add_node(offset + v)
        for u, v in cell.edges:
            G.add_edge(offset + u, offset + v)
        offsets.append(offset + cell.number_of_nodes())
    return G, offsets[:-1]  # offsets per cell


def _build_path(
    cell_specs: List[Tuple[str, Dict[str, Any]]],
    junction_type: str,
    junction_params: Dict[str, Any],
) -> Tuple[nx.Graph, str]:
    """Path of n cells joined by n-1 junctions."""
    cell_graphs, labels, sizes = _instantiate_cells(cell_specs)
    G, offsets = _disjoint_offset_union(cell_graphs)
    k = _junction_anchor_count(junction_type, junction_params)
    for i in range(len(cell_graphs) - 1):
        a_anchors = [
            offsets[i] + v for v in _default_anchors(
                cell_graphs[i], cell_specs[i][0], k,
            )
        ]
        b_anchors = [
            offsets[i + 1] + v for v in _default_anchors(
                cell_graphs[i + 1], cell_specs[i + 1][0], k,
            )
        ]
        _add_junction(G, a_anchors, b_anchors, junction_type, junction_params)
    return G, f"Path[{','.join(labels)}]+{junction_type}"


def _build_cycle(
    cell_specs: List[Tuple[str, Dict[str, Any]]],
    junction_type: str,
    junction_params: Dict[str, Any],
) -> Tuple[nx.Graph, str]:
    """Cycle of n cells joined by n junctions (path + close)."""
    G, lbl = _build_path(cell_specs, junction_type, junction_params)
    cell_graphs, _, sizes = _instantiate_cells(cell_specs)
    offsets = [sum(sizes[:i]) for i in range(len(cell_graphs))]
    k = _junction_anchor_count(junction_type, junction_params)
    last = len(cell_graphs) - 1
    # Closing junction: last cell back to first.
    a_anchors = [
        offsets[last] + v for v in _default_anchors(
            cell_graphs[last], cell_specs[last][0], k,
        )
    ]
    # Use a DIFFERENT anchor set on cell 0 so we don't reuse cell-0's
    # left-junction anchors (avoid degenerate parallel/shared edges).
    # For symmetric cells (K_n, K_{a,b}), shift to the next k vertices.
    first_size = sizes[0]
    other_anchors_0 = _default_anchors(cell_graphs[0], cell_specs[0][0], 2 * k)
    if len(other_anchors_0) >= 2 * k:
        b_anchors = [offsets[0] + v for v in other_anchors_0[k:2 * k]]
    else:
        # Cell too small for separate anchor sets — reuse and accept
        # the degenerate structure (caller's choice).
        b_anchors = [
            offsets[0] + v for v in _default_anchors(
                cell_graphs[0], cell_specs[0][0], k,
            )
        ]
    _add_junction(G, a_anchors, b_anchors, junction_type, junction_params)
    return G, lbl.replace("Path[", "Cycle[")


def _build_grid(
    cell_specs: List[Tuple[str, Dict[str, Any]]],
    junction_type: str,
    junction_params: Dict[str, Any],
    rows: int,
    cols: int,
) -> Tuple[nx.Graph, str]:
    """Grid of rows × cols cells joined by row + col junctions.

    ``cell_specs`` length must equal ``rows * cols`` (one spec per
    grid position, row-major). For homogeneous grids, pass the same
    spec replicated.
    """
    if len(cell_specs) != rows * cols:
        raise ValueError(
            f"grid expects {rows * cols} cell specs, got {len(cell_specs)}"
        )
    cell_graphs, labels, sizes = _instantiate_cells(cell_specs)
    G, offsets = _disjoint_offset_union(cell_graphs)
    k = _junction_anchor_count(junction_type, junction_params)
    # For grids, we need separate anchor sets per direction (horizontal
    # vs vertical). Take anchors [0:k] for horizontal, [k:2k] for vertical
    # when the cell is large enough; otherwise reuse.
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            # Horizontal junction to (r, c+1)
            if c + 1 < cols:
                idx_b = r * cols + (c + 1)
                anchors_a_all = _default_anchors(
                    cell_graphs[idx], cell_specs[idx][0], 2 * k,
                )
                anchors_b_all = _default_anchors(
                    cell_graphs[idx_b], cell_specs[idx_b][0], 2 * k,
                )
                a_h = anchors_a_all[:k]
                b_h = anchors_b_all[:k]
                a_anchors = [offsets[idx] + v for v in a_h]
                b_anchors = [offsets[idx_b] + v for v in b_h]
                _add_junction(G, a_anchors, b_anchors,
                              junction_type, junction_params)
            # Vertical junction to (r+1, c)
            if r + 1 < rows:
                idx_b = (r + 1) * cols + c
                anchors_a_all = _default_anchors(
                    cell_graphs[idx], cell_specs[idx][0], 2 * k,
                )
                anchors_b_all = _default_anchors(
                    cell_graphs[idx_b], cell_specs[idx_b][0], 2 * k,
                )
                # Use second half of anchors for vertical direction;
                # avoids reusing horizontal anchors when cell is big
                # enough (Cm1 / K_{4,4} pattern).
                if len(anchors_a_all) >= 2 * k:
                    a_v = anchors_a_all[k:2 * k]
                    b_v = anchors_b_all[k:2 * k]
                else:
                    a_v = anchors_a_all[:k]
                    b_v = anchors_b_all[:k]
                a_anchors = [offsets[idx] + v for v in a_v]
                b_anchors = [offsets[idx_b] + v for v in b_v]
                _add_junction(G, a_anchors, b_anchors,
                              junction_type, junction_params)
    # Build a compact label
    unique_labels = sorted(set(labels))
    base_lbl = labels[0] if len(unique_labels) == 1 else "Mixed"
    return G, f"Grid[{rows}x{cols},{base_lbl}]+{junction_type}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_cell_graph(
    cell_type: str,
    cell_params: Dict[str, Any],
    junction_type: str,
    junction_params: Dict[str, Any],
    family_type: str,
    family_params: Dict[str, Any],
    alt_cell_type: Optional[str] = None,
    alt_cell_params: Optional[Dict[str, Any]] = None,
) -> Tuple[nx.Graph, str]:
    """Build a cell-structured graph.

    Args:
        cell_type: Primary cell template (``"K_n"``, ``"K_a_b"``,
            ``"C_n"``, ``"P_n"``, ``"Z11"``, ``"Cm1"``, ``"Pm2"``).
        cell_params: Cell template parameters (e.g. ``{"n": 5}`` for
            K_5, ``{"a": 4, "b": 4}`` for K_{4,4}).
        junction_type: Inter-cell junction (``"matching"``, ``"single_edge"``,
            ``"shared_vertex"``, ``"k_a_b_junction"``).
        junction_params: Junction parameters (e.g. ``{"k": 4}`` for M_4).
        family_type: Topology of the cell arrangement (``"path"``,
            ``"cycle"``, ``"grid"``, ``"interleaved"``).
        family_params: Family parameters:
            * path/cycle: ``{"count": n}``
            * grid: ``{"rows": r, "cols": c}``
            * interleaved: ``{"count": n, "pattern": "path"|"cycle"}``
              (alternates between ``cell_type`` and ``alt_cell_type``)
        alt_cell_type: Second cell type for ``"interleaved"`` family.
        alt_cell_params: Parameters for the second cell type.

    Returns:
        ``(networkx.Graph, label)`` where ``label`` is a human-readable
        description of the construction.
    """
    fam = family_type.lower()

    if fam in ("interleaved",):
        count = int(family_params.get("count", 4))
        if alt_cell_type is None:
            raise ValueError("interleaved family requires alt_cell_type")
        cell_specs: List[Tuple[str, Dict[str, Any]]] = []
        for i in range(count):
            if i % 2 == 0:
                cell_specs.append((cell_type, cell_params))
            else:
                cell_specs.append((alt_cell_type, alt_cell_params or {}))
        pattern = family_params.get("pattern", "path").lower()
        if pattern == "cycle":
            return _build_cycle(cell_specs, junction_type, junction_params)
        return _build_path(cell_specs, junction_type, junction_params)

    if fam in ("path",):
        count = int(family_params.get("count", 3))
        cell_specs = [(cell_type, cell_params)] * count
        return _build_path(cell_specs, junction_type, junction_params)

    if fam in ("cycle",):
        count = int(family_params.get("count", 3))
        cell_specs = [(cell_type, cell_params)] * count
        return _build_cycle(cell_specs, junction_type, junction_params)

    if fam in ("grid",):
        rows = int(family_params.get("rows", 2))
        cols = int(family_params.get("cols", 2))
        cell_specs = [(cell_type, cell_params)] * (rows * cols)
        return _build_grid(
            cell_specs, junction_type, junction_params, rows, cols,
        )

    raise ValueError(f"Unknown family_type {family_type!r}")


def compare_to_baseline(
    constructed_poly,
    baseline_poly,
    points: Iterable[Tuple[int, int]] = ((0, 1), (1, 0), (1, 1), (2, 2)),
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Evaluate both polynomials at ``points`` and check the constructed
    graph strictly dominates the baseline.

    Returns a dict keyed by (x, y) with values
    ``{"constructed": int, "baseline": int, "dominates": bool}``.
    A point is dominated when ``constructed >= baseline``; the builder
    surfaces ``all(dominates)`` to the visualizer as a green ✓ vs red ✗.
    """
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for x, y in points:
        c = constructed_poly.evaluate(x, y)
        b = baseline_poly.evaluate(x, y)
        out[(x, y)] = {
            "constructed": c,
            "baseline": b,
            "dominates": c >= b,
        }
    return out


__all__ = [
    "build_cell_graph",
    "compare_to_baseline",
]
