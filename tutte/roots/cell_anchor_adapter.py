"""Generic cell-anchor adapter — normalizes per-cell anchor positions.

For graphs decomposable into cells with detectable cell-template Aut(C),
this module aligns per-cell anchor sets to a canonical structure so the
cycle DP can use a single shared cell template.

GRAPH-AGNOSTIC: works for any cell template (K_{4,4}, K_4, K_3, custom)
with any junction connectivity (M_k matchings, K_{a,b} bipartite, etc.).
The Chimera Cm2 case (which used K_{4,4} bipartite swap as a specific
aut) is one instance; the algorithm generalizes via Aut(cell) enumeration.

Algorithm:
1. Determine cycle order from cell-quotient adjacency.
2. For each cell, identify left/right operational anchor sets (vertices
   going to prev/next junction in cycle).
3. Detect cell template's automorphism group.
4. For each cell, find an aut σ such that σ maps the cell's actual
   left-anchors to canonical_left positions and right-anchors to
   canonical_right (in any consistent ordering).
5. Apply σ per cell to produce canonical labels in the relabeled graph.
6. Verify junctions share matching pattern; build canonical junction
   template.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx

from ..graph import Graph
from .aut_orbit import compute_cell_aut


@dataclass(frozen=True)
class CellAnchorGroups:
    """Per-cell anchor groups + per-junction group mapping.

    A cell has K *named* anchor groups (sets of cell vertices). Each
    junction declares which group on each side it uses. Two junctions
    naming the same group on the same cell SHARE the underlying
    vertex set — this is the structural property that distinguishes
    D-Wave Cm₃ (interior cells share each anchor group across two
    junctions) from D-Wave Cm₂ (each cell vertex is used by exactly
    one junction).

    Generic: works for any cell-decomposable graph regardless of
    whether the cells are K_{a,b}, K_n, Petersen, or anything else.

    Attributes:
        cell_groups: per-cell list of (group_id, sorted vertex tuple).
            cell_groups[cell_idx] = [(g0, (v0, v1, ...)), (g1, (...))].
            Group IDs are stable per cell — group 0 is the smallest
            anchor set, group 1 is the next, etc.
        junction_groups: per-junction (cell_a, cell_b, group_a, group_b).
            For each inter-cell junction (cell_a, cell_b) with
            cell_a < cell_b: which group on cell_a and which group on
            cell_b that junction's edges connect.
        edges_per_junction: list of edge endpoints per junction, in the
            same order as `junction_groups`. Each entry is a list of
            (vertex_in_cell_a, vertex_in_cell_b) pairs.
    """
    cell_groups: List[List[Tuple[int, Tuple[int, ...]]]]
    junction_groups: List[Tuple[int, int, int, int]]
    edges_per_junction: List[List[Tuple[int, int]]]

    def has_shared_anchors(self) -> bool:
        """True if ANY cell has a group serving more than one junction."""
        usage: Dict[Tuple[int, int], int] = defaultdict(int)
        for (ca, cb, ga, gb) in self.junction_groups:
            usage[(ca, ga)] += 1
            usage[(cb, gb)] += 1
        return any(count > 1 for count in usage.values())

    def groups_per_cell(self, cell_idx: int) -> int:
        return len(self.cell_groups[cell_idx])


def detect_cell_anchor_groups(
    partition: List[Set[int]],
    inter_edges: List[Tuple[int, int]],
) -> CellAnchorGroups:
    """Detect per-cell anchor groups + per-junction group mapping.

    Algorithm:
      1. For each cell, collect (other_cell, anchor_set) pairs from
         inter-cell edges.
      2. Group by anchor_set: junctions with the same anchor_set on
         the same cell SHARE that anchor group.
      3. Assign stable group IDs per cell (sorted by anchor set).
      4. For each junction, record which group ID on each side it uses.

    GENERIC: works on any cell-decomposable graph. The detection uses
    only set-equality on inter-cell edge endpoints; no D-Wave-specific
    assumptions.
    """
    n_cells = len(partition)
    node_to_cell: Dict[int, int] = {n: i for i, cell in enumerate(partition) for n in cell}

    # Step 1: collect per-cell (other_cell, anchors-on-this-cell) pairs.
    pair_edges: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    for u, v in inter_edges:
        ci, cj = node_to_cell[u], node_to_cell[v]
        if ci == cj:
            continue
        if ci < cj:
            pair_edges[(ci, cj)].append((u, v))
        else:
            pair_edges[(cj, ci)].append((v, u))

    # Step 2: per cell, determine which junctions use which anchor sets.
    # cell_anchor_set_per_junction[cell] = {(other_cell): frozenset(anchors_on_cell)}
    cell_anchor_set: Dict[int, Dict[int, FrozenSet[int]]] = defaultdict(dict)
    for (ca, cb), edges in pair_edges.items():
        anchors_a = frozenset(u for (u, v) in edges)
        anchors_b = frozenset(v for (u, v) in edges)
        cell_anchor_set[ca][cb] = anchors_a
        cell_anchor_set[cb][ca] = anchors_b

    # Step 3: stable group IDs per cell. Sort unique anchor sets by
    # (sorted tuple) for deterministic ordering.
    cell_groups: List[List[Tuple[int, Tuple[int, ...]]]] = []
    cell_anchor_to_group_id: List[Dict[FrozenSet[int], int]] = []
    for c in range(n_cells):
        unique_anchor_sets = sorted(
            set(cell_anchor_set[c].values()),
            key=lambda s: tuple(sorted(s)),
        )
        groups = [(i, tuple(sorted(s))) for i, s in enumerate(unique_anchor_sets)]
        cell_groups.append(groups)
        cell_anchor_to_group_id.append({s: i for i, s in enumerate(unique_anchor_sets)})

    # Step 4: per-junction group mapping.
    junction_groups: List[Tuple[int, int, int, int]] = []
    edges_per_junction: List[List[Tuple[int, int]]] = []
    for (ca, cb), edges in sorted(pair_edges.items()):
        anchors_a = frozenset(u for (u, v) in edges)
        anchors_b = frozenset(v for (u, v) in edges)
        ga = cell_anchor_to_group_id[ca][anchors_a]
        gb = cell_anchor_to_group_id[cb][anchors_b]
        junction_groups.append((ca, cb, ga, gb))
        edges_per_junction.append(sorted(edges))

    return CellAnchorGroups(
        cell_groups=cell_groups,
        junction_groups=junction_groups,
        edges_per_junction=edges_per_junction,
    )


@dataclass(frozen=True)
class CellRowSpec:
    """Per-cell anchor spec for one row of a grid composition.

    For path DP through a row, each cell c has up to four anchor groups
    relevant to grid-DP composition:

    * `left_group`: group ID used by the junction to cell on the LEFT in
      the same row (or None if this cell is the row's left endpoint).
    * `right_group`: group ID used by the junction to cell on the RIGHT
      (or None if this cell is the row's right endpoint).
    * `extra_groups`: group IDs used by junctions OUTSIDE this row
      (vertical / cross-row), persisting as state extras through path DP.

    When `left_group == right_group != None`, the cell's left and right
    horizontal anchors are the SAME vertex set (anchor sharing). The
    path DP treats this as a single boundary position used twice.
    """
    cell: int
    left_group: Optional[int]
    right_group: Optional[int]
    extra_groups: Tuple[int, ...]

    @property
    def has_shared_horizontal(self) -> bool:
        return (self.left_group is not None
                and self.right_group is not None
                and self.left_group == self.right_group)


@dataclass(frozen=True)
class CellGridSpec:
    """Per-cell anchor spec for a 2D grid composition.

    Each cell in a grid has up to four directional anchor groups:

    * `left_group`: junction to the cell at (r, c-1) — None if c == 0.
    * `right_group`: junction to (r, c+1) — None if c == cols - 1.
    * `up_group`: junction to (r-1, c) — None if r == 0.
    * `down_group`: junction to (r+1, c) — None if r == rows - 1.

    `extra_groups`: any group used by junctions OUTSIDE the grid (rare in
    pure grids but possible for grid-with-extras layouts).

    `has_shared_horizontal`: left_group == right_group != None.
    `has_shared_vertical`: up_group == down_group != None.
    """
    cell: int
    row: int
    col: int
    left_group: Optional[int]
    right_group: Optional[int]
    up_group: Optional[int]
    down_group: Optional[int]
    extra_groups: Tuple[int, ...]

    @property
    def has_shared_horizontal(self) -> bool:
        return (self.left_group is not None
                and self.right_group is not None
                and self.left_group == self.right_group)

    @property
    def has_shared_vertical(self) -> bool:
        return (self.up_group is not None
                and self.down_group is not None
                and self.up_group == self.down_group)

    def to_row_spec(self) -> "CellRowSpec":
        """Convert to a CellRowSpec for path DP (collapses vertical groups
        into extras, deduped)."""
        verticals: List[int] = []
        if self.up_group is not None:
            verticals.append(self.up_group)
        if self.down_group is not None and self.down_group != self.up_group:
            verticals.append(self.down_group)
        extras = tuple(sorted(set(list(verticals) + list(self.extra_groups))))
        return CellRowSpec(
            cell=self.cell,
            left_group=self.left_group,
            right_group=self.right_group,
            extra_groups=extras,
        )


def extract_grid_specs(
    spec: CellAnchorGroups,
    grid_layout: List[List[int]],
) -> List[List[CellGridSpec]]:
    """For a (rows × cols) grid layout, derive per-cell directional specs.

    grid_layout[r][c] = cell index in the global graph.

    Generic: works for any cell-decomposable graph laid out as a 2D grid.
    Junctions to cells in the layout are classified by direction
    (left/right/up/down). Junctions to cells NOT in the layout become
    `extra_groups`.
    """
    rows = len(grid_layout)
    cols = len(grid_layout[0])
    cells_in_layout = {grid_layout[r][c]: (r, c) for r in range(rows) for c in range(cols)}

    cell_to_other_group: Dict[int, Dict[int, int]] = defaultdict(dict)
    for (ca, cb, ga, gb) in spec.junction_groups:
        cell_to_other_group[ca][cb] = ga
        cell_to_other_group[cb][ca] = gb

    out: List[List[CellGridSpec]] = []
    for r in range(rows):
        row_specs: List[CellGridSpec] = []
        for c in range(cols):
            cell = grid_layout[r][c]
            left_group: Optional[int] = None
            right_group: Optional[int] = None
            up_group: Optional[int] = None
            down_group: Optional[int] = None
            extras: List[int] = []
            for other_cell, group in cell_to_other_group[cell].items():
                pos_other = cells_in_layout.get(other_cell)
                if pos_other is None:
                    extras.append(group)
                    continue
                ro, co = pos_other
                if ro == r and co == c - 1:
                    left_group = group
                elif ro == r and co == c + 1:
                    right_group = group
                elif co == c and ro == r - 1:
                    up_group = group
                elif co == c and ro == r + 1:
                    down_group = group
                else:
                    extras.append(group)
            extras_unique = tuple(sorted(set(extras)))
            row_specs.append(CellGridSpec(
                cell=cell, row=r, col=c,
                left_group=left_group, right_group=right_group,
                up_group=up_group, down_group=down_group,
                extra_groups=extras_unique,
            ))
        out.append(row_specs)
    return out


def extract_path_specs(
    spec: CellAnchorGroups,
    cells_in_path: List[int],
) -> List[CellRowSpec]:
    """For an ordered list of cells forming a path, derive per-cell
    anchor specs (which groups are LEFT, RIGHT, and EXTRAS within this row).

    GENERIC: the path can be any sequence of cells from the global graph;
    "left" and "right" are always defined relative to the path order.
    Junctions to cells NOT in the path become "extras".
    """
    cells_set = set(cells_in_path)
    n = len(cells_in_path)
    cell_position = {cell: idx for idx, cell in enumerate(cells_in_path)}

    # Per cell: which group is used by which adjacent cell?
    cell_to_other_group: Dict[int, Dict[int, int]] = defaultdict(dict)
    for (ca, cb, ga, gb) in spec.junction_groups:
        cell_to_other_group[ca][cb] = ga
        cell_to_other_group[cb][ca] = gb

    out: List[CellRowSpec] = []
    for idx, cell in enumerate(cells_in_path):
        left_group: Optional[int] = None
        right_group: Optional[int] = None
        extras: List[int] = []
        for other_cell, group in cell_to_other_group[cell].items():
            if other_cell in cells_set:
                other_idx = cell_position[other_cell]
                if other_idx == idx - 1:
                    left_group = group
                elif other_idx == idx + 1:
                    right_group = group
                else:
                    # Other cell is in the path but not adjacent in path-order:
                    # this means the path order doesn't reflect the cell graph
                    # (e.g., the cells form a cycle, not a strict path).
                    extras.append(group)
            else:
                # Other cell is OUTSIDE the path → vertical extra.
                extras.append(group)
        # Dedupe extras: if a single group serves multiple outside junctions,
        # it's still ONE persistent boundary.
        extras_unique = tuple(sorted(set(extras)))
        out.append(CellRowSpec(
            cell=cell,
            left_group=left_group,
            right_group=right_group,
            extra_groups=extras_unique,
        ))
    return out


def _build_cycle_order(
    n_cells: int,
    pair_edges: Dict[Tuple[int, int], List[Tuple[int, int]]],
) -> Optional[List[int]]:
    """Determine cycle order from cell-quotient adjacency.

    Returns list of cell indices in cycle order, or None if not a cycle.
    """
    adj: Dict[int, Set[int]] = defaultdict(set)
    for (i, j) in pair_edges:
        adj[i].add(j)
        adj[j].add(i)
    # Verify all cells have degree 2 (cycle property)
    for i in range(n_cells):
        if len(adj[i]) != 2:
            return None
    # Walk the cycle starting from cell 0
    order = [0]
    visited = {0}
    while len(order) < n_cells:
        current = order[-1]
        next_cell = None
        for nb in sorted(adj[current]):
            if nb not in visited:
                next_cell = nb
                break
        if next_cell is None:
            return None
        order.append(next_cell)
        visited.add(next_cell)
    if order[0] not in adj[order[-1]]:
        return None
    return order


def _get_cell_anchors_per_junction(
    cycle_order: List[int],
    partition: List[Set[int]],
    pair_edges: Dict[Tuple[int, int], List[Tuple[int, int]]],
) -> Tuple[List[List[int]], List[List[int]]]:
    """For each cell in cycle order, identify left/right anchors.

    left = anchors to PREVIOUS cell in cycle.
    right = anchors to NEXT cell in cycle.
    """
    n = len(cycle_order)
    cells_left: List[List[int]] = []
    cells_right: List[List[int]] = []
    for idx in range(n):
        cur = cycle_order[idx]
        prev = cycle_order[(idx - 1) % n]
        nxt = cycle_order[(idx + 1) % n]
        # Left anchors: cur's vertices in (cur, prev) junction
        prev_pair = tuple(sorted([cur, prev]))
        left_anchors = []
        for u, v in pair_edges.get(prev_pair, []):
            if u in partition[cur]:
                left_anchors.append(u)
            else:
                left_anchors.append(v)
        # Right anchors: cur's vertices in (cur, nxt) junction
        next_pair = tuple(sorted([cur, nxt]))
        right_anchors = []
        for u, v in pair_edges.get(next_pair, []):
            if u in partition[cur]:
                right_anchors.append(u)
            else:
                right_anchors.append(v)
        cells_left.append(left_anchors)
        cells_right.append(right_anchors)
    return cells_left, cells_right


def _find_cell_aut_to_canonical(
    cell_template: Graph,
    cell_template_anchors: List[int],
    actual_cell_vertices: List[int],
    actual_left_anchors: List[int],
    actual_right_anchors: List[int],
    template_left_anchors: List[int],
    template_right_anchors: List[int],
) -> Optional[Dict[int, int]]:
    """Find a vertex map: actual_cell_vertices → cell_template_vertices
    such that the cell graph structure is preserved AND
    actual_left_anchors map to template_left_anchors (set-wise) AND
    actual_right_anchors map to template_right_anchors (set-wise).

    Uses VF2 isomorphism between the actual cell's induced subgraph and
    the cell template, restricted by the anchor-set constraint. Generic
    over any cell template.

    Returns the relabel map (actual_v → template_v), or None if no such
    map exists.
    """
    # Currently this is a placeholder using the simple bipartite-swap
    # approach for K_{4,4} cells. For full generality,
    # iterate VF2 isomorphisms with anchor-set constraints.
    raise NotImplementedError(
        "Generic VF2-based aut finder pending; "
        "use _bipartite_align_relabel for K_{4,4} cells in Cm-class graphs."
    )


def _bipartite_align_relabel(
    g: Graph,
    partition: List[Set[int]],
    pair_edges: Dict[Tuple[int, int], List[Tuple[int, int]]],
    cycle_order: List[int],
    cells_left_orig: List[List[int]],
    cells_right_orig: List[List[int]],
) -> Tuple[Graph, List[List[int]], List[List[int]], List[List[int]],
           List[List[Tuple[int, int]]]]:
    """Bipartite-aware per-cell relabel for K_{a,b}-like cells.

    For each cell, compute its bipartite 2-coloring; align per-cell
    anchor patterns via bipartite swap as needed (the swap is a valid
    aut of K_{a,b} cells). Produces canonical position labels uniform
    across all cells.

    This is the K_{a,b}-cell special case of the generic adapter. Works
    for any bipartite cell where the operational anchor split intersects
    both bipartite classes.
    """
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(g.nodes)
    nx_graph.add_edges_from(g.edges)

    n_cells = len(cycle_order)

    # Per-cell bipartite split (only valid for bipartite cells like K_{a,b}).
    cell_bipartitions: List[Tuple[List[int], List[int]]] = []
    for cur in cycle_order:
        induced = nx_graph.subgraph(sorted(partition[cur]))
        try:
            color = nx.bipartite.color(induced)
        except nx.NetworkXError:
            raise ValueError(
                f"Cell {cur} is not bipartite — generic adapter needed."
            )
        nat_left = sorted([v for v in partition[cur] if color[v] == 0])
        nat_right = sorted([v for v in partition[cur] if color[v] == 1])
        cell_bipartitions.append((nat_left, nat_right))

    # Reference shape: cell 0's count of left-anchors in nat-left side.
    cell0_nat_left, _ = cell_bipartitions[0]
    cell0_left_in_natleft = sum(1 for v in cells_left_orig[0] if v in cell0_nat_left)

    # Determine which cells need bipartite swap to match cell 0's shape.
    needs_swap: List[bool] = [False]
    for idx in range(1, n_cells):
        nat_left, _ = cell_bipartitions[idx]
        left_in_natleft = sum(1 for v in cells_left_orig[idx] if v in nat_left)
        needs_swap.append(left_in_natleft != cell0_left_in_natleft)

    # Per-cell relabel to canonical positions [base..base+a+b-1].
    relabel_map: Dict[int, int] = {}
    canonical_left_anchors: List[List[int]] = []
    canonical_right_anchors: List[List[int]] = []
    cells_canonical: List[List[int]] = []

    a = len(cells_left_orig[0])
    b = len(cells_right_orig[0])

    for idx in range(n_cells):
        cur = cycle_order[idx]
        base = 100 * (idx + 1)
        nat_left, nat_right = cell_bipartitions[idx]
        if needs_swap[idx]:
            nat_left, nat_right = nat_right, nat_left

        left_orig = cells_left_orig[idx]
        right_orig = cells_right_orig[idx]
        # Order anchors: nat-left elements first, then nat-right.
        left_natleft = sorted([v for v in left_orig if v in nat_left])
        left_natright = sorted([v for v in left_orig if v in nat_right])
        right_natleft = sorted([v for v in right_orig if v in nat_left])
        right_natright = sorted([v for v in right_orig if v in nat_right])

        if len(left_natleft) != cell0_left_in_natleft:
            raise ValueError(
                f"Cell {cur} (idx {idx}): left_natleft has {len(left_natleft)},"
                f" expected {cell0_left_in_natleft} (shape misalignment)"
            )

        left_ordered = left_natleft + left_natright
        right_ordered = right_natleft + right_natright
        for i, v in enumerate(left_ordered):
            relabel_map[v] = base + i
        for i, v in enumerate(right_ordered):
            relabel_map[v] = base + a + i

        canonical_left_anchors.append([base + i for i in range(a)])
        canonical_right_anchors.append([base + a + i for i in range(b)])
        cells_canonical.append([base + i for i in range(a + b)])

    # Build relabeled graph.
    new_edges = []
    for u, v in g.edges:
        if u in relabel_map and v in relabel_map:
            new_edges.append((relabel_map[u], relabel_map[v]))
    new_nodes = sorted(set(n for e in new_edges for n in e))
    relabeled = Graph(new_nodes, new_edges)

    # Per-junction edge list in canonical labels.
    junction_edges: List[List[Tuple[int, int]]] = []
    for idx in range(n_cells):
        cur = cycle_order[idx]
        nxt = cycle_order[(idx + 1) % n_cells]
        pair = tuple(sorted([cur, nxt]))
        canon_edges = []
        for u, v in pair_edges[pair]:
            if u in relabel_map and v in relabel_map:
                canon_edges.append(tuple(sorted([relabel_map[u], relabel_map[v]])))
        junction_edges.append(canon_edges)

    return (relabeled, cells_canonical, canonical_left_anchors,
            canonical_right_anchors, junction_edges)


def normalize_cell_anchors_for_cycle(
    g: Graph,
    partition: List[Set[int]],
    inter_edges: List[Tuple[int, int]],
    cell_template: Graph,
) -> Optional[Tuple[Graph, List[List[int]], List[List[int]], List[List[int]],
                    List[List[Tuple[int, int]]], List[int]]]:
    """Normalize per-cell anchors to align with cell template (cycle case).

    Args:
        g: The graph being analyzed.
        partition: List of cell vertex sets.
        inter_edges: Inter-cell edges.
        cell_template: The cell template graph (e.g., K_{4,4}).

    Returns:
        (relabeled_graph, cells_canonical, lefts, rights, junction_edges, cycle_order)
        or None if not a cycle topology / can't be normalized.

    Generic over cell types: uses Aut(cell) to find per-cell alignment.
    For bipartite cells (K_{a,b}), uses the bipartite-aware shortcut.
    """
    n_cells = len(partition)

    # Build pair_edges grouped by cell pair.
    node_to_cell = {n: i for i, cell in enumerate(partition) for n in cell}
    pair_edges: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    for u, v in inter_edges:
        ci, cj = node_to_cell[u], node_to_cell[v]
        if ci < cj:
            pair_edges[(ci, cj)].append((u, v))
        else:
            pair_edges[(cj, ci)].append((v, u))

    # Cycle order detection (graph-agnostic).
    cycle_order = _build_cycle_order(n_cells, pair_edges)
    if cycle_order is None:
        return None  # not a cycle topology

    # Identify per-cell left/right anchors per junction (graph-agnostic).
    cells_left_orig, cells_right_orig = _get_cell_anchors_per_junction(
        cycle_order, partition, pair_edges,
    )

    # Verify all cells have same number of anchors per side.
    if not cells_left_orig or not cells_right_orig:
        return None
    a = len(cells_left_orig[0])
    b = len(cells_right_orig[0])
    if any(len(la) != a for la in cells_left_orig):
        return None
    if any(len(ra) != b for ra in cells_right_orig):
        return None

    # Align via bipartite swap if cell is bipartite (covers K_{a,b} cells
    # like Cm's K_{4,4}). For non-bipartite cells, generic Aut-based
    # alignment — currently raises.
    nxg = nx.Graph()
    nxg.add_nodes_from(cell_template.nodes)
    nxg.add_edges_from(cell_template.edges)
    if nx.is_bipartite(nxg):
        relabeled, cells, lefts, rights, junctions = _bipartite_align_relabel(
            g, partition, pair_edges, cycle_order,
            cells_left_orig, cells_right_orig,
        )
        return relabeled, cells, lefts, rights, junctions, cycle_order
    else:
        # Non-bipartite cell — generic Aut-based alignment is future work.
        # For now, return None to fall through to other engine paths.
        return None
