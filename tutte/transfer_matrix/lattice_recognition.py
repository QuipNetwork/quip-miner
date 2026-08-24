"""Lattice-like periodic strip recognition for transfer-matrix computation.

Detects whether a graph is a periodic lattice-like strip and extracts
its structural properties: width, length, unit cell edge pattern, and vertex
coordinate grid.

All recognition functions perform BOTH O(1) fingerprint checks AND O(V+E)
structural BFS verification. Fingerprint-only checks are insufficient because
non-lattice graphs can match the degree distribution of a lattice family.

Supported families:
    - Grid P_width x P_length (open boundary, planar, period-1)
    - Triangular lattice strip (grid + one diagonal per cell, period-1)
    - Honeycomb / hexagonal lattice strip (brick-wall model, period-2)
    - Square-octagon / truncated square (4.8.8 Archimedean tiling, period-4)
    - Elongated triangular (3.3.3.4.4 semi-regular tiling, period-1)

Unit cell edge encoding:
    3-tuples (row_a, row_b, is_cross_column): is_cross_column=True means old
    boundary row_a connects to new boundary row_b (cross-column).
    is_cross_column=False means new boundary row_a connects to new boundary
    row_b (within-column).

File organization:
    1. Shared BFS isomorphism infrastructure
    2. Grid: detection + unit cell
    3. Triangular: detection + unit cell
    4. Honeycomb: detection + unit cell + helpers
    5. Square-octagon: detection + unit cell + adjacency builder
    6. Elongated triangular: detection + unit cell + adjacency builder
    7. Entry point: detect_periodic_strip

Complexity: O(V + E) for all recognition functions.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

from ..graph import Graph
from ..family_recognition.fingerprint import StructuralFingerprint
from .core import MAX_TRANSFER_MATRIX_WIDTH


# Neighbor offsets for each lattice type.
_GRID_OFFSETS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
_NE_SW_OFFSETS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)]
_NW_SE_OFFSETS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1)]


# =============================================================================
# 1. SHARED BFS ISOMORPHISM INFRASTRUCTURE
# =============================================================================


def _try_lattice_bfs_match(
    graph: Graph,
    expected_adj: Dict[int, Set[int]],
    input_start: int,
    expected_start: int,
) -> Optional[Dict[int, int]]:
    """Greedy BFS isomorphism between an input graph and expected adjacency.

    Starting from a seed pair (input_start, expected_start), extends the
    mapping one BFS layer at a time. At each vertex, unmapped neighbors are
    sorted by degree and paired positionally. Fails if any degree mismatch
    or adjacency inconsistency is found.

    Args:
        graph: Input graph.
        expected_adj: Expected adjacency as {idx: set of neighbor indices}.
        input_start: Starting vertex in input graph.
        expected_start: Starting index in expected graph.

    Returns:
        Mapping input_vertex -> expected_index, or None.

    Complexity: O(V + E).
    """
    inp_to_exp: Dict[int, int] = {input_start: expected_start}
    exp_to_inp: Dict[int, int] = {expected_start: input_start}
    queue: deque = deque([input_start])

    while queue:
        inp_v = queue.popleft()
        exp_v = inp_to_exp[inp_v]

        inp_nbrs = sorted(graph.neighbors(inp_v))
        exp_nbrs = sorted(expected_adj[exp_v])

        if len(inp_nbrs) != len(exp_nbrs):
            return None

        for n in inp_nbrs:
            if n in inp_to_exp and inp_to_exp[n] not in expected_adj[exp_v]:
                return None

        unmapped_inp = [n for n in inp_nbrs if n not in inp_to_exp]
        unmapped_exp = [n for n in exp_nbrs if n not in exp_to_inp]

        if len(unmapped_inp) != len(unmapped_exp):
            return None

        unmapped_inp.sort(key=lambda v: graph.degree(v))
        unmapped_exp.sort(key=lambda v: len(expected_adj[v]))

        for i_n, e_n in zip(unmapped_inp, unmapped_exp):
            if graph.degree(i_n) != len(expected_adj[e_n]):
                return None
            if e_n in exp_to_inp:
                return None
            inp_to_exp[i_n] = e_n
            exp_to_inp[e_n] = i_n
            queue.append(i_n)

    if len(inp_to_exp) != len(expected_adj):
        return None

    return inp_to_exp


def _match_against_expected_adj(
    graph: Graph,
    expected_adj: Dict[int, Set[int]],
) -> bool:
    """Check degree histogram then try BFS isomorphism from min-degree vertices.

    Args:
        graph: Input graph.
        expected_adj: Expected adjacency as {idx: set of neighbor indices}.

    Returns:
        True if the graph matches the expected adjacency, False otherwise.
    """
    expected_deg: Dict[int, int] = {i: len(nbrs) for i, nbrs in expected_adj.items()}

    exp_deg_hist: Dict[int, int] = {}
    for d in expected_deg.values():
        exp_deg_hist[d] = exp_deg_hist.get(d, 0) + 1
    inp_deg_hist: Dict[int, int] = {}
    for v in graph.nodes:
        d = graph.degree(v)
        inp_deg_hist[d] = inp_deg_hist.get(d, 0) + 1
    if exp_deg_hist != inp_deg_hist:
        return False

    exp_by_deg: Dict[int, List[int]] = {}
    for idx, d in expected_deg.items():
        exp_by_deg.setdefault(d, []).append(idx)
    inp_by_deg: Dict[int, List[int]] = {}
    for v in graph.nodes:
        inp_by_deg.setdefault(graph.degree(v), []).append(v)

    min_deg = min(exp_deg_hist.keys())
    input_origin = inp_by_deg[min_deg][0]
    for exp_origin in exp_by_deg[min_deg]:
        mapping = _try_lattice_bfs_match(
            graph, expected_adj, input_origin, exp_origin
        )
        if mapping is not None:
            return True

    return False


def _is_lattice_isomorphic(
    graph: Graph,
    num_rows: int,
    num_cols: int,
    offsets: List[Tuple[int, int]],
) -> bool:
    """Check if graph is isomorphic to a lattice defined by uniform offsets.

    Builds expected adjacency for an num_rows x num_cols lattice where
    neighbors of (r,c) are (r+dr, c+dc) for each (dr,dc) in offsets.
    Only works for lattices with uniform adjacency rules (grid, triangular).
    Honeycomb and square-octagon have position-dependent adjacency and
    must build their own expected adjacency via dedicated functions.

    Args:
        graph: Input graph.
        num_rows: Number of rows in the expected lattice.
        num_cols: Number of columns in the expected lattice.
        offsets: List of (dr, dc) neighbor offsets defining the lattice.

    Returns:
        True if the graph matches the expected lattice, False otherwise.

    Complexity: O(V + E).
    """
    n_expected = num_rows * num_cols
    if graph.node_count() != n_expected:
        return False

    expected_adj: Dict[int, Set[int]] = {i: set() for i in range(n_expected)}
    for r in range(num_rows):
        for c in range(num_cols):
            idx = r * num_cols + c
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < num_rows and 0 <= nc < num_cols:
                    expected_adj[idx].add(nr * num_cols + nc)

    return _match_against_expected_adj(graph, expected_adj)


# =============================================================================
# 2. GRID
# =============================================================================


def detect_grid_dims_with_bfs(
    graph: Graph, fp: StructuralFingerprint
) -> Optional[Tuple[int, int]]:
    """Detect if graph is a grid P_width x P_length with structural BFS.

    Checks bipartiteness, degree distribution, and solves the quadratic
    for dimensions, then verifies via BFS isomorphism.

    Args:
        graph: Input graph (simple, undirected).
        fp: Precomputed structural fingerprint.

    Returns:
        (width, length) where width <= length (shorter side is width),
        or None if the graph is not a grid.

    Complexity: O(V + E).
    """
    num_vertices = fp.node_count
    num_edges = fp.edge_count

    if not fp.is_bipartite:
        return None

    if fp.degree_counts.get(2, 0) != 4:
        return None

    for deg in fp.degree_counts:
        if deg not in (2, 3, 4):
            return None

    # Solve for grid dimensions from V and E.
    # V = w*l, E = 2wl - w - l  =>  w+l = 2V-E, w*l = V
    dim_sum = 2 * num_vertices - num_edges
    discriminant = dim_sum * dim_sum - 4 * num_vertices
    if discriminant < 0:
        return None

    sqrt_disc = math.isqrt(discriminant)
    if sqrt_disc * sqrt_disc != discriminant:
        return None

    dim_large = (dim_sum + sqrt_disc) // 2
    dim_small = (dim_sum - sqrt_disc) // 2

    if dim_large * dim_small != num_vertices or dim_large + dim_small != dim_sum:
        return None
    if dim_small < 1 or dim_large < 1:
        return None

    width, length = dim_small, dim_large

    # Verify degree distribution
    expected_deg3 = 2 * (width - 2) + 2 * (length - 2)
    expected_deg4 = (width - 2) * (length - 2)
    if fp.degree_counts.get(3, 0) != max(0, expected_deg3):
        return None
    if fp.degree_counts.get(4, 0) != max(0, expected_deg4):
        return None

    # Degenerate grids (1xn or 2xn) handled by path/ladder recognizers
    if width <= 2:
        return None

    # Structural BFS verification — try both orientations
    for num_rows, num_cols in [(width, length), (length, width)]:
        if _is_lattice_isomorphic(graph, num_rows, num_cols, _GRID_OFFSETS):
            return (width, length)

    return None


def _grid_unit_cell_edges(width: int) -> List[Tuple[int, int, bool]]:
    """Unit cell edge pattern for grid P_m x P_n.

    m horizontal cross-column edges + (m-1) vertical within-column edges.
    """
    edges: List[Tuple[int, int, bool]] = []
    for i in range(width):
        edges.append((i, i, True))
    for i in range(width - 1):
        edges.append((i, i + 1, False))
    return edges


# =============================================================================
# 3. TRIANGULAR
# =============================================================================


def detect_triangular_dims_with_bfs(
    graph: Graph, fp: StructuralFingerprint
) -> Optional[Tuple[int, int]]:
    """Detect if graph is a triangular lattice strip (grid + one diagonal per cell).

    Both NE-SW and NW-SE diagonal orientations are detected (they are
    isomorphic via column reflection and produce the same Tutte polynomial).

    Args:
        graph: Input graph (simple, undirected).
        fp: Precomputed structural fingerprint.

    Returns:
        (width, length) where width <= length, or None if not triangular.

    Complexity: O(V + E).
    """
    num_vertices = fp.node_count
    num_edges = fp.edge_count

    if fp.is_bipartite:
        return None

    for deg in fp.degree_counts:
        if deg not in (2, 3, 4, 5, 6):
            return None

    # Solve for dimensions: V = m*n, E = 3mn - 2m - 2n + 1
    # => m+n = (3V - E + 1) / 2,  m*n = V
    numerator = 3 * num_vertices - num_edges + 1
    if numerator % 2 != 0:
        return None
    dim_sum = numerator // 2

    discriminant = dim_sum * dim_sum - 4 * num_vertices
    if discriminant < 0:
        return None

    sqrt_disc = math.isqrt(discriminant)
    if sqrt_disc * sqrt_disc != discriminant:
        return None

    dim_large = (dim_sum + sqrt_disc) // 2
    dim_small = (dim_sum - sqrt_disc) // 2

    if dim_large * dim_small != num_vertices or dim_large + dim_small != dim_sum:
        return None
    if dim_small < 1 or dim_large < 1:
        return None

    if dim_small <= 2:
        return None

    width, length = dim_small, dim_large

    # Structural BFS verification — try both orientations and diagonal directions
    for num_rows, num_cols in [(width, length), (length, width)]:
        if num_rows < 3 or num_cols < 3:
            continue
        if (_is_lattice_isomorphic(graph, num_rows, num_cols, _NE_SW_OFFSETS)
                or _is_lattice_isomorphic(graph, num_rows, num_cols, _NW_SE_OFFSETS)):
            return (width, length)

    return None


def _triangular_unit_cell_edges(width: int) -> List[Tuple[int, int, bool]]:
    """Unit cell edge pattern for triangular lattice strip of width m.

    m horizontal cross + (m-1) vertical within + (m-1) diagonal cross.
    """
    edges: List[Tuple[int, int, bool]] = []
    for i in range(width):
        edges.append((i, i, True))
    for i in range(width - 1):
        edges.append((i, i + 1, False))
    for i in range(width - 1):
        edges.append((i, i + 1, True))
    return edges


# =============================================================================
# 4. HONEYCOMB
# =============================================================================

# Brick-wall honeycomb with vertex_rows (even) x vertex_cols grid.
# Edges: all horizontal + column-parity-dependent vertical pairs.
#   Even columns: vertical pairs (0,1), (2,3), (4,5), ...
#   Odd columns:  vertical pairs (1,2), (3,4), (5,6), ...
# Bipartite, mostly degree 2-3, with degree 1 at two corners when
# vertex_cols is even.

HONEYCOMB_MIN_HEX_ROWS: int = 2
HONEYCOMB_MIN_HEX_COLS: int = 2


def detect_honeycomb_dims_with_bfs(
    graph: Graph, fp: StructuralFingerprint
) -> Optional[Tuple[int, int, int, int]]:
    """Detect if graph is a honeycomb (brick-wall) lattice strip.

    Enumerates factor pairs of V with vertex_rows even, checks the
    honeycomb edge count formula, then verifies via BFS isomorphism.

    Args:
        graph: Input graph (simple, undirected).
        fp: Precomputed structural fingerprint.

    Returns:
        (hex_rows, hex_cols, vertex_rows, vertex_cols) or None.
        hex_rows >= 2, hex_cols >= 2.
        vertex_rows = 2 * hex_rows, vertex_cols = hex_cols + 1.
        The "width" for transfer matrix purposes is vertex_rows.

    Complexity: O(V + E).
    """
    num_vertices = fp.node_count
    num_edges = fp.edge_count

    if not fp.is_bipartite:
        return None

    for deg in fp.degree_counts:
        if deg not in (1, 2, 3):
            return None

    if fp.degree_counts.get(1, 0) > 2:
        return None

    for vertex_rows in range(4, num_vertices + 1, 2):
        if num_vertices % vertex_rows != 0:
            continue
        vertex_cols = num_vertices // vertex_rows
        if vertex_cols < 3:
            continue

        expected_edges = _honeycomb_edge_count(vertex_rows, vertex_cols)
        if expected_edges != num_edges:
            continue

        hex_rows = vertex_rows // 2
        hex_cols = vertex_cols - 1
        if hex_rows < HONEYCOMB_MIN_HEX_ROWS or hex_cols < HONEYCOMB_MIN_HEX_COLS:
            continue

        if _is_honeycomb_isomorphic(graph, vertex_rows, vertex_cols):
            return (hex_rows, hex_cols, vertex_rows, vertex_cols)

    return None


def _honeycomb_edge_count(vertex_rows: int, vertex_cols: int) -> int:
    """Expected edge count for a brick-wall honeycomb."""
    num_horizontal = vertex_rows * (vertex_cols - 1)
    half_rows = vertex_rows // 2
    num_even_cols = (vertex_cols + 1) // 2
    num_odd_cols = vertex_cols // 2
    num_vertical = half_rows * num_even_cols + (half_rows - 1) * num_odd_cols
    return num_horizontal + num_vertical


def _build_honeycomb_adj(
    vertex_rows: int, vertex_cols: int,
) -> Dict[int, Set[int]]:
    """Build expected adjacency for a brick-wall honeycomb.

    Cannot use uniform offsets because vertical edges depend on column parity.
    """
    n = vertex_rows * vertex_cols
    expected_adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for r in range(vertex_rows):
        for c in range(vertex_cols):
            idx = r * vertex_cols + c
            if c > 0:
                expected_adj[idx].add(r * vertex_cols + (c - 1))
            if c < vertex_cols - 1:
                expected_adj[idx].add(r * vertex_cols + (c + 1))
            if c % 2 == 0:
                partner = r ^ 1
                if 0 <= partner < vertex_rows:
                    expected_adj[idx].add(partner * vertex_cols + c)
            else:
                if r % 2 == 1 and r + 1 < vertex_rows:
                    expected_adj[idx].add((r + 1) * vertex_cols + c)
                elif r % 2 == 0 and r > 0:
                    expected_adj[idx].add((r - 1) * vertex_cols + c)
    return expected_adj


def _is_honeycomb_isomorphic(
    graph: Graph,
    vertex_rows: int,
    vertex_cols: int,
) -> bool:
    """BFS isomorphism check for honeycomb lattice.

    Builds column-parity-dependent adjacency, then uses the shared
    degree-histogram + BFS matching infrastructure.
    """
    n_expected = vertex_rows * vertex_cols
    if graph.node_count() != n_expected:
        return False

    expected_adj = _build_honeycomb_adj(vertex_rows, vertex_cols)
    return _match_against_expected_adj(graph, expected_adj)


def _honeycomb_unit_cell_edges(vertex_rows: int) -> List[Tuple[int, int, bool]]:
    """Unit cell edge pattern for honeycomb even-column transition.

    Even columns have vertical pairs (0,1), (2,3), (4,5), ...
    Paired with _honeycomb_unit_cell_edges_odd for the alternating pattern.
    """
    edges: List[Tuple[int, int, bool]] = []
    for i in range(vertex_rows):
        edges.append((i, i, True))
    half_rows = vertex_rows // 2
    for k in range(half_rows):
        edges.append((2 * k, 2 * k + 1, False))
    return edges


def _honeycomb_unit_cell_edges_odd(vertex_rows: int) -> List[Tuple[int, int, bool]]:
    """Unit cell edge pattern for honeycomb odd-column transition.

    Odd columns have vertical pairs (1,2), (3,4), (5,6), ...
    """
    edges: List[Tuple[int, int, bool]] = []
    for i in range(vertex_rows):
        edges.append((i, i, True))
    half_rows = vertex_rows // 2
    for k in range(half_rows - 1):
        edges.append((2 * k + 1, 2 * k + 2, False))
    return edges


# =============================================================================
# 5. SQUARE-OCTAGON (4.8.8 TRUNCATED SQUARE TILING)
# =============================================================================

# The truncated square tiling is built from a grid of "small squares":
#   - sq_rows x sq_cols small squares in the grid.
#   - vertex_rows = 2 * sq_rows, vertex_cols = 2 * sq_cols.
#   - Each small square has 4 vertices forming a 4-cycle.
#   - Between adjacent squares, exactly one connector edge exists per corner
#     in a checkerboard pattern (keeping all vertex degrees <= 3).
#   - The octagon faces emerge at each grid cell where 4 squares meet.
#
# Bipartite, vertex degrees 2 or 3, girth 4.
# Period-4 transfer matrix (4 distinct transition patterns per period).

SQUARE_OCTAGON_MIN_SQ_ROWS: int = 2
SQUARE_OCTAGON_MIN_SQ_COLS: int = 2


def _square_octagon_edge_count(vertex_rows: int, vertex_cols: int) -> int:
    """Expected edge count for a truncated square (4.8.8) strip.

    E = (3 * vr * vc - vr - vc) / 2 where vr, vc are both even.
    Derived from: 4*sq_rows*sq_cols within-square edges +
    (2*sq_rows*sq_cols - sq_rows - sq_cols) between-square connectors.
    """
    return (3 * vertex_rows * vertex_cols - vertex_rows - vertex_cols) // 2


def _build_square_octagon_adj(
    vertex_rows: int, vertex_cols: int,
) -> Dict[int, Set[int]]:
    """Build expected adjacency for a truncated square (4.8.8) tiling strip.

    Constructs the tiling from a grid of small squares with checkerboard
    between-square connectors. vertex_rows and vertex_cols must both be even.
    """
    sq_rows = vertex_rows // 2
    sq_cols = vertex_cols // 2
    num_verts = vertex_rows * vertex_cols
    adj: Dict[int, Set[int]] = {i: set() for i in range(num_verts)}

    def idx(r: int, c: int) -> int:
        return r * vertex_cols + c

    for i in range(sq_rows):
        for j in range(sq_cols):
            tl = idx(2 * i, 2 * j)
            tr = idx(2 * i, 2 * j + 1)
            bl = idx(2 * i + 1, 2 * j)
            br = idx(2 * i + 1, 2 * j + 1)

            # Within-square 4-cycle
            adj[tl].add(tr); adj[tr].add(tl)
            adj[tr].add(br); adj[br].add(tr)
            adj[br].add(bl); adj[bl].add(br)
            adj[bl].add(tl); adj[tl].add(bl)

            # Between-square connectors (checkerboard pattern)
            if (i + j) % 2 == 0:
                if i > 0:
                    nb = idx(2 * i - 1, 2 * j)
                    adj[tl].add(nb); adj[nb].add(tl)
                if j + 1 < sq_cols:
                    nb = idx(2 * i, 2 * j + 2)
                    adj[tr].add(nb); adj[nb].add(tr)
                if j > 0:
                    nb = idx(2 * i + 1, 2 * j - 1)
                    adj[bl].add(nb); adj[nb].add(bl)
                if i + 1 < sq_rows:
                    nb = idx(2 * i + 2, 2 * j + 1)
                    adj[br].add(nb); adj[nb].add(br)
            else:
                if j > 0:
                    nb = idx(2 * i, 2 * j - 1)
                    adj[tl].add(nb); adj[nb].add(tl)
                if i > 0:
                    nb = idx(2 * i - 1, 2 * j + 1)
                    adj[tr].add(nb); adj[nb].add(tr)
                if i + 1 < sq_rows:
                    nb = idx(2 * i + 2, 2 * j)
                    adj[bl].add(nb); adj[nb].add(bl)
                if j + 1 < sq_cols:
                    nb = idx(2 * i + 1, 2 * j + 2)
                    adj[br].add(nb); adj[nb].add(br)

    return adj


def _is_square_octagon_isomorphic(
    graph: Graph,
    vertex_rows: int,
    vertex_cols: int,
) -> bool:
    """BFS isomorphism check for truncated square (4.8.8) tiling."""
    if graph.node_count() != vertex_rows * vertex_cols:
        return False
    expected_adj = _build_square_octagon_adj(vertex_rows, vertex_cols)
    return _match_against_expected_adj(graph, expected_adj)


def detect_square_octagon_dims_with_bfs(
    graph: Graph, fp: StructuralFingerprint,
) -> Optional[Tuple[int, int, int, int]]:
    """Detect if graph is a truncated square (4.8.8) lattice strip.

    Enumerates factor pairs of V with both vertex_rows and vertex_cols even,
    checks the edge count formula, then verifies via BFS isomorphism.

    Args:
        graph: Input graph (simple, undirected).
        fp: Precomputed structural fingerprint.

    Returns:
        (sq_rows, sq_cols, vertex_rows, vertex_cols) or None.
        sq_rows >= 2, sq_cols >= 2.
        vertex_rows = 2 * sq_rows, vertex_cols = 2 * sq_cols.

    Complexity: O(V + E).
    """
    num_vertices = fp.node_count
    num_edges = fp.edge_count

    if not fp.is_bipartite:
        return None

    for deg in fp.degree_counts:
        if deg not in (2, 3):
            return None

    # vertex_rows and vertex_cols must both be even.
    for vertex_rows in range(4, num_vertices + 1, 2):
        if num_vertices % vertex_rows != 0:
            continue
        vertex_cols = num_vertices // vertex_rows
        if vertex_cols < 4 or vertex_cols % 2 != 0:
            continue

        expected_edges = _square_octagon_edge_count(vertex_rows, vertex_cols)
        if expected_edges != num_edges:
            continue

        sq_rows = vertex_rows // 2
        sq_cols = vertex_cols // 2
        if sq_rows < SQUARE_OCTAGON_MIN_SQ_ROWS:
            continue
        if sq_cols < SQUARE_OCTAGON_MIN_SQ_COLS:
            continue

        if _is_square_octagon_isomorphic(graph, vertex_rows, vertex_cols):
            return (sq_rows, sq_cols, vertex_rows, vertex_cols)

    return None


def _square_octagon_unit_cell_patterns(
    vertex_rows: int, vertex_cols: int,
) -> Tuple[List[List[Tuple[int, int, bool]]], List[Tuple[int, int]]]:
    """Extract the period-4 unit cell patterns and first column edges.

    Builds the expected adjacency and reads off which edges exist at each
    column transition and in the first column. The pattern repeats every
    4 columns.

    Args:
        vertex_rows: Number of vertex rows (even).
        vertex_cols: Number of vertex columns (even, >= 4).

    Returns:
        (transition_patterns, first_col_edges) where transition_patterns
        is a list of 4 edge-pattern lists (period-4 cycle) and
        first_col_edges is the within-column edges for column 0.
    """
    adj = _build_square_octagon_adj(vertex_rows, vertex_cols)

    def has_edge(r1: int, c1: int, r2: int, c2: int) -> bool:
        return (r2 * vertex_cols + c2) in adj[r1 * vertex_cols + c1]

    # First column within-column edges.
    first_col_edges: List[Tuple[int, int]] = []
    for r in range(vertex_rows - 1):
        if has_edge(r, 0, r + 1, 0):
            first_col_edges.append((r, r + 1))

    # Extract the first 4 transition patterns (period-4).
    transition_patterns: List[List[Tuple[int, int, bool]]] = []
    for c in range(4):
        edges: List[Tuple[int, int, bool]] = []
        # Cross-column edges: old[r] -> new[r]
        for r in range(vertex_rows):
            if has_edge(r, c, r, c + 1):
                edges.append((r, r, True))
        # Within new column: new[r] -> new[r+1]
        for r in range(vertex_rows - 1):
            if has_edge(r, c + 1, r + 1, c + 1):
                edges.append((r, r + 1, False))
        transition_patterns.append(edges)

    return transition_patterns, first_col_edges


# =============================================================================
# 6. ELONGATED TRIANGULAR (3.3.3.4.4 SEMI-REGULAR TILING)
# =============================================================================

# The elongated triangular tiling is a grid with NE-SW diagonals added on
# even-row transitions only (r % 2 == 0):
#   - Same vertex grid as grid: num_rows x num_cols.
#   - Grid edges (horizontal + vertical) everywhere.
#   - Diagonal (r, c) -> (r+1, c+1) only when r is even.
#   - Not bipartite (has triangles), girth 3, vertex degrees 2-5.
#   - Period-1 transfer matrix (single unit cell pattern).
#   - E = 2 * num_rows * num_cols - num_rows - num_cols
#       + (num_rows // 2) * (num_cols - 1).

ELONGATED_TRIANGULAR_MIN_ROWS: int = 3
ELONGATED_TRIANGULAR_MIN_COLS: int = 3


def _elongated_triangular_edge_count(num_rows: int, num_cols: int) -> int:
    """Expected edge count for an elongated triangular (3.3.3.4.4) strip.

    E = 2 * num_rows * num_cols - num_rows - num_cols
      + (num_rows // 2) * (num_cols - 1).

    Grid edges plus NE-SW diagonals on even-row transitions only.
    """
    grid_edges = 2 * num_rows * num_cols - num_rows - num_cols
    # Even-row transitions: r in {0, 2, ..., 2*floor((m-2)/2)}, count = m // 2.
    num_even_transitions = num_rows // 2
    diagonal_edges = num_even_transitions * (num_cols - 1)
    return grid_edges + diagonal_edges


def _build_elongated_triangular_adj(
    num_rows: int, num_cols: int,
) -> Dict[int, Set[int]]:
    """Build expected adjacency for an elongated triangular (3.3.3.4.4) strip.

    Grid edges plus NE-SW diagonals on even-row transitions (r % 2 == 0).
    Cannot use uniform offsets because diagonal edges depend on row parity.
    """
    num_verts = num_rows * num_cols
    adj: Dict[int, Set[int]] = {i: set() for i in range(num_verts)}
    for r in range(num_rows):
        for c in range(num_cols):
            idx = r * num_cols + c
            # Right
            if c + 1 < num_cols:
                nb = r * num_cols + c + 1
                adj[idx].add(nb); adj[nb].add(idx)
            # Down
            if r + 1 < num_rows:
                nb = (r + 1) * num_cols + c
                adj[idx].add(nb); adj[nb].add(idx)
            # NE-SW diagonal: only on even-row transitions
            if r % 2 == 0 and r + 1 < num_rows and c + 1 < num_cols:
                nb = (r + 1) * num_cols + c + 1
                adj[idx].add(nb); adj[nb].add(idx)
    return adj


def _is_elongated_triangular_isomorphic(
    graph: Graph,
    num_rows: int,
    num_cols: int,
) -> bool:
    """BFS isomorphism check for elongated triangular (3.3.3.4.4) lattice."""
    if graph.node_count() != num_rows * num_cols:
        return False
    expected_adj = _build_elongated_triangular_adj(num_rows, num_cols)
    return _match_against_expected_adj(graph, expected_adj)


def detect_elongated_triangular_dims_with_bfs(
    graph: Graph, fp: StructuralFingerprint,
) -> Optional[Tuple[int, int]]:
    """Detect if graph is an elongated triangular (3.3.3.4.4) lattice strip.

    Grid with NE-SW diagonals on even-row transitions only. Produces a
    semi-regular tiling with vertex degrees 2-5 and girth 3.

    Enumerates factor pairs of V, checks the edge count formula,
    then verifies via BFS isomorphism.

    Args:
        graph: Input graph (simple, undirected).
        fp: Precomputed structural fingerprint.

    Returns:
        (num_rows, num_cols) where num_rows is the row count (transfer
        matrix boundary width) and num_cols is the column count (sweep
        direction), or None if not detected. Unlike grid/triangular,
        num_rows is NOT normalized to be <= num_cols since swapping
        dimensions produces a non-isomorphic graph.

    Complexity: O(V + E).
    """
    num_vertices = fp.node_count
    num_edges = fp.edge_count

    if fp.is_bipartite:
        return None

    for deg in fp.degree_counts:
        if deg not in (2, 3, 4, 5):
            return None

    # Enumerate factor pairs. Iterate num_rows from smallest to find
    # the orientation with smallest transfer-matrix width first.
    for num_rows in range(ELONGATED_TRIANGULAR_MIN_ROWS, num_vertices + 1):
        if num_vertices % num_rows != 0:
            continue
        num_cols = num_vertices // num_rows
        if num_cols < ELONGATED_TRIANGULAR_MIN_COLS:
            continue

        expected_edges = _elongated_triangular_edge_count(num_rows, num_cols)
        if expected_edges != num_edges:
            continue

        if _is_elongated_triangular_isomorphic(graph, num_rows, num_cols):
            return (num_rows, num_cols)

    return None


def _elongated_triangular_unit_cell_edges(
    width: int,
) -> List[Tuple[int, int, bool]]:
    """Unit cell edge pattern for elongated triangular (3.3.3.4.4) strip.

    width horizontal cross + (width // 2) diagonal cross +
    (width - 1) vertical within.

    Cross edges: (i, i, True) for all rows, (i, i+1, True) for even i.
    Within: (i, i+1, False) for all rows.
    """
    edges: List[Tuple[int, int, bool]] = []
    # Horizontal cross-column edges
    for i in range(width):
        edges.append((i, i, True))
    # Diagonal cross-column edges (even rows only)
    for i in range(0, width - 1, 2):
        edges.append((i, i + 1, True))
    # Vertical within-column edges
    for i in range(width - 1):
        edges.append((i, i + 1, False))
    return edges


# =============================================================================
# 7. ENTRY POINT
# =============================================================================


class StripProperties(NamedTuple):
    """Properties of a detected periodic lattice strip."""
    width: int
    length: int
    transition_patterns: List[List[Tuple[int, int, bool]]]
    num_vertices: int
    first_col_edges: List[Tuple[int, int]]


def _extract_within_column_edges(
    unit_cell_edges: List[Tuple[int, int, bool]],
) -> List[Tuple[int, int]]:
    """Extract within-column edges from a unit cell pattern.

    Used as the first column's internal edges for the initial state vector.
    """
    return [(a, b) for a, b, is_cross in unit_cell_edges if not is_cross]


def detect_periodic_strip(
    graph: Graph, fp: StructuralFingerprint
) -> Optional[StripProperties]:
    """Detect if graph is a periodic lattice strip and extract transfer-matrix properties.

    Tries each lattice detector in order: grid, triangular, honeycomb,
    square-octagon, elongated triangular. Each detector performs both
    fingerprint checks and structural BFS verification. Returns the
    first match.

    Args:
        graph: Input graph (simple, undirected).
        fp: Precomputed structural fingerprint.

    Returns:
        StripProperties or None if no lattice structure is detected.

    Complexity: O(V + E).
    """
    num_vertices = len(graph.nodes)

    # Grid P_m x P_n
    grid_result = detect_grid_dims_with_bfs(graph, fp)
    if grid_result is not None:
        width, length = grid_result
        if width <= MAX_TRANSFER_MATRIX_WIDTH:
            edges = _grid_unit_cell_edges(width)
            first_col = _extract_within_column_edges(edges)
            return StripProperties(width, length, [edges], num_vertices, first_col)

    # Triangular lattice strip
    tri_result = detect_triangular_dims_with_bfs(graph, fp)
    if tri_result is not None:
        width, length = tri_result
        if width <= MAX_TRANSFER_MATRIX_WIDTH:
            edges = _triangular_unit_cell_edges(width)
            first_col = _extract_within_column_edges(edges)
            return StripProperties(width, length, [edges], num_vertices, first_col)

    # Honeycomb (brick-wall) lattice strip
    hc_result = detect_honeycomb_dims_with_bfs(graph, fp)
    if hc_result is not None:
        _, _, vertex_rows, vertex_cols = hc_result
        if vertex_rows <= MAX_TRANSFER_MATRIX_WIDTH:
            even_edges = _honeycomb_unit_cell_edges(vertex_rows)
            odd_edges = _honeycomb_unit_cell_edges_odd(vertex_rows)
            first_col = _extract_within_column_edges(even_edges)
            return StripProperties(
                vertex_rows, vertex_cols, [odd_edges, even_edges],
                num_vertices, first_col,
            )

    # Square-octagon (4.8.8 truncated square) lattice strip
    so_result = detect_square_octagon_dims_with_bfs(graph, fp)
    if so_result is not None:
        _, _, vertex_rows, vertex_cols = so_result
        if vertex_rows <= MAX_TRANSFER_MATRIX_WIDTH:
            patterns, first_col = _square_octagon_unit_cell_patterns(
                vertex_rows, vertex_cols,
            )
            return StripProperties(
                vertex_rows, vertex_cols, patterns,
                num_vertices, first_col,
            )

    # Elongated triangular (3.3.3.4.4) lattice strip.
    # Checked last: not bipartite (no overlap with honeycomb/square-octagon),
    # degrees 2-5 (no degree 6, so no overlap with triangular), and uses
    # factor-pair enumeration rather than a direct quadratic solve.
    et_result = detect_elongated_triangular_dims_with_bfs(graph, fp)
    if et_result is not None:
        width, length = et_result
        if width <= MAX_TRANSFER_MATRIX_WIDTH:
            edges = _elongated_triangular_unit_cell_edges(width)
            first_col = _extract_within_column_edges(edges)
            return StripProperties(width, length, [edges], num_vertices, first_col)

    return None
