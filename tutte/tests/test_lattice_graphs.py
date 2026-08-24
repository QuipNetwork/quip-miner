"""Test suite for lattice graph Tutte polynomial computation via transfer matrix.

Tests lattice detection and the transfer-matrix pipeline for five lattice
families:
  - Grid (P_m x P_n)
  - Triangular (grid + NE-SW diagonals)
  - Honeycomb (brick-wall, period-2 alternating transfer matrices)
  - Square-octagon (4.8.8 truncated square, period-4 transfer matrices)
  - Elongated triangular (3.3.3.4.4 semi-regular, period-1 transfer matrices)

Covers:
  1. Lattice detection (BFS-based dimension extraction for each family)
  2. Transfer matrix construction (non-crossing partitions, matrix build)
  3. Direct multiply sweep (single-period and multi-period)
  4. Extraction and polynomial correctness
  5. Cross-rejection between lattice families
  6. Edge perturbation and isomorphism invariance

Validation approaches:
  - T(1,1) = spanning tree count (Kirchhoff's matrix-tree theorem)
  - T(2,2) = 2^|E| (universal property)
  - Cross-validation against NetworkX for small grids (<=15 edges)
  - Cross-validation against SynthesisEngine for grids and triangular

Usage:
    pytest tests/test_lattice_graphs.py -v
    pytest tests/test_lattice_graphs.py -v -m "not slow"

    # Interactive lattice comparison (transfer matrix vs Kirchhoff):
    LATTICE=grid ROWS=4 COLS=6 pytest tests/test_lattice_graphs.py::test_lattice_comparison -v -s
"""

import os
import time

import networkx as nx
import pytest

from tutte.graph import Graph
from tutte.polynomial import TuttePolynomial
from tutte.family_recognition import compute_structural_fingerprint
from tutte.transfer_matrix.lattice_recognition import (
    detect_periodic_strip,
    detect_grid_dims_with_bfs,
    detect_triangular_dims_with_bfs,
    detect_honeycomb_dims_with_bfs,
    detect_square_octagon_dims_with_bfs,
    detect_elongated_triangular_dims_with_bfs,
    _grid_unit_cell_edges,
    _triangular_unit_cell_edges,
    _honeycomb_unit_cell_edges,
    _honeycomb_unit_cell_edges_odd,
    _honeycomb_edge_count,
    _square_octagon_edge_count,
    _elongated_triangular_edge_count,
    _elongated_triangular_unit_cell_edges,
)
from tutte.transfer_matrix.core import (
    build_transfer_matrix,
    enumerate_noncrossing_partitions,
    partition_index_map,
    CATALAN_NUMBERS,
    MAX_TRANSFER_MATRIX_WIDTH,
)
from tutte.transfer_matrix.sweep import (
    direct_multiply,
    build_initial_vector,
)
from tutte.transfer_matrix.extraction import extract_tutte_polynomial
from tutte.transfer_matrix import (
    compute_strip_polynomial,
    compute_tutte_via_transfer_matrix,
    detect_periodic_strip,
)
from tutte.family_recognition import compute_structural_fingerprint
from tutte.validation import (
    count_spanning_trees_kirchhoff,
    compute_tutte_networkx,
    _exact_num_spanning_trees,
)

import tutte.transfer_matrix._transfer_matrix_c as _c_ext


def _tm_poly_uncapped(graph):
    """Transfer-matrix polynomial with the width cost gate bypassed.

    `compute_tutte_via_transfer_matrix` applies the `_transfer_wins`
    dispatch heuristic, which defers width >= 6 strips to treewidth_dp
    because Catalan(width)^2 outgrows 2^width. That is the right
    production choice, but it is a COST decision, not a capability limit:
    the sweep itself supports width <= MAX_TRANSFER_MATRIX_WIDTH (8).

    Tests below validate the transfer-matrix machinery across lattice
    families and sizes, so they go through the strip entry point directly
    and stay meaningful for width-6 families (honeycomb hex_rows=3 and
    square-octagon 3xN both yield width 6). See bead riff-ghf.
    """
    strip = detect_periodic_strip(graph, compute_structural_fingerprint(graph))
    if strip is None:
        return None
    return compute_strip_polynomial(strip)


# =============================================================================
# HELPERS
# =============================================================================


def _make_grid_nx(m: int, n: int) -> nx.Graph:
    """Build a grid P_m x P_n as a NetworkX graph with integer labels."""
    return nx.convert_node_labels_to_integers(nx.grid_2d_graph(m, n))


def _make_grid(m: int, n: int) -> Graph:
    """Build a grid P_m x P_n as a tutte Graph."""
    return Graph.from_networkx(_make_grid_nx(m, n))


def _grid_edge_count(m: int, n: int) -> int:
    """Number of edges in P_m x P_n: (m-1)*n + m*(n-1) = 2mn - m - n."""
    return 2 * m * n - m - n


def _compute_pipeline(width, length, unit_cell_edges, num_vertices=None,
                       first_col_edges=None):
    """Run the transfer matrix pipeline directly from raw parameters.

    Chains build_transfer_matrix -> direct_multiply -> extract_tutte_polynomial.
    Used by tests that need to exercise the pipeline with specific edge patterns
    without going through lattice detection.
    """
    if num_vertices is None:
        num_vertices = width * length
    mat = build_transfer_matrix(width, unit_cell_edges)
    vec = direct_multiply(mat, width, length, first_col_edges=first_col_edges)
    return extract_tutte_polynomial(vec, width, num_vertices)


def _clear_all_caches():
    """Clear all caches used by the transfer matrix and treewidth DP pipelines."""
    # Transfer matrix lru_cache functions
    enumerate_noncrossing_partitions.cache_clear()
    partition_index_map.cache_clear()
    # Treewidth DP connect cache
    from tutte.graphs.treewidth import _connect_cache
    _connect_cache.clear()


_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from tutte.synthesis.engine import SynthesisEngine
        _engine = SynthesisEngine()
    return _engine


# =============================================================================
# LATTICE GRAPH BUILDERS
# =============================================================================

# Each builder constructs a NetworkX graph with integer labels, matching the
# adjacency expected by the corresponding BFS recognition function in
# lattice_recognition.py.


def _make_triangular_nx(m: int, n: int) -> nx.Graph:
    """Build a triangular lattice strip (grid + NE-SW diagonals) on m rows x n cols.

    Adjacency: (r,c) neighbours (r,c+-1), (r+-1,c), (r+1,c+1), (r-1,c-1).
    """
    G = nx.Graph()
    for r in range(m):
        for c in range(n):
            G.add_node(r * n + c)
    for r in range(m):
        for c in range(n):
            idx = r * n + c
            # Right
            if c + 1 < n:
                G.add_edge(idx, r * n + c + 1)
            # Down
            if r + 1 < m:
                G.add_edge(idx, (r + 1) * n + c)
            # NE-SW diagonal: (r,c) -> (r+1,c+1)
            if r + 1 < m and c + 1 < n:
                G.add_edge(idx, (r + 1) * n + c + 1)
    return G


def _make_triangular(m: int, n: int) -> Graph:
    return Graph.from_networkx(_make_triangular_nx(m, n))


def _make_triangular_nwse_nx(m: int, n: int) -> nx.Graph:
    """Build a triangular lattice strip with NW-SE diagonals on m rows x n cols.

    Adjacency: (r,c) neighbours (r,c+-1), (r+-1,c), (r+1,c-1), (r-1,c+1).
    This is the mirror image of the NE-SW triangular lattice (column reflection).
    """
    G = nx.Graph()
    for r in range(m):
        for c in range(n):
            G.add_node(r * n + c)
    for r in range(m):
        for c in range(n):
            idx = r * n + c
            # Right
            if c + 1 < n:
                G.add_edge(idx, r * n + c + 1)
            # Down
            if r + 1 < m:
                G.add_edge(idx, (r + 1) * n + c)
            # NW-SE diagonal: (r,c) -> (r+1,c-1)
            if r + 1 < m and c - 1 >= 0:
                G.add_edge(idx, (r + 1) * n + c - 1)
    return G


def _triangular_edge_count(m: int, n: int) -> int:
    """E = 3mn - 2m - 2n + 1."""
    return 3 * m * n - 2 * m - 2 * n + 1


def _make_honeycomb_nx(vertex_rows: int, vertex_cols: int) -> nx.Graph:
    """Build a brick-wall honeycomb lattice with given vertex dimensions.

    vertex_rows must be even. Matches the adjacency in _build_honeycomb_adj.
    """
    G = nx.Graph()
    for r in range(vertex_rows):
        for c in range(vertex_cols):
            G.add_node(r * vertex_cols + c)
    for r in range(vertex_rows):
        for c in range(vertex_cols):
            idx = r * vertex_cols + c
            # Horizontal right
            if c + 1 < vertex_cols:
                G.add_edge(idx, r * vertex_cols + c + 1)
            # Vertical (column-parity dependent)
            if c % 2 == 0:
                # Even column: pairs (0,1),(2,3),(4,5),...
                partner = r ^ 1  # flip lowest bit
                if 0 <= partner < vertex_rows:
                    nidx = partner * vertex_cols + c
                    if nidx > idx:  # avoid double-add
                        G.add_edge(idx, nidx)
            else:
                # Odd column: pairs (1,2),(3,4),(5,6),...
                if r % 2 == 1 and r + 1 < vertex_rows:
                    G.add_edge(idx, (r + 1) * vertex_cols + c)
                # (r even, r > 0 case handled by the r-1 odd case above)
    return G


def _make_honeycomb(vertex_rows: int, vertex_cols: int) -> Graph:
    return Graph.from_networkx(_make_honeycomb_nx(vertex_rows, vertex_cols))


def _make_square_octagon_nx(sq_rows: int, sq_cols: int) -> nx.Graph:
    """Build a truncated square (4.8.8) lattice strip.

    sq_rows x sq_cols small squares in the grid.
    vertex_rows = 2 * sq_rows, vertex_cols = 2 * sq_cols.
    """
    vr, vc = 2 * sq_rows, 2 * sq_cols
    G = nx.Graph()
    G.add_nodes_from(range(vr * vc))

    def idx(r, c):
        return r * vc + c

    for i in range(sq_rows):
        for j in range(sq_cols):
            tl, tr = idx(2 * i, 2 * j), idx(2 * i, 2 * j + 1)
            bl, br = idx(2 * i + 1, 2 * j), idx(2 * i + 1, 2 * j + 1)
            # Within-square 4-cycle
            G.add_edge(tl, tr)
            G.add_edge(tr, br)
            G.add_edge(br, bl)
            G.add_edge(bl, tl)
            # Between-square connectors (checkerboard)
            if (i + j) % 2 == 0:
                if i > 0:
                    G.add_edge(tl, idx(2 * i - 1, 2 * j))
                if j + 1 < sq_cols:
                    G.add_edge(tr, idx(2 * i, 2 * j + 2))
                if j > 0:
                    G.add_edge(bl, idx(2 * i + 1, 2 * j - 1))
                if i + 1 < sq_rows:
                    G.add_edge(br, idx(2 * i + 2, 2 * j + 1))
            else:
                if j > 0:
                    G.add_edge(tl, idx(2 * i, 2 * j - 1))
                if i > 0:
                    G.add_edge(tr, idx(2 * i - 1, 2 * j + 1))
                if i + 1 < sq_rows:
                    G.add_edge(bl, idx(2 * i + 2, 2 * j))
                if j + 1 < sq_cols:
                    G.add_edge(br, idx(2 * i + 1, 2 * j + 2))
    return G


def _make_square_octagon(sq_rows: int, sq_cols: int) -> Graph:
    return Graph.from_networkx(_make_square_octagon_nx(sq_rows, sq_cols))


def _make_elongated_triangular_nx(m: int, n: int) -> nx.Graph:
    """Build an elongated triangular (3.3.3.4.4) lattice strip on m rows x n cols.

    Grid edges + NE-SW diagonals on even-row transitions (r % 2 == 0).
    Adjacency: (r,c) neighbours (r,c+-1), (r+-1,c), plus (r+1,c+1) when r even.
    """
    G = nx.Graph()
    for r in range(m):
        for c in range(n):
            G.add_node(r * n + c)
    for r in range(m):
        for c in range(n):
            idx = r * n + c
            # Right
            if c + 1 < n:
                G.add_edge(idx, r * n + c + 1)
            # Down
            if r + 1 < m:
                G.add_edge(idx, (r + 1) * n + c)
            # NE-SW diagonal: only on even-row transitions
            if r % 2 == 0 and r + 1 < m and c + 1 < n:
                G.add_edge(idx, (r + 1) * n + c + 1)
    return G


def _make_elongated_triangular(m: int, n: int) -> Graph:
    return Graph.from_networkx(_make_elongated_triangular_nx(m, n))


# =============================================================================
# A. GRID DETECTION
# =============================================================================


class TestGridDetection:
    """Test BFS-based grid dimension detection."""

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 4), (3, 5), (3, 6),
        (4, 4), (4, 5), (4, 6),
        (5, 5), (5, 7),
    ])
    def test_detect_grid_dims(self, m, n):
        """detect_grid_dims_with_bfs correctly identifies P_m x P_n grids."""
        graph = _make_grid(m, n)
        fp = compute_structural_fingerprint(graph)
        result = detect_grid_dims_with_bfs(graph, fp)
        assert result is not None, f"Failed to detect grid {m}x{n}"
        width, length = result
        assert width == min(m, n), f"Expected width={min(m, n)}, got {width}"
        assert length == max(m, n), f"Expected length={max(m, n)}, got {length}"

    @pytest.mark.parametrize("m,n", [
        (3, 5), (4, 6), (5, 3), (6, 4),
    ])
    def test_dimension_normalization(self, m, n):
        """Grid detection always returns (width, length) with width <= length."""
        graph = _make_grid(m, n)
        fp = compute_structural_fingerprint(graph)
        result = detect_grid_dims_with_bfs(graph, fp)
        assert result is not None
        width, length = result
        assert width <= length, f"width={width} > length={length}"
        assert width == min(m, n)
        assert length == max(m, n)

    def test_non_grid_petersen(self):
        """Petersen graph is not a grid."""
        graph = Graph.from_networkx(nx.petersen_graph())
        fp = compute_structural_fingerprint(graph)
        result = detect_grid_dims_with_bfs(graph, fp)
        assert result is None

    def test_non_grid_complete(self):
        """Complete graph K_5 is not a grid."""
        graph = Graph.from_networkx(nx.complete_graph(5))
        fp = compute_structural_fingerprint(graph)
        result = detect_grid_dims_with_bfs(graph, fp)
        assert result is None

    def test_non_grid_cycle(self):
        """Cycle C_8 is not a grid."""
        graph = Graph.from_networkx(nx.cycle_graph(8))
        fp = compute_structural_fingerprint(graph)
        result = detect_grid_dims_with_bfs(graph, fp)
        assert result is None

    def test_small_grids_skipped(self):
        """Grids with width <= 2 are skipped (handled by path/ladder)."""
        for m, n in [(1, 5), (2, 5), (2, 3)]:
            graph = _make_grid(m, n)
            fp = compute_structural_fingerprint(graph)
            result = detect_grid_dims_with_bfs(graph, fp)
            assert result is None, f"Grid {m}x{n} should be skipped (width <= 2)"

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 5), (4, 4), (4, 6),
    ])
    def test_detect_periodic_strip(self, m, n):
        """detect_periodic_strip returns strip info for grids."""
        graph = _make_grid(m, n)
        fp = compute_structural_fingerprint(graph)
        result = detect_periodic_strip(graph, fp)
        assert result is not None, f"Failed to detect periodic strip for {m}x{n}"
        width, length, transition_patterns, num_vertices, first_col_edges = result
        assert width == min(m, n)
        assert length == max(m, n)
        assert len(transition_patterns) == 1
        assert len(transition_patterns[0]) == 2 * width - 1
        assert num_vertices == m * n

    def test_degenerate_single_vertex(self):
        """Single vertex is not detected as a grid."""
        G = nx.Graph()
        G.add_node(0)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_grid_dims_with_bfs(graph, fp) is None

    def test_degenerate_single_edge(self):
        """Single edge (K_2) is not detected as a grid."""
        graph = Graph.from_networkx(nx.path_graph(2))
        fp = compute_structural_fingerprint(graph)
        assert detect_grid_dims_with_bfs(graph, fp) is None

    def test_degenerate_empty_graph(self):
        """Empty graph is not detected as a grid."""
        G = nx.Graph()
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_grid_dims_with_bfs(graph, fp) is None

    def test_disconnected_grid_like(self):
        """Two disjoint grids are not detected as a single grid."""
        G1 = _make_grid_nx(3, 4)
        G2 = nx.convert_node_labels_to_integers(_make_grid_nx(3, 4), first_label=12)
        G = nx.compose(G1, G2)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_grid_dims_with_bfs(graph, fp) is None


# =============================================================================
# B. TRANSFER MATRIX CONSTRUCTION
# =============================================================================


class TestTransferMatrixConstruction:
    """Test partition enumeration and transfer matrix building."""

    @pytest.mark.parametrize("m", range(1, 7))
    def test_noncrossing_partition_count(self, m):
        """enumerate_noncrossing_partitions(m) returns Catalan(m) partitions."""
        partitions = enumerate_noncrossing_partitions(m)
        assert len(partitions) == CATALAN_NUMBERS[m], (
            f"Width {m}: expected {CATALAN_NUMBERS[m]} partitions, got {len(partitions)}"
        )

    @pytest.mark.parametrize("m", range(1, 7))
    def test_partition_index_map_bijection(self, m):
        """partition_index_map produces a valid bijection."""
        partitions = enumerate_noncrossing_partitions(m)
        idx_map = partition_index_map(m)
        assert len(idx_map) == len(partitions)
        # All indices are 0..len-1
        assert set(idx_map.values()) == set(range(len(partitions)))
        # All partitions are keys
        for p in partitions:
            assert p in idx_map

    @pytest.mark.parametrize("m", range(1, 7))
    def test_unit_cell_edge_count(self, m):
        """_grid_unit_cell_edges(m) returns 2m-1 edges (m horizontal + m-1 vertical)."""
        edges = _grid_unit_cell_edges(m)
        assert len(edges) == 2 * m - 1, (
            f"Width {m}: expected {2*m-1} edges, got {len(edges)}"
        )

    @pytest.mark.parametrize("m", range(1, 7))
    def test_unit_cell_edge_structure(self, m):
        """Unit cell has m cross-column edges and m-1 within-column adjacent pairs."""
        edges = _grid_unit_cell_edges(m)
        cross = [(a, b) for a, b, is_cross in edges if is_cross]
        within = [(a, b) for a, b, is_cross in edges if not is_cross]
        assert len(cross) == m
        assert len(within) == m - 1
        # Cross-column edges connect same row (horizontal)
        for a, b in cross:
            assert a == b
        # Within-column edges connect adjacent rows
        for a, b in within:
            assert b == a + 1

    @pytest.mark.parametrize("width", range(1, 6))
    def test_transfer_matrix_dimensions(self, width):
        """Transfer matrix is c_m x c_m."""
        edges = _grid_unit_cell_edges(width)
        mat = build_transfer_matrix(width, edges)
        cm = CATALAN_NUMBERS[width]
        assert len(mat) == cm, f"Expected {cm} rows, got {len(mat)}"
        for row_idx, row in enumerate(mat):
            assert len(row) == cm, f"Row {row_idx}: expected {cm} cols, got {len(row)}"

    @pytest.mark.parametrize("width", range(1, 6))
    def test_transfer_matrix_not_all_zero(self, width):
        """Transfer matrix has at least one nonzero entry."""
        edges = _grid_unit_cell_edges(width)
        mat = build_transfer_matrix(width, edges)
        has_nonzero = any(
            not mat[i][j].is_zero()
            for i in range(len(mat))
            for j in range(len(mat[0]))
        )
        assert has_nonzero, f"Transfer matrix for width {width} is all zeros"

    @pytest.mark.parametrize("width", range(3, 6))
    def test_transfer_matrix_triangular_dimensions(self, width):
        """Transfer matrix for triangular edges is c_m x c_m."""
        edges = _triangular_unit_cell_edges(width)
        mat = build_transfer_matrix(width, edges)
        cm = CATALAN_NUMBERS[width]
        assert len(mat) == cm
        for row in mat:
            assert len(row) == cm

    @pytest.mark.parametrize("width", range(3, 6))
    def test_transfer_matrix_triangular_not_all_zero(self, width):
        """Transfer matrix for triangular edges has nonzero entries."""
        edges = _triangular_unit_cell_edges(width)
        mat = build_transfer_matrix(width, edges)
        has_nonzero = any(
            not mat[i][j].is_zero()
            for i in range(len(mat))
            for j in range(len(mat[0]))
        )
        assert has_nonzero


# =============================================================================
# C. PIPELINE STEPS
# =============================================================================


class TestPipelineSteps:
    """Test individual pipeline steps in isolation."""

    @pytest.mark.parametrize("width", range(1, 6))
    def test_initial_vector_length(self, width):
        """build_initial_vector(width) has Catalan(width) entries."""
        vec = build_initial_vector(width)
        assert len(vec) == CATALAN_NUMBERS[width]

    @pytest.mark.parametrize("width", range(1, 6))
    def test_initial_vector_not_all_zero(self, width):
        """Initial vector has at least one nonzero entry."""
        vec = build_initial_vector(width)
        has_nonzero = any(not v.is_zero() for v in vec)
        assert has_nonzero, f"Initial vector for width {width} is all zeros"

    @pytest.mark.parametrize("width", [4, 6])
    def test_initial_vector_custom_first_col_edges(self, width):
        """build_initial_vector with honeycomb-style non-consecutive edges."""
        # Honeycomb even column: edges (0,1), (2,3), (4,5), ...
        hc_edges = [(2 * k, 2 * k + 1) for k in range(width // 2)]
        vec = build_initial_vector(width, first_col_edges=hc_edges)
        assert len(vec) == CATALAN_NUMBERS[width]
        has_nonzero = any(not v.is_zero() for v in vec)
        assert has_nonzero, f"Initial vector with honeycomb edges is all zeros"
        # Should differ from grid-style consecutive edges
        vec_grid = build_initial_vector(width)
        differs = any(vec[i] != vec_grid[i] for i in range(len(vec)))
        assert differs, "Honeycomb first-col edges should differ from grid"

    def test_direct_multiply_length_1(self):
        """direct_multiply with length=1 returns the initial vector (no multiplications)."""
        width = 3
        edges = _grid_unit_cell_edges(width)
        mat = build_transfer_matrix(width, edges)
        vec_length_1 = direct_multiply(mat, width, 1)
        vec_initial = build_initial_vector(width)
        assert len(vec_length_1) == len(vec_initial)
        for i in range(len(vec_length_1)):
            assert vec_length_1[i] == vec_initial[i], (
                f"Index {i}: direct_multiply(length=1) differs from initial vector"
            )

    def test_direct_multiply_length_2_differs(self):
        """direct_multiply with length=2 differs from initial vector."""
        width = 3
        edges = _grid_unit_cell_edges(width)
        mat = build_transfer_matrix(width, edges)
        vec1 = direct_multiply(mat, width, 1)
        vec2 = direct_multiply(mat, width, 2)
        # At least one entry should differ
        differs = any(vec1[i] != vec2[i] for i in range(len(vec1)))
        assert differs, "length=2 result should differ from length=1"

    def test_build_initial_vector_invalid_width(self):
        """build_initial_vector raises on width < 1."""
        with pytest.raises(ValueError, match="width must be >= 1"):
            build_initial_vector(0)

    def test_direct_multiply_invalid_length(self):
        """direct_multiply raises on length < 1."""
        edges = _grid_unit_cell_edges(2)
        mat = build_transfer_matrix(2, edges)
        with pytest.raises(ValueError, match="length must be >= 1"):
            direct_multiply(mat, 2, 0)

    def test_pipeline_invalid_width(self):
        """Pipeline raises on width < 1."""
        with pytest.raises(ValueError, match="width must be >= 1"):
            _compute_pipeline(0, 5, [])

    def test_pipeline_invalid_length(self):
        """Pipeline raises on length < 1."""
        edges = _grid_unit_cell_edges(2)
        with pytest.raises(ValueError, match="length must be >= 1"):
            _compute_pipeline(2, 0, edges)


# =============================================================================
# D. TRANSFER MATRIX TUTTE POLYNOMIAL — MAIN CORRECTNESS TESTS
# =============================================================================


class TestTransferMatrixTutte:
    """End-to-end transfer matrix -> Tutte polynomial correctness."""

    # -- Degenerate cases: P_1 x P_n is a path (tree) --

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8])
    def test_path_graph_width_1(self, n):
        """P_1 x P_n is a path with T = x^(n-1)."""
        edges = _grid_unit_cell_edges(1)
        poly = _compute_pipeline(1, n, edges)
        if n == 1:
            expected = TuttePolynomial.one()
        else:
            expected = TuttePolynomial.x(n - 1)
        assert poly == expected, f"P_1x{n}: got {poly}, expected {expected}"

    # -- Small grids cross-validated against NetworkX --

    @pytest.mark.parametrize("m,n", [
        (2, 2), (2, 3), (2, 4), (3, 3),
    ])
    def test_small_grids_vs_networkx(self, m, n):
        """Small grids (<= 15 edges): exact polynomial match against NetworkX."""
        edges = _grid_unit_cell_edges(min(m, n))
        poly = _compute_pipeline(min(m, n), max(m, n), edges)
        G_nx = _make_grid_nx(m, n)
        nx_poly = compute_tutte_networkx(G_nx)
        assert nx_poly is not None, "NetworkX/sympy not available"
        assert poly == nx_poly, (
            f"Grid {m}x{n}: transfer matrix disagrees with NetworkX.\n"
            f"  TM: {poly}\n  NX: {nx_poly}"
        )

    # -- Spanning tree count via Kirchhoff --

    @pytest.mark.parametrize("m,n", [
        (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
        (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
        (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
        (5, 5), (5, 6), (5, 7), (5, 8),
    ])
    def test_spanning_tree_count(self, m, n):
        """T(1,1) equals the Kirchhoff spanning tree count."""
        width, length = min(m, n), max(m, n)
        edges = _grid_unit_cell_edges(width)
        poly = _compute_pipeline(width, length, edges)
        graph = _make_grid(m, n)
        kirchhoff = count_spanning_trees_kirchhoff(graph)
        t11 = _exact_num_spanning_trees(poly)
        assert t11 == kirchhoff, (
            f"Grid {m}x{n}: T(1,1)={t11} != Kirchhoff={kirchhoff}"
        )

    # -- T(2,2) = 2^|E| property --

    @pytest.mark.parametrize("m,n", [
        (2, 2), (2, 3), (2, 4), (2, 5),
        (3, 3), (3, 4), (3, 5),
        (4, 4), (4, 5),
        (5, 5),
    ])
    def test_two_two_property(self, m, n):
        """T(2,2) = 2^|E| for grid P_m x P_n."""
        width, length = min(m, n), max(m, n)
        edges_list = _grid_unit_cell_edges(width)
        poly = _compute_pipeline(width, length, edges_list)
        E = _grid_edge_count(m, n)
        t22 = poly.evaluate(2, 2)
        assert t22 == 2 ** E, (
            f"Grid {m}x{n}: T(2,2)={t22} != 2^{E}={2**E}"
        )

    # -- Symmetry: T(P_m x P_n) == T(P_n x P_m) --

    @pytest.mark.parametrize("m,n", [
        (3, 4), (3, 5), (3, 7), (4, 5), (4, 6),
    ])
    def test_symmetry(self, m, n):
        """T(P_m x P_n) == T(P_n x P_m) — build both orientations from graphs."""
        graph_mn = _make_grid(m, n)
        graph_nm = _make_grid(n, m)

        # Synthesize via the full transfer-matrix pipeline on each graph
        poly_mn = compute_tutte_via_transfer_matrix(graph_mn)
        poly_nm = compute_tutte_via_transfer_matrix(graph_nm)
        assert poly_mn is not None and poly_nm is not None

        assert poly_mn == poly_nm, (
            f"Grid {m}x{n} vs {n}x{m}: polynomials differ"
        )

    # -- Symmetry: square-octagon T(sq_rows x sq_cols) == T(sq_cols x sq_rows) --

    @pytest.mark.parametrize("sr,sc", [
        (2, 3), (2, 4), (3, 2), (3, 4),
    ])
    def test_square_octagon_symmetry(self, sr, sc):
        """T(sq_rows x sq_cols) == T(sq_cols x sq_rows) for square-octagon."""
        graph_ab = _make_square_octagon(sr, sc)
        graph_ba = _make_square_octagon(sc, sr)

        poly_ab = _tm_poly_uncapped(graph_ab)
        poly_ba = _tm_poly_uncapped(graph_ba)
        assert poly_ab is not None and poly_ba is not None

        assert poly_ab == poly_ba, (
            f"Square-octagon {sr}x{sc} vs {sc}x{sr}: polynomials differ"
        )

    # -- Known values --

    def test_single_vertex(self):
        """P_1 x P_1: single vertex, T = 1."""
        edges = _grid_unit_cell_edges(1)
        poly = _compute_pipeline(1, 1, edges)
        assert poly == TuttePolynomial.one()

    def test_single_edge(self):
        """P_1 x P_2: single edge, T = x."""
        edges = _grid_unit_cell_edges(1)
        poly = _compute_pipeline(1, 2, edges)
        assert poly == TuttePolynomial.x()

    def test_2x2_grid(self):
        """P_2 x P_2 has 4 vertices and 4 edges."""
        edges = _grid_unit_cell_edges(2)
        poly = _compute_pipeline(2, 2, edges)
        # Verify basic properties
        E = _grid_edge_count(2, 2)
        assert E == 4
        assert poly.evaluate(2, 2) == 2 ** 4
        # T(1,1) = spanning tree count for 2x2 grid = 4
        assert _exact_num_spanning_trees(poly) == 4

    # -- Larger grids (slow) --

    @pytest.mark.slow
    @pytest.mark.parametrize("m,n", [
        (5, 10), (6, 6), (6, 8),
    ])
    def test_large_grid_spanning_trees(self, m, n):
        """Large grids: verify T(1,1) = spanning tree count only."""
        width, length = min(m, n), max(m, n)
        edges = _grid_unit_cell_edges(width)
        poly = _compute_pipeline(width, length, edges)
        graph = _make_grid(m, n)
        kirchhoff = count_spanning_trees_kirchhoff(graph)
        t11 = _exact_num_spanning_trees(poly)
        assert t11 == kirchhoff, (
            f"Grid {m}x{n}: T(1,1)={t11} != Kirchhoff={kirchhoff}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("m,n", [
        (5, 10), (6, 6),
    ])
    def test_large_grid_two_two(self, m, n):
        """Large grids: verify T(2,2) = 2^|E|."""
        width, length = min(m, n), max(m, n)
        edges = _grid_unit_cell_edges(width)
        poly = _compute_pipeline(width, length, edges)
        E = _grid_edge_count(m, n)
        t22 = poly.evaluate(2, 2)
        assert t22 == 2 ** E


# =============================================================================
# E. INTEGRATION WITH SYNTHESIS ENGINE
# =============================================================================


class TestIntegrationWithEngine:
    """Compare transfer matrix results against the synthesis engine."""

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 4), (3, 5),
    ])
    def test_grid_vs_engine(self, m, n):
        """Transfer matrix matches SynthesisEngine for small/medium grids."""
        graph = _make_grid(m, n)

        tm_poly = compute_tutte_via_transfer_matrix(graph)
        assert tm_poly is not None

        engine = _get_engine()
        engine_result = engine.synthesize(graph)

        assert tm_poly == engine_result.polynomial, (
            f"Grid {m}x{n}: transfer matrix disagrees with engine.\n"
            f"  TM: {tm_poly}\n  Engine: {engine_result.polynomial}"
        )

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 4),
    ])
    def test_triangular_vs_engine(self, m, n):
        """Transfer matrix matches SynthesisEngine for triangular."""
        graph = _make_triangular(m, n)

        tm_poly = compute_tutte_via_transfer_matrix(graph)
        assert tm_poly is not None

        engine = _get_engine()
        engine_result = engine.synthesize(graph)

        assert tm_poly == engine_result.polynomial, (
            f"Triangular {m}x{n}: transfer matrix disagrees with engine.\n"
            f"  TM: {tm_poly}\n  Engine: {engine_result.polynomial}"
        )


# =============================================================================
# F. TRIANGULAR LATTICE DETECTION
# =============================================================================


class TestTriangularDetection:
    """Test BFS-based triangular lattice detection."""

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 4), (3, 5), (3, 6),
        (4, 4), (4, 5), (4, 6),
        (5, 5), (5, 7),
    ])
    def test_detect_triangular_dims(self, m, n):
        """detect_triangular_dims_with_bfs correctly identifies triangular m x n."""
        graph = _make_triangular(m, n)
        fp = compute_structural_fingerprint(graph)
        result = detect_triangular_dims_with_bfs(graph, fp)
        assert result is not None, f"Failed to detect triangular {m}x{n}"
        width, length = result
        assert width == min(m, n), f"Expected width={min(m, n)}, got {width}"
        assert length == max(m, n), f"Expected length={max(m, n)}, got {length}"

    @pytest.mark.parametrize("m,n", [
        (3, 5), (4, 6), (5, 3), (6, 4),
    ])
    def test_dimension_normalization(self, m, n):
        """Triangular detection always returns (width, length) with width <= length."""
        graph = _make_triangular(m, n)
        fp = compute_structural_fingerprint(graph)
        result = detect_triangular_dims_with_bfs(graph, fp)
        assert result is not None
        width, length = result
        assert width <= length
        assert width == min(m, n)
        assert length == max(m, n)

    def test_vertex_count(self):
        """Triangular m x n has m*n vertices."""
        for m, n in [(3, 4), (4, 5), (5, 6)]:
            graph = _make_triangular(m, n)
            assert graph.node_count() == m * n

    def test_edge_count(self):
        """Triangular m x n has 3mn - 2m - 2n + 1 edges."""
        for m, n in [(3, 3), (3, 4), (4, 5), (5, 6)]:
            graph = _make_triangular(m, n)
            expected = _triangular_edge_count(m, n)
            assert graph.edge_count() == expected, (
                f"Tri {m}x{n}: got {graph.edge_count()} edges, expected {expected}"
            )

    def test_not_bipartite(self):
        """Triangular lattice is not bipartite (has odd cycles / triangles)."""
        graph = _make_triangular(3, 3)
        fp = compute_structural_fingerprint(graph)
        assert not fp.is_bipartite

    def test_interior_degree_6(self):
        """Interior vertices of triangular lattice have degree 6."""
        graph = _make_triangular(5, 5)
        fp = compute_structural_fingerprint(graph)
        assert 6 in fp.degree_counts

    def test_non_triangular_petersen(self):
        """Petersen graph is not a triangular lattice."""
        graph = Graph.from_networkx(nx.petersen_graph())
        fp = compute_structural_fingerprint(graph)
        result = detect_triangular_dims_with_bfs(graph, fp)
        assert result is None

    def test_non_triangular_complete(self):
        """Complete graph K_6 is not a triangular lattice."""
        graph = Graph.from_networkx(nx.complete_graph(6))
        fp = compute_structural_fingerprint(graph)
        result = detect_triangular_dims_with_bfs(graph, fp)
        assert result is None

    def test_small_dims_skipped(self):
        """Triangular with width <= 2 is skipped."""
        for m, n in [(2, 5), (2, 3), (1, 5)]:
            graph = _make_triangular(m, n)
            fp = compute_structural_fingerprint(graph)
            result = detect_triangular_dims_with_bfs(graph, fp)
            assert result is None, f"Tri {m}x{n} should be skipped (width <= 2)"

    def test_symmetry_detection(self):
        """Triangular m x n and n x m should detect the same dimensions."""
        graph_34 = _make_triangular(3, 4)
        graph_43 = _make_triangular(4, 3)
        fp_34 = compute_structural_fingerprint(graph_34)
        fp_43 = compute_structural_fingerprint(graph_43)
        r_34 = detect_triangular_dims_with_bfs(graph_34, fp_34)
        r_43 = detect_triangular_dims_with_bfs(graph_43, fp_43)
        assert r_34 is not None and r_43 is not None
        assert r_34 == r_43

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 4), (4, 4), (4, 5),
    ])
    def test_degree_distribution(self, m, n):
        """Verify the expected degree distribution for triangular m x n.

        Triangular lattice degrees:
        - Corner (0,0) and (m-1,n-1): degree 3 (with diagonal)
        - Corner (0,n-1) and (m-1,0): degree 2 (no diagonal out)
        - Various border: 4 or 5
        - Interior: 6
        """
        graph = _make_triangular(m, n)
        fp = compute_structural_fingerprint(graph)
        # Should have degree-2 vertices
        assert fp.degree_counts.get(2, 0) == 2, (
            f"Tri {m}x{n}: expected 2 deg-2 vertices, got {fp.degree_counts.get(2, 0)}"
        )
        # Max degree should be 6 (interior)
        if m >= 3 and n >= 3:
            assert fp.max_degree == 6

    def test_degenerate_single_vertex(self):
        """Single vertex is not detected as a triangular lattice."""
        G = nx.Graph()
        G.add_node(0)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_triangular_dims_with_bfs(graph, fp) is None

    def test_degenerate_single_edge(self):
        """Single edge is not detected as a triangular lattice."""
        graph = Graph.from_networkx(nx.path_graph(2))
        fp = compute_structural_fingerprint(graph)
        assert detect_triangular_dims_with_bfs(graph, fp) is None

    def test_disconnected_triangular_like(self):
        """Two disjoint triangular lattices are not detected."""
        G1 = _make_triangular_nx(3, 4)
        G2 = nx.convert_node_labels_to_integers(_make_triangular_nx(3, 4), first_label=12)
        G = nx.compose(G1, G2)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_triangular_dims_with_bfs(graph, fp) is None

    # -- NW-SE diagonal orientation --

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 4), (3, 5),
        (4, 4), (4, 5),
        (5, 5),
    ])
    def test_detect_nwse_triangular(self, m, n):
        """Triangular lattice with NW-SE diagonals is detected correctly."""
        G = _make_triangular_nwse_nx(m, n)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        result = detect_triangular_dims_with_bfs(graph, fp)
        assert result is not None, f"NW-SE triangular {m}x{n} not detected"
        width, length = result
        assert width == min(m, n)
        assert length == max(m, n)

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 4), (4, 4),
    ])
    def test_nwse_isomorphic_to_nesw(self, m, n):
        """NW-SE and NE-SW triangular lattices are isomorphic."""
        G_nesw = _make_triangular_nx(m, n)
        G_nwse = _make_triangular_nwse_nx(m, n)
        assert nx.is_isomorphic(G_nesw, G_nwse), (
            f"Triangular {m}x{n}: NE-SW and NW-SE should be isomorphic"
        )

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 4), (4, 4),
    ])
    def test_nwse_kirchhoff_matches(self, m, n):
        """NW-SE triangular via transfer matrix gives correct spanning tree count."""
        G_nwse = _make_triangular_nwse_nx(m, n)
        graph = Graph.from_networkx(G_nwse)
        tm_poly = compute_tutte_via_transfer_matrix(graph)
        assert tm_poly is not None
        kirchhoff = count_spanning_trees_kirchhoff(graph)
        t11 = _exact_num_spanning_trees(tm_poly)
        assert t11 == kirchhoff, (
            f"NW-SE Triangular {m}x{n}: T(1,1)={t11} != Kirchhoff={kirchhoff}"
        )


# =============================================================================
# G. HONEYCOMB LATTICE DETECTION
# =============================================================================


class TestHoneycombDetection:
    """Test BFS-based honeycomb (brick-wall) lattice detection."""

    @pytest.mark.parametrize("hex_rows,hex_cols", [
        (2, 2), (2, 4), (2, 6),
        (3, 2), (3, 4), (3, 6),
        (4, 2), (4, 4),
    ])
    def test_detect_honeycomb_dims(self, hex_rows, hex_cols):
        """detect_honeycomb_dims_with_bfs correctly identifies honeycomb lattice.

        Only even hex_cols produce valid honeycombs (odd vertex_cols avoid
        degree-1 boundary vertices that the detector rejects).
        """
        vertex_rows = 2 * hex_rows
        vertex_cols = hex_cols + 1
        graph = _make_honeycomb(vertex_rows, vertex_cols)
        fp = compute_structural_fingerprint(graph)
        result = detect_honeycomb_dims_with_bfs(graph, fp)
        assert result is not None, (
            f"Failed to detect honeycomb hex_rows={hex_rows}, hex_cols={hex_cols} "
            f"(vr={vertex_rows}, vc={vertex_cols})"
        )
        h, w, vr, vc = result
        assert h == hex_rows, f"Expected hex_rows={hex_rows}, got {h}"
        assert w == hex_cols, f"Expected hex_cols={hex_cols}, got {w}"
        assert vr == vertex_rows
        assert vc == vertex_cols

    def test_vertex_count(self):
        """Honeycomb with vertex_rows x vertex_cols has vertex_rows * vertex_cols vertices."""
        for hex_rows, hex_cols in [(2, 2), (3, 4), (4, 6)]:
            vr, vc = 2 * hex_rows, hex_cols + 1
            graph = _make_honeycomb(vr, vc)
            assert graph.node_count() == vr * vc

    def test_edge_count(self):
        """Honeycomb edge count matches _honeycomb_edge_count formula."""
        for hex_rows, hex_cols in [(2, 2), (2, 4), (3, 2), (3, 4), (4, 4)]:
            vr, vc = 2 * hex_rows, hex_cols + 1
            graph = _make_honeycomb(vr, vc)
            expected = _honeycomb_edge_count(vr, vc)
            assert graph.edge_count() == expected, (
                f"Honeycomb (hr={hex_rows}, hc={hex_cols}): "
                f"got {graph.edge_count()} edges, expected {expected}"
            )

    def test_is_bipartite(self):
        """Honeycomb lattice is bipartite."""
        graph = _make_honeycomb(4, 3)
        fp = compute_structural_fingerprint(graph)
        assert fp.is_bipartite

    def test_degrees_only_2_and_3(self):
        """All honeycomb vertices have degree 2 or 3 (even hex_cols only)."""
        for hex_rows, hex_cols in [(2, 2), (3, 2), (3, 4)]:
            vr, vc = 2 * hex_rows, hex_cols + 1
            graph = _make_honeycomb(vr, vc)
            fp = compute_structural_fingerprint(graph)
            for deg in fp.degree_counts:
                assert deg in (2, 3), (
                    f"Honeycomb (hr={hex_rows}, hc={hex_cols}): unexpected degree {deg}"
                )

    def test_non_honeycomb_petersen(self):
        """Petersen graph is not a honeycomb."""
        graph = Graph.from_networkx(nx.petersen_graph())
        fp = compute_structural_fingerprint(graph)
        result = detect_honeycomb_dims_with_bfs(graph, fp)
        assert result is None

    def test_non_honeycomb_cycle(self):
        """Cycle graph is not a honeycomb (regular degree 2, no degree 3)."""
        graph = Graph.from_networkx(nx.cycle_graph(12))
        fp = compute_structural_fingerprint(graph)
        result = detect_honeycomb_dims_with_bfs(graph, fp)
        assert result is None

    def test_non_honeycomb_path(self):
        """Path graph is not a honeycomb."""
        graph = Graph.from_networkx(nx.path_graph(12))
        fp = compute_structural_fingerprint(graph)
        result = detect_honeycomb_dims_with_bfs(graph, fp)
        assert result is None

    def test_small_hex_skipped(self):
        """Honeycomb with hex_rows < 2 or hex_cols < 2 is skipped."""
        # hex_rows=1, hex_cols=2 => vertex_rows=2, vertex_cols=3
        graph = _make_honeycomb(2, 3)
        fp = compute_structural_fingerprint(graph)
        result = detect_honeycomb_dims_with_bfs(graph, fp)
        assert result is None, "hex_rows=1 should be skipped"

    def test_even_vertex_cols_has_degree_1(self):
        """Even vertex_cols produce degree-1 boundary vertices which are now detected.

        The brick-wall model with even vertex_cols has an odd last column,
        so rows 0 and vertex_rows-1 get no vertical partner and only have
        a single horizontal neighbor (degree 1). The detector accepts this.
        """
        # hex_rows=2, hex_cols=3 => vr=4, vc=4 (even vc => odd last column)
        graph = _make_honeycomb(4, 4)
        fp = compute_structural_fingerprint(graph)
        assert 1 in fp.degree_counts, "Should have degree-1 vertices"
        result = detect_honeycomb_dims_with_bfs(graph, fp)
        assert result is not None, "Even vertex_cols honeycomb should be detected"
        assert result == (2, 3, 4, 4)

    def test_honeycomb_edge_count_formula(self):
        """_honeycomb_edge_count matches direct graph edge count."""
        for vr in range(4, 12, 2):
            for vc in range(3, 8):
                expected = _honeycomb_edge_count(vr, vc)
                graph = _make_honeycomb_nx(vr, vc)
                assert graph.number_of_edges() == expected, (
                    f"vr={vr}, vc={vc}: formula={expected}, graph={graph.number_of_edges()}"
                )

    def test_degenerate_single_vertex(self):
        """Single vertex is not detected as a honeycomb."""
        G = nx.Graph()
        G.add_node(0)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_honeycomb_dims_with_bfs(graph, fp) is None

    def test_degenerate_single_edge(self):
        """Single edge is not detected as a honeycomb."""
        graph = Graph.from_networkx(nx.path_graph(2))
        fp = compute_structural_fingerprint(graph)
        assert detect_honeycomb_dims_with_bfs(graph, fp) is None

    def test_disconnected_honeycomb_like(self):
        """Two disjoint honeycombs are not detected."""
        G1 = _make_honeycomb_nx(4, 3)
        G2 = nx.convert_node_labels_to_integers(_make_honeycomb_nx(4, 3), first_label=20)
        G = nx.compose(G1, G2)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_honeycomb_dims_with_bfs(graph, fp) is None


# =============================================================================
# H. SQUARE-OCTAGON (4.8.8) LATTICE DETECTION
# =============================================================================


class TestSquareOctagonDetection:
    """Test BFS-based square-octagon (truncated square) lattice detection."""

    @pytest.mark.parametrize("sq_rows,sq_cols", [
        (2, 2), (2, 3), (2, 4), (2, 5),
        (3, 2), (3, 3), (3, 4),
    ])
    def test_detect_dims(self, sq_rows, sq_cols):
        """detect_square_octagon_dims_with_bfs correctly identifies the tiling.

        The detector iterates vertex_rows from smallest, so it may return
        the transposed orientation (width <= length normalization).
        """
        graph = _make_square_octagon(sq_rows, sq_cols)
        fp = compute_structural_fingerprint(graph)
        result = detect_square_octagon_dims_with_bfs(graph, fp)
        assert result is not None, (
            f"Failed to detect square-octagon sq={sq_rows}x{sq_cols}"
        )
        sr, sc, vr, vc = result
        # Detector may return transposed dims, so check the set of dimensions
        assert {sr, sc} == {sq_rows, sq_cols}
        assert vr == 2 * sr
        assert vc == 2 * sc

    def test_vertex_count(self):
        """Square-octagon with sq_rows x sq_cols has 4 * sq_rows * sq_cols vertices."""
        for sr, sc in [(2, 3), (3, 4), (2, 5)]:
            graph = _make_square_octagon(sr, sc)
            assert graph.node_count() == 4 * sr * sc

    def test_edge_count(self):
        """Edge count matches _square_octagon_edge_count formula."""
        for sr, sc in [(2, 2), (2, 3), (3, 3), (3, 4), (2, 5)]:
            vr, vc = 2 * sr, 2 * sc
            graph = _make_square_octagon(sr, sc)
            expected = _square_octagon_edge_count(vr, vc)
            assert graph.edge_count() == expected, (
                f"sq={sr}x{sc}: got {graph.edge_count()} edges, expected {expected}"
            )

    def test_is_bipartite(self):
        """Square-octagon lattice is bipartite."""
        graph = _make_square_octagon(2, 3)
        fp = compute_structural_fingerprint(graph)
        assert fp.is_bipartite

    def test_degrees_only_2_and_3(self):
        """All vertices have degree 2 or 3."""
        for sr, sc in [(2, 2), (2, 3), (3, 3)]:
            graph = _make_square_octagon(sr, sc)
            fp = compute_structural_fingerprint(graph)
            for deg in fp.degree_counts:
                assert deg in (2, 3), (
                    f"sq={sr}x{sc}: unexpected degree {deg}"
                )

    def test_girth_is_4(self):
        """Square-octagon has girth 4 (contains 4-cycles from small squares)."""
        G = _make_square_octagon_nx(2, 3)
        assert nx.girth(G) == 4

    def test_non_square_octagon_petersen(self):
        """Petersen graph is not a square-octagon."""
        graph = Graph.from_networkx(nx.petersen_graph())
        fp = compute_structural_fingerprint(graph)
        assert detect_square_octagon_dims_with_bfs(graph, fp) is None

    def test_non_square_octagon_grid(self):
        """Grid is not detected as square-octagon."""
        graph = _make_grid(4, 6)
        fp = compute_structural_fingerprint(graph)
        assert detect_square_octagon_dims_with_bfs(graph, fp) is None

    def test_small_dims_skipped(self):
        """Square-octagon with sq_rows < 2 or sq_cols < 2 is skipped."""
        # sq_rows=1, sq_cols=3 => vr=2, vc=6
        graph = _make_square_octagon(1, 3)
        fp = compute_structural_fingerprint(graph)
        assert detect_square_octagon_dims_with_bfs(graph, fp) is None

    def test_degenerate_single_vertex(self):
        """Single vertex is not detected as square-octagon."""
        G = nx.Graph()
        G.add_node(0)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_square_octagon_dims_with_bfs(graph, fp) is None

    def test_disconnected_rejected(self):
        """Two disjoint square-octagons are not detected."""
        G1 = _make_square_octagon_nx(2, 2)
        G2 = nx.convert_node_labels_to_integers(
            _make_square_octagon_nx(2, 2), first_label=16
        )
        G = nx.compose(G1, G2)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_square_octagon_dims_with_bfs(graph, fp) is None

    def test_edge_count_formula(self):
        """_square_octagon_edge_count matches direct graph edge count."""
        for sr in range(2, 5):
            for sc in range(2, 6):
                vr, vc = 2 * sr, 2 * sc
                expected = _square_octagon_edge_count(vr, vc)
                G = _make_square_octagon_nx(sr, sc)
                assert G.number_of_edges() == expected, (
                    f"sq={sr}x{sc}: formula={expected}, graph={G.number_of_edges()}"
                )


# =============================================================================
# I. ELONGATED TRIANGULAR (3.3.3.4.4) LATTICE DETECTION
# =============================================================================


class TestElongatedTriangularDetection:
    """Test BFS-based elongated triangular (3.3.3.4.4) lattice detection."""

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 4), (3, 5), (3, 6),
        (4, 4), (4, 5), (4, 6),
        (5, 5), (5, 7),
        # Tall orientations (m > n): different graph from (n, m) since
        # diagonal pattern depends on row parity.
        (4, 3), (5, 3), (5, 4), (6, 4), (7, 5),
    ])
    def test_detect_dims(self, m, n):
        """detect_elongated_triangular_dims_with_bfs correctly identifies the lattice."""
        graph = _make_elongated_triangular(m, n)
        fp = compute_structural_fingerprint(graph)
        result = detect_elongated_triangular_dims_with_bfs(graph, fp)
        assert result is not None, f"Failed to detect elongated triangular {m}x{n}"
        width, length = result
        assert width == m, f"Expected width={m}, got {width}"
        assert length == n, f"Expected length={n}, got {length}"

    def test_vertex_count(self):
        """Elongated triangular m x n has m*n vertices."""
        for m, n in [(3, 4), (4, 5), (5, 6)]:
            graph = _make_elongated_triangular(m, n)
            assert graph.node_count() == m * n

    def test_edge_count(self):
        """Elongated triangular m x n has 2mn - m - n + (m//2)*(n-1) edges."""
        for m, n in [(3, 3), (3, 4), (4, 5), (5, 6), (6, 4)]:
            graph = _make_elongated_triangular(m, n)
            expected = _elongated_triangular_edge_count(m, n)
            assert graph.edge_count() == expected, (
                f"ET {m}x{n}: got {graph.edge_count()} edges, expected {expected}"
            )

    def test_not_bipartite(self):
        """Elongated triangular lattice is not bipartite (has triangles)."""
        graph = _make_elongated_triangular(3, 4)
        fp = compute_structural_fingerprint(graph)
        assert not fp.is_bipartite

    def test_girth_is_3(self):
        """Elongated triangular has girth 3 (contains triangles)."""
        G = _make_elongated_triangular_nx(3, 4)
        assert nx.girth(G) == 3

    def test_degrees_in_range(self):
        """All vertices have degree 2, 3, 4, or 5."""
        for m, n in [(3, 4), (4, 5), (5, 6)]:
            graph = _make_elongated_triangular(m, n)
            fp = compute_structural_fingerprint(graph)
            for deg in fp.degree_counts:
                assert deg in (2, 3, 4, 5), (
                    f"ET {m}x{n}: unexpected degree {deg}"
                )

    def test_max_degree_5(self):
        """Interior vertices of large enough lattice reach degree 5."""
        graph = _make_elongated_triangular(5, 5)
        fp = compute_structural_fingerprint(graph)
        assert fp.max_degree == 5

    def test_non_elongated_triangular_petersen(self):
        """Petersen graph is not an elongated triangular."""
        graph = Graph.from_networkx(nx.petersen_graph())
        fp = compute_structural_fingerprint(graph)
        assert detect_elongated_triangular_dims_with_bfs(graph, fp) is None

    def test_non_elongated_triangular_complete(self):
        """Complete graph K_6 is not an elongated triangular."""
        graph = Graph.from_networkx(nx.complete_graph(6))
        fp = compute_structural_fingerprint(graph)
        assert detect_elongated_triangular_dims_with_bfs(graph, fp) is None

    def test_small_dims_skipped(self):
        """Elongated triangular with width < 3 or length < 3 is skipped."""
        for m, n in [(2, 5), (2, 3), (1, 5)]:
            graph = _make_elongated_triangular(m, n)
            fp = compute_structural_fingerprint(graph)
            result = detect_elongated_triangular_dims_with_bfs(graph, fp)
            assert result is None, f"ET {m}x{n} should be skipped (dim < 3)"

    def test_degenerate_single_vertex(self):
        """Single vertex is not detected as elongated triangular."""
        G = nx.Graph()
        G.add_node(0)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_elongated_triangular_dims_with_bfs(graph, fp) is None

    def test_degenerate_single_edge(self):
        """Single edge (K_2) is not detected as elongated triangular."""
        graph = Graph.from_networkx(nx.path_graph(2))
        fp = compute_structural_fingerprint(graph)
        assert detect_elongated_triangular_dims_with_bfs(graph, fp) is None

    def test_disconnected_rejected(self):
        """Two disjoint elongated triangulars are not detected."""
        G1 = _make_elongated_triangular_nx(3, 4)
        G2 = nx.convert_node_labels_to_integers(
            _make_elongated_triangular_nx(3, 4), first_label=12
        )
        G = nx.compose(G1, G2)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_elongated_triangular_dims_with_bfs(graph, fp) is None

    def test_edge_count_formula(self):
        """_elongated_triangular_edge_count matches direct graph edge count."""
        for m in range(3, 8):
            for n in range(3, 8):
                expected = _elongated_triangular_edge_count(m, n)
                G = _make_elongated_triangular_nx(m, n)
                assert G.number_of_edges() == expected, (
                    f"ET {m}x{n}: formula={expected}, graph={G.number_of_edges()}"
                )


# =============================================================================
# K. UNIT CELL EDGE PATTERN TESTS
# =============================================================================


class TestUnitCellEdgePatterns:
    """Test unit cell edge patterns for all lattice types."""

    # -- Grid --

    @pytest.mark.parametrize("width", range(1, 7))
    def test_grid_unit_cell_count(self, width):
        """Grid unit cell has 2m - 1 edges (m horizontal + m-1 vertical)."""
        edges = _grid_unit_cell_edges(width)
        expected = 2 * width - 1
        assert len(edges) == expected, (
            f"Width {width}: expected {expected} grid edges, got {len(edges)}"
        )

    @pytest.mark.parametrize("width", range(1, 7))
    def test_grid_unit_cell_indices_in_range(self, width):
        """All unit cell edge indices stay within [0, width-1]."""
        edges = _grid_unit_cell_edges(width)
        for a, b, _is_cross in edges:
            assert 0 <= a < width, f"Index {a} out of range [0, {width-1}]"
            assert 0 <= b < width, f"Index {b} out of range [0, {width-1}]"

    # -- Triangular --

    @pytest.mark.parametrize("width", range(3, 7))
    def test_triangular_unit_cell_count(self, width):
        """Triangular unit cell has 3m - 2 edges (m horiz + m-1 vert + m-1 diag)."""
        edges = _triangular_unit_cell_edges(width)
        expected = 3 * width - 2
        assert len(edges) == expected, (
            f"Width {width}: expected {expected} tri edges, got {len(edges)}"
        )

    @pytest.mark.parametrize("width", range(3, 7))
    def test_triangular_unit_cell_structure(self, width):
        """Triangular unit cell has correct edge types."""
        edges = _triangular_unit_cell_edges(width)
        cross = [(s, d, x) for s, d, x in edges if x]
        within = [(s, d, x) for s, d, x in edges if not x]
        # Cross-column: m horizontal + m-1 diagonal = 2m - 1
        assert len(cross) == 2 * width - 1
        # Within-column: m-1 vertical
        assert len(within) == width - 1

    @pytest.mark.parametrize("width", range(3, 7))
    def test_triangular_horizontal_edges(self, width):
        """Triangular horizontal edges are (i, i, True) for i in 0..width-1."""
        edges = _triangular_unit_cell_edges(width)
        horizontal = [(s, d, x) for s, d, x in edges if x and s == d]
        assert len(horizontal) == width
        for i, (s, d, _) in enumerate(sorted(horizontal)):
            assert s == i and d == i

    @pytest.mark.parametrize("width", range(3, 7))
    def test_triangular_diagonal_edges(self, width):
        """Triangular diagonal edges are (i, i+1, True) for i in 0..width-2."""
        edges = _triangular_unit_cell_edges(width)
        diag = [(s, d, x) for s, d, x in edges if x and s != d]
        assert len(diag) == width - 1
        for i, (s, d, _) in enumerate(sorted(diag)):
            assert s == i and d == i + 1

    @pytest.mark.parametrize("width", range(3, 7))
    def test_triangular_unit_cell_indices_in_range(self, width):
        """Triangular unit cell edge indices stay within [0, width-1]."""
        edges = _triangular_unit_cell_edges(width)
        for s, d, _ in edges:
            assert 0 <= s < width, f"Index {s} out of range [0, {width-1}]"
            assert 0 <= d < width, f"Index {d} out of range [0, {width-1}]"

    # -- Honeycomb --

    @pytest.mark.parametrize("vertex_rows", range(4, 10, 2))
    def test_honeycomb_even_unit_cell_count(self, vertex_rows):
        """Even-column honeycomb unit cell has vertex_rows + vertex_rows//2 edges."""
        edges = _honeycomb_unit_cell_edges(vertex_rows)
        half = vertex_rows // 2
        expected = vertex_rows + half
        assert len(edges) == expected, (
            f"vr={vertex_rows}: expected {expected} honeycomb even edges, got {len(edges)}"
        )

    @pytest.mark.parametrize("vertex_rows", range(4, 10, 2))
    def test_honeycomb_odd_unit_cell_count(self, vertex_rows):
        """Odd-column honeycomb unit cell has vertex_rows + vertex_rows//2 - 1 edges."""
        edges = _honeycomb_unit_cell_edges_odd(vertex_rows)
        half = vertex_rows // 2
        expected = vertex_rows + half - 1
        assert len(edges) == expected, (
            f"vr={vertex_rows}: expected {expected} honeycomb odd edges, got {len(edges)}"
        )

    @pytest.mark.parametrize("vertex_rows", range(4, 10, 2))
    def test_honeycomb_even_vertical_pairs(self, vertex_rows):
        """Even-column vertical edges pair (0,1),(2,3),(4,5),..."""
        edges = _honeycomb_unit_cell_edges(vertex_rows)
        vert = [(s, d) for s, d, x in edges if not x]
        half = vertex_rows // 2
        assert len(vert) == half
        for k, (s, d) in enumerate(sorted(vert)):
            assert s == 2 * k and d == 2 * k + 1

    @pytest.mark.parametrize("vertex_rows", range(4, 10, 2))
    def test_honeycomb_odd_vertical_pairs(self, vertex_rows):
        """Odd-column vertical edges pair (1,2),(3,4),(5,6),..."""
        edges = _honeycomb_unit_cell_edges_odd(vertex_rows)
        vert = [(s, d) for s, d, x in edges if not x]
        half = vertex_rows // 2
        assert len(vert) == half - 1
        for k, (s, d) in enumerate(sorted(vert)):
            assert s == 2 * k + 1 and d == 2 * k + 2

    # -- Elongated triangular --

    @pytest.mark.parametrize("width", range(3, 7))
    def test_elongated_triangular_unit_cell_count(self, width):
        """ET unit cell has width + width//2 + width-1 edges."""
        edges = _elongated_triangular_unit_cell_edges(width)
        expected = width + width // 2 + width - 1
        assert len(edges) == expected, (
            f"Width {width}: expected {expected} ET edges, got {len(edges)}"
        )

    @pytest.mark.parametrize("width", range(3, 7))
    def test_elongated_triangular_unit_cell_structure(self, width):
        """ET unit cell has correct edge types."""
        edges = _elongated_triangular_unit_cell_edges(width)
        cross = [(s, d, x) for s, d, x in edges if x]
        within = [(s, d, x) for s, d, x in edges if not x]
        # Cross-column: width horizontal + width//2 diagonal
        assert len(cross) == width + width // 2
        # Within-column: width-1 vertical
        assert len(within) == width - 1

    @pytest.mark.parametrize("width", range(3, 7))
    def test_elongated_triangular_horizontal_edges(self, width):
        """ET horizontal edges are (i, i, True) for all i."""
        edges = _elongated_triangular_unit_cell_edges(width)
        horizontal = [(s, d, x) for s, d, x in edges if x and s == d]
        assert len(horizontal) == width
        for i, (s, d, _) in enumerate(sorted(horizontal)):
            assert s == i and d == i

    @pytest.mark.parametrize("width", range(3, 7))
    def test_elongated_triangular_diagonal_edges(self, width):
        """ET diagonal edges are (i, i+1, True) for even i only."""
        edges = _elongated_triangular_unit_cell_edges(width)
        diag = [(s, d, x) for s, d, x in edges if x and s != d]
        expected_count = width // 2
        assert len(diag) == expected_count
        for k, (s, d, _) in enumerate(sorted(diag)):
            assert s == 2 * k and d == 2 * k + 1

    @pytest.mark.parametrize("width", range(3, 7))
    def test_elongated_triangular_unit_cell_indices_in_range(self, width):
        """ET unit cell edge indices stay within [0, width-1]."""
        edges = _elongated_triangular_unit_cell_edges(width)
        for s, d, _ in edges:
            assert 0 <= s < width, f"Index {s} out of range [0, {width-1}]"
            assert 0 <= d < width, f"Index {d} out of range [0, {width-1}]"



# =============================================================================
# K. CROSS-REJECTION TESTS
# =============================================================================


_CROSS_REJECTION_GRAPHS = {
    "grid": lambda: _make_grid(4, 5),
    "triangular": lambda: _make_triangular(4, 5),
    "honeycomb": lambda: _make_honeycomb(4, 3),
    "square_octagon": lambda: _make_square_octagon(2, 3),
    "elongated_triangular": lambda: _make_elongated_triangular(4, 5),
}

_DETECTORS = {
    "grid": detect_grid_dims_with_bfs,
    "triangular": detect_triangular_dims_with_bfs,
    "honeycomb": detect_honeycomb_dims_with_bfs,
    "square_octagon": detect_square_octagon_dims_with_bfs,
    "elongated_triangular": detect_elongated_triangular_dims_with_bfs,
}

_CROSS_REJECTION_CASES = [
    (detector_name, graph_name)
    for detector_name in _DETECTORS
    for graph_name in _CROSS_REJECTION_GRAPHS
    if detector_name != graph_name
]


class TestCrossRejection:
    """Each lattice detector correctly rejects graphs of other lattice types."""

    @pytest.mark.parametrize("detector_name,graph_name", _CROSS_REJECTION_CASES,
                             ids=[f"{d}_rejects_{g}" for d, g in _CROSS_REJECTION_CASES])
    def test_cross_rejection(self, detector_name, graph_name):
        graph = _CROSS_REJECTION_GRAPHS[graph_name]()
        fp = compute_structural_fingerprint(graph)
        result = _DETECTORS[detector_name](graph, fp)
        assert result is None, (
            f"{detector_name} detector should reject {graph_name} graph, got {result}"
        )


# =============================================================================
# L. DETECT_PERIODIC_STRIP INTEGRATION (ALL LATTICE TYPES)
# =============================================================================


class TestPeriodicStripAllLattices:
    """Test detect_periodic_strip dispatches correctly for all lattice types."""

    @pytest.mark.parametrize("m,n", [(3, 4), (4, 5)])
    def test_periodic_strip_triangular(self, m, n):
        """detect_periodic_strip returns correct results for triangular lattice."""
        graph = _make_triangular(m, n)
        fp = compute_structural_fingerprint(graph)
        result = detect_periodic_strip(graph, fp)
        assert result is not None, f"Periodic strip failed for triangular {m}x{n}"
        width, length, patterns, num_verts, first_col = result
        assert width == min(m, n)
        assert length == max(m, n)
        assert len(patterns) == 1
        # Triangular unit cell has 3*width - 2 edges
        assert len(patterns[0]) == 3 * width - 2

    @pytest.mark.parametrize("hex_rows,hex_cols", [(2, 2), (3, 4)])
    def test_periodic_strip_honeycomb(self, hex_rows, hex_cols):
        """detect_periodic_strip returns correct results for honeycomb lattice."""
        vr, vc = 2 * hex_rows, hex_cols + 1
        graph = _make_honeycomb(vr, vc)
        fp = compute_structural_fingerprint(graph)
        result = detect_periodic_strip(graph, fp)
        assert result is not None, (
            f"Periodic strip failed for honeycomb hr={hex_rows}, hc={hex_cols}"
        )
        width, length, patterns, num_verts, first_col = result
        # Honeycomb periodic strip width = vertex_rows
        assert width == vr
        # Length = number of columns
        assert length == vc
        # Honeycomb has two alternating patterns (even/odd columns)
        assert len(patterns) == 2
        # Vertex count matches actual graph
        assert num_verts == vr * vc

    @pytest.mark.parametrize("sq_rows,sq_cols", [(2, 2), (2, 3), (3, 3)])
    def test_periodic_strip_square_octagon(self, sq_rows, sq_cols):
        """detect_periodic_strip returns correct results for square-octagon."""
        graph = _make_square_octagon(sq_rows, sq_cols)
        fp = compute_structural_fingerprint(graph)
        result = detect_periodic_strip(graph, fp)
        vr, vc = 2 * sq_rows, 2 * sq_cols
        assert result is not None, (
            f"Periodic strip failed for square-octagon sq={sq_rows}x{sq_cols}"
        )
        assert result.width == vr
        assert result.length == vc
        # Square-octagon has 4 alternating patterns (period-4)
        assert len(result.transition_patterns) == 4
        assert result.num_vertices == vr * vc

    @pytest.mark.parametrize("m,n", [(3, 4), (4, 5), (5, 5)])
    def test_periodic_strip_elongated_triangular(self, m, n):
        """detect_periodic_strip returns correct results for elongated triangular."""
        graph = _make_elongated_triangular(m, n)
        fp = compute_structural_fingerprint(graph)
        result = detect_periodic_strip(graph, fp)
        assert result is not None, (
            f"Periodic strip failed for elongated triangular {m}x{n}"
        )
        width, length = result.width, result.length
        assert width * length == m * n
        # Period-1: single transition pattern
        assert len(result.transition_patterns) == 1
        # Unit cell: width horizontal + width//2 diagonal + (width-1) vertical
        expected_edges = width + width // 2 + width - 1
        assert len(result.transition_patterns[0]) == expected_edges
        assert result.num_vertices == m * n



    def test_periodic_strip_max_width_enforcement(self):
        """detect_periodic_strip returns None when boundary width exceeds MAX."""
        # Build a grid with width=MAX+1 which would exceed the limit
        width = MAX_TRANSFER_MATRIX_WIDTH + 1
        graph = _make_grid(width, width + 2)
        fp = compute_structural_fingerprint(graph)
        result = detect_periodic_strip(graph, fp)
        assert result is None, (
            f"Grid with width={width} should be rejected "
            f"(MAX_TRANSFER_MATRIX_WIDTH={MAX_TRANSFER_MATRIX_WIDTH})"
        )

    def test_periodic_strip_at_max_width(self):
        """detect_periodic_strip works at exactly MAX_TRANSFER_MATRIX_WIDTH."""
        width = MAX_TRANSFER_MATRIX_WIDTH
        if width < 3:
            pytest.skip("MAX_TRANSFER_MATRIX_WIDTH too small for grid detection")
        graph = _make_grid(width, width + 2)
        fp = compute_structural_fingerprint(graph)
        result = detect_periodic_strip(graph, fp)
        assert result is not None, (
            f"Grid with width={width} should be accepted "
            f"(MAX_TRANSFER_MATRIX_WIDTH={MAX_TRANSFER_MATRIX_WIDTH})"
        )

    def test_periodic_strip_none_for_random(self):
        """Random graph is not detected as any periodic strip."""
        import random
        random.seed(42)
        G = nx.gnm_random_graph(20, 40, seed=42)
        # Make it connected
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                u = next(iter(components[i]))
                v = next(iter(components[i + 1]))
                G.add_edge(u, v)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        result = detect_periodic_strip(graph, fp)
        assert result is None


# =============================================================================
# M. EDGE PERTURBATION TESTS
# =============================================================================


class TestEdgePerturbation:
    """Test that detectors reject lattices with edges added or removed."""

    # -- Grid --

    def test_grid_with_extra_edge_rejected(self):
        """Grid with one diagonal added is not detected as a grid."""
        G = _make_grid_nx(4, 5)
        G.add_edge(0, 6)  # diagonal (0,0)-(1,1)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_grid_dims_with_bfs(graph, fp) is None

    def test_grid_with_removed_edge_rejected(self):
        """Grid with one edge removed is not detected as a grid."""
        G = _make_grid_nx(4, 5)
        edge = list(G.edges())[0]
        G.remove_edge(*edge)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_grid_dims_with_bfs(graph, fp) is None

    # -- Triangular --

    def test_triangular_with_extra_edge_rejected(self):
        """Triangular lattice with an extra edge is not recognized."""
        G = _make_triangular_nx(4, 5)
        nodes = list(G.nodes())
        for u in nodes:
            for v in nodes:
                if u < v and not G.has_edge(u, v):
                    G.add_edge(u, v)
                    graph = Graph.from_networkx(G)
                    fp = compute_structural_fingerprint(graph)
                    assert detect_triangular_dims_with_bfs(graph, fp) is None
                    return

    def test_triangular_with_removed_edge_rejected(self):
        """Triangular lattice with a removed edge is not recognized."""
        G = _make_triangular_nx(4, 5)
        edge = list(G.edges())[0]
        G.remove_edge(*edge)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_triangular_dims_with_bfs(graph, fp) is None

    # -- Honeycomb --

    def test_honeycomb_with_extra_edge_rejected(self):
        """Honeycomb with an extra edge is not recognized."""
        G = _make_honeycomb_nx(4, 3)
        nodes = list(G.nodes())
        for u in nodes:
            for v in nodes:
                if u < v and not G.has_edge(u, v):
                    G.add_edge(u, v)
                    graph = Graph.from_networkx(G)
                    fp = compute_structural_fingerprint(graph)
                    assert detect_honeycomb_dims_with_bfs(graph, fp) is None
                    return

    def test_honeycomb_with_removed_edge_rejected(self):
        """Honeycomb with a removed edge is not recognized."""
        G = _make_honeycomb_nx(4, 3)
        edge = list(G.edges())[0]
        G.remove_edge(*edge)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_honeycomb_dims_with_bfs(graph, fp) is None

    # -- Square-octagon --

    def test_square_octagon_with_extra_edge_rejected(self):
        """Square-octagon with an extra edge is not recognized."""
        G = _make_square_octagon_nx(2, 3)
        nodes = list(G.nodes())
        for u in nodes:
            for v in nodes:
                if u < v and not G.has_edge(u, v):
                    G.add_edge(u, v)
                    graph = Graph.from_networkx(G)
                    fp = compute_structural_fingerprint(graph)
                    assert detect_square_octagon_dims_with_bfs(graph, fp) is None
                    return

    def test_square_octagon_with_removed_edge_rejected(self):
        """Square-octagon with a removed edge is not recognized."""
        G = _make_square_octagon_nx(2, 3)
        edge = list(G.edges())[0]
        G.remove_edge(*edge)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_square_octagon_dims_with_bfs(graph, fp) is None

    # -- Elongated triangular --

    def test_elongated_triangular_with_extra_edge_rejected(self):
        """Elongated triangular with an extra edge is not recognized."""
        G = _make_elongated_triangular_nx(4, 5)
        nodes = list(G.nodes())
        for u in nodes:
            for v in nodes:
                if u < v and not G.has_edge(u, v):
                    G.add_edge(u, v)
                    graph = Graph.from_networkx(G)
                    fp = compute_structural_fingerprint(graph)
                    assert detect_elongated_triangular_dims_with_bfs(graph, fp) is None
                    return

    def test_elongated_triangular_with_removed_edge_rejected(self):
        """Elongated triangular with a removed edge is not recognized."""
        G = _make_elongated_triangular_nx(4, 5)
        edge = list(G.edges())[0]
        G.remove_edge(*edge)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        assert detect_elongated_triangular_dims_with_bfs(graph, fp) is None



# =============================================================================
# N. ISOMORPHISM INVARIANCE TESTS
# =============================================================================


class TestIsomorphismInvariance:
    """Test that detection works regardless of vertex labeling."""

    def _relabel_random(self, G: nx.Graph, seed: int) -> nx.Graph:
        """Relabel graph vertices with a random permutation."""
        import random
        rng = random.Random(seed)
        nodes = list(G.nodes())
        perm = list(range(len(nodes)))
        rng.shuffle(perm)
        mapping = {old: new for old, new in zip(nodes, perm)}
        return nx.relabel_nodes(G, mapping)

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 100])
    def test_grid_relabeled(self, seed):
        """Grid detection works after random vertex relabeling."""
        G = _make_grid_nx(3, 5)
        G2 = self._relabel_random(G, seed)
        graph = Graph.from_networkx(G2)
        fp = compute_structural_fingerprint(graph)
        result = detect_grid_dims_with_bfs(graph, fp)
        assert result is not None, f"Relabeled grid 3x5 (seed={seed}) not detected"
        assert result == (3, 5)

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 100])
    def test_triangular_relabeled(self, seed):
        """Triangular detection works after random vertex relabeling."""
        G = _make_triangular_nx(3, 5)
        G2 = self._relabel_random(G, seed)
        graph = Graph.from_networkx(G2)
        fp = compute_structural_fingerprint(graph)
        result = detect_triangular_dims_with_bfs(graph, fp)
        assert result is not None, f"Relabeled tri 3x5 (seed={seed}) not detected"
        assert result == (3, 5)

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 100])
    def test_honeycomb_relabeled(self, seed):
        """Honeycomb detection works after random vertex relabeling."""
        G = _make_honeycomb_nx(4, 3)  # hex_rows=2, hex_cols=2
        G2 = self._relabel_random(G, seed)
        graph = Graph.from_networkx(G2)
        fp = compute_structural_fingerprint(graph)
        result = detect_honeycomb_dims_with_bfs(graph, fp)
        assert result is not None, f"Relabeled honeycomb (seed={seed}) not detected"
        h, w, vr, vc = result
        assert h == 2 and w == 2

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 100])
    def test_square_octagon_relabeled(self, seed):
        """Square-octagon detection works after random vertex relabeling."""
        G = _make_square_octagon_nx(2, 3)
        G2 = self._relabel_random(G, seed)
        graph = Graph.from_networkx(G2)
        fp = compute_structural_fingerprint(graph)
        result = detect_square_octagon_dims_with_bfs(graph, fp)
        assert result is not None, f"Relabeled sq-oct 2x3 (seed={seed}) not detected"
        sr, sc, _, _ = result
        assert sr == 2 and sc == 3

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 100])
    def test_elongated_triangular_relabeled(self, seed):
        """Elongated triangular detection works after random vertex relabeling."""
        G = _make_elongated_triangular_nx(4, 5)
        G2 = self._relabel_random(G, seed)
        graph = Graph.from_networkx(G2)
        fp = compute_structural_fingerprint(graph)
        result = detect_elongated_triangular_dims_with_bfs(graph, fp)
        assert result is not None, f"Relabeled ET 4x5 (seed={seed}) not detected"
        width, length = result
        assert width * length == 20



# =============================================================================
# O. SPOOFING TESTS (RIGHT STATS, WRONG STRUCTURE)
# =============================================================================


class TestSpoofing:
    """Graphs with correct fingerprint stats but wrong adjacency structure."""

    def test_grid_spoof_degree_histogram(self):
        """Graph with same V, E, degree histogram as a 4x5 grid but wrong structure.

        Build a graph with the same degree distribution as a 4x5 grid
        (corners=degree 2, border=degree 3, interior=degree 4) but wired
        differently so BFS cannot lay out a grid.
        """
        # 4x5 grid: 4 deg-2, 10 deg-3, 6 deg-4, V=20, E=31
        G = nx.Graph()
        G.add_nodes_from(range(20))
        # Wire a random 4-regular-ish graph that hits the same histogram
        # but is NOT a grid (use a known non-grid structure)
        # Strategy: take the grid, then swap two edges to break grid structure
        G = _make_grid_nx(4, 5)
        # Swap edges: remove (0,1) and (2,3), add (0,3) and (1,2)
        # This preserves degrees of 0,1,2,3 but breaks the BFS grid layout
        if G.has_edge(0, 1) and G.has_edge(2, 3) and not G.has_edge(0, 3):
            G.remove_edge(0, 1)
            G.remove_edge(2, 3)
            G.add_edge(0, 3)
            G.add_edge(1, 2)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        result = detect_grid_dims_with_bfs(graph, fp)
        assert result is None, "Edge-swapped grid should not be detected as a grid"

    def test_triangular_spoof_wrong_adjacency(self):
        """Graph with same degree histogram as triangular 3x4 but wrong structure."""
        G = _make_triangular_nx(3, 4)
        # Swap two edges to break triangular structure
        edges = list(G.edges())
        # Find two non-adjacent edges to swap
        e1, e2 = edges[0], edges[-1]
        u1, v1 = e1
        u2, v2 = e2
        if not G.has_edge(u1, v2) and not G.has_edge(u2, v1) and u1 != v2 and u2 != v1:
            G.remove_edge(u1, v1)
            G.remove_edge(u2, v2)
            G.add_edge(u1, v2)
            G.add_edge(u2, v1)
        graph = Graph.from_networkx(G)
        fp = compute_structural_fingerprint(graph)
        result = detect_triangular_dims_with_bfs(graph, fp)
        assert result is None, "Edge-swapped triangular should not be detected"


# =============================================================================
# P. BUILDER VALIDATION TESTS
# =============================================================================


_BUILDER_CASES = [
    ("grid", lambda: _make_grid_nx(4, 5)),
    ("triangular", lambda: _make_triangular_nx(4, 5)),
    ("honeycomb", lambda: _make_honeycomb_nx(4, 3)),
    ("square_octagon", lambda: _make_square_octagon_nx(2, 3)),
    ("elongated_triangular", lambda: _make_elongated_triangular_nx(4, 5)),
]


class TestBuilderValidation:
    """Validate that lattice graph builders produce well-formed graphs."""

    @pytest.mark.parametrize("name,builder", _BUILDER_CASES, ids=[c[0] for c in _BUILDER_CASES])
    def test_builder_connected(self, name, builder):
        """Each lattice builder produces a connected graph."""
        G = builder()
        assert nx.is_connected(G), f"{name} builder produced a disconnected graph"

    @pytest.mark.parametrize("name,builder", _BUILDER_CASES, ids=[c[0] for c in _BUILDER_CASES])
    def test_builder_simple(self, name, builder):
        """Each lattice builder produces a simple graph (no self-loops)."""
        G = builder()
        assert nx.number_of_selfloops(G) == 0, f"{name} builder has self-loops"

    @pytest.mark.parametrize("m,n", [(3, 5), (4, 6)])
    def test_grid_builder_matches_nx(self, m, n):
        """Grid builder is isomorphic to nx.grid_2d_graph."""
        ours = _make_grid_nx(m, n)
        theirs = nx.convert_node_labels_to_integers(nx.grid_2d_graph(m, n))
        assert nx.is_isomorphic(ours, theirs), (
            f"Grid {m}x{n}: builder not isomorphic to nx.grid_2d_graph"
        )

    @pytest.mark.parametrize("m,n", [(3, 4), (4, 5)])
    def test_triangular_builder_vertex_edge_counts(self, m, n):
        """Triangular builder has correct vertex and edge counts."""
        G = _make_triangular_nx(m, n)
        assert G.number_of_nodes() == m * n
        assert G.number_of_edges() == _triangular_edge_count(m, n)

    @pytest.mark.parametrize("m,n", [(3, 4), (4, 5), (5, 6)])
    def test_elongated_triangular_builder_vertex_edge_counts(self, m, n):
        """Elongated triangular builder has correct vertex and edge counts."""
        G = _make_elongated_triangular_nx(m, n)
        assert G.number_of_nodes() == m * n
        assert G.number_of_edges() == _elongated_triangular_edge_count(m, n)


# =============================================================================
# Q. TRANSFER MATRIX KIRCHHOFF VERIFICATION (ALL LATTICE FAMILIES)
# =============================================================================


class TestTransferMatrixKirchhoffAllFamilies:
    """Verify transfer-matrix Tutte polynomials against Kirchhoff spanning tree counts.

    For each lattice family, compute T(G; x, y) via the transfer-matrix
    pipeline and check T(1,1) == Kirchhoff matrix-tree theorem count.
    """

    # -- Triangular lattice --

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 4), (3, 5), (3, 6),
        (4, 4), (4, 5), (4, 6),
    ])
    def test_triangular_spanning_trees(self, m, n):
        """Triangular lattice: T(1,1) matches Kirchhoff count."""
        width, length = min(m, n), max(m, n)
        edges = _triangular_unit_cell_edges(width)
        num_vertices = m * n
        poly = _compute_pipeline(width, length, edges, num_vertices=num_vertices)
        graph = _make_triangular(m, n)
        kirchhoff = count_spanning_trees_kirchhoff(graph)
        t11 = _exact_num_spanning_trees(poly)
        assert t11 == kirchhoff, (
            f"Triangular {m}x{n}: T(1,1)={t11} != Kirchhoff={kirchhoff}"
        )

    # -- T(2,2) = 2^|E| property (triangular) --

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 4), (3, 5),
        (4, 4), (4, 5),
    ])
    def test_triangular_two_two_property(self, m, n):
        """Triangular lattice: T(2,2) = 2^|E|."""
        width, length = min(m, n), max(m, n)
        edges = _triangular_unit_cell_edges(width)
        num_vertices = m * n
        poly = _compute_pipeline(width, length, edges, num_vertices=num_vertices)
        E = _triangular_edge_count(m, n)
        t22 = poly.evaluate(2, 2)
        assert t22 == 2 ** E, (
            f"Triangular {m}x{n}: T(2,2)={t22} != 2^{E}={2**E}"
        )

    # -- Honeycomb lattice --

    @pytest.mark.parametrize("hex_rows,hex_cols", [
        (2, 2), (2, 4),
        (3, 2), (3, 4),
    ])
    def test_honeycomb_spanning_trees(self, hex_rows, hex_cols):
        """Honeycomb lattice: T(1,1) matches Kirchhoff count."""
        vr, vc = 2 * hex_rows, hex_cols + 1
        graph = _make_honeycomb(vr, vc)
        # Exercises the alternating (two-pattern) honeycomb matrices.
        tm_poly = _tm_poly_uncapped(graph)
        assert tm_poly is not None
        kirchhoff = count_spanning_trees_kirchhoff(graph)
        t11 = _exact_num_spanning_trees(tm_poly)
        assert t11 == kirchhoff, (
            f"Honeycomb (hr={hex_rows}, hc={hex_cols}): "
            f"T(1,1)={t11} != Kirchhoff={kirchhoff}"
        )

    @pytest.mark.parametrize("hex_rows,hex_cols", [
        (2, 2), (2, 4),
        (3, 2),
    ])
    def test_honeycomb_two_two_property(self, hex_rows, hex_cols):
        """Honeycomb lattice: T(2,2) = 2^|E|."""
        vr, vc = 2 * hex_rows, hex_cols + 1
        graph = _make_honeycomb(vr, vc)
        tm_poly = _tm_poly_uncapped(graph)
        assert tm_poly is not None
        E = _honeycomb_edge_count(vr, vc)
        t22 = tm_poly.evaluate(2, 2)
        assert t22 == 2 ** E, (
            f"Honeycomb (hr={hex_rows}, hc={hex_cols}): "
            f"T(2,2)={t22} != 2^{E}={2**E}"
        )

    # -- Square-octagon lattice --

    @pytest.mark.parametrize("sq_rows,sq_cols", [
        (2, 2), (2, 3), (2, 4), (2, 5),
        (3, 2), (3, 3), (3, 4),
    ])
    def test_square_octagon_spanning_trees(self, sq_rows, sq_cols):
        """Square-octagon lattice: T(1,1) matches Kirchhoff count."""
        graph = _make_square_octagon(sq_rows, sq_cols)
        tm_poly = _tm_poly_uncapped(graph)
        assert tm_poly is not None, (
            f"Square-octagon sq={sq_rows}x{sq_cols} not detected"
        )
        kirchhoff = count_spanning_trees_kirchhoff(graph)
        t11 = _exact_num_spanning_trees(tm_poly)
        assert t11 == kirchhoff, (
            f"Square-octagon sq={sq_rows}x{sq_cols}: "
            f"T(1,1)={t11} != Kirchhoff={kirchhoff}"
        )

    @pytest.mark.parametrize("sq_rows,sq_cols", [
        (2, 2), (2, 3), (2, 4),
        (3, 2), (3, 3),
    ])
    def test_square_octagon_two_two_property(self, sq_rows, sq_cols):
        """Square-octagon lattice: T(2,2) = 2^|E|."""
        graph = _make_square_octagon(sq_rows, sq_cols)
        tm_poly = _tm_poly_uncapped(graph)
        assert tm_poly is not None
        vr, vc = 2 * sq_rows, 2 * sq_cols
        E = _square_octagon_edge_count(vr, vc)
        t22 = tm_poly.evaluate(2, 2)
        assert t22 == 2 ** E, (
            f"Square-octagon sq={sq_rows}x{sq_cols}: "
            f"T(2,2)={t22} != 2^{E}={2**E}"
        )

    # -- Elongated triangular lattice --

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 4), (3, 5), (3, 6),
        (4, 4), (4, 5), (4, 6),
        (5, 5),
    ])
    def test_elongated_triangular_spanning_trees(self, m, n):
        """Elongated triangular lattice: T(1,1) matches Kirchhoff count."""
        graph = _make_elongated_triangular(m, n)
        tm_poly = _tm_poly_uncapped(graph)
        assert tm_poly is not None, (
            f"Elongated triangular {m}x{n} not detected"
        )
        kirchhoff = count_spanning_trees_kirchhoff(graph)
        t11 = _exact_num_spanning_trees(tm_poly)
        assert t11 == kirchhoff, (
            f"Elongated triangular {m}x{n}: "
            f"T(1,1)={t11} != Kirchhoff={kirchhoff}"
        )

    @pytest.mark.parametrize("m,n", [
        (3, 3), (3, 4), (3, 5),
        (4, 4), (4, 5),
        (5, 5),
    ])
    def test_elongated_triangular_two_two_property(self, m, n):
        """Elongated triangular lattice: T(2,2) = 2^|E|."""
        graph = _make_elongated_triangular(m, n)
        tm_poly = _tm_poly_uncapped(graph)
        assert tm_poly is not None
        E = _elongated_triangular_edge_count(m, n)
        t22 = tm_poly.evaluate(2, 2)
        assert t22 == 2 ** E, (
            f"Elongated triangular {m}x{n}: "
            f"T(2,2)={t22} != 2^{E}={2**E}"
        )



# =============================================================================
# R. INTERACTIVE LATTICE COMPARISON (STANDALONE DIAGNOSTIC)
# =============================================================================


_LATTICE_BUILDERS = {
    "grid": lambda r, c: (_make_grid(r, c), f"Grid P_{r} x P_{c}"),
    "triangular": lambda r, c: (
        Graph.from_networkx(_make_triangular_nx(r, c)),
        f"Triangular {r} x {c}",
    ),
    "honeycomb": lambda r, c: (
        _make_honeycomb(r, c),
        f"Honeycomb vr={r} vc={c}",
    ),
    "square_octagon": lambda r, c: (
        _make_square_octagon(r, c),
        f"Square-octagon sq={r} x {c}",
    ),
    "elongated_triangular": lambda r, c: (
        _make_elongated_triangular(r, c),
        f"Elongated triangular {r} x {c}",
    ),
}

@pytest.mark.slow
def test_lattice_interactive():
    """Verify transfer matrix Tutte polynomial for a user-specified lattice.

    Set env vars to configure:
        LATTICE: grid (default), triangular, honeycomb, square_octagon, elongated_triangular
        ROWS:    row count (default 3; interpretation depends on lattice type)
        COLS:    col count (default 4; interpretation depends on lattice type)

    Width limits (MAX_TRANSFER_MATRIX_WIDTH=8):
        grid/triangular:         width = min(ROWS, COLS), must be <= 8
        honeycomb:               width = ROWS (vertex_rows), must be <= 8
        square_octagon:          width = 2 * ROWS (vertex_rows), must be <= 8
        elongated_triangular:    width = ROWS, must be <= 8

    Examples:
        LATTICE=grid ROWS=4 COLS=6 pytest tests/test_lattice_graphs.py::test_lattice_interactive -v -s
        LATTICE=triangular ROWS=4 COLS=5 pytest tests/test_lattice_graphs.py::test_lattice_interactive -v -s
        LATTICE=honeycomb ROWS=6 COLS=5 pytest tests/test_lattice_graphs.py::test_lattice_interactive -v -s
        LATTICE=square_octagon ROWS=3 COLS=4 pytest tests/test_lattice_graphs.py::test_lattice_interactive -v -s
        LATTICE=elongated_triangular ROWS=5 COLS=10 pytest tests/test_lattice_graphs.py::test_lattice_interactive -v -s
    """
    lattice = os.environ.get("LATTICE", "grid")
    rows = int(os.environ.get("ROWS", 3))
    cols = int(os.environ.get("COLS", 4))

    if lattice not in _LATTICE_BUILDERS:
        pytest.fail(
            f"Unknown lattice type '{lattice}'. "
            f"Supported: {', '.join(_LATTICE_BUILDERS.keys())}"
        )

    graph, label = _LATTICE_BUILDERS[lattice](rows, cols)
    num_vertices = len(graph.nodes)
    num_edges = graph.edge_count()

    print(f"\n{'=' * 64}")
    print(f"  {label}  ({num_vertices} vertices, {num_edges} edges)")
    print(f"{'=' * 64}")

    _clear_all_caches()
    t0 = time.perf_counter()
    tm_poly = compute_tutte_via_transfer_matrix(graph)
    tm_time = time.perf_counter() - t0
    if tm_poly is None:
        pytest.skip(
            f"{label}: transfer matrix boundary width exceeds "
            f"MAX_TRANSFER_MATRIX_WIDTH={MAX_TRANSFER_MATRIX_WIDTH}. "
            f"Reduce ROWS to lower the boundary width."
        )

    kirchhoff = count_spanning_trees_kirchhoff(graph)
    tm_t11 = _exact_num_spanning_trees(tm_poly)
    tm_kirchhoff_ok = (tm_t11 == kirchhoff)
    tm_t22 = tm_poly.evaluate(2, 2)
    t22_ok = (tm_t22 == 2 ** num_edges)

    c_ext = "loaded" if _c_ext._lib is not None else "unavailable"

    print(f"\n[Transfer Matrix]")
    print(f"  Lattice:         {lattice}")
    print(f"  Time:            {tm_time:.4f}s")
    print(f"  C extension:     {c_ext}")
    print(f"  T(1,1):          {tm_t11}")
    print(f"  Kirchhoff OK:    {tm_kirchhoff_ok}")
    print(f"  T(2,2)=2^|E| OK: {t22_ok}")
    print(f"  num_terms:       {tm_poly.num_terms()}")

    print(f"\n[Kirchhoff Ground Truth]")
    print(f"  Spanning trees:  {kirchhoff}")
    print(f"  2^|E|:           {2 ** num_edges}")

    print(f"\n{'─' * 64}")
    print(f"  Kirchhoff OK:    {tm_kirchhoff_ok}")
    print(f"  T(2,2) OK:       {t22_ok}")
    print(f"  Time:            {tm_time:.4f}s")
    print(f"{'=' * 64}")

    assert tm_kirchhoff_ok, (
        f"{label}: T(1,1)={tm_t11} != Kirchhoff={kirchhoff}"
    )
    assert t22_ok, (
        f"{label}: T(2,2)={tm_t22} != 2^{num_edges}={2 ** num_edges}"
    )

