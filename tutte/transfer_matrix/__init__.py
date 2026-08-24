"""Transfer Matrix module for computing Tutte polynomials of lattice strip graphs.

Detects periodic lattice strips (grid, triangular, honeycomb, square-octagon,
elongated triangular) and computes
their Tutte polynomials via transfer-matrix multiplication over non-crossing
partition states. Includes a C-accelerated path (cffi) with CRT fallback
for large coefficients.

Pipeline: lattice detection -> transfer matrix build -> matrix-vector sweep
-> FK extraction -> Tutte polynomial.

Public API:
    compute_tutte_via_transfer_matrix(graph) -> Optional[TuttePolynomial]
    detect_periodic_strip(graph, fp) -> Optional[StripProperties]
"""

from __future__ import annotations

from typing import Optional

from ..graph import Graph
from ..polynomial import TuttePolynomial
from .core import (
    build_transfer_matrix,
    enumerate_noncrossing_partitions,
    partition_index_map,
    CATALAN_NUMBERS,
    MAX_TRANSFER_MATRIX_WIDTH,
)
from .sweep import direct_multiply, build_initial_vector
from .extraction import extract_tutte_polynomial, _convert_ab_to_xy
from .lattice_recognition import StripProperties, detect_periodic_strip
from ..family_recognition import compute_structural_fingerprint

__all__ = [
    'compute_tutte_via_transfer_matrix',
    'compute_strip_polynomial',
    'detect_periodic_strip',
    'build_transfer_matrix',
    'enumerate_noncrossing_partitions',
    'partition_index_map',
    'CATALAN_NUMBERS',
    'MAX_TRANSFER_MATRIX_WIDTH',
    'direct_multiply',
    'build_initial_vector',
    'extract_tutte_polynomial',
]


def _transfer_wins(width: int, length: int) -> bool:
    """Predicate: should transfer_matrix run for a (width, length) strip?

    Per warm-cache benchmark (research/scripts/profile_transfer_vs_tw.py,
    2026-05-25): transfer wins for narrow strips, treewidth_dp wins for
    wider ones because Catalan(width)² grows faster than 2^width.
    """
    if width <= 4:
        return True
    if width == 5:
        return length <= 5
    return False


def compute_tutte_via_transfer_matrix(
    graph: Graph,
) -> Optional[TuttePolynomial]:
    """Compute the Tutte polynomial of a lattice strip via the transfer-matrix pipeline.

    Detects whether the graph is a periodic lattice strip (grid, triangular,
    honeycomb, or square-octagon). If so, builds transfer matrices from the unit cell edge
    pattern, runs matrix-vector multiplication across all columns, and
    extracts the Tutte polynomial from the final FK state vector.

    Tries a C-accelerated path first (full sweep in C, binomial conversion
    in Python). Falls back to pure Python if the C extension is unavailable.

    Args:
        graph: Input graph (simple, undirected).

    Returns:
        TuttePolynomial if the graph is a recognized lattice strip,
        None otherwise

    Time complexity: O(V + E) for detection, O((length - 1) * c_m^2 * P)
        for computation, where c_m = Catalan(width).
    """

    fp = compute_structural_fingerprint(graph)
    strip = detect_periodic_strip(graph, fp)

    if strip is None:
        return None

    # Cost gate: per the warm-cache profile that lived in
    # `tutte/research/scripts/profile_transfer_vs_tw.py` (2026-05-25; that
    # script is LOST, see bead riff-ggi), the transfer-matrix sweep is only
    # competitive against treewidth_dp
    # for narrow strips. Catalan(width)² grows much faster than 2^width:
    #
    #   width  Catalan² / 2^width   transfer wins?
    #     3        25 /     8        yes (4× on square)
    #     4       196 /    16        yes (5× on square; 4× on long strips)
    #     5      1764 /    32        marginal (1.2× on square, loses on length≥10)
    #     6     17424 /    64        no (3.7× slower on square, 5× on length=10)
    #     7    184041 /   128        no (14× slower on square)
    #
    # Heuristic:
    #   - width ≤ 4: always prefer transfer
    #   - width == 5: prefer transfer when length ≤ 5
    #   - width ≥ 6: defer to treewidth_dp
    if not _transfer_wins(strip.width, strip.length):
        return None
    return compute_strip_polynomial(strip)


def compute_strip_polynomial(
    strip: StripProperties,
) -> Optional[TuttePolynomial]:
    """Compute the Tutte polynomial of an ALREADY-DETECTED periodic strip.

    This is the sweep itself, with **no cost gate**. It supports any width
    up to `MAX_TRANSFER_MATRIX_WIDTH`.

    Production dispatch should call `compute_tutte_via_transfer_matrix`,
    which additionally applies the `_transfer_wins` heuristic and defers
    wide strips to treewidth_dp. This entry point exists for callers that
    want the transfer-matrix result regardless of whether it is the
    *cheapest* route — correctness tests over the full supported width
    range, and profiling of the sweep in isolation.

    Args:
        strip: a StripProperties from `detect_periodic_strip`.

    Returns:
        The Tutte polynomial, or None if the C path and the Python
        fallback both fail to produce one.
    """
    width, length, transition_patterns, num_vertices, first_col_edges = strip

    # transition_patterns is a list of edge-pattern lists from detect_periodic_strip.
    # Single-element for grid/triangular, two-element for honeycomb.
    is_single_period = len(transition_patterns) == 1

    # Try C-accelerated path.
    try:
        if is_single_period:
            from ._transfer_matrix_c import c_transfer_matrix_sweep
            ab_result = c_transfer_matrix_sweep(
                width, length, transition_patterns[0], num_vertices
            )
        else:
            from ._transfer_matrix_c import c_transfer_matrix_sweep_multi
            fc = first_col_edges if first_col_edges is not None else [
                (i, i + 1) for i in range(width - 1)
            ]
            ab_result = c_transfer_matrix_sweep_multi(
                width, length, transition_patterns, fc, num_vertices
            )
        if ab_result is not None:
            return _convert_ab_to_xy(ab_result)
    except Exception:
        pass

    # Pure Python fallback.
    if is_single_period:
        transfer_mat = build_transfer_matrix(width, transition_patterns[0])
        final_vector = direct_multiply(
            transfer_mat, width, length,
            first_col_edges=first_col_edges,
        )
    else:
        matrices = [build_transfer_matrix(width, p) for p in transition_patterns]
        final_vector = direct_multiply(
            matrices[0], width, length,
            first_col_edges=first_col_edges,
            additional_matrices=matrices[1:],
        )

    return extract_tutte_polynomial(final_vector, width, num_vertices)