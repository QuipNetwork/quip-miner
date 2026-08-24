"""Direct matrix-vector multiplication sweep for the transfer-matrix pipeline.

Given one or more transfer matrices (c_m x c_m) and a width/length, computes the final FK
state vector by repeated matrix-vector multiplication across columns.

The initial state vector encodes the first column of the lattice strip with
its internal within-column edges already processed. Each subsequent
multiplication by a transfer matrix adds one column transition (cross-column
edges from old column to new column + within-column edges in the new column),
yielding the final state vector after all columns are processed.

For lattices with period > 1 (e.g. honeycomb), multiple transfer matrices
are cycled through successive transitions.

Complexity:
    - Initial vector construction: O(c_m * 2^|first_col_edges|).
    - Per column step: O(c_m^2 * P) where P is polynomial multiplication cost.
    - Total: O((length - 1) * c_m^2 * P).

"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .core import (
    Polynomial,
    partition_index_map,
    canonicalize_partition,
    enumerate_noncrossing_partitions,
)


# =============================================================================
# INITIAL VECTOR
# =============================================================================


def build_initial_vector(
    width: int,
    first_col_edges: Optional[List[Tuple[int, int]]] = None,
) -> List[Polynomial]:
    """Create the initial FK state vector for the first column of a lattice strip.

    The initial vector represents the boundary state after processing the
    first column's internal within-column edges. It starts with the
    all-singletons partition (no edges processed) and enumerates all
    2^|first_col_edges| subsets, accumulating weights into the appropriate
    partition states.

    Each selected edge contributes a factor of (y-1), consistent with the
    FK formulation used by the transfer matrix.

    Args:
        width: Number of boundary vertices (m). Must be >= 1.
        first_col_edges: Within-column edges for the first column, as
            (row_a, row_b) pairs. If None, defaults to grid-style
            consecutive pairs [(0,1), (1,2), ..., (m-2, m-1)].

    Returns:
        State vector of length c_m, where each entry is a Polynomial
        representing the accumulated FK weight for that partition state.

    Raises:
        ValueError: If width < 1.

    Time complexity: O(c_m * 2^|first_col_edges|).
    """
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")

    partition_to_idx = partition_index_map(width)
    num_states = len(enumerate_noncrossing_partitions(width))

    vector: List[Polynomial] = [Polynomial.zero() for _ in range(num_states)]

    # Default to grid-style consecutive vertical edges if not specified
    if first_col_edges is None:
        first_col_edges = [(i, i + 1) for i in range(width - 1)]

    num_first_col_edges = len(first_col_edges)

    # Precompute b^k for edge subset sizes 0..num_first_col_edges.
    # b = (y-1) in (a,b) basis, so b^k is the monomial {(0, k): 1}.
    b_powers: List[Polynomial] = [
        Polynomial.from_coefficients({(0, k): 1}) for k in range(num_first_col_edges + 1)
    ]

    # Enumerate all 2^|first_col_edges| subsets of first column edges.
    # For each subset, merge the connected boundary vertices and record
    # the resulting partition state with weight (y-1)^|selected|.
    for subset_mask in range(1 << num_first_col_edges):
        labels = list(range(width))

        for edge_idx in range(num_first_col_edges):
            if subset_mask & (1 << edge_idx):
                row_a, row_b = first_col_edges[edge_idx]
                label_a = labels[row_a]
                label_b = labels[row_b]
                if label_a != label_b:
                    keep, replace = min(label_a, label_b), max(label_a, label_b)
                    labels = [keep if lbl == replace else lbl for lbl in labels]

        partition = canonicalize_partition(tuple(labels))
        if partition not in partition_to_idx:
            continue

        state_idx = partition_to_idx[partition]
        weight = b_powers[subset_mask.bit_count()]
        vector[state_idx] = vector[state_idx] + weight

    return vector


# =============================================================================
# MATRIX-VECTOR MULTIPLY
# =============================================================================


def matrix_vector_multiply(
    matrix: List[List[Polynomial]],
    vector: List[Polynomial],
) -> List[Polynomial]:
    """Multiply a polynomial matrix by a polynomial vector.

    Computes result[i] = sum_j(matrix[i][j] * vector[j]) for each row i.
    Skips zero vector entries for efficiency.

    Args:
        matrix: c_m x c_m matrix of Polynomials, indexed as matrix[row][col].
        vector: c_m-element vector of Polynomials.

    Returns:
        c_m-element result vector.

    Raises:
        ValueError: If matrix dimensions don't match vector length.

    Time complexity: O(c_m^2 * P) where P is polynomial multiplication cost.
    """
    num_rows = len(matrix)
    if num_rows == 0:
        return []

    num_cols = len(matrix[0])
    if len(vector) != num_cols:
        raise ValueError(
            f"Matrix has {num_cols} columns but vector has {len(vector)} entries"
        )

    result: List[Polynomial] = [Polynomial.zero() for _ in range(num_rows)]

    for row_idx in range(num_rows):
        accumulator = Polynomial.zero()
        for col_idx in range(num_cols):
            if vector[col_idx].is_zero():
                continue
            if matrix[row_idx][col_idx].is_zero():
                continue
            term = matrix[row_idx][col_idx] * vector[col_idx]
            accumulator = accumulator + term
        result[row_idx] = accumulator

    return result


# =============================================================================
# DIRECT MULTIPLY SWEEP
# =============================================================================


def direct_multiply(
    transfer_matrix: List[List[Polynomial]],
    width: int,
    length: int,
    first_col_edges: Optional[List[Tuple[int, int]]] = None,
    additional_matrices: Optional[List[List[List[Polynomial]]]] = None,
) -> List[Polynomial]:
    """Compute the final FK state vector by repeated matrix-vector multiplication.

    Builds the initial state vector (encoding the first column's within-column
    edges), then multiplies by transfer matrices (length - 1) times to
    process all column transitions.

    For a lattice strip with width m and length n:
    - The initial vector encodes column 0 (with its within-column edges).
    - Each of the (length - 1) multiplications adds one column transition.
    - After all multiplications, all edges of the strip have been processed.

    For multi-period lattices (e.g. honeycomb), additional_matrices provides
    extra transfer matrices that are cycled through successive transitions.
    Transition i uses matrix index (i % num_patterns), where the full list
    is [transfer_matrix] + additional_matrices.

    Args:
        transfer_matrix: c_m x c_m transfer matrix from build_transfer_matrix().
        width: Number of rows in the strip (m). Must be >= 1.
        length: Number of columns in the strip (n). Must be >= 1.
        first_col_edges: Within-column edges for the first column, as
            (row_a, row_b) pairs. If None, defaults to grid-style
            consecutive pairs.
        additional_matrices: Extra transfer matrices for multi-period
            lattices. If None, uses transfer_matrix for every step.

    Returns:
        Final state vector of length c_m after processing all columns.

    Raises:
        ValueError: If length < 1.

    Time complexity: O((length - 1) * c_m^2 * P).
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")

    vector = build_initial_vector(width, first_col_edges)

    if additional_matrices:
        all_matrices = [transfer_matrix] + additional_matrices
        num_patterns = len(all_matrices)
        for step in range(length - 1):
            matrix = all_matrices[step % num_patterns]
            vector = matrix_vector_multiply(matrix, vector)
    else:
        for _ in range(length - 1):
            vector = matrix_vector_multiply(transfer_matrix, vector)

    return vector
