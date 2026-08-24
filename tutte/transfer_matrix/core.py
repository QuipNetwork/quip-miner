"""Transfer Matrix Construction for periodic lattice strip graphs.

Builds the c_m x c_m transfer matrix over bivariate polynomial entries
for a periodic lattice strip with a given unit cell edge pattern.

The pipeline works in the (a, b) = (x-1, y-1) basis throughout, matching
the treewidth DP convention (treewidth.py). In this basis, the FK edge
weight b and the forgotten-block factor a*b are simple monomials, and the
final FK-to-Tutte prefactor division reduces to an exponent shift rather
than expensive polynomial long division.

The state space is indexed by non-crossing set partitions of m boundary
vertices, counted by the Catalan number c_m.

The transfer matrix encodes all possible edge-subset transitions between
consecutive columns. Each entry T[new_state, old_state] accumulates
weights b^|S| * (a*b)^f over all edge subsets S that produce the
transition old_state -> new_state, where f is the number of old boundary
blocks that become disconnected from the new boundary during the forget step.

Complexity:
    - Partition enumeration: O(B(m) * m^2), generate all set partitions
      and filter to non-crossing. Cached after first call per width.
    - Transfer matrix construction: O(c_m^2 * 2^|unit_cell_edges|).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Set, Tuple

from ..polynomial import TuttePolynomial

# Type alias: intermediate polynomials in the (a, b) = (x-1, y-1) basis.
# Distinguished from TuttePolynomial (the final result in (x, y) basis)
# to make it clear that these are not yet Tutte polynomials.
Polynomial = TuttePolynomial


# -- Constants -----------------------------------------------------------------

# Maximum boundary width. Catalan numbers grow fast:
#   c_8 = 1430, c_10 = 16796, c_12 = 208012.
# Width 8 is the practical limit for pure Python (seconds). The C extension
# can push to width 10-12 depending on strip length.
MAX_TRANSFER_MATRIX_WIDTH: int = 8

# Catalan numbers c_0..c_12, used to validate partition enumeration.
# c_m = (2m choose m) / (m+1) = count of non-crossing partitions of m elements.
CATALAN_NUMBERS: Tuple[int, ...] = (
    1, 1, 2, 5, 14, 42, 132, 429, 1430,   # c_0 .. c_8
    4862, 16796, 58786, 208012,             # c_9 .. c_12
)


# =============================================================================
# PARTITION UTILITIES
# =============================================================================


def canonicalize_partition(labels: Tuple[int, ...]) -> Tuple[int, ...]:
    """Produce canonical form where labels appear in first-occurrence order.

    Example: (2, 0, 2, 1) -> first-occurrence maps 2->0, 0->1, 1->2
    yielding (0, 1, 0, 2).

    Args:
        labels: A tuple of integer labels representing a set partition.

    Returns:
        Canonical label tuple where the first occurrence of each label
        is in increasing order starting from 0.

    Time complexity: O(m) where m = len(labels).
    """
    if not labels:
        return ()
    mapping: Dict[int, int] = {}
    next_label = 0
    result: List[int] = []
    for lbl in labels:
        if lbl not in mapping:
            mapping[lbl] = next_label
            next_label += 1
        result.append(mapping[lbl])
    return tuple(result)


def is_noncrossing(partition_labels: Tuple[int, ...]) -> bool:
    """Check if a partition (as canonical label tuple) is non-crossing.

    A partition is non-crossing iff no two blocks interleave: there do
    not exist i < j < k < l with labels[i] == labels[k] != labels[j]
    == labels[l].

    Time complexity: O(m^2) where m = len(partition_labels).
    """
    if len(partition_labels) <= 2:
        return True

    # Build position lists for each block.
    block_positions: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(partition_labels):
        block_positions.setdefault(lbl, []).append(idx)

    blocks = list(block_positions.values())

    # Two blocks cross if some element of each lies strictly between
    # the min and max of the other.
    for i, positions_a in enumerate(blocks):
        min_a, max_a = positions_a[0], positions_a[-1]
        if min_a == max_a:
            continue

        for positions_b in blocks[i + 1:]:
            min_b, max_b = positions_b[0], positions_b[-1]
            if min_b == max_b:
                continue

            b_inside_a = any(min_a < pos < max_a for pos in positions_b)
            a_inside_b = any(min_b < pos < max_b for pos in positions_a)

            if b_inside_a and a_inside_b:
                return False

    return True


def _generate_all_partitions(
    num_vertices: int,
    labels: List[int],
    next_fresh: int,
    result: List[Tuple[int, ...]],
) -> None:
    """Generate all set partitions as restricted growth strings.

    Each vertex can join any existing block (labels 0..next_fresh-1)
    or start a new block (label next_fresh).
    """
    if len(labels) == num_vertices:
        result.append(tuple(labels))
        return

    for block_label in range(next_fresh):
        labels.append(block_label)
        _generate_all_partitions(num_vertices, labels, next_fresh, result)
        labels.pop()

    labels.append(next_fresh)
    _generate_all_partitions(num_vertices, labels, next_fresh + 1, result)
    labels.pop()


@lru_cache(maxsize=16)
def enumerate_noncrossing_partitions(
    num_vertices: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Enumerate all non-crossing partitions of [0..num_vertices-1].

    Generates all set partitions as restricted growth strings, then filters
    to keep only non-crossing ones. Count equals Catalan number c_{num_vertices}.

    Args:
        num_vertices: Number of boundary vertices (m).

    Returns:
        Tuple of canonical label tuples, sorted lexicographically.

    Raises:
        ValueError: If num_vertices is negative or exceeds MAX_TRANSFER_MATRIX_WIDTH.

    Time complexity: O(B(m) * m^2) — generate B(m) partitions, filter by
    O(m^2) crossing check. Fast for m <= 8 (B(8) = 4140), slow for
    m >= 10 (B(10) = 115975, B(12) = 4213597).
    """
    if num_vertices < 0:
        raise ValueError(f"num_vertices must be non-negative, got {num_vertices}")
    if num_vertices > MAX_TRANSFER_MATRIX_WIDTH:
        raise ValueError(
            f"num_vertices={num_vertices} exceeds MAX_TRANSFER_MATRIX_WIDTH="
            f"{MAX_TRANSFER_MATRIX_WIDTH}"
        )

    all_partitions: List[Tuple[int, ...]] = []
    _generate_all_partitions(num_vertices, [], 0, all_partitions)

    noncrossing = sorted(p for p in all_partitions if is_noncrossing(p))

    if num_vertices < len(CATALAN_NUMBERS):
        if len(noncrossing) != CATALAN_NUMBERS[num_vertices]:
            raise RuntimeError(
                f"Expected c_{num_vertices}={CATALAN_NUMBERS[num_vertices]} "
                f"non-crossing partitions, got {len(noncrossing)}"
            )

    return tuple(noncrossing)


@lru_cache(maxsize=32)
def partition_index_map(
    num_vertices: int,
) -> Dict[Tuple[int, ...], int]:
    """Build a mapping from canonical partition tuple to its index.

    Used for O(1) lookup of partition state indices during transfer matrix
    construction. The returned mapping is cached and must not be mutated.

    Args:
        num_vertices: Width of the boundary (m).

    Returns:
        Dictionary mapping each canonical partition tuple to its index
        in the enumerated list of non-crossing partitions.
    """
    partitions = enumerate_noncrossing_partitions(num_vertices)
    return {p: i for i, p in enumerate(partitions)}


# =============================================================================
# PARTITION TRANSITION
# =============================================================================


def compute_transition(
    old_partition: Tuple[int, ...],
    selected_edges: List[Tuple[int, int, bool]],
    width: int,
) -> Tuple[Tuple[int, ...], int]:
    """Compute the boundary partition transition for a set of selected edges.

    Models one column transition in a periodic strip. The combined system
    uses 2*width positions:
      - Positions 0..width-1: old boundary (carries old_partition labels).
      - Positions width..2*width-1: new boundary (start as singletons).

    Each edge (row_a, row_b, is_cross_column) is interpreted as:
      - is_cross_column=True: old[row_a] <-> new[row_b].
      - is_cross_column=False: new[row_a] <-> new[row_b].

    After applying all edges, old boundary vertices are "forgotten." Old
    blocks with no connection to the new boundary are counted as forgotten.

    Args:
        old_partition: Canonical partition of the old boundary (width elements).
        selected_edges: Edges present in this edge subset.
        width: Number of boundary vertices (m).

    Returns:
        (new_partition, num_forgotten_blocks).

    Time complexity: O(|selected_edges| * width).
    """
    # Combined partition: old boundary labels + fresh singletons for new boundary.
    max_old_label = max(old_partition) if old_partition else -1
    combined = list(old_partition) + [
        max_old_label + 1 + i for i in range(width)
    ]

    # Merge blocks for each selected edge.
    for row_a, row_b, is_cross in selected_edges:
        if is_cross:
            pos_a, pos_b = row_a, width + row_b
        else:
            pos_a, pos_b = width + row_a, width + row_b

        label_a = combined[pos_a]
        label_b = combined[pos_b]
        if label_a != label_b:
            keep, replace = min(label_a, label_b), max(label_a, label_b)
            combined = [keep if lbl == replace else lbl for lbl in combined]

    # Count forgotten blocks: old labels with no presence in new boundary.
    new_boundary_labels = set(combined[width:])
    forgotten_labels: Set[int] = set()
    for pos in range(width):
        lbl = combined[pos]
        if lbl not in new_boundary_labels:
            forgotten_labels.add(lbl)

    new_partition = canonicalize_partition(tuple(combined[width:]))
    return new_partition, len(forgotten_labels)


# =============================================================================
# TRANSFER MATRIX BUILDER
# =============================================================================


def build_transfer_matrix(
    width: int,
    unit_cell_edges: List[Tuple[int, int, bool]],
) -> List[List[Polynomial]]:
    """Build the c_m x c_m transfer matrix for a periodic lattice strip.

    Enumerates all 2^|unit_cell_edges| edge subsets, computes the boundary
    partition transition for each (old_state, subset) pair, and accumulates
    weights into T[new_state, old_state].

    Each entry accumulates: b^|S| * (a*b)^f, where |S| is the number of
    selected edges and f is the number of forgotten blocks.

    Args:
        width: Number of boundary vertices (m). Must be in [1, MAX_TRANSFER_MATRIX_WIDTH].
        unit_cell_edges: Edge pattern for one column transition.
            Each edge (row_a, row_b, is_cross_column) where is_cross_column=True
            means a cross-column edge (old row_a to new row_b) and False means
            a within-column edge (new row_a to new row_b).

    Returns:
        c_m x c_m matrix of Polynomial entries in (a, b) basis.

    Raises:
        ValueError: If width is < 1 or exceeds MAX_TRANSFER_MATRIX_WIDTH.

    Time complexity: O(c_m^2 * 2^|unit_cell_edges|).
    """
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    if width > MAX_TRANSFER_MATRIX_WIDTH:
        raise ValueError(
            f"width={width} exceeds MAX_TRANSFER_MATRIX_WIDTH="
            f"{MAX_TRANSFER_MATRIX_WIDTH}"
        )

    num_edges = len(unit_cell_edges)
    partitions = enumerate_noncrossing_partitions(width)
    partition_to_idx = partition_index_map(width)
    num_states = len(partitions)

    matrix: List[List[Polynomial]] = [
        [Polynomial.zero() for _ in range(num_states)]
        for _ in range(num_states)
    ]

    # Precompute b^k for edge counts 0..num_edges.
    # b = (y-1) in (a,b) basis, so b^k is the monomial {(0, k): 1}.
    b_powers = [Polynomial.from_coefficients({(0, k): 1}) for k in range(num_edges + 1)]

    # Precompute (a*b)^k for forgotten-block counts 0..width.
    # a*b = (x-1)(y-1), so (a*b)^k is {(k, k): 1}.
    ab_powers = [Polynomial.from_coefficients({(k, k): 1}) for k in range(width + 1)]

    # Enumerate all 2^num_edges edge subsets.
    for subset_mask in range(1 << num_edges):
        selected_edges: List[Tuple[int, int, bool]] = [
            unit_cell_edges[i]
            for i in range(num_edges)
            if subset_mask & (1 << i)
        ]
        edge_weight = b_powers[len(selected_edges)]

        for old_idx, old_partition in enumerate(partitions):
            new_partition, num_forgotten = compute_transition(
                old_partition, selected_edges, width
            )

            if new_partition not in partition_to_idx:
                continue

            new_idx = partition_to_idx[new_partition]

            weight = edge_weight
            if num_forgotten > 0:
                weight = weight * ab_powers[num_forgotten]

            matrix[new_idx][old_idx] = matrix[new_idx][old_idx] + weight

    return matrix