"""Subgraph signature tables for cotree DP.

Defines the signature types and provides the three core operations:
- Leaf: base case table for a single vertex
- Disjoint union combine: no edges between children
- Complete union combine (Algorithm 3.1): all edges between children

A signature is a sorted tuple of component sizes representing the
structure of a spanning subgraph. For example, a spanning subgraph
with components of sizes {3, 2, 1, 1} is represented as (3, 2, 1, 1)
— sorted in non-increasing order for canonical dict keys.

A double-signature tracks how components split across the two sides
of a complete union (⊗) operation. Each entry (f_size, g_size) records
how many F-side and G-side vertices have been absorbed into a merged
component.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import comb
from typing import Dict, Iterable, List, Tuple


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class Signature(tuple):
    """Component-size multiset, always sorted in non-increasing order.

    Represents the sizes of connected components in a spanning subgraph.
    Example: components of sizes {3, 2, 1, 1} → Signature([3, 2, 1, 1])

    The constructor automatically sorts the input, so
    Signature([1, 3, 1, 2]) produces (3, 2, 1, 1).

    Immutable and hashable (inherits from tuple).
    """

    def __new__(cls, parts: Iterable[int] = ()) -> 'Signature':
        return super().__new__(cls, sorted(parts, reverse=True))

    def merge(self, other: 'Signature') -> 'Signature':
        """Multiset union of two signatures."""
        return Signature(list(self) + list(other))

    def to_double(self) -> 'DoubleSig':
        """Convert to initial double-signature: all parts on F-side."""
        return DoubleSig((part_size, 0) for part_size in self)


class DoubleSig(tuple):
    """Multiset of (f_size, g_size) pairs, sorted in non-increasing order.

    Tracks how components split across the F-side and G-side of a
    complete union (⊗) operation. Each pair records how many F-side
    and G-side vertices have been absorbed into a merged component.

    The constructor automatically sorts the input.
    Immutable and hashable (inherits from tuple).
    """

    def __new__(cls, pairs: Iterable[Tuple[int, int]] = ()) -> 'DoubleSig':
        return super().__new__(cls, sorted(pairs, reverse=True))

    def to_signature(self) -> Signature:
        """Convert back to a regular signature.

        Each (f_size, g_size) pair becomes a single component of
        size f_size + g_size.
        """
        return Signature(f_size + g_size for f_size, g_size in self)


# Maps (Signature, edge_count) → count of spanning subgraphs
# with that component structure and exactly that many edges.
SubgraphTable = Dict[Tuple[Signature, int], int]


# =============================================================================
# LEAF
# =============================================================================

def leaf_subgraph_table(vertex: int) -> SubgraphTable:
    """Subgraph signature table for a single vertex.

    A single vertex has one spanning subgraph: itself with 0 edges.
    """
    return {(Signature([1]), 0): 1}


# =============================================================================
# DISJOINT UNION COMBINE
# =============================================================================

def disjoint_union_subgraph_combine(
    table_f: SubgraphTable,
    table_g: SubgraphTable,
) -> SubgraphTable:
    """Combine subgraph tables for disjoint union F | G.

    No edges between F and G — signatures concatenate, edge counts add.
    """
    result: SubgraphTable = defaultdict(int)
    for (sig_f, edges_f), count_f in table_f.items():
        for (sig_g, edges_g), count_g in table_g.items():
            merged_sig = sig_f.merge(sig_g)
            result[(merged_sig, edges_f + edges_g)] += count_f * count_g
    return dict(result)


# =============================================================================
# COMPLETE UNION COMBINE (ALGORITHM 3.1)
# =============================================================================

def complete_union_subgraph_combine(
    table_f: SubgraphTable,
    table_g: SubgraphTable,
) -> SubgraphTable:
    """Algorithm 3.1: Combine subgraph tables for complete union F * G.

    All possible edges exist between F and G. For each pair of F/G
    signatures, enumerate which cross-edges to include and how they
    merge components. Uses CellSel (Algorithm 3.2) for edge counting.

    Args:
        table_f, table_g: Subgraph tables of the two children.
        num_f_vertices: Number of vertices in F.
    """
    result: SubgraphTable = defaultdict(int)

    for (sig_f, edges_f), count_f in table_f.items():
        for (sig_g, edges_g), count_g in table_g.items():
            contributions = complete_union_subgraph_pair(sig_f, sig_g)
            for (merged_sig, extra_edges), contrib_count in contributions.items():
                result[(merged_sig, edges_f + edges_g + extra_edges)] += (
                    count_f * count_g * contrib_count
                )

    return dict(result)


def complete_union_subgraph_pair(
    sig_f: Signature,
    sig_g: Signature,
) -> Dict[Tuple[Signature, int], int]:
    """Core of Algorithm 3.1: complete union combine with edge counting.

    For each G-side component, enumerate which F-side components it merges
    with (via cross-edges), and use CellSel to count the number of ways
    to select the required edges.

    Args:
        sig_f: Signature of subgraph in F.
        sig_g: Signature of subgraph in G.

    Returns:
        Dict mapping (signature, extra_edges) to counts.
    """

    # State table: maps (double_sig, edges_accumulated) -> count
    init_double_sig = sig_f.to_double()
    state: Dict[Tuple[DoubleSig, int], int] = {(init_double_sig, 0): 1}

    for g_comp_size in sig_g:
        next_state: Dict[Tuple[DoubleSig, int], int] = defaultdict(int)

        for (beta, edges_accumulated), beta_count in state.items():
            for gamma, multi_coeff in distinct_submultisets(beta):
                f_total = sum(f_size for f_size, _g in gamma)
                g_total = sum(g_size for _f, g_size in gamma)

                beta_minus_gamma = multiset_diff(beta, gamma)
                merged_entry = (f_total, g_total + g_comp_size)
                beta_prime = DoubleSig(
                    list(beta_minus_gamma) + [merged_entry]
                )

                # Cell sizes: for each selected component with f_size F-side
                # vertices, there are g_comp_size × f_size edges in
                # K_{g_comp_size, f_size}. CellSel counts ways to pick exactly
                # num_edges_to_add edges total with at least one from each cell.
                cell_sizes = [g_comp_size * f_size for f_size, _g in gamma]
                max_possible_edges = g_comp_size * f_total

                for num_edges_to_add in range(len(gamma), max_possible_edges + 1):
                    num_selections = cellsel(cell_sizes, num_edges_to_add)

                    if num_selections > 0:
                        next_state[(beta_prime, edges_accumulated + num_edges_to_add)] += (
                            beta_count * multi_coeff * num_selections
                        )

        state = dict(next_state)

    # Convert double-signatures to regular signatures
    output: Dict[Tuple[Signature, int], int] = defaultdict(int)
    for (double_sig, total_edges), count in state.items():
        merged_sig = double_sig.to_signature()
        output[(merged_sig, total_edges)] += count

    return dict(output)


# ===========================================================================
# COMBINATORICS (merged from former cotree_dp/combinatorics.py)
# ===========================================================================

# =============================================================================
# CELLSEL (ALGORITHM 3.2) — WITH MEMOIZATION
# =============================================================================

# Cache: maps (sorted_cell_sizes_tuple, total_to_select) → result.
# Cell sizes are sorted for canonical keys — cellsel is order-independent
# (the DP result doesn't depend on the order of cells).
#
# The cache is auto-cleared by compute_tutte_cotree_dp after each graph.
# For direct calls to cellsel (e.g. from notebooks or benchmarks), the
# cache is capped at _CELLSEL_CACHE_MAX entries to bound memory.
# At 500K entries the estimated memory usage is ~100 MB.
#
# Empirical peak cache sizes (measured with auto-clear disabled):
#   K_10:     437 entries (~85 KB)
#   K_20:  22,375 entries (~4.3 MB)
#   K_30: 405,613 entries (~77 MB)
_CELLSEL_CACHE_MAX: int = 500_000
_cellsel_cache: Dict[Tuple[Tuple[int, ...], int], int] = {}


def clear_cellsel_cache() -> None:
    """Clear the CellSel memoization cache.

    Call between unrelated graph computations to bound memory usage.
    The cache is also cleared automatically by compute_tutte_cotree_dp.
    """
    _cellsel_cache.clear()


def cellsel(cell_sizes: List[int], total_to_select: int) -> int:
    """Algorithm 3.2: Count cellular selections (memoized).

    Given num_cells pairwise disjoint cells of sizes d_1, d_2, ..., d_k,
    count the number of ways to select exactly total_to_select elements
    such that at least one element is selected from every cell.

    Results are cached by (sorted cell sizes, total_to_select). On K_14,
    this eliminates ~98% of redundant calls (158K duplicates out of 161K).

    Args:
        cell_sizes: List of cell sizes [d_1, d_2, ..., d_k].
        total_to_select: Total number of elements to select.

    Returns:
        Number of cellular selections.

    Complexity: O(num_cells × total_to_select²) on cache miss; O(1) on hit.
    """
    # Fast path for trivial cases (most common call — 3600x on K_14)
    if not cell_sizes:
        return 1 if total_to_select == 0 else 0

    num_cells = len(cell_sizes)
    if total_to_select < num_cells:
        return 0

    # Cache lookup with sorted key for order-independence
    cache_key = (tuple(sorted(cell_sizes)), total_to_select)
    cached = _cellsel_cache.get(cache_key)
    if cached is not None:
        return cached

    result = _cellsel_compute(cell_sizes, num_cells, total_to_select)
    if len(_cellsel_cache) < _CELLSEL_CACHE_MAX:
        _cellsel_cache[cache_key] = result
    return result


def _cellsel_compute(
    cell_sizes: List[int],
    num_cells: int,
    total_to_select: int,
) -> int:
    """Core CellSel DP (uncached).

    Separated from cellsel() so the cache logic doesn't clutter the algorithm.
    """
    # DP: ways_prev[selected] = number of ways to select `selected` elements
    # from the first `cell_idx` cells with at least one from each cell.
    ways_prev = [0] * (total_to_select + 1)
    for selected in range(1, min(cell_sizes[0], total_to_select) + 1):
        ways_prev[selected] = comb(cell_sizes[0], selected)

    cumulative_size = cell_sizes[0]

    for cell_idx in range(1, num_cells):
        current_cell_size = cell_sizes[cell_idx]
        cumulative_size += current_cell_size
        ways_curr = [0] * (total_to_select + 1)
        for selected in range(cell_idx + 1, min(total_to_select, cumulative_size) + 1):
            for from_this_cell in range(1, min(selected - cell_idx, current_cell_size) + 1):
                ways_curr[selected] += (
                    ways_prev[selected - from_this_cell]
                    * comb(current_cell_size, from_this_cell)
                )
        ways_prev = ways_curr

    return ways_prev[total_to_select]


# =============================================================================
# SUBMULTISET ENUMERATION
# =============================================================================

def distinct_submultisets(
    multiset: DoubleSig,
) -> List[Tuple[DoubleSig, int]]:
    """Enumerate distinct submultisets of multiset (including empty).

    Returns list of (submultiset, multinomial_coefficient) pairs.
    The multinomial coefficient accounts for repeated elements:
    if an element appears `multiplicity` times in the multiset and we
    choose it `chosen` times, the coefficient includes C(multiplicity, chosen).

    Includes the empty submultiset (coefficient 1). For forest counting
    (Stage 1), the empty submultiset represents a G-side component that
    stays disconnected from all F-side components — correct for forests
    (which allow disconnected components).
    """
    element_counts = Counter(multiset)
    unique_elements = list(element_counts.keys())

    result: List[Tuple[DoubleSig, int]] = []
    _enum_submultisets(unique_elements, element_counts, 0, [], 1, result)
    return result


def _enum_submultisets(
    unique_elements: list,
    element_counts: dict,
    element_idx: int,
    current_selection: list,
    accumulated_coeff: int,
    result: list,
) -> None:
    """Recursive enumeration of distinct submultisets with coefficients."""
    if element_idx == len(unique_elements):
        submultiset = tuple(sorted(current_selection, reverse=True))
        result.append((submultiset, accumulated_coeff))
        return

    element = unique_elements[element_idx]
    multiplicity = element_counts[element]

    # Choose 0, 1, ..., multiplicity copies of this element
    for num_chosen in range(multiplicity + 1):
        new_coeff = accumulated_coeff * comb(multiplicity, num_chosen)
        new_selection = current_selection + [element] * num_chosen
        _enum_submultisets(
            unique_elements, element_counts,
            element_idx + 1, new_selection, new_coeff, result,
        )


# =============================================================================
# MULTISET DIFFERENCE
# =============================================================================

def multiset_diff(multiset: DoubleSig, to_remove: DoubleSig) -> DoubleSig:
    """Multiset difference: multiset minus to_remove.

    Uses Counter subtraction for O(n + k) instead of O(n × k)
    from repeated list.remove() calls.
    """
    result = Counter(multiset)
    result.subtract(to_remove)
    return DoubleSig(sorted(result.elements(), reverse=True))
