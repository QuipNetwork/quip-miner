"""Generic Aut-based orbit canonicalizer.

For a graph cell C with automorphism group Aut(C), two partitions P_1, P_2
of V(C) are in the same Aut-orbit iff there exists σ ∈ Aut(C) such that
σ(P_1) = P_2 (as a partition).

T_rooted is invariant under Aut(C): T_rooted(C, S)[P_1] = T_rooted(C, S)[P_2]
when P_1, P_2 are in the same orbit. So orbit-compressing T_rooted by the
canonical orbit-rep is correct compression.

This module:
1. Computes Aut(C) via NetworkX VF2 (one computation per cell template).
2. Provides `canonical_partition(P, aut_group) -> Tuple` — the lex-min
   form over all aut applications.

GRAPH-AGNOSTIC: works for any cell with non-trivial automorphism group,
not just K_{4,4} or D-Wave families.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx

from ..graph import Graph
from ..polynomial import TuttePolynomial


_AUT_CACHE: Dict[str, List[Dict[int, int]]] = {}


def compute_cell_aut(cell: Graph) -> List[Dict[int, int]]:
    """Compute the automorphism group of `cell` via NetworkX VF2.

    Returns list of automorphisms; each is a dict mapping
    original vertex → permuted vertex.

    Cached by cell.canonical_key(). For K_{4,4}, returns 1152 auts.
    """
    key = cell.canonical_key()
    if key in _AUT_CACHE:
        return _AUT_CACHE[key]

    nxg = nx.Graph()
    nxg.add_nodes_from(cell.nodes)
    nxg.add_edges_from(cell.edges)
    gm = nx.algorithms.isomorphism.GraphMatcher(nxg, nxg)
    auts = list(gm.isomorphisms_iter())
    _AUT_CACHE[key] = auts
    return auts


def _apply_aut_to_partition(
    P: Tuple[Tuple[int, ...], ...],
    aut: Dict[int, int],
) -> Tuple[Tuple[int, ...], ...]:
    """Apply aut (vertex permutation) to partition P; return canonical form."""
    new_blocks = []
    for block in P:
        permuted = tuple(sorted(aut.get(v, v) for v in block))
        new_blocks.append(permuted)
    return tuple(sorted(new_blocks))


_CANONICAL_CACHE: Dict[Tuple[int, Tuple], Tuple] = {}


def canonical_partition(
    P: Tuple[Tuple[int, ...], ...],
    aut_group: List[Dict[int, int]],
) -> Tuple[Tuple[int, ...], ...]:
    """Return the lex-min partition over all aut applications.

    Two partitions in the same Aut-orbit produce the same canonical form.
    Cached by (aut_group identity, P) — fast for repeated calls with same aut.
    """
    if not aut_group:
        return P
    cache_key = (id(aut_group), P)
    cached = _CANONICAL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    canonical = None
    for aut in aut_group:
        candidate = _apply_aut_to_partition(P, aut)
        if canonical is None or candidate < canonical:
            canonical = candidate
    _CANONICAL_CACHE[cache_key] = canonical
    return canonical


def clear_canonical_cache() -> None:
    """Clear the canonical partition cache (call between independent runs)."""
    _CANONICAL_CACHE.clear()


def aut_compress_t_rooted(
    T_rooted: Dict[Tuple[Tuple[int, ...], ...], TuttePolynomial],
    aut_group: List[Dict[int, int]],
) -> Tuple[Dict[Tuple, TuttePolynomial], Dict[Tuple, List[Tuple]]]:
    """Compress T_rooted dict by Aut-orbit canonical form.

    Returns:
        orbit_T: dict from canonical_partition → polynomial (one per orbit)
        orbit_partitions: dict from canonical → list of all partitions in orbit

    Validates that all partitions in each orbit have the SAME polynomial value
    (uniform within orbit). Raises ValueError otherwise.
    """
    orbit_T: Dict[Tuple, TuttePolynomial] = {}
    orbit_partitions: Dict[Tuple, List[Tuple]] = defaultdict(list)
    for P, val in T_rooted.items():
        canonical = canonical_partition(P, aut_group)
        orbit_partitions[canonical].append(P)
        if canonical in orbit_T:
            if orbit_T[canonical] != val:
                raise ValueError(
                    f"Orbit {canonical} has non-uniform T values: "
                    f"P_existing={orbit_partitions[canonical][0]}, P_new={P}"
                )
        else:
            orbit_T[canonical] = val
    return orbit_T, dict(orbit_partitions)


def per_cell_canonical_key(
    P: Tuple[Tuple[int, ...], ...],
    cell_anchor_groups: List[List[int]],
) -> Tuple[Tuple[int, ...], ...]:
    """Structural canonical key invariant under S_n^N (independent
    permutation within each cell's anchor group).

    For each block B of P, compute the per-cell count tuple
    (|B ∩ cell_anchor_groups[0]|, |B ∩ cell_anchor_groups[1]|, ...).
    The canonical key is the sorted multiset of these tuples.

    Mathematical basis: when each cell c's anchor positions are
    permutable freely (S_{|cell_anchor_groups[c]|} acts on them with
    no cross-cell coupling), the orbit of a partition is fully
    determined by the multiset of per-block per-cell counts. Two
    partitions with the same multiset are S_n^N-equivalent; T_rooted
    is the same on all members of the orbit.

    Cost: O(sum of block sizes) per partition. No aut enumeration.

    Positions in P that don't belong to any cell group are placed in
    a synthetic "outside" group and tracked with a -1 cell index in
    the output tuples.

    PRECONDITION (callers MUST verify, NOT enforced here): the cell
    graph backing each anchor group must support free S_n permutation
    of its anchors — the FULL symmetric group on the anchor positions
    must act on the cell as automorphisms preserving rooted T values.
    Cells satisfying this: K_n (any n), K_{a,b} (anchors all on one
    bipartition side). Cells NOT satisfying this: Petersen, cycles,
    Möbius–Kantor, and other vertex-transitive cells whose Aut group
    is a proper subgroup of S_n on the anchors.

    Using this function with non-K_n / non-K_{a,b} cells over-collapses
    partition orbits and yields silently wrong T values when used as a
    deduplication key. Verified empirically (May 2026):
    `tutte/research/scripts/cascade_audit_per_cell_canonical.py`.
    See `tutte/research/cascade_audit_findings.md` for details.

    Current call sites (all safe — K_{4,4} cells only):
    - cell_quotient_helpers.py
    - cell_quotient_path.py

    Adding any non-K_{a,b} cell type to the roots/ DP requires
    replacing this with `canonical_partition(P, Aut(cell_graph))` from
    this same module, which uses the actual Aut group (correct for any
    cell type, but more expensive).
    """
    pos_to_cell: Dict[int, int] = {}
    for cell_idx, group in enumerate(cell_anchor_groups):
        for p in group:
            pos_to_cell[p] = cell_idx
    n_cells = len(cell_anchor_groups)
    block_shapes = []
    for block in P:
        shape = [0] * (n_cells + 1)  # last slot is "outside" (-1)
        for v in block:
            ci = pos_to_cell.get(v, -1)
            shape[ci if ci >= 0 else n_cells] += 1
        block_shapes.append(tuple(shape))
    return tuple(sorted(block_shapes))


def per_cell_orbit_size(
    canonical_key: Tuple[Tuple[int, ...], ...],
    cell_anchor_groups: List[List[int]],
) -> int:
    """Number of partitions in the per-cell S_n^N orbit indexed by `canonical_key`.

    Computed analytically from the canonical key (sorted multiset of
    per-cell shape tuples) without enumerating partitions:

      For each cell c independently, count the multinomial of placing
      c's anchors into blocks with the per-block shape s_c. Then divide
      by the symmetry of blocks with identical shape (since the blocks
      themselves are unordered).

    This lets us compute output orbit sizes for `M_precompute` /
    `orbit_convolve` without Bell(W) enumeration.
    """
    from collections import Counter
    from math import factorial

    n_cells = len(cell_anchor_groups)
    n_per_cell = [len(g) for g in cell_anchor_groups]
    if n_cells == 0:
        # Special case: no cells => every position is "outside".
        # Treat as a single group with S_n acting (all anchors interchangeable).
        # But canonical_key is over (n_cells + 1) entries; outside is index n_cells.
        outside_n = sum(shape[0] for shape in canonical_key) if canonical_key else 0
        if outside_n == 0:
            return 1
        return _multinomial_then_dedupe(canonical_key, [outside_n], outside_only=True)

    # Compute per-cell multinomial * outside multinomial.
    size = 1
    for c in range(n_cells):
        s_values = [shape[c] for shape in canonical_key]
        if sum(s_values) != n_per_cell[c]:
            # Canonical key inconsistent with cell sizes.
            raise ValueError(
                f"Canonical key column {c} sums to {sum(s_values)} but cell "
                f"has {n_per_cell[c]} anchors"
            )
        mult = factorial(n_per_cell[c])
        for s in s_values:
            mult //= factorial(s)
        size *= mult
    # "Outside" cell (positions not in any anchor group) — index n_cells in shape tuples.
    if canonical_key and len(canonical_key[0]) > n_cells:
        outside_per_block = [shape[n_cells] for shape in canonical_key]
        outside_total = sum(outside_per_block)
        if outside_total > 0:
            mult = factorial(outside_total)
            for s in outside_per_block:
                mult //= factorial(s)
            size *= mult

    # Divide by block-shape-multiset symmetry: blocks with identical
    # shape are interchangeable.
    shape_counts = Counter(canonical_key)
    for s, count in shape_counts.items():
        size //= factorial(count)
    return size


def _multinomial_then_dedupe(canonical_key, sizes, outside_only=False):
    from collections import Counter
    from math import factorial
    s_values = [shape[0] for shape in canonical_key]
    mult = factorial(sizes[0])
    for s in s_values:
        mult //= factorial(s)
    shape_counts = Counter(canonical_key)
    for s, count in shape_counts.items():
        mult //= factorial(count)
    return mult


def per_cell_orbit_rep(
    canonical_key: Tuple[Tuple[int, ...], ...],
    cell_anchor_groups: List[List[int]],
) -> Tuple[Tuple[int, ...], ...]:
    """Construct a representative partition for the per-cell orbit
    indexed by `canonical_key`.

    Greedy: for each block-shape in canonical_key, pick the next
    available positions from each cell's anchor group.
    """
    next_idx = [0] * len(cell_anchor_groups)
    blocks: List[Tuple[int, ...]] = []
    for shape in canonical_key:
        block: List[int] = []
        for c in range(len(cell_anchor_groups)):
            s = shape[c]
            for _ in range(s):
                if next_idx[c] >= len(cell_anchor_groups[c]):
                    raise IndexError(
                        f"per_cell_orbit_rep: cell {c} ran out of positions "
                        f"for canonical_key {canonical_key}"
                    )
                block.append(cell_anchor_groups[c][next_idx[c]])
                next_idx[c] += 1
        # Outside positions (index n_cells in shape) — typically not present
        # since cell_anchor_groups should cover all positions, but handle
        # gracefully.
        blocks.append(tuple(sorted(block)))
    return tuple(sorted(blocks))


def enumerate_per_cell_aut_group(
    cell_anchor_groups: List[List[int]],
) -> List[Dict[int, int]]:
    """Enumerate G = product of S_n per cell-group as Dict[pos→pos] elements.

    G is the per-cell automorphism group: independent S_n permutation
    within each cell-group, no cross-cell coupling. Positions outside any
    cell-group are fixed by every element.

    |G| = ∏ over cells of |cell|!. For 3 cells of 4 positions: |G| = 24³
    = 13 824. For 2 cells of 4 positions: |G| = 576.

    Supports pair-orbit-aware
    convolution where both state and junction are compressed by the same
    per-cell aut group on shared boundary.
    """
    from itertools import permutations, product

    per_group_perms = [list(permutations(group)) for group in cell_anchor_groups]
    elements: List[Dict[int, int]] = []
    for perm_combo in product(*per_group_perms):
        mapping: Dict[int, int] = {}
        for original_group, new_perm in zip(cell_anchor_groups, perm_combo):
            for original, new in zip(original_group, new_perm):
                mapping[original] = new
        elements.append(mapping)
    return elements


def apply_perm_to_partition(
    P: Tuple[Tuple[int, ...], ...],
    perm: Dict[int, int],
) -> Tuple[Tuple[int, ...], ...]:
    """Apply position-permutation `perm` to partition `P`; return canonical sorted form."""
    return tuple(sorted(
        tuple(sorted(perm.get(v, v) for v in block))
        for block in P
    ))


def per_cell_partition_stab(
    P: Tuple[Tuple[int, ...], ...],
    G_elements: List[Dict[int, int]],
) -> List[Dict[int, int]]:
    """Subset of G_elements that fix partition P setwise."""
    P_canonical = tuple(sorted(tuple(sorted(b)) for b in P))
    return [
        σ for σ in G_elements
        if apply_perm_to_partition(P_canonical, σ) == P_canonical
    ]


class PerCellPreconditionViolated(ValueError):
    """Raised by `aut_compress_t_rooted_per_cell` when its precondition
    (free S_n action on cell anchors preserves T_rooted values) is
    violated by the input ``T_rooted`` dict.

    Callers should catch this and fall back to a non-compressed code
    path (e.g., `aut_compress_t_rooted(..., [])` to use full canonical
    partitions with no aut compression). Distinct from generic
    ``ValueError`` so callers can target their except clauses.
    """


def aut_compress_t_rooted_per_cell(
    T_rooted: Dict[Tuple[Tuple[int, ...], ...], TuttePolynomial],
    cell_anchor_groups: List[List[int]],
) -> Tuple[Dict[Tuple, TuttePolynomial], Dict[Tuple, List[Tuple]]]:
    """Compress T_rooted by per-cell structural orbit (S_n^N).

    Returns:
        orbit_T: dict from per_cell_canonical_key → polynomial (one per orbit).
        orbit_partitions: dict from canonical → list of all partitions in orbit.

    Validates that all partitions in each orbit have the SAME
    polynomial value (T_rooted is invariant within S_n^N orbit).
    Raises ``PerCellPreconditionViolated`` (a subclass of ValueError)
    if the precondition fails — callers should catch and fall back
    to non-compressed compression via `aut_compress_t_rooted(..., [])`.
    """
    orbit_T: Dict[Tuple, TuttePolynomial] = {}
    orbit_partitions: Dict[Tuple, List[Tuple]] = defaultdict(list)
    for P, val in T_rooted.items():
        canonical = per_cell_canonical_key(P, cell_anchor_groups)
        orbit_partitions[canonical].append(P)
        if canonical in orbit_T:
            if orbit_T[canonical] != val:
                raise PerCellPreconditionViolated(
                    f"per-cell orbit {canonical} has non-uniform T values: "
                    f"P_existing={orbit_partitions[canonical][0]}, P_new={P}, "
                    f"existing_val={orbit_T[canonical]}, new_val={val}"
                )
        else:
            orbit_T[canonical] = val
    return orbit_T, dict(orbit_partitions)


def build_relabel_aut(
    cell_template: Graph,
    cell_anchors_orig: List[int],
    cell_anchors_pos: List[int],
    preserve_split_index=None,
    preserve_groups=None,
) -> List[Dict[int, int]]:
    """Translate the cell's aut group to use `pos` labels for cell's anchor vertices.

    Args:
        cell_template: graph whose Aut group to use
        cell_anchors_orig: anchor vertices in cell_template's labels
        cell_anchors_pos: corresponding labels in the position space
        preserve_split_index: if not None, indicates the boundary between
            "left" and "right" anchors in cell_anchors_orig. Auts must
            preserve this split (left → left, right → right) to be kept.
        preserve_groups: optional list of index lists into cell_anchors_orig.
            Each list specifies a group of operationally-distinct anchor
            positions that must be preserved (e.g., [[0], [1], [2,3]] for
            a cell with left anchor [0], right anchor [1], extras [2,3]).
            Overrides preserve_split_index when provided.
    """
    orig_to_pos = {cell_anchors_orig[i]: cell_anchors_pos[i]
                   for i in range(len(cell_anchors_orig))}

    preserve_sets = None
    if preserve_groups is not None:
        preserve_sets = [
            set(cell_anchors_orig[i] for i in indices) for indices in preserve_groups
        ]
    elif preserve_split_index is not None:
        left_set = set(cell_anchors_orig[:preserve_split_index])
        right_set = set(cell_anchors_orig[preserve_split_index:])
        preserve_sets = [left_set, right_set]

    cell_auts = compute_cell_aut(cell_template)
    boundary_auts = []
    seen = set()
    anchor_set = set(cell_anchors_orig)
    for aut in cell_auts:
        if not all(aut[v] in anchor_set for v in cell_anchors_orig):
            continue
        if preserve_sets is not None:
            ok = True
            for grp in preserve_sets:
                if not all(aut[v] in grp for v in grp if v in cell_anchors_orig):
                    ok = False
                    break
            if not ok:
                continue
        boundary_aut = {orig_to_pos[v]: orig_to_pos[aut[v]] for v in cell_anchors_orig}
        key = tuple(sorted(boundary_aut.items()))
        if key not in seen:
            seen.add(key)
            boundary_auts.append(boundary_aut)
    return boundary_auts
