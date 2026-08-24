"""Cell-quotient TREE DP — generalizes path DP to tree topologies.

For graphs whose hierarchical decomposition has cell-quotient topology
of a TREE (n cells, n-1 junctions, no cycles).

Architecture: post-order recursive DP over the tree, mirroring the
existing `compute_path_dp` pattern but allowing branching at non-leaf
cells.

This module exposes two DP entry points:

1. `compute_tree_dp_recursive(spec)` — orbit-compressed tree DP for
   tree-topology cell-quotients with NO cross-cell vertex
   identifications. Uses `precompute_M_table` + `orbit_convolve` for
   the (xy-1)^d junction convolution.

2. `compute_corrected_leaf_dp(spec)` — brute-force DP for chord-rule
   leaves of the hybrid cycle-close path, supporting cross-cell vertex
   identifications via the corrected vertex-identification
   convolution rule:

       T_combined[P_comb] = Σ_{(P_1, P_2) → P_comb}
           T_rooted_1[P_1] · T_rooted_2[P_2]
           · (y-1)^{k-m} / (x-1)^{m - c_comp}

   where k = #shared positions, m = merge events on shared positions,
   c_comp = #components in the merged template (1 for connected cells,
   k for an M_k matching template). See
   `tutte/research/data/step3_milestone_b_design.md` for derivation.

References:
- `tutte/roots/cell_quotient_path.py` — path DP this mirrors
- `tutte/roots/cell_quotient_helpers.py` — M-table + convolve primitives
- `tutte/research/data/tree_dp_design.md` — design doc
- `tutte/research/data/step3_milestone_b_design.md` — corrected
  cross-cell-ID convolution rule (RESOLVED 2026-05-07)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from ..graph import Graph
from ..polynomial import TuttePolynomial
from .aut_orbit import (aut_compress_t_rooted, aut_compress_t_rooted_per_cell,
                        build_relabel_aut, canonical_partition,
                        compute_cell_aut, per_cell_canonical_key,
                        per_cell_orbit_rep, per_cell_orbit_size)
from .cell_quotient_helpers import (components_touching,
                                    enumerate_partitions_per_orbit,
                                    orbit_convolve,
                                    precompute_M_and_convolve_streaming_mod,
                                    precompute_M_table)
from .rooted_tutte import (delta, divide_by_x_minus_1_power, join_partitions,
                           relabel_partition_dict, restrict_partition,
                           t_rooted_cached)

# =============================================================================
# Spec / data structures
# =============================================================================


@dataclass
class CellTreeSpec:
    """Specification for a cell-tree topology DP.

    Semantics: linear-path tree topology over isomorphic cells joined by
    a single junction template.

    `cross_cell_identifications`: optional list of
    ``(cell_i, anchor_template_i, cell_j, anchor_template_j)`` tuples
    declaring that two anchor positions in DIFFERENT cells are
    physically the same vertex. Used by the hybrid cycle-close path
    (`tutte/deprecated/cell_quotient_hybrid.py`) to represent contracted
    junction endpoints. Within-cell shared anchors (same template
    index in multiple neighbor groups of one cell) are handled
    automatically by `_allocate_tree_positions` and do NOT need
    entries here. Default empty list reproduces existing behavior.
    """
    cell_template: Graph
    junction_template: Graph
    cell_tree: nx.Graph
    cell_anchor_groups: Dict[int, Dict[int, List[int]]]
    junction_anchors_A: List[int]
    junction_anchors_B: List[int]
    root: int
    cross_cell_identifications: List[Tuple[int, int, int, int]] = field(
        default_factory=list
    )
    # Per-cell list of template anchors to KEEP OPEN as boundary
    # positions BEYOND the junction-anchor set. Used by the hybrid
    # cycle-close path to preserve contracted closing-junction anchors
    # that have no remaining neighbor connection but ARE shared with
    # another cell via ``cross_cell_identifications``. These positions
    # are added to the cell's T_rooted boundary, never marginalized
    # within the cell, and propagate up to the cell's parent until
    # they meet their cross-cell-identified counterpart at a common
    # ancestor.
    extra_open_anchors: Dict[int, List[int]] = field(default_factory=dict)


# =============================================================================
# Linear-path tree DP — mirrors compute_path_dp
# =============================================================================


def compute_tree_dp_simple(spec: CellTreeSpec) -> TuttePolynomial:
    """Tree DP for LINEAR PATH topology (degenerate tree).

    Uses precompute_M_table + orbit_convolve from cell_quotient_helpers
    for correct rooted-Tutte composition. Empty aut groups → no orbit
    compression (correctness over performance for prototype).
    """
    tree = spec.cell_tree
    assert nx.is_tree(tree), "Input must be a tree"
    leaves = [n for n in tree.nodes() if tree.degree(n) == 1]
    assert len(leaves) == 2, (
        f"compute_tree_dp_simple supports only linear path (2 leaves); got {len(leaves)} leaves"
    )

    leaf_a = min(leaves)
    leaf_b = max(leaves)
    order = nx.shortest_path(tree, leaf_a, leaf_b)
    n_cells = len(order)
    assert n_cells >= 2

    cell_template = spec.cell_template
    junction_template = spec.junction_template

    # Allocate per-cell anchor positions for each (cell, neighbor) pair.
    # pos[cell_idx][neighbor_idx] = list of position labels.
    # CRITICAL: when the SAME underlying cell-template vertex appears in
    # multiple neighbor groups (anchor sharing), allocate the SAME position
    # label. This ensures the partition state correctly tracks shared
    # anchors across multiple junctions.
    pos: Dict[int, Dict[int, List[int]]] = {}
    for cell_idx in tree.nodes():
        base = 10000 * (cell_idx + 1)
        pos[cell_idx] = {}
        groups = spec.cell_anchor_groups.get(cell_idx, {})
        # Collect all distinct cell-template vertices used by ANY neighbor
        # at this cell, in deterministic order
        all_vertices_used = set()
        for nbr_anchors in groups.values():
            all_vertices_used.update(nbr_anchors)
        # Assign each unique vertex a position label
        vertex_to_pos: Dict[int, int] = {}
        for i, v in enumerate(sorted(all_vertices_used)):
            vertex_to_pos[v] = base + i
        # Map each neighbor's anchors to the per-vertex position labels
        for nbr in sorted(groups.keys()):
            anchors = groups[nbr]
            pos[cell_idx][nbr] = [vertex_to_pos[a] for a in anchors]

    # === Initialize state with cell 0 (leaf_a) ===
    cell_0 = order[0]
    cell_1 = order[1]
    cell_0_outward_anchors = spec.cell_anchor_groups[cell_0][cell_1]
    cell_0_outward_pos = pos[cell_0][cell_1]

    # T_rooted of cell 0 with its outward anchors as boundary
    T_cell0 = t_rooted_cached(cell_template, cell_0_outward_anchors)
    label_map_0 = {a: p for a, p in zip(cell_0_outward_anchors, cell_0_outward_pos)}
    state_partition = relabel_partition_dict(T_cell0, label_map_0)

    # No aut compression: each partition is its own orbit
    state_orbit_T, state_orbit_partitions = aut_compress_t_rooted(
        state_partition, []
    )

    # State boundary tracking
    state_open_pos = list(cell_0_outward_pos)  # currently-open boundary positions
    total_div = 0
    junction_c_J = components_touching(
        junction_template, list(spec.junction_anchors_A)
    )

    # === Process each subsequent cell ===
    for step in range(1, n_cells):
        cur_cell = order[step]
        prev_cell = order[step - 1]
        next_cell = order[step + 1] if step + 1 < n_cells else None

        # --- Junction step: convolve state with junction at the shared boundary ---
        # state's open boundary == prev_cell's outward to cur_cell == junction's A side
        prev_outward_pos = pos[prev_cell][cur_cell]   # state's current open boundary
        cur_inward_pos = pos[cur_cell][prev_cell]     # junction's B side

        # Build junction T_rooted with positions
        junction_anchor_list = list(spec.junction_anchors_A) + list(spec.junction_anchors_B)
        junction_pos_list = list(prev_outward_pos) + list(cur_inward_pos)
        junction_label_map = {a: p for a, p in zip(junction_anchor_list, junction_pos_list)}
        T_junction = t_rooted_cached(junction_template, junction_anchor_list)
        T_junction_pos = relabel_partition_dict(T_junction, junction_label_map)

        # No aut compression
        junction_orbit_T, junction_orbit_partitions = aut_compress_t_rooted(
            T_junction_pos, []
        )

        # Output boundary after junction step: state extras (none for path) + cur_inward
        # For our case (no persistent extras), out_extra = [], shared = prev_outward, extra (junction) = cur_inward
        out_anchors = list(cur_inward_pos)  # state_extra + extra (no state_extra here)
        out_orbit_partitions = enumerate_partitions_per_orbit(out_anchors, [])
        out_orbit_sizes = {ok: len(parts) for ok, parts in out_orbit_partitions.items()}

        M_j = precompute_M_table(
            state_orbit_partitions, junction_orbit_partitions,
            shared_boundary=list(prev_outward_pos),
            extra_boundary=list(cur_inward_pos),
            out_aut_group=[],
            state_extra_boundary=[],
        )
        state_orbit_T = orbit_convolve(
            state_orbit_T, junction_orbit_T, M_j, out_orbit_sizes,
        )
        total_div += len(prev_outward_pos) - junction_c_J
        state_orbit_partitions = out_orbit_partitions
        state_open_pos = list(cur_inward_pos)

        # --- Cell step: convolve state with cur_cell's T_rooted ---
        if next_cell is not None:
            cur_outward_pos = pos[cur_cell][next_cell]
            cur_anchor_list = (list(spec.cell_anchor_groups[cur_cell][prev_cell])
                               + list(spec.cell_anchor_groups[cur_cell][next_cell]))
            cur_pos_list = list(cur_inward_pos) + list(cur_outward_pos)
        else:
            # Leaf cell: only inward anchors
            cur_outward_pos = []
            cur_anchor_list = list(spec.cell_anchor_groups[cur_cell][prev_cell])
            cur_pos_list = list(cur_inward_pos)

        T_cur = t_rooted_cached(cell_template, cur_anchor_list)
        cur_label_map = {a: p for a, p in zip(cur_anchor_list, cur_pos_list)}
        T_cur_pos = relabel_partition_dict(T_cur, cur_label_map)
        cell_orbit_T, cell_orbit_partitions = aut_compress_t_rooted(T_cur_pos, [])

        out_anchors2 = list(cur_outward_pos)  # output boundary after cell step
        out_orbit_partitions2 = enumerate_partitions_per_orbit(out_anchors2, [])
        out_orbit_sizes2 = {ok: len(parts) for ok, parts in out_orbit_partitions2.items()}

        M_c = precompute_M_table(
            state_orbit_partitions, cell_orbit_partitions,
            shared_boundary=list(cur_inward_pos),
            extra_boundary=list(cur_outward_pos),
            out_aut_group=[],
            state_extra_boundary=[],
        )
        state_orbit_T = orbit_convolve(
            state_orbit_T, cell_orbit_T, M_c, out_orbit_sizes2,
        )
        # cell convolution divisor: |cur_inward| - c_cell
        c_cell = components_touching(
            cell_template, list(spec.cell_anchor_groups[cur_cell][prev_cell])
        )
        total_div += len(cur_inward_pos) - c_cell
        state_orbit_partitions = out_orbit_partitions2
        state_open_pos = list(cur_outward_pos)

    # === Final: state should be over empty boundary (= last cell's empty outward) ===
    # Sum across all partition entries (should be a single entry for empty boundary).
    final_poly = TuttePolynomial.zero()
    for P, val in state_orbit_T.items():
        final_poly = final_poly + val

    # Apply final divisor
    if total_div > 0:
        final_poly = divide_by_x_minus_1_power(final_poly, total_div)
    return final_poly


def _try_chain_recurrence_mod(
    spec: "CellTreeSpec", x_val: int, y_val: int, p: int, path_order: list
) -> "Optional[int]":
    """Attempt to evaluate T(graph) mod p via the chain recurrence
    algebra. Returns None if the chain framework doesn't apply.

    Requires:
    - Linear path topology (already enforced by caller)
    - At least one interior cell (n_cells >= 3) — leaves don't have
      two-sided anchor groups
    """
    n_cells = len(path_order)
    if n_cells < 3:
        return None

    interior_cell = path_order[1]
    interior_groups = spec.cell_anchor_groups.get(interior_cell, {})
    nbr_ids = sorted(interior_groups.keys())
    if len(nbr_ids) != 2:
        return None
    left_nbr, right_nbr = nbr_ids
    left_anchors = list(interior_groups[left_nbr])
    right_anchors = list(interior_groups[right_nbr])

    # Chain framework requires DISJOINT anchor groups. K_3 chain with
    # anchors=[0,1] shares vertices between left and right, which the
    # framework can't handle (needs cross-cell identifications instead).
    if set(left_anchors) & set(right_anchors):
        return None

    chain_cell_anchor_groups = {0: left_anchors, 1: right_anchors}

    from tutte.roots.chain_recurrence import compute_chain_recurrence_mod
    try:
        T_mod, _cache = compute_chain_recurrence_mod(
            cell_template=spec.cell_template,
            cell_anchor_groups=chain_cell_anchor_groups,
            connector_template=spec.junction_template,
            junction_anchors_A=list(spec.junction_anchors_A),
            junction_anchors_B=list(spec.junction_anchors_B),
            left_anchor_group=0,
            right_anchor_group=1,
            n_cells=n_cells,
            x_val=x_val,
            y_val=y_val,
            p=p,
        )
        return T_mod
    except Exception:
        return None


def compute_tree_dp_simple_mod(
    spec: CellTreeSpec, x_val: int, y_val: int, p: int,
) -> int:
    """Modular variant of `compute_tree_dp_simple` —

    Evaluates `T(G; x_val, y_val) mod p` for a linear-path tree-quotient
    cell-decomposable graph WITHOUT materializing the full polynomial.

    Same DP structure as the polynomial version, but each per-cell /
    per-junction T_rooted polynomial is **evaluated at (x_val, y_val)
    mod p before convolution**. The convolution itself uses the modular
    M-table machinery (`precompute_M_and_convolve_streaming_mod`).

    This is the first non-Chimera entrypoint into the modular DP path
    — any graph whose cell-quotient is a linear path can be evaluated
    point-by-point via this function.

    Limitations:
    - Linear path only (2 leaves in cell_tree). Branching trees would
      need a modular port of `compute_tree_dp_recursive`.
    - No orbit compression (empty aut group throughout). Cells with
      non-trivial automorphism could be sped up by porting
      `aut_compress_t_rooted_per_cell`.
    """
    tree = spec.cell_tree
    assert nx.is_tree(tree), "Input must be a tree"
    leaves = [n for n in tree.nodes() if tree.degree(n) == 1]
    assert len(leaves) == 2, (
        f"compute_tree_dp_simple_mod supports linear path only; got {len(leaves)} leaves"
    )

    leaf_a = min(leaves)
    leaf_b = max(leaves)
    order = nx.shortest_path(tree, leaf_a, leaf_b)
    n_cells = len(order)
    assert n_cells >= 2

    # Chain recurrence fast path: for modular evaluation, runs in O(r²)
    # integer ops per cell where r = n_orbits.
    try:
        result = _try_chain_recurrence_mod(spec, x_val, y_val, p, order)
        if result is not None:
            return result
    except Exception:
        pass  # Fall through to standard path DP

    cell_template = spec.cell_template
    junction_template = spec.junction_template

    # Allocate per-cell anchor positions (matches polynomial version).
    pos: Dict[int, Dict[int, List[int]]] = {}
    for cell_idx in tree.nodes():
        base = 10000 * (cell_idx + 1)
        pos[cell_idx] = {}
        groups = spec.cell_anchor_groups.get(cell_idx, {})
        all_vertices_used = set()
        for nbr_anchors in groups.values():
            all_vertices_used.update(nbr_anchors)
        vertex_to_pos: Dict[int, int] = {}
        for i, v in enumerate(sorted(all_vertices_used)):
            vertex_to_pos[v] = base + i
        for nbr in sorted(groups.keys()):
            anchors = groups[nbr]
            pos[cell_idx][nbr] = [vertex_to_pos[a] for a in anchors]

    # === Initialize state with cell 0 ===
    cell_0 = order[0]
    cell_1 = order[1]
    cell_0_outward_anchors = spec.cell_anchor_groups[cell_0][cell_1]
    cell_0_outward_pos = pos[cell_0][cell_1]

    T_cell0 = t_rooted_cached(cell_template, cell_0_outward_anchors)
    label_map_0 = {a: q for a, q in zip(cell_0_outward_anchors, cell_0_outward_pos)}
    state_partition_poly = relabel_partition_dict(T_cell0, label_map_0)

    # Evaluate each partition's polynomial at (x_val, y_val) mod p ONCE.
    state_orbit_T_mod: Dict[Tuple, int] = {
        P: T.evaluate_mod(x_val, y_val, p)
        for P, T in state_partition_poly.items()
    }
    state_orbit_partitions: Dict[Tuple, List[Tuple]] = {P: [P] for P in state_orbit_T_mod}

    total_div = 0
    junction_c_J = components_touching(
        junction_template, list(spec.junction_anchors_A)
    )

    # === Process each subsequent cell ===
    for step in range(1, n_cells):
        cur_cell = order[step]
        prev_cell = order[step - 1]
        next_cell = order[step + 1] if step + 1 < n_cells else None

        prev_outward_pos = pos[prev_cell][cur_cell]
        cur_inward_pos = pos[cur_cell][prev_cell]

        # --- Junction step ---
        junction_anchor_list = list(spec.junction_anchors_A) + list(spec.junction_anchors_B)
        junction_pos_list = list(prev_outward_pos) + list(cur_inward_pos)
        junction_label_map = {a: q for a, q in zip(junction_anchor_list, junction_pos_list)}
        T_junction = t_rooted_cached(junction_template, junction_anchor_list)
        T_junction_pos = relabel_partition_dict(T_junction, junction_label_map)
        junction_orbit_T_mod: Dict[Tuple, int] = {
            P: T.evaluate_mod(x_val, y_val, p)
            for P, T in T_junction_pos.items()
        }
        junction_orbit_partitions: Dict[Tuple, List[Tuple]] = {
            P: [P] for P in junction_orbit_T_mod
        }

        out_orbit_sizes_junc: Dict[Tuple, int] = {}
        state_orbit_T_mod = precompute_M_and_convolve_streaming_mod(
            state_orbit_T_mod, state_orbit_partitions,
            junction_orbit_T_mod, junction_orbit_partitions,
            out_orbit_sizes_junc,
            p=p, x_val=x_val, y_val=y_val,
            shared_boundary=list(prev_outward_pos),
            extra_boundary=list(cur_inward_pos),
            out_aut_group=[],
            state_extra_boundary=[],
        )
        total_div += len(prev_outward_pos) - junction_c_J
        state_orbit_partitions = {P: [P] for P in state_orbit_T_mod}

        # --- Cell step ---
        if next_cell is not None:
            cur_outward_pos = pos[cur_cell][next_cell]
            cur_anchor_list = (list(spec.cell_anchor_groups[cur_cell][prev_cell])
                               + list(spec.cell_anchor_groups[cur_cell][next_cell]))
            cur_pos_list = list(cur_inward_pos) + list(cur_outward_pos)
        else:
            cur_outward_pos = []
            cur_anchor_list = list(spec.cell_anchor_groups[cur_cell][prev_cell])
            cur_pos_list = list(cur_inward_pos)

        T_cur = t_rooted_cached(cell_template, cur_anchor_list)
        cur_label_map = {a: q for a, q in zip(cur_anchor_list, cur_pos_list)}
        T_cur_pos = relabel_partition_dict(T_cur, cur_label_map)
        cell_orbit_T_mod: Dict[Tuple, int] = {
            P: T.evaluate_mod(x_val, y_val, p)
            for P, T in T_cur_pos.items()
        }
        cell_orbit_partitions: Dict[Tuple, List[Tuple]] = {
            P: [P] for P in cell_orbit_T_mod
        }

        out_orbit_sizes_cell: Dict[Tuple, int] = {}
        state_orbit_T_mod = precompute_M_and_convolve_streaming_mod(
            state_orbit_T_mod, state_orbit_partitions,
            cell_orbit_T_mod, cell_orbit_partitions,
            out_orbit_sizes_cell,
            p=p, x_val=x_val, y_val=y_val,
            shared_boundary=list(cur_inward_pos),
            extra_boundary=list(cur_outward_pos),
            out_aut_group=[],
            state_extra_boundary=[],
        )
        c_cell = components_touching(
            cell_template, list(spec.cell_anchor_groups[cur_cell][prev_cell])
        )
        total_div += len(cur_inward_pos) - c_cell
        state_orbit_partitions = {P: [P] for P in state_orbit_T_mod}

    # === Final: sum across remaining state ===
    total = sum(state_orbit_T_mod.values()) % p

    if total_div > 0:
        base = (x_val - 1) % p
        if base == 0:
            raise ValueError(
                f"x_val ({x_val}) ≡ 1 mod {p}; cannot divide by (x-1)^{total_div}"
            )
        denom = pow(base, total_div, p)
        denom_inv = pow(denom, p - 2, p)
        total = total * denom_inv % p

    return total


# =============================================================================
# Recursive tree DP — handles branching
# =============================================================================


def _marginalize_state(
    state_orbit_T: Dict[Tuple, TuttePolynomial],
    state_orbit_partitions: Dict[Tuple, List[Tuple]],
    new_boundary: List[int],
):
    """Marginalize state by restricting partitions to `new_boundary`.

    new_state[P_restricted] = Σ_{P_full: restrict(P_full, new_boundary) = P_restricted}
                              state_orbit_T[orbit(P_full)] / orbit_size

    Returns (new_state_orbit_T, new_state_orbit_partitions).
    """
    new_T: Dict[Tuple, TuttePolynomial] = {}
    new_partitions: Dict[Tuple, List[Tuple]] = {}
    new_set = set(new_boundary)

    for orbit, val in state_orbit_T.items():
        partitions_in_orbit = state_orbit_partitions.get(orbit, [orbit])
        for P_full in partitions_in_orbit:
            P_restricted = restrict_partition(P_full, new_boundary)
            new_partitions.setdefault(P_restricted, []).append(P_full)
            if P_restricted in new_T:
                new_T[P_restricted] = new_T[P_restricted] + val
            else:
                new_T[P_restricted] = val

    # Re-canonicalize as identity-aut orbit (each partition its own orbit)
    new_partitions_norm = {p: [p] for p in new_T}
    return new_T, new_partitions_norm


def _group_positions_by_cell(positions: List[int]) -> List[List[int]]:
    """Group positions by cell-of-origin (decoded from position label).

    Per `_allocate_tree_positions`, position label = 10000 * (cell_idx + 1) + offset.
    So cell_idx = position // 10000 - 1.
    """
    by_cell: Dict[int, List[int]] = {}
    for p in positions:
        cell_idx = p // 10000 - 1
        by_cell.setdefault(cell_idx, []).append(p)
    return [sorted(by_cell[ci]) for ci in sorted(by_cell.keys())]


def _allocate_tree_positions(spec: CellTreeSpec) -> Dict[int, Dict[int, List[int]]]:
    """Allocate positions per (cell, neighbor) with shared-anchor support.

    Within a cell: when the same cell-template vertex appears in
    multiple neighbor groups, allocate the SAME position label.

    Across cells: when ``spec.cross_cell_identifications`` declares
    that ``(cell_i, anchor_i)`` and ``(cell_j, anchor_j)`` are
    physically the same vertex, allocate the SAME position label
    across cells. Used by the hybrid cycle-close path to represent
    contracted junction endpoints.

    Implementation: union-find over (cell, template_anchor) pairs.
    Each connected component gets one allocated position. Component
    representatives are sorted by (smallest cell_idx, smallest
    template_anchor) for determinism, and positions are assigned in
    cell-then-anchor order.
    """
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find(x: Tuple[int, int]) -> Tuple[int, int]:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: Tuple[int, int], b: Tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    used_pairs: List[Tuple[int, int]] = []
    seen_pairs = set()
    for cell_idx in spec.cell_tree.nodes():
        groups = spec.cell_anchor_groups.get(cell_idx, {})
        for nbr_anchors in groups.values():
            for a in nbr_anchors:
                pair = (cell_idx, a)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    used_pairs.append(pair)
                    parent.setdefault(pair, pair)
        # Also include extra_open_anchors so they get position labels
        # even when they're not in any junction's anchor list.
        for a in spec.extra_open_anchors.get(cell_idx, []):
            pair = (cell_idx, a)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                used_pairs.append(pair)
                parent.setdefault(pair, pair)
    for (ci, ai, cj, aj) in spec.cross_cell_identifications:
        union((ci, ai), (cj, aj))

    component_to_pos: Dict[Tuple[int, int], int] = {}
    next_pos = 10000
    for pair in sorted(used_pairs):
        rep = find(pair)
        if rep not in component_to_pos:
            component_to_pos[rep] = next_pos
            next_pos += 1

    pos: Dict[int, Dict[int, List[int]]] = {}
    for cell_idx in spec.cell_tree.nodes():
        pos[cell_idx] = {}
        groups = spec.cell_anchor_groups.get(cell_idx, {})
        for nbr in sorted(groups.keys()):
            anchors = groups[nbr]
            pos[cell_idx][nbr] = [
                component_to_pos[find((cell_idx, a))] for a in anchors
            ]
    return pos


def _allocate_extras_positions(spec: CellTreeSpec) -> Dict[int, List[int]]:
    """Return per-cell list of position labels for extra_open_anchors,
    using the same allocation scheme as ``_allocate_tree_positions``."""
    pos = _allocate_tree_positions(spec)
    # Reconstruct the position lookup by replaying union-find. Cheaper
    # than threading the lookup through; both runs are O(n) on used pairs.
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find(x: Tuple[int, int]) -> Tuple[int, int]:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: Tuple[int, int], b: Tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    used_pairs: List[Tuple[int, int]] = []
    seen_pairs = set()
    for cell_idx in spec.cell_tree.nodes():
        groups = spec.cell_anchor_groups.get(cell_idx, {})
        for nbr_anchors in groups.values():
            for a in nbr_anchors:
                pair = (cell_idx, a)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    used_pairs.append(pair)
                    parent.setdefault(pair, pair)
        for a in spec.extra_open_anchors.get(cell_idx, []):
            pair = (cell_idx, a)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                used_pairs.append(pair)
                parent.setdefault(pair, pair)
    for (ci, ai, cj, aj) in spec.cross_cell_identifications:
        union((ci, ai), (cj, aj))
    component_to_pos: Dict[Tuple[int, int], int] = {}
    next_pos = 10000
    for pair in sorted(used_pairs):
        rep = find(pair)
        if rep not in component_to_pos:
            component_to_pos[rep] = next_pos
            next_pos += 1
    extras_pos: Dict[int, List[int]] = {}
    for cell_idx in spec.cell_tree.nodes():
        extras_pos[cell_idx] = [
            component_to_pos[find((cell_idx, a))]
            for a in spec.extra_open_anchors.get(cell_idx, [])
        ]
    return extras_pos


def _compute_aut_orbits_on_positions(
    cell_template: Graph,
    template_anchors: List[int],
    positions: List[int],
    preserve_anchor_sets: Optional[List[List[int]]] = None,
) -> List[List[int]]:
    """Compute Aut(cell)-orbits on boundary positions (per-cell groups).

    Each group = one orbit of the cell's aut group acting on the
    template anchors, mapped to corresponding positions. Groups are
    DISJOINT (each position in exactly one group), which is the
    precondition for per_cell_canonical_key correctness.

    For K_n: cell aut S_n maps all anchors to all → ONE orbit.
    For K_{a,b}: cell aut S_a × S_b × Z_2 (side swap) → ONE orbit
        without preserve_anchor_sets, TWO orbits when preserve_anchor_sets
        are the bipartition sides (since side swap then filtered out).

    Args:
        preserve_anchor_sets: optional list of template-anchor sets that
            must each be preserved by the aut (mapped to self as a SET).
            Pass per-neighbor anchor sets to ensure aut respects the
            operational structure (which children connect to which
            anchors).
    """
    cell_auts = compute_cell_aut(cell_template)
    anchor_set = set(template_anchors)
    preserve_sets = (
        [set(s) for s in preserve_anchor_sets]
        if preserve_anchor_sets is not None else None
    )
    valid_auts: List[Dict[int, int]] = []
    for aut in cell_auts:
        if not all(aut.get(v) in anchor_set for v in template_anchors):
            continue
        if preserve_sets is not None:
            ok = True
            for s in preserve_sets:
                if not all(aut.get(v) in s for v in s):
                    ok = False
                    break
            if not ok:
                continue
        valid_auts.append(aut)
    template_to_pos = {a: p for a, p in zip(template_anchors, positions)}

    pos_to_orbit: Dict[int, int] = {}
    next_id = 0
    for anchor in template_anchors:
        p = template_to_pos[anchor]
        if p in pos_to_orbit:
            continue
        oid = next_id
        next_id += 1
        # BFS over anchors reachable from `anchor` via valid auts
        seen_anchors = set()
        queue = [anchor]
        while queue:
            curr = queue.pop()
            if curr in seen_anchors:
                continue
            seen_anchors.add(curr)
            curr_pos = template_to_pos.get(curr)
            if curr_pos is not None:
                pos_to_orbit[curr_pos] = oid
            for aut in valid_auts:
                next_a = aut.get(curr)
                if next_a is not None and next_a not in seen_anchors:
                    queue.append(next_a)

    orbits: Dict[int, List[int]] = {}
    for p, oid in pos_to_orbit.items():
        orbits.setdefault(oid, []).append(p)
    return [sorted(orbits[oid]) for oid in sorted(orbits.keys())]


def _initial_cell_groups(
    cell_template: Graph,
    cell_template_anchors: List[int],
    cell_positions_for_anchors: List[int],
    preserve_anchor_sets: Optional[List[List[int]]] = None,
) -> List[List[int]]:
    """state_cell_groups for the initial state of a cell.

    Each group = orbit of cell_template's aut acting on the cell's
    boundary positions. This ensures groups are DISJOINT (the per_cell
    canonical key precondition).

    For K_n cells (full S_n), gives ONE group covering all positions.
    For K_{a,b}, gives one group per bipartite side WHEN
    preserve_anchor_sets includes both sides (filtering out side swap).

    Args:
        preserve_anchor_sets: per-neighbor anchor sets that must be
            preserved (mapped to self as set). Critical for operationally
            distinct anchor groups (e.g., K_{4,4} cells with neighbors
            using A-side vs B-side).
    """
    return _compute_aut_orbits_on_positions(
        cell_template, cell_template_anchors, cell_positions_for_anchors,
        preserve_anchor_sets=preserve_anchor_sets,
    )


def _state_groups_after_junction(
    state_cell_groups: List[List[int]],
    cell_outward_pos: List[int],
    child_inward_pos: List[int],
    keep_shared: bool,
) -> List[List[int]]:
    """Update state_cell_groups after junction step.

    Junction step: convolve state with junction T_rooted at the
    boundary `cell_outward_pos`. If keep_shared, those positions remain
    in state; otherwise they are consumed (subset of some group(s) gets
    removed). The child's inward positions become a new group.
    """
    cell_outward_set = set(cell_outward_pos)
    new_groups: List[List[int]] = []
    if keep_shared:
        for g in state_cell_groups:
            new_groups.append(list(g))
    else:
        for g in state_cell_groups:
            remaining = [p for p in g if p not in cell_outward_set]
            if remaining:
                new_groups.append(remaining)
    new_groups.append(list(child_inward_pos))
    return new_groups


def _state_groups_after_cell_merge(
    state_cell_groups: List[List[int]],
    child_inward_pos: List[int],
) -> List[List[int]]:
    """Update state_cell_groups after cell-merge step.

    Cell-merge consumes the child_inward_pos (shared boundary) via
    convolution. Remove those positions from any group containing them.
    """
    child_inward_set = set(child_inward_pos)
    new_groups: List[List[int]] = []
    for g in state_cell_groups:
        remaining = [p for p in g if p not in child_inward_set]
        if remaining:
            new_groups.append(remaining)
    return new_groups


def _state_groups_after_marginalize(
    state_cell_groups: List[List[int]],
    live_positions: List[int],
) -> List[List[int]]:
    """Update state_cell_groups after marginalization step.

    Marginalization restricts state to `live_positions`. Drop dead
    positions from each group; drop empty groups.
    """
    live_set = set(live_positions)
    new_groups: List[List[int]] = []
    for g in state_cell_groups:
        live_in_group = [p for p in g if p in live_set]
        if not live_in_group:
            continue
        new_groups.append(live_in_group)
    return new_groups


def _expand_per_cell_state(
    state_orbit_T: Dict[Tuple, TuttePolynomial],
    state_cell_groups: List[List[int]],
):
    """Expand per-cell compressed state to uncompressed.

    For each canonical orbit key, enumerate all member partitions by
    applying S_n^N permutations within each per-cell group. Each member
    gets the same T value (orbit invariance).

    Returns (state_T_dict, state_partitions_dict) where each is keyed
    by PARTITION (single-rep orbit per partition).
    """
    from itertools import permutations, product
    state_T_uncompressed: Dict[Tuple, TuttePolynomial] = {}
    state_partitions_uncompressed: Dict[Tuple, List[Tuple]] = {}
    for canonical_key, val in state_orbit_T.items():
        rep = per_cell_orbit_rep(canonical_key, state_cell_groups)
        per_group_perms = [list(permutations(g)) for g in state_cell_groups]
        seen = set()
        for perm_combo in product(*per_group_perms):
            relabel_map = {}
            for orig_g, new_perm in zip(state_cell_groups, perm_combo):
                for orig, new in zip(orig_g, new_perm):
                    relabel_map[orig] = new
            new_blocks = []
            for block in rep:
                new_block = tuple(sorted(relabel_map.get(v, v) for v in block))
                new_blocks.append(new_block)
            new_partition = tuple(sorted(new_blocks))
            if new_partition in seen:
                continue
            seen.add(new_partition)
            state_T_uncompressed[new_partition] = val
            state_partitions_uncompressed[new_partition] = [new_partition]
    return state_T_uncompressed, state_partitions_uncompressed


def _marginalize_state_per_cell(
    state_orbit_T: Dict[Tuple, TuttePolynomial],
    state_cell_groups_old: List[List[int]],
    new_boundary: List[int],
    state_cell_groups_new: List[List[int]],
):
    """Marginalize per-cell-compressed state to new_boundary.

    For each old orbit (canonical key), expand to representative partition,
    enumerate all members of the orbit (S_n^N over old groups), restrict
    each to new_boundary, re-canonicalize under new groups, aggregate.

    Returns new state_orbit_T as {canonical_key: TuttePolynomial}.

    Note: enumerating orbit members is expensive but rare in tree DP.
    """
    from itertools import permutations

    new_T: Dict[Tuple, TuttePolynomial] = {}
    new_set = set(new_boundary)

    for canonical_old, val in state_orbit_T.items():
        rep = per_cell_orbit_rep(canonical_old, state_cell_groups_old)
        # Enumerate all partitions in the orbit by applying all S_n^N permutations.
        # For each cell group, generate permutations of its positions.
        # Each permutation maps rep's positions to new positions per group.
        # The orbit member = rep with those position-relabels.
        per_group_perms = [list(permutations(g)) for g in state_cell_groups_old]
        from itertools import product
        seen_members = set()
        for perm_combo in product(*per_group_perms):
            # Build relabel map: for each group, map original → permuted.
            relabel_map = {}
            for orig_g, new_perm in zip(state_cell_groups_old, perm_combo):
                for orig, new in zip(orig_g, new_perm):
                    relabel_map[orig] = new
            # Apply relabel to rep partition.
            new_partition_blocks = []
            for block in rep:
                new_block = tuple(sorted(relabel_map.get(v, v) for v in block))
                new_partition_blocks.append(new_block)
            new_partition = tuple(sorted(new_partition_blocks))
            if new_partition in seen_members:
                continue
            seen_members.add(new_partition)
            # Restrict to new_boundary.
            P_restricted = restrict_partition(new_partition, new_boundary)
            # Re-canonicalize under new groups.
            canonical_new = per_cell_canonical_key(P_restricted, state_cell_groups_new)
            if canonical_new in new_T:
                new_T[canonical_new] = new_T[canonical_new] + val
            else:
                new_T[canonical_new] = val

    return new_T


def _merge_with_junction_and_identification(
    state_T: Dict[Tuple, TuttePolynomial],
    state_open_pos: List[int],
    child_T: Dict[Tuple, TuttePolynomial],
    child_open_pos: List[int],
    junction_shared: List[int],
    identification_shared: List[int],
    out_positions: List[int],
):
    """Custom cell-merge that distinguishes JUNCTION-shared positions
    (with `(xy-1)^d` deficit accounting) from IDENTIFICATION-shared
    positions (pure vertex identification, no deficit).

    Used by the hybrid cycle-close path when child's subtree carries
    cross-cell-identified anchors that meet their counterpart in
    state at this common ancestor. The two share-types coexist: the
    junction-shared positions arise from cell_anchor_groups
    (junction edges between cells), while identification-shared
    positions arise from `cross_cell_identifications` (pure vertex
    merging with no edges between).

    Algorithm:
    - For each (P_state, P_child) pair:
      - Compute joint = `join_partitions(P_state, P_child, full_universe)`.
      - Compute deficit `d` over JUNCTION-shared only (identification
        positions are not edge-shared so don't contribute to deficit).
      - If `d < 0` (incompatible junction merging): skip pair.
      - Restrict joint to `out_positions` for output partition.
      - Accumulate `state_T · child_T · (xy-1)^d` to output.

    Returns: new_T as Dict[partition_over_out_positions, TuttePolynomial].
    """
    full_universe = sorted(set(state_open_pos) | set(child_open_pos))
    xy_minus_1 = TuttePolynomial.from_coefficients({
        (1, 1): 1, (1, 0): -1, (0, 1): -1, (0, 0): 1,
    })
    junction_shared_list = list(junction_shared)
    new_T_raw: Dict[Tuple, TuttePolynomial] = {}
    for P_state, T_state in state_T.items():
        P_state_S_junc = (
            restrict_partition(P_state, junction_shared_list)
            if junction_shared_list else ()
        )
        for P_child, T_child in child_T.items():
            P_child_S_junc = (
                restrict_partition(P_child, junction_shared_list)
                if junction_shared_list else ()
            )
            d = (delta(P_state_S_junc, P_child_S_junc, junction_shared_list)
                  if junction_shared_list else 0)
            if d < 0:
                continue
            joint = join_partitions(P_state, P_child, full_universe)
            P_out = (restrict_partition(joint, out_positions)
                      if out_positions else ())
            contrib = T_state * T_child
            for _ in range(d):
                contrib = contrib * xy_minus_1
            if P_out in new_T_raw:
                new_T_raw[P_out] = new_T_raw[P_out] + contrib
            else:
                new_T_raw[P_out] = contrib
    return new_T_raw


def _build_combined_aut(
    state_cell_groups: List[List[int]],
    shared_set: set,
    junction_aut_group: List[Dict[int, int]],
) -> List[Dict[int, int]]:
    """Combined aut group on output positions for keep_shared / fully-consumed.

    output_aut element = (σ_state_preserved on PRESERVED state cells) ×
    (σ_junc on consumed shared + child_inward).

    Lets us canonicalize output via the actual aut (state-preserved per-cell
    × junction-aut diagonal on shared+inward) instead of expanding state to
    fully-uncompressed BEFORE the junction step. The latter blows state up
    to Bell(N) before the convolution; the former keeps state per-cell
    compressed during M-table iteration.
    """
    from itertools import permutations, product
    preserved_groups = [g for g in state_cell_groups
                         if not (set(g) & shared_set)]
    if preserved_groups:
        per_group_perms = [list(permutations(g)) for g in preserved_groups]
        preserved_perms = []
        for combo in product(*per_group_perms):
            d: Dict[int, int] = {}
            for orig_g, perm in zip(preserved_groups, combo):
                for orig, new in zip(orig_g, perm):
                    d[orig] = new
            preserved_perms.append(d)
    else:
        preserved_perms = [{}]
    combined: List[Dict[int, int]] = []
    for sp in preserved_perms:
        for ja in junction_aut_group:
            combined_dict: Dict[int, int] = {}
            combined_dict.update(sp)
            combined_dict.update(ja)
            combined.append(combined_dict)
    return combined


def _expand_combined_aut_orbit_members(
    rep: Tuple[Tuple[int, ...], ...],
    cell_anchor_groups: List[List[int]],
):
    """Enumerate all distinct partitions in the per-cell orbit of rep.

    Used by combined-aut junction step to iterate ALL state members
    (per-cell expansion of rep_state) explicitly, since the standard
    M-table aut shortcut breaks when state has consumed cells whose
    aut moves shared positions.
    """
    from itertools import permutations, product
    per_group_perms = [list(permutations(g)) for g in cell_anchor_groups]
    seen = set()
    members = []
    for perm_combo in product(*per_group_perms):
        relabel_map: Dict[int, int] = {}
        for orig_g, new_perm in zip(cell_anchor_groups, perm_combo):
            for orig, new in zip(orig_g, new_perm):
                relabel_map[orig] = new
        new_blocks = []
        for block in rep:
            new_block = tuple(sorted(relabel_map.get(v, v) for v in block))
            new_blocks.append(new_block)
        new_partition = tuple(sorted(new_blocks))
        if new_partition not in seen:
            seen.add(new_partition)
            members.append(new_partition)
    return members


def _run_combined_aut_junction_step(
    state_orbit_T: Dict[Tuple, TuttePolynomial],
    state_orbit_partitions: Dict[Tuple, List[Tuple]],
    state_cell_groups: List[List[int]],
    junction_orbit_T: Dict[Tuple, TuttePolynomial],
    junction_orbit_partitions: Dict[Tuple, List[Tuple]],
    junction_aut: List[Dict[int, int]],
    cell_outward_pos: List[int],
    child_inward_pos: List[int],
    state_extra_pos: List[int],
    cell_outward_is_shared: bool,
    out_anchors: List[int],
):
    """Combined-aut junction step: iterate state members × junction members.

    Replaces the existing fallback (expand state to uncompressed BEFORE
    the junction step) with a path that keeps state per-cell during
    M-table iteration. After this step, state is expanded to fully
    uncompressed (each P own orbit) for subsequent steps.

    Validated correct for:
    - K_4 claw M_2 shared (keep_shared)
    - K_5 claw M_3 shared (fully-consumed + keep_shared)
    - K_{4,4} 3-cell mixed M_4 (fully-consumed without keep_shared)
    - K_{4,4} 5-cell Cm3-pattern M_2 (multi-junction with shared anchors)

    See `tutte/research/data/combined_aut_findings.md`.

    Returns: (new_state_orbit_T, new_state_orbit_partitions) — both
    keyed by individual partitions (each P its own orbit, len=1).
    """
    from collections import defaultdict
    cell_outward_set = set(cell_outward_pos)
    combined_aut = _build_combined_aut(
        state_cell_groups, cell_outward_set, junction_aut,
    )
    full_universe = (list(state_extra_pos) + list(cell_outward_pos)
                      + list(child_inward_pos))
    max_d = len(cell_outward_pos)
    xy_minus_1_dict = {(1, 1): 1, (1, 0): -1, (0, 1): -1, (0, 0): 1}
    xy_powers_dict = [{(0, 0): 1}]
    for k in range(1, max_d + 1):
        prev = xy_powers_dict[-1]
        new = defaultdict(int)
        for (i1, j1), c1 in prev.items():
            for (i2, j2), c2 in xy_minus_1_dict.items():
                new[(i1 + i2, j1 + j2)] += c1 * c2
        xy_powers_dict.append(dict(new))
    M_combined: Dict[Tuple, Dict[Tuple[int, int], int]] = defaultdict(dict)
    for O_state, ps_list in state_orbit_partitions.items():
        rep_state = ps_list[0]
        state_members = _expand_combined_aut_orbit_members(
            rep_state, state_cell_groups,
        )
        for O_junc, junc_pj_list in junction_orbit_partitions.items():
            for P_state in state_members:
                P_state_ext_list = (list(P_state)
                                     + [(v,) for v in child_inward_pos])
                P_state_ext = tuple(sorted(P_state_ext_list))
                P_state_S = restrict_partition(P_state, cell_outward_pos)
                for P_junc in junc_pj_list:
                    P_junc_S = restrict_partition(P_junc, cell_outward_pos)
                    d = delta(P_state_S, P_junc_S, cell_outward_pos)
                    if d < 0:
                        continue
                    P_junc_ext_list = (list(P_junc)
                                        + [(v,) for v in state_extra_pos])
                    P_junc_ext = tuple(sorted(P_junc_ext_list))
                    joint = join_partitions(
                        P_state_ext, P_junc_ext, full_universe,
                    )
                    P_out = restrict_partition(joint, out_anchors)
                    O_out = (canonical_partition(P_out, combined_aut)
                              if out_anchors else ())
                    target = M_combined[(O_state, O_junc, O_out)]
                    for k_pow, v_coeff in xy_powers_dict[d].items():
                        target[k_pow] = target.get(k_pow, 0) + v_coeff
    M_j: Dict[Tuple, TuttePolynomial] = {}
    for key, val_dict in M_combined.items():
        nonzero = {k: v for k, v in val_dict.items() if v != 0}
        M_j[key] = TuttePolynomial.from_coefficients(nonzero)
    out_orbit_partitions = enumerate_partitions_per_orbit(out_anchors, combined_aut)
    out_orbit_sizes = {ok: len(parts)
                        for ok, parts in out_orbit_partitions.items()}
    new_state_orbit_T = orbit_convolve(
        state_orbit_T, junction_orbit_T, M_j, out_orbit_sizes,
    )
    # Expand to fully-uncompressed: each P own orbit, len(ps_list) = 1.
    expanded_T: Dict[Tuple, TuttePolynomial] = {}
    expanded_partitions: Dict[Tuple, List[Tuple]] = {}
    for Oo, parts_list in out_orbit_partitions.items():
        val = new_state_orbit_T.get(Oo)
        if val is None:
            continue
        for P in parts_list:
            expanded_T[P] = val
            expanded_partitions[P] = [P]
    return expanded_T, expanded_partitions


def compute_tree_dp_recursive(
    spec: CellTreeSpec,
    enable_per_cell_compression: bool = False,
) -> TuttePolynomial:
    """Tree DP via post-order recursion. Handles BRANCHING.

    For each cell, recursively compute T_rooted of the subtree rooted at
    each of its children. Then absorb each child's T_rooted into the
    cell's own T_rooted via junction + cell convolution.

    When `enable_per_cell_compression=True`, state is compressed by
    per_cell_canonical_key (S_n^N orbit on cell-anchor groups). One group
    per (cell, neighbor) pair, deduped by position. Required for
    K_{4,4}-cell graphs at M_4 boundary scale.

    Returns T(graph) as a single TuttePolynomial.
    """
    tree = spec.cell_tree
    assert nx.is_tree(tree)

    cell_template = spec.cell_template
    junction_template = spec.junction_template
    pos = _allocate_tree_positions(spec)
    extras_pos = _allocate_extras_positions(spec)
    junction_c_J = components_touching(
        junction_template, list(spec.junction_anchors_A)
    )

    # Cumulative divisor across the entire recursion
    total_div = [0]

    def dp_subtree(cell_idx: int, parent_cell_idx: Optional[int]):
        """Post-order DP. Returns (state_orbit_T, state_orbit_partitions,
        state_open_pos, state_cell_groups) where state is T_rooted indexed
        by partition over cell_idx's PARENT-facing anchor positions (or
        empty boundary if cell_idx is the root).

        state_cell_groups is meaningful only when
        enable_per_cell_compression=True; otherwise it's None.

        CONTRACT: returned state_open_pos = pos[cell_idx][parent_cell_idx]
        (or empty for root). Parent-facing positions are PRESERVED (not
        consumed) throughout the recursion.
        """
        all_neighbors = set(tree.neighbors(cell_idx))
        if parent_cell_idx is not None:
            children = sorted(all_neighbors - {parent_cell_idx})
        else:
            children = sorted(all_neighbors)

        # Build the cell's full anchor list (template-level vertices used by
        # any neighbor: parent + children + extras)
        anchor_lists_per_neighbor: Dict[int, List[int]] = {}
        if parent_cell_idx is not None:
            anchor_lists_per_neighbor[parent_cell_idx] = spec.cell_anchor_groups[cell_idx][parent_cell_idx]
        for child in children:
            anchor_lists_per_neighbor[child] = spec.cell_anchor_groups[cell_idx][child]

        cell_extras = list(spec.extra_open_anchors.get(cell_idx, []))
        cell_extras_pos = list(extras_pos.get(cell_idx, []))
        all_template_verts_used = set(cell_extras)
        for al in anchor_lists_per_neighbor.values():
            all_template_verts_used.update(al)
        cell_template_anchors = sorted(all_template_verts_used)

        # T_rooted of cell with all relevant anchors as boundary
        T_cell = t_rooted_cached(cell_template, cell_template_anchors)
        label_map: Dict[int, int] = {}
        for nbr, al in anchor_lists_per_neighbor.items():
            for tmpl_v, pos_v in zip(al, pos[cell_idx][nbr]):
                if tmpl_v in label_map:
                    assert label_map[tmpl_v] == pos_v, "shared anchor pos mismatch"
                else:
                    label_map[tmpl_v] = pos_v
        for tmpl_v, pos_v in zip(cell_extras, cell_extras_pos):
            if tmpl_v in label_map:
                assert label_map[tmpl_v] == pos_v, "extra anchor pos mismatch"
            else:
                label_map[tmpl_v] = pos_v
        T_cell_pos = relabel_partition_dict(T_cell, label_map)

        # Initial state cell groups for this cell.
        #
        # PER-CELL COMPRESSION CAVEAT (May 2026): per-cell compression
        # works ONLY for cell-tree topologies WITHOUT shared anchors
        # (i.e., no child triggers keep_shared=True at this cell). When
        # keep_shared=True, the junction's aut on (cell_outward,
        # child_inward) is the DIAGONAL S_k (matching swap), but
        # per_cell_canonical_key on output [[cell_outward], [child_inward]]
        # assumes INDEPENDENT S_k × S_k. This over-compresses output
        # orbits (factor k!), causing non-divisible coeffs in
        # orbit_convolve. The proper fix is custom aut compression
        # (junction diagonal + state extras' per-cell), tracked as
        # future work. For now, when keep_shared=True is encountered,
        # we fall back to no compression for that step.
        if enable_per_cell_compression:
            cell_pos_for_anchors = [label_map[a] for a in cell_template_anchors]
            # Per-neighbor anchor sets must be preserved to keep the
            # operational structure (which neighbor connects to which
            # anchors). For K_{4,4} mixed-direction this filters out
            # the side-swap aut.
            preserve_sets = [
                list(al) for al in anchor_lists_per_neighbor.values()
            ]
            state_cell_groups = _initial_cell_groups(
                cell_template, cell_template_anchors, cell_pos_for_anchors,
                preserve_anchor_sets=preserve_sets,
            )
            state_orbit_T, state_orbit_partitions = aut_compress_t_rooted_per_cell(
                T_cell_pos, state_cell_groups,
            )
            # Replace orbit partitions with single rep per orbit (for
            # precompute_M_table per-cell mode).
            state_orbit_partitions = {
                Oo: [per_cell_orbit_rep(Oo, state_cell_groups)]
                for Oo in state_orbit_T
            }
        else:
            state_cell_groups = None
            state_orbit_T, state_orbit_partitions = aut_compress_t_rooted(
                T_cell_pos, []
            )
        # state's open positions = ALL allocated positions for this cell
        state_open_pos: List[int] = sorted(set(label_map.values()))

        # Parent-facing positions (must be PRESERVED throughout).
        # Extras are also preserved — they're cross-cell-shared and
        # propagate up to a common ancestor where they merge with their
        # cross-cell-identified counterpart from another subtree.
        parent_facing_pos: List[int] = (
            list(pos[cell_idx][parent_cell_idx])
            if parent_cell_idx is not None else []
        )
        parent_facing_set = set(parent_facing_pos) | set(cell_extras_pos)

        # Process each child subtree
        for child_idx, child in enumerate(children):
            unprocessed_children = children[child_idx + 1:]
            (child_state_T, child_state_partitions, child_open_pos,
             child_cell_groups) = dp_subtree(child, cell_idx)

            cell_outward_pos = pos[cell_idx][child]
            child_inward_pos = pos[child][cell_idx]

            # Build junction T_rooted with positions
            junction_anchor_list = list(spec.junction_anchors_A) + list(spec.junction_anchors_B)
            junction_pos_list = list(cell_outward_pos) + list(child_inward_pos)
            junction_label_map = {a: p for a, p in zip(junction_anchor_list, junction_pos_list)}
            T_junction = t_rooted_cached(junction_template, junction_anchor_list)
            T_junction_pos = relabel_partition_dict(T_junction, junction_label_map)

            # Junction aut compression deferred until AFTER fallback
            # check (filled in below). State per-cell active → diagonal
            # junction aut compression. State expanded → no compression.

            # Compute "future-live" positions: positions in state_open_pos
            # that will still be needed after this junction (i.e., shared
            # with parent, unprocessed children, or carried up as extras).
            future_live_set = set(parent_facing_pos) | set(cell_extras_pos)
            for c in unprocessed_children:
                future_live_set.update(pos[cell_idx][c])

            # Decide keep_shared: True iff cell_outward_pos has any position
            # shared with future_live_set.
            cell_outward_set = set(cell_outward_pos)
            cell_outward_is_shared = bool(cell_outward_set & future_live_set)

            # PER-CELL COMPRESSION FALLBACK — expand state in two cases:
            #
            # (1) keep_shared=True: junction's diagonal S_k aut on
            # (cell_outward, child_inward) doesn't realize the
            # independent S_k × S_k that per_cell_canonical_key assumes
            # on output [[cell_outward], [child_inward]] →
            # over-compresses, divisibility fails.
            #
            # (2) "fully-consumed state group" (e.g., root cell with no
            # parent): a state per-cell group is ENTIRELY in
            # cell_outward (consumed), so state aut σ on that group
            # doesn't lift to output via convolution iteration. The
            # M-table iterates state_rep × all P_junc; off-diagonal
            # (σ(state_rep) × P_junc) pairs that land in different
            # output orbits are missed → n_state factor over-counts
            # → divisibility fails (e.g., K_5 claw shared M_3 case).
            need_fallback = False
            if (enable_per_cell_compression
                    and state_cell_groups is not None):
                if cell_outward_is_shared:
                    need_fallback = True
                else:
                    # Detect (2): any state group fully consumed AND
                    # has non-trivial per-cell aut (size ≥ 2).
                    for g in state_cell_groups:
                        if (set(g).issubset(cell_outward_set)
                                and len(g) >= 2):
                            for Oo in state_orbit_T:
                                if per_cell_orbit_size(Oo, state_cell_groups) > 1:
                                    need_fallback = True
                                    break
                            if need_fallback:
                                break
            # COMBINED-AUT JUNCTION STEP (validated 6/7 cases incl.
            # K_5 claw shared M_3 + K_{4,4} 5-cell Cm3-pattern M_2 at
            # 23× speedup). Replaces the previous "expand state to
            # uncompressed BEFORE junction step" path: that path blew
            # state up to Bell(N) before the convolution; the combined-aut
            # path keeps state per-cell during M-table iteration and only
            # expands AFTER. See `tutte/research/data/combined_aut_findings.md`.
            use_combined_aut_step = need_fallback

            # Build junction aut. Always non-empty when use_combined_aut_step
            # (we need it to construct the combined output aut). Otherwise
            # only when per-cell still active (existing behavior).
            if use_combined_aut_step or (
                    enable_per_cell_compression
                    and state_cell_groups is not None):
                junction_aut = build_relabel_aut(
                    junction_template,
                    junction_anchor_list,
                    junction_pos_list,
                    preserve_split_index=len(spec.junction_anchors_A),
                )
            else:
                junction_aut = []
            junction_orbit_T, junction_orbit_partitions = aut_compress_t_rooted(
                T_junction_pos, junction_aut
            )

            # === Junction step ===
            other_open = [p for p in state_open_pos if p not in cell_outward_set]
            state_extra_pos = list(other_open)
            if cell_outward_is_shared:
                out_anchors = (list(state_extra_pos) + list(cell_outward_pos)
                                + list(child_inward_pos))
            else:
                out_anchors = list(state_extra_pos) + list(child_inward_pos)

            use_per_cell_this_step = (
                enable_per_cell_compression
                and state_cell_groups is not None
                and not use_combined_aut_step
            )

            if use_combined_aut_step:
                # Combined-aut path: keeps state per-cell during M-table
                # iteration, expands AFTER. Avoids the Bell(N) blowup of
                # the previous "expand-before-junction" fallback.
                state_orbit_T, state_orbit_partitions = (
                    _run_combined_aut_junction_step(
                        state_orbit_T, state_orbit_partitions,
                        state_cell_groups,
                        junction_orbit_T, junction_orbit_partitions,
                        junction_aut,
                        list(cell_outward_pos),
                        list(child_inward_pos),
                        list(state_extra_pos),
                        cell_outward_is_shared,
                        list(out_anchors),
                    )
                )
                state_cell_groups = None
                state_open_pos = list(out_anchors)
                if not cell_outward_is_shared:
                    total_div[0] += len(cell_outward_pos) - junction_c_J
            else:
                if use_per_cell_this_step:
                    out_state_cell_groups_after_junc = _state_groups_after_junction(
                        state_cell_groups, cell_outward_pos, child_inward_pos,
                        cell_outward_is_shared,
                    )
                    out_orbit_partitions = {}
                    out_orbit_sizes = {}
                else:
                    out_state_cell_groups_after_junc = None
                    out_orbit_partitions = enumerate_partitions_per_orbit(out_anchors, [])
                    out_orbit_sizes = {ok: len(parts) for ok, parts in out_orbit_partitions.items()}

                if use_per_cell_this_step:
                    M_j = precompute_M_table(
                        state_orbit_partitions, junction_orbit_partitions,
                        shared_boundary=list(cell_outward_pos),
                        extra_boundary=list(child_inward_pos),
                        out_aut_group=[],
                        state_extra_boundary=list(state_extra_pos),
                        keep_shared=cell_outward_is_shared,
                        out_cell_anchor_groups=out_state_cell_groups_after_junc,
                        state_cell_anchor_groups=state_cell_groups,
                    )
                    # Compute orbit sizes analytically.
                    for (Os, Oj, Oo) in M_j.keys():
                        if Oo not in out_orbit_sizes:
                            out_orbit_sizes[Oo] = per_cell_orbit_size(
                                Oo, out_state_cell_groups_after_junc,
                            )
                else:
                    M_j = precompute_M_table(
                        state_orbit_partitions, junction_orbit_partitions,
                        shared_boundary=list(cell_outward_pos),
                        extra_boundary=list(child_inward_pos),
                        out_aut_group=[],
                        state_extra_boundary=list(state_extra_pos),
                        keep_shared=cell_outward_is_shared,
                    )
                state_orbit_T = orbit_convolve(
                    state_orbit_T, junction_orbit_T, M_j, out_orbit_sizes
                )
                if not cell_outward_is_shared:
                    total_div[0] += len(cell_outward_pos) - junction_c_J

                if use_per_cell_this_step:
                    state_cell_groups = out_state_cell_groups_after_junc
                    state_orbit_partitions = {
                        Oo: [per_cell_orbit_rep(Oo, state_cell_groups)]
                        for Oo in state_orbit_T
                    }
                else:
                    state_orbit_partitions = out_orbit_partitions
                state_open_pos = list(out_anchors)

            # === Cell-merge step ===
            # Child's open positions = child_inward_pos plus any extras
            # propagated up from inside the child's subtree.
            child_inward_set = set(child_inward_pos)
            child_extras_up = [p for p in child_open_pos
                                if p not in child_inward_set]
            assert child_inward_set.issubset(set(child_open_pos)), (
                f"child {child}'s state open missing inward: "
                f"open={sorted(child_open_pos)}, "
                f"inward={sorted(child_inward_pos)}"
            )
            # NOTE: cross-cell merging at common ancestor (extras_already_in_state)
            # requires care — neither (xy-1)^d join nor pure-product is
            # quite right. See `step3_milestone_b_design.md` for the
            # architectural challenge. For now this branch is not
            # active in production; cross-cell extras propagate up
            # purely via extra_boundary in the M-table and are
            # marginalized at root, which gives correct results only
            # for the trivial case (no actual cross-cell sharing
            # involving graph reasoning).
            cell_extra_after = [p for p in state_open_pos
                                 if p not in child_inward_set]
            out_anchors2 = list(cell_extra_after) + list(child_extras_up)

            use_per_cell_merge = (
                enable_per_cell_compression
                and state_cell_groups is not None
            )

            # If state is per-cell but child is uncompressed, the
            # convolution's n_state factor over-counts because state's
            # per-cell aut doesn't act symmetrically on child's partitions.
            # Expand state to uncompressed for correctness.
            if use_per_cell_merge and child_cell_groups is None:
                state_orbit_T, state_orbit_partitions = _expand_per_cell_state(
                    state_orbit_T, state_cell_groups,
                )
                state_cell_groups = None
                use_per_cell_merge = False

            # SYMMETRIC FIX: when both state AND child are per-cell at
            # cell-merge, use the new `junction_cell_anchor_groups`
            # parameter in precompute_M_table so child contributes
            # n_junc analytically (no need to expand). This is correct
            # in TRIVIAL-ORBIT cases (every state and child orbit has
            # size 1) — the typical case when state survived to
            # cell-merge without triggering the junction step's
            # fully-consumed fallback. For non-trivial-orbit cases,
            # state has already been expanded at junction step, so
            # state_cell_groups is None here and this branch doesn't
            # fire (we fall through to the legacy expand-child branch
            # below).
            if (use_per_cell_merge
                    and child_cell_groups is not None):
                # Both per-cell — use junction_cell_anchor_groups in
                # M-table. Skip expansion. Per-cell × per-cell only
                # safe in trivial-orbit cases; verified by audit at
                # junction step's need_fallback check that kept state
                # per-cell only when all orbits are size 1.
                pass  # use_per_cell_merge stays True; M-table call
                      # below passes junction_cell_anchor_groups.
            elif child_cell_groups is not None:
                # State uncompressed but child per-cell — expand child
                # for the M-table to iterate all members correctly.
                child_state_T, child_state_partitions = _expand_per_cell_state(
                    child_state_T, child_cell_groups,
                )
                child_cell_groups = None

            if use_per_cell_merge:
                out_state_cell_groups_after_merge = _state_groups_after_cell_merge(
                    state_cell_groups, child_inward_pos,
                )
                out_orbit_partitions2 = {}
                out_orbit_sizes2 = {}
            else:
                out_state_cell_groups_after_merge = None
                out_orbit_partitions2 = enumerate_partitions_per_orbit(out_anchors2, [])
                out_orbit_sizes2 = {ok: len(parts) for ok, parts in out_orbit_partitions2.items()}

            # Child's state_partitions: if child has cell_groups (per-cell
            # compressed), use them for the M-table convolution. Else child
            # is uncompressed (each partition own orbit, n_junc = 1).
            child_cell_anchor_groups_arg = child_cell_groups if use_per_cell_merge else None

            if use_per_cell_merge:
                M_c = precompute_M_table(
                    state_orbit_partitions, child_state_partitions,
                    shared_boundary=list(child_inward_pos),
                    extra_boundary=list(child_extras_up),
                    out_aut_group=[],
                    state_extra_boundary=list(cell_extra_after),
                    out_cell_anchor_groups=out_state_cell_groups_after_merge,
                    state_cell_anchor_groups=state_cell_groups,
                    junction_cell_anchor_groups=child_cell_groups,
                )
                for (Os, Oj, Oo) in M_c.keys():
                    if Oo not in out_orbit_sizes2:
                        out_orbit_sizes2[Oo] = per_cell_orbit_size(
                            Oo, out_state_cell_groups_after_merge,
                        )
            else:
                M_c = precompute_M_table(
                    state_orbit_partitions, child_state_partitions,
                    shared_boundary=list(child_inward_pos),
                    extra_boundary=list(child_extras_up),
                    out_aut_group=[],
                    state_extra_boundary=list(cell_extra_after),
                )
            state_orbit_T = orbit_convolve(
                state_orbit_T, child_state_T, M_c, out_orbit_sizes2
            )
            child_anchors_template = spec.cell_anchor_groups[child][cell_idx]
            c_cell = components_touching(cell_template, list(child_anchors_template))
            total_div[0] += len(child_inward_pos) - c_cell

            if use_per_cell_merge:
                state_cell_groups = out_state_cell_groups_after_merge
                state_orbit_partitions = {
                    Oo: [per_cell_orbit_rep(Oo, state_cell_groups)]
                    for Oo in state_orbit_T
                }
            else:
                state_orbit_partitions = out_orbit_partitions2
            state_open_pos = list(out_anchors2)

            # === Marginalize dead positions ===
            # After processing this child, some of cell_outward_pos may now be
            # dead (no future use). Remove them from state via marginalization.
            # Live positions = positions in future_live_set ∪ (parent_facing for
            # cell_idx). At this point, future_live_set already includes parent.
            live_positions = sorted(set(state_open_pos) & future_live_set)
            if sorted(state_open_pos) != live_positions:
                if (enable_per_cell_compression
                        and state_cell_groups is not None):
                    new_state_cell_groups = _state_groups_after_marginalize(
                        state_cell_groups, live_positions,
                    )
                    state_orbit_T = _marginalize_state_per_cell(
                        state_orbit_T, state_cell_groups,
                        live_positions, new_state_cell_groups,
                    )
                    state_cell_groups = new_state_cell_groups
                    state_orbit_partitions = {
                        Oo: [per_cell_orbit_rep(Oo, state_cell_groups)]
                        for Oo in state_orbit_T
                    }
                else:
                    state_orbit_T, state_orbit_partitions = _marginalize_state(
                        state_orbit_T, state_orbit_partitions, live_positions
                    )
                state_open_pos = live_positions

        # CONTRACT: returned state_open_pos must:
        # - INCLUDE all of parent_facing_pos (parent needs them to join).
        # - OTHERWISE be a subset of cell_extras_pos (those that haven't
        #   been consumed yet via cross-cell merging in this subtree).
        state_open_set = set(state_open_pos)
        parent_facing_set_local = set(parent_facing_pos)
        extras_set = set(cell_extras_pos)
        assert parent_facing_set_local.issubset(state_open_set), (
            f"cell {cell_idx} (parent {parent_cell_idx}) returned open "
            f"{sorted(state_open_pos)}, missing parent_facing "
            f"{sorted(parent_facing_pos)}"
        )
        leftover = state_open_set - parent_facing_set_local
        assert leftover.issubset(extras_set), (
            f"cell {cell_idx} (parent {parent_cell_idx}) returned open "
            f"{sorted(state_open_pos)} has positions outside "
            f"parent_facing {sorted(parent_facing_pos)} ∪ extras "
            f"{sorted(cell_extras_pos)}"
        )

        return (state_orbit_T, state_orbit_partitions, state_open_pos,
                state_cell_groups)

    # Run recursion from root
    final_state_T, _, final_open, _ = dp_subtree(spec.root, None)

    # If root has parent-facing anchors (impossible since root has no parent),
    # final_open would be those. Otherwise final_open should be empty.
    # Sum across all final partitions to get T(graph).
    final_poly = TuttePolynomial.zero()
    for P, val in final_state_T.items():
        final_poly = final_poly + val
    if total_div[0] > 0:
        final_poly = divide_by_x_minus_1_power(final_poly, total_div[0])
    return final_poly


# =============================================================================
# Corrected leaf DP (chord-rule cross-cell-ID merge).
# =============================================================================
#
# This module's `compute_tree_dp_recursive` above does not correctly handle
# leaves of the chord-rule recursion that have cross-cell vertex identifications
# (Step 3.B.3 architectural blocker). The function below implements the
# corrected convolution rule derived empirically (verified on test cases including
# 2 K_4 + M_1 + 1 shared vertex):
#
#     T_combined[P_comb] = sum_{(P_1, P_2) -> P_comb}
#         T_rooted_1[P_1] * T_rooted_2[P_2] * (y-1)^{k-m} / (x-1)^{m-1}
#
# where k = #shared positions and m = |P_1| + |P_2| - |P_comb| (merge events).
#
# This is a brute-force DP without orbit compression — slower per cell than
# the main tree DP, but correct on cross-cell-ID inputs. Used by the hybrid
# cycle-close path's spec dispatch.


def _t_rooted_full_boundary(
    graph: Graph, boundary: List[int],
) -> Dict[Tuple, TuttePolynomial]:
    """Compute T_rooted of `graph` with full boundary, indexed by partitions
    of `boundary` representing connectivity in the spanning subgraph.

    Mirrors the standard `t_rooted_cached` interface but enumerates partitions
    keyed by the FULL boundary (no orbit compression).
    """
    return t_rooted_cached(graph, boundary)


def _merge_cells_corrected_compressed(
    state_orbit_T: Dict[Tuple, TuttePolynomial],
    state_orbit_partitions: Dict[Tuple, List[Tuple]],
    state_pos: List[int],
    state_cell_groups: List[List[int]],
    child_orbit_T: Dict[Tuple, TuttePolynomial],
    child_orbit_partitions: Dict[Tuple, List[Tuple]],
    child_pos: List[int],
    child_cell_groups: List[List[int]],
    junction_shared: List[int],
    id_shared: List[int],
    out_pos: List[int],
    out_cell_groups: List[List[int]],
    child_full_components: int = 1,
) -> Dict[Tuple, TuttePolynomial]:
    """Orbit-compressed merge using per_cell_canonical_key dedup.

    Iterates over state_orbit reps × all child members (uncompressed
    on child side). Each state orbit's contribution is multiplied by
    its orbit size; the child side enumerates all members explicitly
    via `child_orbit_partitions`. Output partitions are canonicalized
    via per_cell_canonical_key on `out_cell_groups`.

    PRECONDITIONS (caller MUST verify, NOT enforced here):
    - Cells must support full S_n symmetry on their anchor groups
      (K_n or K_{a,b} with anchors on one bipartition side).
    - `state_cell_groups` and `child_cell_groups` must be S_n^N
      compatible with the spanning-subgraph computation that produced
      `state_orbit_T` and `child_orbit_T`.

    See `per_cell_canonical_key` documentation for details on when
    this is safe.
    """
    y_m1 = TuttePolynomial.from_coefficients({(0, 1): 1, (0, 0): -1})
    xy_m1 = TuttePolynomial.from_coefficients({
        (1, 1): 1, (1, 0): -1, (0, 1): -1, (0, 0): 1,
    })
    k_id = len(id_shared)
    junction_list = list(junction_shared)
    id_list = list(id_shared)
    full_universe = sorted(set(state_pos) | set(child_pos))

    new_T: Dict[Tuple, TuttePolynomial] = {}
    x_m1_poly = TuttePolynomial.from_coefficients({(1, 0): 1, (0, 0): -1})
    out_pos_list = list(out_pos)
    has_out = bool(out_pos_list)
    has_groups = bool(out_cell_groups) and has_out

    # Pre-compute child's restrictions (independent of state).
    child_precomp: List[Tuple[Tuple, TuttePolynomial, Tuple, Tuple]] = []
    for P_child_canonical, T_child in child_orbit_T.items():
        child_members = child_orbit_partitions.get(P_child_canonical)
        if not child_members:
            continue
        for P_child in child_members:
            P_child_J = (restrict_partition(P_child, junction_list)
                          if junction_list else ())
            P_child_id = (restrict_partition(P_child, id_list)
                           if id_list else ())
            child_precomp.append((P_child, T_child, P_child_J, P_child_id))

    # Iterate over state orbits (one rep per orbit). For each rep, iterate
    # pre-computed child entries.
    for P_state_canonical, T_state in state_orbit_T.items():
        members = state_orbit_partitions.get(P_state_canonical)
        if not members:
            continue
        n_state = len(members)
        P_state = members[0]
        P_state_J = (restrict_partition(P_state, junction_list)
                      if junction_list else ())
        P_state_id = (restrict_partition(P_state, id_list)
                       if id_list else ())
        # Pre-scale T_state by n_state (hoisted out of inner loop).
        T_state_scaled = T_state if n_state == 1 else n_state * T_state

        for P_child, T_child, P_child_J, P_child_id in child_precomp:
            if junction_list:
                d = delta(P_state_J, P_child_J, junction_list)
                if d < 0:
                    continue
            else:
                d = 0

            joint = join_partitions(P_state, P_child, full_universe)
            if id_list:
                joint_id = restrict_partition(joint, id_list)
                m = len(P_state_id) + len(P_child_id) - len(joint_id)
                if m < 1:
                    continue
            else:
                m = 0

            contrib = T_state_scaled * T_child
            for _ in range(d):
                contrib = contrib * xy_m1
            if id_list:
                for _ in range(k_id - m):
                    contrib = contrib * y_m1
                divisor_exp = m - child_full_components
                if divisor_exp > 0:
                    try:
                        contrib = divide_by_x_minus_1_power(
                            contrib, divisor_exp,
                        )
                    except ValueError:
                        continue
                elif divisor_exp < 0:
                    for _ in range(-divisor_exp):
                        contrib = contrib * x_m1_poly

            if has_out:
                P_out = restrict_partition(joint, out_pos_list)
                P_out_canonical = (
                    per_cell_canonical_key(P_out, out_cell_groups)
                    if has_groups else P_out
                )
            else:
                P_out_canonical = ()

            existing = new_T.get(P_out_canonical)
            if existing is None:
                new_T[P_out_canonical] = contrib
            else:
                new_T[P_out_canonical] = existing + contrib
    return new_T


def _merge_cells_corrected(
    state_T: Dict[Tuple, TuttePolynomial],
    state_pos: List[int],
    child_T: Dict[Tuple, TuttePolynomial],
    child_pos: List[int],
    junction_shared: List[int],
    id_shared: List[int],
    out_pos: List[int],
    child_full_components: int = 1,
) -> Dict[Tuple, TuttePolynomial]:
    """Merge state and child T-tables via vertex-identification convolution.

    - `id_shared`: positions shared via vertex identification (cross-cell-ID
      OR junction template's anchor identification with state). For each
      (P_state, P_child) pair, apply per-pair correction:
          (y-1)^{k-m} / (x-1)^{m - child_full_components}
      where k = |id_shared| and m = merge_events on id_shared.

    - `junction_shared`: positions shared via active junction edges (with
      `(xy-1)^d` factor). Currently unused in the chord-rule leaf path
      (everything is vertex-identification), kept for future generality.

    `child_full_components` = c(G_child) where G_child is the full edge
    set of the component being merged (cell or junction template). For
    cells this is 1 (connected); for junction template M_k it is k
    (matching has k components).

    Mathematical derivation: when state's full graph is connected and
    child's full graph has c_child components, the rank-deficit gain
    from merging is k + 1 - 1 - c_child = k - c_child, so the
    convolution over-counts (x-1) by `m - c_child` per pair (proven
    empirically; see `tutte/research/data/step3_milestone_b_design.md`).
    """
    full_universe = sorted(set(state_pos) | set(child_pos))
    y_m1 = TuttePolynomial.from_coefficients({(0, 1): 1, (0, 0): -1})
    xy_m1 = TuttePolynomial.from_coefficients({
        (1, 1): 1, (1, 0): -1, (0, 1): -1, (0, 0): 1,
    })
    k_id = len(id_shared)
    junction_list = list(junction_shared)
    id_list = list(id_shared)

    new_T: Dict[Tuple, TuttePolynomial] = {}
    for P_state, T_state in state_T.items():
        P_state_J = (restrict_partition(P_state, junction_list)
                      if junction_list else ())
        P_state_id = (restrict_partition(P_state, id_list)
                       if id_list else ())
        for P_child, T_child in child_T.items():
            P_child_J = (restrict_partition(P_child, junction_list)
                          if junction_list else ())
            P_child_id = (restrict_partition(P_child, id_list)
                           if id_list else ())

            if junction_list:
                d = delta(P_state_J, P_child_J, junction_list)
                if d < 0:
                    continue
            else:
                d = 0

            joint = join_partitions(P_state, P_child, full_universe)
            if id_list:
                joint_id = restrict_partition(joint, id_list)
                m = len(P_state_id) + len(P_child_id) - len(joint_id)
            else:
                m = 0

            if id_list and m < 1:
                continue

            contrib = T_state * T_child
            for _ in range(d):
                contrib = contrib * xy_m1
            if id_list:
                for _ in range(k_id - m):
                    contrib = contrib * y_m1
                divisor_exp = m - child_full_components
                if divisor_exp > 0:
                    try:
                        contrib = divide_by_x_minus_1_power(contrib, divisor_exp)
                    except ValueError:
                        continue
                elif divisor_exp < 0:
                    # Multiply by (x-1)^{-divisor_exp}
                    x_m1 = TuttePolynomial.from_coefficients({(1, 0): 1, (0, 0): -1})
                    for _ in range(-divisor_exp):
                        contrib = contrib * x_m1

            P_out = (restrict_partition(joint, out_pos)
                      if out_pos else ())
            if P_out in new_T:
                new_T[P_out] = new_T[P_out] + contrib
            else:
                new_T[P_out] = contrib
    return new_T


def _allocate_positions_with_ids(
    spec: CellTreeSpec,
) -> Tuple[Dict[int, Dict[int, List[int]]], Dict[int, List[int]],
           Dict[Tuple[int, int], int]]:
    """Allocate position labels including cross-cell-ID endpoints.

    Returns ``(junction_pos, extras_pos, all_pos_lookup)`` where:
    - `junction_pos[cell][nbr]` = list of position labels for cell's anchors
      facing neighbor `nbr` (junction edges).
    - `extras_pos[cell]` = list of position labels for cell's extras.
    - `all_pos_lookup[(cell, anchor)]` = position label for every
      `(cell, anchor)` pair appearing in any junction, extra, or
      cross-cell-ID — including pairs that ONLY appear in
      `cross_cell_identifications` (which the standard
      `_allocate_tree_positions` skips).
    """
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    used_pairs: List[Tuple[int, int]] = []
    seen_pairs = set()

    def _add(p):
        if p not in seen_pairs:
            seen_pairs.add(p)
            used_pairs.append(p)
            parent.setdefault(p, p)

    for cell_idx in spec.cell_tree.nodes():
        for nbr_anchors in spec.cell_anchor_groups.get(cell_idx, {}).values():
            for a in nbr_anchors:
                _add((cell_idx, a))
        for a in spec.extra_open_anchors.get(cell_idx, []):
            _add((cell_idx, a))
    # Include cross-cell-ID endpoints even if not in junctions/extras.
    for (ci, ai, cj, aj) in spec.cross_cell_identifications:
        _add((ci, ai))
        _add((cj, aj))
        union((ci, ai), (cj, aj))

    component_to_pos: Dict[Tuple[int, int], int] = {}
    next_pos = 10000
    for pair in sorted(used_pairs):
        rep = find(pair)
        if rep not in component_to_pos:
            component_to_pos[rep] = next_pos
            next_pos += 1

    junction_pos: Dict[int, Dict[int, List[int]]] = {}
    extras_pos: Dict[int, List[int]] = {}
    all_pos: Dict[Tuple[int, int], int] = {}
    for cell_idx in spec.cell_tree.nodes():
        junction_pos[cell_idx] = {}
        for nbr in sorted(spec.cell_anchor_groups.get(cell_idx, {}).keys()):
            anchors = spec.cell_anchor_groups[cell_idx][nbr]
            junction_pos[cell_idx][nbr] = [
                component_to_pos[find((cell_idx, a))] for a in anchors
            ]
        extras_pos[cell_idx] = [
            component_to_pos[find((cell_idx, a))]
            for a in spec.extra_open_anchors.get(cell_idx, [])
        ]
    for pair in used_pairs:
        all_pos[pair] = component_to_pos[find(pair)]
    return junction_pos, extras_pos, all_pos


def build_leaf_graph_from_spec(spec: CellTreeSpec) -> Graph:
    """Construct the actual leaf graph implicitly defined by `spec`.

    Used for ground-truth validation of `compute_corrected_leaf_dp`.
    Each cell gets a copy of `spec.cell_template` with vertices relabeled
    to fresh unique IDs (with cross-cell identifications collapsed via
    union-find on (cell, anchor) pairs). Junction edges between cells
    are added per `spec.cell_anchor_groups` + `spec.junction_template`.
    """
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if ra < rb: parent[rb] = ra
        else: parent[ra] = rb

    # Build all (cell, template_vertex) pairs.
    pairs = set()
    for cell_idx in spec.cell_tree.nodes():
        for v in spec.cell_template.nodes:
            pair = (cell_idx, v)
            pairs.add(pair)
            parent.setdefault(pair, pair)
    # Apply cross-cell-IDs.
    for (ci, ai, cj, aj) in spec.cross_cell_identifications:
        union((ci, ai), (cj, aj))

    # Assign each component a fresh global vertex ID.
    component_id: Dict[Tuple[int, int], int] = {}
    next_id = 0
    for pair in sorted(pairs):
        rep = find(pair)
        if rep not in component_id:
            component_id[rep] = next_id
            next_id += 1

    def vertex_id(cell_idx, template_v):
        return component_id[find((cell_idx, template_v))]

    nodes = set(component_id.values())
    edges = set()

    # Add cell internal edges.
    for cell_idx in spec.cell_tree.nodes():
        for u, v in spec.cell_template.edges:
            gu = vertex_id(cell_idx, u)
            gv = vertex_id(cell_idx, v)
            if gu != gv:
                edges.add((min(gu, gv), max(gu, gv)))

    # Add junction edges per cell_anchor_groups + junction_template.
    junction_anchors_A = spec.junction_anchors_A
    junction_anchors_B = spec.junction_anchors_B
    for u, v in spec.junction_template.edges:
        # u and v are positions in junction_template (0..2k-1).
        # u in junction_anchors_A or junction_anchors_B; same for v.
        # In standard junction_template (matching), one endpoint is in A side, the other in B side.
        pass
    # Simpler: iterate cell_anchor_groups, add the M_k matching edges between paired anchors.
    seen_pairs = set()
    for ci in spec.cell_tree.nodes():
        for cj, anchors_i in spec.cell_anchor_groups.get(ci, {}).items():
            if (ci, cj) in seen_pairs or (cj, ci) in seen_pairs:
                continue
            anchors_j = spec.cell_anchor_groups.get(cj, {}).get(ci, [])
            if len(anchors_j) != len(anchors_i):
                continue
            for ai, aj in zip(anchors_i, anchors_j):
                gu = vertex_id(ci, ai)
                gv = vertex_id(cj, aj)
                if gu != gv:
                    edges.add((min(gu, gv), max(gu, gv)))
            seen_pairs.add((ci, cj))

    return Graph(nodes=frozenset(nodes), edges=frozenset(edges))


def compute_corrected_leaf_dp(
    spec: CellTreeSpec,
) -> TuttePolynomial:
    """Brute-force DP for chord-rule leaf specs with corrected cross-cell-ID
    convolution rule.

    Algorithm:
    1. Allocate position labels (with cross-cell-ID endpoints unioned).
    2. Compute T_rooted of each cell with boundary = union of all positions
       it touches (junction anchors, extras, cross-cell-id endpoints).
    3. Compute T_rooted of each junction template instance with boundary =
       both cells' anchor positions (these edges live BETWEEN cells).
    4. Compose all cell + junction-template T_rooted's via pure
       vertex-identification convolution using the corrected formula.
    5. Sum across partitions to get T(leaf).

    All convolutions are vertex identifications; the corrected formula
    `(y-1)^{k-m} / (x-1)^{m-1}` per (P_1, P_2) pair handles the rank/
    nullity adjustment correctly.
    """
    junction_pos, extras_pos, all_pos = _allocate_positions_with_ids(spec)
    cell_template = spec.cell_template

    # Per-cell boundary = junction-facing + extras + cross-cell-id endpoints.
    cell_boundary: Dict[int, List[int]] = {}
    cell_anchor_to_pos: Dict[int, Dict[int, int]] = {}
    for cell_idx in spec.cell_tree.nodes():
        boundary_set = set()
        anchor_to_pos: Dict[int, int] = {}
        for nbr, anchors in spec.cell_anchor_groups.get(cell_idx, {}).items():
            for a, p in zip(anchors, junction_pos[cell_idx][nbr]):
                boundary_set.add(p)
                anchor_to_pos[a] = p
        for a, p in zip(
            spec.extra_open_anchors.get(cell_idx, []),
            extras_pos[cell_idx],
        ):
            boundary_set.add(p)
            anchor_to_pos[a] = p
        for (ci, ai, cj, aj) in spec.cross_cell_identifications:
            if ci == cell_idx and (ci, ai) in all_pos:
                p = all_pos[(ci, ai)]
                boundary_set.add(p)
                anchor_to_pos[ai] = p
            if cj == cell_idx and (cj, aj) in all_pos:
                p = all_pos[(cj, aj)]
                boundary_set.add(p)
                anchor_to_pos[aj] = p
        cell_boundary[cell_idx] = sorted(boundary_set)
        cell_anchor_to_pos[cell_idx] = anchor_to_pos

    # Compute T_rooted of each cell. Cache at the template level: cells
    # with the SAME template-local boundary share T_rooted (just with
    # different position labels). Compute once per unique boundary set,
    # then translate partition keys per cell.
    #
    # Also compute orbit-compressed forms via per_cell_canonical_key for
    # the K_{a,b} cell case (S_n acts on each anchor side independently).
    cell_T: Dict[int, Dict[Tuple, TuttePolynomial]] = {}
    cell_orbit_T: Dict[int, Dict[Tuple, TuttePolynomial]] = {}
    cell_orbit_partitions: Dict[int, Dict[Tuple, List[Tuple]]] = {}
    cell_groups_per_cell: Dict[int, List[List[int]]] = {}
    template_T_cache: Dict[Tuple[int, ...], Dict[Tuple, TuttePolynomial]] = {}
    # Flag: are all cells safely per-cell compressible?
    # If any cell has shared anchors across groups, per_cell_canonical_key
    # over-collapses — fall back to uncompressed everywhere.
    all_cells_compressible = True
    for cell_idx in spec.cell_tree.nodes():
        anchor_to_pos = cell_anchor_to_pos[cell_idx]
        # Boundary in TEMPLATE-LOCAL labels (anchor template vertices).
        template_boundary = sorted(anchor_to_pos.keys())
        cache_key = tuple(template_boundary)
        if cache_key in template_T_cache:
            T_template = template_T_cache[cache_key]
        else:
            T_template = _t_rooted_full_boundary(cell_template, template_boundary)
            template_T_cache[cache_key] = T_template
        cell_T[cell_idx] = relabel_partition_dict(T_template, anchor_to_pos)
        # Cell groups: split positions by SOURCE (which neighbor / extras /
        # cross-cell-id they come from). For K_{a,b} cells with anchors on
        # one bipartition side per source, each group is a separate
        # bipartition side and S_n acts within it. Verified by
        # aut_compress check.
        groups_for_cell: List[List[int]] = []
        seen_pos = set()
        # Group by neighbor (junction anchors).
        for nbr in sorted(spec.cell_anchor_groups.get(cell_idx, {}).keys()):
            anchors = spec.cell_anchor_groups[cell_idx][nbr]
            grp = [anchor_to_pos[a] for a in anchors if a in anchor_to_pos]
            grp = sorted(set(grp) - seen_pos)
            if grp:
                groups_for_cell.append(grp)
                seen_pos.update(grp)
        # Group extras separately.
        extras_grp = sorted(
            anchor_to_pos[a] for a in spec.extra_open_anchors.get(cell_idx, [])
            if a in anchor_to_pos and anchor_to_pos[a] not in seen_pos
        )
        if extras_grp:
            groups_for_cell.append(extras_grp)
            seen_pos.update(extras_grp)
        # Cross-cell-id endpoints (template anchors not yet in seen_pos).
        cross_grp = sorted(
            pos for tv, pos in anchor_to_pos.items()
            if pos not in seen_pos
        )
        if cross_grp:
            groups_for_cell.append(cross_grp)
        cell_groups_per_cell[cell_idx] = groups_for_cell
        # Detect shared-anchor cells: if any template anchor appears in
        # multiple junction's groups, per_cell_canonical_key over-collapses
        # the orbit. Disable compression entirely in that case.
        anchor_neighbor_count = {}
        for nbr, anchors in spec.cell_anchor_groups.get(cell_idx, {}).items():
            for a in anchors:
                anchor_neighbor_count[a] = anchor_neighbor_count.get(a, 0) + 1
        has_shared_anchors = any(c > 1 for c in anchor_neighbor_count.values())
        if has_shared_anchors:
            all_cells_compressible = False
        # Compress via per-cell canonical key.
        try:
            o_T, o_parts = aut_compress_t_rooted_per_cell(
                cell_T[cell_idx], groups_for_cell,
            )
            cell_orbit_T[cell_idx] = o_T
            cell_orbit_partitions[cell_idx] = o_parts
        except ValueError:
            # Cell doesn't support per-cell compression with these
            # groupings — fall back to uncompressed.
            cell_orbit_T[cell_idx] = cell_T[cell_idx]
            cell_orbit_partitions[cell_idx] = {
                P: [P] for P in cell_T[cell_idx]
            }
            all_cells_compressible = False

    # Compute T_rooted of each junction template instance (one per tree edge).
    # Boundary = positions of both cells' junction anchors (in their respective
    # cell labelings).
    junction_T: Dict[Tuple[int, int], Dict[Tuple, TuttePolynomial]] = {}
    junction_boundary: Dict[Tuple[int, int], List[int]] = {}
    junction_template = spec.junction_template
    for ci, cj in spec.cell_tree.edges():
        a, b = (ci, cj) if ci < cj else (cj, ci)
        anchors_a = spec.cell_anchor_groups.get(a, {}).get(b, [])
        anchors_b = spec.cell_anchor_groups.get(b, {}).get(a, [])
        if len(anchors_a) != len(anchors_b):
            return TuttePolynomial.zero()
        # Junction template uses anchors junction_anchors_A + junction_anchors_B
        # We need to map junction template's vertices to positions:
        # template anchors junction_anchors_A → cell a's anchor positions
        # template anchors junction_anchors_B → cell b's anchor positions
        jan_A = spec.junction_anchors_A
        jan_B = spec.junction_anchors_B
        if len(jan_A) != len(anchors_a) or len(jan_B) != len(anchors_b):
            return TuttePolynomial.zero()
        relabel_J: Dict[int, int] = {}
        for jt_a, cell_anchor_a in zip(jan_A, anchors_a):
            relabel_J[jt_a] = cell_anchor_to_pos[a][cell_anchor_a]
        for jt_b, cell_anchor_b in zip(jan_B, anchors_b):
            relabel_J[jt_b] = cell_anchor_to_pos[b][cell_anchor_b]
        # Internal junction template vertices (if any beyond anchors) get
        # fresh labels.
        junction_internal_base = 2_000_000 + (a * 1000 + b) * 100
        for v in junction_template.nodes:
            if v not in relabel_J:
                relabel_J[v] = junction_internal_base + v
        new_nodes = [relabel_J[v] for v in junction_template.nodes]
        new_edges = [(relabel_J[u], relabel_J[v]) for u, v in junction_template.edges]
        j_graph = Graph(new_nodes, new_edges)
        # Boundary of junction template instance = positions of both cells' anchors.
        anchors_a_pos = [cell_anchor_to_pos[a][ca] for ca in anchors_a]
        anchors_b_pos = [cell_anchor_to_pos[b][cb] for cb in anchors_b]
        boundary = sorted(set(anchors_a_pos) | set(anchors_b_pos))
        junction_T[(a, b)] = _t_rooted_full_boundary(j_graph, boundary)
        junction_boundary[(a, b)] = boundary

    # Compose all components via pure vertex identification.
    # Strategy: walk cell tree in post-order. State accumulates contribution
    # of subtree, with boundary = junction-facing-up positions + un-resolved
    # cross-cell-id positions.

    # First build the list of components in merge order:
    # post-order traversal of cells, with junction templates inserted as
    # bridges between adjacent cells in the tree.

    visited: set = set()
    components: List[Tuple[str, object]] = []

    def post_order(node, parent):
        visited.add(node)
        for nbr in spec.cell_tree.neighbors(node):
            if nbr != parent and nbr not in visited:
                post_order(nbr, node)
                # After recursing into child, add the junction edge between
                # child and current node.
                edge_key = (min(node, nbr), max(node, nbr))
                components.append(("junction", edge_key))
        components.append(("cell", node))

    post_order(spec.root, None)

    # Compose.
    state_T: Optional[Dict[Tuple, TuttePolynomial]] = None
    state_pos: List[int] = []

    # Track for each position which cells/junctions still need to use it.
    # Position lives until all its uses are visited.
    pos_remaining_uses: Dict[int, int] = {}
    # Each cell's boundary positions are uses.
    for cell_idx, bdry in cell_boundary.items():
        for p in bdry:
            pos_remaining_uses[p] = pos_remaining_uses.get(p, 0) + 1
    # Each junction's boundary positions are uses.
    for edge, bdry in junction_boundary.items():
        for p in bdry:
            pos_remaining_uses[p] = pos_remaining_uses.get(p, 0) + 1

    used_so_far: Dict[int, int] = {p: 0 for p in pos_remaining_uses}

    # Compute connected components of cell template and junction template.
    cell_template_nx = nx.Graph()
    cell_template_nx.add_nodes_from(spec.cell_template.nodes)
    cell_template_nx.add_edges_from(spec.cell_template.edges)
    cell_template_components = nx.number_connected_components(cell_template_nx)
    junction_template_nx = nx.Graph()
    junction_template_nx.add_nodes_from(spec.junction_template.nodes)
    junction_template_nx.add_edges_from(spec.junction_template.edges)
    junction_template_components = nx.number_connected_components(
        junction_template_nx,
    )

    # Track per-cell groups for state to enable per_cell_canonical_key
    # compression on output partitions. Each merge produces a new
    # state_cell_groups derived from out_pos's source positions.
    state_cell_groups: List[List[int]] = []

    def _split_pos_into_groups(pos_list, src_groups_list):
        """Split pos_list into groups by which source group each pos belongs
        to. src_groups_list: list of (source_label, list_of_positions)."""
        # For each position, find which source group(s) it belongs to.
        # If it's in multiple, assign to the first encountered (per-cell key
        # is invariant under the chosen split as long as it's consistent).
        pos_set = set(pos_list)
        out: List[List[int]] = []
        seen = set()
        for src_positions in src_groups_list:
            grp = sorted(set(src_positions) & pos_set - seen)
            if grp:
                out.append(grp)
                seen.update(grp)
        # Any leftover positions go into a single "outside" group.
        leftover = sorted(pos_set - seen)
        if leftover:
            out.append(leftover)
        return out

    for kind, key in components:
        if kind == "cell":
            child_T_full = cell_T[key]
            child_pos = cell_boundary[key]
            child_full_c = cell_template_components
            child_orbit_T_local = cell_orbit_T[key]
            child_orbit_partitions_local = cell_orbit_partitions[key]
            child_groups_local = cell_groups_per_cell[key]
        else:
            child_T_full = junction_T[key]
            child_pos = junction_boundary[key]
            child_full_c = junction_template_components
            # Junction template uncompressed (typically small, no benefit).
            child_orbit_T_local = child_T_full
            child_orbit_partitions_local = {P: [P] for P in child_T_full}
            child_groups_local = [list(child_pos)]
        for p in child_pos:
            used_so_far[p] = used_so_far.get(p, 0) + 1

        if state_T is None:
            state_T = child_T_full
            state_pos = child_pos
            state_orbit_T_local = child_orbit_T_local
            state_orbit_partitions_local = child_orbit_partitions_local
            state_cell_groups = child_groups_local
            continue

        state_set = set(state_pos)
        child_set = set(child_pos)
        all_used = state_set | child_set
        id_shared = sorted(state_set & child_set)
        junction_shared: List[int] = []
        out_pos = sorted(
            p for p in all_used
            if used_so_far[p] < pos_remaining_uses[p]
        )

        # Compute out_cell_groups: split out_pos by source (state groups
        # first, then child groups).
        all_src_groups = list(state_cell_groups) + list(child_groups_local)
        out_cell_groups = _split_pos_into_groups(out_pos, all_src_groups)

        # Use compressed merge ONLY if all cells support per-cell
        # compression. Otherwise fall back to uncompressed merge (correct
        # but slower, needed for shared-anchor cells like K_3 M_2 grids).
        if all_cells_compressible:
            try:
                new_state_T = _merge_cells_corrected_compressed(
                    state_orbit_T_local, state_orbit_partitions_local,
                    state_pos, state_cell_groups,
                    child_orbit_T_local, child_orbit_partitions_local,
                    child_pos, child_groups_local,
                    junction_shared=junction_shared,
                    id_shared=id_shared,
                    out_pos=out_pos,
                    out_cell_groups=out_cell_groups,
                    child_full_components=child_full_c,
                )
                state_orbit_T_local = new_state_T
                state_orbit_partitions_local = {
                    P: [per_cell_orbit_rep(P, out_cell_groups)]
                    if out_cell_groups else [P]
                    for P in new_state_T
                }
                state_T = new_state_T
            except (ValueError, IndexError):
                state_T = _merge_cells_corrected(
                    state_T, state_pos,
                    child_T_full, child_pos,
                    junction_shared=junction_shared,
                    id_shared=id_shared,
                    out_pos=out_pos,
                    child_full_components=child_full_c,
                )
                state_orbit_T_local = state_T
                state_orbit_partitions_local = {P: [P] for P in state_T}
                out_cell_groups = []
        else:
            state_T = _merge_cells_corrected(
                state_T, state_pos,
                child_T_full, child_pos,
                junction_shared=junction_shared,
                id_shared=id_shared,
                out_pos=out_pos,
                child_full_components=child_full_c,
            )
            state_orbit_T_local = state_T
            state_orbit_partitions_local = {P: [P] for P in state_T}
            out_cell_groups = []
        state_pos = out_pos
        state_cell_groups = out_cell_groups

    if state_T is None:
        return TuttePolynomial.zero()
    result = TuttePolynomial.zero()
    for P, val in state_T.items():
        result = result + val
    return result
