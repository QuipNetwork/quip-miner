"""Cell-quotient interleaved Hamiltonian-cell DP for grid topologies.

For graphs whose hierarchical decomposition has CELL-QUOTIENT topology
of a 2D grid (with possible cycles), this DP computes T(graph) by:

1. Walking a Hamiltonian path through cells in cell-quotient.
2. At each path step: convolve junction (consuming one state group),
   then convolve new cell (adding new groups to state).
3. At each closing (non-tree) edge: apply cycle DP's proven 2-step
   close formula (convolve at one endpoint with FRESH labels for
   junction's B-side, then identification via union-find).

State is compressed via per-cell structural orbit canonicalization
(block-shape multiset over per-cell anchor groups) — captures S_n^N
joint aut acting independently on each cell's anchor positions.

Junction is compressed via full Aut (via `build_relabel_aut` with
split preservation).

Generic across cell template + junction connectivity. Validated via
the same tests that cover cycle DP plus grid-specific synthetic cases.

Key insight: state size stays bounded by `2 × cell_anchors`
throughout, regardless of grid dimensions. For Cm3 (3×3 K_{4,4} grid):
state has at most 2 K_{4,4} groups = 8 verts → Bell(8) = 4140 partitions
→ ~100 per-cell orbits. Avoids the row-DP's same-boundary convolve wall.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from ..graph import Graph
from ..polynomial import TuttePolynomial
from ..roots.aut_orbit import (aut_compress_t_rooted, aut_compress_t_rooted_per_cell,
                        build_relabel_aut, per_cell_canonical_key,
                        per_cell_orbit_rep, per_cell_orbit_size)
from ..roots.cell_quotient_helpers import (components_touching, orbit_convolve,
                                    precompute_M_table)
from ..roots.rooted_tutte import (all_partitions, divide_by_x_minus_1_power,
                           relabel_partition_dict, t_rooted_cached)


def hamiltonian_path_grid(
    rows: int, cols: int, order: str = "row_zigzag",
) -> List[Tuple[int, int]]:
    """Hamiltonian path through (rows × cols) grid.

    Args:
        order: path traversal style.
          - "row_zigzag": standard row-major zigzag (default).
          - "col_zigzag": column-major zigzag (transpose of row).
            For tall grids (rows > cols) this minimizes peak state size
            since the snake fits within fewer columns.
    """
    path: List[Tuple[int, int]] = []
    if order == "col_zigzag":
        for c in range(cols):
            if c % 2 == 0:
                for r in range(rows):
                    path.append((r, c))
            else:
                for r in range(rows - 1, -1, -1):
                    path.append((r, c))
    else:  # row_zigzag default
        for r in range(rows):
            if r % 2 == 0:
                for c in range(cols):
                    path.append((r, c))
            else:
                for c in range(cols - 1, -1, -1):
                    path.append((r, c))
    return path


def grid_path_and_closing_edges(
    rows: int, cols: int, torus_h: bool = False, torus_v: bool = False,
    order: str = "row_zigzag",
) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int], str]],
           List[Tuple[Tuple[int, int], Tuple[int, int], str]],
           List[Tuple[int, int]]]:
    """Partition all grid edges into Hamiltonian-path edges + closing edges.

    Returns (path_edges, closing_edges, ham_path).
    """
    ham = hamiltonian_path_grid(rows, cols, order=order)
    path_edge_set: set = set()
    for i in range(len(ham) - 1):
        a, b = ham[i], ham[i + 1]
        path_edge_set.add((min(a, b), max(a, b)))

    all_edges: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                all_edges.append(((r, c), (r, c + 1), "horiz"))
            elif torus_h and cols > 2:
                all_edges.append(((r, c), (r, 0), "horiz"))
            if r + 1 < rows:
                all_edges.append(((r, c), (r + 1, c), "vert"))
            elif torus_v and rows > 2:
                all_edges.append(((r, c), (0, c), "vert"))

    path_edges_ordered: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []
    for i in range(len(ham) - 1):
        a, b = ham[i], ham[i + 1]
        for (ea, eb, d) in all_edges:
            if {ea, eb} == {a, b}:
                path_edges_ordered.append((a, b, d))
                break

    closing_edges: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []
    for (ea, eb, d) in all_edges:
        if (min(ea, eb), max(ea, eb)) not in path_edge_set:
            closing_edges.append((ea, eb, d))

    return path_edges_ordered, closing_edges, ham


def _apply_identification_and_compute_outcome(
    P: Tuple[Tuple[int, ...], ...],
    pos_cB_existing: List[int],
    pos_cB_FRESH: List[int],
    a_size: int,
    new_group_positions: set,
    new_state_cell_groups: List[List[int]],
) -> Tuple[int, Tuple]:
    """Apply identification chain to P and compute (actually_same, new_canonical).

    Returns (actually_same, new_canonical) for the partition obtained by:
    1. Building union-find from P's blocks.
    2. Identifying pos_cB_existing[i] ↔ pos_cB_FRESH[i] for i in range(a_size).
    3. Restricting to positions in new_group_positions (drops both pos_cB_existing
       and pos_cB_FRESH).
    4. Computing per-cell canonical key over new_state_cell_groups.
    """
    parent: Dict[int, int] = {}
    for block in P:
        if not block:
            continue
        rep = block[0]
        for v in block:
            parent[v] = rep

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(u: int, v: int) -> int:
        if u not in parent:
            parent[u] = u
        if v not in parent:
            parent[v] = v
        ru, rv = _find(u), _find(v)
        if ru != rv:
            parent[max(ru, rv)] = min(ru, rv)
            return 1
        return 0

    n_merges = 0
    for i in range(a_size):
        n_merges += _union(pos_cB_existing[i], pos_cB_FRESH[i])
    actually_same = a_size - n_merges

    block_of_rep: Dict[int, List[int]] = defaultdict(list)
    for v in new_group_positions:
        if v in parent:
            r = _find(v)
        else:
            r = v
        block_of_rep[r].append(v)
    P_new = tuple(sorted(
        tuple(sorted(b)) for b in block_of_rep.values() if b
    ))
    new_canonical = (
        per_cell_canonical_key(P_new, new_state_cell_groups)
        if new_state_cell_groups else ()
    )
    return actually_same, new_canonical


def _close_step2_bell_n(
    out_boundary_positions: List[int],
    out_state_cell_groups_close: List[List[int]],
    new_state_cell_groups: List[List[int]],
    new_group_positions: set,
    pos_cB_existing: List[int],
    pos_cB_FRESH: List[int],
    a_size: int,
    state_after_close1: Dict[Tuple, TuttePolynomial],
    xy_powers: List[TuttePolynomial],
    state_new: Dict[Tuple, TuttePolynomial],
) -> None:
    """Direct Bell(N) enumeration over the full out boundary.

    For each P over (state_extra + FRESH), look up its per-cell canonical
    in state_after_close1, compute (as, new_canonical), accumulate.

    Fast for small N (≤ 12). Used in the Cm2-like single-closing case.
    """
    for p_list in all_partitions(list(out_boundary_positions)):
        P = tuple(sorted(tuple(sorted(b)) for b in p_list))
        P_canonical = per_cell_canonical_key(P, out_state_cell_groups_close)
        intermediate_val = state_after_close1.get(P_canonical)
        if intermediate_val is None:
            continue

        actually_same, new_canonical = _apply_identification_and_compute_outcome(
            P, pos_cB_existing, pos_cB_FRESH, a_size,
            new_group_positions, new_state_cell_groups,
        )

        contribution = xy_powers[actually_same] * intermediate_val
        state_new[new_canonical] = state_new[new_canonical] + contribution


def _close_step2_sigma_idr(
    out_state_cell_groups_close: List[List[int]],
    new_state_cell_groups: List[List[int]],
    new_group_positions: set,
    pos_cB_existing: List[int],
    pos_cB_FRESH: List[int],
    existing_group_idx: int,
    fresh_group_idx: int,
    a_size: int,
    state_after_close1: Dict[Tuple, TuttePolynomial],
    xy_powers: List[TuttePolynomial],
    state_new: Dict[Tuple, TuttePolynomial],
) -> None:
    """σ_idr enumeration: for each orbit in state_after_close1, enumerate
    permutations of identification-relevant positions only.

    Math: actually_same and new_canonical depend only on σ_idr (S_existing ×
    S_FRESH); other-cell perms σ_other only permute within other groups
    which preserves the per-cell canonical and doesn't touch identification
    pairs.

    For each orbit K with rep partition P_rep:
      Σ_{P in orbit K} (xy)^as(P) × intermediate_val per new_canonical bucket
      = intermediate_val × (orbit_size(K) / |S_idr|) × Σ_σ_idr (xy)^as

    Required when out_boundary > 12 verts (Bell(N) intractable).
    """
    from itertools import permutations
    from math import factorial

    n_existing = len(pos_cB_existing)
    n_fresh = len(pos_cB_FRESH)
    s_idr_size = factorial(n_existing) * factorial(n_fresh)

    perms_existing = list(permutations(range(n_existing)))
    perms_fresh = list(permutations(range(n_fresh)))

    for K_canonical, intermediate_val in state_after_close1.items():
        rep = per_cell_orbit_rep(K_canonical, out_state_cell_groups_close)
        orbit_size_K = per_cell_orbit_size(K_canonical, out_state_cell_groups_close)

        # Enumerate σ_idr ∈ S_existing × S_FRESH, accumulate (as, new_canonical) counts.
        outcome_counts: Dict[Tuple[int, Tuple], int] = defaultdict(int)
        for sigma_e in perms_existing:
            for sigma_f in perms_fresh:
                # Build position remap: within idr cell groups only.
                position_map = {
                    pos_cB_existing[i]: pos_cB_existing[sigma_e[i]]
                    for i in range(n_existing)
                }
                position_map.update({
                    pos_cB_FRESH[i]: pos_cB_FRESH[sigma_f[i]]
                    for i in range(n_fresh)
                })

                P_after_σ = tuple(sorted(
                    tuple(sorted(position_map.get(v, v) for v in block))
                    for block in rep
                ))

                actually_same, new_canonical = _apply_identification_and_compute_outcome(
                    P_after_σ, pos_cB_existing, pos_cB_FRESH, a_size,
                    new_group_positions, new_state_cell_groups,
                )

                outcome_counts[(actually_same, new_canonical)] += 1

        # Multiplicity: each σ_idr corresponds to (orbit_size_K / s_idr_size) orbit members.
        total_count_check = sum(outcome_counts.values())
        assert total_count_check == s_idr_size, (
            f"σ_idr enumeration count mismatch: {total_count_check} != {s_idr_size}"
        )
        # multiplicity = orbit_size_K / s_idr_size; must be integer for integer polys.
        if (orbit_size_K * 1) % s_idr_size != 0:
            # Fall back to enumerating ALL orbit members directly.
            # This handles shape-symmetric orbits where S_idr stab is non-trivial.
            raise NotImplementedError(
                f"Orbit {K_canonical} has non-integer multiplicity "
                f"{orbit_size_K}/{s_idr_size}; needs direct orbit enumeration."
            )
        multiplicity = orbit_size_K // s_idr_size

        for (as_value, new_canonical), count in outcome_counts.items():
            scaled_count = count * multiplicity
            if scaled_count == 0:
                continue
            contribution = scaled_count * xy_powers[as_value] * intermediate_val
            state_new[new_canonical] = state_new[new_canonical] + contribution


def _process_closing_step(
    *,
    a_idx: int,
    b_idx: int,
    g_a: int,
    g_b: int,
    close_id: int,
    T_junction,
    junction_template: Graph,
    junction_anchors_A: List[int],
    junction_anchors_B: List[int],
    junction_c_J: int,
    state_orbit_T: Dict[Tuple, TuttePolynomial],
    state_cell_groups: List[List[int]],
    state_group_meta: List[Tuple[int, int]],
    state_orbit_partitions: Dict[Tuple, List[Tuple]],
    total_div: int,
    state_is_sum_weighted: bool,
    keep_merged: bool,
    xy_powers_cache: Dict[int, List[TuttePolynomial]],
    xy_minus_1: TuttePolynomial,
    stats: Dict[str, float],
    verbose: bool,
    _force_sigma_idr: bool,
) -> Tuple[
    Dict[Tuple, TuttePolynomial],  # state_orbit_T
    List[List[int]],               # state_cell_groups
    List[Tuple[int, int]],         # state_group_meta
    Dict[Tuple, List[Tuple]],      # state_orbit_partitions
    int,                           # total_div
    bool,                          # state_is_sum_weighted
]:
    """Process one closing edge via 2-step cycle DP formula.

    Args:
        a_idx, b_idx: cell indices in Hamiltonian path order.
        g_a, g_b: anchor group ids on each side.
        close_id: unique tag (used to generate FRESH position labels).
        keep_merged: if True, after identification keep pos_cB_existing
            in state under its (b_idx, g_b) tag (the merged-group
            representative). Required when (b_idx, g_b) has more closings
            pending. The "merged" group represents the common vertex
            after identifying pos_cB_existing[i] = pos_cA[i] in G; its
            T_rooted carries the joint partition state for both endpoints.
        state_is_sum_weighted: whether state_orbit_T is SUM-weighted (after
            a prior closing) or per-rep (initial / post-path-DP). Affects
            whether to pass state_cell_anchor_groups to precompute_M_table.

    Returns updated state tuple. After this step, state_is_sum_weighted=True
    (closing always produces sum-weighted state).
    """
    a_size = len(junction_anchors_B)

    # Find positions in current state.
    a_state_idx: Optional[int] = None
    b_state_idx: Optional[int] = None
    for i, meta in enumerate(state_group_meta):
        if meta == (a_idx, g_a):
            a_state_idx = i
        if meta == (b_idx, g_b):
            b_state_idx = i
    if a_state_idx is None or b_state_idx is None:
        raise RuntimeError(
            f"Closing edge {close_id}: missing state group "
            f"({a_idx},{g_a}={a_state_idx}) or ({b_idx},{g_b}={b_state_idx})"
        )
    pos_cA = list(state_cell_groups[a_state_idx])
    pos_cB_existing = list(state_cell_groups[b_state_idx])

    # ----- Step 1: convolve at shared = pos_cA, junction's B-side = FRESH -----
    pos_cB_FRESH = [99000000 + 100 * close_id + i for i in range(a_size)]
    junction_LR = list(junction_anchors_A) + list(junction_anchors_B)
    junction_pos = list(pos_cA) + list(pos_cB_FRESH)
    map_close = {junction_LR[i]: junction_pos[i] for i in range(len(junction_LR))}
    closing_partition = relabel_partition_dict(T_junction, map_close)
    closing_aut = build_relabel_aut(
        junction_template, junction_LR, junction_pos,
        preserve_split_index=len(junction_anchors_A),
    )
    t = time.perf_counter()
    closing_orbit_T, closing_orbit_partitions = aut_compress_t_rooted(
        closing_partition, closing_aut,
    )
    stats["compress"] += time.perf_counter() - t

    # State_extra = state minus pos_cA (the convolved end).
    state_extra_close: List[int] = []
    for i, g in enumerate(state_cell_groups):
        if i != a_state_idx:
            state_extra_close.extend(g)

    # Output: drop pos_cA; add pos_cB_FRESH (will be identified in step 2).
    out_state_cell_groups_close: List[List[int]] = []
    out_state_group_meta_close: List[Tuple[int, int]] = []
    for i, g in enumerate(state_cell_groups):
        if i != a_state_idx:
            out_state_cell_groups_close.append(list(g))
            out_state_group_meta_close.append(state_group_meta[i])
    out_state_cell_groups_close.append(list(pos_cB_FRESH))
    # Synthetic meta tag for FRESH (won't be looked up).
    out_state_group_meta_close.append((-1 - close_id, 0))

    t = time.perf_counter()
    # When state is sum-weighted, pass state_cell_anchor_groups=None so
    # n_state = 1 (no orbit-size multiplication).
    M_close = precompute_M_table(
        state_orbit_partitions, closing_orbit_partitions,
        shared_boundary=pos_cA, extra_boundary=pos_cB_FRESH,
        out_aut_group=[],
        state_extra_boundary=state_extra_close,
        out_cell_anchor_groups=out_state_cell_groups_close,
        state_cell_anchor_groups=None if state_is_sum_weighted else state_cell_groups,
    )
    stats["M_precompute"] += time.perf_counter() - t

    out_orbit_partitions_close: Dict[Tuple, List[Tuple]] = {}
    for (Os, Oj, Oo) in M_close.keys():
        if Oo not in out_orbit_partitions_close:
            out_orbit_partitions_close[Oo] = [
                per_cell_orbit_rep(Oo, out_state_cell_groups_close)
            ]
    seen_out_close: Dict[Tuple, int] = {
        Oo: per_cell_orbit_size(Oo, out_state_cell_groups_close)
        for Oo in out_orbit_partitions_close
    }
    t = time.perf_counter()
    state_after_close1 = orbit_convolve(
        state_orbit_T, closing_orbit_T, M_close, seen_out_close,
    )
    stats["convolve"] += time.perf_counter() - t
    stats["close_step1"] += time.perf_counter() - t
    total_div += len(junction_anchors_B) - junction_c_J

    # ----- Step 2: identification (pos_cB_existing[i] = pos_cB_FRESH[i]) -----
    fresh_idx = len(out_state_cell_groups_close) - 1
    if b_state_idx > a_state_idx:
        b_idx_in_close = b_state_idx - 1
    else:
        b_idx_in_close = b_state_idx
    assert out_state_cell_groups_close[b_idx_in_close] == pos_cB_existing, (
        f"Position mismatch during cycle close: "
        f"{out_state_cell_groups_close[b_idx_in_close]} vs {pos_cB_existing}"
    )

    if a_size not in xy_powers_cache:
        powers = [TuttePolynomial.one()]
        for _ in range(a_size):
            powers.append(powers[-1] * xy_minus_1)
        xy_powers_cache[a_size] = powers
    xy_powers = xy_powers_cache[a_size]

    # Build new state_cell_groups.
    # If keep_merged: keep pos_cB_existing under (b_idx, g_b) tag (the merged
    # vertex represents pos_cA = pos_cB_existing in the post-identification G).
    # Always drop pos_cB_FRESH (those are the synthetic step-1 labels).
    new_state_cell_groups: List[List[int]] = []
    new_state_group_meta: List[Tuple[int, int]] = []
    for i, g in enumerate(out_state_cell_groups_close):
        if i == fresh_idx:
            continue
        if i == b_idx_in_close and not keep_merged:
            continue
        new_state_cell_groups.append(list(g))
        new_state_group_meta.append(out_state_group_meta_close[i])

    new_group_positions = {p for g in new_state_cell_groups for p in g}

    # state_new accumulates {new_per_cell_canonical → polynomial}.
    state_new: Dict[Tuple, TuttePolynomial] = defaultdict(lambda: TuttePolynomial.zero())

    BELL_N_THRESHOLD = 12  # Bell(12) ~ 4M

    out_boundary_positions = list(state_extra_close) + list(pos_cB_FRESH)

    t = time.perf_counter()
    use_bell_n = (not _force_sigma_idr) and (len(out_boundary_positions) <= BELL_N_THRESHOLD)
    if use_bell_n:
        _close_step2_bell_n(
            out_boundary_positions=out_boundary_positions,
            out_state_cell_groups_close=out_state_cell_groups_close,
            new_state_cell_groups=new_state_cell_groups,
            new_group_positions=new_group_positions,
            pos_cB_existing=pos_cB_existing,
            pos_cB_FRESH=pos_cB_FRESH,
            a_size=a_size,
            state_after_close1=state_after_close1,
            xy_powers=xy_powers,
            state_new=state_new,
        )
    else:
        _close_step2_sigma_idr(
            out_state_cell_groups_close=out_state_cell_groups_close,
            new_state_cell_groups=new_state_cell_groups,
            new_group_positions=new_group_positions,
            pos_cB_existing=pos_cB_existing,
            pos_cB_FRESH=pos_cB_FRESH,
            existing_group_idx=b_idx_in_close,
            fresh_group_idx=fresh_idx,
            a_size=a_size,
            state_after_close1=state_after_close1,
            xy_powers=xy_powers,
            state_new=state_new,
        )
    stats["close_step2"] += time.perf_counter() - t

    total_div += a_size

    state_orbit_T = dict(state_new)
    state_orbit_partitions = {
        Os: [per_cell_orbit_rep(Os, new_state_cell_groups)] if new_state_cell_groups else [()]
        for Os in state_orbit_T
    }

    if verbose:
        kept_str = " [keep_merged]" if keep_merged else ""
        print(f"  close edge {close_id} ({a_idx},{g_a})-({b_idx},{g_b}){kept_str}: "
              f"{len(state_orbit_T)} orbits, "
              f"{sum(len(g) for g in new_state_cell_groups)} verts, "
              f"total_div={total_div}",
              file=sys.stderr)

    return (
        state_orbit_T,
        new_state_cell_groups,
        new_state_group_meta,
        state_orbit_partitions,
        total_div,
        True,  # state is now sum-weighted
    )


def compute_grid_dp_interleaved(
    cell_template: Graph,
    cell_anchor_groups_template: Dict[int, List[int]],
    junction_template: Graph,
    junction_anchors_A: List[int],
    junction_anchors_B: List[int],
    rows: int,
    cols: int,
    horiz_groups: Tuple[int, int],
    vert_groups: Tuple[int, int],
    torus_h: bool = False,
    torus_v: bool = False,
    verbose: bool = False,
    order: str = "row_zigzag",
    _force_sigma_idr: bool = False,
) -> Tuple[TuttePolynomial, Dict[str, float]]:
    """Interleaved Hamiltonian-cell DP for 2D grid topology.

    Args:
        cell_template: One cell (e.g., K_{4,4}).
        cell_anchor_groups_template: dict group_id → cell template vertices
            for that group. e.g., {0: [0,1,2,3], 1: [4,5,6,7]} for K_{4,4}.
        junction_template: One junction (e.g., M_4).
        junction_anchors_A, junction_anchors_B: junction anchor split.
        rows, cols: grid dimensions.
        horiz_groups: (cell_a_group_id, cell_b_group_id) for horizontal junctions.
        vert_groups: same for vertical junctions.
        torus_h, torus_v: torus closing edges (rare).
        order: Hamiltonian path traversal — "row_zigzag" (default) or
            "col_zigzag". Choice affects peak state size; for grids that
            are wider than tall, row_zigzag tends to grow state across the
            top row; for tall grids col_zigzag is symmetrically better.
        _force_sigma_idr: testing-only flag to force the σ_idr enumeration
            path even when the Bell(N) path would be tractable. Used to
            cross-validate the two paths produce the same polynomial.

    Returns:
        (T_polynomial, stats_dict)
    """
    a = len(cell_anchor_groups_template[vert_groups[0]])
    b = len(junction_anchors_B)

    stats: Dict[str, float] = {
        "t_rooted": 0.0, "compress": 0.0, "M_precompute": 0.0,
        "convolve": 0.0, "close_step1": 0.0, "close_step2": 0.0,
        "total": 0.0,
    }
    t0 = time.perf_counter()

    # T_rooted of cell + junction (cached).
    t = time.perf_counter()
    template_anchors = list(cell_anchor_groups_template[0]) + list(cell_anchor_groups_template[1])
    T_cell = t_rooted_cached(cell_template, template_anchors)
    T_junction = t_rooted_cached(
        junction_template,
        list(junction_anchors_A) + list(junction_anchors_B),
    )
    stats["t_rooted"] = time.perf_counter() - t

    # c_J for divisors.
    junction_c_J = components_touching(junction_template, list(junction_anchors_A))
    cell_c_J_per_group: Dict[int, int] = {}
    for gid, gverts in cell_anchor_groups_template.items():
        cell_c_J_per_group[gid] = components_touching(cell_template, gverts)

    # Hamiltonian path + closing edges.
    path_edges, closing_edges, ham = grid_path_and_closing_edges(
        rows, cols, torus_h=torus_h, torus_v=torus_v, order=order,
    )
    cell_to_idx = {coord: i for i, coord in enumerate(ham)}

    def cell_positions(cell_idx: int, group_id: int) -> List[int]:
        base = 100000 * (cell_idx + 1)
        return [base + group_id * 1000 + i
                for i in range(len(cell_anchor_groups_template[group_id]))]

    def edge_groups(direction: str) -> Tuple[int, int]:
        return horiz_groups if direction == "horiz" else vert_groups

    # ===== Initialize state with cell 0 (full boundary) =====
    cell0_idx = 0
    cell0_pos_per_group = {
        gid: cell_positions(cell0_idx, gid)
        for gid in cell_anchor_groups_template
    }
    template_to_pos = {}
    for gid, tverts in cell_anchor_groups_template.items():
        for i, tv in enumerate(tverts):
            template_to_pos[tv] = cell0_pos_per_group[gid][i]
    state_partition = relabel_partition_dict(T_cell, template_to_pos)

    # state_cell_groups: list of (cell_idx, group_id, position_list).
    # Stored as parallel lists for the per-cell canonicalizer.
    state_cell_groups: List[List[int]] = [
        list(cell0_pos_per_group[gid]) for gid in sorted(cell_anchor_groups_template)
    ]
    state_group_meta: List[Tuple[int, int]] = [
        (cell0_idx, gid) for gid in sorted(cell_anchor_groups_template)
    ]

    t = time.perf_counter()
    state_orbit_T, state_orbit_partitions = aut_compress_t_rooted_per_cell(
        state_partition, state_cell_groups,
    )
    stats["compress"] += time.perf_counter() - t

    if verbose:
        print(f"  init cell 0: {len(state_orbit_T)} orbits, "
              f"{sum(len(g) for g in state_cell_groups)} verts",
              file=sys.stderr)

    total_div = 0

    # ===== Pre-compute usage tracking for anchor sharing =====
    # usage_remaining[(cell_idx, group_id)] = number of junctions (path + closing)
    # still to process that touch this group. Decrement after each junction step.
    # When usage > 0 after a junction, the group must persist in state.
    usage_remaining: Dict[Tuple[int, int], int] = defaultdict(int)
    for (ca, cb, d) in list(path_edges) + list(closing_edges):
        ai = cell_to_idx[ca]
        bi = cell_to_idx[cb]
        ga, gb = edge_groups(d)
        usage_remaining[(ai, ga)] += 1
        usage_remaining[(bi, gb)] += 1

    visited_cells: set = {0}  # cells whose "other groups" have been added via cell-add

    def find_state_group_idx(cell_idx: int, group_id: int) -> Optional[int]:
        for i, meta in enumerate(state_group_meta):
            if meta == (cell_idx, group_id):
                return i
        return None

    # ===== Pending closings tracker for inline dispatch =====
    pending_closings = list(range(len(closing_edges)))  # indices into closing_edges

    def find_inline_closing() -> Optional[int]:
        """Find a pending closing edge whose BOTH endpoints are in state."""
        for ci in pending_closings:
            ca, cb, d = closing_edges[ci]
            ai = cell_to_idx[ca]
            bi = cell_to_idx[cb]
            ga, gb = edge_groups(d)
            if (find_state_group_idx(ai, ga) is not None
                    and find_state_group_idx(bi, gb) is not None):
                return ci
        return None

    # xy_powers_cache: shared between inline-close and post-path-close steps.
    x_minus_1 = TuttePolynomial.x() + (-1) * TuttePolynomial.one()
    y_minus_1 = TuttePolynomial.y() + (-1) * TuttePolynomial.one()
    xy_minus_1_for_close = x_minus_1 * y_minus_1
    xy_powers_cache: Dict[int, List[TuttePolynomial]] = {}
    state_is_sum_weighted = False  # toggled True after first closing

    # ===== Process path edges (junction step + cell add interleaved) =====
    for step_k, (cell_a, cell_b, direction) in enumerate(path_edges):
        a_idx = cell_to_idx[cell_a]
        b_idx = cell_to_idx[cell_b]
        g_a, g_b = edge_groups(direction)

        # Check if this junction has both endpoints already in state (= closing-style).
        # Happens with anchor sharing in some path orderings.
        b_state_idx_pre = find_state_group_idx(b_idx, g_b)
        if b_state_idx_pre is not None:
            raise NotImplementedError(
                f"Path edge ({cell_a},{cell_b}) {direction} has cell_b's group "
                f"already in state. Path-as-closing dispatch not yet implemented; "
                f"choose a Hamiltonian path order that introduces new cells."
            )

        # ----- Standard Junction step -----
        a_state_idx = find_state_group_idx(a_idx, g_a)
        if a_state_idx is None:
            raise RuntimeError(
                f"Path step {step_k}: cell {cell_a} group {g_a} not in state"
            )
        pos_jA = list(state_cell_groups[a_state_idx])
        pos_jB = cell_positions(b_idx, g_b)

        # keep_shared decision: if (cell_a, g_a) has more uses pending after
        # this junction, keep pos_jA in state for future junctions.
        keep_pos_jA = usage_remaining[(a_idx, g_a)] > 1

        junction_LR = list(junction_anchors_A) + list(junction_anchors_B)
        junction_pos = list(pos_jA) + list(pos_jB)
        map_J = {junction_LR[i]: junction_pos[i] for i in range(len(junction_LR))}
        junction_partition = relabel_partition_dict(T_junction, map_J)
        junction_aut = build_relabel_aut(
            junction_template, junction_LR, junction_pos,
            preserve_split_index=len(junction_anchors_A),
        )
        t = time.perf_counter()
        junction_orbit_T, junction_orbit_partitions = aut_compress_t_rooted(
            junction_partition, junction_aut,
        )
        stats["compress"] += time.perf_counter() - t

        # State_extra: if keep_pos_jA, pos_jA stays as a state extra; else excluded.
        state_extra_pos: List[int] = []
        for i, g in enumerate(state_cell_groups):
            if i != a_state_idx or keep_pos_jA:
                state_extra_pos.extend(g)

        # Output state cell groups.
        out_state_cell_groups: List[List[int]] = []
        out_state_group_meta: List[Tuple[int, int]] = []
        for i, g in enumerate(state_cell_groups):
            if i != a_state_idx or keep_pos_jA:
                out_state_cell_groups.append(list(g))
                out_state_group_meta.append(state_group_meta[i])
        out_state_cell_groups.append(list(pos_jB))
        out_state_group_meta.append((b_idx, g_b))

        # When keep_pos_jA=True, the convolve must keep pos_jA in output too
        # (via keep_shared semantics in precompute_M_table).
        t = time.perf_counter()
        if keep_pos_jA:
            # Use keep_shared=True path: shared positions stay in output.
            # Need state_extra_boundary to NOT include pos_jA (it's the shared).
            state_extra_for_M: List[int] = []
            for i, g in enumerate(state_cell_groups):
                if i != a_state_idx:
                    state_extra_for_M.extend(g)
            M_j = precompute_M_table(
                state_orbit_partitions, junction_orbit_partitions,
                shared_boundary=pos_jA, extra_boundary=pos_jB,
                out_aut_group=[],
                state_extra_boundary=state_extra_for_M,
                keep_shared=True,
                out_cell_anchor_groups=out_state_cell_groups,
                state_cell_anchor_groups=state_cell_groups,
            )
        else:
            M_j = precompute_M_table(
                state_orbit_partitions, junction_orbit_partitions,
                shared_boundary=pos_jA, extra_boundary=pos_jB,
                out_aut_group=[],
                state_extra_boundary=state_extra_pos,
                out_cell_anchor_groups=out_state_cell_groups,
                state_cell_anchor_groups=state_cell_groups,
            )
        stats["M_precompute"] += time.perf_counter() - t

        # Output orbit sizes (analytical).
        seen_out: Dict[Tuple, int] = {}
        for (Os, Oj, Oo) in M_j.keys():
            if Oo not in seen_out:
                seen_out[Oo] = per_cell_orbit_size(Oo, out_state_cell_groups)

        t = time.perf_counter()
        state_orbit_T = orbit_convolve(
            state_orbit_T, junction_orbit_T, M_j, seen_out,
        )
        stats["convolve"] += time.perf_counter() - t

        # Update state.
        state_cell_groups = out_state_cell_groups
        state_group_meta = out_state_group_meta
        state_orbit_partitions = {
            Os: [per_cell_orbit_rep(Os, state_cell_groups)]
            for Os in state_orbit_T
        }
        # Divisor: only when shared is consumed (keep_pos_jA=False), the
        # convolution adds (b - c_J) to total_div. With keep_shared=True, the
        # shared positions are RETAINED so no rank shift from this convolve.
        if not keep_pos_jA:
            total_div += b - junction_c_J

        # Decrement usage for both endpoints (this junction touched both).
        usage_remaining[(a_idx, g_a)] -= 1
        usage_remaining[(b_idx, g_b)] -= 1

        if verbose:
            print(f"  path step {step_k+1} junction {cell_a}->{cell_b}/{direction}"
                  f"{' [keep_shared cell_a]' if keep_pos_jA else ''}: "
                  f"{len(state_orbit_T)} orbits, "
                  f"{sum(len(g) for g in state_cell_groups)} verts",
                  file=sys.stderr)

        # ----- Cell add step (only on first encounter of cell_b) -----
        if b_idx in visited_cells:
            # Cell already had its T_rooted convolved on a prior visit; skip.
            # But the junction added pos_jB which we should marginalize unless
            # there are more uses of (cell_b, g_b).
            keep_pos_jB_no_celladd = usage_remaining[(b_idx, g_b)] > 0
            if not keep_pos_jB_no_celladd:
                # Marginalize pos_jB. For now: implement via single-group cell
                # convolve (no other group). Actually a no-op here would mean
                # pos_jB lingers — TODO for follow-up.
                pass
            # Continue without cell-add; check inline closings below.
        else:
            visited_cells.add(b_idx)

            # Find cell_b's incoming group in state (just added by junction).
            b_in_state_idx = find_state_group_idx(b_idx, g_b)
            if b_in_state_idx is None:
                raise RuntimeError(f"Internal error: b's incoming group not in state")

            # Determine cell_b's OTHER groups (not yet in state).
            other_groups = [
                gid for gid in cell_anchor_groups_template
                if gid != g_b
            ]

            # For each "other group" (typically 1 for 2-group cells like K_{4,4}):
            # convolve cell_b's full T_rooted at shared = pos_jB.
            # If (cell_b, g_b) has more uses pending, keep pos_jB in state via
            # keep_shared=True on cell add.
            keep_pos_jB = usage_remaining[(b_idx, g_b)] > 0

            for other_g in other_groups:
                other_pos = cell_positions(b_idx, other_g)

                cell_relabel: Dict[int, int] = {}
                for i, tv in enumerate(cell_anchor_groups_template[g_b]):
                    cell_relabel[tv] = pos_jB[i]
                for i, tv in enumerate(cell_anchor_groups_template[other_g]):
                    cell_relabel[tv] = other_pos[i]
                cell_partition = relabel_partition_dict(T_cell, cell_relabel)

                t = time.perf_counter()
                cell_orbit_T, cell_orbit_partitions = aut_compress_t_rooted_per_cell(
                    cell_partition, [list(pos_jB), list(other_pos)],
                )
                stats["compress"] += time.perf_counter() - t

                # State_extra = state minus pos_jB (the convolved end).
                state_extra_cell: List[int] = []
                for i, g in enumerate(state_cell_groups):
                    if i != b_in_state_idx:
                        state_extra_cell.extend(g)

                # Output: pos_jB stays IF keep_pos_jB; add other_pos.
                out_state_cell_groups2: List[List[int]] = []
                out_state_group_meta2: List[Tuple[int, int]] = []
                for i, g in enumerate(state_cell_groups):
                    if i != b_in_state_idx or keep_pos_jB:
                        out_state_cell_groups2.append(list(g))
                        out_state_group_meta2.append(state_group_meta[i])
                out_state_cell_groups2.append(list(other_pos))
                out_state_group_meta2.append((b_idx, other_g))

                t = time.perf_counter()
                M_c = precompute_M_table(
                    state_orbit_partitions, cell_orbit_partitions,
                    shared_boundary=pos_jB, extra_boundary=other_pos,
                    out_aut_group=[],
                    state_extra_boundary=state_extra_cell,
                    keep_shared=keep_pos_jB,
                    out_cell_anchor_groups=out_state_cell_groups2,
                    state_cell_anchor_groups=state_cell_groups,
                )
                stats["M_precompute"] += time.perf_counter() - t

                seen_out2: Dict[Tuple, int] = {}
                for (Os, Oj, Oo) in M_c.keys():
                    if Oo not in seen_out2:
                        seen_out2[Oo] = per_cell_orbit_size(Oo, out_state_cell_groups2)

                t = time.perf_counter()
                state_orbit_T = orbit_convolve(
                    state_orbit_T, cell_orbit_T, M_c, seen_out2,
                )
                stats["convolve"] += time.perf_counter() - t

                state_cell_groups = out_state_cell_groups2
                state_group_meta = out_state_group_meta2
                state_orbit_partitions = {
                    Os: [per_cell_orbit_rep(Os, state_cell_groups)]
                    for Os in state_orbit_T
                }
                # Divisor: only when shared (pos_jB) is consumed.
                if not keep_pos_jB:
                    total_div += a - cell_c_J_per_group[g_b]

                # Re-find b_in_state_idx for the next iteration (if any).
                b_in_state_idx = find_state_group_idx(b_idx, other_g)

                if verbose:
                    print(f"  path step {step_k+1} cell {cell_b} (+g{other_g})"
                          f"{' [keep_shared cell_b]' if keep_pos_jB else ''}: "
                          f"{len(state_orbit_T)} orbits, "
                          f"{sum(len(g) for g in state_cell_groups)} verts",
                          file=sys.stderr)

        # ----- Inline closing dispatch -----
        # After each path step, scan pending closings; process any whose
        # endpoints are both in state. Iterate (a closing may unblock another).
        while True:
            ci = find_inline_closing()
            if ci is None:
                break
            close_a, close_b, close_d = closing_edges[ci]
            close_ai = cell_to_idx[close_a]
            close_bi = cell_to_idx[close_b]
            close_ga, close_gb = edge_groups(close_d)
            usage_remaining[(close_ai, close_ga)] -= 1
            usage_remaining[(close_bi, close_gb)] -= 1
            keep_merged_inline = usage_remaining[(close_bi, close_gb)] > 0
            (state_orbit_T, state_cell_groups, state_group_meta,
             state_orbit_partitions, total_div,
             state_is_sum_weighted) = _process_closing_step(
                a_idx=close_ai, b_idx=close_bi,
                g_a=close_ga, g_b=close_gb,
                close_id=ci, T_junction=T_junction,
                junction_template=junction_template,
                junction_anchors_A=junction_anchors_A,
                junction_anchors_B=junction_anchors_B,
                junction_c_J=junction_c_J,
                state_orbit_T=state_orbit_T,
                state_cell_groups=state_cell_groups,
                state_group_meta=state_group_meta,
                state_orbit_partitions=state_orbit_partitions,
                total_div=total_div,
                state_is_sum_weighted=state_is_sum_weighted,
                keep_merged=keep_merged_inline,
                xy_powers_cache=xy_powers_cache,
                xy_minus_1=xy_minus_1_for_close,
                stats=stats,
                verbose=verbose,
                _force_sigma_idr=_force_sigma_idr,
            )
            pending_closings.remove(ci)

    # ===== Post-path closing edges (any pending after inline dispatch) =====
    # Most graphs (Cm2 included) have no inline opportunities and process
    # all closings here. For anchor-shared multi-closing graphs (3x3 K_3,
    # Cm3) inline dispatch above handles closings as endpoints become
    # available; this loop processes whatever remains.
    for close_k, (cell_a, cell_b, direction) in enumerate(closing_edges):
        if close_k not in pending_closings:
            continue
        a_idx = cell_to_idx[cell_a]
        b_idx = cell_to_idx[cell_b]
        g_a, g_b = edge_groups(direction)
        usage_remaining[(a_idx, g_a)] -= 1
        usage_remaining[(b_idx, g_b)] -= 1
        keep_merged_post = usage_remaining[(b_idx, g_b)] > 0
        (state_orbit_T, state_cell_groups, state_group_meta,
         state_orbit_partitions, total_div,
         state_is_sum_weighted) = _process_closing_step(
            a_idx=a_idx, b_idx=b_idx, g_a=g_a, g_b=g_b,
            close_id=close_k, T_junction=T_junction,
            junction_template=junction_template,
            junction_anchors_A=junction_anchors_A,
            junction_anchors_B=junction_anchors_B,
            junction_c_J=junction_c_J,
            state_orbit_T=state_orbit_T,
            state_cell_groups=state_cell_groups,
            state_group_meta=state_group_meta,
            state_orbit_partitions=state_orbit_partitions,
            total_div=total_div,
            state_is_sum_weighted=state_is_sum_weighted,
            keep_merged=keep_merged_post,
            xy_powers_cache=xy_powers_cache,
            xy_minus_1=xy_minus_1_for_close,
            stats=stats,
            verbose=verbose,
            _force_sigma_idr=_force_sigma_idr,
        )
        pending_closings.remove(close_k)

    # ===== Marginalize remaining state =====
    # After closing edges, state_orbit_T accumulates SUM-weighted values
    # (each value already sums over partitions in its orbit). Without
    # closing edges, state_orbit_T stores per-rep values requiring orbit-size
    # multiplication for marginalization.
    if closing_edges:
        sum_T = TuttePolynomial.zero()
        for canonical, val in state_orbit_T.items():
            sum_T = sum_T + val
    else:
        sum_T = TuttePolynomial.zero()
        for canonical, val in state_orbit_T.items():
            if state_cell_groups:
                size = per_cell_orbit_size(canonical, state_cell_groups)
            else:
                size = 1
            sum_T = sum_T + size * val

    T_final = divide_by_x_minus_1_power(sum_T, total_div)
    stats["total"] = time.perf_counter() - t0
    return T_final, stats
