"""Cell-quotient path DP — cycle DP minus the close step.

For graphs whose hierarchical decomposition has cell-quotient topology
of a SIMPLE PATH (n cells, n-1 junctions, no closing edge). Examples:
- A row of cells in a grid (e.g., row 1 of Cm3)
- Pm2's 3-cell linear arrangement

Returns T_rooted indexed by partition over the path's boundary:
- cell_0's left anchors (none if first cell is path-start)
- cell_{N-1}'s right anchors (none if last cell is path-end)
- Per-cell "extra anchors" that persist throughout (e.g., vertical
  anchors when this row is part of a grid)

This is the building block for grid DP: compute T_rooted per row via
path DP, then compose rows via vertical-junction convolution (which is
itself another path DP at the row level).
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..graph import Graph
from ..polynomial import TuttePolynomial
from .aut_orbit import (
    aut_compress_t_rooted,
    aut_compress_t_rooted_per_cell,
    build_relabel_aut,
    per_cell_canonical_key,
    per_cell_orbit_rep,
    per_cell_orbit_size,
)
from .cell_anchor_adapter import CellRowSpec
from .cell_quotient_helpers import (
    components_touching,
    enumerate_partitions_cached,
    enumerate_partitions_per_orbit,
    orbit_convolve,
    precompute_M_table,
)
from .rooted_tutte import (
    relabel_partition_dict,
    t_rooted_cached,
)


# Module-level cache for compute_path_dp_grouped (compressed mode).
#
# The per-cell-compressed canonical keys are POSITION-INVARIANT (they
# encode block-shape multisets, not absolute positions). So two calls
# with identical (cell_template, cell_anchor_groups, junction_template,
# junction_anchors_A/B, cell_specs) but different `label_offset` produce
# IDENTICAL orbit_T dicts (same canonical keys, same polynomials).
#
# Cache key: a hashable spec signature.
# Cache value: (orbit_T_dict, total_div, state_cell_group_sizes).
#   orbit_T_dict is reused as-is (position-invariant).
#   state_cell_group_sizes is a tuple of group sizes; the caller
#   reconstructs the actual position lists from pos_cell + label_offset.
_PATH_DP_CACHE: Dict[Tuple, Tuple[Dict[Tuple, TuttePolynomial], int, Tuple[int, ...]]] = {}
_PATH_DP_CACHE_HITS = 0
_PATH_DP_CACHE_MISSES = 0


def clear_path_dp_cache() -> None:
    """Clear the path DP cache (use between independent benchmarks)."""
    global _PATH_DP_CACHE_HITS, _PATH_DP_CACHE_MISSES
    _PATH_DP_CACHE.clear()
    _PATH_DP_CACHE_HITS = 0
    _PATH_DP_CACHE_MISSES = 0


def path_dp_cache_stats() -> Dict[str, int]:
    """Return cache hit/miss/size stats (for tests/profiling)."""
    return {
        "hits": _PATH_DP_CACHE_HITS,
        "misses": _PATH_DP_CACHE_MISSES,
        "size": len(_PATH_DP_CACHE),
    }


def _state_groups_meta(
    pos_cell: List[Dict[int, List[int]]],
    state_cell_groups: List[List[int]],
) -> List[Optional[Tuple[int, int]]]:
    """Reverse-map each group in state_cell_groups back to (cell_idx, group_id).

    Used by the path-DP cache to store position-invariant group identity.
    On cache hit, the caller reconstructs `state_cell_groups` via
    `[pos_cell[c][g] for (c, g) in meta]`.
    """
    pos_to_cellgroup: Dict[int, Tuple[int, int]] = {}
    for cell_idx, cell_pos in enumerate(pos_cell):
        for group_id, positions in cell_pos.items():
            for p in positions:
                pos_to_cellgroup[p] = (cell_idx, group_id)
    meta: List[Optional[Tuple[int, int]]] = []
    for group in state_cell_groups:
        if not group:
            meta.append(None)
            continue
        meta.append(pos_to_cellgroup[group[0]])
    return meta


def _path_dp_cache_key(
    cell_template: Graph,
    cell_anchor_groups: Dict[int, List[int]],
    junction_template: Graph,
    junction_anchors_A: List[int],
    junction_anchors_B: List[int],
    cell_specs: List[CellRowSpec],
) -> Tuple:
    """Build a cache key invariant under label_offset."""
    cag_serialized = tuple(
        (gid, tuple(sorted(positions)))
        for gid, positions in sorted(cell_anchor_groups.items())
    )
    specs_serialized = tuple(
        (
            spec.left_group, spec.right_group,
            tuple(spec.extra_groups),
            spec.has_shared_horizontal,
        )
        for spec in cell_specs
    )
    return (
        cell_template.canonical_key(),
        cag_serialized,
        junction_template.canonical_key(),
        tuple(junction_anchors_A),
        tuple(junction_anchors_B),
        specs_serialized,
    )


def compute_path_dp(
    cell_template: Graph,
    cell_left_anchors: List[int],
    cell_right_anchors: List[int],
    cell_extra_anchors: List[int],
    junction_template: Graph,
    junction_anchors_A: List[int],
    junction_anchors_B: List[int],
    n_cells: int,
    verbose: bool = False,
) -> Tuple[Dict[Tuple, TuttePolynomial], Dict[str, float], int]:
    """Path DP for n_cells in a path with horizontal junctions.

    Args:
        cell_template: Graph for ONE cell (e.g., K_{4,4}).
        cell_left_anchors: cell vertices going to PREVIOUS junction in path.
            Empty list for the FIRST cell (no left junction).
        cell_right_anchors: cell vertices going to NEXT junction in path.
            Empty list for the LAST cell (no right junction).
        cell_extra_anchors: cell vertices that PERSIST as boundary throughout
            the path DP (e.g., vertical anchors in a row of a grid).
        junction_template: Graph for ONE junction between cells (e.g., M_4).
        junction_anchors_A: junction vertices on the "previous-cell" side.
        junction_anchors_B: junction vertices on the "next-cell" side.
        n_cells: number of cells in the path (n ≥ 1).

    Returns:
        (T_rooted_dict, stats_dict, total_div)
        where T_rooted_dict is a partition → polynomial dict (NOT yet divided
        by (x-1)^total_div — caller does that after additional convolutions),
        and total_div is the accumulated (x-1) divisor power.
    """
    assert n_cells >= 1
    a = len(cell_left_anchors)
    b = len(cell_right_anchors)
    e = len(cell_extra_anchors)
    if n_cells > 1:
        assert len(junction_anchors_A) == b
        assert len(junction_anchors_B) == a

    stats: Dict[str, float] = {
        "t_rooted": 0.0, "compress": 0.0, "M_precompute": 0.0,
        "convolve": 0.0, "enumerate": 0.0, "build_aut": 0.0, "total": 0.0,
    }
    t0 = time.perf_counter()

    cell_LR_anchors = (list(cell_left_anchors)
                       + list(cell_right_anchors)
                       + list(cell_extra_anchors))
    t = time.perf_counter()
    T_cell = t_rooted_cached(cell_template, cell_LR_anchors)
    if n_cells > 1:
        T_junction = t_rooted_cached(
            junction_template,
            list(junction_anchors_A) + list(junction_anchors_B),
        )
    stats["t_rooted"] = time.perf_counter() - t

    # Position labels: each cell gets a unique base.
    pos_cell_left: List[List[int]] = []
    pos_cell_right: List[List[int]] = []
    pos_cell_extra: List[List[int]] = []
    for k in range(n_cells):
        base = 10000 * (k + 1)
        pos_cell_left.append([base + i for i in range(a)])
        pos_cell_right.append([base + 500 + i for i in range(b)])
        pos_cell_extra.append([base + 1000 + i for i in range(e)])

    # Initialize state with cell 0.
    map_LR0 = {}
    for i in range(a):
        map_LR0[cell_left_anchors[i]] = pos_cell_left[0][i]
    for i in range(b):
        map_LR0[cell_right_anchors[i]] = pos_cell_right[0][i]
    for i in range(e):
        map_LR0[cell_extra_anchors[i]] = pos_cell_extra[0][i]
    state_partition = relabel_partition_dict(T_cell, map_LR0)

    # Initial state's aut: cell template aut restricted to preserve THREE
    # operationally-distinct anchor groups (left, right, extra).
    initial_pos = (list(pos_cell_left[0]) + list(pos_cell_right[0])
                   + list(pos_cell_extra[0]))
    # Indices into cell_LR_anchors for each group:
    # left = [0..a-1], right = [a..a+b-1], extra = [a+b..a+b+e-1]
    preserve_groups = []
    if a > 0:
        preserve_groups.append(list(range(a)))
    if b > 0:
        preserve_groups.append(list(range(a, a + b)))
    if e > 0:
        preserve_groups.append(list(range(a + b, a + b + e)))
    # AUT COMPRESSION DISABLED FOR PATH DP: per-cell aut doesn't lift to
    # multi-cell state correctly. After the first junction step, state's
    # boundary spans multiple cells; cell_template's aut would mis-compress.
    # Use identity aut throughout — no compression — to ensure correctness.
    # (Performance optimization can be reintroduced later via a multi-cell-
    # aware aut group computation.)
    cell_aut = []  # was: build_relabel_aut(cell_template, cell_LR_anchors, initial_pos, preserve_groups=preserve_groups,)
    t = time.perf_counter()
    state_orbit_T, state_orbit_partitions = aut_compress_t_rooted(
        state_partition, cell_aut,
    )
    stats["compress"] += time.perf_counter() - t

    # State boundary tracking
    state_left = pos_cell_left[0]  # NOT consumed in path DP (path-start)
    state_right = pos_cell_right[0]
    state_accumulated_extras = list(pos_cell_extra[0])  # persists throughout

    junction_c_J = (
        components_touching(junction_template, list(junction_anchors_A))
        if n_cells > 1 else 1
    )
    cell_c_J = components_touching(cell_template, list(cell_left_anchors)) if a > 0 else 1
    total_div = 0

    if verbose:
        print(f"  path DP: a={a}, b={b}, e={e}, junction_c_J={junction_c_J}",
              file=sys.stderr)

    # Path-DP through (n-1) junction-cell pairs.
    for k in range(n_cells - 1):
        # JUNCTION step: convolve state with junction at pos_jA = state_right.
        pos_jA = state_right
        pos_jB = pos_cell_left[k + 1]
        junction_LR = list(junction_anchors_A) + list(junction_anchors_B)
        junction_pos = list(pos_jA) + list(pos_jB)
        map_J = {junction_LR[i]: junction_pos[i] for i in range(len(junction_LR))}
        junction_partition = relabel_partition_dict(T_junction, map_J)
        junction_aut = []  # disabled: see cell_aut comment above
        t = time.perf_counter()
        junction_orbit_T, junction_orbit_partitions = aut_compress_t_rooted(
            junction_partition, junction_aut,
        )
        stats["compress"] += time.perf_counter() - t

        # Output state's boundary: state_left + accumulated_extras + pos_jB.
        # state_left and accumulated_extras persist (state_extra_boundary in helpers).
        out_extra = list(state_left) + list(state_accumulated_extras)
        out_anchors = out_extra + list(pos_jB)

        # No aut compression for state's output (boundary spans multiple cells'
        # positions; per-cell symmetries don't lift to the state level cleanly).
        # Identity aut: each partition is its own orbit (orbit_size = 1).
        out_aut = []
        t = time.perf_counter()
        out_orbit_partitions = enumerate_partitions_per_orbit(out_anchors, out_aut)
        stats["enumerate"] += time.perf_counter() - t
        out_orbit_sizes = {ok: len(parts) for ok, parts in out_orbit_partitions.items()}

        t = time.perf_counter()
        M_j = precompute_M_table(
            state_orbit_partitions, junction_orbit_partitions,
            shared_boundary=pos_jA, extra_boundary=pos_jB,
            out_aut_group=out_aut, state_extra_boundary=out_extra,
        )
        stats["M_precompute"] += time.perf_counter() - t

        t = time.perf_counter()
        state_orbit_T = orbit_convolve(
            state_orbit_T, junction_orbit_T, M_j, out_orbit_sizes,
        )
        stats["convolve"] += time.perf_counter() - t

        total_div += (b - junction_c_J)
        state_right = pos_jB
        state_orbit_partitions = out_orbit_partitions

        # CELL step: convolve state with next cell at pos_jB.
        cell_idx = k + 1
        pos_next_right = pos_cell_right[cell_idx]
        pos_next_extra = pos_cell_extra[cell_idx]
        map_LR2 = {}
        for i in range(a):
            map_LR2[cell_left_anchors[i]] = pos_jB[i]
        for i in range(b):
            map_LR2[cell_right_anchors[i]] = pos_next_right[i]
        for i in range(e):
            map_LR2[cell_extra_anchors[i]] = pos_next_extra[i]
        cell_partition = relabel_partition_dict(T_cell, map_LR2)
        cell_anchors_pos = list(pos_jB) + list(pos_next_right) + list(pos_next_extra)
        cell_aut2 = []  # disabled: see cell_aut comment above
        t = time.perf_counter()
        cell_orbit_T, cell_orbit_partitions = aut_compress_t_rooted(
            cell_partition, cell_aut2,
        )
        stats["compress"] += time.perf_counter() - t

        out_extra2 = list(state_left) + list(state_accumulated_extras) + list(pos_next_extra)
        out_anchors2 = out_extra2 + list(pos_next_right)
        out_aut2 = []
        t = time.perf_counter()
        out_orbit_partitions2 = enumerate_partitions_per_orbit(out_anchors2, out_aut2)
        stats["enumerate"] += time.perf_counter() - t
        out_orbit_sizes2 = {ok: len(parts) for ok, parts in out_orbit_partitions2.items()}

        t = time.perf_counter()
        # Cell's extra_boundary = right anchors (state's new right) + extras (accumulate).
        cell_extra_boundary = list(pos_next_right) + list(pos_next_extra)
        M_c = precompute_M_table(
            state_orbit_partitions, cell_orbit_partitions,
            shared_boundary=pos_jB, extra_boundary=cell_extra_boundary,
            out_aut_group=out_aut2,
            state_extra_boundary=list(state_left) + list(state_accumulated_extras),
        )
        stats["M_precompute"] += time.perf_counter() - t

        t = time.perf_counter()
        state_orbit_T = orbit_convolve(
            state_orbit_T, cell_orbit_T, M_c, out_orbit_sizes2,
        )
        stats["convolve"] += time.perf_counter() - t

        total_div += (a - cell_c_J)
        state_right = pos_next_right
        state_accumulated_extras += list(pos_next_extra)
        state_orbit_partitions = out_orbit_partitions2

        if verbose:
            print(f"  step {k+1}/{n_cells-1}: state has {len(state_orbit_T)} orbits, "
                  f"{len(state_accumulated_extras)} accumulated extras",
                  file=sys.stderr)

    stats["total"] = time.perf_counter() - t0
    # Return UNCOMPRESSED state: expand each compressed canonical to all
    # partitions in its orbit, sharing the same value (T_rooted is invariant
    # under aut). This makes the result usable by callers (e.g., grid DP)
    # that assume {partition → poly} semantics, not {orbit → poly}.
    state_T_uncompressed: Dict[Tuple, TuttePolynomial] = {}
    for canonical, val in state_orbit_T.items():
        for P in state_orbit_partitions.get(canonical, [canonical]):
            state_T_uncompressed[P] = val
    return state_T_uncompressed, stats, total_div


def compute_path_dp_grouped(
    cell_template: Graph,
    cell_anchor_groups: Dict[int, List[int]],
    junction_template: Graph,
    junction_anchors_A: List[int],
    junction_anchors_B: List[int],
    cell_specs: List[CellRowSpec],
    verbose: bool = False,
    label_offset: int = 0,
    return_pos_layout: bool = False,
    enable_per_cell_compression: bool = False,
    observer: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Dict[Tuple, TuttePolynomial], Dict[str, float], int]:
    """Generic path DP supporting per-cell anchor groups + shared-horizontal.

    Each cell in the path is described by a CellRowSpec naming WHICH groups
    it uses on the left, right, and as extras. When a cell has
    `has_shared_horizontal` (left_group == right_group), its left and right
    anchor positions are physically the same (in the cell graph) AND
    occupy the same boundary positions in the DP — no fresh allocation,
    no consume-on-convolution.

    Args:
        cell_template: Graph for ONE cell.
        cell_anchor_groups: dict group_id → list of cell template vertices
            in that group. Same dict shared across all cells in the path.
        junction_template: Graph for ONE junction between cells.
        junction_anchors_A: junction template vertices on the "previous-cell" side.
        junction_anchors_B: junction template vertices on the "next-cell" side.
        cell_specs: per-cell anchor spec (must have len ≥ 1).
            cell_specs[0].left_group must be None (path-start endpoint).
            cell_specs[-1].right_group must be None (path-end endpoint).
        verbose: print per-step diagnostics.
        observer: optional callable invoked AFTER each junction step and
            each cell step in the main loop. Receives a dict with keys:
              step_index (int, 0-based loop index)
              kind ("junction" or "cell")
              state_orbit_T_before (snapshot dict before orbit_convolve)
              state_orbit_T_after (post-convolve state)
              M_table (the precompute_M_table output for this step)
              orbit_sizes (out_orbit_sizes for this step)
              state_cell_groups_before / state_cell_groups_after
              shared_boundary (pos_jA for junction, pos_jB for cell)
              div_delta (int contribution to total_div)
            When observer is set, the cache is BYPASSED so every step
            fires. Used by extract_chain_transfer_matrix to observe the
            per-step transfer operator. No behavioral change when None.

    Returns:
        (state_T_dict, stats_dict, total_div) — same shape as compute_path_dp.
    """
    n_cells = len(cell_specs)
    assert n_cells >= 1
    assert cell_specs[0].left_group is None, (
        f"first cell must have left_group=None (path endpoint), "
        f"got {cell_specs[0].left_group}"
    )
    assert cell_specs[-1].right_group is None, (
        f"last cell must have right_group=None (path endpoint), "
        f"got {cell_specs[-1].right_group}"
    )

    # Precondition: junctions in the path have consistent A/B side sizes.
    # The junction connects cell k's right_group anchors to cell k+1's
    # left_group anchors via the matching encoded in junction_template.
    if n_cells > 1:
        assert cell_specs[0].right_group is not None, "first cell must have right_group"
        first_right_size = len(cell_anchor_groups[cell_specs[0].right_group])
        first_left_size_next = len(cell_anchor_groups[cell_specs[1].left_group])
        assert len(junction_anchors_A) == first_right_size, (
            f"junction_anchors_A size {len(junction_anchors_A)} must match "
            f"cell 0's right group size {first_right_size}"
        )
        assert len(junction_anchors_B) == first_left_size_next, (
            f"junction_anchors_B size {len(junction_anchors_B)} must match "
            f"cell 1's left group size {first_left_size_next}"
        )

    stats: Dict[str, float] = {
        "t_rooted": 0.0, "compress": 0.0, "M_precompute": 0.0,
        "convolve": 0.0, "enumerate": 0.0, "build_aut": 0.0, "total": 0.0,
    }
    t0 = time.perf_counter()

    # Cache lookup (compressed mode only): per-cell-compressed canonical
    # keys are position-invariant, so identical spec → identical orbit_T
    # regardless of label_offset.
    global _PATH_DP_CACHE_HITS, _PATH_DP_CACHE_MISSES
    cache_hit = False
    cached_state_T: Optional[Dict[Tuple, TuttePolynomial]] = None
    cached_total_div: Optional[int] = None
    cached_group_sizes: Optional[Tuple[int, ...]] = None
    cache_key: Optional[Tuple] = None
    if enable_per_cell_compression and observer is None:
        cache_key = _path_dp_cache_key(
            cell_template, cell_anchor_groups,
            junction_template, junction_anchors_A, junction_anchors_B,
            cell_specs,
        )
        cached = _PATH_DP_CACHE.get(cache_key)
        if cached is not None:
            cached_state_T, cached_total_div, cached_group_sizes = cached
            cache_hit = True
            _PATH_DP_CACHE_HITS += 1
            if verbose:
                print(f"  path DP cache HIT: {len(cached_state_T)} orbits, "
                      f"td={cached_total_div}", file=sys.stderr)
        else:
            _PATH_DP_CACHE_MISSES += 1

    # Allocate positions per cell, per group used. Shared cells reuse
    # left positions for the right group. `label_offset` offsets all
    # positions, so multiple path-DP runs (e.g., per row in a grid) can
    # have disjoint position spaces.
    pos_cell: List[Dict[int, List[int]]] = []
    for k, spec in enumerate(cell_specs):
        cell_pos: Dict[int, List[int]] = {}
        base = 10000 * (k + 1) + label_offset
        if spec.left_group is not None:
            anchors = cell_anchor_groups[spec.left_group]
            cell_pos[spec.left_group] = [base + i for i in range(len(anchors))]
        if spec.right_group is not None:
            if spec.has_shared_horizontal:
                cell_pos[spec.right_group] = cell_pos[spec.left_group]
            else:
                anchors = cell_anchor_groups[spec.right_group]
                cell_pos[spec.right_group] = [
                    base + 500 + i for i in range(len(anchors))
                ]
        offset = 1000
        for g in spec.extra_groups:
            anchors = cell_anchor_groups[g]
            cell_pos[g] = [base + offset + i for i in range(len(anchors))]
            offset += max(len(anchors), 1)
            assert offset <= 9500, "too many extras for the position-base layout"
        pos_cell.append(cell_pos)

    # Cache hit short-circuit: pos_cell now allocated; reconstruct
    # state_cell_groups using the cached metadata; return cached results.
    if cache_hit:
        meta_groups = cached_group_sizes  # actually a list of (cell_idx, group_id) tuples
        state_cell_groups_cached = [
            list(pos_cell[c][g]) for (c, g) in meta_groups
        ]
        stats["total"] = time.perf_counter() - t0
        if return_pos_layout:
            return (
                cached_state_T, stats, cached_total_div,
                pos_cell, state_cell_groups_cached,
            )
        return cached_state_T, stats, cached_total_div, state_cell_groups_cached

    # Precompute T_rooted per cell (cached by (template_key, anchor_tuple)).
    # Boundary order: left_anchors, right_anchors (if not shared), extras.
    cell_T_per_cell: List[Dict[Tuple, TuttePolynomial]] = []
    cell_template_anchors_per_cell: List[List[int]] = []
    cell_position_anchors_per_cell: List[List[int]] = []
    t = time.perf_counter()
    for k, spec in enumerate(cell_specs):
        template_anchors: List[int] = []
        position_anchors: List[int] = []
        if spec.left_group is not None:
            template_anchors.extend(cell_anchor_groups[spec.left_group])
            position_anchors.extend(pos_cell[k][spec.left_group])
        if spec.right_group is not None and not spec.has_shared_horizontal:
            template_anchors.extend(cell_anchor_groups[spec.right_group])
            position_anchors.extend(pos_cell[k][spec.right_group])
        for g in spec.extra_groups:
            template_anchors.extend(cell_anchor_groups[g])
            position_anchors.extend(pos_cell[k][g])
        cell_template_anchors_per_cell.append(template_anchors)
        cell_position_anchors_per_cell.append(position_anchors)
        cell_T_per_cell.append(t_rooted_cached(cell_template, template_anchors))

    if n_cells > 1:
        T_junction = t_rooted_cached(
            junction_template,
            list(junction_anchors_A) + list(junction_anchors_B),
        )
    stats["t_rooted"] = time.perf_counter() - t

    # Initialize state with cell 0.
    spec0 = cell_specs[0]
    map_cell0 = {
        cell_template_anchors_per_cell[0][i]: cell_position_anchors_per_cell[0][i]
        for i in range(len(cell_template_anchors_per_cell[0]))
    }
    state_partition = relabel_partition_dict(cell_T_per_cell[0], map_cell0)

    # Build per-cell groups for state's boundary, used by the structural
    # canonicalizer when enable_per_cell_compression is on. Each group is
    # a list of positions belonging to ONE (cell, anchor-group) pair, which
    # are interchangeable under the cell's S_n aut.
    state_cell_groups: List[List[int]] = []
    if spec0.left_group is not None:
        state_cell_groups.append(list(pos_cell[0][spec0.left_group]))
    if spec0.right_group is not None:
        if not spec0.has_shared_horizontal:
            state_cell_groups.append(list(pos_cell[0][spec0.right_group]))
        else:
            # Shared L=R: positions already in left group, don't re-add.
            pass
    for g in spec0.extra_groups:
        state_cell_groups.append(list(pos_cell[0][g]))

    t = time.perf_counter()
    if enable_per_cell_compression and state_cell_groups:
        # NOTE: this can raise `PerCellPreconditionViolated` (subclass of
        # ValueError) when the input T_rooted dict has partitions
        # whose per-cell canonical key collides with non-equal polynomial
        # values. Symptom seen with persistent T_rooted cache auto-load
        # on Cm_2 modular DP. Task #301 tracks the underlying fix
        # (the downstream key format isn't compatible with a fallback
        # to aut_compress_t_rooted, so a real fix needs to tighten
        # per_cell_canonical_key itself).
        state_orbit_T, state_orbit_partitions = aut_compress_t_rooted_per_cell(
            state_partition, state_cell_groups,
        )
    else:
        cell_aut: List[Dict[int, int]] = []
        state_orbit_T, state_orbit_partitions = aut_compress_t_rooted(
            state_partition, cell_aut,
        )
    stats["compress"] += time.perf_counter() - t

    # State boundary tracking. state_left = positions in state that are
    # NOT in the current right boundary AND NOT extras; in path DP,
    # state_left of the FIRST cell persists throughout (since the path
    # has no closing junction on the left).
    # For the generic case, "state_left" is just the persistent positions
    # carried from the path's left endpoint.
    state_right_pos: List[int] = []
    if spec0.right_group is not None:
        state_right_pos = list(pos_cell[0][spec0.right_group])
    # Persistent extras = path-start positions + accumulated extras across cells.
    persistent_positions: List[int] = []
    if spec0.left_group is not None:
        # Path-start cell can't have left in standard path DP, but if it
        # somehow has one, it persists.
        persistent_positions.extend(pos_cell[0][spec0.left_group])
    for g in spec0.extra_groups:
        persistent_positions.extend(pos_cell[0][g])

    # Junction c_J — connected components of junction template touching
    # the A-side anchors. For matching junctions M_k: c_J = k.
    junction_c_J = (
        components_touching(junction_template, list(junction_anchors_A))
        if n_cells > 1 else 1
    )
    total_div = 0

    if verbose:
        print(f"  path DP grouped: n_cells={n_cells}, "
              f"junction_c_J={junction_c_J}", file=sys.stderr)

    # Path-DP through (n-1) (junction, cell) pairs.
    for k in range(n_cells - 1):
        spec_next = cell_specs[k + 1]

        # JUNCTION step.
        pos_jA = state_right_pos
        pos_jB = list(pos_cell[k + 1][spec_next.left_group])
        junction_LR = list(junction_anchors_A) + list(junction_anchors_B)
        junction_pos = list(pos_jA) + list(pos_jB)
        map_J = {
            junction_LR[i]: junction_pos[i] for i in range(len(junction_LR))
        }
        junction_partition = relabel_partition_dict(T_junction, map_J)
        if enable_per_cell_compression:
            # Junction partition is over (junction A side + junction B side).
            # Use junction template's aut, restricted to preserve the A/B split.
            junction_aut = build_relabel_aut(
                junction_template,
                list(junction_anchors_A) + list(junction_anchors_B),
                junction_pos,
                preserve_split_index=len(junction_anchors_A),
            )
        else:
            junction_aut: List[Dict[int, int]] = []
        t = time.perf_counter()
        junction_orbit_T, junction_orbit_partitions = aut_compress_t_rooted(
            junction_partition, junction_aut,
        )
        stats["compress"] += time.perf_counter() - t

        # Junction step's OUT cell groups: state's existing groups MINUS
        # the consumed cell-k right group, PLUS junction's B-side as a new
        # cell-k+1 left group.
        # state_cell_groups currently includes cell-k's right (= pos_jA),
        # which is consumed. Replace with pos_jB (cell k+1's left).
        out_state_cell_groups: List[List[int]] = []
        consumed_set = set(pos_jA)
        for g_positions in state_cell_groups:
            if set(g_positions) == consumed_set:
                continue  # this group is consumed by junction
            out_state_cell_groups.append(list(g_positions))
        out_state_cell_groups.append(list(pos_jB))

        out_anchors_junc = list(persistent_positions) + list(pos_jB)
        out_aut: List[Dict[int, int]] = []
        t = time.perf_counter()
        if enable_per_cell_compression:
            # Skip enumeration; orbit sizes computed analytically.
            out_orbit_partitions_junc = {}
            out_orbit_sizes_junc = {}
        else:
            out_orbit_partitions_junc = enumerate_partitions_per_orbit(
                out_anchors_junc, out_aut,
            )
            out_orbit_sizes_junc = {
                ok: len(parts) for ok, parts in out_orbit_partitions_junc.items()
            }
        stats["enumerate"] += time.perf_counter() - t

        t = time.perf_counter()
        if enable_per_cell_compression:
            M_j = precompute_M_table(
                state_orbit_partitions, junction_orbit_partitions,
                shared_boundary=pos_jA, extra_boundary=pos_jB,
                out_aut_group=out_aut,
                state_extra_boundary=persistent_positions,
                out_cell_anchor_groups=out_state_cell_groups,
                state_cell_anchor_groups=state_cell_groups,
            )
            # Compute orbit sizes for the new keys analytically.
            for (Os, Oj, Oo) in M_j.keys():
                if Oo not in out_orbit_sizes_junc:
                    out_orbit_sizes_junc[Oo] = per_cell_orbit_size(
                        Oo, out_state_cell_groups,
                    )
        else:
            M_j = precompute_M_table(
                state_orbit_partitions, junction_orbit_partitions,
                shared_boundary=pos_jA, extra_boundary=pos_jB,
                out_aut_group=out_aut,
                state_extra_boundary=persistent_positions,
            )
        stats["M_precompute"] += time.perf_counter() - t

        # Snapshot pre-convolve state for observer.
        if observer is not None:
            state_orbit_T_before_junc = dict(state_orbit_T)
            state_cell_groups_before_junc = [list(g) for g in state_cell_groups]
        t = time.perf_counter()
        state_orbit_T = orbit_convolve(
            state_orbit_T, junction_orbit_T, M_j, out_orbit_sizes_junc,
        )
        stats["convolve"] += time.perf_counter() - t

        # Cell k+1's c_J for the divisor at the cell step's left anchors.
        cell_c_J = components_touching(
            cell_template, list(cell_anchor_groups[spec_next.left_group]),
        )
        total_div += (len(pos_jB) - junction_c_J)
        if observer is not None:
            observer({
                "step_index": k,
                "kind": "junction",
                "state_orbit_T_before": state_orbit_T_before_junc,
                "state_orbit_T_after": dict(state_orbit_T),
                "M_table": M_j,
                "junction_orbit_T": dict(junction_orbit_T),
                "orbit_sizes": dict(out_orbit_sizes_junc),
                "state_cell_groups_before": state_cell_groups_before_junc,
                "state_cell_groups_after": [list(g) for g in out_state_cell_groups],
                "shared_boundary": list(pos_jA),
                "extra_boundary": list(pos_jB),
                "persistent_positions": list(persistent_positions),
                "div_delta": len(pos_jB) - junction_c_J,
                "junction_c_J": junction_c_J,
            })
        state_right_pos = pos_jB
        if enable_per_cell_compression:
            # Construct rep partition per orbit (canonical_key alone isn't a
            # valid partition; precompute_M_table needs an actual rep).
            state_orbit_partitions = {
                Oo: [per_cell_orbit_rep(Oo, out_state_cell_groups)]
                for Oo in state_orbit_T
            }
        else:
            state_orbit_partitions = out_orbit_partitions_junc
        state_cell_groups = out_state_cell_groups

        # CELL step for cell k+1.
        next_template_anchors = cell_template_anchors_per_cell[k + 1]
        next_position_anchors = cell_position_anchors_per_cell[k + 1]
        map_cell_next = {
            next_template_anchors[i]: next_position_anchors[i]
            for i in range(len(next_template_anchors))
        }
        cell_partition = relabel_partition_dict(
            cell_T_per_cell[k + 1], map_cell_next,
        )
        if enable_per_cell_compression:
            # Build cell template aut on next cell's anchors, with
            # preserve_groups for the operationally-distinct anchor groups
            # (left/right/extras) of cell template.
            preserve_groups_next: List[List[int]] = []
            offset = 0
            if spec_next.left_group is not None:
                left_size = len(cell_anchor_groups[spec_next.left_group])
                preserve_groups_next.append(list(range(offset, offset + left_size)))
                offset += left_size
            if (spec_next.right_group is not None
                    and not spec_next.has_shared_horizontal):
                right_size = len(cell_anchor_groups[spec_next.right_group])
                preserve_groups_next.append(list(range(offset, offset + right_size)))
                offset += right_size
            for g in spec_next.extra_groups:
                g_size = len(cell_anchor_groups[g])
                preserve_groups_next.append(list(range(offset, offset + g_size)))
                offset += g_size
            cell_aut2 = build_relabel_aut(
                cell_template,
                next_template_anchors,
                next_position_anchors,
                preserve_groups=preserve_groups_next,
            )
        else:
            cell_aut2: List[Dict[int, int]] = []
        t = time.perf_counter()
        cell_orbit_T, cell_orbit_partitions = aut_compress_t_rooted(
            cell_partition, cell_aut2,
        )
        stats["compress"] += time.perf_counter() - t

        # New cell positions added to state's boundary:
        #   - shared horizontal: only extras (no new right)
        #   - disjoint:          right + extras
        cell_extra_positions: List[int] = []
        if (spec_next.right_group is not None
                and not spec_next.has_shared_horizontal):
            cell_extra_positions.extend(pos_cell[k + 1][spec_next.right_group])
        for g in spec_next.extra_groups:
            cell_extra_positions.extend(pos_cell[k + 1][g])

        if spec_next.has_shared_horizontal:
            # KEEP shared positions in output: cell's right == cell's left,
            # which is state's existing pos_jB. After convolution, pos_jB
            # remains state's right boundary.
            out_anchors_cell = (
                list(persistent_positions) + list(pos_jB)
                + list(cell_extra_positions)
            )
            keep_shared = True
            new_state_right = pos_jB  # unchanged
        else:
            # Standard disjoint: cell's right is fresh; pos_jB consumed.
            out_anchors_cell = (
                list(persistent_positions) + list(cell_extra_positions)
            )
            keep_shared = False
            if spec_next.right_group is not None:
                new_state_right = list(pos_cell[k + 1][spec_next.right_group])
            else:
                new_state_right = []

        # Build out_state_cell_groups for cell step.
        # In disjoint case: state's pos_jB group is consumed; cell's right
        # group + extras are added.
        # In shared case: pos_jB stays (kept_shared); only cell's extras added.
        out_state_cell_groups2: List[List[int]] = []
        consumed_set2 = set(pos_jB) if not spec_next.has_shared_horizontal else set()
        for g_positions in state_cell_groups:
            if set(g_positions) == consumed_set2:
                continue
            out_state_cell_groups2.append(list(g_positions))
        if spec_next.has_shared_horizontal:
            # pos_jB kept (already in state_cell_groups since it was added
            # by the junction step). Add cell k+1's extras.
            for g in spec_next.extra_groups:
                out_state_cell_groups2.append(list(pos_cell[k + 1][g]))
        else:
            # pos_jB consumed (removed above). Add cell k+1's right + extras.
            if spec_next.right_group is not None:
                out_state_cell_groups2.append(list(pos_cell[k + 1][spec_next.right_group]))
            for g in spec_next.extra_groups:
                out_state_cell_groups2.append(list(pos_cell[k + 1][g]))

        out_aut2: List[Dict[int, int]] = []
        t = time.perf_counter()
        if enable_per_cell_compression:
            out_orbit_partitions_cell = {}
            out_orbit_sizes_cell = {}
        else:
            out_orbit_partitions_cell = enumerate_partitions_per_orbit(
                out_anchors_cell, out_aut2,
            )
            out_orbit_sizes_cell = {
                ok: len(parts) for ok, parts in out_orbit_partitions_cell.items()
            }
        stats["enumerate"] += time.perf_counter() - t

        t = time.perf_counter()
        if enable_per_cell_compression:
            M_c = precompute_M_table(
                state_orbit_partitions, cell_orbit_partitions,
                shared_boundary=pos_jB, extra_boundary=cell_extra_positions,
                out_aut_group=out_aut2,
                state_extra_boundary=persistent_positions,
                keep_shared=keep_shared,
                out_cell_anchor_groups=out_state_cell_groups2,
                state_cell_anchor_groups=state_cell_groups,
            )
            for (Os, Oj, Oo) in M_c.keys():
                if Oo not in out_orbit_sizes_cell:
                    out_orbit_sizes_cell[Oo] = per_cell_orbit_size(
                        Oo, out_state_cell_groups2,
                    )
        else:
            M_c = precompute_M_table(
                state_orbit_partitions, cell_orbit_partitions,
                shared_boundary=pos_jB, extra_boundary=cell_extra_positions,
                out_aut_group=out_aut2,
                state_extra_boundary=persistent_positions,
                keep_shared=keep_shared,
            )
        stats["M_precompute"] += time.perf_counter() - t

        # Snapshot pre-convolve state for observer.
        if observer is not None:
            state_orbit_T_before_cell = dict(state_orbit_T)
            state_cell_groups_before_cell = [list(g) for g in state_cell_groups]
        t = time.perf_counter()
        state_orbit_T = orbit_convolve(
            state_orbit_T, cell_orbit_T, M_c, out_orbit_sizes_cell,
        )
        stats["convolve"] += time.perf_counter() - t

        total_div += (len(pos_jB) - cell_c_J)
        if observer is not None:
            observer({
                "step_index": k,
                "kind": "cell",
                "state_orbit_T_before": state_orbit_T_before_cell,
                "state_orbit_T_after": dict(state_orbit_T),
                "M_table": M_c,
                "cell_orbit_T": dict(cell_orbit_T),
                "orbit_sizes": dict(out_orbit_sizes_cell),
                "state_cell_groups_before": state_cell_groups_before_cell,
                "state_cell_groups_after": [list(g) for g in out_state_cell_groups2],
                "shared_boundary": list(pos_jB),
                "extra_boundary": list(cell_extra_positions),
                "persistent_positions": list(persistent_positions),
                "div_delta": len(pos_jB) - cell_c_J,
                "cell_c_J": cell_c_J,
                "keep_shared": keep_shared,
                "has_shared_horizontal": spec_next.has_shared_horizontal,
            })
        state_right_pos = new_state_right
        if enable_per_cell_compression:
            state_orbit_partitions = {
                Oo: [per_cell_orbit_rep(Oo, out_state_cell_groups2)]
                for Oo in state_orbit_T
            }
        else:
            state_orbit_partitions = out_orbit_partitions_cell
        state_cell_groups = out_state_cell_groups2

        # Update persistent positions: extras of cell k+1 join the
        # persistent set (they remain in state through subsequent steps).
        for g in spec_next.extra_groups:
            persistent_positions.extend(pos_cell[k + 1][g])
        # For shared cells: pos_jB stays in state's persistent boundary
        # (it IS state's right boundary still, but it's now also "carried").
        # For disjoint cells: pos_jB is consumed; new state_right is fresh.

        if verbose:
            print(f"  step {k+1}/{n_cells-1}: state {len(state_orbit_T)} orbits, "
                  f"persistent={len(persistent_positions)}, "
                  f"shared={spec_next.has_shared_horizontal}",
                  file=sys.stderr)

    stats["total"] = time.perf_counter() - t0
    if enable_per_cell_compression:
        # Store in cache: orbit_T is position-invariant; group meta lets
        # the caller reconstruct state_cell_groups for any label_offset.
        if cache_key is not None:
            meta = _state_groups_meta(pos_cell, state_cell_groups)
            _PATH_DP_CACHE[cache_key] = (state_orbit_T, total_div, meta)
        # Return compressed state directly. Caller marginalizes via
        # sum(T_orbit * per_cell_orbit_size(canonical, state_cell_groups))
        # without enumerating all partitions in each orbit. To preserve the
        # caller's API expectation of "{partition → poly}", we attach the
        # per-cell groups so the caller can marginalize without expansion.
        # If the caller actually needs the full {partition → poly} dict,
        # they can call expand_per_cell_orbit_dict explicitly.
        if return_pos_layout:
            return state_orbit_T, stats, total_div, pos_cell, state_cell_groups
        return state_orbit_T, stats, total_div, state_cell_groups
    state_T_uncompressed: Dict[Tuple, TuttePolynomial] = {}
    for canonical, val in state_orbit_T.items():
        for P in state_orbit_partitions.get(canonical, [canonical]):
            state_T_uncompressed[P] = val
    if return_pos_layout:
        return state_T_uncompressed, stats, total_div, pos_cell
    return state_T_uncompressed, stats, total_div
