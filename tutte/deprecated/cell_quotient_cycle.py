"""Cell-quotient cycle DP — Tutte polynomial via rooted-Tutte composition.

For graphs whose hierarchical decomposition has CELL-QUOTIENT topology
of a SIMPLE CYCLE (e.g., D-Wave Cm2's 4-cycle of K_{4,4} cells), this DP
computes T(graph) by:
1. Computing T_rooted of one cell (cached, shared across all cells).
2. Path-DP through n-1 cell-junction steps using vertex-sum convolution.
3. Cycle-close at last junction via identification formula.

Key innovations:
- c_J auto-detection for disconnected junctions (M_k matchings):
  divisor = (x-1)^{|S| - c_J(S)} instead of (x-1)^{|S|-1}.
- Orbit-level M_precompute (143× speedup): pick rep_state ∈ O_state.
- Orbit-level close_step2 (26× speedup): collapse poly muls per orbit.
- Position-invariant enumerate caching (7× speedup).

Identification formula at cycle close:
    T(cycle) = (x-1)^{-a} · Σ_P ((x-1)(y-1))^{actually_same(P)} · T_rooted_int[P]
where actually_same = a - n_merges (chain-aware union-find on P).

Validated on K_3, K_4, K_{4,4} cycles + actual D-Wave Cm2 (T(1,1) matches
engine and Kirchhoff). Generic across cell types and junction connectivity.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from ..graph import Graph
from ..polynomial import TuttePolynomial
from ..roots.aut_orbit import aut_compress_t_rooted, build_relabel_aut
from ..roots.cell_quotient_helpers import (components_touching,
                                    enumerate_partitions_cached,
                                    orbit_convolve, precompute_M_table)
from ..roots.rooted_tutte import (divide_by_x_minus_1_power, relabel_partition_dict,
                           t_rooted_cached)


def compute_cycle_dp(
    cell_template: Graph,
    cell_left_anchors: List[int],
    cell_right_anchors: List[int],
    junction_template: Graph,
    junction_anchors_A: List[int],
    junction_anchors_B: List[int],
    n_cells: int,
    verbose: bool = False,
) -> Tuple[TuttePolynomial, Dict[str, float]]:
    """Cycle DP with orbit-compressed rooted-Tutte composition.

    Args:
        cell_template: Graph for ONE cell (e.g., K_{4,4}).
        cell_left_anchors: cell vertices going to PREVIOUS junction in cycle.
        cell_right_anchors: cell vertices going to NEXT junction in cycle.
        junction_template: Graph for ONE junction between cells (e.g., M_4).
        junction_anchors_A: junction vertices on the "previous-cell" side.
        junction_anchors_B: junction vertices on the "next-cell" side.
        n_cells: number of cells in the cycle (n ≥ 2).
        verbose: print per-step diagnostics to stderr.

    Returns:
        (T_polynomial, stats_dict)
    """
    assert n_cells >= 2
    a = len(cell_left_anchors)
    b = len(cell_right_anchors)
    assert len(junction_anchors_A) == b
    assert len(junction_anchors_B) == a

    stats: Dict[str, float] = {
        "t_rooted": 0.0, "compress": 0.0, "M_precompute": 0.0,
        "convolve": 0.0, "close_step1": 0.0, "close_step2": 0.0,
        "enumerate": 0.0, "build_aut": 0.0, "total": 0.0,
    }
    t0 = time.perf_counter()

    # T_rooted of cell + junction (cached by canonical_key).
    t = time.perf_counter()
    cell_LR_anchors = list(cell_left_anchors) + list(cell_right_anchors)
    T_cell = t_rooted_cached(cell_template, cell_LR_anchors)
    T_junction = t_rooted_cached(
        junction_template,
        list(junction_anchors_A) + list(junction_anchors_B),
    )
    stats["t_rooted"] = time.perf_counter() - t

    # Position labels for state's boundary across cells.
    pos_cell_left: List[List[int]] = []
    pos_cell_right: List[List[int]] = []
    for k in range(n_cells):
        base = 10000 * (k + 1)
        pos_cell_left.append([base + i for i in range(a)])
        pos_cell_right.append([base + 500 + i for i in range(b)])

    # Initialize state with cell 0 (full boundary tracked).
    map_LR = {**{cell_left_anchors[i]: pos_cell_left[0][i] for i in range(a)},
              **{cell_right_anchors[i]: pos_cell_right[0][i] for i in range(b)}}
    state_partition = relabel_partition_dict(T_cell, map_LR)
    cell_aut = build_relabel_aut(
        cell_template, cell_LR_anchors,
        list(pos_cell_left[0]) + list(pos_cell_right[0]),
        preserve_split_index=a,
    )
    t = time.perf_counter()
    state_orbit_T, state_orbit_partitions = aut_compress_t_rooted(
        state_partition, cell_aut,
    )
    stats["compress"] += time.perf_counter() - t

    state_left = pos_cell_left[0]
    state_right = pos_cell_right[0]
    total_div = 0

    # Junction's component count touching A-side anchors (handles disconnected
    # junctions like M_k matchings: divisor (b - c_J) instead of (b - 1)).
    junction_c_J = components_touching(junction_template, list(junction_anchors_A))
    cell_c_J = components_touching(cell_template, list(cell_left_anchors))
    if verbose:
        print(f"  junction c_J = {junction_c_J}, cell c_J = {cell_c_J}",
              file=sys.stderr)

    # Path-DP through n-1 (junction, cell) pairs.
    for k in range(n_cells - 1):
        # Junction step: convolve state with junction at pos_jA = state_right.
        pos_jA = state_right
        pos_jB = pos_cell_left[k + 1]
        junction_LR = list(junction_anchors_A) + list(junction_anchors_B)
        junction_pos = list(pos_jA) + list(pos_jB)
        map_J = {junction_LR[i]: junction_pos[i] for i in range(len(junction_LR))}
        junction_partition = relabel_partition_dict(T_junction, map_J)
        junction_aut = build_relabel_aut(
            junction_template, junction_LR, junction_pos,
            preserve_split_index=b,
        )
        t = time.perf_counter()
        junction_orbit_T, junction_orbit_partitions = aut_compress_t_rooted(
            junction_partition, junction_aut,
        )
        stats["compress"] += time.perf_counter() - t

        out_anchors = list(state_left) + list(pos_jB)
        t = time.perf_counter()
        out_aut = build_relabel_aut(
            cell_template, cell_LR_anchors,
            list(state_left) + list(pos_jB),
            preserve_split_index=a,
        )
        stats["build_aut"] += time.perf_counter() - t
        t = time.perf_counter()
        out_orbit_partitions = enumerate_partitions_cached(
            cell_template, cell_LR_anchors, out_anchors,
            preserve_split_index=a,
        )
        stats["enumerate"] += time.perf_counter() - t
        out_orbit_sizes = {ok: len(parts) for ok, parts in out_orbit_partitions.items()}

        t = time.perf_counter()
        M_j = precompute_M_table(
            state_orbit_partitions, junction_orbit_partitions,
            shared_boundary=pos_jA, extra_boundary=pos_jB,
            out_aut_group=out_aut, state_extra_boundary=state_left,
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

        # Cell step: convolve state with next cell at pos_jB.
        cell_idx = k + 1
        pos_next_right = pos_cell_right[cell_idx]
        map_LR2 = {**{cell_left_anchors[i]: pos_jB[i] for i in range(a)},
                   **{cell_right_anchors[i]: pos_next_right[i] for i in range(b)}}
        cell_partition = relabel_partition_dict(T_cell, map_LR2)
        cell_aut2 = build_relabel_aut(
            cell_template, cell_LR_anchors,
            list(pos_jB) + list(pos_next_right),
            preserve_split_index=a,
        )
        t = time.perf_counter()
        cell_orbit_T, cell_orbit_partitions = aut_compress_t_rooted(
            cell_partition, cell_aut2,
        )
        stats["compress"] += time.perf_counter() - t

        out_anchors2 = list(state_left) + list(pos_next_right)
        t = time.perf_counter()
        out_aut2 = build_relabel_aut(
            cell_template, cell_LR_anchors,
            list(state_left) + list(pos_next_right),
            preserve_split_index=a,
        )
        stats["build_aut"] += time.perf_counter() - t
        t = time.perf_counter()
        out_orbit_partitions2 = enumerate_partitions_cached(
            cell_template, cell_LR_anchors, out_anchors2,
            preserve_split_index=a,
        )
        stats["enumerate"] += time.perf_counter() - t
        out_orbit_sizes2 = {ok: len(parts) for ok, parts in out_orbit_partitions2.items()}

        t = time.perf_counter()
        M_c = precompute_M_table(
            state_orbit_partitions, cell_orbit_partitions,
            shared_boundary=pos_jB, extra_boundary=pos_next_right,
            out_aut_group=out_aut2, state_extra_boundary=state_left,
        )
        stats["M_precompute"] += time.perf_counter() - t

        t = time.perf_counter()
        state_orbit_T = orbit_convolve(
            state_orbit_T, cell_orbit_T, M_c, out_orbit_sizes2,
        )
        stats["convolve"] += time.perf_counter() - t

        total_div += (a - cell_c_J)
        state_right = pos_next_right
        state_orbit_partitions = out_orbit_partitions2

    # Cycle close STEP 1: convolve state with closing junction.
    pos_cA = state_right
    pos_cB_FRESH = [99000000 + i for i in range(a)]
    junction_LR = list(junction_anchors_A) + list(junction_anchors_B)
    junction_pos = list(pos_cA) + list(pos_cB_FRESH)
    map_close = {junction_LR[i]: junction_pos[i] for i in range(len(junction_LR))}
    closing_partition = relabel_partition_dict(T_junction, map_close)
    closing_aut = build_relabel_aut(
        junction_template, junction_LR, junction_pos,
        preserve_split_index=b,
    )
    t = time.perf_counter()
    closing_orbit_T, closing_orbit_partitions = aut_compress_t_rooted(
        closing_partition, closing_aut,
    )
    stats["compress"] += time.perf_counter() - t

    out_anchors = list(state_left) + list(pos_cB_FRESH)
    t = time.perf_counter()
    out_aut = build_relabel_aut(
        cell_template, cell_LR_anchors,
        list(state_left) + list(pos_cB_FRESH),
        preserve_split_index=a,
    )
    stats["build_aut"] += time.perf_counter() - t
    t = time.perf_counter()
    out_orbit_partitions = enumerate_partitions_cached(
        cell_template, cell_LR_anchors, out_anchors,
        preserve_split_index=a,
    )
    stats["enumerate"] += time.perf_counter() - t
    out_orbit_sizes = {ok: len(parts) for ok, parts in out_orbit_partitions.items()}

    t = time.perf_counter()
    M_close = precompute_M_table(
        state_orbit_partitions, closing_orbit_partitions,
        shared_boundary=pos_cA, extra_boundary=pos_cB_FRESH,
        out_aut_group=out_aut, state_extra_boundary=state_left,
    )
    stats["M_precompute"] += time.perf_counter() - t

    t = time.perf_counter()
    state_after_close1 = orbit_convolve(
        state_orbit_T, closing_orbit_T, M_close, out_orbit_sizes,
    )
    stats["convolve"] += time.perf_counter() - t
    stats["close_step1"] = time.perf_counter() - t
    total_div += (b - junction_c_J)

    if verbose:
        print(f"  after close step 1: {len(state_after_close1)} orbits",
              file=sys.stderr)

    # Cycle close STEP 2: identification formula
    #   T(cycle) = (x-1)^{-a} · Σ_P ((x-1)(y-1))^{actually_same(P)} · T_rooted_int[P]
    # where actually_same = a - n_merges via chain-aware union-find.
    t = time.perf_counter()
    x_minus_1 = TuttePolynomial.x() + (-1) * TuttePolynomial.one()
    y_minus_1 = TuttePolynomial.y() + (-1) * TuttePolynomial.one()
    xy_minus_1 = x_minus_1 * y_minus_1

    xy_powers = [TuttePolynomial.one()]
    for _ in range(a):
        xy_powers.append(xy_powers[-1] * xy_minus_1)

    # Orbit-level: T_rooted constant within orbit, so collapse poly muls to 1 per orbit.
    T_total = TuttePolynomial.zero()
    for orbit_canonical, intermediate_val in state_after_close1.items():
        actually_same_counts: Dict[int, int] = defaultdict(int)
        for P in out_orbit_partitions.get(orbit_canonical, []):
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
                ru, rv = _find(u), _find(v)
                if ru != rv:
                    parent[max(ru, rv)] = min(ru, rv)
                    return 1
                return 0

            n_merges = 0
            for i in range(a):
                n_merges += _union(state_left[i], pos_cB_FRESH[i])
            actually_same = a - n_merges
            actually_same_counts[actually_same] += 1

        multiplier = TuttePolynomial.zero()
        for k, count in actually_same_counts.items():
            multiplier = multiplier + count * xy_powers[k]
        T_total = T_total + multiplier * intermediate_val
    stats["close_step2"] = time.perf_counter() - t

    total_div += a

    T_final = divide_by_x_minus_1_power(T_total, total_div)
    stats["total"] = time.perf_counter() - t0
    return T_final, stats
