"""Cell-quotient grid DP — extension of cycle DP to grid topologies.

For graphs whose hierarchical decomposition has CELL-QUOTIENT topology of
a 2D grid, computes T(graph) by row-by-row composition:
1. Compute T_rooted of each row via path DP, with vertical anchors as
   `state_extra_boundary`.
2. Compose rows via vertex-sum convolution at vertical-junction shared
   boundaries.
3. Marginalize remaining horizontal-end positions; divide by
   accumulated (x-1)^total_div.

Validated end-to-end on 3x3 K_4 grid (36n 66e): 624t,
T(1,1) = 66,795,331,387,392, MATCHES engine.

GENERIC over cell template + junction connectivity (handles M_k matchings,
K_{a,b} bipartite, etc. via auto-detected component count c_J).

LIMITATION: Cm3 (D-Wave Chimera 3, K_{4,4} cells with anchor sharing in
interior cells) requires a Cm3-specific cell adapter that maps cell_left
== cell_right to the SAME canonical positions. See module docstring of
the historical placeholder for design notes.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ..graph import Graph
from ..polynomial import TuttePolynomial
from .aut_orbit import build_relabel_aut
from .cell_anchor_adapter import CellGridSpec, _build_cycle_order
from .cell_quotient_helpers import (components_touching,
                                    enumerate_partitions_per_orbit,
                                    orbit_convolve,
                                    precompute_M_and_convolve_streaming,
                                    precompute_M_table)
from .cell_quotient_path import compute_path_dp, compute_path_dp_grouped
from .rooted_tutte import divide_by_x_minus_1_power, t_rooted_cached


def is_grid_topology(cell_quotient_adj: dict, n_cells: int) -> Optional[Tuple[int, int]]:
    """Detect if cell-quotient is a 2D grid; return (rows, cols) or None.

    Generic over cell content; checks adjacency structure only via degree
    distribution.
    """
    deg_count = Counter(len(cell_quotient_adj[i]) for i in range(n_cells))
    if 2 not in deg_count or deg_count.get(2, 0) != 4:
        return None
    n_corner = 4
    n_edge = deg_count.get(3, 0)
    n_interior = deg_count.get(4, 0)
    if n_corner + n_edge + n_interior != n_cells:
        return None
    for rows in range(2, n_cells // 2 + 2):
        if n_cells % rows != 0:
            continue
        cols = n_cells // rows
        if rows < 2 or cols < 2:
            continue
        if rows == 2 and cols == 2:
            expected_n_edge = 0
            expected_n_interior = 0
        elif rows == 2 or cols == 2:
            expected_n_edge = 2 * max(rows, cols) - 4
            expected_n_interior = 0
        else:
            expected_n_edge = 2 * (rows - 2) + 2 * (cols - 2)
            expected_n_interior = (rows - 2) * (cols - 2)
        if expected_n_edge == n_edge and expected_n_interior == n_interior:
            return (rows, cols)
    return None


def _grid_cell_layout(
    n_cells: int,
    rows: int,
    cols: int,
    cell_quotient_adj: Dict[int, Set[int]],
) -> Optional[List[List[int]]]:
    """Lay out cells in a (rows, cols) grid such that adjacency matches.

    Returns layout[r][c] = cell index, or None if no valid layout exists.
    Uses BFS from a corner cell.
    """
    corners = [i for i in range(n_cells) if len(cell_quotient_adj[i]) == 2]
    if not corners:
        return None

    start = min(corners)
    layout: List[List[Optional[int]]] = [[None] * cols for _ in range(rows)]
    layout[0][0] = start

    # Pick one of start's two neighbors as the row-0 right-direction.
    nbrs = sorted(cell_quotient_adj[start])
    if len(nbrs) != 2:
        return None
    # Try each as the right neighbor.
    for right_nbr in nbrs:
        down_nbr = nbrs[0] if nbrs[1] == right_nbr else nbrs[1]
        layout[0][0] = start
        layout[0][1] = right_nbr if cols > 1 else None
        layout[1][0] = down_nbr if rows > 1 else None

        ok = True
        # Fill row 0
        for c in range(2, cols):
            prev = layout[0][c - 1]
            prev_prev = layout[0][c - 2]
            # Next cell in row 0: a neighbor of prev not equal to prev_prev,
            # AND not already in column 0..c-2 of any row.
            placed = set()
            for r in range(rows):
                for cc in range(cols):
                    if layout[r][cc] is not None:
                        placed.add(layout[r][cc])
            candidates = [
                n for n in cell_quotient_adj[prev]
                if n not in placed and n != prev_prev
            ]
            # For row 0 (top), the next cell should have degree ≤ 3 (corner or top-edge)
            # Filter by adjacency to expected layout
            if not candidates:
                ok = False
                break
            # Pick the candidate that's a corner if c == cols-1, else edge
            chosen = None
            for cand in candidates:
                if c == cols - 1:
                    if len(cell_quotient_adj[cand]) == 2:  # corner
                        chosen = cand
                        break
                else:
                    if len(cell_quotient_adj[cand]) == 3:  # top-edge
                        chosen = cand
                        break
            if chosen is None:
                chosen = candidates[0]
            layout[0][c] = chosen

        if not ok:
            continue

        # Fill remaining rows by following down neighbors.
        # NOTE: layout[1][0] was pre-set at line 112 (down_nbr); we must
        # skip re-filling cells that are already set, otherwise the
        # `placed` set already contains them and `down_candidates` is empty.
        for r in range(1, rows):
            for c in range(cols):
                if layout[r][c] is not None:
                    continue
                cell_above = layout[r - 1][c]
                if cell_above is None:
                    ok = False
                    break
                placed = set()
                for rr in range(rows):
                    for cc in range(cols):
                        if layout[rr][cc] is not None:
                            placed.add(layout[rr][cc])
                # The down neighbor of cell_above
                down_candidates = [
                    n for n in cell_quotient_adj[cell_above] if n not in placed
                ]
                # Must also be adjacent to layout[r][c-1] if c > 0
                if c > 0 and layout[r][c - 1] is not None:
                    left_cell = layout[r][c - 1]
                    down_candidates = [
                        n for n in down_candidates
                        if n in cell_quotient_adj[left_cell]
                    ]
                if not down_candidates:
                    ok = False
                    break
                layout[r][c] = down_candidates[0]
            if not ok:
                break

        if not ok:
            continue

        # Verify all adjacencies are accounted for
        verified = True
        all_adj_pairs = set()
        for i, nbrs_i in cell_quotient_adj.items():
            for j in nbrs_i:
                if i < j:
                    all_adj_pairs.add((i, j))

        layout_pairs = set()
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    a, b = layout[r][c], layout[r][c + 1]
                    layout_pairs.add((min(a, b), max(a, b)))
                if r + 1 < rows:
                    a, b = layout[r][c], layout[r + 1][c]
                    layout_pairs.add((min(a, b), max(a, b)))

        if all_adj_pairs == layout_pairs:
            return [[layout[r][c] for c in range(cols)] for r in range(rows)]

    return None


def compute_grid_dp_grouped(
    cell_template: Graph,
    cell_anchor_groups: Dict[int, List[int]],
    horiz_junction_template: Graph,
    horiz_junction_anchors_A: List[int],
    horiz_junction_anchors_B: List[int],
    vert_junction_template: Graph,
    vert_junction_anchors_A: List[int],
    vert_junction_anchors_B: List[int],
    grid_specs: List[List[CellGridSpec]],
    verbose: bool = False,
) -> Optional[TuttePolynomial]:
    """Generic grid DP supporting per-cell anchor groups + shared anchors.

    `grid_specs[r][c]` is the CellGridSpec for the cell at (r, c). Cells
    with `has_shared_horizontal` reuse boundary positions across left and
    right horizontal junctions; cells with `has_shared_vertical` reuse
    positions across up and down vertical junctions. This is the
    structural property that distinguishes D-Wave Cm₃ (interior cells
    have both shared-horizontal AND shared-vertical) from synthetic
    K_n grids (all anchors disjoint).

    Algorithm:
      1. For each row r, run `compute_path_dp_grouped` on
         `[spec.to_row_spec() for spec in grid_specs[r]]`. Each row uses a
         disjoint label-offset to keep position spaces separate.
      2. Compose rows top-to-bottom: for r = 1..rows-1,
         (a) Convolve state with `cols` disjoint copies of vert_junction
             at shared = row r-1's down positions.
         (b) Convolve result with row r's T_rooted at shared = row r's up
             positions. If row r contains any cell with shared-vertical
             (up_group == down_group) AND r < rows-1, use keep_shared=True
             so row r's vertical positions persist for the next step.
      3. Marginalize the final state to a scalar.
      4. Divide by accumulated (x-1)^total_div.

    Returns None on structural failure.
    """
    rows = len(grid_specs)
    if rows < 1:
        return None
    cols = len(grid_specs[0])
    if cols < 1:
        return None
    if not all(len(row) == cols for row in grid_specs):
        return None

    # Step 1: per-row path DP.
    row_T_dicts: List[Dict[Tuple, TuttePolynomial]] = []
    row_total_divs: List[int] = []
    row_pos_layouts: List[List[Dict[int, List[int]]]] = []
    for r in range(rows):
        row_specs = [spec.to_row_spec() for spec in grid_specs[r]]
        result = compute_path_dp_grouped(
            cell_template, cell_anchor_groups,
            horiz_junction_template,
            horiz_junction_anchors_A, horiz_junction_anchors_B,
            row_specs, verbose=False,
            label_offset=100000 * r,
            return_pos_layout=True,
        )
        T_dict, _, td, pos_layout = result
        row_T_dicts.append(T_dict)
        row_total_divs.append(td)
        row_pos_layouts.append(pos_layout)
        if verbose:
            print(f"  row {r}: {len(T_dict)} parts, td={td}", file=sys.stderr)

    # Initial state = row 0's T_rooted.
    state_T = row_T_dicts[0]
    total_div = row_total_divs[0]

    # Persistent positions = positions actually present in path DP's output
    # boundary (NOT all allocated positions — junction-consumed positions
    # don't appear in T_dict's keys).
    def _boundary_positions(T_dict) -> List[int]:
        seen: List[int] = []
        seen_set = set()
        if not T_dict:
            return seen
        # Sample one partition to get the boundary positions.
        for P in T_dict.keys():
            for block in P:
                for v in block:
                    if v not in seen_set:
                        seen.append(v)
                        seen_set.add(v)
            break  # one is enough — all partitions share the same boundary.
        return seen

    def _row_persistent_positions(r: int) -> List[int]:
        return _boundary_positions(row_T_dicts[r])

    persistent_positions: List[int] = _row_persistent_positions(0)

    if verbose:
        print(f"  initial: state {len(state_T)} parts, td={total_div}, "
              f"persistent={len(persistent_positions)}", file=sys.stderr)

    # Step 2: row-by-row composition.
    v_a = len(vert_junction_anchors_A)
    v_b = len(vert_junction_anchors_B)
    single_c_J_vert = components_touching(
        vert_junction_template, list(vert_junction_anchors_A),
    )

    for r in range(1, rows):
        # 2a: Combined vertical junction (cols disjoint copies).
        # Anchor mapping: each column c's vert_junction_template has its A-side
        # mapped to row r-1's cell c down positions, and B-side to row r's
        # cell c up positions.
        prev_down_positions: List[int] = []
        next_up_positions: List[int] = []
        for c in range(cols):
            cell_above_spec = grid_specs[r - 1][c]
            cell_below_spec = grid_specs[r][c]
            if cell_above_spec.down_group is None:
                if verbose:
                    print(f"  row {r-1} col {c}: no down_group, abort",
                          file=sys.stderr)
                return None
            if cell_below_spec.up_group is None:
                if verbose:
                    print(f"  row {r} col {c}: no up_group, abort",
                          file=sys.stderr)
                return None
            prev_down_positions.extend(
                row_pos_layouts[r - 1][c][cell_above_spec.down_group]
            )
            next_up_positions.extend(
                row_pos_layouts[r][c][cell_below_spec.up_group]
            )

        if len(prev_down_positions) != cols * v_a:
            if verbose:
                print(f"  vert junc A side size mismatch: "
                      f"{len(prev_down_positions)} vs {cols * v_a}",
                      file=sys.stderr)
            return None
        if len(next_up_positions) != cols * v_b:
            if verbose:
                print(f"  vert junc B side size mismatch: "
                      f"{len(next_up_positions)} vs {cols * v_b}",
                      file=sys.stderr)
            return None

        # Build combined junction graph.
        combined_nodes = list(range(cols * (v_a + v_b)))
        combined_edges = []
        for c in range(cols):
            base_node = c * (v_a + v_b)
            for u, v in vert_junction_template.edges:
                combined_edges.append((base_node + u, base_node + v))
        combined_template = Graph(combined_nodes, combined_edges)
        combined_anchors_A: List[int] = []
        combined_anchors_B: List[int] = []
        for c in range(cols):
            base_node = c * (v_a + v_b)
            combined_anchors_A.extend(
                base_node + vert_junction_anchors_A[i] for i in range(v_a)
            )
            combined_anchors_B.extend(
                base_node + vert_junction_anchors_B[i] for i in range(v_b)
            )

        T_combined = t_rooted_cached(
            combined_template, combined_anchors_A + combined_anchors_B,
        )
        # Relabel combined junction's positions to actual prev_down/next_up.
        map_combined = {
            combined_anchors_A[i]: prev_down_positions[i] for i in range(cols * v_a)
        }
        for i in range(cols * v_b):
            map_combined[combined_anchors_B[i]] = next_up_positions[i]
        T_combined_pos = {}
        for P, val in T_combined.items():
            new_P = tuple(sorted(
                tuple(sorted(map_combined[x] for x in block)) for block in P
            ))
            T_combined_pos[new_P] = T_combined_pos.get(new_P, TuttePolynomial.zero()) + val

        # 2a convolve: state ⊗ combined vert junction at shared = prev_down.
        state_orbit_part = {P: [P] for P in state_T}
        junc_orbit_part = {P: [P] for P in T_combined_pos}
        out_anchors = list(persistent_positions) + list(next_up_positions)
        # Remove prev_down from persistent (consumed in this convolve).
        out_anchors_unique: List[int] = []
        for p in out_anchors:
            if p not in out_anchors_unique:
                out_anchors_unique.append(p)
        out_orbit_part = enumerate_partitions_per_orbit(out_anchors_unique, [])
        out_orbit_sizes = {ok: len(parts) for ok, parts in out_orbit_part.items()}

        state_extra_for_2a: List[int] = [
            p for p in persistent_positions if p not in set(prev_down_positions)
        ]

        M_v = precompute_M_table(
            state_orbit_part, junc_orbit_part,
            shared_boundary=prev_down_positions,
            extra_boundary=next_up_positions,
            out_aut_group=[],
            state_extra_boundary=state_extra_for_2a,
        )
        state_T = orbit_convolve(state_T, T_combined_pos, M_v, out_orbit_sizes)
        combined_c_J = cols * single_c_J_vert
        total_div += (cols * v_a - combined_c_J)
        # Update persistent positions: drop prev_down, add next_up.
        persistent_positions = state_extra_for_2a + list(next_up_positions)

        if verbose:
            print(f"  row {r} after vert junc: {len(state_T)} parts, "
                  f"td={total_div}", file=sys.stderr)

        # 2b convolve: state ⊗ row r's T_rooted at shared = next_up.
        # If any cell in row r has shared-vertical AND r is not the last
        # row, KEEP the up positions in output so they can serve as down
        # positions for the next vertical junction.
        any_shared_vert = any(
            grid_specs[r][c].has_shared_vertical for c in range(cols)
        )
        keep_up = any_shared_vert and (r < rows - 1)

        # All positions row r owns (its full extras boundary, derived from
        # row r's actual T_dict output boundary).
        row_r_all_positions = _row_persistent_positions(r)
        # State currently has persistent_positions (which includes next_up).
        # Row r's T_rooted is over row_r_all_positions (which also includes
        # row r's up positions, since path DP put them in extras).

        # State extra for 2b convolve = persistent positions MINUS the shared
        # (next_up). The shared is consumed (or kept).
        state_extra_for_2b: List[int] = [
            p for p in persistent_positions if p not in set(next_up_positions)
        ]
        # Row r's "extra_boundary" = its positions NOT in shared (next_up).
        row_r_extra_boundary: List[int] = [
            p for p in row_r_all_positions if p not in set(next_up_positions)
        ]

        out_anchors_2b: List[int] = list(state_extra_for_2b) + list(row_r_extra_boundary)
        if keep_up:
            out_anchors_2b = out_anchors_2b + list(next_up_positions)
        # Dedupe (shouldn't happen but safety).
        out_anchors_2b_unique: List[int] = []
        for p in out_anchors_2b:
            if p not in out_anchors_2b_unique:
                out_anchors_2b_unique.append(p)

        out_orbit_part_2b = enumerate_partitions_per_orbit(out_anchors_2b_unique, [])
        out_orbit_sizes_2b = {ok: len(parts) for ok, parts in out_orbit_part_2b.items()}

        row_r_T_orbit = {P: [P] for P in row_T_dicts[r]}
        state_orbit_part_2b = {P: [P] for P in state_T}

        M_r = precompute_M_table(
            state_orbit_part_2b, row_r_T_orbit,
            shared_boundary=next_up_positions,
            extra_boundary=row_r_extra_boundary,
            out_aut_group=[],
            state_extra_boundary=state_extra_for_2b,
            keep_shared=keep_up,
        )
        state_T = orbit_convolve(state_T, row_T_dicts[r], M_r, out_orbit_sizes_2b)

        # Divisor for row r convolution: row r is connected (we treat it as
        # one piece in the convolution); shared is cols * v_b positions;
        # c_J = 1 (the row, viewed as one connected graph in the full graph).
        # Actually: when we do the row r convolution with state, row r might
        # not be connected in isolation (cells joined only through horizontal
        # junctions). But by this point in the DP, state already has row 0..r-1
        # combined; row r is being attached via vert junction (already done).
        # The vertex-sum convolution divisor is (|S| - c_J(row r at S)) where
        # c_J = number of components of row r touching S = next_up.
        # For Cm3 row r: cells joined by horizontal junctions form one connected
        # component (assuming cols >= 2 and horizontal junctions exist).
        # For cols = 1: each row is a single cell, c_J = 1.
        # We approximate c_J = 1 here (works for typical D-Wave-style grids).
        # Refinement: compute c_J exactly from row r's structure.
        total_div += (cols * v_b - 1) + row_total_divs[r]

        if keep_up:
            persistent_positions = (
                state_extra_for_2b + row_r_extra_boundary + list(next_up_positions)
            )
        else:
            persistent_positions = state_extra_for_2b + row_r_extra_boundary

        if verbose:
            print(f"  row {r} after row conv: {len(state_T)} parts, "
                  f"td={total_div}, keep_up={keep_up}",
                  file=sys.stderr)

    # Step 3: marginalize to scalar.
    T_total = TuttePolynomial.zero()
    for val in state_T.values():
        T_total = T_total + val

    try:
        return divide_by_x_minus_1_power(T_total, total_div)
    except ValueError:
        return None


def compute_grid_dp_streamed_kab(
    cell_template: Graph,
    cell_anchor_groups: Dict[int, List[int]],
    horiz_junction_template: Graph,
    horiz_junction_anchors_A: List[int],
    horiz_junction_anchors_B: List[int],
    vert_junction_template: Graph,
    vert_junction_anchors_A: List[int],
    vert_junction_anchors_B: List[int],
    grid_specs: List[List[CellGridSpec]],
    verbose: bool = False,
) -> Optional[TuttePolynomial]:
    """Streamed grid DP for K_{a,b}-style cells.

    Uses per-cell compression on inner row DPs (`enable_per_cell_compression`)
    + `enumerate_junction_internally` + `out_cell_anchor_groups` to keep
    state orbit-compressed throughout the row-by-row composition. The
    M_4 vertical junction lifts state's per-cell S_n on the previous row's
    down-anchors to the next row's up-anchors via the matching-edge
    structure, so the orbit-shortcut on state side is preserved through
    to the output canonical key.

    Preconditions (caller MUST verify before invoking):
    - `cell_template` is bipartite (K_{a,b} family).
    - All cells in `grid_specs` use disjoint anchor groups per direction
      (no shared-anchor interior cells, i.e. NOT Cm₃ interior structure).
    - `vert_junction_template` is a disjoint k-matching M_k (so its
      per-edge cell groups give the matching-preserving canonical key).

    Returns the polynomial on success, `None` on any structural mismatch.

    Validated on Cm₂ (2×2 K_{4,4} grid) — see
    `tutte/research/scripts/cm2_via_v5_streamed.py` and the engine
    benchmark in `tutte/docs/06_5_cell_quotient_grid_dp.md`. 
    T(Cm₂)(1,1) = 11686511179538104320 in ~36 s
    (vs ~55 s via `kmatching_formula`).
    """
    from .aut_orbit import per_cell_canonical_key, per_cell_orbit_rep
    from .rooted_tutte import all_partitions, relabel_partition_dict

    rows = len(grid_specs)
    if rows < 1:
        return None
    cols = len(grid_specs[0])
    if cols < 1:
        return None
    if not all(len(row) == cols for row in grid_specs):
        return None

    # Step 1: per-row compressed path DP.
    row_results: List[Dict] = []
    for r in range(rows):
        row_specs = [spec.to_row_spec() for spec in grid_specs[r]]
        result = compute_path_dp_grouped(
            cell_template, cell_anchor_groups,
            horiz_junction_template,
            horiz_junction_anchors_A, horiz_junction_anchors_B,
            row_specs, verbose=False,
            label_offset=100000 * r, return_pos_layout=True,
            enable_per_cell_compression=True,
        )
        if not isinstance(result, tuple) or len(result) != 5:
            # Compression disabled fallback returned wrong shape.
            return None
        state_orbit_T, _, td, pos_layout, scg = result
        row_results.append({
            "state_T": state_orbit_T, "td": td,
            "pos_layout": pos_layout, "scg": scg,
        })
        if verbose:
            print(f"  row {r}: {len(state_orbit_T)} orbits, td={td}, "
                  f"scg sizes={[len(g) for g in scg]}", file=sys.stderr)

    v_a = len(vert_junction_anchors_A)
    v_b = len(vert_junction_anchors_B)
    single_c_J_vert = components_touching(
        vert_junction_template, list(vert_junction_anchors_A),
    )

    # Initial state: row 0 compressed (single rep per orbit).
    state_orbit_T = row_results[0]["state_T"]
    state_scg = row_results[0]["scg"]
    state_orbit_part = {
        canon: [per_cell_orbit_rep(canon, state_scg)]
        for canon in state_orbit_T
    }
    total_div = row_results[0]["td"]

    # Step 2: compose rows.
    for r in range(1, rows):
        next_scg = row_results[r]["scg"]
        next_orbit_T = row_results[r]["state_T"]
        next_pos_layout = row_results[r]["pos_layout"]

        # 2a: vertical M_k junction (cols disjoint copies).
        prev_down: List[int] = []
        next_up: List[int] = []
        for c in range(cols):
            above = grid_specs[r - 1][c]
            below = grid_specs[r][c]
            if above.down_group is None or below.up_group is None:
                return None
            prev_down.extend(row_results[r - 1]["pos_layout"][c][above.down_group])
            next_up.extend(next_pos_layout[c][below.up_group])

        if len(prev_down) != cols * v_a or len(next_up) != cols * v_b:
            return None

        # Build combined junction template (cols disjoint copies).
        combined_nodes = list(range(cols * (v_a + v_b)))
        combined_edges = []
        for c in range(cols):
            base = c * (v_a + v_b)
            for u, v in vert_junction_template.edges:
                combined_edges.append((base + u, base + v))
        combined = Graph(combined_nodes, combined_edges)
        combined_A = [c * (v_a + v_b) + vert_junction_anchors_A[i]
                      for c in range(cols) for i in range(v_a)]
        combined_B = [c * (v_a + v_b) + vert_junction_anchors_B[i]
                      for c in range(cols) for i in range(v_b)]
        T_combined = t_rooted_cached(combined, combined_A + combined_B)
        map_c = {combined_A[i]: prev_down[i] for i in range(cols * v_a)}
        for i in range(cols * v_b):
            map_c[combined_B[i]] = next_up[i]
        T_combined_pos = relabel_partition_dict(T_combined, map_c)

        # Per-edge cell groups for the M_k junction's matching-preserving aut.
        # Assumes vert_junction is an M_k matching: anchor_A[i] paired with
        # anchor_B[i] via a single edge.
        junc_cell_groups = [[prev_down[i], next_up[i]]
                            for i in range(cols * v_a)]
        junc_orbit_T: Dict[Tuple, TuttePolynomial] = {}
        junc_orbit_part: Dict[Tuple, List[Tuple]] = {}
        for P, val in T_combined_pos.items():
            canon = per_cell_canonical_key(P, junc_cell_groups)
            if canon not in junc_orbit_T:
                junc_orbit_T[canon] = val
                junc_orbit_part[canon] = [P]

        # Out boundary = next_up; compress by next row's per-cell groups.
        # Use streaming M-table+convolve to bound peak memory (M_v can be
        # multi-GB for Cm₃-scale problems). Analytical orbit sizing
        # avoids Bell(|next_up|) enumeration.
        out_orbit_sizes_2a: Dict[Tuple, int] = {}
        state_T_2a = precompute_M_and_convolve_streaming(
            state_orbit_T=state_orbit_T,
            state_orbit_partitions=state_orbit_part,
            junction_orbit_T=junc_orbit_T,
            junction_orbit_partitions=junc_orbit_part,
            out_orbit_sizes=out_orbit_sizes_2a,
            shared_boundary=prev_down,
            extra_boundary=next_up,
            out_aut_group=[],
            state_extra_boundary=[],
            out_cell_anchor_groups=next_scg,
            state_cell_anchor_groups=state_scg,
            junction_cell_anchor_groups=junc_cell_groups,
            enumerate_junction_internally=True,
        )
        total_div += (cols * v_a - cols * single_c_J_vert)
        if verbose:
            print(f"  row {r} 2a: {len(state_T_2a)} orbits, td={total_div}",
                  file=sys.stderr)

        # 2b: row composition (state ⊗ row r).
        state_orbit_part_2b = {
            canon: [per_cell_orbit_rep(canon, next_scg)]
            for canon in state_T_2a
        }
        row_orbit_part = {
            canon: [per_cell_orbit_rep(canon, next_scg)]
            for canon in next_orbit_T
        }

        # Out for 2b: empty if last row (closed); else next row's b-side
        # positions for the next vertical junction.
        if r < rows - 1:
            out_anchors_2b: List[int] = []
            out_scg_2b: List[List[int]] = []
            for c in range(cols):
                spec_below = grid_specs[r][c]
                if spec_below.down_group is None:
                    return None
                grp = next_pos_layout[c][spec_below.down_group]
                out_anchors_2b.extend(grp)
                out_scg_2b.append(list(grp))
        else:
            out_anchors_2b = []
            out_scg_2b = []

        # 2b: streaming M-table + convolve, same memory rationale.
        out_orbit_sizes_2b: Dict[Tuple, int] = (
            {} if out_scg_2b else {(): 1}
        )
        state_final = precompute_M_and_convolve_streaming(
            state_orbit_T=state_T_2a,
            state_orbit_partitions=state_orbit_part_2b,
            junction_orbit_T=next_orbit_T,
            junction_orbit_partitions=row_orbit_part,
            out_orbit_sizes=out_orbit_sizes_2b,
            shared_boundary=next_up,
            extra_boundary=out_anchors_2b,
            out_aut_group=[],
            state_extra_boundary=[],
            out_cell_anchor_groups=out_scg_2b if out_scg_2b else None,
            state_cell_anchor_groups=next_scg,
            junction_cell_anchor_groups=next_scg,
            enumerate_junction_internally=True,
        )
        total_div += (cols * v_b - 1) + row_results[r]["td"]
        if verbose:
            print(f"  row {r} 2b: {len(state_final)} orbits, td={total_div}",
                  file=sys.stderr)

        # State for next row iteration.
        state_orbit_T = state_final
        state_scg = out_scg_2b
        if out_scg_2b:
            state_orbit_part = {
                canon: [per_cell_orbit_rep(canon, out_scg_2b)]
                for canon in state_orbit_T
            }
        else:
            state_orbit_part = {canon: [()] for canon in state_orbit_T}

    # Marginalize.
    total = TuttePolynomial.zero()
    for val in state_orbit_T.values():
        total = total + val
    try:
        return divide_by_x_minus_1_power(total, total_div)
    except ValueError:
        return None
