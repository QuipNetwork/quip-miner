"""Cell-quotient cycle DP helpers: M_precompute + orbit_convolve + caching.

Generic helpers for the cell-quotient cycle DP. Operate on the rooted-Tutte
partition framework with Aut-orbit compression. All generic — work for any
cell template + junction template combination, not D-Wave-specific.

Major optimizations:
1. Orbit-level M_precompute (143× speedup): pick rep_state ∈ O_state,
   iterate all P_junc, multiply by |O_state|. Validated mathematically.
2. Position-invariant `enumerate_partitions_cached` (7× speedup): cache
   canonical orbit partitions; relabel per call.
3. Raw-dict polynomial arithmetic in inner loops (3-5× speedup):
   skip TuttePolynomial encode/decode cycles inside hot loops.

Components-touching helper handles disconnected junctions (e.g., M_k matchings):
divisor for vertex-sum convolution is `(x-1)^{|S| - c_J(S)}` instead of
`(x-1)^{|S|-1}`, where c_J(S) is junction components touching shared boundary.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx

from ..graph import Graph
from ..polynomial import TuttePolynomial
from .aut_orbit import (apply_perm_to_partition, build_relabel_aut,
                        canonical_partition, enumerate_per_cell_aut_group,
                        per_cell_canonical_key, per_cell_orbit_size,
                        per_cell_partition_stab)
from .rooted_tutte import (all_partitions, delta, join_partitions,
                           restrict_partition)


def components_touching(template: Graph, boundary: List[int]) -> int:
    """Count components of template containing at least one boundary vertex.

    For vertex-sum convolution rank-shift: divisor (x-1)^{|S| - c_J(S)} where
    c_J(S) = number of junction components touching shared boundary S.

    Standard formula assumes connected junction (c_J=1, divisor=|S|-1).
    For disconnected junctions (M_k matchings have k components, each
    containing one shared vertex): divisor = (b - c_J).
    """
    nx_g = nx.Graph()
    nx_g.add_nodes_from(template.nodes)
    nx_g.add_edges_from(template.edges)
    boundary_set = set(boundary)
    count = 0
    for component in nx.connected_components(nx_g):
        if component & boundary_set:
            count += 1
    return count


def enumerate_partitions_per_orbit(
    boundary: List[int], aut_group: List[Dict[int, int]],
) -> Dict[Tuple, List[Tuple]]:
    """For each canonical orbit-rep, list all partitions of `boundary`."""
    out: Dict[Tuple, List[Tuple]] = defaultdict(list)
    for p_list in all_partitions(list(boundary)):
        p_tuple = tuple(sorted(tuple(sorted(b)) for b in p_list))
        canonical = canonical_partition(p_tuple, aut_group)
        out[canonical].append(p_tuple)
    return dict(out)


# Cache for canonical orbit partitions, position-invariant.
_CANONICAL_ORBIT_CACHE: Dict[Tuple, Dict[Tuple, List[Tuple]]] = {}


def _relabel_partition(P: Tuple[Tuple[int, ...], ...],
                       pos_map: Dict[int, int]) -> Tuple[Tuple[int, ...], ...]:
    """Apply pos_map to all vertices in partition; return canonical sorted form."""
    return tuple(sorted(tuple(sorted(pos_map[v] for v in block)) for block in P))


def enumerate_partitions_cached(
    cell_template: Graph,
    cell_anchors_orig: List[int],
    boundary_pos: List[int],
    preserve_split_index: Optional[int] = None,
) -> Dict[Tuple, List[Tuple]]:
    """Position-invariant cached enumerate.

    Computes orbit partitions ONCE for canonical positions [0..n-1], then
    relabels to actual boundary_pos. Saves Bell(n)×|aut| work on repeat
    calls with same cell template / anchor structure but different positions.
    """
    boundary_size = len(boundary_pos)
    cache_key = (
        cell_template.canonical_key(),
        tuple(cell_anchors_orig),
        preserve_split_index,
        boundary_size,
    )
    if cache_key not in _CANONICAL_ORBIT_CACHE:
        canonical_pos = list(range(boundary_size))
        canonical_aut = build_relabel_aut(
            cell_template, cell_anchors_orig, canonical_pos,
            preserve_split_index=preserve_split_index,
        )
        _CANONICAL_ORBIT_CACHE[cache_key] = enumerate_partitions_per_orbit(
            canonical_pos, canonical_aut,
        )
    canonical_orbits = _CANONICAL_ORBIT_CACHE[cache_key]
    pos_map = {i: boundary_pos[i] for i in range(boundary_size)}
    return {
        _relabel_partition(ck, pos_map): [_relabel_partition(p, pos_map) for p in parts]
        for ck, parts in canonical_orbits.items()
    }


def precompute_M_table(
    state_orbit_partitions: Dict[Tuple, List[Tuple]],
    junction_orbit_partitions: Dict[Tuple, List[Tuple]],
    shared_boundary: List[int],
    extra_boundary: List[int],
    out_aut_group: List[Dict[int, int]],
    state_extra_boundary: Optional[List[int]] = None,
    keep_shared: bool = False,
    out_cell_anchor_groups: Optional[List[List[int]]] = None,
    state_cell_anchor_groups: Optional[List[List[int]]] = None,
    junction_cell_anchor_groups: Optional[List[List[int]]] = None,
    enumerate_junction_internally: bool = False,
) -> Dict[Tuple[Tuple, Tuple, Tuple], TuttePolynomial]:
    """Precompute M[O_state, O_junction, O_out] using Aut canonical form.

    ORBIT SHORTCUT: pick rep_state ∈ O_state, iterate all P_junc, multiply by
    |O_state|. Math: σ ∈ Aut_state preserving split induces σ' ∈ out_aut
    acting on state_left only; canonical(σ(P_out)) = canonical(P_out).
    Validated correct on K_4 and K_{4,4} cases.

    Optimization: precomputes (xy-1)^d as raw dicts, accumulates raw-dict
    arithmetic to avoid TuttePolynomial encode/decode cycles.

    Args:
        keep_shared: when True, shared_boundary positions are RETAINED in
            the output partition (the standard mode consumes them via the
            vertex-sum convolution, which marginalizes shared positions out).
            Used when the new piece's "right" anchors are physically the
            same vertices as state's existing shared boundary — i.e.,
            anchor-sharing cells (Cm3 interior). The shared positions
            persist for the next junction step.
        out_cell_anchor_groups: when provided, output canonical keys are
            computed via per-cell structural canonicalization (block-shape
            multiset over per-cell anchor groups). Coarser than aut-based
            grouping; orbits captured exactly under S_n^N (independent
            permutation within each cell's anchor group). Use this when
            state spans multiple cells and per-cell aut applies independently.
        state_cell_anchor_groups: when provided AND state_orbit_partitions
            holds only one rep per orbit, n_state is computed analytically
            via per_cell_orbit_size (avoiding the need to enumerate all
            partitions in the orbit explicitly).
        junction_cell_anchor_groups: SYMMETRIC counterpart to
            state_cell_anchor_groups for the junction side. When provided
            AND junction_orbit_partitions holds only one rep per orbit,
            n_junc is computed analytically via per_cell_orbit_size.
            Used by tree DP at cell-merge step where the "junction" is
            actually a child subtree returned in per-cell form. Without
            this, the M-table iterates only [rep] per junction orbit,
            missing other orbit members → wrong convolution result.
            The fallback (expand junction to uncompressed) loses all
            junction-side compression; this parameter preserves it.

            CORRECTNESS NOTE: passing junction_cell_anchor_groups + single
            rep (n_junc > 1) is sound ONLY in the trivial-orbit case
            (every state and junction orbit has size 1) OR when
            enumerate_junction_internally=True. Otherwise pair-orbit
            structure under the diagonal group action is not captured
            by the n_state × n_junc scalar → wrong M-table values.
            Used safely in the tree DP cell-merge step under that
            audit; downstream 2D-grid composition must use
            enumerate_junction_internally instead.
        enumerate_junction_internally: when True AND
            junction_cell_anchor_groups is provided AND each
            junction_orbit_partitions[O_junc] holds a single rep,
            the function ENUMERATES the orbit members of each O_junc
            internally (one orbit at a time, memory-bounded). The
            inner loop then iterates over enumerated members and
            n_junc is set to 1 (orbit-shortcut on junction side
            disabled). This is the correct path for 2D grid
            composition where the caller wants memory-bounded
            enumeration without materializing Bell(boundary) members
            in a single dict. Costs the same time as caller-side
            enumeration but avoids the giant materialized dict.
    """
    if state_extra_boundary is None:
        state_extra_boundary = []
    full_universe = list(state_extra_boundary) + list(shared_boundary) + list(extra_boundary)
    if keep_shared:
        out_boundary = (
            list(state_extra_boundary) + list(shared_boundary) + list(extra_boundary)
        )
    else:
        out_boundary = list(state_extra_boundary) + list(extra_boundary)

    max_d = len(shared_boundary)
    xy_minus_1_dict = {(1, 1): 1, (1, 0): -1, (0, 1): -1, (0, 0): 1}
    xy_powers_dict: List[Dict[Tuple[int, int], int]] = [{(0, 0): 1}]
    for k in range(1, max_d + 1):
        prev = xy_powers_dict[-1]
        new: Dict[Tuple[int, int], int] = defaultdict(int)
        for (i1, j1), c1 in prev.items():
            for (i2, j2), c2 in xy_minus_1_dict.items():
                new[(i1 + i2, j1 + j2)] += c1 * c2
        xy_powers_dict.append(dict(new))

    junc_data: Dict[Tuple, list] = {}
    if enumerate_junction_internally and junction_cell_anchor_groups is not None:
        # Streaming enumeration: expand each orbit's single rep to all
        # members via per-cell S_n^N action. Memory bounded per-orbit.
        for O_junc, pj_list in junction_orbit_partitions.items():
            rep = pj_list[0]
            members = _expand_per_cell_orbit_members(
                rep, junction_cell_anchor_groups,
            )
            per_junc = []
            for P_junc in members:
                P_junc_S = restrict_partition(P_junc, shared_boundary)
                P_junc_ext_list = list(P_junc) + [(v,) for v in state_extra_boundary]
                P_junc_ext = tuple(sorted(P_junc_ext_list))
                per_junc.append((P_junc, P_junc_S, P_junc_ext))
            junc_data[O_junc] = per_junc
    else:
        for O_junc, pj_list in junction_orbit_partitions.items():
            per_junc = []
            for P_junc in pj_list:
                P_junc_S = restrict_partition(P_junc, shared_boundary)
                P_junc_ext_list = list(P_junc) + [(v,) for v in state_extra_boundary]
                P_junc_ext = tuple(sorted(P_junc_ext_list))
                per_junc.append((P_junc, P_junc_S, P_junc_ext))
            junc_data[O_junc] = per_junc

    # Try batched C extension when:
    # (a) per-cell canonicalization is requested (the most expensive case),
    # (b) state × junc total iterations exceed a threshold (otherwise marshal
    #     overhead exceeds savings),
    # (c) the n_junc analytical factor isn't needed — either
    #     junction_cell_anchor_groups is None (each partition is its own
    #     orbit, n_junc=1), OR enumerate_junction_internally=True (junc_data
    #     already contains the expanded orbit members, also n_junc=1 per
    #     entry).
    n_state_total = len(state_orbit_partitions)
    n_junc_total = sum(len(pj) for pj in junc_data.values())
    BATCH_C_THRESHOLD = 100  # low threshold to maximize C ext coverage
    if (out_cell_anchor_groups is not None
            and state_cell_anchor_groups is not None
            and (junction_cell_anchor_groups is None
                 or enumerate_junction_internally)
            and n_state_total * n_junc_total >= BATCH_C_THRESHOLD):
        try:
            from ._partition_c import precompute_M_batched_inner_c
            n_state_per_orbit = {
                O_state: per_cell_orbit_size(O_state, state_cell_anchor_groups)
                for O_state in state_orbit_partitions
            }
            M_dict_c = precompute_M_batched_inner_c(
                state_orbit_partitions=state_orbit_partitions,
                junc_data_per_orbit=junc_data,
                state_extra_boundary=state_extra_boundary,
                extra_boundary=extra_boundary,
                shared_boundary=shared_boundary,
                out_boundary=out_boundary,
                out_cell_anchor_groups=out_cell_anchor_groups,
                n_state_per_orbit=n_state_per_orbit,
                xy_powers_dict=xy_powers_dict,
            )
            if M_dict_c is not None:
                # Convert per-cell canonical keys (tuples of shape tuples)
                # to TuttePolynomial values.
                M_out: Dict[Tuple, TuttePolynomial] = {}
                for key, val_dict in M_dict_c.items():
                    nonzero = {k: v for k, v in val_dict.items() if v != 0}
                    M_out[key] = TuttePolynomial.from_coefficients(nonzero)
                return M_out
        except Exception:
            pass  # fall back to Python

    M_dict: Dict[Tuple, Dict[Tuple[int, int], int]] = defaultdict(dict)
    # Pre-compute n_junc per junction orbit when junction_cell_anchor_groups
    # provided (symmetric to n_state path).
    if junction_cell_anchor_groups is not None:
        n_junc_per_orbit: Dict[Tuple, int] = {
            O_junc: per_cell_orbit_size(O_junc, junction_cell_anchor_groups)
            for O_junc in junction_orbit_partitions
        }
    else:
        n_junc_per_orbit = {}

    for O_state, ps_list in state_orbit_partitions.items():
        if state_cell_anchor_groups is not None:
            n_state = per_cell_orbit_size(O_state, state_cell_anchor_groups)
        else:
            n_state = len(ps_list)
        rep_state = ps_list[0]
        P_state_ext_list = list(rep_state) + [(v,) for v in extra_boundary]
        P_state_ext = tuple(sorted(P_state_ext_list))
        P_state_S = restrict_partition(rep_state, shared_boundary)

        for O_junc, _pj_list in junction_orbit_partitions.items():
            if enumerate_junction_internally:
                # Iterating all members directly; no scalar shortcut needed.
                n_junc = 1
            elif junction_cell_anchor_groups is not None:
                n_junc = n_junc_per_orbit.get(O_junc, 1)
            else:
                n_junc = 1
            for (P_junc, P_junc_S, P_junc_ext) in junc_data[O_junc]:
                d = delta(P_state_S, P_junc_S, shared_boundary)
                if d < 0:
                    continue
                joint = join_partitions(P_state_ext, P_junc_ext, full_universe)
                P_out = restrict_partition(joint, out_boundary)
                if out_cell_anchor_groups is not None:
                    O_out = (
                        per_cell_canonical_key(P_out, out_cell_anchor_groups)
                        if out_boundary else ()
                    )
                else:
                    O_out = canonical_partition(P_out, out_aut_group) if out_boundary else ()
                target = M_dict[(O_state, O_junc, O_out)]
                for k, v in xy_powers_dict[d].items():
                    target[k] = target.get(k, 0) + v * n_state * n_junc

    M: Dict[Tuple, TuttePolynomial] = {}
    for key, val_dict in M_dict.items():
        nonzero = {k: v for k, v in val_dict.items() if v != 0}
        M[key] = TuttePolynomial.from_coefficients(nonzero)
    return M


def precompute_M_and_convolve_streaming(
    state_orbit_T: Dict[Tuple, TuttePolynomial],
    state_orbit_partitions: Dict[Tuple, List[Tuple]],
    junction_orbit_T: Dict[Tuple, TuttePolynomial],
    junction_orbit_partitions: Dict[Tuple, List[Tuple]],
    out_orbit_sizes: Dict[Tuple, int],
    shared_boundary: List[int],
    extra_boundary: List[int],
    out_aut_group: List[Dict[int, int]],
    state_extra_boundary: Optional[List[int]] = None,
    keep_shared: bool = False,
    out_cell_anchor_groups: Optional[List[List[int]]] = None,
    state_cell_anchor_groups: Optional[List[List[int]]] = None,
    junction_cell_anchor_groups: Optional[List[List[int]]] = None,
    enumerate_junction_internally: bool = False,
    chunk_size: int = 500,
    out_orbit_sizes_default: int = 1,
) -> Dict[Tuple, TuttePolynomial]:
    """Build M-table + convolve in CHUNKED state batches to bound peak memory.

    For each state-orbit batch of `chunk_size`: build the partial M-table
    on that slice, convolve into raw coefficient dicts, accumulate into a
    running raw-dict accumulator, then drop the partial M-table. Caps
    peak memory at ~(`chunk_size` / total_state_orbits) × full-M-table size.

    The accumulator stays as `Dict[O_out, Dict[(xpow, ypow), int]]` (raw
    integer coefficient dicts) throughout the streaming loop — only at
    the very end do we apply the orbit-size division and wrap into
    `TuttePolynomial` objects. This avoids the encode/decode cycle that
    `TuttePolynomial.__add__` incurs on every chunk per key (a major
    bottleneck on Cm₃-scale problems before the raw-dict path).

    `out_orbit_sizes` may be incomplete (e.g., when populated analytically
    from M-table keys). Any `O_out` not present uses
    `out_orbit_sizes_default` (= 1, matching `orbit_convolve`'s default).
    """
    canons = list(state_orbit_partitions.keys())
    # Raw-dict accumulator: O_out → coefficient dict (still undivided by
    # out_orbit_size). Division applied once at the end.
    out_raw: Dict[Tuple, Dict[Tuple[int, int], int]] = defaultdict(dict)
    # Pre-encode junction T_rooted into raw dicts once (shared across chunks).
    junc_dict = {ok: _poly_to_dict(v) for ok, v in junction_orbit_T.items()}

    for start in range(0, len(canons), chunk_size):
        chunk = canons[start:start + chunk_size]
        chunk_state_part = {c: state_orbit_partitions[c] for c in chunk}
        chunk_state_T = {c: state_orbit_T[c] for c in chunk}
        M_chunk = precompute_M_table(
            chunk_state_part, junction_orbit_partitions,
            shared_boundary=shared_boundary,
            extra_boundary=extra_boundary,
            out_aut_group=out_aut_group,
            state_extra_boundary=state_extra_boundary,
            keep_shared=keep_shared,
            out_cell_anchor_groups=out_cell_anchor_groups,
            state_cell_anchor_groups=state_cell_anchor_groups,
            junction_cell_anchor_groups=junction_cell_anchor_groups,
            enumerate_junction_internally=enumerate_junction_internally,
        )
        if out_cell_anchor_groups is not None:
            for (_, _, O_out) in M_chunk.keys():
                if O_out not in out_orbit_sizes:
                    out_orbit_sizes[O_out] = per_cell_orbit_size(
                        O_out, out_cell_anchor_groups,
                    )
        # Raw-dict convolution: state × junc × M into out_raw (undivided).
        # Mirrors `orbit_convolve`'s grouping but accumulates into the
        # streaming-level dict.
        state_dict = {ok: _poly_to_dict(chunk_state_T[ok]) for ok in chunk}
        M_dict = {key: _poly_to_dict(val) for key, val in M_chunk.items()}
        grouped: Dict[Tuple[Tuple, Tuple], List[Tuple[Tuple, Dict[Tuple[int, int], int]]]] = (
            defaultdict(list)
        )
        for (O_state, O_junc, O_out), m_val in M_dict.items():
            grouped[(O_state, O_junc)].append((O_out, m_val))
        for (O_state, O_junc), out_entries in grouped.items():
            v1 = state_dict.get(O_state)
            v2 = junc_dict.get(O_junc)
            if v1 is None or v2 is None:
                continue
            sj = _dict_mul(v1, v2)
            if not sj:
                continue
            for (O_out, m_val) in out_entries:
                contrib = _dict_mul(sj, m_val)
                _dict_add(out_raw[O_out], contrib)
        # M_chunk + chunk_out drop out of scope here.

    # Apply orbit-size division + final TuttePolynomial wrap.
    out_T: Dict[Tuple, TuttePolynomial] = {}
    for O_out, val_dict in out_raw.items():
        size = out_orbit_sizes.get(O_out, out_orbit_sizes_default)
        if size > 1:
            divided: Dict[Tuple[int, int], int] = {}
            for k, c in val_dict.items():
                if c % size != 0:
                    raise ValueError(
                        f"Non-divisible coeff {c} by orbit size {size}"
                    )
                if c != 0:
                    divided[k] = c // size
            val_dict = divided
        nonzero = {k: v for k, v in val_dict.items() if v != 0}
        out_T[O_out] = TuttePolynomial.from_coefficients(nonzero)
    return out_T


def _build_junc_data_mod(
    junction_orbit_partitions: Dict[Tuple, List[Tuple]],
    shared_boundary: List[int],
    state_extra_boundary: List[int],
    junction_cell_anchor_groups: Optional[List[List[int]]],
    enumerate_junction_internally: bool,
) -> Dict[Tuple, list]:
    """Build the per-orbit junction partition data (restrictions + extensions).

    Hoisted out of `precompute_M_table_mod` so the streaming wrapper can
    compute it once and reuse across chunks (otherwise the
    `_expand_per_cell_orbit_members` cost is paid once per chunk —
    devastating for Cm₃-scale where there are 4096 junction orbits with
    ~200 ms expansion each).
    """
    import os as _os, time as _t
    _DBG = _os.environ.get("TUTTE_DEBUG_AGG", "0") != "0"
    if _DBG:
        _t0 = _t.time()
        print(f"  [build_junc_data] enum={enumerate_junction_internally} "
              f"n_orbits={len(junction_orbit_partitions)} "
              f"jcag={junction_cell_anchor_groups is not None}", flush=True)
    junc_data: Dict[Tuple, list] = {}
    if enumerate_junction_internally and junction_cell_anchor_groups is not None:
        # Pre-build G_flat ONCE for the cell_anchor_groups; reuse across
        # all orbit expansions. Universe = union of cell_anchor_groups
        # positions + any positions in junction partitions (which are fixed
        # by the per-cell perms but need to appear in the universe).
        universe_set: set = set()
        for grp in junction_cell_anchor_groups:
            for v in grp:
                universe_set.add(v)
        for _, pj_list in junction_orbit_partitions.items():
            for P in pj_list:
                for block in P:
                    for v in block:
                        universe_set.add(v)
        universe = list(universe_set)

        _g_build_t0 = _t.time() if _DBG else 0
        G_flat, n_G = _build_per_cell_aut_flat(
            junction_cell_anchor_groups, universe,
        )
        if _DBG:
            print(f"  [build_junc_data] G_flat built: n_G={n_G} n_universe={len(universe)} "
                  f"({_t.time()-_g_build_t0:.2f}s)", flush=True)

        # Try C-ext path; fall back to Python if unavailable.
        _use_c = True
        try:
            from ._partition_c import _get_lib
            lib, _ffi = _get_lib()
            G_arr_cffi = _ffi.new("int[]", G_flat)
        except Exception:
            _use_c = False

        _expansion_t = 0.0
        _restrict_t = 0.0
        _n_done = 0
        _n_members_total = 0
        _c_used = 0
        _py_fallback = 0
        for O_junc, pj_list in junction_orbit_partitions.items():
            rep = pj_list[0]
            _t1 = _t.time() if _DBG else 0
            members = None
            if _use_c:
                canonical = per_cell_canonical_key(rep, junction_cell_anchor_groups)
                target_size = per_cell_orbit_size(canonical, junction_cell_anchor_groups)
                members = _expand_per_cell_orbit_members_c_batch(
                    rep, target_size, universe, G_arr_cffi, n_G, lib, _ffi,
                )
                if members is not None:
                    _c_used += 1
            if members is None:
                members = _expand_per_cell_orbit_members(
                    rep, junction_cell_anchor_groups,
                )
                _py_fallback += 1
            if _DBG:
                _expansion_t += _t.time() - _t1
                _t2 = _t.time()
            per_junc = []
            for P_junc in members:
                P_junc_S = restrict_partition(P_junc, shared_boundary)
                P_junc_ext_list = list(P_junc) + [(v,) for v in state_extra_boundary]
                P_junc_ext = tuple(sorted(P_junc_ext_list))
                per_junc.append((P_junc, P_junc_S, P_junc_ext))
            if _DBG:
                _restrict_t += _t.time() - _t2
                _n_members_total += len(members)
                _n_done += 1
                if _n_done % 500 == 0:
                    _elapsed = _t.time() - _t0
                    print(f"  [build_junc_data] {_n_done}/{len(junction_orbit_partitions)} orbits "
                          f"({_n_members_total} members), exp={_expansion_t:.1f}s "
                          f"restrict={_restrict_t:.1f}s total={_elapsed:.1f}s "
                          f"c={_c_used} py={_py_fallback}", flush=True)
            junc_data[O_junc] = per_junc
        if _DBG:
            print(f"  [build_junc_data] DONE: {_n_done} orbits, {_n_members_total} members, "
                  f"exp={_expansion_t:.1f}s restrict={_restrict_t:.1f}s total={_t.time()-_t0:.1f}s "
                  f"c={_c_used} py={_py_fallback}",
                  flush=True)
    else:
        for O_junc, pj_list in junction_orbit_partitions.items():
            per_junc = []
            for P_junc in pj_list:
                P_junc_S = restrict_partition(P_junc, shared_boundary)
                P_junc_ext_list = list(P_junc) + [(v,) for v in state_extra_boundary]
                P_junc_ext = tuple(sorted(P_junc_ext_list))
                per_junc.append((P_junc, P_junc_S, P_junc_ext))
            junc_data[O_junc] = per_junc
        if _DBG:
            print(f"  [build_junc_data] DONE (non-enum): {_t.time()-_t0:.1f}s", flush=True)
    return junc_data


def precompute_M_table_mod(
    state_orbit_partitions: Dict[Tuple, List[Tuple]],
    junction_orbit_partitions: Dict[Tuple, List[Tuple]],
    shared_boundary: List[int],
    extra_boundary: List[int],
    out_aut_group: List[Dict[int, int]],
    x_val: int,
    y_val: int,
    p: int,
    state_extra_boundary: Optional[List[int]] = None,
    keep_shared: bool = False,
    out_cell_anchor_groups: Optional[List[List[int]]] = None,
    state_cell_anchor_groups: Optional[List[List[int]]] = None,
    junction_cell_anchor_groups: Optional[List[List[int]]] = None,
    enumerate_junction_internally: bool = False,
    junc_data: Optional[Dict[Tuple, list]] = None,
) -> Dict[Tuple[Tuple, Tuple, Tuple], int]:
    """Fast modular variant of `precompute_M_table`.

    Each `M[O_state, O_junc, O_out]` entry is a single integer mod `p` —
    the polynomial M-table entry evaluated at `(x_val, y_val) mod p` —
    accumulated directly without any polynomial allocation. Replaces the
    path that built a polynomial M-table and then evaluated
    each entry post-hoc. Expected ~100× speedup at Cm₃ scale because the
    inner loop replaces a per-pair polynomial multiply-accumulate (O(terms)
    int operations on ~hundreds-of-term coefficient dicts) with a single
    modular multiply-add.

    All orbit / partition / anchor semantics mirror `precompute_M_table`
    exactly; only the arithmetic is modular.
    """
    if state_extra_boundary is None:
        state_extra_boundary = []
    full_universe = list(state_extra_boundary) + list(shared_boundary) + list(extra_boundary)
    if keep_shared:
        out_boundary = (
            list(state_extra_boundary) + list(shared_boundary) + list(extra_boundary)
        )
    else:
        out_boundary = list(state_extra_boundary) + list(extra_boundary)

    # ((x-1)(y-1))^d evaluated at (x_val, y_val) mod p, for d = 0 .. |S|.
    # Matches `precompute_M_table`'s xy_powers_dict, which despite the
    # legacy name encodes (x-1)(y-1) = xy - x - y + 1, not (xy - 1).
    max_d = len(shared_boundary)
    base_mod = ((x_val - 1) * (y_val - 1)) % p
    xy_pow_mod: List[int] = [1 % p]
    for _ in range(max_d):
        xy_pow_mod.append(xy_pow_mod[-1] * base_mod % p)

    if junc_data is None:
        junc_data = _build_junc_data_mod(
            junction_orbit_partitions, shared_boundary, state_extra_boundary,
            junction_cell_anchor_groups, enumerate_junction_internally,
        )

    if junction_cell_anchor_groups is not None:
        n_junc_per_orbit: Dict[Tuple, int] = {
            O_junc: per_cell_orbit_size(O_junc, junction_cell_anchor_groups)
            for O_junc in junction_orbit_partitions
        }
    else:
        n_junc_per_orbit = {}

    # Try batched C extension (mirrors the polynomial-path dispatch).
    # Same gating: per-cell canonicalization requested, junction either
    # trivially-orbited or expanded internally (n_junc=1 per pair), and
    # iteration count above the marshal-overhead threshold.
    n_state_total = len(state_orbit_partitions)
    n_junc_total = sum(len(pj) for pj in junc_data.values())
    BATCH_C_THRESHOLD = 100
    if (out_cell_anchor_groups is not None
            and state_cell_anchor_groups is not None
            and (junction_cell_anchor_groups is None
                 or enumerate_junction_internally)
            and n_state_total * n_junc_total >= BATCH_C_THRESHOLD):
        try:
            from ._partition_c import precompute_M_batched_inner_c_mod
            n_state_per_orbit = {
                O_state: per_cell_orbit_size(O_state, state_cell_anchor_groups)
                for O_state in state_orbit_partitions
            }
            M_int_c = precompute_M_batched_inner_c_mod(
                state_orbit_partitions=state_orbit_partitions,
                junc_data_per_orbit=junc_data,
                state_extra_boundary=state_extra_boundary,
                extra_boundary=extra_boundary,
                shared_boundary=shared_boundary,
                out_boundary=out_boundary,
                out_cell_anchor_groups=out_cell_anchor_groups,
                n_state_per_orbit=n_state_per_orbit,
                xy_pow_mod=xy_pow_mod,
                p=p,
            )
            if M_int_c is not None:
                return M_int_c
        except Exception:
            pass  # fall back to Python

    M_int: Dict[Tuple, int] = defaultdict(int)
    for O_state, ps_list in state_orbit_partitions.items():
        if state_cell_anchor_groups is not None:
            n_state = per_cell_orbit_size(O_state, state_cell_anchor_groups)
        else:
            n_state = len(ps_list)
        rep_state = ps_list[0]
        P_state_ext_list = list(rep_state) + [(v,) for v in extra_boundary]
        P_state_ext = tuple(sorted(P_state_ext_list))
        P_state_S = restrict_partition(rep_state, shared_boundary)

        for O_junc, _pj_list in junction_orbit_partitions.items():
            if enumerate_junction_internally:
                n_junc = 1
            elif junction_cell_anchor_groups is not None:
                n_junc = n_junc_per_orbit.get(O_junc, 1)
            else:
                n_junc = 1
            ns_nj_mod = n_state * n_junc % p
            if ns_nj_mod == 0:
                continue
            for (P_junc, P_junc_S, P_junc_ext) in junc_data[O_junc]:
                d = delta(P_state_S, P_junc_S, shared_boundary)
                if d < 0:
                    continue
                joint = join_partitions(P_state_ext, P_junc_ext, full_universe)
                P_out = restrict_partition(joint, out_boundary)
                if out_cell_anchor_groups is not None:
                    O_out = (
                        per_cell_canonical_key(P_out, out_cell_anchor_groups)
                        if out_boundary else ()
                    )
                else:
                    O_out = canonical_partition(P_out, out_aut_group) if out_boundary else ()
                key = (O_state, O_junc, O_out)
                add = ns_nj_mod * xy_pow_mod[d] % p
                M_int[key] = (M_int[key] + add) % p

    return dict(M_int)


def precompute_M_and_convolve_streaming_mod(
    state_orbit_T_mod: Dict[Tuple, int],
    state_orbit_partitions: Dict[Tuple, List[Tuple]],
    junction_orbit_T_mod: Dict[Tuple, int],
    junction_orbit_partitions: Dict[Tuple, List[Tuple]],
    out_orbit_sizes: Dict[Tuple, int],
    p: int,
    x_val: int,
    y_val: int,
    shared_boundary: List[int],
    extra_boundary: List[int],
    out_aut_group: List[Dict[int, int]],
    state_extra_boundary: Optional[List[int]] = None,
    keep_shared: bool = False,
    out_cell_anchor_groups: Optional[List[List[int]]] = None,
    state_cell_anchor_groups: Optional[List[List[int]]] = None,
    junction_cell_anchor_groups: Optional[List[List[int]]] = None,
    enumerate_junction_internally: bool = False,
    chunk_size: int = 500,
    out_orbit_sizes_default: int = 1,
) -> Dict[Tuple, int]:
    """Modular variant of `precompute_M_and_convolve_streaming`.

    Operates in integers mod `p` throughout via `precompute_M_table_mod`
    — each chunk's M-table is built as ints
    directly, with no polynomial allocation. `state_orbit_T_mod` and
    `junction_orbit_T_mod` map orbit canonical → `T_orbit(x_val, y_val)
    mod p`. Output: orbit canonical → integer mod p.

    For correctness validation: feeding this with
    `T_oracle.evaluate_mod(x_val, y_val, p)` integer state inputs should
    yield the same integer as `T_engine(Cm).evaluate_mod(x_val, y_val, p)`.

    `out_orbit_sizes`: integer division becomes modular multiplication
    by the inverse (Fermat's little theorem). Each `out_orbit_size` must
    be coprime with `p` (always true when `p` is a prime larger than the
    orbit-size factors).
    """
    canons = list(state_orbit_partitions.keys())
    out_mod: Dict[Tuple, int] = defaultdict(int)

    # Build junction data ONCE (it's chunk-independent). For Cm3-class
    # graphs `_expand_per_cell_orbit_members` is ~200 ms × 4096 orbits =
    # 13 min — would otherwise be paid once per chunk.
    state_extra = [] if state_extra_boundary is None else state_extra_boundary
    junc_data = _build_junc_data_mod(
        junction_orbit_partitions, shared_boundary, state_extra,
        junction_cell_anchor_groups, enumerate_junction_internally,
    )

    # Try single-pass C-ext (collapses M_int build + convolve into one
    # Python loop with 1-tuple O_out key). Only available when full C-ext
    # gating is satisfied AND state/junc values are non-empty.
    n_state_total = len(state_orbit_partitions)
    n_junc_total = sum(len(pj) for pj in junc_data.values())
    can_use_single_pass = (
        out_cell_anchor_groups is not None
        and state_cell_anchor_groups is not None
        and (junction_cell_anchor_groups is None
             or enumerate_junction_internally)
        and n_state_total * n_junc_total >= 100  # BATCH_C_THRESHOLD
        and len(state_orbit_T_mod) > 0
        and len(junction_orbit_T_mod) > 0
    )

    for start in range(0, len(canons), chunk_size):
        chunk = canons[start:start + chunk_size]
        chunk_state_part = {c: state_orbit_partitions[c] for c in chunk}

        if can_use_single_pass:
            try:
                # C-side hash-map aggregation eliminates Python dict ops in
                # the inner aggregation loop. Activated via env var
                # TUTTE_R18_AGGREGATE (default on); falls back transparently
                # to the per-chunk convolve on capacity errors.
                import os
                use_r18 = os.environ.get("TUTTE_R18_AGGREGATE", "1") != "0"
                if use_r18:
                    from ._partition_c import precompute_and_aggregate_c_mod
                    chunk_compute = precompute_and_aggregate_c_mod
                else:
                    from ._partition_c import precompute_and_convolve_c_mod
                    chunk_compute = precompute_and_convolve_c_mod
                n_state_per_orbit = {
                    O_state: per_cell_orbit_size(O_state, state_cell_anchor_groups)
                    for O_state in chunk_state_part
                }
                # R19 H-bucketing activation: pass state_cell_anchor_groups to
                # precompute_and_aggregate_c_mod so the per-state-orbit
                # stabilizer H = stab_G(state_rep|shared_boundary) is computed
                # and used to bucket junction members. For non-R18 path
                # (precompute_and_convolve_c_mod), R19 isn't supported, so we
                # only pass when using R18.
                _aggregate_kwargs = {
                    "state_orbit_partitions": chunk_state_part,
                    "junc_data_per_orbit": junc_data,
                    "state_extra_boundary": state_extra,
                    "extra_boundary": extra_boundary,
                    "shared_boundary": shared_boundary,
                    "out_boundary": (state_extra + list(shared_boundary) + list(extra_boundary))
                                 if keep_shared
                                 else (state_extra + list(extra_boundary)),
                    "out_cell_anchor_groups": out_cell_anchor_groups,
                    "n_state_per_orbit": n_state_per_orbit,
                    "state_orbit_T_mod": state_orbit_T_mod,
                    "junction_orbit_T_mod": junction_orbit_T_mod,
                    "xy_pow_mod": [pow(((x_val - 1) * (y_val - 1)) % p, d, p)
                                for d in range(len(shared_boundary) + 1)],
                    "p": p,
                }
                if use_r18:
                    # R19 only available via precompute_and_aggregate_c_mod.
                    _aggregate_kwargs["state_cell_anchor_groups"] = state_cell_anchor_groups
                chunk_out_mod = chunk_compute(**_aggregate_kwargs)
                if chunk_out_mod is None and use_r18:
                    # Hash-map aggregator returned None (capacity overflow
                    # or similar) — fall back to the per-chunk convolve
                    # for this chunk before resorting to the two-pass
                    # Python path below.
                    from ._partition_c import precompute_and_convolve_c_mod
                    chunk_out_mod = precompute_and_convolve_c_mod(
                        state_orbit_partitions=chunk_state_part,
                        junc_data_per_orbit=junc_data,
                        state_extra_boundary=state_extra,
                        extra_boundary=extra_boundary,
                        shared_boundary=shared_boundary,
                        out_boundary=(state_extra + list(shared_boundary) + list(extra_boundary))
                                     if keep_shared
                                     else (state_extra + list(extra_boundary)),
                        out_cell_anchor_groups=out_cell_anchor_groups,
                        n_state_per_orbit=n_state_per_orbit,
                        state_orbit_T_mod=state_orbit_T_mod,
                        junction_orbit_T_mod=junction_orbit_T_mod,
                        xy_pow_mod=[pow(((x_val - 1) * (y_val - 1)) % p, d, p)
                                    for d in range(len(shared_boundary) + 1)],
                        p=p,
                    )
                if chunk_out_mod is not None:
                    for O_out, val in chunk_out_mod.items():
                        if out_cell_anchor_groups is not None and O_out not in out_orbit_sizes:
                            out_orbit_sizes[O_out] = per_cell_orbit_size(
                                O_out, out_cell_anchor_groups,
                            )
                        out_mod[O_out] = (out_mod[O_out] + val) % p
                    continue  # skip the fallback path for this chunk
            except Exception:
                pass  # fall through to two-pass

        M_chunk = precompute_M_table_mod(
            chunk_state_part, junction_orbit_partitions,
            shared_boundary=shared_boundary,
            extra_boundary=extra_boundary,
            out_aut_group=out_aut_group,
            x_val=x_val, y_val=y_val, p=p,
            state_extra_boundary=state_extra_boundary,
            keep_shared=keep_shared,
            out_cell_anchor_groups=out_cell_anchor_groups,
            state_cell_anchor_groups=state_cell_anchor_groups,
            junction_cell_anchor_groups=junction_cell_anchor_groups,
            enumerate_junction_internally=enumerate_junction_internally,
            junc_data=junc_data,
        )
        if out_cell_anchor_groups is not None:
            for (_, _, O_out) in M_chunk.keys():
                if O_out not in out_orbit_sizes:
                    out_orbit_sizes[O_out] = per_cell_orbit_size(
                        O_out, out_cell_anchor_groups,
                    )
        for (O_state, O_junc, O_out), m_val in M_chunk.items():
            if m_val == 0:
                continue
            sv = state_orbit_T_mod.get(O_state, 0)
            jv = junction_orbit_T_mod.get(O_junc, 0)
            if sv == 0 or jv == 0:
                continue
            contrib = sv * jv % p * m_val % p
            out_mod[O_out] = (out_mod[O_out] + contrib) % p

    # Apply orbit-size division: multiply by modular inverse.
    result: Dict[Tuple, int] = {}
    for O_out, val in out_mod.items():
        size = out_orbit_sizes.get(O_out, out_orbit_sizes_default)
        if size > 1:
            inv = pow(size, p - 2, p)
            val = val * inv % p
        result[O_out] = val
    return result


def _build_per_cell_aut_flat(
    cell_anchor_groups: List[List[int]],
    universe: List[int],
) -> Tuple[List[int], int]:
    """Build flat array of all per-cell S_n^N perm-images for a fixed
    universe. Universe must include all positions in cell_anchor_groups
    plus any other positions that appear in partitions to be canonicalized
    (those are fixed by the perms).

    Returns (G_flat, n_G) where G_flat[g * n_universe + i] = image of
    universe[i] under perm g (encoded as index into universe).
    """
    from itertools import permutations, product
    pos_to_idx = {pos: i for i, pos in enumerate(universe)}
    per_group_perms = [list(permutations(g)) for g in cell_anchor_groups]
    n_G = 1
    for pgp in per_group_perms:
        n_G *= len(pgp)
    G_flat: List[int] = []
    for perm_combo in product(*per_group_perms):
        relabel = {}
        for orig_g, new_perm in zip(cell_anchor_groups, perm_combo):
            for orig, new in zip(orig_g, new_perm):
                relabel[orig] = new
        for pos in universe:
            tgt = relabel.get(pos, pos)
            G_flat.append(pos_to_idx[tgt])
    return G_flat, n_G


def _expand_per_cell_orbit_members_c_batch(
    rep: Tuple[Tuple[int, ...], ...],
    target_size: int,
    universe: List[int],
    G_arr,  # cffi int[] for G_flat
    n_G: int,
    lib, _ffi,
):
    """C-ext orbit expansion with pre-built G array (caller-owned).

    Returns list of canonical-sorted partition tuples, or None on overflow.
    """
    from ..roots._partition_c import _encode_partition, _decode_partition
    n_univ = len(universe)
    if n_univ > 256:
        return None
    pos_to_idx = {pos: i for i, pos in enumerate(universe)}
    idx_to_pos = list(universe)
    try:
        rep_enc = _encode_partition(rep, pos_to_idx)
    except KeyError:
        return None
    rep_arr = _ffi.new("int[]", rep_enc)
    # Output capacity: each member's encoding ≈ rep_enc_len. Allow target_size × 2.
    out_capacity = max(target_size * len(rep_enc) * 2, 1024)
    out_arr = _ffi.new("int[]", out_capacity)
    out_off_arr = _ffi.new("int[]", target_size + 2)
    n_found = lib.expand_orbit_members_c(
        rep_arr, len(rep_enc),
        G_arr, n_G, n_univ,
        target_size,
        out_arr, out_off_arr, out_capacity,
        target_size,
    )
    if n_found < 0:
        return None
    members = []
    for m_idx in range(n_found):
        m_start = out_off_arr[m_idx]
        m_end = out_off_arr[m_idx + 1]
        m_enc = [int(out_arr[i]) for i in range(m_start, m_end)]
        members.append(_decode_partition(m_enc, len(m_enc), idx_to_pos))
    return members


def _expand_per_cell_orbit_members(
    rep: Tuple[Tuple[int, ...], ...],
    cell_anchor_groups: List[List[int]],
):
    """Enumerate all distinct partitions in the per-cell orbit of `rep`
    by applying S_n^N permutations within each cell anchor group.

    Returns a list of distinct partitions in canonical-sorted form.

    Early-terminates once `per_cell_orbit_size` worth of members have
    been found. The orbit size is an analytical invariant of the
    canonical key — for Cm₃ row partitions, most orbits are tiny while
    `|S_n^N|` is huge, so brute-forcing all perm combos wastes ~99% of
    iterations.
    """
    from itertools import permutations, product
    canonical = per_cell_canonical_key(rep, cell_anchor_groups)
    target_size = per_cell_orbit_size(canonical, cell_anchor_groups)
    per_group_perms = [list(permutations(g)) for g in cell_anchor_groups]
    seen = set()
    members = []
    for perm_combo in product(*per_group_perms):
        relabel_map = {}
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
            if len(members) >= target_size:
                return members
    return members


def _poly_to_dict(poly: TuttePolynomial) -> Dict[Tuple[int, int], int]:
    return {(i, j): c for i, j, c in poly.terms()}


# C extension for poly mul; falls back to Python on overflow / unavailable.
try:
    from .._polynomial_c import poly_mul as _poly_mul_dispatch
except Exception:
    _poly_mul_dispatch = None

try:
    from .._polynomial_c import \
        poly_mul_batched_c as _poly_mul_batched_dispatch
except Exception:
    _poly_mul_batched_dispatch = None


_BATCHED_MUL_MIN = 100000  # batching pays off only above ~100K pairs/batch


def _dict_mul(d1: Dict[Tuple[int, int], int],
              d2: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    """Multiply two polynomial coefficient dicts.

    Uses C extension via `_polynomial_c.poly_mul` when available; falls
    back to pure Python on overflow or if extension unavailable.
    """
    if _poly_mul_dispatch is not None and d1 and d2:
        result = _poly_mul_dispatch(d1, d2)
        if result is not None:
            return result
    result_dict: Dict[Tuple[int, int], int] = defaultdict(int)
    for (i1, j1), c1 in d1.items():
        for (i2, j2), c2 in d2.items():
            result_dict[(i1 + i2, j1 + j2)] += c1 * c2
    return result_dict


def _dict_add(d1: Dict[Tuple[int, int], int],
              d2: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    for k, v in d2.items():
        d1[k] = d1.get(k, 0) + v
    return d1


def orbit_convolve(
    state_orbit_T: Dict[Tuple, TuttePolynomial],
    junction_orbit_T: Dict[Tuple, TuttePolynomial],
    M_table: Dict[Tuple, TuttePolynomial],
    out_orbit_sizes: Dict[Tuple, int],
) -> Dict[Tuple, TuttePolynomial]:
    """Convolve at orbit level.

    V_out[O_out] = (Σ V_state · T_junc · M) / |O_out|

    Optimization: raw coefficient dicts in hot loop to avoid encode/decode.
    """
    state_dict = {ok: _poly_to_dict(v) for ok, v in state_orbit_T.items()}
    junc_dict = {ok: _poly_to_dict(v) for ok, v in junction_orbit_T.items()}
    M_dict = {key: _poly_to_dict(v) for key, v in M_table.items()}

    # Group M entries by (O_state, O_junc). For each group, compute
    # `state * junc` exactly ONCE, then multiply by each M-table-entry per
    # O_out. This eliminates redundant `state * junc` poly muls when many
    # O_out share the same (O_state, O_junc) pair.
    out_dict: Dict[Tuple, Dict[Tuple[int, int], int]] = defaultdict(dict)
    grouped: Dict[Tuple[Tuple, Tuple], List[Tuple[Tuple, Dict[Tuple[int, int], int]]]] = (
        defaultdict(list)
    )
    for (O_state, O_junc, O_out), m_val in M_dict.items():
        grouped[(O_state, O_junc)].append((O_out, m_val))

    for (O_state, O_junc), out_entries in grouped.items():
        v1 = state_dict.get(O_state)
        v2 = junc_dict.get(O_junc)
        if v1 is None or v2 is None:
            continue
        sj = _dict_mul(v1, v2)
        if not sj:
            continue
        for (O_out, m_val) in out_entries:
            contrib = _dict_mul(sj, m_val)
            _dict_add(out_dict[O_out], contrib)

    final: Dict[Tuple, TuttePolynomial] = {}
    for O_out, val_dict in out_dict.items():
        size = out_orbit_sizes.get(O_out, 1)
        if size > 1:
            divided = {}
            for k, c in val_dict.items():
                if c % size != 0:
                    raise ValueError(
                        f"Non-divisible coeff {c} by orbit size {size}"
                    )
                if c != 0:
                    divided[k] = c // size
            val_dict = divided
        nonzero = {k: v for k, v in val_dict.items() if v != 0}
        final[O_out] = TuttePolynomial.from_coefficients(nonzero)
    return final
