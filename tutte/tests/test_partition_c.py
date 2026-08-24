"""C extension correctness tests for the partition-DP fast path.

Two C entry points are validated against their pure-Python references:

- `precompute_and_aggregate_c_mod` matches `precompute_and_convolve_c_mod`
  (Python dict accumulator) bit-for-bit.
- `h_canonicalize_c_wrapper` (single) and `h_canonicalize_c_batched`
  (batched) compute the lex-min canonical form across H-permutations of
  a partition, matching `min(apply_perm_to_partition(P, h) for h in H)`.
"""
from __future__ import annotations

import itertools

import pytest

from tutte.roots._partition_c import (
    h_canonicalize_c_batched,
    h_canonicalize_c_wrapper,
    precompute_and_aggregate_c_mod,
    precompute_and_convolve_c_mod,
)
from tutte.roots.aut_orbit import apply_perm_to_partition


def _make_synthetic_inputs(p=1009):
    """Construct a tiny but non-trivial state/junction setup."""
    state_extra_boundary = []
    extra_boundary = [100, 101]
    shared_boundary = [10, 11]
    out_boundary = list(extra_boundary)

    P_state_a = (tuple(shared_boundary),)
    P_state_b = ((shared_boundary[0],), (shared_boundary[1],))

    state_orbit_partitions = {
        P_state_a: [P_state_a],
        P_state_b: [P_state_b],
    }

    P_junc_1 = (tuple(shared_boundary) + tuple(extra_boundary),)
    P_junc_1_S = (tuple(shared_boundary),)

    P_junc_2 = ((shared_boundary[0], extra_boundary[0]),
                (shared_boundary[1], extra_boundary[1]))
    P_junc_2_S = ((shared_boundary[0],), (shared_boundary[1],))

    junc_data_per_orbit = {
        P_junc_1: [(P_junc_1, P_junc_1_S, P_junc_1)],
        P_junc_2: [(P_junc_2, P_junc_2_S, P_junc_2)],
    }

    n_state_per_orbit = {P_state_a: 1, P_state_b: 1}
    state_orbit_T_mod = {P_state_a: 3, P_state_b: 5}
    junction_orbit_T_mod = {P_junc_1: 7, P_junc_2: 11}

    xy_pow_mod = [pow(((2 - 1) * (3 - 1)) % p, d, p)
                  for d in range(len(shared_boundary) + 1)]
    out_cell_anchor_groups = [extra_boundary]

    return dict(
        state_orbit_partitions=state_orbit_partitions,
        junc_data_per_orbit=junc_data_per_orbit,
        state_extra_boundary=state_extra_boundary,
        extra_boundary=extra_boundary,
        shared_boundary=shared_boundary,
        out_boundary=out_boundary,
        out_cell_anchor_groups=out_cell_anchor_groups,
        n_state_per_orbit=n_state_per_orbit,
        state_orbit_T_mod=state_orbit_T_mod,
        junction_orbit_T_mod=junction_orbit_T_mod,
        xy_pow_mod=xy_pow_mod,
        p=p,
    )


def test_matches_r17_synthetic():
    """hash-map aggregation (C ext) bit-for-bit matches Python dict accumulator."""
    kwargs = _make_synthetic_inputs()
    r17 = precompute_and_convolve_c_mod(**kwargs)
    r18 = precompute_and_aggregate_c_mod(**kwargs)
    assert r17 is not None and r18 is not None
    assert r17 == r18


def test_handles_empty_input():
    """Empty state/junction yields {} from both Python and C-ext paths."""
    kwargs = _make_synthetic_inputs()
    kwargs["state_orbit_partitions"] = {}
    kwargs["state_orbit_T_mod"] = {}
    kwargs["n_state_per_orbit"] = {}
    r17 = precompute_and_convolve_c_mod(**kwargs)
    r18 = precompute_and_aggregate_c_mod(**kwargs)
    assert r17 == r18


def test_handles_zero_jv():
    """All zero junction values → both paths return {}."""
    kwargs = _make_synthetic_inputs()
    P_junc_1 = list(kwargs["junction_orbit_T_mod"].keys())[0]
    P_junc_2 = list(kwargs["junction_orbit_T_mod"].keys())[1]
    kwargs["junction_orbit_T_mod"] = {P_junc_1: 0, P_junc_2: 0}
    r17 = precompute_and_convolve_c_mod(**kwargs)
    r18 = precompute_and_aggregate_c_mod(**kwargs)
    assert r17 == r18
    assert len(r18) == 0


def test_cm2_modular_point_matches_r17():
    """End-to-end: Cm_2 T(2,3) mod 1009 = 806 via both Python and C-ext paths."""
    import os
    try:
        import dwave_networkx as dnx
    except ImportError:
        pytest.skip("dwave_networkx unavailable")
    mod = pytest.importorskip(
        "tutte.research.scripts.cm_modular_interp",
        reason="cm_modular_interp lost (uncommitted at machine death); "
               "rebuild tracked in bead riff-ggi",
    )
    build_polynomial_state = mod.build_polynomial_state
    evaluate_modular_at_point = mod.evaluate_modular_at_point

    # Build state once; evaluate under Python then C-ext path.
    state = build_polynomial_state(2)

    import importlib
    import tutte.roots.cell_quotient_helpers as cqh

    os.environ["TUTTE_R18_AGGREGATE"] = "0"
    importlib.reload(cqh)
    v_r17 = evaluate_modular_at_point(state, 2, 3, 1009)

    os.environ["TUTTE_R18_AGGREGATE"] = "1"
    importlib.reload(cqh)
    v_r18 = evaluate_modular_at_point(state, 2, 3, 1009)

    assert v_r17 == 806
    assert v_r18 == 806
    assert v_r17 == v_r18


# =============================================================================
# H-canonicalize tests
# =============================================================================


def _s_n_perms(n):
    """All permutations in S_n as dicts on positions 0..n-1."""
    return [dict(zip(range(n), p)) for p in itertools.permutations(range(n))]


def test_h_canon_single_matches_python_s4():
    """Single-partition H-canonicalize matches Python reference under S_4."""
    H = _s_n_perms(4)
    universe = [0, 1, 2, 3]
    cases = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 1, 2),),
        ((0,), (1,), (2,), (3,)),
        ((0, 1, 2, 3),),
    ]
    for P in cases:
        py = min(apply_perm_to_partition(P, h) for h in H)
        c = h_canonicalize_c_wrapper(P, H, universe)
        assert c == py, f"Mismatch for P={P}: py={py}, c={c}"


def test_h_canon_non_zero_indexed_positions():
    """H-canonicalize handles non-zero-indexed universe (matches Python)."""
    universe = [10, 20, 30, 40, 50]
    H = _s_n_perms(5)
    H_pos = [{universe[i]: universe[h[i]] for i in range(5)} for h in H]
    P = ((10, 20), (30, 40), (50,))
    py = min(apply_perm_to_partition(P, h) for h in H_pos)
    c = h_canonicalize_c_wrapper(P, H_pos, universe)
    assert c == py


def test_h_canon_empty_H_returns_input():
    """Empty H acts as identity → returns canonical form of P."""
    P = ((1, 2), (3,))
    c = h_canonicalize_c_wrapper(P, [], [1, 2, 3])
    assert c == ((1, 2), (3,))


def test_h_canon_batched_matches_single_calls():
    """Batched H-canonicalize matches per-call h_canonicalize_c_wrapper."""
    universe = [0, 1, 2, 3, 4]
    H = _s_n_perms(5)
    P_list = [
        ((0, 1), (2, 3, 4)),
        ((0, 2), (1, 4), (3,)),
        ((0, 4), (1, 2, 3)),
        ((0,), (1,), (2,), (3,), (4,)),
    ]
    single = [h_canonicalize_c_wrapper(P, H, universe) for P in P_list]
    batched = h_canonicalize_c_batched(P_list, H, universe)
    assert batched == single


def test_h_canon_batched_matches_python_reference():
    """Batched C-ext matches Python `min(apply_perm…)` reference."""
    universe = [0, 1, 2, 3, 4, 5]
    # Use a subgroup of S_6: rotations by 2 positions (cycle 0,2,4 and 1,3,5)
    H = []
    for k in range(3):
        H.append({i: (i + 2 * k) % 6 for i in range(6)})
    P_list = [
        ((0, 1), (2, 3), (4, 5)),
        ((0, 3), (1, 4), (2, 5)),
        ((0, 1, 2), (3, 4, 5)),
    ]
    py = [min(apply_perm_to_partition(P, h) for h in H) for P in P_list]
    c = h_canonicalize_c_batched(P_list, H, universe)
    assert c == py


def test_expand_orbit_members_c_matches_python():
    """C-ext orbit expansion matches Python `_expand_per_cell_orbit_members`."""
    from tutte.roots.cell_quotient_helpers import (
        _build_per_cell_aut_flat,
        _expand_per_cell_orbit_members,
        _expand_per_cell_orbit_members_c_batch,
        per_cell_canonical_key,
        per_cell_orbit_size,
    )
    from tutte.roots._partition_c import _get_lib

    cell_anchor_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    universe = [0, 1, 2, 3, 4, 5, 6, 7]

    # Several test partitions of varying structures.
    test_reps = [
        ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)),  # all singletons
        ((0, 1), (2, 3), (4, 5), (6, 7)),  # within-cell pairs
        ((0, 4), (1, 5), (2, 6), (3, 7)),  # cross-cell pairs
        ((0, 1, 2, 3), (4, 5, 6, 7)),  # whole cells
        ((0, 4, 1, 5), (2, 6, 3, 7)),  # mixed
    ]

    G_flat, n_G = _build_per_cell_aut_flat(cell_anchor_groups, universe)
    lib, _ffi = _get_lib()
    G_arr = _ffi.new("int[]", G_flat)

    for rep in test_reps:
        canonical = per_cell_canonical_key(rep, cell_anchor_groups)
        target_size = per_cell_orbit_size(canonical, cell_anchor_groups)
        py_members = sorted(_expand_per_cell_orbit_members(rep, cell_anchor_groups))
        c_members = _expand_per_cell_orbit_members_c_batch(
            rep, target_size, universe, G_arr, n_G, lib, _ffi,
        )
        assert c_members is not None, f"C-ext returned None for rep={rep}"
        c_members_sorted = sorted(c_members)
        assert c_members_sorted == py_members, (
            f"rep={rep}: c={c_members_sorted}, py={py_members}"
        )


def test_expand_orbit_members_c_three_cells():
    """C-ext orbit expansion works for 3 cells (Cm_3 row scale)."""
    from tutte.roots.cell_quotient_helpers import (
        _build_per_cell_aut_flat,
        _expand_per_cell_orbit_members,
        _expand_per_cell_orbit_members_c_batch,
        per_cell_canonical_key,
        per_cell_orbit_size,
    )
    from tutte.roots._partition_c import _get_lib

    # Cm_3 row scale: 3 cells of 4 each (|G| = 24^3 = 13824).
    cell_anchor_groups = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    universe = list(range(12))

    test_reps = [
        ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,)),
        ((0, 4), (1, 5), (2, 6), (3, 7), (8,), (9,), (10,), (11,)),
        ((0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11)),
    ]

    G_flat, n_G = _build_per_cell_aut_flat(cell_anchor_groups, universe)
    assert n_G == 24 ** 3
    lib, _ffi = _get_lib()
    G_arr = _ffi.new("int[]", G_flat)

    for rep in test_reps:
        canonical = per_cell_canonical_key(rep, cell_anchor_groups)
        target_size = per_cell_orbit_size(canonical, cell_anchor_groups)
        py_members = sorted(_expand_per_cell_orbit_members(rep, cell_anchor_groups))
        c_members = _expand_per_cell_orbit_members_c_batch(
            rep, target_size, universe, G_arr, n_G, lib, _ffi,
        )
        assert c_members is not None
        c_members_sorted = sorted(c_members)
        assert c_members_sorted == py_members, (
            f"rep={rep}: |c|={len(c_members_sorted)} |py|={len(py_members)}"
        )


def test_set_stabilizer_c_matches_python():
    """C-ext set_stabilizer_c matches Python `per_cell_partition_stab`."""
    from tutte.roots._partition_c import _encode_partition, _get_lib
    from tutte.roots.aut_orbit import (
        enumerate_per_cell_aut_group,
        per_cell_partition_stab,
    )

    cell_anchor_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    universe = [0, 1, 2, 3, 4, 5, 6, 7]
    G_elements = enumerate_per_cell_aut_group(cell_anchor_groups)
    assert len(G_elements) == 24 * 24  # 576

    # Build G_flat for C call.
    pos_to_idx = {pos: i for i, pos in enumerate(universe)}
    G_flat = []
    for g_perm in G_elements:
        for pos in universe:
            tgt = g_perm.get(pos, pos)
            G_flat.append(pos_to_idx[tgt])

    lib, _ffi = _get_lib()
    G_arr = _ffi.new("int[]", G_flat)
    out_arr = _ffi.new("int[]", len(G_elements))

    test_partitions = [
        ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)),  # singletons → |H| = full
        ((0, 1), (2, 3), (4, 5), (6, 7)),  # 4 pairs
        ((0, 4), (1, 5), (2, 6), (3, 7)),  # cross pairs
        ((0, 1, 2, 3), (4, 5, 6, 7)),  # whole-cell blocks → |H| = full
    ]
    for P in test_partitions:
        py_stab = per_cell_partition_stab(P, G_elements)
        P_enc = _encode_partition(P, pos_to_idx)
        P_arr = _ffi.new("int[]", P_enc)
        n_stab = lib.set_stabilizer_c(
            P_arr, len(P_enc),
            G_arr, len(G_elements), len(universe),
            out_arr, len(G_elements),
        )
        assert n_stab >= 0, f"C error for P={P}"
        assert n_stab == len(py_stab), (
            f"P={P}: C n_stab={n_stab}, py |H|={len(py_stab)}"
        )
