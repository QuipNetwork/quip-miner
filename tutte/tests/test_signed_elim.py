"""Signed-graph elimination-order DP regressions.

Two layers:

1. **High-level DP** (`compute_signed_tutte_elim_mod`): validates against
   brute-force `compute_t_signed` on a range of small signed graphs.
   Covers balanced/unbalanced cycles, multi-edges with opposite signs,
   positive/negative loops, handcuff structures, and historically-buggy
   fuzz cases.
2. **C extension internals** (`encode_state`, `decode_state`,
   `step_edge_batch`, `step_forget_batch`): per-step the C extension
   should be bit-for-bit identical to the pure-Python reference.
"""
from __future__ import annotations

import pytest
import sympy

from tutte.deprecated._signed_elim_c import (
    decode_state, encode_state, step_edge_batch, step_forget_batch,
)
from tutte.deprecated.signed_elim_dp import compute_signed_tutte_elim_mod
from tutte.tests.signed_graph_tutte_prototype import compute_t_signed


# ---------------------------------------------------------------------------
# High-level DP vs brute force
# ---------------------------------------------------------------------------

def _check(nodes, edges, points=((2, 3, 1009), (5, 7, 2017), (-1, 4, 4019))):
    """Compute brute-force T_signed and verify DP agrees at each point."""
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    t_poly = compute_t_signed(nodes, edges, x_sym, y_sym)
    for x_v, y_v, p in points:
        expected = int(t_poly.subs({x_sym: x_v, y_sym: y_v})) % p
        actual, _ = compute_signed_tutte_elim_mod(nodes, edges, x_v, y_v, p)
        assert actual == expected, (
            f"Mismatch at ({x_v},{y_v},{p}): expected {expected}, got {actual}\n"
            f"  T = {t_poly}\n  nodes={nodes}, edges={edges}"
        )


def test_balanced_triangle():
    _check([0, 1, 2], [((0, 1), 0), ((1, 2), 0), ((0, 2), 0)])


def test_unbalanced_triangle():
    _check([0, 1, 2], [((0, 1), 0), ((1, 2), 0), ((0, 2), 1)])


def test_k4_one_sign():
    _check([0, 1, 2, 3], [((0, 1), 0), ((0, 2), 0), ((0, 3), 0),
                          ((1, 2), 0), ((1, 3), 0), ((2, 3), 1)])


def test_multi_edge_unbalanced_cycle():
    """Two parallel edges with opposite signs form an unbalanced cycle."""
    _check([0, 1], [((0, 1), 0), ((0, 1), 1)])


def test_negative_loops_with_bridge():
    """Two unbalanced loops joined by a tree edge — handcuff structure."""
    _check([0, 1], [((0, 0), 1), ((1, 1), 1), ((0, 1), 0)])


def test_two_unbalanced_triangles_bridge():
    """Two unbalanced K_3 joined by a bridge — matroid-equivalent to C_7."""
    _check([0, 1, 2, 3, 4, 5],
           [((0, 1), 0), ((1, 2), 0), ((0, 2), 1),
            ((3, 4), 0), ((4, 5), 0), ((3, 5), 1),
            ((0, 3), 0)])


def test_loop_plus_triangle_plus_bridge():
    """Unbalanced loop on isolated vertex bridged to unbalanced triangle.
    Forces 'merge two unbalanced' trajectory in some elim orderings."""
    _check([0, 1, 2, 3],
           [((0, 0), 1),
            ((1, 2), 0), ((1, 3), 1), ((2, 3), 0),
            ((0, 3), 0)])


def test_4_cycle_unbalanced():
    """4-cycle with one negative edge — matroid-equivalent to C_4."""
    _check([0, 1, 2, 3],
           [((0, 2), 0), ((1, 2), 1), ((1, 3), 1), ((0, 3), 1)])


def test_random_fuzz_5v_8e():
    """Specific previously-failing fuzz case (seed 5)."""
    _check([0, 1, 2, 3, 4],
           [((2, 4), 0), ((0, 1), 0), ((1, 3), 1), ((1, 4), 0),
            ((2, 3), 0), ((1, 3), 0), ((3, 4), 0), ((0, 1), 0)])


# ---------------------------------------------------------------------------
# C extension internals
# ---------------------------------------------------------------------------

def test_encode_decode_roundtrip():
    """Encoded state decodes back to the same tuple."""
    cases = [
        # (partition, monodromy, balance, finalized)
        ((0, 1, 0), (0, 0, 0), (True, True), 0),
        ((0, 0, 1, 1), (0, 1, 0, 1), (True, False), 3),
        ((0,), (0,), (True,), 5),
    ]
    for st in cases:
        flat = encode_state(*st)
        assert decode_state(flat) == st


def test_c_ext_compiles():
    """C extension compiles and is callable."""
    from tutte.deprecated._signed_elim_c import _get_lib
    lib, ffi = _get_lib()
    assert lib is not None
    assert ffi is not None


def test_step_edge_delete_only():
    """Edge within same balanced block, sign 0:
       delete branch keeps state; keep branch ×(y-1)."""
    initial = {((0, 0), (0, 0), (True,), 0): 1}
    new_states = step_edge_batch(
        initial, u_pos=0, v_pos=1, sign=0, is_loop=False,
        y_minus_1=2, p=1009,
    )
    assert new_states is not None
    expected_state = ((0, 0), (0, 0), (True,), 0)
    assert new_states[expected_state] == 3  # 1 + 1*(y-1)


def test_step_edge_tree_merge():
    """Tree edge between two singleton blocks merges them."""
    initial = {((0, 1), (0, 0), (True, True), 0): 1}
    new_states = step_edge_batch(
        initial, u_pos=0, v_pos=1, sign=0, is_loop=False,
        y_minus_1=2, p=1009,
    )
    assert new_states is not None
    delete_state = ((0, 1), (0, 0), (True, True), 0)
    keep_state = ((0, 0), (0, 0), (True,), 0)
    assert new_states[delete_state] == 1
    assert new_states[keep_state] == 1


def test_step_edge_unbalanced_cycle():
    """Unbalanced cycle: block becomes unbalanced; weight unchanged."""
    initial = {((0, 0), (0, 1), (True,), 0): 1}
    new_states = step_edge_batch(
        initial, u_pos=0, v_pos=1, sign=0, is_loop=False,
        y_minus_1=2, p=1009,
    )
    assert new_states is not None
    delete_state = ((0, 0), (0, 1), (True,), 0)
    keep_state = ((0, 0), (0, 0), (False,), 0)
    assert new_states[delete_state] == 1
    assert new_states[keep_state] == 1


def test_step_forget_singleton_balanced():
    """Forgetting a singleton balanced block increments finalized."""
    initial = {((0, 1), (0, 0), (True, True), 0): 5}
    new_states = step_forget_batch(initial, fpos=0, p=1009)
    assert new_states is not None
    expected_state = ((0,), (0,), (True,), 1)
    assert new_states[expected_state] == 5


def test_step_forget_non_singleton():
    """Forgetting non-singleton: drop position, no finalize change."""
    initial = {((0, 0), (0, 0), (True,), 0): 7}
    new_states = step_forget_batch(initial, fpos=0, p=1009)
    assert new_states is not None
    expected_state = ((0,), (0,), (True,), 0)
    assert new_states[expected_state] == 7
