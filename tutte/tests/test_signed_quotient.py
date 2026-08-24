"""Tests for tutte/roots/signed_quotient.py — high-level signed quotient API."""
from __future__ import annotations

import pytest
import sympy
import networkx as nx

from tutte.deprecated.signed_quotient_pipeline import (
    build_quotient_with_monodromy,
    compute_t_fix_sigma_quotient_mod,
    compute_t_signed_quotient_mod,
    derive_t_free_sigma_mod,
    evaluate_t_signed_mod,
    interpolate_t_signed_mod,
)
from tutte.tests.signed_graph_tutte_prototype import compute_t_signed
from tutte.tests.verify_t_fix_sigma import compute_t_fix_sigma_brute


def _validate(g, perm, x_v, y_v, p):
    """Build quotient, compute T_signed via DP, check against brute-force."""
    nodes, edges = build_quotient_with_monodromy(g, perm)
    actual = evaluate_t_signed_mod(nodes, edges, x_v, y_v, p)
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    t_poly = compute_t_signed(nodes, edges, x_sym, y_sym)
    expected = int(t_poly.subs({x_sym: x_v, y_sym: y_v})) % p
    assert actual == expected, (
        f"DP {actual} != brute {expected} on quotient with "
        f"{len(nodes)} verts and {len(edges)} edges"
    )


def test_quotient_c4_rotation():
    """C_4 with rotation σ = (01)(23) has free 2-fold cover structure."""
    g = nx.cycle_graph(4)
    perm = {0: 2, 1: 3, 2: 0, 3: 1}
    _validate(g, perm, 2, 3, 1009)
    _validate(g, perm, 5, 7, 2017)


def test_quotient_k4_swap():
    """K_4 with σ = (01)(23) — paired vertices."""
    g = nx.complete_graph(4)
    perm = {0: 1, 1: 0, 2: 3, 3: 2}
    _validate(g, perm, 2, 3, 1009)


def test_quotient_cube_antipodal():
    """3-cube with antipodal map — every vertex maps to opposite corner."""
    g = nx.hypercube_graph(3)
    # Relabel to integers
    nm = {n: i for i, n in enumerate(sorted(g.nodes()))}
    g = nx.relabel_nodes(g, nm)
    # Antipodal: vertex i ↔ vertex 7-i (in binary, flip all bits)
    perm = {i: 7 - i for i in range(8)}
    _validate(g, perm, 2, 3, 1009)


def test_interpolate_small():
    """Interpolation reproduces the brute-force polynomial bit-for-bit."""
    g = nx.cycle_graph(4)
    perm = {0: 2, 1: 3, 2: 0, 3: 1}
    nodes, edges = build_quotient_with_monodromy(g, perm)

    p = 1009
    # Degree of T_signed: bounded by |V| - 1 in x, |E| - rank in y.
    # For this tiny quotient (2 verts, 2 edges), use small grid.
    x_values = list(range(8))
    y_values = list(range(8))
    coefs = interpolate_t_signed_mod(nodes, edges, x_values, y_values, p)

    # Validate at a fresh point.
    x_v, y_v = 17, 19
    interp_val = 0
    for (a, b), c in coefs.items():
        interp_val = (interp_val + c * pow(x_v, a, p) * pow(y_v, b, p)) % p
    direct_val = evaluate_t_signed_mod(nodes, edges, x_v, y_v, p)
    assert interp_val == direct_val


@pytest.mark.slow
def test_z12_quotient_single_point():
    """End-to-end: compute T_signed for Z(1,2) cell-swap quotient at one point.

    Validates the production API on the actual D-Wave Z(1,2) target.
    Expected mod-1009 value at (2, 3): 430.
    """
    import dwave_networkx as dnx

    g = dnx.zephyr_graph(1, 2)
    # Cell-swap permutation on dnx labeling (matches signed_dp_modular.py).
    perm = {}
    for i in range(24):
        perm[i] = i + 2 if (i // 2) % 2 == 0 else i - 2
    value = compute_t_signed_quotient_mod(g, perm, 2, 3, 1009)
    assert value == 430


def test_t_fix_sigma_c4_free():
    """T_fix^σ DP on C_4 with σ=(02)(13) — free cover, validate vs brute."""
    import sympy
    g = nx.cycle_graph(4)
    perm = {0: 2, 1: 3, 2: 0, 3: 1}
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    expected_poly = compute_t_fix_sigma_brute(g, perm, x_sym, y_sym)
    for x_v, y_v, p in [(2, 3, 1009), (5, 7, 2017), (-1, 4, 4019)]:
        expected = int(expected_poly.subs({x_sym: x_v, y_sym: y_v})) % p
        actual = compute_t_fix_sigma_quotient_mod(g, perm, x_v, y_v, p)
        assert actual == expected, f"({x_v},{y_v},mod {p}): {actual} != {expected}"


def test_t_fix_sigma_cube_antipodal():
    """T_fix^σ on 3-cube with antipodal map — free cover."""
    import sympy
    g = nx.hypercube_graph(3)
    nm = {n: i for i, n in enumerate(sorted(g.nodes()))}
    g = nx.relabel_nodes(g, nm)
    perm = {i: 7 - i for i in range(8)}
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    expected_poly = compute_t_fix_sigma_brute(g, perm, x_sym, y_sym)
    expected = int(expected_poly.subs({x_sym: 2, y_sym: 3})) % 1009
    actual = compute_t_fix_sigma_quotient_mod(g, perm, 2, 3, 1009)
    assert actual == expected


def test_t_fix_sigma_c6_rotation():
    """T_fix^σ on C_6 with rotation-by-3 σ — free cover."""
    import sympy
    g = nx.cycle_graph(6)
    perm = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    expected_poly = compute_t_fix_sigma_brute(g, perm, x_sym, y_sym)
    for x_v, y_v, p in [(2, 3, 1009), (5, 7, 2017)]:
        expected = int(expected_poly.subs({x_sym: x_v, y_sym: y_v})) % p
        actual = compute_t_fix_sigma_quotient_mod(g, perm, x_v, y_v, p)
        assert actual == expected, f"({x_v},{y_v},mod {p}): got {actual} != {expected}"


def test_t_fix_sigma_k33_non_free():
    """T_fix^σ DP now handles K_{3,3} part-swap (has σ-fixed edges → loops in quotient)."""
    import sympy
    g = nx.complete_bipartite_graph(3, 3)
    nm = {n: i for i, n in enumerate(sorted(g.nodes()))}
    g = nx.relabel_nodes(g, nm)
    perm = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    expected_poly = compute_t_fix_sigma_brute(g, perm, x_sym, y_sym)
    for x_v, y_v, p in [(2, 3, 1009), (5, 7, 2017), (-1, 4, 4019)]:
        expected = int(expected_poly.subs({x_sym: x_v, y_sym: y_v})) % p
        actual = compute_t_fix_sigma_quotient_mod(g, perm, x_v, y_v, p)
        assert actual == expected, f"({x_v},{y_v},mod {p}): {actual} != {expected}"


def test_t_fix_sigma_k4_non_free():
    """T_fix^σ DP on K_4 + (01)(23) — non-free cover with 2 σ-fixed edges."""
    import sympy
    g = nx.complete_graph(4)
    perm = {0: 1, 1: 0, 2: 3, 3: 2}
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    expected_poly = compute_t_fix_sigma_brute(g, perm, x_sym, y_sym)
    for x_v, y_v, p in [(2, 3, 1009), (5, 7, 2017)]:
        expected = int(expected_poly.subs({x_sym: x_v, y_sym: y_v})) % p
        actual = compute_t_fix_sigma_quotient_mod(g, perm, x_v, y_v, p)
        assert actual == expected


def test_derive_t_free_sigma_cube():
    """derive_t_free_sigma_mod on Cube antipodal — quick test of the API."""
    import sympy
    g = nx.hypercube_graph(3)
    nm = {n: i for i, n in enumerate(sorted(g.nodes()))}
    g = nx.relabel_nodes(g, nm)
    perm = {i: 7 - i for i in range(8)}
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    poly = nx.tutte_polynomial(g)
    fix_poly = compute_t_fix_sigma_brute(g, perm, x_sym, y_sym)
    for x_v, y_v, p in [(2, 3, 1009), (5, 7, 2017)]:
        expected_g = int(poly.subs({x_sym: x_v, y_sym: y_v})) % p
        expected_fix = int(fix_poly.subs({x_sym: x_v, y_sym: y_v})) % p
        expected_free = (expected_g - expected_fix) % p
        actual_free = derive_t_free_sigma_mod(g, perm, x_v, y_v, p)
        assert actual_free == expected_free


def test_derive_t_free_sigma_via_cover_cube():
    """T_free^σ from-scratch (no engine) via sigma_orbit_dp_full + signed-DP.

    Validates that derive_t_free_sigma_mod_via_cover (no engine dependency)
    matches derive_t_free_sigma_mod (engine-derived) bit-for-bit.

    Cube antipodal σ is the smallest free 2-fold cover for testing.
    """
    from tutte.deprecated.signed_quotient_pipeline import derive_t_free_sigma_mod_via_cover

    g = nx.hypercube_graph(3)
    nm = {n: i for i, n in enumerate(sorted(g.nodes()))}
    g = nx.relabel_nodes(g, nm)
    perm = {i: 7 - i for i in range(8)}
    for x_v, y_v, p in [(2, 3, 1009), (5, 7, 2017), (-1, 4, 4019)]:
        scratch = derive_t_free_sigma_mod_via_cover(g, perm, x_v, y_v, p)
        engine_derived = derive_t_free_sigma_mod(g, perm, x_v, y_v, p)
        assert scratch == engine_derived, (
            f"({x_v},{y_v},{p}): from-scratch {scratch} != engine-derived {engine_derived}"
        )


def test_find_best_sigma_zephyr_chimera():
    """find_best_sigma must return valid free order-2 σ across D-Wave families.

    Regression for cross-family generalization: ensures cell-swap candidates
    in the candidate list discover free σ for Cm_2 and Cm_3 that earlier
    i+n/2-only surveys missed.
    """
    import dwave_networkx as dnx
    from tutte.roots.signed_quotient import find_best_sigma

    # Z(1,2): should find FREE σ (cell-swap ±2)
    g_z12 = dnx.zephyr_graph(1, 2)
    perm = find_best_sigma(g_z12, require_free=True)
    assert perm is not None, "Z(1,2) should have a free order-2 σ"
    # Verify properties.
    n_v = g_z12.number_of_nodes()
    assert all(perm[v] != v for v in range(n_v)), "σ must be free on vertices"
    assert all(perm[perm[v]] == v for v in range(n_v)), "σ must be order 2"
    edges_set = set(tuple(sorted((int(u), int(v)))) for u, v in g_z12.edges())
    assert all(tuple(sorted((perm[u], perm[v]))) in edges_set for u, v in g_z12.edges())
    fixed = sum(1 for u, v in g_z12.edges() if sorted([perm[u], perm[v]]) == sorted([u, v]))
    assert fixed == 0, f"Z(1,2) require_free=True should return σ with 0 fixed edges, got {fixed}"

    # Cm_2: also should find FREE σ (cell-swap variants)
    g_cm2 = dnx.chimera_graph(2)
    perm_cm2 = find_best_sigma(g_cm2, require_free=True)
    assert perm_cm2 is not None, "Cm_2 should have a free order-2 σ (regression for May 2026 update)"

    # If require_free=False, should always return SOMETHING for these graphs.
    for name, g in [("Z(1,1)", dnx.zephyr_graph(1, 1)), ("Z(1,2)", g_z12),
                    ("Cm_1", dnx.chimera_graph(1)), ("Cm_2", g_cm2)]:
        p = find_best_sigma(g, require_free=False)
        assert p is not None, f"{name} should have at least some order-2 σ"


def test_decompose_t_polynomial_via_sigma_cube_full():
    """Full-bidegree polynomial decomposition regression on Cube.

    Verifies decompose_t_polynomial_via_sigma returns (T_fix, T_free, T_total)
    that satisfy T_fix + T_free = T_total at every monomial.

    Cube bidegree ≤ (7, 5). Use 8×6 grid for full recovery.
    """
    from tutte.deprecated.signed_quotient_pipeline import decompose_t_polynomial_via_sigma

    g = nx.hypercube_graph(3)
    nm = {n: i for i, n in enumerate(sorted(g.nodes()))}
    g = nx.relabel_nodes(g, nm)
    perm = {i: 7 - i for i in range(8)}
    x_values = list(range(2, 10))  # 8 values for deg_x ≤ 7
    y_values = list(range(2, 8))   # 6 values for deg_y ≤ 5
    p = 1009

    t_fix, t_free, t_total = decompose_t_polynomial_via_sigma(
        g, perm, x_values, y_values, p, n_workers=1
    )
    # T_fix + T_free == T_total at every monomial.
    all_keys = set(t_fix.keys()) | set(t_free.keys()) | set(t_total.keys())
    for k in all_keys:
        s = (t_fix.get(k, 0) + t_free.get(k, 0)) % p
        t = t_total.get(k, 0) % p
        assert s == t, f"monomial {k}: T_fix+T_free={s} != T_total={t}"

    # Spot-check via direct evaluation against nx.tutte_polynomial.
    import sympy
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    poly = nx.tutte_polynomial(g)
    for x_v, y_v in [(2, 3), (5, 7), (3, 5)]:
        expected = int(poly.subs({x_sym: x_v, y_sym: y_v})) % p
        actual = sum(c * pow(x_v, a, p) * pow(y_v, b, p) for (a, b), c in t_total.items()) % p
        assert actual == expected, f"({x_v},{y_v}): {actual} != {expected}"


def test_compute_t_via_sigma_auto_dnx_lookup_hit(engine):
    """Regression test for the dnx canonical_key bug (May 16, 2026).

    `compute_t_via_sigma_auto` must use `Graph.from_networkx` internally
    so dnx-style tuple/coordinate vertex labels relabel correctly and
    hit the lookup table. Previously a manual frozenset construction
    bypassed lookup → 35s instead of 0.1s for Cm_2.

    Acceptance: Cm_2 single-point evaluates in < 1s with a WARM engine.
    The engine comes from the session fixture so the timed region measures
    the lookup, not one-time table decode (which alone exceeds 5s on the
    current 18.7K-entry table and used to mask this assertion).
    """
    import time
    import dwave_networkx as dnx
    from tutte.deprecated.signed_quotient_pipeline import compute_t_via_sigma_auto

    g = dnx.chimera_graph(2)
    t0 = time.time()
    val = compute_t_via_sigma_auto(g, 2, 3, 1009, engine=engine)
    elapsed = time.time() - t0
    assert val == 806, f"T(Cm_2; 2, 3) mod 1009 expected 806, got {val}"
    assert elapsed < 1.0, (
        f"compute_t_via_sigma_auto(Cm_2) took {elapsed:.3f}s — should be "
        f"sub-second via lookup hit. Likely the dnx canonical_key bug "
        f"regressed (σ-DP fallback takes minutes)."
    )


def test_convenience_wrapper():
    g = nx.cycle_graph(4)
    perm = {0: 2, 1: 3, 2: 0, 3: 1}
    expected = compute_t_signed_quotient_mod(g, perm, 2, 3, 1009)
    # Validate via brute force
    nodes, edges = build_quotient_with_monodromy(g, perm)
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    t_poly = compute_t_signed(nodes, edges, x_sym, y_sym)
    direct = int(t_poly.subs({x_sym: 2, y_sym: 3})) % 1009
    assert expected == direct


def test_compute_t_via_pipeline_z12_lookup(engine):
    """Z(1,2) via unified pipeline — must hit lookup path under 1s.

    Regression-locks the Z(1,2) wiring through the cross-framework
    pipeline. Path 0 (lookup) is the fastest; Z(1,2)'s canonical_key is
    pre-seeded in the rainbow table so this should be sub-second.
    """
    import time
    import dwave_networkx as dnx
    from tutte.deprecated.signed_quotient_pipeline import compute_t_via_pipeline

    g = dnx.zephyr_graph(1, 2)
    t0 = time.time()
    val, framework = compute_t_via_pipeline(g, 2, 3, 1009, engine=engine)
    elapsed = time.time() - t0
    assert val == 629, f"Z(1,2) T(2,3) mod 1009 = {val} != 629"
    assert framework == "lookup", (
        f"Z(1,2) routed via {framework}, expected 'lookup'"
    )
    # Engine init (merger table load + rainbow decode, >5s on the current
    # table) is hoisted into the session `engine` fixture, so this budget
    # measures the lookup itself — sub-millisecond when Path 0 hits.
    assert elapsed < 1.0, f"Z(1,2) pipeline took {elapsed:.3f}s (must be < 1s)"


def test_compute_t_via_pipeline_k22_m2_chain_routes_via_cell_quotient_tree():
    """3-cell K_{2,2}+M_2 chain must route via cell_quotient_tree path.

    Exercises path 2 of the pipeline end-to-end. The cell_quotient_tree
    detector requires rainbow-table entries with `graph` attached so
    `try_hierarchical_partition` can compute cell signatures.
    `_seed_common_cell_templates` augments K_{2,2} with its graph,
    enabling detection. Validates against `nx.tutte_polynomial`.
    """
    from tutte.deprecated.signed_quotient_pipeline import compute_t_via_pipeline
    from tutte.synthesis.engine import SynthesisEngine

    def build_k22_m2_chain(n_cells):
        g = nx.Graph()
        for i in range(n_cells):
            base = 4 * i
            for u in (base, base + 1):
                for v in (base + 2, base + 3):
                    g.add_edge(u, v)
        for i in range(n_cells - 1):
            base = 4 * i
            nxt = 4 * (i + 1)
            g.add_edge(base + 2, nxt)
            g.add_edge(base + 3, nxt + 1)
        return g

    g3 = build_k22_m2_chain(3)
    engine = SynthesisEngine()
    val, framework = compute_t_via_pipeline(g3, 2, 3, 1009, engine=engine)
    # nx oracle
    tp = nx.tutte_polynomial(g3)
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    expected = int(tp.subs({x_sym: 2, y_sym: 3})) % 1009
    assert val == expected, f"3-cell K22+M2 chain: pipeline={val}, sympy={expected}"
    assert framework == "cell_quotient_tree", (
        f"3-cell K22+M2 chain routed via {framework}, expected "
        f"'cell_quotient_tree' (path 1 wiring regression — chain_recurrence "
        f"is the fastest fallback for chain-decomposable graphs)"
    )


def test_compute_t_signed_via_pipeline_z12_under_60s():
    """Z(1,2) signed-Tutte via pipeline — exposes signed-treewidth DP path.

    Locks in the second pipeline entrypoint: `compute_t_signed_via_pipeline`
    computes Zaslavsky's signed Tutte T_signed(G/⟨σ⟩) on the quotient.
    For Z(1,2) at (2, 3, 1009) the value is 430 (per memory
    `project_signed_elim_dp_bugfix.md`). Must complete in <60s — the
    headline target.
    """
    import time
    import dwave_networkx as dnx
    from tutte.deprecated.signed_quotient_pipeline import compute_t_signed_via_pipeline

    g = dnx.zephyr_graph(1, 2)
    t0 = time.time()
    val, sigma_class = compute_t_signed_via_pipeline(g, 2, 3, 1009)
    elapsed = time.time() - t0
    assert val == 430, f"T_signed(Z(1,2); 2, 3) mod 1009 = {val} != 430"
    assert sigma_class == "free", (
        f"Z(1,2) should use free σ; got {sigma_class}"
    )
    assert elapsed < 60.0, (
        f"T_signed pipeline on Z(1,2) took {elapsed:.1f}s (must be < 60s)"
    )


def test_z12_three_scalar_pipeline_entry_points_cross_validate():
    """Z(1,2) cross-validation: 3 scalar entry points + poly T(G) agree.

    Smoke test exercising the wiring surface: T(G) scalar/poly +
    T_signed scalar, all on Z(1, 2). Cross-validates scalar T(G) =
    poly T(G) at point. The polynomial T_signed entry point is excluded
    (too slow to interpolate in a unit-test budget — covered separately
    by `test_compute_t_signed_polynomial_via_pipeline_cm1`).

    All 3 paths must complete in <60s (the headline target).
    """
    import time
    import dwave_networkx as dnx
    from tutte.deprecated.signed_quotient_pipeline import (
        compute_t_polynomial_via_pipeline,
        compute_t_signed_via_pipeline,
        compute_t_via_pipeline,
    )

    g = dnx.zephyr_graph(1, 2)
    x_v, y_v, p = 2, 3, 1009

    # Entry 1: scalar T(G)
    t0 = time.time()
    t_g_scalar, _ = compute_t_via_pipeline(g, x_v, y_v, p)
    assert time.time() - t0 < 60.0
    assert t_g_scalar == 629

    # Entry 2: poly T(G)
    t0 = time.time()
    t_g_poly, _ = compute_t_polynomial_via_pipeline(g)
    assert time.time() - t0 < 60.0
    assert t_g_poly is not None

    # Entry 3: scalar T_signed
    t0 = time.time()
    t_s_scalar, sigma_class = compute_t_signed_via_pipeline(g, x_v, y_v, p)
    assert time.time() - t0 < 60.0
    assert sigma_class == "free"
    assert t_s_scalar == 430

    # Cross-validation: scalar T(G) == poly T(G) at (x_v, y_v) mod p
    poly_val = t_g_poly.evaluate(x_v, y_v) % p
    assert poly_val == t_g_scalar, (
        f"T(G) scalar={t_g_scalar} vs poly@({x_v},{y_v})={poly_val}"
    )


def test_compute_t_signed_polynomial_adaptive_via_pipeline_cm1():
    """Adaptive T_signed polynomial — grows grid until stable.

    Validates `compute_t_signed_polynomial_adaptive_via_pipeline` on Cm_1.
    The recovered polynomial must evaluate to 479 at (2, 3) mod 1009
    (matches the scalar pipeline) and have exactly 9 nonzero terms
    (matches the explicit 10×10 grid recovery).
    """
    import dwave_networkx as dnx
    from tutte.deprecated.signed_quotient_pipeline import \
        compute_t_signed_polynomial_adaptive_via_pipeline

    g = dnx.chimera_graph(1)
    poly, sigma_class, n_evals = compute_t_signed_polynomial_adaptive_via_pipeline(
        g, 1009, max_deg_x=20, max_deg_y=20
    )
    assert poly is not None
    assert sigma_class == "free"
    assert len(poly) == 9, f"Cm_1 T_signed has {len(poly)} terms; expected 9"
    val = sum(c * pow(2, a, 1009) * pow(3, b, 1009) for (a, b), c in poly.items()) % 1009
    assert val == 479, f"adaptive T_signed(Cm_1; 2, 3) = {val} != 479"


def test_compute_t_decomposition_via_pipeline_cm1():
    """Full σ-equivariant decomposition pipeline on Cm_1.

    `compute_t_decomposition_via_pipeline` returns
    `(T_fix^σ, T_free^σ, T, sigma_class)` polynomial dicts. Validates
    the lift identity T_fix + T_free = T at point (2, 3) mod 1009.
    """
    import dwave_networkx as dnx
    from tutte.deprecated.signed_quotient_pipeline import compute_t_decomposition_via_pipeline

    g = dnx.chimera_graph(1)
    xs = list(range(2, 11))
    ys = list(range(2, 11))
    t_fix, t_free, t_tot, sigma_class = compute_t_decomposition_via_pipeline(
        g, xs, ys, 1009
    )
    assert sigma_class == "free", (
        f"Cm_1 decomposition: sigma_class={sigma_class}, expected free"
    )
    assert t_fix is not None and t_free is not None and t_tot is not None

    def eval_poly(poly, x, y, p):
        return sum(c * pow(x, a, p) * pow(y, b, p) for (a, b), c in poly.items()) % p

    val_fix = eval_poly(t_fix, 2, 3, 1009)
    val_free = eval_poly(t_free, 2, 3, 1009)
    val_tot = eval_poly(t_tot, 2, 3, 1009)
    assert val_tot == 600, (
        f"Cm_1 T(2,3) mod 1009 = {val_tot}, expected 600"
    )
    assert (val_fix + val_free) % 1009 == val_tot, (
        f"Lift identity broken: T_fix={val_fix} + T_free={val_free} = "
        f"{(val_fix + val_free) % 1009} != T={val_tot}"
    )


def test_compute_t_polynomial_via_pipeline_z12_lookup():
    """T(G) polynomial pipeline for Z(1,2) — full polynomial via lookup.

    Validates that `compute_t_polynomial_via_pipeline` returns the full
    Tutte polynomial (not just scalar) for Z(1,2). Hits lookup path
    instantly; polynomial has 573 nonzero terms; T(1,1) = #spanning trees
    = 25,117,827,740,467,200; T(2,3) mod 1009 = 629.
    """
    import dwave_networkx as dnx
    from tutte.deprecated.signed_quotient_pipeline import compute_t_polynomial_via_pipeline

    g = dnx.zephyr_graph(1, 2)
    poly, framework = compute_t_polynomial_via_pipeline(g)
    assert framework == "lookup", f"Z(1,2) routed via {framework}, expected lookup"
    assert poly is not None
    assert len(poly._coeffs) == 573, (
        f"Z(1,2) polynomial has {len(poly._coeffs)} terms; expected 573"
    )
    assert poly.evaluate(1, 1) == 25117827740467200
    assert poly.evaluate(2, 3) % 1009 == 629


def test_compute_t_signed_polynomial_via_pipeline_cm1():
    """T_signed polynomial pipeline returns dict — multi-point reconstruction.

    Validates that the polynomial-form pipeline `compute_t_signed_polynomial_via_pipeline`
    interpolates the full T_signed polynomial via 2D Lagrange on the
    user-supplied grid, and that evaluating the dict at (2, 3, 1009)
    matches the single-point value (479) from the scalar pipeline.
    """
    import dwave_networkx as dnx
    from tutte.deprecated.signed_quotient_pipeline import compute_t_signed_polynomial_via_pipeline

    g = dnx.chimera_graph(1)
    xs = list(range(2, 12))
    ys = list(range(2, 12))
    poly, sigma_class = compute_t_signed_polynomial_via_pipeline(
        g, xs, ys, 1009
    )
    assert poly is not None, "Polynomial pipeline returned None on Cm_1"
    assert sigma_class == "free", (
        f"Cm_1 should use free σ; got {sigma_class}"
    )
    # Evaluate polynomial at (2, 3, 1009)
    val_at_23 = sum(
        c * pow(2, a, 1009) * pow(3, b, 1009) for (a, b), c in poly.items()
    ) % 1009
    assert val_at_23 == 479, (
        f"Polynomial T_signed(Cm_1; 2, 3) mod 1009 = {val_at_23} != 479"
    )


def test_compute_t_signed_via_pipeline_cm1_chimera():
    """Cm_1 (K_{4,4}) signed-Tutte via pipeline — cross-family regression.

    Validates that the T_signed pipeline works on Chimera Cm_1 (= K_{4,4}),
    not just Zephyr Z(1, t). Confirms cross-family applicability of the
    σ-equivariant signed-Tutte framework. Cm_1 = 8v 16e — fast enough
    to lock value + free-σ + sub-3s timing.
    """
    import time
    import dwave_networkx as dnx
    from tutte.deprecated.signed_quotient_pipeline import compute_t_signed_via_pipeline

    g = dnx.chimera_graph(1)
    t0 = time.time()
    val, sigma_class = compute_t_signed_via_pipeline(g, 2, 3, 1009)
    elapsed = time.time() - t0
    assert val == 479, f"T_signed(Cm_1; 2, 3) mod 1009 = {val} != 479"
    assert sigma_class == "free", (
        f"Cm_1 should use free σ; got {sigma_class}"
    )
    assert elapsed < 5.0, (
        f"T_signed pipeline on Cm_1 took {elapsed:.1f}s (must be < 5s)"
    )


def test_compute_t_via_pipeline_cube_fallback():
    """Cube via pipeline with empty lookup — must hit σ-DP free path.

    The 3-cube has a free order-2 σ (antipodal map) but no chain-of-cells
    decomposition. With an empty engine table, path 0 (lookup) and path 1
    (cell_quotient_tree) both miss, so it should route through Path 2
    (sigma_orbit_dp_full on cover).
    """
    import time
    from tutte.synthesis.engine import SynthesisEngine
    from tutte.lookup import RainbowTable
    from tutte.deprecated.signed_quotient_pipeline import compute_t_via_pipeline

    g = nx.hypercube_graph(3)
    g = nx.convert_node_labels_to_integers(g)

    empty_engine = SynthesisEngine(table=RainbowTable())
    t0 = time.time()
    val, framework = compute_t_via_pipeline(g, 2, 3, 1009, engine=empty_engine)
    elapsed = time.time() - t0
    assert framework == "sigma_dp_free", (
        f"Cube routed via {framework}, expected 'sigma_dp_free' (path 2 — "
        f"cube has free antipodal σ but no chain decomposition)"
    )
    tp = nx.tutte_polynomial(g)
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    expected = int(tp.subs({x_sym: 2, y_sym: 3})) % 1009
    assert val == expected, (
        f"Pipeline gave {val} via {framework}, sympy gives {expected}"
    )
    assert elapsed < 10.0, f"Cube fallback took {elapsed:.3f}s"
