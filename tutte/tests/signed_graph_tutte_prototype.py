"""Phase 1 prototype: signed-graph Tutte polynomial via direct enumeration.

Given a signed multigraph (G_base = (V, E), χ : E → Z_2):
  - For each A ⊆ E, compute r_signed(A) = |V| - # BALANCED components
  - A component is balanced iff every cycle within it has trivial monodromy
    (sum of χ over cycle edges = 0 mod 2)
  - T_signed(G_base, χ; x, y) = sum_A (x-1)^{r_E - r_signed(A)} (y-1)^{|A| - r_signed(A)}

This is Zaslavsky's frame matroid Tutte polynomial.

Then verify: for 2-fold cover G → G_base with monodromy χ, the σ-fixed
Tutte sum T_fix^σ(G) equals a specific polynomial transformation of
T_signed(G_base, χ).

Specifically, per the lifting framework:
  T_fix^σ(G; x, y) = sum_L (x-1)^{r(E_G) - r_quot(L) - |L| + c(L)} (y-1)^{|A_L| - r_quot(L) - |L| + c(L)}
where c(L) = nullity_G(A_L) - nullity_quot(L) + n_loops(L)
           = (matroid quantity computable via signed-graph rank function)

We verify this by:
1. Taking small graphs (C_4, K_4, Cubical, K_3,3) with known order-2 σ
2. Building quotient G_base with monodromy χ
3. Computing T_fix^σ via DIRECT enumeration over σ-fixed subsets of G
4. Computing it via DP on quotient using r_signed
5. Confirming bit-for-bit polynomial match
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

import sympy
import networkx as nx


def find_(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def rank_subgraph(nodes, edges):
    """Standard graph rank: |V| - # components."""
    parent = {v: v for v in nodes}
    for u, v in edges:
        ru, rv = find_(parent, u), find_(parent, v)
        if ru != rv:
            parent[max(ru, rv)] = min(ru, rv)
    return len(nodes) - len({find_(parent, v) for v in nodes})


def signed_rank(nodes, edges_with_signs):
    """Signed graph rank: |V| - # balanced components.

    edges_with_signs: list of ((u, v), sign) tuples.
    A connected component is balanced iff every cycle in it has sign-sum 0 mod 2.

    Implementation:
    - Build spanning forest
    - For each non-tree edge, check if it creates balanced or unbalanced cycle
      (cycle sign = edge_sign + sum of tree-path signs from u to v)
    - Mark component as unbalanced if any unbalanced cycle exists
    """
    parent = {v: v for v in nodes}
    # Track relative monodromy of each vertex to its component root
    relative_mon = {v: 0 for v in nodes}
    component_unbalanced = {v: False for v in nodes}  # only meaningful at roots

    def find_with_mon(x):
        """Path compression with monodromy tracking."""
        if parent[x] == x:
            return x, 0
        root, mon_to_parent = find_with_mon(parent[x])
        relative_mon[x] = (relative_mon[x] + mon_to_parent) % 2
        parent[x] = root
        return root, relative_mon[x]

    for (u, v), sign in edges_with_signs:
        ru, mon_u = find_with_mon(u)
        rv, mon_v = find_with_mon(v)
        if ru != rv:
            # Merge components
            # Edge (u, v) with sign: its monodromy contribution
            # We want: relative_mon[v] in new root frame = mon_u + sign
            # currently relative_mon[v] = mon_v in v's root frame
            # If we attach rv under ru: new relative_mon[rv] = mon_u + sign + mon_v (XOR)
            target = min(ru, rv)
            other = max(ru, rv)
            if target == ru:
                offset = (mon_u + sign + mon_v) % 2
            else:
                offset = (mon_v + sign + mon_u) % 2
            parent[other] = target
            relative_mon[other] = offset
            # Merge unbalanced flags
            component_unbalanced[target] = (
                component_unbalanced[ru] or component_unbalanced[rv]
            )
        else:
            # Cycle: check balance
            cycle_sign = (mon_u + sign + mon_v) % 2
            if cycle_sign != 0:
                component_unbalanced[ru] = True

    # Count balanced components
    roots = set()
    n_unbalanced = 0
    for v in nodes:
        r, _ = find_with_mon(v)
        roots.add(r)
    for r in roots:
        if component_unbalanced[r]:
            n_unbalanced += 1
    n_balanced = len(roots) - n_unbalanced
    return len(nodes) - n_balanced


def compute_t_signed(nodes, edges_with_signs, x_sym, y_sym):
    """T_signed(G_base, χ; x, y) via direct subset enumeration."""
    n_edges = len(edges_with_signs)
    if n_edges > 20:
        raise ValueError(f"Too many signed edges ({n_edges})")

    r_E = signed_rank(nodes, edges_with_signs)
    total = 0
    for mask in range(1 << n_edges):
        A = [edges_with_signs[i] for i in range(n_edges) if mask & (1 << i)]
        r_A = signed_rank(nodes, A)
        weight = (x_sym - 1) ** (r_E - r_A) * (y_sym - 1) ** (len(A) - r_A)
        total += weight
    return sympy.expand(total)


def compute_t_unsigned(nodes, edges, x_sym, y_sym):
    """Standard Tutte polynomial via direct enumeration."""
    n_edges = len(edges)
    r_E = rank_subgraph(nodes, edges)
    total = 0
    for mask in range(1 << n_edges):
        A = [edges[i] for i in range(n_edges) if mask & (1 << i)]
        r_A = rank_subgraph(nodes, A)
        weight = (x_sym - 1) ** (r_E - r_A) * (y_sym - 1) ** (len(A) - r_A)
        total += weight
    return sympy.expand(total)


def verify_signed_rank_function():
    """Verify signed_rank reduces to standard rank when all signs = 0."""
    print("--- Verify signed_rank with all-zero signs == standard rank ---")
    test_graphs = [
        ("C_4", nx.cycle_graph(4)),
        ("K_4", nx.complete_graph(4)),
        ("Cubical", nx.cubical_graph()),
    ]
    for name, g in test_graphs:
        nodes = list(g.nodes())
        edges_unsigned = sorted(g.edges())
        edges_signed = [(e, 0) for e in edges_unsigned]
        r1 = rank_subgraph(nodes, edges_unsigned)
        r2 = signed_rank(nodes, edges_signed)
        print(f"  {name}: unsigned rank = {r1}, signed rank (all 0) = {r2}, match: {r1 == r2}")


def verify_signed_unbalanced():
    """Test signed rank on a known unbalanced case: triangle with one edge sign 1."""
    print("\n--- Verify signed_rank on triangle with one negative edge ---")
    # Triangle 0-1-2 with edges (0,1)=0, (1,2)=0, (0,2)=1
    # Cycle sign = 0+0+1 = 1 → UNBALANCED component
    # signed_rank should be |V| - 0 (no balanced components, since component is unbalanced)
    # = 3 - 0 = 3
    nodes = [0, 1, 2]
    edges_signed = [((0, 1), 0), ((1, 2), 0), ((0, 2), 1)]
    r = signed_rank(nodes, edges_signed)
    print(f"  triangle with χ=(0,0,1): signed_rank = {r} (expected 3 — fully unbalanced)")
    # Compare to balanced version
    edges_balanced = [((0, 1), 0), ((1, 2), 0), ((0, 2), 0)]
    r2 = signed_rank(nodes, edges_balanced)
    print(f"  triangle with χ=(0,0,0): signed_rank = {r2} (expected 2 — balanced)")


def build_quotient_with_monodromy(g_nx, perm):
    """For 2-fold cover G → G/⟨perm⟩, compute quotient multigraph + monodromy χ.

    Returns:
      quotient_nodes: list of vertex orbit IDs
      quotient_edges: list of ((u_quot, v_quot), sign, is_fixed_loop) per edge orbit
      orbit_sizes: list of edge orbit sizes (1 or 2)
      n_quot_loops: number of σ-fixed edges (becoming loops in quotient)
    """
    nodes = list(g_nx.nodes())
    # Vertex orbits
    seen_v = set()
    orbit_of_v = {}
    next_oid = 0
    for v in nodes:
        if v in seen_v:
            continue
        oid = next_oid
        next_oid += 1
        cur = v
        for _ in range(20):
            if cur in seen_v:
                break
            seen_v.add(cur)
            orbit_of_v[cur] = oid
            cur = perm[cur]
    quotient_nodes = sorted(set(orbit_of_v.values()))

    # Pick canonical sheet labeling: vertex with smallest label in orbit = sheet 0
    sheet_of = {}
    orbit_min = {}
    for v, oid in orbit_of_v.items():
        if oid not in orbit_min or v < orbit_min[oid]:
            orbit_min[oid] = v
    for v, oid in orbit_of_v.items():
        sheet_of[v] = 0 if v == orbit_min[oid] else 1

    # Edge orbits + monodromy
    edges = sorted(g_nx.edges())
    seen_e = set()
    quot_edges = []  # ((u_q, v_q), sign, is_loop_in_quot)
    orbit_sizes = []
    for e in edges:
        ekey = tuple(sorted(e))
        if ekey in seen_e:
            continue
        # Compute σ-orbit of e
        orbit = []
        cur = e
        for _ in range(10):
            ckey = tuple(sorted(cur))
            if ckey in [tuple(sorted(o)) for o in orbit]:
                break
            orbit.append(cur)
            seen_e.add(ckey)
            cur = (perm[cur[0]], perm[cur[1]])
        u, v = e
        u_q, v_q = orbit_of_v[u], orbit_of_v[v]
        is_loop = (u_q == v_q)
        # Monodromy: sheet sum mod 2
        sign = (sheet_of[u] + sheet_of[v]) % 2
        quot_edges.append(((min(u_q, v_q), max(u_q, v_q)), sign, is_loop))
        orbit_sizes.append(len(orbit))

    return quotient_nodes, quot_edges, orbit_sizes


def compute_t_fix_via_signed_dp(g_nx, perm, x_sym, y_sym):
    """Compute T_fix^σ(G) via signed-graph DP on quotient (per the lifting formula).

    For each subset L of σ-orbit edges (= subset of quotient edges):
      |A_L| = sum of orbit_sizes for orbits in L
      n_loops_in_L = # σ-fixed orbits in L
      r_quot(L) = unsigned rank in quotient
      c(L) := nullity_signed(L_signed) ... we test this hypothesis
    """
    nodes = list(g_nx.nodes())
    r_E = rank_subgraph(nodes, sorted(g_nx.edges()))
    quotient_nodes, quot_edges, orbit_sizes = build_quotient_with_monodromy(g_nx, perm)
    n_orb = len(quot_edges)
    if n_orb > 18:
        raise ValueError(f"Too many orbits ({n_orb})")

    total = 0
    for mask in range(1 << n_orb):
        L = [(quot_edges[i][0], quot_edges[i][1]) for i in range(n_orb) if mask & (1 << i)]
        L_loops = [quot_edges[i][2] for i in range(n_orb) if mask & (1 << i)]
        L_sizes = [orbit_sizes[i] for i in range(n_orb) if mask & (1 << i)]
        L_unsigned_edges = [edge for (edge, _sign) in L]
        # Strip loops from unsigned rank computation (loops contribute 0 to rank)
        L_unsigned_no_loops = [e for e in L_unsigned_edges if e[0] != e[1]]
        r_quot = rank_subgraph(quotient_nodes, L_unsigned_no_loops)

        # Signed rank
        r_signed = signed_rank(quotient_nodes, L)
        nullity_signed = sum(1 for e in L) - r_signed
        nullity_unsigned = len(L) - r_quot
        # Test hypothesis: c(L) = nullity_unsigned - nullity_signed + n_loops_already_accounted?
        # Or c(L) = (# unbalanced components in L) (from above analysis)?

        # Per empirical: c(L) = nullity_G(A_L) - nullity_quot(L) + n_loops_in_L
        # nullity_quot(L) here = nullity_unsigned (using strip-loops or include-loops?)

        # Corrected formula: c(L) = nullity_signed(L) = |L| - r_signed(L)
        # This works for both free 2-fold covers AND covers with σ-fixed edges.
        # Derivation:
        #   For free cover: nullity_G = 2*balanced + unbalanced;
        #                   nullity_quot = balanced + unbalanced
        #                   c = nullity_G - nullity_quot = balanced
        #   Also: balanced = nullity_quot - unbalanced = (|L|-r_quot) - (r_signed-r_quot) = |L| - r_signed
        #   For non-free cover (with σ-fixed edges): same identity holds via signed_rank
        #   handling loops correctly (loop with χ=1 unbalances its component).
        c = len(L) - r_signed

        n_loops = sum(1 for is_loop in L_loops if is_loop)
        size_A_L = sum(L_sizes)
        # r_G via formula: r_quot + |L| - c
        # nullity_G = |A_L| - r_G = size_A_L - r_quot - len(L) + c
        nullity_G_proposed = size_A_L - r_quot - len(L) + c
        # exponent of (x-1) = r_E - r_G = r_E - r_quot - len(L) + c
        weight = (x_sym - 1) ** (r_E - r_quot - len(L) + c) * (y_sym - 1) ** nullity_G_proposed
        total += weight
    return sympy.expand(total)


def compute_t_fix_direct(g_nx, perm, x_sym, y_sym):
    """T_fix^σ via direct enumeration over σ-fixed subsets of E(G)."""
    nodes = list(g_nx.nodes())
    edges = sorted(g_nx.edges())
    r_E = rank_subgraph(nodes, edges)

    # Compute σ-orbits of edges
    seen = set()
    orbits = []
    for e in edges:
        ekey = tuple(sorted(e))
        if ekey in seen:
            continue
        orb = []
        cur = e
        for _ in range(20):
            ckey = tuple(sorted(cur))
            if ckey in [tuple(sorted(o)) for o in orb]:
                break
            orb.append(cur)
            seen.add(ckey)
            cur = (perm[cur[0]], perm[cur[1]])
        orbits.append(orb)

    n_orb = len(orbits)
    if n_orb > 18:
        raise ValueError(f"Too many σ-orbits ({n_orb})")
    total = 0
    for mask in range(1 << n_orb):
        A = []
        for i in range(n_orb):
            if mask & (1 << i):
                A.extend(orbits[i])
        r_A = rank_subgraph(nodes, A)
        weight = (x_sym - 1) ** (r_E - r_A) * (y_sym - 1) ** (len(A) - r_A)
        total += weight
    return sympy.expand(total)


def main():
    print("=" * 70)
    print("Signed-graph Tutte prototype + T_fix^σ verification")
    print("=" * 70)

    verify_signed_rank_function()
    verify_signed_unbalanced()

    x = sympy.Symbol('x')
    y = sympy.Symbol('y')

    print("\n--- T_signed sanity ---")
    nodes = [0, 1, 2]
    edges_balanced = [((0, 1), 0), ((1, 2), 0), ((0, 2), 0)]
    T_bal = compute_t_signed(nodes, edges_balanced, x, y)
    T_unsigned = compute_t_unsigned(nodes, [(0,1), (1,2), (0,2)], x, y)
    print(f"  triangle balanced T_signed = {T_bal}, T_unsigned = {T_unsigned}, match: {T_bal == T_unsigned}")

    # KEY VERIFICATION: T_fix^σ via signed DP on quotient = T_fix^σ via direct
    print("\n--- T_fix^σ verification: signed DP on quotient vs direct enumeration ---")
    cases = [
        ("C_4 with rot-2", nx.cycle_graph(4), {0: 2, 1: 3, 2: 0, 3: 1}),
        ("K_4 with σ=(01)(23)", nx.complete_graph(4), {0: 1, 1: 0, 2: 3, 3: 2}),
        ("Cubical with K_2-swap", nx.cubical_graph(),
         {**{i: i+4 for i in range(4)}, **{i+4: i for i in range(4)}}),
    ]
    for name, g, perm in cases:
        nodes_orig = sorted(g.nodes())
        if nodes_orig != list(range(len(nodes_orig))):
            label_map = {n: i for i, n in enumerate(nodes_orig)}
            g = nx.relabel_nodes(g, label_map)
            perm = {label_map[k]: label_map[v] for k, v in perm.items()}
        try:
            T_sig = compute_t_fix_via_signed_dp(g, perm, x, y)
            T_dir = compute_t_fix_direct(g, perm, x, y)
            match = sympy.expand(T_sig - T_dir) == 0
            status = "✓" if match else "✗"
            print(f"  {status} {name}: signed DP = direct: {match}")
            if not match:
                print(f"    signed DP: {T_sig}")
                print(f"    direct:    {T_dir}")
                print(f"    diff:      {sympy.expand(T_sig - T_dir)}")
        except SyntaxError as e:
            print(f"  ✗ {name}: SYNTAX ERROR - {e}")
        except Exception as e:
            print(f"  ✗ {name}: ERROR - {e}")


if __name__ == "__main__":
    main()
