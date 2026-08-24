"""Verify T_fix^σ DP against brute-force on small free 2-fold covers.

For a free 2-fold cover G → G/⟨σ⟩ with monodromy χ:
  T_fix^σ(G) := Σ_{A ⊆ E(G) : σ(A)=A} (x-1)^{r(E_G)-r_G(A)} (y-1)^{|A|-r_G(A)}
              = Σ_L (x-1)^{r(E_G) - r_quot(L) - r_signed(L)} (y-1)^{2|L| - r_quot - r_signed}.

We compute T_fix^σ both ways:
  1. DP via compute_t_fix_sigma_mod on quotient + χ.
  2. Brute force enumerating σ-invariant subsets of E(G).
And verify agreement.
"""
from __future__ import annotations

import sympy
import networkx as nx

from tutte.deprecated.signed_elim_dp import compute_t_fix_sigma_mod
from tutte.tests.signed_graph_tutte_prototype import (
    build_quotient_with_monodromy,
    rank_subgraph,
)


def compute_t_fix_sigma_brute(g_nx: nx.Graph, perm: dict, x_sym, y_sym):
    """Brute force: enumerate σ-invariant edge subsets and sum Whitney terms."""
    edges = sorted(g_nx.edges())
    n_e = len(edges)
    if n_e > 22:
        raise ValueError(f"Too many edges ({n_e}) for brute force")
    nodes = list(g_nx.nodes())
    r_E = rank_subgraph(nodes, edges)
    total = 0
    for mask in range(1 << n_e):
        A = [edges[i] for i in range(n_e) if mask & (1 << i)]
        # Check σ-invariance: σ(A) = A as a set.
        A_set = frozenset(tuple(sorted(e)) for e in A)
        sigma_A_set = frozenset(
            tuple(sorted((perm[e[0]], perm[e[1]]))) for e in A
        )
        if A_set != sigma_A_set:
            continue
        r_A = rank_subgraph(nodes, A)
        weight = (x_sym - 1) ** (r_E - r_A) * (y_sym - 1) ** (len(A) - r_A)
        total += weight
    return sympy.expand(total)


def verify(g_nx: nx.Graph, perm: dict, label: str):
    x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
    t_fix_brute = compute_t_fix_sigma_brute(g_nx, perm, x_sym, y_sym)
    print(f"\n{label}: T_fix^σ (brute) = {t_fix_brute}")

    # DP via quotient
    quot_nodes, quot_edges_full, _ = build_quotient_with_monodromy(g_nx, perm)
    quot_edges = [(e, s) for (e, s, _l) in quot_edges_full]
    r_E_G = rank_subgraph(list(g_nx.nodes()), sorted(g_nx.edges()))

    for x_v, y_v, p in [(2, 3, 1009), (5, 7, 2017), (-1, 4, 4019)]:
        expected = int(t_fix_brute.subs({x_sym: x_v, y_sym: y_v})) % p
        actual, _ = compute_t_fix_sigma_mod(quot_nodes, quot_edges, r_E_G, x_v, y_v, p)
        tag = "OK" if actual == expected else "FAIL"
        print(f"  ({x_v},{y_v},mod {p}): expected={expected}, got={actual} {tag}")


def main():
    # C_4 with σ = (02)(13) — FREE 2-fold cover
    g = nx.cycle_graph(4)
    perm = {0: 2, 1: 3, 2: 0, 3: 1}
    verify(g, perm, "C_4 + (02)(13) [free]")

    # K_4 with σ = (01)(23) — NON-FREE (edges (0,1) and (2,3) are σ-fixed)
    g = nx.complete_graph(4)
    perm = {0: 1, 1: 0, 2: 3, 3: 2}
    verify(g, perm, "K_4 + (01)(23) [non-free]")

    # 3-cube antipodal — FREE
    g = nx.hypercube_graph(3)
    nm = {n: i for i, n in enumerate(sorted(g.nodes()))}
    g = nx.relabel_nodes(g, nm)
    perm = {i: 7 - i for i in range(8)}
    verify(g, perm, "Cube antipodal [free]")

    # K_{3,3} part-swap — NON-FREE (edges (0,3), (1,4), (2,5) are σ-fixed)
    g = nx.complete_bipartite_graph(3, 3)
    nm = {n: i for i, n in enumerate(sorted(g.nodes()))}
    g = nx.relabel_nodes(g, nm)
    perm = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}
    verify(g, perm, "K_{3,3} part-swap [non-free]")


if __name__ == "__main__":
    main()
