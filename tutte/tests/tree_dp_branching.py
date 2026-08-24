"""Recursive tree DP on branching cell-trees.

Test cases:
1. Path topologies (linear cell-trees, simplest case).
2. Y-shape (claw): 4 cells, 3 branching from a center cell.
3. T-shape: 5 cells, balanced branching.
4. Larger trees.
"""

from __future__ import annotations

import time
from typing import Dict, List

import networkx as nx

from tutte.graph import Graph, complete_graph
from tutte.lookup.core import load_default_table
from tutte.synthesis.engine import SynthesisEngine
from tutte.roots.cell_quotient_tree import (
    CellTreeSpec, compute_tree_dp_recursive,
)


def disjoint_union(g1, g2):
    nodes_set = set(g1.nodes)
    nodes_set.update({v + max(g1.nodes) + 1 for v in g2.nodes})
    edges_list = list(g1.edges)
    offset = max(g1.nodes) + 1
    for (u, v) in g2.edges:
        edges_list.append((u + offset, v + offset))
    return Graph(nodes=frozenset(nodes_set), edges=frozenset(edges_list))


def add_edges(g, edges):
    e = set(g.edges)
    for u, v in edges:
        e.add(tuple(sorted([u, v])))
    return Graph(nodes=g.nodes, edges=frozenset(e))


def make_M_k(k: int) -> Graph:
    nodes = frozenset(range(2 * k))
    edges = frozenset((i, i + k) for i in range(k))
    return Graph(nodes=nodes, edges=edges)


def build_tree_graph(cell: Graph, cell_tree: nx.Graph,
                      cell_anchor_groups: Dict[int, Dict[int, List[int]]]):
    """Build the actual graph from cell-tree spec.

    Each cell becomes a copy of `cell` (relabeled). Each junction (i, j) in
    cell_tree becomes M_k matching edges between cell_i's anchors-toward-j
    and cell_j's anchors-toward-i.
    """
    n_cells = cell_tree.number_of_nodes()
    cell_size = cell.node_count()
    # Allocate per-cell vertex offsets
    offsets = {i: i * cell_size for i in range(n_cells)}
    # Build disjoint union of n cells
    g = cell
    for _ in range(n_cells - 1):
        g = disjoint_union(g, cell)
    # Add junction edges
    for (i, j) in cell_tree.edges():
        anchors_i = cell_anchor_groups[i][j]
        anchors_j = cell_anchor_groups[j][i]
        assert len(anchors_i) == len(anchors_j)
        edges_to_add = [(anchors_i[k] + offsets[i], anchors_j[k] + offsets[j])
                         for k in range(len(anchors_i))]
        g = add_edges(g, edges_to_add)
    return g


def test_case(label, cell, cell_tree, cell_anchor_groups, k, root):
    table = load_default_table()
    engine = SynthesisEngine(table=table, verbose=False)
    engine.skip_target_lookup = True

    g = build_tree_graph(cell, cell_tree, cell_anchor_groups)
    print(f"\n[{label}] graph: {g.node_count()}n {g.edge_count()}e")
    direct = engine.synthesize(g).polynomial
    print(f"  engine: {len(direct.to_coefficients())} mons, T(1,1) = {direct.num_spanning_trees()}")

    spec = CellTreeSpec(
        cell_template=cell,
        junction_template=make_M_k(k),
        cell_tree=cell_tree,
        cell_anchor_groups=cell_anchor_groups,
        junction_anchors_A=list(range(k)),
        junction_anchors_B=list(range(k, 2 * k)),
        root=root,
    )
    try:
        t0 = time.time()
        tree_T = compute_tree_dp_recursive(spec)
        wall = time.time() - t0
        match = (tree_T == direct)
        print(f"  tree DP: {len(tree_T.to_coefficients())} mons, "
              f"T(1,1) = {tree_T.num_spanning_trees()}, match={'YES' if match else 'NO'} "
              f"({wall:.1f}s)")
        return match
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  tree DP failed: {type(exc).__name__}: {exc}")
        return False


def main():
    # === Regression: paths via the recursive form ===
    print("=" * 70)
    print("Regression: paths (linear topology baseline)")
    print("=" * 70)

    # 2-cell path
    K3 = complete_graph(3)
    t = nx.Graph(); t.add_nodes_from([0, 1]); t.add_edge(0, 1)
    test_case("K_3 2-cell path M_2", K3, t,
              {0: {1: [0, 1]}, 1: {0: [0, 1]}}, k=2, root=0)

    # 3-cell path
    t = nx.Graph(); t.add_nodes_from([0, 1, 2]); t.add_edges_from([(0,1), (1,2)])
    test_case("K_3 3-cell path M_2", K3, t,
              {0: {1: [0, 1]}, 1: {0: [0, 1], 2: [0, 1]}, 2: {1: [0, 1]}},
              k=2, root=0)

    # === BRANCHING: claw (Y-tree) ===
    print()
    print("=" * 70)
    print("BRANCHING: claw / Y-tree")
    print("=" * 70)

    # 4-cell claw: cell 0 is center, cells 1, 2, 3 are leaves
    t = nx.Graph()
    t.add_nodes_from([0, 1, 2, 3])
    t.add_edges_from([(0, 1), (0, 2), (0, 3)])
    # Cell 0 has 3 neighbors, K_3 cell only has 3 vertices —
    # all 3 anchor groups SHARE the same 2 vertices [0, 1] on cell 0.
    # Wait, that's a problem if K_3 only has 3 verts and we need 3 different
    # anchor sets. Let me use K_4 cells (4 verts, 6 edges) with M_2 junctions.
    K4 = complete_graph(4)
    test_case("K_4 claw M_2 (cell 0 = 3 children with shared anchors)",
              K4, t,
              {
                  0: {1: [0, 1], 2: [0, 1], 3: [0, 1]},  # ALL shared anchors
                  1: {0: [0, 1]},
                  2: {0: [0, 1]},
                  3: {0: [0, 1]},
              },
              k=2, root=1)

    # K_4 claw with DISJOINT anchors per child (cells use [0,1], [0,2], [0,3] respectively)
    test_case("K_4 claw M_2 (cell 0 = 3 children with DISJOINT anchors)",
              K4, t,
              {
                  0: {1: [0, 1], 2: [0, 2], 3: [0, 3]},  # only vertex 0 shared
                  1: {0: [0, 1]},
                  2: {0: [0, 1]},
                  3: {0: [0, 1]},
              },
              k=2, root=1)

    # K_5 claw with bigger anchor sets
    K5 = complete_graph(5)
    test_case("K_5 claw M_3 (cell 0 = 3 children, shared)",
              K5, t,
              {
                  0: {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]},
                  1: {0: [0, 1, 2]},
                  2: {0: [0, 1, 2]},
                  3: {0: [0, 1, 2]},
              },
              k=3, root=1)


if __name__ == "__main__":
    main()
