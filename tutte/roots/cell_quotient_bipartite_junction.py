"""Cell-quotient DP for non-matching bipartite junctions.

Companion to `cell_quotient_tree.py` and `cell_quotient_cycle.py`. The
existing k-matching path requires:

- Each inter-cell anchor used in exactly one inter-cell edge (matching).
- Anchors on each side lie in a single vertex-transitive class.

When the inter-cell junction is DISCONNECTED (multiple components with
disjoint anchor sets on each side), `compute_bipartite_junction_per_component_dp`
splits the junction into per-component sub-junctions and processes each
as a separate step. This avoids materializing the joint
Bell(|junction_boundary|) partition dict (intractable for Z(1, 2) with
24 anchors → Bell(24) ≈ 10^17). For Z(1, 2): junction = 2 components ×
12 anchors each, Bell(12) ≈ 4.2M keys per intermediate state (still
large but feasible if sparse).

These constraints fail for several D-Wave families:

| Family | Inter-cell structure | k-matching? |
|---|---|---|
| Cm_2 | M_4 matching, anchors on one bipartition side | ✓ |
| Z(1, 2) | 2 disjoint bipartite components, anchor degrees [2,2,2,2,2,2,4,4,4,4] | ✗ |
| Pm_2 | Multi-anchor bipartite | ✗ |

This module relaxes both: any inter-cell subgraph (bipartite by
construction since the edges cross cells) becomes a `BipartiteJunction`,
its `junction_template` is the actual subgraph, and per-cell anchors are
the deduplicated lists. T_rooted on the junction template uses
`t_rooted_smart` from `rooted_tutte.py` which factorises over
disconnected components → 2 × 2^16 instead of 2^32 on Z(1, 2).

## Pipeline

```
detect_bipartite_junction_topology(graph, partition, inter_edges)
        ↓ (one BipartiteJunction per cell-pair)
build_bipartite_junction_spec(graph, table)
        ↓ (CellTreeSpec with junction_template = actual junction graph)
compute_tree_dp_simple(spec)  # existing chain DP machinery
        ↓
TuttePolynomial T(G)
```

The detection function lives in `tutte/graphs/covering.py` next to
`detect_kmatching_topology`. This module owns the spec construction +
DP dispatch.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from ..graph import Graph
from ..graphs.covering import (BipartiteJunction, detect_bipartite_junction_topology,
                               try_hierarchical_partition)
from ..polynomial import TuttePolynomial
from .cell_quotient_tree import CellTreeSpec


def build_bipartite_junction_spec(
    graph: Graph, table,
) -> Optional[Tuple[CellTreeSpec, List[BipartiteJunction], nx.Graph, List[Dict[int, int]], List]]:
    """Build a CellTreeSpec for graphs whose cell-pair junctions are
    arbitrary bipartite subgraphs (NOT restricted to k-matchings).

    Returns ``(spec, junctions, cell_topology, isos, partition)`` or None
    if hierarchical partitioning fails or junctions span heterogeneous
    cell sizes.

    Limitations:
    - Currently requires all junctions between the SAME cell-pair to use
      the SAME junction template (verified by canonical_key). Different
      cell pairs may have different junction templates.
    - All cells must be isomorphic to a single cell_template (same as
      the k-matching path).
    """
    result = try_hierarchical_partition(graph, table)
    if result is None:
        return None
    cell_entry, partition, inter_info = result
    if not partition or not inter_info.edges:
        return None

    junctions = detect_bipartite_junction_topology(
        graph, partition, list(inter_info.edges),
    )
    if not junctions:
        return None

    # Cell topology: nodes = cells, edges = pairs with at least one
    # inter-cell edge. Multiple junctions between same pair → single edge
    # in cell_topology (parallel-edge collapsed).
    cell_topology = nx.Graph()
    cell_topology.add_nodes_from(range(len(partition)))
    for j in junctions:
        cell_topology.add_edge(j.cell_i, j.cell_j)
    # We allow cycles in cell_topology (handled by chain framework
    # for n_cells=2 case naturally; the tree DP rejects cycles upstream).

    # Build cell template from cell 0 (relabel to 0..n-1).
    cell0_nodes = sorted(partition[0])
    relabel0 = {v: i for i, v in enumerate(cell0_nodes)}
    cell0_edges = [
        (relabel0[u], relabel0[v]) for u, v in graph.edges
        if u in relabel0 and v in relabel0
    ]
    cell_template = Graph(list(range(len(cell0_nodes))), cell0_edges)
    cell_template_nx = nx.Graph()
    cell_template_nx.add_nodes_from(range(len(cell0_nodes)))
    cell_template_nx.add_edges_from(cell0_edges)

    # Per-cell isomorphism to cell_template.
    isos: List[Dict[int, int]] = []
    for i, cell_nodes in enumerate(partition):
        if i == 0:
            isos.append(relabel0)
            continue
        cell_nx = nx.Graph()
        cell_nx.add_nodes_from(sorted(cell_nodes))
        for u, v in graph.edges:
            if u in cell_nodes and v in cell_nodes:
                cell_nx.add_edge(u, v)
        matcher = GraphMatcher(cell_nx, cell_template_nx)
        if not matcher.is_isomorphic():
            return None
        isos.append(matcher.mapping)

    # Per-cell anchor groups: cell_anchor_groups[cell_idx][neighbor_idx] =
    # list of local-template anchor labels.
    cell_anchor_groups: Dict[int, Dict[int, List[int]]] = {
        i: {} for i in range(len(partition))
    }
    # If a cell pair has MULTIPLE junctions (theoretically possible),
    # we currently collapse them by concatenating anchors. For Z(1, 2):
    # each cell pair has exactly one junction (the inter-cell structure
    # for that pair is one BipartiteJunction, possibly disconnected).
    for j in junctions:
        anchors_i_local = [isos[j.cell_i][a] for a in j.anchors_i]
        anchors_j_local = [isos[j.cell_j][a] for a in j.anchors_j]
        existing_i = cell_anchor_groups[j.cell_i].get(j.cell_j)
        if existing_i is None:
            cell_anchor_groups[j.cell_i][j.cell_j] = anchors_i_local
            cell_anchor_groups[j.cell_j][j.cell_i] = anchors_j_local
        else:
            # Multiple junctions between same cell pair — extend anchor list.
            # The downstream DP will see the union.
            cell_anchor_groups[j.cell_i][j.cell_j] = list(
                dict.fromkeys(existing_i + anchors_i_local)
            )
            cell_anchor_groups[j.cell_j][j.cell_i] = list(dict.fromkeys(
                cell_anchor_groups[j.cell_j][j.cell_i] + anchors_j_local
            ))

    # Junction template: use the actual junction graph from the FIRST
    # junction (we require all junctions to be isomorphic to it for
    # the spec to work — verified below).
    first_j = junctions[0]
    junction_template = first_j.to_junction_graph()
    first_canon = junction_template.canonical_key()
    n_anchors_i = len(first_j.anchors_i)
    junction_anchors_A = list(range(n_anchors_i))
    junction_anchors_B = list(range(n_anchors_i, n_anchors_i + len(first_j.anchors_j)))

    for j in junctions[1:]:
        cand_template = j.to_junction_graph()
        if cand_template.canonical_key() != first_canon:
            # Heterogeneous junction templates — outside current scope.
            return None

    leaf_candidates = [n for n, d in cell_topology.degree() if d == 1]
    root = leaf_candidates[0] if leaf_candidates else 0

    spec = CellTreeSpec(
        cell_template=cell_template,
        junction_template=junction_template,
        cell_tree=cell_topology,
        cell_anchor_groups=cell_anchor_groups,
        junction_anchors_A=junction_anchors_A,
        junction_anchors_B=junction_anchors_B,
        root=root,
    )
    return spec, junctions, cell_topology, isos, partition


_DEFAULT_MAX_CELL_BOUNDARY = 8
"""Default cap on per-cell anchor boundary size.

T_rooted brute force on a cell with `b` boundary vertices iterates
``2^|E|`` subgraph masks AND maintains a partition dict of size up to
``Bell(b)``. ``Bell(8) = 4140`` is tractable in seconds; ``Bell(12) ≈ 4M``
is multi-minute even with sparse keys. The cap defaults to ``8`` to
keep the engine dispatch responsive; raise it explicitly via the
``max_cell_boundary`` argument when calling from a benchmark / research
script that's willing to wait.
"""


def compute_cell_quotient_bipartite_junction_dp(
    graph: Graph, table,
    max_cell_boundary: int = _DEFAULT_MAX_CELL_BOUNDARY,
) -> Optional[TuttePolynomial]:
    """Top-level entry — extends `compute_cell_quotient_tree_dp` to non-matching junctions.

    Detects the hierarchical partition + bipartite junctions, builds a
    CellTreeSpec with the actual junction graph as the template (not M_k),
    and dispatches to `compute_tree_dp_simple` (linear-path case).

    Returns None if:
    - Hierarchical partition fails or junctions span heterogeneous
      templates between cell pairs.
    - The cell topology has a cycle (current implementation handles only
      tree topologies; 2-cell cycle = single junction is OK and treated
      as a linear path).
    - Per-cell anchor boundary exceeds ``max_cell_boundary``. T_rooted
      brute force on a 12-vert boundary (e.g., Z(1, 1) cell with all
      verts as anchors) doesn't complete in minutes — the guard prevents
      the engine dispatch from hanging. Z(1, 2) currently hits this
      guard; lifting it requires either treewidth-aware T_rooted or
      smaller per-junction anchor partitions.
    """
    from .cell_quotient_tree import compute_tree_dp_simple

    built = build_bipartite_junction_spec(graph, table)
    if built is None:
        return None
    spec, _junctions, cell_topology, _isos, _partition = built

    if not nx.is_tree(cell_topology):
        return None
    leaves = [n for n, d in cell_topology.degree() if d == 1]
    if len(leaves) != 2:
        return None
    # Same n_cells > 6 gate as other cell_quotient paths. Per-cell
    # T_rooted boundary × n_cells scales as Bell(boundary × n_cells);
    # 9-cell decompositions hit Bell(18)-class orbit explosion in
    # `compute_tree_dp_simple` downstream.
    if len(_partition) > 6 and graph.node_count() > 36:
        return None

    # Guard against intractable per-cell boundaries.
    #
    # Cache hits on the cell-template T_rooted are necessary but NOT
    # sufficient — the downstream `compute_tree_dp_simple` still iterates
    # the junction's partition dict via `enumerate_partitions_per_orbit`,
    # which is Bell(|junction_boundary|). For Z(1, 2) the junction is
    # 24-anchor disconnected graph; Bell(24) > 10^17 is unreachable even
    # though the cell T_rooted is cached. So we leave the guard at
    # `max_cell_boundary` regardless of cache state — cache hits don't
    # speed up the per-iteration partition processing.
    #
    # Lifting Z(1, 2) requires factoring the junction's per-component
    # T_rooted into the M-table convolution (open research — would need
    # a per-component-junction tree DP variant).
    from .rooted_tutte import _T_ROOTED_CACHE
    cell_canon = spec.cell_template.canonical_key()
    for cell_idx, neighbor_anchors in spec.cell_anchor_groups.items():
        union = sorted({a for anchors in neighbor_anchors.values() for a in anchors})
        cache_key = (cell_canon, tuple(union))
        if cache_key not in _T_ROOTED_CACHE and len(union) > max_cell_boundary:
            return None

    try:
        return compute_tree_dp_simple(spec)
    except Exception:
        return None


# =============================================================================
# Per-component DP: factor the junction into its connected components and
# convolve each independently to avoid the Bell(|junction_boundary|) wall.
# =============================================================================


def _split_junction_by_component(
    junction: BipartiteJunction,
) -> List[BipartiteJunction]:
    """Split a BipartiteJunction into per-connected-component sub-junctions.

    Each returned sub-junction has the same (cell_i, cell_j) ids; its
    `edges` is the subset of edges in one connected component of the
    junction subgraph; `anchors_i` and `anchors_j` are restricted to
    vertices touched by those edges.

    For Z(1, 2) the input has 32 edges in 2 components (16 edges each);
    output is 2 BipartiteJunctions with 6 anchors each side.
    """
    # Build the junction subgraph using GLOBAL vertex labels so that
    # the resulting components map back to junction.edges directly.
    nxg = nx.Graph()
    nxg.add_nodes_from(junction.anchors_i)
    nxg.add_nodes_from(junction.anchors_j)
    nxg.add_edges_from(junction.edges)
    components = list(nx.connected_components(nxg))
    if len(components) <= 1:
        return [junction]

    out: List[BipartiteJunction] = []
    for comp_nodes in components:
        comp_edges = [
            (u, v) for u, v in junction.edges
            if u in comp_nodes and v in comp_nodes
        ]
        if not comp_edges:
            continue
        ai = sorted(set(junction.anchors_i) & comp_nodes)
        aj = sorted(set(junction.anchors_j) & comp_nodes)
        out.append(BipartiteJunction(
            cell_i=junction.cell_i, cell_j=junction.cell_j,
            edges=comp_edges, anchors_i=ai, anchors_j=aj,
        ))
    return out


def _component_subjunction_to_template(
    comp_junction: BipartiteJunction,
) -> Tuple["Graph", List[int], List[int]]:
    """Build a relabeled-from-0 Graph for one junction component plus
    the local-template anchor lists.

    Returns ``(template, anchors_local_i, anchors_local_j)`` where:
    - template: graph with vertices [0..n_anchors_i + n_anchors_j - 1]
    - anchors_local_i: [0..n_anchors_i - 1]
    - anchors_local_j: [n_anchors_i..n_anchors_i + n_anchors_j - 1]
    """
    ai = comp_junction.anchors_i
    aj = comp_junction.anchors_j
    n_i = len(ai)
    n_j = len(aj)
    relabel: Dict[int, int] = {}
    for i, v in enumerate(ai):
        relabel[v] = i
    for i, v in enumerate(aj):
        relabel[v] = n_i + i
    local_edges = [
        (relabel[u], relabel[v]) for u, v in comp_junction.edges
    ]
    template = Graph(list(range(n_i + n_j)), local_edges)
    anchors_local_i = list(range(n_i))
    anchors_local_j = list(range(n_i, n_i + n_j))
    return template, anchors_local_i, anchors_local_j


def compute_bipartite_junction_per_component_dp(
    graph: Graph, table,
    max_cell_boundary: int = 12,
    max_intermediate_state_entries: int = 10_000_000,
    verbose: bool = False,
) -> Optional[TuttePolynomial]:
    """Per-component variant of `compute_cell_quotient_bipartite_junction_dp`.

    For 2-cell linear path with multi-component bipartite junction
    between them (e.g., Z(1, 2)): processes each junction component as
    a separate convolution step, avoiding the Bell(|joint_junction_boundary|)
    wall.

    Returns None if:
    - Hierarchical partition fails / not 2 cells / not 2-cell linear path.
    - Per-cell anchor boundary exceeds ``max_cell_boundary`` (Z(1, 1)
      cells with 12 anchors are at the limit).
    - Intermediate state grows beyond ``max_intermediate_state_entries``
      (sparse-Bell wall — when the joint partition dict after a junction
      component would be too large to materialize).

    Validation entry point. Engine dispatch should call this AFTER
    `compute_cell_quotient_bipartite_junction_dp` (which is faster for
    single-component junctions).
    """
    from .aut_orbit import (aut_compress_t_rooted, build_relabel_aut,
                            compute_cell_aut)
    from .cell_quotient_helpers import (orbit_convolve, precompute_M_table)
    from .rooted_tutte import (_T_ROOTED_CACHE, divide_by_x_minus_1_power,
                                relabel_partition_dict, t_rooted_cached)

    built = build_bipartite_junction_spec(graph, table)
    if built is None:
        return None
    spec, junctions_top, cell_topology, isos, partition = built

    if not nx.is_tree(cell_topology):
        return None
    leaves = [n for n, d in cell_topology.degree() if d == 1]
    if len(leaves) != 2:
        return None
    if len(partition) != 2:
        return None  # current impl: 2-cell linear path only

    cell_a_idx = min(leaves)
    cell_b_idx = max(leaves)

    # The detection emits ONE BipartiteJunction per cell pair.
    pair_junctions = [
        j for j in junctions_top
        if {j.cell_i, j.cell_j} == {cell_a_idx, cell_b_idx}
    ]
    if not pair_junctions:
        return None
    # In the path-topology case there should be exactly one BipartiteJunction
    # that we now split by connected component.
    top_j = pair_junctions[0]
    components = _split_junction_by_component(top_j)
    if len(components) == 0:
        return None

    # If only 1 component the per-component variant adds no value vs the
    # standard `compute_tree_dp_simple`; defer to that path.
    if len(components) == 1:
        try:
            from .cell_quotient_tree import compute_tree_dp_simple
            return compute_tree_dp_simple(spec)
        except Exception:
            return None

    cell_template = spec.cell_template

    cell_a_anchors = spec.cell_anchor_groups[cell_a_idx][cell_b_idx]
    cell_b_anchors = spec.cell_anchor_groups[cell_b_idx][cell_a_idx]
    if (len(cell_a_anchors) > max_cell_boundary
            or len(cell_b_anchors) > max_cell_boundary):
        # Bail unless we already have the cell-template T_rooted in the
        # in-process cache. We do NOT consult the persistent lookup here:
        # even though `t_rooted_cached` would populate the in-process cache
        # for free, the DOWNSTREAM per-component convolution (precompute_M_table
        # × orbit_convolve) is Bell(|cell_anchors|) which dominates the
        # runtime. For Z(1, 2) cells (12 anchors) it is ~10× slower than
        # the engine's treewidth_dp fallback. Engineering this path to
        # be faster than treewidth_dp would require state aut compression
        # with diagonal-subgroup tracking + C-ext for the inner loop;
        # neither is wired yet. Until then, gate on the in-process cache,
        # which is empty at engine start so Z(1, 2)+ defer to treewidth_dp.
        cell_canon = cell_template.canonical_key()
        for anchors in (cell_a_anchors, cell_b_anchors):
            cache_key = (cell_canon, tuple(sorted(anchors)))
            if cache_key not in _T_ROOTED_CACHE:
                return None

    # Position scheme:
    #   cell A's k-th anchor (template-local) → position 10000 + k
    #   cell B's k-th anchor (template-local) → position 20000 + k
    pos_a: Dict[int, int] = {a: 10000 + i for i, a in enumerate(cell_a_anchors)}
    pos_b: Dict[int, int] = {a: 20000 + i for i, a in enumerate(cell_b_anchors)}

    import time as _time
    _t_start = _time.time()

    def _log(msg: str) -> None:
        if verbose:
            elapsed = _time.time() - _t_start
            print(f"[per-comp +{elapsed:.1f}s] {msg}", flush=True)

    # === Initialize state with cell A's T_rooted ===
    _log(f"computing T_rooted(cell, {len(cell_a_anchors)} anchors)...")
    T_cell_a = t_rooted_cached(cell_template, cell_a_anchors)
    _log(f"  cell A T_rooted: {len(T_cell_a)} entries")
    state_T = relabel_partition_dict(T_cell_a, pos_a)

    # No aut compression on the state: tracking the evolving aut across
    # per-component steps is non-trivial (would require the diagonal
    # subgroup of state-aut × junction-aut at each step). Leaving the
    # state uncompressed sacrifices ~8× on Z(1, 1) but keeps the per-
    # component convolution mathematically straightforward.
    state_orbit_T, state_orbit_partitions = aut_compress_t_rooted(state_T, [])
    state_open_pos = list(pos_a.values())
    total_div = 0
    _log(f"  initial state: {len(state_orbit_T)} entries, "
         f"{len(state_open_pos)} open positions")

    iso_a = isos[cell_a_idx]
    iso_b = isos[cell_b_idx]
    inv_iso_a = {v: k for k, v in iso_a.items()}  # local-cell-A label → graph label
    inv_iso_b = {v: k for k, v in iso_b.items()}

    # === Process each junction component ===
    _log(f"processing {len(components)} junction component(s)")
    for k, comp_j in enumerate(components):
        # comp_j.anchors_i / anchors_j are graph-vertex labels.
        # Translate to cell-template-local labels via iso[cell_*].
        L_k_cell_a_local = [iso_a[a] for a in comp_j.anchors_i]
        R_k_cell_b_local = [iso_b[a] for a in comp_j.anchors_j]
        L_k_pos = [pos_a[a] for a in L_k_cell_a_local]
        R_k_pos = [pos_b[a] for a in R_k_cell_b_local]

        # Build the component template (relabeled from 0) and its T_rooted.
        comp_template, comp_anchors_local_i, comp_anchors_local_j = (
            _component_subjunction_to_template(comp_j)
        )
        T_comp = t_rooted_cached(
            comp_template, comp_anchors_local_i + comp_anchors_local_j,
        )
        # Relabel: local_i[k] → L_k_pos[k]; local_j[k] → R_k_pos[k]
        j_label_map: Dict[int, int] = {}
        for i, p in enumerate(L_k_pos):
            j_label_map[comp_anchors_local_i[i]] = p
        for i, p in enumerate(R_k_pos):
            j_label_map[comp_anchors_local_j[i]] = p
        T_comp_pos = relabel_partition_dict(T_comp, j_label_map)
        junction_orbit_T, junction_orbit_partitions = aut_compress_t_rooted(
            T_comp_pos, [],
        )
        _log(f"  comp {k}: junction T_rooted={len(T_comp_pos)} entries, "
             f"L={len(L_k_pos)}, R={len(R_k_pos)}")

        # state_extra: positions currently open but NOT touched by this
        # component's L_k.
        state_extra = [p for p in state_open_pos if p not in set(L_k_pos)]

        _log(f"  comp {k}: building M_table ({len(state_orbit_partitions)}×{len(junction_orbit_partitions)} orbits)...")
        M = precompute_M_table(
            state_orbit_partitions, junction_orbit_partitions,
            shared_boundary=L_k_pos,
            extra_boundary=R_k_pos,
            out_aut_group=[],
            state_extra_boundary=state_extra,
        )
        _log(f"  comp {k}: M_table size = {len(M)}")
        # Derive output orbits from M_table keys (avoid Bell(|out|) enumeration).
        out_orbits = {key for (_, _, key) in M.keys()}
        if len(out_orbits) > max_intermediate_state_entries:
            _log(f"  comp {k}: out_orbits={len(out_orbits)} > max_intermediate; bailing")
            return None  # sparse-Bell wall
        out_orbit_sizes = {ok: 1 for ok in out_orbits}

        _log(f"  comp {k}: orbit_convolve...")
        state_orbit_T = orbit_convolve(
            state_orbit_T, junction_orbit_T, M, out_orbit_sizes,
        )
        _log(f"  comp {k}: new state size = {len(state_orbit_T)}")
        # New state_orbit_partitions: empty aut means each orbit is itself.
        state_orbit_partitions = {ok: [ok] for ok in state_orbit_T.keys()}

        c_J = 1  # one connected component in the per-component template
        total_div += len(L_k_pos) - c_J
        state_open_pos = state_extra + R_k_pos

    # === Final cell-B convolution ===
    # state_open_pos should now match pos_b values (modulo ordering).
    expected_b_pos = set(pos_b.values())
    if set(state_open_pos) != expected_b_pos:
        # Components didn't cover all cell-B anchors (or covered some
        # extras) → cell-B convolution not directly applicable. Bail.
        return None

    _log(f"final: computing T_rooted(cell B, {len(cell_b_anchors)} anchors)...")
    T_cell_b = t_rooted_cached(cell_template, cell_b_anchors)
    _log(f"  cell B T_rooted: {len(T_cell_b)} entries")
    T_cell_b_pos = relabel_partition_dict(T_cell_b, pos_b)
    cell_b_orbit_T, cell_b_orbit_partitions = aut_compress_t_rooted(
        T_cell_b_pos, [],
    )

    shared_b = list(state_open_pos)  # = pos_b values (in some order)
    _log(f"final: building M_b ({len(state_orbit_partitions)}×{len(cell_b_orbit_partitions)} orbits)...")
    M_b = precompute_M_table(
        state_orbit_partitions, cell_b_orbit_partitions,
        shared_boundary=shared_b,
        extra_boundary=[],
        out_aut_group=[],
        state_extra_boundary=[],
    )
    _log(f"final: M_b size = {len(M_b)}")
    out_orbits_b = {key for (_, _, key) in M_b.keys()}
    out_orbit_sizes_b = {ok: 1 for ok in out_orbits_b}
    _log(f"final: orbit_convolve...")
    state_orbit_T = orbit_convolve(
        state_orbit_T, cell_b_orbit_T, M_b, out_orbit_sizes_b,
    )
    _log(f"final: state size = {len(state_orbit_T)}")
    c_cell_b = 1  # cell template is connected
    total_div += len(shared_b) - c_cell_b

    # === Sum over final empty-boundary state ===
    final_poly = TuttePolynomial.zero()
    for _ok, val in state_orbit_T.items():
        final_poly = final_poly + val
    if total_div > 0:
        final_poly = divide_by_x_minus_1_power(final_poly, total_div)
    return final_poly
