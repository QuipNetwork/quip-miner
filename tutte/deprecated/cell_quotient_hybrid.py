"""Cell-quotient HYBRID DP — cycle-close + per-leaf synth for cyclic
cell-quotients.

For graphs with cell-decomposable structure where the cell-quotient
contains cycles (e.g., D-Wave Cm graphs), this module peels one
closing junction at a time using the §4 chord rule and
computes each leaf via a configurable per-leaf synth callable.

The recursion has two paths through `recurse()`:

1. **Path A** (symmetric chord rule): when the closing junction's
   anchors don't share a vertex with any remaining junction's anchors
   (`_junction_symmetric` check), use the standard `C(k, j)` chord
   rule. This is the fast path for K_n / K_{a,b} cells.

2. **Path B** (orbit enumeration): when symmetric pre-check fails,
   enumerate all 2^k matching subsets, group leaves by canonical_key,
   and recurse once per orbit weighted by orbit size. Bridge factors
   are tracked sequentially via `_path_multiplier_fixed` along the
   fixed `present[0..k-1]` walk so Tutte's path-independence holds.
   This handles cross-junction shared anchors (e.g., 2x2 K_3 M_2 grid).

Both paths can dispatch to `compute_corrected_leaf_dp`
(`cell_quotient_tree.py`) at the top level when the closing junction's
removal makes the cell-topology a tree. This bypasses `engine.synthesize`
per leaf using the corrected vertex-identification convolution rule
(see `tutte/research/data/step3_milestone_b_design.md`).

Reuses the chord-rule primitives from
`tutte/graphs/covering.py:apply_kmatching_formula` (`_apply_junction_merge`,
`_is_bridge_junction`). The DIFFERENCE from `apply_kmatching_formula`
is that we BYPASS the P3 gate via `_detect_kmatching_topology_no_p3`
(needed to handle shared-anchor cyclic cell-topologies that the
existing recursion's parallel-edge bookkeeping mishandles).

See plan: `~/.claude/plans/we-are-working-in-magical-forest.md`.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx

from ..graph import Graph, MultiGraph
from ..graphs.covering import (KMatchingJunction, _apply_junction_merge,
                               _is_bridge_junction, _nx_mg_to_mg,
                               detect_kmatching_topology,
                               try_hierarchical_partition)
from ..polynomial import TuttePolynomial
from ..roots.cell_quotient_tree import CellTreeSpec, compute_tree_dp_recursive


def _is_bridge_in_cell_topology(
    cell_topology: nx.Graph, junction: KMatchingJunction,
) -> bool:
    """True iff removing edge (cell_i, cell_j) from cell_topology
    disconnects it. Used to choose chord-rule coefficient family
    (bridge: x, k-1, C(k,j); cycle: C(k,j))."""
    if not cell_topology.has_edge(junction.cell_i, junction.cell_j):
        return False
    test = cell_topology.copy()
    test.remove_edge(junction.cell_i, junction.cell_j)
    return not nx.is_connected(test)


def _pick_closing_junction(
    cell_topology: nx.Graph,
    junctions: List[KMatchingJunction],
) -> int:
    """Return index into ``junctions`` for the next closing edge.

    Heuristic:
    1. Prefer junctions that are CYCLE edges (not bridges) — closing
       a cycle edge produces a tree leaf when only one cycle remains.
    2. Among cycle edges, prefer the one whose closure produces the
       smallest sum of endpoint degrees in the cell-topology
       (smaller leaves, smaller state space).
    3. Tiebreak: lowest (cell_i, cell_j) for determinism + cache hits.
    """
    cycle_indices = [
        i for i, j in enumerate(junctions)
        if not _is_bridge_in_cell_topology(cell_topology, j)
    ]
    candidates = cycle_indices if cycle_indices else list(range(len(junctions)))

    def score(idx: int):
        j = junctions[idx]
        deg = (cell_topology.degree(j.cell_i)
                + cell_topology.degree(j.cell_j))
        return (deg, j.cell_i, j.cell_j)

    return min(candidates, key=score)


def _spec_canonical_key(spec: CellTreeSpec) -> str:
    """Hash a CellTreeSpec for leaf cache. Includes:
    - cell_template canonical key (graph6-equivalent),
    - junction_template canonical key,
    - sorted cell_tree edges,
    - canonicalized cell_anchor_groups (sorted by cell index, then by
      neighbor index, with anchor lists tuple-ified).

    Symmetric leaves (e.g., the same contraction pattern via different
    junction indices) should map to the same key for cache reuse.
    """
    # cell_template + junction_template canonical keys.
    cell_key = spec.cell_template.canonical_key()
    junc_key = spec.junction_template.canonical_key()
    # cell_tree as sorted edge list.
    tree_edges = tuple(sorted(
        (min(u, v), max(u, v)) for u, v in spec.cell_tree.edges()
    ))
    # cell_anchor_groups: nested dict → tuple of (i, tuple of (j, anchor_tuple)).
    anchor_groups_repr = tuple(sorted(
        (i, tuple(sorted(
            (j, tuple(anchors))
            for j, anchors in spec.cell_anchor_groups.get(i, {}).items()
        )))
        for i in spec.cell_tree.nodes()
    ))
    junc_a = tuple(spec.junction_anchors_A)
    junc_b = tuple(spec.junction_anchors_B)
    return repr((cell_key, junc_key, tree_edges, anchor_groups_repr,
                  junc_a, junc_b, spec.root))


def _combine_results_with_chord_rule(
    j_polys: List[TuttePolynomial], k: int, is_bridge: bool,
) -> TuttePolynomial:
    """§4 cycle-close coefficients.

    Bridge case (closing the only path between two cell components):
        T = x · T_0 + (k-1) · T_1 + Σ_{j>=2} C(k, j) · T_j

    Cycle case (closing one edge of a cycle, doesn't disconnect):
        T = Σ_{j=0}^{k} C(k, j) · T_j

    Mirrors the coefficient logic in ``apply_kmatching_formula``
    (`tutte/graphs/covering.py:1611-1634`).
    """
    if len(j_polys) != k + 1:
        raise ValueError(
            f"_combine_results_with_chord_rule expects {k+1} terms; "
            f"got {len(j_polys)}"
        )
    x_poly = TuttePolynomial.x()
    one = TuttePolynomial.one()
    total = TuttePolynomial.zero()
    for j, T_j in enumerate(j_polys):
        if T_j is None or T_j.is_zero():
            continue
        if is_bridge:
            if j == 0:
                coeff = x_poly
            elif j == 1:
                if k - 1 == 0:
                    continue
                coeff = (k - 1) * one
            else:
                coeff = math.comb(k, j) * one
        else:
            coeff = math.comb(k, j) * one
        total = total + coeff * T_j
    return total


def _resolve_edge(
    relabel: Dict[int, int], u: int, v: int,
) -> Tuple[int, int]:
    """Walk the relabel dict to get the current canonical
    representatives of u, v after junction contractions."""
    def _root(x: int) -> int:
        while x in relabel:
            x = relabel[x]
        return x
    return (_root(u), _root(v))


def _nx_mg_to_mg_keep_loops(g: nx.MultiGraph) -> MultiGraph:
    """Like ``_nx_mg_to_mg`` but tracks self-loops separately (the
    original loses them via min/max collapsing). Each self-loop
    contributes a `y` factor in T."""
    edge_counts: Dict[Tuple[int, int], int] = {}
    loop_counts: Dict[int, int] = {}
    for u, v in g.edges():
        if u == v:
            loop_counts[u] = loop_counts.get(u, 0) + 1
        else:
            e = (min(u, v), max(u, v))
            edge_counts[e] = edge_counts.get(e, 0) + 1
    return MultiGraph(
        nodes=frozenset(g.nodes()),
        edge_counts=edge_counts,
        loop_counts=loop_counts,
    )


def _apply_junction_merge_keyed(
    g: nx.MultiGraph,
    junction_edges_keyed: List[Tuple[int, int, int]],
    j: int,
) -> nx.MultiGraph:
    """Like ``_apply_junction_merge_keep_loops`` but operates on
    SPECIFIC edges identified by key (not by vertex pair).

    Each junction edge is `(u, v, key)`. The `key` uniquely identifies
    the edge among all edges in the multigraph (so processing junction
    Y's edge doesn't accidentally remove junction Z's parallel sibling).

    Preserves self-loops from parallel-edge contractions (those at the
    contracted vertex from OTHER edges that happen to also connect u
    and v). Drops exactly the contracted edge itself.

    The first ``j`` keyed edges are contracted (in order); the
    remaining ``k-j`` are deleted.
    """
    new = g.copy()
    # Build a quick lookup from key → (u, v) for the edges we'll process.
    # nx tracks keys so we can target specific edges.
    relabel_local: Dict[int, int] = {}

    def root(x: int) -> int:
        while x in relabel_local:
            x = relabel_local[x]
        return x

    for i, (u_orig, v_orig, key) in enumerate(junction_edges_keyed):
        u = root(u_orig)
        v = root(v_orig)
        if u == v:
            # Edge already collapsed to a self-loop by an earlier
            # contraction; treat as "already processed" — for j-step
            # contraction it would be a self-loop being contracted
            # (no-op for graph; loop disappears). For deletion, also
            # no-op.
            if i < j:
                # Contracting a self-loop just removes it (no merge).
                if new.has_edge(u, u, key=key):
                    new.remove_edge(u, u, key=key)
            else:
                if new.has_edge(u, u, key=key):
                    new.remove_edge(u, u, key=key)
            continue
        if i < j:
            # Contract: remove THIS edge (by key), then merge v into u.
            if new.has_edge(u, v, key=key):
                new.remove_edge(u, v, key=key)
            # Rewire all of v's edges to u.
            if v in new:
                v_edges = list(new.edges(v, keys=True))
                new.remove_node(v)
                for a, b, _ek in v_edges:
                    a2 = u if a == v else a
                    b2 = u if b == v else b
                    # Preserve all rewired edges with their original
                    # keys — including self-loops (parallel siblings
                    # that became (u, u) under contraction).
                    new.add_edge(a2, b2, key=_ek)
                relabel_local[v] = u
        else:
            if new.has_edge(u, v, key=key):
                new.remove_edge(u, v, key=key)
    return new


def _detect_kmatching_topology_no_p3(
    graph: Graph, partition: List, inter_edges: List[Tuple[int, int]],
) -> Optional[List[KMatchingJunction]]:
    """Same as `detect_kmatching_topology` but WITHOUT the P3 gate.

    The standard `detect_kmatching_topology` rejects cyclic
    cell-topologies with shared anchors per its P3 precondition
    (`covering.py:1432-1442`) because the engine's recursive
    `apply_kmatching_formula` has a parallel-edge bookkeeping bug for
    those cases. The hybrid here uses a DIFFERENT recursion (per-leaf
    synth via engine.synthesize) that doesn't have that bug, so we
    bypass P3.

    Returns None if junctions aren't clean k-matchings (per-pair anchor
    uniqueness still required).
    """
    if not inter_edges:
        return []
    node_to_cell: Dict[int, int] = {}
    for i, cell_nodes in enumerate(partition):
        for node in cell_nodes:
            node_to_cell[node] = i
    pair_data: Dict[Tuple[int, int], Tuple[List, set, set]] = {}
    for u, v in inter_edges:
        cu = node_to_cell.get(u)
        cv = node_to_cell.get(v)
        if cu is None or cv is None or cu == cv:
            return None
        if cu < cv:
            ci, cj, ai, aj = cu, cv, u, v
        else:
            ci, cj, ai, aj = cv, cu, v, u
        pair = (ci, cj)
        if pair not in pair_data:
            pair_data[pair] = ([], set(), set())
        edges_list, anchors_i, anchors_j = pair_data[pair]
        edges_list.append((ai, aj))
        anchors_i.add(ai)
        anchors_j.add(aj)
    from ..graphs.covering import _anchors_single_class
    junctions: List[KMatchingJunction] = []
    for (ci, cj), (edges_list, anchors_i, anchors_j) in pair_data.items():
        if len(anchors_i) != len(edges_list) or len(anchors_j) != len(edges_list):
            return None
        if not _anchors_single_class(graph, partition[ci], anchors_i):
            return None
        if not _anchors_single_class(graph, partition[cj], anchors_j):
            return None
        junctions.append(KMatchingJunction(
            cell_i=ci, cell_j=cj,
            edges=edges_list,
            anchors_i=[a for a, _ in edges_list],
            anchors_j=[b for _, b in edges_list],
        ))
    return junctions


def _build_spec_local_no_p3(
    graph: Graph,
    partition: List,
    junctions: List[KMatchingJunction],
    cell_topology: nx.Graph,
) -> Tuple[Optional[CellTreeSpec], Optional[List[Dict[int, int]]]]:
    """Inline `CellTreeSpec` builder using NO-P3 junctions (allows
    shared anchors across junctions in cyclic cell-topologies).

    Mirrors `_build_cell_tree_spec_from_graph` in `__init__.py` but
    starts from junctions already detected via the no-p3 detector,
    so shared-anchor cyclic graphs (2x2 K_3 M_2 grid) build a valid
    spec instead of being rejected.
    """
    from networkx.algorithms.isomorphism import GraphMatcher

    if not junctions:
        return None, None

    cell0_nodes = sorted(partition[0])
    relabel0 = {v: i for i, v in enumerate(cell0_nodes)}
    cell0_edges = [
        (relabel0[u], relabel0[v]) for u, v in graph.edges
        if u in relabel0 and v in relabel0
    ]
    cell_template = Graph(
        list(range(len(cell0_nodes))), cell0_edges,
    )
    cell_template_nx = nx.Graph()
    cell_template_nx.add_nodes_from(range(len(cell0_nodes)))
    cell_template_nx.add_edges_from(cell0_edges)

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
            return None, None
        isos.append(matcher.mapping)

    cell_anchor_groups: Dict[int, Dict[int, List[int]]] = {
        i: {} for i in range(len(partition))
    }
    junction_k = junctions[0].k
    for j in junctions:
        if j.k != junction_k:
            return None, None
        anchors_i_local = [isos[j.cell_i][a] for a in j.anchors_i]
        anchors_j_local = [isos[j.cell_j][a] for a in j.anchors_j]
        if j.cell_j in cell_anchor_groups[j.cell_i]:
            return None, None
        cell_anchor_groups[j.cell_i][j.cell_j] = anchors_i_local
        cell_anchor_groups[j.cell_j][j.cell_i] = anchors_j_local

    junction_template = Graph(
        list(range(2 * junction_k)),
        [(i, i + junction_k) for i in range(junction_k)],
    )
    junction_anchors_A = list(range(junction_k))
    junction_anchors_B = list(range(junction_k, 2 * junction_k))

    leaf_candidates = [n for n, d in cell_topology.degree() if d == 1]
    root = leaf_candidates[0] if leaf_candidates else 0

    spec = CellTreeSpec(
        cell_template=cell_template,
        junction_template=junction_template,
        cell_tree=cell_topology.copy(),
        cell_anchor_groups=cell_anchor_groups,
        junction_anchors_A=junction_anchors_A,
        junction_anchors_B=junction_anchors_B,
        root=root,
    )
    return spec, isos


def compute_cell_quotient_hybrid(
    graph: Graph, table, synth_fn: Optional[Callable] = None,
) -> "Optional[TuttePolynomial]":
    """Engine entry: cyclic cell-quotient via cycle-close + per-leaf synth.

    Detects whether `graph` decomposes hierarchically into K_n cells
    joined by k-matching junctions where the cell-quotient has CYCLES
    (so the cycle DP and tree DP don't apply). Recursively peels
    closing junctions via chord rule, computing each leaf
    via `synth_fn` (defaults to `engine.synthesize` for correctness).

    Returns None if:
    - hierarchical partition fails,
    - junctions aren't k-matchings,
    - cell-topology is a tree (use tree DP for that),
    - building inputs fails for any other reason.
    """
    result = try_hierarchical_partition(graph, table)
    if result is None:
        return None
    _cell_entry, partition, inter_info = result
    if not partition or not inter_info.edges:
        return None
    # Gate large-cell-count graphs with large cells. The orbit explosion in
    # cycle/tree DPs scales with both n_cells AND cell-size (boundary). 3x3
    # K_3 grids (9 cells × 3 nodes) stay tractable; Cm3 (9 cells × 8-node
    # K_{4,4} cells) walls. Use graph size as a proxy for cell complexity.
    if len(partition) > 6 and graph.node_count() > 36:
        return None
    junctions = _detect_kmatching_topology_no_p3(
        graph, partition, list(inter_info.edges),
    )
    if junctions is None or not junctions:
        return None
    cell_topology = nx.Graph()
    cell_topology.add_nodes_from(range(len(partition)))
    for j in junctions:
        cell_topology.add_edge(j.cell_i, j.cell_j)
    if nx.is_tree(cell_topology):
        return None  # use tree DP

    # Plan B (DEAD CODE — see plan note): the spec-leaf dispatch path
    # below is gated to never fire correctly for j>0 cases due to the
    # `(xy-1)^d` architectural blocker in tree DP's cross-cell-ID
    # merge step (see `step3_milestone_b_design.md` Step 3.B.3).
    # Cross-cell-id positions that aren't already in cell_anchor_groups
    # don't get allocated by `_allocate_tree_positions` (line 354), so
    # any `cross_cell_identifications` referencing the closing junction
    # anchors are silently ignored — tree DP returns the j=0 leaf
    # polynomial for ALL j values, which is wrong.
    #
    # Building the spec via `_build_cell_tree_spec_from_graph` (which
    # calls the strict `detect_kmatching_topology`) means initial_spec
    # is None for shared-anchor cyclic graphs anyway. For symmetric
    # cyclic graphs (Cm₂, K_n cycles), Path A handles them faster.
    # The block stays as a hook for future revival once the blocker is
    # resolved.
    initial_spec = None
    isos: Optional[List[Dict[int, int]]] = None
    try:
        initial_spec, isos = _build_spec_local_no_p3(
            graph, partition, junctions, cell_topology,
        )
    except Exception:
        initial_spec = None
        isos = None
    if synth_fn is None:
        from ..synthesis.engine import SynthesisEngine
        engine = SynthesisEngine(table=table)

        def _default_synth(mg: MultiGraph) -> TuttePolynomial:
            # _synthesize_multigraph correctly handles parallel edges
            # (which arise from junction contractions). Falling back to
            # `engine.synthesize(simple_graph)` would lose them and give
            # a wrong polynomial.
            return engine._synthesize_multigraph(mg)
        synth_fn = _default_synth

    # Build initial nx.MultiGraph with explicit keys so we can
    # uniquely identify each edge through contractions (essential
    # when junctions' edges become parallel after contractions).
    g_nx = nx.MultiGraph()
    g_nx.add_nodes_from(graph.nodes)
    next_key = [0]

    def _add_keyed_edge(u: int, v: int) -> int:
        k = next_key[0]
        next_key[0] += 1
        g_nx.add_edge(u, v, key=k)
        return k

    junction_edge_keys: Dict[Tuple[int, int, int], int] = {}
    junction_edges_keyed: List[List[Tuple[int, int, int]]] = []
    for j_idx, junc in enumerate(junctions):
        keyed: List[Tuple[int, int, int]] = []
        for (u, v) in junc.edges:
            k = _add_keyed_edge(u, v)
            keyed.append((u, v, k))
            junction_edge_keys[(j_idx, u, v)] = k
        junction_edges_keyed.append(keyed)
    # All non-junction edges (cell-internal) also need keys for
    # consistent tracking through contractions.
    junction_edge_set = {
        (min(u, v), max(u, v))
        for junc in junctions for (u, v) in junc.edges
    }
    for u, v in graph.edges:
        if (min(u, v), max(u, v)) not in junction_edge_set:
            _add_keyed_edge(u, v)
    # Single recurse cache keyed only on multigraph canonical_key.
    # T(graph) is independent of edge ordering (Tutte's theorem), so the
    # `remaining_idx` and `symmetric_ok` flags do NOT affect the
    # polynomial value — they only steer which recursive path is taken.
    # Engine's _multigraph_cache is the global tier (populated through
    # synth_fn calls), so this local cache is just an in-call shortcut.
    recurse_cache: Dict[str, TuttePolynomial] = {}

    def _t_leaf(mg_nx: nx.MultiGraph) -> TuttePolynomial:
        mg = _nx_mg_to_mg_keep_loops(mg_nx)
        try:
            key = mg.canonical_key()
        except Exception:
            return synth_fn(mg)
        cached = recurse_cache.get(key)
        if cached is not None:
            return cached
        T = synth_fn(mg)
        recurse_cache[key] = T
        return T

    def _canonical_key(g: nx.MultiGraph) -> Optional[str]:
        try:
            return _nx_mg_to_mg_keep_loops(g).canonical_key()
        except Exception:
            return None

    def _try_tree_dp_shortcut(g: nx.MultiGraph) -> Optional[TuttePolynomial]:
        """If g is a simple graph with tree-quotient cell decomposition,
        dispatch to compute_cell_quotient_tree_dp directly. Returns None
        if g has parallel edges, self-loops, or doesn't fit tree DP."""
        # Check for self-loops or parallel edges.
        for u, v in g.edges():
            if u == v:
                return None
        edge_set = set()
        for u, v in g.edges():
            e = (min(u, v), max(u, v))
            if e in edge_set:
                return None
            edge_set.add(e)
        from .. import roots as _roots_mod
        simple = Graph(
            nodes=frozenset(g.nodes()),
            edges=frozenset(edge_set),
        )
        try:
            return _roots_mod.compute_cell_quotient_tree_dp(simple, table)
        except Exception:
            return None

    def _try_spec_leaf_dispatch(
        junc_idx: int,
        S_set: set,
    ) -> Optional[TuttePolynomial]:
        """Plan B: build a `CellTreeSpec` for the chord-rule leaf at
        the TOP-LEVEL closing junction with operations defined by
        `S_set` (contracted indices) on the matching `present`. Edges
        not in S_set are deleted. The leaf spec has:

        - cell_tree minus the closing junction edge.
        - cell_anchor_groups minus the closing junction's entries.
        - cross_cell_identifications for the j contracted edges
          (template-local anchor pairs).
        - extra_open_anchors for the (k-j) deleted edges
          (template-local labels).

        If the leaf cell_tree is a TREE (cycle closed by removing one
        edge), dispatch to `compute_tree_dp_recursive`. Otherwise
        return None to fall back to recursion.
        """
        if initial_spec is None or isos is None:
            return None
        from ..roots.cell_quotient_tree import CellTreeSpec, compute_corrected_leaf_dp
        junc = junctions[junc_idx]
        cell_i = junc.cell_i
        cell_j = junc.cell_j
        # New cell_tree without closing edge.
        new_cell_tree = initial_spec.cell_tree.copy()
        if new_cell_tree.has_edge(cell_i, cell_j):
            new_cell_tree.remove_edge(cell_i, cell_j)
        # If cell_tree is still cyclic (multi-cycle quotient), tree DP
        # cannot dispatch. Caller falls back.
        if not nx.is_tree(new_cell_tree):
            return None
        # Drop the closing junction's entries from cell_anchor_groups.
        new_cell_anchor_groups = {
            i: dict(groups)
            for i, groups in initial_spec.cell_anchor_groups.items()
        }
        if cell_j in new_cell_anchor_groups.get(cell_i, {}):
            new_cell_anchor_groups[cell_i].pop(cell_j)
        if cell_i in new_cell_anchor_groups.get(cell_j, {}):
            new_cell_anchor_groups[cell_j].pop(cell_i)
        # Build cross_cell_identifications and extra_open_anchors from
        # the closing junction's k matching edges.
        new_cross_ids = list(initial_spec.cross_cell_identifications)
        new_extras: Dict[int, List[int]] = {
            i: list(extras)
            for i, extras in initial_spec.extra_open_anchors.items()
        }
        for i, edge in enumerate(list(junc.edges)):
            # Edge may be (u, v) or (u, v, key); we only need the endpoints.
            u, v = edge[0], edge[1]
            # `u, v` are vertices in the raw graph. Translate to the
            # template-local labels via `isos`. Both endpoints could be
            # in either cell_i or cell_j depending on edge orientation.
            if u in isos[cell_i] and v in isos[cell_j]:
                tu = isos[cell_i][u]
                tv = isos[cell_j][v]
            elif v in isos[cell_i] and u in isos[cell_j]:
                tu = isos[cell_i][v]
                tv = isos[cell_j][u]
            else:
                # Edge endpoints not in expected cells (shouldn't happen
                # for a clean junction, but be defensive).
                return None
            if i in S_set:
                new_cross_ids.append((cell_i, tu, cell_j, tv))
            else:
                new_extras.setdefault(cell_i, []).append(tu)
                new_extras.setdefault(cell_j, []).append(tv)
        # Pick a leaf cell as root for the new tree.
        leaf_candidates = [n for n, d in new_cell_tree.degree() if d == 1]
        new_root = (leaf_candidates[0] if leaf_candidates
                     else initial_spec.root)
        leaf_spec = CellTreeSpec(
            cell_template=initial_spec.cell_template,
            junction_template=initial_spec.junction_template,
            cell_tree=new_cell_tree,
            cell_anchor_groups=new_cell_anchor_groups,
            junction_anchors_A=initial_spec.junction_anchors_A,
            junction_anchors_B=initial_spec.junction_anchors_B,
            root=new_root,
            cross_cell_identifications=new_cross_ids,
            extra_open_anchors=new_extras,
        )
        try:
            return compute_corrected_leaf_dp(leaf_spec)
        except Exception:
            return None

    def _build_child_relabel(
        relabel: Dict[int, int],
        ordered: List[Tuple[int, int, int]],
        j: int,
    ) -> Dict[int, int]:
        """Construct child relabel after contracting first j edges of
        ``ordered``. Mirrors the union-find logic of
        ``_apply_junction_merge_keyed`` so that downstream
        ``_resolve_edge`` calls pick up the new vertex identifications."""
        cr = dict(relabel)
        for i, (u, v, _k) in enumerate(ordered):
            if i >= j:
                break
            ru, rv = _resolve_edge(cr, u, v)
            if ru == rv:
                continue
            if ru < rv:
                cr[rv] = ru
            else:
                cr[ru] = rv
        return cr

    def _apply_subset_fixed_order(
        g: nx.MultiGraph,
        edges: List[Tuple[int, int, int]],
        S_set: set,
    ) -> Tuple[nx.MultiGraph, Dict[int, int]]:
        """Apply (contract / delete) operations on ``edges`` in FIXED
        ORDER (``edges[0], edges[1], ..., edges[k-1]``). Index ``i`` is
        contracted iff ``i in S_set``; otherwise deleted.

        This matches Tutte's fixed-order del-contract recursion:
        each subset S corresponds to exactly one path through the
        binary tree, so the multiplier (computed by the same fixed
        walk in ``_path_multiplier_fixed``) and the leaf graph stay
        in lock-step.

        Returns ``(g_after, relabel_added)`` where ``relabel_added``
        maps absorbed vertices to their new (canonical) representative
        — the contractions made in this call only.
        """
        g_cur = g.copy()
        relabel_local: Dict[int, int] = {}

        def root(x: int) -> int:
            while x in relabel_local:
                x = relabel_local[x]
            return x

        for i, (u_orig, v_orig, ek) in enumerate(edges):
            ru = root(u_orig)
            rv = root(v_orig)
            op_is_contract = (i in S_set)
            if ru == rv:
                if g_cur.has_edge(ru, ru, key=ek):
                    g_cur.remove_edge(ru, ru, key=ek)
                continue
            if not g_cur.has_edge(ru, rv, key=ek):
                continue
            if op_is_contract:
                g_cur.remove_edge(ru, rv, key=ek)
                if rv in g_cur:
                    v_edges = list(g_cur.edges(rv, keys=True))
                    g_cur.remove_node(rv)
                    for a, b, _ek in v_edges:
                        a2 = ru if a == rv else a
                        b2 = ru if b == rv else b
                        g_cur.add_edge(a2, b2, key=_ek)
                    relabel_local[rv] = ru
            else:
                g_cur.remove_edge(ru, rv, key=ek)
        return g_cur, relabel_local

    def _path_multiplier_fixed(
        g: nx.MultiGraph,
        edges: List[Tuple[int, int, int]],
        S_set: set,
    ) -> TuttePolynomial:
        """Sequential del-contract multiplier for the FIXED-ORDER walk
        of ``edges`` with ``S_set`` indicating contracted indices.

        Mirrors ``_apply_subset_fixed_order``'s operation order so the
        bridge factors and resulting leaf graph stay in lock-step.
        Returns 0 if the path is invalid (delete a bridge).
        """
        x_poly = TuttePolynomial.x()
        factor = TuttePolynomial.one()
        g_cur = g.copy()
        relabel_local: Dict[int, int] = {}

        def root(x: int) -> int:
            while x in relabel_local:
                x = relabel_local[x]
            return x

        for i, (u_orig, v_orig, ek) in enumerate(edges):
            ru = root(u_orig)
            rv = root(v_orig)
            op_is_contract = (i in S_set)
            if ru == rv:
                if g_cur.has_edge(ru, ru, key=ek):
                    g_cur.remove_edge(ru, ru, key=ek)
                continue
            if not g_cur.has_edge(ru, rv, key=ek):
                continue
            g_minus = g_cur.copy()
            g_minus.remove_edge(ru, rv, key=ek)
            edge_is_bridge = (
                ru not in g_minus
                or rv not in g_minus
                or not nx.has_path(g_minus, ru, rv)
            )
            if edge_is_bridge:
                if op_is_contract:
                    factor = factor * x_poly
                else:
                    return TuttePolynomial.zero()
            if op_is_contract:
                g_cur.remove_edge(ru, rv, key=ek)
                if rv in g_cur:
                    v_edges = list(g_cur.edges(rv, keys=True))
                    g_cur.remove_node(rv)
                    for a, b, _ek in v_edges:
                        a2 = ru if a == rv else a
                        b2 = ru if b == rv else b
                        g_cur.add_edge(a2, b2, key=_ek)
                    relabel_local[rv] = ru
            else:
                g_cur.remove_edge(ru, rv, key=ek)
        return factor

    def _build_child_relabel_fixed(
        relabel: Dict[int, int],
        edges: List[Tuple[int, int, int]],
        S_set: set,
    ) -> Dict[int, int]:
        """Build child relabel for FIXED-ORDER walk with S_set
        indicating contracted indices."""
        cr = dict(relabel)
        for i, (u, v, _k) in enumerate(edges):
            if i not in S_set:
                continue
            ru, rv = _resolve_edge(cr, u, v)
            if ru == rv:
                continue
            if ru < rv:
                cr[rv] = ru
            else:
                cr[ru] = rv
        return cr

    def _path_multiplier(
        g: nx.MultiGraph,
        ordered: List[Tuple[int, int, int]],
        j: int,
    ) -> TuttePolynomial:
        """Sequential del-contract multiplier for the operation pattern
        encoded by ``ordered`` (k matching edges in processing order)
        and ``j`` (number of CONTRACTED edges, taken from the front of
        ``ordered``; remaining ``k-j`` are DELETED).

        Walks ``ordered`` in the GIVEN order (matching
        ``_apply_junction_merge_keyed``'s contract-first convention).
        This consistency is critical: the bridge classifications and
        the leaf graph state must be derived from the same operation
        sequence so that ``mult * T(leaf)`` is a well-defined term in
        the Tutte recursion (different orderings give valid but
        DIFFERENT path decompositions).

        At each step:
        - CHORD: factor unchanged.
        - BRIDGE + contract: factor *= x.
        - BRIDGE + delete: path INVALID (Tutte's bridge case has no
          delete branch). Return 0.
        - self-loop (collapsed by prior op): drop the loop edge (its
          y factor is bookkept at the leaf via ``_t_leaf``).
        """
        x_poly = TuttePolynomial.x()
        factor = TuttePolynomial.one()
        g_cur = g.copy()
        for i, (u_orig, v_orig, ek) in enumerate(ordered):
            op_is_contract = (i < j)
            # Resolve to current vertex labels (prior ops applied).
            ru, rv = u_orig, v_orig
            if ru not in g_cur or rv not in g_cur:
                # Vertex contracted away by a prior operation; look up
                # the edge by key in the current graph.
                found = None
                for a, b, k in g_cur.edges(keys=True):
                    if k == ek:
                        found = (a, b)
                        break
                if found is None:
                    continue
                ru, rv = found
            if ru == rv:
                if g_cur.has_edge(ru, ru, key=ek):
                    g_cur.remove_edge(ru, ru, key=ek)
                continue
            if not g_cur.has_edge(ru, rv, key=ek):
                continue
            # Per-edge bridge check: remove this specific keyed edge
            # and see if endpoints are still connected.
            g_minus = g_cur.copy()
            g_minus.remove_edge(ru, rv, key=ek)
            edge_is_bridge = (
                ru not in g_minus
                or rv not in g_minus
                or not nx.has_path(g_minus, ru, rv)
            )
            if edge_is_bridge:
                if op_is_contract:
                    factor = factor * x_poly
                else:
                    return TuttePolynomial.zero()
            # Apply chosen operation to g_cur for next step.
            if op_is_contract:
                g_cur = _apply_junction_merge_keyed(
                    g_cur, [(ru, rv, ek)], 1,
                )
            else:
                g_cur.remove_edge(ru, rv, key=ek)
        return factor

    def _junction_symmetric(
        present: List[Tuple[int, int, int]],
        other_idx: Tuple[int, ...],
        relabel: Dict[int, int],
    ) -> bool:
        """Chord-rule shortcut valid iff:
        (a) no present-edge endpoint is a "merged" vertex (target of a
            prior contraction), AND
        (b) no remaining-junction anchor overlaps any present-edge
            endpoint at this state.
        Together these ensure the cell template's full automorphism
        group is transitive on this junction's anchors AND the
        contraction patterns from this junction don't leak into
        downstream junctions' anchor structure.
        """
        # Collect present endpoints (resolved through current relabel).
        endpoints: set = set()
        for u, v, _k in present:
            ru, rv = _resolve_edge(relabel, u, v)
            endpoints.add(ru)
            endpoints.add(rv)
        # (a) Prior contractions create "merged" vertices = relabel
        # values (the surviving vertex of a contraction). If a
        # present endpoint has absorbed others, the cell template's
        # automorphism stabilizer at this anchor is reduced.
        merged_vertices = set(relabel.values())
        for v in endpoints:
            if v in merged_vertices:
                return False
        # (b) Remaining junctions sharing an anchor with present
        # break Aut transitivity for downstream junctions.
        for o_idx in other_idx:
            for ou, ov, _ek in junction_edges_keyed[o_idx]:
                rou, rov = _resolve_edge(relabel, ou, ov)
                if rou in endpoints or rov in endpoints:
                    return False
        return True

    def _score_junction(
        idx: int,
        cell_topology_local: nx.Graph,
        present_count: int,
    ) -> int:
        """Order junctions to maximize closed-form-leaf hit rate.
        Bridge in cell-topology gets the highest weight (closure
        factorizes); cycle-rank reduction (does removing this junction
        leave a tree?) gets the next highest; smaller k breaks ties
        toward fewer sub-cases."""
        score = 0
        j = junctions[idx]
        if cell_topology_local.has_edge(j.cell_i, j.cell_j):
            cell_top_after = cell_topology_local.copy()
            cell_top_after.remove_edge(j.cell_i, j.cell_j)
            if not nx.is_connected(cell_top_after):
                # Bridge in cell-topology: closure produces tree leaves.
                score += 1000
            elif nx.is_forest(cell_top_after):
                # Cycle-rank reduction: removing this edge yields a tree.
                score += 500
        score -= present_count * 10
        return score

    def recurse(
        g: nx.MultiGraph,
        remaining_idx: Tuple[int, ...],
        relabel: Dict[int, int],
    ) -> TuttePolynomial:
        """Recursive expansion. For the closing junction at this level:
        - If the matching is symmetric around remaining junctions, use
          the existing C(k, j) chord-rule formula (fast path).
        - Otherwise, enumerate the 2^k matching subsets and group by
          leaf canonical_key (orbit consolidation). Path multipliers
          carry sequential bridge factors; orbit-size multiplicities
          collapse symmetric inputs back to k+1 orbits automatically.
        """
        if not remaining_idx:
            return _t_leaf(g)
        cache_key = _canonical_key(g)
        if cache_key is not None:
            cached = recurse_cache.get(cache_key)
            if cached is not None:
                return cached
        # Tree-quotient short-circuit: dispatch to the tree DP utility
        # when the current state is a clean simple graph with
        # tree-quotient cell decomposition.
        tree_result = _try_tree_dp_shortcut(g)
        if tree_result is not None:
            if cache_key is not None:
                recurse_cache[cache_key] = tree_result
            return tree_result

        def _present_for(idx: int) -> List[Tuple[int, int, int]]:
            resolved: List[Tuple[int, int, int]] = []
            for u, v, k in junction_edges_keyed[idx]:
                ru, rv = _resolve_edge(relabel, u, v)
                if ru == rv:
                    continue
                if g.has_edge(ru, rv, key=k):
                    resolved.append((ru, rv, k))
            return resolved

        # Build a dynamic cell-topology reflecting which cells are
        # currently linked by remaining junctions (in the multigraph).
        # Empty `present` for an index means that junction has been
        # fully consumed by prior operations; skip it.
        cell_topology_local = nx.Graph()
        cell_topology_local.add_nodes_from(range(len(junctions)))
        for idx in remaining_idx:
            if _present_for(idx):
                cell_topology_local.add_edge(
                    junctions[idx].cell_i, junctions[idx].cell_j,
                )

        # Cycle-rank-aware junction picker.
        live = [(idx, _present_for(idx)) for idx in remaining_idx]
        live_with_edges = [(idx, p) for idx, p in live if p]
        if not live_with_edges:
            # All remaining junctions are vacuous (edges contracted/deleted
            # away by prior operations). Drop them and recurse.
            result = _t_leaf(g)
            if cache_key is not None:
                recurse_cache[cache_key] = result
            return result
        chosen_idx, present = max(
            live_with_edges,
            key=lambda ip: _score_junction(
                ip[0], cell_topology_local, len(ip[1]),
            ),
        )
        other_idx = tuple(i for i in remaining_idx if i != chosen_idx)

        k_eff = len(present)
        edges_only = [(u, v) for u, v, _k in present]
        is_bridge = _is_bridge_junction(g, edges_only)
        x_poly = TuttePolynomial.x()
        one = TuttePolynomial.one()

        # Path A — symmetric pre-check + standard chord rule. The
        # `_junction_symmetric` check now covers BOTH prior-contraction
        # asymmetry (merged endpoints) AND remaining-junction anchor
        # overlap, so we no longer need the conservative
        # `relabel == {}` gate. `other_idx` non-empty is still
        # required because last-junction needs full enumeration to
        # cover any subset asymmetry not visible to the check.
        if other_idx and _junction_symmetric(present, other_idx, relabel):
            # Try spec dispatch for each j-class representative; if all
            # succeed, we skip recursing into the engine entirely.
            #
            # Size gate: brute-force `compute_corrected_leaf_dp` scales
            # as Bell(boundary)^2 per merge. For K_{4,4} cells with
            # full 8-position boundary, this is ~4140^2 = 17M pairs per
            # merge with growing polynomials — orders of magnitude
            # slower than letting `engine._synthesize_multigraph` handle
            # the leaf via treewidth DP. Disable spec dispatch when
            # cells are too large for the brute-force leaf DP.
            use_spec_in_path_a = False
            spec_test_tree = None
            if (relabel == {}
                    and len(remaining_idx) == len(junctions)
                    and initial_spec is not None and isos is not None):
                spec_test_tree = initial_spec.cell_tree.copy()
                junc_obj_a = junctions[chosen_idx]
                if spec_test_tree.has_edge(junc_obj_a.cell_i, junc_obj_a.cell_j):
                    spec_test_tree.remove_edge(
                        junc_obj_a.cell_i, junc_obj_a.cell_j,
                    )
                tree_ok = nx.is_tree(spec_test_tree)
                # Per-cell boundary size in the leaf spec = number of
                # UNIQUE template anchor positions touched by either:
                #   (a) remaining junctions (cell_anchor_groups minus closing)
                #   (b) closing junction (re-added as extras + cross-cell-IDs)
                # The closing junction's anchors don't increase boundary
                # because they're already in cell_anchor_groups; they just
                # move from one entry to "extras". Cap total boundary per
                # cell at 6 (Bell(6)=203, manageable); above that
                # brute-force DP is slower than engine.
                if tree_ok:
                    max_cell_boundary = 0
                    for cell_idx, groups in initial_spec.cell_anchor_groups.items():
                        unique_anchors = set()
                        for nbr, anchors in groups.items():
                            unique_anchors.update(anchors)
                        cell_b = len(unique_anchors)
                        if cell_b > max_cell_boundary:
                            max_cell_boundary = cell_b
                    SPEC_DISPATCH_BOUNDARY_CAP = 6
                    use_spec_in_path_a = (
                        max_cell_boundary <= SPEC_DISPATCH_BOUNDARY_CAP
                    )

            total = TuttePolynomial.zero()
            for j in range(0, k_eff + 1):
                T_j = None
                if use_spec_in_path_a:
                    # Symmetric: pick the lex-smallest size-j subset as
                    # representative.
                    S_rep = set(range(j))
                    T_j = _try_spec_leaf_dispatch(chosen_idx, S_rep)
                if T_j is None:
                    g_j = _apply_junction_merge_keyed(g, present, j)
                    child_relabel = _build_child_relabel(relabel, present, j)
                    T_j = recurse(g_j, other_idx, child_relabel)
                if is_bridge:
                    if k_eff == 1:
                        # Use contract path (T = x * T_1) — standard
                        # T = x * T_0 form diverges in chord-rule
                        # recursion; cut-vertex factorization at the
                        # leaf only equates them at depth 0.
                        if j == 1:
                            coeff_poly = x_poly
                        else:
                            continue
                    else:
                        if j == 0:
                            coeff_poly = x_poly
                        elif j == 1:
                            coeff = k_eff - 1
                            if coeff == 0:
                                continue
                            coeff_poly = coeff * one
                        else:
                            coeff_poly = math.comb(k_eff, j) * one
                else:
                    coeff_poly = math.comb(k_eff, j) * one
                total = total + coeff_poly * T_j
            if cache_key is not None:
                recurse_cache[cache_key] = total
            return total

        # Plan B: TOP-LEVEL spec dispatch when removing the closing
        # junction makes the cell-topology a tree (single-cycle input).
        # For each chord-rule sub-case, build a `CellTreeSpec` with
        # `cross_cell_identifications` (j contracted edges) and
        # `extra_open_anchors` (k-j deleted edges), then dispatch to
        # tree DP. Bypasses the slow per-leaf engine call entirely
        # when the leaf cell-tree is acyclic.
        from itertools import combinations
        SPEC_DISPATCH_BOUNDARY_CAP_B = 6
        spec_dispatch_eligible_b = False
        if (relabel == {}
                and len(remaining_idx) == len(junctions)
                and initial_spec is not None and isos is not None):
            test_tree = initial_spec.cell_tree.copy()
            junc_obj = junctions[chosen_idx]
            if test_tree.has_edge(junc_obj.cell_i, junc_obj.cell_j):
                test_tree.remove_edge(junc_obj.cell_i, junc_obj.cell_j)
            tree_ok_b = nx.is_tree(test_tree)
            if tree_ok_b:
                max_cell_b = 0
                for cell_idx, groups in initial_spec.cell_anchor_groups.items():
                    unique_anchors = set()
                    for nbr, anchors in groups.items():
                        unique_anchors.update(anchors)
                    cell_b = len(unique_anchors)
                    if cell_b > max_cell_b:
                        max_cell_b = cell_b
                spec_dispatch_eligible_b = (
                    max_cell_b <= SPEC_DISPATCH_BOUNDARY_CAP_B
                )
        if spec_dispatch_eligible_b:
                spec_total = TuttePolynomial.zero()
                spec_ok = True
                for j in range(0, k_eff + 1):
                    if not spec_ok:
                        break
                    for S_idx in combinations(range(k_eff), j):
                        S_set = set(S_idx)
                        mult = _path_multiplier_fixed(g, present, S_set)
                        if mult.is_zero():
                            continue
                        T_leaf = _try_spec_leaf_dispatch(chosen_idx, S_set)
                        if T_leaf is None:
                            spec_ok = False
                            break
                        spec_total = spec_total + mult * T_leaf
                if spec_ok:
                    if cache_key is not None:
                        recurse_cache[cache_key] = spec_total
                    return spec_total

        # Path B — orbit enumeration for asymmetric junctions.
        # Walk the 2^k matching subsets; for each, build the leaf
        # multigraph and group by canonical_key (orbit consolidation).
        # Each orbit's accumulated multiplier sums all path multipliers
        # of equivalent leaves, so symmetric inputs that happen to fall
        # through this branch still collapse correctly.
        # orbits maps canonical_key -> (orbit_multiplier_polynomial,
        #                                representative_g_leaf,
        #                                representative_child_relabel)
        orbits: Dict[str, Tuple[TuttePolynomial, nx.MultiGraph, Dict[int, int]]] = {}
        unkeyed_total = TuttePolynomial.zero()
        for j in range(0, k_eff + 1):
            for S_idx in combinations(range(k_eff), j):
                S_set = set(S_idx)
                # FIXED-ORDER del-contract walk: this matches Tutte's
                # binary-tree expansion where each subset S is reached
                # by a unique path through the tree (one branch
                # per edge in fixed order). The multiplier and leaf
                # graph use the SAME walk (`_path_multiplier_fixed`
                # and `_apply_subset_fixed_order` mirror each other),
                # ensuring `mult * T(leaf)` is a valid Tutte term.
                mult = _path_multiplier_fixed(g, present, S_set)
                if mult.is_zero():
                    continue
                g_S, _local_rel = _apply_subset_fixed_order(
                    g, present, S_set,
                )
                child_relabel = _build_child_relabel_fixed(
                    relabel, present, S_set,
                )
                leaf_key = _canonical_key(g_S)
                if leaf_key is None:
                    T_S = recurse(g_S, other_idx, child_relabel)
                    unkeyed_total = unkeyed_total + mult * T_S
                    continue
                if leaf_key in orbits:
                    cached_mult, cached_g, cached_rel = orbits[leaf_key]
                    orbits[leaf_key] = (
                        cached_mult + mult, cached_g, cached_rel,
                    )
                else:
                    orbits[leaf_key] = (mult, g_S, child_relabel)
        total = unkeyed_total
        for leaf_key, (mult, g_rep, child_rel) in orbits.items():
            T_rep = recurse(g_rep, other_idx, child_rel)
            total = total + mult * T_rep
        if cache_key is not None:
            recurse_cache[cache_key] = total
        return total

    all_idx = tuple(range(len(junctions)))
    try:
        return recurse(g_nx, all_idx, {})
    except Exception:
        return None
