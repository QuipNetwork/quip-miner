"""Sokal Z generalized chord-junction theorem.

Extends the matching-only unified chord-junction theorem (see
`tutte/roots/chord_junction_closed_form.py`) to **arbitrary** chord
junctions including multi-edge and non-matching bipartite structures.

Theorem (empirically validated, see
`tutte/research/cyclotomic_chord_junction_theorem.md` § "Sokal-Z
Generalized Chord-Junction Theorem"):

  Z(G_1 ⊕_{E_J} G_2; q, v) = Σ_{A_J ⊆ E_J} v^{|A_J|} · Z(G_1 ∪_{φ(A_J)} G_2; q, v)

where φ(A_J) is the equivalence relation on V_k^A ∪ V_k^B induced by
connected components of (V_k^A ∪ V_k^B, A_J).

This module:

1. Computes T(G_1 ⊕_{E_J} G_2; x, y) via the Sokal-Z formula evaluated
   at sufficiently many (x, y) points + Lagrange interpolation back to
   polynomial form.
2. For each evaluation point, sums contributions over A_J subsets. The
   current implementation enumerates A_J directly (2^|E_J| terms);
   tree-DP-over-H_J enumeration is future work for tractability on
   larger |E_J|.
3. Caches T(merger) values per unique merger graph (orbit-compressed).

Cost: O(2^|E_J| · evaluation_points · per_merger_synth). Tractable for
|E_J| ≤ 16 with current prototype. Tree-DP unlocks |E_J| ≤ 32 (Z(1, 2)).

Specialization: when E_J is a perfect matching on V_k, the φ partitions
collapse to V_T subsets and the formula reduces to the original unified
chord-junction theorem after Z → T conversion (see research doc).
"""
from __future__ import annotations

from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx

from ..graph import Graph, MultiGraph
from ..polynomial import TuttePolynomial


def _components_of(
    edges: List[Tuple[int, int]], vertices: List[int],
) -> List[FrozenSet[int]]:
    """Return connected components as a list of frozensets of vertices."""
    if not edges:
        return [frozenset({v}) for v in vertices]
    B = nx.Graph()
    B.add_nodes_from(vertices)
    for (u, v) in edges:
        B.add_edge(u, v)
    return [frozenset(c) for c in nx.connected_components(B)]


def _build_merger_graph(
    cell_A: Graph, cell_B: Graph,
    cell_A_label_offset: int, cell_B_label_offset: int,
    phi_classes: List[FrozenSet[int]],
) -> MultiGraph:
    """Build G_1 ⊔ G_2 with vertices identified per φ classes.

    Args:
        cell_A, cell_B: cell graphs (vertex labels separate, no overlap
            with the offsets).
        cell_A_label_offset, cell_B_label_offset: label offsets so
            cells have disjoint global vertex IDs.
        phi_classes: list of frozensets, each a class of vertex IDs to
            identify into a single vertex.
    """
    # Build per-vertex representative
    rep: Dict[int, int] = {}
    for c in phi_classes:
        sorted_c = sorted(c)
        root = sorted_c[0]
        for v in sorted_c:
            rep[v] = root

    def cl(v: int) -> int:
        return rep.get(v, v)

    # Build the multi-graph directly (MultiGraph is immutable)
    all_vertices = (set(v + cell_A_label_offset for v in cell_A.nodes) |
                    set(v + cell_B_label_offset for v in cell_B.nodes))
    keep = set(cl(v) for v in all_vertices)
    edge_counts: Dict[Tuple[int, int], int] = {}
    loop_counts: Dict[int, int] = {}

    def _add(u: int, v: int) -> None:
        if u == v:
            loop_counts[u] = loop_counts.get(u, 0) + 1
        else:
            e = (min(u, v), max(u, v))
            edge_counts[e] = edge_counts.get(e, 0) + 1

    for (u, v) in cell_A.edges:
        _add(cl(u + cell_A_label_offset), cl(v + cell_A_label_offset))
    for (u, v) in cell_B.edges:
        _add(cl(u + cell_B_label_offset), cl(v + cell_B_label_offset))
    return MultiGraph(
        nodes=frozenset(keep), edge_counts=edge_counts, loop_counts=loop_counts,
    )


def _enumerate_component_phi_terms(
    component_edges: List[Tuple[int, int]],
    component_nodes: List[int],
) -> Dict[Tuple[FrozenSet[int], ...], Dict[int, int]]:
    """Enumerate compatible φ partitions for one H_J component.

    Returns a dict:
      `phi_signature` (tuple of frozenset vertex classes, sorted) →
      `coeff_dict` (dict {|A_J|: count}).

    Each `coeff_dict[k] = c` means there are `c` distinct subsets
    A_J ⊆ component_edges with |A_J| = k that induce this φ partition.
    The polynomial coefficient at variable v is Σ_k c · v^k.
    """
    sigs: Dict[Tuple[FrozenSet[int], ...], Dict[int, int]] = {}
    n_e = len(component_edges)
    for mask in range(1 << n_e):
        A_J = [component_edges[i] for i in range(n_e) if (mask >> i) & 1]
        sig = tuple(sorted(
            _components_of(A_J, component_nodes),
            key=lambda c: (len(c), sorted(c)),
        ))
        d = sigs.setdefault(sig, {})
        d[len(A_J)] = d.get(len(A_J), 0) + 1
    return sigs


def _tree_dp_component_phi_terms(
    component_edges: List[Tuple[int, int]],
    component_nodes: List[int],
) -> Dict[Tuple[FrozenSet[int], ...], Dict[int, int]]:
    """Edge-by-edge DP equivalent to `_enumerate_component_phi_terms`.

    Replaces brute-force 2^|component_edges| enumeration with a DP whose
    state count is bounded by Bell(|component_nodes|): each state is a
    labeled partition of component_nodes. For each edge (a, b) in turn,
    branch on edge ∈ A_J (merge classes of a and b, +1 to polynomial)
    vs edge ∉ A_J (partition unchanged).

    Output is byte-identical to the brute-force function for the same
    inputs — only the enumeration cost differs:
      brute-force: O(2^|E| · |V|)
      tree-DP:     O(|E| · |reachable_partitions|) where reachable
                   ≤ Bell(|V|) and typically much smaller for sparse H_J.

    Use when 2^|component_edges| exceeds a tractable threshold (≥ 16-18).
    Below that, brute force is faster due to constant-factor overhead.
    """
    nodes_sorted = sorted(component_nodes)
    init_partition = tuple(
        frozenset({v}) for v in nodes_sorted
    )
    states: Dict[Tuple[FrozenSet[int], ...], Dict[int, int]] = {
        init_partition: {0: 1}
    }

    def _canonical(parts: Set[FrozenSet[int]]) -> Tuple[FrozenSet[int], ...]:
        return tuple(sorted(parts, key=lambda c: (len(c), sorted(c))))

    for (a, b) in component_edges:
        new_states: Dict[Tuple[FrozenSet[int], ...], Dict[int, int]] = {}
        for partition, coef_dict in states.items():
            # Branch 1: edge NOT in A_J — partition unchanged, polynomial unchanged.
            slot1 = new_states.get(partition)
            if slot1 is None:
                new_states[partition] = dict(coef_dict)
            else:
                for k, c in coef_dict.items():
                    slot1[k] = slot1.get(k, 0) + c

            # Branch 2: edge IN A_J — merge classes containing a and b.
            cls_a = None
            cls_b = None
            for cls in partition:
                if a in cls:
                    cls_a = cls
                if b in cls:
                    cls_b = cls
                if cls_a is not None and cls_b is not None:
                    break
            if cls_a is cls_b:
                # Already merged — partition unchanged, +1 to polynomial.
                slot2 = new_states.setdefault(partition, {})
                for k, c in coef_dict.items():
                    slot2[k + 1] = slot2.get(k + 1, 0) + c
            else:
                merged = cls_a | cls_b
                new_part_set = set(partition)
                new_part_set.discard(cls_a)
                new_part_set.discard(cls_b)
                new_part_set.add(merged)
                new_partition = _canonical(new_part_set)
                slot2 = new_states.setdefault(new_partition, {})
                for k, c in coef_dict.items():
                    slot2[k + 1] = slot2.get(k + 1, 0) + c

        states = new_states

    # Match brute-force output key ordering.
    return {
        tuple(sorted(p, key=lambda c: (len(c), sorted(c)))): coef
        for p, coef in states.items()
    }


def _component_aut_perms(
    component_edges: List[Tuple[int, int]],
    component_nodes: List[int],
    *,
    full_graph: Optional[nx.Graph] = None,
    cell_A_verts: Optional[Set[int]] = None,
    cell_B_verts: Optional[Set[int]] = None,
) -> List[Dict[int, int]]:
    """Return component-vertex permutations from Aut acting on H_J component.

    If `full_graph` is None (legacy, **unsafe** for arbitrary cells):
    returns Aut(H_J component) directly. May include permutations that
    don't extend to merger automorphisms; safe only when the cells have
    rich symmetry (e.g., K_n).

    If `full_graph` is the full chord-joined graph G_1 ⊕ G_2 + E_J AND
    `cell_A_verts`, `cell_B_verts` are provided, restricts to auts that
    preserve the cell bipartition (no cell-A↔cell-B mixing). This is
    required for correctness when the full graph's natural Aut would
    permit vertex moves that break the cell structure (e.g., K_4+K_{4,4}+K_4
    has Aut = S_8 of order 40320, but cell-preserving Aut is only 1152).
    Cells get distinct color labels via `node_match`.

    If `full_graph` is provided WITHOUT cell coloring, falls back to
    "restrict to autos fixing the component set" — UNSAFE for K_n+K_n+K_n
    style graphs because it lets cell-A and cell-B vertices interchange.
    """
    from networkx.algorithms.isomorphism import GraphMatcher
    if full_graph is None:
        comp = nx.Graph()
        comp.add_nodes_from(component_nodes)
        comp.add_edges_from(component_edges)
        return [m for m in GraphMatcher(comp, comp).isomorphisms_iter()]

    # Build a color-attributed copy of full_graph so VF2's node_match
    # forbids cell-A↔cell-B vertex moves.
    if cell_A_verts is not None and cell_B_verts is not None:
        colored = full_graph.copy()
        for v in colored.nodes():
            if v in cell_A_verts:
                colored.nodes[v]['cell'] = 'A'
            elif v in cell_B_verts:
                colored.nodes[v]['cell'] = 'B'
            else:
                colored.nodes[v]['cell'] = 'other'

        def _nm(n1, n2):
            return n1.get('cell') == n2.get('cell')
        gm = GraphMatcher(colored, colored, node_match=_nm)
    else:
        gm = GraphMatcher(full_graph, full_graph)

    comp_set = set(component_nodes)
    perms: List[Dict[int, int]] = []
    seen: Set[Tuple[Tuple[int, int], ...]] = set()
    for m in gm.isomorphisms_iter():
        # Only keep auts that fix the component set (component → component).
        if not all(m[v] in comp_set for v in component_nodes):
            continue
        restricted = {v: m[v] for v in component_nodes}
        sig = tuple(sorted(restricted.items()))
        if sig not in seen:
            seen.add(sig)
            perms.append(restricted)
    return perms


def _phi_canonical_under_aut(
    phi_sig: Tuple[FrozenSet[int], ...],
    aut_perms: List[Dict[int, int]],
) -> Tuple[Tuple[int, ...], ...]:
    """Canonical orbit form of φ under Aut action.

    Returns a tuple-of-tuples representation (avoids frozenset's
    subset-based ordering which is non-lexicographic).
    """
    best = None
    for pm in aut_perms:
        img = tuple(sorted(
            tuple(sorted(pm[v] for v in cls)) for cls in phi_sig
        ))
        if best is None or img < best:
            best = img
    return best


def _aut_orbit_compress_phi_terms(
    sigs: Dict[Tuple[FrozenSet[int], ...], Dict[int, int]],
    aut_perms: List[Dict[int, int]],
) -> Dict[Tuple[FrozenSet[int], ...], Dict[int, int]]:
    """Compress compatible-φ dict by Aut orbits.

    For each orbit, keep one representative φ and sum the coefficient
    polynomials. The representative is the φ whose tuple-canonical
    form is the orbit min.
    """
    if not aut_perms or len(aut_perms) == 1:
        return sigs
    orbit_to_rep: Dict[Tuple[Tuple[int, ...], ...],
                       Tuple[Tuple[FrozenSet[int], ...], Dict[int, int]]] = {}
    for phi, coef in sigs.items():
        canon = _phi_canonical_under_aut(phi, aut_perms)
        if canon not in orbit_to_rep:
            # Take the orbit minimum (canonical) as the representative.
            # We need to convert canonical tuple form back to frozenset form
            # for consistent downstream handling.
            rep_phi = tuple(sorted(
                (frozenset(c) for c in canon),
                key=lambda c: (len(c), sorted(c)),
            ))
            orbit_to_rep[canon] = (rep_phi, dict(coef))
        else:
            existing_coef = orbit_to_rep[canon][1]
            for k, c in coef.items():
                existing_coef[k] = existing_coef.get(k, 0) + c
    return {rep: coef for (rep, coef) in orbit_to_rep.values()}


def _coefficient_polyval(coeff_dict: Dict[int, int], v_val: int) -> int:
    """Evaluate polynomial Σ c_k · v^k at v = v_val."""
    return sum(c * (v_val ** k) for k, c in coeff_dict.items())


def _sokal_z_eval_and_interpolate(terms, c_total, n_total, n_edges_total):
    """Multi-point eval of the Sokal-Z chord-junction formula + bivariate interp.

    Shared tail of the matching (`compute_sokal_z_chord_junction`) and
    per-component (`..._per_component`) entry points. `terms` is a list of
    `(t_poly, c_M, v_M, weight)` where `weight(v_val) -> int` is the per-merger
    coefficient (v**|A_J| in the matching case; a coef-dict polyval in the
    per-component case). Reconstructs T via
    Z_M = (x-1)^c_M (y-1)^v_M T(M;x,y), Z = sum weight(v) Z_M,
    T = Z / ((x-1)^c_total (y-1)^n_total).
    """
    r_total = n_total - c_total
    deg_x_max = r_total
    deg_y_max = n_edges_total - r_total
    # Distinct points avoiding x=1, y=1 (conversion singularities).
    xs = list(range(2, deg_x_max + 3))
    ys = list(range(2, deg_y_max + 3))
    values: Dict[Tuple[int, int], int] = {}
    for x_val in xs:
        for y_val in ys:
            v_val = y_val - 1
            z_sum = 0
            for (t_poly, c_M, v_M, weight) in terms:
                t_val = t_poly.evaluate(x_val, y_val)
                z_M = (x_val - 1) ** c_M * (y_val - 1) ** v_M * t_val
                z_sum += weight(v_val) * z_M
            denom = (x_val - 1) ** c_total * (y_val - 1) ** n_total
            values[(x_val, y_val)] = (z_sum // denom if isinstance(z_sum, int)
                                      else z_sum / denom)
    coeffs = _bivariate_interpolate(values, deg_x_max, deg_y_max, xs, ys)
    return TuttePolynomial.from_coefficients(coeffs)


def compute_sokal_z_chord_junction_per_component(
    cell_A: Graph, cell_B: Graph,
    chord_edges: List[Tuple[int, int]],
    synth_func: Callable[[MultiGraph], TuttePolynomial],
    *,
    max_phi_per_component: int = 200,
    max_phi_cross_product: int = 10_000_000,
    use_aut_compression: bool = True,
    tree_dp_edge_threshold: int = 13,
) -> Optional[TuttePolynomial]:
    """Compute T via per-H_J-component Sokal-Z enumeration.

    This is the tractability extension for |E_J| > 16: instead of
    enumerating 2^|E_J| subsets of chord edges directly, we decompose
    H_J = (V_k^A ∪ V_k^B, E_J) into connected components and enumerate
    compatible φ partitions PER COMPONENT (2^|E_component| each), then
    cross-product the φ tuples.

    Gates:
      - Each H_J component must produce ≤ max_phi_per_component
        distinct compatible φ partitions.
      - The total cross-product Π(component φ counts) must be
        ≤ max_phi_cross_product.

    Returns None when gates fail; caller falls through to other dispatch.

    Note: the cross-product step iterates all φ-tuples and computes the
    merger graph + T(merger). Many tuples produce isomorphic mergers
    (cache by canonical_key). For Z(1, 2) the cross-product is ~3e8
    which exceeds the default gate; further symmetry compression (cell
    Aut orbits on φ) is the next optimization.
    """
    import itertools
    n_A = cell_A.node_count()
    n_B = cell_B.node_count()
    n_total = n_A + n_B
    cell_A_offset = 0
    cell_B_offset = n_A

    chord_global = [(a + cell_A_offset, b + cell_B_offset)
                    for (a, b) in chord_edges]
    n_chord = len(chord_global)

    # Build H_J = bipartite graph on anchors with edges = E_J
    anchors_all = sorted(
        set(a for a, b in chord_global) | set(b for a, b in chord_global)
    )
    H_J = nx.Graph()
    H_J.add_nodes_from(anchors_all)
    for (a, b) in chord_global:
        H_J.add_edge(a, b)

    # Per-component enumeration
    components: List[List[int]] = [
        sorted(c) for c in nx.connected_components(H_J)
    ]
    # Each component has its own A_J edge set
    component_edges: List[List[Tuple[int, int]]] = []
    for comp in components:
        comp_set = set(comp)
        c_edges = [(a, b) for (a, b) in chord_global
                   if a in comp_set and b in comp_set]
        component_edges.append(c_edges)

    # Enumerate compatible φ per component, then orbit-compress under
    # Aut(component). Empirically (Z(1, 2) H_J component): 17236
    # compatible φ → 2297 orbit-distinct φ (7.5× compression). Cross-
    # product drops from 297M to 5.3M — into the tractable range.
    # Build the full chord-joined graph for safe Aut restriction
    full_graph_nx: Optional[nx.Graph] = None
    if use_aut_compression:
        full_graph_nx = nx.Graph()
        full_graph_nx.add_nodes_from(range(n_total))
        for (u, v) in cell_A.edges:
            full_graph_nx.add_edge(u + cell_A_offset, v + cell_A_offset)
        for (u, v) in cell_B.edges:
            full_graph_nx.add_edge(u + cell_B_offset, v + cell_B_offset)
        for (a, b) in chord_global:
            full_graph_nx.add_edge(a, b)

    per_comp_phi: List[Dict[Tuple[FrozenSet[int], ...], Dict[int, int]]] = []
    cross_product = 1
    for i, (comp, c_edges) in enumerate(zip(components, component_edges)):
        if len(c_edges) >= tree_dp_edge_threshold:
            sigs = _tree_dp_component_phi_terms(c_edges, comp)
        else:
            if (1 << len(c_edges)) > 1 << 18:  # brute-force subset cap
                return None
            sigs = _enumerate_component_phi_terms(c_edges, comp)
        if use_aut_compression:
            # Use auts from the FULL chord-joined graph restricted to
            # this component, with cell coloring so cell-A↔cell-B mixing
            # is disallowed (required for correctness on K_n+K_n+K_n style
            # graphs where the naive full-graph Aut is too large).
            cell_A_vset = set(range(cell_A_offset,
                                    cell_A_offset + n_A))
            cell_B_vset = set(range(cell_B_offset,
                                    cell_B_offset + n_B))
            aut_perms = _component_aut_perms(
                c_edges, comp, full_graph=full_graph_nx,
                cell_A_verts=cell_A_vset, cell_B_verts=cell_B_vset,
            )
            sigs = _aut_orbit_compress_phi_terms(sigs, aut_perms)
        if len(sigs) > max_phi_per_component:
            return None
        per_comp_phi.append(sigs)
        cross_product *= len(sigs)
        if cross_product > max_phi_cross_product:
            return None

    # Determine c(total) for original G_1 ⊕ G_2 + E_J
    big = nx.Graph()
    big.add_nodes_from(range(n_total))
    for (u, v) in cell_A.edges:
        big.add_edge(u + cell_A_offset, v + cell_A_offset)
    for (u, v) in cell_B.edges:
        big.add_edge(u + cell_B_offset, v + cell_B_offset)
    for (a, b) in chord_global:
        big.add_edge(a, b)
    c_total = nx.number_connected_components(big)

    # Aggregate cross-product terms by merger canonical key.
    # canonical_key gives strong dedup (catches all iso-but-different
    # mergers) → bounded memory. The per-pair cost is dominated by
    # canonical_key (~25μs) + merger build (~17μs).
    # Memory: stores one (T_poly, c_M, v_M) entry per distinct merger
    # canonical key + one coefficient dict — typically thousands of
    # entries, well-bounded.
    merger_T_cache: Dict[str, Tuple[TuttePolynomial, int, int]] = {}
    contrib: Dict[str, Dict[int, int]] = {}

    phi_sigs_per_comp = [list(d.keys()) for d in per_comp_phi]

    for phi_tuple in itertools.product(*phi_sigs_per_comp):
        phi_classes: List[FrozenSet[int]] = []
        for sig in phi_tuple:
            phi_classes.extend(sig)
        mg = _build_merger_graph(
            cell_A, cell_B, cell_A_offset, cell_B_offset, phi_classes,
        )
        merger_key = mg.canonical_key()
        if merger_key not in merger_T_cache:
            t_poly = synth_func(mg)
            nxm = nx.MultiGraph()
            nxm.add_nodes_from(mg.nodes)
            for (u, v), cnt in mg.edge_counts.items():
                for _ in range(cnt):
                    nxm.add_edge(u, v)
            for u, cnt in mg.loop_counts.items():
                for _ in range(cnt):
                    nxm.add_edge(u, u)
            c_M = nx.number_connected_components(nxm)
            v_M = mg.node_count()
            merger_T_cache[merger_key] = (t_poly, c_M, v_M)
        combined: Dict[int, int] = {0: 1}
        for i, sig in enumerate(phi_tuple):
            coef_d = per_comp_phi[i][sig]
            new: Dict[int, int] = {}
            for k1, c1 in combined.items():
                for k2, c2 in coef_d.items():
                    new[k1 + k2] = new.get(k1 + k2, 0) + c1 * c2
            combined = new
        slot = contrib.setdefault(merger_key, {})
        for k, c in combined.items():
            slot[k] = slot.get(k, 0) + c

    terms = [
        (merger_T_cache[k][0], merger_T_cache[k][1], merger_T_cache[k][2],
         (lambda vv, _cd=contrib[k]: _coefficient_polyval(_cd, vv)))
        for k in contrib
    ]
    n_edges_total = cell_A.edge_count() + cell_B.edge_count() + n_chord
    return _sokal_z_eval_and_interpolate(terms, c_total, n_total, n_edges_total)


def compute_sokal_z_chord_junction(
    cell_A: Graph, cell_B: Graph,
    chord_edges: List[Tuple[int, int]],
    synth_func: Callable[[MultiGraph], TuttePolynomial],
    *,
    max_subsets: int = 65536,
    max_phi_per_component: int = 200,
    max_phi_cross_product: int = 10_000_000,
) -> Optional[TuttePolynomial]:
    """Compute T(G_1 ⊕_{E_J} G_2; x, y) via the Sokal Z generalized formula.

    Args:
        cell_A, cell_B: cell graphs (vertex labels 0..n_A-1 and 0..n_B-1
            respectively). They get reassigned offsets internally.
        chord_edges: list of (a, b) inter-cell chord edges where a ∈
            nodes of cell_A and b ∈ nodes of cell_B. Multi-edges allowed.
        synth_func: callback that synthesizes T(merger_multigraph). The
            engine's `_synthesize_multigraph` is the right callback.
        max_subsets: prototype gate. If 2^|E_J| > max_subsets, returns
            None (tree-DP enumeration not yet implemented). For Z(1, 2)
            this would be 2^32 = 4B; the gate blocks intractable cases.

    Returns:
        T(G_1 ⊕ G_2 + chord_edges; x, y) as a TuttePolynomial, or None
        if intractable.

    Algorithm:
      1. Reassign cell B vertex labels to be disjoint from cell A.
      2. Enumerate A_J ⊆ chord_edges (2^|E_J| subsets).
      3. For each A_J, compute φ partition on V_k^A ∪ V_k^B and build
         the merger multigraph.
      4. Synthesize T(merger) for each unique merger (cached by canonical key).
      5. Use multi-point evaluation: pick (degree_x+1) * (degree_y+1)
         distinct (x, y) points. For each point:
           - q = (x-1)(y-1), v = y-1.
           - Z(merger; q, v) = (x-1)^c(M) (y-1)^(|V(M)|-c(M)) · T(merger; x, y).
           - Z_sum = Σ_{A_J} v^|A_J| · Z(merger(A_J); q, v).
           - T(G_1 ⊕ G_2 + E_J; x, y) = Z_sum / ((x-1)^c(total) (y-1)^(|V|-c(total))).
      6. Bivariate Lagrange interpolation back to polynomial form.
    """
    import itertools
    n_A = cell_A.node_count()
    n_B = cell_B.node_count()
    n_total = n_A + n_B  # cells are vertex-disjoint in G_1 ⊕ G_2
    # Cell B vertex offset to ensure disjoint global IDs
    cell_B_offset = n_A
    cell_A_offset = 0

    # Re-label chord edges to global IDs
    chord_global = [(a + cell_A_offset, b + cell_B_offset)
                    for (a, b) in chord_edges]

    n_chord = len(chord_global)
    if (1 << n_chord) > max_subsets:
        # Try per-component path before giving up.
        return compute_sokal_z_chord_junction_per_component(
            cell_A, cell_B, chord_edges, synth_func,
            max_phi_per_component=max_phi_per_component,
            max_phi_cross_product=max_phi_cross_product,
        )

    # Anchor sets
    anchors_A = sorted(set(a for a, b in chord_global))
    anchors_B = sorted(set(b for a, b in chord_global))
    anchors_all = anchors_A + anchors_B

    # Determine c(total) for the original G_1 ⊕ G_2 + E_J graph
    # by building it as an nx graph and counting components.
    big = nx.Graph()
    big.add_nodes_from(range(n_total))
    for (u, v) in cell_A.edges:
        big.add_edge(u + cell_A_offset, v + cell_A_offset)
    for (u, v) in cell_B.edges:
        big.add_edge(u + cell_B_offset, v + cell_B_offset)
    for (a, b) in chord_global:
        big.add_edge(a, b)
    c_total = nx.number_connected_components(big)

    # ---- Step 1: enumerate A_J subsets, group by canonical merger ----
    # merger_key -> list of (A_J_size, c_merger, |V_merger|, t_poly)
    # Each entry contributes v^|A_J| · (x-1)^c (y-1)^(n-c) · T to Z_sum.
    merger_T_cache: Dict[str, Tuple[TuttePolynomial, int, int]] = {}
    # contrib_terms: list of (|A_J|, c_M, |V_M|, merger_canon_key)
    contrib_terms: List[Tuple[int, int, int, str]] = []

    for mask in range(1 << n_chord):
        A_J = [chord_global[i] for i in range(n_chord) if (mask >> i) & 1]
        # φ partition via connected components of (anchors_all, A_J)
        phi_classes = _components_of(A_J, anchors_all)
        # Build merger multigraph
        mg = _build_merger_graph(
            cell_A, cell_B, cell_A_offset, cell_B_offset, phi_classes,
        )
        merger_key = mg.canonical_key()
        if merger_key not in merger_T_cache:
            t_poly = synth_func(mg)
            # Determine c(M) and |V(M)|: build nx graph and count.
            nxm = nx.MultiGraph()
            nxm.add_nodes_from(mg.nodes)
            for (u, v), cnt in mg.edge_counts.items():
                for _ in range(cnt):
                    nxm.add_edge(u, v)
            for u, cnt in mg.loop_counts.items():
                for _ in range(cnt):
                    nxm.add_edge(u, u)
            c_M = nx.number_connected_components(nxm)
            v_M = mg.node_count()
            merger_T_cache[merger_key] = (t_poly, c_M, v_M)
        _, c_M, v_M = merger_T_cache[merger_key]
        contrib_terms.append((len(A_J), c_M, v_M, merger_key))

    # ---- Step 2: multi-point eval + bivariate Lagrange interpolation ----
    terms = [
        (merger_T_cache[mkey][0], c_M, v_M, (lambda vv, _s=size: vv ** _s))
        for (size, c_M, v_M, mkey) in contrib_terms
    ]
    n_edges_total = cell_A.edge_count() + cell_B.edge_count() + n_chord
    return _sokal_z_eval_and_interpolate(terms, c_total, n_total, n_edges_total)


def _bivariate_interpolate(
    values: Dict[Tuple[int, int], int],
    deg_x: int, deg_y: int,
    xs: List[int], ys: List[int],
) -> Dict[Tuple[int, int], int]:
    """Bivariate Lagrange interpolation from values to coefficient dict."""
    # Strategy: build coefficient matrix and solve linear system.
    # Variables: c_{i,j} for i in [0, deg_x], j in [0, deg_y].
    # Equations: Σ c_{i,j} x^i y^j = values[(x, y)] for each (x, y).
    n_coeffs = (deg_x + 1) * (deg_y + 1)
    pts = [(xv, yv) for xv in xs for yv in ys]
    assert len(pts) >= n_coeffs, (
        f"insufficient evaluation points: {len(pts)} < {n_coeffs}"
    )
    # Build the Vandermonde matrix and RHS.
    import sympy
    A = sympy.Matrix(len(pts), n_coeffs, lambda row, col: 0)
    b = sympy.Matrix(len(pts), 1, lambda row, _: 0)
    for row, (xv, yv) in enumerate(pts):
        for col, (i, j) in enumerate(
            (i, j) for i in range(deg_x + 1) for j in range(deg_y + 1)
        ):
            A[row, col] = sympy.Rational(xv) ** i * sympy.Rational(yv) ** j
        b[row, 0] = sympy.Rational(int(values[(xv, yv)]))
    # Solve (least squares if overdetermined, but should be exact)
    sol = A.solve(b)
    coeffs = {}
    for col, (i, j) in enumerate(
        (i, j) for i in range(deg_x + 1) for j in range(deg_y + 1)
    ):
        c = sol[col, 0]
        if c != 0:
            # Should be an integer
            coeffs[(i, j)] = int(c)
    return coeffs
