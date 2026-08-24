"""Unified bivariate chord-junction closed-form Tutte polynomial.

Implements the **unified bivariate inclusion-exclusion theorem** for
chord-junction Tutte polynomials (proved May 25, 2026; see
``tutte/research/cyclotomic_chord_junction_theorem.md``):

    T(G ⊕_{V_k} G; x, y)
        = (x − 1) · T(G; x, y)²
        + Σ_{∅ ≠ S ⊆ V_k} T(G ∪_{V_S} G; x, y),

where ``G ⊕_{V_k} G`` is two disjoint copies of ``G`` joined by chord
edges between corresponding ``V_k`` vertices, and ``G ∪_{V_S} G`` is
the multigraph obtained by identifying corresponding vertex pairs in
``S`` across the two copies (parallel edges preserved).

The asymmetric form, used when boundary cells in Pegasus or Zephyr have
different cell templates ``G_1 ≠ G_2``, replaces ``(x − 1) T(G)²`` by
``(x − 1) T(G_1) T(G_2)`` and uses ``G_1 ∪_{V_S pairs} G_2`` for the
merger graphs.

This module is the operational hook between the math and the engine:

- Merger values are looked up from the persistent ``MergerTable``
  (``tutte/lookup/merger.py``) when available. The same table backs
  Chimera, Pegasus, Zephyr, and ad-hoc user-graph chord junctions —
  one cache for "all kinds of multigraph mergers".
- On a cache miss, the caller-provided ``synth_multigraph`` callback
  computes the merger value. Optionally the new value is inserted into
  the table so the next call hits.

For the symmetric matching case (each ``V_k`` vertex used by exactly
one chord), the result is mathematically equivalent to the existing
``apply_kmatching_formula`` (covering.py); the equivalence is exercised
by ``tutte/tests/test_unified_chord_junction.py``. Both come from the
cut-vertex identity ``T(G ∪_v G) = T(G)²`` that converts between the
``(x − 1, C(k,1), …, C(k,k))`` coefficients used here and the
``(x, k − 1, C(k,2), …, C(k,k))`` coefficients used by the production
k-matching path.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..graph import Graph, MultiGraph
from ..lookup.merger import MergerEntry, MergerTable, VTTuple
from ..polynomial import TuttePolynomial


SynthMultigraph = Callable[[MultiGraph], TuttePolynomial]
"""Callable that evaluates ``T(mg)`` for a ``tutte.graph.MultiGraph``.

In production this is ``SynthesisEngine._synthesize_multigraph``;
the tests substitute equivalent thin wrappers.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def unified_chord_junction(
    base: Graph,
    V_k: Sequence[int],
    synth_multigraph: SynthMultigraph,
    *,
    merger_table: Optional[MergerTable] = None,
    update_merger_table: bool = False,
    family_tag: Optional[str] = None,
    base_name: Optional[str] = None,
) -> TuttePolynomial:
    """Compute ``T(base ⊕_{V_k} base; x, y)`` via the unified I-E theorem.

    Args:
        base: Base graph ``G``. Two copies are joined by chord edges between
            corresponding ``V_k`` vertices.
        V_k: Chord-position vertices in ``base``. Must be a subset of
            ``base.nodes``. Each ``v ∈ V_k`` produces a single chord edge
            connecting copy-1's ``v`` to copy-2's ``v``.
        synth_multigraph: Callback that returns ``T(mg)`` for a
            ``tutte.graph.MultiGraph``. Used both for the base ``T(G)`` and
            for any merger evaluations that miss ``merger_table``.
        merger_table: Optional persistent merger lookup. When provided,
            entries are checked before invoking ``synth_multigraph``.
            Pass ``None`` to disable caching (each merger is freshly computed).
        update_merger_table: When ``True`` AND ``merger_table`` is non-None,
            insert each newly computed merger into the table for reuse by
            subsequent calls. Default ``False`` (read-only lookups).
        family_tag: D-Wave family tag stored on newly inserted merger
            entries (``"chimera"``, ``"pegasus"``, ``"zephyr"``, …) for
            downstream filtering. Ignored when ``update_merger_table=False``.
        base_name: Human-readable base graph identifier (e.g. ``"K_{4,4}"``)
            stored on newly inserted entries. Ignored when not updating.

    Returns:
        ``T(base ⊕_{V_k} base; x, y)`` as a ``TuttePolynomial``.

    Notes:
        - For ``V_k = []`` returns ``T(2·G)`` (two disjoint copies, no chord
          edges). The I-E sum collapses to ``(x − 1) · T(G)²``, which equals
          ``T(2·G)`` when ``x = 1`` — but is otherwise NOT the disjoint-union
          formula. Callers should not pass empty ``V_k`` for that semantics;
          use ``synth_multigraph`` on the disjoint union directly instead.
        - For ``|V_k| ≥ 1`` the formula is exact for any base graph and any
          chord-position subset.
    """
    base_key = base.canonical_key()
    V_k_sorted: List[int] = sorted(V_k)
    n = base.node_count()

    # Validate V_k
    base_nodes = set(base.nodes)
    for v in V_k_sorted:
        if v not in base_nodes:
            raise ValueError(
                f"V_k contains vertex {v} not in base graph "
                f"(base has {n} vertices: {sorted(base_nodes)})"
            )

    # Compute T(G) once; reused as a square below.
    base_mg = MultiGraph.from_graph(base)
    T_base = synth_multigraph(base_mg)

    x_poly = TuttePolynomial.x()
    one = TuttePolynomial.from_coefficients({(0, 0): 1})
    result = (x_poly + (-1) * one) * T_base * T_base

    for r in range(1, len(V_k_sorted) + 1):
        for S in combinations(V_k_sorted, r):
            S_tuple: VTTuple = tuple(S)
            T_S = _resolve_merger(
                base=base,
                base_key=base_key,
                S=S_tuple,
                synth_multigraph=synth_multigraph,
                merger_table=merger_table,
                update_merger_table=update_merger_table,
                family_tag=family_tag,
                base_name=base_name,
            )
            result = result + T_S
    return result


def unified_chord_junction_asymmetric(
    base_left: Graph,
    base_right: Graph,
    chord_pairs: Sequence[Tuple[int, int]],
    synth_multigraph: SynthMultigraph,
    *,
    merger_table: Optional[MergerTable] = None,
    update_merger_table: bool = False,
    family_tag: Optional[str] = None,
) -> TuttePolynomial:
    """Asymmetric variant: ``T(G_left ⊕_{chord_pairs} G_right; x, y)``.

    Args:
        base_left: Left base graph ``G_1``.
        base_right: Right base graph ``G_2``. May equal ``base_left``.
        chord_pairs: List of ``(left_vertex, right_vertex)`` pairs naming
            the chord edges. Each ``(u, w)`` adds an edge between copy-1's
            ``u`` and copy-2's ``w``.
        synth_multigraph: Callback for evaluating multigraph T values.
        merger_table: Optional persistent merger cache. The asymmetric
            path keys cache lookups by the merger multigraph's canonical
            key — so an asymmetric chord pattern hits the cache whenever
            its merger graph is isomorphic to a symmetric merger that
            was already warmed (common for ``Aut``-rich bases like
            ``K_{4,4}``).
        update_merger_table: When True AND ``merger_table`` is provided,
            insert newly computed mergers into the table for reuse this
            session.
        family_tag: D-Wave family stored on new merger entries.

    Returns:
        ``T(G_left ⊕_{chord_pairs} G_right; x, y)``.
    """
    n_left = base_left.node_count()
    base_left_set = set(base_left.nodes)
    base_right_set = set(base_right.nodes)
    for (u, w) in chord_pairs:
        if u not in base_left_set:
            raise ValueError(f"chord pair ({u}, {w}): {u} not in base_left")
        if w not in base_right_set:
            raise ValueError(f"chord pair ({u}, {w}): {w} not in base_right")

    T_left = synth_multigraph(MultiGraph.from_graph(base_left))
    T_right = synth_multigraph(MultiGraph.from_graph(base_right))

    x_poly = TuttePolynomial.x()
    one = TuttePolynomial.from_coefficients({(0, 0): 1})
    result = (x_poly + (-1) * one) * T_left * T_right

    k = len(chord_pairs)
    indices = list(range(k))
    base_left_key = base_left.canonical_key() if merger_table is not None else None
    for r in range(1, k + 1):
        for S in combinations(indices, r):
            merged = _build_asymmetric_merger(
                base_left, base_right, [chord_pairs[i] for i in S],
            )
            T_merge = None
            merger_key: Optional[str] = None
            if merger_table is not None:
                try:
                    merger_key = merged.canonical_key()
                except Exception:
                    merger_key = None
                if merger_key is not None:
                    entry = merger_table.lookup_by_merger(merger_key)
                    if entry is not None:
                        T_merge = entry.polynomial
            if T_merge is None:
                T_merge = synth_multigraph(merged)
                if (
                    merger_table is not None
                    and update_merger_table
                    and merger_key is not None
                    and merger_table.lookup_by_merger(merger_key) is None
                    and base_left_key is not None
                ):
                    # Stash under a synthetic v_t key derived from the
                    # chord-pair indices; the canonical-key index is the
                    # primary lookup path so the v_t value is informational.
                    v_t_signature: VTTuple = tuple(sorted(
                        chord_pairs[i][0] for i in S
                    ))
                    merger_table.add_entry(MergerEntry(
                        base_canonical_key=base_left_key,
                        v_t=v_t_signature,
                        polynomial=T_merge,
                        merger_canonical_key=merger_key,
                        family_tag=family_tag,
                        base_node_count=base_left.node_count(),
                        base_edge_count=base_left.edge_count(),
                        merger_node_count=merged.node_count(),
                        merger_edge_count=merged.edge_count(),
                    ))
            result = result + T_merge
    return result


# ---------------------------------------------------------------------------
# Internal: merger graph construction + cache resolution
# ---------------------------------------------------------------------------


def _resolve_merger(
    *,
    base: Graph,
    base_key: str,
    S: VTTuple,
    synth_multigraph: SynthMultigraph,
    merger_table: Optional[MergerTable],
    update_merger_table: bool,
    family_tag: Optional[str],
    base_name: Optional[str],
) -> TuttePolynomial:
    """Return ``T(G ∪_{V_S} G)`` from cache, else compute and (optionally)
    insert into the cache."""
    if merger_table is not None:
        entry = merger_table.lookup_by_source(base_key, S)
        if entry is not None:
            return entry.polynomial

    merger = build_symmetric_merger(base, S)
    T_S = synth_multigraph(merger)

    if merger_table is not None and update_merger_table:
        merger_key: Optional[str]
        try:
            merger_key = merger.canonical_key()
        except Exception:
            merger_key = None
        entry = MergerEntry(
            base_canonical_key=base_key,
            v_t=S,
            polynomial=T_S,
            merger_canonical_key=merger_key,
            base_name=base_name,
            family_tag=family_tag,
            base_node_count=base.node_count(),
            base_edge_count=base.edge_count(),
            merger_node_count=merger.node_count(),
            merger_edge_count=merger.edge_count(),
        )
        merger_table.add_entry(entry)

    return T_S


def build_symmetric_merger(base: Graph, V_T: Sequence[int]) -> MultiGraph:
    """Construct ``G ∪_{V_T} G`` as a ``tutte.graph.MultiGraph``.

    Two disjoint copies of ``base`` (copy 1 at vertex IDs in
    ``base.nodes``; copy 2 at IDs shifted by ``n = base.node_count()``).
    For each ``v ∈ V_T`` the pair ``(v, v + n)`` is identified into a
    single vertex (keeping the ``v`` label). Parallel edges arising from
    the identification are preserved.

    Public for use by tests, the warmup script, and the engine dispatch
    in addition to the in-module ``_resolve_merger``.
    """
    n = base.node_count()
    V_T_set = set(V_T)

    def repr_of(node: int) -> int:
        if node < n:
            return node
        original = node - n
        if original in V_T_set:
            return original  # identified with copy-1 representative
        return node

    nodes: set = set()
    for v in base.nodes:
        nodes.add(v)  # copy 1 always present
        if v not in V_T_set:
            nodes.add(v + n)  # copy 2 only present when not identified

    edge_counts: Dict[Tuple[int, int], int] = {}
    loop_counts: Dict[int, int] = {}

    def _add_edge(u: int, w: int) -> None:
        if u == w:
            loop_counts[u] = loop_counts.get(u, 0) + 1
        else:
            key = (min(u, w), max(u, w))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    for (u, w) in base.edges:
        # Copy 1.
        _add_edge(repr_of(u), repr_of(w))
        # Copy 2.
        _add_edge(repr_of(u + n), repr_of(w + n))

    return MultiGraph(
        nodes=frozenset(nodes),
        edge_counts=edge_counts,
        loop_counts=loop_counts,
    )


def _build_asymmetric_merger(
    base_left: Graph,
    base_right: Graph,
    chord_pairs: Sequence[Tuple[int, int]],
) -> MultiGraph:
    """Construct ``G_left ∪_{chord_pairs} G_right`` as a ``MultiGraph``.

    Copy 1 = ``base_left`` (vertex IDs as given). Copy 2 = ``base_right``
    shifted by ``n_left``. For each ``(u, w) ∈ chord_pairs`` we identify
    copy-1's ``u`` with copy-2's ``w + n_left``.
    """
    n_left = base_left.node_count()
    # Build union-find-style mapping {right_vertex_shifted -> kept_id}.
    identifications: Dict[int, int] = {}
    for (u, w) in chord_pairs:
        right_id = w + n_left
        identifications[right_id] = u  # right node folds onto left vertex

    def repr_of_left(v: int) -> int:
        return v

    def repr_of_right(v: int) -> int:
        shifted = v + n_left
        return identifications.get(shifted, shifted)

    nodes: set = set()
    for v in base_left.nodes:
        nodes.add(repr_of_left(v))
    for v in base_right.nodes:
        nodes.add(repr_of_right(v))

    edge_counts: Dict[Tuple[int, int], int] = {}
    loop_counts: Dict[int, int] = {}

    def _add_edge(u: int, w: int) -> None:
        if u == w:
            loop_counts[u] = loop_counts.get(u, 0) + 1
        else:
            key = (min(u, w), max(u, w))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    for (u, w) in base_left.edges:
        _add_edge(repr_of_left(u), repr_of_left(w))
    for (u, w) in base_right.edges:
        _add_edge(repr_of_right(u), repr_of_right(w))

    return MultiGraph(
        nodes=frozenset(nodes),
        edge_counts=edge_counts,
        loop_counts=loop_counts,
    )


__all__ = [
    "SynthMultigraph",
    "unified_chord_junction",
    "unified_chord_junction_asymmetric",
    "build_symmetric_merger",
]
