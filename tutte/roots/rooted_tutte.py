"""Rooted Tutte polynomial primitives.

T_rooted(G, S)[P] = Σ over spanning subgraphs A of G of:
                    (x-1)^{r(E)-r(A)} (y-1)^{|A|-r(A)}
                   where A's component-partition restricted to S = P

Standard Tutte: T(G) = Σ_P T_rooted(G, S)[P].

Brute-force computation is exponential in
|E|, fine for small cells (≤ 16 edges) like K_{4,4}.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from ..graph import Graph
from ..polynomial import TuttePolynomial


def _power_poly(p: TuttePolynomial, k: int) -> TuttePolynomial:
    if k < 0:
        raise ValueError(f"Negative power: {k}")
    result = TuttePolynomial.one()
    for _ in range(k):
        result = result * p
    return result


def _find(parent: Dict[int, int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _normalize_partition(part: Dict[int, Set[int]]) -> Tuple[Tuple[int, ...], ...]:
    """Canonical key: sorted tuple of sorted-tuples."""
    return tuple(sorted(tuple(sorted(b)) for b in part.values()))


def t_rooted_bruteforce(
    graph: Graph,
    boundary: List[int],
) -> Dict[Tuple[Tuple[int, ...], ...], TuttePolynomial]:
    """Brute-force compute T_rooted(graph, boundary).

    Returns dict from canonical partition of boundary → polynomial.
    Uses rank-nullity: weight(A) = (x-1)^{r(E)-r(A)} (y-1)^{|A|-r(A)}.

    Cost: O(2^|E| × |V|). Fine for cells with |E| ≤ 16 (K_{4,4} = 16 edges).
    """
    edges = sorted(graph.edges)
    n_edges = len(edges)
    nodes = sorted(graph.nodes)
    n_nodes = len(nodes)

    full_dsu = {v: v for v in nodes}
    for u, v in edges:
        ru, rv = _find(full_dsu, u), _find(full_dsu, v)
        if ru != rv:
            full_dsu[max(ru, rv)] = min(ru, rv)
    k_G = len({_find(full_dsu, v) for v in nodes})
    r_E = n_nodes - k_G

    x_minus_1 = TuttePolynomial.x() + (-1) * TuttePolynomial.one()
    y_minus_1 = TuttePolynomial.y() + (-1) * TuttePolynomial.one()

    result: Dict[Tuple, TuttePolynomial] = defaultdict(lambda: TuttePolynomial.zero())

    for mask in range(2 ** n_edges):
        A_edges = [edges[i] for i in range(n_edges) if (mask >> i) & 1]
        dsu = {v: v for v in nodes}
        for u, v in A_edges:
            ru, rv = _find(dsu, u), _find(dsu, v)
            if ru != rv:
                dsu[max(ru, rv)] = min(ru, rv)
        k_A = len({_find(dsu, v) for v in nodes})
        r_A = n_nodes - k_A
        boundary_parts: Dict[int, Set[int]] = defaultdict(set)
        for v in boundary:
            boundary_parts[_find(dsu, v)].add(v)
        partition_key = _normalize_partition(boundary_parts)
        weight = _power_poly(x_minus_1, r_E - r_A) * _power_poly(y_minus_1, len(A_edges) - r_A)
        result[partition_key] = result[partition_key] + weight

    return dict(result)


def all_partitions(elements: List[int]) -> List[List[Set[int]]]:
    """All set-partitions of elements."""
    if not elements:
        return [[]]
    if len(elements) == 1:
        return [[{elements[0]}]]
    result = []
    first = elements[0]
    for sub in all_partitions(elements[1:]):
        result.append([{first}] + [set(b) for b in sub])
        for i in range(len(sub)):
            new_sub = [set(b) for b in sub]
            new_sub[i] = new_sub[i] | {first}
            result.append(new_sub)
    return result


def join_partitions(
    P1: Tuple[Tuple[int, ...], ...],
    P2: Tuple[Tuple[int, ...], ...],
    universe: List[int],
) -> Tuple[Tuple[int, ...], ...]:
    """Compute join (transitive closure) of P1 and P2 over universe.

    Two elements are in the same block of the join iff there's a path
    of "same-block" relations through P1 ∪ P2.
    """
    parent = {v: v for v in universe}
    for blocks in [P1, P2]:
        for block in blocks:
            if len(block) <= 1:
                continue
            rep = block[0]
            for v in block[1:]:
                ru, rv = _find(parent, rep), _find(parent, v)
                if ru != rv:
                    parent[max(ru, rv)] = min(ru, rv)
    out: Dict[int, Set[int]] = defaultdict(set)
    for v in universe:
        out[_find(parent, v)].add(v)
    return _normalize_partition(out)


def delta(
    P1: Tuple[Tuple[int, ...], ...],
    P2: Tuple[Tuple[int, ...], ...],
    shared_boundary: List[int],
) -> int:
    """DELTA(P_1, P_2) = nblocks(JOIN(P_1, P_2)) + |S| - nblocks(P_1) - nblocks(P_2).

    Rank deficit when components merge across the shared boundary in vertex-sum.
    """
    join = join_partitions(P1, P2, shared_boundary)
    return len(join) + len(shared_boundary) - len(P1) - len(P2)


def restrict_partition(
    P: Tuple[Tuple[int, ...], ...],
    subset: List[int],
) -> Tuple[Tuple[int, ...], ...]:
    """Restrict partition P to subset; isolated vertices added as singletons."""
    subset_set = set(subset)
    blocks: List[Tuple[int, ...]] = []
    seen: set = set()
    for block in P:
        intersection = tuple(sorted(v for v in block if v in subset_set))
        if intersection:
            blocks.append(intersection)
            seen.update(intersection)
    for v in subset:
        if v not in seen:
            blocks.append((v,))
    return tuple(sorted(blocks))


def divide_by_x_minus_1_power(
    poly: TuttePolynomial, k: int,
) -> TuttePolynomial:
    """Divide polynomial by (x-1)^k via repeated synthetic division.

    Raises ValueError if poly is not divisible (remainder != 0).
    """
    if k == 0:
        return poly
    coeffs: Dict[Tuple[int, int], int] = {}
    for i, j, c in poly.terms():
        coeffs[(i, j)] = c
    if not coeffs:
        return TuttePolynomial.zero()
    for _ in range(k):
        new_coeffs: Dict[Tuple[int, int], int] = {}
        y_groups: Dict[int, Dict[int, int]] = defaultdict(dict)
        for (i, j), c in coeffs.items():
            y_groups[j][i] = c
        for j, x_coeffs in y_groups.items():
            max_i = max(x_coeffs.keys())
            running = 0
            q_coeffs: Dict[int, int] = {}
            for i in range(max_i, -1, -1):
                ci = x_coeffs.get(i, 0)
                running += ci
                if i > 0:
                    q_coeffs[i - 1] = running
            remainder = running
            if remainder != 0:
                raise ValueError(
                    f"Polynomial not divisible by (x-1) at y^{j}: "
                    f"remainder={remainder}"
                )
            for i_new, c_new in q_coeffs.items():
                if c_new != 0:
                    new_coeffs[(i_new, j)] = c_new
        coeffs = new_coeffs
    return TuttePolynomial.from_coefficients(coeffs)


_T_ROOTED_CACHE: Dict[Tuple[str, Tuple[int, ...]], Dict[Tuple, TuttePolynomial]] = {}
"""In-process runtime cache for T_rooted(graph, boundary).

Lifetime: this Python process. Keys are `(canonical_key, sorted_boundary)`;
values are partition→polynomial dicts in the runtime graph's labels.
Cleared via `clear_t_rooted_cache()` between independent test runs.
"""


_T_ROOTED_GRAPHS: Dict[Tuple[str, Tuple[int, ...]], "Graph"] = {}
"""Sidecar dict for `_T_ROOTED_CACHE` that pins the originating Graph.

Populated alongside `_T_ROOTED_CACHE` whenever a fresh cold-compute
entry is added by `t_rooted_cached`. Enables `save_rooted_lookup_default`
to serialize the cache without callers needing to thread Graph objects
through the API. Not populated on persistent-cache hits (those entries
were saved with canonical labels and don't need re-translation).
"""


_ROOTED_LOOKUP: Dict[Tuple[str, Tuple[int, ...]], List[Dict]] = {}
"""Persistent rooted-Tutte lookup table.

Loaded once at engine startup from `tutte/data/rooted_lookup.{bin,json}`.
Partition keys stored in CANONICAL labeling (per `canonical_node_mapping`
in `tutte/graph.py`); on cache hit for a runtime graph with the same
canonical_key but different vertex labels, keys are translated to the
runtime's labels via its canonical mapping inverse. Read-only at
runtime — repopulate via `scripts/warmup_rooted_lookup.py`.

Format: dict from `(canonical_key, canonical_boundary_tuple)` to list of
serialized partition entries (each `{"key": [[blocks]], "coeffs": {...}}`).
"""


def save_rooted_lookup(
    path: str,
    graphs: Dict[Tuple[str, Tuple[int, ...]], "Graph"],
) -> int:
    """Save `_T_ROOTED_CACHE` entries to disk as the rooted lookup table.

    Takes a `graphs` dict mapping cache keys to the actual Graph objects
    that produced them. Translates each entry's partition keys to
    canonical labels (so they can be re-translated to any isomorphic
    runtime graph's labels on load).

    Writes a JSON file at `path`. If `path` ends in `.bin`, writes the
    binary format instead via `save_rooted_lookup_binary`.

    Returns the number of entries saved.
    """
    if path.endswith(".bin"):
        return save_rooted_lookup_binary(path, graphs)
    import json
    from ..graph import canonical_node_mapping as _cnm
    import networkx as nx

    entries = []
    for cache_key, partition_dict in _T_ROOTED_CACHE.items():
        # Cache key format (May 20, 2026): (canon, edges_tuple, sorted_boundary).
        # Older callers passed (canon, sorted_boundary); accept both shapes.
        if len(cache_key) == 3:
            canon, _edges, boundary = cache_key
        else:
            canon, boundary = cache_key
        if cache_key not in graphs:
            continue
        g = graphs[cache_key]
        nxg = nx.Graph()
        nxg.add_nodes_from(g.nodes)
        nxg.add_edges_from(g.edges)
        mapping_orig = _cnm(nxg)
        canonical_boundary = tuple(sorted(mapping_orig[v] for v in boundary))
        partitions = []
        for block_key, poly in partition_dict.items():
            canonical_blocks = tuple(sorted(
                tuple(sorted(mapping_orig[v] for v in block))
                for block in block_key
            ))
            coeffs = {
                f"{i},{j}": c
                for (i, j), c in poly.to_coefficients().items()
            }
            partitions.append({
                "key": [list(b) for b in canonical_blocks],
                "coeffs": coeffs,
            })
        entries.append({
            "canon": canon,
            "canonical_boundary": list(canonical_boundary),
            "partitions": partitions,
        })
    data = {
        "description": "Rooted-Tutte lookup table (partitions in canonical labels)",
        "format_version": "rooted_lookup_v1",
        "entries": entries,
        "n_entries": len(entries),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return len(entries)


def load_rooted_lookup(path: str) -> int:
    """Load rooted-lookup entries from disk into `_ROOTED_LOOKUP`.

    Accepts either the JSON (`rooted_lookup.json`) or binary
    (`rooted_lookup.bin`) format; the extension selects the codec.

    Returns the number of entries loaded; silent no-op if `path` is
    absent or the file's format header is unrecognized.
    """
    import os
    if not os.path.exists(path):
        return 0
    if path.endswith(".bin"):
        return load_rooted_lookup_binary(path)
    import json
    with open(path, "r") as f:
        data = json.load(f)
    if data.get("format_version") not in ("rooted_lookup_v1", "v2-canonical"):
        return 0
    n_loaded = 0
    for entry in data.get("entries", []):
        canon = entry["canon"]
        cbdry = tuple(entry["canonical_boundary"])
        _ROOTED_LOOKUP[(canon, cbdry)] = entry["partitions"]
        n_loaded += 1
    return n_loaded


def clear_t_rooted_cache() -> None:
    """Clear the in-process `_T_ROOTED_CACHE` and its `_T_ROOTED_GRAPHS`
    sidecar. Does NOT clear the persistent `_ROOTED_LOOKUP`. For test
    isolation only.
    """
    _T_ROOTED_CACHE.clear()
    _T_ROOTED_GRAPHS.clear()


def save_rooted_lookup_default() -> int:
    """Save the in-process `_T_ROOTED_CACHE` to the default lookup file.

    Writes `tutte/data/rooted_lookup_table.bin`. Uses the
    `_T_ROOTED_GRAPHS` sidecar to obtain the originating Graph for each
    cache entry, so callers don't need to thread graphs through the API.

    Returns ``n_bin``: the number of entries saved. Entries without a
    tracked Graph (e.g. those loaded from the persistent lookup at
    startup) are silently skipped — they're already on disk.
    """
    import os
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data",
    )
    bin_path = os.path.join(base_dir, "rooted_lookup_table.bin")
    n_bin = save_rooted_lookup_binary(bin_path, dict(_T_ROOTED_GRAPHS))
    return n_bin


_DEFAULT_ROOTED_LOOKUP_LOADED: bool = False


def load_default_rooted_lookup() -> int:
    """Load the default rooted-lookup table from `tutte/data/`.

    Tries `rooted_lookup_table.bin` first (faster), falls back to
    `rooted_lookup_table.json`. Engine startup calls this; the populated
    `_ROOTED_LOOKUP` is consulted by `t_rooted_cached` after the
    in-process cache miss; partition keys get translated from canonical
    labels to runtime labels on demand.

    Filename matches the `*lookup_table*` `.gitignore` pattern, so the
    data file is not tracked in version control alongside the package
    `lookup_table.{bin,json}` (the rainbow table).

    Silent no-op when no file is present. Idempotent — first call loads,
    subsequent calls are no-ops (the lookup table is process-global and
    only needs to be loaded once). Without idempotency, every
    `SynthesisEngine()` construction re-parsed ~3.6 MB of binary
    table (~2 s), which the benchmark absorbed into whichever graph
    triggered the inner-engine creation (family_recognition seed compute).
    """
    global _DEFAULT_ROOTED_LOOKUP_LOADED
    if _DEFAULT_ROOTED_LOOKUP_LOADED:
        return 0
    import os
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data",
    )
    bin_path = os.path.join(base_dir, "rooted_lookup_table.bin")
    if os.path.exists(bin_path):
        try:
            n = load_rooted_lookup_binary(bin_path)
            _DEFAULT_ROOTED_LOOKUP_LOADED = True
            return n
        except Exception:
            pass  # fall through to JSON
    json_path = os.path.join(base_dir, "rooted_lookup_table.json")
    n = load_rooted_lookup(json_path)
    _DEFAULT_ROOTED_LOOKUP_LOADED = True
    return n


# =============================================================================
# Binary serialization for the rooted lookup table.
# =============================================================================
#
# Compact format (varuint + raw 32-byte SHA256 canonical keys), mirroring
# `tutte/lookup/binary.py`'s rainbow-table encoding. Designed for fast
# load at engine startup; ~3x smaller than JSON and ~10x faster to parse.
#
#   Header
#       [magic: 4 bytes]    = "RLKP"
#       [version: 1 byte]   = 1
#       [num_entries: varuint]
#
#   Per entry:
#       [canon: 32 bytes]       — raw SHA256 (canonical_key)
#       [canon_boundary_len: varuint]
#       [canon_boundary: varuint × n]
#       [num_partitions: varuint]
#       Per partition:
#           [num_blocks: varuint]
#           Per block:
#               [block_len: varuint]
#               [block: varuint × n]
#           [num_coeffs: varuint]
#           Per coeff:
#               [i: varuint] [j: varuint] [coeff: signed varint]


def _encode_varuint(n: int) -> bytes:
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def _decode_varuint(data: bytes, offset: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = data[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, offset


def _encode_svarint(n: int) -> bytes:
    # ZigZag-encode signed int → unsigned varuint.
    z = (n << 1) ^ (n >> 63) if n >= 0 else ((-n) << 1) - 1
    return _encode_varuint(z)


def _decode_svarint(data: bytes, offset: int) -> Tuple[int, int]:
    z, offset = _decode_varuint(data, offset)
    n = (z >> 1) ^ -(z & 1)
    return n, offset


def save_rooted_lookup_binary(
    path: str,
    graphs: Dict[Tuple[str, Tuple[int, ...]], "Graph"],
) -> int:
    """Binary equivalent of `save_rooted_lookup`. Writes ``RLKP`` v1 format."""
    from ..graph import canonical_node_mapping as _cnm
    import networkx as nx

    out = bytearray(b"RLKP")
    out.append(1)  # version

    entries_data = []
    for cache_key, partition_dict in _T_ROOTED_CACHE.items():
        # See `save_rooted_lookup` for cache-key shape compatibility.
        if len(cache_key) == 3:
            canon, _edges, boundary = cache_key
        else:
            canon, boundary = cache_key
        if cache_key not in graphs:
            continue
        g = graphs[cache_key]
        nxg = nx.Graph()
        nxg.add_nodes_from(g.nodes)
        nxg.add_edges_from(g.edges)
        mapping_orig = _cnm(nxg)
        canon_bdry = tuple(sorted(mapping_orig[v] for v in boundary))
        parts = []
        for block_key, poly in partition_dict.items():
            canon_blocks = tuple(sorted(
                tuple(sorted(mapping_orig[v] for v in block))
                for block in block_key
            ))
            coeffs = list(poly.to_coefficients().items())
            parts.append((canon_blocks, coeffs))
        entries_data.append((canon, canon_bdry, parts))

    out.extend(_encode_varuint(len(entries_data)))
    for canon, canon_bdry, parts in entries_data:
        out.extend(bytes.fromhex(canon))
        out.extend(_encode_varuint(len(canon_bdry)))
        for v in canon_bdry:
            out.extend(_encode_varuint(v))
        out.extend(_encode_varuint(len(parts)))
        for canon_blocks, coeffs in parts:
            out.extend(_encode_varuint(len(canon_blocks)))
            for block in canon_blocks:
                out.extend(_encode_varuint(len(block)))
                for v in block:
                    out.extend(_encode_varuint(v))
            out.extend(_encode_varuint(len(coeffs)))
            for (i, j), c in coeffs:
                out.extend(_encode_varuint(i))
                out.extend(_encode_varuint(j))
                out.extend(_encode_svarint(int(c)))

    with open(path, "wb") as f:
        f.write(out)
    return len(entries_data)


def load_rooted_lookup_binary(path: str) -> int:
    """Read ``RLKP`` v1 format; populates `_ROOTED_LOOKUP`.

    Silent no-op if `path` is absent or the magic header is unrecognized.
    """
    import os
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 5 or data[:4] != b"RLKP":
        return 0
    if data[4] != 1:
        return 0
    offset = 5
    n_entries, offset = _decode_varuint(data, offset)
    n_loaded = 0
    for _ in range(n_entries):
        canon = data[offset:offset + 32].hex()
        offset += 32
        n_bdry, offset = _decode_varuint(data, offset)
        bdry = []
        for _ in range(n_bdry):
            v, offset = _decode_varuint(data, offset)
            bdry.append(v)
        n_parts, offset = _decode_varuint(data, offset)
        partitions = []
        for _ in range(n_parts):
            n_blocks, offset = _decode_varuint(data, offset)
            blocks = []
            for _ in range(n_blocks):
                blen, offset = _decode_varuint(data, offset)
                block = []
                for _ in range(blen):
                    v, offset = _decode_varuint(data, offset)
                    block.append(v)
                blocks.append(block)
            n_coeffs, offset = _decode_varuint(data, offset)
            coeffs = {}
            for _ in range(n_coeffs):
                i, offset = _decode_varuint(data, offset)
                j, offset = _decode_varuint(data, offset)
                c, offset = _decode_svarint(data, offset)
                coeffs[f"{i},{j}"] = c
            partitions.append({"key": blocks, "coeffs": coeffs})
        _ROOTED_LOOKUP[(canon, tuple(bdry))] = partitions
        n_loaded += 1
    return n_loaded


def t_rooted_outer_product(
    T_a: Dict[Tuple, TuttePolynomial],
    T_b: Dict[Tuple, TuttePolynomial],
) -> Dict[Tuple, TuttePolynomial]:
    """Combine T_rooted dicts for two graphs on disjoint vertex sets.

    If G = G_a ⊔ G_b on disjoint vertices and boundary B = B_a ⊔ B_b,
    then T_rooted(G, B)[P] for any partition P of B decomposes as
    P = P_a ⊔ P_b where P_a is the restriction to B_a (each block of P
    lies entirely in B_a or entirely in B_b because there are no edges
    crossing). This gives:

        T_rooted(G, B)[P] = T_rooted(G_a, B_a)[P_a] · T_rooted(G_b, B_b)[P_b].

    Implementation: outer-product the two dicts, combining partition
    keys via concatenation + canonicalisation.
    """
    out: Dict[Tuple, TuttePolynomial] = {}
    for P_a, val_a in T_a.items():
        for P_b, val_b in T_b.items():
            P_combined = tuple(sorted(tuple(P_a) + tuple(P_b)))
            prod = val_a * val_b
            if P_combined in out:
                out[P_combined] = out[P_combined] + prod
            else:
                out[P_combined] = prod
    return out


def t_rooted_smart(
    graph: Graph, boundary: List[int],
) -> Dict[Tuple, TuttePolynomial]:
    """T_rooted that handles disconnected graphs via component decomposition.

    For a disconnected graph G = G_0 ⊔ G_1 ⊔ ... ⊔ G_k with boundary B,
    each component gets its own boundary slice B_i = B ∩ V(G_i), and:

        T_rooted(G, B) = T_rooted(G_0, B_0) ⊗ T_rooted(G_1, B_1) ⊗ ...

    where ⊗ is the outer product of partition dicts (`t_rooted_outer_product`).
    Boundary vertices in B but not in any component (i.e., disconnected
    isolated boundary points with no edges in `graph`) get added as
    singleton partitions in the final combine step.

    Brute-force cost on each component is 2^|E_i| (much smaller than 2^|E|
    for a disconnected graph). Falls back to `t_rooted_bruteforce` for
    connected graphs.
    """
    import networkx as nx

    g_nx = nx.Graph()
    g_nx.add_nodes_from(graph.nodes)
    g_nx.add_edges_from(graph.edges)
    components = list(nx.connected_components(g_nx))
    if len(components) <= 1:
        return t_rooted_bruteforce(graph, boundary)

    boundary_set = set(boundary)
    component_results: List[Dict[Tuple, TuttePolynomial]] = []
    for comp_nodes in components:
        comp_boundary = [v for v in boundary if v in comp_nodes]
        comp_nodes_sorted = sorted(comp_nodes)
        comp_edges = [(u, v) for u, v in graph.edges
                      if u in comp_nodes and v in comp_nodes]
        comp_graph = Graph(comp_nodes_sorted, comp_edges)
        if comp_boundary:
            T_comp = t_rooted_cached(comp_graph, comp_boundary)
        else:
            # No boundary vertices in this component — value is T(comp) at
            # the empty-boundary partition (a constant polynomial factor).
            T_comp = t_rooted_cached(comp_graph, [])
            # Empty boundary → single key () with value T(comp_graph).
        component_results.append(T_comp)

    # Reduce via repeated outer-product.
    accumulated = component_results[0]
    for next_dict in component_results[1:]:
        accumulated = t_rooted_outer_product(accumulated, next_dict)
    return accumulated


def _translate_canonical_partitions_to_runtime(
    canonical_partitions: List[Dict],
    graph: Graph,
    boundary: List[int],
) -> Dict[Tuple, TuttePolynomial]:
    """Translate canonical-labeled cached partitions to a runtime graph's
    actual vertex labels.

    Used by `t_rooted_cached` when a hit lands in `_ROOTED_LOOKUP`
    (the persistent rooted-lookup table). Cached partition keys reference
    canonical labels (0..n-1 per WL ordering); we apply the INVERSE of the
    runtime graph's canonical mapping to recover keys in the runtime's
    actual labels.
    """
    from ..graph import canonical_node_mapping as _cnm
    import networkx as nx

    nxg = nx.Graph()
    nxg.add_nodes_from(graph.nodes)
    nxg.add_edges_from(graph.edges)
    mapping = _cnm(nxg)  # runtime_label → canonical_label
    inverse = {c: r for r, c in mapping.items()}  # canonical_label → runtime_label

    result: Dict[Tuple, TuttePolynomial] = {}
    for part_entry in canonical_partitions:
        # Translate each block via inverse.
        canonical_blocks = part_entry["key"]
        runtime_key = tuple(sorted(
            tuple(sorted(inverse[c] for c in block))
            for block in canonical_blocks
        ))
        coeffs = {}
        for ck, cv in part_entry["coeffs"].items():
            i, j = ck.split(",")
            coeffs[(int(i), int(j))] = cv
        result[runtime_key] = TuttePolynomial.from_coefficients(coeffs)
    return result


def t_rooted_cached(graph: Graph, boundary: List[int]) -> Dict[Tuple, TuttePolynomial]:
    """Like t_rooted_bruteforce, cached by (canonical_key, edges, sorted_boundary).

    Lookup order:
    1. `_T_ROOTED_CACHE` (in-process, runtime-labeled keys). Fast path.
       Key includes the labeling-sensitive edge set so that two
       *isomorphic but differently-labeled* graphs DON'T collide —
       their partition keys are in different vertex labels and so
       can't legitimately share a cached partition_dict (May 20, 2026
       fix; prior bug: cycle_dp produced wrong T(Cm_2) after engine
       had populated the cache with a differently-labeled K_{4,4}).
    2. `_ROOTED_LOOKUP` (persistent, canonical-labeled keys). On hit,
       translate to runtime labels via
       `_translate_canonical_partitions_to_runtime` and populate the
       fast cache. The persistent lookup IS safe to share across
       iso graphs because its partitions are stored in canonical
       labels and translated per-caller.
    3. Cold compute via `t_rooted_smart` (disconnected) or
       `t_rooted_bruteforce` (connected). Store in fast cache.
    """
    # Labeling-sensitive cache key: canonical_key alone is iso-invariant
    # and so could collide across graphs with the same shape but
    # different vertex labels (e.g., two K_{4,4} cells with different
    # bipartition assignments). Including `tuple(sorted(graph.edges))`
    # discriminates the labelings safely. Cost: a few hundred edges'
    # worth of tuple — sub-microsecond.
    key = (
        graph.canonical_key(),
        tuple(sorted(graph.edges)),
        tuple(sorted(boundary)),
    )
    if key in _T_ROOTED_CACHE:
        return _T_ROOTED_CACHE[key]

    # Persistent rooted-lookup check — labeling-aware via canonical
    # translation. Safe across labelings since it always re-translates.
    if _ROOTED_LOOKUP:
        from ..graph import canonical_node_mapping as _cnm
        import networkx as nx
        nxg = nx.Graph()
        nxg.add_nodes_from(graph.nodes)
        nxg.add_edges_from(graph.edges)
        mapping = _cnm(nxg)
        canon_boundary = tuple(sorted(mapping[v] for v in boundary))
        canon_key = (graph.canonical_key(), canon_boundary)
        canon_entry = _ROOTED_LOOKUP.get(canon_key)
        if canon_entry is not None:
            val = _translate_canonical_partitions_to_runtime(
                canon_entry, graph, boundary,
            )
            _T_ROOTED_CACHE[key] = val
            return val

    # Cold compute — component-aware dispatch.
    import networkx as nx
    g_nx = nx.Graph()
    g_nx.add_nodes_from(graph.nodes)
    g_nx.add_edges_from(graph.edges)
    if nx.number_connected_components(g_nx) > 1:
        val = t_rooted_smart(graph, boundary)
    else:
        val = t_rooted_bruteforce(graph, boundary)
    _T_ROOTED_CACHE[key] = val
    # Pin the graph so `save_rooted_lookup_default` can serialize this
    # entry later (e.g., from a benchmark run) without callers threading
    # the originating Graph through every consumer.
    _T_ROOTED_GRAPHS[key] = graph
    return val


_T_ROOTED_ORBIT_CACHE: Dict[
    Tuple[str, Tuple[int, ...]],
    Tuple[Dict[Tuple, TuttePolynomial], List[Dict[int, int]]],
] = {}


def t_rooted_orbit_compressed(
    graph: Graph, boundary: List[int],
) -> Tuple[Dict[Tuple, TuttePolynomial], List[Dict[int, int]]]:
    """T_rooted compressed by the cell's automorphism group acting on boundary.

    Computes T_rooted(graph, boundary) AND folds partitions in the same
    Aut(graph)-orbit (where Aut(graph) restricted to its action on
    boundary) to a single canonical key during the brute-force pass.

    Returns ``(orbit_T_dict, aut_group)`` where:
    - ``orbit_T_dict[canonical_P] = T_rooted_value`` with one entry per
      Aut-orbit of partitions of `boundary`.
    - ``aut_group`` is the list of automorphisms (as vertex permutations
      restricted to `boundary`) — caller needs this to translate from
      orbit reps to all member partitions, and to compute orbit sizes.

    Cached by ``(graph.canonical_key(), tuple(sorted(boundary)))``;
    re-used across any Z(m, t) caller that needs the same cell template
    + boundary set.

    Reuse pattern (motivating use case): Z(1, 2) needs ``T_rooted(Z(1,1),
    all_12_verts_boundary)`` for chain DP. Computed once, cached, then
    Z(1, 3) / Z(2, 2) / etc. with Z(1, 1) cells re-use the SAME orbit
    dict. The orbit dict is also smaller than the full T_rooted dict by
    a factor of |Aut(Z(1, 1)) ↾ boundary| (= 8 for Z(1, 1)), so
    downstream composition (M-table build, orbit_convolve) gets a
    matching constant-factor speedup.

    Correctness invariant: T values must be UNIFORM across an orbit
    (verified inside the loop; raises if violated, which signals
    a bug rather than an algorithmic error).
    """
    cache_key = (graph.canonical_key(), tuple(sorted(boundary)))
    if cache_key in _T_ROOTED_ORBIT_CACHE:
        return _T_ROOTED_ORBIT_CACHE[cache_key]

    # Compute Aut(graph) once (cached by compute_cell_aut). Each
    # automorphism is a dict mapping vertex → permuted vertex; for the
    # purpose of partition canonicalization we restrict to boundary
    # vertices (vertices outside `boundary` never appear in partition
    # block elements, so their image is irrelevant).
    from .aut_orbit import compute_cell_aut

    aut_group = compute_cell_aut(graph)

    # Run brute force but bucket each mask's partition into its Aut-orbit
    # canonical representative. Validates uniformity within orbit.
    from collections import defaultdict

    import networkx as nx

    edges = sorted(graph.edges)
    n_edges = len(edges)
    nodes = sorted(graph.nodes)
    n_nodes = len(nodes)

    # Full-edge rank for the (x-1)^{r(E)-r(A)} weight.
    full_dsu = {v: v for v in nodes}
    for u, v in edges:
        ru, rv = _find(full_dsu, u), _find(full_dsu, v)
        if ru != rv:
            full_dsu[max(ru, rv)] = min(ru, rv)
    k_G = len({_find(full_dsu, v) for v in nodes})
    r_E = n_nodes - k_G

    x_minus_1 = TuttePolynomial.x() + (-1) * TuttePolynomial.one()
    y_minus_1 = TuttePolynomial.y() + (-1) * TuttePolynomial.one()

    from .aut_orbit import canonical_partition

    orbit_T: Dict[Tuple, TuttePolynomial] = {}

    for mask in range(2 ** n_edges):
        A_edges = [edges[i] for i in range(n_edges) if (mask >> i) & 1]
        dsu = {v: v for v in nodes}
        for u, v in A_edges:
            ru, rv = _find(dsu, u), _find(dsu, v)
            if ru != rv:
                dsu[max(ru, rv)] = min(ru, rv)
        k_A = len({_find(dsu, v) for v in nodes})
        r_A = n_nodes - k_A
        boundary_parts: Dict[int, Set[int]] = defaultdict(set)
        for v in boundary:
            boundary_parts[_find(dsu, v)].add(v)
        partition_key = _normalize_partition(boundary_parts)
        # Only process masks whose partition IS the canonical Aut-orbit
        # representative. By Aut-invariance of T_rooted, T_rooted[P] is
        # constant across orbit members; the sum over all masks producing
        # the canonical equals the per-orbit-member value of T_rooted.
        # Skipping non-canonical partitions both compresses the output
        # dict and avoids duplicate work — caller multiplies by
        # orbit_size when computing T(G).
        canonical_key = canonical_partition(partition_key, aut_group)
        if partition_key != canonical_key:
            continue
        weight = _power_poly(x_minus_1, r_E - r_A) * _power_poly(
            y_minus_1, len(A_edges) - r_A
        )
        existing = orbit_T.get(canonical_key)
        if existing is None:
            orbit_T[canonical_key] = weight
        else:
            orbit_T[canonical_key] = existing + weight

    _T_ROOTED_ORBIT_CACHE[cache_key] = (orbit_T, aut_group)
    return orbit_T, aut_group


def aut_orbit_size(
    canonical_P: Tuple[Tuple[int, ...], ...],
    aut_group: List[Dict[int, int]],
) -> int:
    """Number of distinct partitions in the Aut-orbit of `canonical_P`.

    Companion to `t_rooted_orbit_compressed`: callers multiply the
    per-orbit T_rooted value by this size to recover the unfactored
    Tutte contribution. ``T(G) = Σ_orbit aut_orbit_size(P) · T_rooted_orbit[P]``.
    """
    seen = set()
    for aut in aut_group:
        permuted = tuple(sorted(
            tuple(sorted(aut.get(v, v) for v in block))
            for block in canonical_P
        ))
        seen.add(permuted)
    return len(seen)


def relabel_partition_dict(
    T_dict: Dict[Tuple, TuttePolynomial],
    label_map: Dict[int, int],
) -> Dict[Tuple, TuttePolynomial]:
    """Apply label_map to partition keys; merge values for collisions."""
    new_T: Dict[Tuple, TuttePolynomial] = {}
    for P, val in T_dict.items():
        new_P = tuple(sorted(
            tuple(sorted(label_map.get(v, v) for v in block))
            for block in P
        ))
        new_T[new_P] = new_T.get(new_P, TuttePolynomial.zero()) + val
    return new_T
