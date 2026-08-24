"""Treewidth-Based Tutte Polynomial Computation.

For graphs with bounded treewidth w, computes the Tutte polynomial in
O(n * B(w+1)^2) time using DP on a tree decomposition, where B is the
Bell number. This is vastly faster than deletion-contraction for graphs
with treewidth 4-5 (B(5)=52, B(6)=203).

Algorithm: Fortuin-Kasteleyn / rank-nullity formulation.
We compute T(x,y) = sum_{A subset E} (x-1)^{r(E)-r(A)} * (y-1)^{|A|-r(A)}
via DP on set partitions of bag vertices tracking connectivity.

The DP state is a partition of the bag vertices (encoding which vertices
are connected by selected edges). We work in the (x-1, y-1) basis and
expand to standard (x, y) basis at the end.
"""

from __future__ import annotations

from collections import defaultdict as _defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..graph import MultiGraph
from ..logs import EventType, LogLevel, get_log
from ..polynomial import TuttePolynomial

# Try to import C-accelerated partition merge (5x faster for tw >= 8)
try:
    from ._treewidth_c import c_batch_merge as _c_batch_merge
except Exception:
    _c_batch_merge = None


# =============================================================================
# SET PARTITION UTILITIES
# =============================================================================

def canonicalize(labels: Tuple[int, ...]) -> Tuple[int, ...]:
    """Renumber partition labels in first-appearance order.

    Example: (2, 0, 2, 1) -> (0, 1, 0, 2)
    """
    mapping: Dict[int, int] = {}
    next_id = 0
    result = []
    for lbl in labels:
        if lbl not in mapping:
            mapping[lbl] = next_id
            next_id += 1
        result.append(mapping[lbl])
    return tuple(result)


# =============================================================================
# INTEGER-ENCODED PARTITIONS (fast hashing/equality for bags up to 16 elements)
# =============================================================================
# Encode partition as single int: 4 bits per element, supports up to 16 elements
# with labels 0-15. This makes dict operations ~2x faster than tuple keys.

def encode_partition(labels: Tuple[int, ...]) -> int:
    """Encode canonical partition as single integer. Max 16 elements."""
    result = len(labels)  # Store length in bits 0-3 (low nibble of "header")
    for i, lbl in enumerate(labels):
        result |= (lbl << (4 + i * 4))
    return result


def decode_partition(encoded: int) -> Tuple[int, ...]:
    """Decode integer back to partition tuple."""
    n = encoded & 0xF
    return tuple((encoded >> (4 + i * 4)) & 0xF for i in range(n))


def encode_canonicalize(labels: Tuple[int, ...]) -> int:
    """Canonicalize and encode in one step."""
    return encode_partition(canonicalize(labels))


def encoded_merge_for_conn(enc: int, conn_key: Tuple[Tuple[int, int], ...],
                            shared_positions: List[int]) -> int:
    """Merge partition labels at shared positions according to conn_key, directly on encoded int.

    conn_key is a list of (ci, cj) pairs meaning shared_positions[ci] and
    shared_positions[cj] should be in the same block. Avoids
    decode_partition + list manipulation + encode_canonicalize.
    """
    n = enc & 0xF

    # Apply merges directly on the encoded int using bit manipulation
    for ci, cj in conn_key:
        pi = shared_positions[ci]
        pj = shared_positions[cj]
        lbl_pi = (enc >> (4 + pi * 4)) & 0xF
        lbl_pj = (enc >> (4 + pj * 4)) & 0xF
        if lbl_pi != lbl_pj:
            target = min(lbl_pi, lbl_pj)
            replace = max(lbl_pi, lbl_pj)
            # Replace all occurrences of 'replace' with 'target' in the encoded int
            for k in range(n):
                shift = 4 + k * 4
                if (enc >> shift) & 0xF == replace:
                    enc = (enc & ~(0xF << shift)) | (target << shift)

    # Canonicalize: renumber labels in first-appearance order
    # Build mapping from current labels to canonical labels
    mapping = [0] * 16  # max 16 labels
    seen = [False] * 16
    next_id = 0
    result = n
    for i in range(n):
        lbl = (enc >> (4 + i * 4)) & 0xF
        if not seen[lbl]:
            seen[lbl] = True
            mapping[lbl] = next_id
            next_id += 1
        result |= (mapping[lbl] << (4 + i * 4))
    return result


def encoded_connect(enc: int, i: int, j: int) -> int:
    """Merge blocks containing positions i and j in encoded partition."""
    n = enc & 0xF
    li = (enc >> (4 + i * 4)) & 0xF
    lj = (enc >> (4 + j * 4)) & 0xF
    if li == lj:
        return enc
    target = min(li, lj)
    replace = max(li, lj)
    # Replace labels directly in the integer, then canonicalize
    labels = []
    for k in range(n):
        lbl = (enc >> (4 + k * 4)) & 0xF
        labels.append(target if lbl == replace else lbl)
    return encode_canonicalize(tuple(labels))


# Cache for encoded_connect to avoid recomputation on repeated partition+edge pairs
_connect_cache: Dict[Tuple[int, int, int], int] = {}
_CONNECT_CACHE_MAX = 2_000_000


def encoded_connect_cached(enc: int, i: int, j: int) -> int:
    """Cached version of encoded_connect for hot inner loops."""
    key = (enc, i, j)
    result = _connect_cache.get(key)
    if result is not None:
        return result
    result = encoded_connect(enc, i, j)
    if len(_connect_cache) < _CONNECT_CACHE_MAX:
        _connect_cache[key] = result
    return result


def encoded_forget(enc: int, i: int) -> Tuple[bool, int]:
    """Remove element at position i from encoded partition.

    Returns (was_singleton, new_encoded).
    """
    n = enc & 0xF
    lbl_i = (enc >> (4 + i * 4)) & 0xF
    # Check if singleton
    count = 0
    labels = []
    for k in range(n):
        lbl = (enc >> (4 + k * 4)) & 0xF
        if lbl == lbl_i:
            count += 1
        if k != i:
            labels.append(lbl)
    was_singleton = count == 1
    return was_singleton, encode_canonicalize(tuple(labels))


def connect(labels: Tuple[int, ...], i: int, j: int) -> Tuple[int, ...]:
    """Merge the blocks containing positions i and j."""
    li, lj = labels[i], labels[j]
    if li == lj:
        return labels
    # Replace all occurrences of lj with li
    target = min(li, lj)
    replace = max(li, lj)
    return canonicalize(tuple(target if l == replace else l for l in labels))


def forget(labels: Tuple[int, ...], i: int) -> Tuple[bool, Tuple[int, ...]]:
    """Remove element at position i from the partition.

    Returns (was_singleton, new_partition).
    A singleton is an element whose label appears exactly once.
    """
    lbl = labels[i]
    was_singleton = labels.count(lbl) == 1
    new_labels = labels[:i] + labels[i+1:]
    return was_singleton, canonicalize(new_labels)


# =============================================================================
# POLYNOMIAL ARITHMETIC IN (x-1, y-1) BASIS
# =============================================================================
# We represent polynomials as Dict[(a_pow, b_pow), coeff] where
# a = (x-1), b = (y-1). This avoids division when converting Z -> T.

Poly = Dict[Tuple[int, int], int]


def poly_zero() -> Poly:
    return {}


def poly_one() -> Poly:
    return {(0, 0): 1}


def poly_add(p: Poly, q: Poly) -> Poly:
    result = dict(p)
    for k, v in q.items():
        s = result.get(k, 0) + v
        if s:
            result[k] = s
        elif k in result:
            del result[k]
    return result


def poly_add_inplace(dst: Poly, src: Poly) -> None:
    """Add src into dst in-place. Avoids dict copy for accumulation patterns."""
    for k, v in src.items():
        s = dst.get(k, 0) + v
        if s:
            dst[k] = s
        elif k in dst:
            del dst[k]


def poly_mul_monomial(p: Poly, a_pow: int, b_pow: int) -> Poly:
    """Multiply polynomial by a^a_pow * b^b_pow."""
    return {(k[0] + a_pow, k[1] + b_pow): v for k, v in p.items()}


def _poly_scale(p: Poly, c: int) -> Poly:
    """Multiply polynomial by integer constant."""
    if c == 0:
        return {}
    return {k: v * c for k, v in p.items()}


def _poly_content_key(p: Poly) -> frozenset:
    """Content-based identity key for polynomial deduplication."""
    return frozenset(p.items())


def _power_1_plus_b(k: int) -> Poly:
    """Compute (1 + b)^k = sum C(k,j) b^j via binomial theorem.

    Used for batch introduction of k parallel edges.
    """
    from math import comb
    return {(0, j): comb(k, j) for j in range(k + 1)}


# =============================================================================
# TREE DECOMPOSITION
# =============================================================================

@dataclass
class TreeDecomposition:
    """Tree decomposition of a graph.

    bags: list of frozenset of vertices (each bag)
    tree_adj: adjacency list of the tree (bag index -> set of bag indices)
    root: index of root bag
    bag_edges: for each bag index, list of (u, v, multiplicity) edges assigned to it
    width: treewidth (max bag size - 1)
    """
    bags: List[frozenset]
    tree_adj: Dict[int, Set[int]]
    root: int
    bag_edges: Dict[int, List[Tuple[int, int, int]]]  # bag_idx -> [(u, v, mult)]
    width: int


BELL = [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570, 4213597]


def _elimination_ordering(
    adj: Dict[int, Set[int]],
    nodes: List[int],
    heuristic: str = "minfill",
    seed: Optional[int] = None,
    max_width: int = 20,
) -> Optional[List[int]]:
    """Compute elimination ordering using the given heuristic.

    Heuristics:
    - "minfill": Min fill-in, first-found tie-break (original behavior)
    - "minfill_random": Min fill-in, random tie-break using seed
    - "mindegree": Min degree (fewest neighbors)
    - "minfill_degree": Min fill-in, then min degree tie-break

    Returns None if any bag would exceed max_width + 1.
    """
    import random as _random

    n = len(nodes)
    remaining = set(nodes)
    elim_order = []
    elim_adj = {v: set(adj[v]) for v in nodes}

    rng = _random.Random(seed) if seed is not None else None

    for _ in range(n):
        if heuristic == "mindegree":
            # Pick vertex with fewest remaining neighbors
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = len(elim_adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
        elif heuristic == "minfill_random":
            # Min fill-in with random tie-breaking
            best_fill = float('inf')
            tied = []
            for v in remaining:
                neighbors_in = elim_adj[v] & remaining
                fill = 0
                nb_list = list(neighbors_in)
                for i in range(len(nb_list)):
                    for j in range(i + 1, len(nb_list)):
                        if nb_list[j] not in elim_adj[nb_list[i]]:
                            fill += 1
                if fill < best_fill:
                    best_fill = fill
                    tied = [v]
                elif fill == best_fill:
                    tied.append(v)
            best_v = rng.choice(tied) if rng else tied[0]
        elif heuristic == "minfill_degree":
            # Min fill-in, then min degree tie-break
            best_v = None
            best_fill = float('inf')
            best_deg = float('inf')
            for v in remaining:
                neighbors_in = elim_adj[v] & remaining
                fill = 0
                nb_list = list(neighbors_in)
                for i in range(len(nb_list)):
                    for j in range(i + 1, len(nb_list)):
                        if nb_list[j] not in elim_adj[nb_list[i]]:
                            fill += 1
                deg = len(neighbors_in)
                if fill < best_fill or (fill == best_fill and deg < best_deg):
                    best_fill = fill
                    best_deg = deg
                    best_v = v
        else:
            # Default: minfill with first-found tie-break
            best_v = None
            best_fill = float('inf')
            for v in remaining:
                neighbors_in = elim_adj[v] & remaining
                fill = 0
                nb_list = list(neighbors_in)
                for i in range(len(nb_list)):
                    for j in range(i + 1, len(nb_list)):
                        if nb_list[j] not in elim_adj[nb_list[i]]:
                            fill += 1
                if fill < best_fill:
                    best_fill = fill
                    best_v = v

        # Check bag size
        neighbors_in = elim_adj[best_v] & remaining
        bag_size = 1 + len(neighbors_in)
        if bag_size > max_width + 1:
            return None

        elim_order.append(best_v)

        # Make neighbors a clique (fill-in)
        nb_list = list(neighbors_in)
        for i in range(len(nb_list)):
            for j in range(i + 1, len(nb_list)):
                u, w = nb_list[i], nb_list[j]
                elim_adj[u].add(w)
                elim_adj[w].add(u)

        # Remove v
        remaining.remove(best_v)
        for nb in list(elim_adj[best_v]):
            elim_adj[nb].discard(best_v)

    return elim_order


def _build_decomposition(
    mg: MultiGraph,
    elim_order: List[int],
    adj: Dict[int, Set[int]],
) -> TreeDecomposition:
    """Build a TreeDecomposition from a multigraph and elimination ordering."""
    nodes = sorted(mg.nodes)
    n = len(nodes)

    # Replay elimination to get bags
    remaining = set(nodes)
    elim_bags: List[frozenset] = []
    elim_adj = {v: set(adj[v]) for v in nodes}

    for v in elim_order:
        neighbors_in = elim_adj[v] & remaining
        bag = frozenset({v} | neighbors_in)
        elim_bags.append(bag)

        # Fill-in
        nb_list = list(neighbors_in)
        for i in range(len(nb_list)):
            for j in range(i + 1, len(nb_list)):
                u, w = nb_list[i], nb_list[j]
                elim_adj[u].add(w)
                elim_adj[w].add(u)

        remaining.remove(v)
        for nb in list(elim_adj[v]):
            elim_adj[nb].discard(v)

    # Build tree structure
    elim_index = {v: i for i, v in enumerate(elim_order)}
    bag_list = list(elim_bags)
    tree_adj: Dict[int, Set[int]] = {i: set() for i in range(n)}

    for i in range(n - 1):
        v = elim_order[i]
        bag = bag_list[i]
        remaining_vertices = bag - {v}
        if not remaining_vertices:
            tree_adj[i].add(i + 1)
            tree_adj[i + 1].add(i)
            continue
        parent_idx = min(elim_index[u] for u in remaining_vertices)
        tree_adj[i].add(parent_idx)
        tree_adj[parent_idx].add(i)

    root = n - 1

    # Assign edges to bags
    bag_edges: Dict[int, List[Tuple[int, int, int]]] = {i: [] for i in range(n)}

    for (u, v), mult in mg.edge_counts.items():
        idx_u, idx_v = elim_index[u], elim_index[v]
        bag_idx = max(idx_u, idx_v)
        if u in bag_list[bag_idx] and v in bag_list[bag_idx]:
            bag_edges[bag_idx].append((u, v, mult))
        else:
            for i in range(n):
                if u in bag_list[i] and v in bag_list[i]:
                    bag_edges[i].append((u, v, mult))
                    break

    # Assign loops
    for node, count in mg.loop_counts.items():
        bag_idx = elim_index[node]
        bag_edges[bag_idx].append((node, node, count))

    width = max(len(b) - 1 for b in bag_list)

    return TreeDecomposition(
        bags=bag_list,
        tree_adj=tree_adj,
        root=root,
        bag_edges=bag_edges,
        width=width,
    )


def estimate_dp_cost(td: TreeDecomposition) -> float:
    """Estimate the DP cost of a tree decomposition without running the full DP.

    Cost model: for each bag, Bell(bag_size) * 1.6^n_edges for edge introduction
    (exponential because each edge can roughly double partition count),
    plus Bell(bag_size) * Bell(child_bag_size) for each child merge.
    """
    n_bags = len(td.bags)

    # Build children from rooted tree
    children: Dict[int, List[int]] = {i: [] for i in range(n_bags)}
    visited: Set[int] = set()

    def build_children(bag_idx: int):
        visited.add(bag_idx)
        for nb in td.tree_adj[bag_idx]:
            if nb not in visited:
                children[bag_idx].append(nb)
                build_children(nb)

    build_children(td.root)

    cost = 0.0
    for i in range(n_bags):
        bag_size = len(td.bags[i])
        b = BELL[min(bag_size, len(BELL) - 1)]
        n_edges = len(td.bag_edges[i])
        # Edge introduction cost: exponential in edges (each edge can
        # roughly double active partition count; 1.6 accounts for edges
        # between already-connected vertices not creating new partitions)
        cost += b * (1.6 ** n_edges)
        # Child merge cost: Cartesian product of parent and child tables
        for child in children[i]:
            child_size = len(td.bags[child])
            child_b = BELL[min(child_size, len(BELL) - 1)]
            cost += b * child_b

    return cost


def _redistribute_edges(td: TreeDecomposition, mg: MultiGraph) -> TreeDecomposition:
    """Redistribute edges between bags to minimize DP cost.

    For each edge, find all bags containing both endpoints and assign to the
    bag that minimizes the total estimated cost (fewest edges in the bag).
    """
    n_bags = len(td.bags)

    # Find all valid bags for each edge
    edge_candidates: Dict[Tuple[int, int, int], List[int]] = {}
    all_edges = []
    for bag_idx, edges in td.bag_edges.items():
        for e in edges:
            if e not in edge_candidates:
                edge_candidates[e] = []
                all_edges.append(e)

    # For each edge, find ALL bags that contain both endpoints
    for e in all_edges:
        u, v, mult = e
        edge_candidates[e] = []
        for i in range(n_bags):
            if u in td.bags[i] and v in td.bags[i]:
                edge_candidates[e].append(i)

    # If every edge has only one candidate bag, nothing to redistribute
    if all(len(cands) == 1 for cands in edge_candidates.values()):
        return td

    # Greedy assignment: assign each edge to the candidate bag with fewest
    # already-assigned edges, weighted by Bell number of bag size
    new_bag_edges: Dict[int, List[Tuple[int, int, int]]] = {i: [] for i in range(n_bags)}
    bag_edge_count = {i: 0 for i in range(n_bags)}

    # Sort edges so edges with fewer candidate bags are assigned first
    sorted_edges = sorted(all_edges, key=lambda e: len(edge_candidates[e]))

    for e in sorted_edges:
        candidates = edge_candidates[e]
        if len(candidates) == 1:
            best = candidates[0]
        else:
            # Pick candidate that minimizes Bell(bag_size) * 1.6^(current_edges + 1)
            best = min(
                candidates,
                key=lambda i: BELL[min(len(td.bags[i]), len(BELL) - 1)] * (1.6 ** (bag_edge_count[i] + 1)),
            )
        new_bag_edges[best].append(e)
        bag_edge_count[best] += 1

    return TreeDecomposition(
        bags=td.bags,
        tree_adj=td.tree_adj,
        root=td.root,
        bag_edges=new_bag_edges,
        width=td.width,
    )


def compute_tree_decomposition(mg: MultiGraph, max_width: int = 20) -> Optional[TreeDecomposition]:
    """Compute tree decomposition via greedy min-fill-in elimination.

    Returns None if treewidth exceeds max_width.
    """
    if not mg.nodes:
        return None

    nodes = sorted(mg.nodes)
    n = len(nodes)

    if n == 1:
        bag = frozenset(nodes)
        node = nodes[0]
        loop_edges = []
        if node in mg.loop_counts:
            loop_edges.append((node, node, mg.loop_counts[node]))
        return TreeDecomposition(
            bags=[bag],
            tree_adj={0: set()},
            root=0,
            bag_edges={0: loop_edges},
            width=0,
        )

    # Build simple adjacency
    adj: Dict[int, Set[int]] = {v: set() for v in nodes}
    for (u, v) in mg.edge_counts:
        adj[u].add(v)
        adj[v].add(u)

    ordering = _elimination_ordering(adj, nodes, heuristic="minfill", max_width=max_width)
    if ordering is None:
        return None

    return _build_decomposition(mg, ordering, adj)


def compute_best_tree_decomposition(
    mg: MultiGraph, max_width: int = 20
) -> Optional[TreeDecomposition]:
    """Try multiple elimination orderings and return the best decomposition.

    Strategy: minfill + minfill_degree + 20 random seeds, select by minimum
    treewidth then minimum estimated DP cost. Apply edge redistribution to
    the winner to further reduce cost.

    Overhead: ~20ms for 23-node graphs (vs seconds-minutes for the DP).
    """
    if not mg.nodes:
        return None

    nodes = sorted(mg.nodes)
    n = len(nodes)

    if n == 1:
        return compute_tree_decomposition(mg, max_width)

    # Build simple adjacency
    adj: Dict[int, Set[int]] = {v: set() for v in nodes}
    for (u, v) in mg.edge_counts:
        adj[u].add(v)
        adj[v].add(u)

    best_td: Optional[TreeDecomposition] = None
    best_cost: float = float('inf')

    def _consider(td: TreeDecomposition):
        nonlocal best_td, best_cost
        if td.width > max_width:
            return
        if best_td is not None and td.width > best_td.width:
            return
        cost = estimate_dp_cost(td)
        if best_td is None or td.width < best_td.width or cost < best_cost:
            best_td = td
            best_cost = cost

    # minfill deterministic (good baseline)
    ordering = _elimination_ordering(adj, nodes, heuristic="minfill", max_width=max_width)
    if ordering is not None:
        _consider(_build_decomposition(mg, ordering, adj))

    # minfill_degree deterministic (min fill-in with min degree tie-break)
    ordering = _elimination_ordering(adj, nodes, heuristic="minfill_degree", max_width=max_width)
    if ordering is not None:
        _consider(_build_decomposition(mg, ordering, adj))

    # mindegree deterministic baseline. Sometimes finds a different elimination
    # ordering than minfill (e.g. on graphs with high-fill-in vertices that
    # are nonetheless low-degree). Cheap (~ms) — always worth trying.
    ordering = _elimination_ordering(adj, nodes, heuristic="mindegree", max_width=max_width)
    if ordering is not None:
        _consider(_build_decomposition(mg, ordering, adj))

    # minfill with random tie-breaking.
    # Search budget scales with treewidth — per-bag DP cost is
    # ~Bell(tw+1) × 2^(tw+1), so each width reduction in the chosen
    # ordering compounds the savings. Decomposition-ordering search is
    # O(20ms × n_seeds) which dwarfs the DP cost only for tw ≤ 5; for
    # tw ≥ 12 the search itself amortizes against multi-minute DPs.
    #
    # Early-exit: after N consecutive seeds without improvement OR a
    # time budget, stop. Avoids burning seeds when the heuristic has
    # converged — common for highly symmetric graphs like
    # K_{n,n}-derived families where minfill_random hits the same local
    # optimum repeatedly.
    if best_td is None:
        n_seeds = 20
        no_improve_cap = 20  # essentially: no early exit if nothing found
        time_budget = None
    elif best_td.width >= 12:
        n_seeds = 500   # tw 12-15 band: per-point DP runs in minutes
        no_improve_cap = 80
        time_budget = 30.0  # seconds
    elif best_td.width >= 9:
        n_seeds = 200
        no_improve_cap = 60
        time_budget = 10.0
    elif best_td.width >= 8:
        n_seeds = 50
        no_improve_cap = 30
        time_budget = 5.0
    else:
        n_seeds = 20
        no_improve_cap = 20
        time_budget = None

    import time as _time
    deadline = (_time.time() + time_budget) if time_budget is not None else None
    no_improve = 0
    for seed in range(n_seeds):
        prev_best_width = best_td.width if best_td is not None else float("inf")
        prev_best_cost = best_cost
        ordering = _elimination_ordering(
            adj, nodes, heuristic="minfill_random", seed=seed, max_width=max_width
        )
        if ordering is not None:
            _consider(_build_decomposition(mg, ordering, adj))
        new_best_width = best_td.width if best_td is not None else float("inf")
        if new_best_width < prev_best_width or best_cost < prev_best_cost:
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= no_improve_cap:
            break
        if deadline is not None and _time.time() > deadline:
            break

    # Apply edge redistribution to minimize DP cost
    if best_td is not None:
        best_td = _redistribute_edges(best_td, mg)

    return best_td


# =============================================================================
# DP ON TREE DECOMPOSITION
# =============================================================================

# DP table: maps encoded partition (int) to polynomial in (a, b) basis
# where a = (x-1), b = (y-1).
# Encoded partitions use 4 bits per element for fast hashing/equality.
DPTable = Dict[int, Poly]


def compute_treewidth_tutte(td: TreeDecomposition, mg: MultiGraph) -> TuttePolynomial:
    """Compute Tutte polynomial via DP on tree decomposition.

    Uses the rank-nullity formulation:
    T(x,y) = sum_{A subset E} (x-1)^{r(E)-r(A)} * (y-1)^{|A|-r(A)}

    where r(A) = |V_processed| - num_connected_components(A).
    """
    n_bags = len(td.bags)

    # Compute rank of entire graph: r(E) = |V| - k(G)
    # where k(G) = number of connected components
    visited: Set[int] = set()
    num_components = 0
    for start in mg.nodes:
        if start in visited:
            continue
        num_components += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nb in mg.neighbors(node):
                if nb not in visited:
                    stack.append(nb)
    # Build children for rooted tree
    children: Dict[int, List[int]] = {i: [] for i in range(n_bags)}
    visited_bags: Set[int] = set()

    def build_children(bag_idx: int):
        visited_bags.add(bag_idx)
        for nb in td.tree_adj[bag_idx]:
            if nb not in visited_bags:
                children[bag_idx].append(nb)
                build_children(nb)

    build_children(td.root)

    # Process bottom-up
    def process_bag(bag_idx: int) -> Tuple[DPTable, List[int]]:
        """Process a bag and return (DP table, vertex ordering in partition).

        The vertex ordering maps partition indices to actual vertex IDs.
        """
        bag = td.bags[bag_idx]
        bag_verts = sorted(bag)
        vert_to_idx = {v: i for i, v in enumerate(bag_verts)}

        # Start with trivial table: all singletons, coefficient 1
        init_partition = encode_partition(canonicalize(tuple(range(len(bag_verts)))))
        table: DPTable = {init_partition: poly_one()}

        # Introduce edges assigned to this bag
        # Separate edges by type for batch processing
        loop_edges = []
        simple_edges = []  # mult == 1, non-loop
        parallel_edges = []  # mult > 1, non-loop
        for (u, v, mult) in td.bag_edges[bag_idx]:
            if u not in bag or v not in bag:
                continue
            if u == v:
                loop_edges.append((u, v, mult))
            elif mult == 1:
                simple_edges.append((u, v, mult))
            else:
                parallel_edges.append((u, v, mult))

        # Process loops
        for (u, v, mult) in loop_edges:
            loop_factor = _power_1_plus_b(mult)
            new_table: DPTable = {}
            for partition, poly in table.items():
                new_poly = _poly_mul(poly, loop_factor)
                if new_poly:
                    new_table[partition] = poly_add(
                        new_table.get(partition, poly_zero()), new_poly
                    )
            table = new_table

        # Process parallel edges (mult > 1)
        for (u, v, mult) in parallel_edges:
            idx_u = vert_to_idx[u]
            idx_v = vert_to_idx[v]
            factor_full = _power_1_plus_b(mult)
            factor_minus_1 = poly_add(factor_full, {(0, 0): -1})
            new_table: DPTable = {}
            for partition, poly in table.items():
                connected_part = encoded_connect_cached(partition, idx_u, idx_v)
                if connected_part == partition:
                    # u,v already connected: weight = (1+b)^k
                    new_poly = _poly_mul(poly, factor_full)
                    if new_poly:
                        new_table[partition] = poly_add(
                            new_table.get(partition, poly_zero()), new_poly
                        )
                else:
                    # All k absent: partition unchanged, weight 1
                    new_table[partition] = poly_add(
                        new_table.get(partition, poly_zero()), poly
                    )
                    # >=1 present: connect u,v, weight (1+b)^k - 1
                    new_poly = _poly_mul(poly, factor_minus_1)
                    if new_poly:
                        new_table[connected_part] = poly_add(
                            new_table.get(connected_part, poly_zero()), new_poly
                        )
            table = new_table

        # Batch introduction of simple edges (mult == 1)
        # Instead of introducing one at a time (each doubling table size),
        # iterate over all 2^k subsets of edges simultaneously.
        # This avoids large intermediate tables and collapses partitions early.
        if simple_edges and len(simple_edges) <= 12:
            edge_indices = [(vert_to_idx[u], vert_to_idx[v])
                           for (u, v, _) in simple_edges]
            k = len(edge_indices)
            new_table: DPTable = {}
            for partition, poly in table.items():
                for mask in range(1 << k):
                    # Compute partition from connecting all edges in subset
                    p = partition
                    for bit in range(k):
                        if mask & (1 << bit):
                            p = encoded_connect_cached(
                                p, edge_indices[bit][0], edge_indices[bit][1]
                            )
                    # Weight: b^popcount(mask)
                    bc = bin(mask).count('1')
                    shifted = poly_mul_monomial(poly, 0, bc) if bc > 0 else poly
                    if p in new_table:
                        poly_add_inplace(new_table[p], shifted)
                    else:
                        new_table[p] = dict(shifted)
            table = new_table
        elif simple_edges:
            # Sequential introduction for k > 12
            for (u, v, _) in simple_edges:
                idx_u = vert_to_idx[u]
                idx_v = vert_to_idx[v]
                new_table: DPTable = {}
                for partition, poly in table.items():
                    # Edge absent
                    if partition in new_table:
                        poly_add_inplace(new_table[partition], poly)
                    else:
                        new_table[partition] = dict(poly)
                    # Edge present: connect u,v, multiply by b
                    new_part = encoded_connect_cached(partition, idx_u, idx_v)
                    new_poly = poly_mul_monomial(poly, 0, 1)
                    if new_part in new_table:
                        poly_add_inplace(new_table[new_part], new_poly)
                    else:
                        new_table[new_part] = dict(new_poly)
                table = new_table

        # Process children: merge child tables, then forget child-only vertices
        for child_idx in children[bag_idx]:
            child_table, child_verts = process_bag(child_idx)

            # Precompute forget indices (positions in child_verts to remove,
            # adjusted for earlier removals)
            forget_indices = []
            offset = 0
            for pos, v in enumerate(child_verts):
                if v not in bag:
                    forget_indices.append(pos - offset)
                    offset += 1

            # Forget vertices from child table
            forgotten_table: DPTable = {}
            for partition, poly in child_table.items():
                current_enc = partition
                current_poly = poly

                for idx in forget_indices:
                    was_singleton, new_enc = encoded_forget(current_enc, idx)
                    if was_singleton:
                        current_poly = poly_mul_monomial(current_poly, 1, 1)
                    current_enc = new_enc

                if current_enc in forgotten_table:
                    poly_add_inplace(forgotten_table[current_enc], current_poly)
                else:
                    forgotten_table[current_enc] = dict(current_poly)

            # Now child table has same vertex set as shared vertices
            shared_verts = [v for v in child_verts if v in bag]

            # Merge with current table
            table = _merge_tables(table, forgotten_table, bag_verts, shared_verts)

        return table, bag_verts

    root_table, root_verts = process_bag(td.root)

    # Forget all remaining vertices in root (indices 0,0,0,... since we
    # always forget position 0 after each removal shifts everything down)
    n_root = len(root_verts)
    final_poly = poly_zero()
    for partition, poly in root_table.items():
        current_enc = partition
        current_poly = poly

        for _ in range(n_root):
            was_singleton, new_enc = encoded_forget(current_enc, 0)
            if was_singleton:
                current_poly = poly_mul_monomial(current_poly, 1, 1)
            current_enc = new_enc

        final_poly = poly_add(final_poly, current_poly)

    # We computed sum_A a^{k(A)} * b^{|A|+k(A)} where a=(x-1), b=(y-1).
    # (Each edge contributes b; each completed component contributes a*b.)
    #
    # T(x,y) = sum_A a^{k(A)-k(E)} * b^{|A|+k(A)-|V|}
    #
    # So: final_poly = a^{k(E)} * b^{|V|} * T(x,y)
    # Therefore: T(x,y) = final_poly / (a^{k(E)} * b^{|V|})

    # Divide by (x-1)^{num_components} * (y-1)^{|V|}
    # In our (a, b) basis, this means shifting exponents down
    n_verts = len(mg.nodes)
    a_shift = num_components
    b_shift = n_verts

    tutte_poly: Poly = {}
    for (a_pow, b_pow), coeff in final_poly.items():
        new_a = a_pow - a_shift
        new_b = b_pow - b_shift
        if new_a < 0 or new_b < 0:
            # This shouldn't happen for valid Tutte polynomials
            # but check just in case
            if coeff != 0:
                raise RuntimeError(
                    f"Negative exponent in Tutte polynomial: "
                    f"a^{new_a} * b^{new_b} with coeff {coeff}"
                )
            continue
        if coeff != 0:
            tutte_poly[(new_a, new_b)] = coeff

    # Convert from (x-1, y-1) basis to (x, y) basis
    return _convert_ab_to_xy(tutte_poly)


def _child_connectivity_key(child_enc: int, n_shared: int) -> Tuple[Tuple[int, int], ...]:
    """Extract which shared vertices are in the same block (connectivity pattern).

    Returns tuple of (i,j) pairs where child_part[i] == child_part[j].
    For n_shared vertices, there are at most Bell(n_shared) unique patterns.
    """
    pairs = []
    for i in range(n_shared):
        li = (child_enc >> (4 + i * 4)) & 0xF
        for j in range(i + 1, n_shared):
            lj = (child_enc >> (4 + j * 4)) & 0xF
            if li == lj:
                pairs.append((i, j))
    return tuple(pairs)


def _merge_tables(
    parent_table: DPTable,
    child_table: DPTable,
    parent_verts: List[int],
    shared_verts: List[int],
) -> DPTable:
    """Merge child table into parent table.

    The child table has been reduced to only shared vertices.
    The child's partition encodes connectivity established through the child
    subtree. This connectivity is applied to the parent partition by merging
    blocks of shared vertices that the child says are connected.

    Optimization: group child partitions by connectivity pattern on shared
    vertices and sum their polynomials first. This reduces the Cartesian
    product from |parent| * |child| to |parent| * Bell(|shared|).
    """
    if not shared_verts:
        # No shared vertices: just multiply polynomials
        child_poly = poly_zero()
        for poly in child_table.values():
            child_poly = poly_add(child_poly, poly)

        result: DPTable = {}
        for partition, poly in parent_table.items():
            new_poly = _poly_mul(poly, child_poly)
            if new_poly:
                if partition in result:
                    poly_add_inplace(result[partition], new_poly)
                else:
                    result[partition] = dict(new_poly)
        return result

    parent_vert_idx = {v: i for i, v in enumerate(parent_verts)}
    shared_positions_in_parent = [parent_vert_idx[v] for v in shared_verts]
    n_shared = len(shared_verts)

    # Group child entries by connectivity pattern on shared vertices
    child_groups: Dict[Tuple[Tuple[int, int], ...], Poly] = {}
    for child_enc, child_poly in child_table.items():
        conn_key = _child_connectivity_key(child_enc, n_shared)
        if conn_key in child_groups:
            poly_add_inplace(child_groups[conn_key], child_poly)
        else:
            child_groups[conn_key] = dict(child_poly)

    result: DPTable = {}

    # Fast path: if only one connectivity group with empty key (all disconnected),
    # no merging needed — just multiply each parent entry by the summed child poly.
    # Dedup: group parents by polynomial identity to avoid redundant _poly_mul calls.
    if len(child_groups) == 1 and () in child_groups:
        child_poly = child_groups[()]
        poly_groups: Dict[frozenset, List[int]] = {}
        poly_by_key: Dict[frozenset, Poly] = {}
        for parent_enc, parent_poly in parent_table.items():
            pk = _poly_content_key(parent_poly)
            if pk in poly_groups:
                poly_groups[pk].append(parent_enc)
            else:
                poly_groups[pk] = [parent_enc]
                poly_by_key[pk] = parent_poly

        for pk, encs in poly_groups.items():
            new_poly = _poly_mul(poly_by_key[pk], child_poly)
            if new_poly:
                for enc in encs:
                    result[enc] = new_poly  # Safe: new_poly is never mutated in-place
        return result

    # Parent grouping optimization: for each conn_key, many parent partitions
    # map to the same merged_enc. By distributive law, sum parent polys first
    # (cheap O(N) poly_add), then multiply once (expensive O(N²) _poly_mul).
    # Reduces _poly_mul calls from |parent| * |child_groups| to
    # |unique_outputs| * |child_groups| (e.g. 4,140 -> 877 for TW=7).
    for conn_key, child_poly in child_groups.items():
        # Group parent partitions by their output merged_enc
        if not conn_key:
            # Poly dedup: group parents by polynomial value.
            # Parent grouping has no benefit here (output == input),
            # but many parents share the same polynomial.
            poly_groups_inner: Dict[frozenset, List[int]] = {}
            poly_by_key_inner: Dict[frozenset, Poly] = {}
            for parent_enc, parent_poly in parent_table.items():
                pk = _poly_content_key(parent_poly)
                if pk in poly_groups_inner:
                    poly_groups_inner[pk].append(parent_enc)
                else:
                    poly_groups_inner[pk] = [parent_enc]
                    poly_by_key_inner[pk] = parent_poly

            for pk, encs in poly_groups_inner.items():
                new_poly = _poly_mul(poly_by_key_inner[pk], child_poly)
                if new_poly:
                    for enc in encs:
                        if enc in result:
                            poly_add_inplace(result[enc], new_poly)
                        else:
                            result[enc] = dict(new_poly)
            continue  # Skip the generic multiply loop below

        parents_by_output: Dict[int, Poly] = {}

        # Use C batch merge if available (5x faster for partition encoding).
        # When the C merger returns (None, None, 0) the encoding exceeds
        # uint64 bounds (tw≥15 bag); fall through to the pure-Python branch.
        c_result = None
        if _c_batch_merge is not None:
            c_result = _c_batch_merge(
                parent_table.keys(), conn_key, shared_positions_in_parent)
            if c_result[0] is None:
                c_result = None
        if c_result is not None:
            enc_list, out_buf, n = c_result
            for i in range(n):
                merged_enc = out_buf[i]
                parent_poly = parent_table[enc_list[i]]
                dst = parents_by_output.get(merged_enc)
                if dst is not None:
                    # Inlined poly_add_inplace — avoids function call overhead
                    for k, v in parent_poly.items():
                        s = dst.get(k, 0) + v
                        if s:
                            dst[k] = s
                        elif k in dst:
                            del dst[k]
                else:
                    parents_by_output[merged_enc] = dict(parent_poly)
        else:
            for parent_enc, parent_poly in parent_table.items():
                merged_enc = encoded_merge_for_conn(
                    parent_enc, conn_key, shared_positions_in_parent)
                dst = parents_by_output.get(merged_enc)
                if dst is not None:
                    for k, v in parent_poly.items():
                        s = dst.get(k, 0) + v
                        if s:
                            dst[k] = s
                        elif k in dst:
                            del dst[k]
                else:
                    parents_by_output[merged_enc] = dict(parent_poly)

        # Now multiply once per unique output (expensive _poly_mul)
        for merged_enc, summed_parent in parents_by_output.items():
            new_poly = _poly_mul(summed_parent, child_poly)
            if new_poly:
                dst = result.get(merged_enc)
                if dst is not None:
                    for k, v in new_poly.items():
                        s = dst.get(k, 0) + v
                        if s:
                            dst[k] = s
                        elif k in dst:
                            del dst[k]
                else:
                    result[merged_enc] = dict(new_poly)

    return result


def _poly_mul(p: Poly, q: Poly) -> Poly:
    """Multiply two polynomials.

    Fast paths for common cases: monomial * poly, scalar * poly.
    """
    if not p or not q:
        return {}
    # Fast path: one side is a monomial (very common in DP)
    lp, lq = len(p), len(q)
    if lp == 1:
        (a1, b1), c1 = next(iter(p.items()))
        if a1 == 0 and b1 == 0:
            # Scalar multiply
            return {k: v * c1 for k, v in q.items()}
        return {(a1 + a2, b1 + b2): c1 * c2 for (a2, b2), c2 in q.items()}
    if lq == 1:
        (a2, b2), c2 = next(iter(q.items()))
        if a2 == 0 and b2 == 0:
            return {k: v * c2 for k, v in p.items()}
        return {(a1 + a2, b1 + b2): c1 * c2 for (a1, b1), c1 in p.items()}
    # General case: use defaultdict to avoid per-iteration get/del overhead
    result: Dict[Tuple[int, int], int] = _defaultdict(int)
    for (a1, b1), c1 in p.items():
        for (a2, b2), c2 in q.items():
            result[(a1 + a2, b1 + b2)] += c1 * c2
    return {k: v for k, v in result.items() if v != 0}


def _convert_ab_to_xy(poly_ab: Poly) -> TuttePolynomial:
    """Convert polynomial from (a, b) = (x-1, y-1) basis to (x, y) basis.

    Expand each a^i * b^j = (x-1)^i * (y-1)^j using binomial theorem.
    """
    from math import comb

    coeffs_xy: Dict[Tuple[int, int], int] = {}

    for (a_pow, b_pow), coeff in poly_ab.items():
        # (x-1)^a_pow = sum_{p=0}^{a_pow} C(a_pow,p) * x^p * (-1)^{a_pow-p}
        # (y-1)^b_pow = sum_{q=0}^{b_pow} C(b_pow,q) * y^q * (-1)^{b_pow-q}
        for p in range(a_pow + 1):
            cx = comb(a_pow, p) * ((-1) ** (a_pow - p))
            for q in range(b_pow + 1):
                cy = comb(b_pow, q) * ((-1) ** (b_pow - q))
                contribution = coeff * cx * cy
                if contribution != 0:
                    key = (p, q)
                    coeffs_xy[key] = coeffs_xy.get(key, 0) + contribution

    # Remove zeros
    coeffs_xy = {k: v for k, v in coeffs_xy.items() if v != 0}
    return TuttePolynomial.from_coefficients(coeffs_xy)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def compute_treewidth_tutte_if_applicable(
    mg: MultiGraph, max_width: int = 8
) -> Optional[TuttePolynomial]:
    """Compute Tutte polynomial via tree decomposition if treewidth <= max_width."""
    if not mg.nodes:
        return TuttePolynomial.one()

    if not mg.edge_counts and not mg.loop_counts:
        return TuttePolynomial.one()

    td = compute_best_tree_decomposition(mg, max_width)
    if td is None:
        return None

    _log = get_log()
    cost = estimate_dp_cost(td)
    _log.record(EventType.TREEWIDTH_DP, "treewidth",
                f"TW={td.width}, {len(td.bags)} bags, "
                f"{mg.node_count()}n {mg.edge_count()}e, cost={cost:.0f}",
                LogLevel.DEBUG, graph=mg)

    # Use bulk C DP for 5 <= tw <= 10 when available.
    # (April 2026) Audit measured C vs Python head-to-head:
    #   Z(1,1) tw=5 m=22:   C   6ms vs Py    4ms (Py wins 1.7×)
    #   Cm1    tw=4 m=16:   C   1ms vs Py    2ms (~tie)
    #   Z(1,2) tw=10 m=76:  C 168s vs Py 1235s (C wins 7.4×)
    #   Cm2    tw=11 m=80:  C 695s vs Py  207s (**Py wins 3.4×**)
    # The C path is a clear win for tw=6..10 but a 3.4× regression at tw=11.
    # Likely cause: at tw=11 with int64-overflowing spanning-tree counts
    # (Cm2 has 1.2×10^19), the C path's int128/modular-CRT tier becomes
    # slower than Python bigints. The upper cap (tw <= 10) keeps the win
    # band and lets Python handle the regression cases.
    #
    # Attempted to LOWER this to tw >= 3 but caused a 15% corpus
    # regression (cffi marshaling overhead dominated for tiny graphs);
    # reverted. The right gate is `5 <= tw <= 10`.
    if 5 <= td.width <= 10:
        try:
            from ._treewidth_c import compute_treewidth_tutte_c
            c_result = compute_treewidth_tutte_c(td, mg)
            if c_result is not None:
                _connect_cache.clear()
                return c_result
        except Exception:
            pass  # fall back to Python

    result = compute_treewidth_tutte(td, mg)

    _connect_cache.clear()
    return result
