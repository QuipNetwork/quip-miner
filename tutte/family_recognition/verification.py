"""Structural verification helpers for graph family recognition.

Each function confirms that a graph matching the fingerprint signature actually
has the required structural properties (adjacency pattern, cycle structure, etc.).

All functions are O(n + m).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Optional, Tuple

from ..graph import Graph
from .fingerprint import StructuralFingerprint


def verify_ladder(graph: Graph, k: int) -> bool:
    """Verify graph is P_k × P_2 (ladder on 2k vertices).

    Finds the 4 corner vertices (degree 2), identifies end rungs (pairs of
    adjacent corners), then traces rung-by-rung to verify the full structure.

    Complexity: O(n + m)
    """
    n = graph.node_count()
    if n != 2 * k:
        return False

    if k == 2:
        # L_2 = C_4: 4 vertices, all degree 2, forms a cycle
        return True

    # Find degree-2 vertices (corners)
    corners = [v for v in graph.nodes if graph.degree(v) == 2]
    if len(corners) != 4:
        return False

    # Find pairs of adjacent corners — these are the end rungs
    corner_set = set(corners)
    end_pairs = []
    for c in corners:
        for nb in graph.neighbors(c):
            if nb in corner_set and nb > c:
                end_pairs.append((c, nb))

    if len(end_pairs) != 2:
        return False

    # BFS rung-by-rung from one end to the other
    (a1, a2) = end_pairs[0]
    visited = {a1, a2}
    rungs = [(a1, a2)]

    for _ in range(k - 1):
        curr_a, curr_b = rungs[-1]
        next_of_a = graph.neighbors(curr_a) - visited
        next_of_b = graph.neighbors(curr_b) - visited

        if len(next_of_a) != 1 or len(next_of_b) != 1:
            return False

        na = next(iter(next_of_a))
        nb = next(iter(next_of_b))

        # na and nb should be connected (they form a rung)
        if na == nb:
            return False
        edge = (min(na, nb), max(na, nb))
        if edge not in graph.edges:
            return False

        visited.add(na)
        visited.add(nb)
        rungs.append((na, nb))

    return len(visited) == n


def _verify_helm_with_hub(graph: Graph, hub: int, k: int) -> bool:
    """Check helm structure assuming a specific hub vertex."""
    hub_nbrs = graph.neighbors(hub)
    if len(hub_nbrs) != k:
        return False

    # Hub's neighbors should all be degree 4 (rim vertices)
    for v in hub_nbrs:
        if graph.degree(v) != 4:
            return False

    # Each rim vertex has: hub, 2 rim neighbors, 1 pendant
    for v in hub_nbrs:
        vnbrs = graph.neighbors(v)
        pendant_count = sum(1 for u in vnbrs if graph.degree(u) == 1)
        if pendant_count != 1:
            return False
        rim_nbrs = {u for u in vnbrs if u in hub_nbrs}
        if len(rim_nbrs) != 2:
            return False

    # Verify rim forms a cycle
    start = next(iter(hub_nbrs))
    start_rim_nbrs = graph.neighbors(start) & hub_nbrs
    if len(start_rim_nbrs) != 2:
        return False
    prev = start
    curr = next(iter(start_rim_nbrs))
    visited = {start, curr}
    for _ in range(k - 2):
        rim_nbrs = (graph.neighbors(curr) & hub_nbrs) - {prev}
        if len(rim_nbrs) != 1:
            return False
        nxt = next(iter(rim_nbrs))
        if nxt in visited and nxt != start:
            return False
        prev = curr
        curr = nxt
        visited.add(curr)

    return start in (graph.neighbors(curr) & hub_nbrs)


def verify_helm(graph: Graph, k: int) -> bool:
    """Verify graph is a helm: wheel W_k with pendant at each rim vertex.

    Hub has degree k, rim vertices have degree 4, pendant vertices have degree 1.
    When k=4, hub and rim vertices share the same degree, so all candidates
    are tried.

    Complexity: O(n + m)
    """
    candidates = [v for v in graph.nodes if graph.degree(v) == k]
    if not candidates:
        return False

    # When hub degree is unique, only one candidate
    if len(candidates) == 1:
        return _verify_helm_with_hub(graph, candidates[0], k)

    # k=4: hub and rim share degree 4 — try each candidate as hub
    for hub in candidates:
        if _verify_helm_with_hub(graph, hub, k):
            return True
    return False


def verify_book(graph: Graph, k: int) -> bool:
    """Verify graph is a book: k triangles sharing a common edge.

    Two hub vertices with degree k+1, k leaf vertices with degree 2.
    Each leaf connects to both hubs.

    Complexity: O(n + m)
    """
    hub_deg = k + 1
    hubs = [v for v in graph.nodes if graph.degree(v) == hub_deg]
    if len(hubs) != 2:
        return False

    h1, h2 = hubs
    edge = (min(h1, h2), max(h1, h2))
    if edge not in graph.edges:
        return False

    for v in graph.nodes:
        if v in hubs:
            continue
        if graph.degree(v) != 2:
            return False
        nbrs = graph.neighbors(v)
        if h1 not in nbrs or h2 not in nbrs:
            return False

    return True


def verify_sunlet(graph: Graph, k: int) -> bool:
    """Verify graph is a sunlet: cycle C_k with pendant at each vertex.

    k vertices of degree 3 form a cycle, k vertices of degree 1 are pendants.

    Complexity: O(n + m)
    """
    deg3 = [v for v in graph.nodes if graph.degree(v) == 3]
    if len(deg3) != k:
        return False

    deg3_set = set(deg3)

    for v in deg3:
        nbrs = graph.neighbors(v)
        cycle_nbrs = nbrs & deg3_set
        pendant_nbrs = nbrs - deg3_set
        if len(cycle_nbrs) != 2 or len(pendant_nbrs) != 1:
            return False
        pendant = next(iter(pendant_nbrs))
        if graph.degree(pendant) != 1:
            return False

    # Verify deg3 vertices form a single cycle
    start = deg3[0]
    start_cycle_nbrs = graph.neighbors(start) & deg3_set
    if len(start_cycle_nbrs) != 2:
        return False
    prev = start
    curr = next(iter(start_cycle_nbrs))
    visited = {start, curr}
    for _ in range(k - 2):
        cycle_nbrs = (graph.neighbors(curr) & deg3_set) - {prev}
        if len(cycle_nbrs) != 1:
            return False
        nxt = next(iter(cycle_nbrs))
        if nxt in visited and nxt != start:
            return False
        prev = curr
        curr = nxt
        visited.add(curr)

    return start in (graph.neighbors(curr) & deg3_set)


def _verify_gear_with_hub(graph: Graph, hub: int, k: int) -> bool:
    """Check gear structure assuming a specific hub vertex."""
    hub_nbrs = graph.neighbors(hub)
    if len(hub_nbrs) != k:
        return False

    # Hub neighbors must all be degree 3 (rim vertices)
    for v in hub_nbrs:
        if graph.degree(v) != 3:
            return False

    # Each rim vertex connects to: hub, 2 subdivision vertices (degree 2)
    for v in hub_nbrs:
        vnbrs = graph.neighbors(v) - {hub}
        if len(vnbrs) != 2:
            return False
        for u in vnbrs:
            if graph.degree(u) != 2:
                return False

    # Verify the outer ring: rim and subdivision vertices alternate in a 2k-cycle
    start = next(iter(hub_nbrs))
    start_sub_nbrs = graph.neighbors(start) - {hub}
    if len(start_sub_nbrs) != 2:
        return False

    prev = start
    curr = next(iter(start_sub_nbrs))
    visited = {start, curr}
    for step in range(2 * k - 2):
        nbrs = graph.neighbors(curr) - {prev}
        if hub in nbrs:
            nbrs = nbrs - {hub}
        if len(nbrs) != 1:
            return False
        nxt = next(iter(nbrs))
        if nxt in visited and nxt != start:
            return False
        prev = curr
        curr = nxt
        visited.add(curr)

    return start in graph.neighbors(curr)


def verify_gear(graph: Graph, k: int) -> bool:
    """Verify graph is a gear: wheel W_k with each rim edge subdivided.

    Hub has degree k, k rim vertices have degree 3, k subdivision vertices
    have degree 2. Rim and subdivision vertices alternate in a cycle.
    When k=3, hub and rim vertices share the same degree, so all candidates
    are tried.

    Complexity: O(n + m)
    """
    candidates = [v for v in graph.nodes if graph.degree(v) == k]
    if not candidates:
        return False

    if len(candidates) == 1:
        return _verify_gear_with_hub(graph, candidates[0], k)

    # k=3: hub and rim share degree 3 — try each candidate as hub
    for hub in candidates:
        if _verify_gear_with_hub(graph, hub, k):
            return True
    return False


def _propagate_rungs(
    graph: Graph, start: int, rung_partner: int
) -> Optional[Dict[int, int]]:
    """BFS rung propagation for 3-regular graphs with rung structure.

    Given a start vertex and its assumed rung partner, propagate rung assignments
    to all reachable vertices. At each vertex v with known rung partner r:
      - v's cycle neighbors = neighbors(v) - {r}
      - r's cycle neighbors = neighbors(r) - {v}
      - For each unassigned cycle neighbor c of v, c's rung partner is the
        unassigned cycle neighbor of r that is adjacent to c.

    Works for both prism (two k-cycles + rungs) and Möbius (one 2k-cycle + rungs).

    Complexity: O(n + m)

    Returns:
        Dict mapping each vertex to its rung partner, or None if propagation fails.
    """
    rung: Dict[int, int] = {start: rung_partner, rung_partner: start}
    queue = deque([start])

    while queue:
        v = queue.popleft()
        r = rung[v]
        cycle_nbrs_v = graph.neighbors(v) - {r}
        cycle_nbrs_r = graph.neighbors(r) - {v}

        if len(cycle_nbrs_v) != 2 or len(cycle_nbrs_r) != 2:
            return None

        for c in cycle_nbrs_v:
            if c in rung:
                continue
            # c's rung partner = unassigned neighbor of r that is adjacent to c
            c_rung = None
            for d in cycle_nbrs_r:
                if d not in rung and c in graph.neighbors(d):
                    c_rung = d
                    break
            if c_rung is None:
                return None
            rung[c] = c_rung
            rung[c_rung] = c
            queue.append(c)

    return rung


def verify_prism(graph: Graph, k: int) -> bool:
    """Verify graph is a prism (circular ladder) C_k × K_2.

    3-regular bipartite graph with 2k vertices and 3k edges.
    Two k-cycles connected by k rungs. Each vertex connects to its
    rung partner and two cycle neighbors.

    Uses BFS rung propagation: guess start's rung partner, propagate
    rung assignments, then verify two disjoint k-cycles.

    Complexity: O(n + m)
    """
    n = graph.node_count()
    if n != 2 * k:
        return False

    start = next(iter(graph.nodes))
    start_nbrs = sorted(graph.neighbors(start))
    if len(start_nbrs) != 3:
        return False

    for rung_idx in range(3):
        rung = _propagate_rungs(graph, start, start_nbrs[rung_idx])
        if rung is None or len(rung) != 2 * k:
            continue

        # Partition into two cycles via BFS through non-rung edges
        cycle1: set[int] = {start}
        bfs = deque([start])
        while bfs:
            v = bfs.popleft()
            for u in graph.neighbors(v) - {rung[v]}:
                if u not in cycle1:
                    cycle1.add(u)
                    bfs.append(u)

        if len(cycle1) != k:
            continue

        cycle2 = graph.nodes - cycle1
        if len(cycle2) != k:
            continue

        # Verify each vertex has 2 same-cycle and 1 cross-cycle neighbor
        valid = True
        for v in graph.nodes:
            nbrs = graph.neighbors(v)
            own = cycle1 if v in cycle1 else cycle2
            if len(nbrs & own) != 2 or len(nbrs - own) != 1:
                valid = False
                break

        if not valid:
            continue

        # Verify cycle1 forms a single k-cycle
        c1_start = next(iter(cycle1))
        c1_nbrs = graph.neighbors(c1_start) & cycle1
        if len(c1_nbrs) != 2:
            continue
        prev, curr = c1_start, next(iter(c1_nbrs))
        visited = {c1_start, curr}
        for _ in range(k - 2):
            nxt_set = (graph.neighbors(curr) & cycle1) - {prev}
            if len(nxt_set) != 1:
                valid = False
                break
            prev, curr = curr, next(iter(nxt_set))
            visited.add(curr)

        if valid and len(visited) == k and c1_start in graph.neighbors(curr):
            return True

    return False


def verify_mobius(graph: Graph, k: int) -> bool:
    """Verify graph is a Möbius ladder: 3-regular, non-bipartite, 2k vertices.

    A cycle of 2k vertices with k "rung" edges connecting opposite vertices
    (v_i to v_{i+k}). Equivalently, a prism with one twisted rung.

    Uses BFS rung propagation: guess start's rung partner, propagate
    rung assignments, trace the 2k Hamiltonian cycle through non-rung edges,
    then verify rungs connect vertices k apart.

    Complexity: O(n + m)
    """
    n = graph.node_count()
    if n != 2 * k:
        return False

    start = next(iter(graph.nodes))
    start_nbrs = sorted(graph.neighbors(start))
    if len(start_nbrs) != 3:
        return False

    for rung_idx in range(3):
        rung = _propagate_rungs(graph, start, start_nbrs[rung_idx])
        if rung is None or len(rung) != 2 * k:
            continue

        # Trace 2k-cycle through non-rung (cycle) edges
        cycle_nbrs = graph.neighbors(start) - {rung[start]}
        prev = start
        curr = next(iter(cycle_nbrs))
        cycle = [start, curr]
        valid = True

        for _ in range(2 * k - 2):
            nxt_set = (graph.neighbors(curr) - {rung[curr]}) - {prev}
            if len(nxt_set) != 1:
                valid = False
                break
            prev, curr = curr, next(iter(nxt_set))
            cycle.append(curr)

        if not valid or len(cycle) != 2 * k:
            continue

        # Verify cycle closes
        if start not in (graph.neighbors(cycle[-1]) - {rung[cycle[-1]]}):
            continue

        # Verify rungs connect vertices k apart on the cycle
        valid = True
        for i in range(2 * k):
            if rung[cycle[i]] != cycle[(i + k) % (2 * k)]:
                valid = False
                break

        if valid:
            return True

    return False


def detect_grid_dims(
    graph: Graph, fp: StructuralFingerprint
) -> Optional[Tuple[int, int]]:
    """Detect if graph is a grid P_m × P_n and return (m, n).

    Grid P_m × P_n has m*n vertices, 2mn - m - n edges.
    Degree distribution: 4 corners (degree 2), 2(m-2)+2(n-2) borders (degree 3),
    (m-2)(n-2) interior (degree 4).

    Only returns dimensions if m <= 5 (practical limit for transfer matrix).

    Complexity: O(1) against fingerprint (degree counting + quadratic solve).
    """
    n_v = fp.node_count
    m_e = fp.edge_count

    if not fp.is_bipartite:
        return None

    if fp.degree_counts.get(2, 0) != 4:
        return None

    # Only degrees 2, 3, 4 allowed
    for d in fp.degree_counts:
        if d not in (2, 3, 4):
            return None

    # Solve: m*n = n_v, m+n = 2*n_v - m_e
    s = 2 * n_v - m_e
    disc = s * s - 4 * n_v
    if disc < 0:
        return None

    sqrt_disc = int(math.isqrt(disc))
    if sqrt_disc * sqrt_disc != disc:
        return None

    m1 = (s + sqrt_disc) // 2
    m2 = (s - sqrt_disc) // 2

    if m1 * m2 != n_v or m1 + m2 != s:
        return None
    if m1 < 1 or m2 < 1:
        return None

    m_dim, n_dim = min(m1, m2), max(m1, m2)
    if m_dim <= 1:
        return (m_dim, n_dim)
    if m_dim > 5:
        return None

    # Verify degree counts match grid structure
    expected_deg3 = 2 * (m_dim - 2) + 2 * (n_dim - 2)
    expected_deg4 = (m_dim - 2) * (n_dim - 2)

    if fp.degree_counts.get(3, 0) != expected_deg3:
        return None
    if fp.degree_counts.get(4, 0) != expected_deg4:
        return None

    # Degree counts alone don't pin down grid topology — e.g., the
    # K_{2,2}+M_2 chain (8v, 10e, {2:4, 3:4}) shares the same fingerprint
    # as a 2×4 grid but is not one. Verify actual grid structure.
    if m_dim == 2:
        # P_2 × P_n is exactly the ladder L_n
        return (m_dim, n_dim) if verify_ladder(graph, n_dim) else None
    if not _verify_grid_structure(graph, m_dim, n_dim):
        return None
    return (m_dim, n_dim)


def _verify_grid_structure(graph: Graph, m_dim: int, n_dim: int) -> bool:
    """Verify that `graph` is isomorphic to P_m × P_n by tracing the
    grid structure: identify corners, walk along the boundary, then
    fill in interior rows.

    Complexity: O(n + m) — same as verify_ladder.
    """
    n_v = graph.node_count()
    if n_v != m_dim * n_dim:
        return False
    if m_dim < 2 or n_dim < 2:
        return False

    # Find corner vertices (degree 2). Should be exactly 4.
    corners = [v for v in graph.nodes if graph.degree(v) == 2]
    if len(corners) != 4:
        return False

    # Pick a corner and trace its m-path side (longer dimension first).
    # A corner's two neighbors are: one degree-2 (only at L_2 corner-corner
    # adjacency) or both degree-3 (border vertices for grids m>=3, n>=3).
    corner = corners[0]
    nbrs = list(graph.neighbors(corner))
    if len(nbrs) != 2:
        return False

    # Walk one direction to count the side length.
    def walk_border_from(start, first_step):
        """Walk along the border (degree-2/3 vertices) from start through first_step."""
        path = [start, first_step]
        prev, curr = start, first_step
        while graph.degree(curr) >= 3:
            # On a border, curr has 1 prev and exactly 1 next-border-vertex
            # plus interior connections (if any). Find the unvisited neighbor
            # that's also on the border (degree 2 or 3).
            cand = None
            for nb in graph.neighbors(curr):
                if nb == prev:
                    continue
                if graph.degree(nb) <= 3:  # border vertex
                    if cand is not None:
                        # Multiple border neighbors — not a clean grid border
                        return None
                    cand = nb
            if cand is None:
                return None
            path.append(cand)
            prev, curr = curr, cand
            if graph.degree(curr) == 2:
                # Reached the next corner
                break
        return path

    side1 = walk_border_from(corner, nbrs[0])
    side2 = walk_border_from(corner, nbrs[1])
    if side1 is None or side2 is None:
        return False

    # Sides should match {m_dim, n_dim} in length
    sides = sorted([len(side1), len(side2)])
    if sides != [min(m_dim, n_dim), max(m_dim, n_dim)]:
        return False

    # Cheap final check: total edges = 2mn - m - n
    if graph.edge_count() != 2 * m_dim * n_dim - m_dim - n_dim:
        return False

    return True