"""σ-equivariant elimination-order DP for unsigned Tutte polynomial.

Given a graph G and a FREE σ ∈ Aut(G) of order 2, this DP computes
T(G; x_0, y_0) mod p by processing σ-paired edges as a single
4-branch step (del/del, del/keep, keep/del, keep/keep), maintaining
a σ-invariant active set so σ-canonicalization compresses states
~2× compared to the standard DP.

Algorithm (`compute_tutte_per_orbit_mod`):
  - Build a σ-respecting elimination order on V(G): pairs (v, σv)
    eliminated together so the active set is always σ-invariant.
  - State: (partition over active V(G), finalized_count).
  - For each σ-orbit of edges {e, σe}, apply a 4-branch step:
      del/del, del/keep, keep/del, keep/keep
    with appropriate (y-1)-weighting for cycle edges.
  - After each step, canonicalize the partition under the σ-action
    on positions (lex-min of state vs σ-image).
  - Forget σ-vertex pairs together; singleton blocks finalize.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


def _canonicalize_partition(part_seq):
    """Re-id blocks by first appearance in given sequence."""
    bid = {}
    next_id = 0
    out = []
    for b in part_seq:
        if b not in bid:
            bid[b] = next_id
            next_id += 1
        out.append(bid[b])
    return tuple(out)


def _sigma_image(part_canonical, pos_perm):
    """Compute σ-image of canonical partition tuple.

    `part_canonical[i]` = canonical block id at position i.
    `pos_perm[i]` = position that σ sends position i to.

    σ-image: new_part[pos_perm[i]] = part_canonical[i].
    """
    n = len(part_canonical)
    sigma_seq = [0] * n
    for i in range(n):
        sigma_seq[pos_perm[i]] = part_canonical[i]
    return _canonicalize_partition(sigma_seq)


def compute_tutte_per_orbit_mod(
    nodes: List, edges: List[Tuple[int, int]], perm: Dict[int, int],
    x_val: int, y_val: int, p: int
) -> Tuple[int, int]:
    """Compute T(G; x_val, y_val) mod p via per-orbit batched edge processing.

    For each σ-orbit of edges {e, σe}, process both edges TOGETHER as a
    4-branch step (del/del, del/keep, keep/del, keep/keep). Forget σ-pairs
    together. Maintains σ-invariant active set so σ-canonicalization works.

    Args:
      nodes, edges, perm: same as compute_tutte_sigma_equivariant_mod.
      Must be FREE σ (no fixed vertices or edges).

    Returns: (value_mod_p, max_state_count).
    """
    nm = {v: i for i, v in enumerate(sorted(set(nodes)))}
    n = len(nm)
    edges_int = [(nm[u], nm[v]) for (u, v) in edges]
    perm_int = {nm[v]: nm[perm[v]] for v in nodes}

    # Build σ-paired elim order via HYBRID: min-fill picks σ-PAIRS together.
    # At each step, pick the pair (v, σv) with minimum combined fill-in.
    adj: Dict[int, set] = {v: set() for v in range(n)}
    for u, v in edges_int:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    remaining = set(range(n))
    pair_order = []
    while remaining:
        # Find the unprocessed σ-pair with minimum combined fill-in cost.
        best_pair = None
        best_cost = float('inf')
        seen_pair_keys = set()
        for v in remaining:
            sv = perm_int[v]
            if sv == v:
                continue
            pair_key = (min(v, sv), max(v, sv))
            if pair_key in seen_pair_keys:
                continue
            seen_pair_keys.add(pair_key)
            # Estimate cost: combined neighbors of v and sv among remaining.
            nb_v = adj[v] & remaining
            nb_sv = adj[sv] & remaining
            # Fill-in cost: edges that would need to be added among nb_v - {sv} and nb_sv - {v}.
            cost = len(nb_v) + len(nb_sv)
            if cost < best_cost:
                best_cost = cost
                best_pair = pair_key
        if best_pair is None:
            break
        v, sv = best_pair
        pair_order.append((v, sv))
        # Add fill-in edges among neighbors.
        for w1 in adj[v] & remaining:
            for w2 in adj[v] & remaining:
                if w1 != w2:
                    adj[w1].add(w2)
        for w1 in adj[sv] & remaining:
            for w2 in adj[sv] & remaining:
                if w1 != w2:
                    adj[w1].add(w2)
        remaining.discard(v)
        remaining.discard(sv)
    # Linear elim_pos for the pair-step assignment.
    elim_pos = {}
    for k, (v, sv) in enumerate(pair_order):
        elim_pos[v] = k
        elim_pos[sv] = k

    # Build edge orbits. Each orbit is {e, σe} for free σ.
    edge_orbits = []
    seen_edges = set()
    for (u, v) in edges_int:
        ekey = (min(u, v), max(u, v))
        if ekey in seen_edges:
            continue
        seen_edges.add(ekey)
        sigma_e = (perm_int[u], perm_int[v])
        sigma_e_key = (min(sigma_e), max(sigma_e))
        seen_edges.add(sigma_e_key)
        assert sigma_e_key != ekey, "σ must be free on edges"
        edge_orbits.append((ekey, sigma_e_key))

    # Assign each orbit to a pair-step = max pair-step over endpoints.
    orbit_steps = []
    for (e1, e2) in edge_orbits:
        step = max(elim_pos[e1[0]], elim_pos[e1[1]], elim_pos[e2[0]], elim_pos[e2[1]])
        orbit_steps.append((step, e1, e2))
    orbit_steps.sort()
    n_orbits = len(orbit_steps)

    # Active position tracking.
    active_pos = {v: i for i, (v, _) in enumerate(pair_order)
                  for v in [v, _]}
    # Wait that's wrong. Let me redo.
    active_pos = {}
    pos = 0
    for (v, sv) in pair_order:
        active_pos[v] = pos
        active_pos[sv] = pos + 1
        pos += 2

    def get_sigma_pos_perm():
        n_active = len(active_pos)
        inv = [-1] * n_active
        for v, pp in active_pos.items():
            inv[pp] = v
        pos_perm = [0] * n_active
        for i in range(n_active):
            v = inv[i]
            sv = perm_int[v]
            pos_perm[i] = active_pos[sv]
        return pos_perm

    def canonicalize_state(state, pos_perm):
        part, fin = state
        sigma_part = _sigma_image(part, pos_perm)
        if sigma_part < part:
            return (sigma_part, fin)
        return state

    init_part = tuple(range(n))
    states = {(init_part, 0): 1}
    y_minus_1 = (y_val - 1) % p
    max_states = 1
    orbit_idx = 0

    def apply_edge_to_state(part, fin, w, u, v, keep):
        """Apply edge (u, v) deletion or keep to state. Returns list of (new_state, weight)."""
        if not keep:
            return [(part, fin, w)]
        u_pos = active_pos[u]
        v_pos = active_pos[v]
        bu = part[u_pos]
        bv = part[v_pos]
        if bu == bv:
            # Cycle — factor (y-1).
            return [(part, fin, (w * y_minus_1) % p)]
        # Tree edge — merge bv into bu.
        new_part_list = list(part)
        for i in range(len(part)):
            if part[i] == bv:
                new_part_list[i] = bu
        canon = _canonicalize_partition(new_part_list)
        return [(canon, fin, w)]

    for step in range(len(pair_order)):
        # Process all orbits with orbit_step == step.
        while orbit_idx < n_orbits and orbit_steps[orbit_idx][0] == step:
            _step, (u1, v1), (u2, v2) = orbit_steps[orbit_idx]
            orbit_idx += 1
            pos_perm = get_sigma_pos_perm()
            new_states = defaultdict(int)
            for state, w in states.items():
                part, fin = state
                # 4 branches: (del e1, del e2), (del e1, keep e2),
                # (keep e1, del e2), (keep e1, keep e2).
                # Apply e1, then e2.
                for keep1 in (False, True):
                    for keep2 in (False, True):
                        # Apply e1
                        res1 = apply_edge_to_state(part, fin, w, u1, v1, keep1)
                        for p1, f1, w1 in res1:
                            res2 = apply_edge_to_state(p1, f1, w1, u2, v2, keep2)
                            for p2, f2, w2 in res2:
                                key = canonicalize_state((p2, f2), pos_perm)
                                new_states[key] = (new_states[key] + w2) % p
            states = dict(new_states)
            if len(states) > max_states:
                max_states = len(states)

        # Forget σ-pair at this step.
        v0, sv0 = pair_order[step]
        # Check that no future orbit references v0 or sv0.
        still_needed = False
        for ei in range(orbit_idx, n_orbits):
            _, (a, b), (c, d) = orbit_steps[ei]
            if v0 in (a, b, c, d) or sv0 in (a, b, c, d):
                still_needed = True
                break

        if not still_needed:
            # Forget both v0 and sv0 (as a pair) so active set stays σ-invariant.
            fpos1 = active_pos[v0]
            fpos2 = active_pos[sv0]
            # Re-build active_pos after removing both.
            del active_pos[v0]
            del active_pos[sv0]
            new_active_pos = {}
            forgotten = sorted([fpos1, fpos2])
            for v, pp in active_pos.items():
                shift = sum(1 for f in forgotten if f < pp)
                new_active_pos[v] = pp - shift
            active_pos = new_active_pos

            pos_perm_after = get_sigma_pos_perm()
            new_states = {}
            for state, w in states.items():
                part, fin = state
                # Determine fin increment from singletons forgotten.
                b1 = part[fpos1]
                b2 = part[fpos2]
                size1 = sum(1 for x in part if x == b1)
                size2 = sum(1 for x in part if x == b2)
                inc = 0
                # Finalize block(s) of size 1.
                if size1 == 1:
                    inc += 1
                if b1 != b2 and size2 == 1:
                    inc += 1
                if b1 == b2 and size1 == 2:
                    # Both positions in same block of size 2 — block becomes empty.
                    inc += 1
                new_fin = fin + inc
                # Drop both positions.
                drop_set = {fpos1, fpos2}
                new_part_seq = [part[i] for i in range(len(part)) if i not in drop_set]
                canon = _canonicalize_partition(new_part_seq)
                new_st = canonicalize_state((canon, new_fin), pos_perm_after)
                new_states[new_st] = (new_states.get(new_st, 0) + w) % p
            states = new_states
            if len(states) > max_states:
                max_states = len(states)

    # Aggregation: T(G) = Σ_state weight × (x-1)^{r_E - r_state}.
    parent_full = list(range(n))
    for (u, v) in edges_int:
        ru = u
        while parent_full[ru] != ru:
            ru = parent_full[ru]
        rv = v
        while parent_full[rv] != rv:
            rv = parent_full[rv]
        if ru != rv:
            parent_full[max(ru, rv)] = min(ru, rv)
    roots = set()
    for i in range(n):
        cur = i
        while parent_full[cur] != cur:
            cur = parent_full[cur]
        roots.add(cur)
    r_E = n - len(roots)

    x_minus_1 = (x_val - 1) % p
    total = 0
    for state, w in states.items():
        part, fin = state
        active_blocks = (max(part) + 1) if part else 0
        r_state = n - fin - active_blocks
        total = (total + w * pow(x_minus_1, r_E - r_state, p)) % p
    return total, max_states

