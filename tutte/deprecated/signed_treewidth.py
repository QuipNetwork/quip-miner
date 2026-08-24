"""Signed-graph treewidth DP for σ-equivariant Tutte polynomial.

Computes T_signed(G_base, χ; x_0, y_0, p) at a single integer point mod p
for a signed multigraph (V, E, χ : E → Z_2). Uses bag-by-bag DP on the
existing TreeDecomposition infrastructure with sign-aware partition state.

KEY: at each bag, state tracks ONLY bag vertices. As we move up the tree
decomposition, vertices outside the parent bag are FORGOTTEN, which
permanently consolidates their balance into the "rank deficit accumulator".

State per bag: dict {(partition, monodromy, balance, finalized_balance_deficit):
                     weight_int_mod_p}
  - partition: tuple of block ID per bag vertex (canonicalized)
  - monodromy: tuple of mon bit per bag vertex (rebased per block)
  - balance: tuple of bool per current block
  - finalized_balance_deficit: accumulated rank deficit from forgotten
    components (each forgotten unbalanced block adds 1 to rank, balanced
    adds 0)

Final answer aggregates over the empty-bag state with appropriate factors.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from ..graphs.treewidth import compute_tree_decomposition, TreeDecomposition
from ..graph import MultiGraph


# State: (partition, monodromy, balance, finalized_balance_count)
# finalized_balance_count = # balanced components that have been forgotten
SignedBagState = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[bool, ...], int]


def _canonicalize_partition(partition: List[int]) -> Tuple[List[int], Dict[int, int]]:
    """Relabel block IDs in first-appearance order. Returns (canonical, remap)."""
    relabel = {}
    next_id = 0
    canon = []
    for b in partition:
        if b not in relabel:
            relabel[b] = next_id
            next_id += 1
        canon.append(relabel[b])
    return canon, relabel


def _canonicalize_state(
    partition: List[int], monodromy: List[int], balance: List[bool], finalized: int
) -> SignedBagState:
    """Full state canonicalization."""
    canon_p, remap = _canonicalize_partition(partition)
    n_blocks = len(remap)
    # Rebase monodromy: first vertex in each block has mon 0
    first_mon = [None] * n_blocks
    for i, b in enumerate(canon_p):
        if first_mon[b] is None:
            first_mon[b] = monodromy[i]
    canon_m = tuple((monodromy[i] + first_mon[canon_p[i]]) & 1 for i in range(len(partition)))
    # Reorder balance to match canonical block ordering
    canon_b = [True] * n_blocks
    for old_id, new_id in remap.items():
        canon_b[new_id] = balance[old_id]
    return (tuple(canon_p), canon_m, tuple(canon_b), finalized)


def _introduce_edge(
    state: SignedBagState, u_pos: int, v_pos: int, sign: int, y_minus_1: int, p: int
) -> List[Tuple[SignedBagState, int]]:
    """Branch on delete/keep for edge (u_pos, v_pos) with given sign.

    Returns list of (new_state, weight_factor) tuples.
    """
    partition, monodromy, balance, finalized = state
    n = len(partition)
    bu = partition[u_pos]
    bv = partition[v_pos]
    mu = monodromy[u_pos]
    mv = monodromy[v_pos]

    results = [(state, 1)]  # delete branch, factor 1

    if bu != bv:
        # Merge — rank +1, factor 1
        new_partition = list(partition)
        new_monodromy = list(monodromy)
        new_balance = list(balance)
        offset = (mu + sign + mv) & 1
        # Merge block_v into block_u; adjust mon of block_v vertices
        for i in range(n):
            if partition[i] == bv:
                new_partition[i] = bu
                new_monodromy[i] = (monodromy[i] + offset) & 1
        # Combined balance — leave block_v's entry stale (canonicalize ignores
        # unused block IDs since remap only keys live ones)
        combined = balance[bu] and balance[bv]
        new_balance[bu] = combined
        new_state = _canonicalize_state(new_partition, new_monodromy, new_balance, finalized)
        results.append((new_state, 1))
    else:
        cycle_sign = (mu + sign + mv) & 1
        if balance[bu]:
            if cycle_sign == 0:
                # Balanced cycle in balanced block: rank +0, factor (y-1)
                results.append((state, y_minus_1))
            else:
                # Unbalanced cycle in balanced block: block becomes unbalanced,
                # rank +1, factor 1
                new_balance = list(balance)
                new_balance[bu] = False
                new_state = _canonicalize_state(list(partition), list(monodromy), new_balance, finalized)
                results.append((new_state, 1))
        else:
            # Cycle in unbalanced block: rank +0, factor (y-1)
            results.append((state, y_minus_1))

    return results


def _forget_vertex(state: SignedBagState, pos: int, n_bag: int) -> SignedBagState:
    """Drop vertex at pos. If its block was alone, finalize block's balance.

    Returns new state (n_bag - 1 vertices), with finalized balance count
    incremented if a balanced block was forgotten.
    """
    partition, monodromy, balance, finalized = state
    bid = partition[pos]
    size = sum(1 for b in partition if b == bid)
    new_partition = [partition[i] for i in range(n_bag) if i != pos]
    new_monodromy = [monodromy[i] for i in range(n_bag) if i != pos]

    if size == 1:
        # Block forgotten — add its balance to finalized; leave entry stale
        # (canonicalize will drop it since no remaining partition value references it)
        new_balance = list(balance)
        new_finalized = finalized + (1 if balance[bid] else 0)
    else:
        new_balance = list(balance)
        new_finalized = finalized

    return _canonicalize_state(new_partition, new_monodromy, new_balance, new_finalized)


def _combine_states_uf(state1, state2, verts1, verts2, overlap, combined_verts):
    """Combine two states via union-find on combined_verts.

    state1 is over verts1 (in that order).
    state2 is over verts2 (in that order).
    overlap = verts1 ∩ verts2.
    combined_verts = verts1 ∪ (verts2 - overlap), in some chosen order.

    Combines partition info from both states. Vertices in overlap have
    PRE-AGREED partition/monodromy/balance (caller verified this).

    Returns canonical state over combined_verts, or None if inconsistent.
    """
    n_new = len(combined_verts)
    parent = list(range(n_new))
    mon = [0] * n_new
    bal_per_block = [True] * n_new  # initialized; we'll track per-root later

    pos_in_combined = {v: i for i, v in enumerate(combined_verts)}

    # Helper: find root with monodromy accumulation
    def find_root(x):
        cur = x
        acc = 0
        while parent[cur] != cur:
            acc = (acc + mon[cur]) & 1
            cur = parent[cur]
        return cur, acc

    def do_union(u, v, sign):
        """Union u and v with relative monodromy `sign` (u + sign = v in cover)."""
        ru, mu = find_root(u)
        rv, mv = find_root(v)
        if ru != rv:
            # Merge
            # Set offset so that mon[rv] reflects the new relationship
            offset = (mu + sign + mv) & 1
            parent[rv] = ru
            mon[rv] = offset
            bal_per_block[ru] = bal_per_block[ru] and bal_per_block[rv]
            return False  # merged, no cycle
        else:
            # Cycle: check if balanced
            cycle_sign = (mu + sign + mv) & 1
            if cycle_sign != 0:
                bal_per_block[ru] = False
            return True

    # Apply state1's partition: union verts in same block of state1
    part1, mon1, bal1, fin1 = state1
    block_first_v1 = {}
    for i, v in enumerate(verts1):
        bid = part1[i]
        pos = pos_in_combined[v]
        if bid not in block_first_v1:
            block_first_v1[bid] = (pos, mon1[i])
            bal_per_block[pos] = bal1[bid]
        else:
            # Union with first
            fp, fm = block_first_v1[bid]
            # In state1, mon of i relative to block rep = mon1[i]
            # In state1, mon of first relative to block rep = fm
            # So edge (first, i) has sign mon1[i] XOR fm
            sign = (mon1[i] + fm) & 1
            do_union(fp, pos, sign)
            # Balance carries from state1
            ru, _ = find_root(fp)
            bal_per_block[ru] = bal_per_block[ru] and bal1[bid]

    # Apply state2's partition over verts2: union verts in same block of state2
    part2, mon2, bal2, fin2 = state2
    block_first_v2 = {}
    for i, v in enumerate(verts2):
        bid = part2[i]
        pos = pos_in_combined[v]
        if bid not in block_first_v2:
            block_first_v2[bid] = (pos, mon2[i])
            # If this is a verts2-only vertex, set balance
            ru, _ = find_root(pos)
            if v not in [v1 for v1 in verts1]:
                bal_per_block[ru] = bal_per_block[ru] and bal2[bid]
            else:
                # already covered by state1; balance should match
                pass
        else:
            fp, fm = block_first_v2[bid]
            sign = (mon2[i] + fm) & 1
            do_union(fp, pos, sign)
            ru, _ = find_root(fp)
            bal_per_block[ru] = bal_per_block[ru] and bal2[bid]

    # Build canonical state
    # For each vertex in combined_verts, find its root and monodromy
    out_partition = [0] * n_new
    out_monodromy = [0] * n_new
    block_id_for_root = {}
    next_id = 0
    block_first_mon = []
    for i in range(n_new):
        ri, mi = find_root(i)
        if ri not in block_id_for_root:
            block_id_for_root[ri] = next_id
            block_first_mon.append(mi)
            next_id += 1
        out_partition[i] = block_id_for_root[ri]
        out_monodromy[i] = (mi + block_first_mon[block_id_for_root[ri]]) & 1

    out_balance = [True] * next_id
    for ri, bid in block_id_for_root.items():
        out_balance[bid] = bal_per_block[ri]

    # Combined finalized: sum of both
    combined_fin = fin1 + fin2
    return (tuple(out_partition), tuple(out_monodromy), tuple(out_balance), combined_fin)


def _initial_bag_state(bag_verts: List[int]) -> SignedBagState:
    """Bag with all-singleton blocks, mon=0, balanced."""
    n = len(bag_verts)
    partition = tuple(range(n))
    monodromy = tuple(0 for _ in range(n))
    balance = tuple(True for _ in range(n))
    return (partition, monodromy, balance, 0)


def _project_state(state: SignedBagState, keep_positions: List[int], n_old: int) -> SignedBagState:
    """Marginalize state to only the kept positions (forget all others)."""
    new_state = state
    n_cur = n_old
    forget_positions = sorted(set(range(n_old)) - set(keep_positions), reverse=True)
    for pos in forget_positions:
        new_state = _forget_vertex(new_state, pos, n_cur)
        n_cur -= 1
    return new_state


def compute_signed_tutte_treewidth_mod(
    nodes: List, edges_with_signs: List[Tuple[Tuple, int]],
    x_val: int, y_val: int, p: int
) -> Tuple[int, int]:
    """Compute T_signed(G; x_val, y_val) mod p using treewidth DP on G.

    Returns (T_signed_value_mod_p, max_state_count_observed).
    """
    # Map nodes to 0..n-1
    nm = {v: i for i, v in enumerate(sorted(set(nodes)))}
    n = len(nm)
    edges_int = [((nm[u], nm[v]), sign) for ((u, v), sign) in edges_with_signs]

    # Build underlying multigraph for tree decomposition
    edge_counts = defaultdict(int)
    for (u, v), _sign in edges_int:
        e = (min(u, v), max(u, v))
        edge_counts[e] += 1
    mg = MultiGraph(
        nodes=frozenset(range(n)),
        edge_counts=dict(edge_counts),
        loop_counts={},
    )
    td = compute_tree_decomposition(mg)

    # For each unique (u, v, sign) signed edge, group with its bag assignment.
    # td.bag_edges[bag_idx] gives ((u, v, mult)) — we need to track signs.
    # Group edges (with signs) by which bag they're assigned to:
    edges_by_bag: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    edge_ptr = 0  # round-robin assignment to bags
    # Use td.bag_edges: each unique (u, v) edge appears once with mult.
    # We need to assign each SIGNED edge to a bag. Since td.bag_edges
    # already assigns each (u, v) to some bag, we'll assign all signed
    # variants of (u, v) to that same bag.
    edge_to_bag = {}
    for bag_idx in range(len(td.bags)):
        for (u, v, _mult) in td.bag_edges[bag_idx]:
            edge_to_bag[(min(u, v), max(u, v))] = bag_idx

    for ((u, v), sign) in edges_int:
        e = (min(u, v), max(u, v))
        bag_idx = edge_to_bag.get(e)
        if bag_idx is not None:
            edges_by_bag[bag_idx].append((u, v, sign))

    # Build children adjacency (root = td.root)
    children = defaultdict(list)
    visited_bags = set()
    stack = [(td.root, None)]
    while stack:
        bag_idx, parent_idx = stack.pop()
        if bag_idx in visited_bags:
            continue
        visited_bags.add(bag_idx)
        for nb in td.tree_adj[bag_idx]:
            if nb != parent_idx and nb not in visited_bags:
                children[bag_idx].append(nb)
                stack.append((nb, bag_idx))

    y_minus_1 = (y_val - 1) % p

    # Process bags bottom-up (DFS post-order)
    bag_states: Dict[int, Dict[SignedBagState, int]] = {}
    bag_vert_order: Dict[int, List[int]] = {}

    def process(bag_idx: int):
        bag_verts = sorted(td.bags[bag_idx])
        bag_vert_order[bag_idx] = bag_verts
        # If leaf: start with empty state of bag_verts
        if not children[bag_idx]:
            init = _initial_bag_state(bag_verts)
            states = {init: 1}
        else:
            # Recursively process children
            for c in children[bag_idx]:
                process(c)
            # Combine children states + current bag verts
            # For path-like TDs: typically 1 child. Merge by intersection of bag verts.
            if len(children[bag_idx]) == 1:
                c = children[bag_idx][0]
                child_verts = bag_vert_order[c]
                child_states = bag_states[c]
                # Project child state to vertices in parent bag (forget child-only verts)
                shared = [v for v in child_verts if v in bag_verts]
                keep_positions = [child_verts.index(v) for v in shared]
                # Forget vertices in child not in parent
                projected = defaultdict(int)
                for st, w in child_states.items():
                    proj = _project_state(st, keep_positions, len(child_verts))
                    projected[proj] = (projected[proj] + w) % p
                # Now extend to full parent bag: introduce new vertices as singletons
                new_verts = [v for v in bag_verts if v not in child_verts]
                # For each projected state, add new_verts as singletons
                states = {}
                for st, w in projected.items():
                    partition, monodromy, balance, finalized = st
                    n_old = len(partition)
                    new_partition = list(partition)
                    new_monodromy = list(monodromy)
                    new_balance = list(balance)
                    n_blocks_old = len(balance)
                    # Re-order so that bag_verts order is respected:
                    # current state vertices = `shared` in their order
                    # parent vertices = bag_verts; insert new singletons at proper positions
                    # We need to rebuild state with bag_verts ordering
                    # Construct mapping from bag_verts to state positions
                    # shared maps to existing positions; new_verts get new singletons
                    final_partition = []
                    final_monodromy = []
                    final_balance = list(new_balance)
                    next_new_block = n_blocks_old
                    shared_idx = 0
                    for v in bag_verts:
                        if v in new_verts:
                            final_partition.append(next_new_block)
                            final_monodromy.append(0)
                            final_balance.append(True)
                            next_new_block += 1
                        else:
                            # v is in shared
                            pos_in_shared = shared.index(v)
                            final_partition.append(new_partition[pos_in_shared])
                            final_monodromy.append(new_monodromy[pos_in_shared])
                    new_state = _canonicalize_state(
                        final_partition, final_monodromy, final_balance, finalized
                    )
                    states[new_state] = (states.get(new_state, 0) + w) % p
            else:
                # Multi-child JOIN at parent bag, using union-find combine.
                #
                # Algorithm:
                # 1. Project each child to (C_i ∩ P), getting state over shared_i verts.
                # 2. Sequentially union-find combine: start with empty `covered`,
                #    grow by absorbing each child's projection. Combine merges
                #    partition blocks across boundaries when overlap verts agree.
                # 3. After all children: extend to parent bag (add singletons for
                #    verts not in any child).
                children_list = children[bag_idx]
                projected_children = []
                for c in children_list:
                    c_verts = bag_vert_order[c]
                    shared_c = [v for v in c_verts if v in bag_verts]
                    keep_pos_c = [c_verts.index(v) for v in shared_c]
                    proj_c = defaultdict(int)
                    for st, w in bag_states[c].items():
                        proj_st = _project_state(st, keep_pos_c, len(c_verts))
                        proj_c[proj_st] = (proj_c[proj_st] + w) % p
                    projected_children.append((shared_c, dict(proj_c)))

                # Start with first child as the base
                covered, states = projected_children[0]

                for shared_i, child_states in projected_children[1:]:
                    overlap = [v for v in covered if v in shared_i]
                    # Positions of overlap in current `covered` and `shared_i`
                    cov_overlap_pos = [covered.index(v) for v in overlap]
                    chi_overlap_pos = [shared_i.index(v) for v in overlap]
                    cov_only = [v for v in covered if v not in overlap]
                    chi_only = [v for v in shared_i if v not in overlap]
                    new_covered = list(covered) + chi_only

                    new_states = defaultdict(int)
                    for st, w in states.items():
                        # Project st to overlap (for matching)
                        st_overlap = _project_state(st, cov_overlap_pos, len(covered))
                        p_part, p_mon, p_bal, _ = st_overlap
                        for child_st, child_w in child_states.items():
                            child_overlap = _project_state(child_st, chi_overlap_pos, len(shared_i))
                            c_part, c_mon, c_bal, _ = child_overlap
                            if (p_part, p_mon, p_bal) != (c_part, c_mon, c_bal):
                                continue
                            # Combine st (over covered) with child_st (over shared_i)
                            # using union-find on new_covered = covered + chi_only.
                            combined = _combine_states_uf(
                                st, child_st, covered, shared_i, overlap, new_covered
                            )
                            if combined is None:
                                continue
                            new_w = (w * child_w) % p
                            new_states[combined] = (new_states[combined] + new_w) % p

                    covered = new_covered
                    states = dict(new_states)

                # Finally extend `states` (over `covered`) to bag_verts:
                # add fresh singletons for vertices in bag_verts but not in covered.
                new_verts_final = [v for v in bag_verts if v not in covered]
                final_states = {}
                for st, w in states.items():
                    partition, monodromy, balance, finalized = st
                    n_blocks_old = len(balance)
                    final_partition = []
                    final_monodromy = []
                    final_balance = list(balance)
                    next_new_block = n_blocks_old
                    for v in bag_verts:
                        if v in new_verts_final:
                            final_partition.append(next_new_block)
                            final_monodromy.append(0)
                            final_balance.append(True)
                            next_new_block += 1
                        else:
                            pos_in_cov = covered.index(v)
                            final_partition.append(partition[pos_in_cov])
                            final_monodromy.append(monodromy[pos_in_cov])
                    new_state = _canonicalize_state(
                        final_partition, final_monodromy, final_balance, finalized
                    )
                    final_states[new_state] = (final_states.get(new_state, 0) + w) % p
                states = final_states

        # Process edges assigned to this bag
        for (u, v, sign) in edges_by_bag.get(bag_idx, []):
            u_pos = bag_verts.index(u)
            v_pos = bag_verts.index(v)
            new_states = defaultdict(int)
            for state, w in states.items():
                for new_st, factor in _introduce_edge(state, u_pos, v_pos, sign, y_minus_1, p):
                    new_states[new_st] = (new_states[new_st] + w * factor) % p
            states = dict(new_states)

        bag_states[bag_idx] = states

    process(td.root)

    # Aggregate at root: forget all root-bag verts to get final state
    root_verts = bag_vert_order[td.root]
    root_states = bag_states[td.root]
    final_states = defaultdict(int)
    for st, w in root_states.items():
        proj = _project_state(st, [], len(root_verts))
        final_states[proj] = (final_states[proj] + w) % p

    # final_states should have empty partition; finalized counts the
    # total # balanced components after all forgetting. r_total = n - balanced.
    x_minus_1 = (x_val - 1) % p
    total = 0
    max_states_seen = max((len(s) for s in bag_states.values()), default=1)
    for st, w in final_states.items():
        partition, monodromy, balance, finalized_bal = st
        # All bag verts forgotten; finalized_bal = total balanced components
        # Add current bag's still-active balanced components (should be 0 if
        # all forgotten)
        active_bal = sum(1 for b in balance if b)
        n_balanced = finalized_bal + active_bal
        r_state = n - n_balanced
        # Note: r_E is computed similarly via processing all edges (sum of merges/unbalances)
        # For this DP, "no edge processed = all singletons all balanced = r_state = 0"
        # and "all edges processed = r_E"
        # The final states correspond to specific edge subsets; their rank is r_state
        # The aggregate over all subsets weighted gives sum (x-1)^{r_E - r_state} × w
        # We need r_E to be the max rank achievable (all edges included)
        # But we don't have a single r_E value handy here; compute separately
        pass

    # Compute r_E via direct edge processing
    parent_full = list(range(n))
    mon_full = [0] * n
    bal_full = [True] * n
    for ((u, v), sign) in edges_int:
        cur = u; mu = 0
        while parent_full[cur] != cur:
            mu = (mu + mon_full[cur]) & 1
            cur = parent_full[cur]
        ru = cur
        cur = v; mv = 0
        while parent_full[cur] != cur:
            mv = (mv + mon_full[cur]) & 1
            cur = parent_full[cur]
        rv = cur
        if ru != rv:
            target = min(ru, rv); other = max(ru, rv)
            offset = (mu + sign + mv) & 1 if target == ru else (mv + sign + mu) & 1
            parent_full[other] = target
            mon_full[other] = offset
            bal_full[target] = bal_full[ru] and bal_full[rv]
        else:
            cs = (mu + sign + mv) & 1
            if cs != 0:
                bal_full[ru] = False
    roots = set()
    for i in range(n):
        cur = i
        while parent_full[cur] != cur:
            cur = parent_full[cur]
        roots.add(cur)
    n_balanced_full = sum(1 for r in roots if bal_full[r])
    r_E = n - n_balanced_full

    # Aggregate
    total = 0
    for st, w in final_states.items():
        partition, monodromy, balance, finalized_bal = st
        active_bal = sum(1 for b in balance if b)
        n_balanced = finalized_bal + active_bal
        r_state = n - n_balanced
        total = (total + w * pow(x_minus_1, r_E - r_state, p)) % p

    return total, max_states_seen
