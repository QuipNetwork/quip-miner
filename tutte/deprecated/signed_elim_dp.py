"""Signed-graph Tutte DP via VERTEX ELIMINATION ORDER (path-decomposition style).

For each vertex in elimination order:
  1. Process all edges incident to this vertex (with the other endpoint
     also still "active" at this point or earlier).
  2. After processing those edges, FORGET the vertex (marginalize it out).

State per moment: dict {(partition, monodromy, balance, finalized_count) → weight}
where partition/monodromy/balance only cover currently-active vertices.

For graphs with low treewidth (path-width), this bounds the state count
to ~Bell(pathwidth) × 2^pathwidth × 2^pathwidth, which is manageable.

This sidesteps the multi-child tree-decomp join bug by using a strictly
linear processing order.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from ..graphs.treewidth import _elimination_ordering


SignedActiveState = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[bool, ...], int]


def _canonicalize_active(parent, mon, balance, n_active, finalized):
    """Canonicalize state over n_active vertices (indexed 0..n_active-1).

    parent[i] = first vertex (smallest position) in i's block (caller sets this).
    mon[i] = monodromy of vertex at position i, relative to the merge-tree root
             of its block (NOT relative to the canonical first-position root).
    balance[i] = balance of block whose first vertex is at position i (dict).

    In UNBALANCED blocks, monodromies are irrelevant for future computations
    (cycle/merge cases that touch unbalanced blocks all behave monodromy-invariantly),
    so we canonicalize them to 0 to collapse equivalent states.

    Re-bases monodromies in BALANCED blocks so each block's first vertex has mon 0.
    """
    block_id = {}
    block_bal = []
    next_id = 0
    part = [0] * n_active
    mono = [0] * n_active
    bal_list = []
    for i in range(n_active):
        pi = parent[i]
        bid = block_id.get(pi)
        if bid is None:
            bid = next_id
            block_id[pi] = bid
            next_id += 1
            b = balance.get(pi, True)
            bal_list.append(b)
            block_bal.append(b)
        part[i] = bid
        if block_bal[bid]:
            mono[i] = (mon[i] + mon[pi]) & 1
        # else: mono[i] stays 0 (canonicalized for unbalanced block)
    return (tuple(part), tuple(mono), tuple(bal_list), finalized)


def compute_t_fix_sigma_mod(
    nodes: List, edges_with_signs: List[Tuple[Tuple, int]],
    r_E_G: int, x_val: int, y_val: int, p: int,
    engine: str = "python",
) -> Tuple[int, int]:
    """Compute T_fix^σ(G; x_val, y_val) mod p for a 2-fold cover G → G/σ.

    Given the quotient (nodes, edges_with_signs) with monodromy χ:
      T_fix^σ(G) := Σ_{A ⊆ E(G) : σ(A)=A} (x-1)^{r(E_G)-r_G(A)} (y-1)^{|A|-r_G(A)}.

    Uses the lift identity r_G(A_L) = r_quot(L) + r_signed(L, χ) for 2-fold
    covers (A_L = lift of L from quotient). Handles both FREE covers (every
    cover edge in a σ-pair, |A_L| = 2|L|) and NON-FREE covers (σ-fixed
    cover edges appear as LOOPS in quotient, contributing 1 cover edge each:
    |A_L| = 2|L| - |L_loop|).

    The DP processes quotient edges and accumulates per-edge multipliers.
    Non-loop quotient edge (σ-pair in cover, Δ|A|=2):
      - tree edge (Δr_quot=1, Δr_signed=1): factor (x-1)^{-2}
      - tree edge unbal+unbal merge (Δr_quot=1, Δr_signed=0): factor (x-1)^{-1}(y-1)
      - unbal cycle in balanced (Δr_quot=0, Δr_signed=1): factor (x-1)^{-1}(y-1)
      - bal cycle in any / cycle in unbal (Δr_quot=0, Δr_signed=0): factor (y-1)^2
    Loop quotient edge (σ-fixed in cover, Δ|A|=1):
      - bal cycle in bal: factor (y-1) [one less (y-1) than non-loop]
      - unbal cycle in bal: factor (x-1)^{-1}
      - cycle in unbal: factor (y-1)

    Returns (value_mod_p, max_state_count_observed).

    Caller must supply r_E_G = rank of full edge set in G. This equals
    r_quot(E_quot) + r_signed(E_quot, χ) regardless of free/non-free.
    """
    nm = {v: i for i, v in enumerate(sorted(set(nodes)))}
    n = len(nm)
    edges_int = [((nm[u], nm[v]), sign) for ((u, v), sign) in edges_with_signs]

    adj: Dict[int, set] = {v: set() for v in range(n)}
    for (u, v), _ in edges_int:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    best_order = None
    best_max_bag = float('inf')
    for h in ("minfill", "minfill_degree", "mindegree"):
        order = _elimination_ordering(adj, list(range(n)), heuristic=h, max_width=20)
        if order is None:
            continue
        remaining = set(range(n))
        elim_adj_sim = {v: set(adj[v]) for v in range(n)}
        max_bag = 0
        for v in order:
            nb = elim_adj_sim[v] & remaining
            if len(nb) > max_bag:
                max_bag = len(nb)
            for u in nb:
                for w in nb:
                    if u != w:
                        elim_adj_sim[u].add(w)
            remaining.discard(v)
        if max_bag < best_max_bag:
            best_max_bag = max_bag
            best_order = order
    elim_order = best_order if best_order is not None else list(range(n))

    # Try full DP in C — eliminates per-step Python encode/decode overhead.
    if engine in ("c", "auto"):
        try:
            from ._signed_elim_c import t_fix_dp_full as _t_fix_dp_full
        except ImportError:
            _t_fix_dp_full = None
        if _t_fix_dp_full is not None:
            edges_uv = [(u_, v_) for (u_, v_), _ in edges_int]
            signs_list = [sign_ for (_, sign_) in edges_int]
            is_loop_list = [1 if u_ == v_ else 0 for (u_, v_), _ in edges_int]
            result = _t_fix_dp_full(
                n, edges_uv, signs_list, is_loop_list, elim_order, r_E_G,
                x_val, y_val, p,
            )
            if result is not None:
                return result

    elim_pos = {v: i for i, v in enumerate(elim_order)}
    edge_process_step = []
    for (u, v), sign in edges_int:
        step = max(elim_pos[u], elim_pos[v])
        edge_process_step.append((step, u, v, sign))
    edge_process_step.sort()

    active_pos = {v: v for v in range(n)}

    init_partition = tuple(range(n))
    init_mon = tuple(0 for _ in range(n))
    init_balance = tuple(True for _ in range(n))
    init_state = (init_partition, init_mon, init_balance, 0)
    states = {init_state: 1}

    x_minus_1 = (x_val - 1) % p
    y_minus_1 = (y_val - 1) % p
    x_inv = pow(x_minus_1, p - 2, p) if x_minus_1 != 0 else 0
    # Pre-compute per-edge multipliers (mod p) for NON-loop edges (σ-paired).
    # For non-loop quotient edge: keeping → +2 cover edges, so |A| += 2.
    factor_tree_xbal_xbal = (x_inv * x_inv) % p  # (x-1)^{-2}
    factor_tree_unbal_unbal = (x_inv * y_minus_1) % p  # (x-1)^{-1} (y-1)
    factor_unbal_cycle_in_bal = (x_inv * y_minus_1) % p  # (x-1)^{-1} (y-1)
    factor_cycle_no_rank_change = (y_minus_1 * y_minus_1) % p  # (y-1)^2
    # For LOOP quotient edge (= σ-fixed cover edge): keeping → +1 cover edge.
    # y-exponent contribution is one less than non-loop case.
    factor_loop_bal_cycle_in_bal = y_minus_1  # (y-1)^1
    factor_loop_unbal_cycle_in_bal = x_inv  # (x-1)^{-1}
    factor_loop_cycle_in_unbal = y_minus_1  # (y-1)^1

    max_states = 1
    edge_idx = 0
    n_edges = len(edge_process_step)

    for step in range(len(elim_order)):
        while edge_idx < n_edges and edge_process_step[edge_idx][0] == step:
            _step, u, v, sign = edge_process_step[edge_idx]
            edge_idx += 1
            u_pos = active_pos[u]
            v_pos = active_pos[v]
            new_states = defaultdict(int)
            for state, w in states.items():
                partition, monodromy, balance, finalized = state
                bu = partition[u_pos]
                bv = partition[v_pos]
                mu = monodromy[u_pos]
                mv = monodromy[v_pos]

                # Delete branch: factor 1.
                new_states[state] = (new_states[state] + w) % p

                # Keep branch: depends on whether tree or cycle, and balance.
                if bu != bv:
                    offset = (mu + sign + mv) & 1
                    bal_bu = balance[bu]
                    bal_bv = balance[bv]
                    both_unbal = (not bal_bu) and (not bal_bv)
                    combined = bal_bu and bal_bv
                    if both_unbal:
                        factor = factor_tree_unbal_unbal
                    else:
                        factor = factor_tree_xbal_xbal
                    n_active = len(partition)
                    block_id_map = {}
                    block_first_mon_list = []
                    bal_list = []
                    part = [0] * n_active
                    mono = [0] * n_active
                    for i in range(n_active):
                        old_bid = partition[i]
                        if old_bid == bv:
                            merged_bid = bu
                            merged_mon = (monodromy[i] + offset) & 1
                        else:
                            merged_bid = old_bid
                            merged_mon = monodromy[i]
                        canon_id = block_id_map.get(merged_bid)
                        if canon_id is None:
                            canon_id = len(block_id_map)
                            block_id_map[merged_bid] = canon_id
                            block_first_mon_list.append(merged_mon)
                            if merged_bid == bu:
                                bal_list.append(combined)
                            else:
                                bal_list.append(balance[merged_bid])
                        part[i] = canon_id
                        if bal_list[canon_id]:
                            mono[i] = (merged_mon + block_first_mon_list[canon_id]) & 1
                    new_st = (tuple(part), tuple(mono), tuple(bal_list), finalized)
                    new_states[new_st] = (new_states[new_st] + w * factor) % p
                else:
                    cycle_sign = (mu + sign + mv) & 1
                    # Detect quotient loop (σ-fixed cover edge): u == v.
                    is_loop = (u == v)
                    if balance[bu]:
                        if cycle_sign == 0:
                            # bal cycle in bal: no balance change.
                            # Non-loop: factor (y-1)^2. Loop: factor (y-1).
                            factor = factor_loop_bal_cycle_in_bal if is_loop else factor_cycle_no_rank_change
                            new_states[state] = (new_states[state] + w * factor) % p
                        else:
                            # unbal cycle in bal: block becomes unbal.
                            # Non-loop: factor (x-1)^{-1}(y-1). Loop: factor (x-1)^{-1}.
                            factor = factor_loop_unbal_cycle_in_bal if is_loop else factor_unbal_cycle_in_bal
                            new_balance = list(balance)
                            new_balance[bu] = False
                            new_mono = list(monodromy)
                            for i in range(len(partition)):
                                if partition[i] == bu:
                                    new_mono[i] = 0
                            new_st = (partition, tuple(new_mono), tuple(new_balance), finalized)
                            new_states[new_st] = (new_states[new_st] + w * factor) % p
                    else:
                        # cycle in unbal: no rank/balance change.
                        # Non-loop: factor (y-1)^2. Loop: factor (y-1).
                        factor = factor_loop_cycle_in_unbal if is_loop else factor_cycle_no_rank_change
                        new_states[state] = (new_states[state] + w * factor) % p
            states = dict(new_states)
            if len(states) > max_states:
                max_states = len(states)

        forget_vertex = elim_order[step]
        still_needed = False
        for ei in range(edge_idx, n_edges):
            _, fu, fv, _ = edge_process_step[ei]
            if fu == forget_vertex or fv == forget_vertex:
                still_needed = True
                break

        if not still_needed:
            fpos = active_pos[forget_vertex]
            new_states = {}
            for state, w in states.items():
                partition, monodromy, balance, finalized = state
                n_active = len(partition)
                fpos_bid = partition[fpos]
                fpos_size = 0
                for b in partition:
                    if b == fpos_bid:
                        fpos_size += 1
                if fpos_size == 1:
                    new_fin = finalized + (1 if balance[fpos_bid] else 0)
                else:
                    new_fin = finalized
                block_id_map = {}
                block_first_mon_list = []
                bal_list = []
                n_new = n_active - 1
                part = [0] * n_new
                mono = [0] * n_new
                new_pos = 0
                for i in range(n_active):
                    if i == fpos:
                        continue
                    old_bid = partition[i]
                    cur_mon = monodromy[i]
                    canon_id = block_id_map.get(old_bid)
                    if canon_id is None:
                        canon_id = len(block_id_map)
                        block_id_map[old_bid] = canon_id
                        block_first_mon_list.append(cur_mon)
                        bal_list.append(balance[old_bid])
                    part[new_pos] = canon_id
                    if bal_list[canon_id]:
                        mono[new_pos] = (cur_mon + block_first_mon_list[canon_id]) & 1
                    new_pos += 1
                new_st = (tuple(part), tuple(mono), tuple(bal_list), new_fin)
                new_states[new_st] = (new_states.get(new_st, 0) + w) % p
            states = new_states
            del active_pos[forget_vertex]
            for v in list(active_pos.keys()):
                if active_pos[v] > fpos:
                    active_pos[v] -= 1

    # Final aggregation: T_fix^σ = (x-1)^{r_E_G} × Σ_state weight.
    # Weight already encodes (x-1)^{-r_quot - r_signed} (y-1)^{2|L| - r_quot - r_signed}.
    total_weight = 0
    for state, w in states.items():
        total_weight = (total_weight + w) % p
    result = (pow(x_minus_1, r_E_G, p) * total_weight) % p
    return result, max_states


def compute_signed_tutte_elim_mod(
    nodes: List, edges_with_signs: List[Tuple[Tuple, int]],
    x_val: int, y_val: int, p: int,
    engine: str = "python",
) -> Tuple[int, int]:
    """Compute T_signed(G; x_val, y_val) mod p via elimination-order path DP.

    Forgets each vertex after its last edge is processed, bounding state.
    Returns (value_mod_p, max_state_count_observed).

    Args:
      engine: "python" (default, pure Python) or "c" (use C-ext for
              inner loops; falls back to Python if C-ext unavailable).
              "auto" tries C-ext first.
    """
    nm = {v: i for i, v in enumerate(sorted(set(nodes)))}
    n = len(nm)
    edges_int = [((nm[u], nm[v]), sign) for ((u, v), sign) in edges_with_signs]

    # Determine elimination order via min-fill heuristic
    adj: Dict[int, set] = {v: set() for v in range(n)}
    for (u, v), _ in edges_int:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    # Try multiple heuristics; pick the one with smallest max bag size (proxy for state count).
    best_order = None
    best_max_bag = float('inf')
    for h in ("minfill", "minfill_degree", "mindegree"):
        order = _elimination_ordering(adj, list(range(n)), heuristic=h, max_width=20)
        if order is None:
            continue
        # Estimate max bag size by simulating
        remaining = set(range(n))
        elim_adj_sim = {v: set(adj[v]) for v in range(n)}
        max_bag = 0
        for v in order:
            nb = elim_adj_sim[v] & remaining
            if len(nb) > max_bag:
                max_bag = len(nb)
            for u in nb:
                for w in nb:
                    if u != w:
                        elim_adj_sim[u].add(w)
            remaining.discard(v)
        if max_bag < best_max_bag:
            best_max_bag = max_bag
            best_order = order
    elim_order = best_order if best_order is not None else list(range(n))

    # For each vertex, find when it's LAST seen in any edge
    last_seen = {v: -1 for v in range(n)}
    elim_pos = {v: i for i, v in enumerate(elim_order)}
    # For each edge, decide when to process it: at max elim_pos of endpoints
    edge_process_step = []
    for (u, v), sign in edges_int:
        step = max(elim_pos[u], elim_pos[v])
        edge_process_step.append((step, u, v, sign))
    edge_process_step.sort()  # process in step order

    # Active vertex tracking: each vertex is "active" from start until forgotten.
    # We use a mapping `active_pos[v]` → position in current state's partition tuple.
    # Initially all vertices active at their original index.
    active_pos = {v: v for v in range(n)}

    # Initial state: all singletons, all balanced, no edges processed
    init_partition = tuple(range(n))
    init_mon = tuple(0 for _ in range(n))
    init_balance = tuple(True for _ in range(n))
    init_state = (init_partition, init_mon, init_balance, 0)
    states = {init_state: 1}

    y_minus_1 = (y_val - 1) % p
    max_states = 1

    # Process edges in step order; after each step, forget the vertex
    # whose elim_pos == current step (if no more edges reference it).
    edge_idx = 0
    n_edges = len(edge_process_step)

    use_c = engine in ("c", "auto")
    if use_c:
        try:
            from ._signed_elim_c import step_edge_batch, step_forget_batch
        except ImportError:
            use_c = False

    # Try FULL DP in C — eliminates per-step Python encode/decode overhead.
    if use_c:
        try:
            from ._signed_elim_c import dp_full as _dp_full
        except ImportError:
            _dp_full = None
        if _dp_full is not None:
            # Compute signed rank of full edge set (r_E) — needed by C aggregation.
            parent_full = list(range(n))
            mon_full = [0] * n
            bal_full = [True] * n
            for ((u_, v_), sign_) in edges_int:
                cur = u_; mu_ = 0
                while parent_full[cur] != cur:
                    mu_ = (mu_ + mon_full[cur]) & 1
                    cur = parent_full[cur]
                ru_ = cur
                cur = v_; mv_ = 0
                while parent_full[cur] != cur:
                    mv_ = (mv_ + mon_full[cur]) & 1
                    cur = parent_full[cur]
                rv_ = cur
                if ru_ != rv_:
                    target = min(ru_, rv_); other = max(ru_, rv_)
                    off_ = (mu_ + sign_ + mv_) & 1 if target == ru_ else (mv_ + sign_ + mu_) & 1
                    parent_full[other] = target
                    mon_full[other] = off_
                    bal_full[target] = bal_full[ru_] and bal_full[rv_]
                else:
                    cs_ = (mu_ + sign_ + mv_) & 1
                    if cs_ != 0:
                        bal_full[ru_] = False
            roots_full = set()
            for i in range(n):
                cur = i
                while parent_full[cur] != cur:
                    cur = parent_full[cur]
                roots_full.add(cur)
            n_balanced_full = sum(1 for r in roots_full if bal_full[r])
            r_E = n - n_balanced_full
            edges_uv = [(u_, v_) for (u_, v_), _ in edges_int]
            signs_list = [sign_ for (_, sign_) in edges_int]
            x_minus_1_val = (x_val - 1) % p
            result = _dp_full(
                n, edges_uv, signs_list, elim_order, r_E,
                x_minus_1_val, y_minus_1, p,
            )
            if result is not None:
                return result

    for step in range(len(elim_order)):
        # Process all edges with edge_process_step == step
        while edge_idx < n_edges and edge_process_step[edge_idx][0] == step:
            _step, u, v, sign = edge_process_step[edge_idx]
            edge_idx += 1
            # Lift loop-invariants outside per-state loop
            u_pos = active_pos[u]
            v_pos = active_pos[v]
            # Try C-ext fast path.
            if use_c:
                new_states = step_edge_batch(
                    states, u_pos, v_pos, sign, False, y_minus_1, p
                )
                if new_states is not None:
                    states = new_states
                    if len(states) > max_states:
                        max_states = len(states)
                    continue
            # Process this edge — branch on delete/keep (pure Python fallback)
            new_states = defaultdict(int)
            for state, w in states.items():
                partition, monodromy, balance, finalized = state
                bu = partition[u_pos]
                bv = partition[v_pos]
                mu = monodromy[u_pos]
                mv = monodromy[v_pos]

                # Delete branch
                new_states[state] = (new_states[state] + w) % p

                # Keep branch
                if bu != bv:
                    # Merge two different blocks via tree edge.
                    # Signed-graph matroid rank/nullity:
                    #   - both balanced or one balanced + one unbalanced: Δr = +1, factor 1
                    #   - both UNBALANCED: Δr = 0, factor (y-1) (handcuff circuit)
                    offset = (mu + sign + mv) & 1
                    bal_bu = balance[bu]
                    bal_bv = balance[bv]
                    both_unbalanced = (not bal_bu) and (not bal_bv)
                    combined = bal_bu and bal_bv
                    factor = y_minus_1 if both_unbalanced else 1
                    n_active = len(partition)
                    # Single-pass merge + canonicalize.
                    # Monodromy in unbalanced blocks is canonicalized to 0.
                    block_id_map = {}
                    block_first_mon_list = []
                    bal_list = []
                    part = [0] * n_active
                    mono = [0] * n_active
                    for i in range(n_active):
                        old_bid = partition[i]
                        if old_bid == bv:
                            merged_bid = bu
                            merged_mon = (monodromy[i] + offset) & 1
                        else:
                            merged_bid = old_bid
                            merged_mon = monodromy[i]
                        canon_id = block_id_map.get(merged_bid)
                        if canon_id is None:
                            canon_id = len(block_id_map)
                            block_id_map[merged_bid] = canon_id
                            block_first_mon_list.append(merged_mon)
                            if merged_bid == bu:
                                bal_list.append(combined)
                            else:
                                bal_list.append(balance[merged_bid])
                        part[i] = canon_id
                        if bal_list[canon_id]:
                            mono[i] = (merged_mon + block_first_mon_list[canon_id]) & 1
                        # else: mono[i] stays 0 (canonicalized for unbalanced block)
                    new_st = (tuple(part), tuple(mono), tuple(bal_list), finalized)
                    new_states[new_st] = (new_states[new_st] + w * factor) % p
                else:
                    cycle_sign = (mu + sign + mv) & 1
                    if balance[bu]:
                        if cycle_sign == 0:
                            # Balanced cycle in balanced: factor (y-1)
                            new_states[state] = (new_states[state] + w * y_minus_1) % p
                        else:
                            # Unbalanced cycle in balanced: block becomes unbalanced, factor 1.
                            # Block bu just became unbalanced — zero out its monodromies.
                            new_balance = list(balance)
                            new_balance[bu] = False
                            new_mono = list(monodromy)
                            for i in range(len(partition)):
                                if partition[i] == bu:
                                    new_mono[i] = 0
                            new_st = (partition, tuple(new_mono), tuple(new_balance), finalized)
                            new_states[new_st] = (new_states[new_st] + w) % p
                    else:
                        # Cycle in already-unbalanced: factor (y-1)
                        new_states[state] = (new_states[state] + w * y_minus_1) % p
            states = dict(new_states)
            if len(states) > max_states:
                max_states = len(states)

        # Forget vertex at elim_order[step] (if no more edges reference it)
        forget_vertex = elim_order[step]
        # Check: are there future edges still referencing forget_vertex?
        # Future edges: edge_process_step[edge_idx:] with steps > current step
        # If any of them involves forget_vertex, don't forget.
        still_needed = False
        for ei in range(edge_idx, n_edges):
            _, fu, fv, _ = edge_process_step[ei]
            if fu == forget_vertex or fv == forget_vertex:
                still_needed = True
                break

        if not still_needed:
            # Forget this vertex — single-pass drop + canonicalize.
            fpos = active_pos[forget_vertex]
            if use_c:
                new_states_c = step_forget_batch(states, fpos, p)
                if new_states_c is not None:
                    states = new_states_c
                    # Update active_pos: vertices after fpos shift down by 1
                    del active_pos[forget_vertex]
                    for v_ in list(active_pos.keys()):
                        if active_pos[v_] > fpos:
                            active_pos[v_] -= 1
                    if len(states) > max_states:
                        max_states = len(states)
                    continue
            new_states = {}
            for state, w in states.items():
                partition, monodromy, balance, finalized = state
                n_active = len(partition)
                fpos_bid = partition[fpos]
                # Count active members of fpos's block.
                fpos_size = 0
                for b in partition:
                    if b == fpos_bid:
                        fpos_size += 1
                # If block fully forgotten (size 1), add its balance to finalized.
                if fpos_size == 1:
                    new_fin = finalized + (1 if balance[fpos_bid] else 0)
                else:
                    new_fin = finalized
                # Drop position fpos and canonicalize in single pass.
                # Monodromy in unbalanced blocks is canonicalized to 0.
                block_id_map = {}
                block_first_mon_list = []
                bal_list = []
                n_new = n_active - 1
                part = [0] * n_new
                mono = [0] * n_new
                new_pos = 0
                for i in range(n_active):
                    if i == fpos:
                        continue
                    old_bid = partition[i]
                    cur_mon = monodromy[i]
                    canon_id = block_id_map.get(old_bid)
                    if canon_id is None:
                        canon_id = len(block_id_map)
                        block_id_map[old_bid] = canon_id
                        block_first_mon_list.append(cur_mon)
                        bal_list.append(balance[old_bid])
                    part[new_pos] = canon_id
                    if bal_list[canon_id]:
                        mono[new_pos] = (cur_mon + block_first_mon_list[canon_id]) & 1
                    # else: mono[new_pos] stays 0 (already initialized) for unbalanced
                    new_pos += 1
                new_st = (tuple(part), tuple(mono), tuple(bal_list), new_fin)
                new_states[new_st] = (new_states.get(new_st, 0) + w) % p
            states = new_states
            # Update active_pos: vertices after fpos shift down by 1
            del active_pos[forget_vertex]
            for v in list(active_pos.keys()):
                if active_pos[v] > fpos:
                    active_pos[v] -= 1

    # After all elimination steps, states should be empty-active
    # Sum: T = sum over final states of (x-1)^{r_E - r_state} × weight
    # where r_state = n - finalized_balance_count

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

    x_minus_1 = (x_val - 1) % p
    total = 0
    for state, w in states.items():
        partition, monodromy, balance, finalized = state
        n_active = len(partition)
        # Add active blocks to balanced count
        active_bal = sum(1 for b in balance if b)
        n_balanced_state = finalized + active_bal
        # r_state = n - balanced_state_count; but active partitions also contribute non-balanced
        # Actually: at the end, active should be empty (all forgotten). If not, treat active blocks as components.
        n_total_components = finalized + len([b for b in balance])
        # In the final state, the "rank" of the subset (in cover G) = n - # balanced components - n_active (since active verts not forgotten contribute components)
        # Actually we should think of unforgotten verts as additional components
        # For pure final state (n_active == 0): r_state = n - finalized = n - balanced_components
        r_state = n - n_balanced_state
        # Adjust for active verts: each active vert that's not in a forgotten block is an "extra" component
        # but they should be 0 at end of processing
        total = (total + w * pow(x_minus_1, r_E - r_state, p)) % p

    return total, max_states
