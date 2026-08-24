"""C extension for signed-graph elimination-order DP — inner edge loop.

Hot path: `compute_signed_tutte_elim_mod` / `compute_t_fix_sigma_mod`
iterate per-state work over up to ~10^6 signed-graph DP states. Per-state
work includes:
  - Decoding (partition, monodromy, balance) state tuple
  - Computing branch (delete + keep tree-edge / cycle case)
  - Canonicalizing the new state under block re-labeling + unbalanced-mono-zero
  - Updating a dict[state_tuple → int_mod_p] aggregator

Per `tutte/docs/06_9_signed_equivariant_dp.md`: ~78% of single-point
Z(1,2) runtime (~33s) is in the Python body of these loops. C-ext
target ~5-10× speedup → ~3-7s/point.

State encoding (flat int array):
  [n_active, n_blocks, finalized,
   part[0], part[1], ..., part[n_active-1],
   mono[0], mono[1], ..., mono[n_active-1],
   bal[0], bal[1], ..., bal[n_blocks-1]]
total: 3 + 2*n_active + n_blocks ints.

Pattern mirrors `tutte/roots/_partition_c.py`: lazy auto-compile on first
import; pure-Python fallback if cffi/C compile fails.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import cffi

ffi = cffi.FFI()

ffi.cdef(r"""
    /* Apply one edge step to a batch of signed-DP states.

       Inputs:
         in_states_data:   flat array of encoded states, concatenated.
         in_states_offsets: offsets[i]..offsets[i+1] = state i's slice.
         in_weights:       weights mod p, one per input state (length n_in).
         n_in:             number of input states.
         u_pos, v_pos:     edge endpoint positions in active partition (0..n_active-1).
                           For loop case (u == v in cover), pass u_pos == v_pos.
         sign:             0 or 1 — quotient edge sign (monodromy χ).
         is_loop:          1 if edge is a quotient loop (σ-fixed cover edge), else 0.
         y_minus_1, p:     (y - 1) mod p, prime p.

       Outputs:
         out_states_data:   flat array of new encoded states.
         out_states_offsets: offsets for output states.
         out_weights:       weights mod p, one per output state (after aggregation).
         out_n:             number of distinct output states after aggregation.
         out_capacity_data, out_capacity_weights: caller-allocated capacities.

       Returns 0 on success, negative on error:
         -1: out_capacity_data exceeded
         -2: out_capacity_weights exceeded
         -3: input encoding malformed
         -4: invalid state size (>16 active positions)
    */
    int signed_step_edge_batch_c(
        const int* in_states_data, const int* in_states_offsets,
        const long long* in_weights, int n_in,
        int u_pos, int v_pos, int sign, int is_loop,
        long long y_minus_1, long long p,
        int* out_states_data, int* out_states_offsets,
        long long* out_weights, int* out_n,
        int out_capacity_data, int out_capacity_weights);

    /* Apply forget step to a batch of states.

       Inputs:
         in_states_data:    flat encoded states.
         in_states_offsets: offsets per state.
         in_weights:        weights mod p (length n_in).
         n_in:              # input states.
         fpos:              position to forget (0..n_active-1).
         p:                 prime modulus.

       Outputs same as signed_step_edge_batch_c.
       After forget: n_active decreases by 1, blocks renumbered, finalized
       potentially increments by 1 if forgotten position's block becomes empty
       AND was balanced.

       Returns 0 on success, negative on error (same codes).
    */
    int signed_step_forget_batch_c(
        const int* in_states_data, const int* in_states_offsets,
        const long long* in_weights, int n_in,
        int fpos, long long p,
        int* out_states_data, int* out_states_offsets,
        long long* out_weights, int* out_n,
        int out_capacity_data, int out_capacity_weights);

    /* Full signed-DP loop in C — eliminates per-step Python encode/decode.

       Replaces the entire compute_signed_tutte_elim_mod main loop:
       processes all edges in elim order, forgets vertices when no
       longer referenced, and returns the final T_signed(x, y) mod p.

       Args:
         n_verts:           number of quotient vertices (≤16).
         n_edges:           number of quotient edges.
         edges_uv:          flat array [2*n_edges], pairs (u_e, v_e).
         signs:             [n_edges], 0 or 1 per edge.
         elim_order:        [n_verts], vertex elimination order.
         r_E:               rank of full edge set in cover G.
         x_minus_1, y_minus_1, p: evaluation point.

       Outputs:
         out_total:         T_signed(x, y) mod p.
         out_max_states:    max state count observed.

       Returns 0 on success, negative on error.
    */
    int signed_dp_full_c(
        int n_verts, int n_edges,
        const int* edges_uv, const int* signs,
        const int* elim_order,
        int r_E,
        long long x_minus_1, long long y_minus_1, long long p,
        long long* out_total, int* out_max_states);

    /* σ-equivariant per-orbit DP on cover G — computes T(G) directly via
       σ-canonicalized states. State is (partition, finalized) — no
       monodromy or balance (unsigned Tutte).

       Process σ-orbits of edges as 4-branch steps (del/del, del/keep,
       keep/del, keep/keep). After each transition, apply σ-canonicalization
       (lex-min of partition vs σ-image).

       Args:
         n_verts:        cover G vertex count (must be even for free σ).
         n_orbits:       # σ-orbits of edges = |E| / 2.
         edges_pairs:    flat [4*n_orbits] = [u1,v1,u2,v2, ...] per orbit.
         pair_order:     flat [n_verts] — pairs of (v, σv) consecutive
                         (positions 2k, 2k+1 are pair k).
         perm:           [n_verts] — σ(v) for each v.
         r_E:            rank of full edge set in G.
         x_minus_1, y_minus_1, p: evaluation point.

       Outputs:
         out_total:      T(G; x, y) mod p
         out_max_states: max state count observed
    */
    int sigma_orbit_dp_full_c(
        int n_verts, int n_orbits,
        const int* edges_pairs,
        const int* pair_order,
        const int* perm,
        int r_E,
        long long x_minus_1, long long y_minus_1, long long p,
        long long* out_total, int* out_max_states);

    /* T_fix^σ full DP in C — analogous to signed_dp_full_c with per-edge
       loop / non-loop factor handling and (x-1)^{r_E_G} final aggregation.

       Args (in addition to signed_dp_full_c):
         is_loop:           [n_edges], 1 if quotient edge is a loop (σ-fixed cover edge), else 0.
         r_E_G:             rank of full edge set in cover G.
         factor_tree_xbal_xbal:        (x-1)^{-2} mod p
         factor_tree_unbal_unbal:      (x-1)^{-1}(y-1) mod p
         factor_cycle_no_rank_change:  (y-1)^2 mod p — non-loop bal cycle bal / cycle in unbal
         factor_unbal_cycle_in_bal:    (x-1)^{-1}(y-1) mod p — non-loop unbal cycle in bal
         factor_loop_bal_cycle_in_bal: (y-1) — loop bal cycle in bal
         factor_loop_unbal_cycle_in_bal: (x-1)^{-1} — loop unbal cycle in bal
         factor_loop_cycle_in_unbal:   (y-1) — loop cycle in unbal

       Outputs:
         out_total:        T_fix^σ(x, y) mod p
         out_max_states:   max state count observed
    */
    int t_fix_dp_full_c(
        int n_verts, int n_edges,
        const int* edges_uv, const int* signs, const int* is_loop,
        const int* elim_order,
        int r_E_G,
        long long x_minus_1, long long p,
        long long factor_tree_xbal_xbal, long long factor_tree_unbal_unbal,
        long long factor_cycle_no_rank_change, long long factor_unbal_cycle_in_bal,
        long long factor_loop_bal_cycle_in_bal, long long factor_loop_unbal_cycle_in_bal,
        long long factor_loop_cycle_in_unbal,
        long long* out_total, int* out_max_states);
""")

ffi.set_source("_tutte_signed_elim_cffi", r"""
    #include <stdlib.h>
    #include <string.h>
    #include <stdint.h>
    #include <stdio.h>

    /* =========================================================================
       SIGNED-DP STATE ENCODING
       =========================================================================
       Each state is encoded as flat int array:
         [n_active, n_blocks, finalized,
          part[0..n-1], mono[0..n-1], bal[0..n_blocks-1]]
       Length: 3 + 2*n_active + n_blocks.

       n_active ≤ 16 in practice (treewidth-bounded).
       n_blocks ≤ n_active.
       finalized ∈ [0, n_quotient_verts].
       part[i] ∈ [0, n_blocks).
       mono[i] ∈ {0, 1} (1 bit, but stored as int for simplicity).
                Zeroed for positions in unbalanced blocks.
       bal[b] ∈ {0, 1} (1 bit).
       ========================================================================= */

    /* Decode state header. Returns 0 on success, -1 if malformed. */
    static int decode_header(const int* state, int* n_active, int* n_blocks, int* finalized) {
        *n_active = state[0];
        *n_blocks = state[1];
        *finalized = state[2];
        if (*n_active < 0 || *n_active > 16) return -1;
        if (*n_blocks < 0 || *n_blocks > *n_active) return -1;
        return 0;
    }

    /* Compute state length given header. */
    static int state_length(int n_active, int n_blocks) {
        return 3 + 2 * n_active + n_blocks;
    }

    /* Pointers into a state for part/mono/bal segments. */
    static const int* state_part(const int* state) { return state + 3; }
    static const int* state_mono(const int* state) { int n = state[0]; return state + 3 + n; }
    static const int* state_bal(const int* state)  { int n = state[0]; return state + 3 + 2*n; }

    /* Modular multiplication: (a * b) % p, assumes a, b < 2^62. */
    static long long mod_mul(long long a, long long b, long long p) {
        /* For p < 2^62, a*b can overflow int64. Use __int128 for safety on
           reasonable-sized primes (p < 2^32 fits in int64 directly). */
        if (p < (1LL << 32)) {
            return (a * b) % p;
        }
        return (long long)(((__int128)a * b) % p);
    }

    /* Compute new state after MERGE of blocks bu and bv (both ≠ -1) into bu.
       offset: monodromy offset to add to bv's positions.
       combined_balance: new balance for merged block bu.
       Writes new state to out_buf, returns # ints written. */
    static int build_merged_state(
        const int* state, int bu, int bv, int offset, int combined_balance,
        int* out_buf)
    {
        int n_active = state[0];
        int finalized = state[2];
        const int* part_in = state_part(state);
        const int* mono_in = state_mono(state);
        const int* bal_in = state_bal(state);

        /* Step 1: Compute merged (block_id, mono) per position. */
        int merged_bid[16];
        int merged_mon[16];
        for (int i = 0; i < n_active; i++) {
            if (part_in[i] == bv) {
                merged_bid[i] = bu;
                merged_mon[i] = (mono_in[i] + offset) & 1;
            } else {
                merged_bid[i] = part_in[i];
                merged_mon[i] = mono_in[i];
            }
        }

        /* Step 2: Canonicalize block ids (relabel by first appearance). */
        int block_id_map[16];
        for (int i = 0; i < 16; i++) block_id_map[i] = -1;
        int new_n_blocks = 0;
        int block_first_mon[16];
        int new_bal[16];
        int canon_part[16];
        int canon_mono[16];

        for (int i = 0; i < n_active; i++) {
            int old_bid = merged_bid[i];
            int cid = block_id_map[old_bid];
            if (cid < 0) {
                cid = new_n_blocks++;
                block_id_map[old_bid] = cid;
                block_first_mon[cid] = merged_mon[i];
                if (old_bid == bu) {
                    new_bal[cid] = combined_balance;
                } else {
                    new_bal[cid] = bal_in[old_bid];
                }
            }
            canon_part[i] = cid;
            /* In balanced blocks, rebase monodromy so first position has 0.
               In unbalanced blocks, canonicalize all monodromies to 0. */
            if (new_bal[cid]) {
                canon_mono[i] = (merged_mon[i] + block_first_mon[cid]) & 1;
            } else {
                canon_mono[i] = 0;
            }
        }

        /* Step 3: Write to out_buf. */
        int idx = 0;
        out_buf[idx++] = n_active;
        out_buf[idx++] = new_n_blocks;
        out_buf[idx++] = finalized;
        for (int i = 0; i < n_active; i++) out_buf[idx++] = canon_part[i];
        for (int i = 0; i < n_active; i++) out_buf[idx++] = canon_mono[i];
        for (int b = 0; b < new_n_blocks; b++) out_buf[idx++] = new_bal[b];
        return idx;
    }

    /* Build state with block bu marked unbalanced (mono zero'd for its positions). */
    static int build_unbalanced_state(
        const int* state, int bu,
        int* out_buf)
    {
        int n_active = state[0];
        int n_blocks = state[1];
        int finalized = state[2];
        const int* part_in = state_part(state);
        const int* mono_in = state_mono(state);
        const int* bal_in = state_bal(state);

        int idx = 0;
        out_buf[idx++] = n_active;
        out_buf[idx++] = n_blocks;
        out_buf[idx++] = finalized;
        /* partition unchanged */
        for (int i = 0; i < n_active; i++) out_buf[idx++] = part_in[i];
        /* monodromy: zero for block bu (now unbalanced), unchanged elsewhere */
        for (int i = 0; i < n_active; i++) {
            if (part_in[i] == bu) out_buf[idx++] = 0;
            else                  out_buf[idx++] = mono_in[i];
        }
        /* balance: set block bu false, rest unchanged */
        for (int b = 0; b < n_blocks; b++) {
            if (b == bu) out_buf[idx++] = 0;
            else         out_buf[idx++] = bal_in[b];
        }
        return idx;
    }

    /* Copy state unchanged. */
    static int copy_state(const int* state, int* out_buf) {
        int n_active = state[0];
        int n_blocks = state[1];
        int len = state_length(n_active, n_blocks);
        memcpy(out_buf, state, len * sizeof(int));
        return len;
    }

    /* =========================================================================
       SIMPLE HASH MAP for output state aggregation
       =========================================================================
       Keys = byte-strings (the encoded state). Values = int64 weights mod p.
       Open addressing with linear probing. Capacity ≥ 2 × expected_distinct.
       ========================================================================= */

    typedef struct {
        const int* key_data;  /* pointer into out_states_data */
        int        key_len;
        long long  weight;
        int        in_use;
    } HashEntry;

    /* FNV-1a hash on int array bytes. */
    static uint64_t hash_state(const int* data, int len) {
        uint64_t h = 14695981039346656037ULL;
        const unsigned char* p = (const unsigned char*)data;
        int nbytes = len * (int)sizeof(int);
        for (int i = 0; i < nbytes; i++) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    static int state_eq(const int* a, int alen, const int* b, int blen) {
        if (alen != blen) return 0;
        for (int i = 0; i < alen; i++) if (a[i] != b[i]) return 0;
        return 1;
    }

    /* Insert or accumulate `weight` for key (data, len). */
    static int hash_insert(
        HashEntry* table, int capacity,
        const int* data, int len, long long weight, long long p)
    {
        uint64_t h = hash_state(data, len);
        int probe = (int)(h % (uint64_t)capacity);
        for (int i = 0; i < capacity; i++) {
            int idx = (probe + i) % capacity;
            if (!table[idx].in_use) {
                table[idx].in_use = 1;
                table[idx].key_data = data;
                table[idx].key_len = len;
                table[idx].weight = weight % p;
                return 1; /* new */
            }
            if (state_eq(table[idx].key_data, table[idx].key_len, data, len)) {
                table[idx].weight = (table[idx].weight + weight) % p;
                return 0; /* updated */
            }
        }
        return -1; /* table full */
    }

    /* =========================================================================
       EDGE STEP: process one edge across all input states
       ========================================================================= */
    int signed_step_edge_batch_c(
        const int* in_states_data, const int* in_states_offsets,
        const long long* in_weights, int n_in,
        int u_pos, int v_pos, int sign, int is_loop,
        long long y_minus_1, long long p,
        int* out_states_data, int* out_states_offsets,
        long long* out_weights, int* out_n,
        int out_capacity_data, int out_capacity_weights)
    {
        /* Allocate hash table sized for 2× expected outputs (delete + keep
           per input gives at most 2*n_in distinct states; usually fewer). */
        int hash_cap = 4 * n_in + 16;
        HashEntry* table = (HashEntry*)calloc(hash_cap, sizeof(HashEntry));
        if (!table) return -5;

        /* Scratch buffer for building candidate states before hashing. */
        int scratch[64];

        int out_data_pos = 0;

        for (int i = 0; i < n_in; i++) {
            const int* state = in_states_data + in_states_offsets[i];
            long long w = in_weights[i];

            int n_active, n_blocks, finalized;
            if (decode_header(state, &n_active, &n_blocks, &finalized) < 0) {
                free(table);
                return -3;
            }
            if (n_active > 16) { free(table); return -4; }

            const int* part = state_part(state);
            const int* mono = state_mono(state);
            const int* bal  = state_bal(state);

            int bu = part[u_pos];
            int bv = part[v_pos];
            int mu = mono[u_pos];
            int mv = mono[v_pos];

            /* ===== DELETE BRANCH: state unchanged, weight += w ===== */
            int del_len = copy_state(state, scratch);
            if (out_data_pos + del_len > out_capacity_data) {
                free(table); return -1;
            }
            int* del_target = out_states_data + out_data_pos;
            memcpy(del_target, scratch, del_len * sizeof(int));
            hash_insert(table, hash_cap, del_target, del_len, w, p);
            out_data_pos += del_len;

            /* ===== KEEP BRANCH ===== */
            if (bu != bv) {
                /* Tree edge: merge bu and bv */
                int offset = (mu + sign + mv) & 1;
                int bal_bu = bal[bu];
                int bal_bv = bal[bv];
                int both_unbal = (!bal_bu) && (!bal_bv);
                int combined = bal_bu && bal_bv;
                long long factor = both_unbal ? y_minus_1 : 1LL;
                int merge_len = build_merged_state(
                    state, bu, bv, offset, combined, scratch);
                if (out_data_pos + merge_len > out_capacity_data) {
                    free(table); return -1;
                }
                int* merge_target = out_states_data + out_data_pos;
                memcpy(merge_target, scratch, merge_len * sizeof(int));
                long long w2 = mod_mul(w, factor, p);
                hash_insert(table, hash_cap, merge_target, merge_len, w2, p);
                out_data_pos += merge_len;
            } else {
                /* Cycle: same block bu */
                int cycle_sign = (mu + sign + mv) & 1;
                if (bal[bu]) {
                    if (cycle_sign == 0) {
                        /* Balanced cycle in balanced: factor (y-1) for non-loop, (y-1) for loop too actually...
                           Actually per Python: non-loop has (y-1) once, loop has (y-1).
                           For SIGNED graph Tutte (not T_fix), is_loop only matters for T_fix
                           which has compound factors. For T_signed, just (y-1). */
                        int copy_len = copy_state(state, scratch);
                        if (out_data_pos + copy_len > out_capacity_data) {
                            free(table); return -1;
                        }
                        int* tgt = out_states_data + out_data_pos;
                        memcpy(tgt, scratch, copy_len * sizeof(int));
                        long long w2 = mod_mul(w, y_minus_1, p);
                        hash_insert(table, hash_cap, tgt, copy_len, w2, p);
                        out_data_pos += copy_len;
                    } else {
                        /* Unbalanced cycle in balanced: block becomes unbalanced, factor 1 */
                        int unb_len = build_unbalanced_state(state, bu, scratch);
                        if (out_data_pos + unb_len > out_capacity_data) {
                            free(table); return -1;
                        }
                        int* tgt = out_states_data + out_data_pos;
                        memcpy(tgt, scratch, unb_len * sizeof(int));
                        hash_insert(table, hash_cap, tgt, unb_len, w, p);
                        out_data_pos += unb_len;
                    }
                } else {
                    /* Cycle in unbalanced: factor (y-1) */
                    int copy_len = copy_state(state, scratch);
                    if (out_data_pos + copy_len > out_capacity_data) {
                        free(table); return -1;
                    }
                    int* tgt = out_states_data + out_data_pos;
                    memcpy(tgt, scratch, copy_len * sizeof(int));
                    long long w2 = mod_mul(w, y_minus_1, p);
                    hash_insert(table, hash_cap, tgt, copy_len, w2, p);
                    out_data_pos += copy_len;
                }
            }
        }

        /* Extract distinct entries from hash table into compact output.
           Bug fix: hash table keys are pointers INTO out_states_data; if we
           memcpy compact-direction it overwrites later sources. Allocate a
           temp buffer, copy keys there, then copy back. */
        int* aux = (int*)malloc(out_data_pos * sizeof(int));
        if (!aux && out_data_pos > 0) { free(table); return -5; }
        int n_out = 0;
        int compact_data_pos = 0;
        for (int i = 0; i < hash_cap; i++) {
            if (table[i].in_use) {
                if (n_out >= out_capacity_weights) {
                    free(aux); free(table); return -2;
                }
                if (compact_data_pos + table[i].key_len > out_capacity_data) {
                    free(aux); free(table); return -1;
                }
                out_states_offsets[n_out] = compact_data_pos;
                memcpy(aux + compact_data_pos, table[i].key_data,
                       table[i].key_len * sizeof(int));
                compact_data_pos += table[i].key_len;
                out_weights[n_out] = table[i].weight;
                n_out++;
            }
        }
        out_states_offsets[n_out] = compact_data_pos;
        if (compact_data_pos > 0) {
            memcpy(out_states_data, aux, compact_data_pos * sizeof(int));
        }
        free(aux);
        *out_n = n_out;
        free(table);
        return 0;
    }

    /* =========================================================================
       FORGET STEP: drop one position from each state
       ========================================================================= */
    int signed_step_forget_batch_c(
        const int* in_states_data, const int* in_states_offsets,
        const long long* in_weights, int n_in,
        int fpos, long long p,
        int* out_states_data, int* out_states_offsets,
        long long* out_weights, int* out_n,
        int out_capacity_data, int out_capacity_weights)
    {
        int hash_cap = 2 * n_in + 16;
        HashEntry* table = (HashEntry*)calloc(hash_cap, sizeof(HashEntry));
        if (!table) return -5;

        int scratch[64];
        int out_data_pos = 0;

        for (int i = 0; i < n_in; i++) {
            const int* state = in_states_data + in_states_offsets[i];
            long long w = in_weights[i];

            int n_active, n_blocks, finalized;
            if (decode_header(state, &n_active, &n_blocks, &finalized) < 0) {
                free(table); return -3;
            }
            if (n_active < 1) { free(table); return -3; }

            const int* part = state_part(state);
            const int* mono = state_mono(state);
            const int* bal  = state_bal(state);

            int fpos_bid = part[fpos];
            /* Count members of fpos's block. */
            int fpos_size = 0;
            for (int j = 0; j < n_active; j++) {
                if (part[j] == fpos_bid) fpos_size++;
            }
            int new_finalized = finalized;
            if (fpos_size == 1 && bal[fpos_bid]) {
                new_finalized++;
            }
            int new_n_active = n_active - 1;

            /* Single-pass: drop fpos, relabel blocks by first appearance,
               canonicalize monodromies (zero in unbalanced blocks, rebase in balanced). */
            int block_id_map[16];
            for (int b = 0; b < 16; b++) block_id_map[b] = -1;
            int block_first_mon[16];
            int new_bal[16];
            int new_n_blocks = 0;
            int canon_part[16];
            int canon_mono[16];
            int new_idx = 0;
            for (int j = 0; j < n_active; j++) {
                if (j == fpos) continue;
                int old_bid = part[j];
                int cid = block_id_map[old_bid];
                if (cid < 0) {
                    cid = new_n_blocks++;
                    block_id_map[old_bid] = cid;
                    block_first_mon[cid] = mono[j];
                    new_bal[cid] = bal[old_bid];
                }
                canon_part[new_idx] = cid;
                if (new_bal[cid]) {
                    canon_mono[new_idx] = (mono[j] + block_first_mon[cid]) & 1;
                } else {
                    canon_mono[new_idx] = 0;
                }
                new_idx++;
            }

            /* Build new state into scratch. */
            int idx = 0;
            scratch[idx++] = new_n_active;
            scratch[idx++] = new_n_blocks;
            scratch[idx++] = new_finalized;
            for (int j = 0; j < new_n_active; j++) scratch[idx++] = canon_part[j];
            for (int j = 0; j < new_n_active; j++) scratch[idx++] = canon_mono[j];
            for (int b = 0; b < new_n_blocks; b++)  scratch[idx++] = new_bal[b];

            if (out_data_pos + idx > out_capacity_data) {
                free(table); return -1;
            }
            int* tgt = out_states_data + out_data_pos;
            memcpy(tgt, scratch, idx * sizeof(int));
            hash_insert(table, hash_cap, tgt, idx, w, p);
            out_data_pos += idx;
        }

        /* Extract via temp aux buffer (same overlap reason as edge step). */
        int* aux = (int*)malloc(out_data_pos * sizeof(int));
        if (!aux && out_data_pos > 0) { free(table); return -5; }
        int n_out = 0;
        int compact_data_pos = 0;
        for (int i = 0; i < hash_cap; i++) {
            if (table[i].in_use) {
                if (n_out >= out_capacity_weights) {
                    free(aux); free(table); return -2;
                }
                if (compact_data_pos + table[i].key_len > out_capacity_data) {
                    free(aux); free(table); return -1;
                }
                out_states_offsets[n_out] = compact_data_pos;
                memcpy(aux + compact_data_pos, table[i].key_data,
                       table[i].key_len * sizeof(int));
                compact_data_pos += table[i].key_len;
                out_weights[n_out] = table[i].weight;
                n_out++;
            }
        }
        out_states_offsets[n_out] = compact_data_pos;
        if (compact_data_pos > 0) {
            memcpy(out_states_data, aux, compact_data_pos * sizeof(int));
        }
        free(aux);
        *out_n = n_out;
        free(table);
        return 0;
    }

    /* =========================================================================
       FULL SIGNED-DP LOOP IN C — no per-step Python marshalling
       =========================================================================
       Maintains state hash table internally across all elimination steps.
       Each state stored as: offset into state_buf + length + weight.
       ========================================================================= */

    /* Dynamic hash table for state aggregation. */
    typedef struct {
        int offset;    /* into state_buf */
        int len;       /* length in ints */
        long long weight; /* mod p */
        uint64_t hash;
        int in_use;
    } DynEntry;

    typedef struct {
        DynEntry* entries;
        int capacity;
        int n_used;
        int* state_buf;
        int state_buf_capacity;
        int state_buf_used;
    } DynHashMap;

    static int dynmap_init(DynHashMap* m, int initial_cap, int state_buf_cap) {
        m->capacity = initial_cap;
        m->entries = (DynEntry*)calloc(initial_cap, sizeof(DynEntry));
        m->n_used = 0;
        m->state_buf_capacity = state_buf_cap;
        m->state_buf = (int*)malloc(state_buf_cap * sizeof(int));
        m->state_buf_used = 0;
        return (m->entries && m->state_buf) ? 0 : -1;
    }

    static void dynmap_free(DynHashMap* m) {
        free(m->entries);
        free(m->state_buf);
        m->entries = NULL;
        m->state_buf = NULL;
    }

    static int dynmap_grow_state_buf(DynHashMap* m, int needed) {
        if (m->state_buf_used + needed <= m->state_buf_capacity) return 0;
        int new_cap = m->state_buf_capacity;
        while (new_cap < m->state_buf_used + needed) new_cap *= 2;
        int* new_buf = (int*)realloc(m->state_buf, new_cap * sizeof(int));
        if (!new_buf) return -1;
        m->state_buf = new_buf;
        m->state_buf_capacity = new_cap;
        return 0;
    }

    static int dynmap_grow_table(DynHashMap* m) {
        int new_cap = m->capacity * 2;
        DynEntry* new_entries = (DynEntry*)calloc(new_cap, sizeof(DynEntry));
        if (!new_entries) return -1;
        for (int i = 0; i < m->capacity; i++) {
            if (m->entries[i].in_use) {
                uint64_t h = m->entries[i].hash;
                int probe = (int)(h % (uint64_t)new_cap);
                for (int j = 0; j < new_cap; j++) {
                    int idx = (probe + j) % new_cap;
                    if (!new_entries[idx].in_use) {
                        new_entries[idx] = m->entries[i];
                        break;
                    }
                }
            }
        }
        free(m->entries);
        m->entries = new_entries;
        m->capacity = new_cap;
        return 0;
    }

    static void dynmap_clear(DynHashMap* m) {
        memset(m->entries, 0, m->capacity * sizeof(DynEntry));
        m->n_used = 0;
        m->state_buf_used = 0;
    }

    /* Insert/accumulate. data points to a state in the caller's scratch buf. */
    static int dynmap_insert(
        DynHashMap* m, const int* data, int len, long long weight, long long p)
    {
        if (m->n_used * 2 >= m->capacity) {
            if (dynmap_grow_table(m) < 0) return -1;
        }
        uint64_t h = hash_state(data, len);
        int probe = (int)(h % (uint64_t)m->capacity);
        for (int j = 0; j < m->capacity; j++) {
            int idx = (probe + j) % m->capacity;
            DynEntry* e = &m->entries[idx];
            if (!e->in_use) {
                if (dynmap_grow_state_buf(m, len) < 0) return -1;
                int off = m->state_buf_used;
                memcpy(m->state_buf + off, data, len * sizeof(int));
                m->state_buf_used += len;
                e->offset = off;
                e->len = len;
                e->weight = weight % p;
                e->hash = h;
                e->in_use = 1;
                m->n_used++;
                return 1;
            }
            if (e->hash == h && e->len == len &&
                memcmp(m->state_buf + e->offset, data, len * sizeof(int)) == 0) {
                e->weight = (e->weight + weight) % p;
                return 0;
            }
        }
        return -1;  /* table full (shouldn't happen with grow_table) */
    }

    /* EdgeStep struct + comparator. */
    typedef struct {
        int step;
        int u;
        int v;
        int sign;
    } EdgeStep;

    static int compare_edge_step(const void* a, const void* b) {
        const EdgeStep* ea = (const EdgeStep*)a;
        const EdgeStep* eb = (const EdgeStep*)b;
        return ea->step - eb->step;
    }

    /* Modular pow. */
    static long long mod_pow(long long base, int exp, long long p) {
        if (exp < 0) return 0;  /* shouldn't happen in our use */
        long long result = 1 % p;
        base = ((base % p) + p) % p;
        while (exp > 0) {
            if (exp & 1) result = mod_mul(result, base, p);
            base = mod_mul(base, base, p);
            exp >>= 1;
        }
        return result;
    }

    /* Process one edge transition: read from src into dst hash map.
       u_pos, v_pos in current active partition. */
    static int do_edge_step(
        DynHashMap* src, DynHashMap* dst,
        int u_pos, int v_pos, int sign,
        long long y_minus_1, long long p,
        int scratch[64])
    {
        dynmap_clear(dst);
        for (int i = 0; i < src->capacity; i++) {
            if (!src->entries[i].in_use) continue;
            const int* state = src->state_buf + src->entries[i].offset;
            long long w = src->entries[i].weight;
            int n_active, n_blocks, finalized;
            if (decode_header(state, &n_active, &n_blocks, &finalized) < 0) return -3;
            const int* part = state_part(state);
            const int* mono = state_mono(state);
            const int* bal  = state_bal(state);
            int bu = part[u_pos];
            int bv = part[v_pos];
            int mu = mono[u_pos];
            int mv = mono[v_pos];

            /* Delete branch */
            int del_len = copy_state(state, scratch);
            if (dynmap_insert(dst, scratch, del_len, w, p) < 0) return -1;

            /* Keep branch */
            if (bu != bv) {
                int offset = (mu + sign + mv) & 1;
                int bal_bu = bal[bu];
                int bal_bv = bal[bv];
                int both_unbal = (!bal_bu) && (!bal_bv);
                int combined = bal_bu && bal_bv;
                long long factor = both_unbal ? y_minus_1 : 1LL;
                int merge_len = build_merged_state(state, bu, bv, offset, combined, scratch);
                long long w2 = mod_mul(w, factor, p);
                if (dynmap_insert(dst, scratch, merge_len, w2, p) < 0) return -1;
            } else {
                int cycle_sign = (mu + sign + mv) & 1;
                if (bal[bu]) {
                    if (cycle_sign == 0) {
                        int copy_len = copy_state(state, scratch);
                        long long w2 = mod_mul(w, y_minus_1, p);
                        if (dynmap_insert(dst, scratch, copy_len, w2, p) < 0) return -1;
                    } else {
                        int unb_len = build_unbalanced_state(state, bu, scratch);
                        if (dynmap_insert(dst, scratch, unb_len, w, p) < 0) return -1;
                    }
                } else {
                    int copy_len = copy_state(state, scratch);
                    long long w2 = mod_mul(w, y_minus_1, p);
                    if (dynmap_insert(dst, scratch, copy_len, w2, p) < 0) return -1;
                }
            }
        }
        return 0;
    }

    /* Process one forget step. */
    static int do_forget_step(
        DynHashMap* src, DynHashMap* dst,
        int fpos, long long p,
        int scratch[64])
    {
        dynmap_clear(dst);
        for (int i = 0; i < src->capacity; i++) {
            if (!src->entries[i].in_use) continue;
            const int* state = src->state_buf + src->entries[i].offset;
            long long w = src->entries[i].weight;
            int n_active, n_blocks, finalized;
            if (decode_header(state, &n_active, &n_blocks, &finalized) < 0) return -3;
            if (n_active < 1) return -3;
            const int* part = state_part(state);
            const int* mono = state_mono(state);
            const int* bal  = state_bal(state);

            int fpos_bid = part[fpos];
            int fpos_size = 0;
            for (int j = 0; j < n_active; j++) {
                if (part[j] == fpos_bid) fpos_size++;
            }
            int new_finalized = finalized;
            if (fpos_size == 1 && bal[fpos_bid]) new_finalized++;
            int new_n_active = n_active - 1;

            int block_id_map[16];
            for (int b = 0; b < 16; b++) block_id_map[b] = -1;
            int block_first_mon[16];
            int new_bal[16];
            int new_n_blocks = 0;
            int canon_part[16];
            int canon_mono[16];
            int new_idx = 0;
            for (int j = 0; j < n_active; j++) {
                if (j == fpos) continue;
                int old_bid = part[j];
                int cid = block_id_map[old_bid];
                if (cid < 0) {
                    cid = new_n_blocks++;
                    block_id_map[old_bid] = cid;
                    block_first_mon[cid] = mono[j];
                    new_bal[cid] = bal[old_bid];
                }
                canon_part[new_idx] = cid;
                if (new_bal[cid]) {
                    canon_mono[new_idx] = (mono[j] + block_first_mon[cid]) & 1;
                } else {
                    canon_mono[new_idx] = 0;
                }
                new_idx++;
            }

            int idx = 0;
            scratch[idx++] = new_n_active;
            scratch[idx++] = new_n_blocks;
            scratch[idx++] = new_finalized;
            for (int j = 0; j < new_n_active; j++) scratch[idx++] = canon_part[j];
            for (int j = 0; j < new_n_active; j++) scratch[idx++] = canon_mono[j];
            for (int b = 0; b < new_n_blocks; b++)  scratch[idx++] = new_bal[b];

            if (dynmap_insert(dst, scratch, idx, w, p) < 0) return -1;
        }
        return 0;
    }

    /* =========================================================================
       σ-equivariant per-orbit DP — unsigned Tutte state helpers
       =========================================================================
       State layout: [n_active, n_blocks, finalized, part[0..n_active-1]]
       Length: 3 + n_active. NO monodromy or balance fields.
       ========================================================================= */

    static int unsigned_state_length(int n_active) {
        return 3 + n_active;
    }

    /* Decode header for unsigned state. Returns -1 if malformed. */
    static int decode_unsigned_header(const int* state, int* n_active, int* n_blocks, int* finalized) {
        *n_active = state[0];
        *n_blocks = state[1];
        *finalized = state[2];
        if (*n_active < 0 || *n_active > 32) return -1;
        if (*n_blocks < 0 || *n_blocks > *n_active) return -1;
        return 0;
    }

    /* Copy unsigned state to out_buf. */
    static int copy_unsigned_state(const int* state, int* out_buf) {
        int n_active = state[0];
        int len = unsigned_state_length(n_active);
        memcpy(out_buf, state, len * sizeof(int));
        return len;
    }

    /* Canonicalize partition by first-appearance block relabeling. */
    static void canonicalize_partition_arr(const int* part_in, int n, int* part_out, int* n_blocks_out) {
        int block_map[32];
        for (int i = 0; i < 32; i++) block_map[i] = -1;
        int next_id = 0;
        for (int i = 0; i < n; i++) {
            int old = part_in[i];
            int cid = block_map[old];
            if (cid < 0) {
                cid = next_id++;
                block_map[old] = cid;
            }
            part_out[i] = cid;
        }
        *n_blocks_out = next_id;
    }

    /* Build merged-partition state: merge block bv into bu. */
    static int build_merged_unsigned(
        const int* state, int bu, int bv, int* out_buf)
    {
        int n_active = state[0];
        int finalized = state[2];
        const int* part_in = state + 3;
        int tmp[32];
        for (int i = 0; i < n_active; i++) {
            tmp[i] = (part_in[i] == bv) ? bu : part_in[i];
        }
        int canon_part[32];
        int new_n_blocks;
        canonicalize_partition_arr(tmp, n_active, canon_part, &new_n_blocks);
        int idx = 0;
        out_buf[idx++] = n_active;
        out_buf[idx++] = new_n_blocks;
        out_buf[idx++] = finalized;
        for (int i = 0; i < n_active; i++) out_buf[idx++] = canon_part[i];
        return idx;
    }

    /* σ-canonicalize an unsigned state in-place: replace part with lex-min of
       (part, σ-image of part). pos_perm[i] = position σ sends i to.
       Returns: 1 if state replaced (σ-image was smaller), 0 otherwise. */
    static int sigma_canonicalize_unsigned(int* state, const int* pos_perm) {
        int n_active = state[0];
        int* part = state + 3;
        /* Compute σ-image: sigma_seq[pos_perm[i]] = part[i]. */
        int sigma_seq[32];
        for (int i = 0; i < n_active; i++) {
            sigma_seq[pos_perm[i]] = part[i];
        }
        /* Canonicalize σ-image (relabel by first appearance). */
        int sigma_canon[32];
        int n_blocks_sigma;
        canonicalize_partition_arr(sigma_seq, n_active, sigma_canon, &n_blocks_sigma);
        /* Compare lex: take min(part, sigma_canon). */
        int smaller = 0;
        for (int i = 0; i < n_active; i++) {
            if (sigma_canon[i] < part[i]) { smaller = 1; break; }
            if (sigma_canon[i] > part[i]) { smaller = 0; break; }
        }
        if (smaller) {
            for (int i = 0; i < n_active; i++) part[i] = sigma_canon[i];
            state[1] = n_blocks_sigma;
            return 1;
        }
        return 0;
    }

    /* Process one σ-orbit edge step: 4 branches (del/del, del/keep, keep/del, keep/keep).
       After each branch, σ-canonicalize and insert into dst hash map.
       u1_pos, v1_pos, u2_pos, v2_pos are positions of (u1, v1, u2, v2). */
    static int do_sigma_orbit_step(
        DynHashMap* src, DynHashMap* dst,
        int u1_pos, int v1_pos, int u2_pos, int v2_pos,
        const int* pos_perm,
        long long y_minus_1, long long p,
        int scratch[64], int scratch2[64])
    {
        dynmap_clear(dst);
        for (int i = 0; i < src->capacity; i++) {
            if (!src->entries[i].in_use) continue;
            const int* state = src->state_buf + src->entries[i].offset;
            long long w = src->entries[i].weight;
            int n_active, n_blocks_in, fin;
            if (decode_unsigned_header(state, &n_active, &n_blocks_in, &fin) < 0) return -3;

            /* For each (keep1, keep2) pair, build the resulting state. */
            for (int keep1 = 0; keep1 < 2; keep1++) {
                for (int keep2 = 0; keep2 < 2; keep2++) {
                    /* Apply edge 1: copy or merge. */
                    long long w1 = w;
                    if (!keep1) {
                        (void)copy_unsigned_state(state, scratch);
                    } else {
                        const int* part = state + 3;
                        int bu = part[u1_pos];
                        int bv = part[v1_pos];
                        if (bu == bv) {
                            /* Cycle: factor (y-1), state unchanged. */
                            (void)copy_unsigned_state(state, scratch);
                            w1 = mod_mul(w1, y_minus_1, p);
                        } else {
                            (void)build_merged_unsigned(state, bu, bv, scratch);
                        }
                    }
                    /* Apply edge 2 to scratch result. */
                    int len2;
                    long long w2 = w1;
                    if (!keep2) {
                        len2 = copy_unsigned_state(scratch, scratch2);
                    } else {
                        const int* part = scratch + 3;
                        int bu = part[u2_pos];
                        int bv = part[v2_pos];
                        if (bu == bv) {
                            len2 = copy_unsigned_state(scratch, scratch2);
                            w2 = mod_mul(w2, y_minus_1, p);
                        } else {
                            len2 = build_merged_unsigned(scratch, bu, bv, scratch2);
                        }
                    }
                    /* σ-canonicalize scratch2 in place. */
                    sigma_canonicalize_unsigned(scratch2, pos_perm);
                    if (dynmap_insert(dst, scratch2, len2, w2, p) < 0) return -1;
                }
            }
        }
        return 0;
    }

    /* Forget σ-pair (fpos1, fpos2): drop both positions, update finalized.
       Both positions are simultaneously forgotten so active set stays σ-invariant.
       pos_perm_after is the σ permutation on positions AFTER forget. */
    static int do_sigma_forget_pair(
        DynHashMap* src, DynHashMap* dst,
        int fpos1, int fpos2,
        const int* pos_perm_after,
        long long p,
        int scratch[64])
    {
        if (fpos1 > fpos2) { int t = fpos1; fpos1 = fpos2; fpos2 = t; }
        dynmap_clear(dst);
        for (int i = 0; i < src->capacity; i++) {
            if (!src->entries[i].in_use) continue;
            const int* state = src->state_buf + src->entries[i].offset;
            long long w = src->entries[i].weight;
            int n_active, n_blocks, fin;
            if (decode_unsigned_header(state, &n_active, &n_blocks, &fin) < 0) return -3;
            const int* part = state + 3;
            int b1 = part[fpos1];
            int b2 = part[fpos2];
            int size1 = 0, size2 = 0;
            for (int j = 0; j < n_active; j++) {
                if (part[j] == b1) size1++;
                if (part[j] == b2) size2++;
            }
            int inc = 0;
            if (size1 == 1) inc++;
            if (b1 != b2 && size2 == 1) inc++;
            if (b1 == b2 && size1 == 2) inc++;
            int new_fin = fin + inc;
            int new_n_active = n_active - 2;

            /* Drop fpos1 and fpos2; canonicalize remaining. */
            int dropped[32];
            int idx = 0;
            for (int j = 0; j < n_active; j++) {
                if (j == fpos1 || j == fpos2) continue;
                dropped[idx++] = part[j];
            }
            int canon[32];
            int new_n_blocks;
            canonicalize_partition_arr(dropped, new_n_active, canon, &new_n_blocks);

            scratch[0] = new_n_active;
            scratch[1] = new_n_blocks;
            scratch[2] = new_fin;
            for (int j = 0; j < new_n_active; j++) scratch[3 + j] = canon[j];
            int slen = 3 + new_n_active;
            /* σ-canonicalize after forget. */
            sigma_canonicalize_unsigned(scratch, pos_perm_after);
            if (dynmap_insert(dst, scratch, slen, w, p) < 0) return -1;
        }
        return 0;
    }

    int sigma_orbit_dp_full_c(
        int n_verts, int n_orbits,
        const int* edges_pairs,
        const int* pair_order,
        const int* perm,
        int r_E,
        long long x_minus_1, long long y_minus_1, long long p,
        long long* out_total, int* out_max_states)
    {
        if (n_verts > 32) return -4;
        if (n_verts & 1) return -4;  /* must be even for free σ */
        int n_pairs = n_verts / 2;

        /* elim_pos[v] = the pair-step index of v. */
        int elim_pos[64];
        for (int k = 0; k < n_pairs; k++) {
            int v = pair_order[2 * k];
            int sv = pair_order[2 * k + 1];
            elim_pos[v] = k;
            elim_pos[sv] = k;
        }

        /* Per orbit, compute step = max(elim_pos of all 4 endpoints). Sort by step. */
        typedef struct {
            int step;
            int u1, v1, u2, v2;
        } OrbitStep;
        OrbitStep* orbits = (OrbitStep*)malloc(n_orbits * sizeof(OrbitStep));
        if (!orbits && n_orbits > 0) return -5;
        for (int e = 0; e < n_orbits; e++) {
            int u1 = edges_pairs[4*e + 0];
            int v1 = edges_pairs[4*e + 1];
            int u2 = edges_pairs[4*e + 2];
            int v2 = edges_pairs[4*e + 3];
            int s = elim_pos[u1];
            if (elim_pos[v1] > s) s = elim_pos[v1];
            if (elim_pos[u2] > s) s = elim_pos[u2];
            if (elim_pos[v2] > s) s = elim_pos[v2];
            orbits[e].step = s;
            orbits[e].u1 = u1; orbits[e].v1 = v1;
            orbits[e].u2 = u2; orbits[e].v2 = v2;
        }
        /* Insertion sort by step. */
        for (int i = 1; i < n_orbits; i++) {
            OrbitStep tmp = orbits[i];
            int j = i - 1;
            while (j >= 0 && orbits[j].step > tmp.step) {
                orbits[j + 1] = orbits[j]; j--;
            }
            orbits[j + 1] = tmp;
        }

        /* active_pos[v] = current position; -1 if forgotten.
           Initially: position 2k for pair_order[2k], 2k+1 for pair_order[2k+1]. */
        int active_pos[64];
        for (int i = 0; i < n_verts; i++) active_pos[i] = -1;
        for (int k = 0; k < n_pairs; k++) {
            active_pos[pair_order[2*k]]     = 2*k;
            active_pos[pair_order[2*k + 1]] = 2*k + 1;
        }
        int cur_n_active = n_verts;

        /* Initialize states: single state (init_part, fin=0) with weight 1. */
        DynHashMap m1, m2;
        if (dynmap_init(&m1, 1024, 8192) < 0) { free(orbits); return -5; }
        if (dynmap_init(&m2, 1024, 8192) < 0) { dynmap_free(&m1); free(orbits); return -5; }
        DynHashMap* src = &m1;
        DynHashMap* dst = &m2;

        int init_state[64];
        init_state[0] = n_verts;       /* n_active */
        init_state[1] = n_verts;       /* n_blocks (all singletons) */
        init_state[2] = 0;             /* finalized */
        for (int i = 0; i < n_verts; i++) init_state[3 + i] = i;
        dynmap_insert(src, init_state, 3 + n_verts, 1, p);

        int max_states = 1;
        int scratch[64], scratch2[64];
        int pos_perm[64];

        int orbit_idx = 0;
        for (int step = 0; step < n_pairs; step++) {
            /* Compute pos_perm for current active set. */
            int inv[64];
            for (int i = 0; i < cur_n_active; i++) inv[i] = -1;
            for (int v = 0; v < n_verts; v++) {
                if (active_pos[v] >= 0) inv[active_pos[v]] = v;
            }
            for (int i = 0; i < cur_n_active; i++) {
                int v = inv[i];
                int sv = perm[v];
                pos_perm[i] = active_pos[sv];
            }

            /* Process all orbits at this step. */
            while (orbit_idx < n_orbits && orbits[orbit_idx].step == step) {
                int u1 = orbits[orbit_idx].u1;
                int v1 = orbits[orbit_idx].v1;
                int u2 = orbits[orbit_idx].u2;
                int v2 = orbits[orbit_idx].v2;
                orbit_idx++;
                int u1p = active_pos[u1], v1p = active_pos[v1];
                int u2p = active_pos[u2], v2p = active_pos[v2];
                if (u1p < 0 || v1p < 0 || u2p < 0 || v2p < 0) {
                    dynmap_free(&m1); dynmap_free(&m2); free(orbits); return -3;
                }
                int rc = do_sigma_orbit_step(
                    src, dst, u1p, v1p, u2p, v2p, pos_perm,
                    y_minus_1, p, scratch, scratch2
                );
                if (rc < 0) { dynmap_free(&m1); dynmap_free(&m2); free(orbits); return rc; }
                DynHashMap* t = src; src = dst; dst = t;
                if (src->n_used > max_states) max_states = src->n_used;
            }

            /* Forget σ-pair if not referenced by any future orbit. */
            int v_pair = pair_order[2*step];
            int sv_pair = pair_order[2*step + 1];
            int still_needed = 0;
            for (int e = orbit_idx; e < n_orbits; e++) {
                if (orbits[e].u1 == v_pair || orbits[e].v1 == v_pair ||
                    orbits[e].u2 == v_pair || orbits[e].v2 == v_pair ||
                    orbits[e].u1 == sv_pair || orbits[e].v1 == sv_pair ||
                    orbits[e].u2 == sv_pair || orbits[e].v2 == sv_pair) {
                    still_needed = 1;
                    break;
                }
            }
            if (!still_needed) {
                int fpos1 = active_pos[v_pair];
                int fpos2 = active_pos[sv_pair];
                if (fpos1 < 0 || fpos2 < 0) {
                    dynmap_free(&m1); dynmap_free(&m2); free(orbits); return -3;
                }
                /* Update active_pos: shift down by 1 or 2 depending on relation. */
                active_pos[v_pair] = -1;
                active_pos[sv_pair] = -1;
                int fmin = fpos1 < fpos2 ? fpos1 : fpos2;
                int fmax = fpos1 > fpos2 ? fpos1 : fpos2;
                for (int v = 0; v < n_verts; v++) {
                    int pp = active_pos[v];
                    if (pp < 0) continue;
                    if (pp > fmax) pp -= 2;
                    else if (pp > fmin) pp -= 1;
                    active_pos[v] = pp;
                }
                cur_n_active -= 2;
                /* Recompute pos_perm AFTER forget. */
                int inv2[64];
                for (int i = 0; i < cur_n_active; i++) inv2[i] = -1;
                for (int v = 0; v < n_verts; v++) {
                    if (active_pos[v] >= 0) inv2[active_pos[v]] = v;
                }
                int pos_perm_after[64];
                for (int i = 0; i < cur_n_active; i++) {
                    int vv = inv2[i];
                    int svv = perm[vv];
                    pos_perm_after[i] = active_pos[svv];
                }
                int rc = do_sigma_forget_pair(src, dst, fpos1, fpos2, pos_perm_after, p, scratch);
                if (rc < 0) { dynmap_free(&m1); dynmap_free(&m2); free(orbits); return rc; }
                DynHashMap* t = src; src = dst; dst = t;
                if (src->n_used > max_states) max_states = src->n_used;
            }
        }

        /* Final aggregation: T = Σ w × (x-1)^{r_E - r_state}. */
        long long total = 0;
        for (int i = 0; i < src->capacity; i++) {
            if (!src->entries[i].in_use) continue;
            const int* state = src->state_buf + src->entries[i].offset;
            long long w = src->entries[i].weight;
            int n_active = state[0];
            int n_blocks = state[1];
            int fin = state[2];
            int active_blocks = (n_active > 0) ? n_blocks : 0;
            int r_state = n_verts - fin - active_blocks;
            int exp = r_E - r_state;
            if (exp < 0) exp = 0;
            long long contrib = mod_pow(x_minus_1, exp, p);
            contrib = mod_mul(contrib, w, p);
            total = (total + contrib) % p;
        }
        *out_total = total;
        *out_max_states = max_states;
        dynmap_free(&m1);
        dynmap_free(&m2);
        free(orbits);
        return 0;
    }

    /* Process one T_fix^σ edge transition. Same structure as do_edge_step but
       with factor table parameter and is_loop per-edge handling. */
    typedef struct {
        long long del_factor;                       /* 1 (used for delete) */
        long long tree_xbal_xbal;                   /* (x-1)^{-2} */
        long long tree_unbal_unbal;                 /* (x-1)^{-1}(y-1) */
        long long cycle_no_rank_change_nonloop;     /* (y-1)^2 */
        long long unbal_cycle_in_bal_nonloop;       /* (x-1)^{-1}(y-1) */
        long long loop_bal_cycle_in_bal;            /* (y-1) */
        long long loop_unbal_cycle_in_bal;          /* (x-1)^{-1} */
        long long loop_cycle_in_unbal;              /* (y-1) */
    } TFixFactors;

    static int do_edge_step_tfix(
        DynHashMap* src, DynHashMap* dst,
        int u_pos, int v_pos, int sign, int is_loop,
        const TFixFactors* f, long long p,
        int scratch[64])
    {
        dynmap_clear(dst);
        for (int i = 0; i < src->capacity; i++) {
            if (!src->entries[i].in_use) continue;
            const int* state = src->state_buf + src->entries[i].offset;
            long long w = src->entries[i].weight;
            int n_active, n_blocks, finalized;
            if (decode_header(state, &n_active, &n_blocks, &finalized) < 0) return -3;
            const int* part = state_part(state);
            const int* mono = state_mono(state);
            const int* bal  = state_bal(state);
            int bu = part[u_pos];
            int bv = part[v_pos];
            int mu = mono[u_pos];
            int mv = mono[v_pos];

            /* Delete branch: factor 1. */
            int del_len = copy_state(state, scratch);
            if (dynmap_insert(dst, scratch, del_len, w, p) < 0) return -1;

            /* Keep branch */
            if (bu != bv) {
                int offset = (mu + sign + mv) & 1;
                int bal_bu = bal[bu];
                int bal_bv = bal[bv];
                int both_unbal = (!bal_bu) && (!bal_bv);
                int combined = bal_bu && bal_bv;
                long long factor = both_unbal ? f->tree_unbal_unbal : f->tree_xbal_xbal;
                int merge_len = build_merged_state(state, bu, bv, offset, combined, scratch);
                long long w2 = mod_mul(w, factor, p);
                if (dynmap_insert(dst, scratch, merge_len, w2, p) < 0) return -1;
            } else {
                int cycle_sign = (mu + sign + mv) & 1;
                if (bal[bu]) {
                    if (cycle_sign == 0) {
                        /* bal cycle in bal: no balance change */
                        long long factor = is_loop ? f->loop_bal_cycle_in_bal
                                                   : f->cycle_no_rank_change_nonloop;
                        int copy_len = copy_state(state, scratch);
                        long long w2 = mod_mul(w, factor, p);
                        if (dynmap_insert(dst, scratch, copy_len, w2, p) < 0) return -1;
                    } else {
                        /* unbal cycle in bal: block becomes unbal */
                        long long factor = is_loop ? f->loop_unbal_cycle_in_bal
                                                   : f->unbal_cycle_in_bal_nonloop;
                        int unb_len = build_unbalanced_state(state, bu, scratch);
                        long long w2 = mod_mul(w, factor, p);
                        if (dynmap_insert(dst, scratch, unb_len, w2, p) < 0) return -1;
                    }
                } else {
                    /* cycle in unbal: no rank/balance change */
                    long long factor = is_loop ? f->loop_cycle_in_unbal
                                               : f->cycle_no_rank_change_nonloop;
                    int copy_len = copy_state(state, scratch);
                    long long w2 = mod_mul(w, factor, p);
                    if (dynmap_insert(dst, scratch, copy_len, w2, p) < 0) return -1;
                }
            }
        }
        return 0;
    }

    int t_fix_dp_full_c(
        int n_verts, int n_edges,
        const int* edges_uv, const int* signs, const int* is_loop,
        const int* elim_order,
        int r_E_G,
        long long x_minus_1, long long p,
        long long factor_tree_xbal_xbal, long long factor_tree_unbal_unbal,
        long long factor_cycle_no_rank_change, long long factor_unbal_cycle_in_bal,
        long long factor_loop_bal_cycle_in_bal, long long factor_loop_unbal_cycle_in_bal,
        long long factor_loop_cycle_in_unbal,
        long long* out_total, int* out_max_states)
    {
        if (n_verts > 32) return -4;
        TFixFactors f = {
            1LL,
            factor_tree_xbal_xbal, factor_tree_unbal_unbal,
            factor_cycle_no_rank_change, factor_unbal_cycle_in_bal,
            factor_loop_bal_cycle_in_bal, factor_loop_unbal_cycle_in_bal,
            factor_loop_cycle_in_unbal,
        };

        /* Compute elim_pos. */
        int elim_pos[64];
        for (int i = 0; i < n_verts; i++) elim_pos[elim_order[i]] = i;

        /* Build edge steps with is_loop info. */
        typedef struct { int step, u, v, sign, is_loop; } EdgeStepL;
        EdgeStepL* edge_steps = (EdgeStepL*)malloc(n_edges * sizeof(EdgeStepL));
        if (!edge_steps && n_edges > 0) return -5;
        for (int e = 0; e < n_edges; e++) {
            int u = edges_uv[2*e];
            int v = edges_uv[2*e+1];
            int s = (elim_pos[u] > elim_pos[v]) ? elim_pos[u] : elim_pos[v];
            edge_steps[e].step = s;
            edge_steps[e].u = u;
            edge_steps[e].v = v;
            edge_steps[e].sign = signs[e];
            edge_steps[e].is_loop = is_loop[e];
        }
        /* Stable sort by step. Simple insertion sort (small N). */
        for (int i = 1; i < n_edges; i++) {
            EdgeStepL tmp = edge_steps[i];
            int j = i - 1;
            while (j >= 0 && edge_steps[j].step > tmp.step) {
                edge_steps[j + 1] = edge_steps[j];
                j--;
            }
            edge_steps[j + 1] = tmp;
        }

        int active_pos[64];
        for (int i = 0; i < n_verts; i++) active_pos[i] = i;

        DynHashMap m1, m2;
        if (dynmap_init(&m1, 1024, 8192) < 0) { free(edge_steps); return -5; }
        if (dynmap_init(&m2, 1024, 8192) < 0) {
            dynmap_free(&m1); free(edge_steps); return -5;
        }
        DynHashMap* src = &m1;
        DynHashMap* dst = &m2;

        int init_state[64];
        int idx = 0;
        init_state[idx++] = n_verts;
        init_state[idx++] = n_verts;
        init_state[idx++] = 0;
        for (int i = 0; i < n_verts; i++) init_state[idx++] = i;
        for (int i = 0; i < n_verts; i++) init_state[idx++] = 0;
        for (int i = 0; i < n_verts; i++) init_state[idx++] = 1;
        dynmap_insert(src, init_state, idx, 1, p);

        int max_states = 1;
        int scratch[64];

        int edge_idx = 0;
        for (int step = 0; step < n_verts; step++) {
            while (edge_idx < n_edges && edge_steps[edge_idx].step == step) {
                int u = edge_steps[edge_idx].u;
                int v = edge_steps[edge_idx].v;
                int sign = edge_steps[edge_idx].sign;
                int is_lp = edge_steps[edge_idx].is_loop;
                edge_idx++;
                int u_pos = active_pos[u];
                int v_pos = active_pos[v];
                if (u_pos < 0 || v_pos < 0) {
                    dynmap_free(&m1); dynmap_free(&m2); free(edge_steps); return -3;
                }
                int rc = do_edge_step_tfix(src, dst, u_pos, v_pos, sign, is_lp, &f, p, scratch);
                if (rc < 0) {
                    dynmap_free(&m1); dynmap_free(&m2); free(edge_steps); return rc;
                }
                DynHashMap* tmp = src; src = dst; dst = tmp;
                if (src->n_used > max_states) max_states = src->n_used;
            }

            int forget_v = elim_order[step];
            int still_needed = 0;
            for (int e = edge_idx; e < n_edges; e++) {
                if (edge_steps[e].u == forget_v || edge_steps[e].v == forget_v) {
                    still_needed = 1;
                    break;
                }
            }
            if (!still_needed) {
                int fpos = active_pos[forget_v];
                if (fpos >= 0) {
                    int rc = do_forget_step(src, dst, fpos, p, scratch);
                    if (rc < 0) {
                        dynmap_free(&m1); dynmap_free(&m2); free(edge_steps); return rc;
                    }
                    DynHashMap* tmp = src; src = dst; dst = tmp;
                    if (src->n_used > max_states) max_states = src->n_used;
                    for (int vv = 0; vv < n_verts; vv++) {
                        if (active_pos[vv] > fpos) active_pos[vv]--;
                    }
                    active_pos[forget_v] = -1;
                }
            }
        }

        /* T_fix^σ final aggregation: result = (x-1)^{r_E_G} × Σ weight. */
        long long total_weight = 0;
        for (int i = 0; i < src->capacity; i++) {
            if (!src->entries[i].in_use) continue;
            total_weight = (total_weight + src->entries[i].weight) % p;
        }
        long long x_pow = mod_pow(x_minus_1, r_E_G, p);
        *out_total = mod_mul(x_pow, total_weight, p);
        *out_max_states = max_states;
        dynmap_free(&m1);
        dynmap_free(&m2);
        free(edge_steps);
        return 0;
    }

    int signed_dp_full_c(
        int n_verts, int n_edges,
        const int* edges_uv, const int* signs,
        const int* elim_order,
        int r_E,
        long long x_minus_1, long long y_minus_1, long long p,
        long long* out_total, int* out_max_states)
    {
        if (n_verts > 32) return -4;  /* support up to 32 quotient verts */

        /* Compute elim_pos. */
        int elim_pos[64];
        for (int i = 0; i < n_verts; i++) {
            elim_pos[elim_order[i]] = i;
        }

        /* Build edge steps. */
        EdgeStep* edge_steps = (EdgeStep*)malloc(n_edges * sizeof(EdgeStep));
        if (!edge_steps && n_edges > 0) return -5;
        for (int e = 0; e < n_edges; e++) {
            int u = edges_uv[2*e];
            int v = edges_uv[2*e+1];
            int s = (elim_pos[u] > elim_pos[v]) ? elim_pos[u] : elim_pos[v];
            edge_steps[e].step = s;
            edge_steps[e].u = u;
            edge_steps[e].v = v;
            edge_steps[e].sign = signs[e];
        }
        qsort(edge_steps, n_edges, sizeof(EdgeStep), compare_edge_step);

        /* active_pos[v] = position in current partition; -1 if forgotten. */
        int active_pos[64];
        for (int i = 0; i < n_verts; i++) active_pos[i] = i;

        /* Initialize two hash maps. */
        DynHashMap m1, m2;
        if (dynmap_init(&m1, 1024, 8192) < 0) { free(edge_steps); return -5; }
        if (dynmap_init(&m2, 1024, 8192) < 0) {
            dynmap_free(&m1); free(edge_steps); return -5;
        }
        DynHashMap* src = &m1;
        DynHashMap* dst = &m2;

        /* Initial state: n_verts active, n_verts blocks (all singletons),
           all balanced, finalized=0. partition = [0..n-1]. mono all 0. */
        int init_state[64];
        int idx = 0;
        init_state[idx++] = n_verts;  /* n_active */
        init_state[idx++] = n_verts;  /* n_blocks */
        init_state[idx++] = 0;        /* finalized */
        for (int i = 0; i < n_verts; i++) init_state[idx++] = i;  /* part */
        for (int i = 0; i < n_verts; i++) init_state[idx++] = 0;  /* mono */
        for (int i = 0; i < n_verts; i++) init_state[idx++] = 1;  /* bal */
        dynmap_insert(src, init_state, idx, 1, p);

        int max_states = 1;
        int scratch[64];

        int edge_idx = 0;
        for (int step = 0; step < n_verts; step++) {
            /* Process edges at this step. */
            while (edge_idx < n_edges && edge_steps[edge_idx].step == step) {
                int u = edge_steps[edge_idx].u;
                int v = edge_steps[edge_idx].v;
                int sign = edge_steps[edge_idx].sign;
                edge_idx++;
                int u_pos = active_pos[u];
                int v_pos = active_pos[v];
                if (u_pos < 0 || v_pos < 0) {
                    /* Shouldn't happen with proper elim order */
                    dynmap_free(&m1); dynmap_free(&m2); free(edge_steps); return -3;
                }
                int rc = do_edge_step(src, dst, u_pos, v_pos, sign, y_minus_1, p, scratch);
                if (rc < 0) {
                    dynmap_free(&m1); dynmap_free(&m2); free(edge_steps); return rc;
                }
                /* Swap src/dst. */
                DynHashMap* tmp = src; src = dst; dst = tmp;
                if (src->n_used > max_states) max_states = src->n_used;
            }

            /* Forget? */
            int forget_v = elim_order[step];
            int still_needed = 0;
            for (int e = edge_idx; e < n_edges; e++) {
                if (edge_steps[e].u == forget_v || edge_steps[e].v == forget_v) {
                    still_needed = 1;
                    break;
                }
            }
            if (!still_needed) {
                int fpos = active_pos[forget_v];
                if (fpos >= 0) {
                    int rc = do_forget_step(src, dst, fpos, p, scratch);
                    if (rc < 0) {
                        dynmap_free(&m1); dynmap_free(&m2); free(edge_steps); return rc;
                    }
                    DynHashMap* tmp = src; src = dst; dst = tmp;
                    if (src->n_used > max_states) max_states = src->n_used;
                    /* Update active_pos: positions after fpos shift down. */
                    for (int vv = 0; vv < n_verts; vv++) {
                        if (active_pos[vv] > fpos) active_pos[vv]--;
                    }
                    active_pos[forget_v] = -1;
                }
            }
        }

        /* Final aggregation: T = Σ_state weight × (x-1)^{r_E - r_state}. */
        long long total = 0;
        for (int i = 0; i < src->capacity; i++) {
            if (!src->entries[i].in_use) continue;
            const int* state = src->state_buf + src->entries[i].offset;
            long long w = src->entries[i].weight;
            int n_active, n_blocks, finalized;
            if (decode_header(state, &n_active, &n_blocks, &finalized) < 0) {
                dynmap_free(&m1); dynmap_free(&m2); free(edge_steps); return -3;
            }
            const int* bal = state_bal(state);
            int active_bal = 0;
            for (int b = 0; b < n_blocks; b++) {
                if (bal[b]) active_bal++;
            }
            int n_balanced_state = finalized + active_bal;
            int r_state = n_verts - n_balanced_state;
            int exp = r_E - r_state;
            if (exp < 0) exp = 0;  /* safety */
            long long contrib = mod_pow(x_minus_1, exp, p);
            contrib = mod_mul(contrib, w, p);
            total = (total + contrib) % p;
        }

        *out_total = total;
        *out_max_states = max_states;
        dynmap_free(&m1);
        dynmap_free(&m2);
        free(edge_steps);
        return 0;
    }
""")

_LIB_LOCK = threading.Lock()
_LIB_CACHE = {"lib": None, "ffi": None}


def _get_lib():
    """Lazily compile + import the signed-elim cffi extension.

    Mirrors the pattern from `tutte/roots/_partition_c.py`. Returns
    (lib, ffi). Raises on compile/import failure (caller should catch
    and fall back to pure Python).
    """
    with _LIB_LOCK:
        if _LIB_CACHE["lib"] is not None:
            return _LIB_CACHE["lib"], _LIB_CACHE["ffi"]
        try:
            from _tutte_signed_elim_cffi import ffi as cffi_ffi
            from _tutte_signed_elim_cffi import lib
            _LIB_CACHE["lib"] = lib
            _LIB_CACHE["ffi"] = cffi_ffi
            return lib, cffi_ffi
        except ImportError:
            pass
        import sys
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="tutte_signed_elim_c_")
        ffi.compile(tmpdir=tmpdir)
        sys.path.insert(0, tmpdir)
        from _tutte_signed_elim_cffi import ffi as cffi_ffi
        from _tutte_signed_elim_cffi import lib
        _LIB_CACHE["lib"] = lib
        _LIB_CACHE["ffi"] = cffi_ffi
        return lib, cffi_ffi


# =============================================================================
# State encoding/decoding helpers (Python side)
# =============================================================================


def encode_state(
    partition: Tuple[int, ...],
    monodromy: Tuple[int, ...],
    balance: Tuple[int, ...],
    finalized: int,
) -> List[int]:
    """Encode a signed-DP state as flat int list (matches C-side encoding).

    State format:
      [n_active, n_blocks, finalized,
       part[0..n-1], mono[0..n-1], bal[0..n_blocks-1]]
    """
    n_active = len(partition)
    n_blocks = len(balance)
    out = [n_active, n_blocks, finalized]
    out.extend(partition)
    out.extend(monodromy)
    out.extend(balance)
    return out


def decode_state(
    data: List[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], int]:
    """Decode flat int array → (partition, monodromy, balance, finalized)."""
    n_active = data[0]
    n_blocks = data[1]
    finalized = data[2]
    part = tuple(data[3 : 3 + n_active])
    mono = tuple(data[3 + n_active : 3 + 2 * n_active])
    bal = tuple(bool(b) for b in data[3 + 2 * n_active : 3 + 2 * n_active + n_blocks])
    return part, mono, bal, finalized


# =============================================================================
# Batched step API — Python glue
# =============================================================================


def step_edge_batch(
    states: dict,  # {(part, mono, bal, fin): weight_mod_p}
    u_pos: int,
    v_pos: int,
    sign: int,
    is_loop: bool,
    y_minus_1: int,
    p: int,
) -> Optional[dict]:
    """Apply one edge step (delete + keep branches) to all states via C ext.

    Returns the new states dict, or None if C ext unavailable.

    State key type: tuple (partition: tuple[int], monodromy: tuple[int],
                           balance: tuple[bool], finalized: int).
    Weight type: int (mod p, < 2^62).
    """
    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    n_in = len(states)
    if n_in == 0:
        return {}

    # Encode all input states into one flat array.
    in_data: List[int] = []
    in_offsets: List[int] = []
    in_weights: List[int] = []
    state_keys = []  # parallel list of (part, mono, bal, fin) for fallback
    for st, w in states.items():
        part, mono, bal, fin = st
        in_offsets.append(len(in_data))
        in_data.extend(encode_state(part, mono, bal, fin))
        in_weights.append(int(w) % p)
        state_keys.append(st)
    in_offsets.append(len(in_data))

    # Worst-case output: each input contributes up to 3 branches × state size.
    max_state_size = 64  # 3 + 2*16 + 16 + slack
    out_cap_data = max(n_in * 3 * max_state_size, 1024)
    out_cap_weights = max(n_in * 3, 64)

    in_data_buf = _ffi.new("int[]", in_data)
    in_offsets_buf = _ffi.new("int[]", in_offsets)
    in_weights_buf = _ffi.new("long long[]", in_weights)
    out_data_buf = _ffi.new("int[]", out_cap_data)
    out_offsets_buf = _ffi.new("int[]", out_cap_weights + 1)
    out_weights_buf = _ffi.new("long long[]", out_cap_weights)
    out_n_buf = _ffi.new("int*")

    rc = lib.signed_step_edge_batch_c(
        in_data_buf, in_offsets_buf, in_weights_buf, n_in,
        u_pos, v_pos, sign, 1 if is_loop else 0,
        y_minus_1, p,
        out_data_buf, out_offsets_buf, out_weights_buf, out_n_buf,
        out_cap_data, out_cap_weights,
    )
    if rc != 0:
        return None

    n_out = out_n_buf[0]
    new_states = {}
    for i in range(n_out):
        s = out_offsets_buf[i]
        e = out_offsets_buf[i + 1]
        flat = [out_data_buf[j] for j in range(s, e)]
        key = decode_state(flat)
        new_states[key] = int(out_weights_buf[i])
    return new_states


def sigma_orbit_dp_full(
    n_verts: int,
    edges: List[Tuple[int, int]],
    perm: dict,
    r_E: int,
    x_val: int,
    y_val: int,
    p: int,
) -> Optional[Tuple[int, int]]:
    """σ-equivariant per-orbit DP on cover G — computes T(G) directly.

    Args:
      n_verts:  cover G vertex count (must be even for free σ).
      edges:    list of (u, v) edges, vertices 0..n_verts-1.
      perm:     dict v → σ(v) (free order-2 automorphism).
      r_E:      rank of full edge set in G.
      x_val, y_val, p: evaluation point.

    Returns (T(G; x, y) mod p, max_states) or None on failure.
    """
    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None
    if n_verts % 2 != 0:
        return None

    # Build pair_order via min-fill σ-pair heuristic.
    adj = {v: set() for v in range(n_verts)}
    for u, v in edges:
        if u != v:
            adj[u].add(v); adj[v].add(u)
    remaining = set(range(n_verts))
    pair_order: List[int] = []
    while remaining:
        best_pair = None
        best_cost = float('inf')
        seen = set()
        for v in remaining:
            sv = perm[v]
            if sv == v:
                continue
            key = (min(v, sv), max(v, sv))
            if key in seen:
                continue
            seen.add(key)
            nb_v = adj[v] & remaining
            nb_sv = adj[sv] & remaining
            cost = len(nb_v) + len(nb_sv)
            if cost < best_cost:
                best_cost = cost
                best_pair = key
        if best_pair is None:
            break
        v, sv = best_pair
        pair_order.append(v); pair_order.append(sv)
        # Fill-in edges.
        for w1 in adj[v] & remaining:
            for w2 in adj[v] & remaining:
                if w1 != w2: adj[w1].add(w2)
        for w1 in adj[sv] & remaining:
            for w2 in adj[sv] & remaining:
                if w1 != w2: adj[w1].add(w2)
        remaining.discard(v); remaining.discard(sv)

    # Compute σ-orbits of edges. Each orbit = {e, σe}.
    edges_pairs: List[int] = []
    seen_edges = set()
    n_orbits = 0
    for (u, v) in edges:
        ekey = (min(u, v), max(u, v))
        if ekey in seen_edges:
            continue
        seen_edges.add(ekey)
        sig_e = (perm[u], perm[v])
        sig_ekey = (min(sig_e), max(sig_e))
        seen_edges.add(sig_ekey)
        if sig_ekey == ekey:
            # σ-fixed edge — only valid for non-free covers (not supported here).
            return None
        edges_pairs.extend([u, v, sig_e[0], sig_e[1]])
        n_orbits += 1

    perm_arr = [perm[v] for v in range(n_verts)]

    x_minus_1 = (x_val - 1) % p
    y_minus_1 = (y_val - 1) % p

    edges_buf = _ffi.new("int[]", edges_pairs) if edges_pairs else _ffi.new("int[]", [0])
    pair_buf = _ffi.new("int[]", pair_order)
    perm_buf = _ffi.new("int[]", perm_arr)
    out_total = _ffi.new("long long*")
    out_max_st = _ffi.new("int*")

    rc = lib.sigma_orbit_dp_full_c(
        n_verts, n_orbits, edges_buf, pair_buf, perm_buf,
        r_E, x_minus_1, y_minus_1, p,
        out_total, out_max_st,
    )
    if rc != 0:
        return None
    return int(out_total[0]), int(out_max_st[0])


def t_fix_dp_full(
    n_verts: int,
    edges_uv: List[Tuple[int, int]],
    signs: List[int],
    is_loop: List[int],
    elim_order: List[int],
    r_E_G: int,
    x_val: int,
    y_val: int,
    p: int,
) -> Optional[Tuple[int, int]]:
    """Run the entire T_fix^σ DP in C — eliminates per-step Python encode/decode.

    Args:
      n_verts:    quotient vertex count (≤32).
      edges_uv:   list of quotient edges as (u, v) pairs.
      signs:      monodromy χ per quotient edge.
      is_loop:    1 if quotient edge is a loop (σ-fixed cover edge), 0 else.
      elim_order: vertex elim order.
      r_E_G:      rank of full edge set in cover G.
      x_val, y_val, p: evaluation point.

    Returns (T_fix^σ mod p, max_states) or None on failure.
    """
    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    n_edges = len(edges_uv)
    edges_flat: List[int] = []
    for u, v in edges_uv:
        edges_flat.append(u); edges_flat.append(v)

    x_minus_1 = (x_val - 1) % p
    y_minus_1 = (y_val - 1) % p
    # Modular inverse of (x-1): Fermat for prime p.
    if x_minus_1 == 0:
        return None  # degenerate: x_val == 1, (x-1)^{-1} undefined
    x_inv = pow(x_minus_1, p - 2, p)
    # Precompute factor table.
    factor_tree_xbal_xbal = (x_inv * x_inv) % p  # (x-1)^{-2}
    factor_tree_unbal_unbal = (x_inv * y_minus_1) % p  # (x-1)^{-1}(y-1)
    factor_cycle_no_rank_change = (y_minus_1 * y_minus_1) % p  # (y-1)^2
    factor_unbal_cycle_in_bal = (x_inv * y_minus_1) % p  # (x-1)^{-1}(y-1)
    factor_loop_bal_cycle_in_bal = y_minus_1  # (y-1)
    factor_loop_unbal_cycle_in_bal = x_inv  # (x-1)^{-1}
    factor_loop_cycle_in_unbal = y_minus_1  # (y-1)

    edges_buf = _ffi.new("int[]", edges_flat) if edges_flat else _ffi.new("int[]", [0])
    signs_buf = _ffi.new("int[]", list(signs)) if signs else _ffi.new("int[]", [0])
    is_loop_buf = _ffi.new("int[]", list(is_loop)) if is_loop else _ffi.new("int[]", [0])
    elim_buf = _ffi.new("int[]", list(elim_order))
    out_total = _ffi.new("long long*")
    out_max_st = _ffi.new("int*")

    rc = lib.t_fix_dp_full_c(
        n_verts, n_edges, edges_buf, signs_buf, is_loop_buf, elim_buf,
        r_E_G, x_minus_1, p,
        factor_tree_xbal_xbal, factor_tree_unbal_unbal,
        factor_cycle_no_rank_change, factor_unbal_cycle_in_bal,
        factor_loop_bal_cycle_in_bal, factor_loop_unbal_cycle_in_bal,
        factor_loop_cycle_in_unbal,
        out_total, out_max_st,
    )
    if rc != 0:
        return None
    return int(out_total[0]), int(out_max_st[0])


def dp_full(
    n_verts: int,
    edges_uv: List[Tuple[int, int]],
    signs: List[int],
    elim_order: List[int],
    r_E: int,
    x_minus_1: int,
    y_minus_1: int,
    p: int,
) -> Optional[Tuple[int, int]]:
    """Run the entire signed-DP loop in C — no per-step Python marshalling.

    Args:
      n_verts:     number of quotient vertices (≤32).
      edges_uv:    list of (u, v) edge pairs.
      signs:       list of 0/1 sign per edge, same length as edges_uv.
      elim_order:  vertex elimination order, length n_verts.
      r_E:         rank of full edge set in cover G.
      x_minus_1, y_minus_1, p: evaluation point.

    Returns (total_mod_p, max_states), or None if C-ext unavailable / errored.
    """
    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    n_edges = len(edges_uv)
    edges_flat: List[int] = []
    for u, v in edges_uv:
        edges_flat.append(u)
        edges_flat.append(v)

    edges_buf = _ffi.new("int[]", edges_flat) if edges_flat else _ffi.new("int[]", [0])
    signs_buf = _ffi.new("int[]", list(signs)) if signs else _ffi.new("int[]", [0])
    elim_buf  = _ffi.new("int[]", list(elim_order))
    out_total = _ffi.new("long long*")
    out_max_st = _ffi.new("int*")

    rc = lib.signed_dp_full_c(
        n_verts, n_edges, edges_buf, signs_buf, elim_buf,
        r_E, x_minus_1, y_minus_1, p,
        out_total, out_max_st,
    )
    if rc != 0:
        return None
    return int(out_total[0]), int(out_max_st[0])


def step_forget_batch(
    states: dict,
    fpos: int,
    p: int,
) -> Optional[dict]:
    """Apply one forget step (drop position fpos) to all states via C ext."""
    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    n_in = len(states)
    if n_in == 0:
        return {}

    in_data: List[int] = []
    in_offsets: List[int] = []
    in_weights: List[int] = []
    for st, w in states.items():
        part, mono, bal, fin = st
        in_offsets.append(len(in_data))
        in_data.extend(encode_state(part, mono, bal, fin))
        in_weights.append(int(w) % p)
    in_offsets.append(len(in_data))

    out_cap_data = max(n_in * 64, 1024)
    out_cap_weights = max(n_in, 64)

    in_data_buf = _ffi.new("int[]", in_data)
    in_offsets_buf = _ffi.new("int[]", in_offsets)
    in_weights_buf = _ffi.new("long long[]", in_weights)
    out_data_buf = _ffi.new("int[]", out_cap_data)
    out_offsets_buf = _ffi.new("int[]", out_cap_weights + 1)
    out_weights_buf = _ffi.new("long long[]", out_cap_weights)
    out_n_buf = _ffi.new("int*")

    rc = lib.signed_step_forget_batch_c(
        in_data_buf, in_offsets_buf, in_weights_buf, n_in,
        fpos, p,
        out_data_buf, out_offsets_buf, out_weights_buf, out_n_buf,
        out_cap_data, out_cap_weights,
    )
    if rc != 0:
        return None

    n_out = out_n_buf[0]
    new_states = {}
    for i in range(n_out):
        s = out_offsets_buf[i]
        e = out_offsets_buf[i + 1]
        flat = [out_data_buf[j] for j in range(s, e)]
        key = decode_state(flat)
        new_states[key] = int(out_weights_buf[i])
    return new_states
