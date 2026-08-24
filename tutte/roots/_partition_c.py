"""C extension for partition operations on the path/grid DP hot path.

Hot path: `precompute_M_table` calls `join_partitions` 14M+ times for
Cm3-class graphs (52% of total time per cProfile). Each call is union-find
over a small universe (≤24 verts).

Pattern mirrors `tutte/_polynomial_c.py`: lazy auto-compile on first import.
Pure-Python fallback if cffi compile fails.

Encoding: a partition is encoded as a flat int array
  [block_count, block_1_size, b1_v1, b1_v2, ..., block_2_size, b2_v1, ...]
The C function takes encoded p1, p2, and a universe array (the set of
positions); returns the joined partition in the same encoding.

Positions are relabeled to 0..n_universe-1 in Python before the C call
to avoid hash-table needs in C. Output positions get re-mapped back.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import cffi

ffi = cffi.FFI()

ffi.cdef(r"""
    /* Joins two encoded partitions over a universe of size n_universe.
       Positions in p1, p2 must be in [0, n_universe).
       p1_data: [block_count, b1_size, b1_v1, ..., b2_size, ...]
       p2_data: same encoding.
       out_data: pre-allocated output buffer, capacity ints.
       Returns the number of ints written to out_data, or -1 on overflow.

       Output is canonically sorted (blocks sorted by min element;
       within each block, vertices sorted ascending). */
    int join_partitions_c(
        const int* p1_data, int p1_len,
        const int* p2_data, int p2_len,
        int n_universe,
        int* out_data, int out_capacity);

    /* Batched inner loop for precompute_M_table.

       For ONE state partition (P_state_ext), processes ALL junction
       partitions in batch. For each junction:
         1. Compute delta on shared boundary (positions [0, n_shared)).
         2. If delta < 0: skip (output_deltas[i] = -1).
         3. Else: join over full universe, restrict to out_boundary,
            compute per-cell canonical key.
         4. Output: delta + out canonical key (encoded).

       All position arrays use indices in [0, n_universe). Caller has
       already relabeled. The cell_groups array is also in [0, n_universe).

       boundaries_data: flat array containing all boundary subsets,
         indexed by *_start, *_end positions:
           shared = [0, n_shared)
           extra = [n_shared, n_shared + n_extra)
           state_extra = [n_shared + n_extra, n_shared + n_extra + n_state_extra)
           out = caller computes from above + keep_shared flag, passes separately.

       Returns: total ints written to out_buf (= sum of per-junction output
       segments + one int per junction for the delta), or -1 on overflow.

       Output layout per junction j (in out_buf):
         [delta_j, n_canonical_ints, canonical_data...]
       If delta < 0: just [-1, 0]. */
    int batched_inner_iterations_c(
        /* State partition (extended with extra_boundary as singletons) */
        const int* p_state_ext, int p_state_ext_len,
        /* State partition restricted to shared boundary (precomputed) */
        const int* p_state_S, int p_state_S_len,
        /* All junction partitions in this batch (encoded + extended) */
        const int* junc_parts_ext_data, const int* junc_parts_ext_offsets,
        const int* junc_parts_ext_lens,
        const int* junc_parts_S_data, const int* junc_parts_S_offsets,
        const int* junc_parts_S_lens,
        int n_junc,
        /* Universe parameters */
        int n_universe, int n_shared,
        /* Out boundary indices in [0, n_universe) */
        const int* out_boundary_idx, int n_out_boundary,
        /* Cell anchor groups for canonical key.
           Encoded: [n_cells, c0_size, c0_v0, ..., c1_size, c1_v0, ...] */
        const int* cell_groups_data, int cell_groups_len, int n_cells,
        /* Output buffer: one segment per junction (variable length).
           Returns total ints written, or -1 on capacity exceeded. */
        int* out_buf, int out_capacity,
        int* out_offsets);

    /* Self-test for the C hash map data structure.
       Inserts `n_keys` keys (each `key_lens[k]` ints from `keys_flat`
       at offset `keys_offsets[k]`) with corresponding `values[k]` into
       a hash map of size `capacity`. Modular accumulation mod `p`.
       Writes unique keys + summed values to out_*; returns number of
       unique keys, or -1 on overflow.

       Used to validate the hash-map primitives before wiring into the
       full aggregation function. */
    int hashmap_self_test_c(
        const int* keys_flat, const int* keys_offsets, const int* key_lens,
        const long long* values, int n_keys,
        int capacity, int keys_buffer_cap, long long p,
        int* out_keys_flat, int* out_keys_offsets, int* out_key_lens,
        long long* out_values, int* out_n_unique);

    /* Aggregate one state-orbit's batched_inner_iterations output
       into a caller-allocated hash map. Replaces the Python inner loop
       in `precompute_and_convolve_c_mod` (lines 1142-1162) with a
       single C call per state orbit.

       The hash map uses caller-provided slot arrays so it persists
       across multiple calls (one per state orbit) and accumulates
       contributions until the caller marshals the result.

       Per-junction logic mirrors the Python loop:
           delta = out_buf[seg_start]
           if delta < 0: skip
           n_can = out_buf[seg_start + 1]
           key bytes = canonical_data segment of length n_blocks * shape_size
           contrib = sv_n_state_mod * jv_mod * xy_pow_mod[delta] % p
           hashmap_add(key, contrib)

       Returns 0 on success, -1 on hashmap overflow (caller must size
       hm_capacity and hm_keys_cap larger and retry).

       Performance: replaces ~27 M Python dict ops on Cm₃ with C-side
       open-addressing inserts. Projected 5-8× per the design doc
       `tutte/research/plans/cm3_unlock_design.md`. */
    int aggregate_buf_to_hashmap_c(
        const int* out_buf, const int* out_offsets, int n_junc,
        int shape_size,
        const long long* junc_jv_mod, long long sv_n_state_mod,
        const long long* xy_pow_mod, long long p,
        int* hm_used, int* hm_key_off, int* hm_key_len,
        long long* hm_values, int hm_capacity,
        int* hm_keys_buffer, int hm_keys_cap, int* hm_keys_used,
        int* hm_n_unique);

    /* Marshal a caller-allocated hash map's contents into flat
       output arrays. Iterates the slot table, copying entries for
       used slots only. Returns the number of unique keys written.

       Allows the Python caller to defer marshaling until after all
       state orbits have been aggregated. */
    int hashmap_marshal_c(
        const int* hm_used, const int* hm_key_off, const int* hm_key_len,
        const long long* hm_values, int hm_capacity,
        const int* hm_keys_buffer,
        int* out_keys_flat, int* out_keys_offsets, int* out_key_lens,
        long long* out_values, int out_keys_buffer_cap);

    /* Apply permutation `perm` to encoded partition `P_enc` and
       write its canonical (lex-sorted blocks, sorted-elements-within-blocks)
       form to out_canon. `perm` is length n_universe; perm[v] is the
       image of vertex v under the permutation.

       Encoding: [n_blocks, b1_size, b1_v1, ..., b2_size, b2_v1, ...]
       Returns: out_canon length, or -1 on overflow or invalid input. */
    int apply_perm_canonical_c(
        const int* P_enc, int P_enc_len,
        const int* perm, int n_universe,
        int* out_canon, int out_capacity);

    /* For partition P_enc, compute the lex-min canonical
       encoding across all `n_H` permutations in `H_perms_flat`.
       H_perms_flat is a flat array of length n_H * n_universe: perm
       i is at offset i * n_universe.

       This implements the H-canonicalization inner loop of
       the pair-orbit M-table DP (R7 prototype, since removed) in C. Per
       `tutte/research/plans/cm3_unlock_design.md`: ~14k ops per pair
       in Python → projected ~10× speedup in C.

       Returns: lex-min encoding length, or -1 on overflow. */
    int h_canonicalize_c(
        const int* P_enc, int P_enc_len,
        const int* H_perms_flat, int n_H, int n_universe,
        int* out_canon, int out_capacity);

    /* Batched H-canonicalize: process n_partitions partitions in one
       C call. Inputs:
         P_flat: flat buffer with all partition encodings concatenated.
         P_offsets[i]: start offset of partition i in P_flat (length n+1
                       with P_offsets[n] = total length).
         H_perms_flat, n_H, n_universe: as in h_canonicalize_c.
         out_flat: output buffer for canonical encodings.
         out_offsets[i]: start offset of partition i's canonical in out_flat
                         (length n+1, written by C).
         out_capacity: total size of out_flat.
       Returns: total bytes written to out_flat, or -1 on overflow. */
    int h_canonicalize_batch_c(
        const int* P_flat, const int* P_offsets, int n_partitions,
        const int* H_perms_flat, int n_H, int n_universe,
        int* out_flat, int* out_offsets, int out_capacity);

    /* Compute set-stabilizer of partition P_enc under permutation group G.
       For each perm in G_perms_flat (length n_G * n_universe), check if
       applying it to P leaves the partition setwise invariant. Output:
       out_stab_indices[i] = index into G_perms_flat for the i-th stabilizer
       element (i < returned n_stab).

       Returns: number of stabilizer perms found, or -1 on error.

       This replaces `per_cell_partition_stab` Python loop:
         [σ for σ in G_elements if apply_perm_to_partition(P, σ) == P]
       For Cm_3: 6608 states × 576 G-perms × Python apply_perm = 30+s
       bottleneck. C-side: same arithmetic in microseconds. */
    int set_stabilizer_c(
        const int* P_enc, int P_enc_len,
        const int* G_perms_flat, int n_G, int n_universe,
        int* out_stab_indices, int out_capacity);

    /* Enumerate per-cell orbit members of partition rep under permutation
       group G_perms_flat (n_G perms × n_universe), early-terminating once
       target_size distinct canonical members found.

       Output:
         out_members_flat: concatenated canonical encodings of each member.
         out_members_offsets[i]: start offset of member i in out_members_flat
                                  (length max_members+1; out_members_offsets[n]
                                  = total length).
       Returns: number of distinct members found (≤ target_size), or -1 on
       overflow. */
    int expand_orbit_members_c(
        const int* rep_enc, int rep_enc_len,
        const int* G_perms_flat, int n_G, int n_universe,
        int target_size,
        int* out_members_flat, int* out_members_offsets, int out_capacity,
        int max_members);
""")

ffi.set_source("_tutte_partition_cffi", r"""
    #include <stdlib.h>
    #include <string.h>
    #include <stdint.h>

    /* === open-addressing hash map for O_out aggregation ===
       Keys are variable-length int sequences; values are long long mod p.
       Linear probing with FNV-1a 64-bit hash. Caller provides backing
       arrays for slots and the key buffer.

       Capacity should be ≥ 2 × expected unique keys for low collision rate.
       keys_buffer_cap should be ≥ total ints across all unique keys. */

    typedef struct {
        int* used;          /* slot occupied flag (0/1) */
        int* key_off;       /* offset into keys_buffer */
        int* key_len;       /* key length in ints */
        long long* values;  /* value mod p */
        int capacity;
        int* keys_buffer;
        int keys_cap;
        int keys_used;
        int n_unique;
    } hashmap_t;

    static uint64_t fnv1a_64_ints(const int* data, int n) {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (int i = 0; i < n; i++) {
            uint32_t v = (uint32_t)data[i];
            for (int b = 0; b < 4; b++) {
                h ^= (uint64_t)(v & 0xff);
                h *= 0x100000001b3ULL;
                v >>= 8;
            }
        }
        return h;
    }

    static int hashmap_keys_equal(const int* a, int alen,
                                   const int* b, int blen) {
        if (alen != blen) return 0;
        for (int i = 0; i < alen; i++) if (a[i] != b[i]) return 0;
        return 1;
    }

    /* Returns 0 on success, -1 on keys_buffer overflow (slot table is
       caller-sized so won't overflow if capacity is sufficient). */
    static int hashmap_add(hashmap_t* m, const int* key, int klen,
                           long long val, long long p) {
        if (m->capacity <= 0) return -1;
        uint64_t h = fnv1a_64_ints(key, klen);
        int slot = (int)(h % (uint64_t)m->capacity);
        for (int probe = 0; probe < m->capacity; probe++) {
            if (!m->used[slot]) {
                /* Empty: insert. */
                if (m->keys_used + klen > m->keys_cap) return -1;
                int new_off = m->keys_used;
                for (int i = 0; i < klen; i++) {
                    m->keys_buffer[new_off + i] = key[i];
                }
                m->keys_used += klen;
                m->key_off[slot] = new_off;
                m->key_len[slot] = klen;
                m->values[slot] = ((val % p) + p) % p;
                m->used[slot] = 1;
                m->n_unique++;
                return 0;
            }
            int existing_off = m->key_off[slot];
            int existing_len = m->key_len[slot];
            if (hashmap_keys_equal(m->keys_buffer + existing_off,
                                    existing_len, key, klen)) {
                long long s = m->values[slot] + val;
                s %= p;
                if (s < 0) s += p;
                m->values[slot] = s;
                return 0;
            }
            slot = (slot + 1) % m->capacity;
        }
        return -1;  /* full — caller should size capacity larger */
    }

    int hashmap_self_test_c(
        const int* keys_flat, const int* keys_offsets, const int* key_lens,
        const long long* values, int n_keys,
        int capacity, int keys_buffer_cap, long long p,
        int* out_keys_flat, int* out_keys_offsets, int* out_key_lens,
        long long* out_values, int* out_n_unique)
    {
        /* Allocate slot tables on the caller-provided slots. Use the
           input arrays' last region as scratch... no, allocate inline. */
        if (capacity <= 0 || keys_buffer_cap <= 0) return -1;
        int* used = (int*)calloc(capacity, sizeof(int));
        int* key_off = (int*)calloc(capacity, sizeof(int));
        int* key_len = (int*)calloc(capacity, sizeof(int));
        long long* vals = (long long*)calloc(capacity, sizeof(long long));
        int* keys_buf = (int*)calloc(keys_buffer_cap, sizeof(int));
        if (!used || !key_off || !key_len || !vals || !keys_buf) {
            free(used); free(key_off); free(key_len); free(vals); free(keys_buf);
            return -1;
        }
        hashmap_t m;
        m.used = used; m.key_off = key_off; m.key_len = key_len;
        m.values = vals; m.capacity = capacity;
        m.keys_buffer = keys_buf; m.keys_cap = keys_buffer_cap;
        m.keys_used = 0; m.n_unique = 0;

        int rc = 0;
        for (int k = 0; k < n_keys; k++) {
            const int* kk = keys_flat + keys_offsets[k];
            int kl = key_lens[k];
            if (hashmap_add(&m, kk, kl, values[k], p) < 0) {
                rc = -1;
                break;
            }
        }

        if (rc == 0) {
            /* Marshal results to caller arrays. */
            int out_idx = 0;
            int out_off = 0;
            for (int s = 0; s < capacity; s++) {
                if (!m.used[s]) continue;
                out_keys_offsets[out_idx] = out_off;
                out_key_lens[out_idx] = m.key_len[s];
                for (int i = 0; i < m.key_len[s]; i++) {
                    out_keys_flat[out_off + i] = m.keys_buffer[m.key_off[s] + i];
                }
                out_off += m.key_len[s];
                out_values[out_idx] = m.values[s];
                out_idx++;
            }
            *out_n_unique = out_idx;
        }

        free(used); free(key_off); free(key_len); free(vals); free(keys_buf);
        return rc == 0 ? m.n_unique : -1;
    }
    int aggregate_buf_to_hashmap_c(
        const int* out_buf, const int* out_offsets, int n_junc,
        int shape_size,
        const long long* junc_jv_mod, long long sv_n_state_mod,
        const long long* xy_pow_mod, long long p,
        int* hm_used, int* hm_key_off, int* hm_key_len,
        long long* hm_values, int hm_capacity,
        int* hm_keys_buffer, int hm_keys_cap, int* hm_keys_used,
        int* hm_n_unique)
    {
        if (sv_n_state_mod == 0) return 0;
        hashmap_t m;
        m.used = hm_used; m.key_off = hm_key_off; m.key_len = hm_key_len;
        m.values = hm_values; m.capacity = hm_capacity;
        m.keys_buffer = hm_keys_buffer; m.keys_cap = hm_keys_cap;
        m.keys_used = *hm_keys_used; m.n_unique = *hm_n_unique;

        for (int j = 0; j < n_junc; j++) {
            long long jv = junc_jv_mod[j];
            if (jv == 0) continue;
            int seg_start = out_offsets[j];
            int delta = out_buf[seg_start];
            if (delta < 0) continue;
            int n_can = out_buf[seg_start + 1];
            const int* key_ptr;
            int key_len_ints;
            if (n_can == 0) {
                key_ptr = NULL;
                key_len_ints = 0;
            } else {
                int n_blocks = out_buf[seg_start + 2];
                key_ptr = out_buf + seg_start + 3;
                key_len_ints = n_blocks * shape_size;
            }
            long long contrib = (sv_n_state_mod * jv) % p;
            contrib = (contrib * xy_pow_mod[delta]) % p;
            if (contrib == 0) continue;
            int rc = hashmap_add(&m, key_ptr, key_len_ints, contrib, p);
            if (rc < 0) {
                /* Persist state back even on failure so caller can diagnose. */
                *hm_keys_used = m.keys_used;
                *hm_n_unique = m.n_unique;
                return -1;
            }
        }
        *hm_keys_used = m.keys_used;
        *hm_n_unique = m.n_unique;
        return 0;
    }

    int hashmap_marshal_c(
        const int* hm_used, const int* hm_key_off, const int* hm_key_len,
        const long long* hm_values, int hm_capacity,
        const int* hm_keys_buffer,
        int* out_keys_flat, int* out_keys_offsets, int* out_key_lens,
        long long* out_values, int out_keys_buffer_cap)
    {
        int out_idx = 0;
        int out_off = 0;
        for (int s = 0; s < hm_capacity; s++) {
            if (!hm_used[s]) continue;
            int klen = hm_key_len[s];
            if (out_off + klen > out_keys_buffer_cap) return -1;
            out_keys_offsets[out_idx] = out_off;
            out_key_lens[out_idx] = klen;
            const int* src = hm_keys_buffer + hm_key_off[s];
            for (int i = 0; i < klen; i++) {
                out_keys_flat[out_off + i] = src[i];
            }
            out_off += klen;
            out_values[out_idx] = hm_values[s];
            out_idx++;
        }
        return out_idx;
    }

    /* === end hash map === */



    /* Path-compressing find. */
    static int uf_find(int* parent, int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    /* Union by min-rep (matches Python convention). */
    static void uf_union(int* parent, int a, int b) {
        int ra = uf_find(parent, a);
        int rb = uf_find(parent, b);
        if (ra == rb) return;
        if (ra < rb) parent[rb] = ra;
        else parent[ra] = rb;
    }

    /* Apply unions encoded in `data` (one partition's blocks) to parent. */
    static void apply_blocks(int* parent, const int* data, int n) {
        if (n <= 0) return;
        int block_count = data[0];
        int idx = 1;
        for (int b = 0; b < block_count && idx < n; b++) {
            int bsize = data[idx++];
            if (bsize >= 1 && idx < n) {
                int rep = data[idx];
                for (int i = 1; i < bsize && idx + i < n; i++) {
                    uf_union(parent, rep, data[idx + i]);
                }
            }
            idx += bsize;
        }
    }

    /* Comparator for sorting block-tuples canonically. We sort blocks by
       lexicographic order; each block's vertices are sorted ascending
       beforehand, so block-ordering comparison reduces to comparing
       (size, then per-position values). For our tiny block counts (~24),
       a stable insertion sort is fastest. */

    int join_partitions_c(
        const int* p1_data, int p1_len,
        const int* p2_data, int p2_len,
        int n_universe,
        int* out_data, int out_capacity)
    {
        if (n_universe <= 0) {
            if (out_capacity < 1) return -1;
            out_data[0] = 0;
            return 1;
        }

        /* Stack-allocated parent array (universe ≤ 64 should cover all
           practical cases — Cm3 has 12-24 verts max). */
        int parent[256];
        if (n_universe > 256) return -1;
        for (int i = 0; i < n_universe; i++) parent[i] = i;

        /* Apply unions from both partitions. */
        apply_blocks(parent, p1_data, p1_len);
        apply_blocks(parent, p2_data, p2_len);

        /* Group by component. block_of[v] = index of v's block in output. */
        int block_of[256];
        int blocks[256][256];   /* block contents (overkill but safe) */
        int block_sizes[256];
        int n_blocks = 0;
        int rep_to_block[256];
        for (int i = 0; i < n_universe; i++) rep_to_block[i] = -1;

        for (int v = 0; v < n_universe; v++) {
            int r = uf_find(parent, v);
            if (rep_to_block[r] < 0) {
                rep_to_block[r] = n_blocks;
                block_sizes[n_blocks] = 0;
                n_blocks++;
            }
            int bidx = rep_to_block[r];
            blocks[bidx][block_sizes[bidx]++] = v;
        }

        /* Each block is already sorted (we iterated v in 0..n-1).
           Now sort blocks by (first vertex), which is the canonical order. */
        /* Simple insertion sort on block indices by blocks[idx][0]. */
        int order[256];
        for (int i = 0; i < n_blocks; i++) order[i] = i;
        for (int i = 1; i < n_blocks; i++) {
            int key = order[i];
            int j = i - 1;
            while (j >= 0 && blocks[order[j]][0] > blocks[key][0]) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }

        /* Encode output: [n_blocks, b1_size, b1_v1, ..., b2_size, ...]. */
        int out_idx = 0;
        if (out_capacity < 1) return -1;
        out_data[out_idx++] = n_blocks;
        for (int b = 0; b < n_blocks; b++) {
            int bidx = order[b];
            int bsize = block_sizes[bidx];
            if (out_idx + 1 + bsize > out_capacity) return -1;
            out_data[out_idx++] = bsize;
            for (int i = 0; i < bsize; i++) {
                out_data[out_idx++] = blocks[bidx][i];
            }
        }
        return out_idx;
    }

    /* === Batched inner loop for precompute_M_table === */

    /* Compute delta(P_state_S, P_junc_S) on shared boundary.
       Both partitions use indices in [0, n_shared).
       Returns delta = nblocks(JOIN) + n_shared - nblocks(P_state_S) - nblocks(P_junc_S). */
    static int compute_delta(
        const int* p_state_S, int p_state_S_len,
        const int* p_junc_S, int p_junc_S_len,
        int n_shared)
    {
        if (n_shared == 0) {
            return 0 - 0 - 0;  /* nblocks(empty) = 0 */
        }
        int parent[256];
        for (int i = 0; i < n_shared; i++) parent[i] = i;
        apply_blocks(parent, p_state_S, p_state_S_len);
        apply_blocks(parent, p_junc_S, p_junc_S_len);
        /* Count distinct components */
        int seen[256] = {0};
        int n_join_blocks = 0;
        for (int v = 0; v < n_shared; v++) {
            int r = uf_find(parent, v);
            if (!seen[r]) { seen[r] = 1; n_join_blocks++; }
        }
        int n_state_blocks = (p_state_S_len > 0) ? p_state_S[0] : 0;
        int n_junc_blocks = (p_junc_S_len > 0) ? p_junc_S[0] : 0;
        return n_join_blocks + n_shared - n_state_blocks - n_junc_blocks;
    }

    /* Compute joint partition on full universe (size n_universe).
       Then restrict to out_boundary_idx, then compute per-cell canonical key.
       Output written as: [n_canonical_blocks, shape_0_per_cell..., shape_1_per_cell...]
       where each shape is n_cells+1 ints (last = "outside" count = 0 for valid cases).
       Returns: number of ints written. */
    static int join_restrict_canonical(
        const int* p_state_ext, int p_state_ext_len,
        const int* p_junc_ext, int p_junc_ext_len,
        int n_universe,
        const int* out_boundary_idx, int n_out_boundary,
        const int* cell_groups_data, int n_cells,
        int* out_canonical_buf, int buf_capacity)
    {
        if (n_out_boundary == 0) {
            /* Empty out boundary: canonical key is empty tuple */
            if (buf_capacity < 1) return -1;
            out_canonical_buf[0] = 0;  /* n_blocks = 0 */
            return 1;
        }
        if (n_universe > 256) return -1;

        int parent[256];
        for (int i = 0; i < n_universe; i++) parent[i] = i;
        apply_blocks(parent, p_state_ext, p_state_ext_len);
        apply_blocks(parent, p_junc_ext, p_junc_ext_len);

        /* Build pos_to_cell map: for each index in [0, n_universe),
           which cell does it belong to? -1 = outside. */
        int pos_to_cell[256];
        for (int i = 0; i < n_universe; i++) pos_to_cell[i] = -1;
        int idx = 0;
        if (cell_groups_data[idx++] != n_cells) return -1;  /* sanity */
        for (int c = 0; c < n_cells; c++) {
            int csize = cell_groups_data[idx++];
            for (int i = 0; i < csize; i++) {
                int v = cell_groups_data[idx++];
                if (v >= 0 && v < n_universe) pos_to_cell[v] = c;
            }
        }

        /* Build out_in_universe lookup: is index v in out boundary? */
        int in_out[256] = {0};
        for (int i = 0; i < n_out_boundary; i++) in_out[out_boundary_idx[i]] = 1;

        /* For each component (root), compute its shape over cells +
           outside (n_cells+1 slots), but only counting positions in
           out_boundary. */
        int comp_shape[256][16];  /* n_cells+1 ≤ 16 in practice */
        int comp_seen[256] = {0};
        int n_comps_in_out = 0;
        int comp_root_to_idx[256];
        for (int i = 0; i < n_universe; i++) comp_root_to_idx[i] = -1;
        if (n_cells + 1 > 16) return -1;

        for (int v = 0; v < n_universe; v++) {
            if (!in_out[v]) continue;
            int r = uf_find(parent, v);
            int ci = pos_to_cell[v] >= 0 ? pos_to_cell[v] : n_cells;
            if (!comp_seen[r]) {
                comp_seen[r] = 1;
                comp_root_to_idx[r] = n_comps_in_out;
                for (int j = 0; j < n_cells + 1; j++) comp_shape[n_comps_in_out][j] = 0;
                n_comps_in_out++;
            }
            comp_shape[comp_root_to_idx[r]][ci]++;
        }

        /* Build sorted multiset of shape tuples. Each tuple has n_cells+1 ints.
           Sort blocks lexicographically. Use insertion sort. */
        int order[256];
        for (int i = 0; i < n_comps_in_out; i++) order[i] = i;
        for (int i = 1; i < n_comps_in_out; i++) {
            int key = order[i];
            int j = i - 1;
            while (j >= 0) {
                /* Compare comp_shape[order[j]] vs comp_shape[key] lex. */
                int cmp = 0;
                for (int k = 0; k < n_cells + 1; k++) {
                    if (comp_shape[order[j]][k] < comp_shape[key][k]) { cmp = -1; break; }
                    if (comp_shape[order[j]][k] > comp_shape[key][k]) { cmp = 1; break; }
                }
                if (cmp <= 0) break;
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }

        /* Encode: [n_blocks, shape_0_data..., shape_1_data..., ...] */
        int needed = 1 + n_comps_in_out * (n_cells + 1);
        if (buf_capacity < needed) return -1;
        int out_idx = 0;
        out_canonical_buf[out_idx++] = n_comps_in_out;
        for (int b = 0; b < n_comps_in_out; b++) {
            int bidx = order[b];
            for (int k = 0; k < n_cells + 1; k++) {
                out_canonical_buf[out_idx++] = comp_shape[bidx][k];
            }
        }
        return out_idx;
    }

    int batched_inner_iterations_c(
        const int* p_state_ext, int p_state_ext_len,
        const int* p_state_S, int p_state_S_len,
        const int* junc_parts_ext_data, const int* junc_parts_ext_offsets,
        const int* junc_parts_ext_lens,
        const int* junc_parts_S_data, const int* junc_parts_S_offsets,
        const int* junc_parts_S_lens,
        int n_junc,
        int n_universe, int n_shared,
        const int* out_boundary_idx, int n_out_boundary,
        const int* cell_groups_data, int cell_groups_len, int n_cells,
        int* out_buf, int out_capacity,
        int* out_offsets)
    {
        int total_written = 0;
        for (int j = 0; j < n_junc; j++) {
            out_offsets[j] = total_written;
            const int* p_junc_S = junc_parts_S_data + junc_parts_S_offsets[j];
            int p_junc_S_len = junc_parts_S_lens[j];
            int delta = compute_delta(
                p_state_S, p_state_S_len,
                p_junc_S, p_junc_S_len,
                n_shared
            );
            /* Write delta */
            if (total_written + 1 > out_capacity) return -1;
            out_buf[total_written++] = delta;
            if (delta < 0) {
                /* Skip canonical computation; write 0 for n_canonical_ints */
                if (total_written + 1 > out_capacity) return -1;
                out_buf[total_written++] = 0;
                continue;
            }
            const int* p_junc_ext = junc_parts_ext_data + junc_parts_ext_offsets[j];
            int p_junc_ext_len = junc_parts_ext_lens[j];

            /* Reserve space for canonical_size + canonical_data */
            if (total_written + 1 > out_capacity) return -1;
            int* canonical_size_slot = &out_buf[total_written++];
            int n_can = join_restrict_canonical(
                p_state_ext, p_state_ext_len,
                p_junc_ext, p_junc_ext_len,
                n_universe,
                out_boundary_idx, n_out_boundary,
                cell_groups_data, n_cells,
                out_buf + total_written, out_capacity - total_written
            );
            if (n_can < 0) return -1;
            *canonical_size_slot = n_can;
            total_written += n_can;
        }
        out_offsets[n_junc] = total_written;
        return total_written;
    }

    /* Apply permutation `perm` to partition P_enc and emit
       canonical form. */
    int apply_perm_canonical_c(
        const int* P_enc, int P_enc_len,
        const int* perm, int n_universe,
        int* out_canon, int out_capacity)
    {
        if (P_enc_len == 0) {
            if (out_capacity < 1) return -1;
            out_canon[0] = 0;
            return 1;
        }
        int n_blocks = P_enc[0];
        if (n_blocks > 256) return -1;

        /* Decode + apply perm + sort each block. */
        int blocks[256][256];
        int block_sizes[256];
        int idx = 1;
        for (int b = 0; b < n_blocks; b++) {
            if (idx >= P_enc_len) return -1;
            int sz = P_enc[idx++];
            if (sz > 256) return -1;
            block_sizes[b] = sz;
            for (int i = 0; i < sz; i++) {
                if (idx >= P_enc_len) return -1;
                int v = P_enc[idx++];
                if (v < 0 || v >= n_universe) return -1;
                blocks[b][i] = perm[v];
            }
            /* Insertion sort block elements ascending. */
            for (int i = 1; i < sz; i++) {
                int x = blocks[b][i];
                int j = i - 1;
                while (j >= 0 && blocks[b][j] > x) {
                    blocks[b][j + 1] = blocks[b][j];
                    j--;
                }
                blocks[b][j + 1] = x;
            }
        }

        /* Insertion sort blocks by (first_element, size). Blocks of
           size 0 are placed last (shouldn't occur in valid input). */
        int order[256];
        for (int i = 0; i < n_blocks; i++) order[i] = i;
        for (int i = 1; i < n_blocks; i++) {
            int key = order[i];
            int kfirst = (block_sizes[key] > 0) ? blocks[key][0] : 0x7fffffff;
            int ksize = block_sizes[key];
            int j = i - 1;
            while (j >= 0) {
                int j_first = (block_sizes[order[j]] > 0) ? blocks[order[j]][0] : 0x7fffffff;
                int j_size = block_sizes[order[j]];
                int cmp = 0;
                if (j_first < kfirst) cmp = -1;
                else if (j_first > kfirst) cmp = 1;
                else if (j_size < ksize) cmp = -1;
                else if (j_size > ksize) cmp = 1;
                if (cmp <= 0) break;
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }

        /* Emit. */
        int out_idx = 0;
        if (out_capacity < 1) return -1;
        out_canon[out_idx++] = n_blocks;
        for (int b = 0; b < n_blocks; b++) {
            int bidx = order[b];
            int sz = block_sizes[bidx];
            if (out_idx + 1 + sz > out_capacity) return -1;
            out_canon[out_idx++] = sz;
            for (int i = 0; i < sz; i++) {
                out_canon[out_idx++] = blocks[bidx][i];
            }
        }
        return out_idx;
    }

    /* Lex-min over H. */
    int h_canonicalize_c(
        const int* P_enc, int P_enc_len,
        const int* H_perms_flat, int n_H, int n_universe,
        int* out_canon, int out_capacity)
    {
        if (n_H == 0) {
            /* No permutations — output P_enc itself (assumed already
               canonical by the caller for the identity element). */
            if (out_capacity < P_enc_len) return -1;
            for (int i = 0; i < P_enc_len; i++) out_canon[i] = P_enc[i];
            return P_enc_len;
        }
        /* Worst-case canonical length = same as P_enc_len. */
        int candidate[2048];
        int best[2048];
        int best_len = -1;
        if (P_enc_len > 2048) return -1;

        for (int h = 0; h < n_H; h++) {
            const int* perm = H_perms_flat + (long long)h * n_universe;
            int can_len = apply_perm_canonical_c(P_enc, P_enc_len,
                                                  perm, n_universe,
                                                  candidate, 2048);
            if (can_len < 0) return -1;
            int cmp;
            if (best_len < 0) {
                cmp = -1;
            } else {
                cmp = 0;
                int n_cmp = (best_len < can_len) ? best_len : can_len;
                for (int i = 0; i < n_cmp; i++) {
                    if (candidate[i] < best[i]) { cmp = -1; break; }
                    if (candidate[i] > best[i]) { cmp = 1; break; }
                }
                if (cmp == 0) {
                    if (can_len < best_len) cmp = -1;
                    else if (can_len > best_len) cmp = 1;
                }
            }
            if (cmp < 0) {
                for (int i = 0; i < can_len; i++) best[i] = candidate[i];
                best_len = can_len;
            }
        }
        if (best_len < 0 || out_capacity < best_len) return -1;
        for (int i = 0; i < best_len; i++) out_canon[i] = best[i];
        return best_len;
    }

    /* Batched H-canonicalize: amortizes Python-side overhead by processing
       many partitions in one cffi call. For Cm_3 R19 path: ~6608 state
       orbits × ~4096 junction members each = ~27M partitions, where per-call
       cffi overhead in h_canonicalize_c batched-via-Python was the wall. */
    int h_canonicalize_batch_c(
        const int* P_flat, const int* P_offsets, int n_partitions,
        const int* H_perms_flat, int n_H, int n_universe,
        int* out_flat, int* out_offsets, int out_capacity)
    {
        int out_pos = 0;
        for (int i = 0; i < n_partitions; i++) {
            int p_start = P_offsets[i];
            int p_len = P_offsets[i + 1] - p_start;
            out_offsets[i] = out_pos;
            int remaining = out_capacity - out_pos;
            if (remaining < 0) return -1;
            int n_out;
            if (n_H == 0) {
                if (remaining < p_len) return -1;
                for (int j = 0; j < p_len; j++) out_flat[out_pos + j] = P_flat[p_start + j];
                n_out = p_len;
            } else {
                n_out = h_canonicalize_c(
                    P_flat + p_start, p_len,
                    H_perms_flat, n_H, n_universe,
                    out_flat + out_pos, remaining);
                if (n_out < 0) return -1;
            }
            out_pos += n_out;
        }
        out_offsets[n_partitions] = out_pos;
        return out_pos;
    }

    /* Hash a partition encoding via FNV-1a 64-bit. */
    static uint64_t partition_hash(const int* enc, int enc_len) {
        uint64_t h = 14695981039346656037ULL;
        for (int i = 0; i < enc_len; i++) {
            int v = enc[i];
            h ^= (uint64_t)v;
            h *= 1099511628211ULL;
        }
        return h;
    }

    /* Compare two encoded partitions. */
    static int partition_equal(const int* a, int alen, const int* b, int blen) {
        if (alen != blen) return 0;
        for (int i = 0; i < alen; i++) {
            if (a[i] != b[i]) return 0;
        }
        return 1;
    }

    /* Enumerate distinct canonical orbit members under G perms with
       early termination at target_size. Uses open-addressing hash set
       over canonical encodings. Caller's out_members_flat acts as the
       backing store for hash slots' keys (no extra alloc). */
    int expand_orbit_members_c(
        const int* rep_enc, int rep_enc_len,
        const int* G_perms_flat, int n_G, int n_universe,
        int target_size,
        int* out_members_flat, int* out_members_offsets, int out_capacity,
        int max_members)
    {
        if (rep_enc_len > 2048) return -1;

        /* Hash set: slots × 2 capacity for low collisions. Power of 2. */
        int hs_cap = 1;
        while (hs_cap < max_members * 2 && hs_cap < (1 << 20)) hs_cap <<= 1;
        if (hs_cap < 16) hs_cap = 16;
        /* Allocate small hash-set on stack via VLA-like static buffer.
           For large hs_cap (up to 1M), use heap. */
        int* hs_member_idx = (int*)malloc((size_t)hs_cap * sizeof(int));
        if (hs_member_idx == NULL) return -1;
        for (int i = 0; i < hs_cap; i++) hs_member_idx[i] = -1;
        int hs_mask = hs_cap - 1;

        int candidate[2048];
        int n_found = 0;
        int out_pos = 0;
        out_members_offsets[0] = 0;

        /* Identity perm = rep's canonical form (first member). */
        int identity[256];
        if (n_universe > 256) { free(hs_member_idx); return -1; }
        for (int i = 0; i < n_universe; i++) identity[i] = i;
        int can_len_first = apply_perm_canonical_c(
            rep_enc, rep_enc_len, identity, n_universe, candidate, 2048);
        if (can_len_first < 0) { free(hs_member_idx); return -1; }
        if (out_pos + can_len_first > out_capacity) { free(hs_member_idx); return -1; }
        for (int i = 0; i < can_len_first; i++) out_members_flat[out_pos + i] = candidate[i];
        out_pos += can_len_first;
        out_members_offsets[1] = out_pos;
        n_found = 1;
        uint64_t h_first = partition_hash(candidate, can_len_first);
        int slot = (int)(h_first & (uint64_t)hs_mask);
        while (hs_member_idx[slot] != -1) slot = (slot + 1) & hs_mask;
        hs_member_idx[slot] = 0;  /* index 0 */

        if (n_found >= target_size || n_found >= max_members) {
            free(hs_member_idx);
            return n_found;
        }

        for (int g = 1; g < n_G; g++) {
            const int* perm = G_perms_flat + (long long)g * n_universe;
            int can_len = apply_perm_canonical_c(
                rep_enc, rep_enc_len, perm, n_universe, candidate, 2048);
            if (can_len < 0) { free(hs_member_idx); return -1; }

            /* Hash lookup. */
            uint64_t h = partition_hash(candidate, can_len);
            int slot2 = (int)(h & (uint64_t)hs_mask);
            int found = 0;
            while (hs_member_idx[slot2] != -1) {
                int m_idx = hs_member_idx[slot2];
                int m_start = out_members_offsets[m_idx];
                int m_end = out_members_offsets[m_idx + 1];
                if (partition_equal(
                        out_members_flat + m_start, m_end - m_start,
                        candidate, can_len)) {
                    found = 1;
                    break;
                }
                slot2 = (slot2 + 1) & hs_mask;
            }
            if (found) continue;

            /* New member: append. */
            if (n_found >= max_members) { free(hs_member_idx); return -1; }
            if (out_pos + can_len > out_capacity) { free(hs_member_idx); return -1; }
            for (int i = 0; i < can_len; i++) out_members_flat[out_pos + i] = candidate[i];
            out_pos += can_len;
            hs_member_idx[slot2] = n_found;
            n_found++;
            out_members_offsets[n_found] = out_pos;
            if (n_found >= target_size) {
                free(hs_member_idx);
                return n_found;
            }
        }
        free(hs_member_idx);
        return n_found;
    }

    /* Set-stabilizer: enumerate perms σ in G with σ(P) == P (setwise).
       Stabilizer check: canonicalize P once (sorted blocks, sorted within),
       then for each σ, compute canonicalized σ(P) and compare. Both canonical
       forms are equal iff σ stabilizes P setwise. */
    int set_stabilizer_c(
        const int* P_enc, int P_enc_len,
        const int* G_perms_flat, int n_G, int n_universe,
        int* out_stab_indices, int out_capacity)
    {
        if (P_enc_len > 2048) return -1;
        /* Pre-canonicalize P (apply identity perm). */
        int identity[256];
        for (int i = 0; i < n_universe; i++) identity[i] = i;
        int P_canon[2048];
        int P_canon_len = apply_perm_canonical_c(
            P_enc, P_enc_len, identity, n_universe, P_canon, 2048);
        if (P_canon_len < 0) return -1;

        int candidate[2048];
        int n_stab = 0;
        for (int g = 0; g < n_G; g++) {
            const int* perm = G_perms_flat + (long long)g * n_universe;
            int can_len = apply_perm_canonical_c(
                P_enc, P_enc_len, perm, n_universe, candidate, 2048);
            if (can_len < 0) return -1;
            if (can_len != P_canon_len) continue;
            int match = 1;
            for (int i = 0; i < can_len; i++) {
                if (candidate[i] != P_canon[i]) { match = 0; break; }
            }
            if (match) {
                if (n_stab >= out_capacity) return -1;
                out_stab_indices[n_stab++] = g;
            }
        }
        return n_stab;
    }
""", extra_compile_args=["-O3"])


_lib = None
_lib_lock = threading.Lock()
_ffi = None


def _get_lib():
    global _lib, _ffi
    if _lib is not None:
        return _lib, _ffi
    with _lib_lock:
        if _lib is not None:
            return _lib, _ffi
        try:
            from _tutte_partition_cffi import ffi as cffi_ffi
            from _tutte_partition_cffi import lib
            _lib = lib
            _ffi = cffi_ffi
            return _lib, _ffi
        except ImportError:
            pass
        import sys
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="tutte_partition_c_")
        ffi.compile(tmpdir=tmpdir)
        sys.path.insert(0, tmpdir)
        from _tutte_partition_cffi import ffi as cffi_ffi
        from _tutte_partition_cffi import lib
        _lib = lib
        _ffi = cffi_ffi
        return _lib, _ffi


def _encode_partition(P: Tuple[Tuple[int, ...], ...], pos_to_idx: dict) -> List[int]:
    """Encode partition P as flat int array using pos_to_idx mapping."""
    out = [len(P)]
    for block in P:
        out.append(len(block))
        for v in block:
            out.append(pos_to_idx[v])
    return out


def _decode_partition(data, n: int, idx_to_pos: List[int]) -> Tuple[Tuple[int, ...], ...]:
    """Decode flat int array (cffi int*, length n) into partition tuple."""
    block_count = data[0]
    blocks = []
    idx = 1
    for _ in range(block_count):
        bsize = data[idx]
        idx += 1
        block = tuple(idx_to_pos[data[idx + i]] for i in range(bsize))
        blocks.append(block)
        idx += bsize
    return tuple(blocks)


from .rooted_tutte import restrict_partition as _restrict_partition_py


def _encode_partition_to_idx(P, pos_to_idx):
    """Encode partition with positions already known to be in pos_to_idx."""
    out = [len(P)]
    for block in P:
        out.append(len(block))
        for v in block:
            out.append(pos_to_idx[v])
    return out


def precompute_M_batched_inner_c(
    state_orbit_partitions,           # dict canonical → [rep partition]
    junc_data_per_orbit,              # dict junc_canonical → list of (P_junc, P_junc_S, P_junc_ext)
    state_extra_boundary,             # list[int]
    extra_boundary,                   # list[int]
    shared_boundary,                  # list[int]
    out_boundary,                     # list[int]
    out_cell_anchor_groups,           # list[list[int]] — for canonical key
    n_state_per_orbit,                # dict canonical → n_state (analytical)
    xy_powers_dict,                   # list[dict[(xpow, ypow), int]] indexed by d
):
    """C-batched version of precompute_M_table's inner loop.

    Returns: dict (O_state, O_junc, O_out_canonical_tuple) → coeff dict.
    Returns None if C extension unavailable, sizes exceed limits, or any error.

    For each outer state orbit, batches ALL junction partitions in one C call.
    For Cm3 row composition: 6608 outer × 6608 batched per call → 6608 cffi calls
    + minimal Python overhead per pair. Estimated 10-50x speedup.
    """
    # Build universe = state_extra + shared + extra (matches precompute_M_table).
    full_universe = list(state_extra_boundary) + list(shared_boundary) + list(extra_boundary)
    n_universe = len(full_universe)
    if n_universe > 256:
        return None  # exceeds C buffer
    n_shared = len(shared_boundary)
    n_out_boundary = len(out_boundary)

    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    pos_to_idx = {p: i for i, p in enumerate(full_universe)}
    # Separate mapping for shared_boundary alone — used to encode P_state_S
    # and P_junc_S whose positions live in [0, n_shared) per C's expectation.
    shared_pos_to_idx = {p: i for i, p in enumerate(shared_boundary)}

    # out_boundary indices in [0, n_universe). Some out positions might not be
    # in full_universe (if keep_shared and shared positions are in out but not
    # in extra). For correctness, all out positions must be in full_universe.
    try:
        out_boundary_idx = [pos_to_idx[p] for p in out_boundary]
    except KeyError:
        return None

    # Encode cell_anchor_groups using pos_to_idx. Positions outside universe
    # get index = -1 (treated as "outside" in C).
    n_cells = len(out_cell_anchor_groups)
    cell_groups_data = [n_cells]
    for cell_positions in out_cell_anchor_groups:
        # Filter to positions in universe.
        in_universe = [pos_to_idx[p] for p in cell_positions if p in pos_to_idx]
        cell_groups_data.append(len(in_universe))
        cell_groups_data.extend(in_universe)

    # Pre-encode all junction partitions (once across all outer iterations).
    # Build flat array + offsets for each junction's P_junc_S and P_junc_ext.
    junc_orbit_canonicals = list(junc_data_per_orbit.keys())
    junc_orbit_partition_lists = [junc_data_per_orbit[O_junc] for O_junc in junc_orbit_canonicals]

    # Flatten: total junction partitions = sum over orbits of len(per_junc)
    junc_flat_metadata = []  # list of (orbit_idx, P_junc_idx_in_orbit)
    junc_S_flat = []
    junc_S_offsets = []
    junc_S_lens = []
    junc_ext_flat = []
    junc_ext_offsets = []
    junc_ext_lens = []

    for orbit_idx, per_junc_list in enumerate(junc_orbit_partition_lists):
        for P_junc, P_junc_S, P_junc_ext in per_junc_list:
            junc_flat_metadata.append((orbit_idx, len(junc_flat_metadata)))
            # Encode P_junc_S using shared_pos_to_idx — positions in
            # shared boundary mapped to [0, n_shared).
            try:
                p_S_data = _encode_partition_to_idx(P_junc_S, shared_pos_to_idx)
            except KeyError:
                return None
            junc_S_offsets.append(len(junc_S_flat))
            junc_S_lens.append(len(p_S_data))
            junc_S_flat.extend(p_S_data)
            # Encode P_junc_ext (positions in full_universe)
            try:
                p_ext_data = _encode_partition_to_idx(P_junc_ext, pos_to_idx)
            except KeyError:
                return None
            junc_ext_offsets.append(len(junc_ext_flat))
            junc_ext_lens.append(len(p_ext_data))
            junc_ext_flat.extend(p_ext_data)

    n_junc_total = len(junc_flat_metadata)
    if n_junc_total == 0:
        return {}

    # Allocate cffi arrays for junction data.
    junc_S_arr = _ffi.new("int[]", junc_S_flat)
    junc_S_off_arr = _ffi.new("int[]", junc_S_offsets)
    junc_S_lens_arr = _ffi.new("int[]", junc_S_lens)
    junc_ext_arr = _ffi.new("int[]", junc_ext_flat)
    junc_ext_off_arr = _ffi.new("int[]", junc_ext_offsets)
    junc_ext_lens_arr = _ffi.new("int[]", junc_ext_lens)

    out_boundary_arr = _ffi.new("int[]", out_boundary_idx) if out_boundary_idx else _ffi.new("int[]", 1)
    cell_groups_arr = _ffi.new("int[]", cell_groups_data)

    # Output buffer: per-junction segment is [delta, n_canonical, canonical_data].
    # Worst case: each canonical has n_universe blocks × (n_cells+1) ints + 2.
    per_junc_max = 2 + 1 + n_universe * (n_cells + 1)
    out_capacity = n_junc_total * per_junc_max
    out_buf = _ffi.new("int[]", out_capacity)
    out_offsets = _ffi.new("int[]", n_junc_total + 1)

    # Process each outer state.
    M_dict = {}

    for O_state, ps_list in state_orbit_partitions.items():
        rep_state = ps_list[0]
        n_state = n_state_per_orbit.get(O_state, len(ps_list))

        # Build P_state_S (restrict rep_state to shared_boundary).
        P_state_S = _restrict_partition_py(rep_state, shared_boundary)
        # Build P_state_ext = rep_state + singletons for extra_boundary.
        P_state_ext_list = list(rep_state) + [(v,) for v in extra_boundary]
        P_state_ext = tuple(sorted(P_state_ext_list))

        try:
            p_state_ext_data = _encode_partition_to_idx(P_state_ext, pos_to_idx)
            # P_state_S uses shared_pos_to_idx (in [0, n_shared)).
            p_state_S_data = _encode_partition_to_idx(P_state_S, shared_pos_to_idx)
        except KeyError:
            return None

        p_state_ext_arr = _ffi.new("int[]", p_state_ext_data)
        p_state_S_arr = _ffi.new("int[]", p_state_S_data)

        n_written = lib.batched_inner_iterations_c(
            p_state_ext_arr, len(p_state_ext_data),
            p_state_S_arr, len(p_state_S_data),
            junc_ext_arr, junc_ext_off_arr, junc_ext_lens_arr,
            junc_S_arr, junc_S_off_arr, junc_S_lens_arr,
            n_junc_total,
            n_universe, n_shared,
            out_boundary_arr, n_out_boundary,
            cell_groups_arr, len(cell_groups_data), n_cells,
            out_buf, out_capacity, out_offsets,
        )
        if n_written < 0:
            return None  # capacity exceeded; fall back

        # Decode per-junction results and aggregate. Use buffer protocol
        # to bulk-read cffi int[] into a Python list — avoids 100M+
        # individual cffi attribute accesses.
        out_buf_list = _ffi.unpack(out_buf, n_written) if n_written > 0 else []
        out_offsets_list = _ffi.unpack(out_offsets, n_junc_total + 1)
        shape_size = n_cells + 1
        for j_global in range(n_junc_total):
            orbit_idx, _ = junc_flat_metadata[j_global]
            O_junc = junc_orbit_canonicals[orbit_idx]
            seg_start = out_offsets_list[j_global]
            delta = out_buf_list[seg_start]
            if delta < 0:
                continue
            n_can = out_buf_list[seg_start + 1]
            if n_can == 0:
                O_out = ()
            else:
                n_blocks = out_buf_list[seg_start + 2]
                base0 = seg_start + 3
                # Slice + chunk into block tuples in one Python pass.
                flat = out_buf_list[base0:base0 + n_blocks * shape_size]
                blocks = [tuple(flat[i * shape_size:(i + 1) * shape_size])
                          for i in range(n_blocks)]
                O_out = tuple(sorted(blocks))

            target_key = (O_state, O_junc, O_out)
            if target_key not in M_dict:
                M_dict[target_key] = {}
            target = M_dict[target_key]
            xy_pow = xy_powers_dict[delta]
            for k, v in xy_pow.items():
                target[k] = target.get(k, 0) + v * n_state

    return M_dict


def precompute_M_batched_inner_c_mod(
    state_orbit_partitions,           # dict canonical → [rep partition]
    junc_data_per_orbit,              # dict junc_canonical → list of (P_junc, P_junc_S, P_junc_ext)
    state_extra_boundary,             # list[int]
    extra_boundary,                   # list[int]
    shared_boundary,                  # list[int]
    out_boundary,                     # list[int]
    out_cell_anchor_groups,           # list[list[int]] — for canonical key
    n_state_per_orbit,                # dict canonical → n_state (analytical)
    xy_pow_mod,                       # list[int] — ((x-1)(y-1))^d mod p for d=0..max_d
    p,                                # int — the prime modulus
):
    """Modular variant of `precompute_M_batched_inner_c`.

    Returns: dict (O_state, O_junc, O_out_canonical_tuple) → int mod p.
    Returns None if C extension unavailable, sizes exceed limits, or any error.

    Reuses `batched_inner_iterations_c` for the structural work
    (delta / join / restrict / per-cell canonical key in C). Only the
    final Python aggregation changes: instead of accumulating polynomial
    coefficient dicts per (O_state, O_junc, O_out) bucket, we accumulate
    a single int mod p per bucket. Removes the inner `for k, v in
    xy_pow.items()` loop entirely — one modular mul-add per (state, junc)
    pair.

    Mirrors the polynomial wrapper's behavior exactly except for arithmetic.
    """
    full_universe = list(state_extra_boundary) + list(shared_boundary) + list(extra_boundary)
    n_universe = len(full_universe)
    if n_universe > 256:
        return None
    n_shared = len(shared_boundary)
    n_out_boundary = len(out_boundary)

    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    pos_to_idx = {pos: i for i, pos in enumerate(full_universe)}
    shared_pos_to_idx = {pos: i for i, pos in enumerate(shared_boundary)}

    try:
        out_boundary_idx = [pos_to_idx[pos] for pos in out_boundary]
    except KeyError:
        return None

    n_cells = len(out_cell_anchor_groups)
    cell_groups_data = [n_cells]
    for cell_positions in out_cell_anchor_groups:
        in_universe = [pos_to_idx[pos] for pos in cell_positions if pos in pos_to_idx]
        cell_groups_data.append(len(in_universe))
        cell_groups_data.extend(in_universe)

    junc_orbit_canonicals = list(junc_data_per_orbit.keys())
    junc_orbit_partition_lists = [junc_data_per_orbit[O_junc] for O_junc in junc_orbit_canonicals]

    junc_flat_metadata = []
    junc_S_flat = []
    junc_S_offsets = []
    junc_S_lens = []
    junc_ext_flat = []
    junc_ext_offsets = []
    junc_ext_lens = []

    for orbit_idx, per_junc_list in enumerate(junc_orbit_partition_lists):
        for P_junc, P_junc_S, P_junc_ext in per_junc_list:
            junc_flat_metadata.append((orbit_idx, len(junc_flat_metadata)))
            try:
                p_S_data = _encode_partition_to_idx(P_junc_S, shared_pos_to_idx)
            except KeyError:
                return None
            junc_S_offsets.append(len(junc_S_flat))
            junc_S_lens.append(len(p_S_data))
            junc_S_flat.extend(p_S_data)
            try:
                p_ext_data = _encode_partition_to_idx(P_junc_ext, pos_to_idx)
            except KeyError:
                return None
            junc_ext_offsets.append(len(junc_ext_flat))
            junc_ext_lens.append(len(p_ext_data))
            junc_ext_flat.extend(p_ext_data)

    n_junc_total = len(junc_flat_metadata)
    if n_junc_total == 0:
        return {}

    junc_S_arr = _ffi.new("int[]", junc_S_flat)
    junc_S_off_arr = _ffi.new("int[]", junc_S_offsets)
    junc_S_lens_arr = _ffi.new("int[]", junc_S_lens)
    junc_ext_arr = _ffi.new("int[]", junc_ext_flat)
    junc_ext_off_arr = _ffi.new("int[]", junc_ext_offsets)
    junc_ext_lens_arr = _ffi.new("int[]", junc_ext_lens)

    out_boundary_arr = _ffi.new("int[]", out_boundary_idx) if out_boundary_idx else _ffi.new("int[]", 1)
    cell_groups_arr = _ffi.new("int[]", cell_groups_data)

    per_junc_max = 2 + 1 + n_universe * (n_cells + 1)
    out_capacity = n_junc_total * per_junc_max
    out_buf = _ffi.new("int[]", out_capacity)
    out_offsets = _ffi.new("int[]", n_junc_total + 1)

    M_int = {}

    for O_state, ps_list in state_orbit_partitions.items():
        rep_state = ps_list[0]
        n_state = n_state_per_orbit.get(O_state, len(ps_list))
        n_state_mod = n_state % p
        if n_state_mod == 0:
            continue

        P_state_S = _restrict_partition_py(rep_state, shared_boundary)
        P_state_ext_list = list(rep_state) + [(v,) for v in extra_boundary]
        P_state_ext = tuple(sorted(P_state_ext_list))

        try:
            p_state_ext_data = _encode_partition_to_idx(P_state_ext, pos_to_idx)
            p_state_S_data = _encode_partition_to_idx(P_state_S, shared_pos_to_idx)
        except KeyError:
            return None

        p_state_ext_arr = _ffi.new("int[]", p_state_ext_data)
        p_state_S_arr = _ffi.new("int[]", p_state_S_data)

        n_written = lib.batched_inner_iterations_c(
            p_state_ext_arr, len(p_state_ext_data),
            p_state_S_arr, len(p_state_S_data),
            junc_ext_arr, junc_ext_off_arr, junc_ext_lens_arr,
            junc_S_arr, junc_S_off_arr, junc_S_lens_arr,
            n_junc_total,
            n_universe, n_shared,
            out_boundary_arr, n_out_boundary,
            cell_groups_arr, len(cell_groups_data), n_cells,
            out_buf, out_capacity, out_offsets,
        )
        if n_written < 0:
            return None

        out_buf_list = _ffi.unpack(out_buf, n_written) if n_written > 0 else []
        out_offsets_list = _ffi.unpack(out_offsets, n_junc_total + 1)
        shape_size = n_cells + 1
        for j_global in range(n_junc_total):
            orbit_idx, _ = junc_flat_metadata[j_global]
            O_junc = junc_orbit_canonicals[orbit_idx]
            seg_start = out_offsets_list[j_global]
            delta = out_buf_list[seg_start]
            if delta < 0:
                continue
            n_can = out_buf_list[seg_start + 1]
            if n_can == 0:
                O_out = ()
            else:
                n_blocks = out_buf_list[seg_start + 2]
                base0 = seg_start + 3
                flat = out_buf_list[base0:base0 + n_blocks * shape_size]
                blocks = [tuple(flat[i * shape_size:(i + 1) * shape_size])
                          for i in range(n_blocks)]
                O_out = tuple(sorted(blocks))

            target_key = (O_state, O_junc, O_out)
            add = n_state_mod * xy_pow_mod[delta] % p
            M_int[target_key] = (M_int.get(target_key, 0) + add) % p

    return M_int


def precompute_and_convolve_c_mod(
    state_orbit_partitions,           # dict canonical → [rep partition]
    junc_data_per_orbit,              # dict junc_canonical → list of (P_junc, P_junc_S, P_junc_ext)
    state_extra_boundary,             # list[int]
    extra_boundary,                   # list[int]
    shared_boundary,                  # list[int]
    out_boundary,                     # list[int]
    out_cell_anchor_groups,           # list[list[int]] — for canonical key
    n_state_per_orbit,                # dict canonical → n_state (analytical)
    state_orbit_T_mod,                # dict canonical → int mod p (state value)
    junction_orbit_T_mod,             # dict canonical → int mod p (junction value)
    xy_pow_mod,                       # list[int]
    p,                                # int
):
    """Single-pass C-ext: directly accumulate into out_mod[O_out].

    Combines `precompute_M_batched_inner_c_mod` (build M_int per chunk) +
    streaming-wrapper convolve (M_int × state_T × junc_T → out_mod) into
    one Python loop. Eliminates:
    - 3-tuple `(O_state, O_junc, O_out)` keys (replaced with 1-tuple O_out)
    - Intermediate M_int dict with millions of entries
    - Two dict ops per pair (replaced with one)

    For Cm₃ row composition where per-pair Python dict ops dominate
    wall-clock, this is the primary C-ext speedup.

    Returns: dict O_out_canonical_tuple → int mod p (the per-chunk
    contribution to the final out_mod). Caller accumulates across chunks.
    """
    full_universe = list(state_extra_boundary) + list(shared_boundary) + list(extra_boundary)
    n_universe = len(full_universe)
    if n_universe > 256:
        return None
    n_shared = len(shared_boundary)
    n_out_boundary = len(out_boundary)

    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    pos_to_idx = {pos: i for i, pos in enumerate(full_universe)}
    shared_pos_to_idx = {pos: i for i, pos in enumerate(shared_boundary)}

    try:
        out_boundary_idx = [pos_to_idx[pos] for pos in out_boundary]
    except KeyError:
        return None

    n_cells = len(out_cell_anchor_groups)
    cell_groups_data = [n_cells]
    for cell_positions in out_cell_anchor_groups:
        in_universe = [pos_to_idx[pos] for pos in cell_positions if pos in pos_to_idx]
        cell_groups_data.append(len(in_universe))
        cell_groups_data.extend(in_universe)

    junc_orbit_canonicals = list(junc_data_per_orbit.keys())
    junc_orbit_partition_lists = [junc_data_per_orbit[O_junc] for O_junc in junc_orbit_canonicals]

    junc_flat_metadata = []  # (orbit_idx, _) — but we also need the orbit canonical to lookup jv.
    junc_jv_mod = []  # per-flattened-junc jv mod p value
    junc_S_flat = []
    junc_S_offsets = []
    junc_S_lens = []
    junc_ext_flat = []
    junc_ext_offsets = []
    junc_ext_lens = []

    for orbit_idx, per_junc_list in enumerate(junc_orbit_partition_lists):
        O_junc = junc_orbit_canonicals[orbit_idx]
        jv = junction_orbit_T_mod.get(O_junc, 0)
        if jv == 0:
            # All members of this orbit contribute 0 — skip enumeration.
            for _ in per_junc_list:
                junc_flat_metadata.append((orbit_idx, 0))
                junc_jv_mod.append(0)
                junc_S_offsets.append(len(junc_S_flat))
                junc_S_lens.append(0)
                junc_ext_offsets.append(len(junc_ext_flat))
                junc_ext_lens.append(0)
            continue
        for P_junc, P_junc_S, P_junc_ext in per_junc_list:
            junc_flat_metadata.append((orbit_idx, 0))
            junc_jv_mod.append(jv)
            try:
                p_S_data = _encode_partition_to_idx(P_junc_S, shared_pos_to_idx)
            except KeyError:
                return None
            junc_S_offsets.append(len(junc_S_flat))
            junc_S_lens.append(len(p_S_data))
            junc_S_flat.extend(p_S_data)
            try:
                p_ext_data = _encode_partition_to_idx(P_junc_ext, pos_to_idx)
            except KeyError:
                return None
            junc_ext_offsets.append(len(junc_ext_flat))
            junc_ext_lens.append(len(p_ext_data))
            junc_ext_flat.extend(p_ext_data)

    n_junc_total = len(junc_flat_metadata)
    if n_junc_total == 0:
        return {}

    junc_S_arr = _ffi.new("int[]", junc_S_flat) if junc_S_flat else _ffi.new("int[]", 1)
    junc_S_off_arr = _ffi.new("int[]", junc_S_offsets)
    junc_S_lens_arr = _ffi.new("int[]", junc_S_lens)
    junc_ext_arr = _ffi.new("int[]", junc_ext_flat) if junc_ext_flat else _ffi.new("int[]", 1)
    junc_ext_off_arr = _ffi.new("int[]", junc_ext_offsets)
    junc_ext_lens_arr = _ffi.new("int[]", junc_ext_lens)

    out_boundary_arr = _ffi.new("int[]", out_boundary_idx) if out_boundary_idx else _ffi.new("int[]", 1)
    cell_groups_arr = _ffi.new("int[]", cell_groups_data)

    per_junc_max = 2 + 1 + n_universe * (n_cells + 1)
    out_capacity = n_junc_total * per_junc_max
    out_buf = _ffi.new("int[]", out_capacity)
    out_offsets = _ffi.new("int[]", n_junc_total + 1)

    out_mod = {}

    for O_state, ps_list in state_orbit_partitions.items():
        sv = state_orbit_T_mod.get(O_state, 0)
        if sv == 0:
            continue
        n_state = n_state_per_orbit.get(O_state, len(ps_list))
        n_state_mod = n_state % p
        if n_state_mod == 0:
            continue
        sv_n_state = sv * n_state_mod % p
        if sv_n_state == 0:
            continue

        rep_state = ps_list[0]
        P_state_S = _restrict_partition_py(rep_state, shared_boundary)
        P_state_ext_list = list(rep_state) + [(v,) for v in extra_boundary]
        P_state_ext = tuple(sorted(P_state_ext_list))

        try:
            p_state_ext_data = _encode_partition_to_idx(P_state_ext, pos_to_idx)
            p_state_S_data = _encode_partition_to_idx(P_state_S, shared_pos_to_idx)
        except KeyError:
            return None

        p_state_ext_arr = _ffi.new("int[]", p_state_ext_data)
        p_state_S_arr = _ffi.new("int[]", p_state_S_data)

        n_written = lib.batched_inner_iterations_c(
            p_state_ext_arr, len(p_state_ext_data),
            p_state_S_arr, len(p_state_S_data),
            junc_ext_arr, junc_ext_off_arr, junc_ext_lens_arr,
            junc_S_arr, junc_S_off_arr, junc_S_lens_arr,
            n_junc_total,
            n_universe, n_shared,
            out_boundary_arr, n_out_boundary,
            cell_groups_arr, len(cell_groups_data), n_cells,
            out_buf, out_capacity, out_offsets,
        )
        if n_written < 0:
            return None

        out_buf_list = _ffi.unpack(out_buf, n_written) if n_written > 0 else []
        out_offsets_list = _ffi.unpack(out_offsets, n_junc_total + 1)
        shape_size = n_cells + 1
        # Use bytes as O_out key — much faster construction + hash than
        # nested tuple-of-tuples. The C output is already canonically sorted,
        # so slicing the int run directly yields a stable key.
        for j_global in range(n_junc_total):
            jv = junc_jv_mod[j_global]
            if jv == 0:
                continue
            seg_start = out_offsets_list[j_global]
            delta = out_buf_list[seg_start]
            if delta < 0:
                continue
            n_can = out_buf_list[seg_start + 1]
            if n_can == 0:
                O_out_bytes = b""
            else:
                n_blocks = out_buf_list[seg_start + 2]
                base0 = seg_start + 3
                # bytes(memoryview): O(n) but no per-tuple Python alloc.
                # Each int is 4 bytes (ffi int = C int).
                O_out_bytes = bytes(_ffi.buffer(
                    out_buf + base0, n_blocks * shape_size * 4,
                ))
            contrib = sv_n_state * jv % p * xy_pow_mod[delta] % p
            out_mod[O_out_bytes] = (out_mod.get(O_out_bytes, 0) + contrib) % p

    # Decode bytes keys back to canonical-tuple form for downstream
    # consumers (`per_cell_orbit_size`, `per_cell_orbit_rep` expect
    # `Tuple[Tuple[int, ...], ...]`). One decode per UNIQUE O_out, which
    # is far fewer than per-pair (the whole point of bytes accumulation).
    shape_size = n_cells + 1
    decoded: Dict[Tuple, int] = {}
    for O_out_bytes, val in out_mod.items():
        if not O_out_bytes:
            O_out = ()
        else:
            # Unpack int sequence then group by shape_size.
            n_ints = len(O_out_bytes) // 4
            ints = list(_ffi.unpack(_ffi.cast("int*", _ffi.from_buffer(O_out_bytes)), n_ints))
            n_blocks = n_ints // shape_size
            O_out = tuple(
                tuple(ints[i * shape_size:(i + 1) * shape_size])
                for i in range(n_blocks)
            )
        decoded[O_out] = val
    return decoded


def precompute_and_aggregate_c_mod(
    state_orbit_partitions,
    junc_data_per_orbit,
    state_extra_boundary,
    extra_boundary,
    shared_boundary,
    out_boundary,
    out_cell_anchor_groups,
    n_state_per_orbit,
    state_orbit_T_mod,
    junction_orbit_T_mod,
    xy_pow_mod,
    p,
    hm_capacity_hint: int = 0,
    state_cell_anchor_groups=None,  # enables per-state H-bucketing
):
    """C-side hash-map aggregation drop-in for `precompute_and_convolve_c_mod`.

    Same inputs/outputs as `precompute_and_convolve_c_mod` but the
    per-junction inner aggregation loop runs in C against a
    Python-allocated hash map that persists across state orbits.
    The Python wrapper does no per-pair dict ops — only one C call
    per state orbit (batched_inner_iterations_c + aggregate_buf_to_hashmap_c)
    plus a single final marshal pass.

    Per `tutte/research/plans/cm3_unlock_design.md`: projected 5-8×
    speedup on Cm₃ modular point (75 min → 9-15 min) by eliminating
    ~27 M Python dict updates per point.

    Validates against `precompute_and_convolve_c_mod` on Cm₂; should
    match bit-for-bit. Falls back to None (caller routes to the Python
    path) on C-ext or capacity errors.

    `hm_capacity_hint`: if >0, used as initial hashmap slot count.
    Otherwise sized from junction count × 2 (conservative).

    `state_cell_anchor_groups`: when provided,
    enables per-state-orbit H-bucketing of junction members. For each
    state orbit, compute H = stab_G(rep_state) on shared boundary,
    then group junction members by `h_canonicalize_c_batched`. Only
    one rep per H-bucket is passed to `batched_inner_iterations_c`,
    with the bucket size baked into `jv_mod` as a multiplier. Reduces
    Cm_3 pair-iterations from ~27M → ~2.6M (projected 2-5× end-to-end
    after Python overhead).

    Bit-for-bit invariance: with state_cell_anchor_groups=None
    (default), behavior is identical to the pair-aggregation path. With it
    provided, the new path is mathematically equivalent (H-orbit
    coherence: f(state, J_1) = f(state, J_2) when J_1, J_2 ∈ same
    H-orbit, so summing |bucket| × f(state, rep) ≡ summing f over
    bucket members).
    """
    import os as _os, time as _t
    _DBG = _os.environ.get("TUTTE_DEBUG_AGG", "0") != "0"
    _t_entry = _t.time()
    full_universe = list(state_extra_boundary) + list(shared_boundary) + list(extra_boundary)
    n_universe = len(full_universe)
    if n_universe > 256:
        return None
    n_shared = len(shared_boundary)
    n_out_boundary = len(out_boundary)
    if _DBG:
        n_junc_orbits_dbg = len(junc_data_per_orbit)
        n_junc_members_dbg = sum(len(pl) for pl in junc_data_per_orbit.values())
        print(f"  [agg] entry: n_states={len(state_orbit_partitions)} n_juncs={n_junc_orbits_dbg} members={n_junc_members_dbg} n_universe={n_universe} n_shared={n_shared} scag={state_cell_anchor_groups is not None}", flush=True)

    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    pos_to_idx = {pos: i for i, pos in enumerate(full_universe)}
    shared_pos_to_idx = {pos: i for i, pos in enumerate(shared_boundary)}

    try:
        out_boundary_idx = [pos_to_idx[pos] for pos in out_boundary]
    except KeyError:
        return None

    n_cells = len(out_cell_anchor_groups)
    cell_groups_data = [n_cells]
    for cell_positions in out_cell_anchor_groups:
        in_universe = [pos_to_idx[pos] for pos in cell_positions if pos in pos_to_idx]
        cell_groups_data.append(len(in_universe))
        cell_groups_data.extend(in_universe)

    # R19 H-bucketing setup. When state_cell_anchor_groups is provided,
    # we compute the per-cell aut group G (acting on shared boundary
    # positions per state_cell_anchor_groups). Inside the state loop we
    # compute H = stab_G(rep_state) — junctions in the same H-orbit
    # contribute identically (per per_cell_partition_stab semantics).
    #
    # Bit-for-bit invariance vs the pre-R19 chunk-global path: when |H| = 1
    # the per-state loop uses the chunk-global arrays (no bucketing).
    # When |H| > 1, the per-state path passes only one rep per H-bucket
    # to batched_inner_iterations_c with bucket_size baked into jv_mod.
    # Mathematically: f(state, J_1) = f(state, J_2) when J_1, J_2 ∈
    # same H-orbit, so Σ_{J ∈ bucket} f(state, J) ≡ |bucket| × f(state, rep).
    _r19_use = False
    _G_elements: List[Dict[int, int]] = []
    _shared_set: set = set()
    if state_cell_anchor_groups is not None:
        try:
            from .aut_orbit import (
                apply_perm_to_partition as _apply_perm,
                enumerate_per_cell_aut_group as _enum_aut,
                per_cell_partition_stab as _stab,
            )
            # Restrict cell groups to shared boundary positions. G acts on
            # shared boundary; H bucketing tests junction equivalence under
            # actions that fix the state's shared-boundary projection.
            _shared_set = set(shared_boundary)
            _shared_cell_groups = [
                [p for p in grp if p in _shared_set]
                for grp in state_cell_anchor_groups
            ]
            _shared_cell_groups = [g for g in _shared_cell_groups if g]
            if _shared_cell_groups:
                _G_elements = _enum_aut(_shared_cell_groups)
                if len(_G_elements) > 1:
                    _r19_use = True
        except Exception:
            _r19_use = False

    # R19 gating: only activate when total bucketing cost is amortized.
    # Note that with the batched C-ext h_canonicalize_batch_c the per-state
    # Python overhead is small, so the gate is mostly a safety net.
    #
    # IMPORTANT (May 26 2026 measurement): R19 H-bucketing is a net REGRESSION
    # on both Cm_2 (~21s vs 12s baseline) and Cm_3 (~21s/chunk vs 1.1s/chunk
    # baseline). The C-ext `batched_inner_iterations_c` is already extremely
    # fast (~2.5ms per state for 4096 junction members at Cm_3 scale), so the
    # H-canonicalize overhead in R19 always exceeds the savings.
    #
    # Therefore R19 is now DEFAULT-DISABLED (`TUTTE_R19_ENABLE=0` by default).
    # Set TUTTE_R19_ENABLE=1 to opt in for graphs where the inner loop is
    # genuinely costly (e.g., polynomial-mode pair-orbit DP, not the
    # modular int path).
    if _r19_use:
        import os
        if os.environ.get("TUTTE_R19_ENABLE", "0") == "0":
            _r19_use = False
    if _r19_use:
        import os
        _r19_gate_threshold = int(os.environ.get("TUTTE_R19_MIN_WORK", "100000"))
        # Cheap estimator of bucketing work: state_orbits × G × junc_members
        n_state_orbits_active = sum(
            1 for sv in state_orbit_T_mod.values() if sv
        )
        n_junc_members = sum(
            len(per_junc_list) for per_junc_list in junc_data_per_orbit.values()
        )
        est_work = n_state_orbits_active * max(len(_G_elements), 1) * n_junc_members
        if est_work < _r19_gate_threshold:
            _r19_use = False

    junc_orbit_canonicals = list(junc_data_per_orbit.keys())
    junc_orbit_partition_lists = [junc_data_per_orbit[O_junc] for O_junc in junc_orbit_canonicals]

    junc_jv_mod_list = []
    junc_S_flat = []
    junc_S_offsets = []
    junc_S_lens = []
    junc_ext_flat = []
    junc_ext_offsets = []
    junc_ext_lens = []

    for orbit_idx, per_junc_list in enumerate(junc_orbit_partition_lists):
        O_junc = junc_orbit_canonicals[orbit_idx]
        jv = junction_orbit_T_mod.get(O_junc, 0)
        if jv == 0:
            for _ in per_junc_list:
                junc_jv_mod_list.append(0)
                junc_S_offsets.append(len(junc_S_flat))
                junc_S_lens.append(0)
                junc_ext_offsets.append(len(junc_ext_flat))
                junc_ext_lens.append(0)
            continue
        for P_junc, P_junc_S, P_junc_ext in per_junc_list:
            junc_jv_mod_list.append(jv)
            try:
                p_S_data = _encode_partition_to_idx(P_junc_S, shared_pos_to_idx)
            except KeyError:
                return None
            junc_S_offsets.append(len(junc_S_flat))
            junc_S_lens.append(len(p_S_data))
            junc_S_flat.extend(p_S_data)
            try:
                p_ext_data = _encode_partition_to_idx(P_junc_ext, pos_to_idx)
            except KeyError:
                return None
            junc_ext_offsets.append(len(junc_ext_flat))
            junc_ext_lens.append(len(p_ext_data))
            junc_ext_flat.extend(p_ext_data)

    n_junc_total = len(junc_jv_mod_list)
    if n_junc_total == 0:
        return {}

    junc_S_arr = _ffi.new("int[]", junc_S_flat) if junc_S_flat else _ffi.new("int[]", 1)
    junc_S_off_arr = _ffi.new("int[]", junc_S_offsets)
    junc_S_lens_arr = _ffi.new("int[]", junc_S_lens)
    junc_ext_arr = _ffi.new("int[]", junc_ext_flat) if junc_ext_flat else _ffi.new("int[]", 1)
    junc_ext_off_arr = _ffi.new("int[]", junc_ext_offsets)
    junc_ext_lens_arr = _ffi.new("int[]", junc_ext_lens)
    junc_jv_arr = _ffi.new("long long[]", junc_jv_mod_list)

    # R19: pre-flatten ALL P_junc encodings (full universe = shared+extra)
    # ONCE so per-state batched canonicalization can use them. Universe for
    # H-canonicalization is the FULL partition universe (shared+extra_boundary
    # positions appearing in any P_junc). H acts as identity on positions
    # outside shared_boundary, so they pass through unchanged.
    _r19_P_junc_flat: List[int] = []
    _r19_P_junc_offsets: List[int] = [0]
    _r19_jv_per_member: List[int] = []  # parallel to canonicalize-batch output
    _r19_orbit_idx_per_member: List[int] = []
    _r19_member_in_orbit: List[int] = []  # index of member within its orbit's per_junc_list
    _r19_universe_pos: List[int] = []  # universe positions (shared first, then extra unique)
    _r19_h_universe_idx: Dict[int, int] = {}  # pos → index in universe (for H encoding)
    _r19_canon_capacity = 0  # max single-canonical length (for output buffer sizing)
    if _r19_use:
        try:
            # Build full universe for P_junc encoding: shared_boundary positions
            # first (these are the ones that H permutes), then extra positions
            # discovered in P_junc partitions.
            _r19_universe_pos = list(shared_boundary)
            _r19_h_universe_idx = {pos: i for i, pos in enumerate(_r19_universe_pos)}
            for orbit_idx, per_junc_list in enumerate(junc_orbit_partition_lists):
                O_junc = junc_orbit_canonicals[orbit_idx]
                jv = junction_orbit_T_mod.get(O_junc, 0)
                if jv == 0:
                    continue
                for member_idx, (P_junc, _ps, _pe) in enumerate(per_junc_list):
                    for block in P_junc:
                        for v in block:
                            if v not in _r19_h_universe_idx:
                                _r19_h_universe_idx[v] = len(_r19_universe_pos)
                                _r19_universe_pos.append(v)
            _r19_n_univ = len(_r19_universe_pos)
            # Encode each P_junc against full universe.
            for orbit_idx, per_junc_list in enumerate(junc_orbit_partition_lists):
                O_junc = junc_orbit_canonicals[orbit_idx]
                jv = junction_orbit_T_mod.get(O_junc, 0)
                if jv == 0:
                    continue
                for member_idx, (P_junc, _ps, _pe) in enumerate(per_junc_list):
                    # Encode P_junc using h_universe_idx mapping.
                    enc = [len(P_junc)]
                    for block in P_junc:
                        enc.append(len(block))
                        for v in block:
                            enc.append(_r19_h_universe_idx[v])
                    _r19_P_junc_flat.extend(enc)
                    _r19_P_junc_offsets.append(len(_r19_P_junc_flat))
                    _r19_jv_per_member.append(jv)
                    _r19_orbit_idx_per_member.append(orbit_idx)
                    _r19_member_in_orbit.append(member_idx)
                    if len(enc) > _r19_canon_capacity:
                        _r19_canon_capacity = len(enc)
            _r19_n_members = len(_r19_jv_per_member)
            if _r19_n_members == 0 or _r19_n_univ > 256:
                _r19_use = False
            else:
                _r19_P_flat_arr = _ffi.new("int[]", _r19_P_junc_flat)
                _r19_P_off_arr = _ffi.new("int[]", _r19_P_junc_offsets)
                # Output capacity for canonicals: sum of per-member upper bound.
                _r19_canon_out_cap = sum(
                    _r19_P_junc_offsets[i + 1] - _r19_P_junc_offsets[i]
                    for i in range(_r19_n_members)
                )
                _r19_canon_out_arr = _ffi.new("int[]", max(_r19_canon_out_cap, 1))
                _r19_canon_out_off_arr = _ffi.new("int[]", _r19_n_members + 1)

                # Pre-flatten G_elements as C array for set_stabilizer_c.
                # Each perm maps shared_boundary positions → permuted positions.
                # Encode as length n_G * n_universe (positions outside
                # shared_boundary map to themselves).
                _r19_G_flat: List[int] = []
                _r19_G_bail = False
                for g_perm in _G_elements:
                    for pos in _r19_universe_pos:
                        tgt = g_perm.get(pos, pos)
                        idx = _r19_h_universe_idx.get(tgt)
                        if idx is None:
                            _r19_G_bail = True
                            break
                        _r19_G_flat.append(idx)
                    if _r19_G_bail:
                        break
                if _r19_G_bail:
                    _r19_use_c_stab = False
                else:
                    _r19_G_flat_arr = _ffi.new("int[]", _r19_G_flat)
                    _r19_stab_out_arr = _ffi.new("int[]", max(len(_G_elements), 1))
                    _r19_use_c_stab = True
        except Exception:
            _r19_use = False

    out_boundary_arr = _ffi.new("int[]", out_boundary_idx) if out_boundary_idx else _ffi.new("int[]", 1)
    cell_groups_arr = _ffi.new("int[]", cell_groups_data)

    xy_pow_arr = _ffi.new("long long[]", list(xy_pow_mod))

    per_junc_max = 3 + n_universe * (n_cells + 1)
    out_capacity = n_junc_total * per_junc_max
    out_buf = _ffi.new("int[]", out_capacity)
    out_offsets = _ffi.new("int[]", n_junc_total + 1)

    # Hashmap sizing — caller's state-orbit count drives unique-key bound.
    # Defensive default: 4× n_state_orbits to keep collisions low.
    n_state_orbits = sum(1 for sv in state_orbit_T_mod.values() if sv)
    hm_capacity = max(hm_capacity_hint, 4 * max(n_state_orbits, 1), 1024)
    # Each key is up to n_universe × (n_cells+1) ints. Bound conservatively.
    max_key_ints = n_universe * (n_cells + 1)
    hm_keys_cap = hm_capacity * max(max_key_ints, 1)

    hm_used = _ffi.new("int[]", hm_capacity)
    hm_key_off = _ffi.new("int[]", hm_capacity)
    hm_key_len = _ffi.new("int[]", hm_capacity)
    hm_values = _ffi.new("long long[]", hm_capacity)
    hm_keys_buffer = _ffi.new("int[]", hm_keys_cap)
    hm_keys_used = _ffi.new("int*")
    hm_n_unique = _ffi.new("int*")
    hm_keys_used[0] = 0
    hm_n_unique[0] = 0

    shape_size = n_cells + 1

    if _DBG:
        _t_pre_loop = _t.time()
        print(f"  [agg] pre-loop done: {_t_pre_loop-_t_entry:.2f}s, r19={_r19_use}, n_G={len(_G_elements)}, n_members={len(_r19_jv_per_member) if _r19_use else 0}", flush=True)
    _state_loop_t0 = _t.time()
    _n_processed = 0
    _n_r19_used = 0
    _state_loop_progress_interval = 50

    for O_state, ps_list in state_orbit_partitions.items():
        sv = state_orbit_T_mod.get(O_state, 0)
        if sv == 0:
            continue
        n_state = n_state_per_orbit.get(O_state, len(ps_list))
        n_state_mod = n_state % p
        if n_state_mod == 0:
            continue
        sv_n_state = sv * n_state_mod % p
        if sv_n_state == 0:
            continue

        rep_state = ps_list[0]
        P_state_S = _restrict_partition_py(rep_state, shared_boundary)
        P_state_ext_list = list(rep_state) + [(v,) for v in extra_boundary]
        P_state_ext = tuple(sorted(P_state_ext_list))

        try:
            p_state_ext_data = _encode_partition_to_idx(P_state_ext, pos_to_idx)
            p_state_S_data = _encode_partition_to_idx(P_state_S, shared_pos_to_idx)
        except KeyError:
            return None

        p_state_ext_arr = _ffi.new("int[]", p_state_ext_data)
        p_state_S_arr = _ffi.new("int[]", p_state_S_data)

        # R19: per-state H-bucketing. When _r19_use, compute H stabilizer
        # of rep_state restricted to shared boundary, then group junction
        # members in same H-orbit. Use one rep per bucket with bucket_size
        # baked into jv_mod (mod p multiplier). Mathematically equivalent
        # to summing f(state, J) over all J in the bucket.
        #
        # Implementation: uses batched C-ext `h_canonicalize_batch_c` to
        # avoid per-member Python `apply_perm_to_partition` overhead which
        # dominated the naive R19 wiring (~30s for Cm_2 vs ~5s baseline).
        _per_state_arrays = None
        if _r19_use:
            # C-side set-stabilizer: encode P_state_S on _r19_universe_pos and
            # call set_stabilizer_c, which returns indices into _G_elements.
            # H_arr is then a sub-array of _r19_G_flat_arr indexed by those.
            if _r19_use_c_stab:
                # Encode P_state_S on _r19_h_universe_idx (only shared positions
                # appear; others would not appear in shared-boundary partition).
                try:
                    P_state_S_enc = [len(P_state_S)]
                    for block in P_state_S:
                        P_state_S_enc.append(len(block))
                        for v in block:
                            idx = _r19_h_universe_idx.get(v)
                            if idx is None:
                                raise KeyError(v)
                            P_state_S_enc.append(idx)
                    P_state_S_arr_for_stab = _ffi.new("int[]", P_state_S_enc)
                    n_stab = lib.set_stabilizer_c(
                        P_state_S_arr_for_stab, len(P_state_S_enc),
                        _r19_G_flat_arr, len(_G_elements), _r19_n_univ,
                        _r19_stab_out_arr, len(_G_elements),
                    )
                except (KeyError, Exception):
                    n_stab = -1
            else:
                n_stab = -1

            if n_stab < 0:
                # Fall back to Python stab.
                H = _stab(P_state_S, _G_elements)
                n_H_effective = len(H)
                H_flat: List[int] = []
                bail_h = False
                for perm in H:
                    for pos in _r19_universe_pos:
                        tgt = perm.get(pos, pos)
                        idx = _r19_h_universe_idx.get(tgt)
                        if idx is None:
                            bail_h = True
                            break
                        H_flat.append(idx)
                    if bail_h:
                        break
                if not bail_h and n_H_effective > 1:
                    H_arr = _ffi.new("int[]", H_flat)
                else:
                    H_arr = None
                    n_H_effective = 0
            else:
                # Build H flat array from selected G perms (slice _r19_G_flat).
                n_H_effective = n_stab
                if n_stab > 1:
                    H_flat = []
                    for i in range(n_stab):
                        g_idx = _r19_stab_out_arr[i]
                        H_flat.extend(_r19_G_flat[g_idx * _r19_n_univ:(g_idx + 1) * _r19_n_univ])
                    H_arr = _ffi.new("int[]", H_flat)
                else:
                    H_arr = None  # |H| = 1 ⇒ no bucketing

            # Per-state |H| gate: if |H| is too large, R19 canonicalize cost
            # exceeds savings. For Cm_3 with G=13824, |H| can be in thousands
            # — at which point processing all members directly is faster.
            # Threshold: |H| × n_members × avg_canon_cost < threshold.
            # avg_canon_cost ~ universe_size = 12-24.
            _r19_h_max = int(_os.environ.get("TUTTE_R19_MAX_H", "200"))
            if n_H_effective > _r19_h_max:
                if _DBG and _n_processed < 5:
                    print(f"  [agg] skip R19 for state {_n_processed}: |H|={n_H_effective} > {_r19_h_max}", flush=True)
                n_H_effective = 0
                H_arr = None

            if n_H_effective > 1 and H_arr is not None:
                # Run batched canonicalize over ALL members.
                n_written = lib.h_canonicalize_batch_c(
                    _r19_P_flat_arr, _r19_P_off_arr, _r19_n_members,
                    H_arr, n_H_effective, _r19_n_univ,
                    _r19_canon_out_arr, _r19_canon_out_off_arr,
                    _r19_canon_out_cap,
                )
                if n_written >= 0:
                    # Bucket members by canonical bytes.
                    # canon_bytes per member: slice _r19_canon_out_arr by
                    # _r19_canon_out_off_arr[i:i+1].
                    buckets: Dict[bytes, Tuple[int, int]] = {}
                    for m_idx in range(_r19_n_members):
                        c_start = _r19_canon_out_off_arr[m_idx]
                        c_end = _r19_canon_out_off_arr[m_idx + 1]
                        # Get bytes (use buffer slice for hashable key).
                        canon_bytes = bytes(
                            _ffi.buffer(
                                _r19_canon_out_arr + c_start,
                                (c_end - c_start) * _ffi.sizeof("int")
                            )
                        )
                        existing = buckets.get(canon_bytes)
                        if existing is None:
                            buckets[canon_bytes] = (1, m_idx)
                        else:
                            cnt, idx = existing
                            buckets[canon_bytes] = (cnt + 1, idx)

                    # Build per-state arrays from buckets.
                    ps_jv: List[int] = []
                    ps_S_flat: List[int] = []
                    ps_S_off: List[int] = []
                    ps_S_lens: List[int] = []
                    ps_ext_flat: List[int] = []
                    ps_ext_off: List[int] = []
                    ps_ext_lens: List[int] = []
                    bail_r19 = False
                    for _canon_bytes, (bsize, rep_m_idx) in buckets.items():
                        orbit_idx = _r19_orbit_idx_per_member[rep_m_idx]
                        member_idx = _r19_member_in_orbit[rep_m_idx]
                        per_junc_list = junc_orbit_partition_lists[orbit_idx]
                        _P_junc, P_junc_S, P_junc_ext = per_junc_list[member_idx]
                        jv = _r19_jv_per_member[rep_m_idx]
                        jv_scaled = (jv * bsize) % p
                        if jv_scaled == 0:
                            continue
                        ps_jv.append(jv_scaled)
                        try:
                            p_S_data = _encode_partition_to_idx(P_junc_S, shared_pos_to_idx)
                            p_ext_data = _encode_partition_to_idx(P_junc_ext, pos_to_idx)
                        except KeyError:
                            bail_r19 = True
                            break
                        ps_S_off.append(len(ps_S_flat))
                        ps_S_lens.append(len(p_S_data))
                        ps_S_flat.extend(p_S_data)
                        ps_ext_off.append(len(ps_ext_flat))
                        ps_ext_lens.append(len(p_ext_data))
                        ps_ext_flat.extend(p_ext_data)
                    if not bail_r19 and ps_jv:
                        _per_state_arrays = (ps_jv, ps_S_flat, ps_S_off, ps_S_lens,
                                              ps_ext_flat, ps_ext_off, ps_ext_lens)

        if _per_state_arrays is not None:
            (ps_jv, ps_S_flat, ps_S_off, ps_S_lens,
             ps_ext_flat, ps_ext_off, ps_ext_lens) = _per_state_arrays
            n_junc_state = len(ps_jv)
            ps_S_arr = _ffi.new("int[]", ps_S_flat) if ps_S_flat else _ffi.new("int[]", 1)
            ps_S_off_arr = _ffi.new("int[]", ps_S_off)
            ps_S_lens_arr = _ffi.new("int[]", ps_S_lens)
            ps_ext_arr = _ffi.new("int[]", ps_ext_flat) if ps_ext_flat else _ffi.new("int[]", 1)
            ps_ext_off_arr = _ffi.new("int[]", ps_ext_off)
            ps_ext_lens_arr = _ffi.new("int[]", ps_ext_lens)
            ps_jv_arr = _ffi.new("long long[]", ps_jv)
            ps_out_capacity = n_junc_state * per_junc_max
            ps_out_buf = _ffi.new("int[]", ps_out_capacity)
            ps_out_offsets = _ffi.new("int[]", n_junc_state + 1)

            n_written = lib.batched_inner_iterations_c(
                p_state_ext_arr, len(p_state_ext_data),
                p_state_S_arr, len(p_state_S_data),
                ps_ext_arr, ps_ext_off_arr, ps_ext_lens_arr,
                ps_S_arr, ps_S_off_arr, ps_S_lens_arr,
                n_junc_state,
                n_universe, n_shared,
                out_boundary_arr, n_out_boundary,
                cell_groups_arr, len(cell_groups_data), n_cells,
                ps_out_buf, ps_out_capacity, ps_out_offsets,
            )
            if n_written < 0:
                return None
            rc = lib.aggregate_buf_to_hashmap_c(
                ps_out_buf, ps_out_offsets, n_junc_state,
                shape_size,
                ps_jv_arr, sv_n_state,
                xy_pow_arr, p,
                hm_used, hm_key_off, hm_key_len, hm_values, hm_capacity,
                hm_keys_buffer, hm_keys_cap, hm_keys_used, hm_n_unique,
            )
            if rc < 0:
                return None
        else:
            n_written = lib.batched_inner_iterations_c(
                p_state_ext_arr, len(p_state_ext_data),
                p_state_S_arr, len(p_state_S_data),
                junc_ext_arr, junc_ext_off_arr, junc_ext_lens_arr,
                junc_S_arr, junc_S_off_arr, junc_S_lens_arr,
                n_junc_total,
                n_universe, n_shared,
                out_boundary_arr, n_out_boundary,
                cell_groups_arr, len(cell_groups_data), n_cells,
                out_buf, out_capacity, out_offsets,
            )
            if n_written < 0:
                return None

            rc = lib.aggregate_buf_to_hashmap_c(
                out_buf, out_offsets, n_junc_total,
                shape_size,
                junc_jv_arr, sv_n_state,
                xy_pow_arr, p,
                hm_used, hm_key_off, hm_key_len, hm_values, hm_capacity,
                hm_keys_buffer, hm_keys_cap, hm_keys_used, hm_n_unique,
            )
            if rc < 0:
                return None

        _n_processed += 1
        if _per_state_arrays is not None:
            _n_r19_used += 1
        if _DBG and _n_processed % _state_loop_progress_interval == 0:
            _elapsed = _t.time() - _state_loop_t0
            print(f"  [agg] {_n_processed} states done, r19_used={_n_r19_used}, {_elapsed:.1f}s elapsed", flush=True)

    if _DBG:
        print(f"  [agg] state loop done: {_n_processed} states, r19_used={_n_r19_used}, {_t.time()-_state_loop_t0:.1f}s", flush=True)

    # Marshal hashmap → flat output via C-side dump.
    n_unique = hm_n_unique[0]
    if n_unique == 0:
        return {}
    out_keys_flat = _ffi.new("int[]", hm_keys_used[0])
    out_keys_offsets = _ffi.new("int[]", n_unique)
    out_key_lens = _ffi.new("int[]", n_unique)
    out_values = _ffi.new("long long[]", n_unique)
    n_marshaled = lib.hashmap_marshal_c(
        hm_used, hm_key_off, hm_key_len, hm_values, hm_capacity,
        hm_keys_buffer,
        out_keys_flat, out_keys_offsets, out_key_lens,
        out_values, hm_keys_used[0],
    )
    if n_marshaled != n_unique:
        return None

    # Decode bytes keys to canonical tuples (same format as
    # precompute_and_convolve_c_mod's output).
    decoded: Dict[Tuple, int] = {}
    for k in range(n_unique):
        key_off = out_keys_offsets[k]
        key_len = out_key_lens[k]
        val = int(out_values[k])
        if key_len == 0:
            O_out = ()
        else:
            n_blocks = key_len // shape_size
            O_out = tuple(
                tuple(int(out_keys_flat[key_off + i * shape_size + j])
                      for j in range(shape_size))
                for i in range(n_blocks)
            )
        decoded[O_out] = val
    return decoded


def join_partitions_c_wrapper(
    P1: Tuple[Tuple[int, ...], ...],
    P2: Tuple[Tuple[int, ...], ...],
    universe: List[int],
) -> Optional[Tuple[Tuple[int, ...], ...]]:
    """Drop-in C-extension replacement for `join_partitions`.

    Returns None if C extension unavailable or universe too large
    (caller falls back to pure Python).
    """
    n_univ = len(universe)
    if n_univ == 0:
        return tuple()
    if n_univ > 256:
        return None  # exceeds C buffer; fall back

    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    pos_to_idx = {p: i for i, p in enumerate(universe)}
    idx_to_pos = list(universe)

    p1_data = _encode_partition(P1, pos_to_idx)
    p2_data = _encode_partition(P2, pos_to_idx)

    p1_arr = _ffi.new("int[]", p1_data)
    p2_arr = _ffi.new("int[]", p2_data)

    # Output capacity: worst case = each vertex in own block = 1 + 2*n.
    # But with sizes and counts, allow generous slack.
    out_capacity = 2 * n_univ + 16
    out_arr = _ffi.new("int[]", out_capacity)

    n_out = lib.join_partitions_c(
        p1_arr, len(p1_data),
        p2_arr, len(p2_data),
        n_univ,
        out_arr, out_capacity,
    )
    if n_out < 0:
        return None  # overflow — fall back

    return _decode_partition(out_arr, n_out, idx_to_pos)


# =============================================================================
# H-canonicalize a partition under a permutation group (C-ext)
# =============================================================================


def h_canonicalize_c_wrapper(
    P: Tuple[Tuple[int, ...], ...],
    H_perms: List[Dict[int, int]],
    universe: List[int],
) -> Optional[Tuple[Tuple[int, ...], ...]]:
    """C-extension lex-min over H-permutations of partition P.

    Returns the canonical form (sorted blocks, sorted within block)
    that lexicographically minimizes over all permutations in H, or
    None on C-ext unavailable / overflow.

    Mirrors the pair-orbit H-canonicalization inner loop:
    `h_canon = min(apply_perm_to_partition(P, h) for h in H)`.

    Args:
        P: encoded partition as nested tuple of vertex tuples.
        H_perms: list of dicts mapping position → image. All perms
                 must agree on the universe.
        universe: list of positions covered by H_perms (the support
                  of all perms).

    Returns:
        Canonical-form partition as nested tuple, or None on overflow.
    """
    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    n_univ = len(universe)
    if n_univ > 256:
        return None
    if not P:
        return ()

    pos_to_idx = {pos: i for i, pos in enumerate(universe)}
    idx_to_pos = list(universe)

    # Encode P using pos_to_idx.
    try:
        p_data = _encode_partition(P, pos_to_idx)
    except KeyError:
        return None

    # Encode H_perms: flat array of length n_H * n_univ.
    n_H = len(H_perms)
    if n_H == 0:
        # No permutations — canonicalize P alone (apply identity).
        H_flat = list(range(n_univ))
        n_H = 1
    else:
        H_flat = []
        for perm in H_perms:
            for pos in universe:
                tgt = perm.get(pos, pos)
                if tgt not in pos_to_idx:
                    return None
                H_flat.append(pos_to_idx[tgt])

    p_arr = _ffi.new("int[]", p_data)
    H_arr = _ffi.new("int[]", H_flat)

    out_capacity = max(2 * len(p_data), 32)
    out_arr = _ffi.new("int[]", out_capacity)

    n_out = lib.h_canonicalize_c(
        p_arr, len(p_data),
        H_arr, n_H, n_univ,
        out_arr, out_capacity,
    )
    if n_out < 0:
        return None

    return _decode_partition(out_arr, n_out, idx_to_pos)


def h_canonicalize_c_batched(
    P_list: List[Tuple[Tuple[int, ...], ...]],
    H_perms: List[Dict[int, int]],
    universe: List[int],
) -> Optional[List[Tuple[Tuple[int, ...], ...]]]:
    """Batched H-canonicalize: marshal H ONCE, canonicalize many P's.

    For the pair-orbit H-canonicalization inner loop, the per-cell-pair
    pattern is to canonicalize `len(members)` partitions (~thousands)
    against the SAME H. Per-call marshaling of H in
    `h_canonicalize_c_wrapper` costs ~5ms × n_H × n_univ which dominates;
    this batched variant pre-allocates H_arr once and reuses for all P's.

    Returns list of canonical-form partitions (same length as P_list)
    or None on C-ext unavailable / overflow.
    """
    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    n_univ = len(universe)
    if n_univ > 256:
        return None
    if not P_list:
        return []

    pos_to_idx = {pos: i for i, pos in enumerate(universe)}
    idx_to_pos = list(universe)

    # Marshal H once.
    n_H = len(H_perms)
    if n_H == 0:
        H_flat = list(range(n_univ))
        n_H = 1
    else:
        H_flat = []
        for perm in H_perms:
            for pos in universe:
                tgt = perm.get(pos, pos)
                if tgt not in pos_to_idx:
                    return None
                H_flat.append(pos_to_idx[tgt])
    H_arr = _ffi.new("int[]", H_flat)

    # Per-partition reusable output buffer.
    max_p_data = max(
        (2 * (1 + sum(1 + len(b) for b in P)) for P in P_list), default=32
    )
    out_capacity = max(max_p_data, 32)
    out_arr = _ffi.new("int[]", out_capacity)

    results = []
    for P in P_list:
        try:
            p_data = _encode_partition(P, pos_to_idx)
        except KeyError:
            return None
        p_arr = _ffi.new("int[]", p_data)
        n_out = lib.h_canonicalize_c(
            p_arr, len(p_data),
            H_arr, n_H, n_univ,
            out_arr, out_capacity,
        )
        if n_out < 0:
            return None
        results.append(_decode_partition(out_arr, n_out, idx_to_pos))
    return results
