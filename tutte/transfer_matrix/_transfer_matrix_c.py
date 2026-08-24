"""C extension for transfer matrix Tutte polynomial — full sweep via cffi.

The sweep runs entirely in C: partition enumeration, transfer matrix build,
initial vector, matrix-vector multiply loop, forget + sum + exponent shift.
Returns the polynomial in (a,b) = (x-1, y-1) basis. The (a,b) -> (x,y)
binomial conversion is done in Python to avoid integer overflow in large
binomial coefficients.

When exact int64 arithmetic overflows, the Python wrapper falls back to
modular arithmetic with CRT reconstruction.

Note: The C code uses global mutable state for partition tables and the
overflow flag. This is safe under the Python GIL but not thread-safe for
concurrent calls from multiple threads without the GIL.
"""

import cffi

ffi = cffi.FFI()

ffi.cdef("""
    int transfer_matrix_sweep_c(
        int width, int length,
        const int* unit_cell_edges, int num_edges,
        int num_vertices,
        int* out_ab,
        long long* out_coeffs,
        int* out_n_terms,
        int max_out);

    int transfer_matrix_sweep_modp_c(
        int width, int length,
        const int* unit_cell_edges, int num_edges,
        int num_vertices,
        long long prime,
        int* out_ab,
        long long* out_coeffs,
        int* out_n_terms,
        int max_out);

    int transfer_matrix_sweep_multi_c(
        int width, int length,
        int num_patterns,
        const int* all_edges_flat,
        const int* edges_per_pattern,
        const int* edges_offsets,
        const int* first_col_edges,
        int num_first_col_edges,
        int num_vertices,
        int* out_ab,
        long long* out_coeffs,
        int* out_n_terms,
        int max_out);

    int transfer_matrix_sweep_multi_modp_c(
        int width, int length,
        int num_patterns,
        const int* all_edges_flat,
        const int* edges_per_pattern,
        const int* edges_offsets,
        const int* first_col_edges,
        int num_first_col_edges,
        int num_vertices,
        long long prime,
        int* out_ab,
        long long* out_coeffs,
        int* out_n_terms,
        int max_out);
""")

ffi.set_source("_transfer_matrix_cffi", r"""
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* =========================================================================
   CONFIGURATION
   ========================================================================= */

#define MAX_WIDTH 8
#define MAX_LABEL 16            /* 2 * MAX_WIDTH */
#define MAX_PARTITIONS 1430     /* Catalan(8) */
#define KEY_STRIDE 4096         /* key = a_pow * 4096 + b_pow */
#define PMAP_CAP 4096           /* partition hash map capacity (power of 2) */
#define PMAP_MASK (PMAP_CAP - 1)
#define OVERFLOW_ERR -4         /* int64 overflow detected */

typedef uint32_t polykey_t;

/* Global overflow flag — set on any int64 overflow, checked periodically */
static volatile int g_overflow = 0;

/* Global modular prime — when > 0, all Poly arithmetic works mod g_prime.
   When 0, exact int64 arithmetic with overflow detection. */
static int64_t g_prime = 0;

/* Overflow-safe or modular multiplication */
static inline int64_t val_mul(int64_t a, int64_t b) {
    if (g_prime > 0)
        return (int64_t)((__int128)a * (__int128)b % (__int128)g_prime);
    int64_t r;
    if (__builtin_mul_overflow(a, b, &r)) g_overflow = 1;
    return r;
}

/* Overflow-safe or modular addition */
static inline int64_t val_add(int64_t a, int64_t b) {
    if (g_prime > 0) {
        /* a, b in [0, p-1]; a+b < 2*p < 2^63, fits int64 */
        int64_t s = a + b;
        return s >= g_prime ? s - g_prime : s;
    }
    int64_t r;
    if (__builtin_add_overflow(a, b, &r)) g_overflow = 1;
    return r;
}

/* =========================================================================
   SPARSE POLYNOMIAL (sorted key-value arrays, uint32_t keys)
   ========================================================================= */

typedef struct {
    polykey_t* keys;   /* sorted packed keys */
    int64_t*   vals;   /* parallel coefficients */
    int        n;      /* number of terms */
    int        cap;    /* allocated capacity */
} Poly;

static Poly* poly_alloc(int cap) {
    Poly* p = (Poly*)malloc(sizeof(Poly));
    if (!p) return NULL;
    p->keys = (polykey_t*)calloc(cap, sizeof(polykey_t));
    p->vals = (int64_t*)calloc(cap, sizeof(int64_t));
    p->n = 0;
    p->cap = cap;
    return p;
}

static void poly_free(Poly* p) {
    if (p) { free(p->keys); free(p->vals); free(p); }
}

static void poly_grow(Poly* p, int needed) {
    if (needed <= p->cap) return;
    int new_cap = p->cap * 2;
    if (new_cap < needed) new_cap = needed;
    polykey_t* new_keys = (polykey_t*)realloc(p->keys, new_cap * sizeof(polykey_t));
    if (!new_keys) { g_overflow = 1; return; }
    p->keys = new_keys;
    int64_t* new_vals = (int64_t*)realloc(p->vals, new_cap * sizeof(int64_t));
    if (!new_vals) { g_overflow = 1; return; }
    p->vals = new_vals;
    p->cap = new_cap;
}

static Poly* poly_zero(void) { return poly_alloc(64); }

/* Add a single monomial (key, val) to sorted polynomial */
static void poly_add_monomial(Poly* p, polykey_t key, int64_t val) {
    if (val == 0) return;
    /* Binary search for key */
    int lo = 0, hi = p->n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (p->keys[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    if (lo < p->n && p->keys[lo] == key) {
        p->vals[lo] = val_add(p->vals[lo], val);
        if (p->vals[lo] == 0) {
            memmove(p->keys + lo, p->keys + lo + 1,
                    (p->n - lo - 1) * sizeof(polykey_t));
            memmove(p->vals + lo, p->vals + lo + 1,
                    (p->n - lo - 1) * sizeof(int64_t));
            p->n--;
        }
    } else {
        poly_grow(p, p->n + 1);
        memmove(p->keys + lo + 1, p->keys + lo,
                (p->n - lo) * sizeof(polykey_t));
        memmove(p->vals + lo + 1, p->vals + lo,
                (p->n - lo) * sizeof(int64_t));
        p->keys[lo] = key;
        p->vals[lo] = val;
        p->n++;
    }
}

/* Merge sort for parallel key/val arrays (O(n log n), cache-friendly) */
static void merge_sort_kv(polykey_t* keys, int64_t* vals, int n,
                           polykey_t* tmp_k, int64_t* tmp_v) {
    if (n <= 16) {
        /* Insertion sort for small arrays */
        int i, j;
        for (i = 1; i < n; i++) {
            polykey_t k = keys[i]; int64_t v = vals[i];
            j = i - 1;
            while (j >= 0 && keys[j] > k) {
                keys[j+1] = keys[j]; vals[j+1] = vals[j]; j--;
            }
            keys[j+1] = k; vals[j+1] = v;
        }
        return;
    }
    int mid = n / 2;
    merge_sort_kv(keys, vals, mid, tmp_k, tmp_v);
    merge_sort_kv(keys + mid, vals + mid, n - mid, tmp_k, tmp_v);
    int i = 0, j = mid, w = 0;
    while (i < mid && j < n) {
        if (keys[i] <= keys[j]) {
            tmp_k[w] = keys[i]; tmp_v[w] = vals[i]; i++;
        } else {
            tmp_k[w] = keys[j]; tmp_v[w] = vals[j]; j++;
        }
        w++;
    }
    while (i < mid) { tmp_k[w] = keys[i]; tmp_v[w] = vals[i]; i++; w++; }
    while (j < n)   { tmp_k[w] = keys[j]; tmp_v[w] = vals[j]; j++; w++; }
    memcpy(keys, tmp_k, n * sizeof(polykey_t));
    memcpy(vals, tmp_v, n * sizeof(int64_t));
}

/* dst += src (sorted merge, prune zeros) */
static void poly_add_inplace(Poly* dst, const Poly* src) {
    if (src->n == 0) return;
    if (dst->n == 0) {
        poly_grow(dst, src->n);
        memcpy(dst->keys, src->keys, src->n * sizeof(polykey_t));
        memcpy(dst->vals, src->vals, src->n * sizeof(int64_t));
        dst->n = src->n;
        return;
    }
    int needed = dst->n + src->n;
    polykey_t* tmp_k = (polykey_t*)malloc(needed * sizeof(polykey_t));
    int64_t*   tmp_v = (int64_t*)malloc(needed * sizeof(int64_t));
    int di = 0, si = 0, wi = 0;
    while (di < dst->n && si < src->n) {
        if (dst->keys[di] < src->keys[si]) {
            tmp_k[wi] = dst->keys[di]; tmp_v[wi] = dst->vals[di];
            di++; wi++;
        } else if (dst->keys[di] > src->keys[si]) {
            tmp_k[wi] = src->keys[si]; tmp_v[wi] = src->vals[si];
            si++; wi++;
        } else {
            int64_t s = val_add(dst->vals[di], src->vals[si]);
            if (s != 0) { tmp_k[wi] = dst->keys[di]; tmp_v[wi] = s; wi++; }
            di++; si++;
        }
    }
    while (di < dst->n) {
        tmp_k[wi] = dst->keys[di]; tmp_v[wi] = dst->vals[di];
        di++; wi++;
    }
    while (si < src->n) {
        tmp_k[wi] = src->keys[si]; tmp_v[wi] = src->vals[si];
        si++; wi++;
    }
    poly_grow(dst, wi);
    memcpy(dst->keys, tmp_k, wi * sizeof(polykey_t));
    memcpy(dst->vals, tmp_v, wi * sizeof(int64_t));
    dst->n = wi;
    free(tmp_k); free(tmp_v);
}

/* Shift all keys by delta */
static Poly* poly_shift(const Poly* p, polykey_t delta_key) {
    Poly* r = poly_alloc(p->n > 0 ? p->n : 64);
    int i;
    for (i = 0; i < p->n; i++) {
        r->keys[i] = p->keys[i] + delta_key;
        r->vals[i] = p->vals[i];
    }
    r->n = p->n;
    return r;
}

/* Multiply polynomials: monomial fast path + general hash-accumulate */
static Poly* poly_mul(const Poly* p, const Poly* q) {
    if (p->n == 0 || q->n == 0) return poly_zero();
    /* Monomial fast paths */
    if (p->n == 1) {
        Poly* r = poly_alloc(q->n);
        polykey_t pk = p->keys[0]; int64_t pv = p->vals[0];
        int i;
        for (i = 0; i < q->n; i++) {
            r->keys[i] = pk + q->keys[i];
            r->vals[i] = val_mul(pv, q->vals[i]);
        }
        r->n = q->n;
        return r;
    }
    if (q->n == 1) {
        Poly* r = poly_alloc(p->n);
        polykey_t qk = q->keys[0]; int64_t qv = q->vals[0];
        int i;
        for (i = 0; i < p->n; i++) {
            r->keys[i] = p->keys[i] + qk;
            r->vals[i] = val_mul(p->vals[i], qv);
        }
        r->n = p->n;
        return r;
    }
    /* General: hash table accumulation */
    int total = p->n * q->n;
    int ht_cap = 256;
    while (ht_cap < total * 4) ht_cap <<= 1;
    polykey_t* ht_keys = (polykey_t*)calloc(ht_cap, sizeof(polykey_t));
    int64_t*   ht_vals = (int64_t*)calloc(ht_cap, sizeof(int64_t));
    uint8_t*   ht_used = (uint8_t*)calloc(ht_cap, sizeof(uint8_t));
    int mask = ht_cap - 1;
    int result_n = 0;
    int i, j;
    for (i = 0; i < p->n; i++) {
        polykey_t pk = p->keys[i]; int64_t pv = p->vals[i];
        for (j = 0; j < q->n; j++) {
            polykey_t key = pk + q->keys[j];
            int64_t prod = val_mul(pv, q->vals[j]);
            uint32_t h = ((uint32_t)(key * 2654435761u)) & mask;
            while (ht_used[h]) {
                if (ht_keys[h] == key) {
                    ht_vals[h] = val_add(ht_vals[h], prod);
                    goto next_mul;
                }
                h = (h + 1) & mask;
            }
            ht_used[h] = 1; ht_keys[h] = key; ht_vals[h] = prod;
            result_n++;
            next_mul:;
        }
    }
    /* Extract and sort */
    Poly* r = poly_alloc(result_n > 0 ? result_n : 64);
    int wi = 0;
    for (i = 0; i < ht_cap; i++) {
        if (ht_used[i] && ht_vals[i] != 0) {
            r->keys[wi] = ht_keys[i]; r->vals[wi] = ht_vals[i]; wi++;
        }
    }
    r->n = wi;
    if (wi > 1) {
        polykey_t* tmp_k = (polykey_t*)malloc(wi * sizeof(polykey_t));
        int64_t*   tmp_v = (int64_t*)malloc(wi * sizeof(int64_t));
        merge_sort_kv(r->keys, r->vals, wi, tmp_k, tmp_v);
        free(tmp_k); free(tmp_v);
    }
    free(ht_keys); free(ht_vals); free(ht_used);
    return r;
}

/* =========================================================================
   NON-CROSSING PARTITION ENUMERATION
   ========================================================================= */

static int g_width;
static int g_num_partitions;
static int g_partitions[MAX_PARTITIONS][MAX_WIDTH];
static uint32_t g_partition_enc[MAX_PARTITIONS];

/* Partition hash map: encoded partition -> index */
static struct { uint32_t key; int val; uint8_t used; } g_pmap[PMAP_CAP];

static uint32_t encode_partition(const int* labels, int width) {
    uint32_t enc = 0;
    int i;
    for (i = 0; i < width; i++)
        enc |= ((uint32_t)labels[i]) << (i * 4);
    return enc;
}

static int pmap_get(uint32_t key) {
    uint32_t h = (key * 2654435761u) >> 20;
    h &= PMAP_MASK;
    while (g_pmap[h].used) {
        if (g_pmap[h].key == key) return g_pmap[h].val;
        h = (h + 1) & PMAP_MASK;
    }
    return -1;
}

static void pmap_set(uint32_t key, int val) {
    uint32_t h = (key * 2654435761u) >> 20;
    h &= PMAP_MASK;
    while (g_pmap[h].used) {
        if (g_pmap[h].key == key) { g_pmap[h].val = val; return; }
        h = (h + 1) & PMAP_MASK;
    }
    g_pmap[h].used = 1;
    g_pmap[h].key = key;
    g_pmap[h].val = val;
}

/* Non-crossing check: two blocks cross iff some element of each lies
   strictly between min and max of the other. Must check ALL positions
   of each block, not just min/max. */
static int is_noncrossing(const int* labels, int width) {
    if (width <= 2) return 1;
    int num_blocks = 0;
    int i, a, b;
    for (i = 0; i < width; i++)
        if (labels[i] + 1 > num_blocks) num_blocks = labels[i] + 1;

    for (a = 0; a < num_blocks; a++) {
        int min_a = width, max_a = -1;
        for (i = 0; i < width; i++) {
            if (labels[i] == a) {
                if (i < min_a) min_a = i;
                if (i > max_a) max_a = i;
            }
        }
        if (min_a == max_a) continue;
        for (b = a + 1; b < num_blocks; b++) {
            int min_b = width, max_b = -1;
            for (i = 0; i < width; i++) {
                if (labels[i] == b) {
                    if (i < min_b) min_b = i;
                    if (i > max_b) max_b = i;
                }
            }
            if (min_b == max_b) continue;
            /* Check if any element of b is strictly between min_a and max_a */
            int b_in_a = 0;
            for (i = 0; i < width && !b_in_a; i++)
                if (labels[i] == b && i > min_a && i < max_a) b_in_a = 1;
            if (!b_in_a) continue;
            /* Check if any element of a is strictly between min_b and max_b */
            for (i = 0; i < width; i++)
                if (labels[i] == a && i > min_b && i < max_b) return 0;
        }
    }
    return 1;
}

/* Recursive restricted growth string generator + non-crossing filter */
static void gen_partitions(int* labels, int pos, int next_fresh, int width) {
    if (pos == width) {
        if (is_noncrossing(labels, width) &&
            g_num_partitions < MAX_PARTITIONS) {
            memcpy(g_partitions[g_num_partitions], labels,
                   width * sizeof(int));
            g_num_partitions++;
        }
        return;
    }
    int lbl;
    for (lbl = 0; lbl < next_fresh; lbl++) {
        labels[pos] = lbl;
        gen_partitions(labels, pos + 1, next_fresh, width);
    }
    labels[pos] = next_fresh;
    gen_partitions(labels, pos + 1, next_fresh + 1, width);
}

static int enumerate_partitions(int width) {
    int labels[MAX_WIDTH];
    g_width = width;
    g_num_partitions = 0;
    gen_partitions(labels, 0, 0, width);
    /* Build encoding and hash map */
    memset(g_pmap, 0, sizeof(g_pmap));
    int i;
    for (i = 0; i < g_num_partitions; i++) {
        g_partition_enc[i] = encode_partition(g_partitions[i], width);
        pmap_set(g_partition_enc[i], i);
    }
    return g_num_partitions;
}

/* =========================================================================
   TRANSITION COMPUTATION
   ========================================================================= */

/* Canonicalize labels[0..width-1] in first-occurrence order, return
   the encoded uint32 representation. */
static uint32_t canonicalize_and_encode(int* labels, int width) {
    int mapping[MAX_LABEL];
    memset(mapping, -1, sizeof(mapping));
    int next_label = 0;
    uint32_t enc = 0;
    int i;
    for (i = 0; i < width; i++) {
        int lbl = labels[i];
        if (mapping[lbl] == -1) mapping[lbl] = next_label++;
        labels[i] = mapping[lbl];
        enc |= ((uint32_t)mapping[lbl]) << (i * 4);
    }
    return enc;
}

/* Compute transition for old partition + edge subset mask.
   Returns encoded new partition; sets *out_forgotten and *out_selected. */
static uint32_t compute_transition_c(
    const int* old_labels, int width,
    int subset_mask, const int* unit_cell_edges, int num_edges,
    int* out_forgotten, int* out_selected)
{
    int combined[2 * MAX_WIDTH];
    int max_old = -1;
    int i;
    for (i = 0; i < width; i++) {
        combined[i] = old_labels[i];
        if (old_labels[i] > max_old) max_old = old_labels[i];
    }
    for (i = 0; i < width; i++)
        combined[width + i] = max_old + 1 + i;

    /* Apply selected edges (3 ints per edge: ra, rb, is_cross) */
    *out_selected = 0;
    int e;
    for (e = 0; e < num_edges; e++) {
        if (!(subset_mask & (1 << e))) continue;
        (*out_selected)++;
        int ra = unit_cell_edges[3*e], rb = unit_cell_edges[3*e+1];
        int is_cross = unit_cell_edges[3*e+2];
        int pos_a, pos_b;
        if (is_cross) { pos_a = ra; pos_b = width + rb; }
        else          { pos_a = width + ra; pos_b = width + rb; }

        int la = combined[pos_a], lb = combined[pos_b];
        if (la != lb) {
            int target = la < lb ? la : lb;
            int replace = la < lb ? lb : la;
            for (i = 0; i < 2 * width; i++)
                if (combined[i] == replace) combined[i] = target;
        }
    }

    /* Count forgotten blocks: old boundary labels not in new boundary */
    uint8_t in_new[MAX_LABEL];
    memset(in_new, 0, sizeof(in_new));
    for (i = width; i < 2 * width; i++)
        in_new[combined[i]] = 1;

    uint8_t seen[MAX_LABEL];
    memset(seen, 0, sizeof(seen));
    *out_forgotten = 0;
    for (i = 0; i < width; i++) {
        int lbl = combined[i];
        if (!in_new[lbl] && !seen[lbl]) {
            seen[lbl] = 1;
            (*out_forgotten)++;
        }
    }

    /* Extract and canonicalize new boundary */
    int new_labels[MAX_WIDTH];
    for (i = 0; i < width; i++)
        new_labels[i] = combined[width + i];
    return canonicalize_and_encode(new_labels, width);
}

/* =========================================================================
   TRANSFER MATRIX BUILD
   ========================================================================= */

static Poly** build_transfer_matrix_c(int width,
    const int* unit_cell_edges, int num_edges, int cm)
{
    Poly** matrix = (Poly**)calloc(cm * cm, sizeof(Poly*));
    if (!matrix) return NULL;
    int idx;
    for (idx = 0; idx < cm * cm; idx++)
        matrix[idx] = poly_zero();

    int total_subsets = 1 << num_edges;
    int old_idx, mask;

    for (old_idx = 0; old_idx < cm; old_idx++) {
        const int* old_labels = g_partitions[old_idx];
        for (mask = 0; mask < total_subsets; mask++) {
            int num_forgotten, num_selected;
            uint32_t new_enc = compute_transition_c(
                old_labels, width, mask, unit_cell_edges, num_edges,
                &num_forgotten, &num_selected);

            int new_idx = pmap_get(new_enc);
            if (new_idx < 0) continue;  /* crossing partition, skip */

            /* Weight: a^f * b^(|S|+f), key = f * KEY_STRIDE + (|S|+f) */
            polykey_t key = (polykey_t)num_forgotten * KEY_STRIDE
                          + (polykey_t)(num_selected + num_forgotten);
            poly_add_monomial(matrix[new_idx * cm + old_idx], key, 1);
        }
    }
    return matrix;
}

/* =========================================================================
   INITIAL VECTOR
   ========================================================================= */

static Poly** build_initial_vector_c(int width, int cm,
    const int* unit_cell_edges, int num_edges) {
    Poly** vector = (Poly**)calloc(cm, sizeof(Poly*));
    if (!vector) return NULL;
    int i;
    for (i = 0; i < cm; i++)
        vector[i] = poly_zero();

    /* Extract within-column edges (is_cross == 0) from unit cell.
       These are the first column's internal edges. */
    int first_col_ra[32], first_col_rb[32];
    int n_first = 0;
    int e;
    for (e = 0; e < num_edges && n_first < 32; e++) {
        if (unit_cell_edges[3*e + 2] == 0) { /* not cross-column */
            first_col_ra[n_first] = unit_cell_edges[3*e];
            first_col_rb[n_first] = unit_cell_edges[3*e + 1];
            n_first++;
        }
    }

    int total_subsets = 1 << n_first;
    int mask;

    for (mask = 0; mask < total_subsets; mask++) {
        int labels[MAX_WIDTH];
        for (i = 0; i < width; i++) labels[i] = i;

        int num_selected = __builtin_popcount(mask);
        for (e = 0; e < n_first; e++) {
            if (!(mask & (1 << e))) continue;
            int la = labels[first_col_ra[e]], lb = labels[first_col_rb[e]];
            if (la != lb) {
                int target = la < lb ? la : lb;
                int replace = la < lb ? lb : la;
                for (i = 0; i < width; i++)
                    if (labels[i] == replace) labels[i] = target;
            }
        }

        uint32_t enc = canonicalize_and_encode(labels, width);
        int state_idx = pmap_get(enc);
        if (state_idx < 0) continue;

        /* Weight: b^num_selected (monomial) */
        poly_add_monomial(vector[state_idx], (polykey_t)num_selected, 1);
    }
    return vector;
}

/* =========================================================================
   SPARSE MATRIX-VECTOR MULTIPLY
   ========================================================================= */

typedef struct {
    int* cols;   /* column indices of nonzero entries */
    int  nnz;    /* count */
} SparseRow;

static SparseRow* build_sparse_rows(Poly** matrix, int cm) {
    SparseRow* rows = (SparseRow*)malloc(cm * sizeof(SparseRow));
    int i, j;
    for (i = 0; i < cm; i++) {
        int nnz = 0;
        for (j = 0; j < cm; j++)
            if (matrix[i * cm + j]->n > 0) nnz++;
        rows[i].nnz = nnz;
        rows[i].cols = (int*)malloc(nnz * sizeof(int));
        int k = 0;
        for (j = 0; j < cm; j++)
            if (matrix[i * cm + j]->n > 0) rows[i].cols[k++] = j;
    }
    return rows;
}

static void free_sparse_rows(SparseRow* rows, int cm) {
    int i;
    for (i = 0; i < cm; i++) free(rows[i].cols);
    free(rows);
}

/* Single matrix-vector multiply with sparse row optimization */
static Poly** mat_vec_mul(Poly** matrix, SparseRow* srows,
                           Poly** vector, int cm) {
    Poly** result = (Poly**)calloc(cm, sizeof(Poly*));
    int i;
    for (i = 0; i < cm; i++)
        result[i] = poly_zero();

    for (i = 0; i < cm; i++) {
        int k;
        for (k = 0; k < srows[i].nnz; k++) {
            int j = srows[i].cols[k];
            if (vector[j]->n == 0) continue;

            Poly* prod = poly_mul(matrix[i * cm + j], vector[j]);
            if (prod->n > 0)
                poly_add_inplace(result[i], prod);
            poly_free(prod);
        }
    }
    return result;
}

/* =========================================================================
   MAIN SWEEP
   ========================================================================= */

int transfer_matrix_sweep_c(
    int width, int length,
    const int* unit_cell_edges, int num_edges,
    int num_vertices,
    int* out_ab, long long* out_coeffs, int* out_n_terms, int max_out)
{
    Poly** matrix = NULL;
    SparseRow* srows = NULL;
    Poly** vector = NULL;
    Poly* z_poly = NULL;
    int cm = 0;
    int ret = 0;
    int i;

    if (width < 1 || width > MAX_WIDTH) { ret = -1; goto cleanup; }
    if (length < 1) { ret = -1; goto cleanup; }
    if (num_edges > 20) { ret = -1; goto cleanup; }

    g_overflow = 0;  /* reset overflow flag */

    /* Step 1: Enumerate non-crossing partitions */
    cm = enumerate_partitions(width);
    if (cm == 0) { ret = -1; goto cleanup; }

    /* Step 2: Build transfer matrix */
    matrix = build_transfer_matrix_c(width, unit_cell_edges, num_edges, cm);
    if (!matrix) { ret = -3; goto cleanup; }

    /* Step 3: Pre-compute sparse row structure */
    srows = build_sparse_rows(matrix, cm);
    if (!srows) { ret = -3; goto cleanup; }

    /* Step 4: Build initial vector (derives first-col edges from unit cell) */
    vector = build_initial_vector_c(width, cm, unit_cell_edges, num_edges);
    if (!vector) { ret = -3; goto cleanup; }

    /* Step 5: Matrix-vector multiply (length-1) times */
    for (i = 0; i < length - 1; i++) {
        Poly** new_vec = mat_vec_mul(matrix, srows, vector, cm);
        int j;
        for (j = 0; j < cm; j++) poly_free(vector[j]);
        free(vector);
        vector = new_vec;
    }

    /* Check for overflow after the multiply loop */
    if (g_overflow) { ret = OVERFLOW_ERR; goto cleanup; }

    /* Step 6: Forget final boundary + sum.
       Each partition state is multiplied by (a*b)^num_blocks
       then summed into z_poly. */
    z_poly = poly_zero();
    for (i = 0; i < cm; i++) {
        if (vector[i]->n == 0) continue;
        /* num_blocks = max(labels) + 1 for canonical partition */
        int num_blocks = 0;
        int k;
        for (k = 0; k < width; k++)
            if (g_partitions[i][k] + 1 > num_blocks)
                num_blocks = g_partitions[i][k] + 1;

        if (num_blocks > 0) {
            /* Shift key by num_blocks * (KEY_STRIDE + 1) = a^nb * b^nb */
            polykey_t shift = (polykey_t)num_blocks * (KEY_STRIDE + 1);
            Poly* shifted = poly_shift(vector[i], shift);
            poly_add_inplace(z_poly, shifted);
            poly_free(shifted);
        } else {
            poly_add_inplace(z_poly, vector[i]);
        }
    }

    /* Step 7: Exponent shift by (k(G)=1, |V|=num_vertices).
       Terms with negative shifted exponents are silently dropped.
       In the exact sweep this should never happen (would indicate a bug);
       in the modular sweep, spurious terms can arise from modular
       arithmetic and are correctly filtered here. */
    {
        int a_shift = 1;
        int b_shift = num_vertices;
        int n_out = 0;

        for (i = 0; i < z_poly->n; i++) {
            int a_pow = (int)(z_poly->keys[i] / KEY_STRIDE);
            int b_pow = (int)(z_poly->keys[i] % KEY_STRIDE);
            int new_a = a_pow - a_shift;
            int new_b = b_pow - b_shift;
            if (new_a < 0 || new_b < 0) continue;
            if (n_out >= max_out) { ret = -2; goto cleanup; }
            out_ab[n_out * 2]     = new_a;
            out_ab[n_out * 2 + 1] = new_b;
            out_coeffs[n_out]     = z_poly->vals[i];
            n_out++;
        }
        *out_n_terms = n_out;
    }

cleanup:
    if (z_poly) poly_free(z_poly);
    if (vector) {
        for (i = 0; i < cm; i++)
            if (vector[i]) poly_free(vector[i]);
        free(vector);
    }
    if (srows) free_sparse_rows(srows, cm);
    if (matrix) {
        for (i = 0; i < cm * cm; i++)
            if (matrix[i]) poly_free(matrix[i]);
        free(matrix);
    }
    return ret;
}

/* =========================================================================
   MODULAR SWEEP (same algorithm, arithmetic mod prime)
   ========================================================================= */

int transfer_matrix_sweep_modp_c(
    int width, int length,
    const int* unit_cell_edges, int num_edges,
    int num_vertices,
    long long prime,
    int* out_ab, long long* out_coeffs, int* out_n_terms, int max_out)
{
    g_prime = (int64_t)prime;
    g_overflow = 0;
    int rc = transfer_matrix_sweep_c(width, length, unit_cell_edges, num_edges,
                                      num_vertices,
                                      out_ab, out_coeffs, out_n_terms, max_out);
    g_prime = 0;
    return rc;
}

/* =========================================================================
   MULTI-PATTERN SWEEP (honeycomb and other multi-period lattices)
   ========================================================================= */

/* Build initial vector from explicit first-column edges (not derived from
   unit cell). first_col_edges is a flat array of (row_a, row_b) pairs,
   num_first = number of such pairs. */
static Poly** build_initial_vector_explicit_c(int width, int cm,
    const int* first_col_edges, int num_first) {
    Poly** vector = (Poly**)calloc(cm, sizeof(Poly*));
    if (!vector) return NULL;
    int i;
    for (i = 0; i < cm; i++)
        vector[i] = poly_zero();

    int total_subsets = 1 << num_first;
    int mask;

    for (mask = 0; mask < total_subsets; mask++) {
        int labels[MAX_WIDTH];
        for (i = 0; i < width; i++) labels[i] = i;

        int num_selected = __builtin_popcount(mask);
        int e;
        for (e = 0; e < num_first; e++) {
            if (!(mask & (1 << e))) continue;
            int ra = first_col_edges[2*e];
            int rb = first_col_edges[2*e + 1];
            int la = labels[ra], lb = labels[rb];
            if (la != lb) {
                int target = la < lb ? la : lb;
                int replace = la < lb ? lb : la;
                for (i = 0; i < width; i++)
                    if (labels[i] == replace) labels[i] = target;
            }
        }

        uint32_t enc = canonicalize_and_encode(labels, width);
        int state_idx = pmap_get(enc);
        if (state_idx < 0) continue;

        poly_add_monomial(vector[state_idx], (polykey_t)num_selected, 1);
    }
    return vector;
}

int transfer_matrix_sweep_multi_c(
    int width, int length,
    int num_patterns,
    const int* all_edges_flat,
    const int* edges_per_pattern,
    const int* edges_offsets,
    const int* first_col_edges,
    int num_first_col_edges,
    int num_vertices,
    int* out_ab, long long* out_coeffs, int* out_n_terms, int max_out)
{
    Poly*** matrices = NULL;
    SparseRow** all_srows = NULL;
    Poly** vector = NULL;
    Poly* z_poly = NULL;
    int cm = 0;
    int ret = 0;
    int i, p;

    if (width < 1 || width > MAX_WIDTH) { ret = -1; goto cleanup_multi; }
    if (length < 1) { ret = -1; goto cleanup_multi; }
    if (num_patterns < 1 || num_patterns > 8) { ret = -1; goto cleanup_multi; }
    for (p = 0; p < num_patterns; p++)
        if (edges_per_pattern[p] > 20) { ret = -1; goto cleanup_multi; }

    g_overflow = 0;

    /* Step 1: Enumerate non-crossing partitions */
    cm = enumerate_partitions(width);
    if (cm == 0) { ret = -1; goto cleanup_multi; }

    /* Step 2: Build transfer matrix per pattern */
    matrices = (Poly***)calloc(num_patterns, sizeof(Poly**));
    all_srows = (SparseRow**)calloc(num_patterns, sizeof(SparseRow*));
    if (!matrices || !all_srows) { ret = -3; goto cleanup_multi; }

    for (p = 0; p < num_patterns; p++) {
        const int* pattern_edges = all_edges_flat + edges_offsets[p] * 3;
        int n_edges = edges_per_pattern[p];
        matrices[p] = build_transfer_matrix_c(width, pattern_edges, n_edges, cm);
        if (!matrices[p]) { ret = -3; goto cleanup_multi; }
        all_srows[p] = build_sparse_rows(matrices[p], cm);
        if (!all_srows[p]) { ret = -3; goto cleanup_multi; }
    }

    /* Step 3: Build initial vector from explicit first-col edges */
    vector = build_initial_vector_explicit_c(
        width, cm, first_col_edges, num_first_col_edges);
    if (!vector) { ret = -3; goto cleanup_multi; }

    /* Step 4: Matrix-vector multiply (length-1) times, alternating patterns */
    for (i = 0; i < length - 1; i++) {
        int pat_idx = i % num_patterns;
        Poly** new_vec = mat_vec_mul(
            matrices[pat_idx], all_srows[pat_idx], vector, cm);
        int j;
        for (j = 0; j < cm; j++) poly_free(vector[j]);
        free(vector);
        vector = new_vec;
    }

    if (g_overflow) { ret = OVERFLOW_ERR; goto cleanup_multi; }

    /* Step 5: Forget final boundary + sum */
    z_poly = poly_zero();
    for (i = 0; i < cm; i++) {
        if (vector[i]->n == 0) continue;
        int num_blocks = 0;
        int k;
        for (k = 0; k < width; k++)
            if (g_partitions[i][k] + 1 > num_blocks)
                num_blocks = g_partitions[i][k] + 1;

        if (num_blocks > 0) {
            polykey_t shift = (polykey_t)num_blocks * (KEY_STRIDE + 1);
            Poly* shifted = poly_shift(vector[i], shift);
            poly_add_inplace(z_poly, shifted);
            poly_free(shifted);
        } else {
            poly_add_inplace(z_poly, vector[i]);
        }
    }

    /* Step 6: Exponent shift (see Step 7 comment in single-pattern sweep). */
    {
        int a_shift = 1;
        int b_shift = num_vertices;
        int n_out = 0;

        for (i = 0; i < z_poly->n; i++) {
            int a_pow = (int)(z_poly->keys[i] / KEY_STRIDE);
            int b_pow = (int)(z_poly->keys[i] % KEY_STRIDE);
            int new_a = a_pow - a_shift;
            int new_b = b_pow - b_shift;
            if (new_a < 0 || new_b < 0) continue;
            if (n_out >= max_out) { ret = -2; goto cleanup_multi; }
            out_ab[n_out * 2]     = new_a;
            out_ab[n_out * 2 + 1] = new_b;
            out_coeffs[n_out]     = z_poly->vals[i];
            n_out++;
        }
        *out_n_terms = n_out;
    }

cleanup_multi:
    if (z_poly) poly_free(z_poly);
    if (vector) {
        for (i = 0; i < cm; i++)
            if (vector[i]) poly_free(vector[i]);
        free(vector);
    }
    if (all_srows) {
        for (p = 0; p < num_patterns; p++)
            if (all_srows[p]) free_sparse_rows(all_srows[p], cm);
        free(all_srows);
    }
    if (matrices) {
        for (p = 0; p < num_patterns; p++) {
            if (matrices[p]) {
                for (i = 0; i < cm * cm; i++)
                    if (matrices[p][i]) poly_free(matrices[p][i]);
                free(matrices[p]);
            }
        }
        free(matrices);
    }
    return ret;
}

int transfer_matrix_sweep_multi_modp_c(
    int width, int length,
    int num_patterns,
    const int* all_edges_flat,
    const int* edges_per_pattern,
    const int* edges_offsets,
    const int* first_col_edges,
    int num_first_col_edges,
    int num_vertices,
    long long prime,
    int* out_ab, long long* out_coeffs, int* out_n_terms, int max_out)
{
    g_prime = (int64_t)prime;
    g_overflow = 0;
    int rc = transfer_matrix_sweep_multi_c(
        width, length, num_patterns,
        all_edges_flat, edges_per_pattern, edges_offsets,
        first_col_edges, num_first_col_edges,
        num_vertices,
        out_ab, out_coeffs, out_n_terms, max_out);
    g_prime = 0;
    return rc;
}
""")

# =============================================================================
# BUILD / LOAD
# =============================================================================

_lib = None
_ffi = ffi


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    try:
        from _transfer_matrix_cffi import ffi as _cffi, lib  # noqa: F811
        _lib = lib
        return _lib
    except ImportError:
        pass
    import atexit
    import shutil
    import tempfile
    import sys
    tmpdir = tempfile.mkdtemp(prefix="transfer_matrix_c_")
    atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
    ffi.compile(tmpdir=tmpdir)
    sys.path.insert(0, tmpdir)
    from _transfer_matrix_cffi import ffi as _cffi, lib  # noqa: F811
    _lib = lib
    return _lib


# =============================================================================
# PYTHON WRAPPER
# =============================================================================


def _marshal_edges(unit_cell_edges):
    """Marshal 3-tuple edge list to cffi int array (3 ints per edge)."""
    num_edges = len(unit_cell_edges)
    edges_flat = _ffi.new("int[]", num_edges * 3)
    for i, (ra, rb, is_cross) in enumerate(unit_cell_edges):
        edges_flat[3 * i] = ra
        edges_flat[3 * i + 1] = rb
        edges_flat[3 * i + 2] = 1 if is_cross else 0
    return edges_flat, num_edges


def _unmarshal_poly(out_ab, out_coeffs, n_terms):
    """Unmarshal C output arrays to Python dict."""
    result = {}
    for i in range(n_terms):
        a_pow = out_ab[i * 2]
        b_pow = out_ab[i * 2 + 1]
        coeff = int(out_coeffs[i])
        if coeff != 0:
            result[(a_pow, b_pow)] = coeff
    return result


def _run_sweep_exact(lib, edges_flat, num_edges, width, length, num_vertices):
    """Run exact int64 sweep. Returns dict or None on overflow."""
    max_out = 100000
    out_ab = _ffi.new("int[]", max_out * 2)
    out_coeffs = _ffi.new("long long[]", max_out)
    out_n = _ffi.new("int*")

    rc = lib.transfer_matrix_sweep_c(
        width, length, edges_flat, num_edges, num_vertices,
        out_ab, out_coeffs, out_n, max_out)

    if rc != 0:
        return None
    return _unmarshal_poly(out_ab, out_coeffs, out_n[0])


def _run_sweep_modp(lib, edges_flat, num_edges, width, length, num_vertices,
                    prime):
    """Run modular sweep mod prime. Returns dict of residues."""
    max_out = 100000
    out_ab = _ffi.new("int[]", max_out * 2)
    out_coeffs = _ffi.new("long long[]", max_out)
    out_n = _ffi.new("int*")

    rc = lib.transfer_matrix_sweep_modp_c(
        width, length, edges_flat, num_edges, num_vertices, prime,
        out_ab, out_coeffs, out_n, max_out)

    if rc != 0:
        return None
    return _unmarshal_poly(out_ab, out_coeffs, out_n[0])


# Large primes for CRT (verified prime by Miller-Rabin during generation)
_CRT_PRIMES = None


def _get_crt_primes(count):
    """Generate large primes near 2^62 for CRT reconstruction."""
    global _CRT_PRIMES
    if _CRT_PRIMES is not None and len(_CRT_PRIMES) >= count:
        return _CRT_PRIMES[:count]

    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        d, r = n - 1, 0
        while d % 2 == 0:
            d >>= 1
            r += 1
        # Deterministic witnesses for n < 2^64
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Search sequentially from a large odd starting point to guarantee
    # distinct primes. (1 << 62) - 57 = 4611686018427387847 is odd.
    primes = []
    p = (1 << 62) - 57
    while len(primes) < count:
        if is_prime(p):
            primes.append(p)
        p += 2

    _CRT_PRIMES = primes
    return primes[:count]


def _crt_reconstruct(residues_list, primes):
    """Chinese Remainder Theorem: reconstruct exact integer from residues.

    Args:
        residues_list: list of dicts {key: residue_mod_pi}
        primes: list of primes

    Returns:
        dict {key: exact_coefficient}
    """
    # Collect all keys across all modular results
    all_keys = set()
    for res in residues_list:
        all_keys.update(res.keys())

    M = 1
    for p in primes:
        M *= p
    half_M = M // 2

    result = {}
    for key in all_keys:
        # Garner's algorithm for CRT (more numerically stable)
        x = 0
        for k in range(len(primes)):
            r_k = residues_list[k].get(key, 0)
            # Compute x_k such that x ≡ r_k (mod p_k)
            diff = (r_k - x) % primes[k]
            # Compute product of p_0 * p_1 * ... * p_{k-1}
            prod = 1
            for j in range(k):
                prod = prod * primes[j] % primes[k]
            inv = pow(prod, -1, primes[k])
            coeff = diff * inv % primes[k]
            # Accumulate
            prod_full = 1
            for j in range(k):
                prod_full *= primes[j]
            x += coeff * prod_full

        x %= M
        # Convert to signed: if x > M/2, it's negative
        if x > half_M:
            x -= M
        if x != 0:
            result[key] = x

    return result


def c_transfer_matrix_sweep(width, length, unit_cell_edges, num_vertices=None):
    """C-accelerated transfer matrix sweep.

    Returns dict {(a_pow, b_pow): coeff} in the (a,b) = (x-1, y-1) basis
    after exponent shift, or None if the C extension is unavailable.

    Uses exact int64 arithmetic for small lattices and modular arithmetic
    with CRT reconstruction for large lattices where int64 overflows.

    Args:
        width: Boundary width (m).
        length: Number of columns (n).
        unit_cell_edges: List of (row_a, row_b, is_cross) 3-tuples.
        num_vertices: Total vertex count. Defaults to width * length.
    """
    if num_vertices is None:
        num_vertices = width * length

    try:
        lib = _get_lib()
    except Exception:
        return None

    edges_flat, num_edges = _marshal_edges(unit_cell_edges)

    # Try exact sweep first
    result = _run_sweep_exact(
        lib, edges_flat, num_edges, width, length, num_vertices)
    if result is not None:
        return result

    # Exact sweep overflowed — use modular CRT approach.
    # Bound: max coefficient < 2^(|V| + |E|).
    # Use generous edge count estimate (safe overestimate for CRT).
    n_edges = len(unit_cell_edges) * length
    bits_needed = num_vertices + n_edges + 1  # +1 for sign
    n_primes = (bits_needed + 61) // 62  # each prime gives ~62 bits

    primes = _get_crt_primes(n_primes)
    residues = []
    for p in primes:
        res = _run_sweep_modp(
            lib, edges_flat, num_edges, width, length, num_vertices, p)
        if res is None:
            return None  # unexpected failure
        residues.append(res)

    return _crt_reconstruct(residues, primes)


def _marshal_multi_edges(transition_patterns):
    """Marshal multiple edge patterns into flat arrays for C.

    Returns (all_edges_flat, edges_per_pattern, edges_offsets, total_edges).
    """
    num_patterns = len(transition_patterns)
    edges_per = _ffi.new("int[]", num_patterns)
    offsets = _ffi.new("int[]", num_patterns)
    total = 0
    for i, pat in enumerate(transition_patterns):
        edges_per[i] = len(pat)
        offsets[i] = total
        total += len(pat)

    all_flat = _ffi.new("int[]", total * 3)
    idx = 0
    for pat in transition_patterns:
        for ra, rb, is_cross in pat:
            all_flat[idx] = ra
            all_flat[idx + 1] = rb
            all_flat[idx + 2] = 1 if is_cross else 0
            idx += 3

    return all_flat, edges_per, offsets, total


def _marshal_first_col(first_col_edges):
    """Marshal first-column edge pairs to flat int array."""
    n = len(first_col_edges)
    flat = _ffi.new("int[]", n * 2)
    for i, (ra, rb) in enumerate(first_col_edges):
        flat[2 * i] = ra
        flat[2 * i + 1] = rb
    return flat, n


def _run_sweep_multi_exact(lib, all_edges_flat, edges_per, offsets,
                           num_patterns, first_col_flat, num_first,
                           width, length, num_vertices):
    """Run exact multi-pattern sweep. Returns dict or None on overflow."""
    max_out = 100000
    out_ab = _ffi.new("int[]", max_out * 2)
    out_coeffs = _ffi.new("long long[]", max_out)
    out_n = _ffi.new("int*")

    rc = lib.transfer_matrix_sweep_multi_c(
        width, length, num_patterns,
        all_edges_flat, edges_per, offsets,
        first_col_flat, num_first, num_vertices,
        out_ab, out_coeffs, out_n, max_out)

    if rc != 0:
        return None
    return _unmarshal_poly(out_ab, out_coeffs, out_n[0])


def _run_sweep_multi_modp(lib, all_edges_flat, edges_per, offsets,
                          num_patterns, first_col_flat, num_first,
                          width, length, num_vertices, prime):
    """Run modular multi-pattern sweep. Returns dict of residues."""
    max_out = 100000
    out_ab = _ffi.new("int[]", max_out * 2)
    out_coeffs = _ffi.new("long long[]", max_out)
    out_n = _ffi.new("int*")

    rc = lib.transfer_matrix_sweep_multi_modp_c(
        width, length, num_patterns,
        all_edges_flat, edges_per, offsets,
        first_col_flat, num_first, num_vertices, prime,
        out_ab, out_coeffs, out_n, max_out)

    if rc != 0:
        return None
    return _unmarshal_poly(out_ab, out_coeffs, out_n[0])


def c_transfer_matrix_sweep_multi(width, length, transition_patterns,
                                  first_col_edges, num_vertices=None):
    """C-accelerated transfer matrix sweep for multi-period lattices.

    Handles alternating unit cell patterns (e.g. honeycomb even/odd columns).

    Returns dict {(a_pow, b_pow): coeff} in the (a,b) = (x-1, y-1) basis
    after exponent shift, or None if the C extension is unavailable.

    Args:
        width: Boundary width (m).
        length: Number of columns (n).
        transition_patterns: List of edge pattern lists, cycled through
            successive transitions.
        first_col_edges: List of (row_a, row_b) pairs for initial column.
        num_vertices: Total vertex count. Defaults to width * length.
    """
    if num_vertices is None:
        num_vertices = width * length

    try:
        lib = _get_lib()
    except Exception:
        return None

    num_patterns = len(transition_patterns)
    all_edges_flat, edges_per, offsets, total = _marshal_multi_edges(
        transition_patterns)
    first_col_flat, num_first = _marshal_first_col(first_col_edges)

    # Try exact sweep first
    result = _run_sweep_multi_exact(
        lib, all_edges_flat, edges_per, offsets, num_patterns,
        first_col_flat, num_first, width, length, num_vertices)
    if result is not None:
        return result

    # Exact sweep overflowed — use modular CRT approach.
    max_edges_per_pattern = max(len(p) for p in transition_patterns)
    n_edges = max_edges_per_pattern * length
    bits_needed = num_vertices + n_edges + 1
    n_primes = (bits_needed + 61) // 62

    primes = _get_crt_primes(n_primes)
    residues = []
    for p in primes:
        res = _run_sweep_multi_modp(
            lib, all_edges_flat, edges_per, offsets, num_patterns,
            first_col_flat, num_first, width, length, num_vertices, p)
        if res is None:
            return None
        residues.append(res)

    return _crt_reconstruct(residues, primes)
