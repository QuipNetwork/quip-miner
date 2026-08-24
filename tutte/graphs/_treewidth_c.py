"""C extension for treewidth DP — bulk computation via cffi.

Provides:
1. batch_merge: C-accelerated partition merge for _merge_tables
2. treewidth_tutte_dp: Complete treewidth DP in C (10-20x faster than Python)

The bulk DP runs entirely in C with no Python in the hot loop.
"""

import cffi

ffi = cffi.FFI()

ffi.cdef("""
    /* Partition merge (used by Python _merge_tables) */
    long long merge_and_canonicalize(
        long long enc, const int* conn_pairs, int n_pairs,
        const int* shared_positions, int n_shared);

    void batch_merge(
        const long long* parent_encs, int n_parents,
        const int* conn_pairs, int n_pairs,
        const int* shared_positions, int n_shared,
        long long* out_merged);

    /* Modular determinant via Gaussian elimination in int64.
       Returns det(M) mod prime. M is n×n, row-major in flat array.
       M is MODIFIED in-place. */
    long long modular_det(long long* M, int n, long long prime);

    /* Compute det(M) mod multiple primes in one call.
       M_orig is the original integer matrix (not modified).
       primes[n_primes] = list of primes.
       out_residues[n_primes] = output residues.
       Uses a temp buffer internally for each prime. */
    void modular_det_multi(
        const long long* M_orig, int n,
        const long long* primes, int n_primes,
        long long* out_residues);

    /* Bulk treewidth DP — entire computation in C.
       Returns 0 on success, nonzero on error.
       Output: (x_power, y_power, coefficient) triples in out arrays.
       dp_prime: modulus for DP arithmetic (0 = exact int64).
       conv_prime: modulus for (a,b)→(x,y) basis conversion (0 = exact int64). */
    int treewidth_tutte_dp(
        int n_bags, const int* bag_sizes, const int* bag_verts_flat,
        int root, const int* children_counts, const int* children_flat,
        const int* bag_edge_counts, const int* edges_flat,
        int n_verts, int n_components,
        int* out_xy, long long* out_coeffs, int* out_n_terms, int max_out,
        long long dp_prime, long long conv_prime);

    /* Treewidth DP — returns only the (a,b)-basis polynomial (no conversion).
       Used when we want to run the DP once and convert with multiple primes. */
    int treewidth_tutte_dp_ab(
        int n_bags, const int* bag_sizes, const int* bag_verts_flat,
        int root, const int* children_counts, const int* children_flat,
        const int* bag_edge_counts, const int* edges_flat,
        int n_verts, int n_components,
        int* out_ab, long long* out_coeffs, int* out_n_terms, int max_out);

    /* Modular variant of treewidth_tutte_dp_ab — runs DP mod dp_prime, emits
       (a,b) residues. Used for (a,b)-only CRT reconstruction when coefficients
       exceed int128 (>120 edges). */
    int treewidth_tutte_dp_ab_mod(
        int n_bags, const int* bag_sizes, const int* bag_verts_flat,
        int root, const int* children_counts, const int* children_flat,
        const int* bag_edge_counts, const int* edges_flat,
        int n_verts, int n_components,
        int* out_ab, long long* out_coeffs, int* out_n_terms, int max_out,
        long long dp_prime);

    /* Pseudo-Mersenne variant: dp_prime must equal 2^62 - dp_prime_c.
       Uses fast pseudo-Mersenne reduction in coeff_mul, ~3x faster. */
    int treewidth_tutte_dp_ab_mersenne(
        int n_bags, const int* bag_sizes, const int* bag_verts_flat,
        int root, const int* children_counts, const int* children_flat,
        const int* bag_edge_counts, const int* edges_flat,
        int n_verts, int n_components,
        int* out_ab, long long* out_coeffs, int* out_n_terms, int max_out,
        long long dp_prime, long long dp_prime_c);

    /* __int128 variant — exact arithmetic, coefficients up to ~2^127.
       Outputs each coefficient as (hi64, lo64) pair for Python reconstruction. */
    int treewidth_tutte_dp_ab128(
        int n_bags, const int* bag_sizes, const int* bag_verts_flat,
        int root, const int* children_counts, const int* children_flat,
        const int* bag_edge_counts, const int* edges_flat,
        int n_verts, int n_components,
        int* out_ab, long long* out_coeffs_hi, long long* out_coeffs_lo,
        int* out_n_terms, int max_out);

    /* Convert (a,b)-basis polynomial to (x,y)-basis via binomial expansion.
       If conv_prime > 0, uses modular arithmetic for binomials (handles overflow).
       ab_keys[i] = a_pow * 256 + b_pow, ab_coeffs[i] = coefficient in (a,b) basis.
       Outputs (x,y) coefficients in out_xy/out_coeffs. */
    int basis_convert_ab_to_xy(
        const int* ab_keys, const long long* ab_coeffs, int n_ab_terms,
        int* out_xy, long long* out_coeffs, int* out_n_terms, int max_out,
        long long conv_prime);

    /* Profiling: read/reset/size the per-thread DP stat counters. */
    void treewidth_get_stats(long long* out, int n);
    void treewidth_reset_stats(void);
    int treewidth_stats_size(void);

""")

ffi.set_source("_treewidth_cffi", r"""
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

/* =========================================================================
   PARTITION ENCODING (4 bits per element, max 16 elements)
   ========================================================================= */

static uint64_t canonicalize_enc(uint64_t enc) {
    int n = (int)(enc & 0xF);
    int mapping[16]; int seen[16];
    memset(seen, 0, sizeof(seen));
    int next_id = 0;
    uint64_t result = n;
    int i;
    for (i = 0; i < n; i++) {
        int lbl = (int)((enc >> (4 + i*4)) & 0xF);
        if (!seen[lbl]) { seen[lbl] = 1; mapping[lbl] = next_id++; }
        result |= ((uint64_t)mapping[lbl]) << (4 + i*4);
    }
    return result;
}

static uint64_t encoded_connect(uint64_t enc, int i, int j) {
    int n = (int)(enc & 0xF);
    int li = (int)((enc >> (4 + i*4)) & 0xF);
    int lj = (int)((enc >> (4 + j*4)) & 0xF);
    if (li == lj) return enc;
    int target = li < lj ? li : lj;
    int replace = li < lj ? lj : li;
    int k;
    for (k = 0; k < n; k++) {
        if (((enc >> (4 + k*4)) & 0xF) == (uint64_t)replace)
            enc = (enc & ~((uint64_t)0xF << (4+k*4))) | ((uint64_t)target << (4+k*4));
    }
    return canonicalize_enc(enc);
}

/* Connect cache: open-addressing hash map */
#define CCACHE_CAP (1 << 21)  /* 2M entries */
#define CCACHE_MASK (CCACHE_CAP - 1)
/* Cache uses separate enc + ij fields to avoid key collisions:
   key1 = enc (full partition encoding)
   key2 = (i << 4) | j (packed indices, max 16 each)
   Both must match for a cache hit. */
static __thread struct {
    uint64_t enc;
    uint8_t  ij;   /* (i << 4) | j */
    uint64_t val;
    uint8_t  used;
} ccache[CCACHE_CAP];

/* Profiling counters incremented in hot paths. Read via
   treewidth_get_stats(); reset via treewidth_reset_stats(). All counts
   are per-thread so multi-prime parallel runs have separate counters. */
typedef struct {
    long long n_poly_mul_monomial;
    long long n_poly_mul_general;
    long long n_coeff_mul_inner;     /* cumulative p->n * q->n across general muls */
    long long sum_poly_mul_n_out;    /* cumulative result size */
    long long n_cached_connect_hit;
    long long n_cached_connect_miss;
    long long n_dpt_add;
    long long n_merge_tables;
    long long n_poly_shift;
    long long n_poly_copy;
    long long n_poly_add_inplace;
    long long n_process_bag_calls;
    long long n_encoded_forget;
    long long sum_merge_parent_n;    /* sum of parent.n across all merge calls */
    long long sum_merge_n_groups;    /* sum of n_groups across all merge calls */
    long long sum_merge_pbo_n_total; /* sum of pbo.n across (merge, group) pairs */
    long long n_merge_group_iters;   /* count of (merge, group) pairs */
    /* Per-merge cumulative time (ns) — split into the dominant phases.
       Only populated when treewidth_record_bag_times=1. */
    long long ns_total_merge_tables; /* sum over merge_tables calls */
    long long ns_pbo_build;          /* phase 1: parent → pbo via dpt_add */
    long long ns_pbo_collect;        /* phase 2: pbo → result via poly_mul+dpt_add */
} TWStats;
static __thread TWStats g_tw_stats;

static uint64_t cached_connect(uint64_t enc, int i, int j) {
    uint64_t h64 = enc * 0x9E3779B97F4A7C15ULL + (uint64_t)i * 31 + j;
    uint32_t h = (uint32_t)(h64 >> 43) & CCACHE_MASK;
    uint8_t ij = (uint8_t)((i << 4) | j);
    int probe;
    for (probe = 0; probe < 8; probe++) {
        uint32_t slot = (h + probe) & CCACHE_MASK;
        if (ccache[slot].used && ccache[slot].enc == enc && ccache[slot].ij == ij) {
            g_tw_stats.n_cached_connect_hit++;
            return ccache[slot].val;
        }
        if (!ccache[slot].used) {
            g_tw_stats.n_cached_connect_miss++;
            uint64_t result = encoded_connect(enc, i, j);
            ccache[slot].enc = enc; ccache[slot].ij = ij;
            ccache[slot].val = result; ccache[slot].used = 1;
            return result;
        }
    }
    g_tw_stats.n_cached_connect_miss++;
    return encoded_connect(enc, i, j);
}

typedef struct {
    int is_singleton;
    uint64_t new_enc;
} ForgetResult;

static ForgetResult encoded_forget(uint64_t enc, int pos) {
    g_tw_stats.n_encoded_forget++;
    int n = (int)(enc & 0xF);
    int lbl_i = (int)((enc >> (4 + pos*4)) & 0xF);
    int count = 0;
    int labels[16], new_n = 0;
    int k;
    for (k = 0; k < n; k++) {
        int lbl = (int)((enc >> (4 + k*4)) & 0xF);
        if (lbl == lbl_i) count++;
        if (k != pos) labels[new_n++] = lbl;
    }
    /* Canonicalize */
    int mapping[16]; int seen[16];
    memset(seen, 0, sizeof(seen));
    int next_id = 0;
    uint64_t result = new_n;
    for (k = 0; k < new_n; k++) {
        int lbl = labels[k];
        if (!seen[lbl]) { seen[lbl] = 1; mapping[lbl] = next_id++; }
        result |= ((uint64_t)mapping[lbl]) << (4 + k*4);
    }
    ForgetResult fr;
    fr.is_singleton = (count == 1);
    fr.new_enc = result;
    return fr;
}

/* Partition merge */
long long merge_and_canonicalize(
    long long enc, const int* conn_pairs, int n_pairs,
    const int* shared_positions, int n_shared)
{
    int n = (int)(enc & 0xF);
    int labels[16]; int i;
    for (i = 0; i < n; i++) labels[i] = (int)((enc >> (4 + i*4)) & 0xF);
    int p;
    for (p = 0; p < n_pairs; p++) {
        int pi = shared_positions[conn_pairs[2*p]];
        int pj = shared_positions[conn_pairs[2*p+1]];
        int lpi = labels[pi], lpj = labels[pj];
        if (lpi != lpj) {
            int target = lpi < lpj ? lpi : lpj;
            int replace = lpi < lpj ? lpj : lpi;
            for (i = 0; i < n; i++)
                if (labels[i] == replace) labels[i] = target;
        }
    }
    int mapping[16]; int seen[16];
    memset(seen, 0, sizeof(seen));
    int next_id = 0;
    long long result = n;
    for (i = 0; i < n; i++) {
        int lbl = labels[i];
        if (!seen[lbl]) { seen[lbl] = 1; mapping[lbl] = next_id++; }
        result |= ((long long)mapping[lbl]) << (4 + i*4);
    }
    return result;
}

void batch_merge(
    const long long* parent_encs, int n_parents,
    const int* conn_pairs, int n_pairs,
    const int* shared_positions, int n_shared,
    long long* out_merged)
{
    int i;
    for (i = 0; i < n_parents; i++)
        out_merged[i] = merge_and_canonicalize(
            parent_encs[i], conn_pairs, n_pairs, shared_positions, n_shared);
}

/* =========================================================================
   MODULAR DETERMINANT (Gaussian elimination in int64)
   ========================================================================= */

static inline long long mod_val(long long a, long long p) {
    a %= p;
    if (a < 0) a += p;
    return a;
}

static inline long long mod_inv(long long a, long long p) {
    /* Extended Euclidean algorithm */
    long long g = p, x = 0, y = 1;
    long long ag = a, ax = 1, ay = 0;
    while (ag != 0) {
        long long q = g / ag;
        long long tg = g - q * ag; g = ag; ag = tg;
        long long tx = x - q * ax; x = ax; ax = tx;
        long long ty = y - q * ay; y = ay; ay = ty;
    }
    return mod_val(x, p);
}

long long modular_det(long long* M, int n, long long prime) {
    /* Gaussian elimination with partial pivoting, mod prime.
       M is modified in-place. Returns det(M) mod prime. */
    long long det = 1;
    int i, j, k;

    /* Reduce mod prime */
    for (i = 0; i < n * n; i++)
        M[i] = mod_val(M[i], prime);

    for (k = 0; k < n; k++) {
        /* Find pivot */
        if (M[k * n + k] == 0) {
            int found = 0;
            for (i = k + 1; i < n; i++) {
                if (M[i * n + k] != 0) {
                    /* Swap rows */
                    for (j = 0; j < n; j++) {
                        long long tmp = M[k * n + j];
                        M[k * n + j] = M[i * n + j];
                        M[i * n + j] = tmp;
                    }
                    det = prime - det; /* negate mod prime */
                    found = 1;
                    break;
                }
            }
            if (!found) return 0; /* singular */
        }

        /* Multiply det by pivot */
        __int128 d128 = (__int128)det * (__int128)M[k * n + k];
        det = (long long)(d128 % (__int128)prime);

        /* Eliminate column k */
        long long inv_pivot = mod_inv(M[k * n + k], prime);
        for (i = k + 1; i < n; i++) {
            __int128 factor128 = (__int128)M[i * n + k] * (__int128)inv_pivot;
            long long factor = (long long)(factor128 % (__int128)prime);
            for (j = k; j < n; j++) {
                __int128 prod = (__int128)factor * (__int128)M[k * n + j];
                long long sub = (long long)(prod % (__int128)prime);
                M[i * n + j] = mod_val(M[i * n + j] - sub, prime);
            }
        }
    }

    return det;
}

void modular_det_multi(
    const long long* M_orig, int n,
    const long long* primes, int n_primes,
    long long* out_residues)
{
    /* Allocate temp buffer once */
    long long* M = (long long*)malloc(n * n * sizeof(long long));
    int p;
    for (p = 0; p < n_primes; p++) {
        /* Copy and reduce mod prime */
        memcpy(M, M_orig, n * n * sizeof(long long));
        out_residues[p] = modular_det(M, n, primes[p]);
    }
    free(M);
}

/* =========================================================================
   MODULAR COEFFICIENT ARITHMETIC
   ========================================================================= */

/* Global prime for modular arithmetic. 0 = exact int64.
   Thread-local so parallel CRT prime workers don't race.
   g_prime_c > 0 indicates pseudo-Mersenne form: g_prime = 2^62 - g_prime_c.
   This enables fast reduction without __int128 division. */
static __thread long long g_prime = 0;
static __thread long long g_prime_c = 0;

static inline long long coeff_add(long long a, long long b) {
    if (g_prime == 0) return a + b;
    long long s = a + b;
    if (s >= g_prime) s -= g_prime;
    else if (s <= -g_prime) s += g_prime;
    return s;
}

static inline long long coeff_mul(long long a, long long b) {
    if (g_prime == 0) return a * b;
    __int128 prod = (__int128)a * (__int128)b;
    if (g_prime_c > 0) {
        /* Pseudo-Mersenne: p = 2^62 - c. Substitute 2^62 ≡ c (mod p). */
        long long h = (long long)(prod >> 62);
        long long l = (long long)(prod & ((1LL << 62) - 1));
        __int128 t1 = (__int128)h * (__int128)g_prime_c + (__int128)l;
        long long h1 = (long long)(t1 >> 62);
        long long l1 = (long long)(t1 & ((1LL << 62) - 1));
        long long t2 = h1 * g_prime_c + l1;
        if (t2 >= g_prime) t2 -= g_prime;
        if (t2 >= g_prime) t2 -= g_prime;
        return t2;
    }
    return (long long)(prod % (__int128)g_prime);
}

static inline long long coeff_neg(long long a) {
    if (g_prime == 0) return -a;
    return a == 0 ? 0 : g_prime - a;
}

static inline long long coeff_from_int(long long a) {
    if (g_prime == 0) return a;
    long long r = a % g_prime;
    return r < 0 ? r + g_prime : r;
}

/* =========================================================================
   SPARSE POLYNOMIAL (sorted key-value arrays)
   ========================================================================= */

#define KEY_STRIDE 256  /* key = a_pow * 256 + b_pow */

typedef struct {
    uint16_t* keys;   /* sorted packed keys */
    int64_t*  vals;   /* parallel coefficients */
    int       n;      /* number of terms */
    int       cap;    /* allocated capacity */
} Poly;

static Poly* poly_alloc(int cap) {
    Poly* p = (Poly*)malloc(sizeof(Poly));
    p->keys = (uint16_t*)calloc(cap, sizeof(uint16_t));
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
    p->keys = (uint16_t*)realloc(p->keys, new_cap * sizeof(uint16_t));
    p->vals = (int64_t*)realloc(p->vals, new_cap * sizeof(int64_t));
    p->cap = new_cap;
}

static Poly* poly_zero(void) { return poly_alloc(64); }

static Poly* poly_one(void) {
    Poly* p = poly_alloc(64);
    p->keys[0] = 0; p->vals[0] = coeff_from_int(1); p->n = 1;
    return p;
}

static Poly* poly_copy(const Poly* src) {
    g_tw_stats.n_poly_copy++;
    Poly* dst = poly_alloc(src->n > 0 ? src->n : 64);
    memcpy(dst->keys, src->keys, src->n * sizeof(uint16_t));
    memcpy(dst->vals, src->vals, src->n * sizeof(int64_t));
    dst->n = src->n;
    return dst;
}

/* Thread-local buffer for poly_add_inplace. Avoids a per-call 80KB stack
   frame (8192 × uint16_t + 8192 × int64_t) which touches stack pages
   and hurts L1 cache locality across the 459M+ poly_add_inplace calls
   in a typical Z(1, 2)-class DP. Grown on demand. */
static __thread uint16_t* tl_padd_buf_k = NULL;
static __thread int64_t*  tl_padd_buf_v = NULL;
static __thread int       tl_padd_buf_cap = 0;

static inline void padd_buf_ensure(int needed) {
    if (needed <= tl_padd_buf_cap) return;
    int new_cap = tl_padd_buf_cap > 0 ? tl_padd_buf_cap * 2 : 1024;
    while (new_cap < needed) new_cap *= 2;
    free(tl_padd_buf_k); free(tl_padd_buf_v);
    tl_padd_buf_k = (uint16_t*)malloc(new_cap * sizeof(uint16_t));
    tl_padd_buf_v = (int64_t*)malloc(new_cap * sizeof(int64_t));
    tl_padd_buf_cap = new_cap;
}

/* dst += src (sorted merge) */
static void poly_add_inplace(Poly* dst, const Poly* src) {
    g_tw_stats.n_poly_add_inplace++;
    if (src->n == 0) return;
    int needed = dst->n + src->n;
    padd_buf_ensure(needed);
    uint16_t* tmp_k = tl_padd_buf_k;
    int64_t*  tmp_v = tl_padd_buf_v;
    int di = 0, si = 0, wi = 0;
    int dn = dst->n, sn = src->n;
    while (di < dn && si < sn) {
        if (dst->keys[di] < src->keys[si]) {
            tmp_k[wi] = dst->keys[di]; tmp_v[wi] = dst->vals[di]; di++; wi++;
        } else if (dst->keys[di] > src->keys[si]) {
            tmp_k[wi] = src->keys[si]; tmp_v[wi] = src->vals[si]; si++; wi++;
        } else {
            int64_t s = coeff_add(dst->vals[di], src->vals[si]);
            if (s != 0) { tmp_k[wi] = dst->keys[di]; tmp_v[wi] = s; wi++; }
            di++; si++;
        }
    }
    while (di < dn) { tmp_k[wi] = dst->keys[di]; tmp_v[wi] = dst->vals[di]; di++; wi++; }
    while (si < sn) { tmp_k[wi] = src->keys[si]; tmp_v[wi] = src->vals[si]; si++; wi++; }
    poly_grow(dst, wi);
    memcpy(dst->keys, tmp_k, wi * sizeof(uint16_t));
    memcpy(dst->vals, tmp_v, wi * sizeof(int64_t));
    dst->n = wi;
    /* No heap free needed — buffer is thread-local and reused. */
}

/* Shift all keys by delta */
static Poly* poly_shift(const Poly* p, int delta_key) {
    g_tw_stats.n_poly_shift++;
    Poly* r = poly_alloc(p->n > 0 ? p->n : 64);
    int i;
    for (i = 0; i < p->n; i++) {
        r->keys[i] = p->keys[i] + delta_key;
        r->vals[i] = p->vals[i];
    }
    r->n = p->n;
    return r;
}

/* Thread-local hash-table buffer for poly_mul. Avoids per-call calloc of
   large hash tables (a dominant bottleneck profiled at ~24% of DP time).
   Uses a per-slot generation counter so the buffer can be reused without
   zeroing between calls. */
static __thread uint16_t* tl_htbuf_keys = NULL;
static __thread int64_t*  tl_htbuf_vals = NULL;
static __thread uint32_t* tl_htbuf_gen  = NULL;
static __thread int       tl_htbuf_cap  = 0;
static __thread uint32_t  tl_htbuf_gen_counter = 0;

static inline void htbuf_ensure(int cap) {
    if (cap <= tl_htbuf_cap) return;
    free(tl_htbuf_keys); free(tl_htbuf_vals); free(tl_htbuf_gen);
    tl_htbuf_keys = (uint16_t*)malloc(cap * sizeof(uint16_t));
    tl_htbuf_vals = (int64_t*)malloc(cap * sizeof(int64_t));
    tl_htbuf_gen  = (uint32_t*)calloc(cap, sizeof(uint32_t));
    tl_htbuf_cap = cap;
}

/* Multiply: fast monomial path + general hash-accumulate */
static Poly* poly_mul(const Poly* p, const Poly* q) {
    if (p->n == 0 || q->n == 0) return poly_zero();
    /* Monomial fast paths */
    if (p->n == 1) {
        g_tw_stats.n_poly_mul_monomial++;
        Poly* r = poly_alloc(q->n);
        uint16_t pk = p->keys[0]; int64_t pv = p->vals[0];
        int i;
        for (i = 0; i < q->n; i++) {
            r->keys[i] = pk + q->keys[i];
            r->vals[i] = coeff_mul(pv, q->vals[i]);
        }
        r->n = q->n;
        return r;
    }
    if (q->n == 1) {
        g_tw_stats.n_poly_mul_monomial++;
        Poly* r = poly_alloc(p->n);
        uint16_t qk = q->keys[0]; int64_t qv = q->vals[0];
        int i;
        for (i = 0; i < p->n; i++) {
            r->keys[i] = p->keys[i] + qk;
            r->vals[i] = coeff_mul(p->vals[i], qv);
        }
        r->n = p->n;
        return r;
    }
    /* General: hash table accumulation using thread-local reusable buffer */
    g_tw_stats.n_poly_mul_general++;
    g_tw_stats.n_coeff_mul_inner += (long long)p->n * (long long)q->n;
    int total = p->n * q->n;
    int ht_cap = 1;
    while (ht_cap < total * 4) ht_cap <<= 1;
    if (ht_cap < 256) ht_cap = 256;
    htbuf_ensure(ht_cap);
    uint32_t my_gen = ++tl_htbuf_gen_counter;
    if (my_gen == 0) {
        /* gen_counter wrapped around; clear once and restart */
        memset(tl_htbuf_gen, 0, tl_htbuf_cap * sizeof(uint32_t));
        my_gen = ++tl_htbuf_gen_counter;
    }
    uint16_t* ht_keys = tl_htbuf_keys;
    int64_t*  ht_vals = tl_htbuf_vals;
    uint32_t* ht_gen  = tl_htbuf_gen;
    int mask = ht_cap - 1;
    int result_n = 0;
    int i, j;
    for (i = 0; i < p->n; i++) {
        uint16_t pk = p->keys[i]; int64_t pv = p->vals[i];
        for (j = 0; j < q->n; j++) {
            uint16_t key = pk + q->keys[j];
            int64_t prod = coeff_mul(pv, q->vals[j]);
            uint32_t h = ((uint32_t)key * 2654435761u) & mask;
            while (ht_gen[h] == my_gen) {
                if (ht_keys[h] == key) { ht_vals[h] = coeff_add(ht_vals[h], prod); goto next_mul; }
                h = (h + 1) & mask;
            }
            ht_gen[h] = my_gen; ht_keys[h] = key; ht_vals[h] = prod; result_n++;
            next_mul:;
        }
    }
    /* Extract and sort */
    Poly* r = poly_alloc(result_n > 0 ? result_n : 64);
    int wi = 0;
    for (i = 0; i < ht_cap; i++) {
        if (ht_gen[i] == my_gen && ht_vals[i] != 0) {
            r->keys[wi] = ht_keys[i]; r->vals[wi] = ht_vals[i]; wi++;
        }
    }
    r->n = wi;
    g_tw_stats.sum_poly_mul_n_out += wi;
    /* Insertion sort (polynomials are typically < 500 terms) */
    for (i = 1; i < wi; i++) {
        uint16_t key = r->keys[i]; int64_t val = r->vals[i];
        j = i - 1;
        while (j >= 0 && r->keys[j] > key) {
            r->keys[j+1] = r->keys[j]; r->vals[j+1] = r->vals[j]; j--;
        }
        r->keys[j+1] = key; r->vals[j+1] = val;
    }
    return r;
}

/* =========================================================================
   DP TABLE (open-addressing hash map: uint64_t -> Poly*)
   ========================================================================= */

typedef struct {
    uint64_t* keys;
    Poly**    vals;
    uint8_t*  used;
    int       n;
    int       cap;
    int       mask;
} DPTable;

static DPTable* dpt_alloc(int cap) {
    /* Round up to power of 2 */
    int c = 256;
    while (c < cap) c <<= 1;
    DPTable* t = (DPTable*)malloc(sizeof(DPTable));
    t->keys = (uint64_t*)calloc(c, sizeof(uint64_t));
    t->vals = (Poly**)calloc(c, sizeof(Poly*));
    t->used = (uint8_t*)calloc(c, sizeof(uint8_t));
    t->n = 0; t->cap = c; t->mask = c - 1;
    return t;
}

static void dpt_free(DPTable* t) {
    if (!t) return;
    int i;
    for (i = 0; i < t->cap; i++)
        if (t->used[i]) poly_free(t->vals[i]);
    free(t->keys); free(t->vals); free(t->used); free(t);
}

static Poly* dpt_get(DPTable* t, uint64_t key) {
    uint32_t h = (uint32_t)(key * 0x9E3779B97F4A7C15ULL >> 43) & t->mask;
    while (t->used[h]) {
        if (t->keys[h] == key) return t->vals[h];
        h = (h + 1) & t->mask;
    }
    return NULL;
}

static void dpt_set(DPTable* t, uint64_t key, Poly* val) {
    /* Rehash if > 50% full */
    if (t->n * 2 >= t->cap) {
        int old_cap = t->cap;
        uint64_t* old_keys = t->keys;
        Poly** old_vals = t->vals;
        uint8_t* old_used = t->used;
        int new_cap = old_cap * 2;
        t->keys = (uint64_t*)calloc(new_cap, sizeof(uint64_t));
        t->vals = (Poly**)calloc(new_cap, sizeof(Poly*));
        t->used = (uint8_t*)calloc(new_cap, sizeof(uint8_t));
        t->cap = new_cap; t->mask = new_cap - 1; t->n = 0;
        int i;
        for (i = 0; i < old_cap; i++) {
            if (old_used[i]) dpt_set(t, old_keys[i], old_vals[i]);
        }
        free(old_keys); free(old_vals); free(old_used);
    }
    uint32_t h = (uint32_t)(key * 0x9E3779B97F4A7C15ULL >> 43) & t->mask;
    while (t->used[h]) {
        if (t->keys[h] == key) { poly_free(t->vals[h]); t->vals[h] = val; return; }
        h = (h + 1) & t->mask;
    }
    t->used[h] = 1; t->keys[h] = key; t->vals[h] = val; t->n++;
}

/* Add poly to existing entry (or create new) */
static void dpt_add(DPTable* t, uint64_t key, const Poly* val) {
    g_tw_stats.n_dpt_add++;
    Poly* existing = dpt_get(t, key);
    if (existing) { poly_add_inplace(existing, val); }
    else { dpt_set(t, key, poly_copy(val)); }
}

/* =========================================================================
   CHILD CONNECTIVITY KEY
   ========================================================================= */

/* Extract connectivity pairs from encoded child partition.
   Returns packed pairs and count. */
typedef struct { int pairs[120]; int n_pairs; } ConnKey;

static ConnKey child_conn_key(uint64_t enc, int n_shared) {
    ConnKey ck; ck.n_pairs = 0;
    int i, j;
    for (i = 0; i < n_shared; i++) {
        int li = (int)((enc >> (4 + i*4)) & 0xF);
        for (j = i + 1; j < n_shared; j++) {
            int lj = (int)((enc >> (4 + j*4)) & 0xF);
            if (li == lj) {
                ck.pairs[ck.n_pairs*2] = i;
                ck.pairs[ck.n_pairs*2+1] = j;
                ck.n_pairs++;
            }
        }
    }
    return ck;
}

/* Encode ConnKey into a single uint64 for hashing (up to 10 pairs) */
static uint64_t conn_key_hash(const ConnKey* ck) {
    uint64_t h = 0;
    int i;
    for (i = 0; i < ck->n_pairs; i++) {
        h = h * 31 + ck->pairs[i*2] * 16 + ck->pairs[i*2+1];
    }
    return h;
}

static int conn_key_eq(const ConnKey* a, const ConnKey* b) {
    if (a->n_pairs != b->n_pairs) return 0;
    return memcmp(a->pairs, b->pairs, a->n_pairs * 2 * sizeof(int)) == 0;
}

/* =========================================================================
   MERGE TABLES
   ========================================================================= */

/* Apply conn_key merges to a partition */
static uint64_t apply_conn_merge(uint64_t enc, const ConnKey* ck,
                                  const int* shared_pos) {
    int n = (int)(enc & 0xF);
    int labels[16]; int i;
    for (i = 0; i < n; i++) labels[i] = (int)((enc >> (4+i*4)) & 0xF);
    int p;
    for (p = 0; p < ck->n_pairs; p++) {
        int pi = shared_pos[ck->pairs[p*2]];
        int pj = shared_pos[ck->pairs[p*2+1]];
        int lpi = labels[pi], lpj = labels[pj];
        if (lpi != lpj) {
            int target = lpi < lpj ? lpi : lpj;
            int replace = lpi < lpj ? lpj : lpi;
            for (i = 0; i < n; i++)
                if (labels[i] == replace) labels[i] = target;
        }
    }
    int mapping[16]; int seen[16];
    memset(seen, 0, sizeof(seen));
    int next_id = 0;
    uint64_t result = n;
    for (i = 0; i < n; i++) {
        int lbl = labels[i];
        if (!seen[lbl]) { seen[lbl] = 1; mapping[lbl] = next_id++; }
        result |= ((uint64_t)mapping[lbl]) << (4 + i*4);
    }
    return result;
}

static DPTable* merge_tables(DPTable* parent, DPTable* child,
                              const int* parent_verts, int n_parent,
                              const int* shared_verts, int n_shared) {
    g_tw_stats.n_merge_tables++;
    if (n_shared == 0) {
        /* No shared: multiply by sum of child polys */
        Poly* child_sum = poly_zero();
        int i;
        for (i = 0; i < child->cap; i++)
            if (child->used[i]) poly_add_inplace(child_sum, child->vals[i]);
        DPTable* result = dpt_alloc(parent->n * 2);
        for (i = 0; i < parent->cap; i++) {
            if (!parent->used[i]) continue;
            Poly* prod = poly_mul(parent->vals[i], child_sum);
            if (prod->n > 0) dpt_set(result, parent->keys[i], prod);
            else poly_free(prod);
        }
        poly_free(child_sum);
        return result;
    }

    /* Build shared positions in parent */
    int shared_pos[16];
    int si, pi;
    for (si = 0; si < n_shared; si++) {
        for (pi = 0; pi < n_parent; pi++) {
            if (parent_verts[pi] == shared_verts[si]) {
                shared_pos[si] = pi; break;
            }
        }
    }

    /* Group child entries by connectivity key.
       Use simple array since Bell(11) = 678K possible keys
       but typically only 5-50 unique conn_keys. */
    /* Use heap for conn groups (Bell numbers can exceed stack limits) */
    int max_conn_groups = child->n + 1;
    if (max_conn_groups < 256) max_conn_groups = 256;
    ConnKey* cg_keys = (ConnKey*)malloc(max_conn_groups * sizeof(ConnKey));
    Poly**   cg_polys = (Poly**)malloc(max_conn_groups * sizeof(Poly*));
    int      n_groups = 0;
    int i;

    for (i = 0; i < child->cap; i++) {
        if (!child->used[i]) continue;
        ConnKey ck = child_conn_key(child->keys[i], n_shared);
        /* Find or create group */
        int found = -1, g;
        for (g = 0; g < n_groups; g++) {
            if (conn_key_eq(&cg_keys[g], &ck)) { found = g; break; }
        }
        if (found >= 0) {
            poly_add_inplace(cg_polys[found], child->vals[i]);
        } else {
            if (n_groups >= max_conn_groups) {
                max_conn_groups *= 2;
                cg_keys = (ConnKey*)realloc(cg_keys, max_conn_groups * sizeof(ConnKey));
                cg_polys = (Poly**)realloc(cg_polys, max_conn_groups * sizeof(Poly*));
            }
            cg_keys[n_groups] = ck;
            cg_polys[n_groups] = poly_copy(child->vals[i]);
            n_groups++;
        }
    }

    DPTable* result = dpt_alloc(parent->n * 2);

    g_tw_stats.sum_merge_parent_n += parent->n;
    g_tw_stats.sum_merge_n_groups += n_groups;

    struct timespec ts_a, ts_b, ts_c;
    clock_gettime(CLOCK_MONOTONIC, &ts_a);

    for (int gi = 0; gi < n_groups; gi++) {
        /* Group parents by output merged partition */
        DPTable* pbo = dpt_alloc(parent->n);
        for (i = 0; i < parent->cap; i++) {
            if (!parent->used[i]) continue;
            uint64_t merged = apply_conn_merge(parent->keys[i], &cg_keys[gi], shared_pos);
            dpt_add(pbo, merged, parent->vals[i]);
        }
        g_tw_stats.sum_merge_pbo_n_total += pbo->n;
        g_tw_stats.n_merge_group_iters++;
        clock_gettime(CLOCK_MONOTONIC, &ts_b);
        /* Multiply each grouped parent by child group poly */
        for (i = 0; i < pbo->cap; i++) {
            if (!pbo->used[i]) continue;
            Poly* prod = poly_mul(pbo->vals[i], cg_polys[gi]);
            if (prod->n > 0) dpt_add(result, pbo->keys[i], prod);
            poly_free(prod);
        }
        dpt_free(pbo);
        clock_gettime(CLOCK_MONOTONIC, &ts_c);
        g_tw_stats.ns_pbo_build += (ts_b.tv_sec - ts_a.tv_sec) * 1000000000LL
                                 + (ts_b.tv_nsec - ts_a.tv_nsec);
        g_tw_stats.ns_pbo_collect += (ts_c.tv_sec - ts_b.tv_sec) * 1000000000LL
                                   + (ts_c.tv_nsec - ts_b.tv_nsec);
        ts_a = ts_c;
    }

    for (i = 0; i < n_groups; i++) poly_free(cg_polys[i]);
    free(cg_keys); free(cg_polys);
    return result;
}

/* =========================================================================
   PROCESS BAG (recursive DP)
   ========================================================================= */

typedef struct {
    int n_bags;
    const int* bag_sizes;
    const int* bag_verts_flat;  /* cumulative offset */
    int* bag_verts_offsets;     /* precomputed offsets */
    const int* children_counts;
    const int* children_flat;
    int* children_offsets;
    const int* bag_edge_counts;
    const int* edges_flat;      /* (u, v, mult) triples */
    int* edges_offsets;
} TDInfo;

/* Forward declaration */
static DPTable* process_bag(TDInfo* td, int bag_idx, int** out_verts, int* out_n);

static DPTable* process_bag(TDInfo* td, int bag_idx, int** out_verts, int* out_n) {
    g_tw_stats.n_process_bag_calls++;
    int bag_size = td->bag_sizes[bag_idx];
    const int* bag_v = td->bag_verts_flat + td->bag_verts_offsets[bag_idx];

    /* Sort bag vertices */
    int bag_verts[16]; int bv;
    memcpy(bag_verts, bag_v, bag_size * sizeof(int));
    /* Simple sort */
    for (int a = 0; a < bag_size - 1; a++)
        for (int b = a+1; b < bag_size; b++)
            if (bag_verts[a] > bag_verts[b]) {
                int tmp = bag_verts[a]; bag_verts[a] = bag_verts[b]; bag_verts[b] = tmp;
            }

    int vert_to_idx[10000]; /* sparse: vert_to_idx[v] = index in bag */
    memset(vert_to_idx, -1, sizeof(vert_to_idx));
    for (bv = 0; bv < bag_size; bv++) vert_to_idx[bag_verts[bv]] = bv;

    /* Init table: all singletons, coefficient 1 */
    uint64_t init_part = bag_size;
    for (bv = 0; bv < bag_size; bv++) init_part |= ((uint64_t)bv) << (4 + bv*4);
    DPTable* table = dpt_alloc(256);
    dpt_set(table, init_part, poly_one());

    /* Classify edges */
    int n_edges = td->bag_edge_counts[bag_idx];
    const int* edges = td->edges_flat + td->edges_offsets[bag_idx];

    int loop_u[64], loop_m[64]; int n_loops = 0;
    int par_u[64], par_v[64], par_m[64]; int n_par = 0;
    int sim_u[64], sim_v[64]; int n_sim = 0;

    for (int e = 0; e < n_edges; e++) {
        int u = edges[e*3], v = edges[e*3+1], m = edges[e*3+2];
        if (vert_to_idx[u] < 0 || vert_to_idx[v] < 0) continue;
        if (u == v) { loop_u[n_loops] = u; loop_m[n_loops] = m; n_loops++; }
        else if (m > 1) { par_u[n_par] = u; par_v[n_par] = v; par_m[n_par] = m; n_par++; }
        else { sim_u[n_sim] = u; sim_v[n_sim] = v; n_sim++; }
    }

    /* Process loops */
    for (int li = 0; li < n_loops; li++) {
        int mult = loop_m[li];
        /* factor = (1+b)^mult = sum C(mult,j) b^j */
        Poly* factor = poly_alloc(mult + 1);
        int64_t c = 1;
        for (int j = 0; j <= mult; j++) {
            factor->keys[j] = (uint16_t)j; /* b^j: a=0, b=j, key=j */
            factor->vals[j] = coeff_from_int(c);
            c = c * (mult - j) / (j + 1);
        }
        factor->n = mult + 1;

        DPTable* nt = dpt_alloc(table->n * 2);
        for (int i = 0; i < table->cap; i++) {
            if (!table->used[i]) continue;
            Poly* prod = poly_mul(table->vals[i], factor);
            if (prod->n > 0) dpt_add(nt, table->keys[i], prod);
            poly_free(prod);
        }
        dpt_free(table); table = nt;
        poly_free(factor);
    }

    /* Process parallel edges */
    for (int pi = 0; pi < n_par; pi++) {
        int iu = vert_to_idx[par_u[pi]], iv = vert_to_idx[par_v[pi]];
        int mult = par_m[pi];
        Poly* ff = poly_alloc(mult+1);
        int64_t c = 1;
        for (int j = 0; j <= mult; j++) {
            ff->keys[j] = (uint16_t)j; ff->vals[j] = coeff_from_int(c);
            c = c * (mult - j) / (j + 1);
        }
        ff->n = mult + 1;
        Poly* fm1 = poly_copy(ff);
        fm1->vals[0] = coeff_add(fm1->vals[0], coeff_neg(coeff_from_int(1)));
        if (fm1->vals[0] == 0) {
            /* Remove zero entry by shifting */
            memmove(fm1->keys, fm1->keys+1, (fm1->n-1)*sizeof(uint16_t));
            memmove(fm1->vals, fm1->vals+1, (fm1->n-1)*sizeof(int64_t));
            fm1->n--;
        }

        DPTable* nt = dpt_alloc(table->n * 4);
        for (int i = 0; i < table->cap; i++) {
            if (!table->used[i]) continue;
            uint64_t cp = cached_connect(table->keys[i], iu, iv);
            if (cp == table->keys[i]) {
                Poly* prod = poly_mul(table->vals[i], ff);
                if (prod->n > 0) dpt_add(nt, table->keys[i], prod);
                poly_free(prod);
            } else {
                dpt_add(nt, table->keys[i], table->vals[i]);
                Poly* prod = poly_mul(table->vals[i], fm1);
                if (prod->n > 0) dpt_add(nt, cp, prod);
                poly_free(prod);
            }
        }
        dpt_free(table); table = nt;
        poly_free(ff); poly_free(fm1);
    }

    /* Batch simple edges (k <= 12) */
    if (n_sim > 0 && n_sim <= 12) {
        int eidx[12][2];
        for (int e = 0; e < n_sim; e++) {
            eidx[e][0] = vert_to_idx[sim_u[e]];
            eidx[e][1] = vert_to_idx[sim_v[e]];
        }
        int k = n_sim;
        DPTable* nt = dpt_alloc(table->n * (1 << (k > 4 ? 4 : k)));
        for (int i = 0; i < table->cap; i++) {
            if (!table->used[i]) continue;
            uint64_t base_part = table->keys[i];
            Poly* base_poly = table->vals[i];
            for (int mask = 0; mask < (1 << k); mask++) {
                uint64_t p = base_part;
                for (int bit = 0; bit < k; bit++) {
                    if (mask & (1 << bit))
                        p = cached_connect(p, eidx[bit][0], eidx[bit][1]);
                }
                int bc = __builtin_popcount(mask);
                Poly* shifted;
                if (bc > 0) {
                    shifted = poly_shift(base_poly, (uint16_t)bc);
                } else {
                    shifted = poly_copy(base_poly);
                }
                dpt_add(nt, p, shifted);
                poly_free(shifted);
            }
        }
        dpt_free(table); table = nt;
    } else if (n_sim > 12) {
        for (int e = 0; e < n_sim; e++) {
            int iu = vert_to_idx[sim_u[e]], iv = vert_to_idx[sim_v[e]];
            DPTable* nt = dpt_alloc(table->n * 4);
            for (int i = 0; i < table->cap; i++) {
                if (!table->used[i]) continue;
                dpt_add(nt, table->keys[i], table->vals[i]);
                uint64_t np = cached_connect(table->keys[i], iu, iv);
                Poly* shifted = poly_shift(table->vals[i], 1); /* multiply by b */
                dpt_add(nt, np, shifted);
                poly_free(shifted);
            }
            dpt_free(table); table = nt;
        }
    }

    /* Process children */
    int n_children = td->children_counts[bag_idx];
    const int* child_ids = td->children_flat + td->children_offsets[bag_idx];

    for (int ci = 0; ci < n_children; ci++) {
        int child_idx = child_ids[ci];
        int* child_verts; int child_n;
        DPTable* child_table = process_bag(td, child_idx, &child_verts, &child_n);

        /* Compute forget indices */
        int forget_pos[16]; int n_forget = 0;
        int offset = 0;
        for (int cv = 0; cv < child_n; cv++) {
            int in_bag = 0;
            for (bv = 0; bv < bag_size; bv++)
                if (bag_verts[bv] == child_verts[cv]) { in_bag = 1; break; }
            if (!in_bag) { forget_pos[n_forget++] = cv - offset; offset++; }
        }

        /* Forget vertices from child table */
        DPTable* forgotten = dpt_alloc(child_table->n * 2);
        for (int i = 0; i < child_table->cap; i++) {
            if (!child_table->used[i]) continue;
            uint64_t cur_enc = child_table->keys[i];
            Poly* cur_poly = poly_copy(child_table->vals[i]);
            for (int fi = 0; fi < n_forget; fi++) {
                ForgetResult fr = encoded_forget(cur_enc, forget_pos[fi]);
                if (fr.is_singleton) {
                    Poly* shifted = poly_shift(cur_poly, KEY_STRIDE + 1);
                    poly_free(cur_poly);
                    cur_poly = shifted;
                }
                cur_enc = fr.new_enc;
            }
            dpt_add(forgotten, cur_enc, cur_poly);
            poly_free(cur_poly);
        }

        /* Shared vertices */
        int shared[16]; int n_shared = 0;
        for (int cv = 0; cv < child_n; cv++) {
            for (bv = 0; bv < bag_size; bv++) {
                if (bag_verts[bv] == child_verts[cv]) {
                    shared[n_shared++] = child_verts[cv]; break;
                }
            }
        }

        /* Merge */
        DPTable* merged = merge_tables(table, forgotten, bag_verts, bag_size,
                                        shared, n_shared);
        dpt_free(table); table = merged;
        dpt_free(forgotten);
        dpt_free(child_table);
        free(child_verts);
    }

    /* Return */
    *out_verts = (int*)malloc(bag_size * sizeof(int));
    memcpy(*out_verts, bag_verts, bag_size * sizeof(int));
    *out_n = bag_size;
    return table;
}

/* =========================================================================
   MAIN ENTRY POINT
   ========================================================================= */

int treewidth_tutte_dp(
    int n_bags, const int* bag_sizes, const int* bag_verts_flat,
    int root, const int* children_counts, const int* children_flat,
    const int* bag_edge_counts, const int* edges_flat,
    int n_verts, int n_components,
    int* out_xy, long long* out_coeffs, int* out_n_terms, int max_out,
    long long dp_prime, long long conv_prime)
{
    g_prime = dp_prime;

    /* Clear connect cache */
    memset(ccache, 0, sizeof(ccache));

    /* Precompute offsets */
    TDInfo td;
    td.n_bags = n_bags;
    td.bag_sizes = bag_sizes;
    td.bag_verts_flat = bag_verts_flat;
    td.children_counts = children_counts;
    td.children_flat = children_flat;
    td.bag_edge_counts = bag_edge_counts;
    td.edges_flat = edges_flat;

    td.bag_verts_offsets = (int*)malloc(n_bags * sizeof(int));
    td.children_offsets = (int*)malloc(n_bags * sizeof(int));
    td.edges_offsets = (int*)malloc(n_bags * sizeof(int));

    int off = 0;
    for (int i = 0; i < n_bags; i++) { td.bag_verts_offsets[i] = off; off += bag_sizes[i]; }
    off = 0;
    for (int i = 0; i < n_bags; i++) { td.children_offsets[i] = off; off += children_counts[i]; }
    off = 0;
    for (int i = 0; i < n_bags; i++) { td.edges_offsets[i] = off; off += bag_edge_counts[i] * 3; }

    /* Run DP */
    int* root_verts; int root_n;
    DPTable* root_table = process_bag(&td, root, &root_verts, &root_n);

    /* Forget all root vertices */
    Poly* final_poly = poly_zero();
    for (int i = 0; i < root_table->cap; i++) {
        if (!root_table->used[i]) continue;
        uint64_t cur_enc = root_table->keys[i];
        Poly* cur_poly = poly_copy(root_table->vals[i]);
        for (int j = 0; j < root_n; j++) {
            ForgetResult fr = encoded_forget(cur_enc, 0);
            if (fr.is_singleton) {
                Poly* shifted = poly_shift(cur_poly, KEY_STRIDE + 1);
                poly_free(cur_poly);
                cur_poly = shifted;
            }
            cur_enc = fr.new_enc;
        }
        poly_add_inplace(final_poly, cur_poly);
        poly_free(cur_poly);
    }

    /* Shift exponents: divide by a^{n_components} * b^{n_verts} */
    int a_shift = n_components;
    int b_shift = n_verts;
    /* Build shifted (a,b) polynomial */
    Poly* ab_poly = poly_alloc(final_poly->n);
    for (int i = 0; i < final_poly->n; i++) {
        int a_pow = final_poly->keys[i] / KEY_STRIDE;
        int b_pow = final_poly->keys[i] % KEY_STRIDE;
        int new_a = a_pow - a_shift;
        int new_b = b_pow - b_shift;
        if (new_a < 0 || new_b < 0) continue;
        ab_poly->keys[ab_poly->n] = (uint16_t)(new_a * KEY_STRIDE + new_b);
        ab_poly->vals[ab_poly->n] = final_poly->vals[i];
        ab_poly->n++;
    }

    /* Switch to conversion prime for the basis conversion phase.
       This allows the DP to run in exact int64 while handling binomial overflow. */
    g_prime = conv_prime;

    /* Convert (a,b) = (x-1, y-1) to (x,y) via binomial expansion.
       Binomial coefficients use modular arithmetic (coeff_add) to avoid
       int64 overflow for large n (C(164,82) >> 2^63). */
    int max_ab = 0;
    for (int ti = 0; ti < ab_poly->n; ti++) {
        int a = ab_poly->keys[ti] / KEY_STRIDE;
        int b = ab_poly->keys[ti] % KEY_STRIDE;
        if (a > max_ab) max_ab = a;
        if (b > max_ab) max_ab = b;
    }
    int binom_size = max_ab + 1;
    int64_t* binom_flat = (int64_t*)calloc((size_t)binom_size * binom_size, sizeof(int64_t));
    #define BINOM(n,k) binom_flat[(n) * binom_size + (k)]
    for (int nn = 0; nn < binom_size; nn++) {
        BINOM(nn, 0) = coeff_from_int(1);
        for (int kk = 1; kk <= nn; kk++)
            BINOM(nn, kk) = coeff_add(BINOM(nn-1, kk-1), BINOM(nn-1, kk));
    }

    int ht_cap = 65536;
    uint32_t* ht_keys_xy = (uint32_t*)calloc(ht_cap, sizeof(uint32_t));
    int64_t*  ht_vals_xy = (int64_t*)calloc(ht_cap, sizeof(int64_t));
    uint8_t*  ht_used_xy = (uint8_t*)calloc(ht_cap, sizeof(uint8_t));
    int xy_mask = ht_cap - 1;

    for (int ti = 0; ti < ab_poly->n; ti++) {
        int a_pow = ab_poly->keys[ti] / KEY_STRIDE;
        int b_pow = ab_poly->keys[ti] % KEY_STRIDE;
        int64_t coeff = ab_poly->vals[ti];

        for (int p = 0; p <= a_pow; p++) {
            /* C(a_pow, p) * (-1)^(a_pow - p) */
            int64_t cx = BINOM(a_pow, p);
            if ((a_pow - p) % 2 != 0) cx = coeff_neg(cx);

            for (int q = 0; q <= b_pow; q++) {
                int64_t cy = BINOM(b_pow, q);
                if ((b_pow - q) % 2 != 0) cy = coeff_neg(cy);

                int64_t contribution = coeff_mul(coeff, coeff_mul(cx, cy));
                if (contribution != 0) {
                    uint32_t key = (uint32_t)(p * 256 + q);
                    uint32_t h = (key * 2654435761u) & xy_mask;
                    while (ht_used_xy[h]) {
                        if (ht_keys_xy[h] == key) { ht_vals_xy[h] = coeff_add(ht_vals_xy[h], contribution); goto next_xy; }
                        h = (h + 1) & xy_mask;
                    }
                    ht_used_xy[h] = 1; ht_keys_xy[h] = key; ht_vals_xy[h] = contribution;
                    next_xy:;
                }
            }
        }
    }

    /* Extract results */
    int n_out = 0;
    for (int i = 0; i < ht_cap; i++) {
        if (ht_used_xy[i] && ht_vals_xy[i] != 0 && n_out < max_out) {
            out_xy[n_out * 2] = ht_keys_xy[i] / 256;      /* x power */
            out_xy[n_out * 2 + 1] = ht_keys_xy[i] % 256;  /* y power */
            out_coeffs[n_out] = ht_vals_xy[i];
            n_out++;
        }
    }
    *out_n_terms = n_out;

    /* Cleanup */
    free(ht_keys_xy); free(ht_vals_xy); free(ht_used_xy);
    poly_free(ab_poly);
    free(binom_flat);
    #undef BINOM
    poly_free(final_poly);
    dpt_free(root_table);
    free(root_verts);
    free(td.bag_verts_offsets);
    free(td.children_offsets);
    free(td.edges_offsets);

    return 0;
}

/* Run DP only — return (a,b)-basis polynomial without basis conversion.
   Used when we want to run the DP once (in int64) and convert with
   multiple primes via basis_convert_ab_to_xy. */
int treewidth_tutte_dp_ab(
    int n_bags, const int* bag_sizes, const int* bag_verts_flat,
    int root, const int* children_counts, const int* children_flat,
    const int* bag_edge_counts, const int* edges_flat,
    int n_verts, int n_components,
    int* out_ab, long long* out_coeffs, int* out_n_terms, int max_out)
{
    g_prime = 0;  /* DP in exact int64 */
    memset(ccache, 0, sizeof(ccache));

    TDInfo td;
    td.n_bags = n_bags;
    td.bag_sizes = bag_sizes;
    td.bag_verts_flat = bag_verts_flat;
    td.children_counts = children_counts;
    td.children_flat = children_flat;
    td.bag_edge_counts = bag_edge_counts;
    td.edges_flat = edges_flat;
    td.bag_verts_offsets = (int*)malloc(n_bags * sizeof(int));
    td.children_offsets = (int*)malloc(n_bags * sizeof(int));
    td.edges_offsets = (int*)malloc(n_bags * sizeof(int));
    int off = 0;
    for (int i = 0; i < n_bags; i++) { td.bag_verts_offsets[i] = off; off += bag_sizes[i]; }
    off = 0;
    for (int i = 0; i < n_bags; i++) { td.children_offsets[i] = off; off += children_counts[i]; }
    off = 0;
    for (int i = 0; i < n_bags; i++) { td.edges_offsets[i] = off; off += bag_edge_counts[i] * 3; }

    int* root_verts; int root_n;
    DPTable* root_table = process_bag(&td, root, &root_verts, &root_n);

    /* Forget all root vertices */
    Poly* final_poly = poly_zero();
    for (int i = 0; i < root_table->cap; i++) {
        if (!root_table->used[i]) continue;
        uint64_t cur_enc = root_table->keys[i];
        Poly* cur_poly = poly_copy(root_table->vals[i]);
        for (int j = 0; j < root_n; j++) {
            ForgetResult fr = encoded_forget(cur_enc, 0);
            if (fr.is_singleton) {
                Poly* shifted = poly_shift(cur_poly, KEY_STRIDE + 1);
                poly_free(cur_poly);
                cur_poly = shifted;
            }
            cur_enc = fr.new_enc;
        }
        poly_add_inplace(final_poly, cur_poly);
        poly_free(cur_poly);
    }

    /* Shift exponents: divide by a^{n_components} * b^{n_verts} */
    int a_shift = n_components;
    int b_shift = n_verts;
    int n_out = 0;
    for (int i = 0; i < final_poly->n && n_out < max_out; i++) {
        int a_pow = final_poly->keys[i] / KEY_STRIDE;
        int b_pow = final_poly->keys[i] % KEY_STRIDE;
        int new_a = a_pow - a_shift;
        int new_b = b_pow - b_shift;
        if (new_a < 0 || new_b < 0) continue;
        out_ab[n_out * 2] = new_a;
        out_ab[n_out * 2 + 1] = new_b;
        out_coeffs[n_out] = final_poly->vals[i];
        n_out++;
    }
    *out_n_terms = n_out;

    poly_free(final_poly);
    dpt_free(root_table);
    free(root_verts);
    free(td.bag_verts_offsets);
    free(td.children_offsets);
    free(td.edges_offsets);
    return 0;
}

/* Modular variant of treewidth_tutte_dp_ab. Runs DP mod dp_prime; coefficients
   emitted are residues in [0, dp_prime). For graphs too large for int128 DP;
   Python CRT-reconstructs (a,b) across primes then converts basis. */
int treewidth_tutte_dp_ab_mod(
    int n_bags, const int* bag_sizes, const int* bag_verts_flat,
    int root, const int* children_counts, const int* children_flat,
    const int* bag_edge_counts, const int* edges_flat,
    int n_verts, int n_components,
    int* out_ab, long long* out_coeffs, int* out_n_terms, int max_out,
    long long dp_prime)
{
    g_prime = dp_prime;
    memset(ccache, 0, sizeof(ccache));

    TDInfo td;
    td.n_bags = n_bags;
    td.bag_sizes = bag_sizes;
    td.bag_verts_flat = bag_verts_flat;
    td.children_counts = children_counts;
    td.children_flat = children_flat;
    td.bag_edge_counts = bag_edge_counts;
    td.edges_flat = edges_flat;
    td.bag_verts_offsets = (int*)malloc(n_bags * sizeof(int));
    td.children_offsets = (int*)malloc(n_bags * sizeof(int));
    td.edges_offsets = (int*)malloc(n_bags * sizeof(int));
    int off = 0;
    for (int i = 0; i < n_bags; i++) { td.bag_verts_offsets[i] = off; off += bag_sizes[i]; }
    off = 0;
    for (int i = 0; i < n_bags; i++) { td.children_offsets[i] = off; off += children_counts[i]; }
    off = 0;
    for (int i = 0; i < n_bags; i++) { td.edges_offsets[i] = off; off += bag_edge_counts[i] * 3; }

    int* root_verts; int root_n;
    DPTable* root_table = process_bag(&td, root, &root_verts, &root_n);

    Poly* final_poly = poly_zero();
    for (int i = 0; i < root_table->cap; i++) {
        if (!root_table->used[i]) continue;
        uint64_t cur_enc = root_table->keys[i];
        Poly* cur_poly = poly_copy(root_table->vals[i]);
        for (int j = 0; j < root_n; j++) {
            ForgetResult fr = encoded_forget(cur_enc, 0);
            if (fr.is_singleton) {
                Poly* shifted = poly_shift(cur_poly, KEY_STRIDE + 1);
                poly_free(cur_poly);
                cur_poly = shifted;
            }
            cur_enc = fr.new_enc;
        }
        poly_add_inplace(final_poly, cur_poly);
        poly_free(cur_poly);
    }

    int a_shift = n_components;
    int b_shift = n_verts;
    int n_out = 0;
    for (int i = 0; i < final_poly->n && n_out < max_out; i++) {
        int a_pow = final_poly->keys[i] / KEY_STRIDE;
        int b_pow = final_poly->keys[i] % KEY_STRIDE;
        int new_a = a_pow - a_shift;
        int new_b = b_pow - b_shift;
        if (new_a < 0 || new_b < 0) continue;
        out_ab[n_out * 2] = new_a;
        out_ab[n_out * 2 + 1] = new_b;
        out_coeffs[n_out] = final_poly->vals[i];
        n_out++;
    }
    *out_n_terms = n_out;

    poly_free(final_poly);
    dpt_free(root_table);
    free(root_verts);
    free(td.bag_verts_offsets);
    free(td.children_offsets);
    free(td.edges_offsets);
    return 0;
}

/* Pseudo-Mersenne variant: runs DP mod dp_prime where dp_prime = 2^62 - dp_prime_c.
   coeff_mul uses fast high-low split reduction via dp_prime_c (~3x faster than
   generic __int128 modulo). Emits residues in [0, dp_prime). */
int treewidth_tutte_dp_ab_mersenne(
    int n_bags, const int* bag_sizes, const int* bag_verts_flat,
    int root, const int* children_counts, const int* children_flat,
    const int* bag_edge_counts, const int* edges_flat,
    int n_verts, int n_components,
    int* out_ab, long long* out_coeffs, int* out_n_terms, int max_out,
    long long dp_prime, long long dp_prime_c)
{
    g_prime = dp_prime;
    g_prime_c = dp_prime_c;
    memset(ccache, 0, sizeof(ccache));

    TDInfo td;
    td.n_bags = n_bags;
    td.bag_sizes = bag_sizes;
    td.bag_verts_flat = bag_verts_flat;
    td.children_counts = children_counts;
    td.children_flat = children_flat;
    td.bag_edge_counts = bag_edge_counts;
    td.edges_flat = edges_flat;
    td.bag_verts_offsets = (int*)malloc(n_bags * sizeof(int));
    td.children_offsets = (int*)malloc(n_bags * sizeof(int));
    td.edges_offsets = (int*)malloc(n_bags * sizeof(int));
    int off = 0;
    for (int i = 0; i < n_bags; i++) { td.bag_verts_offsets[i] = off; off += bag_sizes[i]; }
    off = 0;
    for (int i = 0; i < n_bags; i++) { td.children_offsets[i] = off; off += children_counts[i]; }
    off = 0;
    for (int i = 0; i < n_bags; i++) { td.edges_offsets[i] = off; off += bag_edge_counts[i] * 3; }

    int* root_verts; int root_n;
    DPTable* root_table = process_bag(&td, root, &root_verts, &root_n);

    Poly* final_poly = poly_zero();
    for (int i = 0; i < root_table->cap; i++) {
        if (!root_table->used[i]) continue;
        uint64_t cur_enc = root_table->keys[i];
        Poly* cur_poly = poly_copy(root_table->vals[i]);
        for (int j = 0; j < root_n; j++) {
            ForgetResult fr = encoded_forget(cur_enc, 0);
            if (fr.is_singleton) {
                Poly* shifted = poly_shift(cur_poly, KEY_STRIDE + 1);
                poly_free(cur_poly);
                cur_poly = shifted;
            }
            cur_enc = fr.new_enc;
        }
        poly_add_inplace(final_poly, cur_poly);
        poly_free(cur_poly);
    }

    int a_shift = n_components;
    int b_shift = n_verts;
    int n_out = 0;
    for (int i = 0; i < final_poly->n && n_out < max_out; i++) {
        int a_pow = final_poly->keys[i] / KEY_STRIDE;
        int b_pow = final_poly->keys[i] % KEY_STRIDE;
        int new_a = a_pow - a_shift;
        int new_b = b_pow - b_shift;
        if (new_a < 0 || new_b < 0) continue;
        out_ab[n_out * 2] = new_a;
        out_ab[n_out * 2 + 1] = new_b;
        out_coeffs[n_out] = final_poly->vals[i];
        n_out++;
    }
    *out_n_terms = n_out;

    /* Reset g_prime_c to not leak mersenne mode to other calls on this thread. */
    g_prime_c = 0;

    poly_free(final_poly);
    dpt_free(root_table);
    free(root_verts);
    free(td.bag_verts_offsets);
    free(td.children_offsets);
    free(td.edges_offsets);
    return 0;
}

/* =========================================================================
   __int128 DP VARIANTS (for medium-density graphs: 63-120 edges)
   Exact arithmetic with __int128 — no modular reduction, no overflow
   for coefficients up to ~2^127.
   ========================================================================= */

typedef struct {
    uint16_t* keys;
    __int128* vals;
    int n, cap;
} Poly128;

static Poly128* poly_alloc128(int cap) {
    Poly128* p = (Poly128*)malloc(sizeof(Poly128));
    p->keys = (uint16_t*)calloc(cap, sizeof(uint16_t));
    p->vals = (__int128*)calloc(cap, sizeof(__int128));
    p->n = 0; p->cap = cap; return p;
}
static void poly_free128(Poly128* p) {
    if (p) { free(p->keys); free(p->vals); free(p); }
}
static void poly_grow128(Poly128* p, int needed) {
    if (needed <= p->cap) return;
    int new_cap = p->cap * 2;
    if (new_cap < needed) new_cap = needed;
    p->keys = (uint16_t*)realloc(p->keys, new_cap * sizeof(uint16_t));
    p->vals = (__int128*)realloc(p->vals, new_cap * sizeof(__int128));
    p->cap = new_cap;
}
static Poly128* poly_zero128(void) { return poly_alloc128(64); }
static Poly128* poly_one128(void) {
    Poly128* p = poly_alloc128(64);
    p->keys[0] = 0; p->vals[0] = (__int128)1; p->n = 1; return p;
}
static Poly128* poly_copy128(const Poly128* src) {
    Poly128* dst = poly_alloc128(src->n > 0 ? src->n : 64);
    memcpy(dst->keys, src->keys, src->n * sizeof(uint16_t));
    memcpy(dst->vals, src->vals, src->n * sizeof(__int128));
    dst->n = src->n; return dst;
}
static void poly_add_inplace128(Poly128* dst, const Poly128* src) {
    if (src->n == 0) return;
    int needed = dst->n + src->n;
    uint16_t* tmp_k = (uint16_t*)malloc(needed * sizeof(uint16_t));
    __int128* tmp_v = (__int128*)malloc(needed * sizeof(__int128));
    int di = 0, si = 0, wi = 0, dn = dst->n, sn = src->n;
    while (di < dn && si < sn) {
        if (dst->keys[di] < src->keys[si]) {
            tmp_k[wi] = dst->keys[di]; tmp_v[wi] = dst->vals[di]; di++; wi++;
        } else if (dst->keys[di] > src->keys[si]) {
            tmp_k[wi] = src->keys[si]; tmp_v[wi] = src->vals[si]; si++; wi++;
        } else {
            __int128 s = dst->vals[di] + src->vals[si];
            if (s != 0) { tmp_k[wi] = dst->keys[di]; tmp_v[wi] = s; wi++; }
            di++; si++;
        }
    }
    while (di < dn) { tmp_k[wi] = dst->keys[di]; tmp_v[wi] = dst->vals[di]; di++; wi++; }
    while (si < sn) { tmp_k[wi] = src->keys[si]; tmp_v[wi] = src->vals[si]; si++; wi++; }
    poly_grow128(dst, wi);
    memcpy(dst->keys, tmp_k, wi * sizeof(uint16_t));
    memcpy(dst->vals, tmp_v, wi * sizeof(__int128));
    dst->n = wi; free(tmp_k); free(tmp_v);
}
static Poly128* poly_shift128(const Poly128* p, int delta) {
    Poly128* r = poly_alloc128(p->n > 0 ? p->n : 64);
    int i;
    for (i = 0; i < p->n; i++) {
        r->keys[i] = p->keys[i] + delta; r->vals[i] = p->vals[i];
    }
    r->n = p->n; return r;
}
/* Thread-local hash-table buffer for poly_mul128 (int128 variant). */
static __thread uint16_t* tl_htbuf128_keys = NULL;
static __thread __int128* tl_htbuf128_vals = NULL;
static __thread uint32_t* tl_htbuf128_gen  = NULL;
static __thread int       tl_htbuf128_cap  = 0;
static __thread uint32_t  tl_htbuf128_gen_counter = 0;

static inline void htbuf128_ensure(int cap) {
    if (cap <= tl_htbuf128_cap) return;
    free(tl_htbuf128_keys); free(tl_htbuf128_vals); free(tl_htbuf128_gen);
    tl_htbuf128_keys = (uint16_t*)malloc(cap * sizeof(uint16_t));
    tl_htbuf128_vals = (__int128*)malloc(cap * sizeof(__int128));
    tl_htbuf128_gen  = (uint32_t*)calloc(cap, sizeof(uint32_t));
    tl_htbuf128_cap = cap;
}

static Poly128* poly_mul128(const Poly128* p, const Poly128* q) {
    if (p->n == 0 || q->n == 0) return poly_zero128();
    if (p->n == 1) {
        Poly128* r = poly_alloc128(q->n);
        uint16_t pk = p->keys[0]; __int128 pv = p->vals[0];
        int i;
        for (i = 0; i < q->n; i++) {
            r->keys[i] = pk + q->keys[i]; r->vals[i] = pv * q->vals[i];
        }
        r->n = q->n; return r;
    }
    if (q->n == 1) {
        Poly128* r = poly_alloc128(p->n);
        uint16_t qk = q->keys[0]; __int128 qv = q->vals[0];
        int i;
        for (i = 0; i < p->n; i++) {
            r->keys[i] = p->keys[i] + qk; r->vals[i] = p->vals[i] * qv;
        }
        r->n = p->n; return r;
    }
    int total = p->n * q->n;
    int ht_cap = 1; while (ht_cap < total * 4) ht_cap <<= 1;
    if (ht_cap < 256) ht_cap = 256;
    htbuf128_ensure(ht_cap);
    uint32_t my_gen = ++tl_htbuf128_gen_counter;
    if (my_gen == 0) {
        memset(tl_htbuf128_gen, 0, tl_htbuf128_cap * sizeof(uint32_t));
        my_gen = ++tl_htbuf128_gen_counter;
    }
    uint16_t* ht_keys = tl_htbuf128_keys;
    __int128* ht_vals = tl_htbuf128_vals;
    uint32_t* ht_gen  = tl_htbuf128_gen;
    int mask = ht_cap - 1, result_n = 0, i, j;
    for (i = 0; i < p->n; i++) {
        uint16_t pk = p->keys[i]; __int128 pv = p->vals[i];
        for (j = 0; j < q->n; j++) {
            uint16_t key = pk + q->keys[j];
            __int128 prod = pv * q->vals[j];
            uint32_t h = ((uint32_t)key * 2654435761u) & mask;
            while (ht_gen[h] == my_gen) {
                if (ht_keys[h] == key) { ht_vals[h] += prod; goto next_mul128; }
                h = (h + 1) & mask;
            }
            ht_gen[h] = my_gen; ht_keys[h] = key; ht_vals[h] = prod; result_n++;
            next_mul128:;
        }
    }
    Poly128* r = poly_alloc128(result_n > 0 ? result_n : 64);
    int wi = 0;
    for (i = 0; i < ht_cap; i++) {
        if (ht_gen[i] == my_gen && ht_vals[i] != 0) {
            r->keys[wi] = ht_keys[i]; r->vals[wi] = ht_vals[i]; wi++;
        }
    }
    r->n = wi;
    for (i = 1; i < wi; i++) {
        uint16_t key = r->keys[i]; __int128 val = r->vals[i];
        j = i - 1;
        while (j >= 0 && r->keys[j] > key) {
            r->keys[j+1] = r->keys[j]; r->vals[j+1] = r->vals[j]; j--;
        }
        r->keys[j+1] = key; r->vals[j+1] = val;
    }
    return r;
}

/* DPTable128 — mirrors DPTable but with Poly128 values */
typedef struct {
    uint64_t* keys;
    Poly128** vals;
    uint8_t*  used;
    int n, cap, mask;
} DPTable128;

static DPTable128* dpt_alloc128(int cap) {
    int c = 256; while (c < cap) c <<= 1;
    DPTable128* t = (DPTable128*)malloc(sizeof(DPTable128));
    t->keys = (uint64_t*)calloc(c, sizeof(uint64_t));
    t->vals = (Poly128**)calloc(c, sizeof(Poly128*));
    t->used = (uint8_t*)calloc(c, sizeof(uint8_t));
    t->n = 0; t->cap = c; t->mask = c - 1; return t;
}
static void dpt_free128(DPTable128* t) {
    if (!t) return;
    int i;
    for (i = 0; i < t->cap; i++) if (t->used[i]) poly_free128(t->vals[i]);
    free(t->keys); free(t->vals); free(t->used); free(t);
}
static Poly128* dpt_get128(DPTable128* t, uint64_t key) {
    uint32_t h = (uint32_t)(key * 0x9E3779B97F4A7C15ULL >> 43) & t->mask;
    while (t->used[h]) {
        if (t->keys[h] == key) return t->vals[h];
        h = (h + 1) & t->mask;
    }
    return NULL;
}
static void dpt_set128(DPTable128* t, uint64_t key, Poly128* val) {
    if (t->n * 2 >= t->cap) {
        int old_cap = t->cap;
        uint64_t* old_keys = t->keys;
        Poly128** old_vals = t->vals;
        uint8_t* old_used = t->used;
        int new_cap = old_cap * 2;
        t->keys = (uint64_t*)calloc(new_cap, sizeof(uint64_t));
        t->vals = (Poly128**)calloc(new_cap, sizeof(Poly128*));
        t->used = (uint8_t*)calloc(new_cap, sizeof(uint8_t));
        t->cap = new_cap; t->mask = new_cap - 1; t->n = 0;
        int i;
        for (i = 0; i < old_cap; i++) {
            if (old_used[i]) dpt_set128(t, old_keys[i], old_vals[i]);
        }
        free(old_keys); free(old_vals); free(old_used);
    }
    uint32_t h = (uint32_t)(key * 0x9E3779B97F4A7C15ULL >> 43) & t->mask;
    while (t->used[h]) {
        if (t->keys[h] == key) { poly_free128(t->vals[h]); t->vals[h] = val; return; }
        h = (h + 1) & t->mask;
    }
    t->used[h] = 1; t->keys[h] = key; t->vals[h] = val; t->n++;
}
static void dpt_add128(DPTable128* t, uint64_t key, const Poly128* val) {
    Poly128* existing = dpt_get128(t, key);
    if (existing) poly_add_inplace128(existing, val);
    else dpt_set128(t, key, poly_copy128(val));
}

/* merge_tables128 — mirrors merge_tables but uses Poly128 / DPTable128.
   ConnKey helpers are value-type independent so we reuse them. */
static DPTable128* merge_tables128(DPTable128* parent, DPTable128* child,
                                    const int* parent_verts, int n_parent,
                                    const int* shared_verts, int n_shared) {
    if (n_shared == 0) {
        Poly128* child_sum = poly_zero128();
        int i;
        for (i = 0; i < child->cap; i++)
            if (child->used[i]) poly_add_inplace128(child_sum, child->vals[i]);
        DPTable128* result = dpt_alloc128(parent->n * 2);
        for (i = 0; i < parent->cap; i++) {
            if (!parent->used[i]) continue;
            Poly128* prod = poly_mul128(parent->vals[i], child_sum);
            if (prod->n > 0) dpt_set128(result, parent->keys[i], prod);
            else poly_free128(prod);
        }
        poly_free128(child_sum);
        return result;
    }

    int shared_pos[16];
    int si, pi;
    for (si = 0; si < n_shared; si++) {
        for (pi = 0; pi < n_parent; pi++) {
            if (parent_verts[pi] == shared_verts[si]) {
                shared_pos[si] = pi; break;
            }
        }
    }

    int max_conn_groups = child->n + 1;
    if (max_conn_groups < 256) max_conn_groups = 256;
    ConnKey*  cg_keys  = (ConnKey*)malloc(max_conn_groups * sizeof(ConnKey));
    Poly128** cg_polys = (Poly128**)malloc(max_conn_groups * sizeof(Poly128*));
    int       n_groups = 0;
    int i;

    for (i = 0; i < child->cap; i++) {
        if (!child->used[i]) continue;
        ConnKey ck = child_conn_key(child->keys[i], n_shared);
        int found = -1, g;
        for (g = 0; g < n_groups; g++) {
            if (conn_key_eq(&cg_keys[g], &ck)) { found = g; break; }
        }
        if (found >= 0) {
            poly_add_inplace128(cg_polys[found], child->vals[i]);
        } else {
            if (n_groups >= max_conn_groups) {
                max_conn_groups *= 2;
                cg_keys  = (ConnKey*)realloc(cg_keys, max_conn_groups * sizeof(ConnKey));
                cg_polys = (Poly128**)realloc(cg_polys, max_conn_groups * sizeof(Poly128*));
            }
            cg_keys[n_groups] = ck;
            cg_polys[n_groups] = poly_copy128(child->vals[i]);
            n_groups++;
        }
    }

    DPTable128* result = dpt_alloc128(parent->n * 2);

    for (int gi = 0; gi < n_groups; gi++) {
        DPTable128* pbo = dpt_alloc128(parent->n);
        for (i = 0; i < parent->cap; i++) {
            if (!parent->used[i]) continue;
            uint64_t merged = apply_conn_merge(parent->keys[i], &cg_keys[gi], shared_pos);
            dpt_add128(pbo, merged, parent->vals[i]);
        }
        for (i = 0; i < pbo->cap; i++) {
            if (!pbo->used[i]) continue;
            Poly128* prod = poly_mul128(pbo->vals[i], cg_polys[gi]);
            if (prod->n > 0) dpt_add128(result, pbo->keys[i], prod);
            poly_free128(prod);
        }
        dpt_free128(pbo);
    }

    for (i = 0; i < n_groups; i++) poly_free128(cg_polys[i]);
    free(cg_keys); free(cg_polys);
    return result;
}

/* process_bag128 — mirrors process_bag but with __int128 arithmetic */
static DPTable128* process_bag128(TDInfo* td, int bag_idx, int** out_verts, int* out_n);

static DPTable128* process_bag128(TDInfo* td, int bag_idx, int** out_verts, int* out_n) {
    int bag_size = td->bag_sizes[bag_idx];
    const int* bag_v = td->bag_verts_flat + td->bag_verts_offsets[bag_idx];

    int bag_verts[16]; int bv;
    memcpy(bag_verts, bag_v, bag_size * sizeof(int));
    for (int a = 0; a < bag_size - 1; a++)
        for (int b = a+1; b < bag_size; b++)
            if (bag_verts[a] > bag_verts[b]) {
                int tmp = bag_verts[a]; bag_verts[a] = bag_verts[b]; bag_verts[b] = tmp;
            }

    int vert_to_idx[10000];
    memset(vert_to_idx, -1, sizeof(vert_to_idx));
    for (bv = 0; bv < bag_size; bv++) vert_to_idx[bag_verts[bv]] = bv;

    uint64_t init_part = bag_size;
    for (bv = 0; bv < bag_size; bv++) init_part |= ((uint64_t)bv) << (4 + bv*4);
    DPTable128* table = dpt_alloc128(256);
    dpt_set128(table, init_part, poly_one128());

    int n_edges = td->bag_edge_counts[bag_idx];
    const int* edges = td->edges_flat + td->edges_offsets[bag_idx];

    int loop_u[64], loop_m[64]; int n_loops = 0;
    int par_u[64], par_v[64], par_m[64]; int n_par = 0;
    int sim_u[64], sim_v[64]; int n_sim = 0;

    for (int e = 0; e < n_edges; e++) {
        int u = edges[e*3], v = edges[e*3+1], m = edges[e*3+2];
        if (vert_to_idx[u] < 0 || vert_to_idx[v] < 0) continue;
        if (u == v) { loop_u[n_loops] = u; loop_m[n_loops] = m; n_loops++; }
        else if (m > 1) { par_u[n_par] = u; par_v[n_par] = v; par_m[n_par] = m; n_par++; }
        else { sim_u[n_sim] = u; sim_v[n_sim] = v; n_sim++; }
    }

    /* Process loops */
    for (int li = 0; li < n_loops; li++) {
        int mult = loop_m[li];
        Poly128* factor = poly_alloc128(mult + 1);
        __int128 c = (__int128)1;
        for (int j = 0; j <= mult; j++) {
            factor->keys[j] = (uint16_t)j;
            factor->vals[j] = c;
            c = c * (mult - j) / (j + 1);
        }
        factor->n = mult + 1;

        DPTable128* nt = dpt_alloc128(table->n * 2);
        for (int i = 0; i < table->cap; i++) {
            if (!table->used[i]) continue;
            Poly128* prod = poly_mul128(table->vals[i], factor);
            if (prod->n > 0) dpt_add128(nt, table->keys[i], prod);
            poly_free128(prod);
        }
        dpt_free128(table); table = nt;
        poly_free128(factor);
    }

    /* Process parallel edges */
    for (int pi = 0; pi < n_par; pi++) {
        int iu = vert_to_idx[par_u[pi]], iv = vert_to_idx[par_v[pi]];
        int mult = par_m[pi];
        Poly128* ff = poly_alloc128(mult+1);
        __int128 c = (__int128)1;
        for (int j = 0; j <= mult; j++) {
            ff->keys[j] = (uint16_t)j; ff->vals[j] = c;
            c = c * (mult - j) / (j + 1);
        }
        ff->n = mult + 1;
        Poly128* fm1 = poly_copy128(ff);
        fm1->vals[0] = fm1->vals[0] - (__int128)1;
        if (fm1->vals[0] == 0) {
            memmove(fm1->keys, fm1->keys+1, (fm1->n-1)*sizeof(uint16_t));
            memmove(fm1->vals, fm1->vals+1, (fm1->n-1)*sizeof(__int128));
            fm1->n--;
        }

        DPTable128* nt = dpt_alloc128(table->n * 4);
        for (int i = 0; i < table->cap; i++) {
            if (!table->used[i]) continue;
            uint64_t cp = cached_connect(table->keys[i], iu, iv);
            if (cp == table->keys[i]) {
                Poly128* prod = poly_mul128(table->vals[i], ff);
                if (prod->n > 0) dpt_add128(nt, table->keys[i], prod);
                poly_free128(prod);
            } else {
                dpt_add128(nt, table->keys[i], table->vals[i]);
                Poly128* prod = poly_mul128(table->vals[i], fm1);
                if (prod->n > 0) dpt_add128(nt, cp, prod);
                poly_free128(prod);
            }
        }
        dpt_free128(table); table = nt;
        poly_free128(ff); poly_free128(fm1);
    }

    /* Batch simple edges (k <= 12) */
    if (n_sim > 0 && n_sim <= 12) {
        int eidx[12][2];
        for (int e = 0; e < n_sim; e++) {
            eidx[e][0] = vert_to_idx[sim_u[e]];
            eidx[e][1] = vert_to_idx[sim_v[e]];
        }
        int k = n_sim;
        DPTable128* nt = dpt_alloc128(table->n * (1 << (k > 4 ? 4 : k)));
        for (int i = 0; i < table->cap; i++) {
            if (!table->used[i]) continue;
            uint64_t base_part = table->keys[i];
            Poly128* base_poly = table->vals[i];
            for (int mask = 0; mask < (1 << k); mask++) {
                uint64_t p = base_part;
                for (int bit = 0; bit < k; bit++) {
                    if (mask & (1 << bit))
                        p = cached_connect(p, eidx[bit][0], eidx[bit][1]);
                }
                int bc = __builtin_popcount(mask);
                Poly128* shifted;
                if (bc > 0) {
                    shifted = poly_shift128(base_poly, (uint16_t)bc);
                } else {
                    shifted = poly_copy128(base_poly);
                }
                dpt_add128(nt, p, shifted);
                poly_free128(shifted);
            }
        }
        dpt_free128(table); table = nt;
    } else if (n_sim > 12) {
        for (int e = 0; e < n_sim; e++) {
            int iu = vert_to_idx[sim_u[e]], iv = vert_to_idx[sim_v[e]];
            DPTable128* nt = dpt_alloc128(table->n * 4);
            for (int i = 0; i < table->cap; i++) {
                if (!table->used[i]) continue;
                dpt_add128(nt, table->keys[i], table->vals[i]);
                uint64_t np = cached_connect(table->keys[i], iu, iv);
                Poly128* shifted = poly_shift128(table->vals[i], 1);
                dpt_add128(nt, np, shifted);
                poly_free128(shifted);
            }
            dpt_free128(table); table = nt;
        }
    }

    /* Process children */
    int n_children = td->children_counts[bag_idx];
    const int* child_ids = td->children_flat + td->children_offsets[bag_idx];

    for (int ci = 0; ci < n_children; ci++) {
        int child_idx = child_ids[ci];
        int* child_verts; int child_n;
        DPTable128* child_table = process_bag128(td, child_idx, &child_verts, &child_n);

        int forget_pos[16]; int n_forget = 0;
        int offset = 0;
        for (int cv = 0; cv < child_n; cv++) {
            int in_bag = 0;
            for (bv = 0; bv < bag_size; bv++)
                if (bag_verts[bv] == child_verts[cv]) { in_bag = 1; break; }
            if (!in_bag) { forget_pos[n_forget++] = cv - offset; offset++; }
        }

        DPTable128* forgotten = dpt_alloc128(child_table->n * 2);
        for (int i = 0; i < child_table->cap; i++) {
            if (!child_table->used[i]) continue;
            uint64_t cur_enc = child_table->keys[i];
            Poly128* cur_poly = poly_copy128(child_table->vals[i]);
            for (int fi = 0; fi < n_forget; fi++) {
                ForgetResult fr = encoded_forget(cur_enc, forget_pos[fi]);
                if (fr.is_singleton) {
                    Poly128* shifted = poly_shift128(cur_poly, KEY_STRIDE + 1);
                    poly_free128(cur_poly);
                    cur_poly = shifted;
                }
                cur_enc = fr.new_enc;
            }
            dpt_add128(forgotten, cur_enc, cur_poly);
            poly_free128(cur_poly);
        }

        int shared[16]; int n_shared = 0;
        for (int cv = 0; cv < child_n; cv++) {
            for (bv = 0; bv < bag_size; bv++) {
                if (bag_verts[bv] == child_verts[cv]) {
                    shared[n_shared++] = child_verts[cv]; break;
                }
            }
        }

        DPTable128* merged = merge_tables128(table, forgotten, bag_verts, bag_size,
                                              shared, n_shared);
        dpt_free128(table); table = merged;
        dpt_free128(forgotten);
        dpt_free128(child_table);
        free(child_verts);
    }

    *out_verts = (int*)malloc(bag_size * sizeof(int));
    memcpy(*out_verts, bag_verts, bag_size * sizeof(int));
    *out_n = bag_size;
    return table;
}

/* Entry point: run __int128 DP and output (a,b) polynomial.
   Each coefficient emitted as (hi64, lo64) pair (two's complement). */
int treewidth_tutte_dp_ab128(
    int n_bags, const int* bag_sizes, const int* bag_verts_flat,
    int root, const int* children_counts, const int* children_flat,
    const int* bag_edge_counts, const int* edges_flat,
    int n_verts, int n_components,
    int* out_ab, long long* out_coeffs_hi, long long* out_coeffs_lo,
    int* out_n_terms, int max_out)
{
    memset(ccache, 0, sizeof(ccache));

    TDInfo td;
    td.n_bags = n_bags;
    td.bag_sizes = bag_sizes;
    td.bag_verts_flat = bag_verts_flat;
    td.children_counts = children_counts;
    td.children_flat = children_flat;
    td.bag_edge_counts = bag_edge_counts;
    td.edges_flat = edges_flat;
    td.bag_verts_offsets = (int*)malloc(n_bags * sizeof(int));
    td.children_offsets = (int*)malloc(n_bags * sizeof(int));
    td.edges_offsets = (int*)malloc(n_bags * sizeof(int));
    int off = 0;
    for (int i = 0; i < n_bags; i++) { td.bag_verts_offsets[i] = off; off += bag_sizes[i]; }
    off = 0;
    for (int i = 0; i < n_bags; i++) { td.children_offsets[i] = off; off += children_counts[i]; }
    off = 0;
    for (int i = 0; i < n_bags; i++) { td.edges_offsets[i] = off; off += bag_edge_counts[i] * 3; }

    int* root_verts; int root_n;
    DPTable128* root_table = process_bag128(&td, root, &root_verts, &root_n);

    Poly128* final_poly = poly_zero128();
    for (int i = 0; i < root_table->cap; i++) {
        if (!root_table->used[i]) continue;
        uint64_t cur_enc = root_table->keys[i];
        Poly128* cur_poly = poly_copy128(root_table->vals[i]);
        for (int j = 0; j < root_n; j++) {
            ForgetResult fr = encoded_forget(cur_enc, 0);
            if (fr.is_singleton) {
                Poly128* shifted = poly_shift128(cur_poly, KEY_STRIDE + 1);
                poly_free128(cur_poly);
                cur_poly = shifted;
            }
            cur_enc = fr.new_enc;
        }
        poly_add_inplace128(final_poly, cur_poly);
        poly_free128(cur_poly);
    }

    int a_shift = n_components;
    int b_shift = n_verts;
    int n_out = 0;
    for (int i = 0; i < final_poly->n && n_out < max_out; i++) {
        int a_pow = final_poly->keys[i] / KEY_STRIDE;
        int b_pow = final_poly->keys[i] % KEY_STRIDE;
        int new_a = a_pow - a_shift;
        int new_b = b_pow - b_shift;
        if (new_a < 0 || new_b < 0) continue;
        out_ab[n_out * 2] = new_a;
        out_ab[n_out * 2 + 1] = new_b;
        __int128 v = final_poly->vals[i];
        out_coeffs_hi[n_out] = (long long)(v >> 64);
        out_coeffs_lo[n_out] = (long long)((uint64_t)v);
        n_out++;
    }
    *out_n_terms = n_out;

    poly_free128(final_poly);
    dpt_free128(root_table);
    free(root_verts);
    free(td.bag_verts_offsets);
    free(td.children_offsets);
    free(td.edges_offsets);
    return 0;
}

/* Convert (a,b)-basis polynomial to (x,y)-basis via binomial expansion.
   conv_prime > 0: modular arithmetic for binomials (avoids overflow).
   conv_prime == 0: exact int64 arithmetic. */
int basis_convert_ab_to_xy(
    const int* ab_keys, const long long* ab_coeffs, int n_ab_terms,
    int* out_xy, long long* out_coeffs, int* out_n_terms, int max_out,
    long long conv_prime)
{
    g_prime = conv_prime;

    /* Find max degree for binomial table size */
    int max_ab = 0;
    for (int ti = 0; ti < n_ab_terms; ti++) {
        int a = ab_keys[ti * 2];
        int b = ab_keys[ti * 2 + 1];
        if (a > max_ab) max_ab = a;
        if (b > max_ab) max_ab = b;
    }
    int binom_size = max_ab + 1;
    if (binom_size < 1) binom_size = 1;

    int64_t* binom_flat = (int64_t*)calloc((size_t)binom_size * binom_size, sizeof(int64_t));
    #define BINOM2(n,k) binom_flat[(n) * binom_size + (k)]
    for (int nn = 0; nn < binom_size; nn++) {
        BINOM2(nn, 0) = coeff_from_int(1);
        for (int kk = 1; kk <= nn; kk++)
            BINOM2(nn, kk) = coeff_add(BINOM2(nn-1, kk-1), BINOM2(nn-1, kk));
    }

    int ht_cap = 65536;
    uint32_t* ht_keys_xy = (uint32_t*)calloc(ht_cap, sizeof(uint32_t));
    int64_t*  ht_vals_xy = (int64_t*)calloc(ht_cap, sizeof(int64_t));
    uint8_t*  ht_used_xy = (uint8_t*)calloc(ht_cap, sizeof(uint8_t));
    int xy_mask = ht_cap - 1;

    for (int ti = 0; ti < n_ab_terms; ti++) {
        int a_pow = ab_keys[ti * 2];
        int b_pow = ab_keys[ti * 2 + 1];
        int64_t coeff = coeff_from_int(ab_coeffs[ti]);

        for (int p = 0; p <= a_pow; p++) {
            int64_t cx = BINOM2(a_pow, p);
            if ((a_pow - p) % 2 != 0) cx = coeff_neg(cx);

            for (int q = 0; q <= b_pow; q++) {
                int64_t cy = BINOM2(b_pow, q);
                if ((b_pow - q) % 2 != 0) cy = coeff_neg(cy);

                int64_t contribution = coeff_mul(coeff, coeff_mul(cx, cy));
                if (contribution != 0) {
                    uint32_t key = (uint32_t)(p * 256 + q);
                    uint32_t h = (key * 2654435761u) & xy_mask;
                    while (ht_used_xy[h]) {
                        if (ht_keys_xy[h] == key) {
                            ht_vals_xy[h] = coeff_add(ht_vals_xy[h], contribution);
                            goto next_conv;
                        }
                        h = (h + 1) & xy_mask;
                    }
                    ht_used_xy[h] = 1; ht_keys_xy[h] = key; ht_vals_xy[h] = contribution;
                    next_conv:;
                }
            }
        }
    }

    int n_out = 0;
    for (int i = 0; i < ht_cap; i++) {
        if (ht_used_xy[i] && ht_vals_xy[i] != 0 && n_out < max_out) {
            out_xy[n_out * 2] = ht_keys_xy[i] / 256;
            out_xy[n_out * 2 + 1] = ht_keys_xy[i] % 256;
            out_coeffs[n_out] = ht_vals_xy[i];
            n_out++;
        }
    }
    *out_n_terms = n_out;

    free(ht_keys_xy); free(ht_vals_xy); free(ht_used_xy);
    free(binom_flat);
    #undef BINOM2
    return 0;
}

/* =========================================================================
   PROFILING EXPORTERS — read/reset per-thread g_tw_stats counters.
   Order MUST match the TWStats struct field order.
   ========================================================================= */
void treewidth_get_stats(long long* out, int n) {
    long long* src = (long long*)&g_tw_stats;
    int size = (int)(sizeof(TWStats) / sizeof(long long));
    int copy = n < size ? n : size;
    for (int i = 0; i < copy; i++) out[i] = src[i];
}
void treewidth_reset_stats(void) {
    memset(&g_tw_stats, 0, sizeof(g_tw_stats));
}
int treewidth_stats_size(void) {
    return (int)(sizeof(TWStats) / sizeof(long long));
}
""")

# Build/load
_lib = None
_ffi = ffi
import threading as _threading

_lib_lock = _threading.Lock()


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        try:
            from _treewidth_cffi import ffi as _cffi
            from _treewidth_cffi import lib
            _lib = lib
            return _lib
        except ImportError:
            pass
        import sys
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="treewidth_c_")
        ffi.compile(tmpdir=tmpdir)
        sys.path.insert(0, tmpdir)
        from _treewidth_cffi import ffi as _cffi
        from _treewidth_cffi import lib
        _lib = lib
        return _lib


# =============================================================================
# PYTHON WRAPPERS
# =============================================================================

class BatchMerger:
    """Reusable batch merger that caches cffi arrays."""

    def __init__(self):
        self._lib = _get_lib()
        self._enc_buf = None
        self._enc_cap = 0
        self._out_buf = None
        self._pairs_cache = {}
        self._sp_cache = {}

    def merge(self, parent_encs, conn_key, shared_positions):
        n_parents = len(parent_encs)
        if n_parents == 0:
            return [], None, 0

        if n_parents > self._enc_cap:
            new_cap = max(n_parents, self._enc_cap * 2, 1024)
            # `unsigned long long[]` accepts the full uint64 range that
            # `encode_partition` can produce for bags up to 15 elements
            # (tw≤14). Signed `long long[]` here would overflow at the top
            # nibble for tw=14 bags. Buffers are typed once and reused.
            self._enc_buf = _ffi.new("unsigned long long[]", new_cap)
            self._out_buf = _ffi.new("unsigned long long[]", new_cap)
            self._enc_cap = new_cap

        enc_list = list(parent_encs)
        try:
            for i, e in enumerate(enc_list):
                self._enc_buf[i] = e
        except OverflowError:
            # Encoding exceeds uint64 (header nibble already at length 16
            # for tw=15+ bags). Signal Python fallback so the caller's
            # `else` branch uses `encoded_merge_for_conn` instead.
            return None, None, 0

        if conn_key not in self._pairs_cache:
            n_pairs = len(conn_key)
            pairs_flat = _ffi.new("int[]", max(n_pairs * 2, 1))
            for i, (ci, cj) in enumerate(conn_key):
                pairs_flat[2 * i] = ci
                pairs_flat[2 * i + 1] = cj
            self._pairs_cache[conn_key] = (pairs_flat, n_pairs)
        pairs_flat, n_pairs = self._pairs_cache[conn_key]

        sp_key = tuple(shared_positions)
        if sp_key not in self._sp_cache:
            n_shared = len(shared_positions)
            sp = _ffi.new("int[]", max(n_shared, 1))
            for i, p in enumerate(shared_positions):
                sp[i] = p
            self._sp_cache[sp_key] = (sp, n_shared)
        sp, n_shared = self._sp_cache[sp_key]

        # Cast unsigned buffers to long long* for the cdef signature; the
        # bit pattern is preserved and the C code treats encodings as
        # opaque bag-state keys (no arithmetic comparisons on signedness).
        self._lib.batch_merge(
            _ffi.cast("long long*", self._enc_buf), n_parents,
            pairs_flat, n_pairs, sp, n_shared,
            _ffi.cast("long long*", self._out_buf))
        return enc_list, self._out_buf, n_parents


_merger = None


def c_batch_merge(parent_encs, conn_key, shared_positions):
    global _merger
    if _merger is None:
        _merger = BatchMerger()
    return _merger.merge(parent_encs, conn_key, shared_positions)


def compute_treewidth_tutte_c(td, mg):
    """Compute Tutte polynomial via bulk C treewidth DP.

    Args:
        td: TreeDecomposition from treewidth.py
        mg: MultiGraph

    Returns:
        TuttePolynomial or None if C extension unavailable
    """
    from ..polynomial import TuttePolynomial

    lib = _get_lib()

    n_bags = len(td.bags)

    # Build children (rooted tree)
    children = {i: [] for i in range(n_bags)}
    visited = set()

    def build_children(idx):
        visited.add(idx)
        for nb in td.tree_adj[idx]:
            if nb not in visited:
                children[idx].append(nb)
                build_children(nb)

    build_children(td.root)

    # Marshal bag sizes and vertices
    bag_sizes = _ffi.new("int[]", n_bags)
    bag_verts_list = []
    for i in range(n_bags):
        bag_sizes[i] = len(td.bags[i])
        bag_verts_list.extend(sorted(td.bags[i]))
    bag_verts = _ffi.new("int[]", len(bag_verts_list))
    for i, v in enumerate(bag_verts_list):
        bag_verts[i] = v

    # Marshal children
    children_counts = _ffi.new("int[]", n_bags)
    children_list = []
    for i in range(n_bags):
        children_counts[i] = len(children[i])
        children_list.extend(children[i])
    children_flat = _ffi.new("int[]", max(len(children_list), 1))
    for i, c in enumerate(children_list):
        children_flat[i] = c

    # Marshal edges
    bag_edge_counts = _ffi.new("int[]", n_bags)
    edges_list = []
    for i in range(n_bags):
        edges = td.bag_edges.get(i, [])
        bag_edge_counts[i] = len(edges)
        for u, v, m in edges:
            edges_list.extend([u, v, m])
    edges_flat = _ffi.new("int[]", max(len(edges_list), 1))
    for i, val in enumerate(edges_list):
        edges_flat[i] = val

    # Compute connected components
    n_verts = len(mg.nodes)
    visited_nodes = set()
    n_components = 0
    for start in mg.nodes:
        if start in visited_nodes:
            continue
        n_components += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited_nodes:
                continue
            visited_nodes.add(node)
            for nb in mg.neighbors(node):
                if nb not in visited_nodes:
                    stack.append(nb)

    # Tiered DP strategy based on coefficient bounds:
    # (a,b)-basis coefficient bound ≈ 2^{n_edges}; (x,y) bound larger (needs Python ints).
    # 1. ≤62 edges: exact int64 DP + int64 conversion (fastest).
    # 2. >62 edges: full modular DP + CRT (multiple passes, parallelized).
    #
    # (April 2026): the prior "63-120 edges: __int128 DP" path
    # was retired after head-to-head testing showed modular CRT is uniformly
    # faster in that edge range:
    #   Z(1,2) (24n 76e tw=10): int128 168s vs modular 117s (modular wins 30%)
    #   Cm2    (32n 80e tw=11): int128 695s vs modular 801s vs Python 207s
    # On Cm2, ALL C paths lose to Python (caught by `5≤tw≤10` gate at the
    # caller). Modular CRT is also more portable — `__int128` is GCC/Clang
    # only, blocking MSVC/Windows builds. The `treewidth_tutte_dp_ab128` C
    # function and `Poly128` infrastructure remain in source for now but are
    # dead code; safe to delete after the dead-code audit (see
    # `tutte/tests/test_int128_dead_code.py`).
    #
    # Historical note (earlier April 2026): an even earlier "63-76 edges:
    # int64 DP + modular conversion" path silently overflowed int64 in the
    # DP itself for ≥ 63 edges (coefficients can reach 2^E / sqrt(E), well
    # beyond 2^63). Symptom: T(1,1) was correct but other coefficients were
    # off by multiples of 2^64. Replaced by the int128 path then retired
    # entirely in favor of modular CRT.
    n_total_edges = sum(mg.edge_counts.values()) + sum(mg.loop_counts.values())
    needs_dp_modular = n_total_edges > 62
    needs_dp_int128 = False  # int128 path retired (dominated by modular CRT)
    needs_conv_modular = n_total_edges > 62

    max_out = 10000

    if not needs_conv_modular:
        # Fully exact int64 (fastest path)
        out_xy = _ffi.new("int[]", max_out * 2)
        out_coeffs = _ffi.new("long long[]", max_out)
        out_n = _ffi.new("int*")

        rc = lib.treewidth_tutte_dp(
            n_bags, bag_sizes, bag_verts, td.root,
            children_counts, children_flat,
            bag_edge_counts, edges_flat,
            n_verts, n_components,
            out_xy, out_coeffs, out_n, max_out, 0, 0)

        if rc != 0:
            return None

        coeffs = {}
        for i in range(out_n[0]):
            x_pow = out_xy[i * 2]
            y_pow = out_xy[i * 2 + 1]
            coeff = int(out_coeffs[i])
            if coeff != 0:
                coeffs[(x_pow, y_pow)] = coeff

        return TuttePolynomial.from_coefficients(coeffs)

    from ..validation import _crt_multi, _get_modular_primes

    if needs_dp_int128 and not needs_dp_modular:
        # __int128 DP (exact, single pass) + Python basis conversion.
        # Handles coefficients up to ~2^127 (covers up to ~120-edge graphs).
        max_ab = max_out * 2
        ab_keys = _ffi.new("int[]", max_ab * 2)
        ab_hi = _ffi.new("long long[]", max_ab)
        ab_lo = _ffi.new("long long[]", max_ab)
        ab_n = _ffi.new("int*")

        rc = lib.treewidth_tutte_dp_ab128(
            n_bags, bag_sizes, bag_verts, td.root,
            children_counts, children_flat,
            bag_edge_counts, edges_flat,
            n_verts, n_components,
            ab_keys, ab_hi, ab_lo, ab_n, max_ab)
        if rc != 0:
            return None

        # Reconstruct __int128 values from (hi64, lo64) pairs as Python ints.
        # Treat both halves as unsigned, then sign-correct via two's complement.
        MASK64 = (1 << 64) - 1
        poly_ab = {}
        for i in range(ab_n[0]):
            a_pow = ab_keys[i * 2]
            b_pow = ab_keys[i * 2 + 1]
            hi_u = int(ab_hi[i]) & MASK64
            lo_u = int(ab_lo[i]) & MASK64
            val = (hi_u << 64) | lo_u
            if val >= (1 << 127):
                val -= (1 << 128)
            if val != 0:
                poly_ab[(a_pow, b_pow)] = val

        # Basis conversion in Python with arbitrary-precision ints.
        from .treewidth import _convert_ab_to_xy
        return _convert_ab_to_xy(poly_ab)

    if not needs_dp_modular:
        # Exact int64 DP in (a,b) basis, then Python basis conversion with
        # arbitrary-precision ints. The (a,b) coefficients fit int64 for
        # n_edges <= 76; the (x,y) coefficients may overflow int64 but Python
        # math.comb handles them natively.
        max_ab = max_out * 2
        ab_keys = _ffi.new("int[]", max_ab * 2)
        ab_coeffs = _ffi.new("long long[]", max_ab)
        ab_n = _ffi.new("int*")

        rc = lib.treewidth_tutte_dp_ab(
            n_bags, bag_sizes, bag_verts, td.root,
            children_counts, children_flat,
            bag_edge_counts, edges_flat,
            n_verts, n_components,
            ab_keys, ab_coeffs, ab_n, max_ab)
        if rc != 0:
            return None

        poly_ab = {}
        for i in range(ab_n[0]):
            a_pow = ab_keys[i * 2]
            b_pow = ab_keys[i * 2 + 1]
            coeff = int(ab_coeffs[i])
            if coeff != 0:
                poly_ab[(a_pow, b_pow)] = coeff

        from .treewidth import _convert_ab_to_xy
        return _convert_ab_to_xy(poly_ab)

    # (a,b)-only CRT: run DP mod prime k times, reconstruct (a,b) as Python
    # bignums, then convert basis once in Python. The (a,b) coefficient bound
    # is ~2^{n_edges}, much tighter than the (x,y) bound ~2^{2·n_edges + n_verts},
    # so we need roughly half as many primes as the full-modular path.
    # 62-bit pseudo-Mersenne primes (form 2^62 - c) enable fast reduction in
    # coeff_mul, avoiding __int128 division on the hot path.
    log2_ab_bound = n_total_edges + 20
    primes_needed = max(int(log2_ab_bound / 60) + 1, 2)
    primes = _get_modular_primes(primes_needed, bits=62, form='mersenne_below')
    prime_cs = [(1 << 62) - p for p in primes]

    max_ab = max_out * 2

    def _run_one_prime(prime_and_c):
        p_val, p_c = prime_and_c
        ab_keys = _ffi.new("int[]", max_ab * 2)
        ab_coeffs = _ffi.new("long long[]", max_ab)
        ab_n = _ffi.new("int*")
        rc = lib.treewidth_tutte_dp_ab_mersenne(
            n_bags, bag_sizes, bag_verts, td.root,
            children_counts, children_flat,
            bag_edge_counts, edges_flat,
            n_verts, n_components,
            ab_keys, ab_coeffs, ab_n, max_ab, p_val, p_c)
        if rc != 0:
            return None
        return {(ab_keys[i * 2], ab_keys[i * 2 + 1]): int(ab_coeffs[i])
                for i in range(ab_n[0])}

    # Parallelize prime DPs. cffi releases the GIL during C calls; g_prime,
    # g_prime_c, and ccache are __thread so workers don't race.
    import os
    from concurrent.futures import ThreadPoolExecutor
    max_workers = min(len(primes), max(1, (os.cpu_count() or 2) - 1))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        all_residues = list(ex.map(_run_one_prime, zip(primes, prime_cs)))
    if any(r is None for r in all_residues):
        return None

    all_keys = set()
    for rm in all_residues:
        all_keys |= rm.keys()
    poly_ab = {}
    for key in all_keys:
        res = [rm.get(key, 0) for rm in all_residues]
        val = _crt_multi(res, primes)
        if val != 0:
            poly_ab[key] = val

    from .treewidth import _convert_ab_to_xy
    return _convert_ab_to_xy(poly_ab)
