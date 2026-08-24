"""C extension for polynomial multiplication via cffi.

Hot path: TuttePolynomial.__mul__ inner loop is O(n_1 × n_2) dict accumulation
in Python. C extension uses a hash table for O(n_1 × n_2) at native speed,
~10× faster on polynomials with ~hundreds of terms.

Pattern mirrors `tutte/graphs/_treewidth_c.py`: lazy auto-compile on first
import via cffi.compile(tmpdir=...); falls back to Python on compile failure.

Coefficient overflow: int64 suffices for Cm2 (max 58 bits). Larger graphs
(Cm3, Pm3) may need int128 or modular CRT — to be added as needed.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import cffi

ffi = cffi.FFI()

ffi.cdef(r"""
    /* Multiply two polynomials in coefficient-array form.
       Inputs: parallel arrays (xs, ys, cs) of length n_p1 and n_p2.
       Output: pre-allocated arrays out_xs, out_ys, out_cs (capacity);
               returns number of nonzero terms in result.
       Returns -1 on overflow or capacity exceeded. */
    int poly_mul_int64(
        const int* p1_xs, const int* p1_ys, const long long* p1_cs, int n_p1,
        const int* p2_xs, const int* p2_ys, const long long* p2_cs, int n_p2,
        int* out_xs, int* out_ys, long long* out_cs, int out_capacity);

    /* Batched polynomial multiplication.
       Processes n_pairs polynomial multiplications A_i × B_i in one cffi call,
       sharing a single scratch hash table across pairs (reset between pairs).

       Polys A and B are flat-encoded: a_offsets[i]..a_offsets[i+1] gives
       indices into a_xs/a_ys/a_cs for pair i's A poly. Similarly for B.
       Output polys are flat-encoded into out_xs/out_ys/out_cs;
       out_offsets[i] gives the start; out_lens[i] gives length of pair i's
       product.

       Returns 0 on success, -1 on overflow or any per-pair capacity exceeded.

       The pre-allocated `total_out_capacity` must be ≥ Σ n_p1[i] × n_p2[i].
       The hash table capacity (in elements) is reused; caller passes a
       sized scratch buffer (`ht_cap`).
    */
    int poly_mul_batched_int64(
        const int* a_xs, const int* a_ys, const long long* a_cs,
        const int* a_offsets,
        const int* b_xs, const int* b_ys, const long long* b_cs,
        const int* b_offsets,
        int n_pairs,
        int* out_xs, int* out_ys, long long* out_cs,
        int* out_offsets,
        int* out_lens,
        int total_out_capacity);
""")

ffi.set_source("_tutte_polynomial_cffi", r"""
    #include <stdlib.h>
    #include <string.h>
    #include <stdint.h>

    /* Open-addressing hash table on (xpow, ypow) -> coeff index.
       Used as scratch space for accumulating products. */
    #define HT_LOAD_FACTOR 0.7
    #define HT_EMPTY -1

    typedef struct {
        int x, y;
        long long c;
    } Term;

    static int ht_lookup_or_insert(
        int* keys_x, int* keys_y, int* values, int capacity,
        Term* terms, int* term_count, int x, int y
    ) {
        /* Hash: 64-bit fnv-style mix */
        uint64_t h = (uint64_t)x * 2654435761u + (uint64_t)y * 40503u;
        int idx = (int)(h % (uint64_t)capacity);
        while (1) {
            if (values[idx] == HT_EMPTY) {
                values[idx] = *term_count;
                keys_x[idx] = x;
                keys_y[idx] = y;
                terms[*term_count].x = x;
                terms[*term_count].y = y;
                terms[*term_count].c = 0;
                (*term_count)++;
                return values[idx];
            }
            if (keys_x[idx] == x && keys_y[idx] == y) {
                return values[idx];
            }
            idx = (idx + 1) % capacity;
        }
    }

    int poly_mul_int64(
        const int* p1_xs, const int* p1_ys, const long long* p1_cs, int n_p1,
        const int* p2_xs, const int* p2_ys, const long long* p2_cs, int n_p2,
        int* out_xs, int* out_ys, long long* out_cs, int out_capacity
    ) {
        if (n_p1 == 0 || n_p2 == 0) return 0;

        /* Hash table sized for the max possible distinct keys (n_p1 * n_p2). */
        int cap = n_p1 * n_p2 * 2;
        if (cap < 16) cap = 16;
        /* Guard against extreme allocations. */
        if (cap > 16 * 1024 * 1024) return -1;

        int* keys_x = (int*)malloc(cap * sizeof(int));
        int* keys_y = (int*)malloc(cap * sizeof(int));
        int* values = (int*)malloc(cap * sizeof(int));
        Term* terms = (Term*)malloc(cap * sizeof(Term));
        if (!keys_x || !keys_y || !values || !terms) {
            free(keys_x); free(keys_y); free(values); free(terms);
            return -1;
        }
        for (int i = 0; i < cap; i++) values[i] = HT_EMPTY;

        int term_count = 0;

        for (int i = 0; i < n_p1; i++) {
            int x1 = p1_xs[i];
            int y1 = p1_ys[i];
            long long c1 = p1_cs[i];
            for (int j = 0; j < n_p2; j++) {
                int x = x1 + p2_xs[j];
                int y = y1 + p2_ys[j];
                long long c = c1 * p2_cs[j];
                int slot = ht_lookup_or_insert(keys_x, keys_y, values, cap,
                                                terms, &term_count, x, y);
                terms[slot].c += c;
            }
        }

        /* Compact nonzero terms into output arrays. */
        int n_out = 0;
        for (int i = 0; i < term_count; i++) {
            if (terms[i].c != 0) {
                if (n_out >= out_capacity) {
                    free(keys_x); free(keys_y); free(values); free(terms);
                    return -1;
                }
                out_xs[n_out] = terms[i].x;
                out_ys[n_out] = terms[i].y;
                out_cs[n_out] = terms[i].c;
                n_out++;
            }
        }

        free(keys_x); free(keys_y); free(values); free(terms);
        return n_out;
    }

    /* Per-pair scratch: shared HT reused across all n_pairs muls.
       Allocates a worst-case HT once (sized by sum of n_p1[i] × n_p2[i])
       and resets between pairs. Output buffer is contiguous; per-pair
       lengths reported via out_lens. */
    int poly_mul_batched_int64(
        const int* a_xs, const int* a_ys, const long long* a_cs,
        const int* a_offsets,
        const int* b_xs, const int* b_ys, const long long* b_cs,
        const int* b_offsets,
        int n_pairs,
        int* out_xs, int* out_ys, long long* out_cs,
        int* out_offsets,
        int* out_lens,
        int total_out_capacity
    ) {
        if (n_pairs == 0) return 0;

        /* Compute max HT capacity needed across all pairs:
           cap_i = 2 * n_p1[i] * n_p2[i]. Reuse the max. */
        int max_ht_cap = 16;
        for (int p = 0; p < n_pairs; p++) {
            int n1 = a_offsets[p+1] - a_offsets[p];
            int n2 = b_offsets[p+1] - b_offsets[p];
            int cap = n1 * n2 * 2;
            if (cap > max_ht_cap) max_ht_cap = cap;
        }
        if (max_ht_cap > 16 * 1024 * 1024) return -1;

        int* keys_x = (int*)malloc(max_ht_cap * sizeof(int));
        int* keys_y = (int*)malloc(max_ht_cap * sizeof(int));
        int* values = (int*)malloc(max_ht_cap * sizeof(int));
        Term* terms = (Term*)malloc(max_ht_cap * sizeof(Term));
        if (!keys_x || !keys_y || !values || !terms) {
            free(keys_x); free(keys_y); free(values); free(terms);
            return -1;
        }

        int out_pos = 0;
        for (int p = 0; p < n_pairs; p++) {
            int n1 = a_offsets[p+1] - a_offsets[p];
            int n2 = b_offsets[p+1] - b_offsets[p];
            out_offsets[p] = out_pos;
            if (n1 == 0 || n2 == 0) {
                out_lens[p] = 0;
                continue;
            }

            /* Size the HT for this pair: 2 × n1 × n2, clamped. */
            int cap = n1 * n2 * 2;
            if (cap < 16) cap = 16;
            if (cap > max_ht_cap) {
                /* Should not happen; guard. */
                free(keys_x); free(keys_y); free(values); free(terms);
                return -1;
            }
            /* Reset HT entries. */
            for (int i = 0; i < cap; i++) values[i] = HT_EMPTY;

            int term_count = 0;
            int a_start = a_offsets[p];
            int b_start = b_offsets[p];
            for (int i = 0; i < n1; i++) {
                int x1 = a_xs[a_start + i];
                int y1 = a_ys[a_start + i];
                long long c1 = a_cs[a_start + i];
                for (int j = 0; j < n2; j++) {
                    int x = x1 + b_xs[b_start + j];
                    int y = y1 + b_ys[b_start + j];
                    long long c = c1 * b_cs[b_start + j];
                    int slot = ht_lookup_or_insert(keys_x, keys_y, values, cap,
                                                    terms, &term_count, x, y);
                    terms[slot].c += c;
                }
            }

            /* Compact nonzero terms into out arrays. */
            int n_out = 0;
            for (int i = 0; i < term_count; i++) {
                if (terms[i].c != 0) {
                    if (out_pos >= total_out_capacity) {
                        free(keys_x); free(keys_y); free(values); free(terms);
                        return -1;
                    }
                    out_xs[out_pos] = terms[i].x;
                    out_ys[out_pos] = terms[i].y;
                    out_cs[out_pos] = terms[i].c;
                    out_pos++;
                    n_out++;
                }
            }
            out_lens[p] = n_out;
        }
        /* Final offsets sentinel. */
        out_offsets[n_pairs] = out_pos;

        free(keys_x); free(keys_y); free(values); free(terms);
        return 0;
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
            from _tutte_polynomial_cffi import ffi as cffi_ffi
            from _tutte_polynomial_cffi import lib
            _lib = lib
            _ffi = cffi_ffi
            return _lib, _ffi
        except ImportError:
            pass
        import sys
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="tutte_polynomial_c_")
        ffi.compile(tmpdir=tmpdir)
        sys.path.insert(0, tmpdir)
        from _tutte_polynomial_cffi import ffi as cffi_ffi
        from _tutte_polynomial_cffi import lib
        _lib = lib
        _ffi = cffi_ffi
        return _lib, _ffi


# Coefficient overflow guard: int64 max is ~9.2e18 (63-bit).
# Products of two int64 coefficients can overflow. We bail to Python
# if either input has |coeff| > sqrt(int64_max) ~ 3e9 (32-bit-safe range).
_OVERFLOW_GUARD = 1 << 31  # ~2.1e9


def poly_mul_c(
    p1_coeffs: Dict[Tuple[int, int], int],
    p2_coeffs: Dict[Tuple[int, int], int],
) -> Optional[Dict[Tuple[int, int], int]]:
    """Multiply two polynomial coefficient dicts via C extension.

    Returns None if the product would overflow int64 (caller should fall
    back to Python multiplication).

    Inputs: dict mapping (xpow, ypow) -> coeff (int).
    Output: dict mapping (xpow, ypow) -> coeff (nonzero terms only).
    """
    if not p1_coeffs or not p2_coeffs:
        return {}

    n1 = len(p1_coeffs)
    n2 = len(p2_coeffs)

    # Overflow guard: bail if any coefficient could cause int64 overflow.
    for c in p1_coeffs.values():
        if abs(c) >= _OVERFLOW_GUARD:
            return None
    for c in p2_coeffs.values():
        if abs(c) >= _OVERFLOW_GUARD:
            return None

    lib, _ffi = _get_lib()

    p1_xs = _ffi.new("int[]", [k[0] for k in p1_coeffs.keys()])
    p1_ys = _ffi.new("int[]", [k[1] for k in p1_coeffs.keys()])
    p1_cs = _ffi.new("long long[]", list(p1_coeffs.values()))
    p2_xs = _ffi.new("int[]", [k[0] for k in p2_coeffs.keys()])
    p2_ys = _ffi.new("int[]", [k[1] for k in p2_coeffs.keys()])
    p2_cs = _ffi.new("long long[]", list(p2_coeffs.values()))

    out_capacity = n1 * n2 + 16
    out_xs = _ffi.new("int[]", out_capacity)
    out_ys = _ffi.new("int[]", out_capacity)
    out_cs = _ffi.new("long long[]", out_capacity)

    n_out = lib.poly_mul_int64(
        p1_xs, p1_ys, p1_cs, n1,
        p2_xs, p2_ys, p2_cs, n2,
        out_xs, out_ys, out_cs, out_capacity,
    )

    if n_out < 0:
        return None  # overflow or capacity exceeded — fall back

    result: Dict[Tuple[int, int], int] = {}
    for i in range(n_out):
        result[(out_xs[i], out_ys[i])] = out_cs[i]
    return result


def poly_mul_python(
    p1_coeffs: Dict[Tuple[int, int], int],
    p2_coeffs: Dict[Tuple[int, int], int],
) -> Dict[Tuple[int, int], int]:
    """Pure Python fallback. Used when C ext isn't available or overflows."""
    from collections import defaultdict
    result = defaultdict(int)
    for (i1, j1), c1 in p1_coeffs.items():
        for (i2, j2), c2 in p2_coeffs.items():
            result[(i1 + i2, j1 + j2)] += c1 * c2
    return {k: v for k, v in result.items() if v != 0}


def poly_mul(
    p1_coeffs: Dict[Tuple[int, int], int],
    p2_coeffs: Dict[Tuple[int, int], int],
) -> Dict[Tuple[int, int], int]:
    """Multiply two polynomial coefficient dicts. Uses C extension when
    available, falls back to Python on overflow or compile failure."""
    try:
        result = poly_mul_c(p1_coeffs, p2_coeffs)
        if result is not None:
            return result
    except Exception:
        pass
    return poly_mul_python(p1_coeffs, p2_coeffs)


def poly_mul_batched_c(
    pairs: list,
) -> Optional[list]:
    """Batched polynomial multiplication via single cffi call.

    `pairs` is a list of `(A_dict, B_dict)` polynomial pairs. Each
    `_dict` maps `(xpow, ypow) -> coeff`. Returns a list of product
    dicts, in input order.

    Returns None on overflow or any per-pair error; caller falls back to
    per-pair `poly_mul`.

    Shares one C-side hash-table allocation across all pairs (reset
    between pairs). For Cm₃-scale convolves with ~1 M poly muls per
    streaming chunk, this saves the per-call cffi marshaling overhead
    that dominates Python dispatch.
    """
    if not pairs:
        return []

    # Overflow guard: bail if any coefficient might overflow int64.
    for a, b in pairs:
        for c in a.values():
            if abs(c) >= _OVERFLOW_GUARD:
                return None
        for c in b.values():
            if abs(c) >= _OVERFLOW_GUARD:
                return None

    lib, _ffi = _get_lib()

    n_pairs = len(pairs)

    # Flatten A and B into contiguous arrays with offsets.
    a_xs_list: list = []
    a_ys_list: list = []
    a_cs_list: list = []
    a_offsets_list: list = [0]
    b_xs_list: list = []
    b_ys_list: list = []
    b_cs_list: list = []
    b_offsets_list: list = [0]
    total_out_capacity = 0
    for a, b in pairs:
        for (x, y), c in a.items():
            a_xs_list.append(x)
            a_ys_list.append(y)
            a_cs_list.append(c)
        a_offsets_list.append(len(a_xs_list))
        for (x, y), c in b.items():
            b_xs_list.append(x)
            b_ys_list.append(y)
            b_cs_list.append(c)
        b_offsets_list.append(len(b_xs_list))
        total_out_capacity += len(a) * len(b) + 1

    if total_out_capacity > 16 * 1024 * 1024:
        return None  # safety guard

    a_xs = _ffi.new("int[]", a_xs_list)
    a_ys = _ffi.new("int[]", a_ys_list)
    a_cs = _ffi.new("long long[]", a_cs_list)
    a_offsets = _ffi.new("int[]", a_offsets_list)
    b_xs = _ffi.new("int[]", b_xs_list)
    b_ys = _ffi.new("int[]", b_ys_list)
    b_cs = _ffi.new("long long[]", b_cs_list)
    b_offsets = _ffi.new("int[]", b_offsets_list)
    out_xs = _ffi.new("int[]", total_out_capacity)
    out_ys = _ffi.new("int[]", total_out_capacity)
    out_cs = _ffi.new("long long[]", total_out_capacity)
    out_offsets = _ffi.new("int[]", n_pairs + 1)
    out_lens = _ffi.new("int[]", n_pairs)

    rc = lib.poly_mul_batched_int64(
        a_xs, a_ys, a_cs, a_offsets,
        b_xs, b_ys, b_cs, b_offsets,
        n_pairs,
        out_xs, out_ys, out_cs,
        out_offsets, out_lens,
        total_out_capacity,
    )
    if rc != 0:
        return None

    # Decode output: build N product dicts from the flat arrays.
    results: list = []
    for p in range(n_pairs):
        start = out_offsets[p]
        n_out = out_lens[p]
        d: Dict[Tuple[int, int], int] = {}
        for i in range(n_out):
            d[(out_xs[start + i], out_ys[start + i])] = out_cs[start + i]
        results.append(d)
    return results
