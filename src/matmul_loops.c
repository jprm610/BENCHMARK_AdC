/*
 * matmul_loops.c - Six loop-order variants of C = A * B (Phase 1.1).
 *
 * All matrices are row-major. A is m x k_dim, B is k_dim x n, C is m x n.
 * The loop variable over the shared dimension is named p (not k) to avoid
 * shadowing the k_dim parameter.
 *
 * Variants that accumulate partial sums into C across iterations of the
 * outer loops (ikj, jki, kij, kji) memset C to zero before entering the
 * loops.  Variants with the reduction dimension innermost (ijk, jik) use
 * a scalar accumulator and write C once.
 *
 * Locality summary (A m x k_dim row-major, B k_dim x n row-major):
 *   ijk : A stride-1 per i-row, B stride-n (column walk) -- worst for B
 *   ikj : A scalar reuse per (i,p), B stride-1 (row walk) -- best overall
 *   jik : same as ijk but j outer; B column walk, A row walk
 *   jki : B scalar reuse per (p,j), A stride-k_dim (column walk) -- bad A
 *   kij : same as ikj but k outer; good B, but no A scalar reuse
 *   kji : A column walk (stride k_dim), B scalar reuse -- bad A
 */

#include "matmul_loops.h"
#include "matrix_utils.h"   /* xalloc_aligned, xfree */

#include <string.h>         /* memset, memcpy, strcmp */

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static void zero_matrix(scalar_t *C, size_t m, size_t n)
{
    memset(C, 0, m * n * sizeof(scalar_t));
}

/* ------------------------------------------------------------------ */
/* Six loop-order kernels                                               */
/* ------------------------------------------------------------------ */

/*
 * ijk: outer=i, middle=j, inner=k.
 * A: stride-1 within row i. B: stride-n (column j, non-contiguous).
 * Uses per-(i,j) accumulator; no pre-zero needed.
 */
void matmul_ijk(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k_dim, size_t n)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            scalar_t acc = (scalar_t)0;
            for (size_t p = 0; p < k_dim; ++p)
                acc += A[i * k_dim + p] * B[p * n + j];
            C[i * n + j] = acc;
        }
    }
}

/*
 * ikj: outer=i, middle=k, inner=j.
 * A[i,p] reused across all j. B row p is stride-1 (contiguous).
 * C row i is stride-1. Best spatial locality for tall-skinny B (n=128).
 */
void matmul_ikj(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k_dim, size_t n)
{
    zero_matrix(C, m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t p = 0; p < k_dim; ++p) {
            scalar_t a_ip = A[i * k_dim + p];
            for (size_t j = 0; j < n; ++j)
                C[i * n + j] += a_ip * B[p * n + j];
        }
    }
}

/*
 * jik: outer=j, middle=i, inner=k.
 * Same per-element work as ijk; B accessed by column (stride n).
 * Cache behavior identical to ijk at -O0.
 */
void matmul_jik(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k_dim, size_t n)
{
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            scalar_t acc = (scalar_t)0;
            for (size_t p = 0; p < k_dim; ++p)
                acc += A[i * k_dim + p] * B[p * n + j];
            C[i * n + j] = acc;
        }
    }
}

/*
 * jki: outer=j, middle=k, inner=i.
 * B[p,j] scalar reuse across i. A accessed by column (stride k_dim).
 * Bad spatial locality on A.
 */
void matmul_jki(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k_dim, size_t n)
{
    zero_matrix(C, m, n);
    for (size_t j = 0; j < n; ++j) {
        for (size_t p = 0; p < k_dim; ++p) {
            scalar_t b_pj = B[p * n + j];
            for (size_t i = 0; i < m; ++i)
                C[i * n + j] += A[i * k_dim + p] * b_pj;
        }
    }
}

/*
 * kij: outer=k, middle=i, inner=j.
 * A[i,p] scalar reuse across j. B row p stride-1. C row i stride-1.
 * Similar locality to ikj; slightly worse due to outer k breaking A reuse.
 */
void matmul_kij(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k_dim, size_t n)
{
    zero_matrix(C, m, n);
    for (size_t p = 0; p < k_dim; ++p) {
        for (size_t i = 0; i < m; ++i) {
            scalar_t a_ip = A[i * k_dim + p];
            for (size_t j = 0; j < n; ++j)
                C[i * n + j] += a_ip * B[p * n + j];
        }
    }
}

/*
 * kji: outer=k, middle=j, inner=i.
 * B[p,j] scalar reuse. A accessed by column (stride k_dim). Bad A locality.
 */
void matmul_kji(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k_dim, size_t n)
{
    zero_matrix(C, m, n);
    for (size_t p = 0; p < k_dim; ++p) {
        for (size_t j = 0; j < n; ++j) {
            scalar_t b_pj = B[p * n + j];
            for (size_t i = 0; i < m; ++i)
                C[i * n + j] += A[i * k_dim + p] * b_pj;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Lookup by name                                                       */
/* ------------------------------------------------------------------ */

matmul_fn_t matmul_loops_lookup(const char *name)
{
    if (strcmp(name, "ijk") == 0) return matmul_ijk;
    if (strcmp(name, "ikj") == 0) return matmul_ikj;
    if (strcmp(name, "jik") == 0) return matmul_jik;
    if (strcmp(name, "jki") == 0) return matmul_jki;
    if (strcmp(name, "kij") == 0) return matmul_kij;
    if (strcmp(name, "kji") == 0) return matmul_kji;
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Benchmark orchestrator                                               */
/* ------------------------------------------------------------------ */

void benchmark_iterations_loops(scalar_t *B_out,
                                 const scalar_t *A,
                                 const scalar_t *Z,
                                 size_t m, size_t n,
                                 size_t num_iters,
                                 matmul_fn_t kernel)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        kernel(B_next, A, B_curr, m, m, n);

        scalar_t *out_block = B_out + iter * n * n;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                out_block[i * n + j] = B_next[i * n + j];

        scalar_t *tmp = B_curr;
        B_curr        = B_next;
        B_next        = tmp;
    }

    xfree(B_curr);
    xfree(B_next);
}
