/*
 * matmul_morton.c - Recursive matmul with A in Morton layout.
 *
 * Invariants threaded through the recursion:
 *   - A is a square sub-block of side a_block_dim (a power of 2),
 *     occupying the contiguous segment
 *         A_morton[a_morton_offset .. a_morton_offset + a_block_dim^2).
 *   - The current sub-problem operates on m_block rows and k_block
 *     columns of that sub-block; we keep m_block == k_block ==
 *     a_block_dim at every recursive call.
 *   - B and C are row-major sub-blocks with leading dimensions ldb and
 *     ldc (the leading dimensions of the *original* matrices, never the
 *     sub-blocks).
 *
 * The recursion has two productive cases plus a leaf:
 *   - Case N (split n): when n_block > a_block_dim, halve n_block; A is
 *     shared between the two recursive calls. Two regions of C are
 *     disjoint, both overwrite.
 *   - Case MK (split m and k together): when a_block_dim >= 2, halve A
 *     into its four quadrants TL/TR/BL/BR. The Z-order encoding maps
 *     them to four consecutive segments of A_morton at offsets
 *         a_morton_offset + {0, 1, 2, 3} * (half * half).
 *     C splits into its top and bottom halves; B splits into top and
 *     bottom halves; four sub-matrix products combine into the result.
 *   - Leaf: m_block*k_block*n_block <= g_recursion_threshold, or the
 *     degenerate a_block_dim == 1 and n_block == 1 case.
 *
 * Two parallel variants exist for the recursion and the leaf kernel:
 *   - plain (matmul_morton_inner, kernel_base_morton): overwrites C.
 *   - _add (matmul_morton_inner_add, kernel_base_morton_add):
 *     accumulates into C. Used by the second/third/fourth product in
 *     the MK split and by every call inside the _add branch.
 */

#include "matmul_morton.h"
#include "morton.h"
#include "matrix_utils.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Recursion threshold: tunable at run time so that Prompt 2 of Sesion 03
 * can sweep it without recompiling. The default mirrors the Sesion 02
 * value (32 * 32 * 128 = 131072 element products), keeping behaviour
 * unchanged for every caller that does not call the setter. Reads from
 * within the recursion are unsynchronized; in practice the variable is
 * set once before benchmark_iterations_morton_preorganized is invoked,
 * so no fence is needed.
 */
size_t g_recursion_threshold = (size_t)32 * 32 * 128;

void matmul_morton_set_threshold(size_t threshold)
{
    /* Guard against zero: a zero threshold would mean "never recurse",
     * which makes the whole problem fall into the leaf kernel with the
     * full m,k,n dimensions and defeats the purpose of the routine.
     * Treat zero as a request for the default, with a warning so the
     * caller does not silently get unexpected behaviour. */
    if (threshold == 0) {
        fprintf(stderr,
                "Warning: matmul_morton_set_threshold(0) ignored; "
                "keeping previous threshold (%llu).\n",
                (unsigned long long)g_recursion_threshold);
        return;
    }
    g_recursion_threshold = threshold;
}

/* Forward declarations of the internal helpers. */
static void matmul_morton_inner(scalar_t *C,
                                const scalar_t *A_morton,
                                const scalar_t *B,
                                size_t m_block, size_t k_block, size_t n_block,
                                size_t a_morton_offset,
                                size_t a_block_dim,
                                size_t ldc, size_t ldb);

static void matmul_morton_inner_add(scalar_t *C,
                                    const scalar_t *A_morton,
                                    const scalar_t *B,
                                    size_t m_block, size_t k_block, size_t n_block,
                                    size_t a_morton_offset,
                                    size_t a_block_dim,
                                    size_t ldc, size_t ldb);

static void kernel_base_morton(scalar_t *C,
                               const scalar_t *A_morton,
                               const scalar_t *B,
                               size_t m_block, size_t k_block, size_t n_block,
                               size_t a_morton_offset,
                               size_t a_block_dim,
                               size_t ldc, size_t ldb);

static void kernel_base_morton_add(scalar_t *C,
                                   const scalar_t *A_morton,
                                   const scalar_t *B,
                                   size_t m_block, size_t k_block, size_t n_block,
                                   size_t a_morton_offset,
                                   size_t a_block_dim,
                                   size_t ldc, size_t ldb);

/* ------------------------------------------------------------------ */
/* Public wrapper                                                      */
/* ------------------------------------------------------------------ */

void matmul_morton(scalar_t *C,
                   const scalar_t *A_morton,
                   const scalar_t *B,
                   size_t m, size_t k, size_t n)
{
    if (m != k) {
        fprintf(stderr,
                "Error in matmul_morton: A must be square (got m=%llu, k=%llu).\n",
                (unsigned long long)m, (unsigned long long)k);
        exit(EXIT_FAILURE);
    }
    if (!is_power_of_two(m)) {
        fprintf(stderr,
                "Error in matmul_morton: m must be a power of two (got %llu).\n",
                (unsigned long long)m);
        exit(EXIT_FAILURE);
    }

    matmul_morton_inner(C, A_morton, B,
                        m, k, n,
                        /* a_morton_offset = */ 0,
                        /* a_block_dim     = */ m,
                        /* ldc = */ n,
                        /* ldb = */ n);
}

/* ------------------------------------------------------------------ */
/* Overwrite branch of the recursion                                   */
/* ------------------------------------------------------------------ */

static void matmul_morton_inner(scalar_t *C,
                                const scalar_t *A_morton,
                                const scalar_t *B,
                                size_t m_block, size_t k_block, size_t n_block,
                                size_t a_morton_offset,
                                size_t a_block_dim,
                                size_t ldc, size_t ldb)
{
    if (m_block * k_block * n_block <= g_recursion_threshold) {
        kernel_base_morton(C, A_morton, B,
                           m_block, k_block, n_block,
                           a_morton_offset, a_block_dim,
                           ldc, ldb);
        return;
    }

    /* Case N: n is strictly the largest dimension. A is shared between
     * the two recursive calls; B and C halve along the n axis. The two
     * halves of C are disjoint, so both calls overwrite. */
    if (n_block > a_block_dim && n_block >= 2) {
        size_t n_half = n_block / 2;
        matmul_morton_inner(C,          A_morton, B,
                            m_block, k_block, n_half,
                            a_morton_offset, a_block_dim,
                            ldc, ldb);
        matmul_morton_inner(C + n_half, A_morton, B + n_half,
                            m_block, k_block, n_block - n_half,
                            a_morton_offset, a_block_dim,
                            ldc, ldb);
        return;
    }

    /* Case MK: split A into its four Morton quadrants. The recursion
     * keeps m_block == k_block == a_block_dim, so the next level still
     * sees a square sub-block. */
    if (a_block_dim >= 2) {
        assert(m_block == k_block);
        assert(m_block == a_block_dim);

        size_t half          = a_block_dim / 2;
        size_t quadrant_size = half * half;

        /* C_top = A_TL * B_top  (overwrite)
         * Offset 0 selects the top-left Morton quadrant. */
        matmul_morton_inner(C, A_morton, B,
                            half, half, n_block,
                            a_morton_offset + (size_t)0 * quadrant_size,
                            half, ldc, ldb);

        /* C_top += A_TR * B_bot (accumulate over what TL just wrote)
         * Offset 1 selects the top-right quadrant; B_bot = B + half*ldb. */
        matmul_morton_inner_add(C, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)1 * quadrant_size,
                                half, ldc, ldb);

        /* C_bot = A_BL * B_top  (overwrite; C_bot is disjoint from C_top)
         * Offset 2 selects the bottom-left quadrant. */
        matmul_morton_inner(C + half * ldc, A_morton, B,
                            half, half, n_block,
                            a_morton_offset + (size_t)2 * quadrant_size,
                            half, ldc, ldb);

        /* C_bot += A_BR * B_bot (accumulate over what BL just wrote)
         * Offset 3 selects the bottom-right quadrant. */
        matmul_morton_inner_add(C + half * ldc, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)3 * quadrant_size,
                                half, ldc, ldb);
        return;
    }

    /* Degenerate fallback: a_block_dim == 1 and n_block == 1. The leaf
     * kernel handles m_block == k_block == n_block == 1 correctly. */
    kernel_base_morton(C, A_morton, B,
                       m_block, k_block, n_block,
                       a_morton_offset, a_block_dim,
                       ldc, ldb);
}

/* ------------------------------------------------------------------ */
/* Accumulate branch of the recursion                                  */
/* ------------------------------------------------------------------ */

static void matmul_morton_inner_add(scalar_t *C,
                                    const scalar_t *A_morton,
                                    const scalar_t *B,
                                    size_t m_block, size_t k_block, size_t n_block,
                                    size_t a_morton_offset,
                                    size_t a_block_dim,
                                    size_t ldc, size_t ldb)
{
    if (m_block * k_block * n_block <= g_recursion_threshold) {
        kernel_base_morton_add(C, A_morton, B,
                               m_block, k_block, n_block,
                               a_morton_offset, a_block_dim,
                               ldc, ldb);
        return;
    }

    if (n_block > a_block_dim && n_block >= 2) {
        size_t n_half = n_block / 2;
        matmul_morton_inner_add(C,          A_morton, B,
                                m_block, k_block, n_half,
                                a_morton_offset, a_block_dim,
                                ldc, ldb);
        matmul_morton_inner_add(C + n_half, A_morton, B + n_half,
                                m_block, k_block, n_block - n_half,
                                a_morton_offset, a_block_dim,
                                ldc, ldb);
        return;
    }

    if (a_block_dim >= 2) {
        assert(m_block == k_block);
        assert(m_block == a_block_dim);

        size_t half          = a_block_dim / 2;
        size_t quadrant_size = half * half;

        /* All four contributions accumulate: we are already inside the
         * _add branch, so C carries a previous value that every product
         * must add onto. */
        matmul_morton_inner_add(C, A_morton, B,
                                half, half, n_block,
                                a_morton_offset + (size_t)0 * quadrant_size,
                                half, ldc, ldb);
        matmul_morton_inner_add(C, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)1 * quadrant_size,
                                half, ldc, ldb);
        matmul_morton_inner_add(C + half * ldc, A_morton, B,
                                half, half, n_block,
                                a_morton_offset + (size_t)2 * quadrant_size,
                                half, ldc, ldb);
        matmul_morton_inner_add(C + half * ldc, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)3 * quadrant_size,
                                half, ldc, ldb);
        return;
    }

    kernel_base_morton_add(C, A_morton, B,
                           m_block, k_block, n_block,
                           a_morton_offset, a_block_dim,
                           ldc, ldb);
}

/* ------------------------------------------------------------------ */
/* Leaf kernels                                                        */
/*                                                                     */
/* The "Option A" indexing scheme from the prompt: for each (i, k)     */
/* inside the current sub-block, the element of A lives at             */
/*     A_morton[a_morton_offset + morton_encode(i, k)]                 */
/* where i and k are LOCAL indices (0..m_block, 0..k_block). The       */
/* invariant m_block == k_block == a_block_dim guarantees that         */
/* morton_encode(i, k) stays inside [0, a_block_dim^2), so the         */
/* computed A index stays inside the sub-block segment.                */
/* ------------------------------------------------------------------ */

static void kernel_base_morton(scalar_t *C,
                               const scalar_t *A_morton,
                               const scalar_t *B,
                               size_t m_block, size_t k_block, size_t n_block,
                               size_t a_morton_offset,
                               size_t a_block_dim,
                               size_t ldc, size_t ldb)
{
    /* a_block_dim is only needed for the invariant assertion above and
     * for the recursive callers; it is not read by the leaf itself. */
    (void)a_block_dim;

    for (size_t i = 0; i < m_block; ++i) {
        for (size_t j = 0; j < n_block; ++j) {
            scalar_t sum = (scalar_t)0;
            for (size_t k = 0; k < k_block; ++k) {
                uint64_t a_idx = a_morton_offset
                               + morton_encode((uint32_t)i, (uint32_t)k);
                sum += A_morton[a_idx] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;  /* overwrite */
        }
    }
}

static void kernel_base_morton_add(scalar_t *C,
                                   const scalar_t *A_morton,
                                   const scalar_t *B,
                                   size_t m_block, size_t k_block, size_t n_block,
                                   size_t a_morton_offset,
                                   size_t a_block_dim,
                                   size_t ldc, size_t ldb)
{
    (void)a_block_dim;

    for (size_t i = 0; i < m_block; ++i) {
        for (size_t j = 0; j < n_block; ++j) {
            scalar_t sum = (scalar_t)0;
            for (size_t k = 0; k < k_block; ++k) {
                uint64_t a_idx = a_morton_offset
                               + morton_encode((uint32_t)i, (uint32_t)k);
                sum += A_morton[a_idx] * B[k * ldb + j];
            }
            C[i * ldc + j] += sum;  /* accumulate */
        }
    }
}

/* ------------------------------------------------------------------ */
/* Benchmark orchestrators                                             */
/* ------------------------------------------------------------------ */

void benchmark_iterations_morton(scalar_t *B_out,
                                 const scalar_t *A,
                                 const scalar_t *Z,
                                 size_t m, size_t n,
                                 size_t num_iters)
{
    /* Reorganize A into Morton order once; the recurrence reuses it. */
    scalar_t *A_morton = xalloc_aligned(m * m);
    reorganize_to_morton(A, A_morton, m);

    benchmark_iterations_morton_preorganized(B_out, A_morton, Z,
                                             m, n, num_iters);

    xfree(A_morton);
}

void benchmark_iterations_morton_preorganized(scalar_t *B_out,
                                              const scalar_t *A_morton,
                                              const scalar_t *Z,
                                              size_t m, size_t n,
                                              size_t num_iters)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        /* B_next = A * B_curr (A is m x m, B_curr is m x n). */
        matmul_morton(B_next, A_morton, B_curr, m, m, n);

        /* Store the first n rows of B_next into the output buffer. */
        scalar_t *out_block = B_out + iter * n * n;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                out_block[i * n + j] = B_next[i * n + j];
            }
        }

        /* Swap so that next iteration's input is the just-computed value. */
        scalar_t *tmp = B_curr;
        B_curr = B_next;
        B_next = tmp;
    }

    xfree(B_curr);
    xfree(B_next);
}
