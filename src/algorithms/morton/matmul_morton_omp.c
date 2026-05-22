/*
 * matmul_morton_omp.c - OpenMP-parallelized Morton-recursive matmul
 * wired to the AVX-512 microkernel kernel_avx512_4x32 (Zen 5 /
 * EPYC 9R45 main_server variant).
 *
 * Same recursion and same leaf kernel as matmul_morton_avx2.c. The
 * leaf helpers (materialize_a_panel, kernel_base_morton_omp,
 * kernel_base_morton_omp_add, kernel_base_morton_omp_ijk_fallback)
 * are duplicated here rather than #include'd from the sibling
 * translation unit so this module is standalone: no static helper
 * ever crosses the file boundary.
 *
 * The recursion is parallelized as follows:
 *
 *   - Each sub-problem larger than g_parallel_threshold_omp creates
 *     two OpenMP tasks at the recursive split. Sub-problems at or
 *     below that threshold recurse inline, without spawning tasks.
 *     This avoids the per-task overhead at the leaf.
 *
 *   - The MK split produces four sub-products. The two that target
 *     the top half of C must execute in order (TL writes, TR
 *     accumulates), and the two that target the bottom half must
 *     execute in order too. But top and bottom are independent: we
 *     spawn one task for the top sequence and one for the bottom
 *     sequence, then taskwait.
 *
 *   - The N split produces two fully independent sub-products: one
 *     task each, then taskwait.
 *
 *   - The k-only split (none of our recursions take it directly, but
 *     it is inside the MK split as the second product) is sequential
 *     by construction inside the per-half task body.
 *
 * Scratch buffers:
 *
 *   Each leaf materializes a row-major panel of A from the
 *   Morton-of-blocks layout into a per-thread scratch buffer. The
 *   pool is allocated once by the public wrapper using
 *   omp_get_max_threads() and indexed at the leaf by
 *   omp_get_thread_num(). With 8 threads (Zen 5 server: 1 thread/core)
 *   and a 128x128 panel (64 KiB) the total scratch is ~512 KiB,
 *   negligible vs the 32 MiB shared L3.
 */

#include "matmul_morton_omp.h"

#include "morton.h"                /* morton_encode, is_power_of_two */
#include "kernel_avx512_morton.h"  /* kernel_avx512_4x32, geometry */
#include "matrix_utils.h"          /* xalloc_aligned, xfree */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#define MR KERNEL_AVX512_MORTON_MR   /* 4 — tied to MORTON_AVX2_TILE */
#define NR KERNEL_AVX512_MORTON_NR   /* 32 */
#define LEAF_KERNEL kernel_avx512_4x32

/* ------------------------------------------------------------------ */
/* Tunables                                                            */
/* ------------------------------------------------------------------ */

/* Defaults sized for the EPYC 9R45 (L1d 48 KiB / L2 1 MiB / L3 32 MiB
 * shared). The 1048576 threshold gives a leaf side ~90 (A panel
 * ~32 KiB, fits L1d). The makefile may override via -D... at
 * compile time. */
#ifndef MORTON_OMP_RECURSION_THRESHOLD_DEFAULT
#define MORTON_OMP_RECURSION_THRESHOLD_DEFAULT ((size_t)1048576UL)
#endif
#ifndef MORTON_OMP_PARALLEL_THRESHOLD_DEFAULT
#define MORTON_OMP_PARALLEL_THRESHOLD_DEFAULT  ((size_t)1048576UL)
#endif
size_t g_recursion_threshold_omp = MORTON_OMP_RECURSION_THRESHOLD_DEFAULT;
size_t g_parallel_threshold_omp  = MORTON_OMP_PARALLEL_THRESHOLD_DEFAULT;

void matmul_morton_omp_set_threshold(size_t threshold)
{
    if (threshold == 0) {
        fprintf(stderr,
                "Warning: matmul_morton_omp_set_threshold(0) ignored; "
                "keeping previous threshold (%llu).\n",
                (unsigned long long)g_recursion_threshold_omp);
        return;
    }
    g_recursion_threshold_omp = threshold;
}

void matmul_morton_omp_set_parallel_threshold(size_t threshold)
{
    /* Zero is permitted here and means "always run sequentially"
     * (every sub-problem is at or below the parallel threshold).
     * Useful for measuring the serial baseline of this exact module
     * without changing the binary. */
    g_parallel_threshold_omp = threshold;
}

/* ------------------------------------------------------------------ */
/* Leaf helpers (duplicated from matmul_morton_avx2.c on purpose)      */
/* ------------------------------------------------------------------ */

/* Materialize the m_block x k_block panel of A from the
 * Morton-of-blocks layout into a row-major scratch buffer with row
 * stride k_block. Caller guarantees m_block and k_block are multiples
 * of MORTON_AVX2_TILE = 4. */
static void materialize_a_panel(const scalar_t *A_morton,
                                size_t a_morton_offset,
                                size_t a_block_dim,
                                scalar_t *A_local,
                                size_t m_block, size_t k_block)
{
    (void)a_block_dim;

    const size_t blocks_m = m_block / MORTON_AVX2_TILE;
    const size_t blocks_k = k_block / MORTON_AVX2_TILE;
    const size_t tile_sq  = (size_t)MORTON_AVX2_TILE * MORTON_AVX2_TILE;

    for (size_t bi = 0; bi < blocks_m; ++bi) {
        for (size_t bj = 0; bj < blocks_k; ++bj) {
            uint64_t bcode = morton_encode((uint32_t)bi, (uint32_t)bj);
            size_t block_offset = a_morton_offset
                                + (size_t)bcode * tile_sq;

            for (size_t ii = 0; ii < MORTON_AVX2_TILE; ++ii) {
                size_t row_local = bi * MORTON_AVX2_TILE + ii;
                const scalar_t *src = &A_morton[block_offset
                                              + ii * MORTON_AVX2_TILE];
                scalar_t *dst = &A_local[row_local * k_block
                                       + bj * MORTON_AVX2_TILE];
                for (size_t jj = 0; jj < MORTON_AVX2_TILE; ++jj) {
                    dst[jj] = src[jj];
                }
            }
        }
    }
}

/* ijk fallback for non-tile-aligned leaf shapes. */
static void kernel_base_morton_omp_ijk_fallback(
    scalar_t *C, size_t ldc,
    const scalar_t *A_local, size_t lda,
    const scalar_t *B, size_t ldb,
    size_t m_block, size_t k_block, size_t n_block,
    int accumulate)
{
    for (size_t i = 0; i < m_block; ++i) {
        for (size_t j = 0; j < n_block; ++j) {
            scalar_t sum = (scalar_t)0;
            for (size_t p = 0; p < k_block; ++p) {
                sum += A_local[i * lda + p] * B[p * ldb + j];
            }
            if (accumulate) C[i * ldc + j] += sum;
            else            C[i * ldc + j]  = sum;
        }
    }
}

/* Overwrite-variant leaf: zero the C tile, then accumulate via the
 * microkernel; final result equals A * B exactly. */
static void kernel_base_morton_omp(scalar_t *C,
                                   const scalar_t *A_morton,
                                   const scalar_t *B,
                                   size_t m_block, size_t k_block,
                                   size_t n_block,
                                   size_t a_morton_offset,
                                   size_t a_block_dim,
                                   size_t ldc, size_t ldb,
                                   scalar_t *A_local_scratch)
{
    materialize_a_panel(A_morton, a_morton_offset, a_block_dim,
                        A_local_scratch, m_block, k_block);

    if ((m_block % MR) != 0 || (n_block % NR) != 0) {
        for (size_t i = 0; i < m_block; ++i) {
            scalar_t *crow = &C[i * ldc];
            for (size_t j = 0; j < n_block; ++j) crow[j] = (scalar_t)0;
        }
        kernel_base_morton_omp_ijk_fallback(C, ldc,
                                            A_local_scratch, k_block,
                                            B, ldb,
                                            m_block, k_block, n_block,
                                            /*accumulate=*/1);
        return;
    }

    for (size_t i = 0; i < m_block; ++i) {
        scalar_t *crow = &C[i * ldc];
        for (size_t j = 0; j < n_block; ++j) crow[j] = (scalar_t)0;
    }

    for (size_t ii = 0; ii < m_block; ii += MR) {
        for (size_t jj = 0; jj < n_block; jj += NR) {
            LEAF_KERNEL(&C[ii * ldc + jj], ldc,
                        &A_local_scratch[ii * k_block], k_block,
                        &B[jj], ldb,
                        k_block);
        }
    }
}

/* Accumulate-variant leaf: like above but does not zero C, so the
 * microkernel accumulates on top of the previous value. */
static void kernel_base_morton_omp_add(scalar_t *C,
                                       const scalar_t *A_morton,
                                       const scalar_t *B,
                                       size_t m_block, size_t k_block,
                                       size_t n_block,
                                       size_t a_morton_offset,
                                       size_t a_block_dim,
                                       size_t ldc, size_t ldb,
                                       scalar_t *A_local_scratch)
{
    materialize_a_panel(A_morton, a_morton_offset, a_block_dim,
                        A_local_scratch, m_block, k_block);

    if ((m_block % MR) != 0 || (n_block % NR) != 0) {
        kernel_base_morton_omp_ijk_fallback(C, ldc,
                                            A_local_scratch, k_block,
                                            B, ldb,
                                            m_block, k_block, n_block,
                                            /*accumulate=*/1);
        return;
    }

    for (size_t ii = 0; ii < m_block; ii += MR) {
        for (size_t jj = 0; jj < n_block; jj += NR) {
            LEAF_KERNEL(&C[ii * ldc + jj], ldc,
                        &A_local_scratch[ii * k_block], k_block,
                        &B[jj], ldb,
                        k_block);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Recursion                                                           */
/* ------------------------------------------------------------------ */

static void matmul_morton_omp_inner(scalar_t *C,
                                    const scalar_t *A_morton,
                                    const scalar_t *B,
                                    size_t m_block, size_t k_block,
                                    size_t n_block,
                                    size_t a_morton_offset,
                                    size_t a_block_dim,
                                    size_t ldc, size_t ldb,
                                    scalar_t **scratches);

static void matmul_morton_omp_inner_add(scalar_t *C,
                                        const scalar_t *A_morton,
                                        const scalar_t *B,
                                        size_t m_block, size_t k_block,
                                        size_t n_block,
                                        size_t a_morton_offset,
                                        size_t a_block_dim,
                                        size_t ldc, size_t ldb,
                                        scalar_t **scratches);

/* The two MK-split halves are pre-baked into helper routines so each
 * task body is a single statement and the task creation site stays
 * readable. */
static void mk_top_pair(scalar_t *C, const scalar_t *A_morton,
                        const scalar_t *B,
                        size_t half, size_t n_block,
                        size_t a_morton_offset, size_t quadrant_size,
                        size_t ldc, size_t ldb,
                        scalar_t **scratches)
{
    /* TL writes, TR accumulates on top of TL. Sequential. */
    matmul_morton_omp_inner    (C, A_morton, B,
                                half, half, n_block,
                                a_morton_offset + (size_t)0 * quadrant_size,
                                half, ldc, ldb, scratches);
    matmul_morton_omp_inner_add(C, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)1 * quadrant_size,
                                half, ldc, ldb, scratches);
}

static void mk_bot_pair(scalar_t *C, const scalar_t *A_morton,
                        const scalar_t *B,
                        size_t half, size_t n_block,
                        size_t a_morton_offset, size_t quadrant_size,
                        size_t ldc, size_t ldb,
                        scalar_t **scratches)
{
    matmul_morton_omp_inner    (C + half * ldc, A_morton, B,
                                half, half, n_block,
                                a_morton_offset + (size_t)2 * quadrant_size,
                                half, ldc, ldb, scratches);
    matmul_morton_omp_inner_add(C + half * ldc, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)3 * quadrant_size,
                                half, ldc, ldb, scratches);
}

/* Same two helpers but for the _add branch: every product accumulates
 * (the caller already holds a meaningful C value). */
static void mk_top_pair_add(scalar_t *C, const scalar_t *A_morton,
                            const scalar_t *B,
                            size_t half, size_t n_block,
                            size_t a_morton_offset, size_t quadrant_size,
                            size_t ldc, size_t ldb,
                            scalar_t **scratches)
{
    matmul_morton_omp_inner_add(C, A_morton, B,
                                half, half, n_block,
                                a_morton_offset + (size_t)0 * quadrant_size,
                                half, ldc, ldb, scratches);
    matmul_morton_omp_inner_add(C, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)1 * quadrant_size,
                                half, ldc, ldb, scratches);
}

static void mk_bot_pair_add(scalar_t *C, const scalar_t *A_morton,
                            const scalar_t *B,
                            size_t half, size_t n_block,
                            size_t a_morton_offset, size_t quadrant_size,
                            size_t ldc, size_t ldb,
                            scalar_t **scratches)
{
    matmul_morton_omp_inner_add(C + half * ldc, A_morton, B,
                                half, half, n_block,
                                a_morton_offset + (size_t)2 * quadrant_size,
                                half, ldc, ldb, scratches);
    matmul_morton_omp_inner_add(C + half * ldc, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)3 * quadrant_size,
                                half, ldc, ldb, scratches);
}

static void matmul_morton_omp_inner(scalar_t *C,
                                    const scalar_t *A_morton,
                                    const scalar_t *B,
                                    size_t m_block, size_t k_block,
                                    size_t n_block,
                                    size_t a_morton_offset,
                                    size_t a_block_dim,
                                    size_t ldc, size_t ldb,
                                    scalar_t **scratches)
{
    if (m_block * k_block * n_block <= g_recursion_threshold_omp) {
        int tid = omp_get_thread_num();
        kernel_base_morton_omp(C, A_morton, B,
                               m_block, k_block, n_block,
                               a_morton_offset, a_block_dim,
                               ldc, ldb,
                               scratches[tid]);
        return;
    }

    const int spawn = (m_block * k_block * n_block
                       > g_parallel_threshold_omp);

    /* Case N: split n. Both halves are fully independent. */
    if (n_block > a_block_dim && n_block >= 2) {
        size_t n_half = n_block / 2;
        if (spawn) {
            #pragma omp task default(shared) firstprivate(C, B, n_half,           \
                m_block, k_block, n_block, a_morton_offset, a_block_dim,         \
                ldc, ldb)
            matmul_morton_omp_inner(C, A_morton, B,
                                    m_block, k_block, n_half,
                                    a_morton_offset, a_block_dim,
                                    ldc, ldb, scratches);
            #pragma omp task default(shared) firstprivate(C, B, n_half,           \
                m_block, k_block, n_block, a_morton_offset, a_block_dim,         \
                ldc, ldb)
            matmul_morton_omp_inner(C + n_half, A_morton, B + n_half,
                                    m_block, k_block, n_block - n_half,
                                    a_morton_offset, a_block_dim,
                                    ldc, ldb, scratches);
            #pragma omp taskwait
        } else {
            matmul_morton_omp_inner(C, A_morton, B,
                                    m_block, k_block, n_half,
                                    a_morton_offset, a_block_dim,
                                    ldc, ldb, scratches);
            matmul_morton_omp_inner(C + n_half, A_morton, B + n_half,
                                    m_block, k_block, n_block - n_half,
                                    a_morton_offset, a_block_dim,
                                    ldc, ldb, scratches);
        }
        return;
    }

    /* Case MK: top and bottom halves of C are independent; within
     * each half the two products (write then accumulate) must run
     * sequentially. */
    if (a_block_dim >= 2) {
        assert(m_block == k_block);
        assert(m_block == a_block_dim);

        size_t half          = a_block_dim / 2;
        size_t quadrant_size = half * half;

        if (spawn) {
            #pragma omp task default(shared) firstprivate(C, B, half, n_block,    \
                a_morton_offset, quadrant_size, ldc, ldb)
            mk_top_pair(C, A_morton, B, half, n_block,
                        a_morton_offset, quadrant_size,
                        ldc, ldb, scratches);
            #pragma omp task default(shared) firstprivate(C, B, half, n_block,    \
                a_morton_offset, quadrant_size, ldc, ldb)
            mk_bot_pair(C, A_morton, B, half, n_block,
                        a_morton_offset, quadrant_size,
                        ldc, ldb, scratches);
            #pragma omp taskwait
        } else {
            mk_top_pair(C, A_morton, B, half, n_block,
                        a_morton_offset, quadrant_size,
                        ldc, ldb, scratches);
            mk_bot_pair(C, A_morton, B, half, n_block,
                        a_morton_offset, quadrant_size,
                        ldc, ldb, scratches);
        }
        return;
    }

    /* Degenerate fallback: a_block_dim == 1 and n_block == 1. */
    {
        int tid = omp_get_thread_num();
        kernel_base_morton_omp(C, A_morton, B,
                               m_block, k_block, n_block,
                               a_morton_offset, a_block_dim,
                               ldc, ldb,
                               scratches[tid]);
    }
}

static void matmul_morton_omp_inner_add(scalar_t *C,
                                        const scalar_t *A_morton,
                                        const scalar_t *B,
                                        size_t m_block, size_t k_block,
                                        size_t n_block,
                                        size_t a_morton_offset,
                                        size_t a_block_dim,
                                        size_t ldc, size_t ldb,
                                        scalar_t **scratches)
{
    if (m_block * k_block * n_block <= g_recursion_threshold_omp) {
        int tid = omp_get_thread_num();
        kernel_base_morton_omp_add(C, A_morton, B,
                                   m_block, k_block, n_block,
                                   a_morton_offset, a_block_dim,
                                   ldc, ldb,
                                   scratches[tid]);
        return;
    }

    const int spawn = (m_block * k_block * n_block
                       > g_parallel_threshold_omp);

    if (n_block > a_block_dim && n_block >= 2) {
        size_t n_half = n_block / 2;
        if (spawn) {
            #pragma omp task default(shared) firstprivate(C, B, n_half,           \
                m_block, k_block, n_block, a_morton_offset, a_block_dim,         \
                ldc, ldb)
            matmul_morton_omp_inner_add(C, A_morton, B,
                                        m_block, k_block, n_half,
                                        a_morton_offset, a_block_dim,
                                        ldc, ldb, scratches);
            #pragma omp task default(shared) firstprivate(C, B, n_half,           \
                m_block, k_block, n_block, a_morton_offset, a_block_dim,         \
                ldc, ldb)
            matmul_morton_omp_inner_add(C + n_half, A_morton, B + n_half,
                                        m_block, k_block, n_block - n_half,
                                        a_morton_offset, a_block_dim,
                                        ldc, ldb, scratches);
            #pragma omp taskwait
        } else {
            matmul_morton_omp_inner_add(C, A_morton, B,
                                        m_block, k_block, n_half,
                                        a_morton_offset, a_block_dim,
                                        ldc, ldb, scratches);
            matmul_morton_omp_inner_add(C + n_half, A_morton, B + n_half,
                                        m_block, k_block, n_block - n_half,
                                        a_morton_offset, a_block_dim,
                                        ldc, ldb, scratches);
        }
        return;
    }

    if (a_block_dim >= 2) {
        assert(m_block == k_block);
        assert(m_block == a_block_dim);

        size_t half          = a_block_dim / 2;
        size_t quadrant_size = half * half;

        if (spawn) {
            #pragma omp task default(shared) firstprivate(C, B, half, n_block,    \
                a_morton_offset, quadrant_size, ldc, ldb)
            mk_top_pair_add(C, A_morton, B, half, n_block,
                            a_morton_offset, quadrant_size,
                            ldc, ldb, scratches);
            #pragma omp task default(shared) firstprivate(C, B, half, n_block,    \
                a_morton_offset, quadrant_size, ldc, ldb)
            mk_bot_pair_add(C, A_morton, B, half, n_block,
                            a_morton_offset, quadrant_size,
                            ldc, ldb, scratches);
            #pragma omp taskwait
        } else {
            mk_top_pair_add(C, A_morton, B, half, n_block,
                            a_morton_offset, quadrant_size,
                            ldc, ldb, scratches);
            mk_bot_pair_add(C, A_morton, B, half, n_block,
                            a_morton_offset, quadrant_size,
                            ldc, ldb, scratches);
        }
        return;
    }

    {
        int tid = omp_get_thread_num();
        kernel_base_morton_omp_add(C, A_morton, B,
                                   m_block, k_block, n_block,
                                   a_morton_offset, a_block_dim,
                                   ldc, ldb,
                                   scratches[tid]);
    }
}

/* ------------------------------------------------------------------ */
/* Public wrapper                                                      */
/* ------------------------------------------------------------------ */

/* Round up to the next power of two that is at least 64, so scratch
 * sized at side^2 is large enough for any leaf produced by the
 * recursion under the current threshold. Mirror of the helper in
 * matmul_morton_avx2.c. */
static size_t scratch_side_for_threshold(size_t threshold)
{
    size_t target = threshold / (size_t)NR;
    size_t side = 1;
    while (side * side < target) side <<= 1;
    if (side < 64) side = 64;
    return side;
}

void matmul_morton_omp(scalar_t *C,
                       const scalar_t *A_morton,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n)
{
    if (m != k) {
        fprintf(stderr,
                "Error in matmul_morton_omp: A must be square "
                "(got m=%llu, k=%llu).\n",
                (unsigned long long)m, (unsigned long long)k);
        exit(EXIT_FAILURE);
    }
    if (!is_power_of_two(m) || m < (size_t)MR) {
        fprintf(stderr,
                "Error in matmul_morton_omp: m (%llu) must be a power "
                "of two and at least MR = %d.\n",
                (unsigned long long)m, MR);
        exit(EXIT_FAILURE);
    }

    /* omp_get_max_threads returns the upper bound on the team size
     * that any subsequent parallel region in this thread can use.
     * That is the right size for the scratch pool. */
    const int max_threads = omp_get_max_threads();

    size_t side = scratch_side_for_threshold(g_recursion_threshold_omp);
    if (side > m) side = m;

    scalar_t **scratches = (scalar_t **)malloc(
        (size_t)max_threads * sizeof(scalar_t *));
    if (scratches == NULL) {
        fprintf(stderr,
                "Error in matmul_morton_omp: out of memory for "
                "scratch pool (%d threads).\n", max_threads);
        exit(EXIT_FAILURE);
    }
    for (int t = 0; t < max_threads; ++t) {
        scratches[t] = xalloc_aligned(side * side);
    }

    #pragma omp parallel default(shared)
    {
        #pragma omp single
        matmul_morton_omp_inner(C, A_morton, B,
                                m, k, n,
                                /* a_morton_offset = */ 0,
                                /* a_block_dim     = */ m,
                                /* ldc = */ n,
                                /* ldb = */ n,
                                scratches);
    }

    for (int t = 0; t < max_threads; ++t) {
        xfree(scratches[t]);
    }
    free(scratches);
}

/* ------------------------------------------------------------------ */
/* Benchmark orchestrators                                             */
/* ------------------------------------------------------------------ */

void benchmark_iterations_morton_omp(scalar_t *B_out,
                                     const scalar_t *A,
                                     const scalar_t *Z,
                                     size_t m, size_t n,
                                     size_t num_iters)
{
    scalar_t *A_morton = xalloc_aligned(m * m);
    reorganize_to_morton_blocks(A, A_morton, m);

    benchmark_iterations_morton_omp_preorganized(B_out, A_morton, Z,
                                                 m, n, num_iters);

    xfree(A_morton);
}

void benchmark_iterations_morton_omp_preorganized(scalar_t *B_out,
                                                  const scalar_t *A_morton,
                                                  const scalar_t *Z,
                                                  size_t m, size_t n,
                                                  size_t num_iters)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        matmul_morton_omp(B_next, A_morton, B_curr, m, m, n);

        scalar_t *out_block = B_out + iter * n * n;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                out_block[i * n + j] = B_next[i * n + j];
            }
        }

        scalar_t *tmp = B_curr;
        B_curr = B_next;
        B_next = tmp;
    }

    xfree(B_curr);
    xfree(B_next);
}
