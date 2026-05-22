/*
 * matmul_morton_avx2.c - Morton-recursive matmul wired to the AVX2
 * microkernel kernel_avx2_4x16.
 *
 * Differences with respect to matmul_morton (Sesion 02):
 *   - A is stored in Morton-of-BLOCKS layout with tile=MORTON_AVX2_TILE=4
 *     instead of Morton-of-elements. See matmul_morton_avx2.h for the
 *     formal definition.
 *   - The leaf kernel materializes a row-major A_local panel from the
 *     Morton-of-blocks layout, then iterates kernel_avx2_4x16 over
 *     the 4x16 tiles of (A_local, B, C).
 *   - The recursion itself is unchanged: the {0,1,2,3}*(half*half)
 *     quadrant offsets still work because they encode positions in
 *     the Z-order of blocks, not of elements.
 *   - The threshold is separate (g_recursion_threshold_avx2) so the
 *     Sesion 02 default for g_recursion_threshold can stay at 131072
 *     and not affect this module.
 *
 * The scratch A_local buffer is allocated once in the public wrapper
 * and threaded through the recursion to avoid malloc/free per leaf.
 */

#include "matmul_morton_avx2.h"

#include "morton.h"           /* morton_encode, is_power_of_two */
#include "kernel_avx2_morton.h"  /* kernel_avx2_4x16, KERNEL_AVX2_MR/NR */
#include "matrix_utils.h"     /* xalloc_aligned, xfree */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MR KERNEL_AVX2_MR   /* 4 */
#define NR KERNEL_AVX2_NR   /* 16 */

size_t g_recursion_threshold_avx2 = (size_t)64 * 64 * 128;  /* 524288 */

void matmul_morton_avx2_set_threshold(size_t threshold)
{
    /* Same guard as matmul_morton's setter: zero would collapse the
     * whole problem to the leaf and is almost certainly a bug. */
    if (threshold == 0) {
        fprintf(stderr,
                "Warning: matmul_morton_avx2_set_threshold(0) ignored; "
                "keeping previous threshold (%llu).\n",
                (unsigned long long)g_recursion_threshold_avx2);
        return;
    }
    g_recursion_threshold_avx2 = threshold;
}

/* ------------------------------------------------------------------ */
/* Layout reorganization                                               */
/* ------------------------------------------------------------------ */

/*
 * reorganize_to_morton_blocks: pack a row-major m x m matrix into the
 * Morton-of-blocks layout with tile = MORTON_AVX2_TILE. The number of
 * blocks per side, m / MORTON_AVX2_TILE, must be a power of two so
 * that morton_encode applied to block coordinates yields a contiguous
 * permutation of [0, (m/MR)^2). m itself need not be a power of two
 * by itself, but in this project we always have m a power of two and
 * m >= MR, which trivially satisfies both conditions.
 */
void reorganize_to_morton_blocks(const scalar_t *A_row,
                                 scalar_t *A_morton,
                                 size_t m)
{
    if (m == 0 || (m % MORTON_AVX2_TILE) != 0) {
        fprintf(stderr,
                "Error in reorganize_to_morton_blocks: m (%llu) must be "
                "a positive multiple of MORTON_AVX2_TILE (%d).\n",
                (unsigned long long)m, MORTON_AVX2_TILE);
        exit(EXIT_FAILURE);
    }
    size_t n_blocks_per_side = m / MORTON_AVX2_TILE;
    if (!is_power_of_two(n_blocks_per_side)) {
        fprintf(stderr,
                "Error in reorganize_to_morton_blocks: m/MORTON_AVX2_TILE "
                "(%llu) must be a power of two.\n",
                (unsigned long long)n_blocks_per_side);
        exit(EXIT_FAILURE);
    }

    /* Walk in block coordinates so the inner two loops fill a single
     * tile of A_morton at once; the cache behavior of the writes is
     * sequential per tile. */
    for (size_t bi = 0; bi < n_blocks_per_side; ++bi) {
        for (size_t bj = 0; bj < n_blocks_per_side; ++bj) {
            uint64_t bcode = morton_encode((uint32_t)bi, (uint32_t)bj);
            size_t block_offset = (size_t)bcode
                                * (size_t)MORTON_AVX2_TILE
                                * (size_t)MORTON_AVX2_TILE;

            for (size_t ii = 0; ii < MORTON_AVX2_TILE; ++ii) {
                size_t row_global = bi * MORTON_AVX2_TILE + ii;
                for (size_t jj = 0; jj < MORTON_AVX2_TILE; ++jj) {
                    size_t col_global = bj * MORTON_AVX2_TILE + jj;
                    A_morton[block_offset + ii * MORTON_AVX2_TILE + jj]
                        = A_row[row_global * m + col_global];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Leaf kernels                                                        */
/* ------------------------------------------------------------------ */

/* Materialize the m_block x k_block panel of A from the
 * Morton-of-blocks layout into a row-major scratch buffer. Both
 * m_block and k_block must be multiples of MR; the leaf checks
 * before calling. Each output row is contiguous and ready for
 * kernel_avx2_4x16 to consume with leading dimension k_block. */
static void materialize_a_panel(const scalar_t *A_morton,
                                size_t a_morton_offset,
                                size_t a_block_dim,
                                scalar_t *A_local,
                                size_t m_block, size_t k_block)
{
    (void)a_block_dim;  /* invariant: m_block == k_block == a_block_dim */

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
                /* Copy MORTON_AVX2_TILE = 4 floats; small enough for
                 * the compiler to emit straight-line moves. */
                for (size_t jj = 0; jj < MORTON_AVX2_TILE; ++jj) {
                    dst[jj] = src[jj];
                }
            }
        }
    }
}

/* Fallback ijk kernel for leaves whose dimensions do not align to
 * the microkernel geometry (m_block not a multiple of MR or n_block
 * not a multiple of NR). Should not be invoked in the regular sweep
 * (m and n are powers of two with m >= 4, n >= 16) but we keep it
 * for safety. Indexing through A_local lets us reuse the
 * materialization for the slow path too. */
static void kernel_base_morton_avx2_ijk_fallback(
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

/* Overwrite-variant leaf: materializes A_local, zeros the C tile,
 * then runs the microkernel which accumulates into the (zeroed)
 * tile. The combination is equivalent to "C = A * B" on the leaf. */
static void kernel_base_morton_avx2(
    scalar_t *C, const scalar_t *A_morton, const scalar_t *B,
    size_t m_block, size_t k_block, size_t n_block,
    size_t a_morton_offset, size_t a_block_dim,
    size_t ldc, size_t ldb,
    scalar_t *A_local_scratch)
{
    materialize_a_panel(A_morton, a_morton_offset, a_block_dim,
                        A_local_scratch, m_block, k_block);

    if ((m_block % MR) != 0 || (n_block % NR) != 0) {
        /* Fallback: clear C first then accumulate via the slow path. */
        for (size_t i = 0; i < m_block; ++i) {
            scalar_t *crow = &C[i * ldc];
            for (size_t j = 0; j < n_block; ++j) crow[j] = (scalar_t)0;
        }
        kernel_base_morton_avx2_ijk_fallback(C, ldc,
                                             A_local_scratch, k_block,
                                             B, ldb,
                                             m_block, k_block, n_block,
                                             /*accumulate=*/1);
        return;
    }

    /* Zero the C tile so that the microkernel (which accumulates
     * starting from whatever is in C) ends with C = A * B exactly. */
    for (size_t i = 0; i < m_block; ++i) {
        scalar_t *crow = &C[i * ldc];
        for (size_t j = 0; j < n_block; ++j) crow[j] = (scalar_t)0;
    }

    /* Dispatch the microkernel over every 4x16 tile of the leaf. The
     * microkernel's lda is k_block (the row stride of A_local), and
     * its ldb / ldc are the caller-supplied strides (which point
     * into the global B and C). */
    for (size_t ii = 0; ii < m_block; ii += MR) {
        for (size_t jj = 0; jj < n_block; jj += NR) {
            kernel_avx2_4x16(&C[ii * ldc + jj], ldc,
                             &A_local_scratch[ii * k_block], k_block,
                             &B[jj], ldb,
                             k_block);
        }
    }
}

/* Accumulate-variant leaf: like the overwrite version but skips the
 * zeroing of C, so the microkernel accumulates on top of the previous
 * value. Used when the recursion split on k. */
static void kernel_base_morton_avx2_add(
    scalar_t *C, const scalar_t *A_morton, const scalar_t *B,
    size_t m_block, size_t k_block, size_t n_block,
    size_t a_morton_offset, size_t a_block_dim,
    size_t ldc, size_t ldb,
    scalar_t *A_local_scratch)
{
    materialize_a_panel(A_morton, a_morton_offset, a_block_dim,
                        A_local_scratch, m_block, k_block);

    if ((m_block % MR) != 0 || (n_block % NR) != 0) {
        kernel_base_morton_avx2_ijk_fallback(C, ldc,
                                             A_local_scratch, k_block,
                                             B, ldb,
                                             m_block, k_block, n_block,
                                             /*accumulate=*/1);
        return;
    }

    for (size_t ii = 0; ii < m_block; ii += MR) {
        for (size_t jj = 0; jj < n_block; jj += NR) {
            kernel_avx2_4x16(&C[ii * ldc + jj], ldc,
                             &A_local_scratch[ii * k_block], k_block,
                             &B[jj], ldb,
                             k_block);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Recursion                                                           */
/*                                                                     */
/* Identical in shape to matmul_morton_inner / _inner_add: split n     */
/* when it is the largest dimension; otherwise split m and k together  */
/* into Morton quadrants of A. The scratch buffer for A_local is       */
/* threaded through every call so the leaves do not malloc.            */
/* ------------------------------------------------------------------ */

static void matmul_morton_avx2_inner(scalar_t *C,
                                     const scalar_t *A_morton,
                                     const scalar_t *B,
                                     size_t m_block, size_t k_block,
                                     size_t n_block,
                                     size_t a_morton_offset,
                                     size_t a_block_dim,
                                     size_t ldc, size_t ldb,
                                     scalar_t *A_local_scratch);

static void matmul_morton_avx2_inner_add(scalar_t *C,
                                         const scalar_t *A_morton,
                                         const scalar_t *B,
                                         size_t m_block, size_t k_block,
                                         size_t n_block,
                                         size_t a_morton_offset,
                                         size_t a_block_dim,
                                         size_t ldc, size_t ldb,
                                         scalar_t *A_local_scratch);

static void matmul_morton_avx2_inner(scalar_t *C,
                                     const scalar_t *A_morton,
                                     const scalar_t *B,
                                     size_t m_block, size_t k_block,
                                     size_t n_block,
                                     size_t a_morton_offset,
                                     size_t a_block_dim,
                                     size_t ldc, size_t ldb,
                                     scalar_t *A_local_scratch)
{
    if (m_block * k_block * n_block <= g_recursion_threshold_avx2) {
        kernel_base_morton_avx2(C, A_morton, B,
                                m_block, k_block, n_block,
                                a_morton_offset, a_block_dim,
                                ldc, ldb,
                                A_local_scratch);
        return;
    }

    /* Case N: split n. A is shared between the two recursive calls. */
    if (n_block > a_block_dim && n_block >= 2) {
        size_t n_half = n_block / 2;
        matmul_morton_avx2_inner(C,          A_morton, B,
                                 m_block, k_block, n_half,
                                 a_morton_offset, a_block_dim,
                                 ldc, ldb, A_local_scratch);
        matmul_morton_avx2_inner(C + n_half, A_morton, B + n_half,
                                 m_block, k_block, n_block - n_half,
                                 a_morton_offset, a_block_dim,
                                 ldc, ldb, A_local_scratch);
        return;
    }

    /* Case MK: split m and k together into the four Morton quadrants
     * of A. With the Morton-of-blocks layout the quadrants still
     * occupy four consecutive (half*half)-sized segments of A_morton:
     * the Z-order key is computed on block coordinates, but the size
     * of each block is constant (MORTON_AVX2_TILE^2 floats), so the
     * offsets scale exactly like the Morton-of-elements case. */
    if (a_block_dim >= 2) {
        assert(m_block == k_block);
        assert(m_block == a_block_dim);

        size_t half          = a_block_dim / 2;
        size_t quadrant_size = half * half;

        /* C_top = A_TL * B_top  (overwrite) */
        matmul_morton_avx2_inner(C, A_morton, B,
                                 half, half, n_block,
                                 a_morton_offset + (size_t)0 * quadrant_size,
                                 half, ldc, ldb, A_local_scratch);

        /* C_top += A_TR * B_bot (accumulate) */
        matmul_morton_avx2_inner_add(C, A_morton, B + half * ldb,
                                     half, half, n_block,
                                     a_morton_offset + (size_t)1 * quadrant_size,
                                     half, ldc, ldb, A_local_scratch);

        /* C_bot = A_BL * B_top  (overwrite) */
        matmul_morton_avx2_inner(C + half * ldc, A_morton, B,
                                 half, half, n_block,
                                 a_morton_offset + (size_t)2 * quadrant_size,
                                 half, ldc, ldb, A_local_scratch);

        /* C_bot += A_BR * B_bot (accumulate) */
        matmul_morton_avx2_inner_add(C + half * ldc, A_morton, B + half * ldb,
                                     half, half, n_block,
                                     a_morton_offset + (size_t)3 * quadrant_size,
                                     half, ldc, ldb, A_local_scratch);
        return;
    }

    /* Degenerate fallback. */
    kernel_base_morton_avx2(C, A_morton, B,
                            m_block, k_block, n_block,
                            a_morton_offset, a_block_dim,
                            ldc, ldb, A_local_scratch);
}

static void matmul_morton_avx2_inner_add(scalar_t *C,
                                         const scalar_t *A_morton,
                                         const scalar_t *B,
                                         size_t m_block, size_t k_block,
                                         size_t n_block,
                                         size_t a_morton_offset,
                                         size_t a_block_dim,
                                         size_t ldc, size_t ldb,
                                         scalar_t *A_local_scratch)
{
    if (m_block * k_block * n_block <= g_recursion_threshold_avx2) {
        kernel_base_morton_avx2_add(C, A_morton, B,
                                    m_block, k_block, n_block,
                                    a_morton_offset, a_block_dim,
                                    ldc, ldb,
                                    A_local_scratch);
        return;
    }

    if (n_block > a_block_dim && n_block >= 2) {
        size_t n_half = n_block / 2;
        matmul_morton_avx2_inner_add(C,          A_morton, B,
                                     m_block, k_block, n_half,
                                     a_morton_offset, a_block_dim,
                                     ldc, ldb, A_local_scratch);
        matmul_morton_avx2_inner_add(C + n_half, A_morton, B + n_half,
                                     m_block, k_block, n_block - n_half,
                                     a_morton_offset, a_block_dim,
                                     ldc, ldb, A_local_scratch);
        return;
    }

    if (a_block_dim >= 2) {
        assert(m_block == k_block);
        assert(m_block == a_block_dim);

        size_t half          = a_block_dim / 2;
        size_t quadrant_size = half * half;

        /* All four products accumulate: we are already inside the
         * _add branch, so C carries a previous value that every
         * product must add onto. */
        matmul_morton_avx2_inner_add(C, A_morton, B,
                                     half, half, n_block,
                                     a_morton_offset + (size_t)0 * quadrant_size,
                                     half, ldc, ldb, A_local_scratch);
        matmul_morton_avx2_inner_add(C, A_morton, B + half * ldb,
                                     half, half, n_block,
                                     a_morton_offset + (size_t)1 * quadrant_size,
                                     half, ldc, ldb, A_local_scratch);
        matmul_morton_avx2_inner_add(C + half * ldc, A_morton, B,
                                     half, half, n_block,
                                     a_morton_offset + (size_t)2 * quadrant_size,
                                     half, ldc, ldb, A_local_scratch);
        matmul_morton_avx2_inner_add(C + half * ldc, A_morton, B + half * ldb,
                                     half, half, n_block,
                                     a_morton_offset + (size_t)3 * quadrant_size,
                                     half, ldc, ldb, A_local_scratch);
        return;
    }

    kernel_base_morton_avx2_add(C, A_morton, B,
                                m_block, k_block, n_block,
                                a_morton_offset, a_block_dim,
                                ldc, ldb, A_local_scratch);
}

/* ------------------------------------------------------------------ */
/* Public wrapper                                                      */
/* ------------------------------------------------------------------ */

/* Return the maximum panel side that can land in the leaf for the
 * current threshold and (assumed minimum) leaf n. The scratch buffer
 * sized at side^2 is wide enough for any leaf the recursion produces
 * before the threshold cuts it off. We use n_floor = NR = 16 as the
 * smallest n a leaf can have without falling into the fallback path. */
static size_t scratch_side_for_threshold(size_t threshold)
{
    size_t target = threshold / (size_t)NR;
    /* Round up to the next power of two. */
    size_t side = 1;
    while (side * side < target) side <<= 1;
    /* Guarantee a minimum so degenerate calls do not crash. */
    if (side < 64) side = 64;
    return side;
}

void matmul_morton_avx2(scalar_t *C,
                        const scalar_t *A_morton,
                        const scalar_t *B,
                        size_t m, size_t k, size_t n)
{
    if (m != k) {
        fprintf(stderr,
                "Error in matmul_morton_avx2: A must be square "
                "(got m=%llu, k=%llu).\n",
                (unsigned long long)m, (unsigned long long)k);
        exit(EXIT_FAILURE);
    }
    if (!is_power_of_two(m) || m < (size_t)MR) {
        fprintf(stderr,
                "Error in matmul_morton_avx2: m (%llu) must be a power "
                "of two and at least MR = %d.\n",
                (unsigned long long)m, MR);
        exit(EXIT_FAILURE);
    }

    size_t side = scratch_side_for_threshold(g_recursion_threshold_avx2);
    /* Clamp to the actual problem so we never allocate more than m*m. */
    if (side > m) side = m;
    scalar_t *A_local = xalloc_aligned(side * side);

    matmul_morton_avx2_inner(C, A_morton, B,
                             m, k, n,
                             /* a_morton_offset = */ 0,
                             /* a_block_dim     = */ m,
                             /* ldc = */ n,
                             /* ldb = */ n,
                             A_local);

    xfree(A_local);
}

/* ------------------------------------------------------------------ */
/* Benchmark orchestrators                                             */
/* ------------------------------------------------------------------ */

void benchmark_iterations_morton_avx2(scalar_t *B_out,
                                      const scalar_t *A,
                                      const scalar_t *Z,
                                      size_t m, size_t n,
                                      size_t num_iters)
{
    scalar_t *A_morton = xalloc_aligned(m * m);
    reorganize_to_morton_blocks(A, A_morton, m);

    benchmark_iterations_morton_avx2_preorganized(B_out, A_morton, Z,
                                                  m, n, num_iters);

    xfree(A_morton);
}

void benchmark_iterations_morton_avx2_preorganized(scalar_t *B_out,
                                                   const scalar_t *A_morton,
                                                   const scalar_t *Z,
                                                   size_t m, size_t n,
                                                   size_t num_iters)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        matmul_morton_avx2(B_next, A_morton, B_curr, m, m, n);

        /* Store the first n rows of B_next into the output buffer. */
        scalar_t *out_block = B_out + iter * n * n;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                out_block[i * n + j] = B_next[i * n + j];
            }
        }

        /* Swap so next iteration consumes what we just produced. */
        scalar_t *tmp = B_curr;
        B_curr = B_next;
        B_next = tmp;
    }

    xfree(B_curr);
    xfree(B_next);
}
