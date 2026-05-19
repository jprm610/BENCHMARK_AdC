/*
 * matmul_tiled_avx2.h - 6-loop tiled matmul with AVX2+FMA vectorization.
 *
 * Extends the 2D tiling of matmul_tiled (Phase 1.2) by adding a third
 * outer blocking loop over j (jj) and replacing the scalar innermost j
 * loop with an AVX2 broadcast+FMA pass of 8 floats per instruction.
 *
 * Loop structure (all three outer loops tile the same block size BS):
 *
 *   for ii  (step BS, blocks over i)
 *     for kk  (step BS, blocks over k)
 *       for jj  (step BS, blocks over j)
 *         for i in [ii, ii+BS)
 *           for p in [kk, kk+BS)
 *             a_vec = broadcast(A[i,p])
 *             for j in [jj, jj+BS), step 8   <- AVX2
 *               C[i,j:j+8] += a_vec * B[p,j:j+8]
 *             tail loop for remaining j
 *
 * Block size: TILED_AVX2_BS_DEFAULT (64). Must be a positive multiple
 * of 8 so the AVX2 pass always covers at least one full vector; the
 * tail loop handles non-multiple-of-8 remainders when n % BS != 0.
 *
 * The block size can be changed at runtime via matmul_tiled_avx2_set_bs;
 * the bench driver uses this to explore different values from the
 * command line without recompiling.
 */

#ifndef MATMUL_TILED_AVX2_H
#define MATMUL_TILED_AVX2_H

#include <stddef.h>
#include "matmul_naive.h"   /* scalar_t */

#define TILED_AVX2_BS_DEFAULT 64u

extern size_t g_tiled_avx2_bs;

void matmul_tiled_avx2_set_bs(size_t bs);

/*
 * matmul_tiled_avx2: compute C = A * B with 6-loop blocking and AVX2.
 *
 * Same external contract as matmul_tiled / matmul_naive:
 *   C (m x n) is overwritten (memset to zero before the tiled loops).
 *   A (m x k) and B (k x n) are read-only.
 *   No aliasing between C, A, B.
 *
 * Requires: the binary is compiled with -mavx2 -mfma (-O3 -march=znver2
 * or equivalent) so _mm256_fmadd_ps emits a true FMA instruction.
 */
void matmul_tiled_avx2(scalar_t *C,
                       const scalar_t *A,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n);

/*
 * benchmark_iterations_tiled_avx2: same recurrence driver as
 * benchmark_iterations_tiled but delegates each A*B step to
 * matmul_tiled_avx2.
 */
void benchmark_iterations_tiled_avx2(scalar_t *B_out,
                                     const scalar_t *A,
                                     const scalar_t *Z,
                                     size_t m, size_t n,
                                     size_t num_iters);

#endif /* MATMUL_TILED_AVX2_H */
