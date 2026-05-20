/*
 * matmul_tiled_ikj.h - Explicitly tiled ikj matrix multiplication.
 *
 * Applies two-level cache blocking over the i and k dimensions on top of
 * the ikj loop order, which was identified as the best single-order variant
 * in the loop-reorder study (Phase 1.1).
 *
 * Default tile sizes target the L2 cache of the Ryzen 5 4600H (512 KB):
 *   A panel : Mc x Kc floats = 256 x 256 x 4 B = 256 KB
 *   B panel : Kc x n  floats = 256 x 128 x 4 B = 128 KB
 *   C panel : Mc x n  floats = 256 x 128 x 4 B = 128 KB
 *   Total                                        = 512 KB  (= L2)
 *
 * The j dimension is never tiled because n = 128 is small and fixed for
 * this benchmark, so each B row and C row fit in a few cache lines.
 *
 * See docs/API.md for the full public contract.
 */

#ifndef MATMUL_TILED_IKJ_H
#define MATMUL_TILED_IKJ_H

#include <stddef.h>
#include "matmul_naive.h"   /* scalar_t */

#ifndef TILED_IKJ_MC_DEFAULT
#define TILED_IKJ_MC_DEFAULT 256u
#endif
#ifndef TILED_IKJ_KC_DEFAULT
#define TILED_IKJ_KC_DEFAULT 256u
#endif

/*
 * matmul_tiled_ikj: compute C = A * B with explicit Mc x Kc blocking.
 *
 * Uses TILED_IKJ_MC_DEFAULT and TILED_IKJ_KC_DEFAULT as tile sizes.
 * Same signature as matmul_naive; C is overwritten (not accumulated).
 */
void matmul_tiled_ikj(scalar_t *C,
                  const scalar_t *A,
                  const scalar_t *B,
                  size_t m, size_t k, size_t n);

/*
 * benchmark_iterations_tiled_ikj: same semantics as benchmark_iterations
 * (matmul_naive.h) but delegates each A*B step to matmul_tiled_ikj.
 *
 * - B_out    : flat buffer of num_iters * n * n elements (output).
 * - A        : m x m, constant across iterations.
 * - Z        : m x n, initial B_0.
 * - num_iters: number of recurrence steps to measure.
 */
void benchmark_iterations_tiled_ikj(scalar_t *B_out,
                                 const scalar_t *A,
                                 const scalar_t *Z,
                                 size_t m, size_t n,
                                 size_t num_iters);

#endif /* MATMUL_TILED_IKJ_H */
