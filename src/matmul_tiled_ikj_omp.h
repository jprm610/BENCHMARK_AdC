/*
 * matmul_tiled_ikj_omp.h - OpenMP-parallel BLIS-style 6x16 register-blocked
 *                      matmul. Sibling of matmul_tiled_ikj_avx2 with the
 *                      same microkernel; parallelizes the outermost
 *                      mc loop (ic) so each thread owns a contiguous
 *                      range of C rows.
 *
 * Threading model: a single `omp parallel` region is opened in the
 * public function. The pc loop (k tile) runs in lockstep across all
 * threads (sequential semantics, the implicit barrier at the end of
 * `omp for` serves as the sync between pc iterations because all
 * threads must finish accumulating pp before pp+kc reads C). Within
 * each pc iteration, the ic loop is distributed via `omp for
 * schedule(static)`. Residual rows (m % MR != 0) are handled by a
 * single thread inside an `omp single` block.
 *
 * Recommended invocation on Ryzen 5 4600H (Zen 2 Renoir, 2 CCXs x 3
 * cores, L3 4 MiB per CCX):
 *   OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=spread
 * One thread per physical core, distributed across both CCXs. SMT
 * threads (12 total) usually hurt FMA-bound kernels because the two
 * sibling logical CPUs share the FMA pipes.
 */

#ifndef MATMUL_TILED_IKJ_OMP_H
#define MATMUL_TILED_IKJ_OMP_H

#include <stddef.h>
#include "matrix_utils.h"   /* scalar_t */

#ifndef TILED_IKJ_OMP_BS_DEFAULT
#define TILED_IKJ_OMP_BS_DEFAULT 384u
#endif
#define TILED_IKJ_OMP_MR 6u
#define TILED_IKJ_OMP_NR 16u
#ifndef TILED_IKJ_OMP_MC
#define TILED_IKJ_OMP_MC 192u
#endif

extern size_t g_tiled_ikj_omp_bs;

void matmul_tiled_ikj_omp_set_bs(size_t bs);

void matmul_tiled_ikj_omp(scalar_t *C,
                      const scalar_t *A,
                      const scalar_t *B,
                      size_t m, size_t k, size_t n);

void benchmark_iterations_tiled_ikj_omp(scalar_t *B_out,
                                    const scalar_t *A,
                                    const scalar_t *Z,
                                    size_t m, size_t n,
                                    size_t num_iters);

#endif /* MATMUL_TILED_IKJ_OMP_H */
