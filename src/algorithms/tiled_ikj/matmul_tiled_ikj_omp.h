#ifndef MATMUL_TILED_IKJ_OMP_H
#define MATMUL_TILED_IKJ_OMP_H

#include <stddef.h>
#include "matrix_utils.h"   /* scalar_t */

#define TILED_IKJ_OMP_BS_DEFAULT 384u
#define TILED_IKJ_OMP_MR 6u
#define TILED_IKJ_OMP_NR 16u
#define TILED_IKJ_OMP_MC 192u

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
