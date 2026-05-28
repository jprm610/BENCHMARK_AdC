#ifndef MATMUL_TILED_IKJ_OMP_H
#define MATMUL_TILED_IKJ_OMP_H

#include <stddef.h>
#include "matrix_utils.h"   /* scalar_t */

/* Defaults sized for the EPYC 9R45: B panel kc * NR * 4 B fits in
 * L1d (48 KiB) at kc = 256, A panel mc * kc * 4 B fits in L2 (1 MiB)
 * at mc = 288. The Makefile may override at compile time. */
#ifndef TILED_IKJ_OMP_BS_DEFAULT
#define TILED_IKJ_OMP_BS_DEFAULT 256u
#endif
#define TILED_IKJ_OMP_MR 6u
#define TILED_IKJ_OMP_NR 32u
#ifndef TILED_IKJ_OMP_MC
#define TILED_IKJ_OMP_MC 288u
#endif

extern size_t g_tiled_ikj_omp_bs;

void matmul_tiled_ikj_omp_set_bs(size_t bs);

void matmul_tiled_ikj_omp(scalar_t *C,
                      const scalar_t *A,
                      const scalar_t *B,
                      size_t m, size_t k_dim, size_t n);

void benchmark_iterations_tiled_ikj_omp(scalar_t *B_out,
                                    const scalar_t *A,
                                    const scalar_t *Z,
                                    size_t m, size_t n,
                                    size_t num_iters);

#endif /* MATMUL_TILED_IKJ_OMP_H */
