#ifndef MATMUL_TILED_IKJ_AVX2_H
#define MATMUL_TILED_IKJ_AVX2_H

#include <stddef.h>
#include "matrix_utils.h"   /* scalar_t */

// Hiperparámetro para que B (KC x NR) quepa holgadamente en L1d. (Modificable)
#define TILED_IKJ_AVX2_BS_DEFAULT 384u

// Hiperparámetros según arquitectura AVX2.
#define TILED_IKJ_AVX2_MR 6u
#define TILED_IKJ_AVX2_NR 16u
#define TILED_IKJ_AVX2_MC 192u

extern size_t g_tiled_ikj_avx2_bs;

void matmul_tiled_ikj_avx2_set_bs(size_t bs);

void matmul_tiled_ikj_avx2(scalar_t *C,
                       const scalar_t *A,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n);

void benchmark_iterations_tiled_ikj_avx2(scalar_t *B_out,
                                     const scalar_t *A,
                                     const scalar_t *Z,
                                     size_t m, size_t n,
                                     size_t num_iters);

#endif /* MATMUL_TILED_IKJ_AVX2_H */
