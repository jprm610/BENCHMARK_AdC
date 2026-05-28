#ifndef MATMUL_TILED_IKJ_AVX512_H
#define MATMUL_TILED_IKJ_AVX512_H

#include <stddef.h>
#include "matrix_utils.h"   /* scalar_t */

#ifndef TILED_IKJ_AVX512_BS_DEFAULT
#define TILED_IKJ_AVX512_BS_DEFAULT 256u
#endif

#define TILED_IKJ_AVX512_MR 6u
#define TILED_IKJ_AVX512_NR 32u
#ifndef TILED_IKJ_AVX512_MC
#define TILED_IKJ_AVX512_MC 288u
#endif

extern size_t g_tiled_ikj_avx512_bs;

void matmul_tiled_ikj_avx512_set_bs(size_t bs);

void matmul_tiled_ikj_avx512(scalar_t *C,
                       const scalar_t *A,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n);

void benchmark_iterations_tiled_ikj_avx512(scalar_t *B_out,
                                     const scalar_t *A,
                                     const scalar_t *Z,
                                     size_t m, size_t n,
                                     size_t num_iters);

#endif /* MATMUL_TILED_IKJ_AVX512_H */
