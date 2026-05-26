#ifndef MATMUL_TILED_IKJ_H
#define MATMUL_TILED_IKJ_H

#include <stddef.h>
#include "matrix_utils.h"   /* scalar_t */

// Parámetros específicos para L2 cache de 1 MiB por núcleo (AMD EPYC 9R45).
// Mc=Kc=384: A(576 KB) + B(192 KB) + C(192 KB) = 960 KB ≈ 94% del L2.
#ifndef TILED_IKJ_MC_DEFAULT
#define TILED_IKJ_MC_DEFAULT 384u
#endif
#ifndef TILED_IKJ_KC_DEFAULT
#define TILED_IKJ_KC_DEFAULT 384u
#endif


void matmul_tiled_ikj(scalar_t *C,
                  const scalar_t *A,
                  const scalar_t *B,
                  size_t m, size_t k_dim, size_t n);


void benchmark_iterations_tiled_ikj(scalar_t *B_out,
                                 const scalar_t *A,
                                 const scalar_t *Z,
                                 size_t m, size_t n,
                                 size_t num_iters);

#endif /* MATMUL_TILED_IKJ_H */
