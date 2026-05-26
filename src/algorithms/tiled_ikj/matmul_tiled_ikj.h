#ifndef MATMUL_TILED_IKJ_H
#define MATMUL_TILED_IKJ_H

#include <stddef.h>
#include "matrix_utils.h"   /* scalar_t */

// Parámetros específicos para L2 cache de 512 KB (Ryzen 5 4600H).
#define TILED_IKJ_MC_DEFAULT 256u
#define TILED_IKJ_KC_DEFAULT 256u


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
