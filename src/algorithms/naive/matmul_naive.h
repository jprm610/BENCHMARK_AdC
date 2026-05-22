#ifndef MATMUL_NAIVE_H
#define MATMUL_NAIVE_H

#include <stddef.h>
#include "matrix_utils.h"


void matmul_naive(scalar_t *C,
                  const scalar_t *A,
                  const scalar_t *B,
                  size_t m, size_t k, size_t n);


void benchmark_iterations(scalar_t *B_out,
                          const scalar_t *A,
                          const scalar_t *Z,
                          size_t m, size_t n,
                          size_t num_iters);

#endif /* MATMUL_NAIVE_H */
