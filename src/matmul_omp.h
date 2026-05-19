/*
 * matmul_omp.h - 6-loop tiled matmul with AVX2+FMA and OpenMP parallelization.
 *
 * Extends matmul_tiled_avx2 by parallelizing the outermost ii loop with
 * a single #pragma omp parallel for schedule(static). The inner AVX2
 * broadcast+FMA loop is unchanged.
 *
 * Thread count is controlled by OMP_NUM_THREADS at runtime; the binary
 * does not hard-code a thread count.
 */

#ifndef MATMUL_OMP_H
#define MATMUL_OMP_H

#include <stddef.h>
#include "matmul_naive.h"   /* scalar_t */

#define OMP_BS_DEFAULT 64u

extern size_t g_omp_bs;

void matmul_omp_set_bs(size_t bs);

void matmul_omp(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k, size_t n);

void benchmark_iterations_omp(scalar_t *B_out,
                               const scalar_t *A,
                               const scalar_t *Z,
                               size_t m, size_t n,
                               size_t num_iters);

#endif /* MATMUL_OMP_H */
