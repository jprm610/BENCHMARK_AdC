#ifndef MATMUL_LOOPS_H
#define MATMUL_LOOPS_H

#include <stddef.h>
#include "matrix_utils.h"   /* scalar_t */
typedef void (*matmul_fn_t)(scalar_t *C,
                             const scalar_t *A,
                             const scalar_t *B,
                             size_t m, size_t k, size_t n);

// 6 VARIANTES

void matmul_ijk(scalar_t *C, const scalar_t *A, const scalar_t *B,
                size_t m, size_t k, size_t n);

void matmul_ikj(scalar_t *C, const scalar_t *A, const scalar_t *B,
                size_t m, size_t k, size_t n);

void matmul_jik(scalar_t *C, const scalar_t *A, const scalar_t *B,
                size_t m, size_t k, size_t n);

void matmul_jki(scalar_t *C, const scalar_t *A, const scalar_t *B,
                size_t m, size_t k, size_t n);

void matmul_kij(scalar_t *C, const scalar_t *A, const scalar_t *B,
                size_t m, size_t k, size_t n);

void matmul_kji(scalar_t *C, const scalar_t *A, const scalar_t *B,
                size_t m, size_t k, size_t n);

// ---------------------------------------------------------------------
// Auxiliares
matmul_fn_t matmul_loops_lookup(const char *name);

void benchmark_iterations_loops(scalar_t *B_out,
                                 const scalar_t *A,
                                 const scalar_t *Z,
                                 size_t m, size_t n,
                                 size_t num_iters,
                                 matmul_fn_t kernel);

#endif /* MATMUL_LOOPS_H */
