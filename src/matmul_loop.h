/*
 * matmul_loop.h - All six loop-order variants of C = A * B.
 *
 * Each function has the same signature as matmul_naive so that the
 * validation harness can compare them without modification.
 *
 * The six orders are named by the sequence (outer, middle, inner) over
 * the three loop indices:
 *   i  - row index of C and A          (0 .. m-1)
 *   j  - column index of C and B       (0 .. n-1)
 *   k  - shared (reduction) dimension  (0 .. k_dim-1)
 *
 * Because k_dim is already a function parameter, the loop variable over
 * the shared dimension is named p throughout the implementation to avoid
 * shadowing.
 *
 * See docs/API.md for the full public contract.
 */

#ifndef MATMUL_LOOP_H
#define MATMUL_LOOP_H

#include <stddef.h>
#include "matmul_naive.h"   /* scalar_t */

/* Function-pointer type shared by all six variants. */
typedef void (*matmul_fn_t)(scalar_t *C,
                             const scalar_t *A,
                             const scalar_t *B,
                             size_t m, size_t k, size_t n);

/* ---- Six loop-order kernels ---- */

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

/*
 * matmul_loop_lookup: return the function pointer for the named order,
 * or NULL if the name is not one of the six above.
 */
matmul_fn_t matmul_loop_lookup(const char *name);

/*
 * benchmark_iterations_loop: same semantics as benchmark_iterations
 * (matmul_naive.h) but delegates each A*B step to the supplied kernel.
 *
 * - B_out : flat buffer of num_iters * n * n elements (output).
 * - A     : m x m, constant across iterations.
 * - Z     : m x n, initial B_0.
 * - kernel: any of the six variants (or matmul_naive itself).
 */
void benchmark_iterations_loop(scalar_t *B_out,
                                const scalar_t *A,
                                const scalar_t *Z,
                                size_t m, size_t n,
                                size_t num_iters,
                                matmul_fn_t kernel);

#endif /* MATMUL_LOOP_H */
