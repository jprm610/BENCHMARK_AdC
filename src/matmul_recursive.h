/*
 * matmul_recursive.h - Cache-oblivious recursive matrix multiplication.
 *
 * Phase 6, Stage A2 of the project. Implements C = A * B by recursively
 * splitting the largest of the three dimensions until the sub-problem fits
 * in some cache level (controlled by RECURSION_THRESHOLD inside the .c
 * file). No explicit block size is chosen; the cache hierarchy is exploited
 * automatically.
 *
 * The public signature is identical in structure to matmul_naive, so the
 * cross-validation harness can swap one kernel for the other without
 * changes.
 *
 * Layout is row-major. C must not alias A or B. See docs/API.md for the
 * full contract.
 */

#ifndef MATMUL_RECURSIVE_H
#define MATMUL_RECURSIVE_H

#include <stddef.h>

#include "matmul_naive.h"  /* for scalar_t */

/*
 * matmul_recursive: compute C = A * B by cache-oblivious recursion.
 *
 * - C is m x n (output, overwritten).
 * - A is m x k (input).
 * - B is k x n (input).
 *
 * Row-major. C must not alias A or B. The recursion splits the largest
 * dimension of {m, k, n} until m*k*n falls under RECURSION_THRESHOLD; the
 * base case is an ijk triple loop equivalent to matmul_naive.
 *
 * Complexity: 2 * m * k * n floating-point operations, same as the naive
 * kernel, but with much better locality on large m.
 */
void matmul_recursive(scalar_t *C,
                      const scalar_t *A,
                      const scalar_t *B,
                      size_t m, size_t k, size_t n);

/*
 * benchmark_iterations_recursive: same semantics as benchmark_iterations
 * declared in matmul_naive.h, but the inner kernel is matmul_recursive
 * instead of matmul_naive.
 *
 * Lives in matmul_recursive.c rather than matmul_naive.c because the
 * naive translation unit is immutable (Phase 1 baseline rule).
 *
 * - B_out has space for num_iters * n * n elements; block iter occupies
 *   offsets [iter*n*n, (iter+1)*n*n).
 * - A is m x m (constant across iterations).
 * - Z is m x n (acts as B_0).
 *
 * Uses two internal buffers of size m x n with pointer swapping. Both
 * are allocated and freed inside the function.
 */
void benchmark_iterations_recursive(scalar_t *B_out,
                                    const scalar_t *A,
                                    const scalar_t *Z,
                                    size_t m, size_t n,
                                    size_t num_iters);

#endif /* MATMUL_RECURSIVE_H */
