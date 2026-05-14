/*
 * matmul_naive.h - Naive matrix multiplication and iterative benchmark driver.
 *
 * This is the baseline (Phase 1) implementation of the iterated matrix
 * multiplication benchmark. It is intentionally unoptimized: three nested
 * loops in ijk order, no blocking, no transposition, no SIMD intrinsics.
 *
 * See docs/API.md for the public contract.
 */

#ifndef MATMUL_NAIVE_H
#define MATMUL_NAIVE_H

#include <stddef.h>

/*
 * scalar_t is the floating-point type used by every matrix in the project.
 * Centralized here so that switching to double touches only this typedef
 * and the tolerance constants in validate.c.
 */
typedef float scalar_t;

/*
 * matmul_naive: compute C = A * B with three nested loops in ijk order.
 *
 * - C is m x n (output, overwritten).
 * - A is m x k (input).
 * - B is k x n (input).
 *
 * All matrices are row-major. C must not alias A or B. No optimization of
 * any kind. This function is the immovable baseline against which all
 * future versions are compared.
 *
 * Complexity: 2 * m * k * n floating-point operations.
 */
void matmul_naive(scalar_t *C,
                  const scalar_t *A,
                  const scalar_t *B,
                  size_t m, size_t k, size_t n);

/*
 * benchmark_iterations: run the recurrence B_{i+1} = A * B_i with B_0 = Z,
 * storing the first n rows of each B_{i+1} into the flat output buffer.
 *
 * - B_out has space for num_iters * n * n elements.
 *   Block iter (0-based) occupies offsets [iter*n*n, (iter+1)*n*n).
 * - A is m x m (constant across iterations).
 * - Z is m x n (acts as B_0).
 *
 * Uses two internal buffers of size m x n with pointer swapping to avoid
 * copying B between iterations. Both buffers are allocated and freed
 * inside this function.
 */
void benchmark_iterations(scalar_t *B_out,
                          const scalar_t *A,
                          const scalar_t *Z,
                          size_t m, size_t n,
                          size_t num_iters);

#endif /* MATMUL_NAIVE_H */
