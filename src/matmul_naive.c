/*
 * matmul_naive.c - Phase 1 baseline implementation.
 *
 * Deliberately unoptimized. Do not add restrict, blocking, transposition,
 * or any compiler-friendly rewrite to this file. Optimized variants go
 * into separate translation units so we can measure each one in isolation.
 */

#include "matmul_naive.h"
#include "matrix_utils.h"

#include <string.h>

void matmul_naive(scalar_t *C,
                  const scalar_t *A,
                  const scalar_t *B,
                  size_t m, size_t k, size_t n)
{
    /* ijk loop order: walks A row by row (stride 1), B column by column
     * (stride n, bad locality), and accumulates into a scalar.
     * This is the textbook formulation; it is also the worst common
     * layout for spatial locality on B. Kept on purpose as the baseline. */
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            scalar_t acc = (scalar_t)0;
            for (size_t p = 0; p < k; ++p) {
                acc += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = acc;
        }
    }
}

void benchmark_iterations(scalar_t *B_out,
                          const scalar_t *A,
                          const scalar_t *Z,
                          size_t m, size_t n,
                          size_t num_iters)
{
    /* Two working buffers, swapped via pointer reassignment after each
     * multiplication. Avoids copying m*n floats per iteration. */
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    /* Initialize B_curr = Z (a plain memcpy of m*n scalars). */
    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        /* B_next = A * B_curr */
        matmul_naive(B_next, A, B_curr, m, m, n);

        /* Store the first n rows of B_next into the output buffer. */
        scalar_t *out_block = B_out + iter * n * n;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                out_block[i * n + j] = B_next[i * n + j];
            }
        }

        /* Swap: B_curr <- B_next, so the new B_curr is the just-computed
         * B_{iter+1}, and the old B_curr becomes scratch for the next pass. */
        scalar_t *tmp = B_curr;
        B_curr = B_next;
        B_next = tmp;
    }

    xfree(B_curr);
    xfree(B_next);
}
