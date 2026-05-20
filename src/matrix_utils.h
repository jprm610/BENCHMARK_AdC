/*
 * matrix_utils.h - Allocation, initialization, and comparison helpers.
 *
 * These helpers are independent of the kernel implementation and remain
 * unchanged as optimized variants are added later in the project.
 */

#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <stddef.h>

/*
 * scalar_t is the floating-point type used by every matrix in the project.
 * Centralized here so that switching to double touches only this typedef
 * and the tolerance constants in validate.c.
 */
typedef float scalar_t;

/*
 * xalloc_aligned: allocate num_elements * sizeof(scalar_t) bytes with
 * 64-byte alignment (cache-line size on x86-64). Aborts the process on
 * failure with a message on stderr. Memory is uninitialized.
 */
scalar_t *xalloc_aligned(size_t num_elements);

/*
 * xfree: free a pointer returned by xalloc_aligned. Tolerates NULL.
 */
void xfree(scalar_t *ptr);

/*
 * init_matrix_random: fill M with reproducible pseudo-random values in
 * [-1/sqrt(rows), +1/sqrt(rows)]. The scaling keeps the spectral norm of
 * A bounded so that the recurrence B_{i+1} = A * B_i does not overflow.
 *
 * The generator is a simple LCG seeded with `seed`. Same seed produces
 * the same matrix on every platform and compiler.
 */
void init_matrix_random(scalar_t *M,
                        size_t rows, size_t cols,
                        unsigned int seed);

/*
 * init_matrix_zero: set rows * cols elements of M to zero.
 */
void init_matrix_zero(scalar_t *M, size_t rows, size_t cols);

/*
 * init_matrix_identity: fill the n x n matrix M with the identity.
 */
void init_matrix_identity(scalar_t *M, size_t n);

/*
 * matrices_close: element-wise tolerance comparison.
 *
 * Returns 1 if |A_ref[i] - A_test[i]| <= max(abs_tol, rel_tol * |A_ref[i]|)
 * for every i in [0, num_elements); returns 0 otherwise.
 *
 * If the result is 0 and the optional out-parameters are non-NULL, the
 * function writes the first failing index and the two values that differ
 * (useful for debugging).
 */
int matrices_close(const scalar_t *A_ref,
                   const scalar_t *A_test,
                   size_t num_elements,
                   scalar_t abs_tol,
                   scalar_t rel_tol,
                   size_t *first_bad_index,
                   scalar_t *bad_ref,
                   scalar_t *bad_test);

#endif /* MATRIX_UTILS_H */
