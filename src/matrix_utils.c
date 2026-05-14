/*
 * matrix_utils.c - Implementation of allocation and matrix helpers.
 *
 * The PRNG used by init_matrix_random is a tiny LCG: not statistically
 * great, but deterministic across platforms and good enough to fill
 * matrices with non-trivial inputs for performance measurements.
 */

#include "matrix_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <errno.h>

scalar_t *xalloc_aligned(size_t num_elements)
{
    size_t bytes = num_elements * sizeof(scalar_t);

    /* C11 aligned_alloc requires size to be a multiple of the alignment;
     * round up to the next multiple of 64 bytes. */
    size_t rounded = (bytes + 63u) & ~((size_t)63u);
    void *ptr = aligned_alloc(64, rounded);
    if (ptr == NULL) {
        fprintf(stderr,
                "xalloc_aligned: aligned_alloc failed "
                "(elements=%llu, bytes=%llu)\n",
                (unsigned long long)num_elements,
                (unsigned long long)rounded);
        exit(EXIT_FAILURE);
    }
    return (scalar_t *)ptr;
}

void xfree(scalar_t *ptr)
{
    if (ptr != NULL) {
        free(ptr);
    }
}

/* Tiny LCG, parameters from Numerical Recipes. Produces a 32-bit stream. */
static unsigned int lcg_next(unsigned int *state)
{
    *state = (*state) * 1664525u + 1013904223u;
    return *state;
}

void init_matrix_random(scalar_t *M,
                        size_t rows, size_t cols,
                        unsigned int seed)
{
    unsigned int state = seed ? seed : 1u;
    scalar_t scale = (scalar_t)(1.0 / sqrt((double)rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            unsigned int r = lcg_next(&state);
            /* Map r to a float in [-1, 1] and scale by 1/sqrt(rows). */
            scalar_t u = (scalar_t)((double)r / (double)UINT32_MAX) * (scalar_t)2.0 - (scalar_t)1.0;
            M[i * cols + j] = u * scale;
        }
    }
}

void init_matrix_zero(scalar_t *M, size_t rows, size_t cols)
{
    memset(M, 0, rows * cols * sizeof(scalar_t));
}

void init_matrix_identity(scalar_t *M, size_t n)
{
    init_matrix_zero(M, n, n);
    for (size_t i = 0; i < n; ++i) {
        M[i * n + i] = (scalar_t)1;
    }
}

int matrices_close(const scalar_t *A_ref,
                   const scalar_t *A_test,
                   size_t num_elements,
                   scalar_t abs_tol,
                   scalar_t rel_tol,
                   size_t *first_bad_index,
                   scalar_t *bad_ref,
                   scalar_t *bad_test)
{
    for (size_t i = 0; i < num_elements; ++i) {
        scalar_t r = A_ref[i];
        scalar_t t = A_test[i];
        scalar_t diff = (scalar_t)fabs((double)(r - t));
        scalar_t mag = (scalar_t)fabs((double)r);
        scalar_t tol = abs_tol > rel_tol * mag ? abs_tol : rel_tol * mag;
        if (diff > tol) {
            if (first_bad_index) *first_bad_index = i;
            if (bad_ref)         *bad_ref = r;
            if (bad_test)        *bad_test = t;
            return 0;
        }
    }
    return 1;
}
