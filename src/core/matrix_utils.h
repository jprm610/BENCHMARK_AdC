// Include guard para evitar que matrix_utils.h sea incluido múltiples veces.
#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <stddef.h>

/*
 * scalar_t: Switch centralizado para cambiar tipo a los números de la matriz.
 */
typedef float scalar_t;


scalar_t *xalloc_aligned(size_t num_elements);


void xfree(scalar_t *ptr);


void init_matrix_random(scalar_t *M,
                        size_t rows, size_t cols,
                        unsigned int seed);


void init_matrix_zero(scalar_t *M, size_t rows, size_t cols);


void init_matrix_identity(scalar_t *M, size_t n);


int matrices_close(const scalar_t *A_ref,
                   const scalar_t *A_test,
                   size_t num_elements,
                   scalar_t abs_tol,
                   scalar_t rel_tol,
                   size_t *first_bad_index,
                   scalar_t *bad_ref,
                   scalar_t *bad_test);

#endif /* MATRIX_UTILS_H */
