// matrix_utils.c - Implementación para utilidades de matrices, alinearlas en caché,
// generación con lcg (determinístico).

#include "matrix_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <errno.h>


/*
    *xalloc_aligned: Asignar memoria para la matriz.
        INPUTS:
        - num_elements: Número total de elementos (filas * columnas).
        OUTPUTS:
        - Puntero a la memoria asignada, alineada a 64 bytes.
*/
scalar_t *xalloc_aligned(size_t num_elements)
{
    /*
    size_t <= <stddef.h>
    - Para tamaños en memoria
    - int: [0, 2^31 - 1] => Matrices muy grandes >= overflow
    - size_t: [0, 2^64 - 1] en sistemas de 64 bits => Soporta matrices muy grandes (4 TB)
    */
    size_t bytes = num_elements * sizeof(scalar_t);

    /*
    Hacia arriba en múltiplos de 64.
    - Sumar 64 y enmascarar con un not de 63. (0 en los 6 bits menos significativos).
    */
    size_t rounded = (bytes + 63u) & ~((size_t)63u);

    /*
    aligned_alloc: Asigna memoria alineada.
    - Desde un múltiplo de 64.
    - Hasta el rounded calculado. (Supera al tamaño de la matriz).
    */
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


/*
xfree: Liberar memoria asignada para la matriz.
    INPUTS:
    - ptr: Puntero a la memoria asignada.
    OUTPUTS:
    - Ninguno (void).
*/
void xfree(scalar_t *ptr)
{
    if (ptr != NULL) {
        free(ptr);
    }
}


/*
(LCG) Linear Congruential Generator: Generador de números pseudoaleatorios.
    - state: *state * multiplicador + incremento (modulo 2^32).
    - Son unsigned int para evitar problemas de signo y overflow.
    - Devuelve el siguiente número pseudoaleatorio.
*/
static unsigned int lcg_next(unsigned int *state)
{
    *state = (*state) * 1664525u + 1013904223u;
    return *state;
}


/*
init_matrix_random: Inicializar una matriz con valores pseudoaleatorios.
    INPUTS:
    - M: Puntero a la matriz a inicializar.
    - rows: Número de filas de la matriz.
    - cols: Número de columnas de la matriz.
    - seed: Semilla para el generador de números pseudoaleatorios (0 para usar 1).
    OUTPUTS:
    - Ninguno (void).
*/
void init_matrix_random(scalar_t *M,
                        size_t rows, size_t cols,
                        unsigned int seed)
{
    unsigned int state = seed ? seed : 1u;

    /*
    Escala 1/sqrt(m). Evitar overflow en multiplicaciones sucesivas.
    - La norma espectral crece aproximadamente como sqrt(m) cuando m crece >= overflow inevitable.
    - Valor final se puede obtener el valor real multiplicando por sqrt(m)^I.
    */
    scalar_t scale = (scalar_t)(1.0 / sqrt((double)rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            unsigned int r = lcg_next(&state);
            /*
            1. r en [0, 1], dividir por 2^32
            2. Escalar a [-1, 1], multiplicar por 2 y restar 1
            3. Escalar por 1/sqrt(m). (Wigner)
            */
            scalar_t u = (scalar_t)((double)r / (double)UINT32_MAX) * (scalar_t)2.0 - (scalar_t)1.0;
            M[i * cols + j] = u * scale;
        }
    }
}


/*
init_matrix_zero: Inicializar una matriz con ceros. (rows, cols).
    INPUTS:
    - M: Puntero a la matriz a inicializar.
    - rows: Número de filas de la matriz.
    - cols: Número de columnas de la matriz.
    OUTPUTS:
    - Ninguno (void).
*/
void init_matrix_zero(scalar_t *M, size_t rows, size_t cols)
{
    memset(M, 0, rows * cols * sizeof(scalar_t));
}


/*
init_matrix_identity: Inicializar una matriz identidad. (rows, cols).
    INPUTS:
    - M: Puntero a la matriz a inicializar.
    - n: Número de filas y columnas de la matriz.
    OUTPUTS:
    - Ninguno (void).
*/
void init_matrix_identity(scalar_t *M, size_t n)
{
    init_matrix_zero(M, n, n);
    for (size_t i = 0; i < n; ++i) {
        M[i * n + i] = (scalar_t)1;
    }
}


/*
matrices_close: Verificar si dos matrices son "cercanas" dentro de tolerancias.
    INPUTS:
    - A_ref: Puntero a la matriz de referencia.
    - A_test: Puntero a la matriz a comparar.
    - num_elements: Número total de elementos en las matrices.
    - abs_tol: Tolerancia absoluta.
    - rel_tol: Tolerancia relativa.
    OUTPUTS:
    - 1 si las matrices son cercanas, 0 si no lo son.
    - Si no son cercanas, también se pueden retornar el primer índice donde difieren y los valores correspondientes.
*/
int matrices_close(const scalar_t *A_ref,
                   const scalar_t *A_test,
                   size_t num_elements,
                   scalar_t abs_tol,
                   scalar_t rel_tol,
                   size_t *first_bad_index,
                   scalar_t *bad_ref,
                   scalar_t *bad_test)
{
    /*
    Por cada elemento de A_ref y A_test:
    - Verificar si la diferencia absoluta excede una tolerancia.
    - Al primer fallo, resgistar el error y retornar 0.
        |A_ref[i] - A_test[i]| <= max(abs_tol, rel_tol * |A_ref[i]|)
    - De lo contario, (todos correctos) retornar 1.
    */
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
