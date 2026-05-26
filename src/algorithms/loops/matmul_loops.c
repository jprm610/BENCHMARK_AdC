/*
 * matmul_loops.c - 6 variantes de bucle (ijk, ikj, jik, jki, kij, kji)
 * Fase 1.1) Evaluar cuál loop explota mejor la localidad espacial.
 */

#include "matmul_loops.h"
#include "matrix_utils.h"   /* xalloc_aligned, xfree, init_matrix_zero */

#include <string.h>         /* memcpy, strcmp */

// Se itera según el orden del bucle (inner, middle, outer)

/*
matmul_ijk: (i, j, k)
- Realiza la multiplicación igual que naive (matmul_naive.c).
INPUTS:
- C: Puntero a la matriz de salida (m x n).
- A: Puntero a la matriz A (m x k).
- B: Puntero a la matriz B (k x n).
- m: Número de filas de A y C.
- k_dim: Número de columnas de A y filas de B.
- n: Número de columnas de B y C.
OUTPUTS:
- C: Matriz de salida con el resultado de A x B. (Es void pero se entrega en puntero)

- Problema: stride de n al k (inner) iterar por columnas de B.

*/
void matmul_ijk(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k_dim, size_t n)
{
    // Outer i
    for (size_t i = 0; i < m; ++i) {
        // Middle j
        for (size_t j = 0; j < n; ++j) {
            // C[i,j] = sum_k A[i,k] * B[k,j]
            scalar_t acc = (scalar_t)0;
            
            // Inner k
            for (size_t k = 0; k < k_dim; ++k)
                // Stride de n al iterar por columnas de B. 
                // Mala localidad espacial.
                acc += A[i * k_dim + k] * B[k * n + j];
            C[i * n + j] = acc;
        }
    }
}


/*
matmul_jik: (j, i, k)
- Reordena i y j.
INPUTS:
- C: Puntero a la matriz de salida (m x n).
- A: Puntero a la matriz A (m x k).
- B: Puntero a la matriz B (k x n).
- m: Número de filas de A y C.
- k_dim: Número de columnas de A y filas de B.
- n: Número de columnas de B y C.
OUTPUTS:
- C: Matriz de salida con el resultado de A x B. (Es void pero se entrega en puntero)

NOTA: Se reutiliza la colmuna de B.
- Comportamiento similar a ijk debido al stride de k (inner).
*/
void matmul_jik(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k_dim, size_t n)
{
    // Outer j
    // Misma columna de B y C.
    for (size_t j = 0; j < n; ++j) {
        // Middle i
        // Se fija C[i,j], (no necesita incializar C en ceros).
        // Se fija fila de A.
        for (size_t i = 0; i < m; ++i) {
            scalar_t acc = (scalar_t)0;
            // Inner k
            // Stride de n al iterar por filas de B.
            // Mala localidad espacial.
            for (size_t k = 0; k < k_dim; ++k)
                acc += A[i * k_dim + k] * B[k * n + j];
            C[i * n + j] = acc;
        }
    }
}


/*
matmul_jki: (j, k, i)
- Reordena i y j respecto a ikj.
INPUTS:
- C: Puntero a la matriz de salida (m x n).
- A: Puntero a la matriz A (m x k).
- B: Puntero a la matriz B (k x n).
- m: Número de filas de A y C.        
- k_dim: Número de columnas de A y filas de B.
- n: Número de columnas de B y C.
OUTPUTS:
- C: Matriz de salida con el resultado de A x B. (Es void pero se entrega en puntero)

NOTA: De las peores variantes aunque se reutilice B[k,j] debido a la mala localidad espacial de A e C.
*/
void matmul_jki(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k_dim, size_t n)
{
    // Crear C como matriz de ceros,
    // Ir acumulando en C[i,j], ya que k no es inner,
    // Diferente a naive que calcula C[i,j] completo.
    init_matrix_zero(C, m, n);

    // Outer j
    // Fija columnas de C y B.
    for (size_t j = 0; j < n; ++j) {
        // Middle k
        // Stride de k al iterar por filas de A. (Mala localidad espacial).
        // kj fijos, se reusa B[k,j]. (Localidad temporal)
        for (size_t k = 0; k < k_dim; ++k) {
            scalar_t b_kj = B[k * n + j];

            // Inner i
            // stride de n al iterar por filas de C (Mala localidad espacial).
            for (size_t i = 0; i < m; ++i)
                C[i * n + j] += A[i * k_dim + k] * b_kj;
        }
    }
}


/*
matmul_kji: (k, j, i)
- Reordena i y j respecto a kij.
INPUTS:
- C: Puntero a la matriz de salida (m x n).
- A: Puntero a la matriz A (m x k).
- B: Puntero a la matriz B (k x n).
- m: Número de filas de A y C.
- k_dim: Número de columnas de A y filas de B.
- n: Número de columnas de B y C.
OUTPUTS:
- C: Matriz de salida con el resultado de A x B. (Es void pero se entrega en puntero)

NOTA: De los peores junto con jki.
*/
void matmul_kji(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k_dim, size_t n)
{
    // Crear C como matriz de ceros,
    // Ir acumulando en C[i,j], ya que k no es inner,
    // Diferente a naive que calcula C[i,j] completo.
    init_matrix_zero(C, m, n);

    // Outer k
    // Fija columna de A y fila de B.
    for (size_t k = 0; k < k_dim; ++k) {
        // Middle j
        // Fija columna de B y C.
        // kj fijos, se reusa B[k,j]. (Localidad temporal)
        // Stride de k_dim (128) al iterar por filas de B.
        for (size_t j = 0; j < n; ++j) {
            scalar_t b_kj = B[k * n + j];

            // Inner i
            // Stride de n al iterar por filas de A (Mala localidad espacial).
            for (size_t i = 0; i < m; ++i)
                C[i * n + j] += A[i * k_dim + k] * b_kj;
        }
    }
}


/*
matmul_kij: (k, i, j)
- Reordena i y j respecto a kji.
INPUTS:
- C: Puntero a la matriz de salida (m x n).
- A: Puntero a la matriz A (m x k).
- B: Puntero a la matriz B (k x n).
- m: Número de filas de A y C.
- k_dim: Número de columnas de A y filas de B.
- n: Número de columnas de B y C.
OUTPUTS:
- C: Matriz de salida con el resultado de A x B. (Es void pero se entrega en puntero)

NOTA: Detrás de ikj, por stride de k en las columnas de A.
*/
void matmul_kij(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k_dim, size_t n)
{
    // Crear C como matriz de ceros,
    // Ir acumulando en C[i,j], ya que k no es inner,
    // Diferente a naive que calcula C[i,j] completo.
    init_matrix_zero(C, m, n);

    // Outer k
    // Fija columna de A.
    for (size_t k = 0; k < k_dim; ++k) {
        // Middle i
        // Fija fila de A y C.
        for (size_t i = 0; i < m; ++i) {
            // ik fijos, se reusa A[i,k]. (Localidad temporal)
            // Stride de m (tamaño A) al iterar por filas de A. (Mala localidad espacial).
            scalar_t a_ik = A[i * k_dim + k];

            // Inner j
            // Acceso secuencial a columnas de B y C, stride de 1. (Localidad espacial)
            for (size_t j = 0; j < n; ++j)
                C[i * n + j] += a_ik * B[k * n + j];
        }
    }
}


/*
matmul_ikj: (i, k, j)
- Reordena k y j.
INPUTS:
- C: Puntero a la matriz de salida (m x n).
- A: Puntero a la matriz A (m x k).
- B: Puntero a la matriz B (k x n).
- m: Número de filas de A y C.
- k_dim: Número de columnas de A y filas de B.
- n: Número de columnas de B y C.
OUTPUTS:
- C: Matriz de salida con el resultado de A x B. (Es void pero se entrega en puntero)

- NOTA: Se explota localidad temporal y espacial.
*/
void matmul_ikj(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k_dim, size_t n)
{
    // Crear C como matriz de ceros,
    // Ir acumulando en C[i,j], ya que k no es inner,
    // Diferente a naive que calcula C[i,j] completo.
    init_matrix_zero(C, m, n);
    
    // Outer i
    // Misma fila de C.
    for (size_t i = 0; i < m; ++i) {
        // Middle k
        // ik fijos, se reusa A[i,k]. (Localidad temporal)
        // Stride de k_dim (128) al iterar por filas de B.
        // 128 << m a medida que crece el bench.
        for (size_t k = 0; k < k_dim; ++k) {
            scalar_t a_ik = A[i * k_dim + k];

            // Inner j
            // Acceso secuencial a columnas de B y C, stride de 1. (Localidad espacial)
            for (size_t j = 0; j < n; ++j)
                C[i * n + j] += a_ik * B[k * n + j];
        }
    }
}


// ---------------------------------------------------------------------

/*
matmul_loops_lookup: Devuelve un puntero a función de multiplicación de matrices según el nombre.
INPUTS:
- name: Nombre de la variante de multiplicación de matrices ("ijk", "ikj", "jik", "jki", "kij", "kji").
OUTPUTS:
- Puntero a la función correspondiente a la variante de multiplicación de matrices.
*/
matmul_fn_t matmul_loops_lookup(const char *name)
{
    if (strcmp(name, "ijk") == 0) return matmul_ijk;
    if (strcmp(name, "ikj") == 0) return matmul_ikj;
    if (strcmp(name, "jik") == 0) return matmul_jik;
    if (strcmp(name, "jki") == 0) return matmul_jki;
    if (strcmp(name, "kij") == 0) return matmul_kij;
    if (strcmp(name, "kji") == 0) return matmul_kji;
    return NULL;
}


/*
benchmark_iterations: Implementa recurrencia:
        B_{i+1} = A * B_{i} con B_0 = Z.
INPUTS:
- B_out: Puntero a la matriz de salida (num_iters x n x n).
- A: Puntero a la matriz A (m x m).
- Z: Puntero a la matriz Z (m x n).
- m: Número de filas de A y B_i.
- n: Número de columnas de B_i.
- num_iters: Número de iteraciones.
- kernel: Puntero a la función de multiplicación de matrices.
OUTPUTS:
- B_out: Matriz de salida con los resultados de cada iteración. (Es void pero se entrega en puntero)
*/
void benchmark_iterations_loops(scalar_t *B_out,
                                 const scalar_t *A,
                                 const scalar_t *Z,
                                 size_t m, size_t n,
                                 size_t num_iters,
                                 matmul_fn_t kernel)
{
    /*
    - Punteros hacia B_curr y B_next.
    - Más adelante se hará un swap de punteros para evitar copiar B_curr a B_next.
    */
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    /* 
    - Inicializar B_curr con Z, que corresponde a B_0 en la recurrencia. (Una copia)
    - Funciona porque Z es contigua.
    */
    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    // For i = 0, 1, ..., num_iters (I = 2m/n)
    for (size_t iter = 0; iter < num_iters; ++iter) {
        // Calcular B_{i+1} = A * B_{i} usando kernel.
        kernel(B_next, A, B_curr, m, m, n);

        /*
        - Guardar las primeras n filas de B_{i+1} en B_out_{i}.
        - Cada B_out_{i} es contiguo. (Ver cómo se define el buffer completo en bench_loops.c)
        */
        scalar_t *out_block = B_out + iter * n * n;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                out_block[i * n + j] = B_next[i * n + j];

        // Swap B_next y B_curr (Sin copiar).
        scalar_t *tmp = B_curr;
        B_curr        = B_next;
        B_next        = tmp;
    }

    // Liberar memoria al final del benchmark.
    xfree(B_curr);
    xfree(B_next);
}
