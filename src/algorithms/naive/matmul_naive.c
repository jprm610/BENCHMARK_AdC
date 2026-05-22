/*
matmul_naive.c - Implemetación para comparación base.
- Multiplicación de matrices A (m x l) y B (l x n) en C (m x n).
- Benchmark de iteraciones B_{i+1} = A * B_{i} con B_0 = Z.
*/

#include "matmul_naive.h"
#include "matrix_utils.h"

#include <string.h>


/*
matmul_naive: Producto de matrices (naive) A (m x l) y B (l x n) en C (m x n).
INPUTS:
- C: Puntero a la matriz de salida (m x n).
- A: Puntero a la matriz A (m x l).
- B: Puntero a la matriz B (l x n).
- m: Número de filas de A y C.
- l: Número de columnas de A y filas de B.
- n: Número de columnas de B y C.

Nota: La función asume que las matrices están almacenadas en formato row-major.
*/
void matmul_naive(scalar_t *C,
                  const scalar_t *A,
                  const scalar_t *B,
                  size_t m, size_t l, size_t n)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            /*
            - Se implementa la multiplicación de matrices de forma directa,
            - producto punto fila de A por columna de B.
                    C[i][j] = \sum_{k=0}^{l-1} A[i][k] * B[k][j]
            */
            scalar_t acc = (scalar_t)0;
            for (size_t k = 0; k < l; ++k) {

                acc += A[i * l + k] * B[k * n + j];
            }
            C[i * n + j] = acc;
        }
    }
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
*/
void benchmark_iterations(scalar_t *B_out,
                          const scalar_t *A,
                          const scalar_t *Z,
                          size_t m, size_t n,
                          size_t num_iters)
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

        // Calcular B_{i+1} = A * B_{i} usando matmul_naive. 
        matmul_naive(B_next, A, B_curr, m, m, n);

        /*
        - Guardar las primeras n filas de B_{i+1} en B_out_{i}.
        - Cada B_out_{i} es contiguo. (Ver cómo se define el buffer completo en bench_naive.c)
        */
        scalar_t *out_block = B_out + iter * n * n;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                out_block[i * n + j] = B_next[i * n + j];
            }
        }

        // Swap B_next y B_curr (Sin copiar).
        scalar_t *tmp = B_curr;
        B_curr = B_next;
        B_next = tmp;
    }

    // Liberar memoria al final del benchmark.
    xfree(B_curr);
    xfree(B_next);
}
