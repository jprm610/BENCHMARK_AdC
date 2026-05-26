/*
 * matmul_tiled_ikj.c - Implementación de Cache-Blocking (Tiling).
 */

#include "matmul_tiled_ikj.h"
#include "matrix_utils.h"   /* xalloc_aligned, xfree, init_matrix_zero */

#include <string.h>         /* memcpy */

/*
matmul_tiled_ikj: Computar C = A * B, con bloqueos explícitos de Mc x Kc.
INPUT:
- C: matriz m x n
- A: matriz m x k
- B: matriz k x n
- m, k, n: dimensiones de las matrices
OUTPUT:
- C: resultado de A * B (sobrescribe el contenido previo)

NOTA: TILED_IKJ_MC_DEFAULT y TILED_IKJ_KC_DEFAULT son hiperparámetros (matmul_tiled_ikj.h).
- Se debería ver una diferencia notable al crecer más que L2.
*/
void matmul_tiled_ikj(scalar_t *C,
                  const scalar_t *A,
                  const scalar_t *B,
                  size_t m, size_t k_dim, size_t n)
{
    // Tamaños de bloque (tile sizes) para i y k, respectivamente.
    // Según hardware con objetivo 1 MiB de L2 cache por núcleo (AMD EPYC 9R45).
    const size_t Mc = TILED_IKJ_MC_DEFAULT;
    const size_t Kc = TILED_IKJ_KC_DEFAULT;

    // Inicializar C en 0, ya que se acumulará el resultado de cada bloque.
    init_matrix_zero(C, m, n);

    // Se usa el orden de loops ikj, con bloqueos sobre i y k:

    // ii divide filas de A y C en bloques de tamaño Mc.
    for (size_t ii = 0; ii < m; ii += Mc) {
        // Límite cuando el bloque excede el tamaño de la matriz.
        size_t i_end = ii + Mc < m ? ii + Mc : m;

        // kk divide columnas de A y filas de B en bloques de tamaño Kc.
        for (size_t kk = 0; kk < k_dim; kk += Kc) {
            // Límite cuando el bloque excede el tamaño de la matriz.
            size_t k_end = kk + Kc < k_dim ? kk + Kc : k_dim;

            // Se computa bloque 
            //      C[ii:i_end, :] += A[ii:i_end, kk:k_end] * B[kk:k_end, :]
            // usando el orden de loops ikj dentro del bloque.
            for (size_t i = ii; i < i_end; ++i) {
                for (size_t k = kk; k < k_end; ++k) {
                    scalar_t a_ik = A[i * k_dim + k];
                    for (size_t j = 0; j < n; ++j)
                        C[i * n + j] += a_ik * B[k * n + j];
                }
            }
        }
    }
}


/*
benchmark_iterations_tiled_ikj: Implementa recurrencia:
        B_{i+1} = A * B_{i} con B_0 = Z.
INPUTS:
- B_out: Puntero a la matriz de salida (num_iters x n x n).
- A: Puntero a la matriz A (m x m).
- Z: Puntero a la matriz Z (m x n).
- m: Número de filas de A y B_i.
- n: Número de columnas de B_i.
- num_iters: Número de iteraciones.
*/
void benchmark_iterations_tiled_ikj(scalar_t *B_out,
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
        // Calcular B_{i+1} = A * B_{i} usando matmul_tiled_ikj.
        matmul_tiled_ikj(B_next, A, B_curr, m, m, n);

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
