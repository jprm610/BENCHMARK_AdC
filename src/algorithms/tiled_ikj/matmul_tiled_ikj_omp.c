#include "matmul_tiled_ikj_omp.h"
#include "matrix_utils.h"
#include "kernel_avx2_tiled.h"   /* kernel_avx2_tiled_6x16, residual_rows */

#include <omp.h>
#include <stdio.h>
#include <string.h>

size_t g_tiled_ikj_omp_bs = TILED_IKJ_OMP_BS_DEFAULT;

void matmul_tiled_ikj_omp_set_bs(size_t bs)
{
    if (bs == 0) {
        fprintf(stderr,
                "Warning: matmul_tiled_ikj_omp_set_bs(0) ignored; "
                "bs must be positive.\n");
        return;
    }
    g_tiled_ikj_omp_bs = bs;
}


/*
matmul_tiled_ikj_omp: C = A * B con A(m x k), B(k x n), C(m x n) con paralelización de AVX2.
INPUTS:
- C: puntero a la matriz resultado C (m x n)
- A: puntero a la matriz A (m x k)
- B: puntero a la matriz B (k x n)
- m: número de filas de A y C
- k_dim: número de columnas de A y filas de B
- n: número de columnas de B y C
OUTPUTS:
- C: matriz resultado de la multiplicación A * B (Puntero)

NOTA: No se usan los 12 cores lógicos de RYZEN 5 4600H,
Se usan los 6 threads físicos ya que cada par comparte FMA.
Misma estructura de matmul_tiled_ikj_avx2.c
*/
void matmul_tiled_ikj_omp(scalar_t *C,
                      const scalar_t *A,
                      const scalar_t *B,
                      size_t m, size_t k_dim, size_t n)
{
    // Parámetros para AVX2
    const size_t MR = TILED_IKJ_OMP_MR;
    const size_t NR = TILED_IKJ_OMP_NR;
    const size_t MC = TILED_IKJ_OMP_MC;
    const size_t KC = g_tiled_ikj_omp_bs;

    init_matrix_zero(C, m, n);

    const size_t m_aligned = (m / MR) * MR;
    const size_t n_aligned = (n / NR) * NR;

    // Se crean los threads [T0][T1][T2][T3][T4][T5]
    #pragma omp parallel
    {
        // No se paralieliza kk,
        // para evitar conflictos en la escritura de C entre threads.
        for (size_t kk = 0; kk < k_dim; kk += KC) {
            const size_t kc = (kk + KC < k_dim) ? KC : k_dim - kk;

            // Omp distribuye los threads (6 en este caso) para cada ic.
            #pragma omp for schedule(static)
            for (size_t ic = 0; ic < m_aligned; ic += MC) {
                const size_t ic_end = (ic + MC <= m_aligned) ? ic + MC
                                                             : m_aligned;

                for (size_t jr = 0; jr < n_aligned; jr += NR) {
                    for (size_t ir = ic; ir + MR <= ic_end; ir += MR) {
                        kernel_avx2_tiled_6x16(&C[ir * n + jr], n,
                                               &A[ir * k_dim + kk], k_dim,
                                               &B[kk * n + jr], n,
                                               kc);
                    }
                }

                if (n_aligned < n) {
                    kernel_avx2_tiled_residual_rows(
                        &C[ic * n + n_aligned], n,
                        &A[ic * k_dim + kk], k_dim,
                        &B[kk * n + n_aligned], n,
                        ic_end - ic, n - n_aligned, kc);
                }
            }
            if (m_aligned < m) {
                // Un solo thread se encarga de procesar las filas residuales, 
                // para evitar conflictos en la escritura de C.
                #pragma omp single
                {
                    kernel_avx2_tiled_residual_rows(
                        &C[m_aligned * n], n,
                        &A[m_aligned * k_dim + kk], k_dim,
                        &B[kk * n], n,
                        m - m_aligned, n, kc);
                }
            }
        }
    } /* end omp parallel */
}


/*
benchmark_iterations_tiled_ikj_omp: Implementa recurrencia:
        B_{i+1} = A * B_{i} con B_0 = Z.
INPUTS:
- B_out: Puntero a la matriz de salida (num_iters x n x n).
- A: Puntero a la matriz A (m x m).
- Z: Puntero a la matriz Z (m x n).
- m: Número de filas de A y B_i.
- n: Número de columnas de B_i.
- num_iters: Número de iteraciones.
*/
void benchmark_iterations_tiled_ikj_omp(scalar_t *B_out,
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
        // Calcular B_{i+1} = A * B_{i} usando matmul_tiled_ikj_omp.
        matmul_tiled_ikj_omp(B_next, A, B_curr, m, m, n);

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
