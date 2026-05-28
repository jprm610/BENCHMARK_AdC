/*
 * matmul_tiled_ikj_avx512.c - Estilo BLIS con bucles ikj y microkernel AVX-512 6x32.
 */

#include "matmul_tiled_ikj_avx512.h"
#include "matrix_utils.h"
#include "kernel_avx512_tiled.h"   /* kernel_avx512_tiled_6x32, residual */

#include <stdio.h>
#include <string.h>

size_t g_tiled_ikj_avx512_bs = TILED_IKJ_AVX512_BS_DEFAULT;

void matmul_tiled_ikj_avx512_set_bs(size_t bs)
{
    if (bs == 0) {
        fprintf(stderr,
                "Warning: matmul_tiled_ikj_avx512_set_bs(0) ignored; "
                "bs must be positive.\n");
        return;
    }
    g_tiled_ikj_avx512_bs = bs;
}

/*
matmul_tiled_ikj_avx512: Estilo BLIS tiled_ikj matmul (6x32)
INPUTS:
- C: Puntero a la matriz de salida. (m x n)
- A: Puntero a la matriz A. (m x k)
- B: Puntero a la matriz B. (k x n)
- m: Número de filas de A y C.
- k_dim: Número de columnas de A y filas de B.
- n: Número de columnas de B y C.
*/
void matmul_tiled_ikj_avx512(scalar_t *C,
                       const scalar_t *A,
                       const scalar_t *B,
                       size_t m, size_t k_dim, size_t n)
{
    // Hiperparámetros según hardware. (Definidos en matmul_tiled_ikj_avx512.h)
    // Microkernel
    const size_t MR = TILED_IKJ_AVX512_MR;    // Filas microkernel (6)
    const size_t NR = TILED_IKJ_AVX512_NR;    // Columnas microkernel (32)
    // Tiling
    const size_t MC = TILED_IKJ_AVX512_MC;    // Filas bloque L2 (288)
    const size_t KC = g_tiled_ikj_avx512_bs;  // Columnas bloque L1d (256) (ajustable)

    // C en 0s, ya que se acumulará el resultado de cada bloque.
    init_matrix_zero(C, m, n);

    // Necesario para manejar casos donde m o n no son múltiplos de MR o NR.
    const size_t m_aligned = (m / MR) * MR;
    const size_t n_aligned = (n / NR) * NR;

    // kk externo (tamaño KC)
    // Columnas de A y filas de B.
    for (size_t kk = 0; kk < k_dim; kk += KC) {
        const size_t kc = (kk + KC < k_dim) ? KC : k_dim - kk;

        // ic interno (tamaño MC)
        for (size_t ic = 0; ic < m_aligned; ic += MC) {
            const size_t ic_end = (ic + MC <= m_aligned) ? ic + MC
                                                         : m_aligned;

            // jr sobre columnas de B (tamaño NR)
            for (size_t jr = 0; jr < n_aligned; jr += NR) {
                // ir sobre filas de A (tamaño MR)
                // A (MR (6) x KC (256))  <- L2
                // B (KC (256) x NR (32)) <- L1d (Se reusa para todo jr)
                // C (MR (6) x NR (32))   <- registros ZMM
                for (size_t ir = ic; ir + MR <= ic_end; ir += MR) {
                    // C[6×32] += A[6×KC] × B[KC×32]
                    kernel_avx512_tiled_6x32(&C[ir * n + jr], n,
                                             &A[ir * k_dim + kk], k_dim,
                                             &B[kk * n + jr], n,
                                             kc);
                }
            }

            if (n_aligned < n) {
                kernel_avx512_tiled_residual_rows(
                    &C[ic * n + n_aligned], n,
                    &A[ic * k_dim + kk], k_dim,
                    &B[kk * n + n_aligned], n,
                    ic_end - ic, n - n_aligned, kc);
            }
        }

        if (m_aligned < m) {
            kernel_avx512_tiled_residual_rows(
                &C[m_aligned * n], n,
                &A[m_aligned * k_dim + kk], k_dim,
                &B[kk * n], n,
                m - m_aligned, n, kc);
        }
    }
}


/*
benchmark_iterations_tiled_ikj_avx512: Implementa recurrencia:
        B_{i+1} = A * B_{i} con B_0 = Z.
INPUTS:
- B_out: Puntero a la matriz de salida (num_iters x n x n).
- A: Puntero a la matriz A (m x m).
- Z: Puntero a la matriz Z (m x n).
- m: Número de filas de A y B_i.
- n: Número de columnas de B_i.
- num_iters: Número de iteraciones.
*/
void benchmark_iterations_tiled_ikj_avx512(scalar_t *B_out,
                                     const scalar_t *A,
                                     const scalar_t *Z,
                                     size_t m, size_t n,
                                     size_t num_iters)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        matmul_tiled_ikj_avx512(B_next, A, B_curr, m, m, n);

        scalar_t *out_block = B_out + iter * n * n;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                out_block[i * n + j] = B_next[i * n + j];

        scalar_t *tmp = B_curr;
        B_curr        = B_next;
        B_next        = tmp;
    }

    xfree(B_curr);
    xfree(B_next);
}
