#ifndef KERNEL_AVX2_TILED_H
#define KERNEL_AVX2_TILED_H

#include <stddef.h>

#include <immintrin.h>

#include "matrix_utils.h"  /* scalar_t */

// Hiperparámetros microarquitectura AVX2.
#define KERNEL_AVX2_TILED_MR 6u
#define KERNEL_AVX2_TILED_NR 16u


/*
kernel_avx2_tiled_6x16 - 6x16 microkernel para tiled_ikj.
INPUTS:
- C: Puntero a la matriz de salida (m x n).
- A: Puntero a la matriz A (m x k).
- B: Puntero a la matriz B (k x n).
- ldc, lda, ldb: Leading dimensions de C, A, B en unidades de scalar_t (row-major).
- kc: Tamaño del bloque interno (dimensión común de A y B en esta fase).
OUTPUTS:
- C modificado.

NOTA: 
- inline -> garantiza que C vive en registros durante kc pasos
- static -> evita conflicto de símbolos al linkear
*/
static inline void
kernel_avx2_tiled_6x16(scalar_t       *restrict C, size_t ldc,
                       const scalar_t *restrict A, size_t lda,
                       const scalar_t *restrict B, size_t ldb,
                       size_t kc)
{   
    // Tenemos 6x16, por fila 16 floats
    // 1 YMM tiene 256 bits = 8 floats, => 2 YMM para C.
    // => 12 YMM con 8 floats para la acumulación.
    __m256 c00 = _mm256_loadu_ps(&C[0 * ldc + 0]); // fila 0, columnas 0-7
    __m256 c01 = _mm256_loadu_ps(&C[0 * ldc + 8]); // fila 0, columnas 8-15
    __m256 c10 = _mm256_loadu_ps(&C[1 * ldc + 0]); // fila 1, columnas 0-7
    __m256 c11 = _mm256_loadu_ps(&C[1 * ldc + 8]); // fila 1, columnas 8-15
    __m256 c20 = _mm256_loadu_ps(&C[2 * ldc + 0]); // fila 2, columnas 0-7
    __m256 c21 = _mm256_loadu_ps(&C[2 * ldc + 8]); // fila 2, columnas 8-15
    __m256 c30 = _mm256_loadu_ps(&C[3 * ldc + 0]); // fila 3, columnas 0-7
    __m256 c31 = _mm256_loadu_ps(&C[3 * ldc + 8]); // fila 3, columnas 8-15
    __m256 c40 = _mm256_loadu_ps(&C[4 * ldc + 0]); // fila 4, columnas 0-7
    __m256 c41 = _mm256_loadu_ps(&C[4 * ldc + 8]); // fila 4, columnas 8-15
    __m256 c50 = _mm256_loadu_ps(&C[5 * ldc + 0]); // fila 5, columnas 0-7
    __m256 c51 = _mm256_loadu_ps(&C[5 * ldc + 8]); // fila 5, columnas 8-15

    // Para p en [0, kc)
    for (size_t p = 0; p < kc; ++p) {
        // Cargar fila p de B (16 floats) en 2 YMM.
        __m256 b0 = _mm256_loadu_ps(&B[p * ldb + 0]);
        __m256 b1 = _mm256_loadu_ps(&B[p * ldb + 8]);
        __m256 a;

        // Para cada fila de A: broadcast del scalar A[r, p] y 2 FMAs
        // 8 flops por FMA
        a = _mm256_broadcast_ss(&A[0 * lda + p]);   // replica A[0,p] en para 8 columnas
        c00 = _mm256_fmadd_ps(a, b0, c00);          // c00 += A[0,p] * B[p, 0..7]   (1 instrucción)
        c01 = _mm256_fmadd_ps(a, b1, c01);          // c01 += A[0,p] * B[p, 8..15]  (1 instrucción)

        a = _mm256_broadcast_ss(&A[1 * lda + p]);
        c10 = _mm256_fmadd_ps(a, b0, c10);
        c11 = _mm256_fmadd_ps(a, b1, c11);

        a = _mm256_broadcast_ss(&A[2 * lda + p]);
        c20 = _mm256_fmadd_ps(a, b0, c20);
        c21 = _mm256_fmadd_ps(a, b1, c21);

        a = _mm256_broadcast_ss(&A[3 * lda + p]);
        c30 = _mm256_fmadd_ps(a, b0, c30);
        c31 = _mm256_fmadd_ps(a, b1, c31);

        a = _mm256_broadcast_ss(&A[4 * lda + p]);
        c40 = _mm256_fmadd_ps(a, b0, c40);
        c41 = _mm256_fmadd_ps(a, b1, c41);

        a = _mm256_broadcast_ss(&A[5 * lda + p]);
        c50 = _mm256_fmadd_ps(a, b0, c50);
        c51 = _mm256_fmadd_ps(a, b1, c51);

        // 8 flops por FMA * 12 => 96 flops por cada p
    }

    // Reescribe C en memoria con los resultados acumulados.
    _mm256_storeu_ps(&C[0 * ldc + 0], c00);
    _mm256_storeu_ps(&C[0 * ldc + 8], c01);
    _mm256_storeu_ps(&C[1 * ldc + 0], c10);
    _mm256_storeu_ps(&C[1 * ldc + 8], c11);
    _mm256_storeu_ps(&C[2 * ldc + 0], c20);
    _mm256_storeu_ps(&C[2 * ldc + 8], c21);
    _mm256_storeu_ps(&C[3 * ldc + 0], c30);
    _mm256_storeu_ps(&C[3 * ldc + 8], c31);
    _mm256_storeu_ps(&C[4 * ldc + 0], c40);
    _mm256_storeu_ps(&C[4 * ldc + 8], c41);
    _mm256_storeu_ps(&C[5 * ldc + 0], c50);
    _mm256_storeu_ps(&C[5 * ldc + 8], c51);
}


/*
kernel_avx2_tiled_residual_rows - AVX2 fallback para las colas m % MR y n % NR.
- INPUTS:
    - C: puntero a la matriz C.
    - ldc: distancia entre columnas de C.
    - A: puntero a la matriz A.
    - lda: distancia entre columnas de A.
    - B: puntero a la matriz B.
    - ldb: distancia entre columnas de B.
    - mr_eff: número efectivo de filas residuales.
    - n: número de columnas de B.
    - kc: número de iteraciones de p.
- OUTPUTS:
    - C modificado.

NOTA:
- Más lentro que el microkernel pero solo se activa para hasta MR-1 = 5 filas residuales, 
- lo cual es despreciable para m >> MR.
*/
static inline void
kernel_avx2_tiled_residual_rows(scalar_t       *restrict C, size_t ldc,
                                const scalar_t *restrict A, size_t lda,
                                const scalar_t *restrict B, size_t ldb,
                                size_t mr_eff, size_t n, size_t kc)
{
    // Para cada fila residual i
    for (size_t i = 0; i < mr_eff; ++i) {
        // Para cada iteración de p
        for (size_t p = 0; p < kc; ++p) {
            const scalar_t a_val = A[i * lda + p];          // scalar de A
            const __m256   a_vec = _mm256_set1_ps(a_val);   // broadcast a 8 lanes

            // Para cada columna j
            size_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 b = _mm256_loadu_ps(&B[p * ldb + j]); // cargar 8 floats de B
                __m256 c = _mm256_loadu_ps(&C[i * ldc + j]); // cargar 8 floats de C
                c = _mm256_fmadd_ps(a_vec, b, c);            // c += a * b
                _mm256_storeu_ps(&C[i * ldc + j], c);        // guardar C de vuelta
            }
            // Procesar columnas residuales (si n no es múltiplo de 8)
            for (; j < n; ++j)
                C[i * ldc + j] += a_val * B[p * ldb + j];
        }
    }
}

#endif /* KERNEL_AVX2_TILED_H */
