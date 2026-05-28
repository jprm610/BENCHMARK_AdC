#ifndef KERNEL_AVX512_TILED_H
#define KERNEL_AVX512_TILED_H

#include <stddef.h>

#include <immintrin.h>

#include "matrix_utils.h"  /* scalar_t */

// Hiperparámetros microarquitectura AVX-512.
#define KERNEL_AVX512_TILED_MR 6u
#define KERNEL_AVX512_TILED_NR 32u


/*
kernel_avx512_tiled_6x32 - 6x32 microkernel para tiled_ikj.
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
kernel_avx512_tiled_6x32(scalar_t       *restrict C, size_t ldc,
                         const scalar_t *restrict A, size_t lda,
                         const scalar_t *restrict B, size_t ldb,
                         size_t kc)
{
    // Tenemos 6x32, por fila 32 floats.
    // 1 ZMM tiene 512 bits = 16 floats, => 2 ZMM para C.
    // => 12 ZMM con 16 floats para la acumulación.
    __m512 c00 = _mm512_loadu_ps(&C[0 * ldc +  0]); // fila 0, columnas 0-15
    __m512 c01 = _mm512_loadu_ps(&C[0 * ldc + 16]); // fila 0, columnas 16-31
    __m512 c10 = _mm512_loadu_ps(&C[1 * ldc +  0]); // fila 1, columnas 0-15
    __m512 c11 = _mm512_loadu_ps(&C[1 * ldc + 16]); // fila 1, columnas 16-31
    __m512 c20 = _mm512_loadu_ps(&C[2 * ldc +  0]); // fila 2, columnas 0-15
    __m512 c21 = _mm512_loadu_ps(&C[2 * ldc + 16]); // fila 2, columnas 16-31
    __m512 c30 = _mm512_loadu_ps(&C[3 * ldc +  0]); // fila 3, columnas 0-15
    __m512 c31 = _mm512_loadu_ps(&C[3 * ldc + 16]); // fila 3, columnas 16-31
    __m512 c40 = _mm512_loadu_ps(&C[4 * ldc +  0]); // fila 4, columnas 0-15
    __m512 c41 = _mm512_loadu_ps(&C[4 * ldc + 16]); // fila 4, columnas 16-31
    __m512 c50 = _mm512_loadu_ps(&C[5 * ldc +  0]); // fila 5, columnas 0-15
    __m512 c51 = _mm512_loadu_ps(&C[5 * ldc + 16]); // fila 5, columnas 16-31

    // Para p en [0, kc)
    for (size_t p = 0; p < kc; ++p) {
        // Cargar fila p de B (32 floats) en 2 ZMM.
        __m512 b0 = _mm512_loadu_ps(&B[p * ldb +  0]);
        __m512 b1 = _mm512_loadu_ps(&B[p * ldb + 16]);
        __m512 a;

        // Para cada fila de A: broadcast del scalar A[r, p] y 2 FMAs.
        // 16 flops por FMA.
        a = _mm512_set1_ps(A[0 * lda + p]);    // replica A[0,p] en 16 lanes
        c00 = _mm512_fmadd_ps(a, b0, c00);     // c00 += A[0,p] * B[p, 0..15]   (1 instrucción)
        c01 = _mm512_fmadd_ps(a, b1, c01);     // c01 += A[0,p] * B[p, 16..31]  (1 instrucción)

        a = _mm512_set1_ps(A[1 * lda + p]);
        c10 = _mm512_fmadd_ps(a, b0, c10);
        c11 = _mm512_fmadd_ps(a, b1, c11);

        a = _mm512_set1_ps(A[2 * lda + p]);
        c20 = _mm512_fmadd_ps(a, b0, c20);
        c21 = _mm512_fmadd_ps(a, b1, c21);

        a = _mm512_set1_ps(A[3 * lda + p]);
        c30 = _mm512_fmadd_ps(a, b0, c30);
        c31 = _mm512_fmadd_ps(a, b1, c31);

        a = _mm512_set1_ps(A[4 * lda + p]);
        c40 = _mm512_fmadd_ps(a, b0, c40);
        c41 = _mm512_fmadd_ps(a, b1, c41);

        a = _mm512_set1_ps(A[5 * lda + p]);
        c50 = _mm512_fmadd_ps(a, b0, c50);
        c51 = _mm512_fmadd_ps(a, b1, c51);

        // 16 flops por FMA * 12 => 192 flops por cada p
    }

    // Reescribe C en memoria con los resultados acumulados.
    _mm512_storeu_ps(&C[0 * ldc +  0], c00);
    _mm512_storeu_ps(&C[0 * ldc + 16], c01);
    _mm512_storeu_ps(&C[1 * ldc +  0], c10);
    _mm512_storeu_ps(&C[1 * ldc + 16], c11);
    _mm512_storeu_ps(&C[2 * ldc +  0], c20);
    _mm512_storeu_ps(&C[2 * ldc + 16], c21);
    _mm512_storeu_ps(&C[3 * ldc +  0], c30);
    _mm512_storeu_ps(&C[3 * ldc + 16], c31);
    _mm512_storeu_ps(&C[4 * ldc +  0], c40);
    _mm512_storeu_ps(&C[4 * ldc + 16], c41);
    _mm512_storeu_ps(&C[5 * ldc +  0], c50);
    _mm512_storeu_ps(&C[5 * ldc + 16], c51);
}


/*
kernel_avx512_tiled_residual_rows - AVX-512 fallback para las colas m % MR y n % NR.
- INPUTS:
    - C: puntero a la matriz C.
    - ldc: distancia entre filas de C.
    - A: puntero a la matriz A.
    - lda: distancia entre filas de A.
    - B: puntero a la matriz B.
    - ldb: distancia entre filas de B.
    - mr_eff: número efectivo de filas residuales.
    - n: número de columnas de B.
    - kc: número de iteraciones de p.
- OUTPUTS:
    - C modificado.

NOTA:
- Más lento que el microkernel pero solo se activa para hasta MR-1 = 5 filas residuales,
- lo cual es despreciable para m >> MR.
*/
static inline void
kernel_avx512_tiled_residual_rows(scalar_t       *restrict C, size_t ldc,
                                  const scalar_t *restrict A, size_t lda,
                                  const scalar_t *restrict B, size_t ldb,
                                  size_t mr_eff, size_t n, size_t kc)
{
    // Para cada fila residual i
    for (size_t i = 0; i < mr_eff; ++i) {
        // Para cada iteración de p
        for (size_t p = 0; p < kc; ++p) {
            const scalar_t a_val = A[i * lda + p];          // scalar de A
            const __m512   a_vec = _mm512_set1_ps(a_val);   // broadcast a 16 lanes

            // Para cada columna j
            size_t j = 0;
            for (; j + 16 <= n; j += 16) {
                __m512 b = _mm512_loadu_ps(&B[p * ldb + j]); // cargar 16 floats de B
                __m512 c = _mm512_loadu_ps(&C[i * ldc + j]); // cargar 16 floats de C
                c = _mm512_fmadd_ps(a_vec, b, c);            // c += a * b
                _mm512_storeu_ps(&C[i * ldc + j], c);        // guardar C de vuelta
            }
            // Procesar columnas residuales (si n no es múltiplo de 16)
            for (; j < n; ++j)
                C[i * ldc + j] += a_val * B[p * ldb + j];
        }
    }
}

#endif /* KERNEL_AVX512_TILED_H */
