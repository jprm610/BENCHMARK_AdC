// kernel_avx2_morton.h - Microkernel AVX2 + FMA para un tile fijo
// 4x16 de C, usado por la familia Morton-blocked
// (matmul_morton_avx2 y matmul_morton_omp).
//
// Header-only `static inline`: el cuerpo se pega en el call site, asi
// los 8 acumuladores YMM viven en registros durante todo el loop kc y
// nunca se spillean al stack.

#ifndef KERNEL_AVX2_MORTON_H
#define KERNEL_AVX2_MORTON_H

#include <stddef.h>

#include <immintrin.h>

#include "matrix_utils.h"  /* scalar_t */

/*
Geometria del microkernel:
- KERNEL_AVX2_MR = 4 filas por invocacion.
- KERNEL_AVX2_NR = 16 columnas por invocacion.

Balance de registros (Zen 2 tiene 16 YMM nombrados):
    c00..c31  -> 8 YMMs  (acumuladores de C, viven todo el loop kc)
    b0, b1    -> 2 YMMs  (fila p de B, se renuevan en cada iteracion)
    a0..a3    -> 4 YMMs  (broadcasts de A[r,p] para r en 0..3)
    -----------------
                14 de 16 registros YMM usados.

Computa C[4 x 16] += A[4 x kc] * B[kc x 16] con kc parametro runtime.
*/
#define KERNEL_AVX2_MR 4
#define KERNEL_AVX2_NR 16

/*
Responsabilidades del caller (el kernel no verifica nada):
- kc >= 1.
- lda, ldb, ldc son las leading dimensions en las allocations
  ORIGINALES (numero de columnas por fila, row-major).
- ldb >= 16 y ldc >= 16. A se accede solo en A[r*lda + p] para r en
  [0,4) y p en [0,kc), asi que lda >= kc.
- C, A, B no aliasing (restrict).
- Alineacion a 32 bytes preferible pero NO requerida: la implementacion
  usa _mm256_loadu_ps / _mm256_storeu_ps. Las hojas Morton no aterrizan
  siempre en multiplos de 32 bytes porque los offsets navegan a
  posiciones de elemento, no de linea de cache. La penalizacion de
  loadu sobre datos accidentalmente alineados es 0 ciclos en Zen 2.

Semantica: el kernel ACUMULA en C. Callers que quieran C limpio deben
zerarlo antes.

Requiere: -mavx2 -mfma en el call site (CFLAGS_O3_ZEN2 o CFLAGS_OMP_ZEN2).
*/

/*
kernel_avx2_4x16: Microkernel 4x16 AVX2 + FMA para FP32 matmul.
    INPUTS:
    - C: Puntero al tile de C[4 x 16] a actualizar (in/out).
    - ldc: Leading dimension de C en su allocation original.
    - A: Puntero al panel A[4 x kc] con stride lda entre filas.
    - lda: Leading dimension de A.
    - B: Puntero al panel B[kc x 16] con stride ldb entre filas.
    - ldb: Leading dimension de B.
    - kc: Profundidad del producto.
    OUTPUTS:
    - Ninguno (void). C queda actualizado con C += A * B.

Layout de los 8 acumuladores en YMM:

    row 0:  c00  c01      (cols  0..7   cols  8..15)
    row 1:  c10  c11
    row 2:  c20  c21
    row 3:  c30  c31

Cada iteracion del loop kc:
- 2 loads de 256 bits para B (b0, b1) = 16 floats de la fila p de B.
- 4 broadcasts (a0..a3) de A[r, p] para r en 0..3.
- 8 FMAs (8 acumuladores * 1 actualizacion cada uno).

Zen 2 retira hasta 2 FMA ops/ciclo => 8 FMAs son 4 ciclos de compute
por iteracion. Los loads + broadcasts proveen operandos en paralelo
desde la otra ventana de issue. En estado estable el bottleneck es el
FMA throughput, no los loads.
*/
static inline void
kernel_avx2_4x16(scalar_t       *restrict C, size_t ldc,
                 const scalar_t *restrict A, size_t lda,
                 const scalar_t *restrict B, size_t ldb,
                 size_t kc)
{
    /* ===== Fase 1: cargar el tile de C en los 8 acumuladores ===== */
    /* loadu (no load) porque las hojas Morton no garantizan
     * alineacion a 32 bytes; ver comentario arriba del header. */
    __m256 c00 = _mm256_loadu_ps(&C[0 * ldc + 0]);
    __m256 c01 = _mm256_loadu_ps(&C[0 * ldc + 8]);
    __m256 c10 = _mm256_loadu_ps(&C[1 * ldc + 0]);
    __m256 c11 = _mm256_loadu_ps(&C[1 * ldc + 8]);
    __m256 c20 = _mm256_loadu_ps(&C[2 * ldc + 0]);
    __m256 c21 = _mm256_loadu_ps(&C[2 * ldc + 8]);
    __m256 c30 = _mm256_loadu_ps(&C[3 * ldc + 0]);
    __m256 c31 = _mm256_loadu_ps(&C[3 * ldc + 8]);

    /* ===== Fase 2: loop kc (el trabajo real) ===== */
    for (size_t p = 0; p < kc; ++p) {
        /* Una fila de B a profundidad p, partida en las mitades
         * baja y alta del tile de 16 columnas. */
        __m256 b0 = _mm256_loadu_ps(&B[p * ldb + 0]);
        __m256 b1 = _mm256_loadu_ps(&B[p * ldb + 8]);

        /* Broadcast: replica A[r, p] en las 8 lanes del YMM, asi el
         * FMA siguiente aplica ese escalar a las 8 columnas del tile
         * en una sola instruccion. */
        __m256 a0 = _mm256_broadcast_ss(&A[0 * lda + p]);
        __m256 a1 = _mm256_broadcast_ss(&A[1 * lda + p]);
        __m256 a2 = _mm256_broadcast_ss(&A[2 * lda + p]);
        __m256 a3 = _mm256_broadcast_ss(&A[3 * lda + p]);

        /* 8 FMAs independientes (sin dependencia RAW entre si dentro
         * de esta iteracion) => las 2 unidades FMA del Zen 2 las
         * pipelinean sin stalls. */
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }

    /* ===== Fase 3: guardar los acumuladores de vuelta a memoria ===== */
    _mm256_storeu_ps(&C[0 * ldc + 0], c00);
    _mm256_storeu_ps(&C[0 * ldc + 8], c01);
    _mm256_storeu_ps(&C[1 * ldc + 0], c10);
    _mm256_storeu_ps(&C[1 * ldc + 8], c11);
    _mm256_storeu_ps(&C[2 * ldc + 0], c20);
    _mm256_storeu_ps(&C[2 * ldc + 8], c21);
    _mm256_storeu_ps(&C[3 * ldc + 0], c30);
    _mm256_storeu_ps(&C[3 * ldc + 8], c31);
}

#endif /* KERNEL_AVX2_MORTON_H */
