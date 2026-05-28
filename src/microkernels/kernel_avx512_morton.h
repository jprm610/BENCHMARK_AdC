// kernel_avx512_morton.h - Microkernel AVX-512 + FMA para un tile fijo
// 4x32 de C, usado por la familia Morton-blocked
// (matmul_morton_avx512 y matmul_morton_omp) en Zen 5 / EPYC 9R45.
//
// Header-only `static inline`: el cuerpo se pega en el call site, asi
// los 8 acumuladores ZMM viven en registros durante todo el loop kc y
// nunca se spillean al stack.

#ifndef KERNEL_AVX512_MORTON_H
#define KERNEL_AVX512_MORTON_H

#include <stddef.h>

#include <immintrin.h>

#include "matrix_utils.h"  /* scalar_t */

/*
Target: AMD EPYC 9R45 (Zen 5) en AWS c8a.2xlarge.
- 8 cores, 1 thread/core (SMT deshabilitado por el hypervisor).
- L1d 48 KiB / core, L2 1 MiB / core, L3 32 MiB compartido.
- 32 registros ZMM arquitecturales (zmm0..zmm31), dos pipes FMA de
  512 bits => techo teorico de 64 flops/ciclo por core.

Geometria del microkernel:
- KERNEL_AVX512_MORTON_MR = 4 filas por invocacion.
- KERNEL_AVX512_MORTON_NR = 32 columnas por invocacion.

Balance de registros (14 de 32 ZMM en uso):
    c[r][h]   ->  8 ZMMs  (acumuladores de C: r in 0..3, h in {0,1})
    b0, b1    ->  2 ZMMs  (fila p de B: b0 = cols 0..15, b1 = cols 16..31)
    a0..a3    ->  4 ZMMs  (broadcasts de A[r,p] a las 16 lanes)

Cada iteracion del loop kc emite 8 vfmadd231ps independientes. Zen 5
retira 2 FMAs de 512 bits por ciclo => 8 FMAs = 4 ciclos de compute
por paso, sostenidos contra 2 vmovups + 4 broadcasts en la ruta de
loads.

Computa C[4 x 32] += A[4 x kc] * B[kc x 32] con kc parametro runtime.
*/
#define KERNEL_AVX512_MORTON_MR 4u
#define KERNEL_AVX512_MORTON_NR 32u

/*
Responsabilidades del caller (el kernel no verifica nada):
- kc >= 1.
- lda, ldb, ldc son las leading dimensions en unidades scalar_t
  (row-major).
- ldb >= 32, ldc >= 32. A se accede solo en A[r*lda + p] para r en
  [0, MR) y p en [0, kc).
- C, A, B no aliasing (restrict).
- Alineacion a 64 bytes preferible (linea de cache + ZMM) pero NO
  requerida: la implementacion usa _mm512_loadu_ps / _mm512_storeu_ps.

Semantica: el kernel ACUMULA en C. Callers que quieran C limpio deben
zerarlo antes.

Requiere: -march=native (o -mavx512f -mavx512vl) en el call site.
*/

/*
kernel_avx512_4x32: Microkernel 4x32 AVX-512 + FMA para FP32 matmul.
    INPUTS:
    - C: Puntero al tile de C[4 x 32] a actualizar (in/out).
    - ldc: Leading dimension de C en su allocation original.
    - A: Puntero al panel A[4 x kc] con stride lda entre filas.
    - lda: Leading dimension de A.
    - B: Puntero al panel B[kc x 32] con stride ldb entre filas.
    - ldb: Leading dimension de B.
    - kc: Profundidad del producto.
    OUTPUTS:
    - Ninguno (void). C queda actualizado con C += A * B.

Layout de los 8 acumuladores en ZMM:

    row 0:  c00  c01      (cols  0..15   cols 16..31)
    row 1:  c10  c11
    row 2:  c20  c21
    row 3:  c30  c31

Diferencia con el microkernel 6x32 del tiled: aquel reusa un solo
broadcast a traves de las 6 filas (trade-off WAW vs menos registros
vivos, viable con MR mas grande). Este Morton 4x32 mantiene 4
broadcasts independientes (un ZMM por fila), lo que elimina el RAW
entre el broadcast y el siguiente FMA pero gasta 3 ZMM mas. Con 32
ZMM disponibles ese gasto sobra.
*/
static inline void
kernel_avx512_4x32(scalar_t       *restrict C, size_t ldc,
                   const scalar_t *restrict A, size_t lda,
                   const scalar_t *restrict B, size_t ldb,
                   size_t kc)
{
    /* ===== Fase 1: cargar el tile de C en los 8 acumuladores ===== */
    /* loadu (no load) porque las hojas Morton no garantizan
     * alineacion a 64 bytes; los offsets navegan a posiciones de
     * elemento, no de linea de cache. La penalizacion de loadu sobre
     * datos accidentalmente alineados es 0 ciclos en Zen 5. */
    __m512 c00 = _mm512_loadu_ps(&C[0 * ldc +  0]);
    __m512 c01 = _mm512_loadu_ps(&C[0 * ldc + 16]);
    __m512 c10 = _mm512_loadu_ps(&C[1 * ldc +  0]);
    __m512 c11 = _mm512_loadu_ps(&C[1 * ldc + 16]);
    __m512 c20 = _mm512_loadu_ps(&C[2 * ldc +  0]);
    __m512 c21 = _mm512_loadu_ps(&C[2 * ldc + 16]);
    __m512 c30 = _mm512_loadu_ps(&C[3 * ldc +  0]);
    __m512 c31 = _mm512_loadu_ps(&C[3 * ldc + 16]);

    /* ===== Fase 2: loop kc (el trabajo real) ===== */
    for (size_t p = 0; p < kc; ++p) {
        /* Una fila de B a profundidad p, partida en las mitades
         * baja y alta del tile de 32 columnas (16 floats por ZMM). */
        __m512 b0 = _mm512_loadu_ps(&B[p * ldb +  0]);
        __m512 b1 = _mm512_loadu_ps(&B[p * ldb + 16]);

        /* Broadcast: replica A[r, p] a las 16 lanes del ZMM.
         * _mm512_set1_ps en vez de _mm512_broadcast_ss porque AVX-512F
         * no expone vbroadcastss con destino ZMM directamente; set1_ps
         * compila a vpbroadcastd que si existe. El compilador puede
         * interleavear estos 4 broadcasts con los 8 FMAs para
         * esconder la latencia del broadcast detras del throughput FMA. */
        __m512 a0 = _mm512_set1_ps(A[0 * lda + p]);
        __m512 a1 = _mm512_set1_ps(A[1 * lda + p]);
        __m512 a2 = _mm512_set1_ps(A[2 * lda + p]);
        __m512 a3 = _mm512_set1_ps(A[3 * lda + p]);

        /* 8 FMAs independientes (sin dependencia RAW entre si dentro
         * de esta iteracion) => los dos pipes FMA de 512 bits del
         * Zen 5 las pipelinean sin stalls. */
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c01 = _mm512_fmadd_ps(a0, b1, c01);
        c10 = _mm512_fmadd_ps(a1, b0, c10);
        c11 = _mm512_fmadd_ps(a1, b1, c11);
        c20 = _mm512_fmadd_ps(a2, b0, c20);
        c21 = _mm512_fmadd_ps(a2, b1, c21);
        c30 = _mm512_fmadd_ps(a3, b0, c30);
        c31 = _mm512_fmadd_ps(a3, b1, c31);
    }

    /* ===== Fase 3: guardar los acumuladores de vuelta a memoria ===== */
    _mm512_storeu_ps(&C[0 * ldc +  0], c00);
    _mm512_storeu_ps(&C[0 * ldc + 16], c01);
    _mm512_storeu_ps(&C[1 * ldc +  0], c10);
    _mm512_storeu_ps(&C[1 * ldc + 16], c11);
    _mm512_storeu_ps(&C[2 * ldc +  0], c20);
    _mm512_storeu_ps(&C[2 * ldc + 16], c21);
    _mm512_storeu_ps(&C[3 * ldc +  0], c30);
    _mm512_storeu_ps(&C[3 * ldc + 16], c31);
}

#endif /* KERNEL_AVX512_MORTON_H */
