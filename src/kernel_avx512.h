/*
 * kernel_avx512.h - BLIS-style 6x32 AVX-512 register-blocked microkernel.
 *
 * Replaces the 6x16 YMM microkernel (kernel_6x16) when compiled with
 * -DUSE_AVX512=1 and -march=native on a CPU with AVX-512F+FMA support
 * (Zen 4 / Zen 5, Intel Skylake-X and later).
 *
 * Register layout (15 of 32 ZMM registers):
 *   12 ZMM accumulators: c[row][half], row in 0..5, half in {0,1}
 *    2 ZMM for B panels:  b0 = cols 0-15, b1 = cols 16-31
 *    1 ZMM for A bcast:   a  = A[row, p] broadcast to all 16 lanes
 *
 * Each ZMM holds 16 single-precision floats, so NR=32 columns are
 * covered by 2 ZMMs per C row. This doubles FMA throughput per cycle
 * vs the 6x16 YMM kernel on hardware with full-width 512-bit FMA pipes
 * (Zen 4/5 have two 512-bit FMA ports).
 *
 * The kernel is `static inline` so it inlines into the ir loop of each
 * calling translation unit, keeping the 12 ZMM accumulators in registers
 * for the full kc sweep and avoiding ABI save/restore per (ir, jr) call.
 *
 * Included only when USE_AVX512 is defined. The parent .c files guard
 * the include and the NR selection with #ifdef USE_AVX512.
 */

#ifndef KERNEL_AVX512_H
#define KERNEL_AVX512_H

#include <immintrin.h>
#include <stddef.h>
#include "matmul_naive.h"   /* scalar_t */

/* 6x32 kernel geometry (tiled_ikj variants) */
#define KERNEL_AVX512_MR 6u
#define KERNEL_AVX512_NR 32u

/* 4x32 kernel geometry (morton variants — MR=4 matches MORTON_AVX2_TILE) */
#define KERNEL_AVX512_4X32_MR 4u
#define KERNEL_AVX512_4X32_NR 32u

/*
 * kernel_avx512_6x32: accumulate a 6x32 C tile from a kc-deep rank-1
 * update.  C must be pre-zeroed (or hold a partial sum from earlier pc
 * iterations); the result is written back to C on exit.
 *
 * Pre: ldc >= 32, ldb >= 32, lda >= kc. All pointers 4-byte aligned
 * (required by the benchmark's xalloc_aligned).
 */
static inline void kernel_avx512_6x32(scalar_t       *restrict C, size_t ldc,
                                      const scalar_t *restrict A, size_t lda,
                                      const scalar_t *restrict B, size_t ldb,
                                      size_t kc)
{
    __m512 c00 = _mm512_loadu_ps(&C[0 * ldc +  0]);
    __m512 c01 = _mm512_loadu_ps(&C[0 * ldc + 16]);
    __m512 c10 = _mm512_loadu_ps(&C[1 * ldc +  0]);
    __m512 c11 = _mm512_loadu_ps(&C[1 * ldc + 16]);
    __m512 c20 = _mm512_loadu_ps(&C[2 * ldc +  0]);
    __m512 c21 = _mm512_loadu_ps(&C[2 * ldc + 16]);
    __m512 c30 = _mm512_loadu_ps(&C[3 * ldc +  0]);
    __m512 c31 = _mm512_loadu_ps(&C[3 * ldc + 16]);
    __m512 c40 = _mm512_loadu_ps(&C[4 * ldc +  0]);
    __m512 c41 = _mm512_loadu_ps(&C[4 * ldc + 16]);
    __m512 c50 = _mm512_loadu_ps(&C[5 * ldc +  0]);
    __m512 c51 = _mm512_loadu_ps(&C[5 * ldc + 16]);

    for (size_t p = 0; p < kc; ++p) {
        __m512 b0 = _mm512_loadu_ps(&B[p * ldb +  0]);
        __m512 b1 = _mm512_loadu_ps(&B[p * ldb + 16]);
        __m512 a;

        a = _mm512_set1_ps(A[0 * lda + p]);
        c00 = _mm512_fmadd_ps(a, b0, c00);
        c01 = _mm512_fmadd_ps(a, b1, c01);

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
    }

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
 * kernel_avx512_4x32: accumulate a 4x32 C tile from a kc-deep rank-1
 * update. Sibling of kernel_avx512_6x32 tuned for the Morton leaf where
 * MR=4 matches the Morton-of-blocks tile size (MORTON_AVX2_TILE).
 *
 * Register layout (14 of 32 ZMM registers):
 *   8 ZMM accumulators: c[row][half], row in 0..3, half in {0,1}
 *   2 ZMM for B panels:  b0 = cols 0-15, b1 = cols 16-31
 *   4 ZMM for A bcast:   a0..a3 = A[row, p] broadcast to all 16 lanes
 */
static inline void kernel_avx512_4x32(scalar_t       *restrict C, size_t ldc,
                                      const scalar_t *restrict A, size_t lda,
                                      const scalar_t *restrict B, size_t ldb,
                                      size_t kc)
{
    __m512 c00 = _mm512_loadu_ps(&C[0 * ldc +  0]);
    __m512 c01 = _mm512_loadu_ps(&C[0 * ldc + 16]);
    __m512 c10 = _mm512_loadu_ps(&C[1 * ldc +  0]);
    __m512 c11 = _mm512_loadu_ps(&C[1 * ldc + 16]);
    __m512 c20 = _mm512_loadu_ps(&C[2 * ldc +  0]);
    __m512 c21 = _mm512_loadu_ps(&C[2 * ldc + 16]);
    __m512 c30 = _mm512_loadu_ps(&C[3 * ldc +  0]);
    __m512 c31 = _mm512_loadu_ps(&C[3 * ldc + 16]);

    for (size_t p = 0; p < kc; ++p) {
        __m512 b0 = _mm512_loadu_ps(&B[p * ldb +  0]);
        __m512 b1 = _mm512_loadu_ps(&B[p * ldb + 16]);

        __m512 a0 = _mm512_set1_ps(A[0 * lda + p]);
        __m512 a1 = _mm512_set1_ps(A[1 * lda + p]);
        __m512 a2 = _mm512_set1_ps(A[2 * lda + p]);
        __m512 a3 = _mm512_set1_ps(A[3 * lda + p]);

        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c01 = _mm512_fmadd_ps(a0, b1, c01);
        c10 = _mm512_fmadd_ps(a1, b0, c10);
        c11 = _mm512_fmadd_ps(a1, b1, c11);
        c20 = _mm512_fmadd_ps(a2, b0, c20);
        c21 = _mm512_fmadd_ps(a2, b1, c21);
        c30 = _mm512_fmadd_ps(a3, b0, c30);
        c31 = _mm512_fmadd_ps(a3, b1, c31);
    }

    _mm512_storeu_ps(&C[0 * ldc +  0], c00);
    _mm512_storeu_ps(&C[0 * ldc + 16], c01);
    _mm512_storeu_ps(&C[1 * ldc +  0], c10);
    _mm512_storeu_ps(&C[1 * ldc + 16], c11);
    _mm512_storeu_ps(&C[2 * ldc +  0], c20);
    _mm512_storeu_ps(&C[2 * ldc + 16], c21);
    _mm512_storeu_ps(&C[3 * ldc +  0], c30);
    _mm512_storeu_ps(&C[3 * ldc + 16], c31);
}

#endif /* KERNEL_AVX512_H */
