/*
 * kernel_avx2_tiled.h - AVX2 + FMA 6x16 register-blocked microkernel for
 *                       the BLIS-style tiled_ikj family.
 *
 * Header-only by design. Both matmul_tiled_ikj_avx2.c and
 * matmul_tiled_ikj_omp.c include this file and the microkernel is
 * exposed as `static inline` so the compiler can inline it into the
 * innermost ir loop of each translation unit. The reason the kernel
 * lives in a header (not in a .c paired with .h like
 * kernel_avx2_morton) is documented at length in the comments of the
 * tiled_ikj_avx2.c file: forcing a function-call boundary across the
 * 12-YMM accumulator chain risks spilling registers through the ABI,
 * which kills FMA throughput on Zen 2. Keeping the body in this
 * header with `static inline` lets each .c get its own inlined copy
 * with deterministic codegen.
 *
 * Two routines are exposed:
 *   - kernel_avx2_tiled_6x16: 6x16 microkernel that keeps C in 12 YMM
 *     accumulators across the kc sweep (load on entry, FMA chain,
 *     store on exit).
 *   - kernel_avx2_tiled_residual_rows: AVX2 fallback for the m % MR
 *     and n % NR tails. C is not kept in registers here: the
 *     load/FMA/store of one C row happens once per (r, p) pair. Up to
 *     MR-1 = 5 residual rows per matmul invocation, so the throughput
 *     loss is negligible at m >> MR.
 *
 * Caller contract (same as kernel_avx2_morton_4x16):
 *   - kc >= 1.
 *   - lda, ldb, ldc are leading dimensions in scalar_t units (row-major).
 *   - ldb >= NR and ldc >= NR. A is accessed at A[r*lda + p] for
 *     r in [0, MR) and p in [0, kc).
 *   - C, A, B are non-aliasing (restrict).
 *   - 32-byte alignment preferred but not required (uses loadu/storeu).
 *
 * Semantics: ACCUMULATES into C. Caller is responsible for zeroing C
 * before the first invocation if a fresh result is needed.
 *
 * Requires: -mavx2 -mfma at the call site (CFLAGS_O3_ZEN2 or
 * CFLAGS_OMP_ZEN2).
 */

#ifndef KERNEL_AVX2_TILED_H
#define KERNEL_AVX2_TILED_H

#include <stddef.h>

#include <immintrin.h>

#include "matrix_utils.h"  /* scalar_t */

/* Compile-time geometry constants. Exposed so callers (the tiled_ikj
 * orchestrators) can align their mc / kc panels to the tile without
 * re-declaring the magic numbers. KERNEL_AVX2_TILED_MR is the row
 * count per microkernel call; KERNEL_AVX2_TILED_NR the column count. */
#define KERNEL_AVX2_TILED_MR 6u
#define KERNEL_AVX2_TILED_NR 16u

/*
 * kernel_avx2_tiled_6x16: load C tile -> kc FMAs accumulating in
 * registers -> store C.
 *
 * 15 of 16 architectural YMM registers are live in the steady state:
 * 12 for the C accumulators, 2 for the two B halves of row p, 1 for
 * the broadcasted A scalar (reused across the 6 rows). Reusing the
 * same `a` register across rows introduces a WAW hazard that the
 * physical-register renamer (168 entries on Zen 2) resolves without
 * a stall; the per-iteration cost is 6 broadcasts (1 cycle each, 2
 * pipes) and 12 FMAs (5 cycles each, 2 pipes), which balance at 6
 * cycles of FMA throughput per kc step.
 */
static inline void
kernel_avx2_tiled_6x16(scalar_t       *restrict C, size_t ldc,
                       const scalar_t *restrict A, size_t lda,
                       const scalar_t *restrict B, size_t ldb,
                       size_t kc)
{
    __m256 c00 = _mm256_loadu_ps(&C[0 * ldc + 0]);
    __m256 c01 = _mm256_loadu_ps(&C[0 * ldc + 8]);
    __m256 c10 = _mm256_loadu_ps(&C[1 * ldc + 0]);
    __m256 c11 = _mm256_loadu_ps(&C[1 * ldc + 8]);
    __m256 c20 = _mm256_loadu_ps(&C[2 * ldc + 0]);
    __m256 c21 = _mm256_loadu_ps(&C[2 * ldc + 8]);
    __m256 c30 = _mm256_loadu_ps(&C[3 * ldc + 0]);
    __m256 c31 = _mm256_loadu_ps(&C[3 * ldc + 8]);
    __m256 c40 = _mm256_loadu_ps(&C[4 * ldc + 0]);
    __m256 c41 = _mm256_loadu_ps(&C[4 * ldc + 8]);
    __m256 c50 = _mm256_loadu_ps(&C[5 * ldc + 0]);
    __m256 c51 = _mm256_loadu_ps(&C[5 * ldc + 8]);

    for (size_t p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(&B[p * ldb + 0]);
        __m256 b1 = _mm256_loadu_ps(&B[p * ldb + 8]);
        __m256 a;

        a = _mm256_broadcast_ss(&A[0 * lda + p]);
        c00 = _mm256_fmadd_ps(a, b0, c00);
        c01 = _mm256_fmadd_ps(a, b1, c01);

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
    }

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
 * kernel_avx2_tiled_residual_rows: vectorized AVX2 fallback for the
 * m % MR and n % NR tails.
 *
 * Each residual row r accumulates over [pp, pp+kc) iterations of p
 * with C[r, j] += A[r, p] * B[p, j]. C is NOT kept in registers here:
 * the load/FMA/store of one C row happens once per (r, p) pair. This
 * is slower than the microkernel by ~3x but only kicks in for up to
 * MR-1 = 5 residual rows, which is negligible at m >> MR.
 */
static inline void
kernel_avx2_tiled_residual_rows(scalar_t       *restrict C, size_t ldc,
                                const scalar_t *restrict A, size_t lda,
                                const scalar_t *restrict B, size_t ldb,
                                size_t mr_eff, size_t n, size_t kc)
{
    for (size_t i = 0; i < mr_eff; ++i) {
        for (size_t p = 0; p < kc; ++p) {
            const scalar_t a_val = A[i * lda + p];
            const __m256   a_vec = _mm256_set1_ps(a_val);

            size_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 b = _mm256_loadu_ps(&B[p * ldb + j]);
                __m256 c = _mm256_loadu_ps(&C[i * ldc + j]);
                c = _mm256_fmadd_ps(a_vec, b, c);
                _mm256_storeu_ps(&C[i * ldc + j], c);
            }
            for (; j < n; ++j)
                C[i * ldc + j] += a_val * B[p * ldb + j];
        }
    }
}

#endif /* KERNEL_AVX2_TILED_H */
