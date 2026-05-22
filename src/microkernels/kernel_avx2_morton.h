/*
 * kernel_avx2_morton.h - AVX2 + FMA microkernel for a fixed 4x16 tile
 *                       of C, used by the Morton-blocked family
 *                       (matmul_morton_avx2 and matmul_morton_omp).
 *
 * Sesion 03 / Prompt 3 (Stage A4 of the technical plan). Originally
 * shipped as kernel_avx2.{h,c} (a separately compiled .o linked into
 * the morton binaries). Renamed to kernel_avx2_morton.h when the src/
 * tree was reorganized and the sibling header kernel_avx2_tiled.h was
 * added for the 6x16 tile used by the tiled_ikj family. Converted to
 * header-only `static inline` at the same time, for consistency with
 * kernel_avx2_tiled.h and with the unified AVX-512 layout that
 * Juan Pablo introduced in opt_zen5 (kernel_avx512.h exposes both the
 * 6x32 and 4x32 kernels as `static inline` in a single header).
 *
 * Computes C[4 x 16] += A[4 x kc] * B[kc x 16] where kc is a runtime
 * parameter. The 4 x 16 geometry holds the entire C tile in 8 YMM
 * registers (4 rows x 2 vectors of 8 floats each), leaving 8 of the
 * 16 architectural YMM registers free for A broadcasts and B loads.
 * No element of C is written back to memory during the inner loop,
 * which lets the two Zen 2 FMA pipes stay saturated on every cycle
 * that the upstream loads can keep up.
 *
 * Caller responsibilities (the kernel itself checks nothing):
 *   - kc >= 1.
 *   - lda, ldb, ldc are the leading dimensions of A, B, C in their
 *     original allocations (number of columns per row, row-major).
 *   - ldb >= 16 and ldc >= 16. A is accessed only at positions
 *     A[r*lda + p] for r in [0,4) and p in [0,kc), so lda >= kc.
 *   - C, A, B are non-aliasing (restrict).
 *   - 32-byte alignment is preferred but not required; the
 *     implementation uses _mm256_loadu_ps / _mm256_storeu_ps so
 *     unaligned access is correct, just marginally slower.
 *
 * Semantics: the kernel ACCUMULATES into C. Callers that want a fresh
 * C must zero it before invocation.
 *
 * Requires: -mavx2 -mfma at the call site (CFLAGS_O3_ZEN2 or
 * CFLAGS_OMP_ZEN2).
 */

#ifndef KERNEL_AVX2_MORTON_H
#define KERNEL_AVX2_MORTON_H

#include <stddef.h>

#include <immintrin.h>

#include "matrix_utils.h"  /* scalar_t */

/* Compile-time geometry constants, exposed so callers (e.g. the
 * matmul_morton_avx2 leaf kernel) can align their block sizes to the
 * tile without re-declaring the magic numbers. KERNEL_AVX2_MR is the
 * row count handled per call; KERNEL_AVX2_NR the column count. */
#define KERNEL_AVX2_MR 4
#define KERNEL_AVX2_NR 16

/*
 * kernel_avx2_4x16: 4x16 AVX2 + FMA tile kernel for FP32 matmul.
 *
 * Layout of the eight C accumulators across the 16 architectural YMM
 * registers (Zen 2 has 16 named ymm registers and renames out of a
 * 168-entry physical register file, so we can assume no spill if we
 * use 16 or fewer at once):
 *
 *   row 0:  c00  c01      (cols  0..7   cols  8..15)
 *   row 1:  c10  c11
 *   row 2:  c20  c21
 *   row 3:  c30  c31
 *
 * Each iteration of the kc loop reads:
 *   - two 256-bit B vectors (b0, b1) covering 16 floats of row p of B,
 *   - four broadcasts (a0..a3) of A[r,p] for r in 0..3,
 * and issues 8 vfmadd231ps. Zen 2 retires up to two FMA ops per cycle,
 * so 8 FMAs are 4 cycles of compute per kc iteration. The two
 * vmovups + four vbroadcastss provide the operands; in steady state
 * the bottleneck is the FMA throughput, not the loads.
 */
static inline void
kernel_avx2_4x16(scalar_t       *restrict C, size_t ldc,
                 const scalar_t *restrict A, size_t lda,
                 const scalar_t *restrict B, size_t ldb,
                 size_t kc)
{
    /* Load the current C tile into eight YMM accumulators. Using
     * loadu / storeu so the kernel works for any leading dimension
     * (the leaves of matmul_morton_avx2 will not always be 32-byte
     * aligned because the Morton offsets land on element boundaries,
     * not on cache-line boundaries). */
    __m256 c00 = _mm256_loadu_ps(&C[0 * ldc + 0]);
    __m256 c01 = _mm256_loadu_ps(&C[0 * ldc + 8]);
    __m256 c10 = _mm256_loadu_ps(&C[1 * ldc + 0]);
    __m256 c11 = _mm256_loadu_ps(&C[1 * ldc + 8]);
    __m256 c20 = _mm256_loadu_ps(&C[2 * ldc + 0]);
    __m256 c21 = _mm256_loadu_ps(&C[2 * ldc + 8]);
    __m256 c30 = _mm256_loadu_ps(&C[3 * ldc + 0]);
    __m256 c31 = _mm256_loadu_ps(&C[3 * ldc + 8]);

    for (size_t p = 0; p < kc; ++p) {
        /* One row of B at depth p, split into the low and high halves
         * of the 16-column tile. */
        __m256 b0 = _mm256_loadu_ps(&B[p * ldb + 0]);
        __m256 b1 = _mm256_loadu_ps(&B[p * ldb + 8]);

        /* Broadcast a single A element per row of the tile. Each
         * broadcast replicates A[r,p] across all 8 lanes of the YMM
         * register, so the FMA below applies that scalar to every
         * column of the tile in a single instruction. */
        __m256 a0 = _mm256_broadcast_ss(&A[0 * lda + p]);
        __m256 a1 = _mm256_broadcast_ss(&A[1 * lda + p]);
        __m256 a2 = _mm256_broadcast_ss(&A[2 * lda + p]);
        __m256 a3 = _mm256_broadcast_ss(&A[3 * lda + p]);

        /* Eight independent FMAs per kc step. Independent meaning no
         * RAW dependency between them within this iteration, so they
         * pipeline through the two FMA units without stalls. */
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }

    /* Spill the accumulators back to memory. */
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
