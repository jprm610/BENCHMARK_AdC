/*
 * kernel_avx2.c - 4x16 AVX2 + FMA tile kernel for FP32 matmul.
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

#include "kernel_avx2.h"

#include <immintrin.h>

void kernel_avx2_4x16(scalar_t       *restrict C, size_t ldc,
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
