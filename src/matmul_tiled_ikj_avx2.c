/*
 * matmul_tiled_ikj_avx2.c - BLIS-style 6x16 register-blocked matmul.
 *
 * See matmul_tiled_ikj_avx2.h for the loop nest design and the rationale
 * behind the block sizes. This file contains:
 *   - the inline 6x16 microkernel (kernel_6x16) that keeps the C
 *     accumulators in 12 YMM registers throughout the kc sweep;
 *   - a vectorized scalar-tail fallback (accumulate_residual_rows) for
 *     the m % MR != 0 case, which acumulates a single residual row
 *     using broadcast + FMA but does NOT keep C in registers;
 *   - the public matmul_tiled_ikj_avx2 function that orchestrates the
 *     three-level tile loop pc / ic / jr-ir.
 *
 * Correctness contract identical to matmul_naive: C is overwritten
 * via memset before the tile loops; A and B are read-only; no aliasing.
 *
 * Why the kernel is in this file (not a separate .c like kernel_avx2):
 * the FMA chain only stays in registers when the compiler can see the
 * full loop in one translation unit. Putting the microkernel in a
 * separate file would force a function call boundary and would risk
 * spilling the 15 active YMMs through the ABI. With the kernel marked
 * `static inline` here and -O3 enabled, GCC inlines it into the ir
 * loop and keeps all accumulators in registers, which we verified by
 * inspecting objdump on this binary.
 */

#include "matmul_tiled_ikj_avx2.h"
#include "matrix_utils.h"

#include <immintrin.h>
#ifdef USE_AVX512
#include "kernel_avx512.h"
#endif
#include <stdio.h>
#include <string.h>

size_t g_tiled_ikj_avx2_bs = TILED_IKJ_AVX2_BS_DEFAULT;

void matmul_tiled_ikj_avx2_set_bs(size_t bs)
{
    if (bs == 0) {
        fprintf(stderr,
                "Warning: matmul_tiled_ikj_avx2_set_bs(0) ignored; "
                "bs must be positive.\n");
        return;
    }
    g_tiled_ikj_avx2_bs = bs;
}

/*
 * kernel_6x16: load C tile -> kc FMAs accumulating in registers -> store C.
 *
 * Pre: ldc >= 16, ldb >= 16, lda >= kc. The caller has zeroed C in the
 *      outer function so the loaded values just represent the partial
 *      sum from previous pc iterations.
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
static inline void kernel_6x16(scalar_t       *restrict C, size_t ldc,
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
 * accumulate_residual_rows: vectorized AVX2 fallback for m % MR rows.
 *
 * Each residual row r accumulates over [pp, pp+kc) iterations of p
 * with C[r, j] += A[r, p] * B[p, j]. C is NOT kept in registers here:
 * the load/FMA/store of one C row happens once per (r, p) pair. This
 * is slower than the microkernel by ~3x but only kicks in for up to
 * MR-1 = 5 residual rows, which is negligible at m >> MR.
 */
static void accumulate_residual_rows(scalar_t *restrict C, size_t ldc,
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

void matmul_tiled_ikj_avx2(scalar_t *C,
                       const scalar_t *A,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n)
{
    const size_t MR = TILED_IKJ_AVX2_MR;
#ifdef USE_AVX512
    const size_t NR = KERNEL_AVX512_NR;  /* 32: full-width ZMM tile */
#else
    const size_t NR = TILED_IKJ_AVX2_NR;
#endif
    const size_t MC = TILED_IKJ_AVX2_MC;
    const size_t KC = g_tiled_ikj_avx2_bs;

    memset(C, 0, m * n * sizeof(scalar_t));

    const size_t m_aligned = (m / MR) * MR;
    const size_t n_aligned = (n / NR) * NR;

    /* Outer loop on the contracted dimension: kc-deep panels of A and
     * B. Each pp iteration adds its partial sum into C (which is in
     * memory between pp iterations, but loaded into YMMs inside the
     * microkernel for each kc burst). */
    for (size_t pp = 0; pp < k; pp += KC) {
        const size_t kc = (pp + KC < k) ? KC : k - pp;

        /* Block over the m dimension to keep the A panel mc x kc in
         * L2 across all jr iterations of the same ic block. */
        for (size_t ic = 0; ic < m_aligned; ic += MC) {
            const size_t ic_end = (ic + MC <= m_aligned) ? ic + MC
                                                         : m_aligned;

            /* jr outer / ir inner is the BLIS canonical order: each
             * (ic, jr) pair sweeps mc/mr microkernels that share the
             * same kc x nr B panel (12 KiB, fits in L1d), reusing it
             * mc/mr times before moving to the next jr panel. */
            for (size_t jr = 0; jr < n_aligned; jr += NR) {
                for (size_t ir = ic; ir + MR <= ic_end; ir += MR) {
#ifdef USE_AVX512
                    kernel_avx512_6x32(&C[ir * n + jr], n,
                                       &A[ir * k + pp], k,
                                       &B[pp * n + jr], n,
                                       kc);
#else
                    kernel_6x16(&C[ir * n + jr], n,
                                &A[ir * k + pp], k,
                                &B[pp * n + jr], n,
                                kc);
#endif
                }
            }

            /* Tail in n (n % NR != 0). With n=128 in this project this
             * branch is dead, but keep the kernel correct for arbitrary
             * n so the validator can call it at m=256 etc. */
            if (n_aligned < n) {
                accumulate_residual_rows(
                    &C[ic * n + n_aligned], n,
                    &A[ic * k + pp], k,
                    &B[pp * n + n_aligned], n,
                    ic_end - ic, n - n_aligned, kc);
            }
        }

        /* Tail in m (m % MR != 0). Up to MR-1 = 5 residual rows. */
        if (m_aligned < m) {
            accumulate_residual_rows(
                &C[m_aligned * n], n,
                &A[m_aligned * k + pp], k,
                &B[pp * n], n,
                m - m_aligned, n, kc);
        }
    }
}

void benchmark_iterations_tiled_ikj_avx2(scalar_t *B_out,
                                     const scalar_t *A,
                                     const scalar_t *Z,
                                     size_t m, size_t n,
                                     size_t num_iters)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        matmul_tiled_ikj_avx2(B_next, A, B_curr, m, m, n);

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
