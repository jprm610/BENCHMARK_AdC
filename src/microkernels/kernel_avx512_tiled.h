/*
 * kernel_avx512_tiled.h - AVX-512 + FMA 6x32 register-blocked
 *                          microkernel for the BLIS-style tiled_ikj
 *                          family (matmul_tiled_ikj_avx512 and
 *                          matmul_tiled_ikj_omp).
 *
 * Target: AMD EPYC 9R45 (Zen 5) on AWS c8a.2xlarge.
 *   - 8 cores, 1 thread/core (SMT disabled by the hypervisor)
 *   - L1d 48 KiB per core
 *   - L2  1 MiB per core
 *   - L3 32 MiB shared
 *   - 32 architectural ZMM registers, two 512-bit FMA pipes
 *
 * Header-only by design (`static inline`): each translation unit that
 * includes the header gets its own inlined copy at the call site, so
 * the 12 ZMM accumulators stay live in registers for the full kc
 * sweep without ABI save/restore. Forcing a function-call boundary
 * (a separate .c) would spill all 12 ZMMs to the stack and destroy
 * FMA throughput.
 *
 * Two routines are exposed:
 *
 *   - kernel_avx512_tiled_6x32: the BLIS microkernel that keeps a
 *     6x32 tile of C in 12 ZMM accumulators across the kc sweep
 *     (load on entry, FMA chain, store on exit).
 *
 *   - kernel_avx512_tiled_residual_rows: AVX-512 fallback for the
 *     m % MR and n % NR tails. C is not kept in registers here: the
 *     load/FMA/store of one C row happens once per (r, p) pair. Up
 *     to MR-1 = 5 residual rows per matmul invocation, so the
 *     throughput loss vs the microkernel (~3x slower per row) is
 *     negligible at m >> MR.
 *
 * Register budget of kernel_avx512_tiled_6x32 (15 of 32 ZMMs live):
 *   12 ZMM C accumulators : c[row][half], row in 0..5, half in {0,1}
 *    2 ZMM B-panel loads  : b0 = cols 0..15, b1 = cols 16..31
 *    1 ZMM A broadcast    : a = A[row, p] reused across the 6 rows
 *
 * The kernel issues 12 vfmadd231ps per kc iteration. Zen 5 retires
 * two 512-bit FMAs per cycle, so 12 FMAs = 6 cycles of compute per
 * kc step. The single reused `a` register imposes a WAW hazard that
 * the physical-register renamer resolves without stalls.
 *
 * Caller contract:
 *   - kc >= 1.
 *   - lda, ldb, ldc are leading dimensions in scalar_t units (row-major).
 *   - ldb >= NR (= 32), ldc >= NR. A is accessed at A[r*lda + p] for
 *     r in [0, MR) and p in [0, kc).
 *   - C, A, B are non-aliasing (restrict).
 *   - 64-byte alignment is preferred (matches a ZMM line) but not
 *     required: the kernel uses _mm512_loadu_ps / _mm512_storeu_ps.
 *
 * Semantics: ACCUMULATES into C. Caller is responsible for zeroing C
 * before the first invocation if a fresh result is needed.
 *
 * Requires: -march=native (or -mavx512f -mavx512vl) at the call site.
 */

#ifndef KERNEL_AVX512_TILED_H
#define KERNEL_AVX512_TILED_H

#include <stddef.h>

#include <immintrin.h>

#include "matrix_utils.h"  /* scalar_t */

/* Compile-time geometry constants. Exposed so the tiled_ikj
 * orchestrators can align their mc / kc panels to the tile without
 * re-declaring the magic numbers. */
#define KERNEL_AVX512_TILED_MR 6u
#define KERNEL_AVX512_TILED_NR 32u

/*
 * kernel_avx512_tiled_6x32: 6x32 C tile microkernel.
 *
 * Loads 12 ZMMs of C, sweeps kc rank-1 updates accumulating in the 12
 * ZMMs, then stores them back. The single shared `a` broadcast across
 * the 6 rows reduces the live-register count from 18 (6 broadcasts +
 * 2 B-loads + 12 C-accumulators -> 20) to 15, leaving headroom for
 * the renamer.
 */
static inline void
kernel_avx512_tiled_6x32(scalar_t       *restrict C, size_t ldc,
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
 * kernel_avx512_tiled_residual_rows: vectorized AVX-512 fallback for
 * the m % MR (up to 5 rows) and n % NR (less than 32 columns) tails.
 *
 * For each residual row r and each p in [0, kc), accumulates
 *   C[r, j] += A[r, p] * B[p, j]   for j in [0, n)
 * with 16-wide ZMM strides. C is NOT kept in registers; load/FMA/store
 * is paid per (r, p) pair. The scalar tail at the end of each row
 * handles n % 16, so the routine is correct for any n.
 */
static inline void
kernel_avx512_tiled_residual_rows(scalar_t       *restrict C, size_t ldc,
                                  const scalar_t *restrict A, size_t lda,
                                  const scalar_t *restrict B, size_t ldb,
                                  size_t mr_eff, size_t n, size_t kc)
{
    for (size_t i = 0; i < mr_eff; ++i) {
        for (size_t p = 0; p < kc; ++p) {
            const scalar_t a_val = A[i * lda + p];
            const __m512   a_vec = _mm512_set1_ps(a_val);

            size_t j = 0;
            for (; j + 16 <= n; j += 16) {
                __m512 b = _mm512_loadu_ps(&B[p * ldb + j]);
                __m512 c = _mm512_loadu_ps(&C[i * ldc + j]);
                c = _mm512_fmadd_ps(a_vec, b, c);
                _mm512_storeu_ps(&C[i * ldc + j], c);
            }
            /* Scalar tail for n % 16. With n = 128 in the project this
             * never executes, but keep it correct for arbitrary n so
             * the validator can call the kernel at m = 256, etc. */
            for (; j < n; ++j)
                C[i * ldc + j] += a_val * B[p * ldb + j];
        }
    }
}

#endif /* KERNEL_AVX512_TILED_H */
