/*
 * kernel_avx512_morton.h - AVX-512 + FMA microkernel for a fixed 4x32
 *                          tile of C, used by the Morton-blocked family
 *                          (matmul_morton_avx512 and matmul_morton_omp).
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
 * (a separate .c) would spill all 12 ZMMs to the stack on entry and
 * destroy FMA throughput.
 *
 * Register budget of kernel_avx512_4x32 (14 of 32 ZMMs in use):
 *   8 ZMM C accumulators : c[row][half], row in 0..3, half in {0,1}
 *   2 ZMM B-panel loads  : b0 = cols  0..15, b1 = cols 16..31
 *   4 ZMM A broadcasts   : a0..a3 = A[row, p] broadcast to 16 lanes
 *
 * The kernel issues 8 independent vfmadd231ps per kc iteration. Zen 5
 * retires two 512-bit FMAs per cycle, so 8 FMAs = 4 cycles of compute
 * per kc step, matched by 2 vbroadcastss + 2 vmovups + 4 broadcasts
 * on the load side.
 *
 * Caller contract:
 *   - kc >= 1.
 *   - lda, ldb, ldc are leading dimensions in scalar_t units (row-major).
 *   - ldb >= 32, ldc >= 32. A is accessed at A[r*lda + p] for r in
 *     [0, MR) and p in [0, kc).
 *   - C, A, B are non-aliasing (restrict).
 *   - 64-byte alignment is preferred (matches a ZMM line) but not
 *     required: the kernel uses _mm512_loadu_ps / _mm512_storeu_ps.
 *
 * Semantics: ACCUMULATES into C. Caller is responsible for zeroing C
 * before the first invocation if a fresh result is needed.
 *
 * Requires: -march=native (or -mavx512f -mavx512vl) at the call site.
 */

#ifndef KERNEL_AVX512_MORTON_H
#define KERNEL_AVX512_MORTON_H

#include <stddef.h>

#include <immintrin.h>

#include "matrix_utils.h"  /* scalar_t */

/* Compile-time geometry constants. Exposed so callers (the Morton
 * leaf kernel) can align their block sizes to the tile without
 * re-declaring the magic numbers. MR is the row count per call;
 * NR the column count. */
#define KERNEL_AVX512_MORTON_MR 4u
#define KERNEL_AVX512_MORTON_NR 32u

/*
 * kernel_avx512_4x32: load C tile -> kc FMAs accumulating in
 * registers -> store C.
 *
 * 8 independent FMAs per p-step pipeline through the two 512-bit FMA
 * pipes without RAW hazards (each c[r][h] is the only writer of itself
 * in any iteration). This contrasts with the 6x32 tiled kernel which
 * reuses a single broadcast across all 6 rows (and trades the WAW on
 * the broadcast register against fewer live registers, viable on
 * larger MR).
 */
static inline void
kernel_avx512_4x32(scalar_t       *restrict C, size_t ldc,
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

        /* Four independent A broadcasts; the compiler can interleave
         * the four broadcasts with the eight FMAs to hide the
         * broadcast latency behind FMA throughput. */
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

#endif /* KERNEL_AVX512_MORTON_H */
