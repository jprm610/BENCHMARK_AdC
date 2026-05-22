/*
 * kernel_avx2.h - AVX2 + FMA microkernel for a fixed 4x16 tile of C.
 *
 * Sesion 03 / Prompt 3 (Stage A4 of the technical plan).
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
 */

#ifndef KERNEL_AVX2_H
#define KERNEL_AVX2_H

#include <stddef.h>

#include "matrix_utils.h"  /* scalar_t */

void kernel_avx2_4x16(scalar_t       *restrict C, size_t ldc,
                      const scalar_t *restrict A, size_t lda,
                      const scalar_t *restrict B, size_t ldb,
                      size_t kc);

/* Compile-time geometry constants, exposed so callers (e.g. the
 * upcoming matmul_morton_avx2 leaf kernel in Prompt 4) can align
 * their block sizes to the tile without re-declaring the magic
 * numbers. KERNEL_AVX2_MR is the row count handled per call;
 * KERNEL_AVX2_NR the column count. */
#define KERNEL_AVX2_MR 4
#define KERNEL_AVX2_NR 16

#endif /* KERNEL_AVX2_H */
