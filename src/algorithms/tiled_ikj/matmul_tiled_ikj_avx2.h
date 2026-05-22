/*
 * matmul_tiled_ikj_avx2.h - BLIS-style 6x16 register-blocked matmul with
 *                       AVX2 + FMA, tuned for AMD Ryzen 5 4600H (Zen 2).
 *
 * Replaces the previous load/FMA/store-per-(i,p) implementation by a
 * register-blocked microkernel: a 6x16 sub-tile of C lives in 12 YMM
 * accumulators (6 rows x 2 vectors of 8 lanes) during the entire kc
 * sweep, so C touches memory only twice per micro-tile (load on entry,
 * store on exit). Layout of the architectural YMMs:
 *
 *      Active YMM registers per microkernel iteration (15 of 16):
 *        12 = C tile  : c00 c01 .. c50 c51
 *         2 = B panel : b0  b1
 *         1 = A bcast : a   (broadcast of A[r,p], reused across rows)
 *
 * Loop nest (BLIS Goto-style, jc collapsed because n=128 is small):
 *   pc loop   step kc  (== g_tiled_ikj_avx2_bs, configurable, default 192)
 *     ic loop step mc  (= 192, fixed in source)
 *       jr loop step nr (= 16, fixed: microkernel column count)
 *         ir loop step mr (= 6, fixed: microkernel row count)
 *           inner kernel sweeps p in [pc, pc+kc) in registers
 *
 * Block sizes target the Ryzen 5 4600H jerarquia:
 *   L1d: 32 KiB per core    -> B panel kc x nr  = 192*16*4 = 12 KiB
 *   L2:  512 KiB per core   -> A panel mc x kc  = 192*192*4 = 144 KiB
 *   L3:  4 MiB per CCX      -> A + B + C panels with margin
 *
 * Backwards-compat CLI: `[bs]` is still accepted by bench_tiled_ikj_avx2,
 * but now sets g_tiled_ikj_avx2_bs which is interpreted as kc (the inner
 * panel depth). mc / mr / nr are fixed in the source because their
 * optimal values are dictated by the register file geometry and not
 * by tuning. Validators that call matmul_tiled_ikj_avx2_set_bs() with
 * arbitrary values still work; the kernel handles any kc >= 1.
 */

#ifndef MATMUL_TILED_IKJ_AVX2_H
#define MATMUL_TILED_IKJ_AVX2_H

#include <stddef.h>
#include "matrix_utils.h"   /* scalar_t */

/* Default kc (depth of the inner panel of A, in units of scalar_t).
 * 384 chosen empirically on the Ryzen 5 4600H as a compromise across
 * m in {1024, 2048, 4096, 8192}: A panel mc=192 x kc=384 x 4 B = 288 KiB
 * fits comfortably in the 512 KiB per-core L2, while keeping the number
 * of pc iterations (k / kc) small enough that the per-tile C round-trip
 * does not dominate the FMA throughput. Sweeping bs in the validator
 * still works for any positive value. */
#define TILED_IKJ_AVX2_BS_DEFAULT 384u

/* Microkernel geometry (compile-time, not user-tunable). */
#define TILED_IKJ_AVX2_MR 6u
#define TILED_IKJ_AVX2_NR 16u
#define TILED_IKJ_AVX2_MC 192u

extern size_t g_tiled_ikj_avx2_bs;

void matmul_tiled_ikj_avx2_set_bs(size_t bs);

/*
 * matmul_tiled_ikj_avx2: compute C = A * B with BLIS-style register blocking.
 *
 * Same external contract as matmul_naive: C (m x n) is overwritten,
 * A (m x k) and B (k x n) are read-only, no aliasing. C is zeroed
 * internally with memset before the tile loops, so the kernel can
 * accumulate freely without exposing partial state.
 *
 * Edge handling: m % MR and n % NR are handled by a scalar-tail
 * fallback that runs vectorized AVX2 but does not register-block C.
 * In the project's benchmark m varies over {1024, 4096, 8192}; n=128
 * is divisible by NR=16, so only the m-tail (m % 6 in {2, 4}) takes
 * the fallback path.
 *
 * Requires: -mavx2 -mfma (CFLAGS_O3_ZEN2).
 */
void matmul_tiled_ikj_avx2(scalar_t *C,
                       const scalar_t *A,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n);

void benchmark_iterations_tiled_ikj_avx2(scalar_t *B_out,
                                     const scalar_t *A,
                                     const scalar_t *Z,
                                     size_t m, size_t n,
                                     size_t num_iters);

#endif /* MATMUL_TILED_IKJ_AVX2_H */
