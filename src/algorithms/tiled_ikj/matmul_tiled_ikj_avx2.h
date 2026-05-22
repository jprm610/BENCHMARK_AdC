/*
 * matmul_tiled_ikj_avx2.h - BLIS-style 6x32 register-blocked matmul
 *                      with AVX-512 + FMA, tuned for AMD EPYC 9R45
 *                      (Zen 5) on AWS c8a.2xlarge.
 *
 * The "_avx2" suffix in the file name is legacy from the Zen 2
 * variant; the leaf now uses the 6x32 AVX-512 microkernel from
 * kernel_avx512_tiled.h. A 6x32 sub-tile of C lives in 12 ZMM
 * accumulators (6 rows x 2 ZMMs of 16 lanes) during the entire kc
 * sweep, so C touches memory only twice per micro-tile (load on
 * entry, store on exit). Layout of the architectural ZMMs:
 *
 *      Active ZMM registers per microkernel iteration (15 of 32):
 *        12 = C tile  : c00 c01 .. c50 c51
 *         2 = B panel : b0  b1
 *         1 = A bcast : a   (broadcast of A[r,p], reused across rows)
 *
 * Loop nest (BLIS Goto-style, jc collapsed because n=128 is small):
 *   pc loop   step kc  (== g_tiled_ikj_avx2_bs, configurable, default 256)
 *     ic loop step mc  (= 288, fixed in source / -DTILED_IKJ_AVX2_MC)
 *       jr loop step nr (= 32, fixed: microkernel column count)
 *         ir loop step mr (= 6, fixed: microkernel row count)
 *           inner kernel sweeps p in [pc, pc+kc) in registers
 *
 * Block sizes target the EPYC 9R45 hierarchy:
 *   L1d: 48 KiB per core   -> B panel kc x nr = 256 * 32 * 4 = 32 KiB
 *   L2:   1 MiB per core   -> A panel mc x kc = 288 * 256 * 4 = 288 KiB
 *   L3:  32 MiB shared     -> A + B + C panels with very wide margin
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
 * 256 picked so that the B panel kc * NR * 4 B = 256 * 32 * 4 = 32 KiB
 * fits inside L1d (48 KiB) with room for A streaming. The Makefile
 * may override with -DTILED_IKJ_AVX2_BS_DEFAULT=... for sweeps. */
#ifndef TILED_IKJ_AVX2_BS_DEFAULT
#define TILED_IKJ_AVX2_BS_DEFAULT 256u
#endif

/* Microkernel geometry (compile-time, not user-tunable). MR / NR
 * mirror KERNEL_AVX512_TILED_MR / _NR exactly; MC is the outer-A
 * blocking factor sized so that mc * kc * 4 B fits in the per-core
 * L2 (1 MiB). 288 = 48 * MR keeps the block count clean. */
#define TILED_IKJ_AVX2_MR 6u
#define TILED_IKJ_AVX2_NR 32u
#ifndef TILED_IKJ_AVX2_MC
#define TILED_IKJ_AVX2_MC 288u
#endif

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
 * Edge handling: m % MR and n % NR are handled by an AVX-512 fallback
 * (kernel_avx512_tiled_residual_rows) that vectorizes by row but does
 * not register-block C. In the project's benchmark m varies over
 * {1024, 2048, 4096, 8192}; n=128 is divisible by NR=32, so only the
 * m-tail (m % 6 in {2, 4}) takes the fallback path.
 *
 * Requires: -march=native (or -mavx512f -mavx512vl) at the call site.
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
