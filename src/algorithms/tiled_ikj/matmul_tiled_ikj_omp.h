/*
 * matmul_tiled_ikj_omp.h - OpenMP-parallel BLIS-style 6x32
 *                      register-blocked matmul (Zen 5 / EPYC 9R45
 *                      main_server variant). Sibling of
 *                      matmul_tiled_ikj_avx512 with the same AVX-512
 *                      microkernel (kernel_avx512_tiled_6x32);
 *                      parallelizes the outermost mc loop (ic) so
 *                      each thread owns a contiguous range of C rows.
 *
 * Threading model: a single `omp parallel` region is opened in the
 * public function. The pc loop (k tile) runs in lockstep across all
 * threads (sequential semantics, the implicit barrier at the end of
 * `omp for` serves as the sync between pc iterations because all
 * threads must finish accumulating pp before pp+kc reads C). Within
 * each pc iteration, the ic loop is distributed via `omp for
 * schedule(static)`. Residual rows (m % MR != 0) are handled by a
 * single thread inside an `omp single` block.
 *
 * Recommended invocation on EPYC 9R45 (Zen 5, AWS c8a.2xlarge,
 * 8 cores, 1 thread/core, 32 MiB shared L3):
 *   OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
 * The hypervisor disables SMT so there is exactly one logical CPU
 * per physical core; close and spread are topologically equivalent
 * here (single NUMA node, shared L3).
 */

#ifndef MATMUL_TILED_IKJ_OMP_H
#define MATMUL_TILED_IKJ_OMP_H

#include <stddef.h>
#include "matrix_utils.h"   /* scalar_t */

/* Defaults sized for the EPYC 9R45: B panel kc * NR * 4 B fits in
 * L1d (48 KiB) at kc = 256, A panel mc * kc * 4 B fits in L2 (1 MiB)
 * at mc = 288. The Makefile may override at compile time. */
#ifndef TILED_IKJ_OMP_BS_DEFAULT
#define TILED_IKJ_OMP_BS_DEFAULT 256u
#endif
#define TILED_IKJ_OMP_MR 6u
#define TILED_IKJ_OMP_NR 32u
#ifndef TILED_IKJ_OMP_MC
#define TILED_IKJ_OMP_MC 288u
#endif

extern size_t g_tiled_ikj_omp_bs;

void matmul_tiled_ikj_omp_set_bs(size_t bs);

void matmul_tiled_ikj_omp(scalar_t *C,
                      const scalar_t *A,
                      const scalar_t *B,
                      size_t m, size_t k, size_t n);

void benchmark_iterations_tiled_ikj_omp(scalar_t *B_out,
                                    const scalar_t *A,
                                    const scalar_t *Z,
                                    size_t m, size_t n,
                                    size_t num_iters);

#endif /* MATMUL_TILED_IKJ_OMP_H */
