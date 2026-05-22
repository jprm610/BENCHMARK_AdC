/*
 * matmul_tiled_ikj_omp.c - OpenMP-parallel BLIS-style 6x32 matmul
 *                          (Zen 5 / EPYC 9R45 main_server variant).
 *
 * Sibling of matmul_tiled_ikj_avx2.c: same microkernel, same tile
 * geometry, but the public function opens an `omp parallel` region
 * around the pc loop and distributes the ic loop with
 * `omp for schedule(static)`.
 *
 * The 6x32 microkernel and the AVX-512 residual-rows fallback live
 * in kernel_avx512_tiled.h and are exposed as `static inline` so
 * each translation unit (this file and matmul_tiled_ikj_avx2.c)
 * gets its own inlined copy under -O3 without crossing a
 * function-call boundary. The AVX2 dispatch path that the Zen 2
 * version carried in #ifdef branches has been removed because the
 * server CPU supports AVX-512 natively.
 *
 * Correctness: each thread writes a disjoint range of C rows (the
 * outer `omp for` partitions the ic loop and within an ic block one
 * thread owns mc rows of C). A and B are read-only and shared. The
 * memset(C) is executed once before the parallel region.
 */

#include "matmul_tiled_ikj_omp.h"
#include "matrix_utils.h"
#include "kernel_avx512_tiled.h"   /* kernel_avx512_tiled_6x32, residual */

#include <omp.h>
#include <stdio.h>
#include <string.h>

size_t g_tiled_ikj_omp_bs = TILED_IKJ_OMP_BS_DEFAULT;

void matmul_tiled_ikj_omp_set_bs(size_t bs)
{
    if (bs == 0) {
        fprintf(stderr,
                "Warning: matmul_tiled_ikj_omp_set_bs(0) ignored; "
                "bs must be positive.\n");
        return;
    }
    g_tiled_ikj_omp_bs = bs;
}

void matmul_tiled_ikj_omp(scalar_t *C,
                      const scalar_t *A,
                      const scalar_t *B,
                      size_t m, size_t k, size_t n)
{
    const size_t MR = TILED_IKJ_OMP_MR;
    const size_t NR = TILED_IKJ_OMP_NR;
    const size_t MC = TILED_IKJ_OMP_MC;
    const size_t KC = g_tiled_ikj_omp_bs;

    memset(C, 0, m * n * sizeof(scalar_t));

    const size_t m_aligned = (m / MR) * MR;
    const size_t n_aligned = (n / NR) * NR;

    #pragma omp parallel
    {
        for (size_t pp = 0; pp < k; pp += KC) {
            const size_t kc = (pp + KC < k) ? KC : k - pp;

            #pragma omp for schedule(static)
            for (size_t ic = 0; ic < m_aligned; ic += MC) {
                const size_t ic_end = (ic + MC <= m_aligned) ? ic + MC
                                                             : m_aligned;

                for (size_t jr = 0; jr < n_aligned; jr += NR) {
                    for (size_t ir = ic; ir + MR <= ic_end; ir += MR) {
                        kernel_avx512_tiled_6x32(&C[ir * n + jr], n,
                                                 &A[ir * k + pp], k,
                                                 &B[pp * n + jr], n,
                                                 kc);
                    }
                }

                if (n_aligned < n) {
                    kernel_avx512_tiled_residual_rows(
                        &C[ic * n + n_aligned], n,
                        &A[ic * k + pp], k,
                        &B[pp * n + n_aligned], n,
                        ic_end - ic, n - n_aligned, kc);
                }
            }
            /* Implicit barrier at the end of `omp for` ensures all
             * threads have finished writing C for this pp before any
             * thread reads C for pp+kc. */

            if (m_aligned < m) {
                #pragma omp single
                {
                    kernel_avx512_tiled_residual_rows(
                        &C[m_aligned * n], n,
                        &A[m_aligned * k + pp], k,
                        &B[pp * n], n,
                        m - m_aligned, n, kc);
                }
                /* `omp single` has an implicit barrier on exit, so the
                 * residual-rows accumulation is visible to all threads
                 * before the next pp iteration starts. */
            }
        }
    } /* end omp parallel */
}

void benchmark_iterations_tiled_ikj_omp(scalar_t *B_out,
                                    const scalar_t *A,
                                    const scalar_t *Z,
                                    size_t m, size_t n,
                                    size_t num_iters)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        matmul_tiled_ikj_omp(B_next, A, B_curr, m, m, n);

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
