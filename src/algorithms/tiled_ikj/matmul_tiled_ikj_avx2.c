/*
 * matmul_tiled_ikj_avx2.c - BLIS-style 6x32 register-blocked matmul
 *                           (Zen 5 / EPYC 9R45 main_server variant).
 *
 * The "_avx2" suffix in the file name is legacy from the Zen 2
 * version. The leaf now uses the 6x32 AVX-512 microkernel exposed
 * by kernel_avx512_tiled.h. The AVX2 dispatch path (16-wide YMM
 * 6x16 kernel) has been removed because the EPYC 9R45 supports
 * AVX-512 natively and the wider tile doubles FMA throughput per
 * cycle.
 *
 * See matmul_tiled_ikj_avx2.h for the loop nest design and the
 * rationale behind the block sizes. This file contains:
 *   - the public matmul_tiled_ikj_avx2 function that orchestrates the
 *     three-level tile loop pc / ic / jr-ir, calling into the 6x32
 *     microkernel from kernel_avx512_tiled.h for the aligned interior
 *     and into kernel_avx512_tiled_residual_rows for the m % MR /
 *     n % NR tails;
 *   - the public benchmark_iterations_tiled_ikj_avx2 wrapper that
 *     handles the iterated B_{k+1} = A * B_k recurrence with the
 *     standard double-buffer + swap pattern.
 *
 * Correctness contract identical to matmul_naive: C is overwritten
 * via memset before the tile loops; A and B are read-only; no aliasing.
 *
 * Why the microkernel lives in a header (kernel_avx512_tiled.h)
 * instead of a separate .c: the 12-ZMM accumulator chain only stays
 * in registers when the compiler can see the full FMA chain in one
 * translation unit. Putting the microkernel in a separate .c would
 * force a function-call boundary and would spill all 12 ZMMs through
 * the ABI on entry. The header exposes the kernel as `static inline`
 * so this TU (and matmul_tiled_ikj_omp.c) each get their own inlined
 * copy under -O3.
 */

#include "matmul_tiled_ikj_avx2.h"
#include "matrix_utils.h"
#include "kernel_avx512_tiled.h"   /* kernel_avx512_tiled_6x32, residual */

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

void matmul_tiled_ikj_avx2(scalar_t *C,
                       const scalar_t *A,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n)
{
    const size_t MR = TILED_IKJ_AVX2_MR;
    const size_t NR = TILED_IKJ_AVX2_NR;
    const size_t MC = TILED_IKJ_AVX2_MC;
    const size_t KC = g_tiled_ikj_avx2_bs;

    memset(C, 0, m * n * sizeof(scalar_t));

    const size_t m_aligned = (m / MR) * MR;
    const size_t n_aligned = (n / NR) * NR;

    /* Outer loop on the contracted dimension: kc-deep panels of A and
     * B. Each pp iteration adds its partial sum into C (which is in
     * memory between pp iterations, but loaded into ZMMs inside the
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
             * same kc x nr B panel (kc=256, nr=32 -> 32 KiB, fits in
             * L1d=48 KiB), reusing it mc/mr times before moving to
             * the next jr panel. */
            for (size_t jr = 0; jr < n_aligned; jr += NR) {
                for (size_t ir = ic; ir + MR <= ic_end; ir += MR) {
                    kernel_avx512_tiled_6x32(&C[ir * n + jr], n,
                                             &A[ir * k + pp], k,
                                             &B[pp * n + jr], n,
                                             kc);
                }
            }

            /* Tail in n (n % NR != 0). With n=128 in this project this
             * branch is dead, but keep the kernel correct for arbitrary
             * n so the validator can call it at m=256 etc. */
            if (n_aligned < n) {
                kernel_avx512_tiled_residual_rows(
                    &C[ic * n + n_aligned], n,
                    &A[ic * k + pp], k,
                    &B[pp * n + n_aligned], n,
                    ic_end - ic, n - n_aligned, kc);
            }
        }

        /* Tail in m (m % MR != 0). Up to MR-1 = 5 residual rows. */
        if (m_aligned < m) {
            kernel_avx512_tiled_residual_rows(
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
