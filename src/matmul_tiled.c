/*
 * matmul_tiled.c - Explicitly tiled ikj matrix multiplication (Phase 1.2).
 *
 * The outer two loops tile i (step Mc) and k (step Kc) so that the active
 * panels of A, B, and C fit in L2 at the default sizes (Mc=Kc=256, n=128).
 * The innermost three loops are plain ikj, which has stride-1 access to both
 * B and C and scalar reuse of A[i,p] across all j.
 *
 * Locality at default tile sizes (float = 4 B, n = 128):
 *   A panel (Mc x Kc)  : 256 x 256 x 4 = 256 KB  -- reused Kc times in j
 *   B panel (Kc x n)   : 256 x 128 x 4 = 128 KB  -- stride-1 reads in j
 *   C panel (Mc x n)   : 256 x 128 x 4 = 128 KB  -- accumulated, stride-1
 *   Total               :                  512 KB  = L2 of Ryzen 5 4600H
 *
 * Correctness contract: C is fully overwritten (memset to zero before the
 * tiled loops), so the caller must not rely on C's previous contents.
 */

#include "matmul_tiled.h"
#include "matrix_utils.h"   /* xalloc_aligned, xfree */

#include <string.h>         /* memset, memcpy */

void matmul_tiled(scalar_t *C,
                  const scalar_t *A,
                  const scalar_t *B,
                  size_t m, size_t k, size_t n)
{
    const size_t Mc = TILED_MC_DEFAULT;
    const size_t Kc = TILED_KC_DEFAULT;

    memset(C, 0, m * n * sizeof(scalar_t));

    for (size_t ii = 0; ii < m; ii += Mc) {
        size_t i_end = ii + Mc < m ? ii + Mc : m;

        for (size_t pp = 0; pp < k; pp += Kc) {
            size_t p_end = pp + Kc < k ? pp + Kc : k;

            for (size_t i = ii; i < i_end; ++i) {
                for (size_t p = pp; p < p_end; ++p) {
                    scalar_t a_ip = A[i * k + p];
                    for (size_t j = 0; j < n; ++j)
                        C[i * n + j] += a_ip * B[p * n + j];
                }
            }
        }
    }
}

void benchmark_iterations_tiled(scalar_t *B_out,
                                 const scalar_t *A,
                                 const scalar_t *Z,
                                 size_t m, size_t n,
                                 size_t num_iters)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        matmul_tiled(B_next, A, B_curr, m, m, n);

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
