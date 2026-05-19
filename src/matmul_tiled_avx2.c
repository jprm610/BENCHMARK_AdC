/*
 * matmul_tiled_avx2.c - 6-loop tiled matmul with AVX2+FMA vectorization.
 *
 * Adapts the broadcast+FMA pattern from kernel_avx2.c into the 6-loop
 * tiling structure (3 outer block loops + 3 inner loops). The innermost
 * j pass uses _mm256_fmadd_ps with a broadcasted A scalar, equivalent
 * to the per-row step inside kernel_avx2_4x16 but without the 4-row
 * unrolling. A tail loop handles columns when the j-block is not a
 * multiple of 8.
 *
 * Compilation requirements: -mavx2 -mfma (included in CFLAGS_O3_ZEN2).
 * Without -mfma the compiler may emit separate VMULPS + VADDPS instead
 * of a single VFMADD231PS; the result is numerically correct but slower.
 */

#include "matmul_tiled_avx2.h"
#include "matrix_utils.h"   /* xalloc_aligned, xfree */

#include <immintrin.h>
#include <stdio.h>
#include <string.h>

size_t g_tiled_avx2_bs = TILED_AVX2_BS_DEFAULT;

void matmul_tiled_avx2_set_bs(size_t bs)
{
    if (bs == 0 || (bs % 8) != 0) {
        fprintf(stderr,
                "Warning: matmul_tiled_avx2_set_bs(%llu) ignored; "
                "bs must be a positive multiple of 8.\n",
                (unsigned long long)bs);
        return;
    }
    g_tiled_avx2_bs = bs;
}

void matmul_tiled_avx2(scalar_t *C,
                       const scalar_t *A,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n)
{
    const size_t BS = g_tiled_avx2_bs;

    memset(C, 0, m * n * sizeof(scalar_t));

    for (size_t ii = 0; ii < m; ii += BS) {
        size_t i_end = (ii + BS < m) ? ii + BS : m;

        for (size_t kk = 0; kk < k; kk += BS) {
            size_t k_end = (kk + BS < k) ? kk + BS : k;

            for (size_t jj = 0; jj < n; jj += BS) {
                size_t j_end     = (jj + BS < n) ? jj + BS : n;
                size_t j_vec_end = jj + ((j_end - jj) & ~(size_t)7);

                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t p = kk; p < k_end; ++p) {
                        scalar_t a_val = A[i * k + p];
                        __m256 a_vec   = _mm256_set1_ps(a_val);

                        for (size_t j = jj; j < j_vec_end; j += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[p * n + j]);
                            __m256 c_vec = _mm256_loadu_ps(&C[i * n + j]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&C[i * n + j], c_vec);
                        }

                        for (size_t j = j_vec_end; j < j_end; ++j)
                            C[i * n + j] += a_val * B[p * n + j];
                    }
                }
            }
        }
    }
}

void benchmark_iterations_tiled_avx2(scalar_t *B_out,
                                     const scalar_t *A,
                                     const scalar_t *Z,
                                     size_t m, size_t n,
                                     size_t num_iters)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        matmul_tiled_avx2(B_next, A, B_curr, m, m, n);

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
