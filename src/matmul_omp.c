/*
 * matmul_omp.c - 6-loop tiled matmul with AVX2+FMA and OpenMP parallelization.
 *
 * Identical to matmul_tiled_avx2.c except for one pragma on the outermost
 * ii loop. Each ii tile owns rows [ii, ii+BS) of C exclusively, so there
 * are no write conflicts between threads. A and B are read-only and shared.
 * The memset runs before the parallel region, sequentially.
 *
 * Compilation requirements: -mavx2 -mfma -fopenmp.
 */

#include "matmul_omp.h"
#include "matrix_utils.h"   /* xalloc_aligned, xfree */

#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

size_t g_omp_bs = OMP_BS_DEFAULT;

void matmul_omp_set_bs(size_t bs)
{
    if (bs == 0 || (bs % 8) != 0) {
        fprintf(stderr,
                "Warning: matmul_omp_set_bs(%llu) ignored; "
                "bs must be a positive multiple of 8.\n",
                (unsigned long long)bs);
        return;
    }
    g_omp_bs = bs;
}

void matmul_omp(scalar_t *C,
                const scalar_t *A,
                const scalar_t *B,
                size_t m, size_t k, size_t n)
{
    const size_t BS = g_omp_bs;

    memset(C, 0, m * n * sizeof(scalar_t));

    #pragma omp parallel for schedule(static)
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

void benchmark_iterations_omp(scalar_t *B_out,
                               const scalar_t *A,
                               const scalar_t *Z,
                               size_t m, size_t n,
                               size_t num_iters)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        matmul_omp(B_next, A, B_curr, m, m, n);

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
