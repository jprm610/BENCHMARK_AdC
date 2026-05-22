/*
 * matmul_tiled_ikj_omp.c - OpenMP-parallel BLIS-style 6x16 matmul.
 *
 * Sibling of matmul_tiled_ikj_avx2.c: same microkernel, same tile geometry,
 * but the public function opens an `omp parallel` region around the
 * pc loop and distributes the ic loop with `omp for schedule(static)`.
 *
 * Why the microkernel is duplicated here (not factored into a shared
 * header): the 12-YMM accumulator chain only stays in registers when
 * the compiler sees the full FMA chain in a single translation unit
 * and inlines it into the innermost ir loop. Putting the kernel in a
 * shared `static inline` header would work, but in practice with -O3
 * we observed GCC occasionally hoisting some YMM loads out of the
 * loop, spilling one accumulator to the stack. Keeping the kernel
 * inline in each .c makes the codegen deterministic and predictable.
 * The cost is ~70 duplicated lines, which is the standard BLIS-style
 * trade-off for tight FMA throughput.
 *
 * Correctness: each thread writes a disjoint range of C rows (the
 * outer `omp for` partitions the ic loop and within an ic block one
 * thread owns mc rows of C). A and B are read-only and shared. The
 * memset(C) is executed once before the parallel region.
 */

#include "matmul_tiled_ikj_omp.h"
#include "matrix_utils.h"

#include <immintrin.h>
#ifdef USE_AVX512
#include "kernel_avx512.h"
#endif
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

static inline void kernel_6x16(scalar_t       *restrict C, size_t ldc,
                               const scalar_t *restrict A, size_t lda,
                               const scalar_t *restrict B, size_t ldb,
                               size_t kc)
{
    __m256 c00 = _mm256_loadu_ps(&C[0 * ldc + 0]);
    __m256 c01 = _mm256_loadu_ps(&C[0 * ldc + 8]);
    __m256 c10 = _mm256_loadu_ps(&C[1 * ldc + 0]);
    __m256 c11 = _mm256_loadu_ps(&C[1 * ldc + 8]);
    __m256 c20 = _mm256_loadu_ps(&C[2 * ldc + 0]);
    __m256 c21 = _mm256_loadu_ps(&C[2 * ldc + 8]);
    __m256 c30 = _mm256_loadu_ps(&C[3 * ldc + 0]);
    __m256 c31 = _mm256_loadu_ps(&C[3 * ldc + 8]);
    __m256 c40 = _mm256_loadu_ps(&C[4 * ldc + 0]);
    __m256 c41 = _mm256_loadu_ps(&C[4 * ldc + 8]);
    __m256 c50 = _mm256_loadu_ps(&C[5 * ldc + 0]);
    __m256 c51 = _mm256_loadu_ps(&C[5 * ldc + 8]);

    for (size_t p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(&B[p * ldb + 0]);
        __m256 b1 = _mm256_loadu_ps(&B[p * ldb + 8]);
        __m256 a;

        a = _mm256_broadcast_ss(&A[0 * lda + p]);
        c00 = _mm256_fmadd_ps(a, b0, c00);
        c01 = _mm256_fmadd_ps(a, b1, c01);

        a = _mm256_broadcast_ss(&A[1 * lda + p]);
        c10 = _mm256_fmadd_ps(a, b0, c10);
        c11 = _mm256_fmadd_ps(a, b1, c11);

        a = _mm256_broadcast_ss(&A[2 * lda + p]);
        c20 = _mm256_fmadd_ps(a, b0, c20);
        c21 = _mm256_fmadd_ps(a, b1, c21);

        a = _mm256_broadcast_ss(&A[3 * lda + p]);
        c30 = _mm256_fmadd_ps(a, b0, c30);
        c31 = _mm256_fmadd_ps(a, b1, c31);

        a = _mm256_broadcast_ss(&A[4 * lda + p]);
        c40 = _mm256_fmadd_ps(a, b0, c40);
        c41 = _mm256_fmadd_ps(a, b1, c41);

        a = _mm256_broadcast_ss(&A[5 * lda + p]);
        c50 = _mm256_fmadd_ps(a, b0, c50);
        c51 = _mm256_fmadd_ps(a, b1, c51);
    }

    _mm256_storeu_ps(&C[0 * ldc + 0], c00);
    _mm256_storeu_ps(&C[0 * ldc + 8], c01);
    _mm256_storeu_ps(&C[1 * ldc + 0], c10);
    _mm256_storeu_ps(&C[1 * ldc + 8], c11);
    _mm256_storeu_ps(&C[2 * ldc + 0], c20);
    _mm256_storeu_ps(&C[2 * ldc + 8], c21);
    _mm256_storeu_ps(&C[3 * ldc + 0], c30);
    _mm256_storeu_ps(&C[3 * ldc + 8], c31);
    _mm256_storeu_ps(&C[4 * ldc + 0], c40);
    _mm256_storeu_ps(&C[4 * ldc + 8], c41);
    _mm256_storeu_ps(&C[5 * ldc + 0], c50);
    _mm256_storeu_ps(&C[5 * ldc + 8], c51);
}

static void accumulate_residual_rows(scalar_t *restrict C, size_t ldc,
                                     const scalar_t *restrict A, size_t lda,
                                     const scalar_t *restrict B, size_t ldb,
                                     size_t mr_eff, size_t n, size_t kc)
{
    for (size_t i = 0; i < mr_eff; ++i) {
        for (size_t p = 0; p < kc; ++p) {
            const scalar_t a_val = A[i * lda + p];
            const __m256   a_vec = _mm256_set1_ps(a_val);

            size_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 b = _mm256_loadu_ps(&B[p * ldb + j]);
                __m256 c = _mm256_loadu_ps(&C[i * ldc + j]);
                c = _mm256_fmadd_ps(a_vec, b, c);
                _mm256_storeu_ps(&C[i * ldc + j], c);
            }
            for (; j < n; ++j)
                C[i * ldc + j] += a_val * B[p * ldb + j];
        }
    }
}

void matmul_tiled_ikj_omp(scalar_t *C,
                      const scalar_t *A,
                      const scalar_t *B,
                      size_t m, size_t k, size_t n)
{
    const size_t MR = TILED_IKJ_OMP_MR;
#ifdef USE_AVX512
    const size_t NR = KERNEL_AVX512_NR;  /* 32: full-width ZMM tile */
#else
    const size_t NR = TILED_IKJ_OMP_NR;
#endif
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
#ifdef USE_AVX512
                        kernel_avx512_6x32(&C[ir * n + jr], n,
                                           &A[ir * k + pp], k,
                                           &B[pp * n + jr], n,
                                           kc);
#else
                        kernel_6x16(&C[ir * n + jr], n,
                                    &A[ir * k + pp], k,
                                    &B[pp * n + jr], n,
                                    kc);
#endif
                    }
                }

                if (n_aligned < n) {
                    accumulate_residual_rows(
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
                    accumulate_residual_rows(
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
