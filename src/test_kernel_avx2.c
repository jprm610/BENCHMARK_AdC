/*
 * test_kernel_avx2.c - Standalone correctness test for the 4x16 AVX2
 * microkernel of kernel_avx2.c.
 *
 * Four tests, all independent of Morton and recursion. They exercise
 * the kernel as a pure tile multiply with growing kc:
 *
 *   Test 1: kc=1, A=[1,1,1,1]^T, B=known row of 16 distinct values.
 *           Expected: every row of C equals B. Smoke test for the
 *           broadcast + FMA in the simplest possible setting.
 *
 *   Test 2: kc=8, random A and B. Compare against a reference ijk
 *           in double precision. Relative tolerance 1e-4 to absorb
 *           the FP32 accumulation error.
 *
 *   Test 3: kc=128, idem. Representative of the leaf tile size that
 *           matmul_morton_avx2 will dispatch to the kernel.
 *
 *   Test 4: kc=1024, idem. Stress test: 1024 FMA chains per output
 *           lane; confirms the error does not blow up beyond what
 *           FP32 noise predicts.
 *
 * Each test prints "PASS" or "FAIL" and the program exits with code 0
 * only when all four pass.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kernel_avx2.h"
#include "matrix_utils.h"
#include "matmul_naive.h"  /* scalar_t */

#define MR KERNEL_AVX2_MR  /* 4  */
#define NR KERNEL_AVX2_NR  /* 16 */

static int g_failed = 0;

/* Reference matmul that accumulates the product in double precision and
 * stores the result back as scalar_t. This is the ground truth the AVX2
 * kernel must match within FP32-accumulation tolerance.
 *
 * Computes C[MR x NR] = A[MR x kc] * B[kc x NR] (overwrite). */
static void reference_matmul(scalar_t       *C, size_t ldc,
                             const scalar_t *A, size_t lda,
                             const scalar_t *B, size_t ldb,
                             size_t kc)
{
    for (size_t i = 0; i < MR; ++i) {
        for (size_t j = 0; j < NR; ++j) {
            double sum = 0.0;
            for (size_t p = 0; p < kc; ++p) {
                sum += (double)A[i * lda + p] * (double)B[p * ldb + j];
            }
            C[i * ldc + j] = (scalar_t)sum;
        }
    }
}

/* LCG used to fill A and B with reproducible random values in [-1, 1].
 * Independent from matrix_utils's init_matrix_random so the test does
 * not depend on the scaling 1/sqrt(rows) that the recurrence needs. */
static unsigned int lcg_state = 1u;

static void lcg_seed(unsigned int seed) { lcg_state = seed ? seed : 1u; }

static float lcg_next(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    /* Map to [-1, 1] via the top 24 bits. */
    return ((float)(lcg_state >> 8) / (float)(1u << 23)) - 1.0f;
}

static void fill_random(scalar_t *M, size_t n)
{
    for (size_t i = 0; i < n; ++i) M[i] = lcg_next();
}

/* Element-wise comparison with both absolute and relative tolerance.
 * Returns 1 on success, 0 on failure. On failure writes a short report
 * to stderr. */
static int compare_tiles(const scalar_t *expected,
                         const scalar_t *got,
                         size_t ldc_exp,
                         size_t ldc_got,
                         scalar_t abs_tol,
                         scalar_t rel_tol)
{
    for (size_t i = 0; i < MR; ++i) {
        for (size_t j = 0; j < NR; ++j) {
            scalar_t e = expected[i * ldc_exp + j];
            scalar_t g = got[i * ldc_got + j];
            scalar_t diff = (scalar_t)fabsf(e - g);
            scalar_t tol  = abs_tol;
            scalar_t magnitude = (scalar_t)fabsf(e);
            if (rel_tol * magnitude > tol) tol = rel_tol * magnitude;
            if (diff > tol) {
                /* Cast indices to unsigned long long for printf
                 * portability: glibc supports %zu but MinGW UCRT
                 * does not, and the project convention is to cast. */
                fprintf(stderr,
                        "    mismatch at (i=%llu, j=%llu): expected %.7g, "
                        "got %.7g, diff %.3g, tol %.3g\n",
                        (unsigned long long)i, (unsigned long long)j,
                        (double)e, (double)g,
                        (double)diff, (double)tol);
                return 0;
            }
        }
    }
    return 1;
}

static void run_check(const char *name, int passed)
{
    if (passed) {
        printf("  [%s] PASS\n", name);
    } else {
        printf("  [%s] FAIL\n", name);
        g_failed = 1;
    }
}

/* Test 1: kc=1. A is a column of ones, B is a single row with the
 * pattern B[j] = (float)(j + 1). The product C = A * B should be a
 * tile where every one of the 4 rows is identical to B. Smoke test
 * for the broadcast and the FMA path with a single iteration. */
static void test_kc_1(void)
{
    printf("Test 1: kc=1, A=ones column, B=known row\n");
    const size_t kc  = 1;
    const size_t lda = kc;
    const size_t ldb = NR;
    const size_t ldc = NR;

    scalar_t *A = xalloc_aligned(MR * lda);
    scalar_t *B = xalloc_aligned(kc * ldb);
    scalar_t *C = xalloc_aligned(MR * ldc);
    scalar_t *E = xalloc_aligned(MR * ldc);   /* expected */

    for (size_t i = 0; i < MR; ++i) A[i * lda + 0] = 1.0f;
    for (size_t j = 0; j < NR; ++j) B[0 * ldb + j] = (scalar_t)(j + 1);

    init_matrix_zero(C, MR, ldc);

    /* Expected: every row of C equals B. */
    for (size_t i = 0; i < MR; ++i) {
        for (size_t j = 0; j < NR; ++j) {
            E[i * ldc + j] = B[0 * ldb + j];
        }
    }

    kernel_avx2_4x16(C, ldc, A, lda, B, ldb, kc);

    run_check("kc=1 ones x known row", compare_tiles(E, C, ldc, ldc,
                                                     1e-6f, 1e-6f));

    xfree(A); xfree(B); xfree(C); xfree(E);
}

/* Tests 2 through 4: random A and B, compare AVX2 kernel output
 * against the double-precision reference. The relative tolerance is
 * set at 1e-4 to absorb FP32 accumulation noise; the absolute
 * tolerance protects values near zero. */
static void test_random_with_kc(const char *name, size_t kc)
{
    printf("Test: %s\n", name);
    const size_t lda = kc;
    const size_t ldb = NR;
    const size_t ldc = NR;

    scalar_t *A = xalloc_aligned(MR * lda);
    scalar_t *B = xalloc_aligned(kc * ldb);
    scalar_t *C = xalloc_aligned(MR * ldc);
    scalar_t *E = xalloc_aligned(MR * ldc);

    fill_random(A, MR * lda);
    fill_random(B, kc * ldb);

    init_matrix_zero(C, MR, ldc);
    init_matrix_zero(E, MR, ldc);

    reference_matmul(E, ldc, A, lda, B, ldb, kc);
    kernel_avx2_4x16(C, ldc, A, lda, B, ldb, kc);

    /* Tolerance scales mildly with kc because the FP32 accumulation
     * error grows like sqrt(kc) * epsilon. At kc=1024 we have about
     * 32 * 6e-8 ~= 2e-6 of unavoidable noise; 1e-4 relative leaves
     * generous headroom. */
    scalar_t abs_tol = 1e-5f;
    scalar_t rel_tol = 1e-4f;

    run_check(name, compare_tiles(E, C, ldc, ldc, abs_tol, rel_tol));

    xfree(A); xfree(B); xfree(C); xfree(E);
}

int main(void)
{
    printf("=== kernel_avx2_4x16 unit tests ===\n");
    printf("MR = %d, NR = %d\n", MR, NR);

    lcg_seed(0xC0FFEEu);

    test_kc_1();
    test_random_with_kc("kc=8 random",    8);
    test_random_with_kc("kc=128 random",  128);
    test_random_with_kc("kc=1024 random", 1024);

    if (g_failed) {
        printf("RESULT: FAIL\n");
        return EXIT_FAILURE;
    }
    printf("RESULT: PASS (all four)\n");
    return EXIT_SUCCESS;
}
