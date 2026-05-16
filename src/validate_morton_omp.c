/*
 * validate_morton_omp.c - Sanity checks for matmul_morton_omp.
 *
 * Mirror of validate_morton_avx2 (same tolerances, same Morton-of-
 * blocks layout, same five-test pattern) with one extra layer: every
 * cross-validation test is executed with OMP_NUM_THREADS in
 * {1, 4, 12} so we exercise the parallel recursion in three distinct
 * regimes:
 *
 *   - 1 thread:  serial path; should match matmul_morton_avx2 exactly
 *                (modulo -ffast-math reassociation, which the loose
 *                tolerances absorb).
 *   - 4 threads: real parallelism; would expose any race in the
 *                top/bottom MK split or in the n split.
 *   - 12 threads (SMT): the parallel team is larger than the cores
 *                count, so the OS scheduler shuffles tasks; any
 *                latent scratch-buffer reuse bug shows up here as a
 *                random-looking failure.
 *
 * If any thread count fails, the binary returns non-zero so the
 * Makefile's validate_morton_omp target catches the regression.
 *
 * Usage:
 *   validate_morton_omp_O3 [m]
 *
 * m must be a power of two and >= MORTON_AVX2_TILE = 4. Default: 256.
 *
 * Note: OMP_NUM_THREADS from the environment still applies to the
 * "current-env" round (the first time through the tests). The sweep
 * uses omp_set_num_threads() which overrides the env for subsequent
 * parallel regions in this thread, per OpenMP semantics.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#include "matmul_naive.h"
#include "matmul_morton.h"
#include "matmul_morton_avx2.h"
#include "matmul_morton_omp.h"
#include "morton.h"
#include "matrix_utils.h"

#define DEFAULT_M 256u
#define BLOCK_N   128u

/* Same tolerances as validate_morton_avx2: -ffast-math in the
 * microkernel allows reassociation of FP additions. The OMP layer
 * does not add new floating-point ops, but tasks may execute the
 * sub-products in a slightly different order than the serial AVX2
 * version (e.g. top before bottom vs. bottom before top), which can
 * change the order of the final accumulation steps inside the leaf.
 * In practice the difference is tiny but we keep the same loose
 * tolerance for robustness. */
static const scalar_t ABS_TOL = (scalar_t)1.0e-4;
static const scalar_t REL_TOL = (scalar_t)1.0e-3;

static int check_or_report(const char *test_name,
                           const scalar_t *ref,
                           const scalar_t *got,
                           size_t num_elements)
{
    size_t bad_idx = 0;
    scalar_t bad_ref = 0, bad_got = 0;
    int ok = matrices_close(ref, got, num_elements,
                            ABS_TOL, REL_TOL,
                            &bad_idx, &bad_ref, &bad_got);
    if (ok) {
        printf("  [OK]   %s\n", test_name);
        return 0;
    } else {
        printf("  [FAIL] %s: at index %llu, expected %.6e, got %.6e\n",
               test_name, (unsigned long long)bad_idx,
               (double)bad_ref, (double)bad_got);
        return 1;
    }
}

/* ------------------------------------------------------------------ */
/* Standalone algebraic tests (run once with the current OMP env)      */
/* ------------------------------------------------------------------ */

static int test_zero(size_t m, size_t n)
{
    scalar_t *A_row    = xalloc_aligned(m * m);
    scalar_t *A_morton = xalloc_aligned(m * m);
    scalar_t *Z        = xalloc_aligned(m * n);
    scalar_t *C        = xalloc_aligned(m * n);
    scalar_t *expected = xalloc_aligned(m * n);

    init_matrix_random(A_row, m, m, 11u);
    reorganize_to_morton_blocks(A_row, A_morton, m);
    init_matrix_zero(Z, m, n);
    init_matrix_zero(expected, m, n);

    matmul_morton_omp(C, A_morton, Z, m, m, n);
    int rc = check_or_report("A * 0 == 0", expected, C, m * n);

    xfree(A_row); xfree(A_morton); xfree(Z); xfree(C); xfree(expected);
    return rc;
}

static int test_identity(size_t m, size_t n)
{
    scalar_t *I_row    = xalloc_aligned(m * m);
    scalar_t *I_morton = xalloc_aligned(m * m);
    scalar_t *Z        = xalloc_aligned(m * n);
    scalar_t *C        = xalloc_aligned(m * n);

    init_matrix_identity(I_row, m);
    reorganize_to_morton_blocks(I_row, I_morton, m);
    init_matrix_random(Z, m, n, 22u);

    matmul_morton_omp(C, I_morton, Z, m, m, n);
    int rc = check_or_report("I * Z == Z", Z, C, m * n);

    xfree(I_row); xfree(I_morton); xfree(Z); xfree(C);
    return rc;
}

static int test_linearity(size_t m, size_t n)
{
    scalar_t *A_row    = xalloc_aligned(m * m);
    scalar_t *A_morton = xalloc_aligned(m * m);
    scalar_t *Z1  = xalloc_aligned(m * n);
    scalar_t *Z2  = xalloc_aligned(m * n);
    scalar_t *Zs  = xalloc_aligned(m * n);
    scalar_t *C1  = xalloc_aligned(m * n);
    scalar_t *C2  = xalloc_aligned(m * n);
    scalar_t *Cs  = xalloc_aligned(m * n);
    scalar_t *sum = xalloc_aligned(m * n);

    init_matrix_random(A_row, m, m, 31u);
    init_matrix_random(Z1,    m, n, 32u);
    init_matrix_random(Z2,    m, n, 33u);
    reorganize_to_morton_blocks(A_row, A_morton, m);

    for (size_t i = 0; i < m * n; ++i) Zs[i] = Z1[i] + Z2[i];

    matmul_morton_omp(C1, A_morton, Z1, m, m, n);
    matmul_morton_omp(C2, A_morton, Z2, m, m, n);
    matmul_morton_omp(Cs, A_morton, Zs, m, m, n);

    for (size_t i = 0; i < m * n; ++i) sum[i] = C1[i] + C2[i];

    int rc = check_or_report("A * (Z1+Z2) == A*Z1 + A*Z2", sum, Cs, m * n);

    xfree(A_row); xfree(A_morton);
    xfree(Z1); xfree(Z2); xfree(Zs);
    xfree(C1); xfree(C2); xfree(Cs); xfree(sum);
    return rc;
}

/* ------------------------------------------------------------------ */
/* Cross-validation tests (parameterized by thread count for the sweep)*/
/* ------------------------------------------------------------------ */

static int test_cross_naive_one(size_t m, size_t k, size_t n,
                                int n_threads, const char *tag)
{
    scalar_t *A_row    = xalloc_aligned(m * k);
    scalar_t *A_morton = xalloc_aligned(m * k);
    scalar_t *B        = xalloc_aligned(k * n);
    scalar_t *C_naive  = xalloc_aligned(m * n);
    scalar_t *C_omp    = xalloc_aligned(m * n);

    init_matrix_random(A_row, m, k, 51u + (unsigned int)m);
    init_matrix_random(B,     k, n, 71u + (unsigned int)m);
    reorganize_to_morton_blocks(A_row, A_morton, m);

    matmul_naive     (C_naive, A_row,    B, m, k, n);
    matmul_morton_omp(C_omp,   A_morton, B, m, k, n);

    char label[120];
    snprintf(label, sizeof(label),
             "%s morton_omp == naive (m=%llu, k=%llu, n=%llu, T=%d)",
             tag,
             (unsigned long long)m,
             (unsigned long long)k,
             (unsigned long long)n,
             n_threads);

    int rc = check_or_report(label, C_naive, C_omp, m * n);

    xfree(A_row); xfree(A_morton);
    xfree(B); xfree(C_naive); xfree(C_omp);
    return rc;
}

static int test_cross_avx2_one(size_t m, size_t k, size_t n,
                               int n_threads, const char *tag)
{
    scalar_t *A_row    = xalloc_aligned(m * k);
    scalar_t *A_morton = xalloc_aligned(m * k);
    scalar_t *B        = xalloc_aligned(k * n);
    scalar_t *C_avx2   = xalloc_aligned(m * n);
    scalar_t *C_omp    = xalloc_aligned(m * n);

    init_matrix_random(A_row, m, k, 91u + (unsigned int)m);
    init_matrix_random(B,     k, n, 113u + (unsigned int)m);
    reorganize_to_morton_blocks(A_row, A_morton, m);

    matmul_morton_avx2(C_avx2, A_morton, B, m, k, n);
    matmul_morton_omp (C_omp,  A_morton, B, m, k, n);

    char label[120];
    snprintf(label, sizeof(label),
             "%s morton_omp == morton_avx2 (m=%llu, k=%llu, n=%llu, T=%d)",
             tag,
             (unsigned long long)m,
             (unsigned long long)k,
             (unsigned long long)n,
             n_threads);

    int rc = check_or_report(label, C_avx2, C_omp, m * n);

    xfree(A_row); xfree(A_morton);
    xfree(B); xfree(C_omp); xfree(C_avx2);
    return rc;
}

static int test_cross_sweep_at_threads(int n_threads, const char *tag)
{
    static const size_t M_LIST[] = { 4u, 16u, 64u, 256u };
    static const size_t NUM_M = sizeof(M_LIST) / sizeof(M_LIST[0]);

    omp_set_num_threads(n_threads);

    int failures = 0;
    for (size_t i = 0; i < NUM_M; ++i) {
        size_t m = M_LIST[i];
        size_t k = m;
        size_t n = (size_t)BLOCK_N;
        failures += test_cross_naive_one(m, k, n, n_threads, tag);
        failures += test_cross_avx2_one (m, k, n, n_threads, tag);
    }
    return failures;
}

int main(int argc, char **argv)
{
    size_t m = (size_t)DEFAULT_M;
    if (argc >= 2) {
        long long m_in = atoll(argv[1]);
        if (m_in <= 0) {
            fprintf(stderr, "Error: m must be a positive integer.\n");
            return EXIT_FAILURE;
        }
        m = (size_t)m_in;
    }
    if (!is_power_of_two(m) || m < (size_t)MORTON_AVX2_TILE) {
        fprintf(stderr,
                "Error: m (%llu) must be a power of two and >= %d.\n",
                (unsigned long long)m, MORTON_AVX2_TILE);
        return EXIT_FAILURE;
    }

    size_t n = (m < BLOCK_N) ? m : BLOCK_N;

    printf("Validating matmul_morton_omp at m=%llu, n=%llu\n",
           (unsigned long long)m, (unsigned long long)n);
    printf("Tolerances: abs=%.1e, rel=%.1e\n",
           (double)ABS_TOL, (double)REL_TOL);
    printf("Initial OMP_NUM_THREADS = %d (env), max_threads = %d\n",
           omp_get_max_threads(), omp_get_max_threads());

    int failures = 0;

    /* Standalone algebraic tests at the env-provided thread count. */
    printf("\n[env] standalone algebraic tests\n");
    failures += test_zero(m, n);
    failures += test_identity(m, n);
    failures += test_linearity(m, n);

    /* Cross-validation sweep over thread counts. The list intentionally
     * crosses the 4600H's 6-core boundary to exercise SMT scheduling. */
    static const int THREADS[] = { 1, 4, 12 };
    static const size_t NUM_T = sizeof(THREADS) / sizeof(THREADS[0]);
    for (size_t i = 0; i < NUM_T; ++i) {
        char tag[16];
        snprintf(tag, sizeof(tag), "[T=%d]", THREADS[i]);
        printf("\n%s cross-validation sweep\n", tag);
        failures += test_cross_sweep_at_threads(THREADS[i], tag);
    }

    printf("\n");
    if (failures == 0) {
        printf("VALIDATION OK\n");
        return EXIT_SUCCESS;
    } else {
        printf("VALIDATION FAILED (%d test(s))\n", failures);
        return EXIT_FAILURE;
    }
}
