/*
 * validate_recursive.c - Sanity checks for the recursive matmul kernel.
 *
 * Three algebraic invariants (same as validate_naive) plus a fourth
 * cross-validation against matmul_naive on pseudo-random data. The
 * cross-validation is the strongest test: it catches reordering bugs in
 * the recursion that the algebraic invariants would miss (for example,
 * if the second half of the k-split incorrectly overwrote instead of
 * accumulating, A * I = A would still pass but A * B with general B
 * would not).
 *
 * Usage:
 *   validate_recursive_O0 [m]
 *
 * The CLI m controls the size used for the three invariant tests. The
 * cross-validation test (Test 4) always runs the fixed sweep
 * m in {4, 16, 64, 256} with k = m and n = 128.
 *
 * Exit code is 0 on success, 1 on any failure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul_naive.h"
#include "matmul_recursive.h"
#include "matrix_utils.h"

#define DEFAULT_M 256u
#define BLOCK_N   128u

/* Tolerances per Prompt 2: tighter than validate_naive's because the
 * cross-validation has stronger discriminative power. */
static const scalar_t ABS_TOL = (scalar_t)1.0e-5;
static const scalar_t REL_TOL = (scalar_t)1.0e-4;

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

/* Test 1: A * 0 = 0 via matmul_recursive. */
static int test_zero(size_t m, size_t n)
{
    scalar_t *A = xalloc_aligned(m * m);
    scalar_t *Z = xalloc_aligned(m * n);
    scalar_t *C = xalloc_aligned(m * n);
    scalar_t *expected = xalloc_aligned(m * n);

    init_matrix_random(A, m, m, 11u);
    init_matrix_zero(Z, m, n);
    init_matrix_zero(expected, m, n);

    matmul_recursive(C, A, Z, m, m, n);
    int rc = check_or_report("A * 0 == 0", expected, C, m * n);

    xfree(A); xfree(Z); xfree(C); xfree(expected);
    return rc;
}

/* Test 2: I * Z = Z via matmul_recursive. */
static int test_identity(size_t m, size_t n)
{
    scalar_t *I = xalloc_aligned(m * m);
    scalar_t *Z = xalloc_aligned(m * n);
    scalar_t *C = xalloc_aligned(m * n);

    init_matrix_identity(I, m);
    init_matrix_random(Z, m, n, 22u);
    matmul_recursive(C, I, Z, m, m, n);

    int rc = check_or_report("I * Z == Z", Z, C, m * n);

    xfree(I); xfree(Z); xfree(C);
    return rc;
}

/* Test 3: linearity A*(Z1+Z2) = A*Z1 + A*Z2 via matmul_recursive. */
static int test_linearity(size_t m, size_t n)
{
    scalar_t *A   = xalloc_aligned(m * m);
    scalar_t *Z1  = xalloc_aligned(m * n);
    scalar_t *Z2  = xalloc_aligned(m * n);
    scalar_t *Zs  = xalloc_aligned(m * n);
    scalar_t *C1  = xalloc_aligned(m * n);
    scalar_t *C2  = xalloc_aligned(m * n);
    scalar_t *Cs  = xalloc_aligned(m * n);
    scalar_t *sum = xalloc_aligned(m * n);

    init_matrix_random(A,  m, m, 31u);
    init_matrix_random(Z1, m, n, 32u);
    init_matrix_random(Z2, m, n, 33u);

    for (size_t i = 0; i < m * n; ++i) Zs[i] = Z1[i] + Z2[i];

    matmul_recursive(C1, A, Z1, m, m, n);
    matmul_recursive(C2, A, Z2, m, m, n);
    matmul_recursive(Cs, A, Zs, m, m, n);

    for (size_t i = 0; i < m * n; ++i) sum[i] = C1[i] + C2[i];

    int rc = check_or_report("A * (Z1+Z2) == A*Z1 + A*Z2", sum, Cs, m * n);

    xfree(A); xfree(Z1); xfree(Z2); xfree(Zs);
    xfree(C1); xfree(C2); xfree(Cs); xfree(sum);
    return rc;
}

/* Test 4 helper: cross-validate matmul_recursive against matmul_naive on
 * one (m, k, n) triple with reproducible random A and B. */
static int test_cross_one(size_t m, size_t k, size_t n)
{
    scalar_t *A = xalloc_aligned(m * k);
    scalar_t *B = xalloc_aligned(k * n);
    scalar_t *C_naive = xalloc_aligned(m * n);
    scalar_t *C_rec   = xalloc_aligned(m * n);

    /* Per-(m, k, n) seeds keep each sub-test independent and reproducible. */
    init_matrix_random(A, m, k, 51u + (unsigned int)m);
    init_matrix_random(B, k, n, 71u + (unsigned int)m);

    matmul_naive(C_naive, A, B, m, k, n);
    matmul_recursive(C_rec, A, B, m, k, n);

    char label[96];
    snprintf(label, sizeof(label),
             "recursive == naive (m=%llu, k=%llu, n=%llu)",
             (unsigned long long)m,
             (unsigned long long)k,
             (unsigned long long)n);

    int rc = check_or_report(label, C_naive, C_rec, m * n);

    xfree(A); xfree(B); xfree(C_naive); xfree(C_rec);
    return rc;
}

/* Test 4: cross-validate across the fixed sweep {4, 16, 64, 256}, all with
 * k = m and n = 128. */
static int test_cross_sweep(void)
{
    static const size_t M_LIST[] = { 4u, 16u, 64u, 256u };
    static const size_t NUM_M = sizeof(M_LIST) / sizeof(M_LIST[0]);

    int failures = 0;
    for (size_t i = 0; i < NUM_M; ++i) {
        size_t m = M_LIST[i];
        size_t k = m;
        size_t n = (size_t)BLOCK_N;
        failures += test_cross_one(m, k, n);
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
    size_t n = (m < BLOCK_N) ? m : BLOCK_N;

    printf("Validating matmul_recursive at m=%llu, n=%llu\n",
           (unsigned long long)m, (unsigned long long)n);
    printf("Tolerances: abs=%.1e, rel=%.1e\n",
           (double)ABS_TOL, (double)REL_TOL);

    int failures = 0;
    failures += test_zero(m, n);
    failures += test_identity(m, n);
    failures += test_linearity(m, n);
    failures += test_cross_sweep();

    if (failures == 0) {
        printf("VALIDATION OK\n");
        return EXIT_SUCCESS;
    } else {
        printf("VALIDATION FAILED (%d test(s))\n", failures);
        return EXIT_FAILURE;
    }
}
