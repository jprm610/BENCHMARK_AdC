/*
 * validate.c - Sanity checks for the naive matmul kernel.
 *
 * Three algebraic invariants are tested:
 *   1. A * 0 = 0
 *   2. I * Z = Z (identity times anything is anything)
 *   3. A * (Z1 + Z2) = A * Z1 + A * Z2 (linearity)
 *
 * Any kernel that fails one of these has a bug. As more kernel variants
 * are added, this driver can be extended to call them under the same
 * checks; for now it only exercises matmul_naive.
 *
 * Usage:
 *   validate_O0 [m]
 *
 * Exit code is 0 on success, 1 on any failure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul_naive.h"
#include "matrix_utils.h"

#define DEFAULT_M 256u
#define BLOCK_N   128u

/* Tolerances calibrated for float over a few hundred summed products.
 * See docs/API.md section 3.6 for the rationale. */
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

/* Test 1: A * 0 = 0. */
static int test_zero(size_t m, size_t n)
{
    scalar_t *A = xalloc_aligned(m * m);
    scalar_t *Z = xalloc_aligned(m * n);
    scalar_t *C = xalloc_aligned(m * n);
    scalar_t *expected = xalloc_aligned(m * n);

    init_matrix_random(A, m, m, 11u);
    init_matrix_zero(Z, m, n);
    init_matrix_zero(expected, m, n);

    matmul_naive(C, A, Z, m, m, n);
    int rc = check_or_report("A * 0 == 0", expected, C, m * n);

    xfree(A); xfree(Z); xfree(C); xfree(expected);
    return rc;
}

/* Test 2: I * Z = Z, where I is the m x m identity. */
static int test_identity(size_t m, size_t n)
{
    scalar_t *I = xalloc_aligned(m * m);
    scalar_t *Z = xalloc_aligned(m * n);
    scalar_t *C = xalloc_aligned(m * n);

    init_matrix_identity(I, m);
    init_matrix_random(Z, m, n, 22u);
    matmul_naive(C, I, Z, m, m, n);

    int rc = check_or_report("I * Z == Z", Z, C, m * n);

    xfree(I); xfree(Z); xfree(C);
    return rc;
}

/* Test 3: A * (Z1 + Z2) = A * Z1 + A * Z2. */
static int test_linearity(size_t m, size_t n)
{
    scalar_t *A   = xalloc_aligned(m * m);
    scalar_t *Z1  = xalloc_aligned(m * n);
    scalar_t *Z2  = xalloc_aligned(m * n);
    scalar_t *Zs  = xalloc_aligned(m * n);  /* Z1 + Z2 */
    scalar_t *C1  = xalloc_aligned(m * n);  /* A * Z1 */
    scalar_t *C2  = xalloc_aligned(m * n);  /* A * Z2 */
    scalar_t *Cs  = xalloc_aligned(m * n);  /* A * (Z1+Z2) */
    scalar_t *sum = xalloc_aligned(m * n);  /* C1 + C2 */

    init_matrix_random(A,  m, m, 31u);
    init_matrix_random(Z1, m, n, 32u);
    init_matrix_random(Z2, m, n, 33u);

    for (size_t i = 0; i < m * n; ++i) Zs[i] = Z1[i] + Z2[i];

    matmul_naive(C1, A, Z1, m, m, n);
    matmul_naive(C2, A, Z2, m, m, n);
    matmul_naive(Cs, A, Zs, m, m, n);

    for (size_t i = 0; i < m * n; ++i) sum[i] = C1[i] + C2[i];

    int rc = check_or_report("A * (Z1+Z2) == A*Z1 + A*Z2", sum, Cs, m * n);

    xfree(A); xfree(Z1); xfree(Z2); xfree(Zs);
    xfree(C1); xfree(C2); xfree(Cs); xfree(sum);
    return rc;
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
    /* Use a moderate n: smaller than the benchmark default to keep tests
     * fast while still exercising the kernel with realistic shapes. */
    size_t n = (m < BLOCK_N) ? m : BLOCK_N;

    printf("Validating matmul_naive at m=%llu, n=%llu\n",
           (unsigned long long)m, (unsigned long long)n);
    printf("Tolerances: abs=%.1e, rel=%.1e\n",
           (double)ABS_TOL, (double)REL_TOL);

    int failures = 0;
    failures += test_zero(m, n);
    failures += test_identity(m, n);
    failures += test_linearity(m, n);

    if (failures == 0) {
        printf("VALIDATION OK\n");
        return EXIT_SUCCESS;
    } else {
        printf("VALIDATION FAILED (%d test(s))\n", failures);
        return EXIT_FAILURE;
    }
}
