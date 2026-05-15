/*
 * validate_morton.c - Sanity checks for the recursive Morton matmul kernel.
 *
 * Five test groups exercise matmul_morton:
 *   1. A * 0 = 0.
 *   2. I * Z = Z.
 *   3. A * (Z1 + Z2) = A * Z1 + A * Z2 (linearity).
 *   4. Cross-validation against matmul_naive on random data.
 *   5. Cross-validation against matmul_recursive on random data.
 *
 * Test 5 is the strongest because both kernels are recursive: they apply
 * the same recurrence to A and B (but on different layouts), so any bug
 * in the propagation of a_morton_offset, in the quadrant order TL/TR/BL/BR,
 * or in the local/global index handling of kernel_base_morton would show
 * up as a divergence.
 *
 * Usage:
 *   validate_morton_O0 [m]
 *
 * The CLI m controls the size used by the three invariant tests; it must
 * be a power of two. The cross-validation sweep (Tests 4 and 5) always
 * runs over m in {4, 16, 64, 256}.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul_naive.h"
#include "matmul_recursive.h"
#include "matmul_morton.h"
#include "morton.h"
#include "matrix_utils.h"

#define DEFAULT_M 256u
#define BLOCK_N   128u

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

/* Test 1: A * 0 = 0. */
static int test_zero(size_t m, size_t n)
{
    scalar_t *A_row    = xalloc_aligned(m * m);
    scalar_t *A_morton = xalloc_aligned(m * m);
    scalar_t *Z        = xalloc_aligned(m * n);
    scalar_t *C        = xalloc_aligned(m * n);
    scalar_t *expected = xalloc_aligned(m * n);

    init_matrix_random(A_row, m, m, 11u);
    reorganize_to_morton(A_row, A_morton, m);
    init_matrix_zero(Z, m, n);
    init_matrix_zero(expected, m, n);

    matmul_morton(C, A_morton, Z, m, m, n);
    int rc = check_or_report("A * 0 == 0", expected, C, m * n);

    xfree(A_row); xfree(A_morton); xfree(Z); xfree(C); xfree(expected);
    return rc;
}

/* Test 2: I * Z = Z. */
static int test_identity(size_t m, size_t n)
{
    scalar_t *I_row    = xalloc_aligned(m * m);
    scalar_t *I_morton = xalloc_aligned(m * m);
    scalar_t *Z        = xalloc_aligned(m * n);
    scalar_t *C        = xalloc_aligned(m * n);

    init_matrix_identity(I_row, m);
    reorganize_to_morton(I_row, I_morton, m);
    init_matrix_random(Z, m, n, 22u);

    matmul_morton(C, I_morton, Z, m, m, n);
    int rc = check_or_report("I * Z == Z", Z, C, m * n);

    xfree(I_row); xfree(I_morton); xfree(Z); xfree(C);
    return rc;
}

/* Test 3: A * (Z1 + Z2) = A * Z1 + A * Z2. */
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
    reorganize_to_morton(A_row, A_morton, m);

    for (size_t i = 0; i < m * n; ++i) Zs[i] = Z1[i] + Z2[i];

    matmul_morton(C1, A_morton, Z1, m, m, n);
    matmul_morton(C2, A_morton, Z2, m, m, n);
    matmul_morton(Cs, A_morton, Zs, m, m, n);

    for (size_t i = 0; i < m * n; ++i) sum[i] = C1[i] + C2[i];

    int rc = check_or_report("A * (Z1+Z2) == A*Z1 + A*Z2", sum, Cs, m * n);

    xfree(A_row); xfree(A_morton);
    xfree(Z1); xfree(Z2); xfree(Zs);
    xfree(C1); xfree(C2); xfree(Cs); xfree(sum);
    return rc;
}

/* Test 4 helper: morton(C, A, B) == naive(C, A, B) for one shape. */
static int test_cross_naive_one(size_t m, size_t k, size_t n)
{
    scalar_t *A_row    = xalloc_aligned(m * k);
    scalar_t *A_morton = xalloc_aligned(m * k);
    scalar_t *B        = xalloc_aligned(k * n);
    scalar_t *C_naive  = xalloc_aligned(m * n);
    scalar_t *C_morton = xalloc_aligned(m * n);

    init_matrix_random(A_row, m, k, 51u + (unsigned int)m);
    init_matrix_random(B,     k, n, 71u + (unsigned int)m);
    reorganize_to_morton(A_row, A_morton, m);  /* requires m == k */

    matmul_naive (C_naive,  A_row,    B, m, k, n);
    matmul_morton(C_morton, A_morton, B, m, k, n);

    char label[96];
    snprintf(label, sizeof(label),
             "morton == naive (m=%llu, k=%llu, n=%llu)",
             (unsigned long long)m,
             (unsigned long long)k,
             (unsigned long long)n);

    int rc = check_or_report(label, C_naive, C_morton, m * n);

    xfree(A_row); xfree(A_morton);
    xfree(B); xfree(C_naive); xfree(C_morton);
    return rc;
}

/* Test 5 helper: morton(C, A, B) == recursive(C, A, B) for one shape. */
static int test_cross_recursive_one(size_t m, size_t k, size_t n)
{
    scalar_t *A_row    = xalloc_aligned(m * k);
    scalar_t *A_morton = xalloc_aligned(m * k);
    scalar_t *B        = xalloc_aligned(k * n);
    scalar_t *C_rec    = xalloc_aligned(m * n);
    scalar_t *C_morton = xalloc_aligned(m * n);

    /* Different seeds from Test 4 so the two tests probe independent
     * random inputs, in case one of them happens to hide a sign bug. */
    init_matrix_random(A_row, m, k, 91u + (unsigned int)m);
    init_matrix_random(B,     k, n, 113u + (unsigned int)m);
    reorganize_to_morton(A_row, A_morton, m);

    matmul_recursive(C_rec,    A_row,    B, m, k, n);
    matmul_morton   (C_morton, A_morton, B, m, k, n);

    char label[96];
    snprintf(label, sizeof(label),
             "morton == recursive (m=%llu, k=%llu, n=%llu)",
             (unsigned long long)m,
             (unsigned long long)k,
             (unsigned long long)n);

    int rc = check_or_report(label, C_rec, C_morton, m * n);

    xfree(A_row); xfree(A_morton);
    xfree(B); xfree(C_rec); xfree(C_morton);
    return rc;
}

/* Sweep over m in {4, 16, 64, 256} with k = m and n = 128. */
static int test_cross_sweep(int (*test_one)(size_t, size_t, size_t))
{
    static const size_t M_LIST[] = { 4u, 16u, 64u, 256u };
    static const size_t NUM_M = sizeof(M_LIST) / sizeof(M_LIST[0]);

    int failures = 0;
    for (size_t i = 0; i < NUM_M; ++i) {
        size_t m = M_LIST[i];
        size_t k = m;
        size_t n = (size_t)BLOCK_N;
        failures += test_one(m, k, n);
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
    if (!is_power_of_two(m)) {
        fprintf(stderr,
                "Error: m (%llu) must be a power of two for the Morton kernel.\n",
                (unsigned long long)m);
        return EXIT_FAILURE;
    }

    size_t n = (m < BLOCK_N) ? m : BLOCK_N;

    printf("Validating matmul_morton at m=%llu, n=%llu\n",
           (unsigned long long)m, (unsigned long long)n);
    printf("Tolerances: abs=%.1e, rel=%.1e\n",
           (double)ABS_TOL, (double)REL_TOL);

    int failures = 0;
    failures += test_zero(m, n);
    failures += test_identity(m, n);
    failures += test_linearity(m, n);
    failures += test_cross_sweep(test_cross_naive_one);
    failures += test_cross_sweep(test_cross_recursive_one);

    if (failures == 0) {
        printf("VALIDATION OK\n");
        return EXIT_SUCCESS;
    } else {
        printf("VALIDATION FAILED (%d test(s))\n", failures);
        return EXIT_FAILURE;
    }
}
