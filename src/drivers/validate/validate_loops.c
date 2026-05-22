/*
 * validate_loops.c - Algebraic sanity checks for all six loop-order kernels.
 *
 * For each variant (ijk, ikj, jik, jki, kij, kji) runs three invariants:
 *   1. A * 0 == 0
 *   2. I * Z == Z   (identity matrix times Z)
 *   3. A * (Z1 + Z2) == A*Z1 + A*Z2
 *
 * Also cross-validates each variant against matmul_naive for a random
 * (A, B) pair to catch sign or index errors.
 *
 * Usage:
 *   validate_loops_O0 [m]     (default m = 256)
 *
 * Exits 0 if all tests pass, 1 on first failure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul_loops.h"
#include "matmul_naive.h"
#include "matrix_utils.h"

#define ABS_TOL  1e-4f
#define REL_TOL  1e-3f

static const char *ORDERS[] = { "ijk", "ikj", "jik", "jki", "kij", "kji" };
static const int   N_ORDERS = 6;

static int check(const char *label,
                 const scalar_t *ref, const scalar_t *test,
                 size_t num_elements)
{
    size_t bad;
    scalar_t br, bt;
    int ok = matrices_close(ref, test, num_elements,
                            ABS_TOL, REL_TOL, &bad, &br, &bt);
    if (ok) {
        printf("  [OK]   %s\n", label);
    } else {
        printf("  [FAIL] %s  (first bad index %llu: ref=%.6g test=%.6g)\n",
               label, (unsigned long long)bad, (double)br, (double)bt);
    }
    return ok;
}

int main(int argc, char **argv)
{
    size_t m = 256;
    if (argc >= 2) {
        long long m_in = atoll(argv[1]);
        if (m_in <= 0) {
            fprintf(stderr, "Error: m must be a positive integer.\n");
            return EXIT_FAILURE;
        }
        m = (size_t)m_in;
    }
    size_t n = m;   /* square for simplicity: A is m x m, B is m x m */
    size_t k = m;

    printf("Validating loop-order kernels at m=%llu\n",
           (unsigned long long)m);
    printf("Tolerances: abs=%.1e, rel=%.1e\n", (double)ABS_TOL, (double)REL_TOL);

    /* Shared scratch buffers. */
    scalar_t *A    = xalloc_aligned(m * k);
    scalar_t *Z    = xalloc_aligned(k * n);
    scalar_t *Z1   = xalloc_aligned(k * n);
    scalar_t *Z2   = xalloc_aligned(k * n);
    scalar_t *Zsum = xalloc_aligned(k * n);
    scalar_t *I_m  = xalloc_aligned(m * k);   /* identity */
    scalar_t *C    = xalloc_aligned(m * n);
    scalar_t *C2   = xalloc_aligned(m * n);
    scalar_t *Cref = xalloc_aligned(m * n);
    scalar_t *Ctmp = xalloc_aligned(m * n);

    init_matrix_random(A, m, k, 42u);
    init_matrix_random(Z,  k, n, 43u);
    init_matrix_random(Z1, k, n, 44u);
    init_matrix_random(Z2, k, n, 45u);
    init_matrix_identity(I_m, m);

    /* Z1 + Z2 */
    for (size_t idx = 0; idx < k * n; ++idx)
        Zsum[idx] = Z1[idx] + Z2[idx];

    int all_ok = 1;

    for (int o = 0; o < N_ORDERS; ++o) {
        const char *name = ORDERS[o];
        matmul_fn_t fn   = matmul_loops_lookup(name);
        printf("\n--- %s ---\n", name);

        /* Test 1: A * 0 == 0 */
        scalar_t *zero = xalloc_aligned(k * n);
        init_matrix_zero(zero, k, n);
        scalar_t *expected_zero = xalloc_aligned(m * n);
        init_matrix_zero(expected_zero, m, n);
        fn(C, A, zero, m, k, n);
        char label[64];
        snprintf(label, sizeof(label), "(%s)  A * 0 == 0", name);
        all_ok &= check(label, expected_zero, C, m * n);
        xfree(zero);
        xfree(expected_zero);

        /* Test 2: I * Z == Z */
        fn(C, I_m, Z, m, k, n);
        snprintf(label, sizeof(label), "(%s)  I * Z == Z", name);
        all_ok &= check(label, Z, C, m * n);

        /* Test 3: A * (Z1 + Z2) == A*Z1 + A*Z2 */
        fn(C,   A, Zsum, m, k, n);       /* A * (Z1+Z2) */
        fn(C2,  A, Z1,   m, k, n);
        fn(Ctmp,A, Z2,   m, k, n);
        for (size_t idx = 0; idx < m * n; ++idx)
            C2[idx] += Ctmp[idx];         /* A*Z1 + A*Z2 */
        snprintf(label, sizeof(label), "(%s)  A*(Z1+Z2) == A*Z1+A*Z2", name);
        all_ok &= check(label, C2, C, m * n);

        /* Test 4: cross-validate against matmul_naive */
        matmul_naive(Cref, A, Z, m, k, n);
        fn(C, A, Z, m, k, n);
        snprintf(label, sizeof(label), "(%s)  matches matmul_naive", name);
        all_ok &= check(label, Cref, C, m * n);
    }

    xfree(A); xfree(Z); xfree(Z1); xfree(Z2); xfree(Zsum);
    xfree(I_m); xfree(C); xfree(C2); xfree(Cref); xfree(Ctmp);

    printf("\n%s\n", all_ok ? "VALIDATION OK" : "VALIDATION FAILED");
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
