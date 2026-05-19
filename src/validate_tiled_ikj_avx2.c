/*
 * validate_tiled_ikj_avx2.c - Algebraic sanity checks for matmul_tiled_ikj_avx2.
 *
 * Same four tests as validate_tiled_ikj.c:
 *   1. A * 0 == 0
 *   2. I * Z == Z  (identity matrix)
 *   3. A * (Z1 + Z2) == A*Z1 + A*Z2  (linearity)
 *   4. matmul_tiled_ikj_avx2 matches matmul_naive for a random (A, B) pair.
 *
 * Compiled with CFLAGS_O3_ZEN2 (requires -mavx2 -mfma for the intrinsics
 * in matmul_tiled_ikj_avx2.c). Tolerances match validate_tiled_ikj because
 * CFLAGS_O3_ZEN2 does not include -ffast-math and the FMA rounding is
 * deterministic for a fixed instruction sequence.
 *
 * Usage:
 *   validate_tiled_ikj_avx2_O3 [m] [bs]
 *     m  : problem size, default 256
 *     bs : block size passed to matmul_tiled_ikj_avx2_set_bs, default 64
 *
 * Exits 0 if all tests pass, 1 on first failure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul_tiled_ikj_avx2.h"
#include "matmul_naive.h"
#include "matrix_utils.h"

#define ABS_TOL  1e-4f
#define REL_TOL  1e-3f

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
    size_t m  = 256;
    size_t bs = TILED_IKJ_AVX2_BS_DEFAULT;

    if (argc >= 2) {
        long long m_in = atoll(argv[1]);
        if (m_in <= 0) {
            fprintf(stderr, "Error: m must be a positive integer.\n");
            return EXIT_FAILURE;
        }
        m = (size_t)m_in;
    }
    if (argc >= 3) {
        long long bs_in = atoll(argv[2]);
        if (bs_in <= 0) {
            fprintf(stderr, "Error: bs must be a positive integer.\n");
            return EXIT_FAILURE;
        }
        bs = (size_t)bs_in;
    }

    matmul_tiled_ikj_avx2_set_bs(bs);

    size_t n = m;
    size_t k = m;

    printf("Validating matmul_tiled_ikj_avx2 at m=%llu (bs=%llu)\n",
           (unsigned long long)m, (unsigned long long)g_tiled_ikj_avx2_bs);
    printf("Tolerances: abs=%.1e, rel=%.1e\n", (double)ABS_TOL, (double)REL_TOL);

    scalar_t *A    = xalloc_aligned(m * k);
    scalar_t *Z    = xalloc_aligned(k * n);
    scalar_t *Z1   = xalloc_aligned(k * n);
    scalar_t *Z2   = xalloc_aligned(k * n);
    scalar_t *Zsum = xalloc_aligned(k * n);
    scalar_t *I_m  = xalloc_aligned(m * k);
    scalar_t *C    = xalloc_aligned(m * n);
    scalar_t *C2   = xalloc_aligned(m * n);
    scalar_t *Cref = xalloc_aligned(m * n);
    scalar_t *Ctmp = xalloc_aligned(m * n);

    init_matrix_random(A,  m, k, 42u);
    init_matrix_random(Z,  k, n, 43u);
    init_matrix_random(Z1, k, n, 44u);
    init_matrix_random(Z2, k, n, 45u);
    init_matrix_identity(I_m, m);

    for (size_t idx = 0; idx < k * n; ++idx)
        Zsum[idx] = Z1[idx] + Z2[idx];

    int all_ok = 1;

    /* Test 1: A * 0 == 0 */
    scalar_t *zero          = xalloc_aligned(k * n);
    scalar_t *expected_zero = xalloc_aligned(m * n);
    init_matrix_zero(zero, k, n);
    init_matrix_zero(expected_zero, m, n);
    matmul_tiled_ikj_avx2(C, A, zero, m, k, n);
    all_ok &= check("A * 0 == 0", expected_zero, C, m * n);
    xfree(zero);
    xfree(expected_zero);

    /* Test 2: I * Z == Z */
    matmul_tiled_ikj_avx2(C, I_m, Z, m, k, n);
    all_ok &= check("I * Z == Z", Z, C, m * n);

    /* Test 3: A * (Z1 + Z2) == A*Z1 + A*Z2 */
    matmul_tiled_ikj_avx2(C,    A, Zsum, m, k, n);
    matmul_tiled_ikj_avx2(C2,   A, Z1,   m, k, n);
    matmul_tiled_ikj_avx2(Ctmp, A, Z2,   m, k, n);
    for (size_t idx = 0; idx < m * n; ++idx)
        C2[idx] += Ctmp[idx];
    all_ok &= check("A*(Z1+Z2) == A*Z1+A*Z2", C2, C, m * n);

    /* Test 4: matches matmul_naive */
    matmul_naive(Cref, A, Z, m, k, n);
    matmul_tiled_ikj_avx2(C, A, Z, m, k, n);
    all_ok &= check("matches matmul_naive", Cref, C, m * n);

    xfree(A); xfree(Z); xfree(Z1); xfree(Z2); xfree(Zsum);
    xfree(I_m); xfree(C); xfree(C2); xfree(Cref); xfree(Ctmp);

    printf("\n%s\n", all_ok ? "VALIDATION OK" : "VALIDATION FAILED");
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
