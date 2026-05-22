/*
 * validate_morton_avx2.c - Sanity checks for matmul_morton_avx2.
 *
 * Same pattern as validate_morton.c (Sesion 02) but with three
 * differences:
 *
 *   1. The Morton layout used here is Morton-of-blocks with tile=4
 *      (see matmul_morton_avx2.h). A is produced with
 *      reorganize_to_morton_blocks, not reorganize_to_morton.
 *
 *   2. The cross-validation reference is matmul_naive (ground truth)
 *      and matmul_morton (Sesion 02 kernel; uses Morton-of-elements
 *      internally with the OTHER reorganization function). The two
 *      Morton variants must agree on the algebraic result, even
 *      though they index A through totally different physical
 *      layouts.
 *
 *   3. Tolerances are loosened from 1e-5 abs / 1e-4 rel to 1e-4 abs /
 *      1e-3 rel because the microkernel is compiled with
 *      -ffast-math + -funroll-loops, which lets GCC reorder the FMA
 *      chain and introduces more accumulation noise than the strict
 *      ijk reference.
 *
 * Tests:
 *   1. A * 0 = 0.
 *   2. I * Z = Z.
 *   3. A * (Z1 + Z2) = A*Z1 + A*Z2 (linearity).
 *   4. matmul_morton_avx2 == matmul_naive  on m in {4, 16, 64, 256}.
 *   5. matmul_morton_avx2 == matmul_morton on m in {4, 16, 64, 256}.
 *
 * Usage:
 *   validate_morton_avx2_O3 [m]
 *
 * m must be a power of two and >= MORTON_AVX2_TILE = 4. Default: 256.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul_naive.h"
#include "matmul_morton.h"
#include "matmul_morton_avx2.h"
#include "morton.h"
#include "matrix_utils.h"

#define DEFAULT_M 256u
#define BLOCK_N   128u

/* Looser tolerances than validate_morton: -ffast-math in the AVX2
 * microkernel allows reassociation of FP additions, so the
 * acculumated rounding can diverge from a strict ijk reference. */
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

    matmul_morton_avx2(C, A_morton, Z, m, m, n);
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

    matmul_morton_avx2(C, I_morton, Z, m, m, n);
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

    matmul_morton_avx2(C1, A_morton, Z1, m, m, n);
    matmul_morton_avx2(C2, A_morton, Z2, m, m, n);
    matmul_morton_avx2(Cs, A_morton, Zs, m, m, n);

    for (size_t i = 0; i < m * n; ++i) sum[i] = C1[i] + C2[i];

    int rc = check_or_report("A * (Z1+Z2) == A*Z1 + A*Z2", sum, Cs, m * n);

    xfree(A_row); xfree(A_morton);
    xfree(Z1); xfree(Z2); xfree(Zs);
    xfree(C1); xfree(C2); xfree(Cs); xfree(sum);
    return rc;
}

/* Test 4: AVX2 result must match the naive ground truth. */
static int test_cross_naive_one(size_t m, size_t k, size_t n)
{
    scalar_t *A_row    = xalloc_aligned(m * k);
    scalar_t *A_morton = xalloc_aligned(m * k);
    scalar_t *B        = xalloc_aligned(k * n);
    scalar_t *C_naive  = xalloc_aligned(m * n);
    scalar_t *C_avx2   = xalloc_aligned(m * n);

    init_matrix_random(A_row, m, k, 51u + (unsigned int)m);
    init_matrix_random(B,     k, n, 71u + (unsigned int)m);
    reorganize_to_morton_blocks(A_row, A_morton, m);

    matmul_naive       (C_naive, A_row,    B, m, k, n);
    matmul_morton_avx2 (C_avx2,  A_morton, B, m, k, n);

    char label[96];
    snprintf(label, sizeof(label),
             "morton_avx2 == naive (m=%llu, k=%llu, n=%llu)",
             (unsigned long long)m,
             (unsigned long long)k,
             (unsigned long long)n);

    int rc = check_or_report(label, C_naive, C_avx2, m * n);

    xfree(A_row); xfree(A_morton);
    xfree(B); xfree(C_naive); xfree(C_avx2);
    return rc;
}

/* Test 5: AVX2 result must match the Sesion 02 Morton kernel. Both
 * are recursive, but they consume DIFFERENT physical layouts of A
 * (Morton-of-elements vs Morton-of-blocks). Any bug in the offset
 * arithmetic of either side, or in either reorganization function,
 * surfaces here. */
static int test_cross_morton_one(size_t m, size_t k, size_t n)
{
    scalar_t *A_row           = xalloc_aligned(m * k);
    scalar_t *A_morton_elem   = xalloc_aligned(m * k);
    scalar_t *A_morton_blocks = xalloc_aligned(m * k);
    scalar_t *B               = xalloc_aligned(k * n);
    scalar_t *C_morton        = xalloc_aligned(m * n);
    scalar_t *C_avx2          = xalloc_aligned(m * n);

    init_matrix_random(A_row, m, k, 91u + (unsigned int)m);
    init_matrix_random(B,     k, n, 113u + (unsigned int)m);
    reorganize_to_morton       (A_row, A_morton_elem,   m);
    reorganize_to_morton_blocks(A_row, A_morton_blocks, m);

    matmul_morton      (C_morton, A_morton_elem,   B, m, k, n);
    matmul_morton_avx2 (C_avx2,   A_morton_blocks, B, m, k, n);

    char label[96];
    snprintf(label, sizeof(label),
             "morton_avx2 == morton (m=%llu, k=%llu, n=%llu)",
             (unsigned long long)m,
             (unsigned long long)k,
             (unsigned long long)n);

    int rc = check_or_report(label, C_morton, C_avx2, m * n);

    xfree(A_row);
    xfree(A_morton_elem); xfree(A_morton_blocks);
    xfree(B); xfree(C_morton); xfree(C_avx2);
    return rc;
}

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
    if (!is_power_of_two(m) || m < (size_t)MORTON_AVX2_TILE) {
        fprintf(stderr,
                "Error: m (%llu) must be a power of two and >= %d.\n",
                (unsigned long long)m, MORTON_AVX2_TILE);
        return EXIT_FAILURE;
    }

    size_t n = (m < BLOCK_N) ? m : BLOCK_N;

    printf("Validating matmul_morton_avx2 at m=%llu, n=%llu\n",
           (unsigned long long)m, (unsigned long long)n);
    printf("Tolerances: abs=%.1e, rel=%.1e\n",
           (double)ABS_TOL, (double)REL_TOL);

    int failures = 0;
    failures += test_zero(m, n);
    failures += test_identity(m, n);
    failures += test_linearity(m, n);
    failures += test_cross_sweep(test_cross_naive_one);
    failures += test_cross_sweep(test_cross_morton_one);

    if (failures == 0) {
        printf("VALIDATION OK\n");
        return EXIT_SUCCESS;
    } else {
        printf("VALIDATION FAILED (%d test(s))\n", failures);
        return EXIT_FAILURE;
    }
}
