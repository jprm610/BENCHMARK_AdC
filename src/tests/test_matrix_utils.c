/*
 * test_matrix_utils.c - Unit tests for the core building blocks in
 * src/core/matrix_utils.{c,h}. These helpers are used by every
 * benchmark, validation and unit test in the project, so a regression
 * here would silently corrupt every downstream measurement.
 *
 * The tests cover the public API documented in docs/API.md, Section 3:
 *
 *   1. xalloc_aligned    - returns non-NULL pointers aligned to 64
 *                          bytes (the cache-line size assumed by the
 *                          rest of the codebase) for several sizes.
 *   2. init_matrix_zero  - writes exactly zeros over the requested
 *                          range and nothing past it.
 *   3. init_matrix_identity - produces the identity matrix.
 *   4. init_matrix_random - is deterministic (same seed -> same data),
 *                           different seeds give different data, and
 *                           the output respects the documented bound
 *                           |M[i]| <= 1/sqrt(rows).
 *   5. matrices_close    - returns the correct verdict at the boundary
 *                          of the absolute and relative tolerances,
 *                          and populates the optional diagnostic
 *                          out-parameters only when they are non-NULL.
 *
 * Exits 0 and prints MATRIX_UTILS TESTS OK if every check passes.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix_utils.h"

/* Cache-line size assumed throughout the project. xalloc_aligned uses
 * aligned_alloc(64, ...). Keep this in sync with matrix_utils.c. */
#define EXPECTED_ALIGNMENT 64u

static int g_failed = 0;

static void run_check(const char *name, int passed)
{
    if (passed) {
        printf("  [OK]   %s\n", name);
    } else {
        printf("  [FAIL] %s\n", name);
        g_failed = 1;
    }
}

/* ---------- Test 1: xalloc_aligned -------------------------------- */

/* Allocates a handful of buffers of different sizes and verifies that
 * each one is non-NULL and aligned to EXPECTED_ALIGNMENT bytes. The
 * allocator is expected to abort on failure, so a NULL here would only
 * appear if someone weakened the contract. */
static void test_xalloc_aligned(void)
{
    static const size_t sizes[] = {
        1u, 8u, 64u, 4096u, 1u << 16, 1u << 20
    };
    static const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    int all_ok = 1;
    for (size_t i = 0; i < num_sizes; ++i) {
        scalar_t *p = xalloc_aligned(sizes[i]);
        uintptr_t addr = (uintptr_t)p;
        int aligned = (addr % EXPECTED_ALIGNMENT) == 0;
        if (p == NULL || !aligned) {
            fprintf(stderr,
                    "    n=%llu: ptr=%p, addr mod %u = %llu\n",
                    (unsigned long long)sizes[i],
                    (void *)p,
                    (unsigned)EXPECTED_ALIGNMENT,
                    (unsigned long long)(addr % EXPECTED_ALIGNMENT));
            all_ok = 0;
        }
        xfree(p);
    }
    run_check("xalloc_aligned: non-NULL and 64-byte aligned for several sizes",
              all_ok);
}

/* ---------- Test 2: init_matrix_zero ------------------------------ */

/* Fills a buffer with a non-zero sentinel pattern, then calls
 * init_matrix_zero over the inner rows*cols region and checks two
 * things: every element of the region is zero, and the trailing
 * sentinel bytes are untouched (no overrun). */
static void test_init_matrix_zero(void)
{
    const size_t rows = 7;
    const size_t cols = 5;
    const size_t guard = 16;             /* trailing sentinel cells */
    const size_t total = rows * cols + guard;

    scalar_t *M = xalloc_aligned(total);
    /* Sentinel chosen to be easy to spot in case of failure. */
    for (size_t i = 0; i < total; ++i) M[i] = (scalar_t)-7.5;

    init_matrix_zero(M, rows, cols);

    int zeros_ok = 1;
    for (size_t i = 0; i < rows * cols; ++i) {
        if (M[i] != (scalar_t)0) { zeros_ok = 0; break; }
    }
    int guard_ok = 1;
    for (size_t i = rows * cols; i < total; ++i) {
        if (M[i] != (scalar_t)-7.5) { guard_ok = 0; break; }
    }

    run_check("init_matrix_zero: every cell is 0 within range",  zeros_ok);
    run_check("init_matrix_zero: trailing sentinel is untouched", guard_ok);

    xfree(M);
}

/* ---------- Test 3: init_matrix_identity -------------------------- */

/* Builds I_n for n in {1, 4, 16, 33} and checks that the diagonal is
 * 1.0 and every off-diagonal entry is 0.0. Includes a non-power-of-two
 * to exercise the generic case. */
static int test_identity_one(size_t n)
{
    scalar_t *I = xalloc_aligned(n * n);
    init_matrix_identity(I, n);

    int ok = 1;
    for (size_t i = 0; i < n && ok; ++i) {
        for (size_t j = 0; j < n && ok; ++j) {
            scalar_t expected = (i == j) ? (scalar_t)1 : (scalar_t)0;
            if (I[i * n + j] != expected) ok = 0;
        }
    }

    xfree(I);
    return ok;
}

static void test_init_matrix_identity(void)
{
    static const size_t N_LIST[] = { 1u, 4u, 16u, 33u };
    static const size_t NUM_N = sizeof(N_LIST) / sizeof(N_LIST[0]);

    int all_ok = 1;
    for (size_t i = 0; i < NUM_N; ++i) {
        if (!test_identity_one(N_LIST[i])) all_ok = 0;
    }
    run_check("init_matrix_identity: diagonal=1, off-diagonal=0 for n in {1,4,16,33}",
              all_ok);
}

/* ---------- Test 4: init_matrix_random ---------------------------- */

/* Three sub-checks bundled together:
 *   (a) determinism: same seed must produce the same contents on a
 *       second call. The benchmark relies on this to keep runs
 *       reproducible across compilers and platforms.
 *   (b) seed sensitivity: at least one element must differ when the
 *       seed changes. Otherwise the seed is being ignored.
 *   (c) range: every element must lie in [-1/sqrt(rows), +1/sqrt(rows)]
 *       up to FP32 epsilon. The bound is what docs/API.md Section 3.3
 *       advertises and is what keeps the spectral norm of A under
 *       control during the recurrence B_{i+1} = A * B_i.
 *
 * 7 x 11 is non-square on purpose to exercise the row/col indexing of
 * the helper. */
static void test_init_matrix_random(void)
{
    const size_t rows = 7;
    const size_t cols = 11;
    const size_t total = rows * cols;
    const unsigned int seed_a = 42u;
    const unsigned int seed_b = 43u;

    scalar_t *A1 = xalloc_aligned(total);
    scalar_t *A2 = xalloc_aligned(total);
    scalar_t *B  = xalloc_aligned(total);

    init_matrix_random(A1, rows, cols, seed_a);
    init_matrix_random(A2, rows, cols, seed_a);
    init_matrix_random(B,  rows, cols, seed_b);

    int deterministic = (memcmp(A1, A2, total * sizeof(scalar_t)) == 0);
    run_check("init_matrix_random: deterministic for a fixed seed",
              deterministic);

    int seed_changes_output = 0;
    for (size_t i = 0; i < total; ++i) {
        if (A1[i] != B[i]) { seed_changes_output = 1; break; }
    }
    run_check("init_matrix_random: different seeds produce different data",
              seed_changes_output);

    /* The documented bound is |M[i]| <= 1/sqrt(rows). Add a tiny
     * epsilon to absorb the float rounding of the scale factor. */
    const scalar_t bound = (scalar_t)(1.0 / sqrt((double)rows)) * 1.0001f;
    int in_range = 1;
    for (size_t i = 0; i < total; ++i) {
        scalar_t v = A1[i];
        if (v >  bound || v < -bound) { in_range = 0; break; }
    }
    run_check("init_matrix_random: values stay in [-1/sqrt(rows), +1/sqrt(rows)]",
              in_range);

    xfree(A1); xfree(A2); xfree(B);
}

/* ---------- Test 5: matrices_close -------------------------------- */

/* matrices_close uses the mixed tolerance
 *   |A_ref[i] - A_test[i]| <= max(abs_tol, rel_tol * |A_ref[i]|).
 * The boundary cases we want to nail down:
 *   - identical buffers: must return 1, must not touch out-params.
 *   - difference exactly equal to abs_tol on a near-zero element:
 *     accepted (the inequality is non-strict).
 *   - difference slightly above abs_tol on a near-zero element:
 *     rejected, and the optional diagnostic out-params are set to the
 *     first offending index and values.
 *   - large-magnitude difference dominated by the relative tolerance:
 *     accepted when within rel_tol * |A_ref|.
 *   - passing NULL for the optional out-params is tolerated.
 */
static void test_matrices_close(void)
{
    const size_t n = 4;
    const scalar_t abs_tol = 1.0e-5f;
    const scalar_t rel_tol = 1.0e-4f;

    /* Case 1: identical buffers. */
    {
        scalar_t ref[4]  = { 1.0f, -2.0f, 3.0f, 0.0f };
        scalar_t test[4] = { 1.0f, -2.0f, 3.0f, 0.0f };
        size_t   bad_idx = 99;
        scalar_t bad_r = -1, bad_t = -1;
        int ok = matrices_close(ref, test, n, abs_tol, rel_tol,
                                &bad_idx, &bad_r, &bad_t);
        run_check("matrices_close: identical buffers -> 1", ok == 1);
    }

    /* Case 2: difference exactly at abs_tol on a near-zero element. */
    {
        scalar_t ref[4]  = { 0.0f, 0.0f, 0.0f, 0.0f };
        scalar_t test[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        test[2] = abs_tol;  /* diff == abs_tol, must be accepted */
        int ok = matrices_close(ref, test, n, abs_tol, rel_tol,
                                NULL, NULL, NULL);
        run_check("matrices_close: diff == abs_tol -> 1 (boundary accepted)",
                  ok == 1);
    }

    /* Case 3: difference just above abs_tol, near zero -> fail, and
     * the diagnostic out-params point at the right cell. */
    {
        scalar_t ref[4]  = { 0.0f, 0.0f, 0.0f, 0.0f };
        scalar_t test[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        test[2] = abs_tol * 2.0f;
        size_t   bad_idx = 0;
        scalar_t bad_r = 0, bad_t = 0;
        int ok = matrices_close(ref, test, n, abs_tol, rel_tol,
                                &bad_idx, &bad_r, &bad_t);
        int diag_ok = (ok == 0)
                   && (bad_idx == 2)
                   && (bad_r == 0.0f)
                   && (bad_t == abs_tol * 2.0f);
        run_check("matrices_close: diff > abs_tol -> 0 with correct diagnostics",
                  diag_ok);
    }

    /* Case 4: large-magnitude difference dominated by rel_tol. With
     * |ref| = 1.0 and rel_tol = 1e-4, a 5e-5 deviation must be
     * accepted even though it is much bigger than abs_tol. */
    {
        scalar_t ref[4]  = { 1.0f, 1.0f, 1.0f, 1.0f };
        scalar_t test[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        test[1] = 1.0f + 5.0e-5f;  /* diff = 5e-5 = 0.5 * rel_tol * |ref| */
        int ok = matrices_close(ref, test, n, abs_tol, rel_tol,
                                NULL, NULL, NULL);
        run_check("matrices_close: small relative diff at |ref|=1 -> 1",
                  ok == 1);
    }

    /* Case 5: NULL out-params on a failing comparison must not crash. */
    {
        scalar_t ref[4]  = { 0.0f, 0.0f, 0.0f, 0.0f };
        scalar_t test[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
        int ok = matrices_close(ref, test, n, abs_tol, rel_tol,
                                NULL, NULL, NULL);
        run_check("matrices_close: NULL diagnostic pointers tolerated on failure",
                  ok == 0);
    }
}

int main(void)
{
    printf("=== matrix_utils unit tests ===\n");

    test_xalloc_aligned();
    test_init_matrix_zero();
    test_init_matrix_identity();
    test_init_matrix_random();
    test_matrices_close();

    if (g_failed) {
        printf("MATRIX_UTILS TESTS FAILED\n");
        return EXIT_FAILURE;
    }
    printf("MATRIX_UTILS TESTS OK\n");
    return EXIT_SUCCESS;
}
