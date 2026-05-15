/*
 * test_morton.c - Unit tests for the Morton encoding module.
 *
 * Four test groups exercise the module:
 *   1. 4x4 table: spot-check morton_encode against the explicit values
 *      from the technical document.
 *   2. Round-trip of the bit-twiddling: morton_decode(morton_encode(i,j))
 *      must return (i, j) for all (i, j) in [0, 64) x [0, 64).
 *   3. Quadrant contiguity: for m = 8, label each of the four quadrants
 *      of the row-major A and verify that reorganize_to_morton packs them
 *      into four consecutive segments of size m^2 / 4.
 *   4. Round-trip of the matrix reorganization: row-major -> Morton ->
 *      row-major must reproduce the original matrix bit-exactly for
 *      m in {16, 64, 256}.
 *
 * Exits 0 and prints MORTON TESTS OK if every check passes.
 */

#include <stdio.h>
#include <stdlib.h>

#include "morton.h"
#include "matrix_utils.h"
#include "matmul_naive.h"  /* for scalar_t */

/* ---------- Test 1: 4x4 reference table from the technical document. */

static int test_table_4x4(void)
{
    static const struct {
        uint32_t i;
        uint32_t j;
        uint64_t expected;
    } cases[] = {
        {0, 0,  0}, {0, 1,  1}, {1, 0,  2}, {1, 1,  3},
        {0, 2,  4}, {0, 3,  5}, {1, 2,  6}, {1, 3,  7},
        {2, 0,  8}, {2, 1,  9}, {3, 0, 10}, {3, 1, 11},
        {2, 2, 12}, {2, 3, 13}, {3, 2, 14}, {3, 3, 15},
    };
    const size_t num_cases = sizeof(cases) / sizeof(cases[0]);

    int failures = 0;
    for (size_t k = 0; k < num_cases; ++k) {
        uint64_t got = morton_encode(cases[k].i, cases[k].j);
        if (got != cases[k].expected) {
            printf("  [FAIL] morton_encode(%u, %u) = %llu, expected %llu\n",
                   cases[k].i, cases[k].j,
                   (unsigned long long)got,
                   (unsigned long long)cases[k].expected);
            failures++;
        }
    }

    if (failures == 0) {
        printf("  [OK]   4x4 table matches technical document\n");
    }
    return failures;
}

/* ---------- Test 2: round-trip morton_decode . morton_encode = id. */

static int test_roundtrip_encoding(void)
{
    int failures = 0;
    /* Bound on the number of mismatch lines printed, so that a bug does
     * not flood the terminal with 4096 failures. */
    int reported = 0;
    const int max_reports = 5;

    for (uint32_t i = 0; i < 64; ++i) {
        for (uint32_t j = 0; j < 64; ++j) {
            uint64_t code = morton_encode(i, j);
            uint32_t i_back = 0, j_back = 0;
            morton_decode(code, &i_back, &j_back);
            if (i_back != i || j_back != j) {
                if (reported < max_reports) {
                    printf("  [FAIL] roundtrip (%u, %u) -> code=%llu -> (%u, %u)\n",
                           i, j, (unsigned long long)code, i_back, j_back);
                    reported++;
                }
                failures++;
            }
        }
    }

    if (failures == 0) {
        printf("  [OK]   roundtrip encode/decode for (i,j) in [0,64)^2\n");
    }
    return failures;
}

/* ---------- Test 3: quadrant contiguity for m = 8. */

static int test_quadrant_contiguity(void)
{
    const size_t m = 8;
    const size_t quad_size = (m / 2) * (m / 2);  /* 16 elements per quadrant */

    scalar_t *A_row    = xalloc_aligned(m * m);
    scalar_t *A_morton = xalloc_aligned(m * m);

    /* Label every element by the quadrant it belongs to:
     *   1 = TL, 2 = TR, 3 = BL, 4 = BR. */
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            int top  = (i < m / 2);
            int left = (j < m / 2);
            scalar_t label;
            if      ( top &&  left) label = (scalar_t)1;
            else if ( top && !left) label = (scalar_t)2;
            else if (!top &&  left) label = (scalar_t)3;
            else                    label = (scalar_t)4;
            A_row[i * m + j] = label;
        }
    }

    reorganize_to_morton(A_row, A_morton, m);

    /* After reorganization, each block of quad_size consecutive entries
     * in A_morton must hold a single label, in the order TL, TR, BL, BR. */
    static const scalar_t expected_labels[4] = {
        (scalar_t)1, (scalar_t)2, (scalar_t)3, (scalar_t)4
    };
    const char *quadrant_names[4] = { "TL", "TR", "BL", "BR" };

    int failures = 0;
    for (size_t q = 0; q < 4; ++q) {
        for (size_t k = 0; k < quad_size; ++k) {
            scalar_t got = A_morton[q * quad_size + k];
            if (got != expected_labels[q]) {
                printf("  [FAIL] quadrant %s at A_morton[%llu]: got %.1f, expected %.1f\n",
                       quadrant_names[q],
                       (unsigned long long)(q * quad_size + k),
                       (double)got,
                       (double)expected_labels[q]);
                failures++;
                break;  /* one report per quadrant is enough */
            }
        }
    }

    xfree(A_row);
    xfree(A_morton);

    if (failures == 0) {
        printf("  [OK]   quadrant contiguity for m=8 (TL|TR|BL|BR each fill %llu consecutive cells)\n",
               (unsigned long long)quad_size);
    }
    return failures;
}

/* ---------- Test 4: round-trip of reorganization for m in {16, 64, 256}. */

static int test_roundtrip_reorg_one(size_t m)
{
    scalar_t *A_row    = xalloc_aligned(m * m);
    scalar_t *A_morton = xalloc_aligned(m * m);
    scalar_t *A_back   = xalloc_aligned(m * m);

    /* Reproducible random fill; same data has no bearing on the test, the
     * seeds just keep the test deterministic across runs. */
    init_matrix_random(A_row, m, m, 101u + (unsigned int)m);

    reorganize_to_morton(A_row, A_morton, m);
    reorganize_from_morton(A_morton, A_back, m);

    /* The round-trip is a pure copy (no arithmetic); equality must be
     * bit-exact, so the tolerances are set to zero. */
    size_t bad_idx = 0;
    scalar_t bad_ref = 0, bad_got = 0;
    int ok = matrices_close(A_row, A_back, m * m,
                            (scalar_t)0, (scalar_t)0,
                            &bad_idx, &bad_ref, &bad_got);

    xfree(A_row);
    xfree(A_morton);
    xfree(A_back);

    if (!ok) {
        printf("  [FAIL] roundtrip reorg m=%llu at index %llu: expected %.6e, got %.6e\n",
               (unsigned long long)m,
               (unsigned long long)bad_idx,
               (double)bad_ref, (double)bad_got);
        return 1;
    }
    return 0;
}

static int test_roundtrip_reorg(void)
{
    static const size_t M_LIST[] = { 16u, 64u, 256u };
    static const size_t NUM_M = sizeof(M_LIST) / sizeof(M_LIST[0]);

    int failures = 0;
    for (size_t i = 0; i < NUM_M; ++i) {
        failures += test_roundtrip_reorg_one(M_LIST[i]);
    }
    if (failures == 0) {
        printf("  [OK]   roundtrip reorganization for m in {16, 64, 256}\n");
    }
    return failures;
}

int main(void)
{
    printf("Running Morton module tests\n");

    int failures = 0;
    failures += test_table_4x4();
    failures += test_roundtrip_encoding();
    failures += test_quadrant_contiguity();
    failures += test_roundtrip_reorg();

    if (failures == 0) {
        printf("MORTON TESTS OK\n");
        return EXIT_SUCCESS;
    } else {
        printf("MORTON TESTS FAILED (%d failure(s))\n", failures);
        return EXIT_FAILURE;
    }
}
