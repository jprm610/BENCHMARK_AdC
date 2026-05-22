/*
 * bench_morton_avx512.c - Driver that runs the iterated Morton matmul
 * benchmark with the AVX-512 microkernel.
 *
 * Same CLI shape as bench_morton:
 *   bench_morton_avx512_O3 <m> [num_iters] [num_runs] [--threshold N]
 *
 * Differences:
 *   - m must be a power of two AND m >= MORTON_AVX512_TILE = 4
 *     (matmul_morton_avx512 asserts this on its own).
 *   - The Morton reorganization uses reorganize_to_morton_blocks
 *     (tile=4), not reorganize_to_morton.
 *   - The optional --threshold N now overrides
 *     g_recursion_threshold_avx512 via matmul_morton_avx512_set_threshold.
 *
 * Output: m,n,num_iters,median_seconds,gflops (same columns as the
 * other bench drivers so plot_comparison.py / plot_sweep_session_03
 * can ingest a unified table later in Prompt 5).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul_morton_avx512.h"
#include "morton.h"                  /* is_power_of_two */
#include "matrix_utils.h"
#include "timing.h"

#define BLOCK_SIZE_N    128u
#define DEFAULT_RUNS    5
#define MAX_MEAS_ITERS  4

static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void usage(const char *progname)
{
    fprintf(stderr,
            "Usage: %s <m> [num_iters] [num_runs] [--threshold N]\n"
            "  m              : problem size (m x m matrix A; m power of 2, m >= 4)\n"
            "  num_iters      : iterations per measured run "
            "(default: min(2m/n, %d))\n"
            "  num_runs       : measured runs for median (default: %d)\n"
            "  --threshold N  : override g_recursion_threshold_avx512 before warm-up\n"
            "                   (default: matmul_morton_avx512 uses 64*64*128 = 524288)\n",
            progname, MAX_MEAS_ITERS, DEFAULT_RUNS);
}

int main(int argc, char **argv)
{
    /* Two-pass CLI parsing (same approach as bench_morton.c):
     * scan for --threshold first, then the positional args. */
    size_t threshold_override = 0;
    char *positional[3] = {NULL, NULL, NULL};
    int n_positional = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--threshold") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --threshold requires a value.\n");
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            long long t_in = atoll(argv[i + 1]);
            if (t_in <= 0) {
                fprintf(stderr,
                        "Error: --threshold value must be positive (got '%s').\n",
                        argv[i + 1]);
                return EXIT_FAILURE;
            }
            threshold_override = (size_t)t_in;
            ++i;
            continue;
        }
        if (n_positional >= 3) {
            fprintf(stderr, "Error: too many positional arguments.\n");
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        positional[n_positional++] = argv[i];
    }

    if (n_positional < 1) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    long long m_in = atoll(positional[0]);
    if (m_in <= 0) {
        fprintf(stderr, "Error: m must be a positive integer.\n");
        return EXIT_FAILURE;
    }
    size_t m = (size_t)m_in;
    size_t n = BLOCK_SIZE_N;

    if (m < n) {
        fprintf(stderr,
                "Error: m (%llu) must be at least n (%llu).\n",
                (unsigned long long)m, (unsigned long long)n);
        return EXIT_FAILURE;
    }
    if (!is_power_of_two(m)) {
        fprintf(stderr,
                "Error: m (%llu) must be a power of two for the Morton kernel.\n",
                (unsigned long long)m);
        return EXIT_FAILURE;
    }
    if (m < (size_t)MORTON_AVX512_TILE) {
        fprintf(stderr,
                "Error: m (%llu) must be at least %d for the AVX-512 kernel.\n",
                (unsigned long long)m, MORTON_AVX512_TILE);
        return EXIT_FAILURE;
    }

    size_t I_full = 2 * m / n;
    size_t I_meas = (I_full < (size_t)MAX_MEAS_ITERS) ? I_full : (size_t)MAX_MEAS_ITERS;
    if (n_positional >= 2) {
        long long i_in = atoll(positional[1]);
        if (i_in <= 0) {
            fprintf(stderr, "Error: num_iters must be positive.\n");
            return EXIT_FAILURE;
        }
        I_meas = (size_t)i_in;
    }

    size_t num_runs = (size_t)DEFAULT_RUNS;
    if (n_positional >= 3) {
        long long r_in = atoll(positional[2]);
        if (r_in <= 0) {
            fprintf(stderr, "Error: num_runs must be positive.\n");
            return EXIT_FAILURE;
        }
        num_runs = (size_t)r_in;
    }

    /* Apply the threshold override BEFORE the warm-up so every timed
     * iteration sees the same recursion budget. */
    if (threshold_override > 0) {
        matmul_morton_avx512_set_threshold(threshold_override);
    }

    scalar_t *A        = xalloc_aligned(m * m);
    scalar_t *A_morton = xalloc_aligned(m * m);
    scalar_t *Z        = xalloc_aligned(m * n);
    scalar_t *B_out    = xalloc_aligned(I_meas * n * n);

    /* Same seeds as the other bench drivers so cross-kernel timing
     * comparisons run on identical inputs. */
    init_matrix_random(A, m, m, 42u);
    init_matrix_random(Z, m, n, 43u);

    /* Reorganize A to Morton-of-blocks ONCE, outside everything timed. */
    reorganize_to_morton_blocks(A, A_morton, m);

    /* Warm-up (unmeasured). */
    benchmark_iterations_morton_avx512_preorganized(B_out, A_morton, Z, m, n, 1);

    double *times = (double *)malloc(num_runs * sizeof(double));
    if (times == NULL) {
        fprintf(stderr, "Error: out of memory for times array.\n");
        return EXIT_FAILURE;
    }
    for (size_t r = 0; r < num_runs; ++r) {
        double t0 = now_seconds();
        benchmark_iterations_morton_avx512_preorganized(B_out, A_morton, Z,
                                                      m, n, I_meas);
        double t1 = now_seconds();
        times[r] = t1 - t0;
    }

    qsort(times, num_runs, sizeof(double), compare_double);
    double median_seconds = times[num_runs / 2];

    double flops_per_iter = 2.0 * (double)m * (double)m * (double)n;
    double total_flops    = flops_per_iter * (double)I_meas;
    double gflops         = total_flops / median_seconds / 1.0e9;

    printf("%llu,%llu,%llu,%.6f,%.6f\n",
           (unsigned long long)m,
           (unsigned long long)n,
           (unsigned long long)I_meas,
           median_seconds, gflops);

    volatile scalar_t sink = B_out[0];
    (void)sink;

    free(times);
    xfree(A);
    xfree(A_morton);
    xfree(Z);
    xfree(B_out);
    return EXIT_SUCCESS;
}
