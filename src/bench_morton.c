/*
 * bench_morton.c - Driver that runs the iterated Morton matmul benchmark
 * for one problem size m and prints a CSV line to stdout.
 *
 * Same CLI, same warm-up + median pattern, and same CSV format as
 * bench_naive_O0 and bench_recursive_O0. The differences are:
 *   1. m must be a power of two; the driver aborts otherwise.
 *   2. A is reorganized to Morton ONCE, outside the measured region (and
 *      outside the warm-up), so the timing reflects only the Morton
 *      kernel itself. The reorganization is O(m^2); excluding it makes
 *      the GFLOP/s number comparable to the recursive row-major bench.
 *
 * Usage:
 *   bench_morton_O0 <m> [num_iters] [num_runs] [--threshold N]
 *
 * The optional --threshold flag overrides g_recursion_threshold via
 * matmul_morton_set_threshold(N) BEFORE the warm-up runs, so every
 * timed iteration uses the requested value. Used by
 * scripts/run_threshold_sweep.sh (Sesion 03 / Prompt 2).
 *
 * Output: m,n,num_iters,median_seconds,gflops
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul_naive.h"        /* for scalar_t */
#include "matmul_morton.h"
#include "morton.h"              /* for reorganize_to_morton, is_power_of_two */
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
            "  m              : problem size (m x m matrix A; m must be a power of 2)\n"
            "  num_iters      : iterations of the benchmark per run "
            "(default: min(2m/n, %d))\n"
            "  num_runs       : number of measured runs for median timing "
            "(default: %d)\n"
            "  --threshold N  : override g_recursion_threshold before the warm-up\n"
            "                   (default: matmul_morton uses 32*32*128 = 131072)\n",
            progname, MAX_MEAS_ITERS, DEFAULT_RUNS);
}

int main(int argc, char **argv)
{
    /* Two-pass CLI parsing: first sweep the argv for --threshold and
     * pull the value out, then process the remaining tokens as
     * positional arguments. */
    size_t threshold_override = 0;        /* 0 means "do not override" */
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
                        "Error: --threshold value must be a positive integer (got '%s').\n",
                        argv[i + 1]);
                return EXIT_FAILURE;
            }
            threshold_override = (size_t)t_in;
            ++i;  /* skip the value */
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

    /* Apply the threshold override, if any, BEFORE the warm-up so every
     * timed iteration sees the same recursion budget. */
    if (threshold_override > 0) {
        matmul_morton_set_threshold(threshold_override);
    }

    /* Allocate everything once. Same seeds as the other bench drivers
     * so that cross-kernel comparison runs on identical inputs. */
    scalar_t *A        = xalloc_aligned(m * m);
    scalar_t *A_morton = xalloc_aligned(m * m);
    scalar_t *Z        = xalloc_aligned(m * n);
    scalar_t *B_out    = xalloc_aligned(I_meas * n * n);

    init_matrix_random(A, m, m, 42u);
    init_matrix_random(Z, m, n, 43u);

    /* Reorganize A to Morton ONCE, outside everything timed. */
    reorganize_to_morton(A, A_morton, m);

    /* Warm-up (unmeasured) using the preorganized variant. */
    benchmark_iterations_morton_preorganized(B_out, A_morton, Z, m, n, 1);

    double *times = (double *)malloc(num_runs * sizeof(double));
    if (times == NULL) {
        fprintf(stderr, "Error: out of memory for times array.\n");
        return EXIT_FAILURE;
    }
    for (size_t r = 0; r < num_runs; ++r) {
        double t0 = now_seconds();
        benchmark_iterations_morton_preorganized(B_out, A_morton, Z,
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
