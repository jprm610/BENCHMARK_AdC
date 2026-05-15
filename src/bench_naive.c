/*
 * bench_naive.c - Driver that runs the iterated naive matmul benchmark for
 * one problem size m and prints a CSV line to stdout.
 *
 * Usage:
 *   bench_naive_O0 <m> [num_iters] [num_runs]
 *
 * Output: m,n,num_iters,median_seconds,gflops
 *
 * The driver does one warm-up run (unmeasured) and num_runs measured runs;
 * it reports the median wall-clock time and derived sustained gflops.
 *
 * Why median and not mean: a single outlier (cron job, page fault burst,
 * frequency transient) skews the mean but not the median. Five runs is
 * the default minimum to make the median meaningful. Profiling drivers
 * (gprof, perf) typically pass num_runs=1 because they count absolute
 * events and care about a single deterministic execution, not the
 * statistical distribution of wall-clock times.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul_naive.h"
#include "matrix_utils.h"
#include "timing.h"

#define BLOCK_SIZE_N    128u  /* n in the benchmark definition */
#define DEFAULT_RUNS    5
#define MAX_MEAS_ITERS  4     /* Cap iterations for development runs */

static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <m> [num_iters] [num_runs]\n", argv[0]);
        fprintf(stderr,
                "  m         : problem size (m x m matrix A)\n"
                "  num_iters : iterations of the benchmark per run "
                "(default: min(2m/n, %d))\n"
                "  num_runs  : number of measured runs for median timing "
                "(default: %d)\n",
                MAX_MEAS_ITERS, DEFAULT_RUNS);
        return EXIT_FAILURE;
    }

    long long m_in = atoll(argv[1]);
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

    /* Default I_meas: bench definition is I = 2m/n; for development we cap
     * it at MAX_MEAS_ITERS to keep per-run time bounded. The user can
     * override via the second CLI argument. */
    size_t I_full = 2 * m / n;
    size_t I_meas = (I_full < (size_t)MAX_MEAS_ITERS) ? I_full : (size_t)MAX_MEAS_ITERS;
    if (argc >= 3) {
        long long i_in = atoll(argv[2]);
        if (i_in <= 0) {
            fprintf(stderr, "Error: num_iters must be positive.\n");
            return EXIT_FAILURE;
        }
        I_meas = (size_t)i_in;
    }

    /* Third optional argument: how many measured runs. Used by profiling
     * scripts (gprof, perf) that prefer a single deterministic execution. */
    size_t num_runs = (size_t)DEFAULT_RUNS;
    if (argc >= 4) {
        long long r_in = atoll(argv[3]);
        if (r_in <= 0) {
            fprintf(stderr, "Error: num_runs must be positive.\n");
            return EXIT_FAILURE;
        }
        num_runs = (size_t)r_in;
    }

    /* Allocate the three big buffers. A and Z are inputs; B_out collects
     * the first n rows of every B_{i+1}. */
    scalar_t *A     = xalloc_aligned(m * m);
    scalar_t *Z     = xalloc_aligned(m * n);
    scalar_t *B_out = xalloc_aligned(I_meas * n * n);

    /* Reproducible inputs: fixed seeds so every run sees the same data. */
    init_matrix_random(A, m, m, 42u);
    init_matrix_random(Z, m, n, 43u);

    /* Warm-up: one short run to populate caches and resolve any first-touch
     * page faults. The result is discarded. */
    benchmark_iterations(B_out, A, Z, m, n, 1);

    /* Measured runs. Allocated on the heap because num_runs is dynamic. */
    double *times = (double *)malloc(num_runs * sizeof(double));
    if (times == NULL) {
        fprintf(stderr, "Error: out of memory for times array.\n");
        return EXIT_FAILURE;
    }
    for (size_t r = 0; r < num_runs; ++r) {
        double t0 = now_seconds();
        benchmark_iterations(B_out, A, Z, m, n, I_meas);
        double t1 = now_seconds();
        times[r] = t1 - t0;
    }

    /* Sort to extract the median (num_runs / 2 index after sort). With
     * num_runs == 1 the "median" is trivially the single sample, which
     * is fine for profiling-driver callers. */
    qsort(times, num_runs, sizeof(double), compare_double);
    double median_seconds = times[num_runs / 2];

    /* Flop count for the benchmark: each iteration is 2 * m * m * n flops. */
    double flops_per_iter = 2.0 * (double)m * (double)m * (double)n;
    double total_flops    = flops_per_iter * (double)I_meas;
    double gflops         = total_flops / median_seconds / 1.0e9;

    /* CSV line on stdout: header lives in the sweep script. */
    printf("naive,%llu,%llu,%llu,%.6f,%.6f\n",
           (unsigned long long)m,
           (unsigned long long)n,
           (unsigned long long)I_meas,
           median_seconds, gflops);

    /* Force the compiler to "use" some output bytes so it cannot drop
     * the benchmark loop under aggressive optimization. Harmless at -O0
     * but matters when the same source compiles at higher levels later. */
    volatile scalar_t sink = B_out[0];
    (void)sink;

    free(times);
    xfree(A);
    xfree(Z);
    xfree(B_out);
    return EXIT_SUCCESS;
}
