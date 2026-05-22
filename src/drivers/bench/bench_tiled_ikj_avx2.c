/*
 * bench_tiled_ikj_avx2.c - Benchmark driver for matmul_tiled_ikj_avx2.
 *
 * Usage:
 *   bench_tiled_ikj_avx2_O3 <m> [num_iters] [num_runs] [bs]
 *
 *   m         : problem size (A is m x m, B is m x 128)
 *   num_iters : iterations per run  (default: min(2m/n, 4))
 *   num_runs  : measured runs for median (default: 5)
 *   bs        : AVX2 tiling block size, multiple of 8 (default: 64)
 *
 * Output (one CSV line on stdout):
 *   tiled_ikj_avx2,m,n,num_iters,bs,median_seconds,gflops
 *
 * One warm-up run precedes the measured runs to populate caches and
 * resolve first-touch page faults.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul_tiled_ikj_avx2.h"
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

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <m> [num_iters] [num_runs] [bs]\n"
                "  m         : problem size (m x m matrix A)\n"
                "  num_iters : iterations per run (default: min(2m/n,%d))\n"
                "  num_runs  : measured runs for median (default: %d)\n"
                "  bs        : block size, multiple of 8 (default: %u)\n",
                argv[0], MAX_MEAS_ITERS, DEFAULT_RUNS,
                TILED_IKJ_AVX2_BS_DEFAULT);
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

    size_t I_full = 2 * m / n;
    size_t I_meas = (I_full < (size_t)MAX_MEAS_ITERS)
                    ? I_full : (size_t)MAX_MEAS_ITERS;
    if (argc >= 3) {
        long long i_in = atoll(argv[2]);
        if (i_in < 0) {
            fprintf(stderr, "Error: num_iters must be >= 0.\n");
            return EXIT_FAILURE;
        }
        I_meas = (i_in == 0) ? I_full : (size_t)i_in;
    }

    size_t num_runs = (size_t)DEFAULT_RUNS;
    if (argc >= 4) {
        long long r_in = atoll(argv[3]);
        if (r_in <= 0) {
            fprintf(stderr, "Error: num_runs must be positive.\n");
            return EXIT_FAILURE;
        }
        num_runs = (size_t)r_in;
    }

    if (argc >= 5) {
        long long bs_in = atoll(argv[4]);
        if (bs_in <= 0) {
            fprintf(stderr, "Error: bs must be a positive integer.\n");
            return EXIT_FAILURE;
        }
        matmul_tiled_ikj_avx2_set_bs((size_t)bs_in);
    }

    size_t bs = g_tiled_ikj_avx2_bs;

    scalar_t *A     = xalloc_aligned(m * m);
    scalar_t *Z     = xalloc_aligned(m * n);
    scalar_t *B_out = xalloc_aligned(I_meas * n * n);

    init_matrix_random(A, m, m, 42u);
    init_matrix_random(Z, m, n, 43u);

    benchmark_iterations_tiled_ikj_avx2(B_out, A, Z, m, n, 1);

    double *times = (double *)malloc(num_runs * sizeof(double));
    if (times == NULL) {
        fprintf(stderr, "Error: out of memory for times array.\n");
        return EXIT_FAILURE;
    }
    for (size_t r = 0; r < num_runs; ++r) {
        double t0 = now_seconds();
        benchmark_iterations_tiled_ikj_avx2(B_out, A, Z, m, n, I_meas);
        double t1 = now_seconds();
        times[r] = t1 - t0;
    }

    qsort(times, num_runs, sizeof(double), compare_double);
    double median_seconds = times[num_runs / 2];

    double flops_per_iter = 2.0 * (double)m * (double)m * (double)n;
    double total_flops    = flops_per_iter * (double)I_meas;
    double gflops         = total_flops / median_seconds / 1.0e9;

    printf("tiled_ikj_avx2,%llu,%llu,%llu,%llu,%.6f,%.6f\n",
           (unsigned long long)m,
           (unsigned long long)n,
           (unsigned long long)I_meas,
           (unsigned long long)bs,
           median_seconds, gflops);

    volatile scalar_t sink = B_out[0];
    (void)sink;

    free(times);
    xfree(A);
    xfree(Z);
    xfree(B_out);
    return EXIT_SUCCESS;
}
