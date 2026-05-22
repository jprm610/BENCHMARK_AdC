/*
 * bench_naive.c 
    - Driver que ejecuta benchamark iterations (matmul_naive.c) para un m,
    - reporta una línea CSV con m,n,num_iters,median_seconds,gflops
 
 * Uso:
 *   bench_naive_O0 <m> [num_iters] [num_runs]
 *
 * Output: m,n,num_iters,median_seconds,gflops
 *
 * NOTA: Un benck de calentamiento (warm-up) y luego num_runs corridas medidas.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul_naive.h"
#include "matrix_utils.h"
#include "timing.h"

#define BLOCK_SIZE_N    128u  /* n (enunciado proyecto) */
#define DEFAULT_RUNS    5     /* Cap de corridas, es hiperparámetro [num_runs] */
#define MAX_MEAS_ITERS  4     /* Cap iteraciones, también es hiperparámetro [num_iters] */


/*
compare_double: Comparador de doubles (qsort). (Facilitar cálculo de la mediana).
INPUTS:
- a, b: punteros a los doubles a comparar.
OUTPUT:
- -1 -> a primero
-  0 -> iguales
- +1 -> b primero
*/
static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}


/*
main: Implementación benckmark.
1) Args command line: m, num_iters (opcional), num_runs (opcional).
2) Buffers: A, Z, B_out.
3) LCG: A y Z con datos reproducibles (semillas fijas).
4) Warm-up: una corrida corta para poblar caches y resolver page faults.
5) Corridas medidas: num_runs veces, medir el tiempo de benchmark_iterations.
6) Sort: Ordenar los tiempos para extraer la mediana.
7) Stats: Calcular GFLOPS usando el conteo de operaciones y la mediana de tiempo.
8) Save: Imprimir línea CSV con los resultados.
*/
int main(int argc, char **argv)
{   
//---------------------------------------------------------------------------------------------------
    // 1) Args command line

    // 1.1) Check de argumentos mínimos. (Al menos m).
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
    
    // 1.2) m positivo, n fijo (BLOCK_SIZE_N) con m >= n.
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
    
    // 1.3) I_meas (num_iters) opcional, default min(2m/n, MAX_MEAS_ITERS),
    //      a menos que se especifique en [num_iters].
    size_t I_full = 2 * m / n;
    size_t I_meas = (I_full < (size_t)MAX_MEAS_ITERS) ? I_full : (size_t)MAX_MEAS_ITERS;
    if (argc >= 3) {
        long long i_in = atoll(argv[2]);
        if (i_in < 0) {
            fprintf(stderr, "Error: num_iters must be >= 0.\n");
            return EXIT_FAILURE;
        }
        // Si num_iters = 0, usar I_full (2m/n), else usar arg.
        I_meas = (i_in == 0) ? I_full : (size_t)i_in;
    }

    // 1.4) num_runs (opcional), default DEFAULT_RUNS.
    //      a menos que se especifique en [num_runs].
    size_t num_runs = (size_t)DEFAULT_RUNS;
    if (argc >= 4) {
        long long r_in = atoll(argv[3]);
        if (r_in <= 0) {
            fprintf(stderr, "Error: num_runs must be positive.\n");
            return EXIT_FAILURE;
        }
        num_runs = (size_t)r_in;
    }

//---------------------------------------------------------------------------------------------------
    // 2) Buffers: A, Z, B_out.
    scalar_t *A     = xalloc_aligned(m * m);
    scalar_t *Z     = xalloc_aligned(m * n);

    // 2.1) B_out tendrá cada B_{i} (n*n) generado en cada iteración (contiguos).
    scalar_t *B_out = xalloc_aligned(I_meas * n * n);

//---------------------------------------------------------------------------------------------------
    // 3) LCG: A y Z con datos reproducibles (semillas f
    init_matrix_random(A, m, m, 42u);
    init_matrix_random(Z, m, n, 43u);

//---------------------------------------------------------------------------------------------------
    // 4) Warm-up
    benchmark_iterations(B_out, A, Z, m, n, 1);

//---------------------------------------------------------------------------------------------------
    // 5) Corridas medidas
    // 5.1) Memoria para guardar mediciones de tiempo (array)
    double *times = (double *)malloc(num_runs * sizeof(double));
    if (times == NULL) {
        fprintf(stderr, "Error: out of memory for times array.\n");
        return EXIT_FAILURE;
    }
    // 5.2) Loop de corridas medidas
    for (size_t r = 0; r < num_runs; ++r) {
        double t0 = now_seconds();
        benchmark_iterations(B_out, A, Z, m, n, I_meas);
        double t1 = now_seconds();
        times[r] = t1 - t0;
    }

//---------------------------------------------------------------------------------------------------
    // 6) Sort
    // - q_sort (stdlib) con compare_double (array inplace).
    qsort(times, num_runs, sizeof(double), compare_double);
    double median_seconds = times[num_runs / 2];

//---------------------------------------------------------------------------------------------------
    // 7) Stats
    // - FLOP por iteración: 2*m*m*n (producto punto fila x columna).
    // - Total FLOP: FLOP por iteración * I_meas.
    // - GFLOPS: Total FLOP / median_seconds / 1e9.
    double flops_per_iter = 2.0 * (double)m * (double)m * (double)n;
    double total_flops    = flops_per_iter * (double)I_meas;
    double gflops         = total_flops / median_seconds / 1.0e9;

//---------------------------------------------------------------------------------------------------
    // 8) Save: Línea CSV.
    printf("kernel,m,n,num_iters,median_seconds,gflops\n");
    printf("naive,%llu,%llu,%llu,%.6f,%.6f\n",
           (unsigned long long)m,
           (unsigned long long)n,
           (unsigned long long)I_meas,
           median_seconds, gflops);

    /* 
    - Evitar optimizaciones agresivas de memoria.
    - Como B_out no se usa, el compilador podría eliminarlo.
    - Compilador no elimina volatile.
    */
    volatile scalar_t sink = B_out[0];
    (void)sink;
    
    // Liberar memoria y salir.
    free(times);
    xfree(A);
    xfree(Z);
    xfree(B_out);
    return EXIT_SUCCESS;
}
