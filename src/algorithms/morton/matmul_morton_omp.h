// matmul_morton_omp.h - Matmul Morton-recursivo con microkernel AVX2 +
// FMA como leaf, paralelizado con OpenMP tasks. Misma estructura
// recursiva que matmul_morton_avx2; el unico cambio es spawning de
// tasks en los splits arriba de g_parallel_threshold_omp y un scratch
// pool per-thread para el panel A_local.

#ifndef MATMUL_MORTON_OMP_H
#define MATMUL_MORTON_OMP_H

#include <stddef.h>

#include "matrix_utils.h"        /* scalar_t */
#include "matmul_morton_avx2.h"  /* reusa MORTON_AVX2_TILE y
                                  * reorganize_to_morton_blocks */

/*
Topologia del 4600H (Renoir, Zen 2):
- 6 cores fisicos en 2 CCX de 3 cores cada uno; 12 threads logicos via SMT.
- L3 (4 MiB) es PRIVADO por CCX => threads en CCX distintos no comparten
  L3 y pagan Infinity Fabric por cualquier coherencia.

Empiricamente:
- OMP_PROC_BIND=close escala bien hasta 3 threads (un CCX entero) y
  luego carga el L3.
- spread escala mas pero paga trafico cross-CCX.

Precondiciones del wrapper publico (identicas a matmul_morton_avx2):
- m == k, m potencia de 2 con m >= MR = 4.
- A en layout Morton-de-bloques (usar reorganize_to_morton_blocks de
  matmul_morton_avx2.h, el layout es compartido).
*/

/*
matmul_morton_omp: Computa C = A * B paralelizando la recursion con
OpenMP tasks. El numero de threads lo decide el entorno OMP
(OMP_NUM_THREADS, o el numero de CPUs logicos por defecto).
    INPUTS:
    - C, A_morton, B, m, k, n: ver matmul_morton_avx2.h.
    OUTPUTS:
    - Ninguno (void). Aborta con exit(EXIT_FAILURE) en las mismas
      condiciones que matmul_morton_avx2.
*/
void matmul_morton_omp(scalar_t *C,
                       const scalar_t *A_morton,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n);

/*
Threshold de leaf (espejo de g_recursion_threshold_avx2 pero separado
para tunear este modulo sin perturbar al serial AVX2).
*/
extern size_t g_recursion_threshold_omp;
void matmul_morton_omp_set_threshold(size_t threshold);

/*
Threshold de spawn de tasks. Sub-problemas mas grandes que este lanzan
2 tasks OpenMP en el split (top/bot en el caso MK, left/right en el
caso N). Sub-problemas <= a este recursan inline sin tasks.

Default igual al leaf threshold => tasks en todos los splits arriba
del leaf, ninguna en el leaf. set_parallel_threshold(0) permitido =>
fuerza recursion serializada en este modulo (util para aislar el costo
del scaffolding OMP del paralelismo real).
*/
extern size_t g_parallel_threshold_omp;
void matmul_morton_omp_set_parallel_threshold(size_t threshold);

/*
Orquestadores de benchmark, misma forma que los de matmul_morton_avx2
pero ruteados a traves de matmul_morton_omp.
*/
void benchmark_iterations_morton_omp(scalar_t *B_out,
                                     const scalar_t *A,
                                     const scalar_t *Z,
                                     size_t m, size_t n,
                                     size_t num_iters);

void benchmark_iterations_morton_omp_preorganized(scalar_t *B_out,
                                                  const scalar_t *A_morton,
                                                  const scalar_t *Z,
                                                  size_t m, size_t n,
                                                  size_t num_iters);

#endif /* MATMUL_MORTON_OMP_H */
