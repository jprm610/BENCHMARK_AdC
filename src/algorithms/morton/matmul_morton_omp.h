// matmul_morton_omp.h - Matmul Morton-recursivo con microkernel
// AVX-512 + FMA como leaf, paralelizado con OpenMP tasks (Zen 5 /
// EPYC 9R45 main_server). Misma estructura recursiva que
// matmul_morton_avx512; el unico cambio es spawning de tasks en los
// splits arriba de g_parallel_threshold_omp y un scratch pool
// per-thread para el panel A_local.

#ifndef MATMUL_MORTON_OMP_H
#define MATMUL_MORTON_OMP_H

#include <stddef.h>

#include "matrix_utils.h"          /* scalar_t */
#include "matmul_morton_avx512.h"  /* reusa MORTON_AVX512_TILE y
                                    * reorganize_to_morton_blocks */

/*
Topologia del EPYC 9R45 (Zen 5) en AWS c8a.2xlarge:
- 8 vCPUs en un solo nodo NUMA; SMT deshabilitado por el hypervisor =>
  exactamente 1 thread por core fisico.
- L1d 48 KiB / core, L2 1 MiB / core, L3 32 MiB COMPARTIDO por los 8
  cores. A diferencia del 4600H (2 CCX con 4 MiB de L3 cada uno), aqui
  no hay particion de L3 => OMP_PROC_BIND=close y =spread producen el
  mismo resultado topologicamente.

Runtime recomendado:
    OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close

Precondiciones del wrapper publico (identicas a matmul_morton_avx512):
- m == k, m potencia de 2 con m >= MR = 4.
- A en layout Morton-de-bloques (usar reorganize_to_morton_blocks de
  matmul_morton_avx512.h, el layout es compartido).
*/

/*
matmul_morton_omp: Computa C = A * B paralelizando la recursion con
OpenMP tasks. El numero de threads lo decide el entorno OMP
(OMP_NUM_THREADS, o el numero de CPUs logicos por defecto = 8 en el
EPYC 9R45 con SMT off).
    INPUTS:
    - C, A_morton, B, m, k, n: ver matmul_morton_avx512.h.
    OUTPUTS:
    - Ninguno (void). Aborta con exit(EXIT_FAILURE) en las mismas
      condiciones que matmul_morton_avx512.
*/
void matmul_morton_omp(scalar_t *C,
                       const scalar_t *A_morton,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n);

/*
Threshold de leaf (espejo de g_recursion_threshold_avx512 pero separado
para tunear este modulo sin perturbar al serial AVX-512).
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
Orquestadores de benchmark, misma forma que los de matmul_morton_avx512
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
