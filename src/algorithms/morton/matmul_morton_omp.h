/*
 * matmul_morton_omp.h - Morton-recursive matmul with the AVX-512 + FMA
 * microkernel as leaf, parallelized with OpenMP tasks (Zen 5 /
 * EPYC 9R45 main_server variant).
 *
 * The recursion shape and the leaf kernel are the same as
 * matmul_morton_avx512: same Morton-of-blocks layout for A
 * (tile = MORTON_AVX512_TILE = 4), same 4x32 AVX-512 microkernel
 * (kernel_avx512_4x32 from kernel_avx512_morton.h), same overwrite /
 * accumulate split. The only additions are:
 *
 *   - The recursion is wrapped in #pragma omp parallel / #pragma omp
 *     single inside the public wrapper, and each sub-problem above
 *     g_parallel_threshold_omp spawns OpenMP tasks instead of running
 *     sequentially.
 *
 *   - The scratch A_local buffer used by the leaf is no longer a
 *     single threaded buffer; it is a per-thread pool indexed by
 *     omp_get_thread_num() so the parallel leaves never share scratch.
 *
 *   - Two thresholds, both runtime-tunable:
 *       * g_recursion_threshold_omp - sub-problem size at which the
 *         recursion falls to the leaf kernel. Default 524288 (same
 *         shape as matmul_morton_avx512's default).
 *       * g_parallel_threshold_omp - sub-problem size at which the
 *         recursion stops spawning new tasks and runs sequentially.
 *         Default 524288 (same as the leaf threshold: tasks fire at
 *         every recursive split above the leaf, none below it).
 *
 * Topology note (EPYC 9R45 / Zen 5 on AWS c8a.2xlarge): the VM
 * exposes 8 vCPUs from a single NUMA node with no SMT. L3 (32 MiB)
 * is shared by all 8 cores, so OMP_PROC_BIND=close/spread make no
 * topological difference here (in contrast to the Zen 2 4600H, which
 * had two CCXs with 4 MiB L3 each). Recommended runtime config:
 *   OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
 *
 * Pre-conditions for the public wrapper are identical to
 * matmul_morton_avx512: m == k, m a power of two with m >= MR = 4, A in
 * Morton-of-blocks layout (use reorganize_to_morton_blocks from
 * matmul_morton_avx512.h, since the layout is shared).
 */

#ifndef MATMUL_MORTON_OMP_H
#define MATMUL_MORTON_OMP_H

#include <stddef.h>

#include "matrix_utils.h"        /* scalar_t */
#include "matmul_morton_avx512.h"  /* reuse MORTON_AVX512_TILE and
                                  * reorganize_to_morton_blocks */

/*
 * matmul_morton_omp: same shape contract as matmul_morton_avx512 (C is
 * m x n out, A is m x k in Morton-of-blocks, B is k x n row-major)
 * but parallelized with OpenMP tasks across all sub-problems above
 * g_parallel_threshold_omp. The number of threads is whatever the
 * surrounding OMP environment provides (OMP_NUM_THREADS, or the
 * default = number of logical CPUs).
 */
void matmul_morton_omp(scalar_t *C,
                       const scalar_t *A_morton,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n);

/*
 * Leaf threshold (mirror of g_recursion_threshold_avx512 but tracked
 * separately so the parallel module can be tuned without disturbing
 * the serial AVX-512 module's measurements).
 */
extern size_t g_recursion_threshold_omp;
void matmul_morton_omp_set_threshold(size_t threshold);

/*
 * Task-spawn threshold. Recursive calls on sub-problems larger than
 * this threshold split into two OpenMP tasks (one per top/bottom
 * half of the C tile in the MK split, or per left/right half in the
 * N split). Sub-problems at or below this threshold recurse inline
 * without spawning tasks. Setting it equal to the leaf threshold
 * (the default) means tasks fire at every split level above the leaf
 * and never at the leaf itself.
 */
extern size_t g_parallel_threshold_omp;
void matmul_morton_omp_set_parallel_threshold(size_t threshold);

/*
 * Benchmark orchestrators, same shape as the matmul_morton_avx512
 * counterparts but routed through matmul_morton_omp inside.
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
