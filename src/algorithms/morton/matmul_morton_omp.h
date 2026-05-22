/*
 * matmul_morton_omp.h - Morton-recursive matmul with the AVX2 + FMA
 * microkernel as leaf, parallelized with OpenMP tasks.
 *
 * Sesion 03 / Prompt 6 (Stage A5). The recursion shape and the leaf
 * kernel are the same as matmul_morton_avx2 (Prompt 4): same
 * Morton-of-blocks layout for A (tile = MORTON_AVX2_TILE = 4), same
 * 4x16 microkernel, same overwrite / accumulate split. The only
 * additions are:
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
 *         shape as matmul_morton_avx2's default).
 *       * g_parallel_threshold_omp - sub-problem size at which the
 *         recursion stops spawning new tasks and runs sequentially.
 *         Default 524288 (same as the leaf threshold: tasks fire at
 *         every recursive split above the leaf, none below it).
 *
 * Topology note (Renoir / Ryzen 5 4600H): the chip has 2 CCX of 3
 * cores each; L3 (4 MiB) is private per CCX. Threads on different
 * CCXs do not share L3 and pay Infinity Fabric for any coherence
 * traffic. The benchmark script (scripts/run_omp_scaling.sh) compares
 * OMP_PROC_BIND=close (favors same-CCX) and =spread (uses both CCXs);
 * the read-out is that close scales well up to 3 threads then taxes
 * the L3, spread scales further but pays cross-CCX traffic.
 *
 * Pre-conditions for the public wrapper are identical to
 * matmul_morton_avx2: m == k, m a power of two with m >= MR = 4, A in
 * Morton-of-blocks layout (use reorganize_to_morton_blocks from
 * matmul_morton_avx2.h, since the layout is shared).
 */

#ifndef MATMUL_MORTON_OMP_H
#define MATMUL_MORTON_OMP_H

#include <stddef.h>

#include "matrix_utils.h"        /* scalar_t */
#include "matmul_morton_avx2.h"  /* reuse MORTON_AVX2_TILE and
                                  * reorganize_to_morton_blocks */

/*
 * matmul_morton_omp: same shape contract as matmul_morton_avx2 (C is
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
 * Leaf threshold (mirror of g_recursion_threshold_avx2 but tracked
 * separately so the parallel module can be tuned without disturbing
 * the serial AVX2 module's measurements).
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
 * Benchmark orchestrators, same shape as the matmul_morton_avx2
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
