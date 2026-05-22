/*
 * matmul_morton.h - Recursive matrix multiplication with A in Morton
 * (Z-order) layout, B and C in row-major.
 *
 * Phase 6, Stage A3 kernel. The recursion threads the A matrix as a flat
 * Morton-ordered buffer: instead of carrying a pointer + leading
 * dimension for A, the inner function carries an offset into that buffer
 * plus the side length of the current square sub-block. When a sub-block
 * of A is split into its four quadrants (TL, TR, BL, BR), the Z-order
 * encoding (see morton.h) guarantees that those quadrants occupy four
 * consecutive segments in memory, so no pointer arithmetic on A involves
 * non-contiguous strides.
 *
 * Public preconditions:
 *   - A is m x m (square).
 *   - m is a power of 2.
 *   - A_morton has been produced by reorganize_to_morton(A, A_morton, m).
 *   - C is m x n, B is m x n; both row-major.
 *   - C does not alias A_morton or B.
 *
 * See docs/API.md for the full contract.
 */

#ifndef MATMUL_MORTON_H
#define MATMUL_MORTON_H

#include <stddef.h>

#include "matrix_utils.h"  /* scalar_t */

/*
 * matmul_morton: compute C = A * B where A is supplied in Morton layout
 * and B, C are row-major. Aborts on stderr + exit(EXIT_FAILURE) if
 * m != k or if m is not a power of 2.
 *
 * Sub-problems traverse A via a (offset, block_dim) pair instead of a
 * (pointer, stride) pair; the wrapper kicks off the recursion with
 * offset=0 and block_dim=m.
 */
void matmul_morton(scalar_t *C,
                   const scalar_t *A_morton,
                   const scalar_t *B,
                   size_t m, size_t k, size_t n);

/*
 * Runtime-configurable recursion threshold. Default is the Sesion 02
 * value (32 * 32 * 128 = 131072 element products), preserved so that
 * existing callers see the same behaviour as before this knob was added.
 *
 * The setter is the recommended interface for tooling such as
 * scripts/run_threshold_sweep.sh, which iterates over thresholds to
 * find the empirical optimum on the test machine (Sesion 03 / Prompt 2).
 * Reading the variable directly is allowed but not part of the
 * supported API contract.
 */
extern size_t g_recursion_threshold;
void matmul_morton_set_threshold(size_t threshold);

/*
 * benchmark_iterations_morton: run the recurrence B_{i+1} = A * B_i with
 * B_0 = Z, using matmul_morton as the inner kernel. A is supplied in
 * row-major; this function reorganizes it to Morton internally on every
 * call, so the reorganization cost is included in the wall-clock time.
 *
 * Use benchmark_iterations_morton_preorganized below when the caller
 * wants to amortize the reorganization across many runs.
 *
 * B_out has space for num_iters * n * n elements; block iter occupies
 * offsets [iter*n*n, (iter+1)*n*n).
 */
void benchmark_iterations_morton(scalar_t *B_out,
                                 const scalar_t *A,
                                 const scalar_t *Z,
                                 size_t m, size_t n,
                                 size_t num_iters);

/*
 * benchmark_iterations_morton_preorganized: same semantics as
 * benchmark_iterations_morton but receives A already in Morton layout.
 * Used by bench_morton_O0 to keep the reorganization out of the measured
 * region.
 */
void benchmark_iterations_morton_preorganized(scalar_t *B_out,
                                              const scalar_t *A_morton,
                                              const scalar_t *Z,
                                              size_t m, size_t n,
                                              size_t num_iters);

#endif /* MATMUL_MORTON_H */
