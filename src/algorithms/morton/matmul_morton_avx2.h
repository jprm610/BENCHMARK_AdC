/*
 * matmul_morton_avx2.h - Morton-recursive matmul with an AVX2 + FMA
 * microkernel as leaf (Sesion 03 / Prompt 4, Stage A4 integration).
 *
 * Same recursive structure as matmul_morton (Sesion 02 / Stage A3),
 * but the leaf computes the tile by repeatedly invoking
 * kernel_avx2_4x16. To make that kernel useful, A is stored in a
 * DIFFERENT Morton variant than the one used by matmul_morton:
 *
 *   Sesion 02 layout (Morton "fino"):
 *     each element A[i,j] sits at A_morton[morton_encode(i, j)].
 *     Used by matmul_morton; left intact.
 *
 *   Sesion 03 layout (Morton "de bloques", tile = 4):
 *     A is partitioned into MR x MR sub-blocks (MR = 4); the
 *     sub-blocks are Z-ordered between each other, and the four
 *     elements of each sub-block are stored in row-major order.
 *     So A[i,j] sits at
 *         A_morton[ morton_encode(i/MR, j/MR) * MR*MR
 *                 + (i % MR) * MR
 *                 + (j % MR) ].
 *     This is what matmul_morton_avx2 expects.
 *
 * The contiguity property that the recursion relies on is preserved:
 * when a square sub-block of side 2h is split into four quadrants of
 * side h, those four quadrants still occupy four consecutive
 * MR*MR-aligned segments of memory at offsets {0,1,2,3} * (h*h).
 * Only the meaning of the bottom level changes (block of 4x4 floats
 * instead of a single float).
 *
 * Pre-conditions for the public wrapper:
 *   - m == k (A square).
 *   - m is a power of two and m >= MR = 4.
 *   - n >= NR = 16. (n is recommended to be a multiple of NR;
 *     fallback ijk kernel handles non-multiples but slowly.)
 *   - A_morton was produced by reorganize_to_morton_blocks() below.
 *   - C does not alias A_morton or B.
 *
 * Failing m / k assertions abort the process with a stderr message,
 * mirroring matmul_morton.
 */

#ifndef MATMUL_MORTON_AVX2_H
#define MATMUL_MORTON_AVX2_H

#include <stddef.h>

#include "matrix_utils.h"   /* scalar_t */
#include "kernel_avx2.h"    /* for KERNEL_AVX2_MR / KERNEL_AVX2_NR */

/* Tile side used by the Morton-of-blocks layout. Same as the row
 * count of the microkernel, so each tile in A maps to one panel
 * column slice the microkernel will consume. */
#define MORTON_AVX2_TILE KERNEL_AVX2_MR    /* 4 */

/*
 * matmul_morton_avx2: compute C = A_morton * B with A in the
 * Morton-of-blocks layout described above. Same shape contract as
 * matmul_morton: C is m x n (out), A is m x k (in, Morton-blocks),
 * B is k x n (in, row-major).
 */
void matmul_morton_avx2(scalar_t *C,
                        const scalar_t *A_morton,
                        const scalar_t *B,
                        size_t m, size_t k, size_t n);

/*
 * Runtime-configurable recursion threshold, analogous to the Sesion 02
 * knob exposed in matmul_morton.h but tracked separately because the
 * regimes are different (the AVX2 kernel amortizes a much larger leaf
 * than the ijk + morton_encode baseline). Default: 64 * 64 * 128 =
 * 524288 element products, which gives a 64x64 panel of A per leaf
 * (working set ~16 KiB, half of L1d) on the test machine.
 */
extern size_t g_recursion_threshold_avx2;
void matmul_morton_avx2_set_threshold(size_t threshold);

/*
 * Reorganize a row-major m x m matrix into the Morton-of-blocks
 * layout consumed by matmul_morton_avx2. Aborts on stderr +
 * exit(EXIT_FAILURE) if m is not a multiple of MORTON_AVX2_TILE or
 * if (m / MORTON_AVX2_TILE) is not a power of two.
 *
 * Caller allocates A_morton with capacity for m*m elements (e.g.
 * via xalloc_aligned).
 */
void reorganize_to_morton_blocks(const scalar_t *A_row,
                                 scalar_t *A_morton,
                                 size_t m);

/*
 * Orchestrators with the same semantics as the matmul_morton ones
 * but routed through the AVX2 kernel.
 *
 * benchmark_iterations_morton_avx2 takes A in row-major and
 * reorganizes it internally on each call (the reorganization cost
 * is part of the wall-clock time).
 *
 * benchmark_iterations_morton_avx2_preorganized takes A already in
 * the Morton-of-blocks layout and is the one bench_morton_avx2
 * times.
 */
void benchmark_iterations_morton_avx2(scalar_t *B_out,
                                      const scalar_t *A,
                                      const scalar_t *Z,
                                      size_t m, size_t n,
                                      size_t num_iters);

void benchmark_iterations_morton_avx2_preorganized(scalar_t *B_out,
                                                   const scalar_t *A_morton,
                                                   const scalar_t *Z,
                                                   size_t m, size_t n,
                                                   size_t num_iters);

#endif /* MATMUL_MORTON_AVX2_H */
