/*
 * morton.h - Z-order (Morton) bit interleaving and matrix reorganization.
 *
 * Phase 6, Stage A3 support module. Provides:
 *   - bit-interleaving primitives (morton_encode / morton_decode);
 *   - row-major <-> Morton repacking of square matrices;
 *   - a power-of-two predicate used by the kernels that require it.
 *
 * Bit convention (matches the table in the technical document and the
 * quadrant ordering used by matmul_morton):
 *
 *   For (i, j) with i = i_{p-1}...i_1 i_0 and j = j_{p-1}...j_1 j_0 in
 *   binary, the Morton code is the interleaved word
 *
 *       i_{p-1} j_{p-1} ... i_1 j_1 i_0 j_0
 *
 *   so j contributes to even-positioned bits (bit 0, 2, 4, ...) and i
 *   contributes to odd-positioned bits (bit 1, 3, 5, ...). This gives
 *   the canonical quadrant ordering for a 2x2 split:
 *       (0,0) -> 0  top-left
 *       (0,1) -> 1  top-right
 *       (1,0) -> 2  bottom-left
 *       (1,1) -> 3  bottom-right
 *   which is what matmul_morton.c relies on when it indexes the four
 *   sub-quadrants by offsets 0, 1, 2, 3 times the quadrant size.
 */

#ifndef MORTON_H
#define MORTON_H

#include <stddef.h>
#include <stdint.h>

#include "matrix_utils.h"  /* scalar_t */

/*
 * Interleave the bits of i and j to produce the Morton code, following
 * the bit convention documented at the top of this header.
 */
uint64_t morton_encode(uint32_t i, uint32_t j);

/*
 * Inverse of morton_encode. Used only for testing.
 */
void morton_decode(uint64_t code, uint32_t *i, uint32_t *j);

/*
 * Reorganize a row-major square matrix A_row (m x m, m a power of 2)
 * into a Morton-ordered linear buffer A_morton. The caller is
 * responsible for allocating A_morton with capacity for m*m elements.
 *
 * Mapping: A_row[i*m + j] -> A_morton[morton_encode(i, j)].
 *
 * Aborts with fprintf+exit(EXIT_FAILURE) if m is not a power of 2.
 */
void reorganize_to_morton(const scalar_t *A_row,
                          scalar_t *A_morton,
                          size_t m);

/*
 * Inverse of reorganize_to_morton. Same precondition on m.
 * For validation only; the kernel path never uses it.
 */
void reorganize_from_morton(const scalar_t *A_morton,
                            scalar_t *A_row,
                            size_t m);

/*
 * Returns 1 if m is a positive power of two, 0 otherwise.
 */
int is_power_of_two(size_t m);

#endif /* MORTON_H */
