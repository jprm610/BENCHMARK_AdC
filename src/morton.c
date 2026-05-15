/*
 * morton.c - Bit-twiddling implementation of the Morton encoding plus
 * row-major / Morton reorganization for square matrices.
 *
 * Level-1 portable implementation: pure shifts and AND masks. No BMI2
 * intrinsics (pdep/pext), no architecture-specific code. Same code path
 * runs identically on any C11-compliant target, which keeps the
 * cross-machine comparison clean.
 *
 * Convention (see morton.h for the full rationale):
 *   - j contributes to even-positioned bits (bit 0, 2, 4, ...).
 *   - i contributes to odd-positioned bits (bit 1, 3, 5, ...).
 *   - The 4x4 reference table from the technical document matches this
 *     convention; the morton_encode body below mirrors it.
 */

#include "morton.h"

#include <stdio.h>
#include <stdlib.h>

/*
 * spread_bits_32_to_64: take a 32-bit word and spread its bits into the
 * even-positioned slots of a 64-bit word. Bit k of x ends up at bit 2k of
 * the result; odd-positioned slots are zero. Standard five-step
 * bit-twiddling routine; the magic constants are masks of alternating
 * 16/8/4/2/1 ones.
 */
static inline uint64_t spread_bits_32_to_64(uint32_t x)
{
    uint64_t y = x;
    y = (y | (y << 16)) & 0x0000FFFF0000FFFFULL;
    y = (y | (y <<  8)) & 0x00FF00FF00FF00FFULL;
    y = (y | (y <<  4)) & 0x0F0F0F0F0F0F0F0FULL;
    y = (y | (y <<  2)) & 0x3333333333333333ULL;
    y = (y | (y <<  1)) & 0x5555555555555555ULL;
    return y;
}

/*
 * compact_bits_64_to_32: inverse of spread_bits_32_to_64. Collects the
 * bits sitting at even positions of y and packs them into the low 32 bits
 * of the result. Bits at odd positions are discarded by the initial mask.
 */
static inline uint32_t compact_bits_64_to_32(uint64_t y)
{
    y &= 0x5555555555555555ULL;
    y = (y | (y >>  1)) & 0x3333333333333333ULL;
    y = (y | (y >>  2)) & 0x0F0F0F0F0F0F0F0FULL;
    y = (y | (y >>  4)) & 0x00FF00FF00FF00FFULL;
    y = (y | (y >>  8)) & 0x0000FFFF0000FFFFULL;
    y = (y | (y >> 16)) & 0x00000000FFFFFFFFULL;
    return (uint32_t)y;
}

/*
 * morton_encode: j goes to even bits, i to odd bits. This is what makes
 * the four 2x2 quadrants (TL, TR, BL, BR) map to codes 0, 1, 2, 3
 * respectively, which is the property the recursive Morton kernel needs.
 */
uint64_t morton_encode(uint32_t i, uint32_t j)
{
    return spread_bits_32_to_64(j) | (spread_bits_32_to_64(i) << 1);
}

/*
 * morton_decode: invert the interleaving by compacting the even bits
 * (j) and, after a single right shift, the odd bits (i).
 */
void morton_decode(uint64_t code, uint32_t *i, uint32_t *j)
{
    *j = compact_bits_64_to_32(code);
    *i = compact_bits_64_to_32(code >> 1);
}

int is_power_of_two(size_t m)
{
    return m > 0 && (m & (m - 1)) == 0;
}

/*
 * Abort helper shared by both reorganization functions. Defined once so
 * the message format stays identical.
 */
static void require_power_of_two(size_t m, const char *function_name)
{
    if (!is_power_of_two(m)) {
        fprintf(stderr,
                "Error in %s: m must be a power of two (got %llu).\n",
                function_name, (unsigned long long)m);
        exit(EXIT_FAILURE);
    }
}

void reorganize_to_morton(const scalar_t *A_row,
                          scalar_t *A_morton,
                          size_t m)
{
    require_power_of_two(m, "reorganize_to_morton");

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            uint64_t code = morton_encode((uint32_t)i, (uint32_t)j);
            A_morton[code] = A_row[i * m + j];
        }
    }
}

void reorganize_from_morton(const scalar_t *A_morton,
                            scalar_t *A_row,
                            size_t m)
{
    require_power_of_two(m, "reorganize_from_morton");

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            uint64_t code = morton_encode((uint32_t)i, (uint32_t)j);
            A_row[i * m + j] = A_morton[code];
        }
    }
}
