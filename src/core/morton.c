// morton.c - Implementacion portable del interleave Morton via shifts y
// mascaras AND (sin pdep/pext, sin intrinsics arquitectura-especificos).

#include "morton.h"

#include <stdio.h>
#include <stdlib.h>

/*
spread_bits_32_to_64: Lleva un word de 32 bits a las posiciones pares de
un word de 64 bits. Bit k de x queda en bit 2k del resultado; las
posiciones impares quedan en cero.
    INPUTS:
    - x: Word de 32 bits a esparcir.
    OUTPUTS:
    - Word de 64 bits con los bits de x en posiciones pares.

Rutina de 5 pasos. Cada paso duplica la separacion entre bits usando un
OR-shift seguido de AND con una mascara alternante:
    - 0x0000FFFF0000FFFF: bloques de 16 unos separados por 16 ceros.
    - 0x00FF00FF00FF00FF: bloques de 8.
    - 0x0F0F0F0F0F0F0F0F: bloques de 4.
    - 0x3333333333333333: bloques de 2.
    - 0x5555555555555555: 1 bit aislado, separado por 1 cero.
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
compact_bits_64_to_32: Inverso de spread_bits_32_to_64. Recoge los bits
en posiciones pares de y y los empaqueta en los 32 bits bajos del
resultado. Los bits en posiciones impares quedan descartados por la
mascara inicial.
    INPUTS:
    - y: Word de 64 bits a compactar.
    OUTPUTS:
    - Word de 32 bits con los bits pares de y empaquetados.
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

uint64_t morton_encode(uint32_t i, uint32_t j)
{
    /*
    j a posiciones pares, i a impares. El shift << 1 sobre i es lo que
    fija la convencion (ver morton.h). Esa eleccion es la que hace que
    los cuadrantes (TL, TR, BL, BR) caigan en codigos 0, 1, 2, 3.
    */
    return spread_bits_32_to_64(j) | (spread_bits_32_to_64(i) << 1);
}

void morton_decode(uint64_t code, uint32_t *i, uint32_t *j)
{
    *j = compact_bits_64_to_32(code);
    *i = compact_bits_64_to_32(code >> 1);
}

int is_power_of_two(size_t m)
{
    /* Truco clasico: una potencia de 2 tiene un solo bit en 1, asi que
     * m & (m - 1) == 0. La verificacion m > 0 descarta el caso m == 0,
     * que tambien cumple la condicion del AND pero no es una potencia. */
    return m > 0 && (m & (m - 1)) == 0;
}

/*
require_power_of_two: Helper compartido por las dos funciones de
reorganizacion. Definido una sola vez para que el mensaje de error sea
identico.
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

    /*
    Lectura row-major secuencial, escritura Morton dispersa. Se ejecuta
    una sola vez por matriz, fuera de la region medida del benchmark.
    */
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
