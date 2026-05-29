// morton.h - Interleave de bits Z-order (Morton) y reorganizacion
// row-major <-> Morton para matrices cuadradas.

#ifndef MORTON_H
#define MORTON_H

#include <stddef.h>
#include <stdint.h>

#include "matrix_utils.h"  /* scalar_t */

/*
Convencion de bits (vale para todo el modulo):

    Para (i, j) con i = i_{p-1}...i_0 y j = j_{p-1}...j_0 en binario,
    el codigo Morton es la palabra interleaved

        i_{p-1} j_{p-1} ... i_1 j_1 i_0 j_0

    => j contribuye a las posiciones pares (bit 0, 2, 4, ...)
    => i contribuye a las posiciones impares (bit 1, 3, 5, ...)

Esto fija la numeracion de cuadrantes para un split 2x2 que usa el
kernel recursivo matmul_morton:

    (0,0) -> 0  TL  (top-left)
    (0,1) -> 1  TR  (top-right)
    (1,0) -> 2  BL  (bottom-left)
    (1,1) -> 3  BR  (bottom-right)

matmul_morton.c indexa los cuatro sub-cuadrantes con offsets 0, 1, 2, 3
multiplicados por (half*half), apoyandose exactamente en esta tabla.
*/

/*
morton_encode: Interleave los bits de i y j segun la convencion arriba.
    INPUTS:
    - i: Coordenada de fila (32 bits).
    - j: Coordenada de columna (32 bits).
    OUTPUTS:
    - Codigo Morton de 64 bits.
*/
uint64_t morton_encode(uint32_t i, uint32_t j);

/*
morton_decode: Inverso de morton_encode. Solo se usa en tests.
    INPUTS:
    - code: Codigo Morton de 64 bits.
    - i, j: Punteros donde escribir las coordenadas reconstruidas.
    OUTPUTS:
    - Ninguno (void). i y j quedan con los valores reconstruidos.
*/
void morton_decode(uint64_t code, uint32_t *i, uint32_t *j);

/*
reorganize_to_morton: Reordena una matriz row-major m x m en un buffer
Morton lineal. Mapea A_row[i*m + j] -> A_morton[morton_encode(i, j)].
    INPUTS:
    - A_row: Puntero a la matriz row-major de origen (m * m elementos).
    - A_morton: Puntero al buffer destino, capacidad m * m elementos
      (el caller lo aloja con xalloc_aligned).
    - m: Lado de la matriz. Debe ser potencia de 2.
    OUTPUTS:
    - Ninguno (void). Aborta con exit(EXIT_FAILURE) si m no es pot. de 2.
*/
void reorganize_to_morton(const scalar_t *A_row,
                          scalar_t *A_morton,
                          size_t m);

/*
reorganize_from_morton: Inverso de reorganize_to_morton. Solo validacion;
el camino del kernel nunca lo usa.
    INPUTS:
    - A_morton: Puntero a la matriz Morton de origen.
    - A_row: Puntero al buffer row-major destino, m * m elementos.
    - m: Lado de la matriz. Debe ser potencia de 2.
    OUTPUTS:
    - Ninguno (void).
*/
void reorganize_from_morton(const scalar_t *A_morton,
                            scalar_t *A_row,
                            size_t m);

/*
is_power_of_two: Predicado usado por las rutinas de reorganizacion.
    INPUTS:
    - m: Entero a verificar.
    OUTPUTS:
    - 1 si m es potencia positiva de 2, 0 en otro caso.
*/
int is_power_of_two(size_t m);

#endif /* MORTON_H */
