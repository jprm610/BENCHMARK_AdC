// matmul_morton.h - Matmul recursivo con A en layout Morton (Z-order),
// B y C en row-major. Cache-oblivious: la recursion divide A en sus
// cuatro cuadrantes hasta caer a un leaf escalar ijk.

#ifndef MATMUL_MORTON_H
#define MATMUL_MORTON_H

#include <stddef.h>

#include "matrix_utils.h"  /* scalar_t */

/*
Contrato:
- A es m x m (cuadrada), m potencia de 2.
- A_morton se produjo con reorganize_to_morton(A, A_morton, m) en core/.
- B es m x n, C es m x n, ambas row-major.
- C no alias con A_morton ni con B.

La recursion navega A por una pareja (offset, side) en vez de
(puntero, leading dimension): los cuatro cuadrantes ocupan cuatro
segmentos consecutivos de A_morton, asi que dividir en cuadrantes es
solo sumar {0, 1, 2, 3} * (half*half) al offset.

Ver docs/1.6) matmul_morton.md para el detalle conceptual.
*/

/*
matmul_morton: Computa C = A * B con A en layout Morton, B y C row-major.
    INPUTS:
    - C: Buffer de salida (m * n elementos, row-major).
    - A_morton: A en layout Morton-de-elementos (ver core/morton.h).
    - B: Buffer de entrada (m * n elementos, row-major).
    - m, k: Dimensiones de A. Deben cumplir m == k y m potencia de 2.
    - n: Dimensiones de la columna de B y C.
    OUTPUTS:
    - Ninguno (void). Aborta con exit(EXIT_FAILURE) si m != k o si m
      no es potencia de 2.
*/
void matmul_morton(scalar_t *C,
                   const scalar_t *A_morton,
                   const scalar_t *B,
                   size_t m, size_t k, size_t n);

/*
Threshold de recursion (tunable en runtime).
- Default: 32 * 32 * 128 = 131072 elementos producto. A esta
  profundidad el panel de A activo es ~4 KiB (cabe en L1d de 48 KiB
  del EPYC 9R45).
- Subirlo deja que el leaf vea sub-bloques que ya no caben en L1.
- Bajarlo aumenta la profundidad de la recursion y el overhead de
  llamadas; ademas, threshold 0 colapsaria todo al leaf (se ignora con
  un warning).
*/
extern size_t g_recursion_threshold;
void matmul_morton_set_threshold(size_t threshold);

/*
benchmark_iterations_morton: Corre la recurrencia B_{i+1} = A * B_i con
B_0 = Z usando matmul_morton como kernel. A se recibe row-major y se
reorganiza a Morton internamente en cada llamada => el costo de
reorganizar entra en el wall-clock.
    INPUTS:
    - B_out: Buffer de salida con espacio para num_iters * n * n elements.
      El bloque iter ocupa offsets [iter*n*n, (iter+1)*n*n).
    - A: Matriz row-major m x m.
    - Z: Matriz row-major m x n (estado inicial B_0).
    - m, n: Dimensiones.
    - num_iters: Numero de iteraciones a registrar.
    OUTPUTS:
    - Ninguno (void).
*/
void benchmark_iterations_morton(scalar_t *B_out,
                                 const scalar_t *A,
                                 const scalar_t *Z,
                                 size_t m, size_t n,
                                 size_t num_iters);

/*
benchmark_iterations_morton_preorganized: Igual que la anterior pero
recibe A ya en Morton. La usa bench_morton para mantener el costo
de la reorganizacion fuera de la region medida.
    INPUTS:
    - B_out, A_morton, Z, m, n, num_iters: ver arriba.
    OUTPUTS:
    - Ninguno (void).
*/
void benchmark_iterations_morton_preorganized(scalar_t *B_out,
                                              const scalar_t *A_morton,
                                              const scalar_t *Z,
                                              size_t m, size_t n,
                                              size_t num_iters);

#endif /* MATMUL_MORTON_H */
