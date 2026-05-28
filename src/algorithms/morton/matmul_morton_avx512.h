// matmul_morton_avx512.h - Matmul Morton-recursivo con microkernel
// AVX-512 + FMA como leaf (Zen 5 / EPYC 9R45 main_server). Misma
// estructura recursiva de matmul_morton, pero A vive en
// Morton-de-BLOQUES (tile = MR = 4) y el leaf invoca kernel_avx512_4x32
// en vez del ijk escalar.

#ifndef MATMUL_MORTON_AVX512_H
#define MATMUL_MORTON_AVX512_H

#include <stddef.h>

#include "matrix_utils.h"             /* scalar_t */
#include "kernel_avx512_morton.h"     /* KERNEL_AVX512_MORTON_MR / _NR */

/*
Dos layouts conviven en el proyecto:

- Morton "fino" (Fase 1.6, matmul_morton):
    A[i, j] -> A_morton[morton_encode(i, j)]

- Morton "de bloques" (Fase 1.7, este modulo), con tile = MR = 4:
    A[i, j] -> A_morton[
        morton_encode(i/MR, j/MR) * MR*MR + (i%MR)*MR + (j%MR)
    ]
  Es decir: A se particiona en sub-bloques MR x MR; los sub-bloques se
  ordenan entre si en Z-order; los MR*MR elementos de cada sub-bloque
  viven contiguos en row-major.

La propiedad de contiguidad de cuadrantes se preserva: al dividir un
sub-bloque cuadrado de lado 2h en sus 4 cuadrantes de lado h, los
cuadrantes ocupan 4 segmentos consecutivos de A_morton, ahora de
tamano h^2 floats cada uno. Lo unico que cambia es el significado del
nivel mas bajo: un "elemento" Morton es un tile 4x4 de floats, no un
float suelto. Eso da al microkernel filas contiguas dentro del tile,
necesarias para los _mm512_set1_ps.

Esta es la variante AVX-512: el leaf usa el microkernel 4x32
kernel_avx512_4x32 que opera sobre 32 columnas a la vez (2 ZMM por
fila * 4 filas = 8 acumuladores).
*/

/*
Tile lateral del layout Morton-de-bloques. Igual a MR (filas que maneja
el microkernel), para que cada tile en A se mapee a un panel-columna
slice que el microkernel pueda consumir directamente. Coincide con el
MR del kernel AVX-512 (= 4).
*/
#define MORTON_AVX512_TILE KERNEL_AVX512_MORTON_MR    /* 4 */

/*
matmul_morton_avx512: Computa C = A * B con A en Morton-de-bloques.
    INPUTS:
    - C: Buffer de salida (m * n elementos, row-major).
    - A_morton: A en layout Morton-de-bloques (producir con
      reorganize_to_morton_blocks).
    - B: Buffer de entrada (k * n elementos, row-major).
    - m, k: Dimensiones de A. m == k, m potencia de 2 y m >= MR = 4.
    - n: Columnas de B y C. Recomendado n multiplo de NR = 32; el
      fallback ijk maneja otros casos pero lentamente.
    OUTPUTS:
    - Ninguno (void). Aborta con exit(EXIT_FAILURE) si m != k o si las
      precondiciones de m no se cumplen.
*/
void matmul_morton_avx512(scalar_t *C,
                          const scalar_t *A_morton,
                          const scalar_t *B,
                          size_t m, size_t k, size_t n);

/*
Threshold de recursion, separado del de la Fase 1.6 porque los dos
regimenes son distintos: el leaf AVX-512 amortiza una superficie mucho
mayor que el leaf ijk escalar y tolera sub-bloques mas grandes.
- Default: 1048576 elementos producto. Leaf side ~90, panel de A
  activo ~32 KiB, dos tercios del L1d (48 KiB) del EPYC 9R45.
- Override en compile-time via -DMORTON_AVX512_THRESHOLD_DEFAULT=...
*/
extern size_t g_recursion_threshold_avx512;
void matmul_morton_avx512_set_threshold(size_t threshold);

/*
reorganize_to_morton_blocks: Empaqueta una matriz row-major m x m al
layout Morton-de-bloques con tile = MORTON_AVX512_TILE.
    INPUTS:
    - A_row: Puntero a la matriz row-major de origen (m * m elementos).
    - A_morton: Buffer destino, capacidad m * m elementos (el caller lo
      aloja con xalloc_aligned).
    - m: Lado de la matriz. Debe cumplir m % MORTON_AVX512_TILE == 0 y
      (m / MORTON_AVX512_TILE) potencia de 2.
    OUTPUTS:
    - Ninguno (void). Aborta con exit(EXIT_FAILURE) si las
      precondiciones de m no se cumplen.
*/
void reorganize_to_morton_blocks(const scalar_t *A_row,
                                 scalar_t *A_morton,
                                 size_t m);

/*
Orquestadores de benchmark, misma semantica que los de la Fase 1.6
pero ruteados a traves del kernel AVX-512.

benchmark_iterations_morton_avx512 toma A en row-major y la reorganiza
internamente => el costo de reorganizar entra en el wall-clock.

benchmark_iterations_morton_avx512_preorganized toma A ya en
Morton-de-bloques y es la que mide bench_morton_avx512.
*/
void benchmark_iterations_morton_avx512(scalar_t *B_out,
                                        const scalar_t *A,
                                        const scalar_t *Z,
                                        size_t m, size_t n,
                                        size_t num_iters);

void benchmark_iterations_morton_avx512_preorganized(scalar_t *B_out,
                                                     const scalar_t *A_morton,
                                                     const scalar_t *Z,
                                                     size_t m, size_t n,
                                                     size_t num_iters);

#endif /* MATMUL_MORTON_AVX512_H */
