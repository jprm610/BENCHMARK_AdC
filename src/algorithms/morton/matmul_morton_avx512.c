// matmul_morton_avx512.c - Recursion Morton + microkernel AVX-512 4x32
// (Zen 5 / EPYC 9R45). Misma estructura recursiva que matmul_morton.c,
// pero:
// - A vive en Morton-de-BLOQUES (tile MORTON_AVX512_TILE = 4) en vez de
//   Morton-de-elementos.
// - El leaf materializa un panel A_local row-major desde el layout
//   Morton-de-bloques y dispatch el microkernel kernel_avx512_4x32
//   sobre cada tile 4x32 de (A_local, B, C).
// - El threshold es separado (g_recursion_threshold_avx512) para no
//   afectar al modulo de la Fase 1.6.

#include "matmul_morton_avx512.h"

#include "morton.h"                /* morton_encode, is_power_of_two */
#include "kernel_avx512_morton.h"  /* kernel_avx512_4x32, geometry */
#include "matrix_utils.h"          /* xalloc_aligned, xfree */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MR KERNEL_AVX512_MORTON_MR   /* 4 - atado a MORTON_AVX512_TILE */
#define NR KERNEL_AVX512_MORTON_NR   /* 32 */
#define LEAF_KERNEL kernel_avx512_4x32

/*
Default leaf-size threshold para la recursion. Dimensionado para el
EPYC 9R45 con L1d 48 KiB / L2 1 MiB / L3 32 MiB shared.
- 1048576 = leaf side ~90, panel A 90*90*4 = ~32 KiB (cabe en L1d),
  panel B 90*32*4 = ~11 KiB (cabe en L1d).
- Override compile-time via -DMORTON_AVX512_THRESHOLD_DEFAULT=...
- Tambien tunable en runtime via matmul_morton_avx512_set_threshold().
*/
#ifndef MORTON_AVX512_THRESHOLD_DEFAULT
#define MORTON_AVX512_THRESHOLD_DEFAULT ((size_t)1048576UL)
#endif
size_t g_recursion_threshold_avx512 = MORTON_AVX512_THRESHOLD_DEFAULT;

void matmul_morton_avx512_set_threshold(size_t threshold)
{
    /* Mismo guard que matmul_morton: threshold 0 colapsaria todo al
     * leaf con las dimensiones globales. */
    if (threshold == 0) {
        fprintf(stderr,
                "Warning: matmul_morton_avx512_set_threshold(0) ignored; "
                "keeping previous threshold (%llu).\n",
                (unsigned long long)g_recursion_threshold_avx512);
        return;
    }
    g_recursion_threshold_avx512 = threshold;
}

/* ================================================================== */
/* Reorganizacion del layout                                          */
/* ================================================================== */

void reorganize_to_morton_blocks(const scalar_t *A_row,
                                 scalar_t *A_morton,
                                 size_t m)
{
    /*
    Para que morton_encode aplicado a coordenadas-de-bloque produzca
    una permutacion contigua de [0, (m/MR)^2), m/MR debe ser potencia
    de 2. m mismo no tiene que serlo en general, pero el proyecto
    siempre usa m potencia de 2 con m >= MR => ambas condiciones se
    cumplen trivialmente.
    */
    if (m == 0 || (m % MORTON_AVX512_TILE) != 0) {
        fprintf(stderr,
                "Error in reorganize_to_morton_blocks: m (%llu) must be "
                "a positive multiple of MORTON_AVX512_TILE (%d).\n",
                (unsigned long long)m, MORTON_AVX512_TILE);
        exit(EXIT_FAILURE);
    }
    size_t n_blocks_per_side = m / MORTON_AVX512_TILE;
    if (!is_power_of_two(n_blocks_per_side)) {
        fprintf(stderr,
                "Error in reorganize_to_morton_blocks: m/MORTON_AVX512_TILE "
                "(%llu) must be a power of two.\n",
                (unsigned long long)n_blocks_per_side);
        exit(EXIT_FAILURE);
    }

    /* Recorrer en coordenadas de bloque para que las dos iteraciones
     * internas llenen un tile completo de A_morton de una sola vez =>
     * escrituras secuenciales por tile. */
    for (size_t bi = 0; bi < n_blocks_per_side; ++bi) {
        for (size_t bj = 0; bj < n_blocks_per_side; ++bj) {
            uint64_t bcode = morton_encode((uint32_t)bi, (uint32_t)bj);
            size_t block_offset = (size_t)bcode
                                * (size_t)MORTON_AVX512_TILE
                                * (size_t)MORTON_AVX512_TILE;

            for (size_t ii = 0; ii < MORTON_AVX512_TILE; ++ii) {
                size_t row_global = bi * MORTON_AVX512_TILE + ii;
                for (size_t jj = 0; jj < MORTON_AVX512_TILE; ++jj) {
                    size_t col_global = bj * MORTON_AVX512_TILE + jj;
                    A_morton[block_offset + ii * MORTON_AVX512_TILE + jj]
                        = A_row[row_global * m + col_global];
                }
            }
        }
    }
}

/* ================================================================== */
/* Kernels leaf                                                        */
/* ================================================================== */

/*
materialize_a_panel: Copia el panel m_block x k_block de A desde el
layout Morton-de-bloques a un buffer row-major A_local con row stride
k_block. El microkernel asume A row-major, asi que esta materializacion
es la traduccion del layout.

m_block y k_block deben ser multiplos de MORTON_AVX512_TILE (el leaf lo
verifica antes de llamar). Cada fila destino queda contigua y el
microkernel la consume con leading dimension k_block.
*/
static void materialize_a_panel(const scalar_t *A_morton,
                                size_t a_morton_offset,
                                size_t a_block_dim,
                                scalar_t *A_local,
                                size_t m_block, size_t k_block)
{
    (void)a_block_dim;  /* invariante: m_block == k_block == a_block_dim */

    const size_t blocks_m = m_block / MORTON_AVX512_TILE;
    const size_t blocks_k = k_block / MORTON_AVX512_TILE;
    const size_t tile_sq  = (size_t)MORTON_AVX512_TILE * MORTON_AVX512_TILE;

    for (size_t bi = 0; bi < blocks_m; ++bi) {
        for (size_t bj = 0; bj < blocks_k; ++bj) {
            uint64_t bcode = morton_encode((uint32_t)bi, (uint32_t)bj);
            size_t block_offset = a_morton_offset
                                + (size_t)bcode * tile_sq;

            for (size_t ii = 0; ii < MORTON_AVX512_TILE; ++ii) {
                size_t row_local = bi * MORTON_AVX512_TILE + ii;
                const scalar_t *src = &A_morton[block_offset
                                              + ii * MORTON_AVX512_TILE];
                scalar_t *dst = &A_local[row_local * k_block
                                       + bj * MORTON_AVX512_TILE];
                /* Copiar MORTON_AVX512_TILE = 4 floats; suficientemente
                 * pequeno para que el compilador emita straight-line
                 * moves sin loop. */
                for (size_t jj = 0; jj < MORTON_AVX512_TILE; ++jj) {
                    dst[jj] = src[jj];
                }
            }
        }
    }
}

/*
kernel_base_morton_avx512_ijk_fallback: Camino lento para hojas con
dimensiones que no alinean al tile del microkernel. En el sweep
regular m es pot. de 2 con m >= 4 y n = 128 = 4*NR => nunca se invoca;
queda por seguridad.

Indexa A por A_local (row-major gracias a materialize_a_panel), no por
A_morton, para no duplicar la logica de acceso Morton.
*/
static void kernel_base_morton_avx512_ijk_fallback(
    scalar_t *C, size_t ldc,
    const scalar_t *A_local, size_t lda,
    const scalar_t *B, size_t ldb,
    size_t m_block, size_t k_block, size_t n_block,
    int accumulate)
{
    for (size_t i = 0; i < m_block; ++i) {
        for (size_t j = 0; j < n_block; ++j) {
            scalar_t sum = (scalar_t)0;
            for (size_t p = 0; p < k_block; ++p) {
                sum += A_local[i * lda + p] * B[p * ldb + j];
            }
            if (accumulate) C[i * ldc + j] += sum;
            else            C[i * ldc + j]  = sum;
        }
    }
}

/*
kernel_base_morton_avx512: Variante overwrite del leaf. Materializa
A_local, zerea el tile de C, y luego deja al microkernel acumular sobre
el (cero + acumulado = A * B exactamente).
*/
static void kernel_base_morton_avx512(
    scalar_t *C, const scalar_t *A_morton, const scalar_t *B,
    size_t m_block, size_t k_block, size_t n_block,
    size_t a_morton_offset, size_t a_block_dim,
    size_t ldc, size_t ldb,
    scalar_t *A_local_scratch)
{
    materialize_a_panel(A_morton, a_morton_offset, a_block_dim,
                        A_local_scratch, m_block, k_block);

    if ((m_block % MR) != 0 || (n_block % NR) != 0) {
        /* Fallback: zerear C y acumular via camino lento. */
        for (size_t i = 0; i < m_block; ++i) {
            scalar_t *crow = &C[i * ldc];
            for (size_t j = 0; j < n_block; ++j) crow[j] = (scalar_t)0;
        }
        kernel_base_morton_avx512_ijk_fallback(C, ldc,
                                             A_local_scratch, k_block,
                                             B, ldb,
                                             m_block, k_block, n_block,
                                             /*accumulate=*/1);
        return;
    }

    /* Zero del tile de C antes del microkernel: el microkernel acumula
     * sobre C, asi que para que la rama overwrite produzca exactamente
     * A * B hay que partir desde C = 0. */
    for (size_t i = 0; i < m_block; ++i) {
        scalar_t *crow = &C[i * ldc];
        for (size_t j = 0; j < n_block; ++j) crow[j] = (scalar_t)0;
    }

    /* Dispatch del microkernel sobre cada tile 4x32. La lda del
     * microkernel es k_block (row stride de A_local); ldb / ldc son
     * las del caller (apuntan al B y C globales). */
    for (size_t ii = 0; ii < m_block; ii += MR) {
        for (size_t jj = 0; jj < n_block; jj += NR) {
            LEAF_KERNEL(&C[ii * ldc + jj], ldc,
                        &A_local_scratch[ii * k_block], k_block,
                        &B[jj], ldb,
                        k_block);
        }
    }
}

/*
kernel_base_morton_avx512_add: Variante accumulate. Igual que la
anterior pero SIN zerear C, asi el microkernel acumula sobre el valor
previo de C (usado cuando la recursion split en k).
*/
static void kernel_base_morton_avx512_add(
    scalar_t *C, const scalar_t *A_morton, const scalar_t *B,
    size_t m_block, size_t k_block, size_t n_block,
    size_t a_morton_offset, size_t a_block_dim,
    size_t ldc, size_t ldb,
    scalar_t *A_local_scratch)
{
    materialize_a_panel(A_morton, a_morton_offset, a_block_dim,
                        A_local_scratch, m_block, k_block);

    if ((m_block % MR) != 0 || (n_block % NR) != 0) {
        kernel_base_morton_avx512_ijk_fallback(C, ldc,
                                             A_local_scratch, k_block,
                                             B, ldb,
                                             m_block, k_block, n_block,
                                             /*accumulate=*/1);
        return;
    }

    for (size_t ii = 0; ii < m_block; ii += MR) {
        for (size_t jj = 0; jj < n_block; jj += NR) {
            LEAF_KERNEL(&C[ii * ldc + jj], ldc,
                        &A_local_scratch[ii * k_block], k_block,
                        &B[jj], ldb,
                        k_block);
        }
    }
}

/* ================================================================== */
/* Recursion                                                          */
/*                                                                    */
/* Misma forma que matmul_morton_inner / _inner_add: split de n        */
/* cuando es la dimension mas grande; si no, split de m y k juntos en  */
/* los 4 cuadrantes Morton de A. El scratch A_local se hilvana a       */
/* traves de toda la recursion => las hojas no hacen malloc.           */
/* ================================================================== */

static void matmul_morton_avx512_inner(scalar_t *C,
                                     const scalar_t *A_morton,
                                     const scalar_t *B,
                                     size_t m_block, size_t k_block,
                                     size_t n_block,
                                     size_t a_morton_offset,
                                     size_t a_block_dim,
                                     size_t ldc, size_t ldb,
                                     scalar_t *A_local_scratch);

static void matmul_morton_avx512_inner_add(scalar_t *C,
                                         const scalar_t *A_morton,
                                         const scalar_t *B,
                                         size_t m_block, size_t k_block,
                                         size_t n_block,
                                         size_t a_morton_offset,
                                         size_t a_block_dim,
                                         size_t ldc, size_t ldb,
                                         scalar_t *A_local_scratch);

static void matmul_morton_avx512_inner(scalar_t *C,
                                     const scalar_t *A_morton,
                                     const scalar_t *B,
                                     size_t m_block, size_t k_block,
                                     size_t n_block,
                                     size_t a_morton_offset,
                                     size_t a_block_dim,
                                     size_t ldc, size_t ldb,
                                     scalar_t *A_local_scratch)
{
    if (m_block * k_block * n_block <= g_recursion_threshold_avx512) {
        kernel_base_morton_avx512(C, A_morton, B,
                                m_block, k_block, n_block,
                                a_morton_offset, a_block_dim,
                                ldc, ldb,
                                A_local_scratch);
        return;
    }

    /* Caso N: dividir n. A es compartida entre las dos llamadas. */
    if (n_block > a_block_dim && n_block >= 2) {
        size_t n_half = n_block / 2;
        matmul_morton_avx512_inner(C,          A_morton, B,
                                 m_block, k_block, n_half,
                                 a_morton_offset, a_block_dim,
                                 ldc, ldb, A_local_scratch);
        matmul_morton_avx512_inner(C + n_half, A_morton, B + n_half,
                                 m_block, k_block, n_block - n_half,
                                 a_morton_offset, a_block_dim,
                                 ldc, ldb, A_local_scratch);
        return;
    }

    /* Caso MK: dividir m y k juntos en los 4 cuadrantes Morton de A.
     * Con el layout Morton-de-bloques los cuadrantes siguen ocupando 4
     * segmentos consecutivos de (half*half) floats: el Z-order se
     * calcula sobre coordenadas de bloque, pero el tamano de cada
     * bloque es constante (MORTON_AVX512_TILE^2), asi que los offsets
     * escalan igual que en el caso Morton-de-elementos. */
    if (a_block_dim >= 2) {
        assert(m_block == k_block);
        assert(m_block == a_block_dim);

        size_t half          = a_block_dim / 2;
        size_t quadrant_size = half * half;

        /* C_top = A_TL * B_top  (overwrite) */
        matmul_morton_avx512_inner(C, A_morton, B,
                                 half, half, n_block,
                                 a_morton_offset + (size_t)0 * quadrant_size,
                                 half, ldc, ldb, A_local_scratch);

        /* C_top += A_TR * B_bot (accumulate) */
        matmul_morton_avx512_inner_add(C, A_morton, B + half * ldb,
                                     half, half, n_block,
                                     a_morton_offset + (size_t)1 * quadrant_size,
                                     half, ldc, ldb, A_local_scratch);

        /* C_bot = A_BL * B_top  (overwrite) */
        matmul_morton_avx512_inner(C + half * ldc, A_morton, B,
                                 half, half, n_block,
                                 a_morton_offset + (size_t)2 * quadrant_size,
                                 half, ldc, ldb, A_local_scratch);

        /* C_bot += A_BR * B_bot (accumulate) */
        matmul_morton_avx512_inner_add(C + half * ldc, A_morton, B + half * ldb,
                                     half, half, n_block,
                                     a_morton_offset + (size_t)3 * quadrant_size,
                                     half, ldc, ldb, A_local_scratch);
        return;
    }

    /* Fallback degenerado. */
    kernel_base_morton_avx512(C, A_morton, B,
                            m_block, k_block, n_block,
                            a_morton_offset, a_block_dim,
                            ldc, ldb, A_local_scratch);
}

static void matmul_morton_avx512_inner_add(scalar_t *C,
                                         const scalar_t *A_morton,
                                         const scalar_t *B,
                                         size_t m_block, size_t k_block,
                                         size_t n_block,
                                         size_t a_morton_offset,
                                         size_t a_block_dim,
                                         size_t ldc, size_t ldb,
                                         scalar_t *A_local_scratch)
{
    if (m_block * k_block * n_block <= g_recursion_threshold_avx512) {
        kernel_base_morton_avx512_add(C, A_morton, B,
                                    m_block, k_block, n_block,
                                    a_morton_offset, a_block_dim,
                                    ldc, ldb,
                                    A_local_scratch);
        return;
    }

    if (n_block > a_block_dim && n_block >= 2) {
        size_t n_half = n_block / 2;
        matmul_morton_avx512_inner_add(C,          A_morton, B,
                                     m_block, k_block, n_half,
                                     a_morton_offset, a_block_dim,
                                     ldc, ldb, A_local_scratch);
        matmul_morton_avx512_inner_add(C + n_half, A_morton, B + n_half,
                                     m_block, k_block, n_block - n_half,
                                     a_morton_offset, a_block_dim,
                                     ldc, ldb, A_local_scratch);
        return;
    }

    if (a_block_dim >= 2) {
        assert(m_block == k_block);
        assert(m_block == a_block_dim);

        size_t half          = a_block_dim / 2;
        size_t quadrant_size = half * half;

        /* Los 4 productos acumulan: ya estamos dentro de la rama _add,
         * asi que C trae un valor previo que cada producto debe
         * preservar sumandose. */
        matmul_morton_avx512_inner_add(C, A_morton, B,
                                     half, half, n_block,
                                     a_morton_offset + (size_t)0 * quadrant_size,
                                     half, ldc, ldb, A_local_scratch);
        matmul_morton_avx512_inner_add(C, A_morton, B + half * ldb,
                                     half, half, n_block,
                                     a_morton_offset + (size_t)1 * quadrant_size,
                                     half, ldc, ldb, A_local_scratch);
        matmul_morton_avx512_inner_add(C + half * ldc, A_morton, B,
                                     half, half, n_block,
                                     a_morton_offset + (size_t)2 * quadrant_size,
                                     half, ldc, ldb, A_local_scratch);
        matmul_morton_avx512_inner_add(C + half * ldc, A_morton, B + half * ldb,
                                     half, half, n_block,
                                     a_morton_offset + (size_t)3 * quadrant_size,
                                     half, ldc, ldb, A_local_scratch);
        return;
    }

    kernel_base_morton_avx512_add(C, A_morton, B,
                                m_block, k_block, n_block,
                                a_morton_offset, a_block_dim,
                                ldc, ldb, A_local_scratch);
}

/* ================================================================== */
/* Wrapper publico                                                    */
/* ================================================================== */

/*
scratch_side_for_threshold: Devuelve el lado maximo de panel que el
recursion-stop puede producir en la hoja con el threshold actual. El
buffer A_local dimensionado a side^2 es suficiente para cualquier hoja
que la recursion genere antes de que el threshold la corte.

Usamos n_floor = NR = 32 como el n minimo que una hoja puede tener sin
caer al fallback => side ~= sqrt(threshold / NR), redondeado arriba a
potencia de 2 con minimo 64.
*/
static size_t scratch_side_for_threshold(size_t threshold)
{
    size_t target = threshold / (size_t)NR;
    size_t side = 1;
    while (side * side < target) side <<= 1;
    if (side < 64) side = 64;
    return side;
}

void matmul_morton_avx512(scalar_t *C,
                        const scalar_t *A_morton,
                        const scalar_t *B,
                        size_t m, size_t k, size_t n)
{
    if (m != k) {
        fprintf(stderr,
                "Error in matmul_morton_avx512: A must be square "
                "(got m=%llu, k=%llu).\n",
                (unsigned long long)m, (unsigned long long)k);
        exit(EXIT_FAILURE);
    }
    if (!is_power_of_two(m) || m < (size_t)MR) {
        fprintf(stderr,
                "Error in matmul_morton_avx512: m (%llu) must be a power "
                "of two and at least MR = %d.\n",
                (unsigned long long)m, MR);
        exit(EXIT_FAILURE);
    }

    size_t side = scratch_side_for_threshold(g_recursion_threshold_avx512);
    /* Clamp al tamano real del problema para no allocar mas que m*m. */
    if (side > m) side = m;
    scalar_t *A_local = xalloc_aligned(side * side);

    matmul_morton_avx512_inner(C, A_morton, B,
                             m, k, n,
                             /* a_morton_offset = */ 0,
                             /* a_block_dim     = */ m,
                             /* ldc = */ n,
                             /* ldb = */ n,
                             A_local);

    xfree(A_local);
}

/* ================================================================== */
/* Orquestadores de benchmark                                         */
/* ================================================================== */

void benchmark_iterations_morton_avx512(scalar_t *B_out,
                                      const scalar_t *A,
                                      const scalar_t *Z,
                                      size_t m, size_t n,
                                      size_t num_iters)
{
    scalar_t *A_morton = xalloc_aligned(m * m);
    reorganize_to_morton_blocks(A, A_morton, m);

    benchmark_iterations_morton_avx512_preorganized(B_out, A_morton, Z,
                                                  m, n, num_iters);

    xfree(A_morton);
}

void benchmark_iterations_morton_avx512_preorganized(scalar_t *B_out,
                                                   const scalar_t *A_morton,
                                                   const scalar_t *Z,
                                                   size_t m, size_t n,
                                                   size_t num_iters)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        matmul_morton_avx512(B_next, A_morton, B_curr, m, m, n);

        /* Almacena las primeras n filas de B_next en el buffer de salida. */
        scalar_t *out_block = B_out + iter * n * n;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                out_block[i * n + j] = B_next[i * n + j];
            }
        }

        /* Swap: la proxima iteracion consume lo que acabamos de producir. */
        scalar_t *tmp = B_curr;
        B_curr = B_next;
        B_next = tmp;
    }

    xfree(B_curr);
    xfree(B_next);
}
