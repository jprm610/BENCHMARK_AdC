// matmul_morton.c - Implementacion del matmul recursivo con A en
// layout Morton. Dos ramas paralelas (overwrite y _add) y un leaf
// escalar ijk.

#include "matmul_morton.h"
#include "morton.h"
#include "matrix_utils.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
Invariantes que sostiene la recursion:
- A es un sub-bloque cuadrado de lado a_block_dim (potencia de 2) que
  ocupa el segmento contiguo
      A_morton[a_morton_offset .. a_morton_offset + a_block_dim^2).
- El sub-problema actual opera sobre m_block filas y k_block columnas
  de ese sub-bloque. Se mantiene m_block == k_block == a_block_dim en
  cada llamada productiva (las division por 2 mantienen la igualdad).
- B y C son sub-bloques row-major con leading dimensions ldb y ldc
  (las leading dimensions de las matrices ORIGINALES, no de los
  sub-bloques).

Casos:
- Caso N: cuando n_block > a_block_dim, se divide n. A es compartida.
- Caso MK: cuando a_block_dim >= 2, se divide A en sus 4 cuadrantes
  Morton; sus offsets son a_morton_offset + {0,1,2,3} * (half*half).
- Leaf: m_block * k_block * n_block <= g_recursion_threshold, o el
  caso degenerado a_block_dim == 1 con n_block == 1.

Ramas:
- plain (matmul_morton_inner, kernel_base_morton): sobrescribe C.
- _add (matmul_morton_inner_add, kernel_base_morton_add): acumula
  en C. La usa el segundo, tercer y cuarto producto del split MK, y
  cada llamada dentro de la rama _add propaga _add a sus hijos.
*/

size_t g_recursion_threshold = (size_t)32 * 32 * 128;   /* = 131072 */

void matmul_morton_set_threshold(size_t threshold)
{
    /*
    Guard contra threshold == 0: un threshold cero significa "nunca
    recursar", lo que tira el problema entero al leaf con las
    dimensiones globales y derrota el proposito de la rutina. Se trata
    como un bug del caller => warning + mantener el valor previo.
    */
    if (threshold == 0) {
        fprintf(stderr,
                "Warning: matmul_morton_set_threshold(0) ignored; "
                "keeping previous threshold (%llu).\n",
                (unsigned long long)g_recursion_threshold);
        return;
    }
    g_recursion_threshold = threshold;
}

/* Forward declarations de las funciones internas. */
static void matmul_morton_inner(scalar_t *C,
                                const scalar_t *A_morton,
                                const scalar_t *B,
                                size_t m_block, size_t k_block, size_t n_block,
                                size_t a_morton_offset,
                                size_t a_block_dim,
                                size_t ldc, size_t ldb);

static void matmul_morton_inner_add(scalar_t *C,
                                    const scalar_t *A_morton,
                                    const scalar_t *B,
                                    size_t m_block, size_t k_block, size_t n_block,
                                    size_t a_morton_offset,
                                    size_t a_block_dim,
                                    size_t ldc, size_t ldb);

static void kernel_base_morton(scalar_t *C,
                               const scalar_t *A_morton,
                               const scalar_t *B,
                               size_t m_block, size_t k_block, size_t n_block,
                               size_t a_morton_offset,
                               size_t a_block_dim,
                               size_t ldc, size_t ldb);

static void kernel_base_morton_add(scalar_t *C,
                                   const scalar_t *A_morton,
                                   const scalar_t *B,
                                   size_t m_block, size_t k_block, size_t n_block,
                                   size_t a_morton_offset,
                                   size_t a_block_dim,
                                   size_t ldc, size_t ldb);

/* ================================================================== */
/* Wrapper publico                                                    */
/* ================================================================== */

void matmul_morton(scalar_t *C,
                   const scalar_t *A_morton,
                   const scalar_t *B,
                   size_t m, size_t k, size_t n)
{
    if (m != k) {
        fprintf(stderr,
                "Error in matmul_morton: A must be square (got m=%llu, k=%llu).\n",
                (unsigned long long)m, (unsigned long long)k);
        exit(EXIT_FAILURE);
    }
    if (!is_power_of_two(m)) {
        fprintf(stderr,
                "Error in matmul_morton: m must be a power of two (got %llu).\n",
                (unsigned long long)m);
        exit(EXIT_FAILURE);
    }

    /* Lanzar la recursion con offset 0 y a_block_dim = m. */
    matmul_morton_inner(C, A_morton, B,
                        m, k, n,
                        /* a_morton_offset = */ 0,
                        /* a_block_dim     = */ m,
                        /* ldc = */ n,
                        /* ldb = */ n);
}

/* ================================================================== */
/* Rama overwrite                                                     */
/* ================================================================== */

static void matmul_morton_inner(scalar_t *C,
                                const scalar_t *A_morton,
                                const scalar_t *B,
                                size_t m_block, size_t k_block, size_t n_block,
                                size_t a_morton_offset,
                                size_t a_block_dim,
                                size_t ldc, size_t ldb)
{
    if (m_block * k_block * n_block <= g_recursion_threshold) {
        kernel_base_morton(C, A_morton, B,
                           m_block, k_block, n_block,
                           a_morton_offset, a_block_dim,
                           ldc, ldb);
        return;
    }

    /* Caso N: dividir n. A es compartida; las dos mitades de C son
     * disjuntas, asi que ambas pueden sobrescribir sin colisionar. */
    if (n_block > a_block_dim && n_block >= 2) {
        size_t n_half = n_block / 2;
        matmul_morton_inner(C,          A_morton, B,
                            m_block, k_block, n_half,
                            a_morton_offset, a_block_dim,
                            ldc, ldb);
        matmul_morton_inner(C + n_half, A_morton, B + n_half,
                            m_block, k_block, n_block - n_half,
                            a_morton_offset, a_block_dim,
                            ldc, ldb);
        return;
    }

    /* Caso MK: dividir A en sus 4 cuadrantes Morton. m_block == k_block
     * == a_block_dim se mantiene porque la division es exacta (potencia
     * de 2). */
    if (a_block_dim >= 2) {
        assert(m_block == k_block);
        assert(m_block == a_block_dim);

        size_t half          = a_block_dim / 2;
        size_t quadrant_size = half * half;

        /* C_top = A_TL * B_top  (overwrite)
         * Offset 0 = cuadrante top-left por la convencion Morton. */
        matmul_morton_inner(C, A_morton, B,
                            half, half, n_block,
                            a_morton_offset + (size_t)0 * quadrant_size,
                            half, ldc, ldb);

        /* C_top += A_TR * B_bot (accumulate sobre lo que escribio TL)
         * Offset 1 = top-right; B_bot = B + half*ldb. */
        matmul_morton_inner_add(C, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)1 * quadrant_size,
                                half, ldc, ldb);

        /* C_bot = A_BL * B_top  (overwrite; C_bot disjoint de C_top)
         * Offset 2 = bottom-left. */
        matmul_morton_inner(C + half * ldc, A_morton, B,
                            half, half, n_block,
                            a_morton_offset + (size_t)2 * quadrant_size,
                            half, ldc, ldb);

        /* C_bot += A_BR * B_bot (accumulate sobre lo que escribio BL)
         * Offset 3 = bottom-right. */
        matmul_morton_inner_add(C + half * ldc, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)3 * quadrant_size,
                                half, ldc, ldb);
        return;
    }

    /* Fallback degenerado: a_block_dim == 1 y n_block == 1. El leaf
     * maneja correctamente sub-bloques 1x1x1. */
    kernel_base_morton(C, A_morton, B,
                       m_block, k_block, n_block,
                       a_morton_offset, a_block_dim,
                       ldc, ldb);
}

/* ================================================================== */
/* Rama accumulate                                                    */
/* ================================================================== */

static void matmul_morton_inner_add(scalar_t *C,
                                    const scalar_t *A_morton,
                                    const scalar_t *B,
                                    size_t m_block, size_t k_block, size_t n_block,
                                    size_t a_morton_offset,
                                    size_t a_block_dim,
                                    size_t ldc, size_t ldb)
{
    if (m_block * k_block * n_block <= g_recursion_threshold) {
        kernel_base_morton_add(C, A_morton, B,
                               m_block, k_block, n_block,
                               a_morton_offset, a_block_dim,
                               ldc, ldb);
        return;
    }

    if (n_block > a_block_dim && n_block >= 2) {
        size_t n_half = n_block / 2;
        matmul_morton_inner_add(C,          A_morton, B,
                                m_block, k_block, n_half,
                                a_morton_offset, a_block_dim,
                                ldc, ldb);
        matmul_morton_inner_add(C + n_half, A_morton, B + n_half,
                                m_block, k_block, n_block - n_half,
                                a_morton_offset, a_block_dim,
                                ldc, ldb);
        return;
    }

    if (a_block_dim >= 2) {
        assert(m_block == k_block);
        assert(m_block == a_block_dim);

        size_t half          = a_block_dim / 2;
        size_t quadrant_size = half * half;

        /* Los cuatro productos acumulan: ya estamos dentro de la rama
         * _add, asi que C trae un valor previo que cada producto debe
         * preservar sumandose. */
        matmul_morton_inner_add(C, A_morton, B,
                                half, half, n_block,
                                a_morton_offset + (size_t)0 * quadrant_size,
                                half, ldc, ldb);
        matmul_morton_inner_add(C, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)1 * quadrant_size,
                                half, ldc, ldb);
        matmul_morton_inner_add(C + half * ldc, A_morton, B,
                                half, half, n_block,
                                a_morton_offset + (size_t)2 * quadrant_size,
                                half, ldc, ldb);
        matmul_morton_inner_add(C + half * ldc, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)3 * quadrant_size,
                                half, ldc, ldb);
        return;
    }

    kernel_base_morton_add(C, A_morton, B,
                           m_block, k_block, n_block,
                           a_morton_offset, a_block_dim,
                           ldc, ldb);
}

/* ================================================================== */
/* Kernels base (leaf)                                                */
/*                                                                    */
/* Para cada (i, k) dentro del sub-bloque actual, el elemento de A    */
/* vive en A_morton[a_morton_offset + morton_encode(i, k)], donde i,  */
/* k son indices LOCALES (0..m_block, 0..k_block). El invariante      */
/* m_block == k_block == a_block_dim garantiza que morton_encode(i, k)*/
/* nunca sale del segmento del sub-bloque.                            */
/* ================================================================== */

static void kernel_base_morton(scalar_t *C,
                               const scalar_t *A_morton,
                               const scalar_t *B,
                               size_t m_block, size_t k_block, size_t n_block,
                               size_t a_morton_offset,
                               size_t a_block_dim,
                               size_t ldc, size_t ldb)
{
    /* a_block_dim solo se usa para el invariante; el leaf no lo lee. */
    (void)a_block_dim;

    for (size_t i = 0; i < m_block; ++i) {
        for (size_t j = 0; j < n_block; ++j) {
            scalar_t sum = (scalar_t)0;
            for (size_t k = 0; k < k_block; ++k) {
                uint64_t a_idx = a_morton_offset
                               + morton_encode((uint32_t)i, (uint32_t)k);
                sum += A_morton[a_idx] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;  /* overwrite */
        }
    }
}

static void kernel_base_morton_add(scalar_t *C,
                                   const scalar_t *A_morton,
                                   const scalar_t *B,
                                   size_t m_block, size_t k_block, size_t n_block,
                                   size_t a_morton_offset,
                                   size_t a_block_dim,
                                   size_t ldc, size_t ldb)
{
    (void)a_block_dim;

    for (size_t i = 0; i < m_block; ++i) {
        for (size_t j = 0; j < n_block; ++j) {
            scalar_t sum = (scalar_t)0;
            for (size_t k = 0; k < k_block; ++k) {
                uint64_t a_idx = a_morton_offset
                               + morton_encode((uint32_t)i, (uint32_t)k);
                sum += A_morton[a_idx] * B[k * ldb + j];
            }
            C[i * ldc + j] += sum;  /* accumulate */
        }
    }
}

/* ================================================================== */
/* Orquestadores de benchmark                                         */
/* ================================================================== */

void benchmark_iterations_morton(scalar_t *B_out,
                                 const scalar_t *A,
                                 const scalar_t *Z,
                                 size_t m, size_t n,
                                 size_t num_iters)
{
    /* Reorganiza A a Morton una vez; la recurrencia la reusa. */
    scalar_t *A_morton = xalloc_aligned(m * m);
    reorganize_to_morton(A, A_morton, m);

    benchmark_iterations_morton_preorganized(B_out, A_morton, Z,
                                             m, n, num_iters);

    xfree(A_morton);
}

void benchmark_iterations_morton_preorganized(scalar_t *B_out,
                                              const scalar_t *A_morton,
                                              const scalar_t *Z,
                                              size_t m, size_t n,
                                              size_t num_iters)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        /* B_next = A * B_curr (A es m x m, B_curr es m x n). */
        matmul_morton(B_next, A_morton, B_curr, m, m, n);

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
