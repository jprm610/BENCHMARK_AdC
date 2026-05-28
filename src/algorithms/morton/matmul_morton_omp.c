// matmul_morton_omp.c - Implementacion paralela con OpenMP tasks de la
// recursion Morton + microkernel AVX2 4x16.
//
// Los helpers del leaf (materialize_a_panel, kernel_base_morton_omp,
// _add, _ijk_fallback) estan DUPLICADOS aqui (no incluidos desde el
// .c de la Fase 1.7) para que este modulo sea standalone: el AVX2
// serial queda exactamente como Prompt 4 lo dejo, y ningun static
// helper cruza la frontera del archivo.
//
// Paralelizacion:
// - Cada sub-problema > g_parallel_threshold_omp crea 2 tasks OMP en
//   el split. Sub-problemas <= a ese threshold recursan inline => evita
//   el overhead per-task en el leaf.
// - El split MK produce 4 sub-productos. Los dos a C_top deben ejecutar
//   en orden (TL escribe, TR acumula sobre TL); idem C_bot. Pero top y
//   bot son DISJUNTOS => 1 task para la secuencia top, 1 para la
//   secuencia bot, taskwait.
// - El split N produce 2 sub-productos totalmente independientes => 1
//   task por mitad, taskwait.
//
// Scratch buffers:
// - Cada leaf materializa un panel row-major de A en un buffer
//   per-thread. El pool se aloja una vez en el wrapper publico usando
//   omp_get_max_threads() y se indexa en la hoja por
//   omp_get_thread_num(). Con 12 threads y panel 64x64 (16 KiB) el
//   pool total es ~192 KiB, despreciable.

#include "matmul_morton_omp.h"

#include "morton.h"             /* morton_encode, is_power_of_two */
#include "kernel_avx2_morton.h" /* kernel_avx2_4x16, KERNEL_AVX2_MR/NR */
#include "matrix_utils.h"       /* xalloc_aligned, xfree */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#define MR KERNEL_AVX2_MR   /* 4 */
#define NR KERNEL_AVX2_NR   /* 16 */

/* ================================================================== */
/* Tunables                                                            */
/* ================================================================== */

size_t g_recursion_threshold_omp = (size_t)64 * 64 * 128;  /* 524288 */
size_t g_parallel_threshold_omp  = (size_t)64 * 64 * 128;  /* 524288 */

void matmul_morton_omp_set_threshold(size_t threshold)
{
    if (threshold == 0) {
        fprintf(stderr,
                "Warning: matmul_morton_omp_set_threshold(0) ignored; "
                "keeping previous threshold (%llu).\n",
                (unsigned long long)g_recursion_threshold_omp);
        return;
    }
    g_recursion_threshold_omp = threshold;
}

void matmul_morton_omp_set_parallel_threshold(size_t threshold)
{
    /* Zero esta PERMITIDO aqui y significa "siempre serial" (todo
     * sub-problema queda <= threshold). Util para medir el baseline
     * serial de este modulo sin cambiar el binario. */
    g_parallel_threshold_omp = threshold;
}

/* ================================================================== */
/* Helpers del leaf (duplicados a proposito desde matmul_morton_avx2.c)*/
/* ================================================================== */

static void materialize_a_panel(const scalar_t *A_morton,
                                size_t a_morton_offset,
                                size_t a_block_dim,
                                scalar_t *A_local,
                                size_t m_block, size_t k_block)
{
    (void)a_block_dim;

    const size_t blocks_m = m_block / MORTON_AVX2_TILE;
    const size_t blocks_k = k_block / MORTON_AVX2_TILE;
    const size_t tile_sq  = (size_t)MORTON_AVX2_TILE * MORTON_AVX2_TILE;

    for (size_t bi = 0; bi < blocks_m; ++bi) {
        for (size_t bj = 0; bj < blocks_k; ++bj) {
            uint64_t bcode = morton_encode((uint32_t)bi, (uint32_t)bj);
            size_t block_offset = a_morton_offset
                                + (size_t)bcode * tile_sq;

            for (size_t ii = 0; ii < MORTON_AVX2_TILE; ++ii) {
                size_t row_local = bi * MORTON_AVX2_TILE + ii;
                const scalar_t *src = &A_morton[block_offset
                                              + ii * MORTON_AVX2_TILE];
                scalar_t *dst = &A_local[row_local * k_block
                                       + bj * MORTON_AVX2_TILE];
                for (size_t jj = 0; jj < MORTON_AVX2_TILE; ++jj) {
                    dst[jj] = src[jj];
                }
            }
        }
    }
}

/* Camino lento para hojas con dimensiones no alineadas al tile. */
static void kernel_base_morton_omp_ijk_fallback(
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

/* Variante overwrite del leaf: zerea C y luego deja al microkernel
 * acumular sobre el (resultado final = A * B exactamente). */
static void kernel_base_morton_omp(scalar_t *C,
                                   const scalar_t *A_morton,
                                   const scalar_t *B,
                                   size_t m_block, size_t k_block,
                                   size_t n_block,
                                   size_t a_morton_offset,
                                   size_t a_block_dim,
                                   size_t ldc, size_t ldb,
                                   scalar_t *A_local_scratch)
{
    materialize_a_panel(A_morton, a_morton_offset, a_block_dim,
                        A_local_scratch, m_block, k_block);

    if ((m_block % MR) != 0 || (n_block % NR) != 0) {
        for (size_t i = 0; i < m_block; ++i) {
            scalar_t *crow = &C[i * ldc];
            for (size_t j = 0; j < n_block; ++j) crow[j] = (scalar_t)0;
        }
        kernel_base_morton_omp_ijk_fallback(C, ldc,
                                            A_local_scratch, k_block,
                                            B, ldb,
                                            m_block, k_block, n_block,
                                            /*accumulate=*/1);
        return;
    }

    for (size_t i = 0; i < m_block; ++i) {
        scalar_t *crow = &C[i * ldc];
        for (size_t j = 0; j < n_block; ++j) crow[j] = (scalar_t)0;
    }

    for (size_t ii = 0; ii < m_block; ii += MR) {
        for (size_t jj = 0; jj < n_block; jj += NR) {
            kernel_avx2_4x16(&C[ii * ldc + jj], ldc,
                             &A_local_scratch[ii * k_block], k_block,
                             &B[jj], ldb,
                             k_block);
        }
    }
}

/* Variante accumulate del leaf: no zerea C => acumula sobre el valor
 * previo. */
static void kernel_base_morton_omp_add(scalar_t *C,
                                       const scalar_t *A_morton,
                                       const scalar_t *B,
                                       size_t m_block, size_t k_block,
                                       size_t n_block,
                                       size_t a_morton_offset,
                                       size_t a_block_dim,
                                       size_t ldc, size_t ldb,
                                       scalar_t *A_local_scratch)
{
    materialize_a_panel(A_morton, a_morton_offset, a_block_dim,
                        A_local_scratch, m_block, k_block);

    if ((m_block % MR) != 0 || (n_block % NR) != 0) {
        kernel_base_morton_omp_ijk_fallback(C, ldc,
                                            A_local_scratch, k_block,
                                            B, ldb,
                                            m_block, k_block, n_block,
                                            /*accumulate=*/1);
        return;
    }

    for (size_t ii = 0; ii < m_block; ii += MR) {
        for (size_t jj = 0; jj < n_block; jj += NR) {
            kernel_avx2_4x16(&C[ii * ldc + jj], ldc,
                             &A_local_scratch[ii * k_block], k_block,
                             &B[jj], ldb,
                             k_block);
        }
    }
}

/* ================================================================== */
/* Recursion                                                          */
/* ================================================================== */

static void matmul_morton_omp_inner(scalar_t *C,
                                    const scalar_t *A_morton,
                                    const scalar_t *B,
                                    size_t m_block, size_t k_block,
                                    size_t n_block,
                                    size_t a_morton_offset,
                                    size_t a_block_dim,
                                    size_t ldc, size_t ldb,
                                    scalar_t **scratches);

static void matmul_morton_omp_inner_add(scalar_t *C,
                                        const scalar_t *A_morton,
                                        const scalar_t *B,
                                        size_t m_block, size_t k_block,
                                        size_t n_block,
                                        size_t a_morton_offset,
                                        size_t a_block_dim,
                                        size_t ldc, size_t ldb,
                                        scalar_t **scratches);

/*
mk_top_pair / mk_bot_pair: Pre-empaquetan las dos secuencias del split
MK como helpers, asi cada cuerpo de task es una sola sentencia y el
sitio de spawn queda legible. La secuencia top es (TL overwrite, TR
acumula sobre TL); idem bot.
*/
static void mk_top_pair(scalar_t *C, const scalar_t *A_morton,
                        const scalar_t *B,
                        size_t half, size_t n_block,
                        size_t a_morton_offset, size_t quadrant_size,
                        size_t ldc, size_t ldb,
                        scalar_t **scratches)
{
    /* TL escribe, TR acumula sobre TL. Secuencial. */
    matmul_morton_omp_inner    (C, A_morton, B,
                                half, half, n_block,
                                a_morton_offset + (size_t)0 * quadrant_size,
                                half, ldc, ldb, scratches);
    matmul_morton_omp_inner_add(C, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)1 * quadrant_size,
                                half, ldc, ldb, scratches);
}

static void mk_bot_pair(scalar_t *C, const scalar_t *A_morton,
                        const scalar_t *B,
                        size_t half, size_t n_block,
                        size_t a_morton_offset, size_t quadrant_size,
                        size_t ldc, size_t ldb,
                        scalar_t **scratches)
{
    matmul_morton_omp_inner    (C + half * ldc, A_morton, B,
                                half, half, n_block,
                                a_morton_offset + (size_t)2 * quadrant_size,
                                half, ldc, ldb, scratches);
    matmul_morton_omp_inner_add(C + half * ldc, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)3 * quadrant_size,
                                half, ldc, ldb, scratches);
}

/* Las mismas dos helpers pero para la rama _add: cada producto
 * acumula (el caller ya tiene un C con valor significativo). */
static void mk_top_pair_add(scalar_t *C, const scalar_t *A_morton,
                            const scalar_t *B,
                            size_t half, size_t n_block,
                            size_t a_morton_offset, size_t quadrant_size,
                            size_t ldc, size_t ldb,
                            scalar_t **scratches)
{
    matmul_morton_omp_inner_add(C, A_morton, B,
                                half, half, n_block,
                                a_morton_offset + (size_t)0 * quadrant_size,
                                half, ldc, ldb, scratches);
    matmul_morton_omp_inner_add(C, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)1 * quadrant_size,
                                half, ldc, ldb, scratches);
}

static void mk_bot_pair_add(scalar_t *C, const scalar_t *A_morton,
                            const scalar_t *B,
                            size_t half, size_t n_block,
                            size_t a_morton_offset, size_t quadrant_size,
                            size_t ldc, size_t ldb,
                            scalar_t **scratches)
{
    matmul_morton_omp_inner_add(C + half * ldc, A_morton, B,
                                half, half, n_block,
                                a_morton_offset + (size_t)2 * quadrant_size,
                                half, ldc, ldb, scratches);
    matmul_morton_omp_inner_add(C + half * ldc, A_morton, B + half * ldb,
                                half, half, n_block,
                                a_morton_offset + (size_t)3 * quadrant_size,
                                half, ldc, ldb, scratches);
}

static void matmul_morton_omp_inner(scalar_t *C,
                                    const scalar_t *A_morton,
                                    const scalar_t *B,
                                    size_t m_block, size_t k_block,
                                    size_t n_block,
                                    size_t a_morton_offset,
                                    size_t a_block_dim,
                                    size_t ldc, size_t ldb,
                                    scalar_t **scratches)
{
    if (m_block * k_block * n_block <= g_recursion_threshold_omp) {
        int tid = omp_get_thread_num();
        kernel_base_morton_omp(C, A_morton, B,
                               m_block, k_block, n_block,
                               a_morton_offset, a_block_dim,
                               ldc, ldb,
                               scratches[tid]);
        return;
    }

    /* Decide spawning para este split. Sub-problemas grandes spawnean
     * tasks; los chicos recursan inline para no pagar overhead. */
    const int spawn = (m_block * k_block * n_block
                       > g_parallel_threshold_omp);

    /* Caso N: dividir n. Las dos mitades son totalmente independientes. */
    if (n_block > a_block_dim && n_block >= 2) {
        size_t n_half = n_block / 2;
        if (spawn) {
            #pragma omp task default(shared) firstprivate(C, B, n_half,           \
                m_block, k_block, n_block, a_morton_offset, a_block_dim,         \
                ldc, ldb)
            matmul_morton_omp_inner(C, A_morton, B,
                                    m_block, k_block, n_half,
                                    a_morton_offset, a_block_dim,
                                    ldc, ldb, scratches);
            #pragma omp task default(shared) firstprivate(C, B, n_half,           \
                m_block, k_block, n_block, a_morton_offset, a_block_dim,         \
                ldc, ldb)
            matmul_morton_omp_inner(C + n_half, A_morton, B + n_half,
                                    m_block, k_block, n_block - n_half,
                                    a_morton_offset, a_block_dim,
                                    ldc, ldb, scratches);
            #pragma omp taskwait
        } else {
            matmul_morton_omp_inner(C, A_morton, B,
                                    m_block, k_block, n_half,
                                    a_morton_offset, a_block_dim,
                                    ldc, ldb, scratches);
            matmul_morton_omp_inner(C + n_half, A_morton, B + n_half,
                                    m_block, k_block, n_block - n_half,
                                    a_morton_offset, a_block_dim,
                                    ldc, ldb, scratches);
        }
        return;
    }

    /* Caso MK: top y bot de C son independientes; dentro de cada
     * mitad los 2 productos (overwrite + accumulate) deben ir
     * secuenciales. => 1 task top, 1 task bot, taskwait. */
    if (a_block_dim >= 2) {
        assert(m_block == k_block);
        assert(m_block == a_block_dim);

        size_t half          = a_block_dim / 2;
        size_t quadrant_size = half * half;

        if (spawn) {
            #pragma omp task default(shared) firstprivate(C, B, half, n_block,    \
                a_morton_offset, quadrant_size, ldc, ldb)
            mk_top_pair(C, A_morton, B, half, n_block,
                        a_morton_offset, quadrant_size,
                        ldc, ldb, scratches);
            #pragma omp task default(shared) firstprivate(C, B, half, n_block,    \
                a_morton_offset, quadrant_size, ldc, ldb)
            mk_bot_pair(C, A_morton, B, half, n_block,
                        a_morton_offset, quadrant_size,
                        ldc, ldb, scratches);
            #pragma omp taskwait
        } else {
            mk_top_pair(C, A_morton, B, half, n_block,
                        a_morton_offset, quadrant_size,
                        ldc, ldb, scratches);
            mk_bot_pair(C, A_morton, B, half, n_block,
                        a_morton_offset, quadrant_size,
                        ldc, ldb, scratches);
        }
        return;
    }

    /* Fallback degenerado: a_block_dim == 1 y n_block == 1. */
    {
        int tid = omp_get_thread_num();
        kernel_base_morton_omp(C, A_morton, B,
                               m_block, k_block, n_block,
                               a_morton_offset, a_block_dim,
                               ldc, ldb,
                               scratches[tid]);
    }
}

static void matmul_morton_omp_inner_add(scalar_t *C,
                                        const scalar_t *A_morton,
                                        const scalar_t *B,
                                        size_t m_block, size_t k_block,
                                        size_t n_block,
                                        size_t a_morton_offset,
                                        size_t a_block_dim,
                                        size_t ldc, size_t ldb,
                                        scalar_t **scratches)
{
    if (m_block * k_block * n_block <= g_recursion_threshold_omp) {
        int tid = omp_get_thread_num();
        kernel_base_morton_omp_add(C, A_morton, B,
                                   m_block, k_block, n_block,
                                   a_morton_offset, a_block_dim,
                                   ldc, ldb,
                                   scratches[tid]);
        return;
    }

    const int spawn = (m_block * k_block * n_block
                       > g_parallel_threshold_omp);

    if (n_block > a_block_dim && n_block >= 2) {
        size_t n_half = n_block / 2;
        if (spawn) {
            #pragma omp task default(shared) firstprivate(C, B, n_half,           \
                m_block, k_block, n_block, a_morton_offset, a_block_dim,         \
                ldc, ldb)
            matmul_morton_omp_inner_add(C, A_morton, B,
                                        m_block, k_block, n_half,
                                        a_morton_offset, a_block_dim,
                                        ldc, ldb, scratches);
            #pragma omp task default(shared) firstprivate(C, B, n_half,           \
                m_block, k_block, n_block, a_morton_offset, a_block_dim,         \
                ldc, ldb)
            matmul_morton_omp_inner_add(C + n_half, A_morton, B + n_half,
                                        m_block, k_block, n_block - n_half,
                                        a_morton_offset, a_block_dim,
                                        ldc, ldb, scratches);
            #pragma omp taskwait
        } else {
            matmul_morton_omp_inner_add(C, A_morton, B,
                                        m_block, k_block, n_half,
                                        a_morton_offset, a_block_dim,
                                        ldc, ldb, scratches);
            matmul_morton_omp_inner_add(C + n_half, A_morton, B + n_half,
                                        m_block, k_block, n_block - n_half,
                                        a_morton_offset, a_block_dim,
                                        ldc, ldb, scratches);
        }
        return;
    }

    if (a_block_dim >= 2) {
        assert(m_block == k_block);
        assert(m_block == a_block_dim);

        size_t half          = a_block_dim / 2;
        size_t quadrant_size = half * half;

        if (spawn) {
            #pragma omp task default(shared) firstprivate(C, B, half, n_block,    \
                a_morton_offset, quadrant_size, ldc, ldb)
            mk_top_pair_add(C, A_morton, B, half, n_block,
                            a_morton_offset, quadrant_size,
                            ldc, ldb, scratches);
            #pragma omp task default(shared) firstprivate(C, B, half, n_block,    \
                a_morton_offset, quadrant_size, ldc, ldb)
            mk_bot_pair_add(C, A_morton, B, half, n_block,
                            a_morton_offset, quadrant_size,
                            ldc, ldb, scratches);
            #pragma omp taskwait
        } else {
            mk_top_pair_add(C, A_morton, B, half, n_block,
                            a_morton_offset, quadrant_size,
                            ldc, ldb, scratches);
            mk_bot_pair_add(C, A_morton, B, half, n_block,
                            a_morton_offset, quadrant_size,
                            ldc, ldb, scratches);
        }
        return;
    }

    {
        int tid = omp_get_thread_num();
        kernel_base_morton_omp_add(C, A_morton, B,
                                   m_block, k_block, n_block,
                                   a_morton_offset, a_block_dim,
                                   ldc, ldb,
                                   scratches[tid]);
    }
}

/* ================================================================== */
/* Wrapper publico                                                    */
/* ================================================================== */

/* Espejo del helper de matmul_morton_avx2.c. Redondea arriba a
 * potencia de 2 con minimo 64, asi que scratch dimensionado a side^2
 * es suficiente para cualquier hoja que la recursion produzca con el
 * threshold actual. */
static size_t scratch_side_for_threshold(size_t threshold)
{
    size_t target = threshold / (size_t)NR;
    size_t side = 1;
    while (side * side < target) side <<= 1;
    if (side < 64) side = 64;
    return side;
}

void matmul_morton_omp(scalar_t *C,
                       const scalar_t *A_morton,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n)
{
    if (m != k) {
        fprintf(stderr,
                "Error in matmul_morton_omp: A must be square "
                "(got m=%llu, k=%llu).\n",
                (unsigned long long)m, (unsigned long long)k);
        exit(EXIT_FAILURE);
    }
    if (!is_power_of_two(m) || m < (size_t)MR) {
        fprintf(stderr,
                "Error in matmul_morton_omp: m (%llu) must be a power "
                "of two and at least MR = %d.\n",
                (unsigned long long)m, MR);
        exit(EXIT_FAILURE);
    }

    /* omp_get_max_threads devuelve la cota superior del team que
     * cualquier region paralela posterior en este thread puede usar.
     * Ese es el tamano correcto para el pool de scratch. */
    const int max_threads = omp_get_max_threads();

    size_t side = scratch_side_for_threshold(g_recursion_threshold_omp);
    if (side > m) side = m;

    scalar_t **scratches = (scalar_t **)malloc(
        (size_t)max_threads * sizeof(scalar_t *));
    if (scratches == NULL) {
        fprintf(stderr,
                "Error in matmul_morton_omp: out of memory for "
                "scratch pool (%d threads).\n", max_threads);
        exit(EXIT_FAILURE);
    }
    for (int t = 0; t < max_threads; ++t) {
        scratches[t] = xalloc_aligned(side * side);
    }

    /* parallel crea el team UNA sola vez; single garantiza que la
     * recursion raiz la dispara un solo thread. Los demas threads del
     * team se quedan disponibles para ejecutar las tasks que el
     * recursion-root va spawneando. */
    #pragma omp parallel default(shared)
    {
        #pragma omp single
        matmul_morton_omp_inner(C, A_morton, B,
                                m, k, n,
                                /* a_morton_offset = */ 0,
                                /* a_block_dim     = */ m,
                                /* ldc = */ n,
                                /* ldb = */ n,
                                scratches);
    }

    for (int t = 0; t < max_threads; ++t) {
        xfree(scratches[t]);
    }
    free(scratches);
}

/* ================================================================== */
/* Orquestadores de benchmark                                         */
/* ================================================================== */

void benchmark_iterations_morton_omp(scalar_t *B_out,
                                     const scalar_t *A,
                                     const scalar_t *Z,
                                     size_t m, size_t n,
                                     size_t num_iters)
{
    scalar_t *A_morton = xalloc_aligned(m * m);
    reorganize_to_morton_blocks(A, A_morton, m);

    benchmark_iterations_morton_omp_preorganized(B_out, A_morton, Z,
                                                 m, n, num_iters);

    xfree(A_morton);
}

void benchmark_iterations_morton_omp_preorganized(scalar_t *B_out,
                                                  const scalar_t *A_morton,
                                                  const scalar_t *Z,
                                                  size_t m, size_t n,
                                                  size_t num_iters)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        matmul_morton_omp(B_next, A_morton, B_curr, m, m, n);

        scalar_t *out_block = B_out + iter * n * n;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                out_block[i * n + j] = B_next[i * n + j];
            }
        }

        scalar_t *tmp = B_curr;
        B_curr = B_next;
        B_next = tmp;
    }

    xfree(B_curr);
    xfree(B_next);
}
