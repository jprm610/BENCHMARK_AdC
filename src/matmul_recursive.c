/*
 * matmul_recursive.c - Cache-oblivious recursive matrix multiplication.
 *
 * The recursion splits the largest of {m, k, n} in half until the work
 * volume m*k*n drops under RECURSION_THRESHOLD, at which point an ijk
 * triple-loop kernel runs on the sub-block. Sub-blocks are addressed via
 * pointer arithmetic plus the leading dimensions (ldc, lda, ldb) of the
 * original matrices, so no data is ever copied.
 *
 * Two flavors of every internal function exist:
 *   - the plain version overwrites C (C = A * B);
 *   - the _add version accumulates into C (C += A * B).
 *
 * The _add variants are needed when the recursion splits k: the two
 * sub-problems both write to the same region of C, so the first one
 * overwrites and the second one accumulates. When the recursion splits m
 * or n, the two sub-problems target disjoint regions of C and both can
 * overwrite their own region.
 */

#include "matmul_recursive.h"
#include "matrix_utils.h"

#include <string.h>

/*
 * Threshold on the work volume m*k*n below which the recursion stops and
 * the base case runs. 32*32*128 = 131072 elementary operations corresponds
 * to a sub-block whose working set comfortably fits in L1 on typical x86
 * cores (32 KB L1d).
 */
#define RECURSION_THRESHOLD ((size_t)32 * 32 * 128)

/* Forward declarations of the internal helpers. */
static void matmul_recursive_inner(scalar_t *C,
                                   const scalar_t *A,
                                   const scalar_t *B,
                                   size_t m, size_t k, size_t n,
                                   size_t ldc, size_t lda, size_t ldb);

static void matmul_recursive_inner_add(scalar_t *C,
                                       const scalar_t *A,
                                       const scalar_t *B,
                                       size_t m, size_t k, size_t n,
                                       size_t ldc, size_t lda, size_t ldb);

static void kernel_base(scalar_t *C,
                        const scalar_t *A,
                        const scalar_t *B,
                        size_t m, size_t k, size_t n,
                        size_t ldc, size_t lda, size_t ldb);

static void kernel_base_add(scalar_t *C,
                            const scalar_t *A,
                            const scalar_t *B,
                            size_t m, size_t k, size_t n,
                            size_t ldc, size_t lda, size_t ldb);

/*
 * Public wrapper. The caller works with contiguous row-major matrices, so
 * the leading dimensions equal the row lengths of the originals: lda = k,
 * ldb = n, ldc = n.
 */
void matmul_recursive(scalar_t *C,
                      const scalar_t *A,
                      const scalar_t *B,
                      size_t m, size_t k, size_t n)
{
    matmul_recursive_inner(C, A, B, m, k, n,
                           /* ldc = */ n,
                           /* lda = */ k,
                           /* ldb = */ n);
}

/*
 * Overwrite version: C[sub-block] = A[sub-block] * B[sub-block].
 *
 * Splits the largest of {m, k, n} in half. When splitting k, the second
 * recursive call must accumulate (uses _add).
 */
static void matmul_recursive_inner(scalar_t *C,
                                   const scalar_t *A,
                                   const scalar_t *B,
                                   size_t m, size_t k, size_t n,
                                   size_t ldc, size_t lda, size_t ldb)
{
    if (m * k * n <= RECURSION_THRESHOLD) {
        kernel_base(C, A, B, m, k, n, ldc, lda, ldb);
        return;
    }

    if (m >= k && m >= n && m >= 2) {
        /* Split rows of C and rows of A; B is shared. The two halves of
         * C are disjoint, so both calls overwrite their own region. */
        size_t m_half = m / 2;
        matmul_recursive_inner(C,
                               A,
                               B,
                               m_half, k, n,
                               ldc, lda, ldb);
        matmul_recursive_inner(C + m_half * ldc,
                               A + m_half * lda,
                               B,
                               m - m_half, k, n,
                               ldc, lda, ldb);
        return;
    }

    if (n >= m && n >= k && n >= 2) {
        /* Split columns of C and columns of B; A is shared. Two halves
         * of C are disjoint, both overwrite. */
        size_t n_half = n / 2;
        matmul_recursive_inner(C,
                               A,
                               B,
                               m, k, n_half,
                               ldc, lda, ldb);
        matmul_recursive_inner(C + n_half,
                               A,
                               B + n_half,
                               m, k, n - n_half,
                               ldc, lda, ldb);
        return;
    }

    if (k >= 2) {
        /* Split columns of A and rows of B; C is shared by both halves.
         * The first call overwrites C, the second accumulates. */
        size_t k_half = k / 2;
        matmul_recursive_inner(C,
                               A,
                               B,
                               m, k_half, n,
                               ldc, lda, ldb);
        matmul_recursive_inner_add(C,
                                   A + k_half,
                                   B + k_half * ldb,
                                   m, k - k_half, n,
                                   ldc, lda, ldb);
        return;
    }

    /* Fallback for degenerate shapes (e.g. m = k = n = 1) that fail every
     * split predicate above. The base kernel handles it correctly. */
    kernel_base(C, A, B, m, k, n, ldc, lda, ldb);
}

/*
 * Accumulate version: C[sub-block] += A[sub-block] * B[sub-block].
 *
 * Same split logic as matmul_recursive_inner, but every leaf and every
 * recursive call accumulates instead of overwriting.
 */
static void matmul_recursive_inner_add(scalar_t *C,
                                       const scalar_t *A,
                                       const scalar_t *B,
                                       size_t m, size_t k, size_t n,
                                       size_t ldc, size_t lda, size_t ldb)
{
    if (m * k * n <= RECURSION_THRESHOLD) {
        kernel_base_add(C, A, B, m, k, n, ldc, lda, ldb);
        return;
    }

    if (m >= k && m >= n && m >= 2) {
        size_t m_half = m / 2;
        matmul_recursive_inner_add(C,
                                   A,
                                   B,
                                   m_half, k, n,
                                   ldc, lda, ldb);
        matmul_recursive_inner_add(C + m_half * ldc,
                                   A + m_half * lda,
                                   B,
                                   m - m_half, k, n,
                                   ldc, lda, ldb);
        return;
    }

    if (n >= m && n >= k && n >= 2) {
        size_t n_half = n / 2;
        matmul_recursive_inner_add(C,
                                   A,
                                   B,
                                   m, k, n_half,
                                   ldc, lda, ldb);
        matmul_recursive_inner_add(C + n_half,
                                   A,
                                   B + n_half,
                                   m, k, n - n_half,
                                   ldc, lda, ldb);
        return;
    }

    if (k >= 2) {
        /* Both halves accumulate into the same region of C (this is the
         * _add variant, so the caller expects accumulation overall). */
        size_t k_half = k / 2;
        matmul_recursive_inner_add(C,
                                   A,
                                   B,
                                   m, k_half, n,
                                   ldc, lda, ldb);
        matmul_recursive_inner_add(C,
                                   A + k_half,
                                   B + k_half * ldb,
                                   m, k - k_half, n,
                                   ldc, lda, ldb);
        return;
    }

    kernel_base_add(C, A, B, m, k, n, ldc, lda, ldb);
}

/*
 * Base case kernel: ijk triple loop on the sub-block. Strides are taken
 * from the leading dimensions of the original matrices so the pointer
 * arithmetic done by the recursion remains valid.
 *
 * Overwrites C[i*ldc + j].
 */
static void kernel_base(scalar_t *C,
                        const scalar_t *A,
                        const scalar_t *B,
                        size_t m, size_t k, size_t n,
                        size_t ldc, size_t lda, size_t ldb)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            scalar_t acc = (scalar_t)0;
            for (size_t p = 0; p < k; ++p) {
                acc += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] = acc;
        }
    }
}

/*
 * Accumulating base case: same as kernel_base but adds to the existing
 * value of C[i*ldc + j]. Used as the leaf when the recursion has split
 * along k (or anywhere inside the _add branch).
 */
static void kernel_base_add(scalar_t *C,
                            const scalar_t *A,
                            const scalar_t *B,
                            size_t m, size_t k, size_t n,
                            size_t ldc, size_t lda, size_t ldb)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            scalar_t acc = (scalar_t)0;
            for (size_t p = 0; p < k; ++p) {
                acc += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] += acc;
        }
    }
}

/*
 * Iterated benchmark orchestrator using matmul_recursive as the kernel.
 * Mirrors benchmark_iterations from matmul_naive.c verbatim: two working
 * buffers of size m*n, swapped by pointer reassignment after each multiply,
 * and the first n rows of every B_{iter+1} copied into B_out.
 */
void benchmark_iterations_recursive(scalar_t *B_out,
                                    const scalar_t *A,
                                    const scalar_t *Z,
                                    size_t m, size_t n,
                                    size_t num_iters)
{
    scalar_t *B_curr = xalloc_aligned(m * n);
    scalar_t *B_next = xalloc_aligned(m * n);

    memcpy(B_curr, Z, m * n * sizeof(scalar_t));

    for (size_t iter = 0; iter < num_iters; ++iter) {
        /* B_next = A * B_curr */
        matmul_recursive(B_next, A, B_curr, m, m, n);

        /* Store the first n rows of B_next into the output buffer. */
        scalar_t *out_block = B_out + iter * n * n;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                out_block[i * n + j] = B_next[i * n + j];
            }
        }

        /* Swap: the new B_curr is the just-computed B_{iter+1}. */
        scalar_t *tmp = B_curr;
        B_curr = B_next;
        B_next = tmp;
    }

    xfree(B_curr);
    xfree(B_next);
}
