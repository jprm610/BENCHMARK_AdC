# Especificacion de la API del benchmark

**Proyecto:** Benchmark de Multiplicacion Iterada de Matrices
**Curso:** Arquitectura de Computadores - UNAL Medellin
**Hardware de referencia:** Ryzen 5 4600H (Renoir, Zen 2)

Este documento describe el contrato publico de las funciones expuestas en `src/`. Firmas en C, nombres y comentarios en ingles. La organizacion del documento sigue las capas del proyecto: `core`, `microkernels`, `algorithms`, `drivers`, `tests`.

El estado consolidado por fase del proyecto (que kernels existen, que sesion los cerro) vive en la seccion 9 del [`README.md`](../README.md). Aqui solo se documenta el contrato API.

---

## 0. Tabla de contenidos

1. [Convenciones generales](#1-convenciones-generales)
2. [Capa `core`](#2-capa-core)
   - 2.1 [`matrix_utils`](#21-matrix_utils)
   - 2.2 [`timing`](#22-timing)
   - 2.3 [`morton`](#23-morton)
3. [Capa `microkernels`](#3-capa-microkernels)
   - 3.1 [`kernel_avx2_morton`](#31-kernel_avx2_morton)
   - 3.2 [`kernel_avx2_tiled`](#32-kernel_avx2_tiled)
4. [Capa `algorithms`](#4-capa-algorithms)
   - 4.1 [`matmul_naive`](#41-matmul_naive)
   - 4.2 [`matmul_loops`](#42-matmul_loops)
   - 4.3 [`matmul_tiled_ikj`](#43-matmul_tiled_ikj)
   - 4.4 [`matmul_tiled_ikj_avx2`](#44-matmul_tiled_ikj_avx2)
   - 4.5 [`matmul_tiled_ikj_omp`](#45-matmul_tiled_ikj_omp)
   - 4.6 [`matmul_morton`](#46-matmul_morton)
   - 4.7 [`matmul_morton_avx2`](#47-matmul_morton_avx2)
   - 4.8 [`matmul_morton_omp`](#48-matmul_morton_omp)
5. [Capa `drivers`](#5-capa-drivers)
   - 5.1 [Convencion comun de los `bench_*`](#51-convencion-comun-de-los-bench_)
   - 5.2 [Convencion comun de los `validate_*`](#52-convencion-comun-de-los-validate_)
   - 5.3 [Tabla unificada de binarios](#53-tabla-unificada-de-binarios)
6. [Capa `tests`](#6-capa-tests)
7. [Versionado del documento](#7-versionado-del-documento)

---

## 1. Convenciones generales

### 1.1 Tipo escalar

Toda la implementacion usa `float` (IEEE 754 binary32, 4 bytes). El tipo se centraliza con un `typedef` en [`src/core/matrix_utils.h`](../src/core/matrix_utils.h):

```c
typedef float scalar_t;
```

Cambiar a `double` requeriria reemplazar el `typedef` y revisar tolerancias en todos los [`src/drivers/validate/`](../src/drivers/validate).

### 1.2 Tipos enteros

Todos los tamanos, conteos e indices usan `size_t`. Para $m = 2^{20}$, $m \cdot m = 2^{40}$ desborda un `int` de 32 bits, por lo que `size_t` es obligatorio.

### 1.3 Layout de matrices

Por defecto **row-major** en buffers planos `scalar_t *`. Una matriz $M \in \mathbb{R}^{r \times c}$ ocupa `r * c` elementos contiguos:

```c
M[(size_t)i * c + j]
```

La familia Morton (Secciones 2.3, 4.6, 4.7, 4.8) introduce un layout alternativo para $A$; en ese caso el caller usa un buffer producido por `reorganize_to_morton*`. $B$ y $C$ siempre son row-major.

### 1.4 Alineacion

Toda matriz se aloja con alineacion de **64 bytes** (linea de cache en x86_64) via `posix_memalign`. El allocador centralizado es `xalloc_aligned` (Seccion 2.1).

### 1.5 Contrato comun de firma

Todo kernel principal expone exactamente esta firma:

```c
void mm(scalar_t *C,
        const scalar_t *A,
        const scalar_t *B,
        size_t m, size_t k, size_t n);
```

con $C$ de salida ($m \times n$, sobrescrito), $A$ de entrada ($m \times k$), $B$ de entrada ($k \times n$), sin aliasing entre los tres. Las variantes que requieren $A$ en un layout distinto (Morton) reciben el buffer ya convertido pero conservan la misma firma. Esta uniformidad permite que el sistema de validacion compare cualquier kernel nuevo contra `matmul_naive` sin cambios estructurales.

### 1.6 Estilo de comentarios y nombres

Comentarios y nombres en ingles, sin emojis ni caracteres no ASCII. Codigo `-Wall -Wextra -Wpedantic` limpio bajo `-std=c11`.

---

## 2. Capa `core`

Modulos compartidos por todos los algoritmos. No dependen de ningun kernel; los kernels dependen de ellos.

### 2.1 `matrix_utils`

**Archivos:** [`src/core/matrix_utils.h`](../src/core/matrix_utils.h), [`src/core/matrix_utils.c`](../src/core/matrix_utils.c).

Alocacion alineada, inicializacion y comparacion. Define el `typedef scalar_t` de la Seccion 1.1.

#### `xalloc_aligned`

```c
scalar_t *xalloc_aligned(size_t num_elements);
```

Aloja `num_elements * sizeof(scalar_t)` bytes con alineacion de 64 bytes via `posix_memalign`. Si la alocacion falla, imprime mensaje a `stderr` y llama a `exit(EXIT_FAILURE)`. El bloque devuelto **no** esta inicializado. El caller libera con `xfree` o `free`.

#### `xfree`

```c
void xfree(scalar_t *ptr);
```

Wrapper sobre `free` que tolera `NULL`. Sirve para uniformizar el ciclo de vida.

#### `init_matrix_random`

```c
void init_matrix_random(scalar_t *M,
                        size_t rows, size_t cols,
                        unsigned int seed);
```

Llena `M` con valores pseudoaleatorios en $[-1/\sqrt{\text{rows}}, +1/\sqrt{\text{rows}}]$. La escala $1/\sqrt{\text{rows}}$ acota la norma espectral de $A$ en $\Theta(1)$ y evita overflow al iterar $B_{i+1} = A \cdot B_i$ muchas veces. Generador lineal congruencial reproducible: la misma `seed` produce siempre los mismos valores. Semillas convencionales del proyecto: $42$ para $A$, $43$ para $Z$.

#### `init_matrix_zero`

```c
void init_matrix_zero(scalar_t *M, size_t rows, size_t cols);
```

Pone `rows * cols` ceros en `M`. Usado por los kernels que acumulan en $C$ (`tiled_ikj`, las variantes `loops` `ikj/jki/kij/kji`) y por la validacion.

#### `init_matrix_identity`

```c
void init_matrix_identity(scalar_t *M, size_t n);
```

Llena `M` ($n \times n$) con la identidad. Util para el invariante $I \cdot Z = Z$ de los validates.

#### `matrices_close`

```c
int matrices_close(const scalar_t *A_ref,
                   const scalar_t *A_test,
                   size_t num_elements,
                   scalar_t abs_tol,
                   scalar_t rel_tol,
                   size_t *first_bad_index,
                   scalar_t *bad_ref,
                   scalar_t *bad_test);
```

Compara dos buffers usando tolerancia mixta:

$$
|A_{\text{ref}}[i] - A_{\text{test}}[i]| \leq \max(\text{abs\_tol},\ \text{rel\_tol} \cdot |A_{\text{ref}}[i]|)
$$

Devuelve `1` si todos los elementos pasan, `0` en caso contrario. Si devuelve `0` y los punteros `first_bad_index`/`bad_ref`/`bad_test` son no nulos, escribe el primer indice fallido y los dos valores correspondientes para diagnostico.

Las tolerancias concretas con que se invoca `matrices_close` desde cada `validate_*` viven en la Seccion 5.2.

### 2.2 `timing`

**Archivo:** [`src/core/timing.h`](../src/core/timing.h) (solo cabecera, sin `.c`).

#### `now_seconds`

```c
static inline double now_seconds(void);
```

Devuelve el tiempo actual en segundos via `clock_gettime(CLOCK_MONOTONIC, ...)`. `CLOCK_MONOTONIC` es inmune a ajustes de hora del sistema y tiene resolucion de nanosegundos en Linux moderno. Uso tipico:

```c
double t0 = now_seconds();
benchmark_iterations(B_out, A, Z, m, n, I);
double elapsed = now_seconds() - t0;
```

### 2.3 `morton`

**Archivos:** [`src/core/morton.h`](../src/core/morton.h), [`src/core/morton.c`](../src/core/morton.c).

Bit-interleaving Z-order y conversion entre row-major y Morton para matrices cuadradas. Implementacion portatil con magic constants (sin BMI2 `pdep`/`pext`).

**Convencion de bits.** Para $(i, j)$ con representaciones binarias $i_{p-1} \ldots i_0$ y $j_{p-1} \ldots j_0$:

$$
\text{morton}(i, j) = i_{p-1} j_{p-1} \ldots i_1 j_1 i_0 j_0
$$

$j$ contribuye a los bits pares, $i$ a los impares. Esta eleccion produce la tabla canonica para un sub-bloque $2 \times 2$:

| $(i, j)$ | codigo | cuadrante |
|----------|--------|-----------|
| (0, 0)   | 0      | TL |
| (0, 1)   | 1      | TR |
| (1, 0)   | 2      | BL |
| (1, 1)   | 3      | BR |

**Propiedad de contiguidad de cuadrantes.** Al dividir un sub-bloque cuadrado de lado $2h$ en sus cuatro cuadrantes de lado $h$, los cuatro segmentos Morton ocupan offsets $\{0, 1, 2, 3\} \cdot h^2$ desde el padre, todos **contiguos en memoria**. Esta es la propiedad clave que explotan los kernels `matmul_morton*` (Secciones 4.6 a 4.8).

#### `morton_encode`

```c
uint64_t morton_encode(uint32_t i, uint32_t j);
```

Intercala los bits de $i$ y $j$ segun la convencion anterior.

#### `morton_decode`

```c
void morton_decode(uint64_t code, uint32_t *i, uint32_t *j);
```

Inverso de `morton_encode`. Solo se usa en tests.

#### `reorganize_to_morton` / `reorganize_from_morton`

```c
void reorganize_to_morton  (const scalar_t *A_row,    scalar_t *A_morton, size_t m);
void reorganize_from_morton(const scalar_t *A_morton, scalar_t *A_row,    size_t m);
```

Copian elemento a elemento entre row-major y Morton para una matriz cuadrada $m \times m$. **Precondicion:** $m$ potencia de 2. Aborta con mensaje a `stderr` + `exit(EXIT_FAILURE)` si no se cumple. Complejidad: $O(m^2)$.

En el contexto del benchmark, `reorganize_to_morton` se ejecuta **una sola vez** antes de la recurrencia $B_{i+1} = A \cdot B_i$, por lo que su costo amortizado es despreciable ($O(1/I)$ del trabajo total).

#### `is_power_of_two`

```c
int is_power_of_two(size_t m);
```

Devuelve `1` si `m > 0 && (m & (m - 1)) == 0`, `0` en otro caso.

---

## 3. Capa `microkernels`

Microkernels AVX2 + FMA, header-only `static inline`. Ningun `.o` separado: el cuerpo se inline en cada TU que los incluye y se compila con `-O3 -march=znver2 -mavx2 -mfma`.

Cada microkernel tiene la responsabilidad de mantener su tile de $C$ en registros YMM durante toda la pasada $k_c$; el caller garantiza la geometria $(M_R \times N_R)$ y `restrict` no-aliasing.

### 3.1 `kernel_avx2_morton`

**Archivo:** [`src/microkernels/kernel_avx2_morton.h`](../src/microkernels/kernel_avx2_morton.h).

Microkernel $4 \times 16$ usado por la familia Morton-de-bloques (Secciones 4.7 y 4.8). Los $64$ elementos del tile $C$ caben en $8$ registros YMM (4 filas $\times$ 2 vectores de 8 lanes FP32); $4$ YMM extra para los broadcasts de $A$ y $2$ para $B$ totalizan $14$ de $16$ YMMs vivos en estado estable.

#### Geometria

```c
#define KERNEL_AVX2_MR 4    /* filas por tile */
#define KERNEL_AVX2_NR 16   /* columnas por tile */
```

#### `kernel_avx2_4x16`

```c
static inline void
kernel_avx2_4x16(scalar_t       *restrict C, size_t ldc,
                 const scalar_t *restrict A, size_t lda,
                 const scalar_t *restrict B, size_t ldb,
                 size_t kc);
```

**Computa** $C \mathrel{+}= A \cdot B$ sobre el tile $4 \times 16$ (**acumulacion**, no asignacion). Caller debe zerar $C$ antes si necesita reset.

**Precondiciones:**
- `kc >= 1`, `lda >= kc`, `ldb >= 16`, `ldc >= 16`.
- `C`, `A`, `B` no aliasan (`restrict`).
- Alineacion preferida pero no obligatoria (usa `_mm256_loadu_ps`/`_mm256_storeu_ps`); la penalizacion de loadu sobre datos accidentalmente alineados es 0 ciclos en Zen 2.

**Postcondiciones:**
- $C[r, c] \mathrel{+}= \sum_{p=0}^{kc-1} A[r, p] \cdot B[p, c]$ para $r \in [0, 4)$, $c \in [0, 16)$.
- $A$ y $B$ no se modifican.

### 3.2 `kernel_avx2_tiled`

**Archivo:** [`src/microkernels/kernel_avx2_tiled.h`](../src/microkernels/kernel_avx2_tiled.h).

Microkernel BLIS-style $6 \times 16$ usado por la familia `tiled_ikj` (Secciones 4.4 y 4.5). Los $96$ elementos del tile $C$ caben en $12$ registros YMM (6 filas $\times$ 2 vectores de 8 lanes); $2$ YMM para $B$ y $1$ YMM rotativo para los broadcasts de $A$ totalizan $15$ de $16$ YMMs vivos.

#### Geometria

```c
#define KERNEL_AVX2_TILED_MR 6u    /* filas por tile */
#define KERNEL_AVX2_TILED_NR 16u   /* columnas por tile */
```

Estas constantes coexisten con las del modulo de algoritmo (`TILED_IKJ_AVX2_MR`, `TILED_IKJ_OMP_MR`, etc.) que son numericamente iguales. La duplicacion es deliberada: el microkernel publica las suyas en el header del propio microkernel, y cada modulo de algoritmo publica las suyas en su propio header para que sus callers no tengan que incluir el del microkernel.

#### `kernel_avx2_tiled_6x16`

```c
static inline void
kernel_avx2_tiled_6x16(scalar_t       *restrict C, size_t ldc,
                       const scalar_t *restrict A, size_t lda,
                       const scalar_t *restrict B, size_t ldb,
                       size_t kc);
```

**Computa** $C \mathrel{+}= A \cdot B$ sobre el tile $6 \times 16$ (**acumulacion**). Mismo contrato de pre/postcondiciones que `kernel_avx2_4x16`, escalado a $M_R = 6$, $N_R = 16$.

---

## 4. Capa `algorithms`

Cada modulo de esta capa expone un kernel principal con la firma de la Seccion 1.5 y un orquestador `benchmark_iterations_<variant>` que corre la recurrencia $B_{i+1} = A \cdot B_i$ usando doble buffer + swap de punteros.

Los modulos estan ordenados por familia (`naive`, `loops`, `tiled_ikj*`, `morton*`) y, dentro de cada familia, por nivel de optimizacion (escalar $\to$ AVX2 $\to$ OpenMP).

### 4.1 `matmul_naive`

**Archivos:** [`src/algorithms/naive/matmul_naive.h`](../src/algorithms/naive/matmul_naive.h), [`src/algorithms/naive/matmul_naive.c`](../src/algorithms/naive/matmul_naive.c).
**Estado:** baseline obligatorio del proyecto. **No** se modifica: las variantes optimizadas viven en modulos separados con firmas analogas.

#### `matmul_naive`

```c
void matmul_naive(scalar_t *C,
                  const scalar_t *A,
                  const scalar_t *B,
                  size_t m, size_t k, size_t n);
```

**Computa** $C = A \cdot B$ con tres bucles anidados en orden `ijk` (sin optimizacion de localidad: stride $n$ al acceder a $B$). $C$ se **sobrescribe** (no acumula): el orden `ijk` deja la dimension de reduccion en el bucle interno, por lo que cada $C_{ij}$ se completa antes de pasar al siguiente.

**Precondiciones:**
- Los tres punteros son no nulos.
- $C$ no aliasa con $A$ ni con $B$.
- Buffers de tamano suficiente (`m*n`, `m*k`, `k*n`).

**Postcondiciones:**
- $C_{ij} = \sum_{p=0}^{k-1} A_{ip} B_{pj}$ para todo $(i, j)$.
- $A$ y $B$ no se modifican.

**Complejidad:** $2 \cdot m \cdot k \cdot n$ flops.

#### `benchmark_iterations`

```c
void benchmark_iterations(scalar_t *B_out,
                          const scalar_t *A,
                          const scalar_t *Z,
                          size_t m, size_t n,
                          size_t num_iters);
```

**Computa** la recurrencia $B_{i+1} = A \cdot B_i$ con $B_0 = Z$, guardando las primeras $n$ filas de cada $B_{i+1}$ en `B_out`.

**Parametros clave:**
- `B_out` *(out)*: buffer de `num_iters * n * n` elementos. Bloque `iter` ocupa offsets $[\text{iter} \cdot n^2, (\text{iter}+1) \cdot n^2)$.
- `A` *(in)*: matriz cuadrada $m \times m$.
- `Z` *(in)*: matriz $m \times n$, estado inicial $B_0$.
- `num_iters`: tipicamente $I = 2m/n$, pero se acepta cualquier valor positivo.

**Implementacion interna:** doble buffer `B_curr` / `B_next` de tamano $m \times n$ con swap de punteros para evitar copias. Aloja y libera ambos buffers internamente.

**Complejidad:** $2 m^2 n \cdot \text{num\_iters}$ flops, mas $O(mn)$ de copia de salida por iteracion.

### 4.2 `matmul_loops`

**Archivos:** [`src/algorithms/loops/matmul_loops.h`](../src/algorithms/loops/matmul_loops.h), [`src/algorithms/loops/matmul_loops.c`](../src/algorithms/loops/matmul_loops.c).
**Estado:** Fase 1.1 (cache-aware loop reorder).

Las seis permutaciones del orden de bucles para $C = A \cdot B$, expuestas individualmente y via lookup por nombre.

#### Tipo y kernels

```c
typedef void (*matmul_fn_t)(scalar_t *C,
                            const scalar_t *A,
                            const scalar_t *B,
                            size_t m, size_t k, size_t n);

void matmul_ijk(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n);
void matmul_ikj(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n);
void matmul_jik(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n);
void matmul_jki(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n);
void matmul_kij(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n);
void matmul_kji(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n);
```

Pre/postcondiciones identicas a `matmul_naive`. Las variantes con la dimension de reduccion fuera del bucle interno (`ikj`, `jki`, `kij`, `kji`) llaman a `init_matrix_zero(C, m, n)` internamente antes de acumular; `ijk` y `jik` sobrescriben directamente.

#### `matmul_loops_lookup`

```c
matmul_fn_t matmul_loops_lookup(const char *name);
```

Devuelve el puntero de funcion para los nombres `"ijk"`, `"ikj"`, `"jik"`, `"jki"`, `"kij"`, `"kji"`, o `NULL` si el nombre no es reconocido.

#### `benchmark_iterations_loops`

```c
void benchmark_iterations_loops(scalar_t *B_out,
                                const scalar_t *A,
                                const scalar_t *Z,
                                size_t m, size_t n,
                                size_t num_iters,
                                matmul_fn_t kernel);
```

Misma semantica que `benchmark_iterations` (Seccion 4.1), pero delegando cada paso $A \cdot B$ al `kernel` suministrado.

### 4.3 `matmul_tiled_ikj`

**Archivos:** [`src/algorithms/tiled_ikj/matmul_tiled_ikj.h`](../src/algorithms/tiled_ikj/matmul_tiled_ikj.h), [`src/algorithms/tiled_ikj/matmul_tiled_ikj.c`](../src/algorithms/tiled_ikj/matmul_tiled_ikj.c).
**Estado:** Fase 1.2 (tiling explicito apuntando a L2).

Tiling explicito $M_c \times K_c$ sobre el orden `ikj`. Apunta al L2 de $512$ KB del $4600$H.

#### Constantes publicas

```c
#define TILED_IKJ_MC_DEFAULT 256u
#define TILED_IKJ_KC_DEFAULT 256u
```

A $M_c = K_c = 256$, los tres paneles activos ($A$: $256$ KB, $B$: $128$ KB, $C$: $128$ KB en el bloque vivo) llenan exactamente el L2 del $4600$H. Compile-time fijos: no hay setter runtime para esta variante.

#### `matmul_tiled_ikj`

```c
void matmul_tiled_ikj(scalar_t *C,
                      const scalar_t *A,
                      const scalar_t *B,
                      size_t m, size_t k_dim, size_t n);
```

**Computa** $C = A \cdot B$ con tiling explicito. Llama internamente a `init_matrix_zero(C, m, n)` antes de acumular las contribuciones por tile. Misma firma efectiva que `matmul_naive` (el parametro se llama `k_dim` en este modulo).

#### `benchmark_iterations_tiled_ikj`

```c
void benchmark_iterations_tiled_ikj(scalar_t *B_out,
                                    const scalar_t *A,
                                    const scalar_t *Z,
                                    size_t m, size_t n,
                                    size_t num_iters);
```

Misma semantica que `benchmark_iterations` con doble buffer; invoca `matmul_tiled_ikj` en cada iteracion.

### 4.4 `matmul_tiled_ikj_avx2`

**Archivos:** [`src/algorithms/tiled_ikj/matmul_tiled_ikj_avx2.h`](../src/algorithms/tiled_ikj/matmul_tiled_ikj_avx2.h), [`src/algorithms/tiled_ikj/matmul_tiled_ikj_avx2.c`](../src/algorithms/tiled_ikj/matmul_tiled_ikj_avx2.c). Usa [`kernel_avx2_tiled.h`](../src/microkernels/kernel_avx2_tiled.h) (Seccion 3.2).
**Estado:** Fase 1.6 (microkernel BLIS-style $6 \times 16$ register-blocked).

Loop nest Goto-style $p_c \to i_c \to j_r \to i_r$ con microkernel $6 \times 16$ en el centro. $C$ se zera con `memset` al inicio para que cada pasada $p_c$ pueda cargar / acumular / guardar.

#### Constantes publicas

```c
#define TILED_IKJ_AVX2_MR 6u
#define TILED_IKJ_AVX2_NR 16u
#define TILED_IKJ_AVX2_MC 192u
#define TILED_IKJ_AVX2_BS_DEFAULT 384u
extern size_t g_tiled_ikj_avx2_bs;
```

- $M_R = 6$, $N_R = 16$: tile registrado fijo (compile-time).
- $M_C = 192$: tamano del bloque sobre $m$ (multiplo de $M_R$). Panel $A$ activo de $M_C \times k_c$ a $k_c = 384$ ocupa $288$ KiB y cabe en el L2 de $512$ KB.
- $k_c$ (alias `bs`): default $384$, configurable runtime (ver setter abajo). El panel $B$ activo $k_c \times N_R$ a $k_c = 384$ ocupa $24$ KiB y cabe en el L1d.

#### `matmul_tiled_ikj_avx2_set_bs`

```c
void matmul_tiled_ikj_avx2_set_bs(size_t bs);
```

Cambia $k_c$ en runtime. Cualquier valor positivo es valido (el microkernel itera $p$ uno a la vez, no requiere $k_c$ multiplo de $8$). Pasar $0$ es invalido y se rechaza.

#### `matmul_tiled_ikj_avx2`

```c
void matmul_tiled_ikj_avx2(scalar_t *C,
                           const scalar_t *A,
                           const scalar_t *B,
                           size_t m, size_t k, size_t n);
```

**Computa** $C = A \cdot B$. $A$ y $B$ row-major estandar (no requieren reorganizacion previa, a diferencia de la familia Morton).

**Precondiciones:** mismas que `matmul_naive`. Bordes en $m$ y $n$ que no son multiplos del tile caen a un fallback escalar AVX2 sin registrar.

**Compilacion:** requiere `-O3 -march=znver2 -mavx2 -mfma` para que `_mm256_fmadd_ps` emita la instruccion FMA real.

#### `benchmark_iterations_tiled_ikj_avx2`

```c
void benchmark_iterations_tiled_ikj_avx2(scalar_t *B_out,
                                         const scalar_t *A,
                                         const scalar_t *Z,
                                         size_t m, size_t n,
                                         size_t num_iters);
```

Mismo patron de doble buffer.

### 4.5 `matmul_tiled_ikj_omp`

**Archivos:** [`src/algorithms/tiled_ikj/matmul_tiled_ikj_omp.h`](../src/algorithms/tiled_ikj/matmul_tiled_ikj_omp.h), [`src/algorithms/tiled_ikj/matmul_tiled_ikj_omp.c`](../src/algorithms/tiled_ikj/matmul_tiled_ikj_omp.c). Usa [`kernel_avx2_tiled.h`](../src/microkernels/kernel_avx2_tiled.h) (Seccion 3.2).
**Estado:** Fase 1.6 (microkernel $6 \times 16$ + `#pragma omp parallel for schedule(static)` en el bucle $i_c$).

Hermano paralelo de `matmul_tiled_ikj_avx2`. Mismo loop nest y mismo microkernel; lo unico que cambia es que el bucle $i_c$ se distribuye entre threads. Cada thread escribe exclusivamente las filas $[i_c, i_c + M_C)$ de $C$, sin conflictos. La region paralela se abre una sola vez por invocacion.

#### Constantes publicas

```c
#define TILED_IKJ_OMP_MR 6u
#define TILED_IKJ_OMP_NR 16u
#define TILED_IKJ_OMP_MC 192u
#define TILED_IKJ_OMP_BS_DEFAULT 384u
extern size_t g_tiled_ikj_omp_bs;
```

Misma geometria que la version serial; el knob `bs` es independiente para permitir tunear por separado.

#### `matmul_tiled_ikj_omp_set_bs`

```c
void matmul_tiled_ikj_omp_set_bs(size_t bs);
```

Equivalente al de la version serial.

#### `matmul_tiled_ikj_omp`

```c
void matmul_tiled_ikj_omp(scalar_t *C,
                          const scalar_t *A,
                          const scalar_t *B,
                          size_t m, size_t k, size_t n);
```

Mismo contrato externo que `matmul_tiled_ikj_avx2`. El numero de threads lo fija `OMP_NUM_THREADS` antes de invocar el binario (sin override en codigo). **Compilacion:** requiere ademas `-fopenmp`.

#### `benchmark_iterations_tiled_ikj_omp`

```c
void benchmark_iterations_tiled_ikj_omp(scalar_t *B_out,
                                        const scalar_t *A,
                                        const scalar_t *Z,
                                        size_t m, size_t n,
                                        size_t num_iters);
```

Mismo patron de doble buffer.

### 4.6 `matmul_morton`

**Archivos:** [`src/algorithms/morton/matmul_morton.h`](../src/algorithms/morton/matmul_morton.h), [`src/algorithms/morton/matmul_morton.c`](../src/algorithms/morton/matmul_morton.c). Depende del modulo `core/morton` (Seccion 2.3).
**Estado:** Fase 6 / Sesion 02 (Morton fino, cache-oblivious).

Kernel recursivo donde $A$ esta en layout Morton-de-elementos (cada $A[i, j]$ va a la posicion `morton_encode(i, j)`) y $B$, $C$ siguen en row-major. La recursion sobre $A$ se hace via offsets Morton en lugar de via `(puntero, leading dimension)`.

#### `matmul_morton`

```c
void matmul_morton(scalar_t *C,
                   const scalar_t *A_morton,
                   const scalar_t *B,
                   size_t m, size_t k, size_t n);
```

**Precondiciones:**
- $m = k$ ($A$ cuadrada).
- `is_power_of_two(m)`.
- `A_morton` producido por `reorganize_to_morton(A, A_morton, m)`.
- `C` no aliasa con `A_morton` ni con $B$.

Cualquier violacion produce abort con mensaje a `stderr` + `exit(EXIT_FAILURE)`.

**Casos de recursion** (sea `m_block == k_block == a_block_dim` el invariante del cuadrante actual):
1. **Hoja:** `m_block * k_block * n_block <= g_recursion_threshold` $\to$ kernel base ijk escalar con indexing Morton.
2. **Caso N:** $n\_block > a\_block\_dim$ y $n\_block \geq 2$ $\to$ dividir $n$ en mitades. Las dos sub-llamadas comparten $A$; las regiones de $C$ y $B$ son disjuntas.
3. **Caso MK:** $a\_block\_dim \geq 2$ $\to$ dividir $m$ y $k$ simultaneamente en cuatro cuadrantes. Los offsets de los cuatro cuadrantes de $A$ son `a_morton_offset + {0, 1, 2, 3} * half * half`, contiguos en memoria gracias a la propiedad de la Seccion 2.3.
4. **Fallback degenerado:** $a\_block\_dim = 1$ y $n\_block = 1$ $\to$ kernel base.

**Complejidad:** $2mkn$ flops (igual al baseline). La ganancia es de **localidad**, no de operaciones.

#### Constantes y setter

```c
extern size_t g_recursion_threshold;
void matmul_morton_set_threshold(size_t threshold);
```

`g_recursion_threshold` default $= 32 \cdot 32 \cdot 128 = 131072$ flops elementales. A esa profundidad el panel $A$ activo es $\sim 4$ KiB (cabe en L1d de $32$ KiB). Pasar `threshold = 0` al setter imprime warning y mantiene el default.

#### Orquestadores

```c
void benchmark_iterations_morton(scalar_t *B_out,
                                 const scalar_t *A,
                                 const scalar_t *Z,
                                 size_t m, size_t n,
                                 size_t num_iters);

void benchmark_iterations_morton_preorganized(scalar_t *B_out,
                                              const scalar_t *A_morton,
                                              const scalar_t *Z,
                                              size_t m, size_t n,
                                              size_t num_iters);
```

`benchmark_iterations_morton` reorganiza $A$ a Morton **internamente en cada llamada** (el costo entra en el tiempo medido). `benchmark_iterations_morton_preorganized` recibe $A$ ya en Morton y es la que usa `bench_morton_O3` para que la reorganizacion no entre en el tiempo cronometrado.

### 4.7 `matmul_morton_avx2`

**Archivos:** [`src/algorithms/morton/matmul_morton_avx2.h`](../src/algorithms/morton/matmul_morton_avx2.h), [`src/algorithms/morton/matmul_morton_avx2.c`](../src/algorithms/morton/matmul_morton_avx2.c). Usa [`kernel_avx2_morton.h`](../src/microkernels/kernel_avx2_morton.h) (Seccion 3.1).
**Estado:** Sesion 03.

Variante de `matmul_morton` con leaf vectorizado. Misma estructura recursiva, pero cambia el layout de $A$ y el indexing en la hoja.

**Layout Morton-de-bloques** (tile $= M_R = 4$). $A$ se particiona en sub-bloques $4 \times 4$; los sub-bloques se Z-ordenan entre si y los $16$ elementos de cada sub-bloque viven en row-major. La posicion de $A[i, j]$ es:

$$
A\_idx = \text{morton\_encode}(i / M_R,\ j / M_R) \cdot M_R^2 + (i \bmod M_R) \cdot M_R + (j \bmod M_R)
$$

La propiedad de contiguidad de cuadrantes se preserva: lo unico que cambia es el significado del nivel mas bajo (un "elemento" Morton es un tile $4 \times 4$ de floats, no un float suelto). Los layouts fino y de bloques coexisten; `matmul_morton.{c,h}` queda intacto.

#### Constante de tile

```c
#define MORTON_AVX2_TILE KERNEL_AVX2_MR    /* = 4 */
```

#### `reorganize_to_morton_blocks`

```c
void reorganize_to_morton_blocks(const scalar_t *A_row,
                                 scalar_t *A_morton,
                                 size_t m);
```

Empaqueta $A$ row-major al layout Morton-de-bloques. **Precondiciones:** `m % MORTON_AVX2_TILE == 0` y `is_power_of_two(m / MORTON_AVX2_TILE)`. Aborta si no se cumplen. Complejidad: $O(m^2)$.

#### `matmul_morton_avx2`

```c
void matmul_morton_avx2(scalar_t *C,
                        const scalar_t *A_morton,
                        const scalar_t *B,
                        size_t m, size_t k, size_t n);
```

**Precondiciones:**
- $m = k$ y $m$ potencia de $2$, con $m \geq M_R = 4$.
- $n \geq N_R = 16$. Recomendado $n$ multiplo de $N_R$; trozos no alineados caen a un fallback `ijk` sin vectorizar.
- `A_morton` producido por `reorganize_to_morton_blocks`.

#### Constantes y setter

```c
extern size_t g_recursion_threshold_avx2;
void matmul_morton_avx2_set_threshold(size_t threshold);
```

`g_recursion_threshold_avx2` default $= 64 \cdot 64 \cdot 128 = 524288$ flops elementales (panel $A$ activo $\sim 16$ KiB, mitad de L1d). Separado de `g_recursion_threshold` (Seccion 4.6) porque el leaf AVX2 amortiza una hoja mayor.

#### Orquestadores

```c
void benchmark_iterations_morton_avx2(scalar_t *B_out,
                                      const scalar_t *A,
                                      const scalar_t *Z,
                                      size_t m, size_t n,
                                      size_t num_iters);

void benchmark_iterations_morton_avx2_preorganized(scalar_t *B_out,
                                                   const scalar_t *A_morton,
                                                   const scalar_t *Z,
                                                   size_t m, size_t n,
                                                   size_t num_iters);
```

Misma convencion que en `matmul_morton`: la `_preorganized` es la que usa `bench_morton_avx2_O3`.

### 4.8 `matmul_morton_omp`

**Archivos:** [`src/algorithms/morton/matmul_morton_omp.h`](../src/algorithms/morton/matmul_morton_omp.h), [`src/algorithms/morton/matmul_morton_omp.c`](../src/algorithms/morton/matmul_morton_omp.c). Reusa [`kernel_avx2_morton.h`](../src/microkernels/kernel_avx2_morton.h) (Seccion 3.1) y `reorganize_to_morton_blocks` de la Seccion 4.7.
**Estado:** Sesion 03.

Variante paralela de `matmul_morton_avx2`. Misma estructura recursiva y mismo microkernel; agrega `#pragma omp parallel single` en el wrapper publico y emite `omp task` en cada subdivision por encima del threshold de paralelizacion. El scratch buffer del leaf es un pool por-thread indexado por `omp_get_thread_num()` para evitar comparticion entre tasks.

#### `matmul_morton_omp`

```c
void matmul_morton_omp(scalar_t *C,
                       const scalar_t *A_morton,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n);
```

Mismas precondiciones que `matmul_morton_avx2` (Seccion 4.7). **Compilacion:** requiere ademas `-fopenmp`. El numero de threads lo fija `OMP_NUM_THREADS`.

#### Thresholds (dos knobs independientes)

```c
extern size_t g_recursion_threshold_omp;       /* default 524288 */
extern size_t g_parallel_threshold_omp;        /* default 524288 */
void matmul_morton_omp_set_threshold         (size_t threshold);
void matmul_morton_omp_set_parallel_threshold(size_t threshold);
```

- `g_recursion_threshold_omp`: tamano del sub-problema en que la recursion cae al leaf. Mismo rol que `g_recursion_threshold_avx2`.
- `g_parallel_threshold_omp`: tamano por debajo del cual la recursion deja de emitir `omp task` y corre inline. Default igual al leaf threshold $\to$ tasks en cada nivel sobre la hoja y ninguna dentro de ella. Pasar `0` permite serializar para aislar el costo del scaffolding OMP.

Las globals son independientes de las de `matmul_morton_avx2` para tunear la paralela sin alterar las mediciones de la serial.

#### Orquestadores

```c
void benchmark_iterations_morton_omp(scalar_t *B_out,
                                     const scalar_t *A,
                                     const scalar_t *Z,
                                     size_t m, size_t n,
                                     size_t num_iters);

void benchmark_iterations_morton_omp_preorganized(scalar_t *B_out,
                                                  const scalar_t *A_morton,
                                                  const scalar_t *Z,
                                                  size_t m, size_t n,
                                                  size_t num_iters);
```

Misma convencion: la `_preorganized` es la que usa `bench_morton_omp_O3`.

---

## 5. Capa `drivers`

Programas `main` en [`src/drivers/`](../src/drivers): un `bench_<variant>.c` y un `validate_<variant>.c` por cada modulo de la capa `algorithms`. Todos siguen una convencion comun documentada aqui.

### 5.1 Convencion comun de los `bench_*`

**Proposito:** medir tiempo de pared del kernel y reportar GFLOP/s.

**CLI (estandar):**

```
bench_<variant> <m> [num_iters] [num_runs]
```

Las variantes con knob de tile-size adicional (`tiled_ikj_avx2`, `tiled_ikj_omp`) aceptan un cuarto argumento:

```
bench_tiled_ikj_avx2 <m> [num_iters] [num_runs] [bs]
bench_tiled_ikj_omp  <m> [num_iters] [num_runs] [bs]
```

La variante `bench_loops` recibe el orden como primer argumento:

```
bench_loops <order> <m> [num_iters] [num_runs]      # order = ijk | ikj | jik | jki | kij | kji
```

**Defaults:**
- `num_iters`: $\min(2m/n, 4)$ por defecto. Knob `--iters-per-run` desde el sweep (`ITERS_PER_RUN=0` activa $I_{\text{full}} = 2m/n$).
- `num_runs`: `5` para corridas standalone, `1`-`3` para `make results`.

**Protocolo de medicion:** una **corrida de warm-up no medida** seguida de `num_runs` corridas medidas. Se reporta la **mediana** de los tiempos.

**Salida (una linea CSV en `stdout`):**

```
<kernel>,m,n,num_iters,median_seconds,gflops
```

Para `tiled_ikj_avx2` y `tiled_ikj_omp` se inserta una columna `bs` (formato de 7 columnas):

```
<kernel>,m,n,num_iters,bs,median_seconds,gflops
```

Los benches de la familia Morton (`bench_morton_O3`, `bench_morton_avx2_O3`, `bench_morton_omp_O3`) **abortan** si $m$ no es potencia de 2; la reorganizacion a Morton se ejecuta **una sola vez antes del warm-up**, fuera del tiempo cronometrado.

### 5.2 Convencion comun de los `validate_*`

**Proposito:** verificar correctitud del kernel sobre invariantes algebraicos.

**CLI:**

```
validate_<variant> [m]
```

Las variantes con knob de bs aceptan un segundo argumento:

```
validate_tiled_ikj_avx2 [m] [bs]
validate_tiled_ikj_omp  [m] [bs]
```

Default: $m = 256$.

**Invariantes basicos (todos los validates):**

1. $A \cdot 0 = 0$
2. $I \cdot Z = Z$
3. $A \cdot (Z_1 + Z_2) = A \cdot Z_1 + A \cdot Z_2$

Algunos validates anaden tests adicionales:

- `validate_loops`: 4 tests por variante (3 invariantes + cross-validation contra `matmul_naive`), barrido sobre los 6 ordenes.
- `validate_morton`: 3 invariantes + cross-validation contra `matmul_naive` en $m \in \{4, 16, 64, 256\}$ (7 tests totales).
- `validate_morton_avx2`, `validate_morton_omp`: 3 invariantes + cross-validation contra `matmul_naive` y contra `matmul_morton`.
- `validate_tiled_ikj{,_avx2,_omp}`: 3 invariantes + cross-validation contra `matmul_naive`.

**Tolerancias (`matrices_close`):**

| Validate | `ABS_TOL` | `REL_TOL` |
|----------|-----------|-----------|
| `validate_naive` | $10^{-4}$ | $10^{-3}$ |
| `validate_loops` | $10^{-4}$ | $10^{-3}$ |
| `validate_tiled_ikj` | $10^{-4}$ | $10^{-3}$ |
| `validate_tiled_ikj_avx2` | $10^{-4}$ | $10^{-3}$ |
| `validate_tiled_ikj_omp` | $10^{-4}$ | $10^{-3}$ |
| `validate_morton` | $10^{-5}$ | $10^{-4}$ |
| `validate_morton_avx2` | $10^{-4}$ | $10^{-3}$ |
| `validate_morton_omp` | $10^{-4}$ | $10^{-3}$ |

`validate_morton` mantiene tolerancia mas estricta porque el kernel escalar Morton fino no reasocia la suma FP; los demas relajan a $10^{-4} / 10^{-3}$ por el reorden inducido por AVX2 (`-O3`, intrinsics FMA) y por OpenMP.

**Salida:** mensajes por test (`[OK]` / `[FAIL]`) y, al cierre, `VALIDATION OK` o `VALIDATION FAILED`. Codigo de salida: `0` en exito, `1` si algun test falla. Cuando falla, se reporta el primer indice fallido y los dos valores (`A_ref`, `A_test`) para diagnostico.

### 5.3 Tabla unificada de binarios

Una fila por driver. Rutas relativas a la raiz del repo tras `make build`. Targets de compilacion individuales documentados en [`docs/0.0) makefile.md`](<0.0) makefile.md>).

| Familia | Bench | Validate |
|---------|-------|----------|
| `naive` (Sesion 01) | `bin/bench/bench_naive_O3` | `bin/validate/validate_naive_O0` |
| `loops` (Fase 1.1) | `bin/bench/bench_loops_O3` | `bin/validate/validate_loops_O0` |
| `tiled_ikj` (Fase 1.2) | `bin/bench/bench_tiled_ikj_O3` | `bin/validate/validate_tiled_ikj_O0` |
| `tiled_ikj_avx2` (Fase 1.6) | `bin/bench/bench_tiled_ikj_avx2_O3` | `bin/validate/validate_tiled_ikj_avx2_O3` |
| `tiled_ikj_omp` (Fase 1.6) | `bin/bench/bench_tiled_ikj_omp_O3` | `bin/validate/validate_tiled_ikj_omp_O3` |
| `morton` (Fase 6) | `bin/bench/bench_morton_O3` | `bin/validate/validate_morton_O0` |
| `morton_avx2` (Sesion 03) | `bin/bench/bench_morton_avx2_O3` | `bin/validate/validate_morton_avx2_O3` |
| `morton_omp` (Sesion 03) | `bin/bench/bench_morton_omp_O3` | `bin/validate/validate_morton_omp_O3` |

**Flags de compilacion por sufijo:**

- `_O0`: `-std=c11 -Wall -Wextra -Wpedantic -O0 -g -fno-omit-frame-pointer -D_POSIX_C_SOURCE=200809L`.
- `_O3`: igual al anterior pero con `-O3 -march=znver2 -mavx2 -mfma`.
- `_O3` ademas con `-fopenmp` para las variantes `_omp`.

Las variantes que requieren AVX2 + FMA usan `_O3` tambien para el `validate_*` (no `_O0`) porque los microkernels son `static inline` con intrinsics y a `-O0` no se materializan las instrucciones FMA.

**Pipeline de medicion:** el unico flujo soportado es `make results` (orquesta `scripts/run_perf_zen2_sweep.sh` + `scripts/consolidate_perf_zen2.py`). Salida canonica: `results/metrics.csv`. Knobs (`VARIANTS`, `MS`, `ITERS_PER_RUN`, `RUNS`) documentados en [`docs/0.0) makefile.md`](<0.0) makefile.md>).

---

## 6. Capa `tests`

Tests unitarios standalone para las primitivas de [`src/core/`](../src/core) y los microkernels de [`src/microkernels/`](../src/microkernels). Binarios en `bin/tests/`. Documentacion detallada (estructura de cada test, casos cubiertos, contrato de exit code) en [`docs/1.9) tests.md`](<1.9) tests.md>).

| Binario | Cubre |
|---------|-------|
| `bin/tests/test_matrix_utils` | `xalloc_aligned`, `init_matrix_zero/identity/random`, `matrices_close` |
| `bin/tests/test_morton` | `morton_encode`, `morton_decode`, `reorganize_to_morton`, `reorganize_from_morton` |
| `bin/tests/test_kernel_avx2_morton` | Microkernel $4 \times 16$ de [`kernel_avx2_morton.h`](../src/microkernels/kernel_avx2_morton.h) |
| `bin/tests/test_kernel_avx2_tiled` | Microkernel $6 \times 16$ de [`kernel_avx2_tiled.h`](../src/microkernels/kernel_avx2_tiled.h) |

La piramide completa de verificacion es `tests` $\to$ `validate` $\to$ `bench`. `make validate` depende de `make tests`: un fallo en los unit-tests aborta antes de correr los `validate_*`.

---

## 7. Versionado del documento

Este documento se actualiza con cada PR que toque la API publica. La regla es: **si una firma de funcion, una constante publica o un binario cambia, este documento debe cambiar en el mismo commit**.
