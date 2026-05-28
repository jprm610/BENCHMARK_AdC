# Especificacion de la API del benchmark

**Proyecto:** Benchmark de Multiplicacion Iterada de Matrices
**Curso:** Arquitectura de Computadores - UNAL Medellin
**Estado:** Version inicial (Fase 1: baseline + profiling + escalamiento con $m$)

Este documento describe el contrato publico de las funciones implementadas en `src/`. Las firmas estan en C (codigo y nombres en ingles). Las versiones futuras (transposicion, tiling, vectorizacion, OpenMP) **deben respetar las mismas firmas** para que el sistema de validacion siga funcionando sin modificacion.

---

## 1. Convenciones generales

### 1.1 Tipo escalar

Toda la implementacion usa `float` (IEEE 754 binary32, 4 bytes). Se centraliza con un `typedef`:

```c
typedef float scalar_t;
```

Definido en [`src/core/matrix_utils.h`](../src/core/matrix_utils.h). Cambiar a `double` requeriria reemplazar este `typedef` y revisar tolerancias en `src/drivers/validate/validate_naive.c`.

### 1.2 Tipos enteros

Todos los tamanos, conteos e indices usan `size_t` (entero sin signo, ancho de palabra del sistema). Para $m = 2^{20}$, $m \cdot m = 2^{40}$ desborda un `int` de 32 bits, por lo que el uso de `size_t` es obligatorio.

### 1.3 Layout de matrices

Todas las matrices estan en **row-major** y se almacenan en buffers planos de tipo `scalar_t *`. Una matriz $M \in \mathbb{R}^{r \times c}$ ocupa `r * c` elementos contiguos. El elemento $M_{ij}$ se accede como:

```c
M[(size_t)i * c + j]
```

Si una version futura introduce padding ("leading dimension" $> c$) o un layout distinto (Morton, column-major), debera exponer una nueva firma; no se modificara la firma de las funciones aqui descritas.

### 1.4 Alineacion

Toda la memoria para matrices se aloja con alineacion de **64 bytes** (linea de cache en x86_64) usando `posix_memalign`. El allocador `xalloc_aligned` se encarga.

### 1.5 Estilo de comentarios

Comentarios y nombres en ingles, sin emojis ni caracteres no ASCII (compatibilidad con teclado en espanol y portabilidad de fuentes).

---

## 2. Modulo `matmul_naive`

**Archivo:** [`src/algorithms/naive/matmul_naive.h`](../src/algorithms/naive/matmul_naive.h), [`src/algorithms/naive/matmul_naive.c`](../src/algorithms/naive/matmul_naive.c).

Contiene el kernel ingenuo $C = A \cdot B$ y el orquestador de la recurrencia $B_{i+1} = A \cdot B_i$.

### 2.1 `matmul_naive`

```c
void matmul_naive(scalar_t *C,
                  const scalar_t *A,
                  const scalar_t *B,
                  size_t m, size_t k, size_t n);
```

**Computa** $C = A \cdot B$ con tres bucles anidados en orden `ijk`.

**Parametros:**

- `C` *(out)*: matriz de salida, tamano $m \times n$, row-major. El contenido previo se sobrescribe.
- `A` *(in)*: matriz de entrada, tamano $m \times k$.
- `B` *(in)*: matriz de entrada, tamano $k \times n$.
- `m, k, n`: dimensiones de las matrices.

**Precondiciones:**

- Los tres punteros son no nulos y referencian buffers correctamente alojados.
- $C$ no aliasa con $A$ ni con $B$.
- Los buffers tienen suficiente memoria (`m * n`, `m * k`, `k * n` elementos respectivamente).

**Postcondiciones:**

- $C_{ij} = \sum_{p=0}^{k-1} A_{ip} B_{pj}$ para todo $(i, j)$.
- $A$ y $B$ no son modificados.

**Complejidad:** $2 \cdot m \cdot k \cdot n$ flops. Sin optimizacion de localidad (orden `ijk` produce stride $n$ al acceder a $B$).

**Notas:** esta funcion es el baseline obligatorio del proyecto. **No** debe modificarse para introducir optimizaciones; las versiones optimizadas iran en nuevos modulos (`matmul_reordered.c`, `matmul_tiled_ikj.c`, etc.) con firmas analogas.

### 2.2 `benchmark_iterations`

```c
void benchmark_iterations(scalar_t *B_out,
                          const scalar_t *A,
                          const scalar_t *Z,
                          size_t m, size_t n,
                          size_t num_iters);
```

**Computa** la recurrencia $B_{i+1} = A \cdot B_i$ con $B_0 = Z$, almacenando las primeras $n$ filas de cada $B_{i+1}$ en el buffer de salida.

**Parametros:**

- `B_out` *(out)*: buffer plano de tamano `num_iters * n * n` elementos. Los primeros $n^2$ elementos contienen las $n$ primeras filas de $B_1$ en row-major, los siguientes $n^2$ las de $B_2$, etc.
- `A` *(in)*: matriz cuadrada $m \times m$, constante durante toda la ejecucion.
- `Z` *(in)*: matriz inicial $m \times n$ ($B_0 = Z$).
- `m, n`: dimensiones del benchmark. $n$ es el tamano del bloque (tipicamente 128).
- `num_iters`: numero de iteraciones $I$ a ejecutar. Tipicamente $I = 2m/n$, pero se acepta cualquier valor positivo para permitir mediciones rapidas.

**Implementacion interna:** usa dos buffers `B_curr` y `B_next` de tamano $m \times n$ con swap de punteros para evitar copias entre iteraciones. Aloja y libera ambos buffers internamente.

**Precondiciones:**

- $n \leq m$ (no se valida; el caller es responsable).
- `B_out` tiene espacio para `num_iters * n * n` elementos.

**Postcondiciones:**

- Para cada `iter` en $[0, \text{num\_iters})$: `B_out[iter * n * n + i * n + j]` contiene $B_{\text{iter}+1}[i, j]$ para $0 \leq i < n$, $0 \leq j < n$.

**Complejidad:** $2 m^2 n \cdot$ `num_iters` flops, mas $O(mn)$ copia de salida por iteracion.

---

## 3. Modulo `matrix_utils`

**Archivo:** [`src/core/matrix_utils.h`](../src/core/matrix_utils.h), [`src/core/matrix_utils.c`](../src/core/matrix_utils.c).

Utilidades de alocacion, inicializacion y comparacion. Independiente de la implementacion del kernel.

### 3.1 `xalloc_aligned`

```c
scalar_t *xalloc_aligned(size_t num_elements);
```

Aloja `num_elements * sizeof(scalar_t)` bytes con alineacion de 64 bytes via `posix_memalign`. Si la alocacion falla, imprime un mensaje a `stderr` y llama a `exit(EXIT_FAILURE)`. El bloque devuelto **no** esta inicializado.

El caller libera con `free(ptr)`.

### 3.2 `xfree`

```c
void xfree(scalar_t *ptr);
```

Wrapper sobre `free` que tolera `NULL`. Sirve para uniformizar el ciclo de vida.

### 3.3 `init_matrix_random`

```c
void init_matrix_random(scalar_t *M,
                        size_t rows, size_t cols,
                        unsigned int seed);
```

Llena `M` con valores pseudoaleatorios en $[-1/\sqrt{\text{rows}}, +1/\sqrt{\text{rows}}]$. La escala $1/\sqrt{\text{rows}}$ acota la norma espectral de $A$ en $\Theta(1)$ y evita overflow al iterar $B_{i+1} = A \cdot B_i$ muchas veces.

Usa un generador lineal congruencial con semilla `seed` para que la inicializacion sea **reproducible**: la misma semilla produce siempre los mismos valores, independientemente del compilador y la plataforma.

### 3.4 `init_matrix_zero`

```c
void init_matrix_zero(scalar_t *M, size_t rows, size_t cols);
```

Pone `rows * cols` ceros en `M`. Util para validacion (multiplicacion por matriz cero).

### 3.5 `init_matrix_identity`

```c
void init_matrix_identity(scalar_t *M, size_t n);
```

Llena `M` ($n \times n$) con la matriz identidad. Util para validacion (multiplicacion por identidad).

### 3.6 `matrices_close`

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

Compara dos buffers elemento a elemento usando una tolerancia mixta:

$$
|A_{\text{ref}}[i] - A_{\text{test}}[i]| \leq \max(\text{abs\_tol}, \text{rel\_tol} \cdot |A_{\text{ref}}[i]|)
$$

**Devuelve** 1 si todos los elementos pasan la prueba, 0 en caso contrario. Si se devuelve 0 y los punteros opcionales (`first_bad_index`, `bad_ref`, `bad_test`) son no nulos, escribe el primer indice fallido y los valores correspondientes (para diagnostico).

**Tolerancias recomendadas para `float`:**

- `abs_tol = 1e-5`
- `rel_tol = 1e-4`

Despues de varias iteraciones la suma de errores de redondeo crece como $\sqrt{m} \cdot \epsilon_{\text{mach}}$. Con $\epsilon_{\text{mach}} \approx 6 \cdot 10^{-8}$ y $m = 4096$, el error esperado es $\sim 4 \cdot 10^{-6}$. Las tolerancias dejan margen para variaciones de orden de operaciones.

---

## 4. Modulo `timing`

**Archivo:** [`src/core/timing.h`](../src/core/timing.h) (solo cabecera, sin `.c`).

### 4.1 `now_seconds`

```c
static inline double now_seconds(void);
```

Devuelve el tiempo actual en segundos como `double`, usando `clock_gettime(CLOCK_MONOTONIC, ...)`. `CLOCK_MONOTONIC` es inmune a ajustes de hora del sistema y tiene resolucion de nanosegundos en Linux moderno.

**Uso tipico:**

```c
double t0 = now_seconds();
benchmark_iterations(B_out, A, Z, m, n, I);
double t1 = now_seconds();
double elapsed = t1 - t0;
```

---

## 5. Binarios producidos

### 5.1 `bin/bench_naive_O0`

Compilado con `gcc -O0 -g`. Es el baseline obligatorio del proyecto.

**Uso:**

```
./bin/bench_naive_O0 <m> [num_iters] [num_runs]
```

- `<m>`: tamano del problema (entero positivo).
- `[num_iters]`: opcional. Iteraciones del benchmark dentro de cada corrida medida. Por defecto se usa $I_{\text{meas}} = \min(2m/n, 4)$ para mantener tiempos de medicion razonables durante el desarrollo.
- `[num_runs]`: opcional. Numero de corridas medidas para la mediana. Por defecto 5 (estabilidad estadistica). Los scripts de profiling usan 1 (una corrida determinista basta, ya que `gprof`/`perf` cuentan eventos absolutos, no estiman distribuciones).

**Salida:** una linea CSV en `stdout`:

```
kernel,m,n,num_iters,median_seconds,gflops
```

Internamente ejecuta una corrida de warm-up (no medida) y luego `num_runs` corridas medidas, reportando la mediana de los tiempos. Cuando `num_runs == 1` la "mediana" es trivialmente esa unica muestra.

### 5.2 `bin/validate_naive_O0`

Valida la implementacion sobre tres invariantes algebraicos: $A \cdot 0 = 0$, $I \cdot Z = Z$, $A \cdot (Z_1 + Z_2) = A \cdot Z_1 + A \cdot Z_2$. Imprime `VALIDATION OK` y retorna 0 si todas pasan; imprime detalles del fallo y retorna 1 en caso contrario.

**Uso:**

```
./bin/validate_naive_O0 [m]
```

Por defecto $m = 256$.

### 5.3 Sweep unificado: `make results`

El flujo unico de medicion vive en `make results`, que orquesta `scripts/run_perf_zen5_sweep.sh` (un sweep de hardware counters por celda `(variant, m)`) y `scripts/consolidate_perf_zen5.py` (union de los grupos A+B de eventos perf en una sola fila por celda). Salida canonica: `results/metrics.csv`.

Knobs (variables de entorno o argumentos del target):

| Variable | Default | Efecto |
|----------|---------|--------|
| `MS` | `"1024 2048 4096"` | Lista de tamaños $m$ a barrer. |
| `VARIANTS` | todas | Subconjunto de variantes a correr. |

---

## 6. Modulo `matmul_loops` (Fase 1.1 — cache-aware)

**Archivo:** [`src/algorithms/loops/matmul_loops.h`](../src/algorithms/loops/matmul_loops.h), [`src/algorithms/loops/matmul_loops.c`](../src/algorithms/loops/matmul_loops.c).

Las seis variantes de orden de bucles de $C = A \cdot B$. Misma firma que `matmul_naive`.

### 6.1 Tipo funcion-puntero

```c
typedef void (*matmul_fn_t)(scalar_t *C,
                             const scalar_t *A,
                             const scalar_t *B,
                             size_t m, size_t k, size_t n);
```

### 6.2 Seis kernels

```c
void matmul_ijk(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n);
void matmul_ikj(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n);
void matmul_jik(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n);
void matmul_jki(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n);
void matmul_kij(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n);
void matmul_kji(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n);
```

Precondiciones y postcondiciones identicas a `matmul_naive`. Las variantes con la dimension de reduccion no en el bucle interno (ikj, jki, kij, kji) hacen `memset(C, 0, ...)` internamente antes de acumular.

### 6.3 `matmul_loops_lookup`

```c
matmul_fn_t matmul_loops_lookup(const char *name);
```

Devuelve el puntero de funcion para el nombre dado (`"ijk"`, `"ikj"`, `"jik"`, `"jki"`, `"kij"`, `"kji"`), o `NULL` si el nombre no es reconocido.

### 6.4 `benchmark_iterations_loops`

```c
void benchmark_iterations_loops(scalar_t *B_out,
                                 const scalar_t *A,
                                 const scalar_t *Z,
                                 size_t m, size_t n,
                                 size_t num_iters,
                                 matmul_fn_t kernel);
```

Misma semantica que `benchmark_iterations` (Seccion 2.2) pero delegando cada paso $A \cdot B$ al `kernel` suministrado. Doble buffer + swap de punteros; aloja y libera los buffers internamente.

### 6.5 Binarios y scripts

| Binario | CLI | Salida |
|---------|-----|--------|
| `bin/bench_loops_O0` | `<order> <m> [num_iters] [num_runs]` | `kernel,m,n,num_iters,median_seconds,gflops` |
| `bin/validate_loops_O0` | `[m]` (default 256) | 4 tests por variante (3 invariantes + cross-val vs naive) |

`run_sweep_loops.sh` requiere un orden como argumento obligatorio para evitar que los kernels se midan en el mismo proceso (lo que contamina el estado de cache y el presupuesto termico entre ordenes):

```
scripts/run_sweep_loops.sh <order> ["<m list>"]
```

- `<order>`: uno de `ijk ikj jik jki kij kji` (obligatorio)
- `"<m list>"`: lista separada por espacios (opcional; default `256 384 512 768 1024 1536 2048 3072 4096`)
- Salida: `results/loop_<order>.csv`

Para correr los seis ordenes y obtener un CSV combinado usar `make sweep_loops_all`, que los encadena como procesos separados y concatena los resultados en `results/loop_order.csv`.

---

## 6.5 Roadmap de modulos por fase

A medida que se avanzan las fases del proyecto se anaden modulos manteniendo el mismo estilo. El estado consolidado y actualizado de cada fase vive en la seccion 9 del [`README.md`](../README.md); aqui se documenta el contrato de firma que todo nuevo kernel debe respetar:

Cada nuevo kernel debe exponer la firma:

```c
void mm(scalar_t *C,
        const scalar_t *A,
        const scalar_t *B,
        size_t m, size_t k, size_t n);
```

para que los binarios `validate_*` puedan compararlo contra `matmul_naive` sin cambios estructurales. Las precondiciones y postcondiciones son las de la Seccion 2.1.

---

## 8. Modulo `morton` (Fase 6, Etapa A3 - support)

**Archivo:** [`src/core/morton.h`](../src/core/morton.h), [`src/core/morton.c`](../src/core/morton.c).

Bit-interleaving Z-order y conversion entre layout row-major y Morton para matrices cuadradas. Implementacion Nivel 1 portatil (magic constants y shifts; sin BMI2 `pdep`/`pext`).

### 8.1 Convencion de bits

Para $(i, j)$ con representaciones binarias $i_{p-1} \ldots i_1 i_0$ y $j_{p-1} \ldots j_1 j_0$, el codigo Morton es:

$$
\text{morton}(i, j) = i_{p-1} j_{p-1} \ldots i_1 j_1 i_0 j_0
$$

**$j$ contribuye a los bits pares** (posiciones 0, 2, 4, ...) y **$i$ a los impares** (1, 3, 5, ...). Esta eleccion produce la tabla canonica para un sub-bloque 2x2:

| $(i, j)$ | codigo | cuadrante |
|---------|--------|-----------|
| (0, 0) | 0 | top-left (TL) |
| (0, 1) | 1 | top-right (TR) |
| (1, 0) | 2 | bottom-left (BL) |
| (1, 1) | 3 | bottom-right (BR) |

Esta es la **propiedad de contiguidad de cuadrantes** que el kernel `matmul_morton` (Seccion 9) explota: cuando un sub-bloque cuadrado de $A$ de lado `a_block_dim = 2 h` se divide en cuatro cuadrantes de lado $h$, los cuatro segmentos Morton respectivos ocupan posiciones $\{0, 1, 2, 3\} \cdot h^2$ a partir del offset del bloque padre, todas **contiguas en memoria**. No hay strides al recurrir.

### 8.2 `morton_encode`

```c
uint64_t morton_encode(uint32_t i, uint32_t j);
```

Intercala los bits de $i$ y $j$ segun la convencion anterior. Internamente usa `spread_bits_32_to_64`, un spread con magic constants de 5 pasos:

```c
y = (y | (y << 16)) & 0x0000FFFF0000FFFFULL;
y = (y | (y <<  8)) & 0x00FF00FF00FF00FFULL;
y = (y | (y <<  4)) & 0x0F0F0F0F0F0F0F0FULL;
y = (y | (y <<  2)) & 0x3333333333333333ULL;
y = (y | (y <<  1)) & 0x5555555555555555ULL;
```

El resultado es `spread(j) | (spread(i) << 1)`.

### 8.3 `morton_decode`

```c
void morton_decode(uint64_t code, uint32_t *i, uint32_t *j);
```

Inverso por compactacion de bits (mascaras y shifts en orden inverso). Solo para validacion.

### 8.4 `reorganize_to_morton` / `reorganize_from_morton`

```c
void reorganize_to_morton  (const scalar_t *A_row,   scalar_t *A_morton, size_t m);
void reorganize_from_morton(const scalar_t *A_morton, scalar_t *A_row,   size_t m);
```

Copian elemento a elemento entre row-major y Morton para una matriz cuadrada $m \times m$. El caller aloja el buffer destino con `xalloc_aligned(m * m)`. Ambas funciones llaman a `is_power_of_two(m)` y abortan con `fprintf(stderr, ...) + exit(EXIT_FAILURE)` si la pre-condicion no se cumple.

Complejidad: $O(m^2)$. En el contexto del benchmark, `reorganize_to_morton` se ejecuta **una sola vez** antes de la recurrencia $B_{i+1} = A \cdot B_i$ (que tiene $I = 2m/n$ iteraciones de $O(m^2 n)$ flops cada una), por lo que el costo amortizado es $O(1/I)$ del trabajo total y se considera despreciable.

### 8.5 `is_power_of_two`

```c
int is_power_of_two(size_t m);
```

Returns 1 if `m > 0 && (m & (m - 1)) == 0`, 0 otherwise.

---

## 9. Modulo `matmul_morton` (Fase 6, Etapa A3 - kernel)

**Archivo:** [`src/algorithms/morton/matmul_morton.h`](../src/algorithms/morton/matmul_morton.h), [`src/algorithms/morton/matmul_morton.c`](../src/algorithms/morton/matmul_morton.c).

Kernel recursivo donde $A$ esta en layout Morton (Z-order) y $B$, $C$ siguen en row-major. La recursion sobre $A$ se hace via offsets Morton en lugar de via `(puntero, leading dimension)`.

### 9.1 `matmul_morton`

```c
void matmul_morton(scalar_t *C,
                   const scalar_t *A_morton,
                   const scalar_t *B,
                   size_t m, size_t k, size_t n);
```

**Preconditciones.** $m == k$ ($A$ debe ser cuadrada) y `is_power_of_two(m)` (el indexing Z-order requiere subdivisiones exactas en mitades). Cualquier violacion produce abort con mensaje claro a stderr + `exit(EXIT_FAILURE)`. $A\_morton$ debe haber sido producido por `reorganize_to_morton(A, A_morton, m)`.

**Casos de recursion.** Sea `m_block`, `k_block`, `n_block` las dimensiones del sub-problema actual y `a_block_dim` el lado del sub-bloque cuadrado actual de $A$ (invariante: `m_block == k_block == a_block_dim`).

1. **Hoja.** $m\_block \cdot k\_block \cdot n\_block \leq$ `RECURSION_THRESHOLD`: kernel base con indexing Morton.
2. **Caso N.** $n\_block > a\_block\_dim$ y $n\_block \geq 2$: dividir $n$. Las dos sub-llamadas comparten $A$ (mismo `a_morton_offset`); las regiones de $C$ y $B$ son disjuntas, ambas sobreescriben.
3. **Caso MK.** $a\_block\_dim \geq 2$: dividir $m$ y $k$ simultaneamente. Cuatro productos sobre los cuadrantes Morton de $A$:

   $$
   C_{\text{top}} = A_{TL} B_{\text{top}} + A_{TR} B_{\text{bot}}, \qquad
   C_{\text{bot}} = A_{BL} B_{\text{top}} + A_{BR} B_{BR}
   $$

   Implementado como: TL sobreescribe $C_{\text{top}}$, TR acumula sobre $C_{\text{top}}$, BL sobreescribe $C_{\text{bot}}$, BR acumula sobre $C_{\text{bot}}$. Los offsets de los cuatro cuadrantes son `a_morton_offset + {0, 1, 2, 3} \cdot (\text{half} \cdot \text{half})`, todos contiguos en memoria por la propiedad de contiguidad de la Seccion 8.1.

4. **Fallback degenerado.** $a\_block\_dim = 1$ y $n\_block = 1$: kernel base.

**Indexing del kernel base.** El kernel hoja, para indices locales $(i, k)$ dentro del sub-bloque:

$$
A\_idx = a\_morton\_offset + \text{morton\_encode}(i, k)
$$

El invariante `m_block == k_block == a_block_dim` garantiza que `morton_encode(i, k)` se mantiene dentro de $[0, a\_block\_dim^2)$, por lo que `A_idx` queda dentro del segmento del sub-bloque.

**Complejidad.** $2 \cdot m \cdot k \cdot n$ flops (igual al baseline). La ganancia respecto a `matmul_naive` viene de **localidad** (no de operaciones): los cuatro cuadrantes del sub-bloque cuadrado de $A$ son contiguos en memoria gracias a la propiedad de la Seccion 8.1, eliminando los strides que un layout row-major produciria al dividir por $m$ y $k$ cuando los sub-bloques son mas anchos que la linea de cache. El esquema cache-oblivious recursivo alcanza asintoticamente $\Omega(n^3 / \sqrt{M})$ transferencias (Hong y Kung, 1981) **sin** conocer el tamano de cache $M$.

### 9.2 `benchmark_iterations_morton`

```c
void benchmark_iterations_morton(scalar_t *B_out,
                                 const scalar_t *A,
                                 const scalar_t *Z,
                                 size_t m, size_t n,
                                 size_t num_iters);
```

Misma semantica que `benchmark_iterations` pero usando `matmul_morton` como kernel. Internamente reorganiza $A$ a Morton una vez (cuenta dentro del tiempo total) y delega en `benchmark_iterations_morton_preorganized`. Util cuando la conversion es parte de la medicion.

### 9.3 `benchmark_iterations_morton_preorganized`

```c
void benchmark_iterations_morton_preorganized(scalar_t *B_out,
                                              const scalar_t *A_morton,
                                              const scalar_t *Z,
                                              size_t m, size_t n,
                                              size_t num_iters);
```

Misma logica que la anterior pero recibiendo $A$ **ya en Morton**. Usada por `bench_morton_O0` para que la reorganizacion no entre en el tiempo medido.

---

## 10. Binarios y scripts nuevos de la Fase 6

### 10.1 Binarios

| Binario | Archivo fuente | CLI | Salida |
|---------|----------------|-----|--------|
| `bin/bench_morton_O0`       | `bench_morton.c`       | `<m> [num_iters] [num_runs]`           | linea CSV `morton,m,n,num_iters,median_seconds,gflops`; aborta si $m$ no es potencia de 2 |
| `bin/validate_morton_O0`    | `validate_morton.c`    | `[m]` (default 256, potencia de 2)     | 7 tests: 3 invariantes + 4 cross-validation contra `matmul_naive` (en $m \in \{4, 16, 64, 256\}$) |
| `bin/test_morton`           | `test_morton.c`        | sin args                               | 4 grupos: tabla 4x4, round-trip encode/decode (4096 pares), contiguidad de cuadrantes para $m=8$, round-trip de reorganizacion para $m \in \{16, 64, 256\}$ |

Los binarios de bench reusan el patron del baseline: 1 warm-up + `num_runs` corridas medidas con mediana, `num_iters` default = $\min(2m/n, 4)$, semillas 42 ($A$) y 43 ($Z$). En `bench_morton_O0` la reorganizacion a Morton se ejecuta una sola vez **antes** del warm-up para que el tiempo cronometrado sea solo el del kernel.

### 10.2 Sweeps

| Script | Lista por defecto de $m$ | Salida CSV |
|--------|--------------------------|------------|
| `scripts/run_sweep_morton.sh`    | $\{1024, 2048, 4096, 8192\}$; filtra y omite no-potencias-de-2 con `Warning:` a stderr | `results/morton_O0.csv` |

Misma CLI que `run_sweep_naive.sh`: el listado de $m$ se puede pasar como primer argumento.

### 10.3 Profiling con perf

Para comparaciones entre kernels usar el pipeline unificado de la Sesion 03: `make profile_zen2` (o `make results`) corre `scripts/run_perf_zen5_sweep.sh` sobre las 13 variantes activas en $m \in \{1024, 4096, 8192\}$ y el consolidador `scripts/consolidate_perf_zen5.py` produce `results/metrics.csv`. Ese pipeline reemplaza el comparador ad-hoc (`profile_perf_compare.sh` / `plot_perf_compare.py`) que se uso en Sesion 02.

Si `perf_event_paranoid` esta demasiado restrictivo, el script aborta con mensaje claro indicando el comando exacto para arreglarlo y la referencia a la seccion 3.3 del `README.md`.

### 10.4 Targets de Makefile

```
make test_morton                -> bin/test_morton
make bench_morton               -> bin/bench_morton_O0
make validate_morton            -> bin/validate_morton_O0
make sweep_morton_run           -> bash scripts/run_sweep_morton.sh
```

Targets de Fase 1.1 (loop-reorder):

```
make bench_loops                -> bin/bench_loops_O0
make validate_loops             -> bin/validate_loops_O0
make sweep_loops_ijk            -> results/loop_ijk.csv  (proceso independiente)
make sweep_loops_ikj            -> results/loop_ikj.csv
make sweep_loops_jik            -> results/loop_jik.csv
make sweep_loops_jki            -> results/loop_jki.csv
make sweep_loops_kij            -> results/loop_kij.csv
make sweep_loops_kji            -> results/loop_kji.csv
make sweep_loops_all            -> los seis anteriores + results/loop_order.csv (combinado)
make plot_naive                 -> plots/naive_O0.png  (solo baseline)
make plot_loop                  -> plots/loop_orders.png  (6 ordenes desde loop_order.csv)
make plot_loop_vs_naive         -> plots/loop_vs_naive.png  (naive + 6 ordenes)
```

Targets de Fase 1.3 (tiled_ikj_avx512):

```
make bench_tiled_ikj_avx512_ZEN5      -> bin/bench_tiled_ikj_avx512_ZEN5
make validate_tiled_ikj_avx512_ZEN5   -> bin/validate_tiled_ikj_avx512_ZEN5
```

Ambos se compilan con `CFLAGS_O3_ZEN5` (`-O3 -march=native` mas los `-D...` de los thresholds del Makefile), que es obligatorio para que `_mm512_fmadd_ps` emita la instruccion FMA-512 real. El target `make results` incluye `bench_tiled_ikj_avx512_ZEN5` como dependencia y `scripts/run_perf_zen5_sweep.sh` incluye `tiled_ikj_avx512` en su lista de variantes por defecto.

Todos extienden el Makefile **al final**, sin modificar las recetas del baseline (`bench_naive_O0`, `bench_naive_pg`, `validate_naive`, `sweep_naive`, `profile_*_naive`, `clean`, `distclean`).

---

## 11. Microkernels AVX-512 (rama `main_server`, Zen 5)

La rama `main_server` retira por completo los microkernels AVX2 (4x16 y 6x16) que servian para el Ryzen 5 4600H (Zen 2) y los reemplaza por dos microkernels AVX-512 sintonizados al servidor AWS c8a.2xlarge (AMD EPYC 9R45, Zen 5):

| Header | Tile | Familia | Acumuladores | Broadcasts | Comentario |
|---|---|---|---|---|---|
| `src/microkernels/kernel_avx512_morton.h` | $4 \times 32$ | Morton | 8 ZMM | 4 ZMM | `MR = MORTON_AVX512_TILE = 4` |
| `src/microkernels/kernel_avx512_tiled.h`  | $6 \times 32$ | tiled_ikj | 12 ZMM | 1 ZMM (reusado) | + `residual_rows` AVX-512 |

Ambos son **header-only** con `static inline`: cada TU que los incluye obtiene una copia inlineada bajo `-O3`. No hay `.o` separado, y por lo tanto no hay riesgo de spill de ZMMs a traves del ABI en una llamada de funcion. Ese era el modelo bajo el que estaba el 6x16 en Zen 2 (`kernel_avx2_tiled.h`), y es el patron unificado que tambien usa Juan Pablo en la rama `opt_zen5`.

### 11.1 `kernel_avx512_4x32` (familia Morton)

```c
static inline void
kernel_avx512_4x32(scalar_t       *restrict C, size_t ldc,
                   const scalar_t *restrict A, size_t lda,
                   const scalar_t *restrict B, size_t ldb,
                   size_t kc);
```

Acumula $C \mathrel{+}= A \cdot B$ sobre un tile fijo $4 \times 32$. Layout de los 8 ZMM acumuladores:

$$
\begin{array}{c|cc}
 & \text{cols } 0..15 & \text{cols } 16..31 \\\hline
\text{row 0} & c_{00} & c_{01} \\
\text{row 1} & c_{10} & c_{11} \\
\text{row 2} & c_{20} & c_{21} \\
\text{row 3} & c_{30} & c_{31}
\end{array}
$$

Cada paso de $kc$ emite 8 vfmadd231ps independientes. El EPYC 9R45 tiene dos pipes FMA de $512$ bits, asi que 8 FMAs son 4 ciclos de computo por paso (igual al kernel 4x16 de Zen 2, pero con el doble de ancho por instruccion: $32$ lanes vs $16$).

**Precondiciones:** `kc >= 1`, `lda >= kc`, `ldb >= 32`, `ldc >= 32`, no aliasing. Alineacion a $64$ bytes preferida pero no obligatoria (usa `_mm512_loadu_ps`).

### 11.2 `kernel_avx512_tiled_6x32` (familia tiled_ikj)

```c
static inline void
kernel_avx512_tiled_6x32(scalar_t       *restrict C, size_t ldc,
                         const scalar_t *restrict A, size_t lda,
                         const scalar_t *restrict B, size_t ldb,
                         size_t kc);
```

Acumula $C \mathrel{+}= A \cdot B$ sobre un tile fijo $6 \times 32$. Layout de los 12 ZMM acumuladores ($15$ ZMM activos de los $32$ totales):

$$
\begin{array}{c|cc}
 & \text{cols } 0..15 & \text{cols } 16..31 \\\hline
\text{row 0} & c_{00} & c_{01} \\
\text{row 1} & c_{10} & c_{11} \\
\text{row 2} & c_{20} & c_{21} \\
\text{row 3} & c_{30} & c_{31} \\
\text{row 4} & c_{40} & c_{41} \\
\text{row 5} & c_{50} & c_{51}
\end{array}
$$

A diferencia del kernel Morton, el broadcast de $A$ se **reusa** sobre las 6 filas (un solo registro ZMM); el renombrador fisico resuelve el WAW hazard sin penalizacion. Cada paso de $kc$ emite 12 FMAs ($6$ ciclos a 2 FMAs/ciclo).

### 11.3 `kernel_avx512_tiled_residual_rows`

Fallback AVX-512 para las colas $m \bmod \text{MR}$ y $n \bmod \text{NR}$. Vectoriza por filas con `_mm512_*` pero no register-blockea $C$: paga `load/FMA/store` por cada par $(r, p)$. Solo se activa para hasta $\text{MR} - 1 = 5$ filas residuales por invocacion, asi que el costo es despreciable a $m \gg \text{MR}$.

### 11.4 Constantes de geometria

```c
/* kernel_avx512_morton.h */
#define KERNEL_AVX512_MORTON_MR  4u
#define KERNEL_AVX512_MORTON_NR 32u

/* kernel_avx512_tiled.h */
#define KERNEL_AVX512_TILED_MR  6u
#define KERNEL_AVX512_TILED_NR 32u
```

Expuestas para que `matmul_morton_avx512`, `matmul_tiled_ikj_avx512` y los tests calcen sus bloques al tile sin redeclarar las magic numbers.

### 11.5 Compilacion

No hay `.o` separado. Cada TU que incluye los headers se compila con los flags Zen 5 estandar del proyecto:

```
-O3 -march=native -D_POSIX_C_SOURCE=200809L \
   -DMORTON_AVX512_THRESHOLD_DEFAULT=1048576UL \
   -DTILED_IKJ_AVX512_BS_DEFAULT=256u -DTILED_IKJ_AVX512_MC=288u  ...
```

`-march=native` activa AVX-512F / VL / BW / DQ / IFMA, BMI2, AVX2, FMA, y todo lo demas que el EPYC 9R45 expone vista CPUID. GCC 11 puede no reconocer `-march=znver4/5` por nombre; `-march=native` es la forma portable y conservadora.

### 11.6 Performance esperada (techo single-core)

Zen 5 retira 2 FMA de 512 bits por ciclo = $2 \times 16 \times 2 = 64$ flops por ciclo en FP32. A frecuencia turbo aprox. $3.5$ GHz bajo carga AVX-512 sostenida, el techo es $\sim 224$ GFLOPS por core. Con 8 cores y L3 compartida de 32 MiB el techo agregado en el chip es $\sim 1.8$ TFLOPS. Cuanto de ese techo toca el kernel depende del bandwidth de los paneles de $B$ desde L2/L3 y del costo de materializacion del panel de $A$ en el leaf de Morton.

---

## 12. Modulo `matmul_morton_avx512` (Sesion 03, Etapa A4 integracion)

**Archivo:** [`src/algorithms/morton/matmul_morton_avx512.h`](../src/algorithms/morton/matmul_morton_avx512.h), [`src/algorithms/morton/matmul_morton_avx512.c`](../src/algorithms/morton/matmul_morton_avx512.c).

Variante de `matmul_morton` cuyo leaf invoca el microkernel AVX-512 (`kernel_avx512_4x32`) de la Seccion 11. La recursion sigue la misma estructura cache-oblivious (Caso N independiente, Caso MK acoplado en cuatro cuadrantes) pero el layout de $A$ cambia.

### 12.1 Layout Morton-de-bloques (tile = 4)

Sesion 02 (Morton "fino"): cada elemento $A[i, j]$ ocupa la posicion `morton_encode(i, j)`. Sesion 03 (Morton "de bloques"): $A$ se particiona en sub-bloques $\text{MR} \times \text{MR}$ con $\text{MR} = 4$; los sub-bloques se Z-ordenan entre si, y los $16$ elementos de cada sub-bloque quedan en row-major. La posicion de $A[i, j]$ es:

$$
A\_idx = \text{morton\_encode}(i / \text{MR}, j / \text{MR}) \cdot \text{MR}^2 + (i \bmod \text{MR}) \cdot \text{MR} + (j \bmod \text{MR})
$$

La **propiedad de contiguidad** (Seccion 8.1) se preserva: cuatro cuadrantes de lado $h$ siguen ocupando offsets $\{0, 1, 2, 3\} \cdot h^2$ desde el padre. Solo cambia el significado del nivel hoja: bloque $4 \times 4$ de floats en lugar de un solo float. El layout fino y el de bloques **coexisten**; `matmul_morton.{c,h}` queda intacto.

### 12.2 `matmul_morton_avx512`

```c
void matmul_morton_avx512(scalar_t *C,
                        const scalar_t *A_morton,
                        const scalar_t *B,
                        size_t m, size_t k, size_t n);
```

**Computa** $C = A_{\text{morton}} \cdot B$ donde `A_morton` esta en Morton-de-bloques (Seccion 12.1). Misma forma que `matmul_naive`: $C$ es $m \times n$ (out), $A$ es $m \times k$ (in, Morton-de-bloques), $B$ es $k \times n$ (in, row-major).

**Precondiciones:**

- $m = k$ (matriz cuadrada).
- $m$ potencia de $2$ y $m \geq \text{MR} = 4$.
- $n \geq \text{NR} = 16$ (recomendado: $n$ multiplo de $\text{NR}$; las trozas no alineadas caen al fallback `ijk` sin vectorizar).
- `A_morton` producido por `reorganize_to_morton_blocks` (Seccion 12.4).
- `C` no aliasa con `A_morton` ni con $B$.

Cualquier violacion de los chequeos sobre $m$ y $k$ aborta con `fprintf(stderr, ...) + exit(EXIT_FAILURE)`, igual que `matmul_morton`.

**Tolerancia de validacion.** $\text{abs\_tol} = 10^{-3}$ relativo (contra $10^{-4}$ del Morton fino). El relajamiento es necesario porque `-ffast-math` autoriza reasociacion de suma FP en el kernel, acumulando mas error.

### 12.3 Threshold de hoja y ajuste empirico

```c
extern size_t g_recursion_threshold_avx512;
void matmul_morton_avx512_set_threshold(size_t threshold);
```

Variable global con default $64 \cdot 64 \cdot 128 = 524288$ flops elementales, equivalente a una hoja de $64 \times 64$ floats por panel de $A$ (working set $\sim 16$ KiB, mitad de L1d en el $4600$H). El setter acepta cualquier valor positivo; pasar $0$ imprime un warning y deja el default. La constante esta separada de `g_recursion_threshold` (Sesion 02) porque los regimenes son distintos: el microkernel AVX-512 amortiza una hoja mucho mas grande que el `ijk + morton_encode` ingenuo, asi que el threshold optimo es mayor.

### 12.4 `reorganize_to_morton_blocks`

```c
void reorganize_to_morton_blocks(const scalar_t *A_row,
                                 scalar_t *A_morton,
                                 size_t m);
```

Reorganiza una matriz row-major $m \times m$ al layout Morton-de-bloques consumido por `matmul_morton_avx512`. Aborta si $m$ no es multiplo de $\text{MR}$ o si $m / \text{MR}$ no es potencia de $2$. El caller aloja `A_morton` con capacidad para $m^2$ elementos (tipicamente `xalloc_aligned`).

Complejidad: $O(m^2)$. Se ejecuta una sola vez antes de la recurrencia $B_{i+1} = A \cdot B_i$, igual que en Sesion 02.

### 12.5 Orquestadores

```c
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
```

Misma semantica que sus contrapartes en `matmul_morton`. El primero reorganiza $A$ internamente (la conversion entra en el tiempo medido); el segundo recibe $A$ ya reorganizado y es el que usa `bench_morton_avx512_ZEN5`.

---

## 13. Modulo `matmul_morton_omp` (Sesion 03, Etapa A5)

**Archivo:** [`src/algorithms/morton/matmul_morton_omp.h`](../src/algorithms/morton/matmul_morton_omp.h), [`src/algorithms/morton/matmul_morton_omp.c`](../src/algorithms/morton/matmul_morton_omp.c).

Variante paralela de `matmul_morton_avx512`. Reusa el microkernel AVX-512 y el layout Morton-de-bloques; agrega `#pragma omp parallel single` en el wrapper publico y emite OpenMP tasks en cada subdivision recursiva por encima del threshold de paralelizacion. El scratch buffer del leaf pasa a ser un pool por-thread indexado por `omp_get_thread_num()` para que las hojas paralelas no compartan memoria intermedia.

### 13.1 `matmul_morton_omp`

```c
void matmul_morton_omp(scalar_t *C,
                       const scalar_t *A_morton,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n);
```

**Contrato de forma** identico a `matmul_morton_avx512`: $C$ es $m \times n$ (out), `A_morton` es $m \times k$ en Morton-de-bloques, $B$ es $k \times n$ row-major, mismas precondiciones ($m = k$ potencia de $2$, $m \geq \text{MR}$).

### 13.2 Thresholds (dos knobs independientes)

```c
extern size_t g_recursion_threshold_omp;
extern size_t g_parallel_threshold_omp;
void matmul_morton_omp_set_threshold         (size_t threshold);
void matmul_morton_omp_set_parallel_threshold(size_t threshold);
```

- `g_recursion_threshold_omp` (default $524288$): tamano del sub-problema en que la recursion cae al leaf kernel. Mismo rol que `g_recursion_threshold_avx512`.
- `g_parallel_threshold_omp` (default $524288$): tamano por debajo del cual la recursion deja de emitir `omp task` y corre inline. Con el default igual al leaf threshold, las tasks disparan en cada nivel sobre la hoja y nunca dentro de ella.

Las globals se mantienen separadas de las de `matmul_morton_avx512` para poder tunear la variante paralela sin alterar las mediciones del modulo serial.

### 13.3 Variables de entorno relevantes

| Variable | Efecto | Default usado en el bench |
|----------|--------|---------------------------|
| `OMP_NUM_THREADS` | Numero de threads. Si se omite, OpenMP usa todos los logicos (8 en c8a.2xlarge: el hipervisor desactiva SMT). | 8 (un thread por core). |
| `OMP_PROC_BIND`   | `close` y `spread` son topologicamente equivalentes en c8a.2xlarge (1 NUMA node, L3 compartido). | `close`. |
| `OMP_PLACES`      | `cores` une cada thread a un core fisico. | `cores`. |

### 13.4 Recomendacion para el EPYC 9R45 (AWS c8a.2xlarge, Zen 5)

La VM expone $8$ vCPUs en un solo NUMA node, sin SMT (1 thread por core). L3 ($32$ MiB) es compartida por todos los cores, asi que la topologia "CCX private L3" del Zen 2 no aplica aqui: cualquier thread puede aprovechar el L3 completo cuando los otros lo dejan libre. Configuracion recomendada:

- **`OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close`**: usa todos los cores, un thread por core. Es el default usado por el sweep `scripts/run_omp_scaling.sh` y por `scripts/profile_perf_zen5.sh` cuando `VARIANT=morton_omp`. Speedup cercano a lineal hasta saturar el ancho de banda a L3 / DRAM.
- **`OMP_NUM_THREADS=4`**: util para diagnostico de scaling (mitad del chip). Spawn de tasks reducido, menos presion sobre L3.

No hay regimen `OMP_NUM_THREADS=16` aqui: SMT esta desactivado a nivel hipervisor y `8` es el limite duro del VM.

### 13.5 Orquestadores

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

Misma estructura que en los modulos anteriores: el primero reorganiza $A$ internamente y la conversion entra en el tiempo medido; el segundo recibe $A$ ya en Morton-de-bloques y es el que usa `bench_morton_omp_O3`.

---

## 14. Modulo `matmul_tiled_ikj_avx512` (BLIS-style 6x32 register-blocked, Zen 5)

**Archivo:** [`src/algorithms/tiled_ikj/matmul_tiled_ikj_avx512.h`](../src/algorithms/tiled_ikj/matmul_tiled_ikj_avx512.h), [`src/algorithms/tiled_ikj/matmul_tiled_ikj_avx512.c`](../src/algorithms/tiled_ikj/matmul_tiled_ikj_avx512.c). El microkernel 6x32 AVX-512 vive en [`src/microkernels/kernel_avx512_tiled.h`](../src/microkernels/kernel_avx512_tiled.h) (header-only `static inline`, compartido con `matmul_tiled_ikj_omp`).

Kernel BLIS-style con micropanel registrado $6 \times 32$, sintonizado al EPYC 9R45 (Zen 5). Reemplaza la version Zen 2 (microkernel $6 \times 16$ con YMM) por un microkernel inline que mantiene un sub-tile $6 \times 32$ de $C$ en $12$ registros ZMM durante toda la pasada $k_c$; $C$ toca memoria solo dos veces por micro-tile (load al entrar, store al salir).

### 14.1 Geometria del microkernel (compile-time)

```c
#define TILED_IKJ_AVX512_MR 6u
#define TILED_IKJ_AVX512_NR 32u
#define TILED_IKJ_AVX512_MC 288u
#define TILED_IKJ_AVX512_BS_DEFAULT 256u
extern size_t g_tiled_ikj_avx512_bs;
```

- $\text{MR} = 6$, $\text{NR} = 32$: filas y columnas del tile registrado. Activan $15$ de los $32$ registros ZMM arquitecturales ($12$ acumuladores $C$ + $2$ vectores $B$ + $1$ broadcast $A$ reusado entre filas). El renombrador fisico del Zen 5 resuelve la dependencia WAW sobre el broadcast sin stall.
- $\text{MC} = 288 = 48 \times \text{MR}$: tamano del bloque sobre $m$ (multiplo de $\text{MR}$). El panel $A$ activo $\text{MC} \times k_c$ a $k_c = 256$ ocupa $288$ KiB y cabe holgado en el L2 de $1$ MiB por core.
- $\text{BS}$ (sinonimo `kc`): tamano del bloque sobre $k$. Default $256$, configurable runtime via `matmul_tiled_ikj_avx512_set_bs(bs)`. El panel $B$ activo $k_c \times \text{NR}$ a $k_c = 256$ ocupa $32$ KiB, que cabe en el L1d de $48$ KiB con margen para los acumuladores activos de $C$.

### 14.2 `matmul_tiled_ikj_avx512_set_bs`

```c
void matmul_tiled_ikj_avx512_set_bs(size_t bs);
```

Cambia $k_c$ en runtime. Solo se rechaza `bs == 0` (cualquier valor positivo es valido; no se requiere multiplo de $8$ porque el microkernel itera $p$ uno a la vez). Util para el `sweep_threshold` y para el barrido manual del bs.

### 14.3 `matmul_tiled_ikj_avx512`

```c
void matmul_tiled_ikj_avx512(scalar_t *C,
                       const scalar_t *A,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n);
```

**Computa** $C = A \cdot B$ usando el loop nest BLIS Goto-style:

```
pc  loop  step kc  (= g_tiled_ikj_avx512_bs, default 256)
  ic loop step mc  (= TILED_IKJ_AVX512_MC, default 288)
    jr loop step nr (= TILED_IKJ_AVX512_NR, fixed 32)
      ir loop step mr (= TILED_IKJ_AVX512_MR, fixed 6)
        kernel_avx512_tiled_6x32: kc FMAs accumulating in 12 ZMM registers
```

**Parametros y precondiciones.** Identicos a `matmul_naive` (Seccion 2.1): $C$ es $m \times n$ (out, sobrescrito), $A$ es $m \times k$, $B$ es $k \times n$, todos row-major, sin aliasing. El kernel maneja bordes en $m$ y $n$ con un fallback vectorizado AVX-512 que no registra $C$ (`kernel_avx512_tiled_residual_rows` en [`kernel_avx512_tiled.h`](../src/microkernels/kernel_avx512_tiled.h)).

**Postcondiciones.** $C_{ij} = \sum_{p=0}^{k-1} A_{ip} B_{pj}$ para todo $(i, j)$.

**Microkernel inline** (esquema, $15$ ZMM vivos por iteracion del bucle $p$):

```c
// 12 acumuladores de C cargados una sola vez por (ir, jr) micro-tile
__m512 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
for (size_t p = 0; p < kc; ++p) {
    __m512 b0 = _mm512_loadu_ps(&B[p*ldb +  0]);
    __m512 b1 = _mm512_loadu_ps(&B[p*ldb + 16]);
    __m512 a  = _mm512_set1_ps(A[0*lda + p]);
    c00 = _mm512_fmadd_ps(a, b0, c00);
    c01 = _mm512_fmadd_ps(a, b1, c01);
    // ... idem para filas 1..5 ...
}
// store de los 12 acumuladores una sola vez
```

`memset(C, 0, m*n*sizeof(scalar_t))` al inicio permite que cada iteracion del bucle `pc` cargue $C$, acumule sobre el, y lo escriba de vuelta — agregando consistentemente las contribuciones parciales de cada panel $k_c$.

**Compilacion.** Requiere `-O3 -march=native` (AVX-512F mas vecinos). El microkernel vive en [`kernel_avx512_tiled.h`](../src/microkernels/kernel_avx512_tiled.h) como `static inline`; GCC lo inline en el bucle `ir` y mantiene los $12$ acumuladores en ZMMs sin spilling al stack.

**Complejidad.** $2 \cdot m \cdot k \cdot n$ flops (identica al baseline). La ganancia frente a la version $6$-loop simple es de **densidad aritmetica**: por iteracion del bucle interno $p$ se hacen $12$ FMAs ZMM (retire $6$ ciclos en los dos pipes FMA-512 del Zen 5) contra $2$ loads + $6$ broadcasts (no en el camino critico). El techo single-core teorico del EPYC 9R45 es $\sim 224$ GFLOPS FP32 a $3.5$ GHz turbo bajo AVX-512 sostenido.

### 14.4 `benchmark_iterations_tiled_ikj_avx512`

```c
void benchmark_iterations_tiled_ikj_avx512(scalar_t *B_out,
                                          const scalar_t *A,
                                          const scalar_t *Z,
                                          size_t m, size_t n,
                                          size_t num_iters);
```

Mismo patron que `benchmark_iterations` (Seccion 2.2): doble buffer + swap de punteros, aloja y libera internamente. Invoca `matmul_tiled_ikj_avx512` en cada iteracion de la recurrencia.

### 14.5 Binarios

| Binario | CLI | Salida CSV |
|---------|-----|------------|
| `bin/bench_tiled_ikj_avx512_ZEN5` | `<m> [num_iters] [num_runs] [bs]` | `tiled_ikj_avx512,m,n,num_iters,bs,median_seconds,gflops` (7 columnas) |
| `bin/validate_tiled_ikj_avx512_ZEN5` | `[m] [bs]` (defaults: m=256, bs=256) | 4 tests: `A*0==0`, `I*Z==Z`, linealidad, cross contra naive |

El cuarto argumento opcional `[bs]` llama a `matmul_tiled_ikj_avx512_set_bs(bs)` antes de las corridas. La columna `bs` del CSV preserva el formato de 7 columnas que ya manejaba el consolidador `consolidate_perf_zen5.py` (rama `len(parts) >= 7`).

**Tolerancias de validacion:** `ABS_TOL = 1e-4f`, `REL_TOL = 1e-3f`.

---

## 15. Modulo `matmul_tiled_ikj_omp` (kernel 6x32 AVX-512 + OpenMP `parallel for` en `ic`, Zen 5)

**Archivo header:** [`src/algorithms/tiled_ikj/matmul_tiled_ikj_omp.h`](../src/algorithms/tiled_ikj/matmul_tiled_ikj_omp.h)
**Implementacion:** [`src/algorithms/tiled_ikj/matmul_tiled_ikj_omp.c`](../src/algorithms/tiled_ikj/matmul_tiled_ikj_omp.c) (microkernel compartido en [`src/microkernels/kernel_avx512_tiled.h`](../src/microkernels/kernel_avx512_tiled.h))
**Compilacion requerida:** `-O3 -march=native -fopenmp`

Hermano paralelo de `matmul_tiled_ikj_avx512` (Modulo 14). Mismo microkernel registrado $6 \times 32$ AVX-512, mismo loop nest, **pero el bucle externo $i_c$ esta distribuido entre threads** con `#pragma omp for schedule(static)`. La region `omp parallel` se abre una sola vez por invocacion y abarca el bucle $p_c$ entero; el barrier implicito al final de cada `omp for` sincroniza las pasadas $p_c$ (necesario porque $C$ se acumula entre pasadas).

### 15.1 Constantes y global

```c
#define TILED_IKJ_OMP_MR 6u
#define TILED_IKJ_OMP_NR 32u
#define TILED_IKJ_OMP_MC 288u
#define TILED_IKJ_OMP_BS_DEFAULT 256u
extern size_t g_tiled_ikj_omp_bs;
```

Misma geometria que `matmul_tiled_ikj_avx512`. El microkernel vive en el header `kernel_avx512_tiled.h` como `static inline`; cada TU que lo incluye obtiene su copia inlineada bajo `-O3`, evitando el spilling de los $12$ acumuladores ZMM a traves del ABI.

### 15.2 Funciones publicas

```c
void matmul_tiled_ikj_omp_set_bs(size_t bs);

void matmul_tiled_ikj_omp(scalar_t *C,
                      const scalar_t *A,
                      const scalar_t *B,
                      size_t m, size_t k, size_t n);

void benchmark_iterations_tiled_ikj_omp(scalar_t *B_out,
                                    const scalar_t *A,
                                    const scalar_t *Z,
                                    size_t m, size_t n,
                                    size_t num_iters);
```

Mismo contrato externo que las contrapartes seriales:

- $C$ ($m \times n$) se inicializa con `memset` a cero secuencialmente (fuera del region paralelo).
- $A$ ($m \times k$) y $B$ ($k \times n$) son solo lectura, compartidas entre threads.
- Cada thread procesa un rango disjunto del loop $i_c$, escribiendo exclusivamente en filas $[i_c, i_c + m_c)$ de $C$ — sin conflictos de escritura ni false sharing entre tiles diferentes ($\text{MC} \cdot n \cdot 4 = 192 \cdot 128 \cdot 4 = 96$ KiB por bloque, varios ordenes de magnitud por encima de la linea de cache).
- El numero de threads lo fija `OMP_NUM_THREADS` antes de invocar el binario.

### 15.3 Recomendacion para el EPYC 9R45 (AWS c8a.2xlarge, Zen 5)

Configuracion recomendada: **`OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close`**. La VM expone $8$ vCPUs en un solo NUMA node con SMT deshabilitada por el hipervisor, asi que `close` y `spread` son topologicamente equivalentes (todos los cores comparten el mismo L3 de $32$ MiB).

### 15.4 Binarios

| Binario | Target make | Flags |
|---------|-------------|-------|
| `bin/bench_tiled_ikj_omp_ZEN5`    | `bench_tiled_ikj_omp_ZEN5`    | `CFLAGS_OMP_ZEN5` (`-O3 -march=native -fopenmp` + thresholds) |
| `bin/validate_tiled_ikj_omp_ZEN5` | `validate_tiled_ikj_omp_ZEN5` | idem |

**CLI bench:** `bench_tiled_ikj_omp_ZEN5 <m> [num_iters] [num_runs] [bs]`

**Salida CSV** (7 columnas, identica a `tiled_ikj_avx512`):
```
tiled_ikj_omp,m,n,num_iters,bs,median_seconds,gflops
```

**Integracion en el pipeline perf:** `scripts/profile_perf_zen5.sh` fija `OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close` cuando `VARIANT=tiled_ikj_omp`. `scripts/run_perf_zen5_sweep.sh` incluye `tiled_ikj_omp` en su array `VARIANTS` por defecto. `scripts/consolidate_perf_zen5.py` parsea la salida.

---

## 16. Cambios y versionado

Este documento se actualiza con cada PR que toque la API publica. La regla es: **si una firma de funcion cambia, este documento debe cambiar en el mismo commit**.
