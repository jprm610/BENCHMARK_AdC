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

Definido en [`src/matmul_naive.h`](../src/matmul_naive.h). Cambiar a `double` requeriria reemplazar este `typedef` y revisar tolerancias en `validate_naive.c`.

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

**Archivo:** [`src/matmul_naive.h`](../src/matmul_naive.h), [`src/matmul_naive.c`](../src/matmul_naive.c).

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

**Notas:** esta funcion es el baseline obligatorio del proyecto. **No** debe modificarse para introducir optimizaciones; las versiones optimizadas iran en nuevos modulos (`matmul_reordered.c`, `matmul_tiled.c`, etc.) con firmas analogas.

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

**Archivo:** [`src/matrix_utils.h`](../src/matrix_utils.h), [`src/matrix_utils.c`](../src/matrix_utils.c).

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

**Archivo:** [`src/timing.h`](../src/timing.h) (solo cabecera, sin `.c`).

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

### 5.2 `bin/bench_naive_pg`

Igual que `bench_naive_O0` pero compilado adicionalmente con `-pg` para soportar `gprof`. Misma CLI. Produce `gmon.out` en el cwd al ejecutarse.

### 5.3 `bin/validate_naive_O0`

Valida la implementacion sobre tres invariantes algebraicos: $A \cdot 0 = 0$, $I \cdot Z = Z$, $A \cdot (Z_1 + Z_2) = A \cdot Z_1 + A \cdot Z_2$. Imprime `VALIDATION OK` y retorna 0 si todas pasan; imprime detalles del fallo y retorna 1 en caso contrario.

**Uso:**

```
./bin/validate_naive_O0 [m]
```

Por defecto $m = 256$.

### 5.4 `scripts/run_sweep_naive.sh`

Orquesta los tres primeros pasos del proyecto en una sola pasada. Para cada $m$ del listado:

1. Ejecuta `bin/bench_naive_O0 <m>` (5 corridas + mediana) y agrega la linea CSV a `results/naive_O0.csv`.
2. Ejecuta `bin/bench_naive_pg <m> <PROFILE_ITERS> <PROFILE_RUNS>` bajo `gprof`, guardando `results/gprof_naive_m<m>.txt`.
3. Ejecuta `bin/bench_naive_O0 <m> <PROFILE_ITERS> <PROFILE_RUNS>` bajo `perf stat`, guardando `results/perf_naive_m<m>.txt`.

**Variables de entorno:**

| Variable | Default | Efecto |
|----------|---------|--------|
| `PROFILING` | `full` | `full`/`gprof`/`perf`/`0` para escoger que profilers correr. |
| `PROFILE_ITERS` | `1` | Iteraciones del benchmark dentro de cada corrida profileada. |
| `PROFILE_RUNS` | `1` | Corridas medidas bajo el profiler (usar 1 minimiza overhead). |

**Argumento posicional:**

```
scripts/run_sweep_naive.sh [m_list]
```

Si se omite, usa el listado por defecto $\{256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192\}$.

### 5.5 `scripts/profile_gprof_naive.sh` y `scripts/profile_perf_naive.sh`

Scripts standalone equivalentes a un paso del sweep. CLI uniforme:

```
scripts/profile_gprof_naive.sh <m> [num_iters] [num_runs]
scripts/profile_perf_naive.sh  <m> [num_iters] [num_runs]
```

Defaults: `m=2048, num_iters=1, num_runs=1`. Ambos respetan el contrato CLI extendido de `bench_naive_O0`/`bench_naive_pg`.

### 5.6 `scripts/plot_results.py`

Graficador general para cualquier combinacion de CSVs del proyecto. Acepta uno o mas archivos CSV como argumentos posicionales, agrupa las filas por la columna `kernel` y produce tres archivos con el prefijo `--out`:

- `<out>.png` — GFLOP/s vs m, una curva por kernel, eje x logaritmico base 2
- `<out>_time.png` — tiempo por iteracion vs m (log-log) + referencia teorica $O(m^2 n)$
- `<out>_combined.csv` — union de todas las filas de entrada, deduplicadas por `(kernel, m)`

```
scripts/plot_results.py [csvs ...]
                        [--out BASE_PATH]
                        [--title STRING]
                        [--l1-kb K] [--l2-kb K] [--l3-kb K]
                        [--cpu-label STRING]
```

Sin argumentos posicionales lee `results/naive_O0.csv` (comportamiento compatible con versiones anteriores). Los defaults de cache estan calibrados para el Ryzen 5 4600H: `--l1-kb 32 --l2-kb 512 --l3-kb 4096`; ajusta los flags en otra CPU.

Ejemplos:

```bash
# solo los seis ordenes de bucles
python3 scripts/plot_results.py results/loop_order.csv \
    --out plots/loop_orders --title "Loop-order kernels"

# ordenes de bucles mas naive en la misma grafica
python3 scripts/plot_results.py results/naive_O0.csv results/loop_order.csv \
    --out plots/loop_vs_naive
```

---

## 6. Modulo `matmul_loop` (Fase 1.1 — cache-aware)

**Archivo:** [`src/matmul_loop.h`](../src/matmul_loop.h), [`src/matmul_loop.c`](../src/matmul_loop.c).

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

### 6.3 `matmul_loop_lookup`

```c
matmul_fn_t matmul_loop_lookup(const char *name);
```

Devuelve el puntero de funcion para el nombre dado (`"ijk"`, `"ikj"`, `"jik"`, `"jki"`, `"kij"`, `"kji"`), o `NULL` si el nombre no es reconocido.

### 6.4 `benchmark_iterations_loop`

```c
void benchmark_iterations_loop(scalar_t *B_out,
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
| `bin/bench_loop_O0` | `<order> <m> [num_iters] [num_runs]` | `kernel,m,n,num_iters,median_seconds,gflops` |
| `bin/validate_loop_O0` | `[m]` (default 256) | 4 tests por variante (3 invariantes + cross-val vs naive) |

`run_sweep_loop.sh` requiere un orden como argumento obligatorio para evitar que los kernels se midan en el mismo proceso (lo que contamina el estado de cache y el presupuesto termico entre ordenes):

```
scripts/run_sweep_loop.sh <order> ["<m list>"]
```

- `<order>`: uno de `ijk ikj jik jki kij kji` (obligatorio)
- `"<m list>"`: lista separada por espacios (opcional; default `256 384 512 768 1024 1536 2048 3072 4096`)
- Salida: `results/loop_<order>.csv`

Para correr los seis ordenes y obtener un CSV combinado usar `make sweep_loop_all`, que los encadena como procesos separados y concatena los resultados en `results/loop_order.csv`.

---

## 7. Roadmap de la API

A medida que se avancen las fases del proyecto se anadiran modulos manteniendo el mismo estilo:

| Fase | Nuevo modulo | Estado | Razon |
|------|--------------|--------|-------|
| 2 (reorden de bucles) | `matmul_reordered.c` | pendiente | Probar `ikj`, `kij`, etc. |
| 3 (transposicion + tiling) | `matmul_tiled.c` | pendiente | Pre-transposicion y blocking de L2 |
| 4 (vectorizacion) | igual `matmul_tiled.c` con flags | pendiente | Auto-vectorizacion |
| 5 (OpenMP) | `matmul_parallel.c` | pendiente | `#pragma omp parallel for` |
| Opcional / Fase 6 (Morton) | `matmul_recursive.c` + `morton.c` + `matmul_morton.c` | **COMPLETADA** (codigo y validacion; mediciones masivas en Sesion 03) | Recursion cache-oblivious sobre row-major y sobre layout Z-order |
| Fase 1.3 (tiled_avx2) | `matmul_tiled_avx2.c` | **COMPLETADA** (Sesion 03, integracion en `make results`) | 6-loop tiling (ii, kk, jj + i, p, j) con broadcast AVX2 + FMA; BS=64 configurable en runtime |

Cada nuevo kernel debera tener la firma `void mm(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n)` para que `validate.c` lo pueda probar sin cambios.

---

## 7. Modulo `matmul_recursive` (Fase 6, Etapa A2)

**Archivo:** [`src/matmul_recursive.h`](../src/matmul_recursive.h), [`src/matmul_recursive.c`](../src/matmul_recursive.c).

Kernel cache-oblivious que computa $C = A \cdot B$ recursivamente, dividiendo la dimension mas grande de $\{m, k, n\}$ en cada nivel hasta que el sub-problema cae bajo `RECURSION_THRESHOLD = 32 \cdot 32 \cdot 128 = 131072` flops elementales, donde un kernel base `ijk` cierra la recursion.

### 7.1 `matmul_recursive`

```c
void matmul_recursive(scalar_t *C,
                      const scalar_t *A,
                      const scalar_t *B,
                      size_t m, size_t k, size_t n);
```

**Computa** $C = A \cdot B$ por recursion cache-oblivious. Firma identica en estructura a `matmul_naive`: $C$ es $m \times n$ (out), $A$ es $m \times k$, $B$ es $k \times n$, todos row-major, $C$ no aliasa $A$ ni $B$.

**Algoritmo.** Wrapper publico que delega en `matmul_recursive_inner` con `ldc=n, lda=k, ldb=n`. La funcion interna evalua en orden:

1. Si $m \cdot k \cdot n \leq$ `RECURSION_THRESHOLD`, llamar al kernel base.
2. Si $m \geq k$ y $m \geq n$ y $m \geq 2$: dividir $m$ por la mitad. Las dos sub-llamadas trabajan sobre regiones disjuntas de $C$; ambas sobreescriben.
3. Si $n \geq m$ y $n \geq k$ y $n \geq 2$: dividir $n$. Igual, regiones disjuntas en $C$.
4. Si $k \geq 2$: dividir $k$. **No** son sub-llamadas independientes — la primera sobreescribe $C$, la segunda acumula sobre $C$ (variante `_inner_add` + `kernel_base_add`).
5. Fallback degenerado ($m = k = n = 1$): kernel base directo.

**Complejidad teorica.** $2 \cdot m \cdot k \cdot n$ flops, **identica** a la del baseline. La ganancia respecto a `matmul_naive` es de **localidad**, no de operaciones: el teorema de Hong y Kung [1981] muestra que cualquier algoritmo $\Theta(n^3)$ de multiplicacion de matrices realiza al menos $\Omega(n^3 / \sqrt{M})$ transferencias entre dos niveles de memoria de tamano $M$ palabras. El esquema recursivo cache-oblivious alcanza este limite asintotico **sin** conocer $M$: al dividir hasta que el sub-problema cabe en cualquier nivel de la jerarquia, automaticamente respeta el tradeoff transferencias-vs-trabajo a todas las escalas. Esto es lo que distingue al cache-oblivious del tiling explicito (que requiere conocer el tamano de cache para elegir el block size).

**Precondiciones y postcondiciones.** Identicas a `matmul_naive`.

**Notas.** El kernel base interno es un triple bucle `ijk` por consistencia conceptual con el baseline — optimizar el leaf no es objetivo de la Etapa A2.

### 7.2 `benchmark_iterations_recursive`

```c
void benchmark_iterations_recursive(scalar_t *B_out,
                                    const scalar_t *A,
                                    const scalar_t *Z,
                                    size_t m, size_t n,
                                    size_t num_iters);
```

Misma semantica que `benchmark_iterations` (Seccion 2.2) pero invocando `matmul_recursive` como kernel. Doble buffer + swap de punteros; aloja y libera los buffers internamente. Vive en `matmul_recursive.c` porque `matmul_naive.c` es inmutable.

---

## 8. Modulo `morton` (Fase 6, Etapa A3 - support)

**Archivo:** [`src/morton.h`](../src/morton.h), [`src/morton.c`](../src/morton.c).

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

**Archivo:** [`src/matmul_morton.h`](../src/matmul_morton.h), [`src/matmul_morton.c`](../src/matmul_morton.c).

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

**Complejidad.** $2 \cdot m \cdot k \cdot n$ flops (igual al baseline). La ganancia esperada respecto a `matmul_recursive` es localidad espacial adicional: en `matmul_recursive` row-major los sub-bloques al dividir por $m$ y $k$ producen strides cuando los sub-bloques son mas anchos que la linea de cache; en `matmul_morton` los cuatro cuadrantes son contiguos, eliminando ese mismatch.

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
| `bin/bench_recursive_O0`    | `bench_recursive.c`    | `<m> [num_iters] [num_runs]`           | linea CSV `recursive,m,n,num_iters,median_seconds,gflops` |
| `bin/validate_recursive_O0` | `validate_recursive.c` | `[m]` (default 256)                    | 7 tests: 3 invariantes + 4 cross-validation contra `matmul_naive` |
| `bin/bench_morton_O0`       | `bench_morton.c`       | `<m> [num_iters] [num_runs]`           | linea CSV `morton,m,n,num_iters,median_seconds,gflops`; aborta si $m$ no es potencia de 2 |
| `bin/validate_morton_O0`    | `validate_morton.c`    | `[m]` (default 256, potencia de 2)     | 11 tests: 3 invariantes + 4 cross contra `matmul_naive` + 4 cross contra `matmul_recursive` |
| `bin/test_morton`           | `test_morton.c`        | sin args                               | 4 grupos: tabla 4x4, round-trip encode/decode (4096 pares), contiguidad de cuadrantes para $m=8$, round-trip de reorganizacion para $m \in \{16, 64, 256\}$ |

Los binarios de bench reusan el patron del baseline: 1 warm-up + `num_runs` corridas medidas con mediana, `num_iters` default = $\min(2m/n, 4)$, semillas 42 ($A$) y 43 ($Z$). En `bench_morton_O0` la reorganizacion a Morton se ejecuta una sola vez **antes** del warm-up para que el tiempo cronometrado sea solo el del kernel.

### 10.2 Sweeps

| Script | Lista por defecto de $m$ | Salida CSV |
|--------|--------------------------|------------|
| `scripts/run_sweep_recursive.sh` | $\{256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192\}$ | `results/recursive_O0.csv` |
| `scripts/run_sweep_morton.sh`    | $\{1024, 2048, 4096, 8192\}$; filtra y omite no-potencias-de-2 con `Warning:` a stderr | `results/morton_O0.csv` |

Misma CLI que `run_sweep_naive.sh`: el listado de $m$ se puede pasar como primer argumento.

### 10.3 Comparacion entre kernels

| Script | Lee | Produce |
|--------|-----|---------|
| `scripts/plot_comparison.py` | `naive_O0.csv`, `recursive_O0.csv`, `morton_O0.csv` | `results/comparison_all.csv` (consolidado con columna `kernel` al inicio) + 4 PNG en `plots/`: `comparison_gflops_vs_m.png`, `comparison_time_vs_m.png`, `speedup_morton_vs_recursive.png`, `speedup_morton_vs_naive.png` |

CLI con defaults calibrados para Ryzen 5 4600H (`--l1-kb 32 --l2-kb 512 --l3-kb 4096`). Si `naive_O0.csv` o `recursive_O0.csv` falta, imprime mensaje claro con el comando para regenerarlo. Morton ausente produce solo un warning (no es bloqueante).

### 10.4 Profiling con perf

| Script | Eventos | Salida |
|--------|---------|--------|
| `scripts/profile_perf_compare.sh` | `L1-dcache-loads`, `L1-dcache-load-misses`, `LLC-loads`, `LLC-load-misses`, `dTLB-load-misses`, `cycles`, `instructions` | `results/perf_compare.csv` con columnas `m,variant,l1_loads,l1_misses,llc_loads,llc_misses,dtlb_misses,cycles,instructions` |
| `scripts/plot_perf_compare.py` | `perf_compare.csv` | 3 PNG (`perf_l1_misses.png`, `perf_llc_misses.png`, `perf_dtlb_misses.png`) + `plots/perf_summary_table.txt` con tasas L1 miss / LLC miss / dTLB-per-instruction |

Si `perf_event_paranoid` esta demasiado restrictivo, el script aborta con mensaje claro indicando el comando exacto para arreglarlo y la referencia a la seccion 3.3 del `README.md`.

### 10.5 Targets de Makefile

```
make bench_recursive            -> bin/bench_recursive_O0
make validate_recursive         -> bin/validate_recursive_O0
make test_morton                -> bin/test_morton
make bench_morton               -> bin/bench_morton_O0
make validate_morton            -> bin/validate_morton_O0
make sweep_recursive_run        -> bash scripts/run_sweep_recursive.sh
make sweep_morton_run           -> bash scripts/run_sweep_morton.sh
make plots_comparison           -> python3 scripts/plot_comparison.py
make sweep_full_santiago        -> los tres anteriores en cadena
make perf_compare               -> bash scripts/profile_perf_compare.sh
make plots_perf                 -> python3 scripts/plot_perf_compare.py
```

Targets de Fase 1.1 (loop-reorder):

```
make bench_loop                 -> bin/bench_loop_O0
make validate_loop              -> bin/validate_loop_O0
make sweep_loop_ijk             -> results/loop_ijk.csv  (proceso independiente)
make sweep_loop_ikj             -> results/loop_ikj.csv
make sweep_loop_jik             -> results/loop_jik.csv
make sweep_loop_jki             -> results/loop_jki.csv
make sweep_loop_kij             -> results/loop_kij.csv
make sweep_loop_kji             -> results/loop_kji.csv
make sweep_loop_all             -> los seis anteriores + results/loop_order.csv (combinado)
make plot_naive                 -> plots/naive_O0.png  (solo baseline)
make plot_loop                  -> plots/loop_orders.png  (6 ordenes desde loop_order.csv)
make plot_loop_vs_naive         -> plots/loop_vs_naive.png  (naive + 6 ordenes)
```

Targets de Fase 1.3 (tiled_avx2):

```
make bench_tiled_avx2           -> bin/bench_tiled_avx2_O3
make validate_tiled_avx2        -> bin/validate_tiled_avx2_O3
```

Ambos se compilan con `CFLAGS_O3_ZEN2` (`-O3 -march=znver2 -mavx2 -mfma`), que es obligatorio para que `_mm256_fmadd_ps` emita la instruccion FMA real. El `make results` incluye `bench_tiled_avx2_O3` como dependencia y `run_perf_zen2_sweep.sh` incluye `tiled_avx2` en su lista de variantes por defecto.

Todos extienden el Makefile **al final**, sin modificar las recetas del baseline (`bench_naive_O0`, `bench_naive_pg`, `validate_naive`, `sweep_naive`, `profile_*_naive`, `clean`, `distclean`).

---

## 11. Modulo `kernel_avx2` (Sesion 03, Etapa A4)

**Archivo:** [`src/kernel_avx2.h`](../src/kernel_avx2.h), [`src/kernel_avx2.c`](../src/kernel_avx2.c).

Microkernel AVX2 + FMA que acumula un tile fijo de $4 \times 16$ de $C$. Los $4 \times 16 = 64$ elementos del tile viven en $8$ registros YMM (4 filas $\times$ 2 vectores de 8 lanes FP32). Queda mitad del banco de YMM libre para los broadcasts de $A$ y los loads de $B$, condicion necesaria para mantener los dos pipes FMA del Zen 2 saturados sin spill.

### 11.1 `kernel_avx2_4x16`

```c
void kernel_avx2_4x16(scalar_t       *restrict C, size_t ldc,
                      const scalar_t *restrict A, size_t lda,
                      const scalar_t *restrict B, size_t ldb,
                      size_t kc);
```

**Computa** $C \mathrel{+}= A \cdot B$ sobre el tile fijo $4 \times 16$. La semantica es de **acumulacion**: el caller que necesite un $C$ limpio debe inicializarlo en cero antes de la invocacion.

**Parametros:**

- `C` *(in / out)*: matriz destino, $4$ filas $\times \geq 16$ columnas, row-major. Acumulado en sitio.
- `lda`, `ldb`, `ldc`: leading dimensions de $A$, $B$, $C$ en sus buffers originales (numero de columnas por fila).
- `A` *(in)*: panel de $4 \times kc$. Acceso interno solo a `A[r * lda + p]` con $r \in [0, 4)$ y $p \in [0, kc)$.
- `B` *(in)*: panel de $kc \times 16$.
- `kc`: longitud de la dimension contraida ($kc \geq 1$).

**Precondiciones:**

- `kc >= 1`, `lda >= kc`, `ldb >= 16`, `ldc >= 16`.
- `C`, `A`, `B` no aliasan (`restrict`).
- Alineacion a $32$ bytes es preferida pero **no obligatoria**: la implementacion usa `_mm256_loadu_ps` / `_mm256_storeu_ps`. El caller que pueda garantizar alineacion paga menos en el front-end.

**Postcondiciones:**

- $C[r, c] \mathrel{+}= \sum_{p=0}^{kc-1} A[r, p] \cdot B[p, c]$ para todo $(r, c)$ con $r \in [0, 4)$ y $c \in [0, 16)$. $A$ y $B$ no se modifican.

**Compilacion.** El objeto `build/obj/kernel_avx2.o` se compila aparte con flags Stage A4:

```
-O3 -march=znver2 -mavx2 -mfma -funroll-loops -ffast-math
```

`-Wpedantic` se omite porque los tipos `__m256` son extensiones GCC. `-ffast-math` autoriza reasociacion de la suma FP, lo que el microkernel necesita para emitir las cadenas FMA, pero a cambio acumula un poco mas de error de redondeo (relevante para las tolerancias de validacion del modulo $matmul\_morton\_avx2$).

**Performance esperada.** El techo single-core del Zen 2 es $2$ FMA $\times$ $8$ lanes $\times$ $2$ flops/op $\times$ $4.0$ GHz $= 128$ GFLOPS en FP32. En hojas cuyo working set cabe en L1d ($\leq 32$ KiB) el microkernel toca $\sim 8$–$10$ FMA-ops por ciclo (medido `fp_ret_sse_avx_ops.all`/cycle en Prompt 7), lo que se traduce en $\sim 80$ a $90$ GFLOPS sostenidos a la frecuencia turbo bajo carga AVX2. Es el techo computacional real para una sola hoja; el throughput de `matmul_morton_avx2` sobre la matriz completa es menor por el costo de materializacion de paneles y el trafico de $B$ desde L2/L3.

### 11.2 Constantes de geometria

```c
#define KERNEL_AVX2_MR 4    /* filas por tile */
#define KERNEL_AVX2_NR 16   /* columnas por tile */
```

Expuestas para que `matmul_morton_avx2` y el test de unidad calcen sus bloques al tile sin redeclarar las magic numbers.

---

## 12. Modulo `matmul_morton_avx2` (Sesion 03, Etapa A4 integracion)

**Archivo:** [`src/matmul_morton_avx2.h`](../src/matmul_morton_avx2.h), [`src/matmul_morton_avx2.c`](../src/matmul_morton_avx2.c).

Variante de `matmul_morton` cuyo leaf invoca el microkernel AVX2 de la Seccion 11. La recursion sigue la misma estructura cache-oblivious (Caso N independiente, Caso MK acoplado en cuatro cuadrantes) pero el layout de $A$ cambia.

### 12.1 Layout Morton-de-bloques (tile = 4)

Sesion 02 (Morton "fino"): cada elemento $A[i, j]$ ocupa la posicion `morton_encode(i, j)`. Sesion 03 (Morton "de bloques"): $A$ se particiona en sub-bloques $\text{MR} \times \text{MR}$ con $\text{MR} = 4$; los sub-bloques se Z-ordenan entre si, y los $16$ elementos de cada sub-bloque quedan en row-major. La posicion de $A[i, j]$ es:

$$
A\_idx = \text{morton\_encode}(i / \text{MR}, j / \text{MR}) \cdot \text{MR}^2 + (i \bmod \text{MR}) \cdot \text{MR} + (j \bmod \text{MR})
$$

La **propiedad de contiguidad** (Seccion 8.1) se preserva: cuatro cuadrantes de lado $h$ siguen ocupando offsets $\{0, 1, 2, 3\} \cdot h^2$ desde el padre. Solo cambia el significado del nivel hoja: bloque $4 \times 4$ de floats en lugar de un solo float. El layout fino y el de bloques **coexisten**; `matmul_morton.{c,h}` queda intacto.

### 12.2 `matmul_morton_avx2`

```c
void matmul_morton_avx2(scalar_t *C,
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
extern size_t g_recursion_threshold_avx2;
void matmul_morton_avx2_set_threshold(size_t threshold);
```

Variable global con default $64 \cdot 64 \cdot 128 = 524288$ flops elementales, equivalente a una hoja de $64 \times 64$ floats por panel de $A$ (working set $\sim 16$ KiB, mitad de L1d en el $4600$H). El setter acepta cualquier valor positivo; pasar $0$ imprime un warning y deja el default. La constante esta separada de `g_recursion_threshold` (Sesion 02) porque los regimenes son distintos: el microkernel AVX2 amortiza una hoja mucho mas grande que el `ijk + morton_encode` ingenuo, asi que el threshold optimo es mayor.

### 12.4 `reorganize_to_morton_blocks`

```c
void reorganize_to_morton_blocks(const scalar_t *A_row,
                                 scalar_t *A_morton,
                                 size_t m);
```

Reorganiza una matriz row-major $m \times m$ al layout Morton-de-bloques consumido por `matmul_morton_avx2`. Aborta si $m$ no es multiplo de $\text{MR}$ o si $m / \text{MR}$ no es potencia de $2$. El caller aloja `A_morton` con capacidad para $m^2$ elementos (tipicamente `xalloc_aligned`).

Complejidad: $O(m^2)$. Se ejecuta una sola vez antes de la recurrencia $B_{i+1} = A \cdot B_i$, igual que en Sesion 02.

### 12.5 Orquestadores

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

Misma semantica que sus contrapartes en `matmul_morton`. El primero reorganiza $A$ internamente (la conversion entra en el tiempo medido); el segundo recibe $A$ ya reorganizado y es el que usa `bench_morton_avx2_O3`.

---

## 13. Modulo `matmul_morton_omp` (Sesion 03, Etapa A5)

**Archivo:** [`src/matmul_morton_omp.h`](../src/matmul_morton_omp.h), [`src/matmul_morton_omp.c`](../src/matmul_morton_omp.c).

Variante paralela de `matmul_morton_avx2`. Reusa el microkernel AVX2 y el layout Morton-de-bloques; agrega `#pragma omp parallel single` en el wrapper publico y emite OpenMP tasks en cada subdivision recursiva por encima del threshold de paralelizacion. El scratch buffer del leaf pasa a ser un pool por-thread indexado por `omp_get_thread_num()` para que las hojas paralelas no compartan memoria intermedia.

### 13.1 `matmul_morton_omp`

```c
void matmul_morton_omp(scalar_t *C,
                       const scalar_t *A_morton,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n);
```

**Contrato de forma** identico a `matmul_morton_avx2`: $C$ es $m \times n$ (out), `A_morton` es $m \times k$ en Morton-de-bloques, $B$ es $k \times n$ row-major, mismas precondiciones ($m = k$ potencia de $2$, $m \geq \text{MR}$).

### 13.2 Thresholds (dos knobs independientes)

```c
extern size_t g_recursion_threshold_omp;
extern size_t g_parallel_threshold_omp;
void matmul_morton_omp_set_threshold         (size_t threshold);
void matmul_morton_omp_set_parallel_threshold(size_t threshold);
```

- `g_recursion_threshold_omp` (default $524288$): tamano del sub-problema en que la recursion cae al leaf kernel. Mismo rol que `g_recursion_threshold_avx2`.
- `g_parallel_threshold_omp` (default $524288$): tamano por debajo del cual la recursion deja de emitir `omp task` y corre inline. Con el default igual al leaf threshold, las tasks disparan en cada nivel sobre la hoja y nunca dentro de ella.

Las globals se mantienen separadas de las de `matmul_morton_avx2` para poder tunear la variante paralela sin alterar las mediciones del modulo serial.

### 13.3 Variables de entorno relevantes

| Variable | Efecto | Default usado en el bench |
|----------|--------|---------------------------|
| `OMP_NUM_THREADS` | Numero de threads. Si se omite, OpenMP usa todos los logicos (12 en el 4600H con SMT). | 6 (un thread por core fisico). |
| `OMP_PROC_BIND`   | `close` mantiene threads en el mismo CCX (3 cores + L3 4 MiB privada). `spread` los reparte entre los 2 CCXs. | Ver Seccion 13.4. |
| `OMP_PLACES`      | `cores` une cada thread a un core fisico (evita migracion entre core y SMT sibling). | `cores`. |

### 13.4 Recomendacion para el Ryzen 5 4600H

El chip Renoir tiene **2 CCX de 3 cores cada uno**, con L3 de $4$ MiB privada por CCX. Threads que cruzan CCX pierden la coherencia de L3 y pagan trafico por el Infinity Fabric. Esto define dos regimenes:

- **`OMP_NUM_THREADS=3 OMP_PROC_BIND=close`**: la opcion mas limpia para validaciones single-CCX y para diagnostico de scaling intra-cluster. Speedup cercano a lineal hasta $3$ threads; mas alla mete trafico cross-CCX y no escala.
- **`OMP_NUM_THREADS=6 OMP_PROC_BIND=close`**: usa los $6$ cores fisicos repartidos entre los dos CCXs respetando la afinidad de cada thread a su core. Es el **default recomendado** y el usado en el sweep de Prompt 6 (`scripts/run_omp_scaling.sh`). En `omp_scaling.csv` se observan picos cercanos a este modo a $m = 8192$.
- **`OMP_NUM_THREADS=6 OMP_PROC_BIND=spread`**: distribuye los threads para maximizar L3 compartido por thread, util cuando el working set por thread es grande. Comparable a `close` en GFLOPS sostenidos para $m \geq 4096$.

El SMT a $12$ threads aporta poco en este kernel: AVX2 + FMA ya satura los recursos de retirement; los hilos SMT extra se traducen en `cycles` mayores con la misma `fp_ops_per_cycle`.

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

## 14. Modulo `matmul_tiled_avx2` (Fase 1.3 — 6-loop tiling con AVX2+FMA)

**Archivo:** [`src/matmul_tiled_avx2.h`](../src/matmul_tiled_avx2.h), [`src/matmul_tiled_avx2.c`](../src/matmul_tiled_avx2.c).

Kernel de tiling explicito de seis bucles que extiende `matmul_tiled` (Fase 1.2, orden `ikj`, 2D) agregando un tercer nivel de bloque sobre la dimension `j`. El loop interno de `j` usa instrinsics AVX2 broadcast+FMA para procesar 8 elementos `float` por iteracion.

### 14.1 Constante y variable de block size

```c
#define TILED_AVX2_BS_DEFAULT 64u
extern size_t g_tiled_avx2_bs;
```

El bloque por defecto es `BS = 64`. Con tres paneles activos de `BS x BS` floats el working set es `3 x 64 x 64 x 4 B = 48 KiB`, que cabe en el L2 de 512 KB del Ryzen 5 4600H con margen para B y C. El block size debe ser multiplo de 8 (ancho de un vector AVX2 FP32); se rechaza cualquier valor que viole esta condicion.

### 14.2 `matmul_tiled_avx2_set_bs`

```c
void matmul_tiled_avx2_set_bs(size_t bs);
```

Cambia el block size global en runtime. Si `bs == 0` o `bs % 8 != 0`, imprime un aviso a `stderr` y retorna sin modificar `g_tiled_avx2_bs`. Util para barrer block sizes desde los scripts de tuning sin recompilar.

### 14.3 `matmul_tiled_avx2`

```c
void matmul_tiled_avx2(scalar_t *C,
                        const scalar_t *A,
                        const scalar_t *B,
                        size_t m, size_t k, size_t n);
```

**Computa** $C = A \cdot B$ con 6 bucles anidados: tres exteriores de tiling `(ii, kk, jj)` y tres interiores `(i, p, j)`.

**Parametros y precondiciones.** Identicos a `matmul_naive` (Seccion 2.1): `C` es $m \times n$ (out, sobrescrito), `A` es $m \times k$, `B` es $k \times n$, todos row-major, sin aliasing. No requiere que `m`, `k` ni `n` sean potencias de 2 ni multiplos de `BS`; el kernel maneja colas con un tail loop escalar.

**Postcondiciones.** $C_{ij} = \sum_{p=0}^{k-1} A_{ip} B_{pj}$ para todo $(i, j)$.

**Algoritmo vectorial.** Para cada bloque `(ii, kk, jj)` y cada par interior `(i, p)`:

```c
__m256 a_vec = _mm256_set1_ps(A[i*k + p]);          // broadcast del escalar a_ip
for (j = jj; j < j_vec_end; j += 8) {               // j_vec_end = mayor multiplo de 8 <= j_end
    __m256 b = _mm256_loadu_ps(&B[p*n + j]);
    __m256 c = _mm256_loadu_ps(&C[i*n + j]);
    c = _mm256_fmadd_ps(a_vec, b, c);
    _mm256_storeu_ps(&C[i*n + j], c);
}
for (j = j_vec_end; j < j_end; ++j)                 // tail escalar
    C[i*n + j] += A[i*k + p] * B[p*n + j];
```

`memset(C, 0, m*n*sizeof(scalar_t))` al inicio de la funcion garantiza que la acumulacion parcial por bloques sea correcta.

**Compilacion.** Requiere `-O3 -march=znver2 -mavx2 -mfma`. Sin `-mavx2 -mfma` el compilador rechaza `_mm256_fmadd_ps`.

**Complejidad.** $2 \cdot m \cdot k \cdot n$ flops (identica al baseline). Ganancia frente a `loop_ikj`: mejor reutilizacion de cache en los tres niveles gracias al tiling 3D, a costa de instrucciones adicionales de control de bloque.

### 14.4 `benchmark_iterations_tiled_avx2`

```c
void benchmark_iterations_tiled_avx2(scalar_t *B_out,
                                      const scalar_t *A,
                                      const scalar_t *Z,
                                      size_t m, size_t n,
                                      size_t num_iters);
```

Mismo patron que `benchmark_iterations` (Seccion 2.2): doble buffer + swap de punteros, aloja y libera internamente. Invoca `matmul_tiled_avx2` en cada iteracion.

### 14.5 Binarios

| Binario | CLI | Salida CSV |
|---------|-----|------------|
| `bin/bench_tiled_avx2_O3` | `<m> [num_iters] [num_runs] [bs]` | `tiled_avx2,m,n,num_iters,bs,median_seconds,gflops` (7 columnas) |
| `bin/validate_tiled_avx2_O3` | `[m] [bs]` (defaults: m=256, bs=64) | 4 tests: `A*0==0`, `I*Z==Z`, linealidad, cross contra naive |

El cuarto argumento opcional `[bs]` llama a `matmul_tiled_avx2_set_bs(bs)` antes de las corridas. La salida CSV de bench tiene **7 columnas** (una mas que los drivers de 6 columnas de `loop_*` y `tiled_ikj`) porque incluye `bs` entre `num_iters` y `median_seconds`; el consolidador `consolidate_perf_zen2.py` ya maneja este formato con la rama `len(parts) >= 7`.

**Tolerancias de validacion:** `ABS_TOL = 1e-4f`, `REL_TOL = 1e-3f` (igual que `validate_tiled`).

---

## 15. Cambios y versionado

Este documento se actualiza con cada PR que toque la API publica. La regla es: **si una firma de funcion cambia, este documento debe cambiar en el mismo commit**.
