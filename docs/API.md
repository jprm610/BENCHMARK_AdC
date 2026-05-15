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
m,n,num_iters,median_seconds,gflops
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

Lee `results/naive_O0.csv` y produce las graficas de paso 3 en `plots/`.

```
scripts/plot_results.py [--csv ...] [--out-dir ...]
                        [--l1-kb K] [--l2-kb K] [--l3-kb K]
                        [--cpu-label STRING]
```

Los defaults estan calibrados para la maquina de pruebas (AMD Ryzen 5 4600H): `--l1-kb 32 --l2-kb 512 --l3-kb 4096`. Ajusta los flags si corres en otra CPU.

---

## 6. Roadmap de la API

A medida que se avancen las fases del proyecto se anadiran modulos manteniendo el mismo estilo:

| Fase | Nuevo modulo | Razon |
|------|--------------|-------|
| 2 (reorden de bucles) | `matmul_reordered.c` | Probar `ikj`, `kij`, etc. |
| 3 (transposicion + tiling) | `matmul_tiled.c` | Pre-transposicion y blocking de L2 |
| 4 (vectorizacion) | igual `matmul_tiled.c` con flags | Auto-vectorizacion |
| 5 (OpenMP) | `matmul_parallel.c` | `#pragma omp parallel for` |
| Opcional (Morton) | `matmul_morton.c` | Layout Z-order recursivo |

Cada nuevo kernel debera tener la firma `void mm(scalar_t *C, const scalar_t *A, const scalar_t *B, size_t m, size_t k, size_t n)` para que `validate.c` lo pueda probar sin cambios.

---

## 7. Cambios y versionado

Este documento se actualiza con cada PR que toque la API publica. La regla es: **si una firma de funcion cambia, este documento debe cambiar en el mismo commit**.
