# Plan de la Fase 6: Morton vs recursivo vs naive

**Fecha:** 2026-05-15
**Rama:** `claude/santiago-recursive-morton`
**Predecesor:** PR de rename baseline (mergeado en `main`, commit `82f9777`).
**Documento maestro:** `docs/PROMPTS_SESION_02.md` (vive fuera de este worktree, en la copia de trabajo de `main`).
**Estado:** plan aprobado; sin codigo todavia.

---

## 1. Objetivo cientifico

Comparar tres kernels de multiplicacion de matrices en funcion del tamano $m$ del problema, manteniendo fija la recurrencia $B_{i+1} = A \cdot B_i$ con $n = 128$ y $A \in \mathbb{R}^{m \times m}$:

1. **`matmul_naive`** (baseline existente, orden `ijk`, layout row-major).
2. **`matmul_recursive`** (divide-and-conquer cache-oblivious sobre row-major; Etapa A2 del documento tecnico).
3. **`matmul_morton`** (divide-and-conquer cache-oblivious con $A$ en layout Z-order / Morton; Etapa A3).

**Hipotesis.** La recursion cache-oblivious aprovecha automaticamente toda la jerarquia de memoria porque los sub-problemas eventualmente caben en L1, L2 y L3 sin elegir un tamano de bloque explicito. El layout Morton encima de la recursion garantiza que los cuatro cuadrantes de cualquier sub-bloque cuadrado de $A$ son cuatro segmentos consecutivos contiguos en memoria, mejorando todavia mas la localidad espacial cuando $A$ no cabe en cache. Se espera por lo tanto:

$$
\text{GFLOP/s}_{\text{morton}}(m) \gtrsim \text{GFLOP/s}_{\text{recursive}}(m) \gg \text{GFLOP/s}_{\text{naive}}(m)
$$

con la separacion entre las tres curvas creciendo a partir del primer cliff de cache. La hipotesis se valida o se refuta combinando dos analisis: GFLOP/s y tiempo/iteracion (Prompt 6) mas eventos de hardware (L1/LLC/dTLB misses) via `perf` (Prompt 7).

**Cuatro CSV de salida, mas un CSV consolidado al cierre.** Para mantener la rastreabilidad de cada variante por separado, cada kernel escribe su propio archivo de resultados; al cierre del Prompt 6 se genera un CSV unificado para alimentar las graficas comparativas.

---

## 2. Archivos a crear

Lista tomada literalmente de la Seccion 1 ("Archivos que tu creas") de `docs/PROMPTS_SESION_02.md`, sincronizada con los Prompts 1 a 8 del mismo documento.

### Codigo C (`src/`) -- 11 archivos

| Archivo | Origen | Proposito |
|---------|--------|-----------|
| `src/matmul_recursive.h` | Prompt 1 | Declaracion publica del kernel recursivo row-major + del orquestador `benchmark_iterations_recursive`. |
| `src/matmul_recursive.c` | Prompt 1+2 | Recursion cache-oblivious con `RECURSION_THRESHOLD = 32*32*128`; cuatro funciones internas (`matmul_recursive_inner`, `matmul_recursive_inner_add`, `kernel_base`, `kernel_base_add`). |
| `src/morton.h` | Prompt 3 | API publica de bit-interleaving y reorganizacion: `morton_encode`, `morton_decode`, `reorganize_to_morton`, `reorganize_from_morton`, `is_power_of_two`. |
| `src/morton.c` | Prompt 3 | Implementacion Nivel 1 portatil (bit twiddling con magic constants, sin BMI2 `pdep`/`pext`). |
| `src/matmul_morton.h` | Prompt 4 | Declaracion publica de `matmul_morton`, `benchmark_iterations_morton` y `benchmark_iterations_morton_preorganized`. |
| `src/matmul_morton.c` | Prompt 4 | Recursion con offsets Morton para $A$; casos N (dividir $n$) y MK (dividir $m$ y $k$ simultaneamente, los cuatro cuadrantes en offsets `0/1/2/3 * half*half`). |
| `src/test_morton.c` | Prompt 3 | Pruebas unitarias del modulo `morton`: tabla 4x4 conocida, round-trip encode/decode, contiguidad de cuadrantes, round-trip de reorganizacion. |
| `src/bench_recursive.c` | Prompt 2 | Driver `bench_recursive_O0` con la misma CLI y CSV que `bench_naive_O0`. |
| `src/validate_recursive.c` | Prompt 2 | Cuatro tests: tres invariantes algebraicos heredados + cross-validation contra `matmul_naive` en $m \in \{4, 16, 64, 256\}$. |
| `src/bench_morton.c` | Prompt 5 | Driver `bench_morton_O0`. Reorganizacion a Morton **fuera** del bucle medido (warm-up incluido); usa `benchmark_iterations_morton_preorganized` para que el tiempo refleje solo el kernel. |
| `src/validate_morton.c` | Prompt 5 | Cinco tests: los tres invariantes + cross-validation contra `matmul_naive` + cross-validation contra `matmul_recursive` (la prueba mas fuerte, ambos recursivos). |

### Scripts (`scripts/`) -- 5 archivos

| Archivo | Origen | Proposito |
|---------|--------|-----------|
| `scripts/run_sweep_recursive.sh` | Prompt 6 | Sweep de `bench_recursive_O0` en la lista completa de Fase 1; escribe `results/recursive_O0.csv`. |
| `scripts/run_sweep_morton.sh` | Prompt 6 | Sweep de `bench_morton_O0` restringido a potencias de 2; escribe `results/morton_O0.csv`. Si recibe un $m$ no potencia de 2, lo omite con mensaje a stderr y continua. |
| `scripts/plot_comparison.py` | Prompt 6 | Lee los tres CSV (`naive_O0.csv`, `recursive_O0.csv`, `morton_O0.csv`), genera el CSV consolidado `results/comparison_all.csv` y cuatro PNG. |
| `scripts/profile_perf_compare.sh` | Prompt 7 | Corre `perf stat -x,` para los tres binarios en $m \in \{1024, 2048, 4096, 8192\}$ con eventos L1/LLC/dTLB; escribe `results/perf_compare.csv`. |
| `scripts/plot_perf_compare.py` | Prompt 7 | Tres PNG de misses vs $m$ y `plots/perf_summary_table.txt` con las tasas de miss por variante. |

### Documentacion (`docs/`)

| Archivo | Origen | Proposito |
|---------|--------|-----------|
| `docs/PLAN_FASE6.md` | Prompt 0 | Este documento. |
| `docs/API.md` | Prompt 8 | **Se extiende** (no se reescribe) con secciones de los modulos nuevos antes de "Cambios y versionado"; se marca "Opcional (Morton)" como COMPLETADA en la tabla Roadmap. |
| `docs/SESION_02_RESUMEN.md` | Prompt 8 | Resumen analogo al de Sesion 01 (10 secciones + resumen ejecutivo de resultados). |
| `README.md` | Prompt 8 | Se actualiza la seccion de uso con los nuevos make targets. |

### Build (`Makefile`) -- solo extension

Targets nuevos al final del archivo, **sin tocar** los del baseline (`bench_naive_O0`, `bench_naive_pg`, `validate_naive_O0`, `sweep_naive`, `profile_*_naive`, `clean`, `distclean`):

```
bench_recursive       -> bin/bench_recursive_O0
validate_recursive    -> bin/validate_recursive_O0
test_morton           -> bin/test_morton
bench_morton          -> bin/bench_morton_O0
validate_morton       -> bin/validate_morton_O0
sweep_recursive_run   -> bash scripts/run_sweep_recursive.sh
sweep_morton_run      -> bash scripts/run_sweep_morton.sh
plots_comparison      -> python3 scripts/plot_comparison.py
sweep_full_santiago   -> sweep_recursive_run + sweep_morton_run + plots_comparison
perf_compare          -> bash scripts/profile_perf_compare.sh
plots_perf            -> python3 scripts/plot_perf_compare.py
```

Reglas de link explicitas (validate_morton enlaza tanto `matmul_recursive.c` como `matmul_naive.c` porque hace cross-validation contra ambos).

---

## 3. Archivos que NO se tocan

Por acuerdo de coordinacion con Juan Pablo (Camino B, Fase 2 en paralelo) y por la regla del proyecto de baseline inmutable:

**Codigo baseline (intacto):**

- `src/matmul_naive.h`, `src/matmul_naive.c`
- `src/bench_naive.c`, `src/validate_naive.c`

**Utilidades compartidas (intactas):**

- `src/matrix_utils.h`, `src/matrix_utils.c`
- `src/timing.h`

**Scripts del baseline (intactos):**

- `scripts/run_sweep_naive.sh`
- `scripts/profile_gprof_naive.sh`, `scripts/profile_perf_naive.sh`
- `scripts/plot_results.py`

**Resultados del baseline (intactos pero reusados como input):**

- `results/naive_O0.csv` se **lee** desde `plot_comparison.py` y `profile_perf_compare.sh`, pero nunca se sobreescribe.

**Archivos que solo se extienden:**

- `Makefile`: agregar targets al final, no modificar los existentes.
- `docs/API.md`: agregar secciones nuevas antes de "Cambios y versionado"; las secciones 1 a 5 escritas en Sesion 01 + rename quedan intactas.

---

## 4. Politica de salidas CSV (4 separados + 1 consolidado)

Conforme a la preferencia explicita confirmada al inicio de la sesion. Cada kernel y cada experimento escribe su propio CSV; el script de plotting hace la union al final.

| CSV | Generador | Columnas | Cuando |
|-----|-----------|----------|--------|
| `results/naive_O0.csv` | Sweep Fase 1 (existente) | `m,n,num_iters,median_seconds,gflops` | Ya existe; no se regenera. Si falta, `make sweep_naive` lo recrea. |
| `results/recursive_O0.csv` | `run_sweep_recursive.sh` | `m,n,num_iters,median_seconds,gflops` | Prompt 6. |
| `results/morton_O0.csv` | `run_sweep_morton.sh` | `m,n,num_iters,median_seconds,gflops` | Prompt 6, restringido a $m \in \{1024, 2048, 4096, 8192\}$. |
| `results/perf_compare.csv` | `profile_perf_compare.sh` | `m,variant,l1_loads,l1_misses,llc_loads,llc_misses,dtlb_misses,cycles,instructions` | Prompt 7. |
| **`results/comparison_all.csv`** | `plot_comparison.py` (cierre del Prompt 6) | `kernel,m,n,num_iters,median_seconds,gflops` | Concatena los tres CSV de bench agregando la columna `kernel` al principio. Es el archivo de referencia para el reporte final y para futuras comparaciones contra la Fase 2 de Juan Pablo. |

**Cuatro graficas comparativas** del Prompt 6, alimentadas por `comparison_all.csv`:

- `plots/comparison_gflops_vs_m.png` -- tres curvas (eje $x$ log, guias L1/L2/L3 a 32 KB / 512 KB / 4 MB).
- `plots/comparison_time_vs_m.png` -- tres curvas tiempo/iter vs $m$ en log-log, con curva teorica $O(m^2 n)$ anclada al menor $m$ de naive.
- `plots/speedup_morton_vs_recursive.png` -- cociente solo donde ambos existen (4 puntos).
- `plots/speedup_morton_vs_naive.png` -- analogo.

**Tres graficas adicionales** del Prompt 7 (eventos de hardware), alimentadas por `perf_compare.csv`:

- `plots/perf_l1_misses.png`, `plots/perf_llc_misses.png`, `plots/perf_dtlb_misses.png`.
- `plots/perf_summary_table.txt` (tabla de tasas de miss).

**Restriccion grafica.** En las graficas que mezclan naive/recursive (11 puntos) con morton (4 puntos), Morton se dibuja con linea discontinua + marcadores; las otras dos como linea continua. La leyenda lo explicita.

---

## 5. Convenciones reusadas

Todas las convenciones del proyecto se mantienen sin excepcion:

- **Tipo escalar.** `typedef float scalar_t` heredado de `src/matmul_naive.h`. Todas las cabeceras nuevas incluyen ese header para heredar el tipo.
- **Layout publico row-major** para $B$, $C$ y los buffers de la recurrencia $B_{i+1} = A \cdot B_i$. El layout Morton es **interno** a `matmul_morton.c`: $A$ se reorganiza a Morton una vez (por `reorganize_to_morton`) y se conserva en ese formato para todas las iteraciones; la reorganizacion no se cuenta en el tiempo medido por `bench_morton_O0`.
- **Tipos enteros.** `size_t` para todos los tamanos, indices y conteos. Manipulacion de bits Morton con `uint64_t` explicito (y `uint32_t` para los pares $(i, j)$ de entrada de `morton_encode`).
- **Alineacion 64 B** via `xalloc_aligned` de `matrix_utils`.
- **Comentarios y nombres en ingles**, sin emojis ni caracteres no ASCII.
- **Firmas canonicas** equivalentes a `matmul_naive` para los kernels:

  ```c
  void matmul_recursive(scalar_t *C,
                        const scalar_t *A,
                        const scalar_t *B,
                        size_t m, size_t k, size_t n);

  void matmul_morton(scalar_t *C,
                     const scalar_t *A_morton,   /* A already in Morton layout */
                     const scalar_t *B,
                     size_t m, size_t k, size_t n);
  ```

  Las dos funciones siguen el contrato de `matmul_naive`: $C$ es out, $A$ y $B$ son in, sin alias entre $C$ y los inputs. La diferencia semantica de `matmul_morton` (recibe $A$ ya reorganizado) se documenta en `docs/API.md` y se enforce con `assert` al inicio.
- **Tolerancias de validacion.** `abs_tol = 1e-5`, `rel_tol = 1e-4` para los invariantes algebraicos. Las cross-validations (Test 4 de `validate_recursive`, Tests 4-5 de `validate_morton`) usan las mismas tolerancias y se evaluan en $m \in \{4, 16, 64, 256\}$.
- **Seeds reproducibles.** 42 para $A$, 43 para $Z$.
- **CLI de los binarios.** Identica al baseline: `<m> [num_iters] [num_runs]` con `num_iters = min(2*m/n, MAX_MEAS_ITERS=4)` y `num_runs = 5` por defecto. 1 warm-up + 5 corridas medidas + mediana.
- **CSV individual.** Mismas cinco columnas que `naive_O0.csv`: `m,n,num_iters,median_seconds,gflops`. El CSV consolidado `comparison_all.csv` agrega `kernel` al principio.
- **Reloj.** `now_seconds()` (CLOCK_MONOTONIC) de `src/timing.h`.

---

## 6. Restriccion explicita sobre Morton

- **Pre-condicion del kernel.** `matmul_morton` solo se invoca con $m$ **potencia de 2** y $m = k$ ($A$ cuadrada). La razon: el indexing Z-order por bit-interleaving requiere subdivisiones exactas en mitades en cada nivel recursivo. Tamanos no potencia de 2 o no cuadrados rompen la propiedad de contiguidad de cuadrantes y por lo tanto la justificacion algoritmica del layout. El kernel hace `assert(m == k && is_power_of_two(m))` al inicio y aborta con `fprintf(stderr, ...) + exit(EXIT_FAILURE)` si la pre-condicion no se cumple.

- **Sweep restringido.** El sweep de Morton se evalua **solo** en

  $$
  m \in \{1024,\ 2048,\ 4096,\ 8192\}.
  $$

  Esos cuatro puntos cubren las transiciones L1 -> L2 -> L3 -> DRAM en el Ryzen 5 4600H (32 KB / 512 KB / 4 MB) y son suficientes para mostrar el comportamiento asintotico de Morton vs los otros dos kernels. `run_sweep_morton.sh` filtra automaticamente: si el usuario pasa un $m$ no potencia de 2, lo omite con mensaje a stderr y sigue.

- **Sweep de naive y recursive.** Lista completa de Fase 1:

  $$
  m \in \{256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192\}.
  $$

  En los cuatro puntos comunes con Morton ($1024, 2048, 4096, 8192$) se computan los speedups relativos (`speedup_morton_vs_recursive`, `speedup_morton_vs_naive`) y se reportan en tabla y grafica.

- **Reorganizacion fuera del tiempo medido.** El costo $O(m^2)$ de `reorganize_to_morton` se amortiza sobre las $I = 2m/n$ iteraciones de la recurrencia (cada una $O(m^2 n)$), por lo que en el regimen del proyecto la conversion es despreciable. Para hacerla explicitamente despreciable en la grafica, `bench_morton_O0` la ejecuta una sola vez **antes** del warm-up; el tiempo cronometrado es el de `benchmark_iterations_morton_preorganized`, que recibe $A$ ya en Morton.

---

## 7. Decisiones tecnicas tomadas en el documento maestro

Estas decisiones ya estan especificadas en `docs/PROMPTS_SESION_02.md` y no quedan abiertas:

| Decision | Valor | Justificacion |
|----------|-------|---------------|
| Threshold de la recursion | `RECURSION_THRESHOLD = 32 * 32 * 128 = 131072` flops elementales | Sub-bloque base con working set en L1. Misma constante para recursive y morton. |
| Kernel base | Tres bucles `ijk` (mismo orden que `matmul_naive`) | Consistencia conceptual; el objetivo en Fase 6 **no** es optimizar el kernel base. |
| Estrategia de division en recursive | Dividir la dimension mas grande de $\{m, k, n\}$ | Cache-oblivious clasico. |
| Estrategia de division en morton | Caso N independiente (dividir $n$); caso MK acoplado (dividir $m$ y $k$ simultaneamente, cuatro cuadrantes con offsets `0/1/2/3 * half*half`) | Garantiza que los sub-bloques de $A$ permanecen cuadrados y por lo tanto Morton-indexable. |
| Algoritmo de `morton_encode` | Nivel 1 portatil con magic constants y shifts (`spread_bits_32_to_64`) | Reproducible entre maquinas; sin dependencia de BMI2. |
| Strides explicitos | `ldc`, `lda`, `ldb` se propagan a las funciones internas | Permite trabajar con sub-bloques sin copiar. |

---

## 8. Decisiones aun por tomar al inicio de la implementacion

Pequenos detalles que no afectan el alcance ni la lista de archivos:

1. **Orden de implementacion dentro de cada modulo.** Ambos modulos siguen el patron wrapper publico -> funcion `_inner` con strides -> kernel base. Se escribe primero el kernel base (lo mas simple, mas facil de testear con `m` pequeno), luego el `_inner`, finalmente el wrapper.
2. **Manejo de errores en `assert`.** Decidir si compilar con `-DNDEBUG` o no para el sweep. Default: mantener los `assert` activos en `-O0` (consistente con el regimen de debugging del proyecto).
3. **Granularidad del swap de buffers** en `benchmark_iterations_recursive` y `benchmark_iterations_morton`: misma logica de doble buffer + swap del baseline.

---

## 9. Riesgos y mitigaciones

| Riesgo | Mitigacion |
|--------|------------|
| `-O0` enmascara el efecto de localidad porque el overhead de control de flujo domina. | Si el primer plot lo sugiere, replicar el sweep con `-O2 -fno-tree-vectorize` para aislar localidad sin auto-vectorizacion. Decision al ver `comparison_gflops_vs_m.png`. |
| `bench_morton_O0` con $m = 8192$ ocupa $\sim 256$ MiB en $A$ Morton mas buffers; cabe en el Ryzen 4600H (16 GiB RAM). | Si en otra maquina no cabe, truncar el sweep Morton a $\{1024, 2048, 4096\}$ y documentarlo. |
| Cuatro puntos de Morton dan curva visualmente pobre. | Aceptado por diseno: la conclusion descansa en la **separacion** entre curvas (graficas (c) y (d)), no en la densidad de puntos. |
| `perf` requiere ajuste de `perf_event_paranoid` en WSL2 y permisos de kernel; falla silenciosa con perfiles vacios. | `profile_perf_compare.sh` detecta `perf` fallido y emite mensaje claro referenciando la seccion 3.3 del `README.md`. |
| Acumulacion de error numerico en la cross-validation morton vs recursive por reordenamiento de sumas. | Las tolerancias `abs_tol=1e-5`, `rel_tol=1e-4` y la restriccion a $m \leq 256$ en los Tests 4-5 dejan margen amplio ($\sqrt{256} \cdot \epsilon_{\text{mach}} \approx 10^{-6}$). |
| Juan Pablo mergea Fase 2 mientras esta rama esta abierta. | Rebase frecuente sobre `origin/main` (Apendice C del documento maestro). Como solo se crean archivos nuevos y se extiende Makefile/API.md/README.md, los conflictos se resuelven aceptando ambos lados con bloques `# === Fase 2 ===` / `# === Fase 6 ===` en el Makefile. |

---

## 10. Mapa de prompts y entregables

Resumen de los Prompts 0 a 8 del documento maestro y lo que produce cada uno. Sirve de checklist al ejecutar.

| Prompt | Tema | Archivos producidos | Verificacion clave |
|--------|------|---------------------|---------------------|
| 0 | Bootstrap | `docs/PLAN_FASE6.md` (este archivo) | rama limpia, baseline valida |
| 1 | Kernel recursivo row-major | `src/matmul_recursive.{h,c}` | `gcc -fsyntax-only` sin warnings |
| 2 | Bench + validate recursive | `src/bench_recursive.c`, `src/validate_recursive.c`, Makefile | `./bin/validate_recursive_O0 256` -> OK; `./bin/bench_recursive_O0 1024 1 1` -> CSV |
| 3 | Modulo morton + test | `src/morton.{h,c}`, `src/test_morton.c` | `./bin/test_morton` -> MORTON TESTS OK |
| 4 | Kernel matmul_morton | `src/matmul_morton.{h,c}` | `gcc -fsyntax-only` sin warnings |
| 5 | Bench + validate morton | `src/bench_morton.c`, `src/validate_morton.c` | `./bin/validate_morton_O0 256` -> OK; rechazo claro en $m$ no potencia de 2 |
| 6 | Sweeps y graficas comparativas | `scripts/run_sweep_recursive.sh`, `scripts/run_sweep_morton.sh`, `scripts/plot_comparison.py`, **`results/recursive_O0.csv`**, **`results/morton_O0.csv`**, **`results/comparison_all.csv`**, 4 PNG | Tres CSV de bench coherentes; consolidado generado; lectura preliminar del speedup |
| 7 | perf comparativo | `scripts/profile_perf_compare.sh`, `scripts/plot_perf_compare.py`, **`results/perf_compare.csv`**, 3 PNG + tabla | Tasas de miss distinguibles entre variantes |
| 8 | Documentacion final + PR | `docs/API.md` extendido, `docs/SESION_02_RESUMEN.md`, `README.md` actualizado | PR creado y mergeable |

---

## 11. Entregables al cierre de la Fase 6

1. **Codigo:** 11 archivos C nuevos y 5 scripts nuevos, mergeados a `main` por PR.
2. **Makefile** con 11 targets nuevos sin modificar los existentes.
3. **Cuatro CSV** primarios (`naive_O0`, `recursive_O0`, `morton_O0`, `perf_compare`) y **uno consolidado** (`comparison_all.csv`).
4. **Siete PNG** (4 comparativos de Prompt 6 + 3 de perf de Prompt 7) y la tabla de tasas de miss.
5. **`docs/API.md`** extendido con tres secciones nuevas (matmul_recursive, morton, matmul_morton) y la tabla Roadmap actualizada.
6. **`docs/SESION_02_RESUMEN.md`** con 10 secciones + resumen ejecutivo de resultados experimentales.
7. **`README.md`** con los nuevos `make` targets en la seccion de uso.
8. **PR** "feat: Fase 6 - implementacion recursiva cache-oblivious con Morton layout" abierto en GitHub.

---

*Plan preparatorio de la Sesion 02. Ninguna linea de codigo se ha escrito todavia; la implementacion empieza al confirmar este plan e iniciar el Prompt 1.*
