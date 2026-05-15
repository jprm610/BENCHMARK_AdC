# Prompts de la Sesion 02 - Implementacion recursiva + Morton

**Proyecto:** BENCHMARK_AdC - Camino A (cache-oblivious recursivo + Z-order/Morton layout)
**Responsable:** Santiago
**Contraparte en paralelo:** Juan Pablo (Camino B - Fase 2 en adelante)
**Fecha objetivo:** Sesion 02
**Base:** `main` posterior al merge de PR #1 (Fase 1 completa) + PR de rename (ver Seccion 2).

Este documento contiene los prompts en orden para alimentar a Claude Code durante la Sesion 02. Cada prompt esta diseniado para ser autosuficiente: incluye contexto, archivos a tocar, firmas, restricciones y criterio de aceptacion.

---

## 0. Contexto resumido

La Fase 1 produjo el baseline ingenuo (`matmul_naive`, orden `ijk`) con su infraestructura de medicion, validacion y profiling. La API esta documentada en `docs/API.md` y el resumen completo en `docs/SESION_01_RESUMEN.md`.

Tu trabajo en la Sesion 02 es la Etapa A2 + A3 del documento tecnico del proyecto, equivalente a la Fase 6 opcional del plan ejecutivo: implementar una multiplicacion recursiva cache-oblivious sobre row-major, y luego una segunda version con la matriz A reorganizada en layout Z-order (Morton). El objetivo cientifico es **comparar Morton vs row-major recursivo vs naive (y mas adelante vs las optimizaciones de Juan Pablo) en funcion de $m$**.

Convenciones del proyecto que se mantienen sin excepcion:

- Tipo escalar: `scalar_t` (`float`).
- Layout: row-major, plano, alineado a 64 bytes con `posix_memalign`.
- Indices y tamanos: `size_t`.
- Codigo y comentarios en ingles; mensajes a stdout y documentos en espanol.
- Sin emojis ni caracteres no ASCII decorativos.
- Estilo de comentarios y formato segun `docs/API.md`.

---

## 1. Estrategia de coordinacion con Juan Pablo

Juan Pablo va a trabajar en paralelo en su propia rama implementando la Fase 2 (reorden de bucles + pre-transposicion de A). Para evitar conflictos de merge, se aplican estas reglas:

**Archivos que tu NO modificas en la Sesion 02 (despues del rename):**

- `src/matmul_naive.c`, `src/matmul_naive.h`
- `src/bench_naive.c` (renombrado, ver Seccion 2)
- `src/validate_naive.c` (renombrado, ver Seccion 2)
- `scripts/run_sweep_naive.sh` (renombrado)
- `scripts/profile_gprof_naive.sh`, `scripts/profile_perf_naive.sh` (renombrados)
- `scripts/plot_results.py`

**Archivos que tu solo extiendes (anades cosas, no cambias lo existente):**

- `Makefile`: solo agregas targets nuevos al final, no tocas los existentes.
- `docs/API.md`: solo agregas las secciones de tus modulos nuevos, no tocas lo existente.

**Archivos que tu creas:**

- `src/matmul_recursive.h`, `src/matmul_recursive.c`
- `src/morton.h`, `src/morton.c`
- `src/matmul_morton.h`, `src/matmul_morton.c`
- `src/validate_recursive.c`, `src/validate_morton.c`
- `src/bench_recursive.c`, `src/bench_morton.c`
- `src/test_morton.c`
- `scripts/run_sweep_recursive.sh`, `scripts/run_sweep_morton.sh`
- `scripts/plot_comparison.py`
- `scripts/profile_perf_compare.sh`, `scripts/plot_perf_compare.py`
- `docs/PLAN_FASE6.md`, `docs/SESION_02_RESUMEN.md`

---

## 2. Pre-requisito: pasada de rename de la baseline

Antes de empezar la Sesion 02 propiamente, hay que hacer una pasada mecanica de renombrado para que la baseline de la Fase 1 lleve sufijo `_naive_` consistente con los modulos que se van a anadir (`_recursive_`, `_morton_`). Esto es **un cambio puramente nominal, sin alteracion funcional**.

**Por que se hace antes que todo lo demas:**

- Si Santiago y Juan Pablo cada uno introducen modulos nuevos sin renombrar el baseline, el repo termina con `bench_O0` (naive), `bench_reordered_O0` (Juan Pablo), `bench_recursive_O0`, `bench_morton_O0` (Santiago) — todos consistentes excepto el primero. Inconsistente y confuso.
- Si se hace al final, hay que esperar a que ambas ramas mergeen y luego hacer la pasada, generando conflictos masivos.
- Si se hace al principio como PR independiente a `main`, ambos brancheas desde el main ya renombrado y nadie paga el costo individualmente.

**Tabla de renombrados:**

| De | A |
|----|---|
| `src/benchmark.c` | `src/bench_naive.c` |
| `src/validate.c` | `src/validate_naive.c` |
| `bin/bench_O0` | `bin/bench_naive_O0` |
| `bin/bench_pg` | `bin/bench_naive_pg` |
| `bin/validate_O0` | `bin/validate_naive_O0` |
| `scripts/run_sweep.sh` | `scripts/run_sweep_naive.sh` |
| `scripts/profile_gprof.sh` | `scripts/profile_gprof_naive.sh` |
| `scripts/profile_perf.sh` | `scripts/profile_perf_naive.sh` |
| `results/baseline_O0.csv` | `results/naive_O0.csv` |
| `results/gprof_m<m>.txt` | `results/gprof_naive_m<m>.txt` |
| `results/perf_m<m>.txt` | `results/perf_naive_m<m>.txt` |
| `plots/baseline_gflops_vs_m.png` | `plots/naive_gflops_vs_m.png` |
| `plots/baseline_time_vs_m.png` | `plots/naive_time_vs_m.png` |

**Lo que NO se renombra** (porque ya esta correctamente namespaceado o porque es generico):

- `src/matmul_naive.{c,h}` — ya lleva `naive` en el nombre.
- `src/matrix_utils.{c,h}` — utilidad compartida, sin variante.
- `src/timing.h` — utilidad compartida.
- Makefile target `sweep` y `clean` — son verbos genericos; pueden quedarse o pasar a `sweep_naive` segun decision (ver el prompt de rename).

Despues del rename, el repositorio tiene una nomenclatura paralela limpia:

```
bench_naive_O0      validate_naive_O0      run_sweep_naive.sh      naive_O0.csv
bench_recursive_O0  validate_recursive_O0  run_sweep_recursive.sh  recursive_O0.csv
bench_morton_O0     validate_morton_O0     run_sweep_morton.sh     morton_O0.csv
```

---

## 3. Roadmap de los prompts

| # | Prompt | Salida principal |
|---|--------|------------------|
| PRE | Rename pass de la baseline | PR a `main`, mergeado antes de seguir |
| 0 | Bootstrap | Rama nueva, `docs/PLAN_FASE6.md` |
| 1 | Kernel recursivo sobre row-major | `matmul_recursive` + compila limpio |
| 2 | Bench y validate del recursivo | `bin/bench_recursive_O0`, `bin/validate_recursive_O0` |
| 3 | Modulo Morton: encoding + reorganizacion | `morton.h/c`, `bin/test_morton` |
| 4 | Kernel recursivo con A en Morton | `matmul_morton` |
| 5 | Bench y validate del Morton | `bin/bench_morton_O0`, `bin/validate_morton_O0` |
| 6 | Sweep comparativo y graficas | `plots/comparison_*.png` |
| 7 | Analisis con perf (eventos de hardware) | Tabla de misses por variante |
| 8 | Documentacion final | `API.md` actualizada + `SESION_02_RESUMEN.md` |

Tiempo estimado: PRE toma 15-20 minutos (mecanico). Los Prompts 0 a 8 entre 20 y 45 minutos cada uno mas tu revision intercalada. Sesion total: medio dia a un dia completo.

---

## 4. Prompts en orden

A continuacion los prompts. Cada bloque entre lineas dobles es lo que tu copias y pegas a Claude Code.

---

### Prompt PRE - Pasada de rename de la baseline

```text
Antes de iniciar la Sesion 02 propiamente, vamos a hacer una pasada de renombrado
mecanico sobre los archivos baseline para que el repositorio tenga nomenclatura
consistente cuando se anadan los modulos recursive y morton. Este cambio NO altera
ninguna logica funcional: solo renombra archivos, targets, binarios y referencias.

IMPORTANTE: este prompt se ejecuta en una rama propia que se mergea a main ANTES
de abrir la rama de trabajo de la Sesion 02. Asi Juan Pablo tambien puede branchear
desde el main renombrado y evitamos conflictos dobles.

PASOS PRELIMINARES:

1. git checkout main && git pull
2. git checkout -b claude/rename-naive-baseline

RENOMBRADOS DE ARCHIVOS (usar `git mv` para preservar historial):

- src/benchmark.c                -> src/bench_naive.c
- src/validate.c                 -> src/validate_naive.c
- scripts/run_sweep.sh           -> scripts/run_sweep_naive.sh
- scripts/profile_gprof.sh       -> scripts/profile_gprof_naive.sh
- scripts/profile_perf.sh        -> scripts/profile_perf_naive.sh

ACTUALIZACIONES EN EL MAKEFILE:

- Target `bench_O0` -> `bench_naive_O0`. El binario `bin/bench_O0` -> `bin/bench_naive_O0`.
- Target `bench_pg` -> `bench_naive_pg`. El binario `bin/bench_pg` -> `bin/bench_naive_pg`.
- Target `validate_O0` -> `validate_naive_O0`. El binario `bin/validate_O0` -> `bin/validate_naive_O0`.
- Target `sweep` -> `sweep_naive` (mas claro cuando convivan tres sweeps).
- Las recetas deben referenciar src/bench_naive.c y src/validate_naive.c.
- El target `all` o `default` (si existe) actualiza sus dependencias.
- `clean` y `distclean` se quedan con el mismo nombre pero deben borrar los
  binarios renombrados (bench_naive_O0, etc.).

ACTUALIZACIONES EN LOS SCRIPTS RENOMBRADOS:

- run_sweep_naive.sh:
  - `bin/bench_O0` -> `bin/bench_naive_O0`
  - `bin/bench_pg` -> `bin/bench_naive_pg`
  - Salida CSV: `results/baseline_O0.csv` -> `results/naive_O0.csv`
  - Llamadas a profile_*.sh: `scripts/profile_gprof.sh` -> `scripts/profile_gprof_naive.sh`,
    idem para perf.
  - Reportes: `results/gprof_m<m>.txt` -> `results/gprof_naive_m<m>.txt`,
    `results/perf_m<m>.txt` -> `results/perf_naive_m<m>.txt`.

- profile_gprof_naive.sh: `bench_pg` -> `bench_naive_pg`, reporte
  `gprof_m<m>.txt` -> `gprof_naive_m<m>.txt`.

- profile_perf_naive.sh: `bench_O0` -> `bench_naive_O0`, reporte
  `perf_m<m>.txt` -> `perf_naive_m<m>.txt`.

ACTUALIZACIONES EN scripts/plot_results.py:

- Default del CSV de entrada: `results/baseline_O0.csv` -> `results/naive_O0.csv`.
- Default de las imagenes de salida:
    `plots/baseline_gflops_vs_m.png` -> `plots/naive_gflops_vs_m.png`
    `plots/baseline_time_vs_m.png`   -> `plots/naive_time_vs_m.png`
- Cualquier label, titulo o leyenda interna que diga "baseline" puede quedarse
  asi (es una descripcion conceptual, no un nombre de archivo) O ajustarse a
  "naive" si quieres consistencia visual. Recomiendo dejarlo como "baseline" en
  titulos y "naive" en nombres de archivo, porque "baseline" sigue siendo el rol
  conceptual.

ACTUALIZACIONES EN docs/API.md:

- Seccion 5.1: titulo y todo el cuerpo cambian de `bin/bench_O0` a `bin/bench_naive_O0`.
- Seccion 5.2: `bin/bench_pg` -> `bin/bench_naive_pg`.
- Seccion 5.3: `bin/validate_O0` -> `bin/validate_naive_O0`.
- Seccion 5.4: actualizar los paths de scripts y CSV.
- Seccion 5.5: actualizar nombres de los scripts standalone.
- Seccion 5.6: actualizar los defaults de plot_results.py.
- En cualquier ejemplo de comando que aparezca, actualizar los nombres.
- En la tabla del Roadmap (Seccion 6), no hace falta cambiar nada porque las
  filas siguen siendo modulos.

ACTUALIZACIONES EN README.md:

- Toda la seccion de uso debe reflejar los nuevos nombres. Hacer una pasada con
  grep y reemplazar uno a uno.

ACTUALIZACIONES EN docs/SESION_01_RESUMEN.md:

- Actualizar las referencias en las tablas y comandos. Las menciones a
  `bin/bench_O0`, `bin/bench_pg`, `bin/validate_O0`, `baseline_O0.csv`,
  `baseline_*.png`, `run_sweep.sh`, `profile_*.sh`, `benchmark.c`, `validate.c`
  deben quedar con los nombres nuevos.
- Anadir una nota al final indicando: "Los archivos baseline se renombraron en el
  PR de rename (claude/rename-naive-baseline) antes de iniciar la Sesion 02, sin
  cambios funcionales. Las referencias en este documento estan actualizadas."

ACTUALIZACIONES EN .gitignore:

- Si las entradas usan los nombres viejos (`bench_O0`, `baseline_*.csv`, etc.),
  actualizarlas al nuevo esquema.

PRESERVACION DE DATOS LOCALES:

Si existen en el filesystem (puede que si, de la ejecucion de la Sesion 01):

    results/baseline_O0.csv -> results/naive_O0.csv
    results/gprof_m*.txt    -> results/gprof_naive_m*.txt
    results/perf_m*.txt     -> results/perf_naive_m*.txt
    plots/baseline_*.png    -> plots/naive_*.png

Como `results/` y `plots/` estan gitignored, esto no afecta el repositorio pero
preserva las mediciones existentes. Hazlo con `mv` simple en el filesystem.

VERIFICACION COMPLETA POST-RENAME:

Despues del rename, ejecuta:

    make clean
    make
    ./bin/validate_naive_O0 256

Debe imprimir VALIDATION OK.

    ./bin/bench_naive_O0 1024 1 1

Debe imprimir una linea CSV bien formada.

    make bench_naive_pg
    bash scripts/profile_perf_naive.sh 1024 1 1
    bash scripts/profile_gprof_naive.sh 1024 1 1

Ambos deben generar sus reportes en results/ con los nuevos nombres.

    make sweep_naive

Debe ejecutar el sweep completo y producir results/naive_O0.csv con varias lineas.

    source ~/venvs/matmul/bin/activate  # si aplica
    python3 scripts/plot_results.py

Debe generar plots/naive_gflops_vs_m.png y plots/naive_time_vs_m.png.

GREP DE SANIDAD:

Antes de commitear, corre este grep para asegurar que no queda ninguna referencia
colgada a los nombres viejos:

    grep -rn --include="*.c" --include="*.h" --include="*.md" --include="*.sh" \
         --include="*.py" --include="Makefile" \
         -e "bench_O0" \
         -e "bench_pg" \
         -e "validate_O0" \
         -e "baseline_O0.csv" \
         -e "run_sweep.sh" \
         -e "profile_gprof.sh" \
         -e "profile_perf.sh" \
         -e "src/benchmark.c" \
         -e "src/validate.c"

El resultado debe ser vacio. Si aparece algo, corregirlo antes de commitear.

COMMIT:

Un solo commit con mensaje:

    refactor: rename baseline files and binaries with _naive_ suffix

    Mechanical rename to prepare for the introduction of _recursive_ and _morton_
    modules. No functional changes.

    Files renamed:
    - src/benchmark.c   -> src/bench_naive.c
    - src/validate.c    -> src/validate_naive.c
    - scripts/run_sweep.sh, profile_gprof.sh, profile_perf.sh -> *_naive.sh
    - Binaries: bench_O0, bench_pg, validate_O0 -> bench_naive_O0, bench_naive_pg,
      validate_naive_O0
    - CSVs and reports under results/ prefixed with naive_
    - Plot defaults aligned

    Verified: validate_naive_O0 256 -> OK; sweep_naive run successfully.

PR:

Titulo: "refactor: rename baseline to naive for consistency with upcoming modules"

Cuerpo:
- Motivacion (paralelismo nominal con los modulos recursive y morton que vienen).
- Lista de renombrados.
- Verificaciones realizadas.
- Nota de que esto debe mergearse ANTES de que Santiago abra la rama Fase 6 y
  ANTES de que Juan Pablo abra la rama Fase 2.

Si gh CLI esta disponible, abre el PR con `gh pr create`. Si no, imprime el URL
de comparacion.

CRITERIO DE ACEPTACION:

1. Grep de sanidad retorna vacio.
2. make clean && make compila sin warnings.
3. ./bin/validate_naive_O0 256 imprime VALIDATION OK.
4. ./bin/bench_naive_O0 1024 1 1 produce CSV bien formado.
5. make sweep_naive ejecuta sin errores.
6. python3 scripts/plot_results.py produce plots/naive_*.png.
7. PR creado o URL impreso.

UNA VEZ MERGEADO: avisame para iniciar el Prompt 0 desde el main renombrado.
```

---

### Prompt 0 - Bootstrap de la Sesion 02

```text
El PR de rename ya esta mergeado en main. Inicio formal de la Sesion 02.

Antes de empezar:

1. Lee completos `docs/API.md` y `docs/SESION_01_RESUMEN.md` para confirmar que las
   referencias estan actualizadas a los nombres _naive_ y para tener el contexto
   completo de las firmas y convenciones.
2. Asegurate de estar en main actualizado:
       git fetch && git checkout main && git pull
3. Crea una rama nueva `claude/santiago-recursive-morton` desde main.
4. Verifica que el baseline renombrado compila y valida limpio:
       make clean && make && ./bin/validate_naive_O0 256
   Debe imprimir VALIDATION OK.
5. Crea `docs/PLAN_FASE6.md` con un resumen ejecutivo de una pagina del plan que
   viene:
   - Objetivo cientifico (comparar Morton vs recursive vs naive en funcion de m).
   - Lista de archivos que se van a crear (los enumerados en la Seccion 1 de
     PROMPTS_SESION_02.md).
   - Lista de archivos que NO se tocan (los _naive y los scripts del baseline).
   - Convenciones reusadas (scalar_t, row-major, size_t, alineacion 64B,
     comentarios en ingles).
   - Restriccion explicita: Morton solo corre en m potencia de 2; el sweep Morton
     se limita a {1024, 2048, 4096, 8192}.

No implementes nada de codigo todavia. Solo crea la rama y el documento de plan.

Confirma cuando este todo listo y muestra el contenido de docs/PLAN_FASE6.md.
```

---

### Prompt 1 - Kernel recursivo sobre row-major (Etapa A2)

```text
Implementa la multiplicacion recursiva cache-oblivious sobre matrices en row-major
(sin Morton todavia). Es la Etapa A2 del documento tecnico. La motivacion es: dividir
el problema por la mitad en su dimension mas grande hasta que el sub-problema quepa
en algun nivel de cache. No se elige tamano de bloque; la jerarquia se aprovecha
automaticamente.

ARCHIVOS A CREAR:

- src/matmul_recursive.h: declaracion publica.
- src/matmul_recursive.c: implementacion.

FIRMA PUBLICA (debe ser identica en estructura a matmul_naive):

    void matmul_recursive(scalar_t *C,
                          const scalar_t *A,
                          const scalar_t *B,
                          size_t m, size_t k, size_t n);

Semantica: computa C = A * B con A de tamano m x k, B de k x n, C de m x n, todos en
row-major. Sobreescribe C. Misma precondicion de no-alias que matmul_naive.

ALGORITMO:

La funcion publica matmul_recursive es un wrapper que delega en una funcion interna
con strides explicitos:

    static void matmul_recursive_inner(scalar_t *C, const scalar_t *A, const scalar_t *B,
                                       size_t m, size_t k, size_t n,
                                       size_t ldc, size_t lda, size_t ldb);

donde ldc, lda, ldb son las leading dimensions de las matrices ORIGINALES (no de los
sub-bloques). El wrapper publico llama con ldc=n, lda=k, ldb=n.

Logica de la recursion en matmul_recursive_inner:

1. Si m * k * n <= THRESHOLD (con THRESHOLD = 32 * 32 * 128 = 131072 expuesto como
   constante de archivo nombrada RECURSION_THRESHOLD), llamar al kernel base.

2. Caso 1: si m >= k y m >= n y m >= 2, dividir m por 2.
       size_t m_half = m / 2;
       matmul_recursive_inner(C,                        A,                        B,
                              m_half, k, n, ldc, lda, ldb);
       matmul_recursive_inner(C + m_half * ldc,         A + m_half * lda,         B,
                              m - m_half, k, n, ldc, lda, ldb);

3. Caso 3: si n >= m y n >= k y n >= 2, dividir n por 2.
       size_t n_half = n / 2;
       matmul_recursive_inner(C,                        A, B,
                              m, k, n_half, ldc, lda, ldb);
       matmul_recursive_inner(C + n_half,               A, B + n_half,
                              m, k, n - n_half, ldc, lda, ldb);

4. Caso 2: k es la mas grande, dividir k por 2. NO son independientes: la segunda suma
   sobre C.
       size_t k_half = k / 2;
       matmul_recursive_inner(C, A,                        B,
                              m, k_half, n, ldc, lda, ldb);
       matmul_recursive_inner_add(C, A + k_half,           B + k_half * ldb,
                                  m, k - k_half, n, ldc, lda, ldb);

5. Si despues de aplicar las reglas no se entra en ningun caso (por ejemplo m=k=n=1),
   llamar al kernel base con esos tamanos.

KERNEL BASE:

    static void kernel_base(scalar_t *C, const scalar_t *A, const scalar_t *B,
                            size_t m, size_t k, size_t n,
                            size_t ldc, size_t lda, size_t ldb);

Tres bucles anidados en orden ijk (igual que matmul_naive para consistencia
conceptual; el objetivo en esta fase NO es optimizar el kernel base). C[i*ldc + j]
se SOBRESCRIBE.

KERNEL BASE ADD:

    static void kernel_base_add(scalar_t *C, const scalar_t *A, const scalar_t *B,
                                size_t m, size_t k, size_t n,
                                size_t ldc, size_t lda, size_t ldb);

Igual que kernel_base, pero ACUMULA: C[i*ldc + j] += suma. Se usa cuando la
recursion divide por k.

NOTA IMPORTANTE: cuando se divide por k, la primera llamada de recursion debe
escribir C desde cero (matmul_recursive_inner, kernel sobreescribe), y la segunda
debe sumarle encima (matmul_recursive_inner_add, kernel acumula). Cuando se divide
por m o n, las regiones de C son disjuntas y ambas llamadas pueden sobreescribir
sus sub-bloques.

Implementa tambien matmul_recursive_inner_add con la misma logica pero llamando a
kernel_base_add en las hojas y propagando _add a la segunda llamada del caso k.

ARCHIVOS QUE NO TOCAS:

- src/matmul_naive.c, src/matmul_naive.h
- src/bench_naive.c, src/validate_naive.c
- Makefile (en este prompt todavia no; eso es el Prompt 2)

VERIFICACION INTERNA:

Compila con:

    gcc -O0 -g -Wall -Wextra -Wpedantic -fsyntax-only src/matmul_recursive.c

Debe pasar sin warnings.

CRITERIO DE ACEPTACION:

- src/matmul_recursive.h y src/matmul_recursive.c existen.
- Compilan sin warnings.
- El header expone exactamente la firma indicada arriba.
- Los simbolos internos (matmul_recursive_inner, kernel_base, etc.) son `static`.
- RECURSION_THRESHOLD esta definido como una constante.
- No se modifico ningun archivo de Juan Pablo ni de la baseline _naive.

Cuando termines, muestra el contenido completo de src/matmul_recursive.h y un
resumen de la estructura de src/matmul_recursive.c (cuantas lineas, que funciones,
sin pegarme todo el cuerpo). Voy a revisar antes del Prompt 2.
```

---

### Prompt 2 - Validacion y binario para el recursivo

```text
Ahora vamos a hacer compilable y validable el modulo matmul_recursive. Esto implica
anadir targets al Makefile, escribir un programa de validacion cruzada contra
matmul_naive, y escribir un binario de benchmark con la misma CLI que bench_naive_O0.

ARCHIVOS A CREAR:

1. src/validate_recursive.c: programa standalone que ejercita los tres invariantes
   algebraicos del proyecto (A*0=0, I*Z=Z, linealidad) sobre matmul_recursive, MAS
   una cuarta verificacion crucial: cross-validation contra matmul_naive sobre
   datos aleatorios.

   - CLI: `./validate_recursive_O0 [m]`, default m=256.
   - Test 1: A * 0 = 0 (init_matrix_zero para Z).
   - Test 2: I * Z = Z (init_matrix_identity para A).
   - Test 3: A*(Z1+Z2) = A*Z1 + A*Z2 (linealidad).
   - Test 4: matmul_recursive(C_rec, A, B, m, k, n) y matmul_naive(C_naive, ...) con
     datos aleatorios deben dar matrices iguales dentro de tolerancia. Probar para
     m chicos: m=4, 16, 64, 256, con k=m y n=128.
   - Si todos pasan: imprime VALIDATION OK y retorna 0.

   Tolerancias: abs_tol = 1e-5, rel_tol = 1e-4.

2. src/bench_recursive.c: driver de benchmark con la misma CLI y formato CSV que
   bench_naive_O0 pero usando matmul_recursive como kernel.

   - CLI: `./bench_recursive_O0 <m> [num_iters] [num_runs]`.
   - num_iters default: min(2*m/n, MAX_MEAS_ITERS) con MAX_MEAS_ITERS=4.
   - num_runs default: 5.
   - 1 warm-up + num_runs corridas medidas, reporta mediana.
   - Internamente usa una funcion auxiliar `benchmark_iterations_recursive` que
     hace lo mismo que `benchmark_iterations` del modulo naive pero invocando
     matmul_recursive. Como no podemos modificar matmul_naive.c, esta funcion vive
     en matmul_recursive.c. Anadir su declaracion a matmul_recursive.h:

         void benchmark_iterations_recursive(scalar_t *B_out,
                                             const scalar_t *A,
                                             const scalar_t *Z,
                                             size_t m, size_t n,
                                             size_t num_iters);

     Misma semantica que benchmark_iterations (doble buffer + swap).

   - Salida CSV: `m,n,num_iters,median_seconds,gflops`.
   - gflops = 2 * m * m * n * num_iters / median_seconds / 1e9.

ARCHIVOS A MODIFICAR (extension minima):

- src/matmul_recursive.h: agregar declaracion de benchmark_iterations_recursive.
- src/matmul_recursive.c: implementacion de benchmark_iterations_recursive.
- Makefile: agregar targets al final, despues de los existentes. NO tocar los
  targets bench_naive_O0, bench_naive_pg, validate_naive_O0, sweep_naive.

  Nuevos targets:

      bench_recursive: bin/bench_recursive_O0
      validate_recursive: bin/validate_recursive_O0

      bin/bench_recursive_O0: src/bench_recursive.c src/matmul_recursive.c \
                              src/matrix_utils.c src/matmul_naive.c
          $(CC) $(CFLAGS_O0) -o $@ $^

      bin/validate_recursive_O0: src/validate_recursive.c src/matmul_recursive.c \
                                 src/matrix_utils.c src/matmul_naive.c
          $(CC) $(CFLAGS_O0) -o $@ $^

  validate_recursive_O0 enlaza tambien matmul_naive.c porque la cross-validation
  lo necesita.

ARCHIVOS QUE NO SE TOCAN:

- src/matmul_naive.c, src/matmul_naive.h, src/bench_naive.c, src/validate_naive.c.
- Los targets existentes del Makefile.

CRITERIO DE ACEPTACION:

1. `make validate_recursive` compila sin warnings.
2. `./bin/validate_recursive_O0 256` imprime VALIDATION OK.
3. `make bench_recursive` compila sin warnings.
4. `./bin/bench_recursive_O0 1024 1 1` produce CSV bien formado con gflops > 0.
5. Regresion del baseline: `./bin/validate_naive_O0 256` sigue dando VALIDATION OK.

Cuando termines, ejecuta los cinco pasos del criterio y muestrame los outputs.
```

---

### Prompt 3 - Modulo Morton: encoding + reorganizacion

```text
Implementa el modulo de codificacion Morton (Z-order) y las funciones de
reorganizacion de la matriz A entre row-major y Morton.

ARCHIVOS A CREAR:

1. src/morton.h: declaraciones publicas.
2. src/morton.c: implementaciones.
3. src/test_morton.c: programa de prueba unitaria del modulo.

API PUBLICA EN morton.h:

    #include <stdint.h>
    #include "matmul_naive.h"  // for scalar_t

    /*
     * Interleave the bits of i and j to produce the Morton code.
     * For (i, j) with i = i_{p-1}...i_1 i_0 and j = j_{p-1}...j_1 j_0 in binary,
     * the result is i_{p-1} j_{p-1} ... i_1 j_1 i_0 j_0.
     * i goes to even-positioned bits (bit 0, 2, 4, ...), j to odd ones.
     */
    uint64_t morton_encode(uint32_t i, uint32_t j);

    /*
     * Inverse of morton_encode. Used only for testing.
     */
    void morton_decode(uint64_t code, uint32_t *i, uint32_t *j);

    /*
     * Reorganize a row-major matrix A_row (m x m, m power of 2) into a
     * Morton-ordered linear buffer A_morton. Destination allocated by caller
     * with capacity for m*m elements.
     * Convention: A_row[i*m + j] -> A_morton[morton_encode((uint32_t)i, (uint32_t)j)].
     * Aborts with fprintf+exit(EXIT_FAILURE) if m is not a power of 2.
     */
    void reorganize_to_morton(const scalar_t *A_row, scalar_t *A_morton, size_t m);

    /*
     * Inverse of reorganize_to_morton. For validation only.
     */
    void reorganize_from_morton(const scalar_t *A_morton, scalar_t *A_row, size_t m);

    /*
     * Utility: returns 1 if m is a power of 2 (m > 0), 0 otherwise.
     */
    int is_power_of_two(size_t m);

IMPLEMENTACION DE morton_encode (Nivel 1, portable, bit twiddling):

    static inline uint64_t spread_bits_32_to_64(uint32_t x) {
        uint64_t y = x;
        y = (y | (y << 16)) & 0x0000FFFF0000FFFFULL;
        y = (y | (y <<  8)) & 0x00FF00FF00FF00FFULL;
        y = (y | (y <<  4)) & 0x0F0F0F0F0F0F0F0FULL;
        y = (y | (y <<  2)) & 0x3333333333333333ULL;
        y = (y | (y <<  1)) & 0x5555555555555555ULL;
        return y;
    }

    uint64_t morton_encode(uint32_t i, uint32_t j) {
        return spread_bits_32_to_64(i) | (spread_bits_32_to_64(j) << 1);
    }

IMPLEMENTACION DE morton_decode: aplicar las mascaras y shifts inversos para
compactar los bits pares (i) e impares (j).

IMPLEMENTACION DE is_power_of_two:

    int is_power_of_two(size_t m) {
        return m > 0 && (m & (m - 1)) == 0;
    }

IMPLEMENTACION DE reorganize_to_morton y reorganize_from_morton: doble bucle,
validar is_power_of_two(m) al inicio.

PROGRAMA DE PRUEBA src/test_morton.c:

1. Tabla 4x4: para i,j en {0,1,2,3}, verificar que morton_encode produce:
       (0,0)=0  (0,1)=1  (1,0)=2  (1,1)=3
       (0,2)=4  (0,3)=5  (1,2)=6  (1,3)=7
       (2,0)=8  (2,1)=9  (3,0)=10 (3,1)=11
       (2,2)=12 (2,3)=13 (3,2)=14 (3,3)=15

2. Round-trip encoding: para i,j en {0..63}, morton_decode(morton_encode(i,j))
   debe devolver (i,j).

3. Contiguidad de cuadrantes: para m=8, llenar A_row[i*m+j] con valores
   distinguibles, reorganizar a Morton, verificar que los cuatro cuadrantes
   ocupan posiciones consecutivas en el arreglo Morton.

4. Round-trip de reorganizacion: para m=16, 64, 256, A_row -> A_morton ->
   A_back, comparar A_row vs A_back con matrices_close.

Si todos los tests pasan, imprime MORTON TESTS OK y retorna 0.

MAKEFILE:

Agregar al final:

    test_morton: bin/test_morton

    bin/test_morton: src/test_morton.c src/morton.c src/matrix_utils.c
        $(CC) $(CFLAGS_O0) -o $@ $^

CRITERIO DE ACEPTACION:

1. `make test_morton` compila sin warnings.
2. `./bin/test_morton` imprime MORTON TESTS OK.
3. Los binarios anteriores (validate_naive_O0, validate_recursive_O0,
   bench_naive_O0, bench_recursive_O0) siguen funcionando.
4. La tabla 4x4 coincide con la del documento tecnico (Etapa A3).

Cuando termines, corre el test, muestrame el output completo y pega el contenido
de src/morton.h.
```

---

### Prompt 4 - Kernel recursivo con A en layout Morton (Etapa A3)

```text
La pieza central de la Sesion 02: el kernel recursivo donde A esta en layout
Morton y B, C siguen en row-major. La idea clave es que la recursion pasa OFFSETS
MORTON para A en lugar de un (puntero, lda) tradicional. Cuando un sub-bloque
cuadrado de A se divide en cuatro cuadrantes, los offsets Morton de esos cuadrantes
son cuatro segmentos contiguos consecutivos en memoria.

ARCHIVOS A CREAR:

- src/matmul_morton.h
- src/matmul_morton.c

FIRMA INTERNA:

    static void matmul_morton_inner(scalar_t *C, const scalar_t *A_morton, const scalar_t *B,
                                    size_t m_block, size_t k_block, size_t n_block,
                                    size_t a_morton_offset,
                                    size_t a_block_dim,
                                    size_t ldc, size_t ldb);

Donde:
- C es el sub-bloque actual de la matriz de salida row-major (leading dim ldc).
- A_morton es el puntero al ARREGLO MORTON COMPLETO de A.
- B es el sub-bloque actual de B row-major (leading dim ldb).
- m_block, k_block, n_block son las dimensiones del sub-problema actual.
- a_morton_offset es el indice en A_morton donde empieza el bloque cuadrado actual.
- a_block_dim es la dimension del bloque cuadrado actual (m_block == k_block ==
  a_block_dim cuando se divide simultaneamente por m y k).

GEOMETRIA DE LA RECURSION:

A es cuadrada m x m con m potencia de 2, almacenada en Morton. Los sub-bloques de
A en cada nivel son siempre cuadrados de tamano potencia de 2. La recursion sobre
A divide simultaneamente filas y columnas (Caso MK), mientras que B y C se dividen
independientemente.

CASOS DE RECURSION:

1. Caso base: si m_block * k_block * n_block <= RECURSION_THRESHOLD, llamar al
   kernel base.

2. Caso N (dividir n): si n_block es la dimension mas grande y n_block >= 2.
   Sub-llamadas independientes con offsets en C y B, MISMO sub-bloque de A:

       n_half = n_block / 2;
       matmul_morton_inner(C,          A_morton, B,
                           m_block, k_block, n_half,
                           a_morton_offset, a_block_dim, ldc, ldb);
       matmul_morton_inner(C + n_half, A_morton, B + n_half,
                           m_block, k_block, n_block - n_half,
                           a_morton_offset, a_block_dim, ldc, ldb);

3. Caso MK (dividir m y k simultaneamente): cuando m_block == k_block ==
   a_block_dim y m_block >= 2.

       half = a_block_dim / 2;
       quadrant_size = half * half;  // numero de elementos en cada cuadrante Morton

       // C_top += A_top_left * B_top   (sobreescribe; primera contribucion)
       matmul_morton_inner(C, A_morton, B,
                           half, half, n_block,
                           a_morton_offset + 0 * quadrant_size, half, ldc, ldb);

       // C_top += A_top_right * B_bot  (acumula)
       matmul_morton_inner_add(C, A_morton, B + half * ldb,
                               half, half, n_block,
                               a_morton_offset + 1 * quadrant_size, half, ldc, ldb);

       // C_bot += A_bot_left * B_top   (sobreescribe)
       matmul_morton_inner(C + half * ldc, A_morton, B,
                           half, half, n_block,
                           a_morton_offset + 2 * quadrant_size, half, ldc, ldb);

       // C_bot += A_bot_right * B_bot  (acumula)
       matmul_morton_inner_add(C + half * ldc, A_morton, B + half * ldb,
                               half, half, n_block,
                               a_morton_offset + 3 * quadrant_size, half, ldc, ldb);

   La correspondencia offset 0/1/2/3 -> cuadrante TL/TR/BL/BR es la propiedad
   fundamental del Z-order y debe quedar documentada con un comentario en el codigo.

KERNEL BASE PARA MORTON:

    static void kernel_base_morton(scalar_t *C, const scalar_t *A_morton,
                                   const scalar_t *B,
                                   size_t m_block, size_t k_block, size_t n_block,
                                   size_t a_morton_offset,
                                   size_t a_block_dim,
                                   size_t ldc, size_t ldb);

Tres bucles ijk. Para indexar A, usa la Opcion A (mas simple, suficiente para esta
fase):

    for (size_t i = 0; i < m_block; i++) {
        for (size_t j = 0; j < n_block; j++) {
            scalar_t sum = 0;
            for (size_t k = 0; k < k_block; k++) {
                uint64_t a_idx = a_morton_offset
                               + morton_encode((uint32_t)i, (uint32_t)k);
                sum += A_morton[a_idx] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;  // sobrescribe
        }
    }

ANALOGO kernel_base_morton_add: identico pero `C[i*ldc + j] += sum;`.

Tambien implementar matmul_morton_inner_add con la misma logica pero propagando
_add a las llamadas recursivas y al kernel base.

FUNCION PUBLICA (wrapper):

    void matmul_morton(scalar_t *C,
                       const scalar_t *A_morton,
                       const scalar_t *B,
                       size_t m, size_t k, size_t n);

Validar al inicio:
- m == k (A debe ser cuadrada).
- is_power_of_two(m).
Si fallan, abortar con mensaje a stderr y exit(EXIT_FAILURE).

Llama a matmul_morton_inner con a_morton_offset=0, a_block_dim=m, ldc=n, ldb=n.

ORQUESTADOR PARA EL BENCHMARK:

    void benchmark_iterations_morton(scalar_t *B_out,
                                     const scalar_t *A,        // row-major
                                     const scalar_t *Z,
                                     size_t m, size_t n,
                                     size_t num_iters);

Internamente:
1. Aloja A_morton con xalloc_aligned(m * m).
2. reorganize_to_morton(A, A_morton, m).
3. Aloja B_curr, B_next con xalloc_aligned(m * n) cada uno.
4. Copia Z a B_curr.
5. Para iter = 0..num_iters-1:
       matmul_morton(B_next, A_morton, B_curr, m, m, n);
       // copia primeras n filas de B_next a B_out + iter * n * n
       // swap B_curr <-> B_next
6. Libera todo.

Y una segunda funcion que recibe A_morton ya reorganizada (para que bench_morton
pueda excluir la reorganizacion del tiempo medido):

    void benchmark_iterations_morton_preorganized(scalar_t *B_out,
                                                  const scalar_t *A_morton,
                                                  const scalar_t *Z,
                                                  size_t m, size_t n,
                                                  size_t num_iters);

Misma logica que la anterior pero SIN reorganize_to_morton al inicio.

CRITERIO DE ACEPTACION:

- src/matmul_morton.h y src/matmul_morton.c existen y compilan sin warnings con
  `gcc -O0 -Wall -Wextra -Wpedantic -fsyntax-only`.
- El header expone matmul_morton, benchmark_iterations_morton y
  benchmark_iterations_morton_preorganized.
- Aborto claro cuando m no es potencia de 2 o m != k.
- No hay ejecucion del binario todavia; eso es el Prompt 5.

Cuando termines, muestra el contenido completo de src/matmul_morton.h y un
resumen de la estructura de src/matmul_morton.c.
```

---

### Prompt 5 - Bench y validate del Morton

```text
Cierre del lado de implementacion: bench y validate para el modulo Morton.

ARCHIVOS A CREAR:

1. src/validate_morton.c:
   - CLI: `./validate_morton_O0 [m]`, default m=256 (potencia de 2).
   - Aborta si m no es potencia de 2.
   - Test 1: A * 0 = 0.
   - Test 2: I * Z = Z.
   - Test 3: Linealidad.
   - Test 4: Cross-validation contra matmul_naive (m=4, 16, 64, 256).
   - Test 5: Cross-validation contra matmul_recursive (m=4, 16, 64, 256). Esta es
     la prueba mas fuerte porque ambos son recursivos.
   - Si todos pasan: VALIDATION OK.

2. src/bench_morton.c:
   - CLI: `./bench_morton_O0 <m> [num_iters] [num_runs]`.
   - Aborta si m no es potencia de 2.
   - Reorganizacion a Morton se hace UNA SOLA VEZ fuera del bucle de corridas
     medidas (incluye warm-up): aloja A, init_matrix_random, aloja A_morton,
     reorganize_to_morton. Despues entra el bucle de num_runs corridas que
     llaman a benchmark_iterations_morton_preorganized.
   - Salida CSV: m,n,num_iters,median_seconds,gflops igual que los otros bench.

MAKEFILE:

    bench_morton: bin/bench_morton_O0
    validate_morton: bin/validate_morton_O0

    bin/bench_morton_O0: src/bench_morton.c src/matmul_morton.c src/morton.c \
                         src/matrix_utils.c src/matmul_naive.c
        $(CC) $(CFLAGS_O0) -o $@ $^

    bin/validate_morton_O0: src/validate_morton.c src/matmul_morton.c \
                            src/matmul_recursive.c src/morton.c \
                            src/matrix_utils.c src/matmul_naive.c
        $(CC) $(CFLAGS_O0) -o $@ $^

CRITERIO DE ACEPTACION:

1. make validate_morton compila sin warnings.
2. ./bin/validate_morton_O0 256 imprime VALIDATION OK.
3. ./bin/validate_morton_O0 384 aborta con mensaje claro sobre la potencia de 2.
4. make bench_morton compila sin warnings.
5. ./bin/bench_morton_O0 1024 1 1 produce CSV con gflops positivos.
6. ./bin/bench_morton_O0 1024 1 1 y ./bin/bench_recursive_O0 1024 1 1 producen
   gflops dentro de un factor 5 entre si.
7. Todos los binarios anteriores siguen funcionando (validate_naive_O0,
   validate_recursive_O0, bench_naive_O0, bench_recursive_O0).

Cuando termines, ejecuta los siete pasos y pega los outputs. Si el Test 5 falla,
revisa la propagacion de a_morton_offset en kernel_base_morton: el bug clasico es
confundir indices locales y globales. Avisame si no logras resolverlo.
```

---

### Prompt 6 - Sweep comparativo y graficas

```text
Genera los sweeps de mediciones y las graficas comparativas.

ARCHIVOS A CREAR:

1. scripts/run_sweep_recursive.sh: analogo a run_sweep_naive.sh pero llama a
   bin/bench_recursive_O0 y escribe results/recursive_O0.csv.
   - Lista de m por defecto: {256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
     6144, 8192} (igual que el sweep naive).
   - Solo bench + CSV; sin profiling integrado (eso es el Prompt 7).

2. scripts/run_sweep_morton.sh: analogo pero para Morton.
   - Lista de m por defecto: solo potencias de 2: {1024, 2048, 4096, 8192}.
   - Escribe en results/morton_O0.csv.
   - Si el usuario pasa un m que no es potencia de 2, lo omite con mensaje a
     stderr y continua.

3. scripts/plot_comparison.py:
   - Lee results/naive_O0.csv, results/recursive_O0.csv, results/morton_O0.csv.
   - Genera:
     (a) plots/comparison_gflops_vs_m.png: tres curvas, eje x log, guias L1/L2/L3.
     (b) plots/comparison_time_vs_m.png: tres curvas tiempo vs m en log-log, mas
         curva teorica O(m^2 n) anclada al menor m de naive.
     (c) plots/speedup_morton_vs_recursive.png: cociente gflops(morton)/gflops(recursive)
         vs m, solo donde ambos existen.
     (d) plots/speedup_morton_vs_naive.png: analogo, morton vs naive.
   - CLI con defaults sensatos. Reusa estilo visual de plot_results.py.

MAKEFILE:

    sweep_recursive_run:
        bash scripts/run_sweep_recursive.sh

    sweep_morton_run:
        bash scripts/run_sweep_morton.sh

    plots_comparison:
        python3 scripts/plot_comparison.py

    sweep_full_santiago: sweep_recursive_run sweep_morton_run plots_comparison

ASUMPCION: results/naive_O0.csv ya existe (de la Sesion 01 + rename). Si no,
plot_comparison.py debe imprimir mensaje claro sugiriendo correr `make sweep_naive`.

CRITERIO DE ACEPTACION:

1. bash scripts/run_sweep_recursive.sh produce results/recursive_O0.csv con una
   linea por m, gflops > 0 en cada una.
2. bash scripts/run_sweep_morton.sh produce results/morton_O0.csv para las
   potencias de 2. Pasarle un m no potencia de 2 como argumento lo omite con
   mensaje.
3. python3 scripts/plot_comparison.py produce los cuatro PNG.
4. Las graficas se ven legibles (titulo, leyenda, ejes labeled).
5. Lectura preliminar coherente: la curva morton vs recursive no esta al reves
   ni tiene valores cero/negativos. Si la hipotesis no se cumple, documentarlo;
   eso es ciencia tambien.

Cuando termines, corre los sweeps + plotting y muestrame:
- Primeras 5 y ultimas 5 lineas de cada CSV.
- Lista de archivos en plots/.
- Tu lectura preliminar del speedup Morton vs recursive: desde que m se observa
  diferencia, en que magnitud.
```

---

### Prompt 7 - Analisis con perf (eventos de hardware)

```text
Profundizamos el experimento con eventos de hardware. Si Morton da speedup, debe
justificarse con menos cache misses y menos TLB misses (lo predice la teoria). Si
no da speedup, los eventos deben confirmar que el hardware prefetcher tapa todo.

ARCHIVOS A CREAR:

1. scripts/profile_perf_compare.sh:
   - CLI: `bash scripts/profile_perf_compare.sh [m_list]`, default
     "1024 2048 4096 8192".
   - Eventos: L1-dcache-loads, L1-dcache-load-misses, LLC-loads, LLC-load-misses,
     dTLB-load-misses, cycles, instructions.
   - Para cada m, corre con perf stat -x,:
       ./bin/bench_naive_O0     <m> 1 1
       ./bin/bench_recursive_O0 <m> 1 1
       ./bin/bench_morton_O0    <m> 1 1
   - Salida: results/perf_compare.csv con columnas:
       m, variant, l1_loads, l1_misses, llc_loads, llc_misses, dtlb_misses,
       cycles, instructions
     donde variant in {naive, recursive, morton}.
   - Si perf falla por permisos, imprimir mensaje claro indicando ajuste de
     perf_event_paranoid (referencia al README de la Sesion 01).

2. scripts/plot_perf_compare.py:
   - plots/perf_l1_misses.png, perf_llc_misses.png, perf_dtlb_misses.png:
     tres curvas (naive, recursive, morton) vs m.
   - plots/perf_summary_table.txt: tabla texto plano:
       m | variant | L1 miss rate | LLC miss rate | dTLB misses/instr

MAKEFILE:

    perf_compare:
        bash scripts/profile_perf_compare.sh

    plots_perf:
        python3 scripts/plot_perf_compare.py

CRITERIO DE ACEPTACION:

1. bash scripts/profile_perf_compare.sh produce results/perf_compare.csv sin
   errores, valores positivos.
2. python3 scripts/plot_perf_compare.py produce las tres graficas + la tabla.
3. Las tres variantes muestran patrones distinguibles en al menos un evento.

Cuando termines, muestrame:
- Primeras 10 lineas de results/perf_compare.csv.
- Contenido completo de plots/perf_summary_table.txt.
- Tu lectura preliminar: que evento separa mejor las variantes?
```

---

### Prompt 8 - Documentacion final y cierre de la Sesion 02

```text
Cerramos la Sesion 02 con documentacion y PR.

ARCHIVOS A MODIFICAR/CREAR:

1. docs/API.md: agregar secciones de los modulos nuevos al final, ANTES de la
   seccion "Cambios y versionado":
   - matmul_recursive (firmas, semantica, complejidad, conexion teorica con
     cache-oblivious y Hong-Kung).
   - morton (encode/decode/reorganize, propiedad de contiguidad de cuadrantes).
   - matmul_morton (restriccion m potencia de 2, reorganizacion amortizada).
   - Binarios nuevos: bench_recursive_O0, bench_morton_O0, validate_recursive_O0,
     validate_morton_O0, test_morton, con CLIs.
   - Scripts nuevos: run_sweep_recursive.sh, run_sweep_morton.sh,
     plot_comparison.py, profile_perf_compare.sh, plot_perf_compare.py.
   - Actualizar la tabla Roadmap marcando "Opcional (Morton)" como COMPLETADA.
   NO TOQUES las secciones 1-5 escritas en la Sesion 01 + rename.

2. docs/SESION_02_RESUMEN.md: analogo a SESION_01_RESUMEN.md con secciones:
   1. Contexto del proyecto (breve).
   2. Decisiones tomadas en la Sesion 02 (tabla: rename como primer paso,
      restriccion a potencia de 2 para Morton, Nivel 1 portable para
      morton_encode, kernel base en ijk, exclusion de la reorganizacion del
      tiempo medido, etc.).
   3. Estructura de archivos nuevos.
   4. Que hacen los componentes clave.
   5. Verificaciones realizadas.
   6. Commits y PRs.
   7. Como reanudar.
   8. Que sigue: plan para la Sesion 03 (opciones: integracion con Fase 2 de
      Juan Pablo, Etapa A4 kernel base optimizado, Etapa A5 OpenMP, reporte y
      presentacion).
   9. Riesgos conocidos para la Sesion 03.
   10. Referencia rapida de comandos.
   + Resumen ejecutivo de resultados experimentales (gflops, hipotesis Morton
     confirmada o no, lectura de eventos de hardware).

3. README.md: actualizar seccion de uso con los nuevos make targets.

COMMITS:

Si se hicieron commits por modulo a lo largo de los prompts, perfecto. Si todo
quedo en un solo commit grande, ahora hacer commits incrementales con git rebase
-i o (mas pragmatico) un commit final que agrupe lo no commiteado:

    feat: Phase 6 - cache-oblivious recursive and Morton implementations

    Adds matmul_recursive (Stage A2), morton module, matmul_morton (Stage A3),
    their validation and benchmark drivers, comparative sweeps, perf analysis,
    and updated documentation. No changes to naive baseline or to Juan Pablo's
    Phase 2 module path.

PR:

Titulo: "feat: Fase 6 - implementacion recursiva cache-oblivious con Morton layout"

Cuerpo:
- Que se anadio (lista de archivos nuevos).
- Resultados experimentales preliminares.
- Coordinacion con Fase 2 de Juan Pablo (no hay cruces de archivos).
- Como validar el PR (criterios de aceptacion de cada prompt).

Si gh CLI esta disponible: `gh pr create ...`. Si no, imprime el URL de
comparacion.

CRITERIO DE ACEPTACION:

1. docs/API.md tiene las secciones nuevas, sin tocar las viejas.
2. docs/SESION_02_RESUMEN.md cubre las 10 secciones.
3. README.md tiene los nuevos make targets.
4. Hay al menos un commit por modulo o un commit final agrupado bien descrito.
5. PR creado o URL de comparacion impreso.

Cuando termines:
- Tabla de contenidos de docs/SESION_02_RESUMEN.md.
- Diff de README.md.
- URL del PR.
```

---

## 5. Apendice A: rubrica resumida de validacion incremental

| Despues de | Comando | Salida esperada |
|------------|---------|-----------------|
| PRE | `./bin/validate_naive_O0 256` | VALIDATION OK |
| PRE | `make sweep_naive` | results/naive_O0.csv con varias lineas |
| 0 | `git status` desde la rama nueva | rama claude/santiago-recursive-morton limpia |
| 1 | `gcc -fsyntax-only src/matmul_recursive.c` | sin errores |
| 2 | `make validate_recursive && ./bin/validate_recursive_O0 256` | VALIDATION OK |
| 2 | `./bin/bench_recursive_O0 1024 1 1` | linea CSV bien formada |
| 2 | `./bin/validate_naive_O0 256` (regresion) | VALIDATION OK |
| 3 | `make test_morton && ./bin/test_morton` | MORTON TESTS OK |
| 4 | `gcc -fsyntax-only src/matmul_morton.c` | sin errores |
| 5 | `make validate_morton && ./bin/validate_morton_O0 256` | VALIDATION OK |
| 5 | `./bin/bench_morton_O0 1024 1 1` | linea CSV con gflops > 0 |
| 6 | `make sweep_full_santiago` | tres CSV + cuatro PNG generados |
| 7 | `make perf_compare && make plots_perf` | results/perf_compare.csv + tabla |
| 8 | inspeccion de docs/ | secciones nuevas, viejas intactas |

---

## 6. Apendice B: heuristicas de debugging para los puntos calientes

**Prompt PRE (rename) - sintomas tipicos:**

- Algun grep de sanidad retorna referencias colgadas: revisar bloques de codigo en
  Markdown (los snippets de comandos en README, SESION_01_RESUMEN, API), tablas
  Markdown, y comentarios al inicio de los scripts (donde se mencionan los
  binarios producidos).
- `make sweep_naive` falla porque alguna receta interna del Makefile aun referencia
  `bench_O0`: hacer grep dentro del Makefile mismo.
- Profiling scripts no encuentran el binario: revisar variables internas en
  los scripts; `BENCH_BIN`, `BENCH_PG_BIN` o similar.

**Prompt 1 (recursivo row-major) - sintomas y causas tipicas:**

- Resultado difiere de matmul_naive en un solo bloque (cuadrante inferior
  derecho): puntero mal desplazado en la recursion. Imprime los punteros C, A, B
  y tamanos en cada llamada y verifica a mano para m=k=n=4.
- Correcto para m chico, diverge para m grande: RECURSION_THRESHOLD lleva tarde
  al kernel base; el caso de division por k no acumula bien. Inspecciona
  kernel_base_add.
- Stack overflow: la recursion no llega al kernel base. Verifica la comparacion
  con RECURSION_THRESHOLD y el manejo del caso m=k=n=1.

**Prompt 3 (morton encoding) - sintomas:**

- Tabla 4x4 da numeros raros: mascaras de spread_bits_32_to_64 mal escritas. El
  bug clasico es 0x5555... vs 0xAAAA... o el orden de los shifts invertido.
- morton_decode no es inversa: el compactador aplica mascaras y shifts en orden
  inverso al spread, paso a paso.

**Prompt 4 (morton kernel) - los bugs mas dolorosos:**

- a_morton_offset mal propagado: dos llamadas con el mismo offset hacen que dos
  cuadrantes contribuyan a la misma region de C. Resultado "casi correcto" pero
  diverge en partes especificas. Regla mnemotecnica: offset 0=TL, 1=TR, 2=BL,
  3=BR. Esto es porque morton_encode(0,0)=0, morton_encode(0,1)=1,
  morton_encode(1,0)=2, morton_encode(1,1)=3.
- Confusion entre indices locales y globales en el kernel base: dentro del
  kernel, i va de 0 a m_block-1 (local). El offset global es
  a_morton_offset + morton_encode(i_local, k_local).
- B se desplaza mal cuando se divide por k: el B inferior es `B + half * ldb`,
  NO `B + half`. ldb es la leading dimension de la matriz B ORIGINAL.

**Prompt 5 (cross-validation Morton vs recursive) - como leer fallos:**

Si el Test 5 falla pero los Tests 1-4 pasan, el bug esta especificamente en el
manejo del a_morton_offset o en kernel_base_morton. Estrategia diagnostica:
temporalmente reemplaza kernel_base_morton por una version que indexa A como
row-major (calculando indices globales i_global, k_global desde i_local,
k_local y los offsets de bloque) y verifica si los tests pasan. Si pasan, el bug
esta en morton_encode dentro del kernel. Si no pasan, el bug esta en la
propagacion de offsets en la recursion.

---

## 7. Apendice C: que hacer si Juan Pablo mergea su Fase 2 mientras trabajas

Despues del rename PRE, ambos brancheais desde el main renombrado. Tu solo creas
archivos nuevos y extiendes Makefile/API.md. Juan Pablo solo crea archivos nuevos
(matmul_reordered.c, etc.) y extiende los mismos archivos compartidos. Los
conflictos posibles son:

**(a) Conflicto en Makefile:** mas probable. Resolucion manual: aceptar ambos
lados, agrupar targets de Juan Pablo y los tuyos en bloques separados con
comentarios `# === Fase 2 targets ===` y `# === Fase 6 targets ===`.

**(b) Conflicto en docs/API.md:** improbable porque ambos solo agregan secciones
al final. Si pasa, aceptar ambos cambios y reordenar la tabla del Roadmap.

**(c) Conflicto en README.md:** depende. Si ambos extienden la seccion de uso,
ordenarlos en bloques.

Recomendacion: rebasea tu rama sobre main al menos una vez al dia mientras la
Fase 2 esta abierta:

    git fetch
    git rebase origin/main

Cuanto mas frecuentemente rebases, mas baratos seran los conflictos.

---

*Documento generado al inicio de la Sesion 02. Edita conforme avances o si surgen
ajustes a los prompts.*
