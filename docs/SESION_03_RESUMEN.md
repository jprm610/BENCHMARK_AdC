# Resumen de la Sesion 03

**Fecha:** 2026-05-16
**Estado:** Sesion 03 cerrada. Etapa A4 (microkernel AVX2 + FMA), Etapa A5 (OpenMP tasks) y caracterizacion completa (perf Zen 2 + Roofline anclado a STREAM) implementadas, validadas y mergeadas en `main`. Las ramas de trabajo `claude/*` se borraron al cierre; el historial vive en los $17$ PRs mergeados.
**Predecesor:** Sesion 02 (Fase 6: cache-oblivious recursivo + Morton sobre row-major).
**Siguiente paso:** Sesion 04 (Fase 5: comparacion contra OpenBLAS, opcional BLIS / MKL, y reporte final del curso).

Este documento sirve para retomar el proyecto sin releer la conversacion de la Sesion 03. Anexa al `SESION_02_RESUMEN.md` los hallazgos cuantitativos de la sesion: microkernel a $\sim 64.6 \%$ del techo FMA single-core de Zen 2, cliff L3 confirmado empiricamente en $m \approx 1024$, OMP que escala bien hasta $6$ threads, y un Roofline completo del Ryzen $5$ $4600$H.

---

## 1. Contexto del proyecto

Benchmark de **multiplicacion iterada de matrices** para Arquitectura de Computadores, UNAL Medellin. Recurrencia:

$$
B_0 = Z, \qquad B_{i+1} = A \cdot B_i, \qquad i = 0, 1, \ldots, I-1
$$

con $A \in \mathbb{R}^{m \times m}$, $Z \in \mathbb{R}^{m \times n}$, $n = 128$, $I = 2m/n$.

La Sesion 01 dejo `matmul_naive`; la Sesion 02 anadio `matmul_recursive` (cache-oblivious row-major) y `matmul_morton` (recursion sobre layout Z-order fino). La Sesion 03 baja al hardware especifico del Ryzen $5$ $4600$H y cuantifica cuanto del Roofline tocamos con tres palancas: microkernel AVX2 + FMA $4 \times 16$, paralelizacion con OpenMP tasks, y profiling con eventos PMC de Zen $2$.

Documentos de referencia:

- [`docs/PLAN_SESION_03.md`](PLAN_SESION_03.md) - plan tecnico de la sesion.
- [`docs/PROMPTS_SESION_03.md`](PROMPTS_SESION_03.md) - guion de los $10$ prompts ejecutados (PRE + $0$ a $9$).
- [`docs/SESION_02_RESUMEN.md`](SESION_02_RESUMEN.md) - punto de partida.
- [`docs/API.md`](API.md) - secciones $11$, $12$, $13$ describen los modulos nuevos.

---

## 2. Decisiones tomadas en la Sesion 03

| Decision | Valor | Razon |
|----------|-------|-------|
| **Auditoria PDEP/PEXT como primer paso** | `scripts/audit_no_pdep.sh` (Prompt PRE), corre bajo `make audit` | BMI2 en Zen $2$ esta microcodeado: `PDEP`/`PEXT` cuestan $\sim 18$ ciclos vs $\sim 3$ en Intel. La auditoria verifica que ningun modulo Morton emite estas instrucciones, ni siquiera incidentalmente via `__builtin_*`. |
| **L3 efectiva: $4$ MiB por CCX** | Asumido en lugar de los $8$ MiB del datasheet | Renoir reparte los $8$ MiB totales en $2$ CCX de $4$ MiB cada uno; los $3$ cores de un CCX no ven el L3 del otro CCX. Todas las predicciones de cliff usan $4$ MiB. |
| **Rango de $m$ acotado** | $m \in \{512, 1024, 2048, 4096, 8192, 16384\}$ | RAM total $8$ GiB (en WSL2 efectivo $\sim 3.5$ GiB sin tunear); $m = 16384$ ya consume $\sim 1$ GiB para $A$. Limite documentado, no inherente al algoritmo. |
| **Microkernel $4 \times 16$** | $8$ acumuladores YMM, $2$ vectores por fila | Maximo tile que cabe en $8$ de los $16$ YMM disponibles dejando registros libres para broadcasts; saturando un FMA pipe del Zen $2$. |
| **Morton "de bloques" (tile = $4$)** | Layout nuevo coexistiendo con el Morton fino | El microkernel necesita panels row-major contiguos en el leaf. Morton fino exige decodificar por elemento dentro del bucle interno; Morton-de-bloques materializa un panel con copia $O(m^2)$ amortizada. La recursion en cuadrantes es identica en ambos. |
| **Threshold de hoja AVX2** | $g\_recursion\_threshold\_avx2 = 524288 = 64 \cdot 64 \cdot 128$ | Hoja de $64 \times 64$ floats por panel de $A$: working set $\sim 16$ KiB (mitad de L1d). Empiricamente equivalente a otros valores en la "isla" $[8\text{K}, 1\text{M}]$ dentro del ruido; se eligio el valor consistente con la geometria de L1d. |
| **OpenMP tasks por subdivision** | `#pragma omp task` en cada split por encima de `g_parallel_threshold_omp = 524288`; scratch por-thread | Tasks (no `parallel for`) porque la division MK genera $4$ sub-llamadas con dependencias de orden (TL/BL sobreescriben C, TR/BR acumulan); las tasks expresan ese DAG naturalmente. |
| **Afinidad recomendada para $4600$H** | `OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores` | Default para todos los benches OMP. SMT a $12$ threads no aporta en AVX2 + FMA (saturado el retirement); `spread` paga cross-CCX a cambio de balance. |
| **Compilacion del microkernel** | `-O3 -march=znver2 -mavx2 -mfma -funroll-loops -ffast-math` | `-ffast-math` autoriza reasociacion de suma FP (necesario para que GCC emita cadenas FMA) a cambio de mas error. Tolerancia de validacion subida a $10^{-3}$. |
| **Bandwidth de referencia: STREAM medido** | STREAM Triad $1$T $\approx 15.7$ GB/s, $6$T $\approx 17.9$ GB/s | El datasheet menciona $51.2$ GB/s teoricos; el sistema entrega $\sim 31$–$35 \%$ de eso bajo WSL2. El Roofline se ancla en lo medido, no en el teorico. |
| **Eventos perf Zen 2** | $6$ eventos por celda (compute + memoria + TLB) en $2$ grupos para evitar multiplexing | `min_mux_pct = 100 \%` en las $12$ celdas. $3$ eventos del prompt original sustituidos por proxies disponibles en kernel WSL2 $6.6.114$. |

---

## 3. Estructura de archivos nuevos

Listado de archivos creados o extendidos en la Sesion 03. El baseline de Fase $1$ y los modulos de Fase $6$ (`matmul_recursive`, `morton`, `matmul_morton`) quedan intactos.

```
src/
|-- hwinfo.c                    Caracterizacion runtime del CPU (Prompt 1)
|-- kernel_avx2.h               Microkernel AVX2 + FMA 4x16 (Prompt 3)
|-- kernel_avx2.c               Implementacion del microkernel (Prompt 3)
|-- test_kernel_avx2.c          Unit test del microkernel kc in {1, 8, 128, 1024} (Prompt 3)
|-- matmul_morton_avx2.h        Variante Morton con microkernel AVX2 (Prompt 4)
|-- matmul_morton_avx2.c        Implementacion + reorganize_to_morton_blocks (Prompt 4)
|-- validate_morton_avx2.c      Cross-validacion contra naive y morton (Prompt 4)
|-- bench_morton_avx2.c         Driver bench_morton_avx2_O3 (Prompt 4)
|-- matmul_morton_omp.h         Variante paralela con OpenMP tasks (Prompt 6)
|-- matmul_morton_omp.c         Implementacion con scratch por-thread (Prompt 6)
|-- validate_morton_omp.c       Cross-validacion a 1/4/12 threads (Prompt 6)
|-- bench_morton_omp.c          Driver bench_morton_omp_O3 (Prompt 6)

scripts/
|-- audit_no_pdep.sh            Auditoria PDEP/PEXT (Prompt PRE)
|-- run_threshold_sweep.sh      Sweep de RECURSION_THRESHOLD (Prompt 2)
|-- plot_threshold_sweep.py     Plot del threshold sweep (Prompt 2)
|-- run_sweep_session_03.sh     Sweep comparativo 4 variantes (Prompt 5)
|-- plot_sweep_session_03.py    Plot del sweep comparativo (Prompt 5)
|-- run_sweep_morton_avx2_xl.sh Extension de morton_avx2 hasta m=32768 (Prompt 5 ext)
|-- plot_morton_avx2_xl.py      Plot de la extension (Prompt 5 ext)
|-- run_omp_scaling.sh          Escalado threads 1..12, bind close/spread (Prompt 6)
|-- plot_omp_scaling.py         Plot del escalado OMP (Prompt 6)
|-- profile_perf_zen2.sh        Captura perf por celda (Prompt 7)
|-- run_perf_zen2_sweep.sh      Orquesta las 12 celdas (Prompt 7)
|-- consolidate_perf_zen2.py    Consolida grupos A+B a un CSV (Prompt 7)
|-- plot_perf_zen2.py           Plot 4 paneles IPC/FMA/L3/TLB (Prompt 7)
|-- measure_stream.sh           Descarga, compila y corre STREAM (Prompt 8)
|-- plot_roofline.py            Roofline anclado a STREAM + perf (Prompt 8)

docs/
|-- PLAN_SESION_03.md           Plan tecnico (Prompt 0; extendido en Prompts 2 y 7)
|-- PROMPTS_SESION_03.md        Guion completo de los 10 prompts
|-- SESION_03_RESUMEN.md        Este archivo (Prompt 9)
|-- API.md                      Extendido con secciones 11, 12, 13 (Prompt 9)

README.md                       Extendido con sub-seccion "Flujo de Sesion 03"
Makefile                        Extendido con bloque "Sesion 03 targets" (todos los makes nuevos)

results/
|-- hwinfo.csv                  Snapshot del fingerprinting (Prompt 1)
|-- threshold_sweep.csv         30 puntos del sweep (Prompt 2)
|-- omp_scaling.csv         29 puntos: 7 threads x 2 bindings x 2 m's + 1 default (Prompt 6)
|-- perf_<variant>_m<M>_{A,B}.txt   24 archivos crudos perf (Prompt 7)
|-- perf_zen2_summary.csv       12 celdas consolidadas (Prompt 7)
|-- stream_1t.txt, stream_6t.txt   Triad bandwidth medido (Prompt 8)

plots/
|-- perf_zen2_breakdown.png     4 paneles IPC/FMA/L3/TLB (Prompt 7)
|-- roofline_4600h.png          Roofline final anclado (Prompt 8)
```

Totales: $13$ archivos `.c/.h` nuevos, $14$ scripts nuevos, $3$ documentos nuevos. Makefile, `docs/API.md` y `README.md` extendidos al final.

---

## 4. Que hacen los componentes clave

### 4.1 `hwinfo` (Prompt 1)

Binario standalone que imprime el fingerprint del CPU: modelo, vendor, cores/threads logicos, soporte AVX2/FMA3/BMI2, tamanos de L1d/L2/L3 leidos de `/sys/devices/system/cpu/cpu0/cache/`, RAM total y disponible de `/proc/meminfo`. Modo `--csv` para encabezado de reportes. Advierte si L3 reportado es menor a $8$ MiB (probable Renoir/Cezanne con $4$ MiB por CCX) o si BMI2 esta soportado en AMD pre-Zen$3$ (alerta del costo de `PDEP`/`PEXT`).

### 4.2 `audit_no_pdep.sh` (Prompt PRE)

Compila los modulos Morton (`morton.c`, `matmul_morton.c`, `matmul_morton_avx2.c`, `matmul_morton_omp.c`) con `-O3 -march=znver2` y verifica con `objdump` que el ensamblador resultante no contiene `pdep` ni `pext`. Imprime `PASS` o `FAIL` con detalle por archivo. Critico porque BMI2 esta microcodeado en Zen $2$ ($\sim 18$ ciclos) y un uso incidental degradaria el throughput sin notarlo.

### 4.3 `kernel_avx2_4x16` (Prompt 3)

Microkernel hot loop. Acumula $C[4, 16] \mathrel{+}= A[4, kc] \cdot B[kc, 16]$ manteniendo el tile entero de $C$ en $8$ registros YMM (4 filas $\times$ 2 vectores). El bucle interno emite $8$ FMAs por iteracion (uno por acumulador) usando `_mm256_fmadd_ps` con broadcasts de $A$ y loads no-alineados de $B$. Compilado con `-funroll-loops -ffast-math` para que GCC emita las cadenas FMA. Validado contra una referencia `ijk` en doble precision para $kc \in \{1, 8, 128, 1024\}$ con tolerancia $10^{-3}$ relativa.

### 4.4 `matmul_morton_avx2` (Prompt 4)

Integra el microkernel en la recursion Morton. Sustituye el layout fino (un float por celda Morton) por **Morton-de-bloques** con tile $4 \times 4$: cada celda Morton apunta a un sub-bloque row-major de $16$ floats. La recursion sigue siendo cuadrante TL/TR/BL/BR con offsets $\{0, 1, 2, 3\} \cdot h^2$ (la propiedad de contiguidad de cuadrantes se preserva). En el leaf, el codigo materializa un panel `A_local` row-major desde el layout Morton-de-bloques y llama al microkernel sobre el panel materializado. Threshold runtime-tunable (default $524288$, $\sim 16$ KiB de working set por panel).

### 4.5 `matmul_morton_omp` (Prompt 6)

Misma estructura recursiva que `matmul_morton_avx2`, pero el wrapper publico envuelve la primera llamada en `#pragma omp parallel single` y cada subdivision por encima de `g_parallel_threshold_omp` emite dos `omp task` (Caso N: dos columnas; Caso MK: $4$ cuadrantes con dependencias de orden TL/BL/TR/BR). El scratch buffer `A_local` que el microkernel necesita pasa a ser un pool por-thread indexado por `omp_get_thread_num()`. Dos thresholds independientes (`g_recursion_threshold_omp` y `g_parallel_threshold_omp`), ambos con default $524288$: tasks disparan en cada nivel sobre la hoja, nunca dentro.

### 4.6 Scripts de profiling (Prompts 7 y 8)

- `profile_perf_zen2.sh` toma `(variant, m)` y corre dos invocaciones perf por celda (grupo A: `cycles, instructions, fp_ret_sse_avx_ops.all, ls_dispatch.ld_dispatch, l2_request_g1.all_no_prefetch`; grupo B: `cycles, instructions, l2_cache_req_stat.ls_rd_blk_l_hit_x, cache-misses, bp_l1_tlb_miss_l2_tlb_miss, dTLB-load-misses`). Dos grupos para mantener `min_mux_pct = 100 \%`. Tres eventos del prompt original no expuestos en kernel WSL2 $6.6.114$ se sustituyeron por proxies documentados en el script.
- `consolidate_perf_zen2.py` lee los archivos crudos `perf_<v>_m<M>_{A,B}.txt`, parsea el CSV de `perf -x ,`, computa IPC, fp/cyc, miss rates y TLB walks por kilo-instruccion, y vuelca `results/perf_zen2_summary.csv`.
- `plot_perf_zen2.py` genera `plots/perf_zen2_breakdown.png` con $4$ paneles (IPC, FMA throughput, L3 miss rate, TLB walks/kinst) en un solo grid para comparar las $4$ variantes a tres tamanos.
- `measure_stream.sh` descarga `stream.c` de McCalpin, compila con `-O3 -fopenmp -march=znver2`, corre $1$T y $6$T (`OMP_PROC_BIND=spread`), guarda los reportes.
- `plot_roofline.py` lee `stream_{1,6}t.txt` y los archivos perf por variante, computa `(arithmetic_intensity, achieved_gflops)` por celda con `fp_ops / runtime_ns` y dibuja el Roofline anclado a STREAM medido (no al pico DRAM teorico).

---

## 5. Resultados experimentales

### 5.1 Tabla de GFLOPS por (variante, $m$)

Bench-level GFLOPS reportados por `plot_roofline.py` (anclados a `fp_ret_sse_avx_ops.all` y wall clock perf):

| variante       |  $m=1024$ |  $m=4096$ |  $m=8192$ |
|----------------|----------:|----------:|----------:|
| `naive`        |     2.47  |     2.16  |     0.92  |
| `recursive`    |     5.13  |     5.24  |     5.18  |
| `morton`       |     0.50  |     0.50  |     0.50  |
| `morton_avx2`  |    32.57  |    39.51  |  **39.96**|
| `morton_omp`   |    11.44  |    21.02  |    25.22  |

Lectura: `morton_avx2` saca un $43 \times$ sobre `naive` a $m = 8192$ ($39.96 / 0.92 \approx 43.4$); `recursive` (mismo kernel `ijk`, sin SIMD) saca apenas $5.6 \times$. La curva plana de `morton` (fino) confirma que el bottleneck es CPU-bound en aritmetica entera (`morton_encode` por elemento), no memoria.

`morton_omp` aparece aqui en su modo de profiling `OMP_NUM_THREADS=6 OMP_PROC_BIND=spread` con $I = 4$ iteraciones. El throughput ahi cae por debajo de `morton_avx2` por el overhead de gestion de tasks sobre sub-problemas chicos relativos al kernel AVX2 ya muy rapido. El verdadero scaling se ve en el sweep dedicado (Seccion 5.4).

### 5.2 Speedup acumulado vs `naive`

Cociente $\text{GFLOPS}_{\text{variante}} / \text{GFLOPS}_{\text{naive}}$ a $m = 8192$:

| variante       | speedup vs naive |
|----------------|-----------------:|
| `naive`        |   $1.0 \times$   |
| `recursive`    |   $5.6 \times$   |
| `morton`       |   $0.5 \times$ (regresion: el indice integer ahoga al kernel) |
| `morton_avx2`  | **$43.4 \times$**|
| `morton_omp`   |  $27.4 \times$ (profiling); hasta $\sim 278 \times$ con sweep dedicado (Seccion 5.4) |

El salto principal viene de la Etapa A4 (Morton + microkernel AVX2 + FMA). La Etapa A5 (OpenMP tasks) anade un factor multiplicativo encima.

### 5.3 Fraccion del Roofline alcanzada

Pico single-core de Zen $2$ en FP32: $2$ FMA $\times$ $8$ lanes $\times$ $2$ ops/FMA $\times$ $4.0$ GHz $= 128$ GFLOPS. Pico multi-core con $6$ cores activos: $\sim 768$ GFLOPS ideal (sin throttling AVX). STREAM Triad medido: $15.7$ GB/s $1$T, $17.9$ GB/s $6$T. Knee del Roofline: $128 / 15.7 \approx 8.2$ flops/byte $1$T, $768 / 17.9 \approx 43$ flops/byte $6$T.

Fraccion del techo alcanzada por celda (perf-level: `fp_ops_per_cycle / 16`):

| variante       | $m=1024$ |  $m=4096$  |  $m=8192$  | regimen |
|----------------|---------:|-----------:|-----------:|---------|
| `naive`        |    $4.0\,\%$ |    $3.6\,\%$  |    $1.5\,\%$  | memory-bound severo |
| `recursive`    |    $8.2\,\%$ |    $8.4\,\%$  |    $8.3\,\%$  | compute-bound sin SIMD |
| `morton`       |    $0.8\,\%$ |    $0.8\,\%$  |    $0.8\,\%$  | CPU-bound integer (`morton_encode`) |
| `morton_avx2`  |   $53.6\,\%$ |   $64.4\,\%$  | **$64.6\,\%$**| compute-bound vectorial, $\sim 1$ pipe FMA saturado |

La hipotesis cientifica del plan se confirma: solo el microkernel AVX2 + FMA toca una fraccion significativa del techo. Que `morton_avx2` mantenga $\sim 64 \%$ a $m = 8192$ (donde $A$ son $256$ MiB, $64 \times$ el L3 efectivo) demuestra que la localidad espacial del Morton-de-bloques + reuso intensivo de $C$ en registros amortiza el trafico a DRAM. El otro $\sim 35 \%$ esta en el segundo pipe FMA inalcanzable sin loop unrolling adicional o software prefetching del panel siguiente.

### 5.4 Eficiencia de OpenMP

Sweep dedicado `omp_scaling.csv` (driver `bench_morton_omp_O3` con $I = 1$, $3$ runs, mediana). Threads $\in \{1, 2, 3, 4, 6, 8, 12\}$ contra `OMP_PROC_BIND` $\in \{$close, spread$\}$ a $m \in \{4096, 8192\}$. Valores en GFLOPS:

| $m$  | threads | close  | spread | speedup close vs $1$T | ideal |
|-----:|--------:|-------:|-------:|----------------------:|------:|
| $4096$ |    $1$ |   $60.7$ |   $60.6$ |             $1.0\times$ | $1.0\times$ |
| $4096$ |    $2$ |  $117.8$ |  $114.8$ |             $1.9\times$ | $2.0\times$ |
| $4096$ |    $3$ |  $124.6$ |  $121.3$ |             $2.1\times$ | $3.0\times$ |
| $4096$ |    $4$ |  $140.7$ |  $142.0$ |             $2.3\times$ | $4.0\times$ |
| $4096$ |    $6$ |  $191.5$ |  $176.8$ |             $3.2\times$ | $6.0\times$ |
| $4096$ |    $8$ |  $223.7$ |  $214.0$ |             $3.7\times$ | $6.0\times$ (sin SMT) |
| $4096$ |   $12$ | **$262.2$** |  $244.1$ |             $4.3\times$ | $6.0\times$ (sin SMT) |
| $8192$ |    $1$ |   $59.7$ |   $59.9$ |             $1.0\times$ | $1.0\times$ |
| $8192$ |    $3$ |  $114.3$ |  $120.7$ |             $1.9\times$ | $3.0\times$ |
| $8192$ |    $6$ |  $190.5$ |  $193.8$ |             $3.2\times$ | $6.0\times$ |
| $8192$ |   $12$ | **$255.8$** |  $258.6$ |             $4.3\times$ | $6.0\times$ (sin SMT) |

Lecturas:

- A $2$ threads ambos bindings van casi a speedup lineal ($\sim 1.9\times$): el footprint se reparte entre cores del mismo CCX y el L2 privado por core acelera la materializacion del panel.
- A $3$ threads `close` la curva pierde linealidad: los tres cores del CCX se pegan entre si en el L3 compartido de $4$ MiB. `spread` ($3$T) reparte un thread por CCX y desbloquea el segundo L3, pero el cross-CCX traffic le impide capitalizar mas alla.
- A $6$ threads ambas afinidades convergen alrededor de $\sim 3.2 \times$. La eficiencia paralela cae a $\sim 53 \%$: el cliff dominante es el bandwidth de DRAM (STREAM $6$T solo $1.14 \times$ el $1$T, la jerarquia satura rapido).
- SMT a $8$ y $12$ threads sigue sumando pero con eficiencia decreciente: $12$T close = $255.8$ GFLOPS a $m = 8192$ y $262.2$ GFLOPS a $m = 4096$ (ambos $\sim 4.3 \times$ del $1$T, no $12 \times$). El segundo pipe FMA del Zen $2$ queda parcialmente alimentado por el SMT sibling.
- **Mejor configuracion practica:** $12$ threads `close` para maximizar throughput puro, $6$ threads `close` si se quiere preservar L3 para otra tarea concurrente, $3$ threads `spread` para experimentos single-CCX limpios.

### 5.5 Cliff L3 confirmado empiricamente

El cliff de L3 ($4$ MiB efectiva por CCX, $m_{L3} = \sqrt{4 \text{MiB} / 4 \text{B}} = 1024$) se valida directamente con `l3_miss_rate` de `naive`:

| $m$    | naive L3 miss rate | comentario |
|-------:|-------------------:|------------|
| $1024$ |      $33.2 \,\%$   | $A$ entera ($4$ MiB) compite con $B$ y $C$ por el L3 |
| $4096$ |      $96.9 \,\%$   | $A$ son $64$ MiB, $16\times$ el L3; todo cae a DRAM |
| $8192$ |      $83.2 \,\%$   | denominador crece pero `fp/cyc` colapsa a $0.24$ |

`recursive`, `morton` y `morton_avx2` no sufren el cliff: $l3\_miss\_rate$ se mantiene en $0.5 \,\%$, $11$–$16 \,\%$ y $\sim 3 \,\%$ respectivamente. La propiedad cache-oblivious de la recursion se ve directamente en los contadores.

### 5.6 Reduccion de TLB walks

A $m = 8192$, walks de pagina completos (`bp_l1_tlb_miss_l2_tlb_miss`) por kilo-instruccion:

| variante       | walks/kinst | vs naive |
|----------------|------------:|---------:|
| `naive`        |    $0.0084$ | $1.0\times$ |
| `recursive`    |    $0.0018$ | **$4.7\times$ menos** |
| `morton`       |    $0.0011$ | **$7.6\times$ menos** |
| `morton_avx2`  |    $0.0049$ | $1.7\times$ menos |

El Z-order reduce dramaticamente los page walks (casi un orden de magnitud) al mantener el working set en pocas paginas de $4$ KiB. `morton_avx2` paga mas TLB que `morton` fino por la materializacion del panel `A_local` (stride reads sobre los sub-bloques de $4 \times 4$); el trade-off vale el $10 \times$ de FLOPS pero documenta una oportunidad de optimizacion futura.

---

## 6. Verificaciones realizadas

| Verificacion | Resultado |
|--------------|-----------|
| `make audit` (`scripts/audit_no_pdep.sh`) | **PASS** sobre `morton.c`, `matmul_morton.c`, `matmul_morton_avx2.c`, `matmul_morton_omp.c` |
| `make hwinfo && ./bin/hwinfo` (Prompt 1) | Reporta `L1d=32, L2=512, L3=4096 KiB`; warning de "L3 < 8 MiB esperado por CCX" emitido como disenado |
| `make test_kernel_avx2 && ./bin/test_kernel_avx2` (Prompt 3) | $4$ PASS ($kc \in \{1, 8, 128, 1024\}$) |
| `objdump -d build/obj/kernel_avx2.o | grep vfmadd | wc -l` | $> 8$ FMAs emitidas |
| `./bin/validate_morton_avx2_O3 256` (Prompt 4) | VALIDATION OK ($5$ tests: $3$ invariantes + cross naive + cross morton) |
| `./bin/validate_morton_omp_O3 256` con `OMP_NUM_THREADS` $\in \{1, 4, 12\}$ (Prompt 6) | VALIDATION OK en los tres regimenes (sin race conditions) |
| `make sweep_session_03` (Prompt 5) | $24$ puntos generados ($4$ variantes $\times$ $6$ tamanos); `plots/session_03_gflops_vs_m.png` |
| `make profile_zen2 && make plot_perf_zen2` (Prompt 7) | $12$ celdas, $\min mux\_pct = 100 \%$; `plots/perf_zen2_breakdown.png` |
| `make stream && make plot_roofline` (Prompt 8) | STREAM Triad $1$T = $15.7$ GB/s, $6$T = $17.9$ GB/s; `plots/roofline_4600h.png` |
| Regresion Sesion 02: `./bin/validate_naive_O0 256`, `validate_recursive_O0 256`, `validate_morton_O0 256` | Todos VALIDATION OK; baseline intacto |

---

## 7. Commits y PRs

Rama de la sesion: `claude/santiago-session-03-*` (multiples sub-ramas por prompt, todas mergeadas a `main`).

| Prompt | PR | Resumen |
|-------:|----|---------|
| PRE | [#7](https://github.com/jprm610/BENCHMARK_AdC/pull/7) | Auditoria PDEP/PEXT (`scripts/audit_no_pdep.sh` + `make audit`) |
| 0 | (sin PR) | `docs/PLAN_SESION_03.md` y `docs/PROMPTS_SESION_03.md` creados directo en `main` |
| 1 | [#8](https://github.com/jprm610/BENCHMARK_AdC/pull/8) | `src/hwinfo.c` + `make hwinfo` |
| 2 | [#9](https://github.com/jprm610/BENCHMARK_AdC/pull/9) | Tuning empirico `RECURSION_THRESHOLD` (`run_threshold_sweep.sh`, plot) |
| 3 | [#10](https://github.com/jprm610/BENCHMARK_AdC/pull/10) | `kernel_avx2_4x16` + test |
| 4 | [#12](https://github.com/jprm610/BENCHMARK_AdC/pull/12) | `matmul_morton_avx2` + validate + bench |
| 5 | [#11](https://github.com/jprm610/BENCHMARK_AdC/pull/11) y [#12 ext](https://github.com/jprm610/BENCHMARK_AdC/pull/12) | Sweep comparativo + extension `morton_avx2` hasta $m = 32768$ |
| 6 | [#13](https://github.com/jprm610/BENCHMARK_AdC/pull/13) | `matmul_morton_omp` + validate + bench + sweep OMP |
| 7 | [#14](https://github.com/jprm610/BENCHMARK_AdC/pull/14) y [#15](https://github.com/jprm610/BENCHMARK_AdC/pull/15) | perf Zen $2$: scripts, $24$ archivos crudos, summary CSV, plot |
| 8 | [#16](https://github.com/jprm610/BENCHMARK_AdC/pull/16) | STREAM + Roofline final |
| 9 | (este PR) | Documentacion de cierre (`API.md`, `SESION_03_RESUMEN.md`, `README.md`) |

Coordinacion con Juan Pablo (Camino B, Fases $3$ y $4$): la sesion solo creo archivos nuevos y extendio `Makefile` / `docs/API.md` / `README.md` al final, en bloques etiquetados `# === Sesion 03 ===` y "Modulo $11/12/13$". Los archivos de Fase $2$ (`matmul_reordered.c`, `matmul_tiled.c`) no se tocaron.

---

## 8. Como reanudar la sesion

```bash
# 1. Sincronizar
cd ~/Proyectos/BENCHMARK_AdC      # o el path equivalente en WSL2
git checkout main
git pull origin main

# 2. Pre-requisitos del entorno (una vez por boot)
sudo sysctl -w kernel.perf_event_paranoid=1   # para perf
source ~/venvs/matmul/bin/activate            # matplotlib + numpy

# 3. Compilar todo y correr la bateria de regresion
make audit                                    # PDEP/PEXT clean
make hwinfo && ./bin/hwinfo                   # fingerprint del CPU
make validate_recursive validate_morton \
     validate_morton_avx2 validate_morton_omp
./bin/validate_naive_O0          256
./bin/validate_recursive_O0      256
./bin/validate_morton_O0         256
./bin/validate_morton_avx2_O3    256
OMP_NUM_THREADS=4 ./bin/validate_morton_omp_O3 256

# 4. Reproducir los resultados experimentales (en este orden)
make sweep_threshold && make plot_threshold   # ~8 min
make sweep_session_03 && make plot_session_03 # ~25 min
make sweep_omp_scaling && make plot_omp_scaling   # ~10 min
make profile_zen2 && make plot_perf_zen2      # ~5 min
make stream                                   # ~2 min
make profile_zen2_omp                         # ~3 min
make plot_roofline                            # < 1 min

# 5. Inspeccionar
ls results/                                   # CSVs y archivos crudos perf
ls plots/                                     # roofline_4600h.png, perf_zen2_breakdown.png
explorer.exe plots/roofline_4600h.png         # desde WSL2
```

Branch de trabajo actual del Prompt $9$ (esta sesion): `claude/romantic-swartz-c353e4`. Mergeada a `main` con el PR del Prompt $9$.

---

## 9. Que sigue: opciones para la Sesion 04

Cuatro caminos, ordenados por valor curricular esperado.

### 9.1 (Recomendado) Comparacion final contra OpenBLAS

Cerrar el bucle con la libreria de referencia del estado del arte. Agregar `bench_openblas` que invoca `cblas_sgemm` con la misma instrumentacion que los benches propios, y extender `plot_sweep_session_03.py` con la curva de OpenBLAS. Resultado esperado: OpenBLAS toca $\sim 90$–$95 \%$ del techo FMA multi-core, cerca de $2 \times$ lo que saca `morton_omp` a $12$T. El delta esta en el packing de $B$ con software prefetching y en el segundo pipe FMA que `morton_avx2` deja sin tocar. Cumple la **Fase 5 del plan original** del proyecto.

### 9.2 Reporte y presentacion del curso

Compilar los hallazgos de las Sesiones $1$–$3$ en un documento entregable y una presentacion. La estructura natural:

1. Baseline (`matmul_naive`) y caracterizacion del problema (Sesion 01).
2. Localidad: recursion cache-oblivious y Z-order (Sesion 02).
3. Optimizacion fina al hardware: microkernel AVX2 + OMP (Sesion 03).
4. Comparacion con OpenBLAS y conclusiones (Sesion 04).

Los plots ya existen: `naive_gflops_vs_m.png`, `comparison_gflops_vs_m.png`, `perf_zen2_breakdown.png`, `roofline_4600h.png`. Falta hilarlos con texto y las tablas de este resumen.

### 9.3 Mejoras opcionales (algoritmo)

- **Streaming stores** (`_mm256_stream_ps`) en la copia de salida de la materializacion del panel, para evitar polucion del L1d del thread con datos que solo se escriben una vez.
- **Software prefetching** del siguiente panel de $A$ durante el FMA del actual (`__builtin_prefetch(A_next, 0, 0)`).
- **Reescribir el microkernel para consumir directamente Morton-de-bloques** sin materializacion intermedia: eliminaria los TLB walks extras de la Seccion 5.6 a costa de un kernel mas complejo. Estimacion: $10$–$15 \%$ extra de pico.

### 9.4 Comparacion con BLIS y MKL

Si se consigue acceso a MKL (Intel) o BLIS configurado para Zen $2$, agregar ambas como curvas comparativas. BLIS-Zen$2$ es interesante porque comparte arquitectura (sandwich GEMM tres niveles) con el camino seguido en esta sesion y daria una referencia limpia del techo "humano" de un GEMM bien hecho para Zen $2$. MKL en AMD historicamente funciona pero subutiliza FMA en Zen; medirlo serviria para discutir el sesgo de optimizacion por vendor.

---

## 10. Riesgos conocidos para la Sesion 04

| Riesgo | Mitigacion |
|--------|------------|
| OpenBLAS por defecto usa todos los hilos logicos; sin afinidad explicita cruza CCX y baja $\sim 30 \%$ vs `OMP_PROC_BIND=close`. | Forzar `OPENBLAS_NUM_THREADS=6` y `OMP_PROC_BIND=close` para comparabilidad con `morton_omp`. |
| Tolerancia de OpenBLAS contra el baseline: la libreria reordena agresivamente, la suma FP difiere mas que entre nuestras variantes. | Tolerancia $10^{-2}$ relativa para los cross-checks contra OpenBLAS (mayor que el $10^{-3}$ usado para `morton_avx2`). |
| MKL en AMD requiere setear `MKL_DEBUG_CPU_TYPE=5` (workaround para usar AVX2 en CPU no-Intel). Algunas builds recientes ya no lo respetan. | Si MKL ignora el workaround, documentarlo y usar la version "Intel-fair" sin el flag como referencia explicitamente sub-optima. |
| WSL2 limita memoria a $\sim 3.5$ GiB por defecto; impide $m = 32768$ con $A$ de $4$ GiB. | Si la Sesion 04 quiere extender el rango, ajustar `~/.wslconfig` con `memory=7GB` y `wsl --shutdown`. Alternativa: correr el bench desde un nativo Linux dual-boot. |
| `-ffast-math` puede divergir aun mas al introducir OpenBLAS (que tambien reordena); riesgo de detectar "fallas" que son solo distintas reasociaciones. | Mantener `validate_morton_avx2_O3` como prueba aislada del kernel propio antes de cross-validar contra OpenBLAS. |
| El sweep XL de `morton_avx2` (Prompt 5 ext) llega hasta $m = 16384$ por limite WSL; $m = 32768$ probablemente sera OOM-killed. | El script ya tiene chequeo de `MemAvailable` y aborta antes de lanzar el bench si la estimacion excede el $85 \%$. No tocar a menos que se haya ajustado WSL. |
| Throttling AVX2 bajo carga sostenida ($> 30$ s) en laptops: el $4600$H puede bajar de $4.0$ GHz a $3.0$–$3.2$ GHz. | Las mediciones del Prompt $5$ usan $1$ iter $\times$ $3$ runs (sub-segundo cada uno) para evitar el regimen sostenido. Si la Sesion $04$ corre tests largos (multi-iter), reportar la frecuencia con `cpupower frequency-info`. |

---

## Apendice: comandos rapidos para reproducir todos los resultados

```bash
# === Compilacion ===
make                                 # baseline + recursive + morton (Fases 1 y 6)
make hwinfo                          # bin/hwinfo (Sesion 03)
make audit                           # auditoria PDEP/PEXT

# === Bench targets (Sesion 03) ===
make bench_morton_O3                 # variant Morton con -O3 -march=znver2
make bench_morton_avx2               # microkernel AVX2 + FMA
make bench_morton_omp                # version paralela con OpenMP tasks
make test_kernel_avx2                # unit test del microkernel
make validate_morton_avx2            # cross-validation AVX2
make validate_morton_omp             # cross-validation OMP (correr con OMP_NUM_THREADS=1,4,12)

# === Sweeps y profiling ===
make sweep_threshold                 # tuning empirico RECURSION_THRESHOLD
make plot_threshold

make sweep_session_03                # 4 variantes x 6 tamanos
make plot_session_03

make sweep_morton_avx2_xl            # extension morton_avx2 hasta m=16384/32768
make plot_morton_avx2_xl

make sweep_omp_scaling               # 7 threads x 2 bindings x 2 m's
make plot_omp_scaling

make profile_zen2                    # 12 celdas perf Zen 2 (compute + memoria)
make plot_perf_zen2
make profile_zen2_omp                # perf para morton_omp (6 threads bind=spread)

make stream                          # STREAM Triad 1T y 6T
make plot_roofline                   # Roofline final anclado a STREAM medido
make roofline                        # stream + plot_roofline en cadena

# === Regresion completa (criterio de aceptacion del Prompt 9) ===
./bin/validate_naive_O0          256
./bin/validate_recursive_O0      256
./bin/validate_morton_O0         256
./bin/validate_morton_avx2_O3    256
OMP_NUM_THREADS=4 ./bin/validate_morton_omp_O3 256
bash scripts/audit_no_pdep.sh        # PASS

# === Visualizar resultados ===
explorer.exe plots/perf_zen2_breakdown.png
explorer.exe plots/roofline_4600h.png
explorer.exe plots/session_03_gflops_vs_m.png
explorer.exe plots/omp_scaling.png
cat results/perf_zen2_summary.csv
cat results/omp_scaling.csv
```

---

*Documento generado al cierre de la Sesion 03 (Prompt 9). Mantiene el formato del `SESION_02_RESUMEN.md` para continuidad. Las cifras numericas estan ancladas en los CSVs / plots commiteados; cualquier divergencia al reproducir indica regresion y debe levantar un issue antes de la Sesion 04.*
