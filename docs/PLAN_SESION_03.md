# Plan de la Sesion 03: optimizacion fina de Morton al Ryzen 5 4600H

**Fecha:** 2026-05-15
**Rama:** `claude/santiago-session-03-tuning`
**Predecesor:** PR #7 (auditoria PDEP/PEXT), mergeado en `main` (commit `e1dee41`).
**Documento maestro:** `docs/PROMPTS_SESION_03.md` (en la copia de trabajo de `main`, fuera de este worktree).
**Estado:** plan aprobado; sin codigo todavia.

---

## 1. Objetivo cientifico

La Sesion 02 dejo `matmul_morton` "funcionando y validado" pero con un `kernel_base` ingenuo: tres bucles `ijk` que dependen unicamente de lo que el compilador logre auto-vectorizar bajo `-O3`. En la practica esto deja **mas del 90 % del techo de Zen 2 sobre la mesa**: el procesador puede retirar dos FMA de 256 bits por ciclo, equivalentes a $16$ flops/ciclo por core en FP32, y el `ijk` ingenuo apenas se acerca.

El objetivo de la Sesion 03 es **bajar al hardware especifico** y cuantificar cuanto del Roofline del Ryzen 5 4600H podemos tocar con tres palancas sucesivas:

1. **Microkernel AVX2 + FMA** (Etapa A4) con tile fijo $4 \times 16$, 8 acumuladores en registros YMM, dirigido a saturar las dos unidades FMA del Zen 2.
2. **Paralelizacion OpenMP tasks** (Etapa A5) respetando la topologia de 2 CCX del Renoir (3 cores + L3 de 4 MiB privada por CCX); cualquier hilo que cruce CCX paga coherencia y pierde la ventaja de L3.
3. **Tuning empirico del threshold de recursion** sobre la maquina real, en lugar de la constante de manual de la Sesion 02.

Las preguntas que la sesion debe responder cuantitativamente, ancladas en mediciones de la maquina:

- ¿Que fraccion del pico FMA single-core ($128$ GFLOPS) toca `matmul_morton_avx2` en su mejor $m$?
- ¿Donde esta el cliff de L3 efectiva (4 MiB por CCX) y como se separan las curvas naive / recursive / morton / morton_avx2 a partir de el?
- ¿Cuanto escala `matmul_morton_omp` con `OMP_PROC_BIND=close` (single-CCX) vs `spread` (cross-CCX)? ¿SMT a 12 threads aporta o es ruido?
- ¿Donde cae cada variante en el Roofline anclado a un STREAM medido (no al $51.2$ GB/s teorico)?

Los cuellos de botella reales se identifican con `perf` Zen 2 (eventos `fp_ret_sse_avx_ops.all`, `l3_lookup_state.l3_miss`, `bp_l1_tlb_miss_l2_tlb_miss`, etc.) y se cierran con un diagrama Roofline `plots/roofline_4600h.png` que es el entregable cientifico final de la sesion.

---

## 2. Hardware objetivo: especificaciones medidas

Estas son las **especificaciones medidas en la maquina real**, no las del datasheet generico del 4600H. Cualquier decision de tamano de bloque, tile, threshold o numero de threads se deriva de esta tabla.

| Recurso | Valor medido | Asociatividad / detalles | Implicacion para el codigo |
|---------|--------------|---------------------------|----------------------------|
| Modelo | AMD Ryzen 5 4600H | Renoir, Zen 2, 7 nm | `-march=znver2 -mtune=znver2` |
| Cores / threads | 6 / 12 (SMT) | Distribuidos en 2 CCX de 3 cores | OpenMP: probar 6 vs 12 threads |
| Frecuencia | 3.0 GHz base, 4.0 GHz turbo | Reduce bajo carga AVX2 sostenida | Fijar gobernador `performance` |
| L1d por core | 32 KiB | 8-way, lineas de 64 B | Hoja recursiva: working set $\leq 16$ KiB |
| L1i por core | 32 KiB | 8-way | Microkernel debe caber en code cache |
| L2 por core | 512 KiB | 8-way, privada al core | Tiling intermedio opcional |
| L3 efectiva | **4 MiB por CCX** | Compartida entre 3 cores del mismo CCX | Cliff L3 a partir de $m \approx 1024$ |
| Memoria | 8 GiB (7.37 GiB usables) | DDR4-3200 dual channel | Rango factible: $m$ hasta $2^{14} = 16384$ |
| SIMD | AVX2 + FMA3 | Datapath nativo de 256 bits | `__m256`, `_mm256_fmadd_ps` |
| BMI2 | Soportado pero microcodeado | PDEP/PEXT con latencia $\sim 18$ ciclos | **NO usar PDEP/PEXT (auditado en Prompt PRE)** |
| Pagina | 4 KiB | L1 DTLB 64 entries, L2 DTLB 2048 entries | Morton ayuda con TLB para $m$ grande |

### 2.1 Numeros derivados (referencias del Roofline)

- **Pico FP32 por core**: $2\ \text{FMA} \times 8\ \text{lanes} \times 2\ \text{ops} \times 4.0\ \text{GHz} = 128$ GFLOPS.
- **Pico FP32 multi-core (6 cores activos)**: $\sim 770$ GFLOPS (ideal; el throttling AVX bajara este numero bajo carga sostenida).
- **Bandwidth DRAM teorico**: $3200 \cdot 8 \cdot 2 \approx 51.2$ GB/s. Realista (STREAM, a medir en Prompt 8): $\sim 35$ a $40$ GB/s.
- **Balance maquina single-core**: $128 / 40 \approx 3.2$ flops/byte. Multi-core: $770 / 40 \approx 19$ flops/byte.
- **Cliff L1d**: working set de `A_block + B_block + C_block` $\leq 32$ KiB.
- **Cliff L2**: working set $\leq 512$ KiB.
- **Cliff L3 (single-thread)**: $A$ entera deja de caber cuando $m^2 \cdot 4\ \text{B} > 4\ \text{MiB}$, es decir $m > 1024$.
- **Limite RAM**: para evitar swap, $A$ no debe exceder $\sim 2$ GiB, es decir $m \lesssim 22000$. Practico para el sweep: $m \leq 16384$ (ver Seccion 3).

---

## 3. Rango de $m$ del sweep

Se redefine el rango respecto a la Sesion 02 para alinearse al cliff L3 real y al limite de RAM de 8 GiB:

$$
m_{\text{sweep}} = \{512,\ 1024,\ 2048,\ 4096,\ 8192,\ 16384\}
$$

Todos son potencias de 2, asi que el sweep de Morton se ejecuta sobre los seis puntos sin restriccion adicional.

### 3.1 Justificacion del limite superior

El consumo de memoria de $A$ ($m \times m$, FP32) escala con $4 m^2$ bytes:

| $m$ | Tamano de $A$ | Comentario |
|-----|--------------|------------|
| $4096$ | $64$ MiB | Cabe holgado |
| $8192$ | $256$ MiB | Cabe; deja $\sim 7$ GiB para el resto del sistema |
| $16384$ | $1$ GiB | Limite practico con 8 GiB de RAM totales |
| $32768$ | $4$ GiB | Excede el margen practico; el sistema empieza a swap |

El proyecto original menciona barrer hasta $m = 2^{20}$; con 8 GiB es inviable y se documenta como restriccion explicita del hardware de pruebas. Si en el futuro se corre en una maquina con $\geq 32$ GiB de RAM, el sweep se puede extender sin tocar codigo.

### 3.1.1 Hallazgo de `hwinfo` sobre la RAM visible al benchmark

El binario `bin/hwinfo` reporta `mem_total_kib = 3683428` ($\approx 3.51$ GiB) en lugar de los $\sim 7.37$ GiB del host. La causa: el benchmark corre en WSL2, que por defecto limita la memoria de la VM a la mitad de la RAM del host (con techo de 8 GiB segun la version de WSL). Las implicaciones:

- $m = 16384$ requiere $\sim 1$ GiB para $A$ + $\sim 16$ MiB para cada buffer de $B$ + $\sim 16$ MiB para $C$; sumando $\approx 1.05$ GiB. Cabe en 3.5 GiB, pero el margen contra el sistema y el shell es estrecho (riesgo de OOM-killer si la maquina hace algo mas en paralelo).
- Mitigacion opcional: crear `C:\Users\<usuario>\.wslconfig` con `[wsl2]\nmemory=7GB` y reiniciar WSL (`wsl --shutdown`) para subir el limite. Solo es necesario si $m = 16384$ falla por memoria.
- El sweep se ejecuta normalmente hasta $m = 8192$ sin riesgo (max $\sim 260$ MiB de working set total). Para $m = 16384$, verificar `free -h` antes y dejar margen.

### 3.2 Cliff esperado de L3 e hipotesis Morton

A partir de $m \approx 1024$, $A$ entera deja de caber en L3 efectiva (4 MiB por CCX). La hipotesis cientifica es:

> Las curvas de naive y morton (sin AVX2) empiezan a separarse visiblemente a partir de $m = 1024$; el cliff es donde la localidad espacial de Z-order pasa de "marginalmente util" a "claramente dominante". `morton_avx2` deberia mantener una pendiente mas plana hasta $m = 8192$ porque la combinacion de localidad espacial + reuso intensivo de C en registros amortiza el trafico hacia DRAM.

El sweep comparativo del Prompt 5 produce la grafica que confirma o refuta la hipotesis.

---

## 4. Cliff esperado de L3 y plan de medicion

La transicion donde $A$ entera deja de caber en L3 (4 MiB efectivos por CCX) ocurre en:

$$
m_{L3} = \sqrt{\frac{4\ \text{MiB}}{4\ \text{B}}} = \sqrt{2^{20}} = 1024
$$

Hipotesis de comportamiento esperado para cada variante en el sweep:

| Variante | Antes del cliff ($m \leq 512$) | Despues del cliff ($m \geq 2048$) |
|----------|--------------------------------|-------------------------------------|
| `naive` | Limitado por overhead de control de flujo (`-O0`) o por auto-vectorizacion debil | Limitado por bandwidth DRAM; GFLOPS plano o decreciente |
| `recursive` | Similar a naive (mismo kernel base, recursion barata) | Mejora moderada por menor presion de cache |
| `morton` | Similar o ligeramente peor que recursive (overhead de `morton_encode`) | Mejora clara sobre recursive en TLB misses (confirmado por Sesion 02: factor $\sim 100\times$ menos L1 miss rate) |
| `morton_avx2` | Domina por amplio margen gracias al microkernel | Mantiene fraccion alta del techo FMA si el threshold esta bien tuneado |
| `morton_omp` | Overhead de tasks excede beneficio (sub-problemas chicos) | Speedup cercano a $6\times$ con `bind=close`; $\leq 6\times$ con `bind=spread` por costo cross-CCX |

El Prompt 5 valida la primera y la tercera fila; el Prompt 6 valida la quinta. El Prompt 7 (`perf`) provee la prueba directa: si Morton tiene menor `l3_lookup_state.l3_miss` que naive a $m \geq 2048$, la hipotesis del cliff se confirma directamente con eventos de hardware.

---

## 5. Archivos a crear

Lista tomada literalmente de la Seccion 2 ("Archivos que tu creas") de `docs/PROMPTS_SESION_03.md`. El script `scripts/audit_no_pdep.sh` ya fue creado en el Prompt PRE (PR #7, mergeado), por lo que no aparece en esta lista de pendientes.

### 5.1 Codigo C (`src/`)

| Archivo | Origen (Prompt) | Proposito |
|---------|-----------------|-----------|
| `src/hwinfo.c` | 1 | Caracterizacion automatica de cache sizes, RAM, soporte AVX2/FMA/BMI2; modo legible y `--csv` para encabezado de reportes. |
| `src/kernel_avx2.h`, `src/kernel_avx2.c` | 3 | Microkernel AVX2 + FMA para tile $4 \times 16$; 8 acumuladores YMM; bucle interno saturado en FMA. |
| `src/test_kernel_avx2.c` | 3 | Unit test standalone del microkernel: $kc \in \{1, 8, 128, 1024\}$, comparacion contra `ijk` doble precision. |
| `src/matmul_morton_avx2.h`, `src/matmul_morton_avx2.c` | 4 | Variante de Morton que usa el microkernel en las hojas; threshold ajustado para que la hoja sea multiplo de $4 \times 16$. |
| `src/validate_morton_avx2.c` | 4 | Cross-validation contra `matmul_naive` con tolerancia $1\text{e}-3$ relativo (AVX2 + `ffast-math` introduce mas error). |
| `src/bench_morton_avx2.c` | 4 | Driver de benchmark con la misma CLI que `bench_morton`. |
| `src/matmul_morton_omp.h`, `src/matmul_morton_omp.c` | 6 | Variante paralela con OpenMP tasks; threshold de paralelizacion configurable; respeta la regla "no paralelizar division por $k$". |
| `src/validate_morton_omp.c` | 6 | Cross-validation con `OMP_NUM_THREADS in {1, 4, 12}` para detectar condiciones de carrera. |
| `src/bench_morton_omp.c` | 6 | Driver de benchmark de la variante paralela. |

### 5.2 Scripts (`scripts/`)

| Archivo | Origen (Prompt) | Proposito |
|---------|-----------------|-----------|
| `scripts/run_threshold_sweep.sh` | 2 | Sweep de `RECURSION_THRESHOLD` $\in \{2K, 4K, ..., 1M\}$ elementos para $m \in \{1024, 2048, 4096\}$; salida `results/threshold_sweep.csv`. |
| `scripts/plot_threshold_sweep.py` | 2 | Plot del threshold sweep con una curva por $m$; identifica maximo empirico. |
| `scripts/run_sweep_session_03.sh` | 5 | Sweep comparativo de las 4 variantes (naive, recursive, morton, morton_avx2) en $m_{\text{sweep}}$. |
| `scripts/plot_sweep_session_03.py` | 5 | Graficas `session_03_gflops_vs_m.png` (con anotaciones de cliffs L3 y TLB) y `session_03_speedup_vs_naive.png`. |
| `scripts/run_omp_scaling.sh` | 6 | Escalado de threads $\in \{1, 2, 3, 4, 6, 8, 12\}$ con `OMP_PROC_BIND` en `close` y `spread`. |
| `scripts/plot_omp_scaling.py` | 6 | Plot de speedup vs threads para ambas configuraciones de afinidad. |
| `scripts/profile_perf_zen2.sh` | 7 | Captura eventos `perf` Zen 2 (`fp_ret_sse_avx_ops.all`, `l3_lookup_state.l3_miss`, `bp_l1_tlb_miss_l2_tlb_miss`, etc.) para cada combinacion (variante, $m$). |
| `scripts/consolidate_perf_zen2.py` | 7 | Parser de los reportes de `perf stat` -> `results/perf_zen2_summary.csv` con IPC, FP-ops/ciclo, miss rates. |
| `scripts/plot_perf_zen2.py` | 7 | Plot con 4 subplots: IPC, FMA throughput, L3 miss rate, TLB walks per kinst. |
| `scripts/measure_stream.sh` | 8 | Compila STREAM (McCalpin), corre `OMP_NUM_THREADS in {1, 6}` con afinidad spread; salida `results/stream_{1t,6t}.txt`. |
| `scripts/plot_roofline.py` | 8 | Roofline anclado a bandwidth STREAM medido y FLOPs `perf` medidos; ubica las 5 variantes en (intensity, GFLOPS). |

### 5.3 Documentacion (`docs/`)

| Archivo | Origen (Prompt) | Proposito |
|---------|-----------------|-----------|
| `docs/PLAN_SESION_03.md` | 0 | Este documento. |
| `docs/SESION_03_RESUMEN.md` | 9 | Resumen ejecutivo de cierre (10 secciones + apendice de comandos). |
| `docs/API.md` | 9 | **Se extiende** con tres secciones nuevas (`kernel_avx2`, `matmul_morton_avx2`, `matmul_morton_omp`). |
| `README.md` | 9 | Se extiende la seccion de uso con los nuevos `make` targets. |

### 5.4 Build (`Makefile`) - solo extension

Todos los targets nuevos van bajo un bloque etiquetado `# === Sesion 03 targets ===` (la auditoria ya tiene un bloque con ese nombre desde PR #7; los nuevos targets se agregan al mismo bloque o en un sub-bloque). Sin modificar ningun target del baseline ni de Fase 6.

```
hwinfo                     -> bin/hwinfo (caracterizacion del CPU)
sweep_threshold            -> bash scripts/run_threshold_sweep.sh
plot_threshold             -> python3 scripts/plot_threshold_sweep.py
test_kernel_avx2           -> bin/test_kernel_avx2
validate_morton_avx2       -> bin/validate_morton_avx2_O3
bench_morton_avx2          -> bin/bench_morton_avx2_O3
sweep_session_03           -> bash scripts/run_sweep_session_03.sh
plot_session_03            -> python3 scripts/plot_sweep_session_03.py
validate_morton_omp        -> bin/validate_morton_omp_O3
bench_morton_omp           -> bin/bench_morton_omp_O3
sweep_omp_scaling          -> bash scripts/run_omp_scaling.sh
plot_omp_scaling           -> python3 scripts/plot_omp_scaling.py
profile_zen2               -> bash scripts/profile_perf_zen2.sh
plot_perf_zen2             -> python3 scripts/plot_perf_zen2.py
stream                     -> bash scripts/measure_stream.sh
plot_roofline              -> python3 scripts/plot_roofline.py
```

---

## 6. Archivos que NO se tocan

Por la regla del proyecto "baseline inmutable" y por coordinacion en paralelo con la rama de Juan Pablo (Fases 3 y 4 del Camino B):

**Codigo baseline (intacto):**

- `src/matmul_naive.h`, `src/matmul_naive.c`
- `src/bench_naive.c`, `src/validate_naive.c`

**Codigo Fase 6 sin AVX2 (intacto):**

- `src/matmul_recursive.h`, `src/matmul_recursive.c`
- `src/bench_recursive.c`, `src/validate_recursive.c`
- `src/morton.h`, `src/morton.c`
- `src/matmul_morton.h`, `src/matmul_morton.c` (excepcion documentada: el Prompt 2 puede convertir `RECURSION_THRESHOLD` de macro a variable global con setter, manteniendo el default. Si el cambio es invasivo, se prefiere duplicar la indireccion en el modulo nuevo `matmul_morton_avx2`.)
- `src/bench_morton.c`, `src/validate_morton.c`, `src/test_morton.c`

**Utilidades compartidas (intactas):**

- `src/matrix_utils.h`, `src/matrix_utils.c`
- `src/timing.h`

**Scripts existentes (intactos):**

- `scripts/run_sweep_naive.sh`, `scripts/profile_gprof_naive.sh`, `scripts/profile_perf_naive.sh`, `scripts/plot_results.py`
- `scripts/run_sweep_recursive.sh`, `scripts/run_sweep_morton.sh`, `scripts/plot_comparison.py`
- `scripts/profile_perf_compare.sh`, `scripts/plot_perf_compare.py`
- `scripts/audit_no_pdep.sh` (Prompt PRE, mergeado)

**Codigo de Juan Pablo (intacto):**

- `src/matmul_reordered.{c,h}`, `src/matmul_tiled.{c,h}` y cualquier `bench_*` / `validate_*` correspondiente, si Juan Pablo los introdujo en `main`.

**Archivos que solo se extienden al final:**

- `Makefile`: nuevos targets en bloque `# === Sesion 03 targets ===`. Sin modificar recetas existentes.
- `docs/API.md`: nuevas secciones al final, antes de "Cambios y versionado".
- `README.md`: la seccion "Como usar" se extiende con los nuevos comandos; las secciones de instalacion y conceptos quedan intactas.

---

## 7. Restricciones conocidas

| Restriccion | Origen | Implicacion para la sesion |
|-------------|--------|---------------------------|
| **Morton requiere $m$ potencia de 2** | Sesion 02 (heredada) | El sweep usa solo potencias de 2 ($\{512, 1024, ..., 16384\}$); no hay puntos intermedios para Morton. Naive y recursive pueden correr en cualquier $m$ pero el sweep comparativo se mantiene en potencias de 2 para que las 4 curvas tengan los mismos puntos. |
| **Microkernel AVX2 asume FP32** | Diseno del tile $4 \times 16$ con 8 acumuladores YMM (8 lanes de FP32 por vector) | Cambiar a FP64 requiere reescribir el microkernel: el tile pasaria a $4 \times 8$ y los broadcasts a `_mm256_broadcast_sd`. El proyecto entero usa `typedef float scalar_t`, asi que no es un problema actual; solo se documenta para futuras sesiones. |
| **OpenMP tasks: topologia 2 CCX** | Hardware Renoir | Cada CCX tiene 3 cores y 4 MiB L3 privada. Threads que se separan entre CCX pierden coherencia de L3 y pagan trafico Infinity Fabric. Por eso `run_omp_scaling.sh` compara `OMP_PROC_BIND=close` (mismo CCX, escalado bueno hasta 3 threads) contra `bind=spread` (escalado hasta 6 threads pero con mas overhead). La recomendacion default sera `bind=close` para benchmarks single-CCX y `bind=spread` para los que requieran los 6 cores. |
| **Hoja del microkernel multiplo de $4 \times 16$** | Geometria del tile | El threshold del Prompt 4 se ajusta a un valor que garantice $m_b, n_b$ multiplos de $4, 16$ respectivamente. Sugerencia inicial: $64 \cdot 64 \cdot 128 = 524288$, o el valor empirico del Prompt 2 redondeado al multiplo adecuado. Casos no-alineados caen al fallback `ijk` sin vectorizar. |
| **Layout Morton "de bloques" (tile=4) en lugar de Morton "fino"** | Sub-tarea introducida en Prompt 4 | Morton "fino" (Sesion 02): cada elemento $A[i,j]$ en `A_morton[morton_encode(i,j)]`. Morton "de bloques" (Sesion 03): el $A$ se particiona en sub-bloques de $4 \times 4$ floats, los sub-bloques son Z-ordenados entre si, y los 16 elementos de cada sub-bloque estan en row-major. El microkernel AVX2 puede materializar un panel row-major desde este layout con un costo $O(m^2)$ amortizado, vs. el costo $O(m^2 n)$ que tendria decodificar Morton fino por elemento dentro del bucle interno. **Ambos layouts coexisten**: `morton.{c,h}` y `matmul_morton.{c,h}` (Sesion 02) quedan intactos; el nuevo `matmul_morton_avx2.{c,h}` expone su propia `reorganize_to_morton_blocks`. La recursion es identica en ambos kernels porque los offsets de los cuadrantes (`{0,1,2,3} * (half * half)`) dependen del Z-order de bloques, no de elementos. |
| **`-ffast-math` cambia tolerancias de validacion** | Sesion 03 | `kernel_avx2.c` se compila con `-O3 -march=znver2 -mavx2 -mfma -funroll-loops -ffast-math`. Esto autoriza reasociacion de suma FP y acumula mas error que el morton sin AVX2. Tolerancia de `validate_morton_avx2.c`: $1\text{e}-3$ relativo (frente a $1\text{e}-4$ del morton sin AVX2). |
| **`perf` en WSL2 requiere `perf_event_paranoid`** | Kernel custom de WSL2 | El script `profile_perf_zen2.sh` aborta con mensaje claro si los eventos no estan disponibles; ajuste manual con `sudo sysctl -w kernel.perf_event_paranoid=1` (no persiste entre reboots; mismo procedimiento que en Sesion 02 para `perf_compare`). |
| **Limite practico de RAM: $m \leq 16384$** | Hardware del laptop (8 GiB) | El sweep se trunca en $m = 16384$ aunque la consigna original mencione $2^{20}$. Si naive con $m = 16384$ tarda mas de 5 minutos, se permite saltarlo solo para naive (variable `NAIVE_M_MAX` en `run_sweep_session_03.sh`). |
| **PDEP/PEXT prohibidos** | Auditoria PR #7 mergeada | `make audit` debe seguir pasando despues de cualquier cambio del microkernel. El target esta en CI local; cualquier intrinseco BMI2 introducido en `kernel_avx2.c` o `morton.c` rompe el build de la sesion. |

---

*Plan preparatorio de la Sesion 03. Ninguna linea de codigo se ha escrito todavia; la implementacion empieza al confirmar este plan e iniciar el Prompt 1 (`hwinfo`).*

---

## 8. Resultados del Prompt 2: tuning empirico de `RECURSION_THRESHOLD`

Sweep ejecutado en la maquina real con `bin/bench_morton_O3` (compilado con `-O3 -march=znver2 -mavx2 -mfma`), `NUM_ITERS=1`, `NUM_RUNS=3` (1 warm-up + 3 corridas medidas, mediana). Diez thresholds en escala log $\in \{2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576\}$ elementos cruzados con $m \in \{1024, 2048, 4096\}$ = 30 mediciones. Tiempo total: 7 min 44 s (~15 s por punto). Archivo de datos: `results/threshold_sweep.csv`. Plot: `plots/threshold_sweep.png`.

### 8.1 Argmax estricto por $m$

| $m$ | argmax threshold (elementos) | GFLOPS mediana | min GFLOPS observado | variacion |
|----:|-----------------------------:|---------------:|---------------------:|----------:|
| 1024 | 1048576 | 0.5491 | 0.5057 | 7.9 % |
| 2048 | 524288 | 0.5452 | 0.5078 | 6.8 % |
| 4096 | 8192 | 0.5455 | 0.5104 | 6.4 % |

Los tres argmaxes caen en thresholds distintos. En un primer sweep de control (mismo binario, otro instante) los argmaxes fueron 4096 / 4096 / 8192 — diferencia atribuible al ruido de timing y al ruido del scheduler del SO. La metrica de "argmax estricto" **no es robusta** en este regimen.

### 8.2 Metrica robusta: media de GFLOPS por threshold sobre los tres $m$

| threshold | media GFLOPS | observacion |
|----------:|-------------:|-------------|
| **1048576** | **0.5446** | top de cluster grande |
| **524288**  | **0.5432** | top de cluster grande |
| **8192**    | **0.5427** | top de cluster chico |
| **16384**   | **0.5427** | top de cluster chico |
| 262144 | 0.5381 | mid |
| 65536 | 0.5331 | mid |
| 2048 | 0.5322 | bottom |
| 131072 | 0.5276 | bottom |
| 32768 | 0.5273 | bottom |
| 4096 | 0.5190 | worst |

Aparecen dos **islas de buen rendimiento** separadas por un valle central:

- **Isla chica** ($8192$–$16384$): leaf con working set $\sim 4$ a $8$ KiB, cabe holgado en L1d (32 KiB). Coherente con la prediccion teorica del Prompt 2 ($m_b \approx 32$ a $50$).
- **Isla grande** ($524288$–$1048576$): casi sin recursion para $m \leq 4096$ (basta una o dos divisiones para caer al kernel base con bloques grandes). El compilador autovectoriza partes del kernel grande ligeramente mejor en este regimen.

La diferencia entre las dos islas y el valle es del orden de $3$–$5$ %, **comparable al ruido entre runs** ($\sim 8$ % observado). En la practica los datos no distinguen un optimo claro.

### 8.3 Lectura: por que la sensibilidad es baja

Las GFLOPS se mueven en $\sim 0.53$ a $0.55$ a lo largo de todo el sweep porque el kernel base actual (`kernel_base_morton`: tres bucles `ijk` con `morton_encode` dentro del loop interno) impide la autovectorizacion de GCC. El bottleneck es el calculo del indice Morton elemento-a-elemento, no la profundidad de la recursion ni la localidad del sub-bloque. Consecuencias:

- El threshold optimo es **insensible** a la jerarquia de cache en este regimen.
- La curva esperada (knee abajo por overhead de recursion, knee arriba por spill de L1d) no aparece porque el kernel, saturado en el calculo del indice, nunca se vuelve memory-bound a esta escala.
- Cualquier threshold en $\{8192, 16384, 524288, 1048576\}$ da rendimiento estadisticamente equivalente.

### 8.4 Decision para el default de Sesion 03

**Threshold optimo empirico para 4600H con el kernel ingenuo: $\mathbf{8192}$ elementos.** Justificacion:

1. Es el menor de los thresholds de la "isla chica", consistente con la **prediccion teorica** (leaf working set en L1d).
2. **Anticipa** el cambio de regimen que introduce el Prompt 3: cuando el microkernel AVX2 + FMA reemplace el `ijk + morton_encode` por un tile $4 \times 16$ con 8 acumuladores YMM, el kernel base sera $10$–$30\times$ mas rapido y el bottleneck pasara de "computo del indice" a "trafico desde L1d/L2/L3". En ese regimen los thresholds chicos (que mantienen el leaf en L1d) ganan claramente sobre los grandes.
3. Es el valor mas chico que ya cabe en la geometria del microkernel: $4 \times 16 = 64$ multiplicaciones por iteracion del kernel interno; $8192 / 128 = 64$ elementos de $A$ por bloque hoja, suficiente para no fragmentar la recursion en exceso.

`g_recursion_threshold` mantiene su default de codigo en $131072$ (Sesion 02) por compatibilidad binaria con `validate_morton_O0` y los scripts de Sesion 02. Para los benchmarks de Sesion 03 con el kernel ingenuo se pasa `--threshold 8192` explicito. Cuando llegue el microkernel AVX2 del Prompt 3, el sweep se repite con `bench_morton_avx2_O3` y el nuevo optimo se vuelve el default del modulo `matmul_morton_avx2`.

### 8.5 Caveat sobre la utilidad del sweep en este regimen

El sweep cumple su funcion de criterio de aceptacion del Prompt 2, pero el **valor cientifico** del numero elegido es marginal mientras el kernel base sea `ijk + morton_encode` sin vectorizar: cualquier threshold "razonable" produce el mismo rendimiento dentro del ruido. El **verdadero** sweep de tuning ocurre en el Prompt 4, donde se mide el threshold sobre `matmul_morton_avx2`. El procedimiento ya esta listo (`scripts/run_threshold_sweep.sh` parametrizado por `BENCH_BIN`, plot reutilizable), asi que el Prompt 4 solo necesita cambiar el binario y reescribir esta seccion.

---

## 9. Resultados perf Zen 2 (Prompt 7)

Sweep de contadores de hardware sobre las cuatro variantes a $m \in \{1024, 4096, 8192\}$ usando `perf stat -x ,` con dos grupos de eventos por celda (compute / memoria + TLB) para evitar multiplexing. Los **12 cells / 24 invocaciones quedaron al 100% de cobertura** (`min_mux_pct = 100`), lo que significa que los conteos son medidos directamente sin scaling. Tres eventos del prompt original no estan expuestos en este kernel (`Linux 6.6.114.1-microsoft-standard-WSL2`, perf 6.18) y se sustituyeron con proxies documentados en [scripts/profile_perf_zen2.sh](../scripts/profile_perf_zen2.sh).

Datos crudos: [results/perf_zen2_summary.csv](../results/perf_zen2_summary.csv). Grafica: [plots/perf_zen2_breakdown.png](../plots/perf_zen2_breakdown.png).

### 9.1 Tabla resumen

| variant       | $m$  | IPC  | fp/cyc | l1d_miss | l3_miss | tlb_walk/kinst |
|---------------|-----:|-----:|-------:|---------:|--------:|---------------:|
| `naive`       | 1024 | 1.18 |   0.64 |    66.9 % | 33.2 % |        0.0208  |
| `naive`       | 4096 | 1.05 |   0.57 |    66.9 % | 96.9 % |        0.0070  |
| `naive`       | 8192 | 0.44 |   0.24 |    67.6 % | 83.2 % |        0.0084  |
| `recursive`   | 1024 | 2.54 |   1.32 |    28.7 % |  0.5 % |        0.0038  |
| `recursive`   | 4096 | 2.58 |   1.34 |    28.4 % |  0.5 % |        0.0020  |
| `recursive`   | 8192 | 2.56 |   1.33 |    28.4 % |  0.5 % |        0.0018  |
| `morton`      | 1024 | 3.97 |   0.13 |     0.4 % | 16.4 % |        0.0014  |
| `morton`      | 4096 | 3.97 |   0.13 |     0.5 % | 14.5 % |        0.0010  |
| `morton`      | 8192 | 3.97 |   0.13 |     0.7 % | 11.4 % |        0.0011  |
| `morton_avx2` | 1024 | 1.72 |   8.58 |     9.8 % |  4.2 % |        0.0224  |
| `morton_avx2` | 4096 | 2.00 |  10.30 |     9.5 % |  3.2 % |        0.0059  |
| `morton_avx2` | 8192 | 2.00 |  10.34 |     9.5 % |  3.2 % |        0.0049  |

Convenciones: `l1d_miss` = `l2_request_g1.all_no_prefetch / ls_dispatch.ld_dispatch`; `l3_miss` = `cache-misses / l2_request_g1.all_no_prefetch` (proxy de `l3_lookup_state.l3_miss`); `tlb_walk/kinst` = `bp_l1_tlb_miss_l2_tlb_miss * 1000 / instructions`. Cobertura `min_mux_pct = 100 %` en las 12 celdas.

### 9.2 Fraccion del techo FMA por variante

El techo single-core del Zen 2 son $2 \text{ FMA} \times 8 \text{ lanes} = 16$ ops/ciclo en FP32. La fraccion alcanzada en cada celda es `fp_ops_per_cycle / 16`:

| variant       | $m=1024$ | $m=4096$ | $m=8192$ | comentario |
|---------------|---------:|---------:|---------:|------------|
| `naive`       |  4.0 %   |   3.6 %  |   1.5 %  | colapso por DRAM-bound a $m$ grande |
| `recursive`   |  8.2 %   |   8.4 %  |   8.3 %  | plano: cache OK, kernel `ijk` sin SIMD |
| `morton`      |  0.8 %   |   0.8 %  |   0.8 %  | ahogado por el `morton_encode` integer |
| `morton_avx2` | 53.6 %   |  64.4 %  | **64.6 %** | mejor caso del sweep; saturando un FMA pipe |

La conclusion concuerda con la hipotesis cientifica del plan: solo el microkernel AVX2 + FMA toca una fraccion significativa del techo. **`morton_avx2` mantiene $\sim 64 \%$ del peak a $m = 8192$**, una vez que $A$ no cabe ni en L3 ni en RAM caching trivial; la combinacion de localidad espacial (Morton-de-bloques) + reuso intensivo de $C$ en registros amortiza el trafico hacia DRAM.

Caveat sobre `morton` (fino): IPC = 3.97 confirma que el procesador **no esta stalleado** en cache (l1d_miss = 0.5 %, l3_miss < 17 %), simplemente esta ejecutando casi 4 instrucciones por ciclo donde casi todas son integer (calculo de `morton_encode` por elemento). El bottleneck es CPU-bound en aritmetica entera, no memoria.

### 9.3 Cliff L3 en `naive`

`l3_miss_rate` para `naive`:

- $m = 1024$: **33.2 %** — $A$ entera ($4$ MiB) entra justo en el L3 efectivo por CCX, pero las dos $B$ y los buffers de salida la desplazan parcialmente.
- $m = 4096$: **96.9 %** — $A$ entera son $64$ MiB, $16\times$ el L3; practicamente todo lo que falla L2 termina en DRAM.
- $m = 8192$: **83.2 %** — el numero baja paradojicamente porque hay mas trafico agregado a L2 (el denominador crece), pero el efecto sobre el throughput es brutal: `fp/cyc` cae de $0.57$ ($m{=}4096$) a $0.24$ ($m{=}8192$) — un colapso de $2.4\times$.

El cliff de L3 se cruza **entre $m = 1024$ y $m = 4096$**, exactamente donde el modelo teorico de la Seccion 4 lo predijo ($m_{L3} = \sqrt{4\ \text{MiB} / 4\ \text{B}} = 1024$). La validacion experimental del cliff es directa: el ratio `cache-misses / l2_request` salta de $33 \%$ a $97 \%$ al cruzar esa frontera.

`recursive` y `morton` (fino) NO sufren este cliff: `l3_miss_rate` se mantiene en $0.5 \%$ y $11$–$16 \%$ respectivamente a traves de los tres $m$, validando la propiedad cache-oblivious de la recursion. `morton_avx2` tambien lo evita ($\sim 3 \%$ estable).

### 9.4 Reduccion de TLB walks por Morton a $m = 8192$

`bp_l1_tlb_miss_l2_tlb_miss` cuenta los TLB misses severos que requieren un **page walk completo** (decenas de ciclos cada uno). A $m = 8192$:

| variant       | walks / kinst | reduccion vs `naive` |
|---------------|--------------:|---------------------:|
| `naive`       | 0.0084 | 1.00x (baseline) |
| `recursive`   | 0.0018 | **4.67x menos** |
| `morton`      | 0.0011 | **7.64x menos** |
| `morton_avx2` | 0.0049 | 1.72x menos |

La reduccion de `morton` (fino) frente a `naive` es de **casi un orden de magnitud**, exactamente el efecto que motivo el Camino A del proyecto: el Z-order mantiene el working set agrupado en pocas paginas de 4 KiB, dramaticamente menos que el row-major naive que recorre filas enteras y atraviesa muchas paginas distintas por iteracion.

`recursive` ya logra un $4.7\times$ por la sola virtud de la recursion cache-oblivious (los bloques chicos tocan pocas paginas a la vez). `morton` agrega encima la localidad espacial 2D del Z-order.

`morton_avx2` tiene **mas TLB walks que `morton` fino** ($0.0049$ vs $0.0011$). El culpable es la **materializacion del panel `A_local`** en el leaf: para cada llamada al microkernel, el codigo copia un sub-bloque desde el layout Morton-de-bloques a un buffer row-major contiguo. Esa copia hace stride reads sobre los bloques de 4x4, lo que toca multiples paginas. El trade-off vale la pena (10x mas FLOPS), pero documenta una oportunidad de optimizacion: prefetch explicito del panel siguiente, o evitar la materializacion convirtiendo el microkernel para que consuma directamente el layout Morton.

### 9.5 Anomalia metodologica: `l2_load_hit_rate > 1` en `morton`

La celda `morton m=1024` reporta `l2_load_hit_rate = 1.167` ($> 100 \%$), lo cual es matematicamente imposible si los dos contadores midieran lo mismo. La causa: el numerador `l2_cache_req_stat.ls_rd_blk_l_hit_x` **incluye hits servidos por el hardware prefetcher de L2**, mientras que el denominador `l2_request_g1.all_no_prefetch` **excluye los prefetches**. En cargas con prefetching agresivo (como `morton` fino, que tiene patron de acceso muy regular), el numerador puede exceder al denominador.

Solo afecta a la metrica `l2_load_hit_rate` (no a las demas). Documenta que `naive` no tiene este artefacto porque su l1d_miss rate es tan alto que los prefetchers no pueden adelantarse. Las conclusiones de las Secciones 9.2–9.4 no cambian.

### 9.6 Lectura agregada

Las cuatro variantes ocupan **cuatro regimes distintos** del Roofline:

| variant       | regimen efectivo | cuello de botella dominante |
|---------------|------------------|-----------------------------|
| `naive`       | memory-bound severo | DRAM bandwidth ($l3\_miss > 80 \%$ a $m \geq 4096$) |
| `recursive`   | compute-bound sin SIMD | front-end del decoder (IPC 2.5, sin vectorizacion) |
| `morton`      | compute-bound integer | `morton_encode` por elemento (IPC 4, fp/cyc 0.13) |
| `morton_avx2` | compute-bound vectorial | saturando $\sim 1$ de los $2$ FMA pipes (fp/cyc 10.3) |

Para el reporte de cierre (Prompt 9): la **tabla 9.1 + las tres observaciones 9.2/9.3/9.4** son los hallazgos cuantitativos centrales de la sesion. La grafica `perf_zen2_breakdown.png` muestra los cuatro paneles en una sola figura.
