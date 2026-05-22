# Prompts de la Sesion 03 - Optimizacion fina de Morton para Ryzen 5 4600H

**Proyecto:** BENCHMARK_AdC - Camino A (cache-oblivious recursivo + Z-order/Morton layout)
**Responsable:** Santiago
**Contraparte en paralelo:** Juan Pablo (Camino B - Fases 3 y 4)
**Hardware objetivo:** AMD Ryzen 5 4600H (Renoir / Zen 2, 6 cores / 12 threads)
**Fecha objetivo:** Sesion 03
**Base:** `main` posterior al merge del PR de Sesion 02 (matmul_recursive + matmul_morton funcionales).

Este documento contiene los prompts en orden para alimentar a Claude Code durante la Sesion 03. La Sesion 02 dejo Morton "funcionando y validado" pero sin optimizar a nivel de microkernel. El proposito de esta sesion es **bajar al hardware especifico**: explotar AVX2 + FMA del Zen 2, tunear tamanos de bloque a los caches reales del 4600H, paralelizar con OpenMP respetando la topologia de 2 CCX, y cerrar con un Roofline ubicado en la maquina real.

> **Nota (2026-05-21): reorganizacion del arbol `src/`.** Los prompts
> y fragmentos de Makefile abajo mencionan rutas planas (`src/foo.c`,
> `src/kernel_avx2.{h,c}`) que corresponden al layout original. Tras
> el refactor del 2026-05-21 los archivos se reubicaron en
> subdirectorios funcionales y `kernel_avx2.{h,c}` se renombro a
> `kernel_avx2_morton.{h,c}` (porque ahora hay un segundo microkernel
> AVX2 `kernel_avx2_tiled.h` para la familia tiled). Para el mapeo
> completo ver el aviso al inicio de `docs/PLAN_SESION_03.md`. El
> contenido textual de los prompts se conserva tal cual por valor de
> registro historico.

---

## 0. Contexto resumido

La Sesion 02 produjo dos modulos validados:

- `matmul_recursive` (Etapa A2): recursion cache-oblivious sobre row-major, kernel base `ijk` sin optimizar.
- `matmul_morton` (Etapa A3): mismo motor recursivo pero con A reorganizada en Z-order.

Ambos comparten un `kernel_base` ingenuo: tres bucles `ijk` que se apoyan unicamente en lo que el compilador logre auto-vectorizar con `-O3`. En la practica, esto deja **mas del 90% del rendimiento teorico sobre la mesa**: el procesador puede ejecutar dos FMA de 256 bits por ciclo (16 flops/ciclo por core), y el `ijk` ingenuo apenas se acerca.

El objetivo cientifico de la Sesion 03 es **demostrar el techo de un kernel intencionalmente optimizado al hardware del 4600H** y verificar cuanto del Roofline somos capaces de tocar con Morton + AVX2 + paralelizacion.

Convenciones del proyecto que se mantienen sin excepcion:

- Tipo escalar: `scalar_t` (`float`).
- Layout: row-major, plano, alineado a 64 bytes con `posix_memalign`.
- Indices y tamanos: `size_t`.
- Codigo y comentarios en ingles; mensajes a stdout y documentos en espanol.
- Sin emojis ni caracteres no ASCII decorativos.
- Estilo de comentarios y formato segun `docs/API.md`.

---

## 1. Hardware objetivo: especificaciones reales de la maquina

Estas son las **especificaciones medidas en la maquina real** (no las del datasheet generico del 4600H). Cualquier decision de tamano de bloque, tile, threshold o threads debe derivarse de aqui.

| Recurso | Valor medido | Asociatividad / detalles | Implicacion para el codigo |
|---------|--------------|---------------------------|----------------------------|
| Modelo | AMD Ryzen 5 4600H | Renoir, Zen 2, 7 nm | `-march=znver2 -mtune=znver2` |
| Cores / threads | 6 / 12 (SMT) | Distribuidos en 2 CCX de 3 cores | OpenMP: probar 6 vs 12 threads |
| Frecuencia | 3.0 GHz base, 4.0 GHz turbo | Reduce bajo carga AVX2 sostenida | Fijar gobernador `performance` |
| L1d por core | 32 KiB | 8-way, lineas de 64 B | Hoja recursiva: working set <= 16 KiB |
| L1i por core | 32 KiB | 8-way | Microkernel debe caber en code cache |
| L2 por core | 512 KiB | 8-way, privada al core | Tiling intermedio opcional |
| L3 efectiva | **4 MiB por CCX** | Compartida entre 3 cores del mismo CCX | Cliff L3 a partir de m ~ 1024 |
| Memoria | 8 GiB (7.37 GiB usables) | DDR4-3200 dual channel | Rango factible: m hasta 2^14 = 16384 |
| SIMD | AVX2 + FMA3 | Datapath nativo de 256 bits | `__m256`, `_mm256_fmadd_ps` |
| BMI2 | Soportado pero microcodeado | PDEP/PEXT con latencia ~18 ciclos | **NO usar PDEP/PEXT (Prompt PRE)** |
| Pagina | 4 KiB | L1 DTLB 64 entries, L2 DTLB 2048 entries | Morton ayuda con TLB para m grandes |

### 1.1 Numeros derivados que se citan en los prompts

- **Pico FP32 por core**: $2 \text{ FMA} \times 8 \text{ lanes} \times 2 \text{ ops} \times 4.0 \text{ GHz} = 128$ GFLOPS.
- **Pico FP32 multi-core (6 cores activos)**: ~770 GFLOPS (en condiciones ideales; el throttling AVX bajara este numero).
- **Bandwidth DRAM teorico**: $3200 \cdot 8 \cdot 2 \approx 51.2$ GB/s. Realista (STREAM): ~35-40 GB/s.
- **Balance maquina single-core**: $128 / 40 \approx 3.2$ flops/byte. Multi-core: $770 / 40 \approx 19$ flops/byte.
- **Cliff L1d**: working set de A_block + B_block + C_block <= 32 KiB.
- **Cliff L2**: working set <= 512 KiB.
- **Cliff L3 (single-thread)**: A entera deja de caber cuando $m^2 \cdot 4 > 4 \text{ MiB}$, es decir $m > 1024$.
- **Limite RAM**: para evitar swap, A no debe exceder 2 GiB, es decir $m \leq 22000$ aprox. Practico: $m \leq 16384$.

### 1.2 Rango de m del sweep para esta sesion

Se redefine el rango respecto a Sesion 02 para alinearse al cliff L3 real y al limite de RAM:

`m_sweep = {512, 1024, 2048, 4096, 8192, 16384}`

El proyecto original menciona hasta $2^{20}$, pero con 8 GiB de RAM es inviable. Esta restriccion debe quedar documentada en `docs/PLAN_SESION_03.md`.

---

## 2. Estrategia de coordinacion con Juan Pablo

Juan Pablo en paralelo trabaja en su rama en las Fases 3 y 4 (tiling explicito + vectorizacion manual del Camino B). Las reglas son las mismas que en Sesion 02:

**Archivos que tu NO modificas en la Sesion 03:**

- `src/matmul_naive.{c,h}`, `src/bench_naive.c`, `src/validate_naive.c`.
- `src/matmul_reordered.{c,h}`, `src/matmul_tiled.{c,h}` (si Juan Pablo los introdujo).
- Cualquier `bench_*` o `validate_*` que pertenezca a las variantes de Juan Pablo.

**Archivos que tu solo extiendes (anades cosas, no cambias lo existente):**

- `Makefile`: nuevos targets al final, en bloque `# === Sesion 03 targets ===`.
- `docs/API.md`: nuevas secciones al final.
- `README.md`: extender la seccion de uso solo con tus comandos nuevos.

**Archivos que tu creas:**

- `src/hwinfo.c` (caracterizacion automatica de cache sizes).
- `src/kernel_avx2.{h,c}` (microkernel AVX2 + FMA para hojas).
- `src/matmul_morton_avx2.{h,c}` (variante de Morton que usa el microkernel).
- `src/matmul_morton_omp.{h,c}` (variante paralela con OpenMP tasks).
- `src/validate_morton_avx2.c`, `src/validate_morton_omp.c`.
- `src/bench_morton_avx2.c`, `src/bench_morton_omp.c`.
- `src/test_kernel_avx2.c` (unit test del microkernel solo).
- `scripts/audit_no_pdep.sh` (verifica que no se emite PDEP/PEXT).
- `scripts/run_threshold_sweep.sh`, `scripts/plot_threshold_sweep.py`.
- `scripts/run_sweep_morton_avx2.sh`, `scripts/run_sweep_morton_omp.sh`.
- `scripts/run_omp_scaling.sh`, `scripts/plot_omp_scaling.py`.
- `scripts/profile_perf_zen2.sh`, `scripts/plot_perf_zen2.py`.
- `scripts/measure_stream.sh`, `scripts/plot_roofline.py`.
- `docs/PLAN_SESION_03.md`, `docs/SESION_03_RESUMEN.md`.

---

## 3. Roadmap de los prompts

| # | Prompt | Salida principal |
|---|--------|------------------|
| PRE | Auditoria PDEP/PEXT del codigo Sesion 02 | `scripts/audit_no_pdep.sh` + posibles fixes |
| 0 | Bootstrap Sesion 03 | Rama, `docs/PLAN_SESION_03.md` |
| 1 | Hardware fingerprinting | `src/hwinfo.c`, `bin/hwinfo` |
| 2 | Tuning empirico de RECURSION_THRESHOLD | Sweep + grafico, threshold elegido |
| 3 | Microkernel AVX2 + FMA (Etapa A4) | `src/kernel_avx2.{h,c}` + unit test |
| 4 | Integracion del microkernel con Morton | `matmul_morton_avx2`, validado vs naive |
| 5 | Bench comparativo: naive vs morton vs morton_avx2 | CSVs y graficos GFLOPS vs m |
| 6 | Paralelizacion OpenMP (Etapa A5) | `matmul_morton_omp`, sweep de threads |
| 7 | Profiling con eventos perf Zen 2 | Tabla de FMA throughput, cache misses, TLB |
| 8 | Roofline final (STREAM + variantes) | `plots/roofline_4600h.png` |
| 9 | Documentacion final Sesion 03 | `docs/API.md` extendida, `docs/SESION_03_RESUMEN.md` |

Tiempo estimado: PRE 10-15 minutos. Prompts 0 a 9 entre 20 y 60 minutos cada uno mas tu revision intercalada. Sesion total: dia completo o dia y medio.

---

## 4. Prompts en orden

A continuacion los prompts. Cada bloque entre lineas dobles es lo que tu copias y pegas a Claude Code.

---

### Prompt PRE - Auditoria PDEP/PEXT

```text
Antes de iniciar la Sesion 03, tenemos que confirmar que el codigo de la Sesion 02
NO esta usando intrinsecos PDEP/PEXT, ni el compilador los esta emitiendo
espontaneamente. Esto es critico para AMD Zen 2 (Renoir, Ryzen 5 4600H), donde
estas instrucciones existen pero estan implementadas en microcodigo con latencia
de ~18 ciclos en lugar de los ~3 ciclos de Zen 3 e Intel Haswell+. Usarlas haria
que el calculo del indice Morton consumiera mas tiempo que el beneficio del
layout.

PASOS PRELIMINARES:

1. git checkout main && git pull
2. git checkout -b claude/audit-no-pdep

CREAR scripts/audit_no_pdep.sh con el siguiente contenido y comportamiento:

#!/usr/bin/env bash
# audit_no_pdep.sh
# Verifica que el codigo fuente no usa intrinsecos PDEP/PEXT y que el assembler
# generado para los binarios principales no contiene pdep/pext.
# Falla con exit code distinto de cero si encuentra cualquiera de los dos.

set -euo pipefail

echo "=== Auditoria PDEP/PEXT ==="

echo
echo "[1/3] Buscando intrinsecos PDEP/PEXT en codigo fuente..."
HITS_SOURCE=$(grep -rn -E "_pdep_u(32|64)|_pext_u(32|64)" src/ || true)
if [ -n "$HITS_SOURCE" ]; then
    echo "FAIL: encontrados intrinsecos PDEP/PEXT en codigo fuente:"
    echo "$HITS_SOURCE"
    exit 1
fi
echo "OK: no hay intrinsecos en src/."

echo
echo "[2/3] Compilando binarios objetivo con -S para inspeccion..."
mkdir -p build/audit
for src in src/matmul_morton.c src/morton.c; do
    base=$(basename "$src" .c)
    gcc -O3 -march=znver2 -mavx2 -mfma -mbmi -mbmi2 \
        -S -o "build/audit/${base}.s" \
        -Isrc "$src" || { echo "FAIL: no se pudo compilar $src"; exit 1; }
done
echo "OK: assembler generado en build/audit/"

echo
echo "[3/3] Buscando pdep/pext en el assembler generado..."
HITS_ASM=$(grep -in -E "\b(pdep|pext)[a-z]?\b" build/audit/*.s || true)
if [ -n "$HITS_ASM" ]; then
    echo "FAIL: el compilador emitio pdep/pext en algun modulo:"
    echo "$HITS_ASM"
    exit 1
fi
echo "OK: el compilador no emitio pdep/pext en los modulos auditados."

echo
echo "=== Auditoria PDEP/PEXT: PASS ==="

Darle permisos de ejecucion (chmod +x).

VERIFICACION:

1. Ejecutar `./scripts/audit_no_pdep.sh` desde la raiz del repo.
2. Si imprime PASS y exit code 0: documentar el resultado en el PR.
3. Si imprime FAIL en el paso 1 (intrinsecos en fuente): editar src/morton.c
   reemplazando cualquier uso de _pdep_u64 / _pdep_u32 por la implementacion
   "magic bits" (mascaras shift-or-and). La firma esperada de la implementacion
   alternativa es:

       static inline uint64_t spread_bits_32_to_64(uint32_t x) {
           uint64_t r = x;
           r = (r | (r << 16)) & 0x0000FFFF0000FFFFULL;
           r = (r | (r <<  8)) & 0x00FF00FF00FF00FFULL;
           r = (r | (r <<  4)) & 0x0F0F0F0F0F0F0F0FULL;
           r = (r | (r <<  2)) & 0x3333333333333333ULL;
           r = (r | (r <<  1)) & 0x5555555555555555ULL;
           return r;
       }

       uint64_t morton_encode(uint32_t i, uint32_t j) {
           return (spread_bits_32_to_64(i) << 1) | spread_bits_32_to_64(j);
       }

   Despues de la edicion, re-ejecutar la auditoria.

4. Si imprime FAIL en el paso 3 (assembler con pdep/pext espontaneo): es muy
   improbable, pero si ocurre, revisar los flags de compilacion: probablemente
   el compilador esta intentando optimizar algun bit-twiddle con BMI2. Solucion:
   anotar la funcion afectada con `__attribute__((target("no-bmi2")))` solo si
   esta confirmado y documentar la razon.

INTEGRAR LA AUDITORIA AL MAKEFILE:

Anadir al final del Makefile, en un bloque etiquetado:

    # === Sesion 03 targets ===
    .PHONY: audit
    audit:
    	bash scripts/audit_no_pdep.sh

Y agregar `audit` como dependencia de `all` o, mas conservador, dejarlo como
target manual y mencionarlo en README.md como paso de CI local.

CRITERIO DE ACEPTACION:

1. `scripts/audit_no_pdep.sh` existe y es ejecutable.
2. `./scripts/audit_no_pdep.sh` imprime PASS.
3. El target `make audit` funciona.
4. Si fue necesario editar morton.c, los tests previos siguen pasando:
       make test_morton && ./bin/test_morton    -> MORTON TESTS OK
       make validate_morton && ./bin/validate_morton_O0 256    -> VALIDATION OK
5. PR titulado "chore: audit PDEP/PEXT absence for Zen 2 correctness" con
   resumen de hallazgos.

UNA VEZ MERGEADO: avisame para iniciar el Prompt 0 desde el main auditado.
```

---

### Prompt 0 - Bootstrap de la Sesion 03

```text
La auditoria PDEP/PEXT esta mergeada. Inicio formal de la Sesion 03.

PASOS PRELIMINARES:

1. git fetch && git checkout main && git pull
2. Verificar que la base esta sana:
       make clean && make
       ./bin/validate_naive_O0 256       -> VALIDATION OK
       ./bin/validate_recursive_O0 256   -> VALIDATION OK
       ./bin/validate_morton_O0 256      -> VALIDATION OK
       ./scripts/audit_no_pdep.sh         -> PASS
3. Crear rama nueva: git checkout -b claude/santiago-session-03-tuning

CREAR docs/PLAN_SESION_03.md con un resumen ejecutivo (1-2 paginas) que cubra:

1. Objetivo cientifico de la Sesion 03: cuanto del Roofline del 4600H podemos
   tocar con Morton + AVX2 + paralelizacion, y donde estan los cuellos de
   botella reales.

2. Tabla de especificaciones del hardware objetivo (copiar tal cual la Tabla 1.1
   del documento PROMPTS_SESION_03.md, citando que son las medidas en la maquina
   real y no las del datasheet generico).

3. Rango de m del sweep: {512, 1024, 2048, 4096, 8192, 16384}. Justificar el
   limite superior con el tamano de A en RAM:
       m=16384 -> A = 16384^2 * 4 bytes = 1 GiB
       m=32768 -> A = 4 GiB (excede el margen practico con 8 GiB totales)

4. Cliff esperado de L3: a partir de m ~ 1024 (L3 efectiva 4 MiB por CCX).
   Hipotesis: Morton empieza a separarse de naive a partir de aqui.

5. Lista de archivos que se van a crear (los enumerados en la Seccion 2 de
   PROMPTS_SESION_03.md).

6. Lista de archivos que NO se tocan (todos los _naive, _recursive sin avx2,
   _morton sin avx2, y cualquier archivo de Juan Pablo).

7. Restricciones conocidas:
   - Morton requiere m potencia de 2 (heredado de Sesion 02).
   - Microkernel AVX2 asume FP32; el cambio a FP64 requeriria reescribirlo.
   - OpenMP tasks: en hardware Zen 2 Renoir hay 2 CCX (cada uno con 3 cores y
     4 MiB L3 privada). Threads que se separan entre CCX pierden coherencia de
     L3. Usar OMP_PROC_BIND=close en los benchmarks single-CCX.

No implementes nada de codigo todavia. Solo crea la rama y el documento de plan.

CRITERIO DE ACEPTACION:

1. Rama claude/santiago-session-03-tuning existe.
2. docs/PLAN_SESION_03.md tiene las 7 secciones.
3. No se modifico ningun otro archivo.

Cuando termines, muestra el contenido completo de docs/PLAN_SESION_03.md.
```

---

### Prompt 1 - Hardware fingerprinting

```text
Vamos a crear un binario que mida las caracteristicas del hardware en tiempo de
ejecucion. Esto sirve para dos cosas: (a) validar que las especificaciones del
PLAN_SESION_03 coinciden con la maquina real, y (b) producir un encabezado
estandar en todos los reportes y CSVs ("medido en CPU X con L1d/L2/L3 = Y/Z/W").

ARCHIVOS A CREAR:

- src/hwinfo.c (programa standalone, no se vincula a otros binarios).

FUNCIONALIDAD:

El programa imprime en stdout, en formato legible y tambien en formato CSV
(opcion --csv) los siguientes campos:

- cpu_model: leido de /proc/cpuinfo, campo "model name".
- num_cores: numero de cores fisicos (sysconf(_SC_NPROCESSORS_ONLN) y leyendo
  /proc/cpuinfo).
- num_threads: numero de threads SMT.
- l1d_kib: tamano de L1d en KiB. Leido de sysconf(_SC_LEVEL1_DCACHE_SIZE) si
  esta disponible; si no, fallback a parsear /sys/devices/system/cpu/cpu0/cache.
- l1d_assoc: asociatividad de L1d.
- l1d_line: tamano de linea L1d.
- l2_kib, l2_assoc, l2_line: idem para L2.
- l3_kib, l3_assoc, l3_line: idem para L3.
- mem_total_kib: RAM total, leido de /proc/meminfo (MemTotal).
- mem_avail_kib: RAM disponible, leido de /proc/meminfo (MemAvailable).
- avx2_supported, fma_supported, bmi2_supported: detectados con
  __builtin_cpu_supports("avx2") etc.

ADICIONAL: el programa imprime warnings explicitos si:

- L3 reportada es < 8 MiB (sugiere que es la L3 por CCX, no la total).
- BMI2 esta soportado pero el CPU no es Zen 3 o superior (vendor AMD, family <
  0x19): emitir el warning "PDEP/PEXT microcoded, do not use in hot loops".

INTEGRACION CON MAKEFILE:

Anadir al bloque "Sesion 03 targets":

    bin/hwinfo: src/hwinfo.c | bin
    	$(CC) $(CFLAGS_O3) -o $@ $<

Y un target conveniente:

    .PHONY: hwinfo
    hwinfo: bin/hwinfo
    	./bin/hwinfo

EJEMPLO DE SALIDA ESPERADA EN ESTA MAQUINA:

    === Hardware fingerprint ===
    cpu_model       : AMD Ryzen 5 4600H with Radeon Graphics
    num_cores       : 6
    num_threads     : 12
    l1d_kib         : 32   (assoc=8, line=64)
    l2_kib          : 512  (assoc=8, line=64)
    l3_kib          : 4096 (assoc=16, line=64)
    mem_total_kib   : 7733184  (~7.37 GiB)
    avx2_supported  : yes
    fma_supported   : yes
    bmi2_supported  : yes (WARNING: Zen 2 implementa PDEP/PEXT en microcodigo, evitar)
    [WARNING]: L3 reportada (4 MiB) es menor que el datasheet (8 MiB). Probable
    medicion por CCX. Para single-thread, el L3 efectivo es 4 MiB.

VERIFICACION INTERNA:

    make hwinfo
    ./bin/hwinfo
    ./bin/hwinfo --csv > results/hwinfo.csv

CRITERIO DE ACEPTACION:

1. src/hwinfo.c compila sin warnings con `-O3 -Wall -Wextra -Wpedantic`.
2. ./bin/hwinfo imprime un bloque con todos los campos en la maquina del
   profesor (espera que tenga AMD/Intel; el programa debe ser robusto a ambos).
3. ./bin/hwinfo --csv produce una linea CSV bien formada con todos los campos.
4. Los warnings aparecen cuando corresponde.
5. results/hwinfo.csv queda guardado en el repo (committed).

Cuando termines, muestra el contenido de results/hwinfo.csv y el output legible
del ./bin/hwinfo (sin --csv).
```

---

### Prompt 2 - Tuning empirico de RECURSION_THRESHOLD

```text
En la Sesion 02 elegimos RECURSION_THRESHOLD = 32 * 32 * 128 = 131072 como un
valor de manual. Ahora vamos a medirlo en la maquina real.

CONTEXTO:

El threshold determina cuando la recursion deja de subdividir y delega al kernel
base. Si es muy chico, hay overhead de recursion innecesario. Si es muy grande,
el sub-problema base no cabe en L1d y perdemos el beneficio cache-oblivious.

Para el 4600H, L1d = 32 KiB. Working set de la hoja (A_block + B_block, asumiendo
que C se mantiene en registros / streaming): 4 * (m_b * k_b + k_b * n_b) bytes.
Para n_b = 128 y m_b = k_b, queremos:

    4 * (m_b^2 + m_b * 128) <= 32768
    m_b^2 + 128 m_b <= 8192
    m_b <= ~50

Asi que el cuadrado de la hoja deberia estar en el rango m_b = k_b = 32 a 48.
Vamos a barrer thresholds y medir cual da mejor GFLOPS.

ARCHIVOS A CREAR / MODIFICAR:

- Modificar src/matmul_morton.c (o introducir una capa de indireccion): exponer
  RECURSION_THRESHOLD como variable global configurable en tiempo de ejecucion,
  no como constante de compilacion. Patron:

      // En matmul_morton.h:
      extern size_t g_recursion_threshold;
      void matmul_morton_set_threshold(size_t t);

      // En matmul_morton.c:
      size_t g_recursion_threshold = 32 * 32 * 128;
      void matmul_morton_set_threshold(size_t t) { g_recursion_threshold = t; }

  Y dentro de la recursion, usar g_recursion_threshold en lugar de la macro.

  IMPORTANTE: NO romper la firma publica de matmul_morton, y NO romper Sesion 02.
  El cambio es solo: la constante macro se convierte en variable con setter; el
  default se preserva.

- src/bench_morton.c (extender, no crear nuevo): aceptar un argumento opcional
  adicional al final `--threshold N`. Si esta presente, llamar
  matmul_morton_set_threshold(N) antes del primer bench.

- scripts/run_threshold_sweep.sh: bash script que ejecuta bench_morton_O3 para
  cada threshold en {2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M} (en
  numeros de elementos, no bytes) y para cada m en {1024, 2048, 4096}. Salida:
  results/threshold_sweep.csv con columnas threshold,m,gflops_median.

  Patron: para cada (threshold, m), 5 corridas, mediana.

- scripts/plot_threshold_sweep.py: genera plots/threshold_sweep.png con una
  linea por m, eje x = threshold (escala log), eje y = GFLOPS.

INTEGRACION MAKEFILE:

    sweep_threshold: bench_morton_O3
    	bash scripts/run_threshold_sweep.sh

    plot_threshold: results/threshold_sweep.csv
    	python3 scripts/plot_threshold_sweep.py

VERIFICACION INTERNA:

    make sweep_threshold
    make plot_threshold
    ls plots/threshold_sweep.png

CRITERIO DE ACEPTACION:

1. matmul_morton sigue pasando validate_morton_O0 256 (regression check).
2. El parametro --threshold del bench funciona.
3. El sweep genera results/threshold_sweep.csv con 30 lineas (10 thresholds * 3
   ms).
4. La grafica plots/threshold_sweep.png muestra un maximo claro (no monotono).
5. En docs/PLAN_SESION_03.md anadir una seccion al final con el resultado:
   "Threshold optimo empirico para 4600H: <valor> elementos, gflops <valor>."

NOTA: este valor empirico se usara como default en el kernel AVX2 del Prompt 3,
pero el setter se mantendra disponible para experimentacion.

Cuando termines, muestra plots/threshold_sweep.png (asciinema o describir el
shape) y el threshold elegido.
```

---

### Prompt 3 - Microkernel AVX2 + FMA (Etapa A4)

```text
Implementacion del microkernel base optimizado con AVX2 + FMA, dirigido al
Ryzen 5 4600H (Zen 2). Este es el corazon de la sesion: el codigo que
realmente toca el techo del hardware.

ARCHIVOS A CREAR:

- src/kernel_avx2.h: declaracion publica del microkernel.
- src/kernel_avx2.c: implementacion con intrinsecos.
- src/test_kernel_avx2.c: unit test standalone que verifica correctness del
  microkernel sin involucrar Morton ni recursion.

DISENIO DEL MICROKERNEL:

Geometria fija: el microkernel computa exactamente una multiplicacion
C(4 x 16) += A(4 x kc) * B(kc x 16), donde kc es parametro variable.

Por que 4 x 16:
- 16 columnas = 2 vectores __m256 (8 floats cada uno).
- 4 filas: mantienen 4 * 2 = 8 vectores __m256 para C, que caben holgados
  en los 16 registros YMM disponibles.
- Quedan 8 registros libres para A (broadcasts) y B (cargas).

Por que es importante mantener todo C en registros: durante el bucle interno
sobre kc, ningun elemento de C se escribe a memoria. Solo se hace fmadd. Esto
satura las dos unidades FMA del Zen 2.

FIRMA PUBLICA:

    // src/kernel_avx2.h
    #ifndef KERNEL_AVX2_H
    #define KERNEL_AVX2_H
    #include <stddef.h>
    #include "common.h"   // define scalar_t como float

    // Computa C[4 x 16] += A[4 x kc] * B[kc x 16].
    // A: row-major, leading dimension lda.
    // B: row-major, leading dimension ldb.
    // C: row-major, leading dimension ldc.
    // Asume kc >= 1. No verifica nada.
    // Asume lda, ldb, ldc >= 16. Asume A,B,C alineados a 32 bytes (carga unaligned
    // tambien funciona pero es mas lenta).
    void kernel_avx2_4x16(scalar_t       * restrict C, size_t ldc,
                          const scalar_t * restrict A, size_t lda,
                          const scalar_t * restrict B, size_t ldb,
                          size_t kc);

    #endif

IMPLEMENTACION:

    // src/kernel_avx2.c
    #include "kernel_avx2.h"
    #include <immintrin.h>

    void kernel_avx2_4x16(scalar_t       * restrict C, size_t ldc,
                          const scalar_t * restrict A, size_t lda,
                          const scalar_t * restrict B, size_t ldb,
                          size_t kc)
    {
        // Cargar C[4][0..15] a 8 registros (cada fila son 2 vectores).
        __m256 c00 = _mm256_loadu_ps(&C[0*ldc +  0]);
        __m256 c01 = _mm256_loadu_ps(&C[0*ldc +  8]);
        __m256 c10 = _mm256_loadu_ps(&C[1*ldc +  0]);
        __m256 c11 = _mm256_loadu_ps(&C[1*ldc +  8]);
        __m256 c20 = _mm256_loadu_ps(&C[2*ldc +  0]);
        __m256 c21 = _mm256_loadu_ps(&C[2*ldc +  8]);
        __m256 c30 = _mm256_loadu_ps(&C[3*ldc +  0]);
        __m256 c31 = _mm256_loadu_ps(&C[3*ldc +  8]);

        for (size_t p = 0; p < kc; ++p) {
            __m256 b0 = _mm256_loadu_ps(&B[p*ldb + 0]);
            __m256 b1 = _mm256_loadu_ps(&B[p*ldb + 8]);

            __m256 a0 = _mm256_broadcast_ss(&A[0*lda + p]);
            __m256 a1 = _mm256_broadcast_ss(&A[1*lda + p]);
            __m256 a2 = _mm256_broadcast_ss(&A[2*lda + p]);
            __m256 a3 = _mm256_broadcast_ss(&A[3*lda + p]);

            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);
        }

        _mm256_storeu_ps(&C[0*ldc +  0], c00);
        _mm256_storeu_ps(&C[0*ldc +  8], c01);
        _mm256_storeu_ps(&C[1*ldc +  0], c10);
        _mm256_storeu_ps(&C[1*ldc +  8], c11);
        _mm256_storeu_ps(&C[2*ldc +  0], c20);
        _mm256_storeu_ps(&C[2*ldc +  8], c21);
        _mm256_storeu_ps(&C[3*ldc +  0], c30);
        _mm256_storeu_ps(&C[3*ldc +  8], c31);
    }

UNIT TEST (src/test_kernel_avx2.c):

Prueba 1: kernel_avx2_4x16 con kc=1, C=zeros, A=identidad parcial, B=conocida.
Verifica que C resulta igual a las primeras 4 filas y 16 columnas de B.

Prueba 2: kernel_avx2_4x16 con kc=8, A y B aleatorias. Comparar con un loop de
referencia ijk en doble precision. Tolerancia: 1e-4 relativo (acumulacion en FP32
introduce error).

Prueba 3: idem con kc=128 (caso tipico de hoja).

Prueba 4: idem con kc=1024 (stress test, verifica que no se acumula error
catastrofico).

Cada prueba imprime "PASS" o "FAIL". El programa retorna 0 solo si todos pasan.

FLAGS DE COMPILACION:

El archivo src/kernel_avx2.c debe compilarse con:

    -O3 -march=znver2 -mavx2 -mfma -funroll-loops -ffast-math

Modificar el Makefile para que el target del bench AVX2 use estos flags. Aplica
solo a este archivo si es mas conveniente:

    src/kernel_avx2.o: src/kernel_avx2.c src/kernel_avx2.h
    	$(CC) -O3 -march=znver2 -mavx2 -mfma -funroll-loops -ffast-math \
    	      -Wall -Wextra -c -o $@ $<

VERIFICACION POST-COMPILE:

Confirmar que el assembler generado usa vfmadd231ps (la FMA AVX2):

    objdump -d build/kernel_avx2.o | grep vfmadd | head -20

Deberia mostrar muchas instancias.

CRITERIO DE ACEPTACION:

1. src/kernel_avx2.{h,c} y src/test_kernel_avx2.c existen.
2. Compilan limpio con los flags indicados.
3. `make test_kernel_avx2 && ./bin/test_kernel_avx2` imprime 4 PASS.
4. objdump confirma que se emitio vfmadd231ps (y NO pdep/pext).
5. ./scripts/audit_no_pdep.sh sigue pasando.

Cuando termines, muestra el output de:
    objdump -d build/kernel_avx2.o | grep -E "vfmadd|vmovups|vbroadcast" | wc -l
y el output completo de ./bin/test_kernel_avx2.
```

---

### Prompt 4 - Integracion del microkernel con Morton

```text
Crear una nueva variante de matmul_morton que use el microkernel AVX2 + FMA en
las hojas. La logica de recursion y el layout Morton de A se mantienen iguales.
Solo cambia el kernel_base.

ARCHIVOS A CREAR:

- src/matmul_morton_avx2.h
- src/matmul_morton_avx2.c
- src/validate_morton_avx2.c
- src/bench_morton_avx2.c

FIRMA PUBLICA:

    void matmul_morton_avx2(scalar_t *C,
                            const scalar_t *A_morton,
                            const scalar_t *B,
                            size_t m, size_t k, size_t n);

Identica a matmul_morton pero con el sufijo _avx2.

LOGICA:

Copia matmul_morton.c y reemplaza:

1. El kernel_base_morton (que era tres bucles ijk) se reemplaza por una funcion
   que llama al microkernel cuando el sub-problema es del tamano exacto
   4 x kc x 16, y por un loop sobre el microkernel cuando es mas grande pero
   multiplo de (4, _, 16).

   Patron:

       static void kernel_base_morton_avx2(scalar_t *C, const scalar_t *A_morton,
                                           const scalar_t *B,
                                           size_t mb, size_t kb, size_t nb,
                                           size_t ldc, size_t ldb,
                                           size_t a_morton_offset)
       {
           // Reorganizamos A_morton[a_morton_offset ...] como un panel mb x kb
           // row-major LOCAL (despues del decoding del bloque Morton).
           // Si mb es multiplo de 4 y nb es multiplo de 16, iterar:
           if ((mb % 4 == 0) && (nb % 16 == 0)) {
               for (size_t ii = 0; ii < mb; ii += 4) {
                   for (size_t jj = 0; jj < nb; jj += 16) {
                       kernel_avx2_4x16(&C[ii*ldc + jj], ldc,
                                        &A_local[ii*kb], kb,
                                        &B[jj], ldb,        // ojo: B no se subdivide
                                                            // en j si lo accedemos por
                                                            // ldb. Ajustar segun como
                                                            // el matmul_morton actual
                                                            // pasa los punteros de B.
                                        kb);
                   }
               }
           } else {
               // Fallback al kernel ijk no vectorizado para tamanos no multiplos.
               kernel_base_morton_ijk_fallback(C, A_morton, B, mb, kb, nb,
                                               ldc, ldb, a_morton_offset);
           }
       }

   IMPORTANTE: el offset de A_morton apunta al inicio de un bloque cuyo layout
   interno (despues del Morton de los bloques) es row-major LOCAL. Si en
   Sesion 02 implementaste Morton "puro" (cada elemento individual en orden
   Morton), tendras que hacer una decodificacion por elemento o reorganizar la
   estrategia. La forma estandar y mas practica es "Morton de bloques":
       - Los bloques de M_TILE x M_TILE (con M_TILE = 4) se ordenan en Z-order.
       - Dentro de cada bloque, los elementos estan en row-major puro.
   Esto simplifica el acceso del microkernel: una vez que sabes que estas en
   el bloque (i_block, j_block), su offset Morton es
   morton_encode_blocks(i_block, j_block) * M_TILE * M_TILE, y dentro del
   bloque accedes A_block[i_local * M_TILE + j_local] directamente.

   Si esto no coincide con lo que se hizo en Sesion 02, hay dos opciones:
   (a) cambiar el layout Morton actual a "Morton de bloques" (recomendado).
   (b) mantener el Morton "fino" actual y en el microkernel hacer broadcasts
       con direcciones calculadas por morton_encode en cada iteracion (NO
       recomendado: el calculo dentro del bucle anula el beneficio AVX2).

   Si hay que migrar de (b) a (a), documentarlo en docs/PLAN_SESION_03.md como
   sub-tarea adicional. La preferencia es mantener el codigo Sesion 02 intacto
   y duplicar (con la migracion de layout) en el nuevo modulo _avx2.

2. RECURSION_THRESHOLD se ajusta a un valor que garantice que la hoja sea
   multiplo de 4 x 16. Sugerencia: threshold = 64 * 64 * 128 = 524288, o el
   valor optimo encontrado en Prompt 2 redondeado al multiplo mas cercano.

3. La firma matmul_morton_avx2 deberia tener una precondicion explicita en su
   docstring: "m debe ser potencia de 2 y m >= 4; n debe ser >= 16."

VALIDATE Y BENCH:

src/validate_morton_avx2.c: copia de src/validate_morton.c con el binario
generado en bin/validate_morton_avx2_O3, usando matmul_morton_avx2 como
referencia y matmul_naive como ground truth. Tolerancia: 1e-3 relativo
(AVX2 + ffast-math + acumulacion FP32 introduce mas error que el morton
no-vectorizado).

src/bench_morton_avx2.c: copia de src/bench_morton.c, vincula
matmul_morton_avx2, genera bin/bench_morton_avx2_O3.

INTEGRACION MAKEFILE:

    # === Sesion 03: Morton AVX2 ===

    src/kernel_avx2.o: src/kernel_avx2.c src/kernel_avx2.h
    	$(CC) -O3 -march=znver2 -mavx2 -mfma -funroll-loops -ffast-math \
    	      -Wall -Wextra -c -o $@ $<

    bin/validate_morton_avx2_O3: src/validate_morton_avx2.c \
                                  src/matmul_morton_avx2.c \
                                  src/kernel_avx2.o \
                                  src/morton.o src/matrix_utils.o | bin
    	$(CC) -O3 -march=znver2 -mavx2 -mfma -Wall -Wextra \
    	      -o $@ $^ -lm

    bin/bench_morton_avx2_O3: src/bench_morton_avx2.c \
                              src/matmul_morton_avx2.c \
                              src/kernel_avx2.o \
                              src/morton.o src/matrix_utils.o | bin
    	$(CC) -O3 -march=znver2 -mavx2 -mfma -Wall -Wextra \
    	      -o $@ $^ -lm

    .PHONY: validate_morton_avx2 bench_morton_avx2
    validate_morton_avx2: bin/validate_morton_avx2_O3
    bench_morton_avx2: bin/bench_morton_avx2_O3

VERIFICACION INTERNA:

    make validate_morton_avx2
    ./bin/validate_morton_avx2_O3 256          -> VALIDATION OK
    ./bin/validate_morton_avx2_O3 1024         -> VALIDATION OK

    make bench_morton_avx2
    ./bin/bench_morton_avx2_O3 2048 3 5        -> CSV con gflops > los de
                                                  matmul_morton sin avx2

CRITERIO DE ACEPTACION:

1. Todos los validate previos siguen pasando (regresion):
       validate_naive_O0 256, validate_recursive_O0 256, validate_morton_O0 256
2. validate_morton_avx2_O3 256 imprime VALIDATION OK.
3. bench_morton_avx2_O3 1024 1 3 produce gflops al menos 4x mayor que
   bench_morton_O3 1024 1 3 (esperamos 5-10x).
4. audit_no_pdep.sh sigue pasando.

Cuando termines, muestra la linea CSV de:
    ./bin/bench_morton_O3 2048 3 5
y la de:
    ./bin/bench_morton_avx2_O3 2048 3 5
para que veamos el speedup directo del microkernel.
```

---

### Prompt 5 - Bench comparativo y graficas

```text
Construir el sweep comparativo que produce el primer entregable cientifico
serio de la sesion: una grafica con cuatro curvas (naive, recursive, morton,
morton_avx2) en funcion de m.

ARCHIVOS A CREAR:

- scripts/run_sweep_session_03.sh
- scripts/plot_sweep_session_03.py

run_sweep_session_03.sh:

#!/usr/bin/env bash
set -euo pipefail

MS=(512 1024 2048 4096 8192 16384)
WARMUP=2
RUNS=5

mkdir -p results

# Asegurarse de que el gobernador este en performance.
# (Linux solamente; no falla si no tiene permisos.)
sudo cpupower frequency-set -g performance 2>/dev/null || true

for variant in naive recursive morton morton_avx2; do
    case $variant in
        naive)        BIN=./bin/bench_naive_O3 ;;
        recursive)    BIN=./bin/bench_recursive_O3 ;;
        morton)       BIN=./bin/bench_morton_O3 ;;
        morton_avx2)  BIN=./bin/bench_morton_avx2_O3 ;;
    esac

    CSV=results/session_03_${variant}.csv
    : > "$CSV"
    echo "variant,m,warmup,runs,gflops_median,time_median_s" >> "$CSV"

    for m in "${MS[@]}"; do
        # Morton requiere potencia de 2; m=16384 puede consumir mucha RAM.
        if [ "$variant" = "morton" ] || [ "$variant" = "morton_avx2" ]; then
            # Validar potencia de 2.
            if (( m & (m - 1) )); then
                echo "Saltando $variant m=$m (no potencia de 2)"
                continue
            fi
        fi

        echo "=== $variant m=$m ==="
        "$BIN" "$m" "$WARMUP" "$RUNS" >> "$CSV" || {
            echo "Fallo $variant m=$m, continuando..."
            continue
        }
    done
done

echo "Sweep completo. CSVs en results/session_03_*.csv"

plot_sweep_session_03.py:

Lee los 4 CSVs y genera plots/session_03_gflops_vs_m.png con:

- Eje x: m (escala log2).
- Eje y: GFLOPS.
- Una curva por variante con leyenda.
- Anotaciones verticales en los cliffs:
    - m=1024 (cliff L3 efectiva, 4 MiB / sizeof(float) / m^2).
    - m=8192 (region donde TLB empieza a presionarse).
- Linea horizontal punteada con el techo computacional single-core
  (128 GFLOPS) y otra con el techo memory-bound estimado (40 GB/s * intensity).

Tambien generar plots/session_03_speedup_vs_naive.png:

- Eje x: m.
- Eje y: speedup respecto a naive (gflops_variant / gflops_naive).
- Una curva por variante (recursive, morton, morton_avx2). Naive no aparece (es
  la referencia = 1.0).

INTEGRACION MAKEFILE:

    .PHONY: sweep_session_03 plot_session_03
    sweep_session_03: bench_naive_O3 bench_recursive_O3 bench_morton_O3 bench_morton_avx2_O3
    	bash scripts/run_sweep_session_03.sh

    plot_session_03:
    	python3 scripts/plot_sweep_session_03.py

VERIFICACION INTERNA:

    make sweep_session_03
    make plot_session_03
    ls plots/session_03_*.png

Esperar tiempos de ejecucion: m=16384 puede tardar minutos en naive. Si naive
con m=16384 tarda demasiado (>5 min), permitir saltarse ese punto solo para
naive (configurable con variable NAIVE_M_MAX en el script).

ANALISIS QUE DEBE QUEDAR REGISTRADO:

En docs/PLAN_SESION_03.md anadir al final una seccion "Resultados sweep
comparativo" con:

1. Tabla de GFLOPS por (variante, m).
2. Tabla de speedup respecto a naive.
3. Observacion del cliff L3: a partir de que m las curvas naive y recursive se
   separan visiblemente.
4. Speedup de morton_avx2 vs morton: cuanto aporta solo el microkernel.

CRITERIO DE ACEPTACION:

1. Los 4 CSVs existen y tienen al menos una linea por cada m.
2. plots/session_03_gflops_vs_m.png muestra las 4 curvas, anotaciones de
   cliffs, y techos del Roofline.
3. plots/session_03_speedup_vs_naive.png muestra que morton_avx2 supera al
   resto en todos los m >= 1024.
4. La seccion "Resultados sweep comparativo" esta en PLAN_SESION_03.md.

Cuando termines, muestra las 2 imagenes (al menos describiendo su shape) y la
tabla de GFLOPS por variante.
```

---

### Prompt 6 - Paralelizacion OpenMP (Etapa A5)

```text
Variante paralela de matmul_morton_avx2 usando OpenMP tasks. Considera la
topologia de 2 CCX del Renoir: cada CCX tiene 3 cores y 4 MiB de L3 privada
al CCX. Threads en CCX distintos no comparten L3; trafico entre CCX cuesta
mucho mas que dentro del mismo CCX.

ARCHIVOS A CREAR:

- src/matmul_morton_omp.{h,c}
- src/validate_morton_omp.c
- src/bench_morton_omp.c
- scripts/run_omp_scaling.sh
- scripts/plot_omp_scaling.py

FIRMA PUBLICA:

    void matmul_morton_omp(scalar_t *C,
                           const scalar_t *A_morton,
                           const scalar_t *B,
                           size_t m, size_t k, size_t n);

ESTRATEGIA DE PARALELIZACION:

Aprovechar la estructura recursiva: las divisiones por m o por n producen
sub-problemas INDEPENDIENTES (no dependen del resultado del otro). Las
divisiones por k tienen dependencia (la segunda ACUMULA sobre C) y NO se
paralelizan a este nivel.

Patron OpenMP tasks:

    static void matmul_morton_omp_inner(...)
    {
        if (subproblema_pequeno) {
            kernel_base_morton_avx2(...);
            return;
        }
        if (dividir_por_m) {
            #pragma omp task firstprivate(...) shared(C, A_morton, B)
            matmul_morton_omp_inner(C_top,    A_top,    B, ...);
            #pragma omp task firstprivate(...) shared(C, A_morton, B)
            matmul_morton_omp_inner(C_bottom, A_bottom, B, ...);
            #pragma omp taskwait
        } else if (dividir_por_n) {
            // analogo
        } else {  // dividir por k
            matmul_morton_omp_inner(C, A_left,  B_top,    ...);   // escribe
            matmul_morton_omp_inner_add(C, A_right, B_bottom, ...); // acumula
        }
    }

    void matmul_morton_omp(...)
    {
        #pragma omp parallel
        {
            #pragma omp single
            matmul_morton_omp_inner(...);
        }
    }

UMBRAL DE PARALELIZACION:

Crear tasks indefinidamente genera overhead. Tipico: parar de crear tasks
cuando el sub-problema es chico y delegar al kernel secuencial. Threshold
sugerido: 64 * 64 * 128 = 524288 elementos (mas grande que el threshold de
hojas, para evitar enxambres de tasks). Exponer como variable global con
setter, igual que en Prompt 2.

COMPILACION CON OPENMP:

Agregar -fopenmp a CFLAGS de los targets relevantes:

    bin/bench_morton_omp_O3: src/bench_morton_omp.c \
                             src/matmul_morton_omp.c \
                             src/kernel_avx2.o \
                             src/morton.o src/matrix_utils.o | bin
    	$(CC) -O3 -march=znver2 -mavx2 -mfma -fopenmp -Wall -Wextra \
    	      -o $@ $^ -lm

run_omp_scaling.sh:

Ejecuta bench_morton_omp_O3 con OMP_NUM_THREADS in {1, 2, 3, 4, 6, 8, 12} para
m in {4096, 8192}. Tambien con dos configuraciones de afinidad:

    OMP_PLACES=cores  OMP_PROC_BIND=close   (mismo CCX prioritario)
    OMP_PLACES=cores  OMP_PROC_BIND=spread  (distribuir entre CCX)

Salida: results/omp_scaling.csv con columnas (m, threads, bind, gflops_median).

plot_omp_scaling.py:

Genera plots/omp_scaling.png con:

- Eje x: numero de threads.
- Eje y: speedup respecto a 1 thread.
- Una serie por (m, bind), idealmente 4 curvas: m=4096 close, m=4096 spread,
  m=8192 close, m=8192 spread.
- Linea diagonal ideal (speedup = threads) como referencia.

INTERPRETACION ESPERADA:

- Con close: speedup razonable hasta 3 threads (un CCX), luego baja la
  eficiencia. Saturacion en 6 cores: ~4-5x.
- Con spread: deberia escalar mejor hasta 6 cores pero con mas overhead de
  sincronizacion.
- SMT (12 threads): probablemente NO mejora, las dos unidades FMA por core ya
  estan saturadas.

VERIFICACION INTERNA:

    make validate_morton_omp
    OMP_NUM_THREADS=4 ./bin/validate_morton_omp_O3 256    -> VALIDATION OK

    make bench_morton_omp
    OMP_NUM_THREADS=6 OMP_PROC_BIND=close \
        ./bin/bench_morton_omp_O3 4096 1 3                -> gflops alto

    make sweep_omp_scaling
    make plot_omp_scaling

CRITERIO DE ACEPTACION:

1. validate_morton_omp con OMP_NUM_THREADS=1, 4 y 12 imprime VALIDATION OK
   (sin condiciones de carrera).
2. bench_morton_omp_O3 con 6 threads supera a bench_morton_avx2_O3 con factor
   3-5x.
3. plots/omp_scaling.png muestra la curva de speedup.
4. audit_no_pdep.sh sigue pasando.

Cuando termines, muestra la tabla resumen:
    m, threads, bind, gflops, speedup_vs_1t

para m=8192 y bind=close.
```

---

### Prompt 7 - Profiling con eventos perf de Zen 2

```text
Capturar metricas de hardware especificas del Zen 2 para validar las hipotesis
del proyecto: cliff L3, presion TLB, utilizacion de FMA. Estos numeros
alimentan la discusion del reporte.

ARCHIVOS A CREAR:

- scripts/profile_perf_zen2.sh
- scripts/plot_perf_zen2.py

EVENTOS DE INTERES (Zen 2 / Family 17h):

| Evento | Que mide | Por que importa |
|--------|----------|------------------|
| cycles | Ciclos totales | Denominador de IPC |
| instructions | Instrucciones retiradas | IPC = inst / cycles |
| fp_ret_sse_avx_ops.all | FLOPs vectoriales retirados | Numerador del Roofline real |
| ls_dispatch.ld_dispatch | Loads | Trafico de memoria |
| l1_data_cache_fills_all | Fills de L1d | Misses de L1d |
| l2_request_g1.all_no_prefetch | Requests a L2 sin prefetch | Misses reales de L1d |
| l2_cache_req_stat.ls_rd_blk_l_hit_x | Hits en L2 | Eficacia del tiling L1 |
| l3_lookup_state.l3_miss | Misses de L3 | Cuando A no cabe en L3 |
| bp_l1_tlb_miss_l2_tlb_hit | TLB misses servidos en L2 | Presion TLB intermedia |
| bp_l1_tlb_miss_l2_tlb_miss | TLB misses que llegan a page walk | Presion TLB severa |

Estos nombres son los de perf en Linux para Zen 2. Si alguno no esta disponible,
documentar en el script y usar el alternativo mas cercano.

profile_perf_zen2.sh:

#!/usr/bin/env bash
set -euo pipefail

VARIANT=${1:-morton_avx2}
M=${2:-4096}
BIN=./bin/bench_${VARIANT}_O3

EVENTS="cycles,instructions,fp_ret_sse_avx_ops.all,\
ls_dispatch.ld_dispatch,l1_data_cache_fills_all,\
l2_request_g1.all_no_prefetch,l3_lookup_state.l3_miss,\
bp_l1_tlb_miss_l2_tlb_hit,bp_l1_tlb_miss_l2_tlb_miss"

mkdir -p results

OUTPUT=results/perf_${VARIANT}_m${M}.txt

# Run con perf stat. Usar -x , para CSV-friendly output.
perf stat -e "$EVENTS" -o "$OUTPUT" -- "$BIN" "$M" 1 3 2>&1 | tail -20

echo "Resultado en $OUTPUT"

EJECUTAR PARA CADA COMBINACION RELEVANTE:

for variant in naive recursive morton morton_avx2; do
    for m in 1024 4096 8192; do
        ./scripts/profile_perf_zen2.sh $variant $m
    done
done

(Para morton_omp, perf por defecto suma sobre todos los threads; OK.)

CONSOLIDAR EN UN CSV:

scripts/consolidate_perf_zen2.py (Python): parsea los archivos
results/perf_*.txt y produce results/perf_zen2_summary.csv con columnas:

    variant, m, cycles, instructions, ipc,
    fp_ops, fp_ops_per_cycle,
    l1d_miss_rate, l2_miss_rate, l3_miss_rate,
    tlb_walk_per_kinst

Donde:
- ipc = instructions / cycles.
- fp_ops_per_cycle = fp_ret_sse_avx_ops.all / cycles.
- l1d_miss_rate = l2_request_g1.all_no_prefetch / ls_dispatch.ld_dispatch.
- l3_miss_rate = l3_lookup_state.l3_miss / l2_request_g1.all_no_prefetch.
- tlb_walk_per_kinst = bp_l1_tlb_miss_l2_tlb_miss / (instructions / 1000).

plot_perf_zen2.py: genera plots/perf_zen2_breakdown.png con varias subplots:

- (a) IPC por (variante, m).
- (b) FP ops por ciclo por (variante, m). Linea horizontal a 16 (techo Zen 2:
  2 FMA x 8 lanes = 16 flops/ciclo en FP32).
- (c) L3 miss rate vs m, una curva por variante. Espero que naive y recursive
  diverjan despues de m=1024.
- (d) TLB walks por mil instrucciones, una curva por variante. Espero que
  morton este claramente debajo de naive para m grande.

INTERPRETACION QUE DEBE QUEDAR EN PLAN_SESION_03.md:

Anadir seccion "Resultados perf Zen 2" con:

1. Que fraccion del techo FMA toca cada variante (fp_ops_per_cycle / 16).
2. A partir de que m los misses de L3 empiezan a explotar para naive.
3. Cuanto reduce Morton los TLB walks para m=8192.

CRITERIO DE ACEPTACION:

1. profile_perf_zen2.sh se ejecuta sin errores para todas las combinaciones.
2. results/perf_zen2_summary.csv existe con 12 filas (4 variantes x 3 ms).
3. plots/perf_zen2_breakdown.png tiene los 4 subplots.
4. La seccion "Resultados perf Zen 2" esta en PLAN_SESION_03.md.

NOTAS:

- Si el sistema bloquea perf por seguridad (paranoid level), tendras que pedirme
  ejecutar uno de estos antes:
      sudo sysctl -w kernel.perf_event_paranoid=1
  o agregar el usuario al grupo perf (mas permanente). Documentar en
  README.md la dependencia.
- Algunos eventos pueden no estar disponibles en una version vieja del kernel.
  Si perf reporta "<not supported>" para alguno, sustituir por el evento
  generico (cache-misses, dTLB-load-misses, etc.) y documentar el cambio.

Cuando termines, muestra results/perf_zen2_summary.csv completo y describe lo
que se ve en plots/perf_zen2_breakdown.png.
```

---

### Prompt 8 - Roofline final

```text
Construir el diagrama Roofline definitivo de la Sesion 03, anclado en mediciones
reales de bandwidth (STREAM) y de FLOPs (perf), no en numeros teoricos.

ARCHIVOS A CREAR:

- scripts/measure_stream.sh
- scripts/measure_stream.py (alternativa puramente en Python si no quieres
  compilar STREAM)
- scripts/plot_roofline.py

MEDICION DE BANDWIDTH (STREAM):

Opcion A (recomendada): compilar el benchmark STREAM original de McCalpin.

scripts/measure_stream.sh:

#!/usr/bin/env bash
set -euo pipefail

mkdir -p build/stream
cd build/stream
if [ ! -f stream.c ]; then
    wget -O stream.c https://www.cs.virginia.edu/stream/FTP/Code/stream.c
fi

# STREAM_ARRAY_SIZE: debe ser al menos 4x mas grande que L3 para no caber.
# L3 = 4 MiB = 1M doubles = 1M floats. 4x = 4M floats = 16M bytes.
# Mas conservador: 64M floats = 256 MiB de working set total (3 arrays).
gcc -O3 -fopenmp -march=znver2 -mavx2 -mfma \
    -DSTREAM_ARRAY_SIZE=64000000 \
    -DNTIMES=20 \
    stream.c -o stream

# Single-thread
OMP_NUM_THREADS=1 ./stream > ../../results/stream_1t.txt

# Multi-thread (saturar la jerarquia de memoria con 6 cores)
OMP_NUM_THREADS=6 OMP_PROC_BIND=spread ./stream > ../../results/stream_6t.txt

echo "Stream completo. Reportes en results/stream_*.txt"
echo "==== Single thread ===="
grep -E "(Copy|Scale|Add|Triad)" ../../results/stream_1t.txt
echo "==== 6 threads ===="
grep -E "(Copy|Scale|Add|Triad)" ../../results/stream_6t.txt

Opcion B: implementar un STREAM minimo en Python con numpy (suficiente para
estimacion gruesa). NO recomendado para el reporte final porque numpy llama
internamente a BLAS y la medicion no es comparable.

INTEGRAR AL MAKEFILE:

    .PHONY: stream
    stream:
    	bash scripts/measure_stream.sh

CONSTRUIR EL ROOFLINE:

plot_roofline.py:

Lee:
- results/stream_1t.txt y stream_6t.txt para el bandwidth medido.
- results/perf_zen2_summary.csv para el fp_ops_per_cycle y los flops totales por
  variante.

Para cada variante, calcula:

- arithmetic_intensity = total_flops / total_bytes_moved
  donde total_bytes_moved = (l1_data_cache_fills_all + similar para L2/L3)
  o mejor, usar la formula simplificada:
      bytes = 4 * (m^2 + 2 * m * n) por iteracion (lectura de A + dos buffers
      de B + escritura)
- gflops_alcanzados = fp_ret_sse_avx_ops.all / tiempo

Grafica:

- Eje x: arithmetic intensity (flops/byte), escala log.
- Eje y: GFLOPS, escala log.
- Linea horizontal en pico FMA medido (esperamos ~700 GFLOPS multi-core).
- Linea diagonal con pendiente = bandwidth medido (STREAM Triad) hasta el knee.
- Puntos para cada variante en (intensity, gflops) con colores distintos.
- Variante OpenMP en multi-core, otras en single-core.

Anotar el knee (ridge point) del Roofline:
    ridge_intensity = peak_gflops / peak_bandwidth

Para el 4600H (estimacion): ridge ~ 770 / 40 = 19 flops/byte (multi-core),
~ 128 / 40 = 3.2 flops/byte (single-core). Matmul tiene intensidad >> 19 cuando
esta bien bloqueado, asi que esperamos estar en la region compute-bound (encima
del knee).

INTERPRETACION QUE DEBE QUEDAR EN PLAN_SESION_03.md:

Anadir seccion "Roofline final" con:

1. Bandwidth STREAM medido (Triad single-thread y 6-thread).
2. Pico FMA observado (mejor variante).
3. Posicion de cada variante en el Roofline: cuanto del techo computacional
   tocan en su mejor m.
4. Si alguna variante esta en la region memory-bound, identificar por que (
   probablemente naive y recursive para m grande).

CRITERIO DE ACEPTACION:

1. results/stream_1t.txt y stream_6t.txt existen con numeros razonables
   (Triad 1t: 8-15 GB/s; Triad 6t: 25-40 GB/s).
2. plots/roofline_4600h.png muestra el Roofline con todas las variantes.
3. La seccion "Roofline final" esta en PLAN_SESION_03.md.

Cuando termines, muestra el Triad bandwidth (1t y 6t) y describe la posicion
de morton_omp en el Roofline (compute-bound o memory-bound, fraccion del
techo).
```

---

### Prompt 9 - Documentacion final de la Sesion 03

```text
Cerrar la Sesion 03 con la documentacion completa y abrir el PR a main.

ARCHIVOS A ACTUALIZAR / CREAR:

1. docs/API.md (extender, NO reescribir): anadir tres secciones nuevas en orden:

   ### kernel_avx2

   - Firma: void kernel_avx2_4x16(...).
   - Precondiciones (alineacion, leading dimensions, kc).
   - Comportamiento: C += A * B sobre un tile fijo 4x16.
   - Performance esperada: ~120 GFLOPS single-core en hojas que caben en L1d.

   ### matmul_morton_avx2

   - Firma: void matmul_morton_avx2(...).
   - Precondiciones (m potencia de 2, m >= 4, n multiplo de 16).
   - Notas sobre el threshold de hoja y como ajustarlo via
     matmul_morton_avx2_set_threshold().

   ### matmul_morton_omp

   - Firma: void matmul_morton_omp(...).
   - Variables de entorno relevantes: OMP_NUM_THREADS, OMP_PROC_BIND.
   - Recomendacion para 4600H: OMP_NUM_THREADS=6 OMP_PROC_BIND=close.

2. docs/SESION_03_RESUMEN.md (crear, analogo a SESION_02_RESUMEN.md): secciones:

   1. Contexto del proyecto (breve).
   2. Decisiones tomadas en la Sesion 03 (tabla):
      - Auditoria PDEP/PEXT como primer paso.
      - L3 efectiva 4 MiB asumida en lugar de 8 MiB del datasheet.
      - Rango de m limitado a {512..16384} por RAM 8 GiB.
      - Microkernel 4x16 con 8 acumuladores.
      - Morton "de bloques" si se migro desde Morton fino.
      - OpenMP tasks con OMP_PROC_BIND=close.
   3. Estructura de archivos nuevos (lista).
   4. Que hacen los componentes clave (5-10 lineas cada uno):
      - hwinfo, audit_no_pdep, kernel_avx2, matmul_morton_avx2,
        matmul_morton_omp, scripts de profiling.
   5. Resultados experimentales:
      - Tabla de GFLOPS por (variante, m).
      - Speedup acumulado vs naive.
      - Fraccion del Roofline alcanzada.
      - Eficiencia de OpenMP (speedup vs ideal a 6 threads).
   6. Verificaciones realizadas (lista exhaustiva: validate_* OK, audit OK,
      regression vs Sesion 02 OK).
   7. Commits y PRs (lista).
   8. Como reanudar la sesion (rama, comandos clave).
   9. Que sigue (Sesion 04): opciones
      - Comparacion final con OpenBLAS (Fase 5 del plan original).
      - Reporte y presentacion.
      - Mejoras opcionales: streaming stores, prefetching manual,
        comparacion con BLIS y MKL si se consigue.
   10. Riesgos conocidos para la Sesion 04.
   + Apendice: comandos rapidos para reproducir todos los resultados.

3. README.md: extender la seccion "Como usar" con:

   - make audit                          # verifica que no se emite PDEP/PEXT
   - make hwinfo                         # imprime caracteristicas del CPU
   - make sweep_threshold                # mide threshold optimo
   - make validate_morton_avx2           # valida la variante AVX2
   - make bench_morton_avx2              # ejecuta el bench AVX2
   - OMP_NUM_THREADS=6 make bench_morton_omp  # version paralela
   - make sweep_session_03               # sweep comparativo de las 4 variantes
   - make stream                         # mide bandwidth con STREAM
   - make profile_zen2                   # captura eventos perf Zen 2
   - make plot_roofline                  # genera el Roofline final

COMMITS:

Si los commits incrementales no se hicieron por modulo, hacer un commit final
agrupando lo no commiteado:

    feat: Session 03 - AVX2 microkernel + OpenMP + Roofline for Ryzen 5 4600H

    Adds:
    - kernel_avx2_4x16 (Stage A4, AVX2+FMA microkernel for 4x16 tile)
    - matmul_morton_avx2 wiring the microkernel into the Morton recursion
    - matmul_morton_omp parallel variant with OpenMP tasks (Stage A5)
    - hwinfo for runtime cache fingerprinting
    - PDEP/PEXT audit script (Zen 2 correctness)
    - perf-based profiling for Zen 2 events
    - STREAM-anchored Roofline visualization

    Results on AMD Ryzen 5 4600H (Renoir, Zen 2, 6c/12t, L3 4MiB/CCX, 8 GiB RAM):
    - Single-core morton_avx2: <X> GFLOPS at m=4096 (<Y>% of FMA peak)
    - 6-thread morton_omp: <X> GFLOPS at m=8192 (<Y>x speedup vs 1 thread)
    - Cliff L3 confirmed empirically at m ~ 1024.

PR:

Titulo: "feat: Sesion 03 - microkernel AVX2 + OpenMP + Roofline anclado al 4600H"

Cuerpo:
- Que se anadio (lista de archivos).
- Resultados experimentales (tabla de GFLOPS).
- Confirmacion de los criterios de aceptacion de cada prompt anterior.
- Coordinacion con el avance de Juan Pablo (sin cruces de archivos).
- Como validar localmente (lista de comandos).

CRITERIO DE ACEPTACION FINAL:

1. docs/API.md tiene las tres secciones nuevas, sin tocar las viejas.
2. docs/SESION_03_RESUMEN.md cubre las 10 secciones + apendice de comandos.
3. README.md tiene los nuevos make targets documentados.
4. Todos los validates pasan (regresion completa):
       ./bin/validate_naive_O0 256
       ./bin/validate_recursive_O0 256
       ./bin/validate_morton_O0 256
       ./bin/validate_morton_avx2_O3 256
       OMP_NUM_THREADS=4 ./bin/validate_morton_omp_O3 256
5. ./scripts/audit_no_pdep.sh imprime PASS.
6. PR creado y URL impreso.

Cuando termines:
- Tabla de contenidos de docs/SESION_03_RESUMEN.md.
- Diff de README.md.
- URL del PR.
- Resumen de una linea con la mejor GFLOPS alcanzada y a que fraccion del
  Roofline corresponde.
```

---

## 5. Apendice A: rubrica resumida de validacion incremental

| Despues de | Comando | Salida esperada |
|------------|---------|-----------------|
| PRE | `./scripts/audit_no_pdep.sh` | PASS |
| PRE | `make audit` (alias) | PASS |
| 0 | inspeccion de docs/PLAN_SESION_03.md | 7 secciones presentes |
| 1 | `make hwinfo && ./bin/hwinfo` | bloque con L1d=32, L2=512, L3=4096 KiB |
| 1 | `./bin/hwinfo --csv > results/hwinfo.csv` | CSV con todos los campos |
| 2 | `make sweep_threshold && make plot_threshold` | plots/threshold_sweep.png con maximo claro |
| 3 | `make test_kernel_avx2 && ./bin/test_kernel_avx2` | 4 PASS |
| 3 | `objdump -d build/kernel_avx2.o \| grep vfmadd \| wc -l` | > 8 (FMAs emitidas) |
| 4 | `./bin/validate_morton_avx2_O3 256` | VALIDATION OK |
| 4 | `./bin/bench_morton_avx2_O3 2048 3 5` | gflops 4x+ que morton no-avx2 |
| 5 | `make sweep_session_03 && make plot_session_03` | 4 CSVs + 2 PNG |
| 6 | `OMP_NUM_THREADS=6 ./bin/validate_morton_omp_O3 256` | VALIDATION OK |
| 6 | `make sweep_omp_scaling && make plot_omp_scaling` | plots/omp_scaling.png |
| 7 | `make profile_zen2 && make plot_perf_zen2` | results/perf_zen2_summary.csv + plots |
| 8 | `make stream` | Triad 1t en 8-15 GB/s, 6t en 25-40 GB/s |
| 8 | `make plot_roofline` | plots/roofline_4600h.png con knees y puntos |
| 9 | inspeccion de docs/ | API.md + SESION_03_RESUMEN.md completos |

---

## 6. Apendice B: heuristicas de debugging para puntos calientes

**Prompt PRE (auditoria) - sintomas tipicos:**

- audit_no_pdep.sh falla con HITS_SOURCE no vacio: alguien (probablemente la
  Sesion 02) uso `_pdep_u64` en src/morton.c. Reemplazar por la implementacion
  "magic bits" mostrada en el prompt.
- audit_no_pdep.sh falla con HITS_ASM no vacio: el compilador emitio PDEP por
  su cuenta. Improbable; investigar el bucle donde aparece y posiblemente
  desactivar BMI2 solo en esa funcion.

**Prompt 2 (threshold sweep) - sintomas:**

- La grafica es plana (sin maximo): los warmups no son suficientes o el sweep
  esta dominado por overhead de I/O. Aumentar WARMUP y RUNS.
- El maximo esta en el extremo izquierdo (threshold muy chico): la recursion
  esta llamando muy rapido al kernel base, el cual es ijk no vectorizado y poco
  eficiente. Recordar que el optimo cambiara DESPUES del Prompt 3 (microkernel
  AVX2). Repetir el sweep al final si hay tiempo.

**Prompt 3 (microkernel AVX2) - bugs tipicos:**

- vfmadd se emite pero el GFLOPS sigue bajo: comprobar que A se accede por
  `&A[fila*lda + p]`, no `&A[fila + p*lda]` (transpuesta accidental). En el
  layout esperado, A es row-major y `lda` es el numero de columnas, NO de
  filas.
- Error de validacion 1e-2 o peor: muy probablemente kc se interpreto mal
  (extra/missing iteration). Imprimir el primer elemento de C antes y despues
  del kernel para kc=1.
- Crash con SIGSEGV: alineacion. Aunque uso loadu (unaligned), si el puntero
  esta cerca del final de una pagina mapeada, falla. Verificar con
  `posix_memalign` que C/A/B esten alineados a 32 bytes minimo.

**Prompt 4 (integracion Morton + microkernel) - bugs comunes:**

- Validacion falla solo para m grande: probablemente la migracion de Morton
  "fino" a Morton "de bloques" se hizo a medias. Comparar el offset Morton
  calculado para un bloque conocido entre Sesion 02 y Sesion 03.
- GFLOPS solo 1.5x mejor que el morton sin avx2 (esperabamos 5-10x): el
  microkernel no se esta usando en realidad. Verificar con perf que
  fp_ret_sse_avx_ops.all es alto y que vfmadd aparece en el assembler del
  binario final (no solo en kernel_avx2.o).

**Prompt 6 (OpenMP) - bugs comunes:**

- Validacion falla solo con threads > 1: race condition. Lo mas probable es
  que dos tasks escriban en el mismo sub-bloque de C porque el caso de
  division por k se paralelizo accidentalmente. Verificar que el caso k es
  secuencial.
- Speedup negativo o cercano a 1: overhead de tasks. Subir el threshold de
  paralelizacion. Tambien revisar OMP_PROC_BIND.
- Speedup ideal hasta 3 threads y luego se aplana: con OMP_PROC_BIND=close,
  los threads quedan en el mismo CCX y el segundo CCX no se usa. Probar con
  bind=spread y comparar.

**Prompt 7 (perf Zen 2) - eventos no disponibles:**

- "<not supported>" en algun evento: el kernel de Linux puede no tener el
  mapping correcto para Zen 2. Eventos equivalentes genericos:
      fp_ret_sse_avx_ops.all -> r5300c7 (raw event hex; vale solo para Zen 2)
                                o cancelacion del evento si no funciona.
      l3_lookup_state.l3_miss -> LLC-load-misses (mas grueso pero portable).
      bp_l1_tlb_miss_l2_tlb_miss -> dTLB-load-misses (idem).
- perf bloqueado: `sudo sysctl -w kernel.perf_event_paranoid=1` o agregar
  CAP_PERFMON a Claude (no aplica) / al usuario.

**Prompt 8 (Roofline) - numeros raros:**

- Triad 1t reporta 2-3 GB/s: STREAM_ARRAY_SIZE muy chico, el array esta
  cabiendo en L2. Subir a 64M floats minimo.
- Triad 6t no supera a 1t: NUMA o afinidad mala. Probar OMP_PROC_BIND=spread y
  OMP_PLACES=cores explicito.
- Las variantes caen en region memory-bound siempre: probablemente la formula
  de bytes movidos es muy pesimista. Usar la formula DRAM_bytes_moved
  derivada de l3_lookup_state.l3_miss * 64 (bytes por linea), no el total
  teorico.

---

## 7. Apendice C: tabla de referencia de eventos perf en Zen 2

| Nombre en perf | Codigo raw | Que es |
|----------------|------------|--------|
| cycles | (generico) | Ciclos no-halt |
| instructions | (generico) | Instrucciones retiradas |
| fp_ret_sse_avx_ops.all | r5300c7 | FLOPs SSE/AVX retirados (todos los tipos) |
| fp_ret_sse_avx_ops.mac_flops | r5304c7 | Solo FMA mac flops |
| ls_dispatch.ld_dispatch | r5301a9 | Despachos de load |
| ls_dispatch.st_dispatch | r5302a9 | Despachos de store |
| l1_data_cache_fills_all | r530143 | Fills a L1d desde todos los origenes |
| l2_request_g1.all_no_prefetch | r53fc60 | Requests a L2 (no prefetch) |
| l2_cache_req_stat.ls_rd_blk_l_hit_x | r530864 | Hits en L2 de load reads |
| l3_lookup_state.l3_miss | r530f04 | Misses de L3 |
| bp_l1_tlb_miss_l2_tlb_hit | r530245 | TLB miss en L1 servido por L2 |
| bp_l1_tlb_miss_l2_tlb_miss | r530145 | TLB miss en ambos niveles -> page walk |
| de_dis_uops_from_decoder.opcache_dispatched | r53aa | Uops desde op cache |
| de_dis_dispatch_token_stalls.fp_sch_rsrc_stall | r5304af | Stalls FP por scheduler lleno |

Para usar estos eventos en perf:

    perf stat -e cycles,instructions,r5300c7,r53fc60,r530f04 ./bin/...

Si perf no acepta los nombres simbolicos, los codigos raw siempre funcionan en
Family 17h (Zen 2). Documentar en el script si caes a raw.

---

## 8. Apendice D: orden cronologico recomendado de los prompts

Para mantener una sesion productiva sin abrumarse:

**Dia 1 (manana):** PRE, 0, 1, 2. Estos son rapidos (15-30 min cada uno) y dejan
la base para el trabajo pesado.

**Dia 1 (tarde):** 3 y 4. Aqui esta la mitad cientifica de la sesion. El
microkernel AVX2 es el codigo mas delicado de toda la sesion: si no funciona,
el resto se queda sin sustento.

**Dia 2 (manana):** 5 y 6. Bench comparativo + OpenMP. Una vez funcional, el
sweep tarda solo (es maquina la que trabaja).

**Dia 2 (tarde):** 7, 8, 9. Profiling, Roofline y documentacion. Cierre.

Si el tiempo aprieta, los prompts opcionales son 6 (OpenMP) y 7 (perf
detallado). El nucleo cientifico de la sesion son 3, 4, 5, 8.

---

*Documento generado al inicio de la Sesion 03. Edita conforme avances o si
surgen ajustes a los prompts.*
