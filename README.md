# Benchmark de Multiplicacion Iterada de Matrices (`main_server`)

**Curso:** Arquitectura de Computadores
**Universidad:** Universidad Nacional de Colombia, Sede Medellin
**Autores:** Santiago Uribe Echavarría - Juan Pablo Robledo Meza
**Fecha:** 29 Mayo 2026
**Hardware de referencia:** AMD EPYC 9R45 (Zen 5) en AWS c8a.2xlarge
**ISA SIMD principal:** AVX-512 (F + VL + BW + DQ + IFMA)

Esta rama porta el benchmark al servidor de produccion. Implementa y mide la recurrencia $B_{i+1} = A \cdot B_i$ sobre ocho kernels distintos (baseline, reordenamientos, tiling, vectorizacion AVX-512 y paralelizacion OpenMP) con un pipeline de profiling con `perf` calibrado para el hardware Zen 5.

El contrato publico de las funciones esta en [`docs/API.md`](docs/API.md). La rama paralela `main` contiene la version Zen 2 / AVX2 del proyecto, pensada para desarrollo local en WSL2 sobre un Ryzen 5 4600H; ambas comparten estructura, API y pipeline de validacion.

---

## 0. Tabla de contenidos

1. [Que hace este proyecto](#1-que-hace-este-proyecto)
2. [Hardware de referencia](#2-hardware-de-referencia)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Setup en Amazon Linux](#4-setup-en-amazon-linux)
5. [Quickstart](#5-quickstart)
6. [Compilar](#6-compilar)
7. [Validar correctitud](#7-validar-correctitud)
8. [Ejecutar benchmarks](#8-ejecutar-benchmarks)
9. [Profiling con `perf`](#9-profiling-con-perf)
10. [Reproducibilidad y estabilidad](#10-reproducibilidad-y-estabilidad)
11. [Adaptar el proyecto a otro hardware o microkernel](#11-adaptar-el-proyecto-a-otro-hardware-o-microkernel)
12. [Documentacion adicional](#12-documentacion-adicional)

---

## 1. Que hace este proyecto

Mide la recurrencia

$$
B_{i+1} = A \cdot B_i, \quad i = 0, 1, \ldots, I-1, \quad B_0 = Z
$$

con $A \in \mathbb{R}^{m \times m}$, $Z \in \mathbb{R}^{m \times 128}$, $I = 2m/n$. Cada iteracion guarda las primeras $n$ filas del producto en un buffer de salida.

El objetivo del proyecto no es la matmul en si, sino **medir el efecto incremental** de cada tecnica de optimizacion (orden de bucles, tiling, register-blocking, SIMD, OpenMP) sobre un baseline ingenuo. Los ocho kernels comparten la misma firma publica (Seccion 1.5 de `docs/API.md`) y se evaluan con el mismo pipeline de hardware counters.

La especializacion de `main_server` respecto a `main` es exclusivamente la microarquitectura objetivo (Zen 5 / AVX-512 vs Zen 2 / AVX2) y los hiperparametros que dependen de ella (tamanos de tile, thresholds de recursion, default OMP). Todo lo demas (estructura del repo, API, validacion, pipeline perf) es identico.

---

## 2. Hardware de referencia

Todas las decisiones de hiperparametros del proyecto (tamanos de tile, configuracion OMP, threshold de recursion) se eligieron para el EPYC 9R45 en AWS c8a.2xlarge:

| Recurso | Valor |
|---|---|
| Microarquitectura | Zen 5 (Genoa-X family) |
| vCPUs visibles | $8$ (1 thread por core; SMT deshabilitado por el hipervisor) |
| Topologia NUMA | $1$ unico nodo |
| L1d | $48$ KiB por core |
| L2 | $1$ MiB por core |
| L3 | $32$ MiB **compartido** por los $8$ cores |
| ISA SIMD | AVX-512F + VL + BW + DQ + IFMA, BMI2, FMA |
| Registros ZMM arquitecturales | $32$ ($zmm0..zmm31$) |
| Throughput FMA | $2$ pipes de $512$ bits $\Rightarrow$ $64$ flops/ciclo por core en FP32 |
| Frecuencia turbo bajo AVX-512 sostenido | $\sim 3.5$ GHz |

Cada decision queda justificada en el doc del modulo correspondiente: $M_c = K_c = 384$ ([`docs/1.3) matmul_tiled_ikj.md`](<docs/1.3) matmul_tiled_ikj.md>)) hace que los tres paneles activos llenen el L2 de $1$ MiB; $M_C = 288,\ k_c = 256$ ([`docs/1.4) matmul_tiled_ikj_avx512.md`](<docs/1.4) matmul_tiled_ikj_avx512.md>)) deja el panel $B$ activo de $32$ KiB holgado en el L1d de $48$ KiB; el default OpenMP de $8$ threads ([`docs/1.5) matmul_tiled_ikj_omp.md`](<docs/1.5) matmul_tiled_ikj_omp.md>)) usa un thread por core porque el hipervisor ya quito SMT.

Diferencia clave respecto a Zen 2 (rama `main`): como L3 es **compartida** entre los $8$ cores (no privada por CCX), `OMP_PROC_BIND=close` y `OMP_PROC_BIND=spread` son topologicamente equivalentes. Si vas a portar a otro chip, ver [Seccion 11](#11-adaptar-el-proyecto-a-otro-hardware-o-microkernel).

---

## 3. Estructura del repositorio

```
.
|-- README.md
|-- Makefile
|-- docs/                                 Documentacion del proyecto (ver Seccion 12)
|-- src/
|   |-- core/                             Modulos compartidos por todos los algoritmos
|   |   |-- matrix_utils.{h,c}            Alocacion alineada, init_*, matrices_close, scalar_t
|   |   |-- timing.h                      now_seconds() inline sobre CLOCK_MONOTONIC
|   |   `-- morton.{h,c}                  Encoding Z-order y reorganizacion row-major <-> Morton
|   |-- microkernels/                     Tiles AVX-512 header-only (static inline)
|   |   |-- kernel_avx512_morton.h        4x32, usado por la familia Morton
|   |   `-- kernel_avx512_tiled.h         6x32, usado por la familia tiled_ikj
|   |-- algorithms/
|   |   |-- naive/                        matmul_naive (baseline ijk)
|   |   |-- loops/                        matmul_loops (6 ordenes seleccionables por nombre)
|   |   |-- tiled_ikj/                    matmul_tiled_ikj{,_avx512,_omp}
|   |   `-- morton/                       matmul_morton{,_avx512,_omp}
|   |-- drivers/                          Programas main
|   |   |-- bench/                        bench_*.c (uno por kernel)
|   |   `-- validate/                     validate_*.c (uno por kernel)
|   `-- tests/                            Unit tests standalone (test_*.c)
|-- scripts/
|   |-- profile_perf_zen5.sh              Captura perf por celda (variant, m)
|   |-- run_perf_zen5_sweep.sh            Orquesta el sweep completo
|   |-- consolidate_perf_zen5.py          Une grupos A+B -> results/metrics.csv
|   `-- plot_metrics_perf_zen5.py         Renderiza las 4 figuras a plots/
|-- bin/                                  Binarios compilados (gitignored)
|   |-- bench/                            bench_<kernel>_ZEN5
|   |-- validate/                         validate_<kernel>_O0 o _ZEN5
|   `-- tests/                            test_<modulo>
|-- results/                              CSV y reportes de profiling (gitignored)
`-- plots/                                Imagenes generadas (gitignored)
```

---

## 4. Setup en Amazon Linux

El servidor de referencia corre **Amazon Linux 2023** sobre AWS c8a.2xlarge (Linux nativo, no virtualizado a traves de WSL2). Los comandos asumen acceso SSH al servidor.

### 4.1 Acceso al servidor

```bash
ssh -i <key.pem> ec2-user@<hostname>
```

Verificar la distribucion:

```bash
cat /etc/os-release        # debe decir Amazon Linux 2023
uname -a
```

### 4.2 Paquetes obligatorios

```bash
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y gcc make binutils
```

`Development Tools` agrupa `gcc`, `make`, `binutils` y `git`. El segundo `dnf install` es redundante pero asegura las versiones individuales si el grupo no esta disponible en tu AMI.

Provee `gcc` (compilador), `make` (build), `gprof` (perfilado por funcion, parte de `binutils`), `ar` y `objdump` (analisis).

### 4.3 perf

En Amazon Linux 2023 el paquete `perf` se instala directamente y funciona con el kernel del host sin compilacion adicional:

```bash
sudo dnf install -y perf
perf --version
```

Ajusta `perf_event_paranoid` para que los contadores PMU sean accesibles sin root (una vez por boot, o persistente):

```bash
sudo sysctl -w kernel.perf_event_paranoid=1                            # temporal
echo "kernel.perf_event_paranoid=1" | sudo tee /etc/sysctl.d/99-perf.conf && sudo sysctl --system   # persistente
```

A diferencia de WSL2, en Amazon Linux todos los eventos PMC del Zen 5 estan expuestos sin restricciones; no es necesario recompilar `perf` desde el kernel.

### 4.4 Python + matplotlib

Para los plots de profiling:

```bash
sudo dnf install -y python3 python3-pip
python3 -m venv ~/venvs/matmul
source ~/venvs/matmul/bin/activate
pip install matplotlib numpy
```

Activa el venv (`source ~/venvs/matmul/bin/activate`) cada vez que abras una nueva sesion SSH antes de correr cualquier script de plotting.

### 4.5 Cachegrind (opcional)

Para diagnosticar cache misses por linea de codigo (mas lento que `perf` pero util para inspeccion fina):

```bash
sudo dnf install -y valgrind
```

Uso basico:

```bash
valgrind --tool=cachegrind --cache-sim=yes ./bin/bench/bench_naive_ZEN5 1024 1
cg_annotate cachegrind.out.<pid> | less
```

### 4.6 Verificacion del entorno

```bash
gcc --version
make --version
gprof --version
perf --version
python3 -c "import matplotlib, numpy; print('python OK')"
```

---

## 5. Quickstart

Desde la raiz del repositorio:

```bash
make build       # 1. compila los 20 binarios (8 bench + 8 validate + 4 tests)
make validate    # 2. corre la piramide tests -> validate sobre los 8 kernels
make results     # 3. sweep perf Zen 5 -> results/metrics.csv
make plots       # 4. renderiza las 4 figuras en plots/
```

Si los tres pasos terminan con `OK`, el repo esta sano y `results/metrics.csv` tiene una fila por celda `(variant, m)` con todos los contadores. Usar los knobs de la [Seccion 8.3](#83-sweep-unificado-make-results) para limitar el alcance durante desarrollo.

---

## 6. Compilar

### 6.1 Targets agregados

| Target | Que hace |
|---|---|
| `make build` (alias `make all`) | Compila los $20$ binarios sin ejecutar nada |
| `make tests` | Compila y corre los $4$ unit-tests en orden |
| `make validate` | Depende de `tests`; corre los $8$ `validate_*` en orden |
| `make results` | Compila los $8$ benches y lanza el sweep perf $\to$ `results/metrics.csv` |
| `make plots` | Renderiza las $4$ figuras a `plots/` desde `results/metrics.csv` (no relanza `results`) |
| `make clean` | Borra `bin/` y `build/` |
| `make distclean` | `clean` + borra `results/*.csv` y `plots/*` |

### 6.2 Targets individuales

| Kernel | Target bench | Target validate |
|---|---|---|
| `naive` | `make bench_naive_ZEN5` | `make validate_naive` |
| `loops` | `make bench_loops_ZEN5` | `make validate_loops` |
| `tiled_ikj` | `make bench_tiled_ikj_ZEN5` | `make validate_tiled_ikj` |
| `tiled_ikj_avx512` | `make bench_tiled_ikj_avx512_ZEN5` | `make validate_tiled_ikj_avx512` |
| `tiled_ikj_omp` | `make bench_tiled_ikj_omp_ZEN5` | `make validate_tiled_ikj_omp` |
| `morton` | `make bench_morton_ZEN5` | `make validate_morton` |
| `morton_avx512` | `make bench_morton_avx512_ZEN5` | `make validate_morton_avx512` |
| `morton_omp` | `make bench_morton_omp_ZEN5` | `make validate_morton_omp` |

Referencia completa de targets, incluidos los de profiling (`profile_zen5`, `profile_zen5_one`, `profile_zen5_omp`, `consolidate_zen5`) y de analisis (`plots`), en [`docs/0.0) makefile.md`](<docs/0.0) makefile.md>).

### 6.3 Flags por sufijo

| Sufijo | Variable | Flags |
|---|---|---|
| `_O0` | `BASE_CFLAGS` | `-std=c11 -Wall -Wextra -Wpedantic -O0 -g -fno-omit-frame-pointer -D_POSIX_C_SOURCE=200809L` |
| `_ZEN5` | `CFLAGS_O3_ZEN5` | `-std=c11 -Wall -Wextra -Wpedantic -O3 -march=native -D_POSIX_C_SOURCE=200809L` + thresholds Zen 5 via `-D...` |
| `_ZEN5` + OMP | `CFLAGS_OMP_ZEN5` | `CFLAGS_O3_ZEN5` + `-fopenmp` |

`-O0 -g` se usa donde queremos que el codigo refleje el algoritmo sin reordenamientos del compilador (el `validate_naive`, el `validate_loops`, etc.). Los benches y los validates de las variantes vectorizadas usan `_ZEN5` porque los microkernels son `static inline` con intrinsics AVX-512: a `-O0` no se materializan las instrucciones FMA-$512$ y el test mediria otra cosa.

Los `-D...` que `CFLAGS_O3_ZEN5` inyecta sobrescriben en compile-time los defaults `#ifndef`-guarded de cada modulo. El bloque concreto vive en el Makefile y configura:

- `TILED_IKJ_MC_DEFAULT = 384`, `TILED_IKJ_KC_DEFAULT = 384` (apuntan al L2 de $1$ MiB).
- `TILED_IKJ_AVX512_MC = 288`, `TILED_IKJ_AVX512_BS_DEFAULT = 256` (panel $B$ de $32$ KiB en L1d).
- Igual para `TILED_IKJ_OMP_*`.
- `MORTON_AVX512_THRESHOLD_DEFAULT = 1048576` (leaf $A$ activo de $\sim 32$ KiB).
- `MORTON_OMP_RECURSION_THRESHOLD_DEFAULT = MORTON_OMP_PARALLEL_THRESHOLD_DEFAULT = 1048576`.

---

## 7. Validar correctitud

### 7.1 Piramide de tres niveles

El proyecto verifica con tres capas, de fina a gruesa:

```
tests (unit)     -> piezas individuales en aislamiento
validate         -> algoritmos completos contra invariantes algebraicos + cross-check vs naive
bench            -> performance (no correctitud)
```

`make validate` corre primero `tests` y aborta si falla. Detalle de las capas en [`docs/1.9) tests.md`](<docs/1.9) tests.md>).

### 7.2 Como ejecutar

```bash
make tests        # 4 unit-tests sobre core/ y microkernels/
make validate     # tests + 8 validate_* en orden, aborta al primer fallo
```

Individualmente:

```bash
./bin/validate/validate_naive_O0                       # m = 256 por defecto
./bin/validate/validate_naive_O0 512                   # m custom
./bin/validate/validate_tiled_ikj_avx512_ZEN5 256 256  # m, bs (solo tiled_*_avx512/omp)
./bin/tests/test_morton                                # tests del modulo morton
```

### 7.3 Invariantes cubiertos

Todos los `validate_*` chequean los tres invariantes algebraicos basicos:

1. $A \cdot 0 = 0$
2. $I \cdot Z = Z$
3. $A \cdot (Z_1 + Z_2) = A \cdot Z_1 + A \cdot Z_2$

Los `validate_*` distintos al baseline anaden ademas cross-validation contra `matmul_naive` sobre un barrido de tamanos. Tolerancias por validate y detalles de los tests adicionales en `docs/API.md` Seccion 5.2.

Salida esperada:

```
Validating matmul_naive at m=256, n=128
Tolerances: abs=1.0e-04, rel=1.0e-03
  [OK]   A * 0 == 0
  [OK]   I * Z == Z
  [OK]   A * (Z1+Z2) == A*Z1 + A*Z2
VALIDATION OK
```

Codigo de salida `0` en exito, `1` al primer fallo, con reporte del primer indice y los valores divergentes.

---

## 8. Ejecutar benchmarks

### 8.1 Convencion comun

**CLI estandar:**

```
bench_<kernel>_ZEN5 <m> [num_iters] [num_runs]
```

Variantes con knob adicional:

```
bench_loops_ZEN5 <order> <m> [num_iters] [num_runs]              # order = ijk|ikj|jik|jki|kij|kji
bench_tiled_ikj_avx512_ZEN5 <m> [num_iters] [num_runs] [bs]      # bs = k_c, default 256
bench_tiled_ikj_omp_ZEN5    <m> [num_iters] [num_runs] [bs]
```

**Defaults:** `num_iters = min(2m/n, 4)`, `num_runs = 5`.

**Protocolo:** una corrida de warm-up no medida + `num_runs` corridas medidas, se reporta la **mediana** de los tiempos.

**Salida:** una linea CSV en `stdout`:

```
<kernel>,m,n,num_iters,median_seconds,gflops
```

Para `tiled_ikj_avx512` y `tiled_ikj_omp` se inserta una columna `bs` (formato de $7$ columnas).

**Restriccion de la familia Morton:** $m$ debe ser potencia de $2$ y $\geq 4$ (el indexing Z-order requiere subdivisiones exactas). El binario aborta con mensaje claro si no se cumple. La reorganizacion a Morton se ejecuta **una sola vez antes del warm-up**, fuera del tiempo cronometrado.

### 8.2 Ejemplos

```bash
# Naive
./bin/bench/bench_naive_ZEN5 1024 4 1

# Loops, orden ikj
./bin/bench/bench_loops_ZEN5 ikj 1024 4 1

# Tiled_ikj_avx512 con bs custom
./bin/bench/bench_tiled_ikj_avx512_ZEN5 4096 4 1 384

# Tiled_ikj_omp con 8 threads, un thread por core
OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
    ./bin/bench/bench_tiled_ikj_omp_ZEN5 4096

# Morton (m potencia de 2)
./bin/bench/bench_morton_avx512_ZEN5 4096 4 1

# Morton OpenMP
OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
    ./bin/bench/bench_morton_omp_ZEN5 4096
```

En este chip `OMP_PROC_BIND=close` y `OMP_PROC_BIND=spread` son topologicamente equivalentes (L3 compartido, $1$ NUMA node), pero conviene mantener uno fijo durante un sweep para no introducir varianza entre celdas.

### 8.3 Sweep unificado (`make results`)

Para comparar kernels y producir un CSV consolidado:

```bash
make results                                            # defaults
make results MS="1024 4096"                             # solo esos tamanos
make results VARIANTS="tiled_ikj_avx512 morton_avx512"  # solo esas variantes
make results ITERS_PER_RUN=0                            # I_full = 2m/n, 1 run
```

**Knobs:**

| Variable | Default | Que controla |
|---|---|---|
| `VARIANTS` | $13$ activas | Lista de variantes a correr (separadas por espacio) |
| `MS` | `1024 2048 4096 8192 16384 32768` | Lista de tamanos |
| `ITERS_PER_RUN` | `1` | Iteraciones por run. `0` activa $I_{\text{full}} = 2m/n$ |
| `RUNS` | `3` (o `1` si `ITERS_PER_RUN=0`) | Runs medidos para la mediana |

**Variantes activas** (las que `run_perf_zen5_sweep.sh` corre por defecto): `naive`, `loop_ijk`, `loop_ikj`, `loop_jik`, `loop_jki`, `loop_kij`, `loop_kji`, `tiled_ikj`, `tiled_ikj_avx512`, `tiled_ikj_omp`, `morton`, `morton_avx512`, `morton_omp`.

**Salida:** `results/metrics.csv` con una fila por celda `(variant, m)` y todas las columnas de hardware counters consolidadas.

---

## 9. Profiling con `perf`

### 9.1 Que captura

`make results` lanza `scripts/run_perf_zen5_sweep.sh`, que por cada celda `(variant, m)` ejecuta `perf stat` dos veces para evitar multiplexing de contadores. Los `.txt` crudos quedan en `results/<variant>/perf_<variant>_m<M>_{A,B}.txt`; el consolidador `scripts/consolidate_perf_zen5.py` los une en `results/metrics.csv`.

**Eventos por grupo** (los que el script pide a `perf`; nombres exactos del PMC de Zen 5):

| Grupo | Eventos |
|---|---|
| A (compute side) | `cycles`, `instructions`, `fp_ret_sse_avx_ops.all`, `ls_dispatch.ld_dispatch`, `l2_request_g1.all_no_prefetch` |
| B (memory + TLB) | `cycles`, `instructions`, `l2_cache_req_stat.ls_rd_blk_l_hit_x`, `cache-misses`, `bp_l1_tlb_miss_l2_tlb_miss`, `dTLB-load-misses` |

`cycles` e `instructions` se duplican en ambos grupos para cross-check entre pasadas. La cabecera de `scripts/profile_perf_zen5.sh` documenta cada evento y sus equivalentes si alguno no esta disponible en una microcode revision concreta.

### 9.2 Como lanzar

```bash
sudo sysctl -w kernel.perf_event_paranoid=1     # una vez por boot

make profile_zen5                               # sweep completo (todas las variantes y MS)
make profile_zen5_one VARIANT=tiled_ikj_avx512 M=4096   # una sola celda
make profile_zen5_omp                           # morton_omp con varios threads
```

### 9.3 Plots

`scripts/plot_metrics_perf_zen5.py` lee `results/metrics.csv` y escribe en `plots/` cuatro figuras que resumen el sweep:

1. `gflops_vs_m.png` -- throughput vs $m$, ejes log-log, una linea por variante.
2. `best_per_family.png` -- arco de optimizacion (subset curado: `naive`, `loop_ikj`, `tiled_ikj{,_avx512,_omp}`, `morton{,_avx512,_omp}`) con lineas de techo teorico single-core y all-core para el 9R45.
3. `llc_misses_vs_m.png` -- LLC misses por kilo-instruccion vs $m$, log-log. Reemplaza la figura por niveles L1/L2/L3 del pipeline de Zen 2 porque bajo KVM AMD solo `cache-misses` esta expuesto.
4. `omp_scaling.png` -- pares `morton_avx512` vs `morton_omp` y `tiled_ikj_avx512` vs `tiled_ikj_omp` con speedup y eficiencia anotados sobre los $m$ comunes.

Estilo, paleta (verde oscuro tiled, azul oscuro morton, amarillos loops, gris naive) y marcadores (circulo plain, triangulo SIMD, estrella OMP) son los mismos que en `scripts/plot_metrics_perf_zen2.py` de la rama `main`.

```bash
source ~/venvs/matmul/bin/activate
make plots
```

`make plots` es el atajo canonico; internamente ejecuta `python3 scripts/plot_metrics_perf_zen5.py --csv results/metrics.csv --out-dir plots`. El target no depende de `results`, asi que un replot es barato y no relanza el sweep de perf. El header del CSV (`seconds` con BOM UTF-8 que emite el consolidador en el servidor) y el alias a `median_seconds` se manejan internamente.

---

## 10. Reproducibilidad y estabilidad

### 10.1 Documenta el hardware

Guarda esta info junto con tus resultados:

```bash
lscpu
cat /sys/devices/system/cpu/cpu0/cache/index{0,1,2,3}/size
free -h
cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
```

### 10.2 Aisla la instancia

Una instancia AWS dedicada no comparte cores con otros tenants en c8a.2xlarge, pero conviene asegurarse de que no hay carga concurrente del usuario durante las mediciones:

- Cerrar procesos pesados (builds, indexadores, monitorizadores como `htop` activo, etc.).
- No usar la misma instancia para builds y bench en paralelo.
- Verificar que ningun otro `bench_*` esta en background con `pgrep bench_`.

### 10.3 Mediana de varias corridas

Los benches ya reportan mediana de $5$ corridas por defecto. Para el sweep, ajusta con `RUNS=<n>`.

---

## 11. Adaptar el proyecto a otro hardware o microkernel

Los defaults estan calibrados para Zen 5 / EPYC 9R45. Si vas a correr en otro chip, estos son los puntos donde tocar:

| Que cambiar | Donde | Por que |
|---|---|---|
| `-march=native` | `CFLAGS_O3_ZEN5` y `CFLAGS_OMP_ZEN5` en el `Makefile` | Selecciona el ISA tuning de GCC para el chip de destino |
| Thresholds `-D...` del Makefile (`TILED_IKJ_MC_DEFAULT`, `TILED_IKJ_AVX512_MC`, `BS_DEFAULT`, `MORTON_*_THRESHOLD_DEFAULT`) | Bloque `CFLAGS_O3_ZEN5` del `Makefile` | Tienen que cuadrar con el L1d, L2 y registros del chip nuevo |
| Tile de microkernel $M_R, N_R$ | `kernel_avx512_morton.h` ($4 \times 32$) y `kernel_avx512_tiled.h` ($6 \times 32$) | Dependen del numero de registros vectoriales y del ancho de SIMD (32 ZMM en Zen 5, $16$ YMM en Zen 2) |
| Default `OMP_NUM_THREADS` | `scripts/profile_perf_zen5.sh` (casos `tiled_ikj_omp` / `morton_omp`) | Debe coincidir con el numero de cores fisicos del chip |
| Politica `OMP_PROC_BIND` | idem | Topologia: $1$ NUMA node con L3 compartido $\to$ `close` $==$ `spread`; varios CCX o NUMA $\to$ elegir segun el modelo de localidad |

Para reemplazar un microkernel con tu propia version: copiar `src/microkernels/kernel_avx512_*.h` a un header nuevo, ajustar las constantes $M_R$, $N_R$ y el cuerpo del kernel, y cambiar el `#include` en el modulo de algoritmo que lo consume (`matmul_morton_avx512.c`, `matmul_tiled_ikj_avx512.c`, etc.). El contrato externo de cada `matmul_*` no cambia, asi que los `validate_*` siguen sirviendo sin tocarlos.

La rama paralela `main` es un ejemplo de port a otra microarquitectura: lleva el proyecto a AVX2 sobre Zen 2 con los mismos kernels reescalados a tiles que aprovechan los $16$ YMM (tile $4 \times 16$ para Morton, $6 \times 16$ para tiled) y con setup en WSL2 sobre un Ryzen 5 4600H. Sirve como referencia de hasta donde llega un port "completo" en direccion opuesta a la actual.

---

## 12. Documentacion adicional

| Documento | Cubre |
|---|---|
| [`docs/API.md`](docs/API.md) | Contrato publico (firmas, tipos, layout, constantes, binarios) |
| [`docs/0.0) makefile.md`](<docs/0.0) makefile.md>) | Referencia completa de targets y knobs del `Makefile` |
| [`docs/0.1) matrix_utils.md`](<docs/0.1) matrix_utils.md>) | Helpers de `core/matrix_utils` |
| [`docs/0.2) morton.md`](<docs/0.2) morton.md>) | Encoding Z-order y reorganizacion |
| [`docs/1.1) matmul_naive.md`](<docs/1.1) matmul_naive.md>) | Baseline `ijk` |
| [`docs/1.2) matmul_loops.md`](<docs/1.2) matmul_loops.md>) | Reordenamiento de bucles ($6$ variantes) |
| [`docs/1.3) matmul_tiled_ikj.md`](<docs/1.3) matmul_tiled_ikj.md>) | Tiling explicito apuntando a L2 |
| [`docs/1.4) matmul_tiled_ikj_avx512.md`](<docs/1.4) matmul_tiled_ikj_avx512.md>) | Microkernel BLIS-style $6 \times 32$ con AVX-512 + FMA |
| [`docs/1.5) matmul_tiled_ikj_omp.md`](<docs/1.5) matmul_tiled_ikj_omp.md>) | $6 \times 32$ + OpenMP `parallel for` en $i_c$ |
| [`docs/1.6) matmul_morton.md`](<docs/1.6) matmul_morton.md>) | Recursion Morton fina cache-oblivious |
| [`docs/1.7) matmul_morton_avx512.md`](<docs/1.7) matmul_morton_avx512.md>) | Morton-de-bloques + microkernel AVX-512 $4 \times 32$ |
| [`docs/1.8) matmul_morton_omp.md`](<docs/1.8) matmul_morton_omp.md>) | Morton-de-bloques + OpenMP tasks |
| [`docs/1.9) tests.md`](<docs/1.9) tests.md>) | Piramide de unit-tests (estructura y casos cubiertos) |
