# Benchmark de Multiplicacion Iterada de Matrices

**Curso:** Arquitectura de Computadores
**Autores:** Santiago Uribe Echavarría - Juan Pablo Robledo Meza
**Universidad:** Universidad Nacional de Colombia, Sede Medellin
**Fecha:**29 de Mayo 2026
**Hardware de referencia:** Ryzen 5 4600H (Renoir, Zen 2)

Repositorio que implementa y mide la recurrencia $B_{i+1} = A \cdot B_i$ sobre ocho kernels distintos (baseline, reordenamientos, tiling, vectorizacion AVX2 y paralelizacion OpenMP) y un pipeline de profiling con `perf` en hardware Zen 2.

| Kernel | Fase | Tecnica principal |
|---|---|---|
| `matmul_naive` | Sesion 01 | Baseline ijk, $-O0$ |
| `matmul_loops` | Fase 1.1 | Seis permutaciones del orden de bucles |
| `matmul_tiled_ikj` | Fase 1.2 | Tiling explicito $M_c \times K_c = 256 \times 256$ apuntando a L2 |
| `matmul_tiled_ikj_avx2` | Fase 1.6 | Microkernel BLIS-style $6 \times 16$ con AVX2 + FMA |
| `matmul_tiled_ikj_omp` | Fase 1.6 | $6 \times 16$ + `#pragma omp parallel for` en $i_c$ |
| `matmul_morton` | Fase 6 | Recursion cache-oblivious con $A$ en Morton fino |
| `matmul_morton_avx2` | Sesion 03 | Morton-de-bloques + microkernel AVX2 $4 \times 16$ |
| `matmul_morton_omp` | Sesion 03 | Morton-de-bloques + AVX2 + OpenMP tasks |

El contrato publico de las funciones esta en [`docs/API.md`](docs/API.md). La rama `main_server` contiene un port a AVX-512 / Zen 5 que se editara en una siguiente fase.

---

## 0. Tabla de contenidos

1. [Que hace este proyecto](#1-que-hace-este-proyecto)
2. [Hardware de referencia](#2-hardware-de-referencia)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Setup en WSL2](#4-setup-en-wsl2)
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

---

## 2. Hardware de referencia

Todas las decisiones de hiperparametros del proyecto (tamanos de tile, configuracion OMP, threshold de recursion) se eligieron para el Ryzen 5 $4600$H:

| Recurso | Valor |
|---|---|
| Microarquitectura | Zen 2 (Renoir) |
| Cores fisicos | $6$, organizados en $2$ CCX de $3$ cores |
| Threads logicos | $12$ (SMT) |
| L1d | $32$ KiB por core |
| L2 | $512$ KiB por core |
| L3 | $4$ MiB **privado por CCX** |
| ISA SIMD | AVX2 + FMA (no AVX-512) |
| Frecuencia turbo bajo AVX2 | $\sim 4.0$ GHz |

Cada decision queda justificada en el doc del modulo correspondiente: $M_c = K_c = 256$ ([`docs/1.3) matmul_tiled_ikj.md`](<docs/1.3) matmul_tiled_ikj.md>)) hace que los tres paneles activos llenen el L2; $M_C = 192,\ k_c = 384$ ([`docs/1.4) matmul_tiled_ikj_avx2.md`](<docs/1.4) matmul_tiled_ikj_avx2.md>)) deja el panel $A$ activo de $288$ KiB en L2 y el panel $B$ de $24$ KiB en L1d; el default OpenMP de $6$ threads ([`docs/1.5) matmul_tiled_ikj_omp.md`](<docs/1.5) matmul_tiled_ikj_omp.md>)) usa un thread por core fisico para no compartir las pipas FMA con el sibling SMT.

Si vas a correr el proyecto en otro PC, ver [Seccion 11](#11-adaptar-el-proyecto-a-otro-hardware-o-microkernel) para los puntos donde tocar.

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
|   |-- microkernels/                     Tiles AVX2 header-only (static inline)
|   |   |-- kernel_avx2_morton.h          4x16, usado por la familia Morton
|   |   `-- kernel_avx2_tiled.h           6x16, usado por la familia tiled_ikj
|   |-- algorithms/
|   |   |-- naive/                        matmul_naive (baseline ijk)
|   |   |-- loops/                        matmul_loops (6 ordenes seleccionables por nombre)
|   |   |-- tiled_ikj/                    matmul_tiled_ikj{,_avx2,_omp}
|   |   `-- morton/                       matmul_morton{,_avx2,_omp}
|   |-- drivers/                          Programas main
|   |   |-- bench/                        bench_*.c (uno por kernel)
|   |   `-- validate/                     validate_*.c (uno por kernel)
|   `-- tests/                            Unit tests standalone (test_*.c)
|-- scripts/
|   |-- profile_perf_zen2.sh              Captura perf por celda (variant, m)
|   |-- run_perf_zen2_sweep.sh            Orquesta el sweep completo
|   |-- consolidate_perf_zen2.py          Une grupos A+B -> results/metrics.csv
|   `-- plot_perf_zen2.py                 Plots de IPC, FMA, cache misses, TLB walks
|-- bin/                                  Binarios compilados (gitignored)
|   |-- bench/                            bench_<kernel>_O3
|   |-- validate/                         validate_<kernel>_O0 o _O3
|   `-- tests/                            test_<modulo>
|-- results/                              CSV y reportes de profiling (gitignored)
`-- plots/                                Imagenes generadas (gitignored)
```

---

## 4. Setup en WSL2

El proyecto **requiere WSL2 con Ubuntu** por compatibilidad con `gprof` y `perf`. Probado en Ubuntu $22.04$ y $24.04$.

### 4.1 Instalar WSL2 + Ubuntu

Desde **PowerShell como administrador**:

```powershell
wsl --install -d Ubuntu
```

Tras el reinicio, abre Ubuntu desde el menu Inicio y crea tu usuario. Verifica:

```bash
uname -a   # debe decir Linux ... microsoft-standard-WSL2
```

### 4.2 Paquetes obligatorios

```bash
sudo apt update
sudo apt install -y build-essential gcc make binutils
```

Provee `gcc` (compilador), `make` (build), `gprof` (perfilado por funcion), `ar`/`objdump` (analisis).

### 4.3 perf

Primer intento con el paquete generico:

```bash
sudo apt install -y linux-tools-generic linux-tools-common
perf --version
```

Si falla con "perf not found for kernel ...", compila desde el repositorio del kernel de WSL2:

```bash
sudo apt install -y flex bison libelf-dev libdwarf-dev libdw-dev libnuma-dev pkg-config
cd /usr/src
sudo git clone --depth=1 https://github.com/microsoft/WSL2-Linux-Kernel.git
cd WSL2-Linux-Kernel/tools/perf
sudo make
sudo cp perf /usr/local/bin/
perf --version
```

Ajusta `perf_event_paranoid` para que los contadores PMU sean accesibles sin root (una vez por boot, o persistente):

```bash
sudo sysctl -w kernel.perf_event_paranoid=1                            # temporal
echo "kernel.perf_event_paranoid=1" | sudo tee /etc/sysctl.d/99-perf.conf && sudo sysctl --system   # persistente
```

En WSL2 algunos eventos PMU pueden devolver `<not supported>` (limitacion del hipervisor); el script de profiling lo registra y continua.

### 4.4 Python + matplotlib

Para los plots (`make plot_perf_zen2`):

```bash
sudo apt install -y python3 python3-pip python3-venv
python3 -m venv ~/venvs/matmul
source ~/venvs/matmul/bin/activate
pip install matplotlib numpy
```

Activa el venv (`source ~/venvs/matmul/bin/activate`) cada vez que abras una terminal antes de correr los plots.

### 4.5 Cachegrind (opcional)

Para diagnosticar cache misses por linea de codigo (mas lento que `perf` pero independiente del hipervisor):

```bash
sudo apt install -y valgrind
```

Uso basico:

```bash
valgrind --tool=cachegrind --cache-sim=yes ./bin/bench/bench_naive_O3 1024 1
cg_annotate cachegrind.out.<pid> | less
```

### 4.6 Verificacion del entorno

```bash
gcc --version
make --version
gprof --version
perf --version || echo "perf no esta listo"
python3 -c "import matplotlib, numpy; print('python OK')"
```

---

## 5. Quickstart

Desde la raiz del repositorio en WSL2:

```bash
make build       # 1. compila los 20 binarios (8 bench + 8 validate + 4 tests)
make validate    # 2. corre la piramide tests -> validate sobre los 8 kernels
make results     # 3. sweep perf Zen 2 -> results/metrics.csv
```

Si los tres pasos terminan con `OK`, el repo esta sano y `results/metrics.csv` tiene una fila por celda `(variant, m)` con todos los contadores. El sweep completo tarda alrededor de $20$-$30$ minutos en el $4600$H; usar los knobs de la [Seccion 8.3](#83-sweep-unificado-make-results) para limitar el alcance durante desarrollo.

---

## 6. Compilar

### 6.1 Targets agregados

| Target | Que hace |
|---|---|
| `make build` (alias `make all`) | Compila los $20$ binarios sin ejecutar nada |
| `make tests` | Compila y corre los $4$ unit-tests en orden |
| `make validate` | Depende de `tests`; corre los $8$ `validate_*` en orden |
| `make results` | Compila los $8$ benches y lanza el sweep perf $\to$ `results/metrics.csv` |
| `make clean` | Borra `bin/` y `build/` |
| `make distclean` | `clean` + borra `results/*.csv` y `plots/*` |

### 6.2 Targets individuales

| Kernel | Target bench | Target validate |
|---|---|---|
| `naive` | `make bench_naive_O3` | `make validate_naive` |
| `loops` | `make bench_loops_O3` | `make validate_loops` |
| `tiled_ikj` | `make bench_tiled_ikj_O3` | `make validate_tiled_ikj` |
| `tiled_ikj_avx2` | `make bench_tiled_ikj_avx2_O3` | `make validate_tiled_ikj_avx2` |
| `tiled_ikj_omp` | `make bench_tiled_ikj_omp_O3` | `make validate_tiled_ikj_omp` |
| `morton` | `make bench_morton_O3` | `make validate_morton` |
| `morton_avx2` | `make bench_morton_avx2_O3` | `make validate_morton_avx2` |
| `morton_omp` | `make bench_morton_omp_O3` | `make validate_morton_omp` |

Referencia completa de targets, incluidos los de profiling (`profile_zen2`, `profile_zen2_one`, `profile_zen2_omp`, `consolidate_zen2`, `plot_perf_zen2`), en [`docs/0.0) makefile.md`](<docs/0.0) makefile.md>).

### 6.3 Flags por sufijo

| Sufijo | Variable | Flags |
|---|---|---|
| `_O0` | `BASE_CFLAGS` | `-std=c11 -Wall -Wextra -Wpedantic -O0 -g -fno-omit-frame-pointer -D_POSIX_C_SOURCE=200809L` |
| `_O3` | `CFLAGS_O3_ZEN2` | `BASE_CFLAGS` con `-O3 -march=znver2 -mavx2 -mfma` (sin `-O0` ni `-g`) |
| `_O3` + OMP | `CFLAGS_OMP_ZEN2` | `CFLAGS_O3_ZEN2` + `-fopenmp` |

`-O0 -g` se usa donde queremos que el codigo refleje el algoritmo sin reordenamientos del compilador (el `validate_naive`, el `validate_loops`, etc.). Los benches y los validates de las variantes vectorizadas usan `_O3` porque los microkernels son `static inline` con intrinsics: a `-O0` no se materializan las instrucciones FMA y el test medirsia otra cosa.

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
./bin/validate/validate_naive_O0           # m = 256 por defecto
./bin/validate/validate_naive_O0 512       # m custom
./bin/validate/validate_tiled_ikj_avx2_O3 256 384   # m, bs (solo tiled_*_avx2/omp)
./bin/tests/test_morton                    # tests del modulo morton
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
bench_<kernel>_O3 <m> [num_iters] [num_runs]
```

Variantes con knob adicional:

```
bench_loops_O3 <order> <m> [num_iters] [num_runs]              # order = ijk|ikj|jik|jki|kij|kji
bench_tiled_ikj_avx2_O3 <m> [num_iters] [num_runs] [bs]        # bs = k_c, default 384
bench_tiled_ikj_omp_O3  <m> [num_iters] [num_runs] [bs]
```

**Defaults:** `num_iters = min(2m/n, 4)`, `num_runs = 5`.

**Protocolo:** una corrida de warm-up no medida + `num_runs` corridas medidas, se reporta la **mediana** de los tiempos.

**Salida:** una linea CSV en `stdout`:

```
<kernel>,m,n,num_iters,median_seconds,gflops
```

Para `tiled_ikj_avx2` y `tiled_ikj_omp` se inserta una columna `bs` (formato de $7$ columnas).

**Restriccion de la familia Morton:** $m$ debe ser potencia de $2$ y $\geq 4$ (el indexing Z-order requiere subdivisiones exactas). El binario aborta con mensaje claro si no se cumple. La reorganizacion a Morton se ejecuta **una sola vez antes del warm-up**, fuera del tiempo cronometrado.

### 8.2 Ejemplos

```bash
# Naive
./bin/bench/bench_naive_O3 1024 4 1

# Loops, orden ikj
./bin/bench/bench_loops_O3 ikj 1024 4 1

# Tiled_ikj_avx2 con bs custom
./bin/bench/bench_tiled_ikj_avx2_O3 4096 4 1 256

# Tiled_ikj_omp con 6 threads y binding close
OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close \
    ./bin/bench/bench_tiled_ikj_omp_O3 4096

# Morton (m potencia de 2)
./bin/bench/bench_morton_avx2_O3 4096 4 1

# Morton OpenMP
OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close \
    ./bin/bench/bench_morton_omp_O3 4096
```

### 8.3 Sweep unificado (`make results`)

Para comparar kernels y producir un CSV consolidado:

```bash
make results                                        # defaults
make results MS="1024 4096"                         # solo esos tamanos
make results VARIANTS="tiled_ikj_avx2 morton_avx2"  # solo esas variantes
make results ITERS_PER_RUN=0                        # I_full = 2m/n, 1 run
```

**Knobs:**

| Variable | Default | Que controla |
|---|---|---|
| `VARIANTS` | $13$ activas | Lista de variantes a correr (separadas por espacio) |
| `MS` | `1024 2048 4096 8192 16384 32768` | Lista de tamanos |
| `ITERS_PER_RUN` | `1` | Iteraciones por run. `0` activa $I_{\text{full}} = 2m/n$ |
| `RUNS` | `3` (o `1` si `ITERS_PER_RUN=0`) | Runs medidos para la mediana |

**Variantes activas** (las que `run_perf_zen2_sweep.sh` corre por defecto): `naive`, `loop_ijk`, `loop_ikj`, `loop_jik`, `loop_jki`, `loop_kij`, `loop_kji`, `tiled_ikj`, `tiled_ikj_avx2`, `tiled_ikj_omp`, `morton`, `morton_avx2`, `morton_omp`.

**Salida:** `results/metrics.csv` con una fila por celda `(variant, m)` y todas las columnas de hardware counters consolidadas.

---

## 9. Profiling con `perf`

### 9.1 Que captura

`make results` lanza `scripts/run_perf_zen2_sweep.sh`, que por cada celda `(variant, m)` ejecuta `perf stat` dos veces para evitar multiplexing de contadores. Los `.txt` crudos quedan en `results/<variant>/perf_<variant>_m<M>_{A,B}.txt`; el consolidador `scripts/consolidate_perf_zen2.py` los une en `results/metrics.csv`.

**Eventos por grupo (los que el script pide a `perf`):**

| Grupo | Eventos |
|---|---|
| A (compute side) | `cycles`, `instructions`, `fp_ret_sse_avx_ops.all`, `ls_dispatch.ld_dispatch`, `l2_request_g1.all_no_prefetch` |
| B (memory + TLB) | `cycles`, `instructions`, `l2_cache_req_stat.ls_rd_blk_l_hit_x`, `cache-misses`, `bp_l1_tlb_miss_l2_tlb_miss`, `dTLB-load-misses` |

`cycles` e `instructions` se duplican en ambos grupos para cross-check entre pasadas. Algunos eventos especificados por la rubrica del curso no estan expuestos por el kernel WSL2 + microcode actual; el script los sustituye por proxies equivalentes y deja el detalle en la cabecera de `scripts/profile_perf_zen2.sh`.

### 9.2 Como lanzar

```bash
sudo sysctl -w kernel.perf_event_paranoid=1     # una vez por boot

make profile_zen2                               # sweep completo (todas las variantes y MS)
make profile_zen2_one VARIANT=tiled_ikj_avx2 M=4096   # una sola celda
make profile_zen2_omp                           # morton_omp con varios threads
```

### 9.3 Plots

```bash
source ~/venvs/matmul/bin/activate
make plot_perf_zen2                             # plots/perf_zen2_breakdown.png
```

---

## 10. Reproducibilidad y estabilidad

### 10.1 Documenta el hardware

Guarda esta info junto con tus resultados:

```bash
lscpu
cat /sys/devices/system/cpu/cpu0/cache/index{0,1,2,3}/size
free -h
```

### 10.2 Fija el plan de energia en Windows

WSL2 hereda el gobernador del host. Desde PowerShell:

```powershell
powercfg /setactive SCHEME_MIN   # Maximum performance
```

### 10.3 Cierra programas pesados

Chrome con muchas pestanas, Slack, Zoom, Docker Desktop, OneDrive sincronizando, etc. compiten por L3 y memoria y meten ruido. Cierralos antes de medir.

### 10.4 Mediana de varias corridas

Los benches ya reportan mediana de $5$ corridas por defecto. Para el sweep, ajusta con `RUNS=<n>`.

---

## 11. Adaptar el proyecto a otro hardware o microkernel

Los defaults estan calibrados para Zen 2. Si vas a correr en otro PC, estos son los puntos donde tocar:

| Que cambiar | Donde | Por que |
|---|---|---|
| `-march=znver2` | `CFLAGS_O3_ZEN2` y `CFLAGS_OMP_ZEN2` en el `Makefile` | Selecciona el ISA tuning de GCC para el chip de destino |
| Tamanos de tile $M_C$, $k_c$ | `TILED_IKJ_AVX2_MC`, `TILED_IKJ_AVX2_BS_DEFAULT` y sus equivalentes `_OMP` en `src/algorithms/tiled_ikj/*.h` | Tienen que cuadrar con el L1d y el L2 del nuevo chip |
| Tile de microkernel $M_R, N_R$ | `kernel_avx2_morton.h` ($4 \times 16$) y `kernel_avx2_tiled.h` ($6 \times 16$) | Dependen del numero de registros vectoriales disponibles (16 YMM en Zen 2, 32 ZMM en Zen 5) |
| Default `OMP_NUM_THREADS` | `profile_perf_zen2.sh` (caso `tiled_ikj_omp` / `morton_omp`) | Debe coincidir con el numero de cores fisicos del nuevo chip |
| Thresholds Morton | `g_recursion_threshold*` en `src/algorithms/morton/*.c` | Calibrados al L1d del 4600H; cambian con el tamano de L1d |

Para reemplazar un microkernel con tu propia version: copiar `src/microkernels/kernel_avx2_*.h` a un header nuevo, ajustar las constantes $M_R$, $N_R$ y el cuerpo del kernel, y cambiar el `#include` en el modulo de algoritmo que lo consume (`matmul_morton_avx2.c`, `matmul_tiled_ikj_avx2.c`, etc.). El contrato externo de cada `matmul_*` no cambia, asi que los `validate_*` siguen sirviendo sin tocarlos.

La rama paralela `main_server` es un ejemplo de port a otra microarquitectura: lleva el proyecto a AVX-512 sobre Zen 5 con los mismos kernels reescalados a tiles que aprovechan los 32 ZMM. Sirve como referencia de hasta donde llega un port "completo".

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
| [`docs/1.4) matmul_tiled_ikj_avx2.md`](<docs/1.4) matmul_tiled_ikj_avx2.md>) | Microkernel BLIS-style $6 \times 16$ con AVX2 + FMA |
| [`docs/1.5) matmul_tiled_ikj_omp.md`](<docs/1.5) matmul_tiled_ikj_omp.md>) | $6 \times 16$ + OpenMP `parallel for` en $i_c$ |
| [`docs/1.6) matmul_morton.md`](<docs/1.6) matmul_morton.md>) | Recursion Morton fina cache-oblivious |
| [`docs/1.7) matmul_morton_avx2.md`](<docs/1.7) matmul_morton_avx2.md>) | Morton-de-bloques + microkernel AVX2 $4 \times 16$ |
| [`docs/1.8) matmul_morton_omp.md`](<docs/1.8) matmul_morton_omp.md>) | Morton-de-bloques + OpenMP tasks |
| [`docs/1.9) tests.md`](<docs/1.9) tests.md>) | Piramide de unit-tests (estructura y casos cubiertos) |
