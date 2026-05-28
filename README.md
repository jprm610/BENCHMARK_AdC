# Benchmark de Multiplicacion Iterada de Matrices

**Curso:** Arquitectura de Computadores
**Universidad:** Universidad Nacional de Colombia, Sede Medellin
**Fecha:** Mayo 2026
**Estado:** Fase 1 cerrada (baseline + profiling + escalamiento con $m$); Fase 1.1 cerrada (reordenamiento de bucles, 6 variantes); Fase 1.2 cerrada (tiling explicito `ikj` apuntando a L2); Fase 1.3 cerrada (`tiled_ikj_avx2`, integrada en `make results`); Fase 1.4 cerrada (`tiled_ikj_omp`, integrada en `make results` y sweep perf); Fase 1.6 cerrada (microkernel BLIS-style $6 \times 16$ inline para `tiled_ikj_avx2` y `tiled_ikj_omp`, $M_R = 6$, $N_R = 16$, $M_C = 192$, $BS = 384$ por defecto, default OMP a 6 threads `close`); Fase 1.6.1 cerrada (rename `tiled` $\to$ `tiled_ikj` por consistencia); Fase 6 cerrada (Morton Z-order cache-oblivious); Sesion 03 cerrada (microkernel AVX2 + FMA, OpenMP tasks, perf Zen 2, Roofline anclado al $4600$H).

---

## 1. Que hace este proyecto

Implementa y mide el benchmark de multiplicacion iterada de matrices:

$$
B_{i+1} = A \cdot B_i, \quad i = 0, 1, \ldots, I-1, \quad B_0 = Z
$$

donde $A \in \mathbb{R}^{m \times m}$, $Z \in \mathbb{R}^{m \times 128}$ e $I = 2m/n$. En cada iteracion se almacenan las primeras $n$ filas del producto.

La fase actual cubre los **tres primeros pasos** del proyecto:

1. **Implementacion en C compilada sin optimizacion de compilador** (`-O0`).
2. **Profiling** con `gprof` (perfil por funcion) y `perf` (contadores de hardware: instrucciones, IPC, fallos de pagina, efectividad de branching).
3. **Evaluacion de desempeno a medida que crece $m$**, identificando los tamanos donde aparecen cambios importantes y comparando con la complejidad teorica $2 m^2 n$ por iteracion.

La especificacion completa de la API publica esta en [`docs/API.md`](docs/API.md). Es un documento vivo: cada cambio a las firmas se reflejara alli en el mismo commit.

---

## 2. Estructura del repositorio

```
.
|-- README.md                          -> Este archivo
|-- Makefile                           -> Targets de compilacion, profiling y graficas
|-- docs/
|   |-- 0.0) makefile.md               -> Referencia de uso del Makefile (targets y flags)
|   |-- 0.1) matrix_utils.md           -> Helpers compartidos (matrix_utils)
|   |-- 1.1) matmul_naive.md           -> Algoritmo naive (baseline ijk)
|   |-- 1.2) matmul_loops.md           -> Reordenamiento de bucles (6 variantes)
|   |-- 1.3) matmul_tiled_ikj.md       -> Tiling explicito sobre ikj
|   `-- API.md                         -> Contrato publico de las funciones
|-- src/
|   |-- core/                              -> Modulos compartidos por todos los algoritmos
|   |   |-- matrix_utils.{h,c}             -> Helpers (alocacion, init, comparacion)
|   |   |-- timing.h                       -> clock_gettime(CLOCK_MONOTONIC) inline
|   |   `-- morton.{h,c}                   -> Encoding Z-order + reorganizacion (Etapa A3)
|   |-- microkernels/                      -> Tiles AVX2, ambos header-only (static inline)
|   |   |-- kernel_avx2_morton.h           -> Microkernel 4x16 usado por la familia Morton (Etapa A4)
|   |   `-- kernel_avx2_tiled.h            -> Microkernel 6x16 usado por tiled_ikj_avx2 y tiled_ikj_omp
|   |-- algorithms/                        -> Una carpeta por familia de algoritmo
|   |   |-- naive/matmul_naive.{h,c}       -> Baseline ijk (Sesion 01)
|   |   |-- loops/matmul_loops.{h,c}       -> 6 ordenes de loop con lookup por nombre (Fase 1.1)
|   |   |-- morton/                        -> Fase 6 / Sesion 02-03: kernel recursivo + AVX2 + OMP
|   |   |   |-- matmul_morton.{h,c}            -> Kernel recursivo con A en Morton fino (A3)
|   |   |   |-- matmul_morton_avx2.{h,c}       -> Morton-de-bloques + microkernel (A4)
|   |   |   `-- matmul_morton_omp.{h,c}        -> Variante paralela OpenMP tasks (A5)
|   |   `-- tiled_ikj/                     -> Fase 1.2-1.6: tiling explicito + BLIS 6x16 + OpenMP
|   |       |-- matmul_tiled_ikj.{h,c}         -> Tiling Mc x Kc sobre ikj, apunta a L2 (Mc=Kc=256)
|   |       |-- matmul_tiled_ikj_avx2.{h,c}    -> Microkernel inline 6x16 (MR=6, NR=16, MC=192), BS=384, AVX2+FMA
|   |       `-- matmul_tiled_ikj_omp.{h,c}     -> Microkernel 6x16 + #pragma omp parallel for schedule(static) en ic
|   |-- drivers/                           -> Programas main: medicion (bench) y verificacion (validate)
|   |   |-- bench/                         -> bench_naive.c, bench_loops.c, bench_morton{,_avx2,_omp}.c, bench_tiled_ikj{,_avx2,_omp}.c
|   |   `-- validate/                      -> validate_naive.c, validate_loops.c, validate_morton{,_avx2,_omp}.c, validate_tiled_ikj{,_avx2,_omp}.c
|   `-- tests/                             -> Tests unitarios standalone (ver docs/1.9) tests.md)
|       |-- test_matrix_utils.c            -> xalloc_aligned, init_matrix_*, matrices_close
|       |-- test_morton.c                  -> Round-trip encode/decode + contiguidad de cuadrantes
|       |-- test_kernel_avx2_morton.c      -> Microkernel 4x16 (kernel_avx2_morton.h)
|       `-- test_kernel_avx2_tiled.c       -> Microkernel 6x16 (kernel_avx2_tiled.h)
|-- scripts/
|   |-- profile_perf_zen2.sh           -> Captura perf por celda (variant, m)
|   |-- run_perf_zen2_sweep.sh         -> Orquesta el sweep de variantes x tamanos
|   |-- consolidate_perf_zen2.py       -> Consolida grupos A+B -> results/metrics.csv
|   `-- plot_perf_zen2.py              -> 4 paneles: IPC, FMA, L3 miss, TLB walks
|-- results/                            -> CSV y reportes de profiling (gitignored)
|-- plots/                              -> Imagenes generadas (gitignored)
`-- bin/                                -> Binarios compilados (gitignored)
```

---

## 3. Prerrequisitos: que instalar en WSL2

Este proyecto asume que estas trabajando en **WSL2 con Ubuntu** (decision tomada al inicio del proyecto). Las instrucciones que siguen estan probadas en Ubuntu 22.04 y 24.04.

### 3.1 Si todavia no tienes WSL2

Abre **PowerShell como administrador** desde Windows y ejecuta:

```powershell
wsl --install -d Ubuntu
```

Tras el reinicio, abre Ubuntu desde el menu Inicio y crea tu usuario. Para verificar:

```bash
uname -a   # debe decir Linux ... microsoft-standard-WSL2
lsb_release -a
```

### 3.2 Paquetes obligatorios

Dentro del shell de Ubuntu (WSL2):

```bash
sudo apt update
sudo apt install -y build-essential gcc make binutils
```

Esto te da:

| Comando | Paquete | Por que lo necesitamos |
|---------|---------|------------------------|
| `gcc`   | `gcc`               | Compilador C, paso 1 del proyecto |
| `make`  | `make`              | Orquesta la compilacion |
| `gprof` | `binutils`          | Perfilado por funcion, paso 2 |
| `ar`, `objdump` | `binutils`  | Utiles para analisis posteriores |

### 3.3 Paquetes para profiling de hardware (paso 2)

Para `perf` (contadores de instrucciones, IPC, fallos de pagina, branch misses):

```bash
sudo apt install -y linux-tools-generic linux-tools-common
```

En WSL2 el kernel es propio de Microsoft, por lo que el paquete generico no siempre incluye un `perf` directamente ejecutable. Si al ejecutar `perf` aparece un mensaje del tipo "perf not found for kernel ...", instala desde fuente:

```bash
sudo apt install -y flex bison libelf-dev libdwarf-dev libdw-dev libnuma-dev pkg-config
cd /usr/src
sudo git clone --depth=1 https://github.com/microsoft/WSL2-Linux-Kernel.git
cd WSL2-Linux-Kernel/tools/perf
sudo make
sudo cp perf /usr/local/bin/
perf --version
```

**Importante para WSL2:** los contadores de PMU disponibles dependen del soporte del hipervisor. En la practica funcionan al menos `instructions`, `cycles`, `branches`, `branch-misses`, `task-clock`, `page-faults`. Si algun evento devuelve `<not supported>`, no es un error tuyo, simplemente el evento no esta expuesto.

Para bajar la restriccion de seguridad (necesario en muchos kernels):

```bash
echo "kernel.perf_event_paranoid=1" | sudo tee /etc/sysctl.d/99-perf.conf
sudo sysctl --system
```

### 3.4 Cachegrind (opcional, recomendado)

`cachegrind` simula la jerarquia de cache y te da fallos por linea de codigo. Es **mucho mas lento** (10-50x) pero independiente del hardware, asi que funciona bien dentro de WSL2:

```bash
sudo apt install -y valgrind
```

Uso basico (no incluido en los scripts, util para diagnostico fino):

```bash
valgrind --tool=cachegrind --cache-sim=yes ./bin/bench_naive_O0 1024 1
cg_annotate cachegrind.out.<pid>
```

### 3.5 Python + matplotlib (paso 3, graficas)

```bash
sudo apt install -y python3 python3-pip python3-venv
python3 -m venv ~/venvs/matmul
source ~/venvs/matmul/bin/activate
pip install matplotlib numpy
```

Recuerda activar el venv (`source ~/venvs/matmul/bin/activate`) cada vez que abras una nueva terminal antes de ejecutar los scripts de plotting (`scripts/plot_perf_zen2.py`).

Alternativa rapida sin venv (no recomendada para entornos compartidos):

```bash
sudo apt install -y python3-matplotlib python3-numpy
```

### 3.6 Verificacion rapida del entorno

```bash
gcc --version
make --version
gprof --version
perf --version || echo "perf no esta listo"
python3 -c "import matplotlib, numpy; print('python OK')"
```

---

## 4. Compilar el proyecto

Desde la raiz del repositorio (en WSL2):

```bash
make
```

Esto produce dos binarios en `bin/`:

| Binario | Compilado con | Para |
|---------|---------------|------|
| `bin/bench_naive_O0`    | `-O0 -g`      | Benchmark baseline, paso 1 y paso 3 |
| `bin/validate_naive_O0` | `-O0 -g`      | Verificador de correctitud |

Targets individuales del baseline (Fase 1):

```bash
make bench_naive_O3      # solo el benchmark baseline
make validate_naive      # solo el verificador baseline
make clean               # borra bin/ y build/
make distclean           # clean + borra results/*.csv y plots/*
```

Targets de Fase 1.1 (loop reorder), Fase 1.2 (tiling) y Fase 1.3 (tiled_ikj_avx2):

```bash
# Fase 1.1 - loop reorder
make bench_loops              # bin/bench_loops_O0 y bin/bench_loops_O3
make validate_loops           # bin/validate_loops_O0

# Fase 1.2 - tiling explicito
make bench_tiled_ikj              # bin/bench_tiled_ikj_O3
make validate_tiled_ikj           # bin/validate_tiled_ikj_O0

# Fase 1.3 - tiled_ikj_avx2 (6-loop tiling con AVX2+FMA, compilado con -O3 -march=znver2)
make bench_tiled_ikj_avx2         # bin/bench_tiled_ikj_avx2_O3
make validate_tiled_ikj_avx2      # bin/validate_tiled_ikj_avx2_O3

# Fase 1.4 - tiled_ikj_omp (tiled_ikj_avx2 + OpenMP parallel for, compilado con -O3 -march=znver2 -fopenmp)
make bench_tiled_ikj_omp          # bin/bench_tiled_ikj_omp_O3
make validate_tiled_ikj_omp       # bin/validate_tiled_ikj_omp_O3
```

Targets de Fase 6 (Morton Z-order cache-oblivious):

```bash
make bench_morton_O3          # bin/bench_morton_O3   (m debe ser potencia de 2)
make validate_morton          # bin/validate_morton_O0 (idem)
```

Unit tests (capa por debajo de validate, ver [`docs/1.9) tests.md`](docs/1.9\)%20tests.md)):

```bash
make tests                       # bin/tests/* — los 4 unit-tests, en orden
make test_matrix_utils           # solo bin/tests/test_matrix_utils
make test_morton                 # solo bin/tests/test_morton
make test_kernel_avx2_morton     # solo bin/tests/test_kernel_avx2_morton
make test_kernel_avx2_tiled      # solo bin/tests/test_kernel_avx2_tiled
```

`make validate_all` depende de `make tests`, asi que un fallo en los unit-tests aborta antes de correr los `validate_*`.

Para comparaciones entre kernels (naive + loops + tiled_ikj* + morton*) usar el pipeline unificado de la Sesion 03:

```bash
make results                  # sweep perf Zen 2 + consolida -> results/metrics.csv
```

**Flags fijos en el Makefile** (`BASE_CFLAGS`):

```
-std=c11 -Wall -Wextra -Wpedantic -O0 -g -fno-omit-frame-pointer
```

- `-O0`: usado por los `validate_*` para que las comparaciones reflejen el algoritmo sin reordenamientos del compilador.
- `-g`: simbolos de debug, necesarios para que `perf report` muestre nombres legibles.
- `-fno-omit-frame-pointer`: deja el stack pointer en su sitio para que las herramientas de profiling resuelvan call graphs sin DWARF unwinding.

Los binarios de medicion (`bench_*_O3`) se compilan con `CFLAGS_O3_ZEN2` (`-O3 -march=znver2 -mavx2 -mfma`).

---

## 5. Correr y validar

### 5.1 Validacion (verifica que el kernel computa bien)

```bash
./bin/validate_naive_O0          # m = 256 por defecto
./bin/validate_naive_O0 512      # m custom
```

Pasa tres invariantes algebraicos:

1. $A \cdot 0 = 0$
2. $I \cdot Z = Z$
3. $A \cdot (Z_1 + Z_2) = A \cdot Z_1 + A \cdot Z_2$

Salida esperada:

```
Validating matmul_naive at m=256, n=128
Tolerances: abs=1.0e-04, rel=1.0e-03
  [OK]   A * 0 == 0
  [OK]   I * Z == Z
  [OK]   A * (Z1+Z2) == A*Z1 + A*Z2
VALIDATION OK
```

Si alguno falla, el codigo de salida es 1 y se reporta el primer indice donde difieren los valores.

### 5.2 Una sola corrida del benchmark

```bash
./bin/bench_naive_O0 1024            # m=1024, iteraciones y corridas default
./bin/bench_naive_O0 1024 4          # m=1024, 4 iteraciones medidas por corrida
./bin/bench_naive_O0 1024 4 1        # m=1024, 4 iteraciones, 1 sola corrida medida
```

Los tres argumentos posicionales son:

1. `m`: tamano del problema (obligatorio).
2. `num_iters`: iteraciones del benchmark dentro de cada corrida medida. Default: $\min(2m/n, 4)$.
3. `num_runs`: corridas medidas (sobre las que se toma la mediana). Default: 5.

Independientemente de `num_runs`, el binario hace siempre **1 corrida de warm-up** (no medida) antes de medir.

Salida (una linea CSV en stdout):

```
m,n,num_iters,median_seconds,gflops
1024,128,4,X.XXXXXX,X.XXXXXX
```

### 5.3 Sweep completo y consolidacion

El flujo unificado vive en `make results`. Lanza el sweep de hardware counters sobre todas las variantes activas y consolida en `results/metrics.csv`:

```bash
make results                                        # defaults completos
make results MS="1024 4096"                         # solo esos tamanos
make results VARIANTS="tiled_ikj_avx2 morton_avx2"  # solo esas variantes
make results ITERS_PER_RUN=0                        # I_full = 2m/n, 1 run
```

La especificacion completa de knobs (`VARIANTS`, `MS`, `ITERS_PER_RUN`, `RUNS`) y todos los targets relacionados estan en [`docs/0.0) makefile.md`](docs/0.0\)%20makefile.md).

### 5.4 Flujo de Fase 6: Morton (Z-order) cache-oblivious

#### 5.4.1 Validacion

```bash
./bin/validate_morton_O0    256   # 3 invariantes + cross-validation contra naive (m potencia de 2)
./bin/tests/test_morton           # tests del modulo Morton (encode/decode/reorganize)
```

Cada uno imprime `VALIDATION OK` (o `MORTON TESTS OK`) y retorna 0 cuando todo pasa.

#### 5.4.2 Bench individual

```bash
./bin/bench_morton_O3    1024 4 1        # m debe ser potencia de 2
```

Misma CLI y mismo CSV de salida que `bench_naive_O3`. `bench_morton_O3` ejecuta `reorganize_to_morton(A)` una sola vez antes del warm-up, fuera del tiempo medido, para que las GFLOP/s reflejen solo el kernel.

#### 5.4.3 Comparacion entre kernels

Para comparar Morton contra el resto del pipeline (naive, loops, tiled_ikj*, morton_avx2, morton_omp) se usa el pipeline unificado:

```bash
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'   # una vez por boot
make results              # sweep perf Zen 2 sobre las variantes activas -> results/metrics.csv
```

### 5.7 Flujo de Fase 1.2: tiling explicito (`tiled_ikj`)

```bash
# Compilar
make bench_tiled_ikj
make validate_tiled_ikj

# Validar correctitud
./bin/validate_tiled_ikj_O0 256

# Bench individual
./bin/bench_tiled_ikj_O3 1024          # m=1024, defaults
./bin/bench_tiled_ikj_O3 1024 4 1      # m, iters, runs
```

Salida CSV:
```
tiled_ikj,1024,128,4,X.XXXXXX,X.XXXXXX
```

Para incluir `tiled_ikj` en el sweep de perf completo y regenerar `results/metrics.csv`:

```bash
make results
```

El target `results` ya incluye `bench_tiled_ikj_O3` como dependencia y `run_perf_zen2_sweep.sh`
incluye `tiled_ikj` en su lista de variantes por defecto.

Para correr solo la celda de tiling sin relanzar todo el sweep:

```bash
VARIANTS="tiled_ikj" MS="1024 2048" bash scripts/run_perf_zen2_sweep.sh
python3 scripts/consolidate_perf_zen2.py
```

**Tamanos de tile:** `Mc = Kc = 256`, elegidos para que los tres panels activos (A: 256 KB, B: 128 KB, C: 128 KB) llenen exactamente el L2 de 512 KB del Ryzen 5 4600H. El beneficio sobre `loop_ikj` es visible a partir de $m \geq 4096$, cuando $A$ supera el L3 y el tiling evita los cache misses masivos que sufre el orden sin bloques.

---

### 5.8 Flujo de Fase 1.3 / 1.6: tiled_ikj_avx2 (microkernel BLIS-style 6x16 con AVX2+FMA)

```bash
# Compilar (requiere -O3 -march=znver2 -mavx2 -mfma)
make bench_tiled_ikj_avx2
make validate_tiled_ikj_avx2

# Validar correctitud
./bin/validate_tiled_ikj_avx2_O3 256        # m=256, BS=384 (default)
./bin/validate_tiled_ikj_avx2_O3 256 128    # m=256, BS=128 custom

# Bench individual
./bin/bench_tiled_ikj_avx2_O3 1024          # m=1024, defaults (iters auto, runs=5, BS=384)
./bin/bench_tiled_ikj_avx2_O3 1024 4 1 256  # m=1024, 4 iters, 1 corrida, BS=256
```

Salida CSV (7 columnas, incluye `bs`):
```
tiled_ikj_avx2,1024,128,4,384,X.XXXXXX,X.XXXXXX
```

Para incluir `tiled_ikj_avx2` en el sweep de perf completo y regenerar `results/metrics.csv`:

```bash
make results
```

El target `results` incluye `bench_tiled_ikj_avx2_O3` como dependencia y `run_perf_zen2_sweep.sh` incluye `tiled_ikj_avx2` en su lista de variantes por defecto.

Para correr solo la celda de `tiled_ikj_avx2` sin relanzar todo el sweep:

```bash
VARIANTS="tiled_ikj_avx2" MS="1024 2048" bash scripts/run_perf_zen2_sweep.sh
python3 scripts/consolidate_perf_zen2.py
```

**Geometria del microkernel BLIS-style (Fase 1.6):** $M_R = 6$, $N_R = 16$, $M_C = 192$ fijos en source; $BS$ (= $k_c$) configurable runtime, default $384$. Los $12$ acumuladores YMM del tile $6 \times 16$ de $C$ se mantienen vivos durante toda la pasada $k_c$ (verificado con `objdump` que GCC no spillea ningun YMM). El panel $A$ activo $M_C \times k_c = 192 \times 384$ ocupa $288$ KiB y cabe en el L2 de $512$ KB del Ryzen 5 4600H; el panel $B$ activo $k_c \times N_R = 384 \times 16$ ocupa $24$ KiB y cabe en el L1d de $32$ KB. Cambiar $BS$ via 4.o argumento del bench o `matmul_tiled_ikj_avx2_set_bs(bs)` en runtime (cualquier valor positivo es valido; el microkernel itera $p$ uno a la vez).

---

### 5.9 Flujo de Fase 1.4 / 1.6: tiled_ikj_omp (microkernel 6x16 + OpenMP)

```bash
# Compilar (requiere -O3 -march=znver2 -mavx2 -mfma -fopenmp)
make bench_tiled_ikj_omp
make validate_tiled_ikj_omp

# Validar correctitud
./bin/validate_tiled_ikj_omp_O3 256        # m=256, BS=384 (default)
./bin/validate_tiled_ikj_omp_O3 256 128    # m=256, BS=128 custom

# Bench individual (OMP_NUM_THREADS controla el numero de threads)
OMP_NUM_THREADS=6 OMP_PROC_BIND=close ./bin/bench_tiled_ikj_omp_O3 4096          # m=4096, BS=384, 6 threads close
OMP_NUM_THREADS=6 OMP_PROC_BIND=close ./bin/bench_tiled_ikj_omp_O3 4096 4 5 256  # m, iters, runs, bs
```

Salida CSV (7 columnas, mismo formato que `tiled_ikj_avx2`):
```
tiled_ikj_omp,4096,128,4,384,X.XXXXXX,X.XXXXXX
```

Para incluir `tiled_ikj_omp` en el sweep de perf completo con 6 threads fijos y regenerar `results/metrics.csv`:

```bash
make results
```

El target `results` incluye `bench_tiled_ikj_omp_O3` como dependencia. `run_perf_zen2_sweep.sh` incluye `tiled_ikj_omp` en su lista de variantes y `profile_perf_zen2.sh` fija `OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close` automaticamente para esa variante (cambiado en Fase 1.6 desde $8$/close, que era suboptimo para el microkernel FMA-bound del $6 \times 16$).

Para correr solo la celda de `tiled_ikj_omp` sin relanzar todo el sweep:

```bash
bash scripts/profile_perf_zen2.sh tiled_ikj_omp 4096
python3 scripts/consolidate_perf_zen2.py --out results/metrics.csv
```

**Paralelizacion:** un unico `#pragma omp parallel for schedule(static)` sobre el bucle externo $i_c$ (tiles de filas de altura $M_C = 192$). Cada tile escribe exclusivamente las filas $[i_c, i_c + M_C)$ de $C$; no hay conflictos de escritura entre threads. $A$ y $B$ son `const` y compartidas. El `memset` inicial de $C$ corre fuera de la region paralela. SMT a $12$ threads degrada $\sim 60\%$ porque los dos hilos comparten las pipas FMA del core fisico, por eso el default empirico es $6$ threads.

---

### 5.6 Flujo de Sesion 03: microkernel AVX2 + OpenMP

La Sesion 03 lleva el proyecto al hardware del Ryzen $5$ $4600$H (Zen $2$): microkernel AVX2 + FMA $4 \times 16$, paralelizacion con OpenMP tasks y profiling con eventos PMC de Zen $2$. Todos los targets nuevos viven bajo el bloque `# === Sesion 03 targets ===` del `Makefile` y no tocan los pipelines de Fases $1$ ni $6$.

#### 5.6.1 Targets de un solo comando

```bash
make validate_morton_avx2             # cross-valida la variante AVX2 contra naive y morton
make bench_morton_avx2_O3             # bench single-core del microkernel AVX2
OMP_NUM_THREADS=6 make bench_morton_omp_O3   # version paralela (OpenMP)
make results                          # sweep perf Zen 2 sobre las variantes activas -> results/metrics.csv
make profile_zen2                     # captura eventos perf Zen 2 (group A + group B por celda)
make plot_perf_zen2                   # genera plots/perf_zen2_breakdown.png
```

`make profile_zen2` requiere `kernel.perf_event_paranoid <= 2`. Ajustar una vez por boot con:

```bash
sudo sysctl -w kernel.perf_event_paranoid=1
```

#### 5.6.2 Variables de entorno para `bench_morton_omp`

| Variable | Default util en el $4600$H | Efecto |
|----------|-----------------------------|--------|
| `OMP_NUM_THREADS` | `6` (un thread por core fisico) | $12$ activa SMT, suma rendimiento pero con eficiencia baja. |
| `OMP_PROC_BIND`   | `close` (recomendado)           | Mantiene threads en el mismo CCX. `spread` los reparte entre los $2$ CCXs. |
| `OMP_PLACES`      | `cores`                         | Une cada thread a un core fisico. |

Mejor combinacion empirica para throughput puro: `OMP_NUM_THREADS=12 OMP_PROC_BIND=close` toca $\sim 256$ GFLOPS a $m = 8192$ y $\sim 262$ GFLOPS a $m = 4096$. Para single-CCX limpio (e.g. compartiendo el laptop con otras cargas): `OMP_NUM_THREADS=3 OMP_PROC_BIND=close`.

#### 5.6.3 Reproducir el sweep completo

```bash
source ~/venvs/matmul/bin/activate
sudo sysctl -w kernel.perf_event_paranoid=1

make results                           # ~25 min, sweep perf Zen 2 -> results/metrics.csv
make plot_perf_zen2                    # ~1  min, plots/perf_zen2_breakdown.png
make profile_zen2_omp                  # ~3  min, perf de morton_omp con varios threads
```

Mejor resultado esperado al cierre: `morton_avx2` a $\sim 40$ GFLOPS bench-wide a $m = 8192$ ($\sim 64 \%$ del techo FMA single-core medido con perf), y `morton_omp` a $\sim 260$ GFLOPS con $12$ threads `close`.

---

## 6. Profiling: contadores de hardware con perf

El profiling de hardware counters se hace via `make profile_zen2` (sweep completo) o `make profile_zen2_one VARIANT=<v> M=<m>` (una sola celda). Ambos invocan `scripts/profile_perf_zen2.sh`, que captura los grupos A y B de eventos PMC de Zen 2 por celda `(variant, m)` y deja los `.txt` crudos en `results/perf_<variant>_m<M>_{A,B}.txt`. El consolidador `scripts/consolidate_perf_zen2.py` los une en `results/metrics.csv`.

Eventos cubiertos (grupos A + B):

| Pregunta | Eventos de perf |
|----------|-----------------|
| Numero de instrucciones | `instructions` |
| IPC promedio | `instructions` / `cycles` |
| FMA throughput | `fp_ret_sse_avx_ops.all` (uops AVX/FMA retirados) |
| Cache misses | `l1d_misses`, `l2_misses`, `l3_misses` |
| TLB walks | `dtlb_walks`, `itlb_walks` |
| Branching | `branches`, `branch-misses` |
| Fallos de pagina | `page-faults`, `minor-faults`, `major-faults` |

`make profile_zen2` requiere `kernel.perf_event_paranoid <= 2`. Ajustar una vez por boot con `sudo sysctl -w kernel.perf_event_paranoid=1`.

Si un evento aparece como `<not supported>` en WSL2 es normal (limitacion del hipervisor); el script lo registra y continua.

### 6.1 (Opcional) Cachegrind

```bash
valgrind --tool=cachegrind --cache-sim=yes ./bin/bench_naive_O3 1024 1
ls cachegrind.out.*
cg_annotate cachegrind.out.<pid> | less
```

Cachegrind simula L1/LL caches y genera fallos por **linea de codigo**. Mucho mas lento que `perf` pero util cuando quieres saber exactamente que linea del kernel produce los misses.

---

## 7. Reproducibilidad: estabilizar las mediciones

Antes de tomar mediciones para el reporte:

### 7.1 Documenta el hardware

```bash
lscpu                              # modelo, frecuencias, caches
cat /proc/cpuinfo | grep "model name" | head -1
cat /sys/devices/system/cpu/cpu0/cache/index{0,1,2,3}/size
free -h                            # RAM total
```

Guarda esta salida junto con tus resultados.

### 7.2 Fija el gobernador de CPU (si tu instalacion lo permite)

WSL2 hereda el gobernador del host. En Windows, asegurate que el plan de energia este en "Maximo rendimiento":

```powershell
powercfg /list
powercfg /setactive SCHEME_MIN   # Maximum performance
```

### 7.3 Cierra programas pesados

Cualquier proceso (Chrome con 80 pestanas, Slack, Zoom, Docker Desktop, OneDrive sincronizando) compite por L3 y memoria y mete ruido. Cierralos antes de medir.

### 7.4 Reporta la mediana

`bench_naive_O3` ya hace cinco corridas y reporta la mediana. Si quieres ser mas estricto, edita `DEFAULT_RUNS` en `src/drivers/bench/bench_naive.c` y recompila. El sweep `make results` controla el numero de runs medidos via el knob `RUNS` (default 3, o 1 cuando `ITERS_PER_RUN=0` activa `I_full`).

---

## 8. Flujo de trabajo completo

```bash
# 1. Compilar todo
make build

# 2. Verificar correctitud de las 8 variantes
make validate_all

# 3. Sweep perf + consolidacion
make results                                        # defaults completos
# o con knobs:
make results MS="1024 4096" VARIANTS="naive loop_ikj morton_avx2 tiled_ikj_avx2"

# 4. (Opcional) Plots de los contadores perf
source ~/venvs/matmul/bin/activate
make plot_perf_zen2                                 # plots/perf_zen2_breakdown.png
```

Al terminar, en `results/` esta `metrics.csv` con una fila por celda `(variant, m)` y todos los contadores; en `plots/` los PNG correspondientes.

---

## 9. Que viene despues (no incluido en esta fase)

Las fases siguientes mantendran la misma API descrita en `docs/API.md` y se sumaran como modulos independientes:

| Fase | Que se agregara | Estado |
|------|-----------------|--------|
| 1.1 | Reordenamiento de bucles (6 ordenes seleccionables por nombre) | **COMPLETADO** (`matmul_loops`, integrado en `make results` y sweep perf Zen 2) |
| 1.2 | Tiling explicito de un nivel para L2 | **COMPLETADO** (`matmul_tiled_ikj`, $M_c = K_c = 256$ apuntando al L2 del $4600$H) |
| 1.3 / 1.6 | Tiling con AVX2+FMA y microkernel BLIS-style | **COMPLETADO** (`tiled_ikj_avx2`: microkernel inline $6 \times 16$, $M_C = 192$, $BS = 384$ default; integrado en `make results`) |
| 1.4 / 1.6 | OpenMP sobre microkernel $6 \times 16$ | **COMPLETADO** (`tiled_ikj_omp`: `#pragma omp parallel for` en bucle $i_c$, $6$ threads `close` default) |
| 4 | Flags de compilador y auto-vectorizacion (`-O3 -march=native`) | **COMPLETADO** como parte de la Sesion 03 (microkernel AVX2 + FMA explicito sobre Zen $2$) |
| 5 | OpenMP + comparacion con OpenBLAS | OpenMP **COMPLETADO** (Sesion 03 `matmul_morton_omp`; Fase 1.4/1.6 `tiled_ikj_omp`); comparacion contra OpenBLAS pendiente para Sesion 04 |
| 6 / Opcional | Matmul recursivo cache-oblivious + layout Morton sobre $A$ | **COMPLETADO** (Sesion 02 codigo y validacion; Sesion 03 sweep masivo, perf compare, Roofline) |

El proyecto **esta disenado para que cada fase se entregue de forma incremental** y se pueda comparar contra el baseline producido aqui.

---

## 10. Preguntas frecuentes

**No tengo WSL2, solo tengo Windows.**
Necesitas instalarlo. Es la opcion acordada por compatibilidad con `gprof` y `perf`. La instalacion son dos comandos en PowerShell (seccion 3.1).

**perf no se compila / no encuentra el kernel.**
WSL2 usa un kernel propio de Microsoft. La opcion mas robusta es compilar `perf` desde el repositorio `WSL2-Linux-Kernel` como muestra la seccion 3.3. Si no es viable, `cachegrind` (Seccion 6.1) sigue siendo accesible para diagnostico fino.

**El sweep tarda mucho.**
Reduce el alcance con los knobs: `make results MS="1024 2048" VARIANTS="naive morton_avx2"` mientras desarrollas. El sweep completo se lanza una vez al final.

**Quiero medir tambien con `-O3`.**
Eso es parte de la Fase 4. Aqui no se hace para mantener el baseline limpio y el paso 1 explicito.

**Como cambio de `float` a `double`?**
Edita `typedef float scalar_t;` en `src/core/matrix_utils.h` (alias compartido) y revisa las constantes `ABS_TOL`, `REL_TOL` en `src/drivers/validate/validate_naive.c`. La API queda igual gracias al alias.
