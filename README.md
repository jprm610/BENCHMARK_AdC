# Benchmark de Multiplicacion Iterada de Matrices

**Curso:** Arquitectura de Computadores
**Universidad:** Universidad Nacional de Colombia, Sede Medellin
**Fecha:** Mayo 2026
**Estado:** Fase 1 cerrada (baseline + profiling + escalamiento con $m$); Fase 1.1 cerrada (reordenamiento de bucles, 6 variantes); Fase 1.2 cerrada (tiling explicito `ikj` apuntando a L2); Fase 1.3 cerrada (`tiled_ikj_avx512`, integrada en `make results`); Fase 1.4 cerrada (`tiled_ikj_omp`, integrada en `make results` y sweep perf); Fase 1.6 cerrada (microkernel BLIS-style $6 \times 16$ inline para `tiled_ikj_avx512` y `tiled_ikj_omp`, $M_R = 6$, $N_R = 16$, $M_C = 192$, $BS = 384$ por defecto, default OMP a 6 threads `close`); Fase 1.6.1 cerrada (rename `tiled` $\to$ `tiled_ikj` por consistencia); Fase 6 cerrada (Morton Z-order cache-oblivious); Sesion 03 cerrada (microkernel AVX2 + FMA, OpenMP tasks, perf Zen 2, Roofline anclado al $4600$H).

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
|   |-- microkernels/                      -> Kernels AVX-512 header-only (static inline)
|   |   |-- kernel_avx512_morton.h         -> Microkernel 4x32 AVX-512 (familia Morton, Zen 5)
|   |   `-- kernel_avx512_tiled.h          -> Microkernel 6x32 AVX-512 + residual (familia tiled_ikj, Zen 5)
|   |-- algorithms/                        -> Una carpeta por familia de algoritmo
|   |   |-- naive/matmul_naive.{h,c}       -> Baseline ijk
|   |   |-- loops/matmul_loops.{h,c}       -> 6 ordenes de loop con lookup por nombre (Fase 1.1)
|   |   |-- morton/                        -> Fase 6: kernel recursivo + AVX-512 + OMP
|   |   |   |-- matmul_morton.{h,c}            -> Kernel recursivo con A en Morton fino (A3)
|   |   |   |-- matmul_morton_avx512.{h,c}     -> Morton-de-bloques + microkernel 4x32 (Zen 5)
|   |   |   `-- matmul_morton_omp.{h,c}        -> Variante paralela OpenMP tasks (Zen 5)
|   |   `-- tiled_ikj/                     -> Fase 1.2-1.6: tiling explicito + BLIS 6x32 + OpenMP
|   |       |-- matmul_tiled_ikj.{h,c}         -> Tiling Mc x Kc sobre ikj, apunta a L2 (escalar)
|   |       |-- matmul_tiled_ikj_avx512.{h,c}  -> 6x32 AVX-512 (MR=6, NR=32, MC=288), BS=256 default
|   |       `-- matmul_tiled_ikj_omp.{h,c}     -> 6x32 AVX-512 + #pragma omp parallel for schedule(static) en ic
|   |-- drivers/                           -> Programas main: medicion (bench) y verificacion (validate)
|   |   |-- bench/                         -> bench_naive.c, bench_loops.c, bench_morton{,_avx512,_omp}.c, bench_tiled_ikj{,_avx512,_omp}.c
|   |   `-- validate/                      -> validate_naive.c, validate_loops.c, validate_morton{,_avx512,_omp}.c, validate_tiled_ikj{,_avx512,_omp}.c
|   `-- tests/                             -> Tests unitarios standalone (ver docs/1.9) tests.md)
|       |-- test_matrix_utils.c            -> xalloc_aligned, init_matrix_*, matrices_close
|       |-- test_morton.c                  -> Round-trip encode/decode + contiguidad de cuadrantes
|       |-- test_kernel_avx512_morton.c    -> Microkernel 4x32 (kernel_avx512_morton.h)
|       `-- test_kernel_avx512_tiled.c     -> Microkernel 6x32 (kernel_avx512_tiled.h)
|-- scripts/
|   |-- profile_perf_zen5.sh           -> Captura perf por celda (variant, m)
|   |-- run_perf_zen5_sweep.sh         -> Orquesta las celdas (variantes x tamaños)
|   |-- consolidate_perf_zen5.py       -> Consolida grupos A+B -> results/metrics.csv
|   `-- plot_perf_zen2.py              -> 4 paneles: IPC, FMA, L3 miss, TLB walks
|-- results/                            -> CSV y reportes de profiling (gitignored)
|-- plots/                              -> Imagenes generadas (gitignored salvo perf_zen2 / roofline)
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

**Importante para WSL2:** los contadores de PMU disponibles dependen del soporte del hipervisor. En la practica funcionan al menos `instructions`, `cycles`, `branches`, `branch-misses`, `task-clock`, `page-faults`. Si algun evento devuelve `<not supported>`, no es un error tuyo, simplemente el evento no esta expuesto. El script `profile_perf_naive.sh` ignora esos eventos.

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

Recuerda activar el venv (`source ~/venvs/matmul/bin/activate`) cada vez que abras una nueva terminal antes de ejecutar `scripts/plot_results.py`.

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
make bench_naive_O0      # solo el benchmark baseline
make validate_naive      # solo el verificador baseline
make bench_naive_pg      # version con -pg para gprof, paso 2
make clean               # borra bin/ y build/
make distclean           # clean + borra results/*.csv y plots/*
```

Targets de Fase 1.1 (loop reorder), Fase 1.2 (tiling) y Fase 1.3 (tiled_ikj_avx512):

```bash
# Fase 1.1 - loop reorder
make bench_loops              # bin/bench_loops_O0 y bin/bench_loops_O3
make validate_loops           # bin/validate_loops_O0

# Fase 1.2 - tiling explicito
make bench_tiled_ikj              # bin/bench_tiled_ikj_O3
make validate_tiled_ikj           # bin/validate_tiled_ikj_O0

# Fase 1.3 - tiled_ikj_avx512 (6-loop tiling con AVX-512+FMA, compilado con -O3 -march=native)
make bench_tiled_ikj_avx512_ZEN5    # bin/bench_tiled_ikj_avx512_ZEN5
make validate_tiled_ikj_avx512_ZEN5 # bin/validate_tiled_ikj_avx512_ZEN5

# Fase 1.4 - tiled_ikj_omp (tiled_ikj_avx512 + OpenMP parallel for, compilado con -O3 -march=native -fopenmp)
make bench_tiled_ikj_omp          # bin/bench_tiled_ikj_omp_O3
make validate_tiled_ikj_omp       # bin/validate_tiled_ikj_omp_O3
```

Targets de Fase 6 (Morton Z-order cache-oblivious):

```bash
make bench_morton             # bin/bench_morton_O0   (m debe ser potencia de 2)
make validate_morton          # bin/validate_morton_O0 (idem)
```

Unit tests (capa por debajo de validate, ver [`docs/1.9) tests.md`](docs/1.9\)%20tests.md)):

```bash
make tests                       # bin/tests/* — los 4 unit-tests, en orden
make test_matrix_utils           # solo bin/tests/test_matrix_utils
make test_morton                 # solo bin/tests/test_morton
make test_kernel_avx512_morton   # solo bin/tests/test_kernel_avx512_morton
make test_kernel_avx512_tiled    # solo bin/tests/test_kernel_avx512_tiled
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

- `-O0`: requisito del proyecto. Sin optimizacion del compilador.
- `-g`: simbolos de debug, necesarios para que `gprof` y `perf report` muestren nombres legibles.
- `-fno-omit-frame-pointer`: deja el stack pointer en su sitio para que las herramientas de profiling resuelvan call graphs sin DWARF unwinding.

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

### 5.3 Sweep completo: paso 3 del proyecto

```bash
make sweep_naive
```

Equivalente a `bash scripts/run_sweep_naive.sh`. Corre el benchmark para los valores por defecto $m \in \{256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192\}$ y por cada uno produce **tres archivos**:

1. Una linea en `results/naive_O0.csv` con la medicion de gflops.
2. Un reporte `results/gprof_naive_m<m>.txt` (perfil por funcion).
3. Un reporte `results/perf_naive_m<m>.txt` (contadores de hardware).

Asi tienes registro completo de paso 1 (timing baseline), paso 2 (profiling) y paso 3 (escalamiento) en una sola corrida.

Para un rango custom:

```bash
bash scripts/run_sweep_naive.sh "256 512 1024 2048"
```

Controlar el profiling:

```bash
PROFILING=0     bash scripts/run_sweep_naive.sh                  # solo CSV, sin profiling
PROFILING=gprof bash scripts/run_sweep_naive.sh                  # CSV + solo gprof
PROFILING=perf  bash scripts/run_sweep_naive.sh                  # CSV + solo perf
PROFILE_ITERS=2 PROFILE_RUNS=1 bash scripts/run_sweep_naive.sh   # mas iteraciones para el profile
```

Por defecto el profiling usa `iters=1, runs=1` (una sola corrida determinista) para no multiplicar los tiempos. La medicion del CSV sigue usando 5 corridas internas con mediana para tener un valor estadisticamente estable.

**Nota sobre tiempos esperados a `-O0`:** una corrida completa de baseline a `-O0` con $m = 8192$ puede tomar varios minutos. Para el reporte final, el sweep completo del rango por defecto puede tardar entre 1 y 2 horas dependiendo del hardware. Conviene lanzarlo y dejarlo correr. Si solo necesitas el CSV (sin profiling), usa `PROFILING=0` para reducir el tiempo al minimo.

### 5.4 Graficas a partir del CSV

Con el venv activado (`source ~/venvs/matmul/bin/activate`) y el CSV ya generado:

```bash
python3 scripts/plot_results.py
```

Produce dos PNG en `plots/`:

- `plots/naive_gflops_vs_m.png`: gflops sostenidos vs $m$, con marcas verticales para las transiciones de cache.
- `plots/naive_time_vs_m.png`: tiempo por iteracion en escala log-log, con la curva teorica $2 m^2 n$ anclada en el $m$ mas pequeno. Sirve para comparar con la complejidad esperada.

**Defaults calibrados para AMD Ryzen 5 4600H** (la maquina de pruebas inicial):

- L1d: 32 KB por core (192 KiB totales / 6 cores)
- L2 : 512 KB por core (3 MiB totales / 6 cores)
- L3 : 4 MB compartida (4 MiB / 1 instancia)

Si corres en otra maquina personaliza los argumentos:

```bash
python3 scripts/plot_results.py \
    --cpu-label "Intel Core i7-XXXX" \
    --l1-kb 32 --l2-kb 1024 --l3-kb 8192
```

Para conocer los tamanos exactos de tu maquina:

```bash
lscpu | grep -E "cache|Model name"
# o, mas explicito:
getconf -a | grep CACHE
```

**Importante:** los valores que pides en el script son **por core** para L1 y L2, y **totales (compartido)** para L3. `lscpu` reporta el total de L1/L2 sumado a traves de los cores ("192 KiB (6 instances)"); divide entre el numero de instancias para sacar el valor por core.

### 5.5 Flujo de Fase 6: Morton (Z-order) cache-oblivious

#### 5.5.1 Validacion

```bash
./bin/validate_morton_O0    256   # 3 invariantes + cross-validation contra naive (m potencia de 2)
./bin/tests/test_morton           # tests del modulo Morton (encode/decode/reorganize)
```

Cada uno imprime `VALIDATION OK` (o `MORTON TESTS OK`) y retorna 0 cuando todo pasa.

#### 5.5.2 Bench individual

```bash
./bin/bench_morton_O0    1024 4 1        # m debe ser potencia de 2
```

Misma CLI y mismo CSV de salida que `bench_naive_O0`. `bench_morton_O0` ejecuta `reorganize_to_morton(A)` una sola vez antes del warm-up, fuera del tiempo medido, para que las GFLOP/s reflejen solo el kernel.

#### 5.5.3 Sweep individual

```bash
make sweep_morton_run                         # produce results/morton_O0.csv
# equivalente a:
bash scripts/run_sweep_morton.sh              # 4 potencias de 2, ~50 min a -O0
```

Para un rango custom:

```bash
bash scripts/run_sweep_morton.sh    "1024 2048 4096 8192"
```

`run_sweep_morton.sh` filtra y omite con warning a stderr cualquier $m$ que no sea potencia de 2.

#### 5.5.4 Comparacion entre kernels

Para comparar Morton contra el resto del pipeline (naive, loops, tiled_ikj*, morton_avx512, morton_omp) se usa el pipeline unificado de la Sesion 03 (seccion 5.6):

```bash
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'   # una vez por boot
make results              # sweep perf Zen 2 sobre 13 variantes -> results/metrics.csv
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

El target `results` ya incluye `bench_tiled_ikj_O3` como dependencia y `run_perf_zen5_sweep.sh`
incluye `tiled_ikj` en su lista de variantes por defecto.

Para correr solo la celda de tiling sin relanzar todo el sweep:

```bash
VARIANTS="tiled_ikj" MS="1024 2048" bash scripts/run_perf_zen5_sweep.sh
python3 scripts/consolidate_perf_zen5.py
```

**Tamanos de tile:** `Mc = Kc = 256`, elegidos para que los tres panels activos (A: 256 KB, B: 128 KB, C: 128 KB) llenen exactamente el L2 de 512 KB del Ryzen 5 4600H. El beneficio sobre `loop_ikj` es visible a partir de $m \geq 4096$, cuando $A$ supera el L3 y el tiling evita los cache misses masivos que sufre el orden sin bloques.

---

### 5.8 Flujo de Fase 1.3 / 1.6: tiled_ikj_avx512 (microkernel BLIS-style 6x32 con AVX-512+FMA)

```bash
# Compilar (requiere -O3 -march=native)
make bench_tiled_ikj_avx512
make validate_tiled_ikj_avx512

# Validar correctitud
./bin/validate_tiled_ikj_avx512_ZEN5 256        # m=256, BS=384 (default)
./bin/validate_tiled_ikj_avx512_ZEN5 256 128    # m=256, BS=128 custom

# Bench individual
./bin/bench_tiled_ikj_avx512_ZEN5 1024          # m=1024, defaults (iters auto, runs=5, BS=384)
./bin/bench_tiled_ikj_avx512_ZEN5 1024 4 1 256  # m=1024, 4 iters, 1 corrida, BS=256
```

Salida CSV (7 columnas, incluye `bs`):
```
tiled_ikj_avx512,1024,128,4,384,X.XXXXXX,X.XXXXXX
```

Para incluir `tiled_ikj_avx512` en el sweep de perf completo y regenerar `results/metrics.csv`:

```bash
make results
```

El target `results` incluye `bench_tiled_ikj_avx512_ZEN5` como dependencia y `run_perf_zen5_sweep.sh` incluye `tiled_ikj_avx512` en su lista de variantes por defecto.

Para correr solo la celda de `tiled_ikj_avx512` sin relanzar todo el sweep:

```bash
VARIANTS="tiled_ikj_avx512" MS="1024 2048" bash scripts/run_perf_zen5_sweep.sh
python3 scripts/consolidate_perf_zen5.py
```

**Geometria del microkernel BLIS-style (Fase 1.6):** $M_R = 6$, $N_R = 16$, $M_C = 192$ fijos en source; $BS$ (= $k_c$) configurable runtime, default $384$. Los $12$ acumuladores YMM del tile $6 \times 16$ de $C$ se mantienen vivos durante toda la pasada $k_c$ (verificado con `objdump` que GCC no spillea ningun YMM). El panel $A$ activo $M_C \times k_c = 192 \times 384$ ocupa $288$ KiB y cabe en el L2 de $512$ KB del Ryzen 5 4600H; el panel $B$ activo $k_c \times N_R = 384 \times 16$ ocupa $24$ KiB y cabe en el L1d de $32$ KB. Cambiar $BS$ via 4.o argumento del bench o `matmul_tiled_ikj_avx512_set_bs(bs)` en runtime (cualquier valor positivo es valido; el microkernel itera $p$ uno a la vez).

---

### 5.9 Flujo de Fase 1.4 / 1.6: tiled_ikj_omp (microkernel 6x16 + OpenMP)

```bash
# Compilar (requiere -O3 -march=native -fopenmp)
make bench_tiled_ikj_omp
make validate_tiled_ikj_omp

# Validar correctitud
./bin/validate_tiled_ikj_omp_O3 256        # m=256, BS=384 (default)
./bin/validate_tiled_ikj_omp_O3 256 128    # m=256, BS=128 custom

# Bench individual (OMP_NUM_THREADS controla el numero de threads)
OMP_NUM_THREADS=6 OMP_PROC_BIND=close ./bin/bench_tiled_ikj_omp_O3 4096          # m=4096, BS=384, 6 threads close
OMP_NUM_THREADS=6 OMP_PROC_BIND=close ./bin/bench_tiled_ikj_omp_O3 4096 4 5 256  # m, iters, runs, bs
```

Salida CSV (7 columnas, mismo formato que `tiled_ikj_avx512`):
```
tiled_ikj_omp,4096,128,4,384,X.XXXXXX,X.XXXXXX
```

Para incluir `tiled_ikj_omp` en el sweep de perf completo con 6 threads fijos y regenerar `results/metrics.csv`:

```bash
make results
```

El target `results` incluye `bench_tiled_ikj_omp_O3` como dependencia. `run_perf_zen5_sweep.sh` incluye `tiled_ikj_omp` en su lista de variantes y `profile_perf_zen5.sh` fija `OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close` automaticamente para esa variante (cambiado en Fase 1.6 desde $8$/close, que era suboptimo para el microkernel FMA-bound del $6 \times 16$).

Para correr solo la celda de `tiled_ikj_omp` sin relanzar todo el sweep:

```bash
bash scripts/profile_perf_zen5.sh tiled_ikj_omp 4096
python3 scripts/consolidate_perf_zen5.py --out results/metrics.csv
```

**Paralelizacion:** un unico `#pragma omp parallel for schedule(static)` sobre el bucle externo $i_c$ (tiles de filas de altura $M_C = 192$). Cada tile escribe exclusivamente las filas $[i_c, i_c + M_C)$ de $C$; no hay conflictos de escritura entre threads. $A$ y $B$ son `const` y compartidas. El `memset` inicial de $C$ corre fuera de la region paralela. SMT a $12$ threads degrada $\sim 60\%$ porque los dos hilos comparten las pipas FMA del core fisico, por eso el default empirico es $6$ threads.

---

### 5.6 Flujo de Sesion 03: microkernel AVX2 + OpenMP + Roofline

La Sesion 03 lleva el proyecto al hardware del Ryzen $5$ $4600$H (Zen $2$): microkernel AVX2 + FMA $4 \times 16$, paralelizacion con OpenMP tasks, profiling con eventos PMC de Zen $2$ y Roofline anclado al bandwidth STREAM medido. Todos los targets nuevos viven bajo el bloque `# === Sesion 03 targets ===` del `Makefile` y no tocan los pipelines de Fases $1$ ni $6$.

#### 5.6.1 Targets de un solo comando

```bash
make audit                            # auditoria PDEP/PEXT (imprime PASS / FAIL)
make hwinfo                           # bin/hwinfo: caracteristicas del CPU en runtime
make sweep_threshold                  # mide el RECURSION_THRESHOLD optimo de Morton
make validate_morton_avx512             # cross-valida la variante AVX-512 contra naive y morton
make bench_morton_avx512                # bench single-core del microkernel AVX-512
OMP_NUM_THREADS=6 make bench_morton_omp   # version paralela (OpenMP tasks)
make results                          # sweep perf Zen 5 sobre 13 variantes -> results/metrics.csv
make stream                           # mide DRAM bandwidth con STREAM (Triad 1T y 8T)
make profile_zen5                     # captura eventos perf Zen 5 (group A + group B por celda)
make plot_roofline                    # genera plots/roofline_9R45.png anclado al STREAM medido
```

`make audit` debe ejecutarse antes de cualquier bench: BMI2 en Zen $2$ esta microcodeado ($\sim 18$ ciclos para `PDEP`/`PEXT`) y un uso incidental degradaria el throughput sin notarlo. El script verifica que ningun modulo Morton emite `pdep` ni `pext` en el ensamblador.

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

Mejor combinacion empirica para throughput puro (sweep de Prompt $6$, `results/omp_scaling.csv`): `OMP_NUM_THREADS=12 OMP_PROC_BIND=close` toca $\sim 256$ GFLOPS a $m = 8192$ y $\sim 262$ GFLOPS a $m = 4096$. Para single-CCX limpio (e.g. compartiendo el laptop con otras cargas): `OMP_NUM_THREADS=3 OMP_PROC_BIND=close`.

#### 5.6.3 Reproducir el Roofline completo

```bash
source ~/venvs/matmul/bin/activate
sudo sysctl -w kernel.perf_event_paranoid=1

make audit                            # PASS
make results                           # ~25 min, sweep perf Zen 2 -> results/metrics.csv
make plot_perf_zen2                    # ~1  min, plots/perf_zen2_breakdown.png
make stream                            # ~2  min
make profile_zen2_omp                  # ~3  min, perf de morton_omp para el Roofline
make plot_roofline                     # < 1 min, plots/roofline_4600h.png
```

Mejor resultado esperado al cierre: `morton_avx512` a $\sim 40$ GFLOPS bench-wide a $m = 8192$ ($\sim 64 \%$ del techo FMA single-core medido con perf), y `morton_omp` a $\sim 260$ GFLOPS con $12$ threads `close` segun `omp_scaling.csv`. El reporte completo de hallazgos esta en [`docs/SESION_03_RESUMEN.md`](docs/SESION_03_RESUMEN.md).

---

## 6. Paso 2: profiling

**Atajo:** `make sweep_naive` corre los tres (CSV + gprof + perf) por cada valor de $m$ automaticamente. Las dos subsecciones siguientes describen como correr cada profiler por separado para un solo $m$, util durante el desarrollo o para inspeccionar un cliff concreto.

### 6.1 Perfil por funcion con gprof

```bash
make bench_naive_pg                              # compila bench con -pg
bash scripts/profile_gprof_naive.sh              # m=2048, iters=1, runs=1 por defecto
bash scripts/profile_gprof_naive.sh 1024         # m custom
bash scripts/profile_gprof_naive.sh 1024 2 1     # m, iteraciones, corridas medidas
```

El reporte queda en `results/gprof_naive_m<M>.txt`. Es esperable que **mas del 95% del tiempo** caiga en `matmul_naive`; eso confirma que esa funcion es el cuello de botella.

Para inspeccionar manualmente:

```bash
gprof bin/bench_naive_pg gmon.out > results/gprof_manual.txt
less results/gprof_manual.txt
```

### 6.2 Contadores de hardware con perf

```bash
bash scripts/profile_perf_naive.sh               # m=2048, iters=1, runs=1
bash scripts/profile_perf_naive.sh 1024          # m custom
bash scripts/profile_perf_naive.sh 1024 2 1      # m, iteraciones, corridas
```

El reporte queda en `results/perf_naive_m<M>.txt`. Los eventos solicitados cubren los cuatro puntos del paso 2:

| Pregunta del proyecto | Eventos de perf |
|------------------------|-----------------|
| Numero de instrucciones | `instructions` |
| IPC promedio            | `instructions` / `cycles` |
| Fallos de pagina        | `page-faults`, `minor-faults`, `major-faults` |
| Efectividad del branching | `branches`, `branch-misses` |
| (Bonus) memoria         | `cache-references`, `cache-misses`, `L1-dcache-load-misses`, `LLC-load-misses` |

**Interpretacion tipica a `-O0`:** IPC bajo (entre 0.3 y 0.8 sobre un peak teorico de 4-6), branch miss rate $< 0.5\%$, mucha actividad en cache misses. El cuello de botella es memoria, no CPU.

Si un evento aparece como `<not supported>` en WSL2 es normal (limitacion del hipervisor). Los eventos basicos (`instructions`, `cycles`, `branches`, `branch-misses`, `page-faults`) suelen funcionar.

### 6.3 (Opcional) Cachegrind

```bash
valgrind --tool=cachegrind --cache-sim=yes ./bin/bench_naive_O0 1024 1
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

`bench_naive_O0` ya hace cinco corridas y reporta la mediana. Si quieres ser mas estricto, edita `DEFAULT_RUNS` en `src/drivers/bench/bench_naive.c` y recompila.

---

## 8. Flujo de trabajo completo para la entrega de la Fase 1

Resumen de un ciclo completo para los **tres primeros pasos**:

```bash
# 1. Compilar todo (incluye bench_naive_O0, bench_naive_pg y validate_naive_O0)
make
make bench_naive_pg

# 2. Verificar correctitud
./bin/validate_naive_O0 256

# 3. Paso 1, 2 y 3 de una sola pasada:
#    - CSV con gflops vs m (paso 3)
#    - gprof por cada m (paso 2.a)
#    - perf por cada m (paso 2.b)
make sweep_naive

# 4. Generar las graficas (paso 3)
source ~/venvs/matmul/bin/activate
python3 scripts/plot_results.py
```

Al terminar, en `results/` y `plots/` deberian estar:

- `results/naive_O0.csv` (datos del sweep)
- `results/gprof_naive_m256.txt`, `results/gprof_naive_m384.txt`, ..., `results/gprof_naive_m8192.txt`
- `results/perf_naive_m256.txt`, `results/perf_naive_m384.txt`, ..., `results/perf_naive_m8192.txt`
- `plots/naive_gflops_vs_m.png`
- `plots/naive_time_vs_m.png`

---

## 9. Que viene despues (no incluido en esta fase)

Las fases siguientes mantendran la misma API descrita en `docs/API.md` y se sumaran como modulos independientes:

| Fase | Que se agregara | Estado |
|------|-----------------|--------|
| 1.1 | Reordenamiento de bucles (6 ordenes seleccionables por nombre) | **COMPLETADO** (`matmul_loops`, integrado en `make results` y sweep perf Zen 2) |
| 1.2 | Tiling explicito de un nivel para L2 | **COMPLETADO** (`matmul_tiled_ikj`, $M_c = K_c = 256$ apuntando al L2 del $4600$H) |
| 1.3 / 1.6 | Tiling con AVX2+FMA y microkernel BLIS-style | **COMPLETADO** (`tiled_ikj_avx512`: microkernel inline $6 \times 16$, $M_C = 192$, $BS = 384$ default; integrado en `make results`) |
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
WSL2 usa un kernel propio de Microsoft. La opcion mas robusta es compilar `perf` desde el repositorio `WSL2-Linux-Kernel` como muestra la seccion 3.3. Si no es viable, los tres puntos del paso 2 que se pueden medir tambien con `gprof` y `cachegrind` siguen siendo accesibles.

**El sweep tarda mucho.**
A `-O0` es esperable. Reduce el rango con `bash scripts/run_sweep_naive.sh "256 512 1024 2048"` mientras desarrollas. El sweep completo se lanza una vez al final.

**Quiero medir tambien con `-O3`.**
Eso es parte de la Fase 4. Aqui no se hace para mantener el baseline limpio y el paso 1 explicito.

**Como cambio de `float` a `double`?**
Edita `typedef float scalar_t;` en `src/core/matrix_utils.h` (alias compartido) y revisa las constantes `ABS_TOL`, `REL_TOL` en `src/drivers/validate/validate_naive.c`. La API queda igual gracias al alias.
