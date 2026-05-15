# Benchmark de Multiplicacion Iterada de Matrices

**Curso:** Arquitectura de Computadores
**Universidad:** Universidad Nacional de Colombia, Sede Medellin
**Fecha:** Mayo 2026
**Estado:** Fase 1 (baseline + profiling + escalamiento con $m$)

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
|-- README.md               -> Este archivo
|-- Makefile                -> Targets de compilacion y profiling
|-- docs/
|   `-- API.md              -> Contrato publico de las funciones
|-- src/
|   |-- matmul_naive.h      -> Declaraciones del kernel y typedef scalar_t
|   |-- matmul_naive.c      -> Kernel ijk + orquestador de iteraciones
|   |-- matrix_utils.h      -> Helpers (alocacion, init, comparacion)
|   |-- matrix_utils.c      -> Implementacion de los helpers
|   |-- timing.h            -> clock_gettime(CLOCK_MONOTONIC) inline
|   |-- bench_naive.c       -> Driver de medicion (binario bench_naive_O0)
|   `-- validate_naive.c    -> Verificador algebraico (binario validate_naive_O0)
|-- scripts/
|   |-- run_sweep_naive.sh        -> Corre el benchmark variando m, escribe CSV
|   |-- profile_gprof_naive.sh    -> Lanza gmon.out y produce el reporte de gprof
|   |-- profile_perf_naive.sh     -> Captura contadores de hardware con perf
|   `-- plot_results.py           -> Genera graficas a partir del CSV
|-- results/                -> CSV y reportes de profiling (gitignore)
|-- plots/                  -> Imagenes generadas (gitignore)
`-- bin/                    -> Binarios compilados (gitignore)
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

Targets individuales:

```bash
make bench_naive_O0      # solo el benchmark
make validate_naive      # solo el verificador
make bench_naive_pg      # version con -pg para gprof, paso 2
make clean               # borra bin/ y build/
make distclean           # clean + borra results/*.csv y plots/*
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

`bench_naive_O0` ya hace cinco corridas y reporta la mediana. Si quieres ser mas estricto, edita `DEFAULT_RUNS` en `src/bench_naive.c` y recompila.

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

| Fase | Que se agregara |
|------|-----------------|
| 2 | Reordenamiento de bucles (ikj, kij) + pre-transposicion de $A$ |
| 3 | Tiling de un nivel para L2 + padding anti-conflict-misses |
| 4 | Flags de compilador y auto-vectorizacion (`-O3 -march=native`) |
| 5 | OpenMP + comparacion con OpenBLAS |
| Opcional | Layout Morton sobre $A$ + matmul recursivo |

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
Edita `typedef float scalar_t;` en `src/matmul_naive.h` y revisa las constantes `ABS_TOL`, `REL_TOL` en `src/validate_naive.c`. La API queda igual gracias al alias.
