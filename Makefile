#
# Makefile for the iterated matmul benchmark - Phase 1 (naive baseline).
#
# Targets:
#   make                  -> bench_naive_O0 and validate_naive_O0 (default)
#   make bench_naive_O0   -> baseline benchmark binary (no compiler optimization)
#   make bench_naive_pg   -> same as bench_naive_O0 but compiled with -pg for gprof
#   make validate_naive   -> validate_naive_O0 binary that runs the algebraic sanity tests
#   make sweep_naive      -> runs scripts/run_sweep_naive.sh after building bench_naive_O0
#   make clean            -> remove binaries and object files (keeps CSV/plots)
#   make distclean        -> clean plus remove results/*.csv and plots/*
#
# The point of having both bench_naive_O0 and bench_naive_pg as separate binaries
# is so that the gprof instrumentation does not contaminate timing on the
# non-profiled benchmark.
#

CC      := gcc
CSTD    := -std=c11
WARN    := -Wall -Wextra -Wpedantic
INCS    := -Isrc
LIBS    := -lm

# -O0 is mandatory for Phase 1. -g for debugging symbols and meaningful
# function names in gprof/perf reports. -fno-omit-frame-pointer makes
# perf call-graphs work without DWARF unwinding.
# _POSIX_C_SOURCE=200809L is required so that glibc exposes clock_gettime
# and CLOCK_MONOTONIC under -std=c11 (which otherwise hides POSIX-only
# symbols).
BASE_CFLAGS := $(CSTD) $(WARN) $(INCS) -O0 -g -fno-omit-frame-pointer \
               -D_POSIX_C_SOURCE=200809L

SRC_DIR  := src
BIN_DIR  := bin
OBJ_DIR  := build

# Common module list (kernel + helpers) used by both bench and validate.
COMMON_SRCS := $(SRC_DIR)/matmul_naive.c $(SRC_DIR)/matrix_utils.c

BENCH_SRCS    := $(COMMON_SRCS) $(SRC_DIR)/bench_naive.c
VALIDATE_SRCS := $(COMMON_SRCS) $(SRC_DIR)/validate_naive.c

BENCH_NAIVE_O0     := $(BIN_DIR)/bench_naive_O0
BENCH_NAIVE_PG     := $(BIN_DIR)/bench_naive_pg
VALIDATE_NAIVE_O0  := $(BIN_DIR)/validate_naive_O0

.PHONY: all bench_naive_O0 bench_naive_pg validate_naive sweep_naive profile_gprof_naive profile_perf_naive clean distclean

all: $(BENCH_NAIVE_O0) $(VALIDATE_NAIVE_O0)

bench_naive_O0: $(BENCH_NAIVE_O0)
bench_naive_pg: $(BENCH_NAIVE_PG)
validate_naive: $(VALIDATE_NAIVE_O0)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BENCH_NAIVE_O0): $(BENCH_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(BENCH_SRCS) -o $@ $(LIBS)

# -pg instruments the binary so that gprof can read gmon.out.
$(BENCH_NAIVE_PG): $(BENCH_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) -pg $(BENCH_SRCS) -o $@ $(LIBS)

$(VALIDATE_NAIVE_O0): $(VALIDATE_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_SRCS) -o $@ $(LIBS)

sweep_naive: $(BENCH_NAIVE_O0) $(BENCH_NAIVE_PG)
	bash scripts/run_sweep_naive.sh

profile_gprof_naive: $(BENCH_NAIVE_PG)
	bash scripts/profile_gprof_naive.sh

profile_perf_naive: $(BENCH_NAIVE_O0)
	bash scripts/profile_perf_naive.sh

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR) build/audit gmon.out perf.data perf.data.old cachegrind.out.*

distclean: clean
	rm -f results/*.csv plots/*

# =====================================================================
# Fase 6 / Etapa A2 targets - cache-oblivious recursive kernel
#
# These build on top of the naive baseline without modifying any of
# the targets above. The validate binary links matmul_naive.c too
# because the cross-validation test compares the two kernels element
# by element.
# =====================================================================

RECURSIVE_COMMON_SRCS    := $(COMMON_SRCS) $(SRC_DIR)/matmul_recursive.c
BENCH_RECURSIVE_SRCS     := $(RECURSIVE_COMMON_SRCS) $(SRC_DIR)/bench_recursive.c
VALIDATE_RECURSIVE_SRCS  := $(RECURSIVE_COMMON_SRCS) $(SRC_DIR)/validate_recursive.c

BENCH_RECURSIVE_O0       := $(BIN_DIR)/bench_recursive_O0
VALIDATE_RECURSIVE_O0    := $(BIN_DIR)/validate_recursive_O0

.PHONY: bench_recursive validate_recursive

bench_recursive: $(BENCH_RECURSIVE_O0)
validate_recursive: $(VALIDATE_RECURSIVE_O0)

$(BENCH_RECURSIVE_O0): $(BENCH_RECURSIVE_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(BENCH_RECURSIVE_SRCS) -o $@ $(LIBS)

$(VALIDATE_RECURSIVE_O0): $(VALIDATE_RECURSIVE_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_RECURSIVE_SRCS) -o $@ $(LIBS)

# =====================================================================
# Fase 6 / Etapa A3 support module - Morton (Z-order) encoding tests
# =====================================================================

TEST_MORTON_SRCS := $(SRC_DIR)/morton.c $(SRC_DIR)/matrix_utils.c $(SRC_DIR)/test_morton.c
TEST_MORTON      := $(BIN_DIR)/test_morton

.PHONY: test_morton

test_morton: $(TEST_MORTON)

$(TEST_MORTON): $(TEST_MORTON_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(TEST_MORTON_SRCS) -o $@ $(LIBS)

# =====================================================================
# Fase 6 / Etapa A3 targets - Morton kernel bench and validate
#
# validate_morton links matmul_recursive.c too because Test 5 cross-
# validates the Morton kernel against the recursive row-major one.
# =====================================================================

MORTON_KERNEL_SRCS       := $(SRC_DIR)/matmul_morton.c $(SRC_DIR)/morton.c
BENCH_MORTON_SRCS        := $(COMMON_SRCS) $(MORTON_KERNEL_SRCS) $(SRC_DIR)/bench_morton.c
VALIDATE_MORTON_SRCS     := $(COMMON_SRCS) $(MORTON_KERNEL_SRCS) \
                            $(SRC_DIR)/matmul_recursive.c \
                            $(SRC_DIR)/validate_morton.c

BENCH_MORTON_O0          := $(BIN_DIR)/bench_morton_O0
VALIDATE_MORTON_O0       := $(BIN_DIR)/validate_morton_O0

.PHONY: bench_morton validate_morton

bench_morton: $(BENCH_MORTON_O0)
validate_morton: $(VALIDATE_MORTON_O0)

$(BENCH_MORTON_O0): $(BENCH_MORTON_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(BENCH_MORTON_SRCS) -o $@ $(LIBS)

$(VALIDATE_MORTON_O0): $(VALIDATE_MORTON_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_MORTON_SRCS) -o $@ $(LIBS)

# =====================================================================
# Fase 6 / Prompt 6 - comparative sweeps and plots
# =====================================================================

.PHONY: sweep_recursive_run sweep_morton_run plots_comparison sweep_full_santiago

sweep_recursive_run: $(BENCH_RECURSIVE_O0)
	bash scripts/run_sweep_recursive.sh

sweep_morton_run: $(BENCH_MORTON_O0)
	bash scripts/run_sweep_morton.sh

plots_comparison:
	python3 scripts/plot_comparison.py

sweep_full_santiago: sweep_recursive_run sweep_morton_run plots_comparison

# =====================================================================
# Fase 6 / Prompt 7 - hardware-event comparison via perf
# =====================================================================

.PHONY: perf_compare plots_perf

perf_compare: $(BENCH_NAIVE_O0) $(BENCH_RECURSIVE_O0) $(BENCH_MORTON_O0)
	bash scripts/profile_perf_compare.sh

plots_perf:
	python3 scripts/plot_perf_compare.py

# =====================================================================
# Fase 1.1 - Cache-aware: loop reorder
#
# bench_loop_O0  : benchmark that selects kernel by name at runtime
# validate_loop  : algebraic + cross-validation for all 6 orders
# sweep_loop     : runs scripts/run_sweep_loop.sh -> results/loop_order.csv
# =====================================================================

LOOP_COMMON_SRCS   := $(COMMON_SRCS) $(SRC_DIR)/matmul_loop.c
BENCH_LOOP_SRCS    := $(LOOP_COMMON_SRCS) $(SRC_DIR)/bench_loop.c
VALIDATE_LOOP_SRCS := $(LOOP_COMMON_SRCS) $(SRC_DIR)/validate_loop.c

BENCH_LOOP_O0      := $(BIN_DIR)/bench_loop_O0
VALIDATE_LOOP_O0   := $(BIN_DIR)/validate_loop_O0

.PHONY: bench_loop validate_loop \
        sweep_loop_ijk sweep_loop_ikj sweep_loop_jik \
        sweep_loop_jki sweep_loop_kij sweep_loop_kji \
        sweep_loop_all

bench_loop: $(BENCH_LOOP_O0)
validate_loop: $(VALIDATE_LOOP_O0)

$(BENCH_LOOP_O0): $(BENCH_LOOP_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(BENCH_LOOP_SRCS) -o $@ $(LIBS)

$(VALIDATE_LOOP_O0): $(VALIDATE_LOOP_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_LOOP_SRCS) -o $@ $(LIBS)

# Per-order targets: each runs in its own process to avoid cross-contamination.
sweep_loop_ijk: $(BENCH_LOOP_O0)
	bash scripts/run_sweep_loop.sh ijk

sweep_loop_ikj: $(BENCH_LOOP_O0)
	bash scripts/run_sweep_loop.sh ikj

sweep_loop_jik: $(BENCH_LOOP_O0)
	bash scripts/run_sweep_loop.sh jik

sweep_loop_jki: $(BENCH_LOOP_O0)
	bash scripts/run_sweep_loop.sh jki

sweep_loop_kij: $(BENCH_LOOP_O0)
	bash scripts/run_sweep_loop.sh kij

sweep_loop_kji: $(BENCH_LOOP_O0)
	bash scripts/run_sweep_loop.sh kji

# Runs all six orders sequentially (each as a separate process) and
# concatenates the results into a single results/loop_order.csv.
sweep_loop_all: sweep_loop_ijk sweep_loop_ikj sweep_loop_jik \
                sweep_loop_jki sweep_loop_kij sweep_loop_kji
	@echo "kernel,m,n,num_iters,median_seconds,gflops" > results/loop_order.csv
	@for f in results/loop_ijk.csv results/loop_ikj.csv results/loop_jik.csv \
	           results/loop_jki.csv results/loop_kij.csv results/loop_kji.csv; do \
	    tail -n +2 "$$f" >> results/loop_order.csv; \
	done
	@echo "  -> results/loop_order.csv (combined)"

# Plot targets for loop-order results.
.PHONY: plot_loop plot_loop_vs_naive plot_naive

plot_naive:
	python3 scripts/plot_results.py

plot_loop:
	python3 scripts/plot_results.py results/loop_order.csv \
	    --out plots/loop_orders --title "Loop-order kernels"

plot_loop_vs_naive:
	python3 scripts/plot_results.py results/naive_O0.csv results/loop_order.csv \
	    --out plots/loop_vs_naive --title "Loop orders vs naive baseline"

# =====================================================================
# Sesion 03 targets
#
# Bloque para los entregables de la Sesion 03. Por ahora contiene:
#   audit  - verifica que los modulos Morton no usan PDEP/PEXT ni los
#            recibe del compilador (critico para AMD Zen 2; ver
#            scripts/audit_no_pdep.sh para el rationale completo).
#   hwinfo - imprime un fingerprint del hardware en tiempo de ejecucion
#            (CPU model, cores/threads, tamanos L1d/L2/L3, RAM, soporte
#            AVX2/FMA/BMI2). Modo --csv para el encabezado de los CSV
#            de mediciones. Salida warning si L3 < 8 MiB (medicion por
#            CCX) o si BMI2 esta soportado en AMD pre-Zen3.
# =====================================================================

CFLAGS_O3       := $(CSTD) $(WARN) $(INCS) -O3
CFLAGS_O3_ZEN2  := $(CSTD) $(WARN) $(INCS) -O3 -march=znver2 -mavx2 -mfma \
                   -D_POSIX_C_SOURCE=200809L

HWINFO_BIN := $(BIN_DIR)/hwinfo

.PHONY: audit hwinfo

audit:
	bash scripts/audit_no_pdep.sh

hwinfo: $(HWINFO_BIN)
	./$(HWINFO_BIN)

$(HWINFO_BIN): $(SRC_DIR)/hwinfo.c | $(BIN_DIR)
	$(CC) $(CFLAGS_O3) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------
# Sesion 03 / Prompt 2 - threshold sweep
#
# bench_morton_O3 is the same translation units as bench_morton_O0 but
# compiled with -O3 -march=znver2 -mavx2 -mfma so that the sweep
# measures the regime that will be used from Prompt 4 onward. The
# baseline binaries (bench_morton_O0, validate_morton_O0) remain
# unchanged for regression and for Sesion 02 reproducibility.
# ---------------------------------------------------------------------

BENCH_MORTON_O3 := $(BIN_DIR)/bench_morton_O3

.PHONY: bench_morton_O3 sweep_threshold plot_threshold

bench_morton_O3: $(BENCH_MORTON_O3)

$(BENCH_MORTON_O3): $(BENCH_MORTON_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_MORTON_SRCS) -o $@ $(LIBS)

sweep_threshold: $(BENCH_MORTON_O3)
	bash scripts/run_threshold_sweep.sh

plot_threshold:
	python3 scripts/plot_threshold_sweep.py

# ---------------------------------------------------------------------
# Sesion 03 / Prompt 3 - AVX2 + FMA microkernel
#
# kernel_avx2.c uses immintrin.h. We compile it to an object with the
# stage-A4 flag set (-march=znver2 -mavx2 -mfma -funroll-loops
# -ffast-math) and link the test driver against it. -Wpedantic is
# dropped here because immintrin types are GCC extensions and trigger
# pedantic warnings on perfectly valid code. The audit script keeps
# verifying that pdep/pext do not get emitted (criterion 4 / 5).
# ---------------------------------------------------------------------

CFLAGS_AVX2_KERNEL := $(CSTD) -Wall -Wextra $(INCS) \
                      -O3 -march=znver2 -mavx2 -mfma \
                      -funroll-loops -ffast-math

KERNEL_AVX2_OBJ   := $(OBJ_DIR)/kernel_avx2.o
TEST_KERNEL_AVX2  := $(BIN_DIR)/test_kernel_avx2

.PHONY: test_kernel_avx2

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(KERNEL_AVX2_OBJ): $(SRC_DIR)/kernel_avx2.c $(SRC_DIR)/kernel_avx2.h \
                    $(SRC_DIR)/matmul_naive.h | $(OBJ_DIR)
	$(CC) $(CFLAGS_AVX2_KERNEL) -c -o $@ $<

test_kernel_avx2: $(TEST_KERNEL_AVX2)
	./$(TEST_KERNEL_AVX2)

$(TEST_KERNEL_AVX2): $(SRC_DIR)/test_kernel_avx2.c $(KERNEL_AVX2_OBJ) \
                     $(SRC_DIR)/matrix_utils.c $(SRC_DIR)/matrix_utils.h \
                     $(SRC_DIR)/matmul_naive.h | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) \
	      $(SRC_DIR)/test_kernel_avx2.c \
	      $(SRC_DIR)/matrix_utils.c \
	      $(KERNEL_AVX2_OBJ) \
	      -o $@ $(LIBS)

# ---------------------------------------------------------------------
# Sesion 03 / Prompt 4 - Morton + AVX2 integration
#
# matmul_morton_avx2 wires the AVX2 microkernel into the recursive
# Morton skeleton, using a Morton-of-blocks layout for A (tile = 4)
# that materializes nicely to a row-major panel in the leaf. The
# baseline _O0 binaries remain untouched for regression.
#
# validate_morton_avx2 links matmul_morton.c too, because Test 5
# cross-checks the AVX2 variant against the Sesion 02 Morton kernel.
# ---------------------------------------------------------------------

MORTON_AVX2_SHARED_SRCS  := $(SRC_DIR)/matmul_morton_avx2.c \
                             $(SRC_DIR)/morton.c \
                             $(SRC_DIR)/matrix_utils.c \
                             $(SRC_DIR)/matmul_naive.c

BENCH_MORTON_AVX2_SRCS    := $(MORTON_AVX2_SHARED_SRCS) \
                             $(SRC_DIR)/bench_morton_avx2.c

VALIDATE_MORTON_AVX2_SRCS := $(MORTON_AVX2_SHARED_SRCS) \
                             $(SRC_DIR)/matmul_morton.c \
                             $(SRC_DIR)/validate_morton_avx2.c

BENCH_MORTON_AVX2_O3    := $(BIN_DIR)/bench_morton_avx2_O3
VALIDATE_MORTON_AVX2_O3 := $(BIN_DIR)/validate_morton_avx2_O3

.PHONY: bench_morton_avx2 bench_morton_avx2_O3 validate_morton_avx2

# bench_morton_avx2_O3 is an alias that matches the naming convention
# of bench_naive_O3 / bench_recursive_O3 / bench_morton_O3, so the
# sweep_session_03 target and downstream scripts can talk about the
# four bench binaries with a uniform name.
bench_morton_avx2:    $(BENCH_MORTON_AVX2_O3)
bench_morton_avx2_O3: $(BENCH_MORTON_AVX2_O3)
validate_morton_avx2: $(VALIDATE_MORTON_AVX2_O3)

$(BENCH_MORTON_AVX2_O3): $(BENCH_MORTON_AVX2_SRCS) $(KERNEL_AVX2_OBJ) \
                         | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) \
	      $(BENCH_MORTON_AVX2_SRCS) \
	      $(KERNEL_AVX2_OBJ) \
	      -o $@ $(LIBS)

$(VALIDATE_MORTON_AVX2_O3): $(VALIDATE_MORTON_AVX2_SRCS) $(KERNEL_AVX2_OBJ) \
                            | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) \
	      $(VALIDATE_MORTON_AVX2_SRCS) \
	      $(KERNEL_AVX2_OBJ) \
	      -o $@ $(LIBS)

# ---------------------------------------------------------------------
# Sesion 03 / Prompt 5 - comparative sweep across four variants
#
# The sweep compares naive, recursive, morton and morton_avx2 at the
# same -O3 -march=znver2 -mavx2 -mfma flag set, so the only difference
# between curves is the algorithm and the layout (not the optimization
# regime). To make that comparison possible we need -O3 versions of
# the naive and recursive bench drivers; bench_morton_O3 already
# exists from Prompt 2, and bench_morton_avx2_O3 from Prompt 4. The
# _O0 binaries remain untouched because validate_*_O0 and the Sesion
# 01/02 reproducibility artifacts depend on them.
# ---------------------------------------------------------------------

BENCH_NAIVE_O3     := $(BIN_DIR)/bench_naive_O3
BENCH_RECURSIVE_O3 := $(BIN_DIR)/bench_recursive_O3

.PHONY: bench_naive_O3 bench_recursive_O3 sweep_session_03 plot_session_03

bench_naive_O3:     $(BENCH_NAIVE_O3)
bench_recursive_O3: $(BENCH_RECURSIVE_O3)

$(BENCH_NAIVE_O3): $(BENCH_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_SRCS) -o $@ $(LIBS)

$(BENCH_RECURSIVE_O3): $(BENCH_RECURSIVE_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_RECURSIVE_SRCS) -o $@ $(LIBS)

sweep_session_03: $(BENCH_NAIVE_O3) $(BENCH_RECURSIVE_O3) \
                  $(BENCH_MORTON_O3) $(BENCH_MORTON_AVX2_O3)
	bash scripts/run_sweep_session_03.sh

plot_session_03:
	python3 scripts/plot_sweep_session_03.py

# ---------------------------------------------------------------------
# Sesion 03 / extension de Prompt 5 - matmul_morton_avx2 hasta m=32768
#
# Corre solo bench_morton_avx2_O3 sobre el grid extendido para observar
# como escala el microkernel cuando A pasa de 1 GiB (m=16384) a 4 GiB
# (m=32768). El script tiene chequeo de memoria via /proc/meminfo y
# avisa antes de cada m si el working set estimado excede el 85% de
# MemAvailable; en WSL2 con el limite por defecto (~3.5 GiB) el bench
# a m=32768 muy probablemente sera OOM-killed a menos que se eleve
# memory= en ~/.wslconfig.
# ---------------------------------------------------------------------

.PHONY: sweep_morton_avx2_xl plot_morton_avx2_xl

sweep_morton_avx2_xl: $(BENCH_MORTON_AVX2_O3)
	bash scripts/run_sweep_morton_avx2_xl.sh

plot_morton_avx2_xl:
	python3 scripts/plot_morton_avx2_xl.py

# ---------------------------------------------------------------------
# Sesion 03 / Prompt 6 - matmul_morton_omp (OpenMP tasks)
#
# Parallel variant of matmul_morton_avx2 using OpenMP tasks at every
# recursive split above g_parallel_threshold_omp. Reuses the AVX2
# microkernel and the Morton-of-blocks layout from Prompt 4.
#
# CFLAGS_OMP_ZEN2: same flags as CFLAGS_O3_ZEN2 plus -fopenmp. The
# OMP runtime library is linked automatically by gcc with -fopenmp.
#
# validate_morton_omp links matmul_morton.c too (cross-validation
# Test 5 of the avx2 module is the bridge; the omp validate does its
# own cross-check against matmul_morton_avx2 and matmul_naive, but
# matmul_morton.c is needed by matmul_morton_avx2.h's reuse).
# ---------------------------------------------------------------------

CFLAGS_OMP_ZEN2 := $(CFLAGS_O3_ZEN2) -fopenmp

MORTON_OMP_SHARED_SRCS  := $(SRC_DIR)/matmul_morton_omp.c \
                            $(SRC_DIR)/matmul_morton_avx2.c \
                            $(SRC_DIR)/morton.c \
                            $(SRC_DIR)/matrix_utils.c \
                            $(SRC_DIR)/matmul_naive.c

BENCH_MORTON_OMP_SRCS    := $(MORTON_OMP_SHARED_SRCS) \
                            $(SRC_DIR)/bench_morton_omp.c

VALIDATE_MORTON_OMP_SRCS := $(MORTON_OMP_SHARED_SRCS) \
                            $(SRC_DIR)/matmul_morton.c \
                            $(SRC_DIR)/validate_morton_omp.c

BENCH_MORTON_OMP_O3    := $(BIN_DIR)/bench_morton_omp_O3
VALIDATE_MORTON_OMP_O3 := $(BIN_DIR)/validate_morton_omp_O3

.PHONY: bench_morton_omp bench_morton_omp_O3 validate_morton_omp \
        sweep_omp_scaling plot_omp_scaling

bench_morton_omp:    $(BENCH_MORTON_OMP_O3)
bench_morton_omp_O3: $(BENCH_MORTON_OMP_O3)
validate_morton_omp: $(VALIDATE_MORTON_OMP_O3)

$(BENCH_MORTON_OMP_O3): $(BENCH_MORTON_OMP_SRCS) $(KERNEL_AVX2_OBJ) | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN2) \
	      $(BENCH_MORTON_OMP_SRCS) \
	      $(KERNEL_AVX2_OBJ) \
	      -o $@ $(LIBS)

$(VALIDATE_MORTON_OMP_O3): $(VALIDATE_MORTON_OMP_SRCS) $(KERNEL_AVX2_OBJ) | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN2) \
	      $(VALIDATE_MORTON_OMP_SRCS) \
	      $(KERNEL_AVX2_OBJ) \
	      -o $@ $(LIBS)

sweep_omp_scaling: $(BENCH_MORTON_OMP_O3)
	bash scripts/run_omp_scaling.sh

plot_omp_scaling:
	python3 scripts/plot_omp_scaling.py

# ---------------------------------------------------------------------
# Sesion 03 / Prompt 7 - perf Zen 2 hardware counter sweep
#
# Crosses (variant, m) for 4 variants x 3 m's = 12 cells, two perf
# invocations per cell (group A: compute side, group B: memory + TLB
# side) to keep multiplexing percentages close to 100%. The
# consolidator merges both groups into one CSV row per cell.
#
# Dependencies:
#   - perf (linux-tools-generic on Ubuntu, or built from the kernel tree)
#   - kernel.perf_event_paranoid <= 2 (see README.md / the script for
#     instructions if you hit a permission error)
# ---------------------------------------------------------------------

.PHONY: profile_zen2 consolidate_zen2 plot_perf_zen2 profile_zen2_one

# profile_zen2_one is a thin wrapper for ad-hoc single-cell profiling
# during development. Use as: make profile_zen2_one VARIANT=morton M=4096
VARIANT ?= morton_avx2
M       ?= 4096
profile_zen2_one: $(BENCH_NAIVE_O3) $(BENCH_RECURSIVE_O3) \
                  $(BENCH_MORTON_O3) $(BENCH_MORTON_AVX2_O3)
	bash scripts/profile_perf_zen2.sh $(VARIANT) $(M)

profile_zen2: $(BENCH_NAIVE_O3) $(BENCH_RECURSIVE_O3) \
              $(BENCH_MORTON_O3) $(BENCH_MORTON_AVX2_O3)
	bash scripts/run_perf_zen2_sweep.sh
	python3 scripts/consolidate_perf_zen2.py

consolidate_zen2:
	python3 scripts/consolidate_perf_zen2.py

plot_perf_zen2:
	python3 scripts/plot_perf_zen2.py

# Profile morton_omp at multi-thread for the Roofline plot. Default to
# 6 threads with bind=spread (uses both CCXs without crowding). Pass
# THREADS_OMP / BIND_OMP / MS_OMP to override.
THREADS_OMP ?= 6
BIND_OMP    ?= spread
MS_OMP      ?= 1024 4096 8192
profile_zen2_omp: $(BENCH_MORTON_OMP_O3)
	@for m in $(MS_OMP); do \
	  echo "=== morton_omp m=$$m (OMP_NUM_THREADS=$(THREADS_OMP) bind=$(BIND_OMP)) ==="; \
	  OMP_NUM_THREADS=$(THREADS_OMP) OMP_PLACES=cores OMP_PROC_BIND=$(BIND_OMP) \
	    bash scripts/profile_perf_zen2.sh morton_omp $$m; \
	done

# ---------------------------------------------------------------------
# Sesion 03 / Prompt 8 - STREAM bandwidth + Roofline final
# ---------------------------------------------------------------------

.PHONY: stream plot_roofline roofline

stream:
	bash scripts/measure_stream.sh

plot_roofline:
	python3 scripts/plot_roofline.py

# Convenience target: STREAM + Roofline in one shot. Assumes
# profile_zen2 (and optionally profile_zen2_omp) ran beforehand so
# the perf files exist.
roofline: stream plot_roofline
