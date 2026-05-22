#
# Makefile for the iterated matmul benchmark.
#
# Source layout (post src/ reorg):
#
#   src/core/          matrix_utils.{c,h}, timing.h, morton.{c,h}
#   src/microkernels/  kernel_avx2_morton.h         (4x16, Morton family,
#                                                    header-only static inline)
#                      kernel_avx2_tiled.h          (6x16, tiled_ikj family,
#                                                    header-only static inline)
#   src/algorithms/
#     naive/           matmul_naive.{c,h}
#     loops/           matmul_loops.{c,h}
#     morton/          matmul_morton{,_avx2,_omp}.{c,h}
#     tiled_ikj/       matmul_tiled_ikj{,_avx2,_omp}.{c,h}
#   src/drivers/
#     bench/           bench_*.c       (one per algorithm)
#     validate/        validate_*.c    (one per algorithm)
#   src/tests/         test_morton.c, test_kernel_avx2.c
#   src/tools/         hwinfo.c
#
# All -I flags below are added together so that #include "foo.h" works
# regardless of which subdirectory the header lives in. This keeps the
# source files free of relative paths like "../../core/matrix_utils.h"
# and lets each .c stay in its conceptual subdir without referring to
# its on-disk neighbors explicitly.
#
# Common targets:
#   make                   -> bench_naive_O0 and validate_naive_O0 (default)
#   make bench_naive_O0    -> baseline benchmark binary (no compiler optimization)
#   make bench_naive_pg    -> same as bench_naive_O0 but compiled with -pg for gprof
#   make validate_naive    -> validate_naive_O0 binary that runs the algebraic tests
#   make sweep_naive       -> runs scripts/run_sweep_naive.sh after building
#   make results [M=...]   -> full perf-counter sweep (variant x m grid)
#   make clean             -> remove binaries and object files (keeps CSV/plots)
#   make distclean         -> clean plus remove results/*.csv and plots/*
#

CC      := gcc
CSTD    := -std=c11
WARN    := -Wall -Wextra -Wpedantic

# ---------------------------------------------------------------------
# Source tree layout
# ---------------------------------------------------------------------

SRC_DIR        := src

CORE_DIR       := $(SRC_DIR)/core
MK_DIR         := $(SRC_DIR)/microkernels
ALG_DIR        := $(SRC_DIR)/algorithms
NAIVE_DIR      := $(ALG_DIR)/naive
LOOPS_DIR      := $(ALG_DIR)/loops
MORTON_DIR     := $(ALG_DIR)/morton
TILED_DIR      := $(ALG_DIR)/tiled_ikj
BENCH_DIR      := $(SRC_DIR)/drivers/bench
VALIDATE_DIR   := $(SRC_DIR)/drivers/validate
TESTS_DIR      := $(SRC_DIR)/tests
TOOLS_DIR      := $(SRC_DIR)/tools

BIN_DIR        := bin
OBJ_DIR        := build

# Aggregate -I flag so every #include "foo.h" resolves regardless of
# which subdirectory the header lives in.
INCS := -I$(CORE_DIR) -I$(MK_DIR) \
        -I$(NAIVE_DIR) -I$(LOOPS_DIR) -I$(MORTON_DIR) -I$(TILED_DIR)

LIBS := -lm

# -O0 is mandatory for Phase 1. -g for debugging symbols and meaningful
# function names in gprof/perf reports. -fno-omit-frame-pointer makes
# perf call-graphs work without DWARF unwinding.
# _POSIX_C_SOURCE=200809L is required so that glibc exposes clock_gettime
# and CLOCK_MONOTONIC under -std=c11 (which otherwise hides POSIX-only
# symbols).
BASE_CFLAGS := $(CSTD) $(WARN) $(INCS) -O0 -g -fno-omit-frame-pointer \
               -D_POSIX_C_SOURCE=200809L

# Common module list (kernel + helpers) used by both bench and validate
# of every algorithm.
COMMON_SRCS := $(NAIVE_DIR)/matmul_naive.c $(CORE_DIR)/matrix_utils.c

BENCH_SRCS    := $(COMMON_SRCS) $(BENCH_DIR)/bench_naive.c
VALIDATE_SRCS := $(COMMON_SRCS) $(VALIDATE_DIR)/validate_naive.c

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
# Fase 6 / Etapa A3 support module - Morton (Z-order) encoding tests
# =====================================================================

TEST_MORTON_SRCS := $(CORE_DIR)/morton.c $(CORE_DIR)/matrix_utils.c $(TESTS_DIR)/test_morton.c
TEST_MORTON      := $(BIN_DIR)/test_morton

.PHONY: test_morton

test_morton: $(TEST_MORTON)

$(TEST_MORTON): $(TEST_MORTON_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(TEST_MORTON_SRCS) -o $@ $(LIBS)

# =====================================================================
# Fase 6 / Etapa A3 targets - Morton kernel bench and validate
#
# validate_morton cross-validates the Morton kernel against matmul_naive
# (Test 4); matmul_naive is the canonical reference baseline.
# =====================================================================

MORTON_KERNEL_SRCS       := $(MORTON_DIR)/matmul_morton.c $(CORE_DIR)/morton.c
BENCH_MORTON_SRCS        := $(COMMON_SRCS) $(MORTON_KERNEL_SRCS) $(BENCH_DIR)/bench_morton.c
VALIDATE_MORTON_SRCS     := $(COMMON_SRCS) $(MORTON_KERNEL_SRCS) \
                            $(VALIDATE_DIR)/validate_morton.c

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
# Fase 6 / Prompt 6 - Morton sweep
# =====================================================================

.PHONY: sweep_morton_run

sweep_morton_run: $(BENCH_MORTON_O0)
	bash scripts/run_sweep_morton.sh

# =====================================================================
# Fase 1.1 - Cache-aware: loop reorder
#
# bench_loops_O0 : benchmark that selects kernel by name at runtime
# validate_loops : algebraic + cross-validation for all 6 orders
# sweep_loops    : runs scripts/run_sweep_loops.sh -> results/loop_order.csv
# =====================================================================

LOOPS_COMMON_SRCS   := $(COMMON_SRCS) $(LOOPS_DIR)/matmul_loops.c
BENCH_LOOPS_SRCS    := $(LOOPS_COMMON_SRCS) $(BENCH_DIR)/bench_loops.c
VALIDATE_LOOPS_SRCS := $(LOOPS_COMMON_SRCS) $(VALIDATE_DIR)/validate_loops.c

BENCH_LOOPS_O0      := $(BIN_DIR)/bench_loops_O0
BENCH_LOOPS_O3      := $(BIN_DIR)/bench_loops_O3
VALIDATE_LOOPS_O0   := $(BIN_DIR)/validate_loops_O0

.PHONY: bench_loops bench_loops_O3 validate_loops \
        sweep_loops_ijk sweep_loops_ikj sweep_loops_jik \
        sweep_loops_jki sweep_loops_kij sweep_loops_kji \
        sweep_loops_all

bench_loops: $(BENCH_LOOPS_O0)
bench_loops_O3: $(BENCH_LOOPS_O3)
validate_loops: $(VALIDATE_LOOPS_O0)

$(BENCH_LOOPS_O0): $(BENCH_LOOPS_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(BENCH_LOOPS_SRCS) -o $@ $(LIBS)

$(BENCH_LOOPS_O3): $(BENCH_LOOPS_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_LOOPS_SRCS) -o $@ $(LIBS)

$(VALIDATE_LOOPS_O0): $(VALIDATE_LOOPS_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_LOOPS_SRCS) -o $@ $(LIBS)

# Per-order targets: each runs in its own process to avoid cross-contamination.
sweep_loops_ijk: $(BENCH_LOOPS_O0)
	bash scripts/run_sweep_loops.sh ijk

sweep_loops_ikj: $(BENCH_LOOPS_O0)
	bash scripts/run_sweep_loops.sh ikj

sweep_loops_jik: $(BENCH_LOOPS_O0)
	bash scripts/run_sweep_loops.sh jik

sweep_loops_jki: $(BENCH_LOOPS_O0)
	bash scripts/run_sweep_loops.sh jki

sweep_loops_kij: $(BENCH_LOOPS_O0)
	bash scripts/run_sweep_loops.sh kij

sweep_loops_kji: $(BENCH_LOOPS_O0)
	bash scripts/run_sweep_loops.sh kji

# Runs all six orders sequentially (each as a separate process) and
# concatenates the results into a single results/loop_order.csv.
sweep_loops_all: sweep_loops_ijk sweep_loops_ikj sweep_loops_jik \
                 sweep_loops_jki sweep_loops_kij sweep_loops_kji
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

$(HWINFO_BIN): $(TOOLS_DIR)/hwinfo.c | $(BIN_DIR)
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
# Sesion 03 / Prompt 3 - AVX2 + FMA microkernel for the Morton family
#
# kernel_avx2_morton.h is now header-only (`static inline`), matching
# the layout of kernel_avx2_tiled.h and of Juan Pablo's kernel_avx512.h
# in opt_zen5. There is no kernel_avx2_morton.o anymore: each .c that
# includes the header (matmul_morton_avx2.c, matmul_morton_omp.c,
# tests/test_kernel_avx2.c) gets its own inlined copy under -O3.
# Bench/validate binaries are compiled with -mavx2 -mfma via
# CFLAGS_O3_ZEN2, which is what the immintrin intrinsics require.
#
# The audit script keeps verifying that pdep/pext do not get emitted
# (criterion 4 / 5).
# ---------------------------------------------------------------------

TEST_KERNEL_AVX2 := $(BIN_DIR)/test_kernel_avx2

.PHONY: test_kernel_avx2

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

test_kernel_avx2: $(TEST_KERNEL_AVX2)
	./$(TEST_KERNEL_AVX2)

$(TEST_KERNEL_AVX2): $(TESTS_DIR)/test_kernel_avx2.c \
                     $(MK_DIR)/kernel_avx2_morton.h \
                     $(CORE_DIR)/matrix_utils.c $(CORE_DIR)/matrix_utils.h \
                     $(NAIVE_DIR)/matmul_naive.h | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) \
	      $(TESTS_DIR)/test_kernel_avx2.c \
	      $(CORE_DIR)/matrix_utils.c \
	      -o $@ $(LIBS)

# ---------------------------------------------------------------------
# Sesion 03 / Prompt 4 - Morton + AVX2 integration
#
# matmul_morton_avx2 wires the AVX2 microkernel (kernel_avx2_morton)
# into the recursive Morton skeleton, using a Morton-of-blocks layout
# for A (tile = 4) that materializes nicely to a row-major panel in
# the leaf. The baseline _O0 binaries remain untouched for regression.
#
# validate_morton_avx2 links matmul_morton.c too, because Test 5
# cross-checks the AVX2 variant against the Sesion 02 Morton kernel.
# ---------------------------------------------------------------------

MORTON_AVX2_SHARED_SRCS  := $(MORTON_DIR)/matmul_morton_avx2.c \
                             $(CORE_DIR)/morton.c \
                             $(CORE_DIR)/matrix_utils.c \
                             $(NAIVE_DIR)/matmul_naive.c

BENCH_MORTON_AVX2_SRCS    := $(MORTON_AVX2_SHARED_SRCS) \
                             $(BENCH_DIR)/bench_morton_avx2.c

VALIDATE_MORTON_AVX2_SRCS := $(MORTON_AVX2_SHARED_SRCS) \
                             $(MORTON_DIR)/matmul_morton.c \
                             $(VALIDATE_DIR)/validate_morton_avx2.c

BENCH_MORTON_AVX2_O3    := $(BIN_DIR)/bench_morton_avx2_O3
VALIDATE_MORTON_AVX2_O3 := $(BIN_DIR)/validate_morton_avx2_O3

.PHONY: bench_morton_avx2 bench_morton_avx2_O3 validate_morton_avx2

# bench_morton_avx2_O3 is an alias that matches the naming convention
# of bench_naive_O3 / bench_morton_O3, so the perf Zen 2 sweep and
# downstream scripts can talk about the bench binaries with a uniform
# name.
bench_morton_avx2:    $(BENCH_MORTON_AVX2_O3)
bench_morton_avx2_O3: $(BENCH_MORTON_AVX2_O3)
validate_morton_avx2: $(VALIDATE_MORTON_AVX2_O3)

$(BENCH_MORTON_AVX2_O3): $(BENCH_MORTON_AVX2_SRCS) \
                         $(MK_DIR)/kernel_avx2_morton.h | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_MORTON_AVX2_SRCS) -o $@ $(LIBS)

$(VALIDATE_MORTON_AVX2_O3): $(VALIDATE_MORTON_AVX2_SRCS) \
                            $(MK_DIR)/kernel_avx2_morton.h | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(VALIDATE_MORTON_AVX2_SRCS) -o $@ $(LIBS)

# ---------------------------------------------------------------------
# Sesion 03 / Prompt 5 - bench_naive_O3 driver for fair flag-set
# comparisons across the perf Zen 2 sweep
#
# The perf Zen 2 sweep compares all variants at the same -O3 -march=znver2
# -mavx2 -mfma flag set so the only difference between curves is the
# algorithm and the layout, not the optimization regime. To make that
# comparison possible we need an -O3 version of the naive bench driver;
# bench_morton_O3 already exists from Prompt 2, and bench_morton_avx2_O3
# from Prompt 4. The _O0 binary remains untouched because
# validate_naive_O0 and the Sesion 01 reproducibility artifacts depend
# on it.
# ---------------------------------------------------------------------

BENCH_NAIVE_O3 := $(BIN_DIR)/bench_naive_O3

.PHONY: bench_naive_O3

bench_naive_O3: $(BENCH_NAIVE_O3)

$(BENCH_NAIVE_O3): $(BENCH_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_SRCS) -o $@ $(LIBS)

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

MORTON_OMP_SHARED_SRCS  := $(MORTON_DIR)/matmul_morton_omp.c \
                            $(MORTON_DIR)/matmul_morton_avx2.c \
                            $(CORE_DIR)/morton.c \
                            $(CORE_DIR)/matrix_utils.c \
                            $(NAIVE_DIR)/matmul_naive.c

BENCH_MORTON_OMP_SRCS    := $(MORTON_OMP_SHARED_SRCS) \
                            $(BENCH_DIR)/bench_morton_omp.c

VALIDATE_MORTON_OMP_SRCS := $(MORTON_OMP_SHARED_SRCS) \
                            $(MORTON_DIR)/matmul_morton.c \
                            $(VALIDATE_DIR)/validate_morton_omp.c

BENCH_MORTON_OMP_O3    := $(BIN_DIR)/bench_morton_omp_O3
VALIDATE_MORTON_OMP_O3 := $(BIN_DIR)/validate_morton_omp_O3

.PHONY: bench_morton_omp bench_morton_omp_O3 validate_morton_omp \
        sweep_omp_scaling plot_omp_scaling

bench_morton_omp:    $(BENCH_MORTON_OMP_O3)
bench_morton_omp_O3: $(BENCH_MORTON_OMP_O3)
validate_morton_omp: $(VALIDATE_MORTON_OMP_O3)

$(BENCH_MORTON_OMP_O3): $(BENCH_MORTON_OMP_SRCS) \
                        $(MK_DIR)/kernel_avx2_morton.h | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN2) $(BENCH_MORTON_OMP_SRCS) -o $@ $(LIBS)

$(VALIDATE_MORTON_OMP_O3): $(VALIDATE_MORTON_OMP_SRCS) \
                           $(MK_DIR)/kernel_avx2_morton.h | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN2) $(VALIDATE_MORTON_OMP_SRCS) -o $@ $(LIBS)

sweep_omp_scaling: $(BENCH_MORTON_OMP_O3)
	bash scripts/run_omp_scaling.sh

plot_omp_scaling:
	python3 scripts/plot_omp_scaling.py

# ---------------------------------------------------------------------
# Sesion 03 / Prompt 7 - perf Zen 2 hardware counter sweep
#
# Crosses (variant, m) for 13 variants x 3 m's = 39 cells, two perf
# invocations per cell (group A: compute side, group B: memory + TLB
# side) to keep multiplexing percentages close to 100%. The
# consolidator merges both groups into one CSV row per cell.
# Variants: naive morton morton_avx2 morton_omp + 6 loop orders + tiled_ikj* family.
#
# Dependencies:
#   - perf (linux-tools-generic on Ubuntu, or built from the kernel tree)
#   - kernel.perf_event_paranoid <= 2 (see README.md / the script for
#     instructions if you hit a permission error)
# ---------------------------------------------------------------------

.PHONY: profile_zen2 consolidate_zen2 plot_perf_zen2 profile_zen2_one results

# profile_zen2_one is a thin wrapper for ad-hoc single-cell profiling
# during development. Use as: make profile_zen2_one VARIANT=loop_ikj M=4096
VARIANT ?= morton_avx2
M       ?= 4096
profile_zen2_one: $(BENCH_NAIVE_O3) \
                  $(BENCH_MORTON_O3) $(BENCH_MORTON_AVX2_O3) $(BENCH_LOOPS_O3)
	bash scripts/profile_perf_zen2.sh $(VARIANT) $(M)

profile_zen2: $(BENCH_NAIVE_O3) \
              $(BENCH_MORTON_O3) $(BENCH_MORTON_AVX2_O3) $(BENCH_LOOPS_O3)
	$(if $(M),MS="$(M)" )bash scripts/run_perf_zen2_sweep.sh
	python3 scripts/consolidate_perf_zen2.py

# Forward declaration of the tiled_ikj* bench output paths so the
# `results` target prerequisite list expands correctly. GNU Make expands
# variables in target prerequisites during the read phase; if a variable
# is defined later in the file, it expands to empty here and the
# corresponding binary is silently dropped from the dependency set. The
# canonical recipes for these binaries live further down in the
# Fase 1.2 / 1.3 / 1.4 blocks and consume the variables defined here.
BENCH_TILED_IKJ_O3      := $(BIN_DIR)/bench_tiled_ikj_O3
BENCH_TILED_IKJ_AVX2_O3 := $(BIN_DIR)/bench_tiled_ikj_avx2_O3
BENCH_TILED_IKJ_OMP_O3  := $(BIN_DIR)/bench_tiled_ikj_omp_O3

results: $(BENCH_NAIVE_O3) \
         $(BENCH_MORTON_O3) $(BENCH_MORTON_AVX2_O3) $(BENCH_MORTON_OMP_O3) \
         $(BENCH_LOOPS_O3) \
         $(BENCH_TILED_IKJ_O3) $(BENCH_TILED_IKJ_AVX2_O3) $(BENCH_TILED_IKJ_OMP_O3)
	$(if $(M),MS="$(M)" )bash scripts/run_perf_zen2_sweep.sh
	python3 scripts/consolidate_perf_zen2.py --out results/metrics.csv

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

# =====================================================================
# Fase 1.2 - Cache-aware: explicit loop tiling (tilled_ikj)
#
# bench_tiled_O3   : benchmark with the same flags as the loop-reorder
#                    study (CFLAGS_O3_ZEN2) so results are directly
#                    comparable with bench_loops_O3 and bench_naive_O3.
# validate_tiled_ikj   : algebraic + cross-validation against matmul_naive,
#                    compiled at -O0 for deterministic numerical output.
# =====================================================================

TILED_IKJ_COMMON_SRCS    := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj.c
BENCH_TILED_IKJ_SRCS     := $(TILED_IKJ_COMMON_SRCS) $(BENCH_DIR)/bench_tiled_ikj.c
VALIDATE_TILED_IKJ_SRCS  := $(TILED_IKJ_COMMON_SRCS) $(VALIDATE_DIR)/validate_tiled_ikj.c

# BENCH_TILED_IKJ_O3 is forward-declared above (just before the `results`
# target) so its expansion in `results`'s prerequisite list works.
VALIDATE_TILED_IKJ_O0    := $(BIN_DIR)/validate_tiled_ikj_O0

.PHONY: bench_tiled_ikj validate_tiled_ikj

bench_tiled_ikj:    $(BENCH_TILED_IKJ_O3)
validate_tiled_ikj: $(VALIDATE_TILED_IKJ_O0)

$(BENCH_TILED_IKJ_O3): $(BENCH_TILED_IKJ_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_TILED_IKJ_SRCS) -o $@ $(LIBS)

$(VALIDATE_TILED_IKJ_O0): $(VALIDATE_TILED_IKJ_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_TILED_IKJ_SRCS) -o $@ $(LIBS)

# =====================================================================
# Fase 1.3 - tiled_ikj_avx2: 6-loop blocking with AVX2+FMA vectorization
#
# Extends matmul_tiled_ikj (Phase 1.2) by adding a third outer blocking
# loop over j and replacing the scalar innermost j pass with an AVX2
# broadcast+FMA vector loop. Both binaries are compiled with
# CFLAGS_O3_ZEN2 because matmul_tiled_ikj_avx2.c uses immintrin
# intrinsics (transitively through kernel_avx2_tiled.h) that require
# -mavx2 -mfma; compiling at -O0 without those flags would produce an
# assembler error on the _mm256_fmadd_ps call.
#
# Note: kernel_avx2_tiled.h is header-only (`static inline`), so there
# is no kernel_avx2_tiled.o object to link against. The header is
# included directly by matmul_tiled_ikj_avx2.c.
# =====================================================================

TILED_IKJ_AVX2_COMMON_SRCS   := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj_avx2.c
BENCH_TILED_IKJ_AVX2_SRCS    := $(TILED_IKJ_AVX2_COMMON_SRCS) $(BENCH_DIR)/bench_tiled_ikj_avx2.c
VALIDATE_TILED_IKJ_AVX2_SRCS := $(TILED_IKJ_AVX2_COMMON_SRCS) $(VALIDATE_DIR)/validate_tiled_ikj_avx2.c

# BENCH_TILED_IKJ_AVX2_O3 is forward-declared above; only the validate
# binary path is defined here.
VALIDATE_TILED_IKJ_AVX2_O3 := $(BIN_DIR)/validate_tiled_ikj_avx2_O3

.PHONY: bench_tiled_ikj_avx2 validate_tiled_ikj_avx2

bench_tiled_ikj_avx2:    $(BENCH_TILED_IKJ_AVX2_O3)
validate_tiled_ikj_avx2: $(VALIDATE_TILED_IKJ_AVX2_O3)

$(BENCH_TILED_IKJ_AVX2_O3): $(BENCH_TILED_IKJ_AVX2_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_TILED_IKJ_AVX2_SRCS) -o $@ $(LIBS)

$(VALIDATE_TILED_IKJ_AVX2_O3): $(VALIDATE_TILED_IKJ_AVX2_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(VALIDATE_TILED_IKJ_AVX2_SRCS) -o $@ $(LIBS)

# =====================================================================
# Fase 1.4 - tiled_ikj_omp: tiled_ikj_avx2 + OpenMP parallel for on ii loop
#
# Parallelizes the outermost ii tile loop with a single pragma omp
# parallel for schedule(static). Each ii tile writes to disjoint rows
# of C, so there are no write conflicts. A and B are shared read-only.
#
# Uses CFLAGS_OMP_ZEN2 (defined above as CFLAGS_O3_ZEN2 + -fopenmp)
# so the hardware event comparisons remain apples-to-apples with the
# other O3_ZEN2 variants. Shares kernel_avx2_tiled.h with the AVX2
# sibling; no extra object file required.
# =====================================================================

TILED_IKJ_OMP_COMMON_SRCS   := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj_omp.c
BENCH_TILED_IKJ_OMP_SRCS    := $(TILED_IKJ_OMP_COMMON_SRCS) $(BENCH_DIR)/bench_tiled_ikj_omp.c
VALIDATE_TILED_IKJ_OMP_SRCS := $(TILED_IKJ_OMP_COMMON_SRCS) $(VALIDATE_DIR)/validate_tiled_ikj_omp.c

# BENCH_TILED_IKJ_OMP_O3 is forward-declared above; only the validate
# binary path is defined here.
VALIDATE_TILED_IKJ_OMP_O3 := $(BIN_DIR)/validate_tiled_ikj_omp_O3

.PHONY: bench_tiled_ikj_omp bench_tiled_ikj_omp_O3 validate_tiled_ikj_omp

bench_tiled_ikj_omp:    $(BENCH_TILED_IKJ_OMP_O3)
bench_tiled_ikj_omp_O3: $(BENCH_TILED_IKJ_OMP_O3)
validate_tiled_ikj_omp: $(VALIDATE_TILED_IKJ_OMP_O3)

$(BENCH_TILED_IKJ_OMP_O3): $(BENCH_TILED_IKJ_OMP_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN2) $(BENCH_TILED_IKJ_OMP_SRCS) -o $@ $(LIBS)

$(VALIDATE_TILED_IKJ_OMP_O3): $(VALIDATE_TILED_IKJ_OMP_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN2) $(VALIDATE_TILED_IKJ_OMP_SRCS) -o $@ $(LIBS)
