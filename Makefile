#
# Makefile for the iterated matmul benchmark - main_server variant.
#
# Target hardware: AMD EPYC 9R45 (Zen 5) on AWS c8a.2xlarge.
#   8 cores, 1 thread/core (SMT disabled by hypervisor)
#   L1d 48 KiB/core, L2 1 MiB/core, L3 32 MiB shared
#   Full AVX-512 ISA (F + VL + BW + DQ + IFMA + ...)
#
# Source layout:
#   src/core/             matrix_utils.{c,h}, timing.h, morton.{c,h}
#   src/microkernels/     kernel_avx512_morton.h, kernel_avx512_tiled.h
#   src/algorithms/
#     naive/              matmul_naive.{c,h}
#     loops/              matmul_loops.{c,h}
#     morton/             matmul_morton{,_avx512,_omp}.{c,h}
#     tiled_ikj/          matmul_tiled_ikj{,_avx512,_omp}.{c,h}
#   src/drivers/bench/    bench_*.c  (one per algorithm)
#   src/drivers/validate/ validate_*.c
#   src/tests/            test_matrix_utils.c, test_morton.c,
#                         test_kernel_avx512_morton.c, test_kernel_avx512_tiled.c
#
# Key targets:
#   make build         -> compile all bench + validate + test binaries
#   make tests         -> build + run all unit tests (fast, run first)
#   make validate      -> tests + run all validate_* (correctness gate)
#   make results       -> build + perf sweep -> results/metrics.csv
#   make plots         -> render 4 figures from results/metrics.csv -> plots/
#   make clean         -> remove binaries and object files
#   make distclean     -> clean + remove results/*.csv and plots/*
#
# Binary layout under bin/:
#   bin/bench/      bench_*_ZEN5      (benchmark drivers, -O3 -march=native)
#   bin/validate/   validate_*_O0     (correctness drivers, mostly -O0 for clarity)
#   bin/tests/      test_*            (unit tests for core/ and microkernels/)
#
# Sweep knobs (pass on the command line):
#   MS=<sizes>   size list for run_perf_zen5_sweep.sh  (e.g. MS="1024 4096")
#   VARIANT=<v>  single variant for profile_zen5_one   (default: morton_avx512)
#   M=<size>     single size   for profile_zen5_one    (default: 4096)
#

# ── 1. TOOLCHAIN ──────────────────────────────────────────────────────
CC   := gcc
CSTD := -std=c11
WARN := -Wall -Wextra -Wpedantic
LIBS := -lm

# ── 2. PATHS ──────────────────────────────────────────────────────────
SRC_DIR      := src
CORE_DIR     := $(SRC_DIR)/core
MK_DIR       := $(SRC_DIR)/microkernels
ALG_DIR      := $(SRC_DIR)/algorithms
NAIVE_DIR    := $(ALG_DIR)/naive
LOOPS_DIR    := $(ALG_DIR)/loops
MORTON_DIR   := $(ALG_DIR)/morton
TILED_DIR    := $(ALG_DIR)/tiled_ikj
BENCH_DIR    := $(SRC_DIR)/drivers/bench
VALIDATE_DIR := $(SRC_DIR)/drivers/validate
TESTS_DIR    := $(SRC_DIR)/tests

BIN_DIR          := bin
BIN_BENCH_DIR    := $(BIN_DIR)/bench
BIN_VALIDATE_DIR := $(BIN_DIR)/validate
BIN_TESTS_DIR    := $(BIN_DIR)/tests
OBJ_DIR          := build

INCS := -I$(CORE_DIR) -I$(MK_DIR) \
        -I$(NAIVE_DIR) -I$(LOOPS_DIR) -I$(MORTON_DIR) -I$(TILED_DIR)

# ── 3. COMPILER FLAGS ─────────────────────────────────────────────────
# -D_POSIX_C_SOURCE=200809L exposes clock_gettime under -std=c11.
# -fno-omit-frame-pointer keeps perf call-graphs working.
BASE_CFLAGS    := $(CSTD) $(WARN) $(INCS) -O0 -g -fno-omit-frame-pointer \
                  -D_POSIX_C_SOURCE=200809L
CFLAGS_O3      := $(CSTD) $(WARN) $(INCS) -O3

# -march=native reads CPUID at compile time and enables the full AVX-512
# extension set (F/VL/BW/DQ/IFMA) available on the EPYC 9R45.
# The -D flags set algorithm tuning defaults for the server's cache
# hierarchy (L1d=48 KiB, L2=1 MiB): kc=256 (B panel 32 KiB fits L1d),
# mc=288 (A panel 288 KiB fits L2).
CFLAGS_O3_ZEN5  := $(CSTD) $(WARN) $(INCS) -O3 -march=native             \
                   -D_POSIX_C_SOURCE=200809L                               \
                   -DMORTON_AVX512_THRESHOLD_DEFAULT=1048576UL             \
                   -DMORTON_OMP_RECURSION_THRESHOLD_DEFAULT=1048576UL      \
                   -DMORTON_OMP_PARALLEL_THRESHOLD_DEFAULT=1048576UL       \
                   -DTILED_IKJ_MC_DEFAULT=384u                             \
                   -DTILED_IKJ_KC_DEFAULT=384u                             \
                   -DTILED_IKJ_AVX512_BS_DEFAULT=256u                      \
                   -DTILED_IKJ_AVX512_MC=288u                              \
                   -DTILED_IKJ_OMP_BS_DEFAULT=256u                         \
                   -DTILED_IKJ_OMP_MC=288u
CFLAGS_OMP_ZEN5 := $(CFLAGS_O3_ZEN5) -fopenmp

# ── 4. SOURCES & BINARY PATHS ─────────────────────────────────────────
# Modules shared by every algorithm family (reference + matrix helpers).
COMMON_SRCS := $(NAIVE_DIR)/matmul_naive.c $(CORE_DIR)/matrix_utils.c

# Header-only microkernels: listed as explicit prerequisites so Make
# recompiles all dependents when a kernel header changes.
KERNEL_MORTON_H := $(MK_DIR)/kernel_avx512_morton.h
KERNEL_TILED_H  := $(MK_DIR)/kernel_avx512_tiled.h

# --- naive ---
BENCH_NAIVE_SRCS    := $(COMMON_SRCS) $(BENCH_DIR)/bench_naive.c
VALIDATE_NAIVE_SRCS := $(COMMON_SRCS) $(VALIDATE_DIR)/validate_naive.c
BENCH_NAIVE_ZEN5    := $(BIN_BENCH_DIR)/bench_naive_ZEN5
VALIDATE_NAIVE_O0   := $(BIN_VALIDATE_DIR)/validate_naive_O0

# --- loops ---
BENCH_LOOPS_SRCS    := $(COMMON_SRCS) $(LOOPS_DIR)/matmul_loops.c \
                       $(BENCH_DIR)/bench_loops.c
VALIDATE_LOOPS_SRCS := $(COMMON_SRCS) $(LOOPS_DIR)/matmul_loops.c \
                       $(VALIDATE_DIR)/validate_loops.c
BENCH_LOOPS_ZEN5    := $(BIN_BENCH_DIR)/bench_loops_ZEN5
VALIDATE_LOOPS_O0   := $(BIN_VALIDATE_DIR)/validate_loops_O0

# --- morton ---
BENCH_MORTON_SRCS    := $(COMMON_SRCS) $(MORTON_DIR)/matmul_morton.c \
                        $(CORE_DIR)/morton.c $(BENCH_DIR)/bench_morton.c
VALIDATE_MORTON_SRCS := $(COMMON_SRCS) $(MORTON_DIR)/matmul_morton.c \
                        $(CORE_DIR)/morton.c $(VALIDATE_DIR)/validate_morton.c
BENCH_MORTON_ZEN5    := $(BIN_BENCH_DIR)/bench_morton_ZEN5
VALIDATE_MORTON_O0   := $(BIN_VALIDATE_DIR)/validate_morton_O0

# --- morton_avx512 ---
MORTON_AVX512_BASE_SRCS := $(MORTON_DIR)/matmul_morton_avx512.c \
                           $(CORE_DIR)/morton.c $(CORE_DIR)/matrix_utils.c \
                           $(NAIVE_DIR)/matmul_naive.c
BENCH_MORTON_AVX512_SRCS    := $(MORTON_AVX512_BASE_SRCS) \
                                $(BENCH_DIR)/bench_morton_avx512.c
VALIDATE_MORTON_AVX512_SRCS := $(MORTON_AVX512_BASE_SRCS) \
                                $(MORTON_DIR)/matmul_morton.c \
                                $(VALIDATE_DIR)/validate_morton_avx512.c
BENCH_MORTON_AVX512_ZEN5    := $(BIN_BENCH_DIR)/bench_morton_avx512_ZEN5
VALIDATE_MORTON_AVX512_ZEN5 := $(BIN_VALIDATE_DIR)/validate_morton_avx512_ZEN5

# --- morton_omp ---
MORTON_OMP_BASE_SRCS := $(MORTON_DIR)/matmul_morton_omp.c \
                        $(MORTON_DIR)/matmul_morton_avx512.c \
                        $(CORE_DIR)/morton.c $(CORE_DIR)/matrix_utils.c \
                        $(NAIVE_DIR)/matmul_naive.c
BENCH_MORTON_OMP_SRCS    := $(MORTON_OMP_BASE_SRCS) \
                             $(BENCH_DIR)/bench_morton_omp.c
VALIDATE_MORTON_OMP_SRCS := $(MORTON_OMP_BASE_SRCS) \
                             $(MORTON_DIR)/matmul_morton.c \
                             $(VALIDATE_DIR)/validate_morton_omp.c
BENCH_MORTON_OMP_ZEN5    := $(BIN_BENCH_DIR)/bench_morton_omp_ZEN5
VALIDATE_MORTON_OMP_ZEN5 := $(BIN_VALIDATE_DIR)/validate_morton_omp_ZEN5

# --- tiled_ikj ---
BENCH_TILED_IKJ_SRCS    := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj.c \
                            $(BENCH_DIR)/bench_tiled_ikj.c
VALIDATE_TILED_IKJ_SRCS := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj.c \
                            $(VALIDATE_DIR)/validate_tiled_ikj.c
BENCH_TILED_IKJ_ZEN5    := $(BIN_BENCH_DIR)/bench_tiled_ikj_ZEN5
VALIDATE_TILED_IKJ_O0   := $(BIN_VALIDATE_DIR)/validate_tiled_ikj_O0

# --- tiled_ikj_avx512 ---
BENCH_TILED_IKJ_AVX512_SRCS    := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj_avx512.c \
                                   $(BENCH_DIR)/bench_tiled_ikj_avx512.c
VALIDATE_TILED_IKJ_AVX512_SRCS := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj_avx512.c \
                                   $(VALIDATE_DIR)/validate_tiled_ikj_avx512.c
BENCH_TILED_IKJ_AVX512_ZEN5    := $(BIN_BENCH_DIR)/bench_tiled_ikj_avx512_ZEN5
VALIDATE_TILED_IKJ_AVX512_ZEN5 := $(BIN_VALIDATE_DIR)/validate_tiled_ikj_avx512_ZEN5

# --- tiled_ikj_omp ---
BENCH_TILED_IKJ_OMP_SRCS    := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj_omp.c \
                                $(BENCH_DIR)/bench_tiled_ikj_omp.c
VALIDATE_TILED_IKJ_OMP_SRCS := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj_omp.c \
                                $(VALIDATE_DIR)/validate_tiled_ikj_omp.c
BENCH_TILED_IKJ_OMP_ZEN5    := $(BIN_BENCH_DIR)/bench_tiled_ikj_omp_ZEN5
VALIDATE_TILED_IKJ_OMP_ZEN5 := $(BIN_VALIDATE_DIR)/validate_tiled_ikj_omp_ZEN5

# --- tests ---
# Unit tests for the building blocks (core/, microkernels/). Output goes
# to bin/tests/, parallel to bin/bench/ and bin/validate/ so each binary
# family lives in its own subdirectory.
TEST_MATRIX_UTILS_SRCS := $(CORE_DIR)/matrix_utils.c \
                          $(TESTS_DIR)/test_matrix_utils.c
TEST_MATRIX_UTILS      := $(BIN_TESTS_DIR)/test_matrix_utils

TEST_MORTON_SRCS          := $(CORE_DIR)/morton.c $(CORE_DIR)/matrix_utils.c \
                             $(TESTS_DIR)/test_morton.c
TEST_MORTON               := $(BIN_TESTS_DIR)/test_morton

TEST_KERNEL_AVX512_MORTON := $(BIN_TESTS_DIR)/test_kernel_avx512_morton
TEST_KERNEL_AVX512_TILED  := $(BIN_TESTS_DIR)/test_kernel_avx512_tiled

# Aggregate lists used by build / validate / results.
ALL_BENCH := \
    $(BENCH_NAIVE_ZEN5) \
    $(BENCH_LOOPS_ZEN5) \
    $(BENCH_MORTON_ZEN5) \
    $(BENCH_MORTON_AVX512_ZEN5) \
    $(BENCH_MORTON_OMP_ZEN5) \
    $(BENCH_TILED_IKJ_ZEN5) \
    $(BENCH_TILED_IKJ_AVX512_ZEN5) \
    $(BENCH_TILED_IKJ_OMP_ZEN5)

ALL_VALIDATE := \
    $(VALIDATE_NAIVE_O0) \
    $(VALIDATE_LOOPS_O0) \
    $(VALIDATE_MORTON_O0) \
    $(VALIDATE_TILED_IKJ_O0) \
    $(VALIDATE_MORTON_AVX512_ZEN5) \
    $(VALIDATE_MORTON_OMP_ZEN5) \
    $(VALIDATE_TILED_IKJ_AVX512_ZEN5) \
    $(VALIDATE_TILED_IKJ_OMP_ZEN5)

ALL_TESTS := \
    $(TEST_MATRIX_UTILS) \
    $(TEST_MORTON) \
    $(TEST_KERNEL_AVX512_MORTON) \
    $(TEST_KERNEL_AVX512_TILED)

# ── 5. .PHONY ─────────────────────────────────────────────────────────
.PHONY: all build tests validate results \
        bench_naive_ZEN5 validate_naive \
        bench_loops_ZEN5 validate_loops \
        bench_morton_ZEN5 validate_morton \
        bench_morton_avx512_ZEN5 validate_morton_avx512 \
        bench_morton_omp_ZEN5 validate_morton_omp \
        bench_tiled_ikj_ZEN5 validate_tiled_ikj \
        bench_tiled_ikj_avx512_ZEN5 validate_tiled_ikj_avx512 \
        bench_tiled_ikj_omp_ZEN5 validate_tiled_ikj_omp \
        test_matrix_utils test_morton \
        test_kernel_avx512_morton test_kernel_avx512_tiled \
        profile_zen5 profile_zen5_one profile_zen5_omp \
        consolidate_zen5 plots \
        clean distclean

# ── 6. MAIN TARGETS ───────────────────────────────────────────────────

# Compile all bench + validate + test binaries without running anything.
build: $(ALL_BENCH) $(ALL_VALIDATE) $(ALL_TESTS)

all: build

# Build and run all unit tests in sequence; stops on first failure.
# The unit tests guard the building blocks (matrix_utils, morton, the
# two AVX-512 microkernels) in isolation. They are cheap, so they run
# before validate as a fast pre-flight check.
tests: $(ALL_TESTS)
	@echo "=== test_matrix_utils ==="
	./$(TEST_MATRIX_UTILS)
	@echo "=== test_morton ==="
	./$(TEST_MORTON)
	@echo "=== test_kernel_avx512_morton ==="
	./$(TEST_KERNEL_AVX512_MORTON)
	@echo "=== test_kernel_avx512_tiled ==="
	./$(TEST_KERNEL_AVX512_TILED)
	@echo "All unit tests passed."

# Build and run all validate_* in sequence; stops on first failure.
# Depends on tests: if the building blocks are broken there is no point
# in running the algebraic invariants.
validate: tests $(ALL_VALIDATE)
	@echo "=== validate_naive ==="
	./$(VALIDATE_NAIVE_O0)
	@echo "=== validate_loops ==="
	./$(VALIDATE_LOOPS_O0)
	@echo "=== validate_morton ==="
	./$(VALIDATE_MORTON_O0)
	@echo "=== validate_tiled_ikj ==="
	./$(VALIDATE_TILED_IKJ_O0)
	@echo "=== validate_morton_avx512 ==="
	./$(VALIDATE_MORTON_AVX512_ZEN5)
	@echo "=== validate_morton_omp ==="
	./$(VALIDATE_MORTON_OMP_ZEN5)
	@echo "=== validate_tiled_ikj_avx512 ==="
	./$(VALIDATE_TILED_IKJ_AVX512_ZEN5)
	@echo "=== validate_tiled_ikj_omp ==="
	./$(VALIDATE_TILED_IKJ_OMP_ZEN5)
	@echo "All validations passed."

VARIANTS ?=
MS       ?=
VARIANT  ?= morton_avx512
M        ?= 4096

# Full benchmark run -> results/metrics.csv.
results: $(ALL_BENCH)
	$(if $(VARIANTS),VARIANTS="$(VARIANTS)" )$(if $(MS),MS="$(MS)" )bash scripts/run_perf_zen5_sweep.sh
	python3 scripts/consolidate_perf_zen5.py --out results/metrics.csv

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR) gmon.out perf.data perf.data.old cachegrind.out.*

distclean: clean
	rm -f results/*.csv plots/*

# ── 7. BUILD RULES ────────────────────────────────────────────────────
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_BENCH_DIR): | $(BIN_DIR)
	mkdir -p $(BIN_BENCH_DIR)

$(BIN_VALIDATE_DIR): | $(BIN_DIR)
	mkdir -p $(BIN_VALIDATE_DIR)

$(BIN_TESTS_DIR): | $(BIN_DIR)
	mkdir -p $(BIN_TESTS_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# naive
bench_naive_ZEN5: $(BENCH_NAIVE_ZEN5)
validate_naive:    $(VALIDATE_NAIVE_O0)

$(BENCH_NAIVE_ZEN5): $(BENCH_NAIVE_SRCS) | $(BIN_BENCH_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(BENCH_NAIVE_SRCS) -o $@ $(LIBS)

$(VALIDATE_NAIVE_O0): $(VALIDATE_NAIVE_SRCS) | $(BIN_VALIDATE_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_NAIVE_SRCS) -o $@ $(LIBS)

# loops
bench_loops_ZEN5: $(BENCH_LOOPS_ZEN5)
validate_loops:    $(VALIDATE_LOOPS_O0)

$(BENCH_LOOPS_ZEN5): $(BENCH_LOOPS_SRCS) | $(BIN_BENCH_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(BENCH_LOOPS_SRCS) -o $@ $(LIBS)

$(VALIDATE_LOOPS_O0): $(VALIDATE_LOOPS_SRCS) | $(BIN_VALIDATE_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_LOOPS_SRCS) -o $@ $(LIBS)

# morton
bench_morton_ZEN5: $(BENCH_MORTON_ZEN5)
validate_morton:    $(VALIDATE_MORTON_O0)

$(BENCH_MORTON_ZEN5): $(BENCH_MORTON_SRCS) | $(BIN_BENCH_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(BENCH_MORTON_SRCS) -o $@ $(LIBS)

$(VALIDATE_MORTON_O0): $(VALIDATE_MORTON_SRCS) | $(BIN_VALIDATE_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_MORTON_SRCS) -o $@ $(LIBS)

# morton_avx512
bench_morton_avx512_ZEN5: $(BENCH_MORTON_AVX512_ZEN5)
validate_morton_avx512:    $(VALIDATE_MORTON_AVX512_ZEN5)

$(BENCH_MORTON_AVX512_ZEN5): $(BENCH_MORTON_AVX512_SRCS) $(KERNEL_MORTON_H) | $(BIN_BENCH_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(BENCH_MORTON_AVX512_SRCS) -o $@ $(LIBS)

$(VALIDATE_MORTON_AVX512_ZEN5): $(VALIDATE_MORTON_AVX512_SRCS) $(KERNEL_MORTON_H) | $(BIN_VALIDATE_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(VALIDATE_MORTON_AVX512_SRCS) -o $@ $(LIBS)

# morton_omp
bench_morton_omp_ZEN5: $(BENCH_MORTON_OMP_ZEN5)
validate_morton_omp:    $(VALIDATE_MORTON_OMP_ZEN5)

$(BENCH_MORTON_OMP_ZEN5): $(BENCH_MORTON_OMP_SRCS) $(KERNEL_MORTON_H) | $(BIN_BENCH_DIR)
	$(CC) $(CFLAGS_OMP_ZEN5) $(BENCH_MORTON_OMP_SRCS) -o $@ $(LIBS)

$(VALIDATE_MORTON_OMP_ZEN5): $(VALIDATE_MORTON_OMP_SRCS) $(KERNEL_MORTON_H) | $(BIN_VALIDATE_DIR)
	$(CC) $(CFLAGS_OMP_ZEN5) $(VALIDATE_MORTON_OMP_SRCS) -o $@ $(LIBS)

# tiled_ikj
bench_tiled_ikj_ZEN5: $(BENCH_TILED_IKJ_ZEN5)
validate_tiled_ikj:    $(VALIDATE_TILED_IKJ_O0)

$(BENCH_TILED_IKJ_ZEN5): $(BENCH_TILED_IKJ_SRCS) | $(BIN_BENCH_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(BENCH_TILED_IKJ_SRCS) -o $@ $(LIBS)

$(VALIDATE_TILED_IKJ_O0): $(VALIDATE_TILED_IKJ_SRCS) | $(BIN_VALIDATE_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_TILED_IKJ_SRCS) -o $@ $(LIBS)

# tiled_ikj_avx512
bench_tiled_ikj_avx512_ZEN5: $(BENCH_TILED_IKJ_AVX512_ZEN5)
validate_tiled_ikj_avx512:    $(VALIDATE_TILED_IKJ_AVX512_ZEN5)

$(BENCH_TILED_IKJ_AVX512_ZEN5): $(BENCH_TILED_IKJ_AVX512_SRCS) $(KERNEL_TILED_H) | $(BIN_BENCH_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(BENCH_TILED_IKJ_AVX512_SRCS) -o $@ $(LIBS)

$(VALIDATE_TILED_IKJ_AVX512_ZEN5): $(VALIDATE_TILED_IKJ_AVX512_SRCS) $(KERNEL_TILED_H) | $(BIN_VALIDATE_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(VALIDATE_TILED_IKJ_AVX512_SRCS) -o $@ $(LIBS)

# tiled_ikj_omp
bench_tiled_ikj_omp_ZEN5: $(BENCH_TILED_IKJ_OMP_ZEN5)
validate_tiled_ikj_omp:    $(VALIDATE_TILED_IKJ_OMP_ZEN5)

$(BENCH_TILED_IKJ_OMP_ZEN5): $(BENCH_TILED_IKJ_OMP_SRCS) $(KERNEL_TILED_H) | $(BIN_BENCH_DIR)
	$(CC) $(CFLAGS_OMP_ZEN5) $(BENCH_TILED_IKJ_OMP_SRCS) -o $@ $(LIBS)

$(VALIDATE_TILED_IKJ_OMP_ZEN5): $(VALIDATE_TILED_IKJ_OMP_SRCS) $(KERNEL_TILED_H) | $(BIN_VALIDATE_DIR)
	$(CC) $(CFLAGS_OMP_ZEN5) $(VALIDATE_TILED_IKJ_OMP_SRCS) -o $@ $(LIBS)

# ── 8. PROFILE / SWEEP ────────────────────────────────────────────────

# Single-cell ad-hoc profiling.  Usage: make profile_zen5_one VARIANT=loop_ikj M=4096
profile_zen5_one: $(ALL_BENCH)
	bash scripts/profile_perf_zen5.sh $(VARIANT) $(M)

# Full hardware-counter sweep across all variants and sizes.
profile_zen5: $(ALL_BENCH)
	$(if $(VARIANTS),VARIANTS="$(VARIANTS)" )$(if $(MS),MS="$(MS)" )bash scripts/run_perf_zen5_sweep.sh
	python3 scripts/consolidate_perf_zen5.py

# OMP multi-thread perf at several sizes.  Knobs: THREADS_OMP BIND_OMP MS_OMP
THREADS_OMP ?= 8
BIND_OMP    ?= spread
MS_OMP      ?= 1024 4096 8192

profile_zen5_omp: $(BENCH_MORTON_OMP_ZEN5)
	@for m in $(MS_OMP); do \
	    echo "=== morton_omp m=$$m (OMP_NUM_THREADS=$(THREADS_OMP) bind=$(BIND_OMP)) ==="; \
	    OMP_NUM_THREADS=$(THREADS_OMP) OMP_PLACES=cores OMP_PROC_BIND=$(BIND_OMP) \
	        bash scripts/profile_perf_zen5.sh morton_omp $$m; \
	done

# Re-run the Python consolidator without repeating the perf sweep.
consolidate_zen5:
	python3 scripts/consolidate_perf_zen5.py

# ── 9. ANALYSIS & AUXILIARY ───────────────────────────────────────────

# Render the 4 figures (gflops_vs_m, best_per_family, llc_misses_vs_m,
# omp_scaling) into plots/ from results/metrics.csv. Requires
# matplotlib in the active Python environment (see README section 4.4
# for the venv setup). Does NOT depend on `results` so a replot is
# cheap and does not retrigger the multi-minute perf sweep; run
# `make results` first if results/metrics.csv is missing or stale.
plots:
	python3 scripts/plot_metrics_perf_zen5.py \
	    --csv results/metrics.csv --out-dir plots

# Unit tests (build + run). Each target builds + runs a single test;
# the aggregate target `tests` (Section 6) runs all of them in order.
# Test binaries live in bin/tests/, parallel to bin/bench/ and
# bin/validate/.
test_matrix_utils: $(TEST_MATRIX_UTILS)
	./$(TEST_MATRIX_UTILS)

$(TEST_MATRIX_UTILS): $(TEST_MATRIX_UTILS_SRCS) | $(BIN_TESTS_DIR)
	$(CC) $(BASE_CFLAGS) $(TEST_MATRIX_UTILS_SRCS) -o $@ $(LIBS)

test_morton: $(TEST_MORTON)
	./$(TEST_MORTON)

$(TEST_MORTON): $(TEST_MORTON_SRCS) | $(BIN_TESTS_DIR)
	$(CC) $(BASE_CFLAGS) $(TEST_MORTON_SRCS) -o $@ $(LIBS)

test_kernel_avx512_morton: $(TEST_KERNEL_AVX512_MORTON)
	./$(TEST_KERNEL_AVX512_MORTON)

$(TEST_KERNEL_AVX512_MORTON): $(TESTS_DIR)/test_kernel_avx512_morton.c \
                               $(KERNEL_MORTON_H) \
                               $(CORE_DIR)/matrix_utils.c \
                               $(CORE_DIR)/matrix_utils.h | $(BIN_TESTS_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) \
	      $(TESTS_DIR)/test_kernel_avx512_morton.c \
	      $(CORE_DIR)/matrix_utils.c \
	      -o $@ $(LIBS)

test_kernel_avx512_tiled: $(TEST_KERNEL_AVX512_TILED)
	./$(TEST_KERNEL_AVX512_TILED)

$(TEST_KERNEL_AVX512_TILED): $(TESTS_DIR)/test_kernel_avx512_tiled.c \
                              $(KERNEL_TILED_H) \
                              $(CORE_DIR)/matrix_utils.c \
                              $(CORE_DIR)/matrix_utils.h | $(BIN_TESTS_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) \
	      $(TESTS_DIR)/test_kernel_avx512_tiled.c \
	      $(CORE_DIR)/matrix_utils.c \
	      -o $@ $(LIBS)

