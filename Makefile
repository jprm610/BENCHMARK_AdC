#
# Makefile for the iterated matmul benchmark (Ryzen 5 4600H / Zen 2).
#
# Source layout:
#   src/core/             matrix_utils.{c,h}, timing.h, morton.{c,h}
#   src/microkernels/     kernel_avx2_morton.h, kernel_avx2_tiled.h
#   src/algorithms/
#     naive/              matmul_naive.{c,h}
#     loops/              matmul_loops.{c,h}
#     morton/             matmul_morton{,_avx2,_omp}.{c,h}
#     tiled_ikj/          matmul_tiled_ikj{,_avx2,_omp}.{c,h}
#   src/drivers/bench/    bench_*.c  (one per algorithm)
#   src/drivers/validate/ validate_*.c
#   src/tests/            test_morton.c, test_kernel_avx2.c
#
# Key targets:
#   make build         -> compile all bench + validate binaries
#   make validate_all  -> build + run all validate_* (correctness gate)
#   make results       -> build + perf sweep -> results/metrics.csv
#   make clean         -> remove binaries and object files
#   make distclean     -> clean + remove results/*.csv and plots/*
#
# Sweep knobs (pass on the command line):
#   MS=<sizes>   size list for run_perf_zen2_sweep.sh  (e.g. MS="1024 4096")
#   VARIANT=<v>  single variant for profile_zen2_one   (default: morton_avx2)
#   M=<size>     single size   for profile_zen2_one    (default: 4096)
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

BIN_DIR := bin
OBJ_DIR := build

INCS := -I$(CORE_DIR) -I$(MK_DIR) \
        -I$(NAIVE_DIR) -I$(LOOPS_DIR) -I$(MORTON_DIR) -I$(TILED_DIR)

# ── 3. COMPILER FLAGS ─────────────────────────────────────────────────
# -D_POSIX_C_SOURCE=200809L exposes clock_gettime under -std=c11.
# -fno-omit-frame-pointer keeps perf call-graphs working.
BASE_CFLAGS     := $(CSTD) $(WARN) $(INCS) -O0 -g -fno-omit-frame-pointer \
                   -D_POSIX_C_SOURCE=200809L
CFLAGS_O3       := $(CSTD) $(WARN) $(INCS) -O3
CFLAGS_O3_ZEN2  := $(CSTD) $(WARN) $(INCS) -O3 -march=znver2 -mavx2 -mfma \
                   -D_POSIX_C_SOURCE=200809L
CFLAGS_OMP_ZEN2 := $(CFLAGS_O3_ZEN2) -fopenmp

# ── 4. SOURCES & BINARY PATHS ─────────────────────────────────────────
# Modules shared by every algorithm family (reference + matrix helpers).
COMMON_SRCS := $(NAIVE_DIR)/matmul_naive.c $(CORE_DIR)/matrix_utils.c

# Header-only microkernels: listed as explicit prerequisites so Make
# recompiles all dependents when a kernel header changes.
KERNEL_MORTON_H := $(MK_DIR)/kernel_avx2_morton.h
KERNEL_TILED_H  := $(MK_DIR)/kernel_avx2_tiled.h

# --- naive ---
BENCH_NAIVE_SRCS    := $(COMMON_SRCS) $(BENCH_DIR)/bench_naive.c
VALIDATE_NAIVE_SRCS := $(COMMON_SRCS) $(VALIDATE_DIR)/validate_naive.c
BENCH_NAIVE_O3    := $(BIN_DIR)/bench_naive_O3
VALIDATE_NAIVE_O0 := $(BIN_DIR)/validate_naive_O0

# --- loops ---
BENCH_LOOPS_SRCS    := $(COMMON_SRCS) $(LOOPS_DIR)/matmul_loops.c \
                       $(BENCH_DIR)/bench_loops.c
VALIDATE_LOOPS_SRCS := $(COMMON_SRCS) $(LOOPS_DIR)/matmul_loops.c \
                       $(VALIDATE_DIR)/validate_loops.c
BENCH_LOOPS_O3    := $(BIN_DIR)/bench_loops_O3
VALIDATE_LOOPS_O0 := $(BIN_DIR)/validate_loops_O0

# --- morton ---
BENCH_MORTON_SRCS    := $(COMMON_SRCS) $(MORTON_DIR)/matmul_morton.c \
                        $(CORE_DIR)/morton.c $(BENCH_DIR)/bench_morton.c
VALIDATE_MORTON_SRCS := $(COMMON_SRCS) $(MORTON_DIR)/matmul_morton.c \
                        $(CORE_DIR)/morton.c $(VALIDATE_DIR)/validate_morton.c
BENCH_MORTON_O3    := $(BIN_DIR)/bench_morton_O3
VALIDATE_MORTON_O0 := $(BIN_DIR)/validate_morton_O0

# --- morton_avx2 ---
MORTON_AVX2_BASE_SRCS := $(MORTON_DIR)/matmul_morton_avx2.c \
                         $(CORE_DIR)/morton.c $(CORE_DIR)/matrix_utils.c \
                         $(NAIVE_DIR)/matmul_naive.c
BENCH_MORTON_AVX2_SRCS    := $(MORTON_AVX2_BASE_SRCS) \
                              $(BENCH_DIR)/bench_morton_avx2.c
VALIDATE_MORTON_AVX2_SRCS := $(MORTON_AVX2_BASE_SRCS) \
                              $(MORTON_DIR)/matmul_morton.c \
                              $(VALIDATE_DIR)/validate_morton_avx2.c
BENCH_MORTON_AVX2_O3    := $(BIN_DIR)/bench_morton_avx2_O3
VALIDATE_MORTON_AVX2_O3 := $(BIN_DIR)/validate_morton_avx2_O3

# --- morton_omp ---
MORTON_OMP_BASE_SRCS := $(MORTON_DIR)/matmul_morton_omp.c \
                        $(MORTON_DIR)/matmul_morton_avx2.c \
                        $(CORE_DIR)/morton.c $(CORE_DIR)/matrix_utils.c \
                        $(NAIVE_DIR)/matmul_naive.c
BENCH_MORTON_OMP_SRCS    := $(MORTON_OMP_BASE_SRCS) \
                             $(BENCH_DIR)/bench_morton_omp.c
VALIDATE_MORTON_OMP_SRCS := $(MORTON_OMP_BASE_SRCS) \
                             $(MORTON_DIR)/matmul_morton.c \
                             $(VALIDATE_DIR)/validate_morton_omp.c
BENCH_MORTON_OMP_O3    := $(BIN_DIR)/bench_morton_omp_O3
VALIDATE_MORTON_OMP_O3 := $(BIN_DIR)/validate_morton_omp_O3

# --- tiled_ikj ---
BENCH_TILED_IKJ_SRCS    := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj.c \
                            $(BENCH_DIR)/bench_tiled_ikj.c
VALIDATE_TILED_IKJ_SRCS := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj.c \
                            $(VALIDATE_DIR)/validate_tiled_ikj.c
BENCH_TILED_IKJ_O3    := $(BIN_DIR)/bench_tiled_ikj_O3
VALIDATE_TILED_IKJ_O0 := $(BIN_DIR)/validate_tiled_ikj_O0

# --- tiled_ikj_avx2 ---
BENCH_TILED_IKJ_AVX2_SRCS    := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj_avx2.c \
                                 $(BENCH_DIR)/bench_tiled_ikj_avx2.c
VALIDATE_TILED_IKJ_AVX2_SRCS := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj_avx2.c \
                                 $(VALIDATE_DIR)/validate_tiled_ikj_avx2.c
BENCH_TILED_IKJ_AVX2_O3    := $(BIN_DIR)/bench_tiled_ikj_avx2_O3
VALIDATE_TILED_IKJ_AVX2_O3 := $(BIN_DIR)/validate_tiled_ikj_avx2_O3

# --- tiled_ikj_omp ---
BENCH_TILED_IKJ_OMP_SRCS    := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj_omp.c \
                                $(BENCH_DIR)/bench_tiled_ikj_omp.c
VALIDATE_TILED_IKJ_OMP_SRCS := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj_omp.c \
                                $(VALIDATE_DIR)/validate_tiled_ikj_omp.c
BENCH_TILED_IKJ_OMP_O3    := $(BIN_DIR)/bench_tiled_ikj_omp_O3
VALIDATE_TILED_IKJ_OMP_O3 := $(BIN_DIR)/validate_tiled_ikj_omp_O3

# --- tests ---
TEST_MORTON_SRCS := $(CORE_DIR)/morton.c $(CORE_DIR)/matrix_utils.c \
                    $(TESTS_DIR)/test_morton.c
TEST_MORTON      := $(BIN_DIR)/test_morton
TEST_KERNEL_AVX2 := $(BIN_DIR)/test_kernel_avx2

# Aggregate lists used by build / validate_all / results.
ALL_BENCH := \
    $(BENCH_NAIVE_O3) \
    $(BENCH_LOOPS_O3) \
    $(BENCH_MORTON_O3) \
    $(BENCH_MORTON_AVX2_O3) \
    $(BENCH_MORTON_OMP_O3) \
    $(BENCH_TILED_IKJ_O3) \
    $(BENCH_TILED_IKJ_AVX2_O3) \
    $(BENCH_TILED_IKJ_OMP_O3)

ALL_VALIDATE := \
    $(VALIDATE_NAIVE_O0) \
    $(VALIDATE_LOOPS_O0) \
    $(VALIDATE_MORTON_O0) \
    $(VALIDATE_MORTON_AVX2_O3) \
    $(VALIDATE_MORTON_OMP_O3) \
    $(VALIDATE_TILED_IKJ_O0) \
    $(VALIDATE_TILED_IKJ_AVX2_O3) \
    $(VALIDATE_TILED_IKJ_OMP_O3)

# ── 5. .PHONY ─────────────────────────────────────────────────────────
.PHONY: all build validate_all results \
        bench_naive_O3 validate_naive \
        bench_loops_O3 validate_loops \
        bench_morton_O3 validate_morton \
        bench_morton_avx2_O3 validate_morton_avx2 \
        bench_morton_omp_O3 validate_morton_omp \
        bench_tiled_ikj_O3 validate_tiled_ikj \
        bench_tiled_ikj_avx2_O3 validate_tiled_ikj_avx2 \
        bench_tiled_ikj_omp_O3 validate_tiled_ikj_omp \
        test_morton test_kernel_avx2 \
        profile_zen2 profile_zen2_one profile_zen2_omp \
        consolidate_zen2 plot_perf_zen2 \
        clean distclean

# ── 6. MAIN TARGETS ───────────────────────────────────────────────────

# Compile all bench + validate binaries without running anything.
build: $(ALL_BENCH) $(ALL_VALIDATE)

all: build

# Build and run all validate_* in sequence; stops on first failure.
validate_all: $(ALL_VALIDATE)
	@echo "=== validate_naive ==="
	./$(VALIDATE_NAIVE_O0)
	@echo "=== validate_loops ==="
	./$(VALIDATE_LOOPS_O0)
	@echo "=== validate_morton ==="
	./$(VALIDATE_MORTON_O0)
	@echo "=== validate_morton_avx2 ==="
	./$(VALIDATE_MORTON_AVX2_O3)
	@echo "=== validate_morton_omp ==="
	./$(VALIDATE_MORTON_OMP_O3)
	@echo "=== validate_tiled_ikj ==="
	./$(VALIDATE_TILED_IKJ_O0)
	@echo "=== validate_tiled_ikj_avx2 ==="
	./$(VALIDATE_TILED_IKJ_AVX2_O3)
	@echo "=== validate_tiled_ikj_omp ==="
	./$(VALIDATE_TILED_IKJ_OMP_O3)
	@echo "All validations passed."

VARIANTS ?=
MS       ?=
VARIANT  ?= morton_avx2
M        ?= 4096

# Full benchmark run -> results/metrics.csv.
results: $(ALL_BENCH)
	$(if $(VARIANTS),VARIANTS="$(VARIANTS)" )$(if $(MS),MS="$(MS)" )bash scripts/run_perf_zen2_sweep.sh
	python3 scripts/consolidate_perf_zen2.py --out results/metrics.csv

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR) gmon.out perf.data perf.data.old cachegrind.out.*

distclean: clean
	rm -f results/*.csv plots/*

# ── 7. BUILD RULES ────────────────────────────────────────────────────
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# naive
bench_naive_O3: $(BENCH_NAIVE_O3)
validate_naive:  $(VALIDATE_NAIVE_O0)

$(BENCH_NAIVE_O3): $(BENCH_NAIVE_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_NAIVE_SRCS) -o $@ $(LIBS)

$(VALIDATE_NAIVE_O0): $(VALIDATE_NAIVE_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_NAIVE_SRCS) -o $@ $(LIBS)

# loops
bench_loops_O3: $(BENCH_LOOPS_O3)
validate_loops:  $(VALIDATE_LOOPS_O0)

$(BENCH_LOOPS_O3): $(BENCH_LOOPS_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_LOOPS_SRCS) -o $@ $(LIBS)

$(VALIDATE_LOOPS_O0): $(VALIDATE_LOOPS_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_LOOPS_SRCS) -o $@ $(LIBS)

# morton
bench_morton_O3: $(BENCH_MORTON_O3)
validate_morton:  $(VALIDATE_MORTON_O0)

$(BENCH_MORTON_O3): $(BENCH_MORTON_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_MORTON_SRCS) -o $@ $(LIBS)

$(VALIDATE_MORTON_O0): $(VALIDATE_MORTON_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_MORTON_SRCS) -o $@ $(LIBS)

# morton_avx2
bench_morton_avx2_O3: $(BENCH_MORTON_AVX2_O3)
validate_morton_avx2:  $(VALIDATE_MORTON_AVX2_O3)

$(BENCH_MORTON_AVX2_O3): $(BENCH_MORTON_AVX2_SRCS) $(KERNEL_MORTON_H) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_MORTON_AVX2_SRCS) -o $@ $(LIBS)

$(VALIDATE_MORTON_AVX2_O3): $(VALIDATE_MORTON_AVX2_SRCS) $(KERNEL_MORTON_H) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(VALIDATE_MORTON_AVX2_SRCS) -o $@ $(LIBS)

# morton_omp
bench_morton_omp_O3: $(BENCH_MORTON_OMP_O3)
validate_morton_omp:  $(VALIDATE_MORTON_OMP_O3)

$(BENCH_MORTON_OMP_O3): $(BENCH_MORTON_OMP_SRCS) $(KERNEL_MORTON_H) | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN2) $(BENCH_MORTON_OMP_SRCS) -o $@ $(LIBS)

$(VALIDATE_MORTON_OMP_O3): $(VALIDATE_MORTON_OMP_SRCS) $(KERNEL_MORTON_H) | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN2) $(VALIDATE_MORTON_OMP_SRCS) -o $@ $(LIBS)

# tiled_ikj
bench_tiled_ikj_O3: $(BENCH_TILED_IKJ_O3)
validate_tiled_ikj:  $(VALIDATE_TILED_IKJ_O0)

$(BENCH_TILED_IKJ_O3): $(BENCH_TILED_IKJ_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_TILED_IKJ_SRCS) -o $@ $(LIBS)

$(VALIDATE_TILED_IKJ_O0): $(VALIDATE_TILED_IKJ_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_TILED_IKJ_SRCS) -o $@ $(LIBS)

# tiled_ikj_avx2
bench_tiled_ikj_avx2_O3: $(BENCH_TILED_IKJ_AVX2_O3)
validate_tiled_ikj_avx2:  $(VALIDATE_TILED_IKJ_AVX2_O3)

$(BENCH_TILED_IKJ_AVX2_O3): $(BENCH_TILED_IKJ_AVX2_SRCS) $(KERNEL_TILED_H) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(BENCH_TILED_IKJ_AVX2_SRCS) -o $@ $(LIBS)

$(VALIDATE_TILED_IKJ_AVX2_O3): $(VALIDATE_TILED_IKJ_AVX2_SRCS) $(KERNEL_TILED_H) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) $(VALIDATE_TILED_IKJ_AVX2_SRCS) -o $@ $(LIBS)

# tiled_ikj_omp
bench_tiled_ikj_omp_O3: $(BENCH_TILED_IKJ_OMP_O3)
validate_tiled_ikj_omp:  $(VALIDATE_TILED_IKJ_OMP_O3)

$(BENCH_TILED_IKJ_OMP_O3): $(BENCH_TILED_IKJ_OMP_SRCS) $(KERNEL_TILED_H) | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN2) $(BENCH_TILED_IKJ_OMP_SRCS) -o $@ $(LIBS)

$(VALIDATE_TILED_IKJ_OMP_O3): $(VALIDATE_TILED_IKJ_OMP_SRCS) $(KERNEL_TILED_H) | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN2) $(VALIDATE_TILED_IKJ_OMP_SRCS) -o $@ $(LIBS)

# ── 8. PROFILE / SWEEP ────────────────────────────────────────────────

# Single-cell ad-hoc profiling.  Usage: make profile_zen2_one VARIANT=loop_ikj M=4096
profile_zen2_one: $(ALL_BENCH)
	bash scripts/profile_perf_zen2.sh $(VARIANT) $(M)

# Full hardware-counter sweep across all variants and sizes.
profile_zen2: $(ALL_BENCH)
	$(if $(VARIANTS),VARIANTS="$(VARIANTS)" )$(if $(MS),MS="$(MS)" )bash scripts/run_perf_zen2_sweep.sh
	python3 scripts/consolidate_perf_zen2.py

# OMP multi-thread perf at several sizes.  Knobs: THREADS_OMP BIND_OMP MS_OMP
THREADS_OMP ?= 6
BIND_OMP    ?= spread
MS_OMP      ?= 1024 4096 8192

profile_zen2_omp: $(BENCH_MORTON_OMP_O3)
	@for m in $(MS_OMP); do \
	    echo "=== morton_omp m=$$m (OMP_NUM_THREADS=$(THREADS_OMP) bind=$(BIND_OMP)) ==="; \
	    OMP_NUM_THREADS=$(THREADS_OMP) OMP_PLACES=cores OMP_PROC_BIND=$(BIND_OMP) \
	        bash scripts/profile_perf_zen2.sh morton_omp $$m; \
	done

# Re-run the Python consolidator without repeating the perf sweep.
consolidate_zen2:
	python3 scripts/consolidate_perf_zen2.py

# ── 9. ANALYSIS & AUXILIARY ───────────────────────────────────────────

plot_perf_zen2:
	python3 scripts/plot_perf_zen2.py

# Unit tests (build + run).
test_morton: $(TEST_MORTON)
	./$(TEST_MORTON)

$(TEST_MORTON): $(TEST_MORTON_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(TEST_MORTON_SRCS) -o $@ $(LIBS)

test_kernel_avx2: $(TEST_KERNEL_AVX2)
	./$(TEST_KERNEL_AVX2)

$(TEST_KERNEL_AVX2): $(TESTS_DIR)/test_kernel_avx2.c \
                     $(KERNEL_MORTON_H) \
                     $(CORE_DIR)/matrix_utils.c $(CORE_DIR)/matrix_utils.h \
                     $(NAIVE_DIR)/matmul_naive.h | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN2) \
	      $(TESTS_DIR)/test_kernel_avx2.c \
	      $(CORE_DIR)/matrix_utils.c \
	      -o $@ $(LIBS)
