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
	rm -rf $(BIN_DIR) $(OBJ_DIR) gmon.out perf.data perf.data.old cachegrind.out.*

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
