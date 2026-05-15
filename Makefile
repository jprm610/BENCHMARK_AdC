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
