#
# Makefile for the iterated matmul benchmark - Phase 1 (baseline).
#
# Targets:
#   make            -> bench_O0 and validate_O0 (default)
#   make bench_O0   -> baseline benchmark binary (no compiler optimization)
#   make bench_pg   -> same as bench_O0 but compiled with -pg for gprof
#   make validate   -> validate_O0 binary that runs the algebraic sanity tests
#   make sweep      -> runs scripts/run_sweep.sh after building bench_O0
#   make clean      -> remove binaries and object files (keeps CSV/plots)
#   make distclean  -> clean plus remove results/*.csv and plots/*
#
# The point of having both bench_O0 and bench_pg as separate binaries is
# so that the gprof instrumentation does not contaminate timing on the
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
BASE_CFLAGS := $(CSTD) $(WARN) $(INCS) -O0 -g -fno-omit-frame-pointer

SRC_DIR  := src
BIN_DIR  := bin
OBJ_DIR  := build

# Common module list (kernel + helpers) used by both bench and validate.
COMMON_SRCS := $(SRC_DIR)/matmul_naive.c $(SRC_DIR)/matrix_utils.c

BENCH_SRCS    := $(COMMON_SRCS) $(SRC_DIR)/benchmark.c
VALIDATE_SRCS := $(COMMON_SRCS) $(SRC_DIR)/validate.c

BENCH_O0     := $(BIN_DIR)/bench_O0
BENCH_PG     := $(BIN_DIR)/bench_pg
VALIDATE_O0  := $(BIN_DIR)/validate_O0

.PHONY: all bench_O0 bench_pg validate sweep profile_gprof profile_perf clean distclean

all: $(BENCH_O0) $(VALIDATE_O0)

bench_O0: $(BENCH_O0)
bench_pg: $(BENCH_PG)
validate: $(VALIDATE_O0)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BENCH_O0): $(BENCH_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(BENCH_SRCS) -o $@ $(LIBS)

# -pg instruments the binary so that gprof can read gmon.out.
$(BENCH_PG): $(BENCH_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) -pg $(BENCH_SRCS) -o $@ $(LIBS)

$(VALIDATE_O0): $(VALIDATE_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_SRCS) -o $@ $(LIBS)

sweep: $(BENCH_O0)
	bash scripts/run_sweep.sh

profile_gprof: $(BENCH_PG)
	bash scripts/profile_gprof.sh

profile_perf: $(BENCH_O0)
	bash scripts/profile_perf.sh

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR) gmon.out perf.data perf.data.old cachegrind.out.*

distclean: clean
	rm -f results/*.csv plots/*
