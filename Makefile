#
# Makefile for the iterated matmul benchmark - main_server variant.
#
# Target hardware: AMD EPYC 9R45 (Zen 5) on AWS c8a.2xlarge.
#   8 cores, 1 thread/core (SMT disabled by hypervisor)
#   L1d 48 KiB per core, L2 1 MiB per core, L3 32 MiB shared
#   Full AVX-512 ISA (F + VL + BW + DQ + IFMA + ...)
#
# All vectorized targets are AVX-512 (kernel_avx512_morton.h and
# kernel_avx512_tiled.h, both header-only `static inline`). The Zen 2
# / AVX2 dispatch path was removed because the server CPU supports
# AVX-512 natively and the wider tile delivers ~2x the FMA throughput
# per cycle. Only baseline (-O0, scalar) and Zen 5 (-march=native +
# AVX-512) targets remain.
#
# Source layout:
#
#   src/core/          matrix_utils.{c,h}, timing.h, morton.{c,h}
#   src/microkernels/  kernel_avx512_morton.h   (4x32, Morton family)
#                      kernel_avx512_tiled.h    (6x32, tiled_ikj family)
#   src/algorithms/
#     naive/           matmul_naive.{c,h}
#     loops/           matmul_loops.{c,h}
#     morton/          matmul_morton{,_avx512,_omp}.{c,h}
#     tiled_ikj/       matmul_tiled_ikj{,_avx512,_omp}.{c,h}
#   src/drivers/
#     bench/           bench_*.c        (one per algorithm)
#     validate/        validate_*.c     (one per algorithm)
#   src/tests/         test_morton.c, test_kernel_avx512_morton.c
#   src/tools/         hwinfo.c
#
# Common targets:
#   make                    -> bench_naive_O0 and validate_naive_O0
#   make bench_naive_O0     -> baseline benchmark (no optimization)
#   make bench_naive_pg     -> baseline + -pg for gprof
#   make validate_naive     -> baseline algebraic sanity tests
#   make sweep_naive        -> scripts/run_sweep_naive.sh
#   make audit              -> PDEP/PEXT auditor (Zen-family hygiene)
#   make hwinfo             -> runtime CPU fingerprint
#   make test_morton        -> Morton unit tests
#   make test_kernel_avx512_morton -> AVX-512 microkernel unit test
#
#   make results [M=...]      -> build everything + run perf sweep (Zen 5)
#                                + write results/metrics.csv
#                                e.g. `make results M=2048` runs just m=2048.
#   make profile_zen5 [M=...] -> same sweep but assuming bins are cached
#   make profile_zen5_one VARIANT=... M=...
#                             -> single (variant, m) cell
#   make consolidate_zen5     -> re-parse existing perf files
#
#   make clean              -> remove bin/, build/, perf artifacts
#   make distclean          -> clean + remove results/ and plots/
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

# Baseline flags: -O0 + debug symbols for gprof/perf, frame pointer
# preserved so perf call graphs work without DWARF unwinding.
# _POSIX_C_SOURCE=200809L exposes clock_gettime under -std=c11.
BASE_CFLAGS := $(CSTD) $(WARN) $(INCS) -O0 -g -fno-omit-frame-pointer \
               -D_POSIX_C_SOURCE=200809L

# Common modules (kernel + helpers) used by every bench and validate.
COMMON_SRCS := $(NAIVE_DIR)/matmul_naive.c $(CORE_DIR)/matrix_utils.c

BENCH_SRCS    := $(COMMON_SRCS) $(BENCH_DIR)/bench_naive.c
VALIDATE_SRCS := $(COMMON_SRCS) $(VALIDATE_DIR)/validate_naive.c

BENCH_NAIVE_O0     := $(BIN_DIR)/bench_naive_O0
BENCH_NAIVE_PG     := $(BIN_DIR)/bench_naive_pg
VALIDATE_NAIVE_O0  := $(BIN_DIR)/validate_naive_O0

.PHONY: all bench_naive_O0 bench_naive_pg validate_naive sweep_naive \
        profile_gprof_naive profile_perf_naive clean distclean

all: $(BENCH_NAIVE_O0) $(VALIDATE_NAIVE_O0)

bench_naive_O0: $(BENCH_NAIVE_O0)
bench_naive_pg: $(BENCH_NAIVE_PG)
validate_naive: $(VALIDATE_NAIVE_O0)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BENCH_NAIVE_O0): $(BENCH_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(BENCH_SRCS) -o $@ $(LIBS)

# -pg instruments the binary so gprof can read gmon.out.
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
# Baseline support module - Morton (Z-order) encoding tests
# =====================================================================

TEST_MORTON_SRCS := $(CORE_DIR)/morton.c $(CORE_DIR)/matrix_utils.c \
                    $(TESTS_DIR)/test_morton.c
TEST_MORTON      := $(BIN_DIR)/test_morton

.PHONY: test_morton

test_morton: $(TEST_MORTON)

$(TEST_MORTON): $(TEST_MORTON_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(TEST_MORTON_SRCS) -o $@ $(LIBS)

# =====================================================================
# Baseline Morton kernel bench and validate (-O0, no vectorization)
#
# validate_morton cross-validates the Morton kernel against
# matmul_naive (Test 4). matmul_naive is the canonical reference.
# =====================================================================

MORTON_KERNEL_SRCS       := $(MORTON_DIR)/matmul_morton.c $(CORE_DIR)/morton.c
BENCH_MORTON_SRCS        := $(COMMON_SRCS) $(MORTON_KERNEL_SRCS) \
                            $(BENCH_DIR)/bench_morton.c
VALIDATE_MORTON_SRCS     := $(COMMON_SRCS) $(MORTON_KERNEL_SRCS) \
                            $(VALIDATE_DIR)/validate_morton.c

BENCH_MORTON_O0          := $(BIN_DIR)/bench_morton_O0
VALIDATE_MORTON_O0       := $(BIN_DIR)/validate_morton_O0

.PHONY: bench_morton validate_morton sweep_morton_run

bench_morton: $(BENCH_MORTON_O0)
validate_morton: $(VALIDATE_MORTON_O0)

$(BENCH_MORTON_O0): $(BENCH_MORTON_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(BENCH_MORTON_SRCS) -o $@ $(LIBS)

$(VALIDATE_MORTON_O0): $(VALIDATE_MORTON_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_MORTON_SRCS) -o $@ $(LIBS)

sweep_morton_run: $(BENCH_MORTON_O0)
	bash scripts/run_sweep_morton.sh

# =====================================================================
# Baseline loop-reorder bench and validate (-O0)
#
# bench_loops_O0 : runtime kernel selection by name (6 orders)
# validate_loops : algebraic + cross-validation for all 6 orders
# =====================================================================

LOOPS_COMMON_SRCS   := $(COMMON_SRCS) $(LOOPS_DIR)/matmul_loops.c
BENCH_LOOPS_SRCS    := $(LOOPS_COMMON_SRCS) $(BENCH_DIR)/bench_loops.c
VALIDATE_LOOPS_SRCS := $(LOOPS_COMMON_SRCS) $(VALIDATE_DIR)/validate_loops.c

BENCH_LOOPS_O0      := $(BIN_DIR)/bench_loops_O0
VALIDATE_LOOPS_O0   := $(BIN_DIR)/validate_loops_O0

.PHONY: bench_loops validate_loops \
        sweep_loops_ijk sweep_loops_ikj sweep_loops_jik \
        sweep_loops_jki sweep_loops_kij sweep_loops_kji \
        sweep_loops_all

bench_loops: $(BENCH_LOOPS_O0)
validate_loops: $(VALIDATE_LOOPS_O0)

$(BENCH_LOOPS_O0): $(BENCH_LOOPS_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(BENCH_LOOPS_SRCS) -o $@ $(LIBS)

$(VALIDATE_LOOPS_O0): $(VALIDATE_LOOPS_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_LOOPS_SRCS) -o $@ $(LIBS)

# Per-order targets: each runs in its own process to avoid
# cross-contamination through page caches.
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

sweep_loops_all: sweep_loops_ijk sweep_loops_ikj sweep_loops_jik \
                 sweep_loops_jki sweep_loops_kij sweep_loops_kji
	@echo "kernel,m,n,num_iters,median_seconds,gflops" > results/loop_order.csv
	@for f in results/loop_ijk.csv results/loop_ikj.csv results/loop_jik.csv \
	           results/loop_jki.csv results/loop_kij.csv results/loop_kji.csv; do \
	    tail -n +2 "$$f" >> results/loop_order.csv; \
	done
	@echo "  -> results/loop_order.csv (combined)"

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
# Baseline tiled_ikj bench and validate (-O0, no vectorization)
#
# The intrinsic-using siblings (matmul_tiled_ikj_avx512 / _omp) include
# kernel_avx512_tiled.h, which requires AVX-512 ISA. They are built
# under the Zen 5 block further below; at -O0 / baseline we only build
# matmul_tiled_ikj (scalar tiling).
# =====================================================================

TILED_IKJ_COMMON_SRCS    := $(COMMON_SRCS) $(TILED_DIR)/matmul_tiled_ikj.c
BENCH_TILED_IKJ_SRCS     := $(TILED_IKJ_COMMON_SRCS) \
                            $(BENCH_DIR)/bench_tiled_ikj.c
VALIDATE_TILED_IKJ_SRCS  := $(TILED_IKJ_COMMON_SRCS) \
                            $(VALIDATE_DIR)/validate_tiled_ikj.c

VALIDATE_TILED_IKJ_O0    := $(BIN_DIR)/validate_tiled_ikj_O0

.PHONY: validate_tiled_ikj

validate_tiled_ikj: $(VALIDATE_TILED_IKJ_O0)

$(VALIDATE_TILED_IKJ_O0): $(VALIDATE_TILED_IKJ_SRCS) | $(BIN_DIR)
	$(CC) $(BASE_CFLAGS) $(VALIDATE_TILED_IKJ_SRCS) -o $@ $(LIBS)

# =====================================================================
# Audit and hardware info (Zen-family hygiene)
#
#   audit  : verifies that no Morton TU uses PDEP/PEXT intrinsics nor
#            triggers the compiler to emit them. PDEP/PEXT are
#            microcoded on Zen 2 (~18 cycles) and remain non-optimal
#            on Zen 5; the audit is a precaution that travels with
#            every architecture.
#   hwinfo : runtime CPU fingerprint (model, cores/threads, cache
#            sizes, RAM, AVX-512 + AVX2 + FMA + BMI2 support). Useful
#            as the first line of every results CSV.
# =====================================================================

CFLAGS_O3 := $(CSTD) $(WARN) $(INCS) -O3

HWINFO_BIN := $(BIN_DIR)/hwinfo

.PHONY: audit hwinfo

audit:
	bash scripts/audit_no_pdep.sh

hwinfo: $(HWINFO_BIN)
	./$(HWINFO_BIN)

$(HWINFO_BIN): $(TOOLS_DIR)/hwinfo.c | $(BIN_DIR)
	$(CC) $(CFLAGS_O3) -o $@ $< $(LIBS)

# =====================================================================
# AWS c8a.2xlarge - AMD EPYC 9R45 / Zen 5
#
# GCC 11 may not recognise -march=znver4/5 by name; -march=native
# reads CPUID at compile time and enables AVX-512F/BW/VL/DQ/IFMA and
# the rest of the EPYC 9R45 extensions automatically.
#
# Compile-time tunables (override on the make command line):
#
#   -DMORTON_AVX512_THRESHOLD_DEFAULT     leaf size for matmul_morton_avx512
#   -DMORTON_OMP_RECURSION_THRESHOLD_DEFAULT  leaf size for matmul_morton_omp
#   -DMORTON_OMP_PARALLEL_THRESHOLD_DEFAULT   task-spawn cutoff
#   -DTILED_IKJ_MC_DEFAULT  outer mc tile for the scalar tiled_ikj
#   -DTILED_IKJ_KC_DEFAULT  outer kc tile for the scalar tiled_ikj
#   -DTILED_IKJ_AVX512_BS_DEFAULT  inner kc panel depth for tiled_avx512
#                                (B panel = kc * NR * 4 B; want <= L1d=48 KiB)
#   -DTILED_IKJ_AVX512_MC          outer mc tile for tiled_avx512
#                                (A panel = mc * kc * 4 B; want <= L2=1 MiB)
#   -DTILED_IKJ_OMP_BS_DEFAULT   inner kc panel depth for tiled_omp
#   -DTILED_IKJ_OMP_MC           outer mc tile for tiled_omp
#
# Defaults below picked for the EPYC 9R45: kc=256 (B panel 32 KiB,
# fits L1d=48 KiB), mc=288 (A panel 288 KiB, fits L2=1 MiB).
#
# The KVM on AWS exposes only generic hardware events; no AMD raw
# events. Perf collection uses a single uncontested group, no
# multiplexing required (see scripts/profile_perf_zen5.sh).
#
# Main targets:
#   make results  -> build everything + run sweep + write metrics.csv
#   make profile_zen5  -> sweep only (binaries already built)
#   make profile_zen5_one VARIANT=... M=...
#   make consolidate_zen5
# =====================================================================

CFLAGS_O3_ZEN5  := $(CSTD) $(WARN) $(INCS) -O3 -march=native              \
                    -D_POSIX_C_SOURCE=200809L                              \
                    -DMORTON_AVX512_THRESHOLD_DEFAULT=1048576UL              \
                    -DMORTON_OMP_RECURSION_THRESHOLD_DEFAULT=1048576UL     \
                    -DMORTON_OMP_PARALLEL_THRESHOLD_DEFAULT=1048576UL      \
                    -DTILED_IKJ_MC_DEFAULT=384u                            \
                    -DTILED_IKJ_KC_DEFAULT=384u                            \
                    -DTILED_IKJ_AVX512_BS_DEFAULT=256u                       \
                    -DTILED_IKJ_AVX512_MC=288u                               \
                    -DTILED_IKJ_OMP_BS_DEFAULT=256u                        \
                    -DTILED_IKJ_OMP_MC=288u
CFLAGS_OMP_ZEN5 := $(CFLAGS_O3_ZEN5) -fopenmp

# Microkernel unit test (AVX-512 4x32). Compiled with the Zen 5 flag
# set so the immintrin intrinsics resolve.
TEST_KERNEL_AVX512_MORTON := $(BIN_DIR)/test_kernel_avx512_morton

.PHONY: test_kernel_avx512_morton

test_kernel_avx512_morton: $(TEST_KERNEL_AVX512_MORTON)
	./$(TEST_KERNEL_AVX512_MORTON)

$(TEST_KERNEL_AVX512_MORTON): $(TESTS_DIR)/test_kernel_avx512_morton.c \
                              $(MK_DIR)/kernel_avx512_morton.h \
                              $(CORE_DIR)/matrix_utils.c \
                              $(CORE_DIR)/matrix_utils.h | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) \
	      $(TESTS_DIR)/test_kernel_avx512_morton.c \
	      $(CORE_DIR)/matrix_utils.c \
	      -o $@ $(LIBS)

# ---- Source sets shared with the bench / validate targets below ----

MORTON_AVX512_SHARED_SRCS  := $(MORTON_DIR)/matmul_morton_avx512.c \
                            $(CORE_DIR)/morton.c \
                            $(CORE_DIR)/matrix_utils.c \
                            $(NAIVE_DIR)/matmul_naive.c

BENCH_MORTON_AVX512_SRCS    := $(MORTON_AVX512_SHARED_SRCS) \
                             $(BENCH_DIR)/bench_morton_avx512.c

VALIDATE_MORTON_AVX512_SRCS := $(MORTON_AVX512_SHARED_SRCS) \
                             $(MORTON_DIR)/matmul_morton.c \
                             $(VALIDATE_DIR)/validate_morton_avx512.c

MORTON_OMP_SHARED_SRCS    := $(MORTON_DIR)/matmul_morton_omp.c \
                             $(MORTON_DIR)/matmul_morton_avx512.c \
                             $(CORE_DIR)/morton.c \
                             $(CORE_DIR)/matrix_utils.c \
                             $(NAIVE_DIR)/matmul_naive.c

BENCH_MORTON_OMP_SRCS     := $(MORTON_OMP_SHARED_SRCS) \
                             $(BENCH_DIR)/bench_morton_omp.c

VALIDATE_MORTON_OMP_SRCS  := $(MORTON_OMP_SHARED_SRCS) \
                             $(MORTON_DIR)/matmul_morton.c \
                             $(VALIDATE_DIR)/validate_morton_omp.c

TILED_IKJ_AVX512_COMMON_SRCS   := $(COMMON_SRCS) \
                                $(TILED_DIR)/matmul_tiled_ikj_avx512.c
BENCH_TILED_IKJ_AVX512_SRCS    := $(TILED_IKJ_AVX512_COMMON_SRCS) \
                                $(BENCH_DIR)/bench_tiled_ikj_avx512.c
VALIDATE_TILED_IKJ_AVX512_SRCS := $(TILED_IKJ_AVX512_COMMON_SRCS) \
                                $(VALIDATE_DIR)/validate_tiled_ikj_avx512.c

TILED_IKJ_OMP_COMMON_SRCS    := $(COMMON_SRCS) \
                                $(TILED_DIR)/matmul_tiled_ikj_omp.c
BENCH_TILED_IKJ_OMP_SRCS     := $(TILED_IKJ_OMP_COMMON_SRCS) \
                                $(BENCH_DIR)/bench_tiled_ikj_omp.c
VALIDATE_TILED_IKJ_OMP_SRCS  := $(TILED_IKJ_OMP_COMMON_SRCS) \
                                $(VALIDATE_DIR)/validate_tiled_ikj_omp.c

# ---- Zen 5 bench binaries ----

BENCH_NAIVE_ZEN5           := $(BIN_DIR)/bench_naive_ZEN5
BENCH_MORTON_ZEN5          := $(BIN_DIR)/bench_morton_ZEN5
BENCH_MORTON_AVX512_ZEN5     := $(BIN_DIR)/bench_morton_avx512_ZEN5
BENCH_MORTON_OMP_ZEN5      := $(BIN_DIR)/bench_morton_omp_ZEN5
BENCH_LOOPS_ZEN5           := $(BIN_DIR)/bench_loops_ZEN5
BENCH_TILED_IKJ_ZEN5       := $(BIN_DIR)/bench_tiled_ikj_ZEN5
BENCH_TILED_IKJ_AVX512_ZEN5  := $(BIN_DIR)/bench_tiled_ikj_avx512_ZEN5
BENCH_TILED_IKJ_OMP_ZEN5   := $(BIN_DIR)/bench_tiled_ikj_omp_ZEN5

# Zen 5 validate binaries (cross-validation against matmul_naive).
VALIDATE_MORTON_AVX512_ZEN5     := $(BIN_DIR)/validate_morton_avx512_ZEN5
VALIDATE_MORTON_OMP_ZEN5      := $(BIN_DIR)/validate_morton_omp_ZEN5
VALIDATE_TILED_IKJ_AVX512_ZEN5  := $(BIN_DIR)/validate_tiled_ikj_avx512_ZEN5
VALIDATE_TILED_IKJ_OMP_ZEN5   := $(BIN_DIR)/validate_tiled_ikj_omp_ZEN5

ALL_ZEN5_BINS := $(BENCH_NAIVE_ZEN5) \
                 $(BENCH_MORTON_ZEN5) $(BENCH_MORTON_AVX512_ZEN5) \
                 $(BENCH_MORTON_OMP_ZEN5) \
                 $(BENCH_LOOPS_ZEN5) \
                 $(BENCH_TILED_IKJ_ZEN5) $(BENCH_TILED_IKJ_AVX512_ZEN5) \
                 $(BENCH_TILED_IKJ_OMP_ZEN5)

ALL_ZEN5_VALIDATE := $(VALIDATE_MORTON_AVX512_ZEN5) $(VALIDATE_MORTON_OMP_ZEN5) \
                     $(VALIDATE_TILED_IKJ_AVX512_ZEN5) \
                     $(VALIDATE_TILED_IKJ_OMP_ZEN5)

.PHONY: results profile_zen5 profile_zen5_one consolidate_zen5 \
        bench_naive_ZEN5 bench_morton_ZEN5 bench_morton_avx512_ZEN5 \
        bench_morton_omp_ZEN5 bench_loops_ZEN5 bench_tiled_ikj_ZEN5 \
        bench_tiled_ikj_avx512_ZEN5 bench_tiled_ikj_omp_ZEN5 \
        validate_morton_avx512_ZEN5 validate_morton_omp_ZEN5 \
        validate_tiled_ikj_avx512_ZEN5 validate_tiled_ikj_omp_ZEN5 \
        validate_all_zen5

bench_naive_ZEN5:          $(BENCH_NAIVE_ZEN5)
bench_morton_ZEN5:         $(BENCH_MORTON_ZEN5)
bench_morton_avx512_ZEN5:    $(BENCH_MORTON_AVX512_ZEN5)
bench_morton_omp_ZEN5:     $(BENCH_MORTON_OMP_ZEN5)
bench_loops_ZEN5:          $(BENCH_LOOPS_ZEN5)
bench_tiled_ikj_ZEN5:      $(BENCH_TILED_IKJ_ZEN5)
bench_tiled_ikj_avx512_ZEN5: $(BENCH_TILED_IKJ_AVX512_ZEN5)
bench_tiled_ikj_omp_ZEN5:  $(BENCH_TILED_IKJ_OMP_ZEN5)

validate_morton_avx512_ZEN5:    $(VALIDATE_MORTON_AVX512_ZEN5)
validate_morton_omp_ZEN5:     $(VALIDATE_MORTON_OMP_ZEN5)
validate_tiled_ikj_avx512_ZEN5: $(VALIDATE_TILED_IKJ_AVX512_ZEN5)
validate_tiled_ikj_omp_ZEN5:  $(VALIDATE_TILED_IKJ_OMP_ZEN5)
validate_all_zen5: $(ALL_ZEN5_VALIDATE)

# ---- Zen 5 bench compile recipes ----
#
# The AVX-512 microkernels are header-only `static inline`, so each
# .c file that includes them gets its own inlined copy. There is no
# kernel_*.o object to link.

$(BENCH_NAIVE_ZEN5): $(BENCH_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(BENCH_SRCS) -o $@ $(LIBS)

$(BENCH_MORTON_ZEN5): $(BENCH_MORTON_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(BENCH_MORTON_SRCS) -o $@ $(LIBS)

$(BENCH_MORTON_AVX512_ZEN5): $(BENCH_MORTON_AVX512_SRCS) \
                           $(MK_DIR)/kernel_avx512_morton.h | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(BENCH_MORTON_AVX512_SRCS) -o $@ $(LIBS)

$(BENCH_MORTON_OMP_ZEN5): $(BENCH_MORTON_OMP_SRCS) \
                          $(MK_DIR)/kernel_avx512_morton.h | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN5) $(BENCH_MORTON_OMP_SRCS) -o $@ $(LIBS)

$(BENCH_LOOPS_ZEN5): $(BENCH_LOOPS_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(BENCH_LOOPS_SRCS) -o $@ $(LIBS)

$(BENCH_TILED_IKJ_ZEN5): $(BENCH_TILED_IKJ_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(BENCH_TILED_IKJ_SRCS) -o $@ $(LIBS)

$(BENCH_TILED_IKJ_AVX512_ZEN5): $(BENCH_TILED_IKJ_AVX512_SRCS) \
                              $(MK_DIR)/kernel_avx512_tiled.h | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(BENCH_TILED_IKJ_AVX512_SRCS) -o $@ $(LIBS)

$(BENCH_TILED_IKJ_OMP_ZEN5): $(BENCH_TILED_IKJ_OMP_SRCS) \
                             $(MK_DIR)/kernel_avx512_tiled.h | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN5) $(BENCH_TILED_IKJ_OMP_SRCS) -o $@ $(LIBS)

# ---- Zen 5 validate compile recipes ----

$(VALIDATE_MORTON_AVX512_ZEN5): $(VALIDATE_MORTON_AVX512_SRCS) \
                              $(MK_DIR)/kernel_avx512_morton.h | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(VALIDATE_MORTON_AVX512_SRCS) -o $@ $(LIBS)

$(VALIDATE_MORTON_OMP_ZEN5): $(VALIDATE_MORTON_OMP_SRCS) \
                             $(MK_DIR)/kernel_avx512_morton.h | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN5) $(VALIDATE_MORTON_OMP_SRCS) -o $@ $(LIBS)

$(VALIDATE_TILED_IKJ_AVX512_ZEN5): $(VALIDATE_TILED_IKJ_AVX512_SRCS) \
                                 $(MK_DIR)/kernel_avx512_tiled.h | $(BIN_DIR)
	$(CC) $(CFLAGS_O3_ZEN5) $(VALIDATE_TILED_IKJ_AVX512_SRCS) -o $@ $(LIBS)

$(VALIDATE_TILED_IKJ_OMP_ZEN5): $(VALIDATE_TILED_IKJ_OMP_SRCS) \
                                $(MK_DIR)/kernel_avx512_tiled.h | $(BIN_DIR)
	$(CC) $(CFLAGS_OMP_ZEN5) $(VALIDATE_TILED_IKJ_OMP_SRCS) -o $@ $(LIBS)

# ---- Zen 5 perf-counter sweep ----

# Variables used by profile_zen5_one. Override via:
#   make profile_zen5_one VARIANT=morton_avx512 M=4096
VARIANT ?= morton_avx512
M       ?= 4096

results: $(ALL_ZEN5_BINS)
	$(if $(M),MS="$(M)" )bash scripts/run_perf_zen5_sweep.sh
	python3 scripts/consolidate_perf_zen5.py --out results/metrics.csv

profile_zen5: $(ALL_ZEN5_BINS)
	$(if $(M),MS="$(M)" )bash scripts/run_perf_zen5_sweep.sh
	python3 scripts/consolidate_perf_zen5.py --out results/metrics.csv

profile_zen5_one: $(ALL_ZEN5_BINS)
	bash scripts/profile_perf_zen5.sh $(VARIANT) $(M)

consolidate_zen5:
	python3 scripts/consolidate_perf_zen5.py --out results/metrics.csv
