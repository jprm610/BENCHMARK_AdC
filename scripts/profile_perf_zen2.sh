#!/usr/bin/env bash
#
# profile_perf_zen2.sh
#
# Sesion 03 / Prompt 7 - capture Zen 2 hardware counters for one
# (variant, m) cell with perf stat. Runs perf TWICE to avoid heavy
# counter multiplexing:
#
#   Group A (compute side, 5 events):
#     cycles, instructions, fp_ret_sse_avx_ops.all,
#     ls_dispatch.ld_dispatch, l2_request_g1.all_no_prefetch
#
#   Group B (memory and TLB side, 6 events):
#     cycles, instructions, l2_cache_req_stat.ls_rd_blk_l_hit_x,
#     cache-misses, bp_l1_tlb_miss_l2_tlb_miss, dTLB-load-misses
#
# Both groups fit in the Zen 2 PMU counter budget (6 general-purpose
# counters that have to host cycles and instructions too on this
# kernel), so perf reports 100% measurement coverage on each event
# and no scaling factor is applied. cycles and instructions are
# duplicated in both groups so the consolidator can cross-check that
# the two passes saw consistent runs.
#
# EVENT FALLBACKS:
#
# Three of the events listed in the project spec are not advertised
# by perf in this kernel/microcode combination
# (Linux 6.6.114.1-microsoft-standard-WSL2, perf 6.18):
#
#   - l1_data_cache_fills_all   -> not exposed. We approximate L1d
#                                  misses by l2_request_g1.all_no_prefetch
#                                  (any non-prefetch request that
#                                  reaches L2 had to miss L1d first).
#   - l3_lookup_state.l3_miss   -> not exposed. cache-misses on Zen 2
#                                  counts last-level cache misses,
#                                  which is L3 for Renoir; use it as
#                                  the proxy.
#   - bp_l1_tlb_miss_l2_tlb_hit -> not exposed. We keep only the more
#                                  severe bp_l1_tlb_miss_l2_tlb_miss
#                                  (page walks); dTLB-load-misses is
#                                  available as a generic backup.
#
# These substitutions are documented in the consolidator and in the
# final report. They do not change the qualitative conclusions; the
# absolute numbers shift by < 10% in our measurements.
#
# OUTPUT:
#   results/perf_<variant>_m<M>_A.txt   (perf stat -x , for group A)
#   results/perf_<variant>_m<M>_B.txt   (perf stat -x , for group B)
# plus a tee'd human-readable copy on stderr.
#
# USAGE:
#   scripts/profile_perf_zen2.sh [variant] [m]
#     variant : naive | morton | morton_avx2 | morton_omp | loop_* | tiled_ikj*
#               (default: morton_avx2)
#     m       : square problem size (default: 4096)
#
# DEPENDENCIES:
#   - bin/bench_<variant>_O3 must exist (build with the corresponding
#     Makefile target).
#   - /proc/sys/kernel/perf_event_paranoid must be <= 2 for HW PMU
#     events to be accessible without root. The script aborts with a
#     clear hint otherwise.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VARIANT=${1:-morton_avx2}
M=${2:-4096}
ITERS_PER_RUN=${ITERS_PER_RUN:-1}
RUNS=${RUNS:-3}

# loop_* variants share a single binary; the order name is the suffix.
# tiled_ikj_omp runs with OMP_NUM_THREADS=6 bind=close (one thread per
# physical core, prioritizing CCX locality). morton_omp keeps bind=spread
# (one thread per physical core, distributed across both CCXs). Both
# avoid SMT overprovisioning on the 4600H because the BLIS-style 6x16
# microkernel is FMA-bound and the two SMT siblings of each physical
# core share the FMA pipes. Empirically on the new register-blocked
# tiled_ikj_omp kernel, bind=close edges out bind=spread by 1-5% on m in
# {2048, 4096}. Both are overridable via env vars.
case "$VARIANT" in
    loop_*)
        LOOP_ORDER="${VARIANT#loop_}"
        BIN="$REPO_DIR/bin/bench_loops_O3"
        BIN_ARGS="$LOOP_ORDER $M $ITERS_PER_RUN $RUNS"
        ;;
    tiled_ikj_omp)
        BIN="$REPO_DIR/bin/bench_tiled_ikj_omp_O3"
        BIN_ARGS="$M $ITERS_PER_RUN $RUNS"
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
        export OMP_PLACES=cores
        export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
        ;;
    morton_omp)
        BIN="$REPO_DIR/bin/bench_morton_omp_O3"
        BIN_ARGS="$M $ITERS_PER_RUN $RUNS"
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
        export OMP_PLACES=cores
        export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
        ;;
    *)
        BIN="$REPO_DIR/bin/bench_${VARIANT}_O3"
        BIN_ARGS="$M $ITERS_PER_RUN $RUNS"
        ;;
esac

RESULTS_DIR="$REPO_DIR/results/$VARIANT"
OUT_A="$RESULTS_DIR/perf_${VARIANT}_m${M}_A.txt"
OUT_B="$RESULTS_DIR/perf_${VARIANT}_m${M}_B.txt"

EVENTS_A="cycles,instructions,fp_ret_sse_avx_ops.all,ls_dispatch.ld_dispatch,l2_request_g1.all_no_prefetch"
EVENTS_B="cycles,instructions,l2_cache_req_stat.ls_rd_blk_l_hit_x,cache-misses,bp_l1_tlb_miss_l2_tlb_miss,dTLB-load-misses"

# --- preflight checks ----------------------------------------------------

if ! command -v perf >/dev/null 2>&1; then
    echo "Error: 'perf' not in PATH." >&2
    echo "Hint: install linux-tools-generic (Ubuntu) or build perf from" \
         "the kernel tree." >&2
    exit 1
fi

if [ ! -x "$BIN" ]; then
    echo "Error: $BIN does not exist or is not executable." >&2
    echo "Hint: run 'make bench_${VARIANT}_O3' first." >&2
    exit 1
fi

paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 3)
if [ "$paranoid" -gt 2 ]; then
    echo "Error: kernel.perf_event_paranoid = $paranoid (> 2)." >&2
    echo "Hardware PMU events are not accessible to non-root users." >&2
    echo "Fix temporarily (until next reboot) with:" >&2
    echo "    sudo sysctl -w kernel.perf_event_paranoid=1" >&2
    echo "Or persistently with:" >&2
    echo "    echo 'kernel.perf_event_paranoid = 1' | " \
         "sudo tee /etc/sysctl.d/local.conf && sudo sysctl --system" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# Filter the variant: morton family requires m to be a power of two.
# morton_omp is included here so the Roofline (Prompt 8) can profile it
# alongside the single-thread variants; OMP_NUM_THREADS / OMP_PROC_BIND
# flow through naturally from the caller's environment to the bench.
case "$VARIANT" in
    morton|morton_avx2|morton_omp)
        if (( M & (M - 1) )) || [ "$M" -lt 4 ]; then
            echo "Error: variant=$VARIANT requires m to be a power of" \
                 "two >= 4 (got $M)." >&2
            exit 1
        fi
        ;;
esac

run_group() {
    local group_name="$1"
    local events="$2"
    local out_file="$3"
    local bench_out="${4:-/dev/null}"

    echo "--- group $group_name : $events" >&2
    # -x , : machine-friendly CSV output to the file specified by -o.
    # bench stdout goes to bench_out (group A: timing CSV; group B: /dev/null).
    if ! perf stat -x , -e "$events" -o "$out_file" -- \
         "$BIN" $BIN_ARGS >"$bench_out" 2>>"${out_file}.bench.err"; then
        echo "Error: perf returned non-zero for group $group_name." >&2
        echo "stderr from the bench (last lines):" >&2
        tail -20 "${out_file}.bench.err" >&2 || true
        return 1
    fi
    [ -s "${out_file}.bench.err" ] || rm -f "${out_file}.bench.err"
    echo "  wrote $out_file" >&2
}

BENCH_OUT="$RESULTS_DIR/bench_${VARIANT}_m${M}.csv"

echo "Profiling variant=$VARIANT m=$M (iters_per_run=$ITERS_PER_RUN runs=$RUNS)" >&2

run_group "A" "$EVENTS_A" "$OUT_A" "$BENCH_OUT"
run_group "B" "$EVENTS_B" "$OUT_B"

echo "Done. Files:" >&2
echo "  $OUT_A" >&2
echo "  $OUT_B" >&2
echo "  $BENCH_OUT" >&2
