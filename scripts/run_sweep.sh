#!/usr/bin/env bash
#
# run_sweep.sh - Run the baseline benchmark over a range of m values,
# collect the gflops results into a CSV file, and (optionally) produce a
# per-m gprof and perf report.
#
# Step 3 of the project: evaluate performance as m grows and look for the
# transitions where the working set crosses each level of the cache
# hierarchy.
#
# Usage:
#   scripts/run_sweep.sh                       # default sweep, profiling ON
#   scripts/run_sweep.sh "256 512 1024"        # custom m values
#   PROFILING=0 scripts/run_sweep.sh           # skip gprof and perf
#   PROFILING=gprof scripts/run_sweep.sh       # only gprof per m
#   PROFILING=perf  scripts/run_sweep.sh       # only perf per m
#
# Environment variables:
#   PROFILING   = full | gprof | perf | 0        (default: full)
#   PROFILE_ITERS                                 (default: 1)
#       Iterations of the benchmark to run under perf/gprof. One is
#       usually enough; absolute event counts are what matters.
#   PROFILE_RUNS                                  (default: 1)
#       Measured runs under perf/gprof. Profiling does not need
#       statistical robustness; one deterministic run is the norm.
#

set -euo pipefail

# Resolve repo root from this script's location (works from any cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BENCH_BIN="$REPO_DIR/bin/bench_O0"
BENCH_PG="$REPO_DIR/bin/bench_pg"
RESULTS_DIR="$REPO_DIR/results"
OUTPUT="$RESULTS_DIR/baseline_O0.csv"

# Default sweep: powers of 2 and a few in-between points to expose the
# transitions in finer detail.
DEFAULT_M_LIST="256 384 512 768 1024 1536 2048 3072 4096 6144 8192"
M_LIST="${1:-$DEFAULT_M_LIST}"

PROFILING="${PROFILING:-full}"
PROFILE_ITERS="${PROFILE_ITERS:-1}"
PROFILE_RUNS="${PROFILE_RUNS:-1}"

if [ ! -x "$BENCH_BIN" ]; then
    echo "Error: $BENCH_BIN not found. Run 'make bench_O0' first." >&2
    exit 1
fi

# Decide which profilers to attempt.
WITH_GPROF=0
WITH_PERF=0
case "$PROFILING" in
    full)   WITH_GPROF=1; WITH_PERF=1 ;;
    gprof)  WITH_GPROF=1 ;;
    perf)   WITH_PERF=1 ;;
    0|off|no|none) ;;
    *) echo "Warning: unrecognized PROFILING='$PROFILING', defaulting to 'full'." >&2
       WITH_GPROF=1; WITH_PERF=1 ;;
esac

# Make sure prerequisites for each requested profiler exist; if not,
# downgrade gracefully so the sweep still runs.
if [ "$WITH_GPROF" -eq 1 ]; then
    if [ ! -x "$BENCH_PG" ]; then
        echo "Warning: $BENCH_PG missing - skipping gprof. Run 'make bench_pg' to enable."
        WITH_GPROF=0
    elif ! command -v gprof >/dev/null 2>&1; then
        echo "Warning: gprof not installed - skipping gprof per m."
        WITH_GPROF=0
    fi
fi
if [ "$WITH_PERF" -eq 1 ]; then
    if ! command -v perf >/dev/null 2>&1; then
        echo "Warning: perf not installed - skipping perf per m."
        WITH_PERF=0
    fi
fi

mkdir -p "$RESULTS_DIR"
echo "m,n,num_iters,median_seconds,gflops" > "$OUTPUT"

echo "Sweep over m: $M_LIST"
echo "Output CSV  : $OUTPUT"
echo "Profiling   : gprof=$WITH_GPROF perf=$WITH_PERF (iters=$PROFILE_ITERS runs=$PROFILE_RUNS)"
echo

for m in $M_LIST; do
    printf "  [m=%-6s]\n" "$m"

    # 1. Timed measurement that feeds the CSV (full statistical setup:
    #    warm-up + DEFAULT_RUNS measured runs inside the binary).
    printf "    timing       ... "
    line="$("$BENCH_BIN" "$m")"
    echo "$line" >> "$OUTPUT"
    gflops="$(echo "$line" | awk -F, '{print $5}')"
    secs="$(echo "$line"  | awk -F, '{print $4}')"
    printf "median=%ss  gflops=%s\n" "$secs" "$gflops"

    # 2. gprof report for this m (one deterministic run).
    if [ "$WITH_GPROF" -eq 1 ]; then
        printf "    gprof        ... "
        if bash "$SCRIPT_DIR/profile_gprof.sh" "$m" "$PROFILE_ITERS" "$PROFILE_RUNS" \
                > /dev/null 2> "$RESULTS_DIR/gprof_m${m}.err"; then
            rm -f "$RESULTS_DIR/gprof_m${m}.err"
            echo "results/gprof_m${m}.txt"
        else
            echo "FAILED (see results/gprof_m${m}.err)"
        fi
    fi

    # 3. perf report for this m (one deterministic run).
    if [ "$WITH_PERF" -eq 1 ]; then
        printf "    perf         ... "
        if bash "$SCRIPT_DIR/profile_perf.sh" "$m" "$PROFILE_ITERS" "$PROFILE_RUNS" \
                > /dev/null 2> "$RESULTS_DIR/perf_m${m}.err"; then
            rm -f "$RESULTS_DIR/perf_m${m}.err"
            echo "results/perf_m${m}.txt"
        else
            echo "FAILED (see results/perf_m${m}.err)"
        fi
    fi
done

echo
echo "Done."
echo "  CSV         : $OUTPUT"
[ "$WITH_GPROF" -eq 1 ] && echo "  gprof files : $RESULTS_DIR/gprof_m*.txt"
[ "$WITH_PERF"  -eq 1 ] && echo "  perf files  : $RESULTS_DIR/perf_m*.txt"
