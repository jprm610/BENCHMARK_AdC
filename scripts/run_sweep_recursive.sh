#!/usr/bin/env bash
#
# run_sweep_recursive.sh - Run the cache-oblivious recursive matmul
# benchmark over a range of m values and collect the gflops results into
# results/recursive_O0.csv.
#
# Phase 6, Stage A2 sweep. No profiling is integrated here; perf
# comparison across kernels lives in profile_perf_compare.sh.
#
# Usage:
#   scripts/run_sweep_recursive.sh                    # default sweep
#   scripts/run_sweep_recursive.sh "256 512 1024"     # custom m values
#
# Output: results/recursive_O0.csv with header
#   m,n,num_iters,median_seconds,gflops
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BENCH_BIN="$REPO_DIR/bin/bench_recursive_O0"
RESULTS_DIR="$REPO_DIR/results"
OUTPUT="$RESULTS_DIR/recursive_O0.csv"

# Same default sweep as the naive baseline so the two CSVs share the
# same x-axis points in plot_comparison.py.
DEFAULT_M_LIST="256 384 512 768 1024 1536 2048 3072 4096 6144 8192"
M_LIST="${1:-$DEFAULT_M_LIST}"

if [ ! -x "$BENCH_BIN" ]; then
    echo "Error: $BENCH_BIN not found. Run 'make bench_recursive' first." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
echo "m,n,num_iters,median_seconds,gflops" > "$OUTPUT"

echo "Sweep over m: $M_LIST"
echo "Output CSV  : $OUTPUT"
echo

for m in $M_LIST; do
    printf "  [m=%-6s] timing ... " "$m"
    line="$("$BENCH_BIN" "$m")"
    echo "$line" >> "$OUTPUT"
    gflops="$(echo "$line" | awk -F, '{print $5}')"
    secs="$(echo "$line"  | awk -F, '{print $4}')"
    printf "median=%ss  gflops=%s\n" "$secs" "$gflops"
done

echo
echo "Done. CSV: $OUTPUT"
