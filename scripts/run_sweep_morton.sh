#!/usr/bin/env bash
#
# run_sweep_morton.sh - Run the Morton (Z-order) recursive matmul
# benchmark over a list of m values restricted to powers of two and
# collect the gflops results into results/morton_O0.csv.
#
# Phase 6, Stage A3 sweep. The Morton kernel asserts m is a power of two
# (see src/matmul_morton.c), so this script filters the input list and
# emits a stderr warning for any m that does not qualify.
#
# Usage:
#   scripts/run_sweep_morton.sh                       # default sweep
#   scripts/run_sweep_morton.sh "1024 2048 3000"      # 3000 is skipped
#
# Output: results/morton_O0.csv with header
#   m,n,num_iters,median_seconds,gflops
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BENCH_BIN="$REPO_DIR/bin/bench_morton_O0"
RESULTS_DIR="$REPO_DIR/results"
OUTPUT="$RESULTS_DIR/morton_O0.csv"

# Restricted default sweep: only powers of two cover the L1/L2/L3/DRAM
# transitions on the Ryzen 5 4600H and are the only sizes the Morton
# kernel accepts.
DEFAULT_M_LIST="1024 2048 4096 8192"
M_LIST="${1:-$DEFAULT_M_LIST}"

is_power_of_two() {
    local v="$1"
    if [ "$v" -le 0 ]; then
        return 1
    fi
    if [ $(( v & (v - 1) )) -eq 0 ]; then
        return 0
    fi
    return 1
}

if [ ! -x "$BENCH_BIN" ]; then
    echo "Error: $BENCH_BIN not found. Run 'make bench_morton' first." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
echo "kernel,m,n,num_iters,median_seconds,gflops" > "$OUTPUT"

echo "Sweep over m: $M_LIST (only powers of two will run)"
echo "Output CSV  : $OUTPUT"
echo

for m in $M_LIST; do
    if ! is_power_of_two "$m"; then
        echo "Warning: skipping m=$m (not a power of two; required by the Morton kernel)" >&2
        continue
    fi
    printf "  [m=%-6s] timing ... " "$m"
    line="$("$BENCH_BIN" "$m")"
    echo "$line" >> "$OUTPUT"
    gflops="$(echo "$line" | awk -F, '{print $6}')"
    secs="$(echo "$line"  | awk -F, '{print $5}')"
    printf "median=%ss  gflops=%s\n" "$secs" "$gflops"
done

echo
echo "Done. CSV: $OUTPUT"
