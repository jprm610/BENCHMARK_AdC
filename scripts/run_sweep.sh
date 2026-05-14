#!/usr/bin/env bash
#
# run_sweep.sh - Run the baseline benchmark over a range of m values and
# collect the results into a CSV file.
#
# Step 3 of the project: evaluate performance as m grows and look for the
# transitions where the working set crosses each level of the cache
# hierarchy.
#
# Usage:
#   scripts/run_sweep.sh                  # default sweep, 256..8192
#   scripts/run_sweep.sh "256 512 1024"   # custom m values
#

set -euo pipefail

# Resolve repo root from this script's location (works from any cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BENCH_BIN="$REPO_DIR/bin/bench_O0"
RESULTS_DIR="$REPO_DIR/results"
OUTPUT="$RESULTS_DIR/baseline_O0.csv"

# Default sweep: powers of 2 and a few in-between points to expose the
# transitions in finer detail.
DEFAULT_M_LIST="256 384 512 768 1024 1536 2048 3072 4096 6144 8192"
M_LIST="${1:-$DEFAULT_M_LIST}"

if [ ! -x "$BENCH_BIN" ]; then
    echo "Error: binary $BENCH_BIN not found. Run 'make bench_O0' first." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
echo "m,n,num_iters,median_seconds,gflops" > "$OUTPUT"

echo "Sweep over m: $M_LIST"
echo "Output CSV: $OUTPUT"
echo

for m in $M_LIST; do
    printf "  m=%-6s ... " "$m"
    line="$("$BENCH_BIN" "$m")"
    echo "$line" >> "$OUTPUT"
    # Show the gflops value so the user has feedback during long runs.
    gflops="$(echo "$line" | awk -F, '{print $5}')"
    secs="$(echo "$line" | awk -F, '{print $4}')"
    printf "median=%ss  gflops=%s\n" "$secs" "$gflops"
done

echo
echo "Done. CSV written to $OUTPUT"
