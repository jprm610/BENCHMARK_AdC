#!/usr/bin/env bash
#
# profile_gprof.sh - Run the -pg instrumented binary, then process gmon.out
# with gprof to produce a flat and call-graph profile.
#
# Step 2 of the project: gprof gives a function-level time breakdown.
# Note that gprof reports wall time apportioned by sampling and call
# counts; for instructions/IPC/page-faults/branch effectiveness we use
# the perf script instead.
#
# Usage:
#   scripts/profile_gprof.sh           # default m=2048, iters=2
#   scripts/profile_gprof.sh 1024      # custom m
#   scripts/profile_gprof.sh 1024 4    # custom m and iters
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_PG="$REPO_DIR/bin/bench_pg"
RESULTS_DIR="$REPO_DIR/results"

M="${1:-2048}"
ITERS="${2:-2}"
OUT_TXT="$RESULTS_DIR/gprof_m${M}.txt"

if [ ! -x "$BENCH_PG" ]; then
    echo "Error: $BENCH_PG not found. Run 'make bench_pg' first." >&2
    exit 1
fi

if ! command -v gprof >/dev/null 2>&1; then
    echo "Error: gprof is not installed. On Debian/Ubuntu: sudo apt install binutils" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# gprof writes gmon.out in the *current* working directory of the process.
# Run from REPO_DIR so the file lands somewhere predictable, and clean
# any previous one to avoid mixing profiles.
cd "$REPO_DIR"
rm -f gmon.out

echo "Running $BENCH_PG with m=$M iters=$ITERS (this may take a while at -O0)..."
"$BENCH_PG" "$M" "$ITERS" > /dev/null

if [ ! -f gmon.out ]; then
    echo "Error: gmon.out was not produced. Did you compile with -pg?" >&2
    exit 1
fi

echo "Generating flat profile..."
gprof "$BENCH_PG" gmon.out > "$OUT_TXT"

# Keep gmon.out around so it can be re-analyzed with different flags.
echo
echo "Gprof report saved to: $OUT_TXT"
echo "Top of the flat profile:"
echo "------------------------"
head -n 40 "$OUT_TXT"
