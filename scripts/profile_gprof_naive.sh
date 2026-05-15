#!/usr/bin/env bash
#
# profile_gprof_naive.sh - Run the -pg instrumented naive binary, then
# process gmon.out with gprof to produce a flat and call-graph profile.
#
# Step 2 of the project: gprof gives a function-level time breakdown.
# Note that gprof reports wall time apportioned by sampling and call
# counts; for instructions/IPC/page-faults/branch effectiveness we use
# the perf script instead.
#
# Usage:
#   scripts/profile_gprof_naive.sh                    # default m=2048, iters=1, runs=1
#   scripts/profile_gprof_naive.sh 1024               # custom m
#   scripts/profile_gprof_naive.sh 1024 2             # custom m and iters
#   scripts/profile_gprof_naive.sh 1024 2 3           # full control: m, iters, runs
#
# For sweeps, the typical invocation is `<m> 1 1` - one iteration of the
# benchmark, one measured run. That keeps profiling cost bounded; the
# absolute event counts (calls, samples) are what matter, not statistics.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_PG="$REPO_DIR/bin/bench_naive_pg"
RESULTS_DIR="$REPO_DIR/results"

M="${1:-2048}"
ITERS="${2:-1}"
RUNS="${3:-1}"
OUT_TXT="$RESULTS_DIR/gprof_naive_m${M}.txt"

if [ ! -x "$BENCH_PG" ]; then
    echo "Error: $BENCH_PG not found. Run 'make bench_naive_pg' first." >&2
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

echo "Running $BENCH_PG with m=$M iters=$ITERS runs=$RUNS (this may take a while at -O0)..."
"$BENCH_PG" "$M" "$ITERS" "$RUNS" > /dev/null

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
