#!/usr/bin/env bash
#
# profile_perf.sh - Run the baseline binary under 'perf stat' to capture
# hardware counters required by step 2 of the project:
#
#   * total instructions
#   * average IPC (instructions per cycle)
#   * page faults (major + minor)
#   * branch effectiveness (branches + branch-misses + miss rate)
#   * cache references / misses (as bonus context)
#
# Usage:
#   scripts/profile_perf.sh                  # default m=2048, iters=1, runs=1
#   scripts/profile_perf.sh 1024             # custom m
#   scripts/profile_perf.sh 1024 2           # custom m and iters
#   scripts/profile_perf.sh 1024 2 3         # full control: m, iters, runs
#
# Notes for WSL2:
#   - perf must be installed (sudo apt install linux-tools-generic or
#     linux-tools-$(uname -r) on a real Linux). On WSL2 you may need to
#     compile perf from the WSL2-Linux-Kernel repo; see README section 3.3.
#   - WSL2 currently exposes a limited set of hardware PMCs. If a counter
#     is unsupported it will print "<not supported>"; the script still
#     produces output for the ones that work.
#   - kernel.perf_event_paranoid may need to be lowered (see README).
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_BIN="$REPO_DIR/bin/bench_O0"
RESULTS_DIR="$REPO_DIR/results"

M="${1:-2048}"
ITERS="${2:-1}"
RUNS="${3:-1}"
OUT_TXT="$RESULTS_DIR/perf_m${M}.txt"

if [ ! -x "$BENCH_BIN" ]; then
    echo "Error: $BENCH_BIN not found. Run 'make bench_O0' first." >&2
    exit 1
fi

if ! command -v perf >/dev/null 2>&1; then
    echo "Error: perf is not installed." >&2
    echo "  On Ubuntu/WSL2: see README section 3.3 for installation steps." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# Curated event list: covers the four bullet points of step 2 and adds
# cache-miss counters because they are essential to interpret the cliffs.
EVENTS="task-clock,cycles,instructions,\
branches,branch-misses,\
cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,\
page-faults,minor-faults,major-faults"

echo "perf stat with m=$M iters=$ITERS runs=$RUNS"
echo "events: $EVENTS"
echo

# 'perf stat -d' would add extra cache events automatically; we keep the
# explicit list for repeatability.
perf stat -e "$EVENTS" -- "$BENCH_BIN" "$M" "$ITERS" "$RUNS" 2> "$OUT_TXT" >/dev/null || true

echo "Perf report saved to: $OUT_TXT"
echo "--------------------------------"
cat "$OUT_TXT"
