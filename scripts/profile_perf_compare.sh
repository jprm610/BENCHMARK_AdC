#!/usr/bin/env bash
#
# profile_perf_compare.sh - Collect hardware-event counts for the three
# matmul kernels (naive, recursive, morton) at several problem sizes and
# emit a single CSV that plot_perf_compare.py can consume.
#
# Phase 6, Prompt 7. Goal: if Morton gives a speedup, the event counts
# must justify it as fewer cache and TLB misses, as theory predicts. If
# Morton does not speed up, the events must confirm that the hardware
# prefetcher is masking the locality difference.
#
# Events captured (one perf invocation per (m, variant)):
#   L1-dcache-loads, L1-dcache-load-misses,
#   LLC-loads, LLC-load-misses,
#   dTLB-load-misses,
#   cycles, instructions
#
# Usage:
#   bash scripts/profile_perf_compare.sh                       # default m list
#   bash scripts/profile_perf_compare.sh "1024 2048 4096"      # custom list
#
# Output: results/perf_compare.csv with header
#   m,variant,l1_loads,l1_misses,llc_loads,llc_misses,dtlb_misses,cycles,instructions
#
# perf-event-paranoid handling: WSL2 ships with /proc/sys/kernel/perf_event_paranoid
# typically at 4, which blocks every counter. If that is the case, this
# script aborts early with the exact fix and a pointer to README section 3.3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BIN_DIR="$REPO_DIR/bin"
RESULTS_DIR="$REPO_DIR/results"
OUTPUT="$RESULTS_DIR/perf_compare.csv"

NAIVE_BIN="$BIN_DIR/bench_naive_O0"
REC_BIN="$BIN_DIR/bench_recursive_O0"
MOR_BIN="$BIN_DIR/bench_morton_O0"

DEFAULT_M_LIST="1024 2048 4096 8192"
M_LIST="${1:-$DEFAULT_M_LIST}"

# perf stat -e takes a comma-separated event list.
EVENTS="L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-load-misses,cycles,instructions"

# ----- environment checks ---------------------------------------------

if ! command -v perf >/dev/null 2>&1; then
    echo "Error: 'perf' not installed." >&2
    echo "On WSL2 you need to build perf from the WSL2-Linux-Kernel sources" >&2
    echo "(see README section 3.3 of Session 01)." >&2
    exit 1
fi

# Smoke-test perf permissions. Failure here usually means
# perf_event_paranoid is set too high; bail out with the fix.
if ! perf stat -e cycles true >/dev/null 2>&1; then
    echo "Error: perf cannot read counters." >&2
    echo "Most likely cause: /proc/sys/kernel/perf_event_paranoid is too restrictive." >&2
    echo "Fix:" >&2
    echo "    sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'" >&2
    echo "(applies until the next reboot; see README section 3.3 of Session 01)." >&2
    exit 1
fi

for b in "$NAIVE_BIN" "$REC_BIN" "$MOR_BIN"; do
    if [ ! -x "$b" ]; then
        echo "Error: $b not built." >&2
        echo "Run 'make bench_naive_O0 bench_recursive bench_morton' first." >&2
        exit 1
    fi
done

is_power_of_two() {
    local v="$1"
    [ "$v" -gt 0 ] && [ $(( v & (v - 1) )) -eq 0 ]
}

# perf with -x, may emit "<not supported>" or "<not counted>" instead of
# a number on systems where the event is unavailable. Normalize to 0 so
# the CSV stays parseable; the plot script reports them as such.
sanitize() {
    local v="$1"
    if [[ "$v" =~ ^[0-9]+$ ]]; then
        echo "$v"
    else
        echo "0"
    fi
}

# Extract a single event count from a perf -x, output file.
# Args: file, event_name
extract_count() {
    local file="$1" ev="$2"
    awk -F, -v ev="$ev" '$3==ev {print $1; exit}' "$file"
}

# Run one (m, variant, bin) triple under perf and append a CSV line.
run_one() {
    local m="$1" variant="$2" bin="$3"
    local tmp
    tmp="$(mktemp)"

    # perf stat -x,  -> CSV-like to stderr (one line per event).
    # The benchmark itself prints one CSV line to stdout; discard it so
    # only the perf output lands in $tmp.
    if ! perf stat -x, -e "$EVENTS" "$bin" "$m" 1 1 >/dev/null 2> "$tmp"; then
        echo "  WARNING: perf stat failed for variant=$variant m=$m" >&2
        rm -f "$tmp"
        return 1
    fi

    local L1L L1M LLCL LLCM DTLBM CYC INS
    L1L=$(sanitize  "$(extract_count "$tmp" L1-dcache-loads)")
    L1M=$(sanitize  "$(extract_count "$tmp" L1-dcache-load-misses)")
    LLCL=$(sanitize "$(extract_count "$tmp" LLC-loads)")
    LLCM=$(sanitize "$(extract_count "$tmp" LLC-load-misses)")
    DTLBM=$(sanitize "$(extract_count "$tmp" dTLB-load-misses)")
    CYC=$(sanitize  "$(extract_count "$tmp" cycles)")
    INS=$(sanitize  "$(extract_count "$tmp" instructions)")

    echo "$m,$variant,$L1L,$L1M,$LLCL,$LLCM,$DTLBM,$CYC,$INS" >> "$OUTPUT"
    rm -f "$tmp"
    return 0
}

# ----- main loop ------------------------------------------------------

mkdir -p "$RESULTS_DIR"
echo "m,variant,l1_loads,l1_misses,llc_loads,llc_misses,dtlb_misses,cycles,instructions" > "$OUTPUT"

echo "perf comparison sweep"
echo "  events : $EVENTS"
echo "  m list : $M_LIST"
echo "  output : $OUTPUT"
echo

for m in $M_LIST; do
    echo "[m=$m]"

    printf "  naive     ... "
    if run_one "$m" naive "$NAIVE_BIN"; then echo "done"; else echo "skipped"; fi

    printf "  recursive ... "
    if run_one "$m" recursive "$REC_BIN"; then echo "done"; else echo "skipped"; fi

    if is_power_of_two "$m"; then
        printf "  morton    ... "
        if run_one "$m" morton "$MOR_BIN"; then echo "done"; else echo "skipped"; fi
    else
        echo "  morton    ... skipped (m=$m not a power of two)"
    fi
done

echo
echo "Done. CSV: $OUTPUT"
