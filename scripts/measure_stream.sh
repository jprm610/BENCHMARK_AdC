#!/usr/bin/env bash
#
# measure_stream.sh
#
# Sesion 03 / Prompt 8 - measure sustained DRAM bandwidth of the test
# machine with the McCalpin STREAM benchmark. The Triad bandwidth is
# the realistic ceiling that anchors the diagonal of the Roofline
# diagram (results/perf_zen2_summary.csv supplies the y-axis points,
# this script supplies the slope).
#
# STREAM_ARRAY_SIZE is set so the three arrays (a, b, c, each of
# STREAM_ARRAY_SIZE doubles) total 1.5 GiB, which is ~ 400x the
# Ryzen 5 4600H L3 (4 MiB per CCX). At this size every iteration of
# the kernel must round-trip to DRAM; cache cannot hide any of it.
#
# Single-thread Triad on Zen 2 with DDR4-3200 dual-channel typically
# lands in 8-15 GB/s. 6-thread Triad with spread (one thread per
# physical core) saturates the two memory controllers at roughly
# 25-40 GB/s. The Ryzen 5 4600H peak DDR4-3200 dual-channel is 51.2
# GB/s in theory.
#
# Output:
#   results/stream_1t.txt   - single-thread run
#   results/stream_6t.txt   - 6-thread run with OMP_PROC_BIND=spread
#
# Source of STREAM:
#   https://www.cs.virginia.edu/stream/FTP/Code/stream.c
# downloaded once to build/stream/stream.c and cached there.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="$REPO_DIR/build/stream"
RESULTS_DIR="$REPO_DIR/results"

STREAM_URL="https://www.cs.virginia.edu/stream/FTP/Code/stream.c"
STREAM_SIZE=${STREAM_SIZE:-64000000}   # 64M doubles per array -> ~1.5 GiB total
STREAM_NTIMES=${STREAM_NTIMES:-20}     # default in STREAM, 20 iters
THREADS_MULTI=${THREADS_MULTI:-6}      # physical cores on 4600H

mkdir -p "$BUILD_DIR" "$RESULTS_DIR"

# --- fetch ---------------------------------------------------------------

if [ ! -f "$BUILD_DIR/stream.c" ]; then
    echo "Downloading STREAM source to $BUILD_DIR/stream.c ..."
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$STREAM_URL" -o "$BUILD_DIR/stream.c"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$STREAM_URL" -O "$BUILD_DIR/stream.c"
    else
        echo "Error: neither curl nor wget available; cannot fetch STREAM." >&2
        echo "Manual fix: download $STREAM_URL into $BUILD_DIR/stream.c" >&2
        exit 1
    fi
fi

# --- build ---------------------------------------------------------------

# -DSTREAM_ARRAY_SIZE pins the per-array size at compile time. -fopenmp
# enables the parallel loops STREAM uses internally. -O3 -mavx2 -mfma
# match the rest of the project's measurement regime.
echo "Compiling STREAM (size=$STREAM_SIZE doubles, ntimes=$STREAM_NTIMES)..."
gcc -O3 -fopenmp -march=znver2 -mavx2 -mfma \
    -DSTREAM_ARRAY_SIZE="$STREAM_SIZE" \
    -DNTIMES="$STREAM_NTIMES" \
    "$BUILD_DIR/stream.c" -o "$BUILD_DIR/stream"

# --- run -----------------------------------------------------------------

echo
echo "Running STREAM single-thread ..."
OMP_NUM_THREADS=1 "$BUILD_DIR/stream" > "$RESULTS_DIR/stream_1t.txt"

echo "Running STREAM $THREADS_MULTI-thread (OMP_PROC_BIND=spread) ..."
OMP_NUM_THREADS="$THREADS_MULTI" OMP_PLACES=cores OMP_PROC_BIND=spread \
    "$BUILD_DIR/stream" > "$RESULTS_DIR/stream_${THREADS_MULTI}t.txt"

# --- summary -------------------------------------------------------------

echo
echo "==== Single thread ===="
grep -E "^(Copy|Scale|Add|Triad):" "$RESULTS_DIR/stream_1t.txt" || true
echo
echo "==== $THREADS_MULTI threads ===="
grep -E "^(Copy|Scale|Add|Triad):" "$RESULTS_DIR/stream_${THREADS_MULTI}t.txt" || true

echo
echo "Files:"
echo "  $RESULTS_DIR/stream_1t.txt"
echo "  $RESULTS_DIR/stream_${THREADS_MULTI}t.txt"
echo "Next: python3 scripts/plot_roofline.py"
