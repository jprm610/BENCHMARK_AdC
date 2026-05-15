#!/usr/bin/env bash
#
# run_threshold_sweep.sh
#
# Sesion 03 / Prompt 2 - empirical tuning of g_recursion_threshold for
# the Morton kernel on the test machine (Ryzen 5 4600H, Renoir / Zen 2).
#
# Iterates over a logarithmic ladder of thresholds (in number of element
# products, NOT bytes) and a small set of representative problem sizes,
# running bin/bench_morton_O3 for each combination. The bench binary
# already does warm-up + median over 5 runs internally, so a single
# invocation per (threshold, m) is enough.
#
# The grid is:
#   thresholds = {2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M}
#   m          = {1024, 2048, 4096}
#   -> 10 x 3 = 30 rows.
#
# Output:
#   results/threshold_sweep.csv with columns:
#     threshold,m,gflops_median
#
# Reads from each bench invocation the last line on stdout (CSV row),
# pulls field 5 (gflops), and writes the {threshold, m, gflops} triple.

set -euo pipefail

BIN=${BENCH_BIN:-./bin/bench_morton_O3}
CSV=${CSV_OUT:-results/threshold_sweep.csv}

THRESHOLDS=(2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)
MS=(1024 2048 4096)

# Number of measured iterations per bench invocation. The Morton kernel
# at -O3 still spends meaningful time per iteration at m=4096, so we
# keep it small to bound the total sweep time. The bench averages 5
# runs internally regardless.
NUM_ITERS=${NUM_ITERS:-1}
# 3 runs are enough for a reliable median when each run is already long
# enough to dominate timer noise. Larger values bloat the sweep without
# changing the qualitative shape. Override with NUM_RUNS=5 if desired.
NUM_RUNS=${NUM_RUNS:-3}

if [ ! -x "$BIN" ]; then
    echo "Error: $BIN is missing or not executable. Run 'make bench_morton_O3' first." >&2
    exit 1
fi

mkdir -p "$(dirname "$CSV")"

# Write into a tmp file first; only rename to the final path when the
# loop completes. This way a Ctrl-C or an OOM-killer in the middle of
# the sweep leaves the previous, complete CSV alone instead of leaving
# a half-written file behind. If a previous CSV exists, also keep a
# .bak copy as a safety net.
TMP_CSV="${CSV}.tmp"
if [ -f "$CSV" ]; then
    cp "$CSV" "${CSV}.bak"
fi
trap 'rm -f "$TMP_CSV"' EXIT

echo "threshold,m,gflops_median" > "$TMP_CSV"

total=$(( ${#THRESHOLDS[@]} * ${#MS[@]} ))
count=0

for threshold in "${THRESHOLDS[@]}"; do
    for m in "${MS[@]}"; do
        count=$(( count + 1 ))
        printf '[%2d/%d] threshold=%-8d m=%-6d ... ' \
               "$count" "$total" "$threshold" "$m" >&2

        # The bench prints exactly one CSV line:
        #   m,n,num_iters,median_seconds,gflops
        line=$("$BIN" "$m" "$NUM_ITERS" "$NUM_RUNS" --threshold "$threshold")
        gflops=$(echo "$line" | awk -F, '{print $5}')

        if [ -z "$gflops" ]; then
            echo "FAILED (empty gflops from bench)" >&2
            echo "$threshold,$m,nan" >> "$TMP_CSV"
            continue
        fi

        printf 'gflops=%s\n' "$gflops" >&2
        echo "$threshold,$m,$gflops" >> "$TMP_CSV"
    done
done

mv "$TMP_CSV" "$CSV"
trap - EXIT

n_data_rows=$(( $(wc -l < "$CSV") - 1 ))
echo >&2
echo "Sweep complete. $n_data_rows data rows written to $CSV." >&2
echo "Expected: $total." >&2
echo "Run: python3 scripts/plot_threshold_sweep.py" >&2
