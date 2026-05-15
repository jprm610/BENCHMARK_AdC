#!/usr/bin/env bash
# run_sweep_loop.sh - Measure all six loop-order kernels across a range of m.
#
# Usage:
#   bash scripts/run_sweep_loop.sh [m_list]
#
#   m_list : space-separated list of m values (default below).
#
# Output: results/loop_order.csv
#   Columns: kernel,m,n,num_iters,median_seconds,gflops
#
# The script runs bench_loop_O0 for each (order, m) pair and appends the
# CSV line.  A single header line is written at the start.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH="${REPO_ROOT}/bin/bench_loop_O0"
RESULTS_DIR="${REPO_ROOT}/results"
OUT_CSV="${RESULTS_DIR}/loop_order.csv"

if [[ ! -x "${BENCH}" ]]; then
    echo "Error: ${BENCH} not found. Run 'make bench_loop' first." >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

# Default m values: representative points across L1/L2/L3/DRAM regimes.
DEFAULT_M="512 1024 2048 4096"
M_LIST="${1:-${DEFAULT_M}}"

ORDERS="ijk ikj jik jki kij kji"

# Write header (overwrite any previous run).
echo "kernel,m,n,num_iters,median_seconds,gflops" > "${OUT_CSV}"

for order in ${ORDERS}; do
    for m in ${M_LIST}; do
        echo -n "  ${order}  m=${m} ... " >&2
        line=$("${BENCH}" "${order}" "${m}")
        echo "${line}" >> "${OUT_CSV}"
        gflops=$(echo "${line}" | cut -d',' -f6)
        echo "${gflops} GFLOPS" >&2
    done
done

echo "" >&2
echo "Results written to ${OUT_CSV}" >&2
