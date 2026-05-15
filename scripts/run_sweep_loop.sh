#!/usr/bin/env bash
# run_sweep_loop.sh - Measure loop-order kernels across a range of m.
#
# Usage:
#   bash scripts/run_sweep_loop.sh                      # all 6 orders, default m list
#   bash scripts/run_sweep_loop.sh <order>              # single order, default m list
#   bash scripts/run_sweep_loop.sh <order> "<m list>"   # single order, custom m list
#
#   <order>  : one of ijk ikj jik jki kij kji  (omit for all six)
#   <m list> : space-separated sizes, e.g. "512 1024 2048"
#
# Output:
#   results/loop_<order>.csv  -- one file per order run
#   results/loop_order.csv    -- all orders combined (only when running all)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH="${REPO_ROOT}/bin/bench_loop_O0"
RESULTS_DIR="${REPO_ROOT}/results"
COMBINED_CSV="${RESULTS_DIR}/loop_order.csv"

if [[ ! -x "${BENCH}" ]]; then
    echo "Error: ${BENCH} not found. Run 'make bench_loop' first." >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

ALL_ORDERS="ijk ikj jik jki kij kji"
DEFAULT_M="256 384 512 768 1024 1536 2048 3072 4096"
HEADER="kernel,m,n,num_iters,median_seconds,gflops"

# Parse arguments: optional order, optional m list.
if [[ $# -eq 0 ]]; then
    ORDERS="${ALL_ORDERS}"
    M_LIST="${DEFAULT_M}"
    RUN_ALL=1
elif [[ $# -eq 1 ]]; then
    ORDERS="$1"
    M_LIST="${DEFAULT_M}"
    RUN_ALL=0
else
    ORDERS="$1"
    M_LIST="$2"
    RUN_ALL=0
fi

# Validate order names.
for order in ${ORDERS}; do
    case "${order}" in
        ijk|ikj|jik|jki|kij|kji) ;;
        *) echo "Error: unknown order '${order}'. Valid: ijk ikj jik jki kij kji" >&2
           exit 1 ;;
    esac
done

# Initialize CSVs.
for order in ${ORDERS}; do
    echo "${HEADER}" > "${RESULTS_DIR}/loop_${order}.csv"
done
if [[ "${RUN_ALL}" -eq 1 ]]; then
    echo "${HEADER}" > "${COMBINED_CSV}"
fi

# Run measurements.
for order in ${ORDERS}; do
    echo "=== ${order} ===" >&2
    for m in ${M_LIST}; do
        echo -n "  m=${m} ... " >&2
        line=$("${BENCH}" "${order}" "${m}")
        echo "${line}" >> "${RESULTS_DIR}/loop_${order}.csv"
        if [[ "${RUN_ALL}" -eq 1 ]]; then
            echo "${line}" >> "${COMBINED_CSV}"
        fi
        gflops=$(echo "${line}" | cut -d',' -f6)
        echo "${gflops} GFLOPS" >&2
    done
    echo "  -> ${RESULTS_DIR}/loop_${order}.csv" >&2
done

if [[ "${RUN_ALL}" -eq 1 ]]; then
    echo "  -> ${COMBINED_CSV} (combined)" >&2
fi
