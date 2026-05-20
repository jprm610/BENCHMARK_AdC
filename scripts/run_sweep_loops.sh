#!/usr/bin/env bash
# run_sweep_loops.sh - Measure one loop-order kernel across a range of m.
#
# Each order must be run as a separate invocation so that process state
# (cache, branch predictor, thermal budget) does not bleed between orders.
#
# Usage:
#   bash scripts/run_sweep_loops.sh <order>              # default m list
#   bash scripts/run_sweep_loops.sh <order> "<m list>"   # custom m list
#
#   <order>  : one of ijk ikj jik jki kij kji
#   <m list> : space-separated sizes, e.g. "512 1024 2048"
#
# Output:
#   results/loop_<order>.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH="${REPO_ROOT}/bin/bench_loops_O0"
RESULTS_DIR="${REPO_ROOT}/results"

if [[ ! -x "${BENCH}" ]]; then
    echo "Error: ${BENCH} not found. Run 'make bench_loops' first." >&2
    exit 1
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <order> [\"<m list>\"]" >&2
    echo "  order  : ijk | ikj | jik | jki | kij | kji" >&2
    echo "  Run each order separately to avoid cross-contamination." >&2
    exit 1
fi

ORDER="$1"
case "${ORDER}" in
    ijk|ikj|jik|jki|kij|kji) ;;
    *) echo "Error: unknown order '${ORDER}'. Valid: ijk ikj jik jki kij kji" >&2
       exit 1 ;;
esac

DEFAULT_M="256 384 512 768 1024 1536 2048 3072 4096"
M_LIST="${2:-${DEFAULT_M}}"
HEADER="kernel,m,n,num_iters,median_seconds,gflops"
OUT_CSV="${RESULTS_DIR}/loop_${ORDER}.csv"

mkdir -p "${RESULTS_DIR}"
echo "${HEADER}" > "${OUT_CSV}"

echo "=== ${ORDER} ===" >&2
for m in ${M_LIST}; do
    echo -n "  m=${m} ... " >&2
    line=$("${BENCH}" "${ORDER}" "${m}")
    echo "${line}" >> "${OUT_CSV}"
    gflops=$(echo "${line}" | cut -d',' -f6)
    echo "${gflops} GFLOPS" >&2
done
echo "  -> ${OUT_CSV}" >&2
