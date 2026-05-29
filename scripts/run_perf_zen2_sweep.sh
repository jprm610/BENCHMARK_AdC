#!/usr/bin/env bash
#
# run_perf_zen2_sweep.sh
#
# Sesion 03 / Prompt 7 - sweep profile_perf_zen2.sh over all
# (variant, m) cells required by the report:
#
#   variants : naive morton morton_avx2 morton_omp    (4)
#              loop_ijk loop_ikj loop_jik              (6)
#              loop_jki loop_kij loop_kji
#              tiled_ikj tiled_ikj_avx2 tiled_ikj_omp  (3)
#   m        : 1024 4096 8192                          (3)
#   -> 13 variants x 3 m = 39 cells, each cell = 2 perf invocations
#      (group A + group B) = 78 perf stat runs.
#
# Total wall time: dominated by naive at m=8192, which is the
# memory-bound corner. Expect ~10-20 minutes on the 4600H.
#
# This script is the convenience entry point. profile_perf_zen2.sh
# is the per-cell primitive and can be called directly with custom
# (variant, m) arguments if needed.
#
# Overridable:
#   VARIANTS, MS, ITERS_PER_RUN, RUNS  via env vars.
#
# After the sweep:
#   python3 scripts/consolidate_perf_zen2.py
#       -> results/metrics.csv
#   python3 scripts/plot_metrics_perf_zen2.py
#       -> plots/{gflops_vs_m,speedup_vs_naive,best_per_family,
#                 efficiency_pct_peak,perf_breakdown,
#                 cache_hierarchy,omp_scaling,roofline}.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VARIANTS=(${VARIANTS:-naive morton morton_avx2 morton_omp loop_ijk loop_ikj loop_jik loop_jki loop_kij loop_kji tiled_ikj tiled_ikj_avx2 tiled_ikj_omp})
MS=(${MS:-1024 2048 4096 8192 16384 32768})
ITERS_PER_RUN=${ITERS_PER_RUN:-1}
if [ "${ITERS_PER_RUN}" -eq 0 ]; then
    RUNS=${RUNS:-1}
else
    RUNS=${RUNS:-3}
fi

export ITERS_PER_RUN
export RUNS

total=$(( ${#VARIANTS[@]} * ${#MS[@]} ))
count=0

for variant in "${VARIANTS[@]}"; do
    for m in "${MS[@]}"; do
        count=$(( count + 1 ))
        echo "============================================================"
        echo "[$count/$total] variant=$variant m=$m"
        echo "============================================================"
        bash "$SCRIPT_DIR/profile_perf_zen2.sh" "$variant" "$m" \
            || echo "WARN: cell ($variant, $m) failed; continuing." >&2
    done
done

echo
echo "Sweep complete."
echo "Files in $REPO_DIR/results/perf_<variant>_m<M>_{A,B}.txt"
echo "Next:"
echo "  python3 scripts/consolidate_perf_zen2.py"
echo "  python3 scripts/plot_metrics_perf_zen2.py"
