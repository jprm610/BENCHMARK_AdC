#!/usr/bin/env bash
#
# run_perf_zen2_sweep.sh
#
# Sesion 03 / Prompt 7 - sweep profile_perf_zen2.sh over all
# (variant, m) cells required by the report:
#
#   variants : naive recursive morton morton_avx2     (4)
#              loop_ijk loop_ikj loop_jik              (6)
#              loop_jki loop_kij loop_kji
#   m        : 1024 4096 8192                          (3)
#   -> 30 cells, each cell = 2 perf invocations (group A + group B)
#      = 60 perf stat runs.
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
#       -> results/perf_zen2_summary.csv
#   python3 scripts/plot_perf_zen2.py
#       -> plots/perf_zen2_breakdown.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VARIANTS=(${VARIANTS:-naive recursive morton morton_avx2 loop_ijk loop_ikj loop_jik loop_jki loop_kij loop_kji tiled_ikj tiled_avx2 tiled_omp morton_omp})
MS=(${MS:-1024 4096 8192})
ITERS_PER_RUN=${ITERS_PER_RUN:-1}
RUNS=${RUNS:-3}

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
echo "  python3 scripts/plot_perf_zen2.py"
