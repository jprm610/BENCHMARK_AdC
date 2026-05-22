#!/usr/bin/env bash
#
# run_perf_zen5_sweep.sh
#
# Recorre profile_perf_zen5.sh sobre todas las celdas (variant, m)
# requeridas para el servidor AWS c8a.2xlarge (AMD EPYC 9R45 / Zen 5).
#
#   variants : 13 (mismas que el sweep de Zen 2)
#   m        : 1024 4096 8192  (3 tamanos)
#   -> 39 celdas, 1 invocacion de perf por celda = 39 perf stat runs.
#
# Sobreescribible via env vars: VARIANTS, MS, ITERS_PER_RUN, RUNS.
#
# Despues del sweep:
#   python3 scripts/consolidate_perf_zen5.py --out results/metrics.csv
#
# O usar directamente:
#   make results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VARIANTS=(${VARIANTS:-naive morton morton_avx512 morton_omp loop_ijk loop_ikj loop_jik loop_jki loop_kij loop_kji tiled_ikj tiled_ikj_avx512 tiled_ikj_omp})
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
        bash "$SCRIPT_DIR/profile_perf_zen5.sh" "$variant" "$m" \
            || echo "WARN: cell ($variant, $m) failed; continuing." >&2
    done
done

echo
echo "Sweep complete."
echo "Files in $REPO_DIR/results/<variant>/perf_<variant>_m<M>_A.txt"
echo "Next:"
echo "  python3 scripts/consolidate_perf_zen5.py --out results/metrics.csv"
