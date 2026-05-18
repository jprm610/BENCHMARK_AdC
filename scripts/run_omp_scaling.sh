#!/usr/bin/env bash
#
# run_omp_scaling.sh
#
# Sesion 03 / Prompt 6 - OpenMP scaling sweep for matmul_morton_omp.
#
# Crosses (m, threads, bind) and records the median GFLOPS produced
# by bench_morton_omp_O3 for each cell. The point of crossing bind
# in {close, spread} is to expose the CCX topology of the 4600H:
#
#   close  : threads are packed into the lowest-numbered cores first,
#            so 1-3 threads stay inside CCX0 (3 cores + 4 MiB L3
#            private). Above 3 threads close has to spill into CCX1
#            and pays Infinity Fabric for coherence traffic.
#
#   spread : threads are distributed across places, so 2 threads land
#            in different CCXs immediately. This loses L3 sharing at
#            T=2 but uses both CCXs for T<=6 without any thread
#            crowding the same L3 partition.
#
# The expected reading:
#   - T<=3 : close ahead of spread (better L3 reuse inside CCX0).
#   - T=4..6: spread ahead of close (close already crossed CCX, and
#             spread keeps the two CCXs balanced).
#   - T=8..12 (SMT): neither bind helps; the two FMA pipes per core
#             are already saturated by the AVX2 microkernel.
#
# Sweep grid (overrideable via env vars):
#   MS=(4096 8192)
#   THREADS=(1 2 3 4 6 8 12)
#   BINDS=(close spread)
#   ITERS_PER_RUN=1     (each measured run already iterates I=2m/n
#                       internally inside the bench up to cap 4; for
#                       the scaling sweep we use 1 to keep each cell
#                       short)
#   RUNS=3              (median over 3 runs per cell)
#
# Output: results/omp_scaling.csv with columns
#   m,threads,bind,gflops_median,time_median_s,leaf_thr,par_thr
#
# Plot: python3 scripts/plot_omp_scaling.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BIN="$REPO_DIR/bin/bench_morton_omp_O3"
CSV="$REPO_DIR/results/omp_scaling.csv"
TMP_CSV="${CSV}.tmp"

MS=(${MS:-4096 8192})
THREADS=(${THREADS:-1 2 3 4 6 8 12})
BINDS=(${BINDS:-close spread})
ITERS_PER_RUN=${ITERS_PER_RUN:-1}
RUNS=${RUNS:-3}

# Leaf and parallel thresholds passed to the bench; left at the defaults
# unless the caller overrides. Exposing them here lets us experiment
# without touching the source.
LEAF_THR=${LEAF_THR:-524288}
PAR_THR=${PAR_THR:-524288}

mkdir -p "$(dirname "$CSV")"

if [ ! -x "$BIN" ]; then
    echo "Error: $BIN no existe o no es ejecutable." >&2
    echo "Hint: 'make bench_morton_omp_O3' antes de correr este script." >&2
    exit 1
fi

if [ -f "$CSV" ]; then
    cp "$CSV" "${CSV}.bak"
fi
trap 'rm -f "$TMP_CSV"' EXIT

echo "m,threads,bind,gflops_median,time_median_s,leaf_thr,par_thr" > "$TMP_CSV"

# Try to set the CPU governor to performance. Silent failure (WSL2
# normally does not expose cpupower).
if command -v cpupower >/dev/null 2>&1; then
    sudo -n cpupower frequency-set -g performance >/dev/null 2>&1 || true
fi

echo "Bench binary  : $BIN"
echo "MS            : ${MS[*]}"
echo "THREADS       : ${THREADS[*]}"
echo "BINDS         : ${BINDS[*]}"
echo "ITERS_PER_RUN : $ITERS_PER_RUN"
echo "RUNS          : $RUNS"
echo "Leaf/par thr  : $LEAF_THR / $PAR_THR"
echo "Output CSV    : $CSV"
echo

total=$(( ${#MS[@]} * ${#THREADS[@]} * ${#BINDS[@]} ))
count=0

for m in "${MS[@]}"; do
    for bind in "${BINDS[@]}"; do
        for t in "${THREADS[@]}"; do
            count=$(( count + 1 ))
            printf '[%2d/%d] m=%-6d bind=%-7s threads=%-3d ... ' \
                   "$count" "$total" "$m" "$bind" "$t" >&2

            # OMP_PLACES=cores keeps each "place" pinned to one logical
            # CPU, which is the cleanest setup for measuring bind=close
            # vs bind=spread (places coincide with physical thread
            # candidates and the runtime decides packing).
            line=$(env OMP_NUM_THREADS="$t" \
                       OMP_PLACES=cores \
                       OMP_PROC_BIND="$bind" \
                       "$BIN" "$m" "$ITERS_PER_RUN" "$RUNS" \
                       --threshold "$LEAF_THR" \
                       --parallel-threshold "$PAR_THR") || {
                echo "FAILED (bench non-zero)" >&2
                continue
            }

            gflops=$(echo "$line" | awk -F, '{print $5}')
            secs=$(echo   "$line" | awk -F, '{print $4}')

            echo "$m,$t,$bind,$gflops,$secs,$LEAF_THR,$PAR_THR" >> "$TMP_CSV"
            printf 'gflops=%s  t_med=%ss\n' "$gflops" "$secs" >&2
        done
    done
done

mv "$TMP_CSV" "$CSV"
trap - EXIT

n_data_rows=$(( $(wc -l < "$CSV") - 1 ))
echo
echo "Sweep complete. $n_data_rows filas escritas en $CSV."
echo "Esperado: $total."
echo "Next: python3 scripts/plot_omp_scaling.py"
