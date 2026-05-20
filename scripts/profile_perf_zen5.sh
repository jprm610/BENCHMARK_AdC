#!/usr/bin/env bash
#
# profile_perf_zen5.sh
#
# Captura hardware counters para una celda (variant, m) en el servidor
# AWS c8a.2xlarge (AMD EPYC 9R45 / Zen 5, hypervisor KVM).
#
# El KVM de esta instancia expone solo eventos hardware genericos. Se
# usan dos grupos de 4 eventos cada uno para evitar multiplexing
# (misma estrategia que profile_perf_zen2.sh):
#
#   Grupo A (4 eventos - compute y LLC):
#     cycles, instructions, cache-references, cache-misses
#
#   Grupo B (4 eventos - branches):
#     cycles, instructions, branch-instructions, branch-misses
#
# stalled-cycles-frontend/backend se omiten: en AMD Zen bajo KVM
# siempre retornan 0 sin valor diagnostico.
#
# METRICAS derivadas en el consolidador:
#   ipc                  = instructions / cycles           (grupo A)
#   llc_misses_per_kinst = cache-misses / instructions * 1000  (grupo A;
#                          cache-references no esta disponible en KVM)
#   branch_miss_rate     = branch-misses / branch-instructions (grupo B)
#
# OUTPUT:
#   results/<variant>/perf_<variant>_m<M>_A.txt   (perf stat -x , grupo A)
#   results/<variant>/perf_<variant>_m<M>_B.txt   (perf stat -x , grupo B)
#   results/<variant>/bench_<variant>_m<M>.csv     (bench stdout, desde grupo A)
#
# USAGE:
#   scripts/profile_perf_zen5.sh [variant] [m]
#     variant : naive | morton | morton_avx2 | morton_omp | loop_* | tiled_ikj*
#               (default: morton_avx2)
#     m       : tamano cuadrado del problema (default: 1024)
#
# PREREQUISITOS:
#   - bin/bench_<variant>_ZEN5 debe existir.
#     Construir con: make results_zen5   (todos)
#                    make bench_<variant>_ZEN5  (uno solo)
#   - kernel.perf_event_paranoid <= 2.
#     Fix: sudo sysctl -w kernel.perf_event_paranoid=1
#     Persistente: echo 'kernel.perf_event_paranoid = 1' | \
#                  sudo tee /etc/sysctl.d/local.conf && sudo sysctl --system

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VARIANT=${1:-morton_avx2}
M=${2:-1024}
ITERS_PER_RUN=${ITERS_PER_RUN:-1}
RUNS=${RUNS:-3}

# 8 cores fisicos, sin SMT. OMP_NUM_THREADS sobreescribible via env var.
case "$VARIANT" in
    loop_*)
        LOOP_ORDER="${VARIANT#loop_}"
        BIN="$REPO_DIR/bin/bench_loops_ZEN5"
        BIN_ARGS="$LOOP_ORDER $M $ITERS_PER_RUN $RUNS"
        ;;
    tiled_ikj_omp)
        BIN="$REPO_DIR/bin/bench_tiled_ikj_omp_ZEN5"
        BIN_ARGS="$M $ITERS_PER_RUN $RUNS"
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
        export OMP_PLACES=cores
        export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
        ;;
    morton_omp)
        BIN="$REPO_DIR/bin/bench_morton_omp_ZEN5"
        BIN_ARGS="$M $ITERS_PER_RUN $RUNS"
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
        export OMP_PLACES=cores
        export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
        ;;
    *)
        BIN="$REPO_DIR/bin/bench_${VARIANT}_ZEN5"
        BIN_ARGS="$M $ITERS_PER_RUN $RUNS"
        ;;
esac

RESULTS_DIR="$REPO_DIR/results/$VARIANT"
OUT_A="$RESULTS_DIR/perf_${VARIANT}_m${M}_A.txt"
OUT_B="$RESULTS_DIR/perf_${VARIANT}_m${M}_B.txt"
BENCH_OUT="$RESULTS_DIR/bench_${VARIANT}_m${M}.csv"

EVENTS_A="cycles,instructions,cache-references,cache-misses"
EVENTS_B="cycles,instructions,branch-instructions,branch-misses"

# --- preflight checks ------------------------------------------------

if ! command -v perf >/dev/null 2>&1; then
    echo "Error: 'perf' not in PATH." >&2
    echo "Hint: sudo yum install perf  (Amazon Linux 2023)" >&2
    exit 1
fi

if [ ! -x "$BIN" ]; then
    echo "Error: $BIN does not exist or is not executable." >&2
    echo "Hint: run 'make bench_${VARIANT}_ZEN5' (or 'make results_zen5') first." >&2
    exit 1
fi

paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 3)
if [ "$paranoid" -gt 2 ]; then
    echo "Error: kernel.perf_event_paranoid = $paranoid (> 2)." >&2
    echo "Fix (hasta el proximo reboot):" >&2
    echo "    sudo sysctl -w kernel.perf_event_paranoid=1" >&2
    echo "Fix persistente:" >&2
    echo "    echo 'kernel.perf_event_paranoid = 1' | \\" >&2
    echo "    sudo tee /etc/sysctl.d/local.conf && sudo sysctl --system" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

case "$VARIANT" in
    morton|morton_avx2|morton_omp)
        if (( M & (M - 1) )) || [ "$M" -lt 4 ]; then
            echo "Error: variant=$VARIANT requires m to be a power of two >= 4 (got $M)." >&2
            exit 1
        fi
        ;;
esac

run_group() {
    local group_name="$1"
    local events="$2"
    local out_file="$3"
    local bench_out="${4:-/dev/null}"

    echo "--- group $group_name : $events" >&2
    if ! perf stat -x , -e "$events" -o "$out_file" -- \
         "$BIN" $BIN_ARGS >"$bench_out" 2>>"${out_file}.bench.err"; then
        echo "Error: perf returned non-zero for group $group_name." >&2
        tail -20 "${out_file}.bench.err" >&2 || true
        return 1
    fi
    [ -s "${out_file}.bench.err" ] || rm -f "${out_file}.bench.err"
    echo "  wrote $out_file" >&2
}

echo "Profiling variant=$VARIANT m=$M (iters_per_run=$ITERS_PER_RUN runs=$RUNS)" >&2

run_group "A" "$EVENTS_A" "$OUT_A" "$BENCH_OUT"
run_group "B" "$EVENTS_B" "$OUT_B"

echo "Done. Files:" >&2
echo "  $OUT_A" >&2
echo "  $OUT_B" >&2
echo "  $BENCH_OUT" >&2
