#!/usr/bin/env bash
#
# profile_perf_zen5.sh
#
# Captura hardware counters para una celda (variant, m) en el servidor
# AWS c8a.2xlarge (AMD EPYC 9R45 / Zen 5, hypervisor KVM).
#
# El KVM de esta instancia solo expone 8 eventos hardware genericos;
# los eventos raw AMD (fp_ret_sse_avx_ops, ls_dispatch, l2_request_g1,
# l2_cache_req_stat, bp_l1_tlb_miss_l2_tlb_miss) no estan disponibles.
# Los 8 eventos genericos caben en el budget del PMU virtual en una
# sola invocacion de perf sin multiplexing.
#
# EVENTOS (grupo unico):
#   cycles, instructions,
#   cache-references      (proxy de accesos L3),
#   cache-misses          (proxy de misses L3),
#   stalled-cycles-frontend, stalled-cycles-backend,
#   branch-instructions, branch-misses
#
# METRICAS derivadas en el consolidador:
#   ipc                  = instructions / cycles
#   llc_miss_rate        = cache-misses / cache-references
#   frontend_stall_rate  = stalled-cycles-frontend / cycles
#   backend_stall_rate   = stalled-cycles-backend  / cycles
#   branch_miss_rate     = branch-misses / branch-instructions
#
# OUTPUT:
#   results/<variant>/perf_<variant>_m<M>_A.txt   (perf stat -x ,)
#   results/<variant>/bench_<variant>_m<M>.csv     (bench stdout)
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
BENCH_OUT="$RESULTS_DIR/bench_${VARIANT}_m${M}.csv"

EVENTS="cycles,instructions,cache-references,cache-misses,stalled-cycles-frontend,stalled-cycles-backend,branch-instructions,branch-misses"

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

echo "Profiling variant=$VARIANT m=$M (iters_per_run=$ITERS_PER_RUN runs=$RUNS)" >&2

if ! perf stat -x , -e "$EVENTS" -o "$OUT_A" -- \
     "$BIN" $BIN_ARGS >"$BENCH_OUT" 2>"${OUT_A}.bench.err"; then
    echo "Error: perf returned non-zero." >&2
    echo "stderr from bench (last lines):" >&2
    tail -20 "${OUT_A}.bench.err" >&2 || true
    exit 1
fi
[ -s "${OUT_A}.bench.err" ] || rm -f "${OUT_A}.bench.err"

echo "  wrote $OUT_A" >&2
echo "  wrote $BENCH_OUT" >&2
echo "Done." >&2
