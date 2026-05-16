#!/usr/bin/env bash
#
# run_sweep_session_03.sh
#
# Sesion 03 / Prompt 5 - comparative sweep across the four variants
# that the session contrasts: naive, recursive, morton and morton_avx2.
# All four are built at the same -O3 -march=znver2 -mavx2 -mfma flag
# set (Makefile targets bench_naive_O3, bench_recursive_O3,
# bench_morton_O3, bench_morton_avx2_O3) so that any difference in
# GFLOPS reflects the algorithm and the layout, not the optimization
# regime.
#
# Sweep grid:
#   m  in {512, 1024, 2048, 4096, 8192, 16384}
#   ITERS_PER_RUN = 2     (the bench's 2nd positional arg)
#   RUNS          = 5     (the bench's 3rd positional arg; median over RUNS)
#
# Note on the names: the original Prompt 5 spec calls these WARMUP and
# RUNS, but the bench drivers actually accept (num_iters, num_runs) as
# the 2nd and 3rd positional args. There is one fixed warm-up run
# baked into each bench main(), independent of any CLI value. We keep
# the spec's variable name "WARMUP" as ITERS_PER_RUN here to avoid
# confusion: 2 measured iterations per run x 5 runs is what gets
# averaged via the median, on top of the 1 fixed internal warm-up.
#
# Power-of-two filter: matmul_morton and matmul_morton_avx2 require m
# to be a power of 2. Every m in the grid above already satisfies that
# (it is the whole point of the grid), but the check stays as a safety
# net in case someone overrides MS to include arbitrary sizes.
#
# Naive cliff: for very large m, the naive O3 bench can take many
# minutes per (m, runs) combination because its working set thrashes
# DRAM. NAIVE_M_MAX caps the naive sweep so the whole comparison
# completes in a reasonable wall time; the other three variants
# continue beyond NAIVE_M_MAX. Default is the largest m in the grid
# (i.e. no cap by default); override via NAIVE_M_MAX=8192 if needed.
#
# Output: one CSV per variant in results/session_03_<variant>.csv with
# the bench-native header
#   m,n,num_iters,median_seconds,gflops
# This matches every other bench CSV in the project (Sesion 01 and 02)
# so plot_sweep_session_03.py and the existing plot_comparison.py can
# share the same reader.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# The grid is overridable but defaults to the Sesion 03 canonical sweep.
MS=(${MS:-512 1024 2048 4096 8192 16384})
ITERS_PER_RUN=${ITERS_PER_RUN:-2}
RUNS=${RUNS:-5}

# Maximum m to feed to naive. Defaults to "no cap" (largest grid point).
# Set NAIVE_M_MAX=8192 from the shell to skip m=16384 for naive only.
NAIVE_M_MAX_DEFAULT=$(printf '%s\n' "${MS[@]}" | sort -n | tail -1)
NAIVE_M_MAX=${NAIVE_M_MAX:-$NAIVE_M_MAX_DEFAULT}

RESULTS_DIR="$REPO_DIR/results"
mkdir -p "$RESULTS_DIR"

# Try to nudge the CPU governor to performance. Silent failure is fine:
# WSL2 may not have cpupower, or sudo may not be available without a
# prompt, and the bench is still informative on the default governor.
if command -v cpupower >/dev/null 2>&1; then
    sudo -n cpupower frequency-set -g performance >/dev/null 2>&1 || true
fi

VARIANTS=(naive recursive morton morton_avx2)

is_power_of_two() {
    local v="$1"
    [ "$v" -gt 0 ] && [ $(( v & (v - 1) )) -eq 0 ]
}

bin_for_variant() {
    case "$1" in
        naive)        echo "$REPO_DIR/bin/bench_naive_O3" ;;
        recursive)    echo "$REPO_DIR/bin/bench_recursive_O3" ;;
        morton)       echo "$REPO_DIR/bin/bench_morton_O3" ;;
        morton_avx2)  echo "$REPO_DIR/bin/bench_morton_avx2_O3" ;;
        *)            echo "" ;;
    esac
}

# Up-front check: refuse to start the sweep if any required binary is
# missing, so we do not spend 20 minutes on three variants only to
# discover the fourth was never built.
missing=""
for variant in "${VARIANTS[@]}"; do
    bin=$(bin_for_variant "$variant")
    if [ ! -x "$bin" ]; then
        missing="$missing $bin"
    fi
done
if [ -n "$missing" ]; then
    echo "Error: missing binaries:$missing" >&2
    echo "Hint: run 'make sweep_session_03' which depends on all four." >&2
    exit 1
fi

echo "Sweep grid       : m = ${MS[*]}"
echo "Iters per run    : $ITERS_PER_RUN"
echo "Measured runs    : $RUNS"
echo "NAIVE_M_MAX      : $NAIVE_M_MAX"
echo

for variant in "${VARIANTS[@]}"; do
    BIN=$(bin_for_variant "$variant")
    CSV="$RESULTS_DIR/session_03_${variant}.csv"
    TMP_CSV="${CSV}.tmp"

    if [ -f "$CSV" ]; then
        cp "$CSV" "${CSV}.bak"
    fi
    trap 'rm -f "$TMP_CSV"' EXIT

    echo "m,n,num_iters,median_seconds,gflops" > "$TMP_CSV"

    for m in "${MS[@]}"; do
        if { [ "$variant" = "morton" ] || [ "$variant" = "morton_avx2" ]; } \
            && ! is_power_of_two "$m"; then
            echo "  [$variant m=$m] SKIP (not a power of two)" >&2
            continue
        fi
        if [ "$variant" = "naive" ] && [ "$m" -gt "$NAIVE_M_MAX" ]; then
            echo "  [$variant m=$m] SKIP (above NAIVE_M_MAX=$NAIVE_M_MAX)" >&2
            continue
        fi

        printf '  [%-11s m=%-6d] running ... ' "$variant" "$m" >&2
        line=$("$BIN" "$m" "$ITERS_PER_RUN" "$RUNS") || {
            echo "FAILED (bench returned non-zero)" >&2
            continue
        }
        gflops=$(echo "$line" | awk -F, '{print $5}')
        secs=$(echo "$line"   | awk -F, '{print $4}')
        echo "$line" >> "$TMP_CSV"
        printf 't_med=%ss  gflops=%s\n' "$secs" "$gflops" >&2
    done

    mv "$TMP_CSV" "$CSV"
    trap - EXIT
    echo "  -> wrote $CSV"
    echo
done

echo "Sweep complete. CSVs in $RESULTS_DIR/session_03_*.csv"
echo "Next: python3 scripts/plot_sweep_session_03.py"
