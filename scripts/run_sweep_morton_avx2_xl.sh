#!/usr/bin/env bash
#
# run_sweep_morton_avx2_xl.sh
#
# Sesion 03 / extension de Prompt 5: corre SOLO bench_morton_avx2_O3
# sobre un grid extendido que llega a m=32768, una potencia de 2 mas
# alla del techo del sweep comparativo de Prompt 5. El proposito es
# observar el comportamiento del microkernel AVX2 cuando A pasa de
# 1 GiB (m=16384) a 4 GiB (m=32768), es decir cuando el working set
# ya no cabe en la RAM efectiva de WSL2 segun esta configurada por
# defecto.
#
# Sweep grid:
#   m  in {512, 1024, 2048, 4096, 8192, 16384, 32768}
#   ITERS_PER_RUN = 2
#   RUNS          = 5
#
# Output:
#   results/session_03_morton_avx2_xl.csv
# con la cabecera nativa del bench:
#   m,n,num_iters,median_seconds,gflops
#
# CHEQUEO DE MEMORIA:
#
# A m=32768 los buffers principales son:
#   A         : 32768^2 * 4 = 4 GiB
#   A_morton  : 4 GiB (reorganizacion de A en bloques Morton; no es
#               in-place, vive en paralelo durante el bench)
#   B_curr    : 32768 * 128 * 4 = 16 MiB
#   B_next    : 16 MiB
#   B_out     : I_meas * 128 * 128 * 4 ~= 256 KiB para I_meas=4
#
# Suma minima ~ 8.03 GiB. La instalacion estandar de WSL2 limita la
# VM a la mitad de la RAM del host (techo 8 GiB). En un Ryzen 5
# 4600H con 8 GiB de RAM fisica el limite cae en ~3.5 GiB, lo que
# garantiza un OOM-killer si se intenta m=32768 sin elevar el limite.
#
# El script:
#   1. Lee MemAvailable de /proc/meminfo.
#   2. Estima el working set por m (4 * m^2 * 2 + overhead constante).
#   3. Antes de cada m, si el working set estimado excede el 85% de
#      MemAvailable, imprime un WARNING claro con instrucciones para
#      subir el limite de WSL2 (crear ~/.wslconfig con [wsl2]
#      memory=10GB y reiniciar via "wsl --shutdown"), y le da al
#      usuario 5 segundos para cancelar con Ctrl-C antes de seguir.
#   4. Variable SKIP_MEM_CHECK=1 para saltarse el chequeo (testing).
#
# El bench mismo no captura OOM: si el kernel mata el proceso, el
# script lo nota porque "$BIN" devuelve non-zero, salta ese m, y
# continua con la siguiente potencia de 2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MS=(${MS:-512 1024 2048 4096 8192 16384 32768})
ITERS_PER_RUN=${ITERS_PER_RUN:-2}
RUNS=${RUNS:-5}
SKIP_MEM_CHECK=${SKIP_MEM_CHECK:-0}
N_FIXED=128  # bench_morton_avx2 hardcodea n=128 (BLOCK_SIZE_N)

BIN="$REPO_DIR/bin/bench_morton_avx2_O3"
CSV="$REPO_DIR/results/session_03_morton_avx2_xl.csv"
TMP_CSV="${CSV}.tmp"

mkdir -p "$(dirname "$CSV")"

if [ ! -x "$BIN" ]; then
    echo "Error: $BIN no existe o no es ejecutable." >&2
    echo "Hint: 'make bench_morton_avx2_O3' antes de correr este script." >&2
    exit 1
fi

# Backup del CSV previo, si lo hubiera. Tmp + rename para no dejarlo a medias.
if [ -f "$CSV" ]; then
    cp "$CSV" "${CSV}.bak"
fi
trap 'rm -f "$TMP_CSV"' EXIT
echo "m,n,num_iters,median_seconds,gflops" > "$TMP_CSV"

mem_available_kib() {
    # MemAvailable es lo que el kernel reporta como "razonable para usar
    # sin entrar a swap", incluyendo cache reclaimable. Mucho mas honesto
    # que MemFree.
    awk '/^MemAvailable:/ {print $2}' /proc/meminfo
}

estimate_ws_bytes() {
    # A + A_morton + 2 * B + B_out, en bytes. n=128 (BLOCK_SIZE_N), 4
    # bytes por scalar (FP32). B_out asume I_meas=4 (cap del bench).
    local m=$1
    local b_per_scalar=4
    local a_bytes=$(( m * m * b_per_scalar ))
    local a_morton_bytes=$a_bytes
    local b_bytes=$(( m * N_FIXED * b_per_scalar ))
    local bout_bytes=$(( 4 * N_FIXED * N_FIXED * b_per_scalar ))
    echo $(( a_bytes + a_morton_bytes + 2 * b_bytes + bout_bytes ))
}

human_bytes() {
    local b=$1
    awk -v b="$b" 'BEGIN {
        units[0]="B"; units[1]="KiB"; units[2]="MiB"; units[3]="GiB"; units[4]="TiB"
        u=0
        while (b >= 1024 && u < 4) { b /= 1024; u++ }
        printf "%.2f %s", b, units[u]
    }'
}

echo "Sweep grid    : m = ${MS[*]}"
echo "Iters per run : $ITERS_PER_RUN"
echo "Measured runs : $RUNS"
echo "Output CSV    : $CSV"
echo

for m in "${MS[@]}"; do
    # Sanidad: el bench_morton_avx2 requiere m potencia de 2 y m>=4.
    if [ "$m" -lt 4 ] || (( m & (m - 1) )); then
        echo "  [m=$m] SKIP (no potencia de 2 o menor a 4)" >&2
        continue
    fi

    ws=$(estimate_ws_bytes "$m")
    ws_h=$(human_bytes "$ws")

    if [ "$SKIP_MEM_CHECK" = "0" ]; then
        avail_kib=$(mem_available_kib)
        avail_bytes=$(( avail_kib * 1024 ))
        avail_h=$(human_bytes "$avail_bytes")
        # 85% como margen contra kernel + procesos + buffers del bench.
        budget=$(( avail_bytes * 85 / 100 ))
        if [ "$ws" -gt "$budget" ]; then
            echo "  [m=$m] WARNING: working set estimado $ws_h > 85% MemAvailable ($avail_h)" >&2
            echo "  >>> Riesgo de OOM-killer. Para subir el limite de WSL2:" >&2
            echo "      1. Crear C:\\Users\\<usuario>\\.wslconfig con:" >&2
            echo "         [wsl2]" >&2
            echo "         memory=10GB" >&2
            echo "      2. PowerShell: wsl --shutdown && wsl" >&2
            echo "      3. Re-ejecutar este script." >&2
            echo "  Para saltarse este chequeo y aceptar el riesgo:" >&2
            echo "      SKIP_MEM_CHECK=1 bash scripts/run_sweep_morton_avx2_xl.sh" >&2
            echo "  Continuando en 5 segundos (Ctrl-C para abortar)..." >&2
            sleep 5
        fi
    fi

    printf '  [m=%-6d ws_est=%-10s] running ... ' "$m" "$ws_h" >&2
    line=$("$BIN" "$m" "$ITERS_PER_RUN" "$RUNS") || {
        echo "FAILED (bench non-zero, posible OOM o RAM insuficiente)" >&2
        continue
    }
    gflops=$(echo "$line" | awk -F, '{print $5}')
    secs=$(echo "$line"   | awk -F, '{print $4}')
    echo "$line" >> "$TMP_CSV"
    printf 't_med=%ss  gflops=%s\n' "$secs" "$gflops" >&2
done

mv "$TMP_CSV" "$CSV"
trap - EXIT

n_data_rows=$(( $(wc -l < "$CSV") - 1 ))
echo
echo "Sweep complete. $n_data_rows filas escritas en $CSV."
echo "Next: python3 scripts/plot_morton_avx2_xl.py"
