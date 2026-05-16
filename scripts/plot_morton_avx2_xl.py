#!/usr/bin/env python3
"""plot_morton_avx2_xl.py

Sesion 03 / extension de Prompt 5 -- visualiza la corrida extendida
de matmul_morton_avx2 que llega a m=32768.

Reads:
    results/session_03_morton_avx2_xl.csv
con la cabecera nativa de los bench: m,n,num_iters,median_seconds,gflops.

Writes:
    plots/session_03_morton_avx2_xl.png

El plot tiene una sola curva (morton_avx2) con:
  - Eje x: m en escala log2.
  - Eje y: GFLOPS sostenido.
  - Etiqueta numerica sobre cada punto.
  - Techo horizontal punteado en el pico FMA single-core (default 128
    GFLOPS para Zen 2 a 4 GHz).
  - Techo memory-bound estimado (default 40 GB/s x intensidad
    aritmetica del matmul evaluada en el m geometricamente medio del
    sweep). El STREAM real reemplaza este numero en el Prompt 8.
  - Guias verticales en los cliffs relevantes para el 4600H:
      * L1d (32 KiB)  -> A fila entera ~ m = 8 (no util en este rango)
      * L2  (512 KiB) -> A entera cabe hasta m ~ 362
      * L3  (4 MiB)   -> A entera cabe hasta m ~ 1024 (cliff L3 por CCX)
      * RAM WSL2 limite tipico (~3.5 GiB) -> ws total >= 8 GiB para
        m=32768; etiquetado como "riesgo OOM" si la curva existe alli.

Si la fila para m=32768 esta ausente en el CSV (porque el bench fue
saltado por OOM o por el chequeo de memoria del script de sweep),
la guia "riesgo OOM" se sigue dibujando pero el plot lo refleja
naturalmente porque la curva termina antes.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append({
                    "m": int(row["m"]),
                    "n": int(row["n"]),
                    "num_iters": int(row["num_iters"]),
                    "median_seconds": float(row["median_seconds"]),
                    "gflops": float(row["gflops"]),
                })
            except (KeyError, ValueError) as exc:
                print(f"Skipping malformed row {row!r}: {exc}",
                      file=sys.stderr)
    rows.sort(key=lambda r: r["m"])
    return rows


def m_at_full_a_fits(cache_bytes: float, bytes_per_scalar: int = 4) -> float:
    """m donde A m x m llena exactamente el cache dado."""
    return math.sqrt(cache_bytes / bytes_per_scalar)


def add_ceilings(ax, args, m_min: float, m_max: float) -> None:
    peak = args.peak_fma
    ax.axhline(peak, color="black", linestyle=":", linewidth=1,
               alpha=0.8, label=f"pico FMA single-core ({peak:.0f} GFLOPS)")

    # Intensidad aritmetica del matmul iterado evaluada en sqrt(m_min*m_max).
    # bytes_per_iter ~ 4 * (m^2 + 2 * m * n), flops_per_iter = 2 * m^2 * n.
    bw = args.bw_gbs * 1.0e9
    m_ref = math.sqrt(m_min * m_max)
    n = 128.0
    bytes_per_iter = 4.0 * (m_ref * m_ref + 2.0 * m_ref * n)
    flops_per_iter = 2.0 * m_ref * m_ref * n
    intensity = flops_per_iter / bytes_per_iter
    mem_ceiling = bw * intensity / 1.0e9
    ax.axhline(
        mem_ceiling, color="gray", linestyle="--", linewidth=1, alpha=0.6,
        label=(f"memory-bound estimado "
               f"({args.bw_gbs:.0f} GB/s x {intensity:.1f} flops/B "
               f"= {mem_ceiling:.0f} GFLOPS)"),
    )


def add_cliffs(ax, args) -> None:
    cliffs = [
        ("L2 (A entera, 512 KiB)",
         m_at_full_a_fits(args.l2_kib * 1024), "tab:olive"),
        (f"L3 (A entera, {args.l3_mib} MiB)",
         m_at_full_a_fits(args.l3_mib * 1024 * 1024), "tab:orange"),
        (f"WSL2 RAM limit ({args.ram_gib:.1f} GiB)",
         m_at_full_a_fits(args.ram_gib * 1024 ** 3 / 2.0),
         # /2 porque A + A_morton coexisten; el cliff se cruza cuando
         # cada uno equivale a ~RAM/2, no a RAM completa.
         "tab:red"),
    ]
    y_top = ax.get_ylim()[1]
    for label, m_val, color in cliffs:
        if not (1.0 < m_val < 1.0e7):
            continue
        ax.axvline(m_val, color=color, linestyle=":",
                   linewidth=1, alpha=0.7)
        ax.text(m_val, y_top * 0.93, f"  {label}\n  m ~= {m_val:.0f}",
                color=color, rotation=90, ha="left", va="top", fontsize=8)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path,
                   default=Path("results/session_03_morton_avx2_xl.csv"),
                   help="CSV de entrada (default: %(default)s)")
    p.add_argument("--out", type=Path,
                   default=Path("plots/session_03_morton_avx2_xl.png"),
                   help="PNG de salida (default: %(default)s)")
    p.add_argument("--cpu-label",
                   default="AMD Ryzen 5 4600H (Renoir, Zen 2)")
    p.add_argument("--peak-fma", type=float, default=128.0,
                   help="pico FMA single-core en GFLOPS (default: %(default)s)")
    p.add_argument("--bw-gbs", type=float, default=40.0,
                   help="bandwidth DDR4 realista en GB/s para la guia "
                        "memory-bound (default: %(default)s)")
    p.add_argument("--l2-kib", type=int, default=512,
                   help="L2 por core en KiB (default: %(default)s)")
    p.add_argument("--l3-mib", type=int, default=4,
                   help="L3 efectiva por CCX en MiB (default: %(default)s)")
    p.add_argument("--ram-gib", type=float, default=3.5,
                   help="RAM disponible para el bench en GiB; default 3.5 "
                        "asume el limite WSL2 por defecto (mitad de 8 GiB).")
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"Error: {args.csv} no existe. "
              f"Corre 'make sweep_morton_avx2_xl' primero.", file=sys.stderr)
        return 1

    rows = read_csv(args.csv)
    if not rows:
        print(f"Error: {args.csv} no tiene filas validas.", file=sys.stderr)
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        print("Error: matplotlib no esta disponible. "
              "Activa el venv con: source ~/venvs/matmul/bin/activate",
              file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    ms = np.array([r["m"] for r in rows], dtype=float)
    gflops = np.array([r["gflops"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.plot(ms, gflops, marker="^", linestyle="-", linewidth=1.8,
            markersize=8, color="tab:purple",
            label="morton_avx2 (microkernel 4x16 + Morton de bloques)")
    for m_v, g_v in zip(ms, gflops):
        ax.annotate(f"{g_v:.1f}", (m_v, g_v),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("m (log2 scale)")
    ax.set_ylabel("Sustained GFLOPS")
    ax.set_title(
        "matmul_morton_avx2 extendido a m=32768\n"
        f"{args.cpu_label}",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.3)

    add_ceilings(ax, args, float(ms.min()), float(ms.max()))
    add_cliffs(ax, args)

    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}")

    print()
    print(f"{'m':>8}  {'gflops':>10}  {'t_median_s':>12}")
    for r in rows:
        print(f"{r['m']:>8}  {r['gflops']:>10.4f}  {r['median_seconds']:>12.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
