#!/usr/bin/env python3
"""plot_omp_scaling.py

Sesion 03 / Prompt 6 - visualize the OpenMP scaling sweep.

Reads:
    results/omp_scaling.csv
with columns
    m,threads,bind,gflops_median,time_median_s,leaf_thr,par_thr

Writes:
    plots/omp_scaling.png

Plot layout:
  - x axis : threads
  - y axis : speedup vs 1 thread (computed separately per (m, bind))
  - one series per (m, bind), so the canonical 2x2 sweep yields four
    curves (m=4096 close, m=4096 spread, m=8192 close, m=8192 spread)
  - dashed gray diagonal for ideal speedup (y = x)
  - vertical guide at the 3-thread (single-CCX) and 6-thread (full
    physical cores) boundaries

Tolerant to missing cells (failed bench runs): a series silently skips
the absent thread values.
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
                    "threads": int(row["threads"]),
                    "bind": row["bind"],
                    "gflops": float(row["gflops_median"]),
                    "time": float(row["time_median_s"]),
                })
            except (KeyError, ValueError) as exc:
                print(f"Skipping malformed row {row!r}: {exc}",
                      file=sys.stderr)
    return rows


def group_by_series(rows: list[dict]) -> dict[tuple[int, str], list[dict]]:
    """Group rows by (m, bind) and sort each series by threads."""
    series: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        key = (r["m"], r["bind"])
        series.setdefault(key, []).append(r)
    for key in series:
        series[key].sort(key=lambda r: r["threads"])
    return series


# Visual style: each (m, bind) gets a color from m and a linestyle
# from bind. The same m thus uses two related curves (solid vs dashed)
# so the reader can see the bind contrast at a glance.
M_COLORS = {
    4096: "tab:blue",
    8192: "tab:red",
}
BIND_STYLE = {
    "close":  dict(linestyle="-",  marker="o"),
    "spread": dict(linestyle="--", marker="s"),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path,
                   default=Path("results/omp_scaling.csv"),
                   help="CSV de entrada (default: %(default)s)")
    p.add_argument("--out", type=Path,
                   default=Path("plots/omp_scaling.png"),
                   help="PNG de salida (default: %(default)s)")
    p.add_argument("--cpu-label",
                   default="AMD Ryzen 5 4600H (Renoir, Zen 2)")
    p.add_argument("--cores-per-ccx", type=int, default=3,
                   help="cores por CCX para la guia vertical "
                        "(default: %(default)s)")
    p.add_argument("--physical-cores", type=int, default=6,
                   help="cores fisicos totales para la guia vertical "
                        "(default: %(default)s)")
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"Error: {args.csv} no existe. "
              "Corre 'make sweep_omp_scaling' primero.", file=sys.stderr)
        return 1

    rows = read_csv(args.csv)
    if not rows:
        print(f"Error: {args.csv} no tiene filas validas.", file=sys.stderr)
        return 1

    series = group_by_series(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        print("Error: matplotlib no disponible. "
              "source ~/venvs/matmul/bin/activate", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    max_threads_seen = 1
    summary_lines: list[str] = []
    for (m, bind), pts in sorted(series.items()):
        ts = [p["threads"] for p in pts]
        gs = [p["gflops"]  for p in pts]
        if not ts:
            continue
        base_pts = [p for p in pts if p["threads"] == 1]
        if not base_pts:
            print(f"Warning: no T=1 sample for (m={m}, bind={bind}); "
                  "skipping series (speedup ill-defined).", file=sys.stderr)
            continue
        base = base_pts[0]["gflops"]
        if base <= 0:
            print(f"Warning: base GFLOPS for (m={m}, bind={bind}) is "
                  f"non-positive ({base}); skipping series.", file=sys.stderr)
            continue
        speedups = [g / base for g in gs]
        max_threads_seen = max(max_threads_seen, max(ts))

        color = M_COLORS.get(m, "tab:gray")
        style = BIND_STYLE.get(bind, dict(linestyle="-", marker="x"))
        ax.plot(ts, speedups,
                linewidth=1.8, markersize=7,
                color=color, **style,
                label=f"m={m}, bind={bind}")

        for t_v, s_v in zip(ts, speedups):
            ax.annotate(f"{s_v:.2f}", (t_v, s_v),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8)

        summary_lines.append(
            f"m={m:<5} bind={bind:<6} "
            f"T=1 base={base:.2f} GFLOPS  "
            f"T={max(ts):<2} speedup={max(speedups):.2f}x"
        )

    # Ideal diagonal speedup = threads.
    t_ideal = np.arange(1, max_threads_seen + 1)
    ax.plot(t_ideal, t_ideal, linestyle=":", color="gray",
            linewidth=1, alpha=0.7, label="ideal (speedup = threads)")

    # CCX and physical-core guides.
    for x_v, lbl in [(args.cores_per_ccx, f"1 CCX ({args.cores_per_ccx}c)"),
                     (args.physical_cores,
                      f"physical cores ({args.physical_cores})")]:
        if 1 <= x_v <= max_threads_seen:
            ax.axvline(x_v, color="black", linestyle=":",
                       linewidth=0.8, alpha=0.5)
            ax.text(x_v, ax.get_ylim()[1] * 0.05,
                    f" {lbl}", rotation=90, ha="left", va="bottom",
                    fontsize=8, color="dimgray")

    ax.set_xlabel("OMP_NUM_THREADS")
    ax.set_ylabel("Speedup vs 1 thread")
    ax.set_title(
        "matmul_morton_omp - escalado vs threads y bind\n"
        f"{args.cpu_label}",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8.5)
    ax.set_xticks(sorted(set(int(t) for t in t_ideal)))

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}")

    print()
    print("Resumen por serie:")
    for line in summary_lines:
        print(f"  {line}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
