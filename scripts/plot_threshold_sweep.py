#!/usr/bin/env python3
"""plot_threshold_sweep.py

Sesion 03 / Prompt 2 -- visualize results/threshold_sweep.csv as a
GFLOPS-vs-threshold plot, one line per problem size m.

Reads:
    results/threshold_sweep.csv   (columns: threshold,m,gflops_median)

Writes:
    plots/threshold_sweep.png

The x axis is logarithmic in the threshold (element products); the
expected shape is non-monotonic with a maximum somewhere between
~16K and ~256K elements: too small a threshold pays recursion overhead,
too large lets the leaf working set spill out of L1d (32 KiB on the
test machine). The annotated argmax (per m) helps choose the empirical
default for the AVX2 microkernel in Prompt 3.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def parse_csv(path: Path) -> dict[int, list[tuple[int, float]]]:
    """Return {m: sorted [(threshold, gflops), ...]} from the CSV."""
    series: dict[int, list[tuple[int, float]]] = {}
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                threshold = int(row["threshold"])
                m = int(row["m"])
                gflops_raw = row["gflops_median"]
                gflops = math.nan if gflops_raw == "nan" else float(gflops_raw)
            except (KeyError, ValueError) as exc:
                print(f"Skipping malformed row {row!r}: {exc}", file=sys.stderr)
                continue
            series.setdefault(m, []).append((threshold, gflops))
    for m in series:
        series[m].sort(key=lambda pair: pair[0])
    return series


def find_argmax(points: list[tuple[int, float]]) -> tuple[int, float]:
    """Return the (threshold, gflops) pair with the maximum gflops,
    ignoring NaN entries. Falls back to the first finite point if all
    are NaN."""
    finite = [(t, g) for t, g in points if math.isfinite(g)]
    if not finite:
        return points[0]
    return max(finite, key=lambda pair: pair[1])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot GFLOPS vs RECURSION_THRESHOLD for the Morton kernel."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/threshold_sweep.csv"),
        help="Input CSV from scripts/run_threshold_sweep.sh "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plots/threshold_sweep.png"),
        help="Output PNG path (default: %(default)s)",
    )
    parser.add_argument(
        "--cpu-label",
        default="AMD Ryzen 5 4600H (Renoir, Zen 2)",
        help="CPU label shown in the subtitle.",
    )
    parser.add_argument(
        "--l1-kib",
        type=int,
        default=32,
        help="L1d size in KiB; drawn as a vertical guide at the threshold "
        "that just fills it for n=128 and m_b=k_b leaves "
        "(default: %(default)s).",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: CSV not found at {args.csv}.", file=sys.stderr)
        print("Run scripts/run_threshold_sweep.sh first.", file=sys.stderr)
        return 1

    series = parse_csv(args.csv)
    if not series:
        print("Error: no usable rows in the CSV.", file=sys.stderr)
        return 1

    # Import here so --help and CSV parsing work without matplotlib.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            "Error: matplotlib not installed in the active Python.\n"
            "Activate the project virtualenv first:\n"
            "    source ~/venvs/matmul/bin/activate\n"
            "or install matplotlib in the current interpreter:\n"
            "    python3 -m pip install matplotlib",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    argmaxes: list[tuple[int, int, float]] = []  # (m, threshold*, gflops*)
    for m in sorted(series):
        points = series[m]
        xs = [t for t, _ in points]
        ys = [g for _, g in points]
        ax.plot(
            xs, ys,
            marker="o", linewidth=1.6,
            label=f"m = {m}",
        )
        t_star, g_star = find_argmax(points)
        argmaxes.append((m, t_star, g_star))
        if math.isfinite(g_star):
            ax.scatter([t_star], [g_star], marker="*", s=140,
                       zorder=5, edgecolors="black", linewidths=0.7)

    # Cache-fit guide: with n_b = 128 and m_b = k_b, the working set in
    # bytes is 4 * (m_b^2 + m_b * 128). Solving for m_b that just fills
    # L1d gives m_b ~= 50, threshold = m_b^2 * 128 ~= 320000. Below that
    # the leaf fits in L1d.
    n_b = 128
    l1_bytes = args.l1_kib * 1024
    # 4 * (m_b^2 + m_b * n_b) = l1_bytes  =>
    # m_b = (-n_b + sqrt(n_b^2 + l1_bytes)) / 2
    disc = n_b * n_b + l1_bytes
    m_b_fit = (-n_b + math.sqrt(disc)) / 2.0
    threshold_l1_fit = int(m_b_fit * m_b_fit * n_b)
    ax.axvline(
        threshold_l1_fit, color="gray", linestyle="--", linewidth=1,
        alpha=0.6,
    )
    ax.text(
        threshold_l1_fit, ax.get_ylim()[1] * 0.05 if ax.get_ylim()[1] else 0,
        f" leaf fits L1d\n at threshold ~ {threshold_l1_fit:,}",
        rotation=90, va="bottom", ha="left",
        fontsize=8, color="gray",
    )

    ax.set_xscale("log")
    ax.set_xlabel("g_recursion_threshold (element products)")
    ax.set_ylabel("GFLOPS (median over runs)")
    ax.set_title(
        "Morton kernel: GFLOPS vs recursion threshold\n"
        f"{args.cpu_label}",
        fontsize=11,
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(title="problem size", loc="best")

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Wrote {args.out}")

    print()
    print("Per-m argmax:")
    print(f"{'m':>6}  {'threshold*':>12}  {'gflops*':>10}")
    for m, t_star, g_star in argmaxes:
        print(f"{m:>6}  {t_star:>12,}  {g_star:>10.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
