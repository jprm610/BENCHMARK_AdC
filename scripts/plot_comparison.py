#!/usr/bin/env python3
"""
plot_comparison.py - Generate the Phase 6 comparison plots from the
three per-kernel CSVs plus a consolidated comparison CSV.

Inputs (default paths under results/):
    naive_O0.csv      (full sweep)
    recursive_O0.csv  (full sweep)
    morton_O0.csv     (powers of two only)

Outputs under plots/:
    comparison_gflops_vs_m.png         three curves, log-x, L1/L2/L3 guides
    comparison_time_vs_m.png           three curves, log-log + O(m^2 n) ref
    speedup_morton_vs_recursive.png    ratio gflops(morton)/gflops(recursive)
    speedup_morton_vs_naive.png        ratio gflops(morton)/gflops(naive)

Outputs under results/:
    comparison_all.csv                 union of the three CSVs with a
                                       new leading 'kernel' column. This
                                       is the canonical reference table
                                       for the Phase 6 report.

If results/naive_O0.csv is missing, prints a clear hint suggesting
'make sweep_naive' and exits with code 1. Recursive is treated the
same way (suggesting 'make sweep_recursive_run'). Morton is optional:
its absence produces a warning but the rest of the plots still run.

CLI defaults are calibrated for the test machine (Ryzen 5 4600H):
    --l1-kb 32 --l2-kb 512 --l3-kb 4096
Override on a different CPU.
"""

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KERNELS = ("naive", "recursive", "morton")

# Visual style per kernel. Morton uses a dashed line because its sweep
# only has four points; the dashes signal to the reader that the curve
# is not directly comparable to the eleven-point continuous ones.
STYLE = {
    "naive":     dict(color="tab:blue",   marker="o", linestyle="-",
                      label="Naive (ijk, row-major)"),
    "recursive": dict(color="tab:green",  marker="s", linestyle="-",
                      label="Recursive (cache-oblivious, row-major)"),
    "morton":    dict(color="tab:red",    marker="D", linestyle="--",
                      label="Morton (cache-oblivious, Z-order)"),
}


def read_csv(path):
    """Read one bench CSV into a list of dicts sorted by m. Returns []
    if the file is missing or empty."""
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "m": int(row["m"]),
                "n": int(row["n"]),
                "num_iters": int(row["num_iters"]),
                "median_seconds": float(row["median_seconds"]),
                "gflops": float(row["gflops"]),
            })
    rows.sort(key=lambda r: r["m"])
    return rows


def write_comparison_csv(data, out_path):
    """Concatenate the three per-kernel CSVs into one with a leading
    'kernel' column. Only kernels actually present are written."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kernel", "m", "n", "num_iters",
                         "median_seconds", "gflops"])
        for kernel in KERNELS:
            rows = data.get(kernel) or []
            for r in rows:
                writer.writerow([kernel, r["m"], r["n"], r["num_iters"],
                                 f"{r['median_seconds']:.6f}",
                                 f"{r['gflops']:.6f}"])
    print(f"Wrote {out_path}")


def m_at_full_a(cache_bytes, bytes_per_scalar=4):
    """m at which the full m x m matrix A fills the given cache level."""
    return math.sqrt(cache_bytes / bytes_per_scalar)


def m_at_row(cache_bytes, bytes_per_scalar=4):
    """m at which one row of A fills the given cache level."""
    return cache_bytes / bytes_per_scalar


def add_cache_guides(ax, args):
    """Vertical dashed lines for L1 (row of A), L2 (full A), L3 (full A)."""
    transitions = [
        ("L1 (row of A)", m_at_row(args.l1_kb * 1024),     "tab:red"),
        ("L2 (full A)",   m_at_full_a(args.l2_kb * 1024),  "tab:orange"),
        ("L3 (full A)",   m_at_full_a(args.l3_kb * 1024),  "tab:purple"),
    ]
    y_top = ax.get_ylim()[1]
    for label, m_val, color in transitions:
        if m_val <= 0:
            continue
        ax.axvline(m_val, color=color, linestyle=":", alpha=0.5)
        ax.text(m_val, y_top * 0.95, f"{label}  m={m_val:.0f}",
                color=color, rotation=90,
                ha="right", va="top", fontsize=8)


def plot_gflops(data, args, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for kernel in KERNELS:
        rows = data.get(kernel)
        if not rows:
            continue
        ms = np.array([r["m"] for r in rows], dtype=float)
        gflops = np.array([r["gflops"] for r in rows], dtype=float)
        ax.plot(ms, gflops, linewidth=1.6, markersize=6, **STYLE[kernel])

    ax.set_xscale("log", base=2)
    ax.set_xlabel("m")
    ax.set_ylabel("Sustained GFLOP/s")
    subtitle = (f"{args.cpu_label}  |  "
                f"L1d={args.l1_kb} KB/core, "
                f"L2={args.l2_kb} KB/core, "
                f"L3={args.l3_kb // 1024} MB shared")
    ax.set_title("Naive vs Recursive vs Morton: sustained GFLOP/s vs m\n"
                 + subtitle, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    add_cache_guides(ax, args)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_time(data, args, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    n_for_ref = None

    for kernel in KERNELS:
        rows = data.get(kernel)
        if not rows:
            continue
        ms = np.array([r["m"] for r in rows], dtype=float)
        t_iter = np.array([r["median_seconds"] / r["num_iters"]
                           for r in rows], dtype=float)
        ax.loglog(ms, t_iter, linewidth=1.6, markersize=6, **STYLE[kernel])
        if kernel == "naive":
            n_for_ref = rows[0]["n"]
            ms_naive = ms
            t_naive = t_iter

    # Theoretical O(m^2 * n) curve anchored at the smallest m of naive.
    if n_for_ref is not None:
        flops_iter = 2.0 * ms_naive * ms_naive * n_for_ref
        scale = t_naive[0] / flops_iter[0]
        ax.loglog(ms_naive, scale * flops_iter, "--", color="gray",
                  linewidth=1.2, alpha=0.7,
                  label=r"Theoretical $O(m^{2} n)$ (anchored at min m of naive)")

    ax.set_xlabel("m")
    ax.set_ylabel("Time per iteration [s]")
    ax.set_title("Time per iteration vs m (log-log)\n" + args.cpu_label,
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_speedup(data, args, numerator, denominator, out_path):
    """Bar/line plot of gflops(num) / gflops(den) at the m values where
    both kernels were measured."""
    num_rows = data.get(numerator) or []
    den_rows = data.get(denominator) or []
    den_by_m = {r["m"]: r["gflops"] for r in den_rows}

    pairs = []
    for r in num_rows:
        if r["m"] in den_by_m:
            pairs.append((r["m"], r["gflops"] / den_by_m[r["m"]]))
    if not pairs:
        print(f"Skipping {out_path}: no overlapping m between "
              f"{numerator} and {denominator}.")
        return

    ms, ratios = zip(*pairs)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ms, ratios, "o-", color="tab:red", linewidth=1.6, markersize=7)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7,
               label="parity (ratio = 1)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m")
    ax.set_ylabel(f"GFLOP/s({numerator}) / GFLOP/s({denominator})")
    ax.set_title(f"Speedup of {numerator} relative to {denominator}\n"
                 f"{args.cpu_label}", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    for m_v, r_v in pairs:
        ax.annotate(f"{r_v:.2f}", (m_v, r_v),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results",
                   help="directory containing the per-kernel CSVs")
    p.add_argument("--out-dir", default="plots",
                   help="directory for the generated PNGs")
    p.add_argument("--comparison-csv", default=None,
                   help="path of the consolidated CSV "
                        "(default: <results-dir>/comparison_all.csv)")
    p.add_argument("--l1-kb", type=int, default=32,
                   help="L1d size per core in KB (default 32, Ryzen 5 4600H)")
    p.add_argument("--l2-kb", type=int, default=512,
                   help="L2 size per core in KB (default 512)")
    p.add_argument("--l3-kb", type=int, default=4096,
                   help="L3 size shared in KB (default 4096 = 4 MB)")
    p.add_argument("--cpu-label", default="AMD Ryzen 5 4600H",
                   help="CPU model string for plot subtitles")
    return p.parse_args()


def main():
    args = parse_args()

    paths = {
        "naive":     os.path.join(args.results_dir, "naive_O0.csv"),
        "recursive": os.path.join(args.results_dir, "recursive_O0.csv"),
        "morton":    os.path.join(args.results_dir, "morton_O0.csv"),
    }

    data = {}
    missing_required = False
    for kernel in KERNELS:
        rows = read_csv(paths[kernel])
        if rows is None:
            if kernel == "naive":
                print(f"Error: {paths[kernel]} not found.", file=sys.stderr)
                print("Hint: run 'make sweep_naive' first.", file=sys.stderr)
                missing_required = True
            elif kernel == "recursive":
                print(f"Error: {paths[kernel]} not found.", file=sys.stderr)
                print("Hint: run 'make sweep_recursive_run' first.",
                      file=sys.stderr)
                missing_required = True
            else:
                print(f"Warning: {paths[kernel]} not found "
                      f"(Morton sweep is optional; continuing without it).",
                      file=sys.stderr)
                data[kernel] = []
        else:
            data[kernel] = rows
    if missing_required:
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    comparison_csv = args.comparison_csv or os.path.join(
        args.results_dir, "comparison_all.csv")
    write_comparison_csv(data, comparison_csv)

    plot_gflops(data, args,
                os.path.join(args.out_dir, "comparison_gflops_vs_m.png"))
    plot_time(data, args,
              os.path.join(args.out_dir, "comparison_time_vs_m.png"))
    plot_speedup(data, args, "morton", "recursive",
                 os.path.join(args.out_dir, "speedup_morton_vs_recursive.png"))
    plot_speedup(data, args, "morton", "naive",
                 os.path.join(args.out_dir, "speedup_morton_vs_naive.png"))


if __name__ == "__main__":
    main()
