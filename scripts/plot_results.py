#!/usr/bin/env python3
"""
plot_results.py - General benchmark plotter for any matmul sweep CSVs.

All benchmark CSVs in this project share the format:
    kernel,m,n,num_iters,median_seconds,gflops

This script accepts one or more such files, groups rows by the 'kernel'
column, and produces:
  1. <out>.png           sustained GFLOP/s vs m, one line per kernel
  2. <out>_time.png      time-per-iteration vs m (log-log) + O(m^2 n) ref
  3. <out>_combined.csv  union of all input rows (deduped by kernel+m)

Usage examples:
  # naive baseline only (backwards-compatible default):
  python3 scripts/plot_results.py

  # all six loop-order kernels:
  python3 scripts/plot_results.py results/loop_*.csv \\
      --out plots/loop_orders --title "Loop-order kernels"

  # loop orders vs naive on one plot:
  python3 scripts/plot_results.py results/naive_O0.csv results/loop_*.csv \\
      --out plots/loop_vs_naive --title "Loop orders vs naive"

Cache-guide defaults match the AMD Ryzen 5 4600H test machine:
  L1d  32 KB/core  |  L2  512 KB/core  |  L3  4 MB shared
Override with --l1-kb / --l2-kb / --l3-kb (pass 0 to suppress a guide).
"""

import argparse
import csv
import itertools
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Visual palette                                                                #
# --------------------------------------------------------------------------- #
_COLORS  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<"]


def _styles():
    return zip(itertools.cycle(_COLORS), itertools.cycle(_MARKERS))


# --------------------------------------------------------------------------- #
# I/O                                                                           #
# --------------------------------------------------------------------------- #

def _read_one(path):
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "kernel":         row["kernel"],
                "m":              int(row["m"]),
                "n":              int(row["n"]),
                "num_iters":      int(row["num_iters"]),
                "median_seconds": float(row["median_seconds"]),
                "gflops":         float(row["gflops"]),
            })
    return rows


def load_all(paths):
    """
    Read every path, merge rows, deduplicate by (kernel, m) — last file
    wins — and return  kernel -> [rows sorted by m].
    """
    by_kernel = {}
    for path in paths:
        for row in _read_one(path):
            k = row["kernel"]
            if k not in by_kernel:
                by_kernel[k] = {}
            by_kernel[k][row["m"]] = row
    return {k: sorted(v.values(), key=lambda r: r["m"])
            for k, v in by_kernel.items()}


def write_combined_csv(data, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kernel", "m", "n", "num_iters", "median_seconds", "gflops"])
        for rows in data.values():
            for r in rows:
                w.writerow([r["kernel"], r["m"], r["n"], r["num_iters"],
                             f"{r['median_seconds']:.6f}", f"{r['gflops']:.6f}"])
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------- #
# Cache guides                                                                  #
# --------------------------------------------------------------------------- #

def _cache_transitions(args):
    bps = 4  # bytes per float
    out = []
    if args.l1_kb > 0:
        m_row = (args.l1_kb * 1024) / bps
        out.append((f"L1 (row of A)  m={m_row:.0f}", m_row, "tab:red"))
    if args.l2_kb > 0:
        m_full = math.sqrt((args.l2_kb * 1024) / bps)
        out.append((f"L2 (full A)  m={m_full:.0f}", m_full, "tab:orange"))
    if args.l3_kb > 0:
        m_full = math.sqrt((args.l3_kb * 1024) / bps)
        out.append((f"L3 (full A)  m={m_full:.0f}", m_full, "tab:purple"))
    return out


def _add_cache_guides(ax, args):
    y_top = ax.get_ylim()[1]
    for label, m_val, color in _cache_transitions(args):
        ax.axvline(m_val, color=color, linestyle="--", alpha=0.6)
        ax.text(m_val, y_top * 0.95, label,
                color=color, rotation=90, ha="right", va="top", fontsize=8)


# --------------------------------------------------------------------------- #
# Plots                                                                         #
# --------------------------------------------------------------------------- #

def plot_gflops(data, args, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    for (color, marker), (kernel, rows) in zip(_styles(), data.items()):
        ms     = np.array([r["m"]      for r in rows], dtype=float)
        gflops = np.array([r["gflops"] for r in rows], dtype=float)
        ax.plot(ms, gflops, color=color, marker=marker,
                linewidth=1.6, markersize=6, label=kernel)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("m  (matrix dimension)")
    ax.set_ylabel("Sustained GFLOP/s")
    title    = args.title or "Benchmark: sustained GFLOP/s vs m"
    subtitle = args.cpu_label
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    _add_cache_guides(ax, args)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_time(data, args, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    ref_rows = None
    for (color, marker), (kernel, rows) in zip(_styles(), data.items()):
        ms     = np.array([r["m"] for r in rows], dtype=float)
        t_iter = np.array([r["median_seconds"] / r["num_iters"]
                           for r in rows], dtype=float)
        ax.loglog(ms, t_iter, color=color, marker=marker,
                  linewidth=1.6, markersize=6, label=kernel)
        if ref_rows is None:
            ref_rows = rows

    if ref_rows is not None:
        ms_r  = np.array([r["m"] for r in ref_rows], dtype=float)
        n_r   = ref_rows[0]["n"]
        flops = 2.0 * ms_r ** 2 * n_r
        t_r   = np.array([r["median_seconds"] / r["num_iters"]
                          for r in ref_rows], dtype=float)
        scale = t_r[0] / flops[0]
        ax.loglog(ms_r, scale * flops, "--", color="gray",
                  linewidth=1.2, alpha=0.7,
                  label=r"$O(m^{2}n)$ ref (anchored at min m)")

    ax.set_xlabel("m  (matrix dimension)")
    ax.set_ylabel("Time per iteration [s]")
    title    = args.title or "Benchmark: time per iteration vs m (log-log)"
    subtitle = args.cpu_label
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("csvs", nargs="*",
                   help="Benchmark CSV files (default: results/naive_O0.csv)")
    p.add_argument("--out", default=None,
                   help="Output base path without extension "
                        "(default: plots/<stem-of-first-csv>)")
    p.add_argument("--title", default=None,
                   help="Plot title (auto-generated if omitted)")
    p.add_argument("--l1-kb", type=int, default=32,
                   help="L1d size per core in KB; 0 disables guide (default 32)")
    p.add_argument("--l2-kb", type=int, default=512,
                   help="L2 size per core in KB; 0 disables guide (default 512)")
    p.add_argument("--l3-kb", type=int, default=4096,
                   help="L3 total size in KB; 0 disables guide (default 4096)")
    p.add_argument("--cpu-label", default="AMD Ryzen 5 4600H",
                   help="CPU model string for plot subtitles")
    return p.parse_args()


def main():
    args = parse_args()

    paths = args.csvs if args.csvs else ["results/naive_O0.csv"]

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        for p in missing:
            print(f"Error: file not found: {p}", file=sys.stderr)
        if not args.csvs:
            print("Hint: run 'make sweep_naive' first.", file=sys.stderr)
        sys.exit(1)

    data = load_all(paths)
    if not data:
        print("Error: no data rows found in the provided CSVs.", file=sys.stderr)
        sys.exit(1)

    if args.out is None:
        stem     = os.path.splitext(os.path.basename(paths[0]))[0]
        args.out = os.path.join("plots", stem)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    write_combined_csv(data, args.out + "_combined.csv")
    plot_gflops(data, args, args.out + ".png")
    plot_time(data,   args, args.out + "_time.png")


if __name__ == "__main__":
    main()
