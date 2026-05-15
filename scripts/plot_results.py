#!/usr/bin/env python3
"""
plot_results.py - Generate scaling plots from a baseline CSV.

Reads results/baseline_O0.csv and produces:
  1. plots/baseline_gflops_vs_m.png  - sustained gflops as a function of m,
     with vertical guides for L1, L2, L3 working-set transitions.
  2. plots/baseline_time_vs_m.png    - per-iteration time on a log-log axis
     with the theoretical 2*m^2*n cubic-in-m reference line, so the user
     can compare measured complexity against the expected one.

Default cache sizes are calibrated for the test machine:
  AMD Ryzen 5 4600H (Renoir, Zen 2, 6 cores)
    L1d : 32 KB per core   (192 KiB total / 6 instances)
    L2  : 512 KB per core  (3 MiB total / 6 instances)
    L3  : 4 MB shared      (4 MiB / 1 instance)

The user can override via CLI for a different machine:
  --l1-kb 32 --l2-kb 1024 --l3-kb 32768
"""

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend; no DISPLAY needed inside WSL.
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: str):
    """Read the baseline CSV into a list of dicts keyed by column name."""
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


def m_at_working_set(cache_bytes: float, n: int, bytes_per_scalar: int = 4):
    """Return the m at which row(A) reaches the given cache.

    The dominant working set for the tall-skinny matmul at the row level
    is a row of A (m * bytes_per_scalar) plus the matrix B (m * n *
    bytes_per_scalar). We use row(A) alone as a first approximation of
    the L1 transition since rows of A iterate fastest in the ikj-like
    pattern; for L2/L3 we use the full A matrix m*m*bytes_per_scalar.
    """
    # m_l1 where one row of A no longer fits in L1:
    # m * bytes_per_scalar = cache_bytes  -> m = cache_bytes / bytes_per_scalar
    return cache_bytes / bytes_per_scalar


def m_at_full_a(cache_bytes: float, bytes_per_scalar: int = 4):
    """Return the m at which the full m*m matrix A stops fitting in cache."""
    # m^2 * bytes_per_scalar = cache_bytes -> m = sqrt(cache_bytes/bytes)
    return math.sqrt(cache_bytes / bytes_per_scalar)


def plot_gflops(rows, args, out_path):
    ms = np.array([r["m"] for r in rows], dtype=float)
    gflops = np.array([r["gflops"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.plot(ms, gflops, "o-", linewidth=1.6, markersize=6,
            label="Baseline -O0 (naive ijk)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m")
    ax.set_ylabel("Sustained gflops")
    subtitle = (f"{args.cpu_label}  |  "
                f"L1d={args.l1_kb} KB/core, "
                f"L2={args.l2_kb} KB/core, "
                f"L3={args.l3_kb // 1024} MB shared")
    ax.set_title("Naive matmul benchmark: sustained gflops vs m\n" + subtitle,
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

    # Vertical guides for cache transitions. Two flavors:
    #   * A full A stops fitting in level X (m = sqrt(size / 4))
    #   * A row of A stops fitting in L1 (m = size / 4)
    n = rows[0]["n"] if rows else 128
    transitions = [
        ("L2 (full A)",    m_at_full_a(args.l2_kb * 1024),          "tab:orange"),
        ("L3 (full A)",    m_at_full_a(args.l3_kb * 1024),          "tab:purple"),
        ("L1 (row of A)",  m_at_working_set(args.l1_kb * 1024, n),  "tab:red"),
    ]
    y_top = ax.get_ylim()[1]
    for label, m_val, color in transitions:
        if m_val is None or m_val <= 0:
            continue
        ax.axvline(m_val, color=color, linestyle="--", alpha=0.6)
        ax.text(m_val, y_top * 0.95, f"{label}  m={m_val:.0f}",
                color=color, rotation=90,
                ha="right", va="top", fontsize=8)

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_time(rows, args, out_path):
    """Per-iteration time vs m on log-log, with theoretical O(m^2 * n) ref."""
    ms = np.array([r["m"] for r in rows], dtype=float)
    n = rows[0]["n"] if rows else 128
    # Time per iteration = total_time / num_iters
    t_iter = np.array([r["median_seconds"] / r["num_iters"] for r in rows])

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.loglog(ms, t_iter, "o-", linewidth=1.6, markersize=6,
              label="Measured time per iteration")

    # Theoretical cubic-in-m-but-actually-quadratic-times-n curve.
    # Each iteration does 2*m*m*n flops. Fit the constant by anchoring at
    # the smallest m so the shape (slope = 2 on log-log) is the comparison.
    flops_iter = 2.0 * ms * ms * n
    anchor_idx = 0
    scale = t_iter[anchor_idx] / flops_iter[anchor_idx]
    ax.loglog(ms, scale * flops_iter, "--",
              label="Theoretical O(m^2 * n) (anchored at smallest m)",
              alpha=0.7)

    ax.set_xlabel("m")
    ax.set_ylabel("Time per iteration [s]")
    ax.set_title("Measured time per iteration vs theoretical complexity\n"
                 + args.cpu_label, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="results/baseline_O0.csv",
                   help="path to the CSV with sweep results")
    p.add_argument("--out-dir", default="plots",
                   help="directory for the generated plots")
    # Defaults match the AMD Ryzen 5 4600H test machine. Override on the
    # command line if you run on a different CPU.
    p.add_argument("--l1-kb", type=int, default=32,
                   help="L1 data cache size per core in KB "
                        "(default 32, Ryzen 5 4600H)")
    p.add_argument("--l2-kb", type=int, default=512,
                   help="L2 cache size per core in KB "
                        "(default 512, Ryzen 5 4600H)")
    p.add_argument("--l3-kb", type=int, default=4096,
                   help="L3 cache size (shared) in KB "
                        "(default 4096 = 4 MB, Ryzen 5 4600H)")
    p.add_argument("--cpu-label", default="AMD Ryzen 5 4600H",
                   help="CPU model name to embed in the plot subtitle")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.csv):
        print(f"Error: CSV file not found at {args.csv}", file=sys.stderr)
        print("Hint: run 'make sweep' first.", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_csv(args.csv)
    if not rows:
        print("Error: CSV is empty.", file=sys.stderr)
        sys.exit(1)

    plot_gflops(rows, args, os.path.join(args.out_dir, "baseline_gflops_vs_m.png"))
    plot_time(rows, args, os.path.join(args.out_dir, "baseline_time_vs_m.png"))


if __name__ == "__main__":
    main()
