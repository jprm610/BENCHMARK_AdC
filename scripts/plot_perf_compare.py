#!/usr/bin/env python3
"""
plot_perf_compare.py - Render the Phase 6 perf comparison from
results/perf_compare.csv.

Inputs (default path):
    results/perf_compare.csv with columns
        m,variant,l1_loads,l1_misses,llc_loads,llc_misses,dtlb_misses,
        cycles,instructions

Outputs under plots/:
    perf_l1_misses.png    - L1-dcache-load-misses vs m, one curve per variant
    perf_llc_misses.png   - LLC-load-misses        vs m, one curve per variant
    perf_dtlb_misses.png  - dTLB-load-misses       vs m, one curve per variant
    perf_summary_table.txt - plain-text table with three derived metrics:
        L1 miss rate    = l1_misses / l1_loads
        LLC miss rate   = llc_misses / llc_loads
        dTLB / instr    = dtlb_misses / instructions

If results/perf_compare.csv is missing, the script prints a clear hint
pointing at 'make perf_compare' and exits with code 1.
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("naive", "recursive", "morton")

STYLE = {
    "naive":     dict(color="tab:blue",  marker="o", linestyle="-",
                      label="naive"),
    "recursive": dict(color="tab:green", marker="s", linestyle="-",
                      label="recursive"),
    "morton":    dict(color="tab:red",   marker="D", linestyle="--",
                      label="morton"),
}


def read_csv(path):
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "m":            int(row["m"]),
                "variant":      row["variant"],
                "l1_loads":     int(row["l1_loads"]),
                "l1_misses":    int(row["l1_misses"]),
                "llc_loads":    int(row["llc_loads"]),
                "llc_misses":   int(row["llc_misses"]),
                "dtlb_misses":  int(row["dtlb_misses"]),
                "cycles":       int(row["cycles"]),
                "instructions": int(row["instructions"]),
            })
    return rows


def group_by_variant(rows):
    """Returns {variant: [row, row, ...]} sorted by m within each list."""
    out = {v: [] for v in VARIANTS}
    for r in rows:
        if r["variant"] in out:
            out[r["variant"]].append(r)
    for v in out:
        out[v].sort(key=lambda r: r["m"])
    return out


def plot_event(grouped, event_key, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    plotted_any = False
    for variant in VARIANTS:
        series = grouped.get(variant) or []
        if not series:
            continue
        ms = np.array([r["m"] for r in series], dtype=float)
        ys = np.array([r[event_key] for r in series], dtype=float)
        ax.plot(ms, ys, linewidth=1.6, markersize=6, **STYLE[variant])
        plotted_any = True

    if not plotted_any:
        print(f"Skipping {out_path}: no data for any variant.")
        plt.close(fig)
        return

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("m")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def write_summary_table(rows, out_path):
    """Plain-text aligned table: m | variant | L1 miss rate | LLC miss rate | dTLB / instr."""

    def safe_ratio(num, den):
        return (num / den) if den > 0 else float("nan")

    # Stable order: by m, then by variant in the canonical order.
    variant_order = {v: i for i, v in enumerate(VARIANTS)}
    rows_sorted = sorted(
        rows,
        key=lambda r: (r["m"], variant_order.get(r["variant"], 99))
    )

    lines = []
    header = f"{'m':>6}  {'variant':<10}  {'L1 miss rate':>14}  {'LLC miss rate':>15}  {'dTLB/instr':>13}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows_sorted:
        l1_rate   = safe_ratio(r["l1_misses"],   r["l1_loads"])
        llc_rate  = safe_ratio(r["llc_misses"],  r["llc_loads"])
        dtlb_rate = safe_ratio(r["dtlb_misses"], r["instructions"])
        lines.append(
            f"{r['m']:>6}  {r['variant']:<10}  "
            f"{l1_rate:>14.4e}  {llc_rate:>15.4e}  {dtlb_rate:>13.4e}"
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")
    print(f"Wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", default="results/perf_compare.csv",
                   help="path to the perf comparison CSV "
                        "(default: results/perf_compare.csv)")
    p.add_argument("--out-dir", default="plots",
                   help="directory for the generated PNGs and table "
                        "(default: plots)")
    return p.parse_args()


def main():
    args = parse_args()

    rows = read_csv(args.csv)
    if rows is None:
        print(f"Error: {args.csv} not found.", file=sys.stderr)
        print("Hint: run 'make perf_compare' first.", file=sys.stderr)
        sys.exit(1)
    if not rows:
        print(f"Error: {args.csv} is empty.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    grouped = group_by_variant(rows)

    plot_event(grouped, "l1_misses",
               "L1-dcache-load-misses vs m",
               "L1-dcache-load-misses",
               os.path.join(args.out_dir, "perf_l1_misses.png"))
    plot_event(grouped, "llc_misses",
               "LLC-load-misses vs m",
               "LLC-load-misses",
               os.path.join(args.out_dir, "perf_llc_misses.png"))
    plot_event(grouped, "dtlb_misses",
               "dTLB-load-misses vs m",
               "dTLB-load-misses",
               os.path.join(args.out_dir, "perf_dtlb_misses.png"))

    write_summary_table(rows,
                        os.path.join(args.out_dir, "perf_summary_table.txt"))


if __name__ == "__main__":
    main()
