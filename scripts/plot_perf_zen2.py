#!/usr/bin/env python3
"""plot_perf_zen2.py

Sesion 03 / Prompt 7 - render the four subplots that summarize the
Zen 2 hardware counter sweep.

Reads:
    results/perf_zen2_summary.csv

Writes:
    plots/perf_zen2_breakdown.png

Layout (2 rows x 2 cols):
    (a) IPC                  vs m, one bar group per variant
    (b) FP ops / cycle       vs m, idem; horizontal line at 16 (Zen 2
                             single-core peak: 2 FMA pipes x 8 FP32 lanes)
    (c) L3 miss rate         vs m, one curve per variant; cliff visible
                             at m ~ 1024 for naive / recursive (4 MiB
                             effective L3 per CCX)
    (d) TLB walks / kinst    vs m, one curve per variant; expectation is
                             that morton variants stay well below naive
                             at m=8192 because Z-order keeps the working
                             set inside fewer 4 KiB pages

The "TLB walks" metric is bp_l1_tlb_miss_l2_tlb_miss * 1000 /
instructions. The "L3 miss rate" is cache-misses / l2_request (loads
that missed L1 and went to L2; of those, what fraction also missed
the last level cache).

Tolerant to missing rows: any variant absent for a given m is silently
skipped in that subplot.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


VARIANTS = ("naive", "recursive", "morton", "morton_avx2",
            "loop_ijk", "loop_ikj", "loop_jik",
            "loop_jki", "loop_kij", "loop_kji")

STYLE = {
    "naive":       dict(color="tab:blue",   marker="o",
                        label="naive"),
    "recursive":   dict(color="tab:green",  marker="s",
                        label="recursive"),
    "morton":      dict(color="tab:red",    marker="D",
                        label="morton (fino)"),
    "morton_avx2": dict(color="tab:purple", marker="^",
                        label="morton_avx2 (4x16 + bloques)"),
    "loop_ijk":    dict(color="#e67e00",    marker="o",
                        label="loop ijk"),
    "loop_ikj":    dict(color="#b35900",    marker="s",
                        label="loop ikj"),
    "loop_jik":    dict(color="#cc9900",    marker="D",
                        label="loop jik"),
    "loop_jki":    dict(color="#997300",    marker="^",
                        label="loop jki"),
    "loop_kij":    dict(color="#e6b800",    marker="P",
                        label="loop kij"),
    "loop_kji":    dict(color="#b38f00",    marker="X",
                        label="loop kji"),
}

VARIANT_ORDER = {v: i for i, v in enumerate(VARIANTS)}


def read_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            r = dict(row)
            for key in ("m", "cycles", "instructions",
                        "ipc", "fp_ops", "fp_ops_per_cycle",
                        "l1d_miss_rate", "l2_load_hit_rate", "l3_miss_rate",
                        "tlb_walk_per_kinst", "dtlb_load_miss_per_kinst",
                        "min_mux_pct"):
                if key in r:
                    try:
                        if key == "m":
                            r[key] = int(r[key])
                        else:
                            r[key] = (math.nan if r[key] == "nan"
                                      else float(r[key]))
                    except (TypeError, ValueError):
                        r[key] = math.nan if key != "m" else 0
            rows.append(r)
    return rows


def group_by_variant(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["variant"], []).append(r)
    for variant in out:
        out[variant].sort(key=lambda r: r["m"])
    return out


def bar_grouped(ax, by_variant, ms, key, ylabel, title, ylim_top=None):
    """Generic grouped-bar helper: one cluster of bars per m, one bar
    per variant inside the cluster."""
    import numpy as np
    n_var = len(VARIANTS)
    width = 0.8 / n_var
    x = np.arange(len(ms))
    for idx, variant in enumerate(VARIANTS):
        rows = by_variant.get(variant, [])
        by_m = {r["m"]: r[key] for r in rows}
        ys = [by_m.get(m, math.nan) for m in ms]
        offset = (idx - (n_var - 1) / 2.0) * width
        ax.bar(x + offset, ys, width,
               color=STYLE[variant]["color"],
               label=STYLE[variant]["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("m")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top)


def line_per_variant(ax, by_variant, ms, key, ylabel, title,
                     log_y: bool = False):
    for variant in VARIANTS:
        rows = by_variant.get(variant, [])
        by_m = {r["m"]: r[key] for r in rows}
        ys = [by_m.get(m, math.nan) for m in ms]
        # Drop NaN tails so the line does not jump.
        clean = [(m, y) for m, y in zip(ms, ys)
                 if isinstance(y, float) and math.isfinite(y)]
        if not clean:
            continue
        xs_clean, ys_clean = zip(*clean)
        st = STYLE[variant]
        ax.plot(xs_clean, ys_clean,
                color=st["color"], marker=st["marker"],
                linewidth=1.6, markersize=7,
                label=st["label"])
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("m (log2 scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path,
                   default=Path("results/perf_zen2_summary.csv"))
    p.add_argument("--out", type=Path,
                   default=Path("plots/perf_zen2_breakdown.png"))
    p.add_argument("--cpu-label",
                   default="AMD Ryzen 5 4600H (Renoir, Zen 2)")
    p.add_argument("--fp-peak", type=float, default=16.0,
                   help="Zen 2 single-core FP32 peak in ops/cycle "
                        "(default: %(default)s = 2 FMA pipes x 8 lanes)")
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"Error: {args.csv} no existe. "
              "Corre 'make profile_zen2' primero.", file=sys.stderr)
        return 1

    rows = read_csv(args.csv)
    if not rows:
        print(f"Error: {args.csv} no tiene filas validas.", file=sys.stderr)
        return 1

    by_variant = group_by_variant(rows)
    ms = sorted({r["m"] for r in rows})
    if not ms:
        print("Error: no m values in the CSV.", file=sys.stderr)
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Error: matplotlib no esta disponible. "
              "source ~/venvs/matmul/bin/activate", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    (ax_ipc, ax_fp), (ax_l3, ax_tlb) = axes

    # (a) IPC
    bar_grouped(ax_ipc, by_variant, ms, "ipc",
                "Instructions per cycle (IPC)",
                "(a) IPC por variante y m")

    # (b) FP ops / cycle with Zen 2 peak line
    bar_grouped(ax_fp, by_variant, ms, "fp_ops_per_cycle",
                "FP ops por ciclo",
                "(b) Throughput FMA (FP32 ops / ciclo)")
    ax_fp.axhline(args.fp_peak, color="black", linestyle=":",
                  linewidth=1.2, alpha=0.8,
                  label=f"peak {args.fp_peak:.0f} (2 FMA x 8 lanes)")
    ax_fp.legend(loc="upper left", fontsize=8)

    # (c) L3 miss rate vs m
    line_per_variant(ax_l3, by_variant, ms, "l3_miss_rate",
                     "L3 miss rate (cache-misses / l2_request)",
                     "(c) L3 miss rate vs m (cliff esperado en m ~ 1024)")

    # (d) TLB walks per kinst
    line_per_variant(ax_tlb, by_variant, ms, "tlb_walk_per_kinst",
                     "TLB walks por 1000 instrucciones",
                     "(d) Presion TLB severa (bp_l1_tlb_miss_l2_tlb_miss)",
                     log_y=True)

    # Shared legend at the bottom.
    handles, labels = ax_ipc.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=len(VARIANTS),
               bbox_to_anchor=(0.5, -0.01),
               frameon=False, fontsize=10)

    fig.suptitle(
        f"Sesion 03 - perf Zen 2 breakdown\n{args.cpu_label}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")

    # Summary table on stdout.
    print()
    print(f"{'variant':<14} {'m':>6} {'ipc':>6} {'fp/cyc':>8} "
          f"{'l3_miss':>10} {'tlb_walks/kinst':>16} {'mux%':>6}")
    for r in sorted(rows, key=lambda r: (VARIANT_ORDER.get(r["variant"], 99),
                                         r["m"])):
        print(f"{r['variant']:<14} {r['m']:>6} "
              f"{r['ipc']:>6.2f} {r['fp_ops_per_cycle']:>8.2f} "
              f"{r['l3_miss_rate']:>10.4f} "
              f"{r['tlb_walk_per_kinst']:>16.4f} "
              f"{r['min_mux_pct']:>6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
