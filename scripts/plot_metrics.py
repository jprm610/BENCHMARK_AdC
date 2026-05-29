#!/usr/bin/env python3
"""plot_metrics.py

Render the figures that summarize the matmul benchmark sweep recorded
in results/metrics.csv.

Reads:
    results/metrics.csv     (one row per (variant, m), produced by
                             scripts/consolidate_perf_zen2.py)

Writes (into plots/):
    1. gflops_vs_m.png        Lines, one per variant; log-log axes.
                              Shows raw throughput across problem sizes.
    2. best_per_family.png    Curated subset (naive, loop_ikj,
                              tiled_ikj, tiled_ikj_avx2, tiled_ikj_omp,
                              morton, morton_avx2, morton_omp) telling
                              the optimization story in one figure,
                              with single-core / all-core peak GFLOPS
                              ceiling lines.
    3. cache_hierarchy.png    3 panels: L1D miss rate, L2 load hit
                              rate, L3 miss rate vs m.
    4. omp_scaling.png        Side-by-side bars: avx2 vs omp counterpart
                              for morton and tiled_ikj; parallel speedup
                              and efficiency labeled per pair.

Hardware model: AMD Ryzen 5 4600H (Renoir, Zen 2; 6 cores; sustained
AVX2 boost ~3.3 GHz all-core, ~3.7 GHz single-core; DDR4-3200 dual
channel).

Tolerant to:
    - missing m for any variant: that point is dropped, line continues.
    - low-quality cells (min_mux_pct < 80): not specially marked here,
      the helper alpha_for is available if a future plot needs it.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# 1. Variants, ordering and visual style
# ---------------------------------------------------------------------

# Canonical order: baseline -> loop reorder family -> tiled family ->
# morton family. Within each family, plain -> avx2 -> omp.
VARIANTS = (
    "naive",
    "loop_ijk", "loop_ikj", "loop_jik",
    "loop_jki", "loop_kij", "loop_kji",
    "tiled_ikj", "tiled_ikj_avx2", "tiled_ikj_omp",
    "morton", "morton_avx2", "morton_omp",
)

# Family classification, used to pick the color and the marker.
FAMILY = {
    "naive":           "baseline",
    "loop_ijk":        "loops",
    "loop_ikj":        "loops",
    "loop_jik":        "loops",
    "loop_jki":        "loops",
    "loop_kij":        "loops",
    "loop_kji":        "loops",
    "tiled_ikj":       "tiled",
    "tiled_ikj_avx2":  "tiled",
    "tiled_ikj_omp":   "tiled",
    "morton":          "morton",
    "morton_avx2":     "morton",
    "morton_omp":      "morton",
}

# Palette by family:
#   - tiled       -> dark green
#   - morton      -> dark blue
#   - loops       -> warm yellows / browns
#   - baseline    -> neutral gray
# Marker by sub-variant: plain = circle, avx2 = triangle, omp = star.
STYLE = {
    "naive":          dict(color="#444444", marker="o", label="naive"),
    "loop_ijk":       dict(color="#e67e00", marker="o", label="loop ijk"),
    "loop_ikj":       dict(color="#b35900", marker="s", label="loop ikj"),
    "loop_jik":       dict(color="#cc9900", marker="D", label="loop jik"),
    "loop_jki":       dict(color="#997300", marker="v", label="loop jki"),
    "loop_kij":       dict(color="#e6b800", marker="P", label="loop kij"),
    "loop_kji":       dict(color="#b38f00", marker="X", label="loop kji"),
    "tiled_ikj":      dict(color="#1b5e20", marker="o",
                           label="tiled ikj"),
    "tiled_ikj_avx2": dict(color="#1b5e20", marker="^",
                           label="tiled ikj + AVX2"),
    "tiled_ikj_omp":  dict(color="#1b5e20", marker="*",
                           label="tiled ikj + AVX2 + OMP"),
    "morton":         dict(color="#0d47a1", marker="o", label="morton"),
    "morton_avx2":    dict(color="#0d47a1", marker="^",
                           label="morton + AVX2"),
    "morton_omp":     dict(color="#0d47a1", marker="*",
                           label="morton + AVX2 + OMP"),
}

# Subset used for the "executive" plot: best representative of each
# family + the baseline.
BEST_PER_FAMILY = (
    "naive",
    "loop_ikj",
    "tiled_ikj",
    "tiled_ikj_avx2",
    "tiled_ikj_omp",
    "morton",
    "morton_avx2",
    "morton_omp",
)


# ---------------------------------------------------------------------
# 2. Hardware model (Ryzen 5 4600H, Zen 2 / Renoir)
# ---------------------------------------------------------------------

CPU_LABEL = "AMD Ryzen 5 4600H (Renoir, Zen 2, 6 cores, DDR4-3200)"

# FMA throughput per core. Zen 2 has 2 256-bit FMA pipes per core.
# A 256-bit FMA on FP32 processes 8 lanes simultaneously. Counting
# each FMA as one retired uop (AMD perf convention for
# fp_ret_sse_avx_ops.all), the peak is:
#     PEAK_FMA_OPS_PER_CYCLE_PER_CORE = 2 pipes x 8 lanes = 16
# Each FMA is mul + add, so each retired op corresponds to 2 FLOPs in
# the bench's GFLOPS convention (gflops = 2 m^2 n / time):
#     FLOPS_PER_FMA_OP = 2
PEAK_FMA_OPS_PER_CYCLE_PER_CORE = 16
FLOPS_PER_FMA_OP                = 2

# Sustained AVX2 clocks. Empirical figures for the 4600H.
FREQ_SINGLE_CORE_GHZ = 3.7
FREQ_ALL_CORE_GHZ    = 3.3
NUM_CORES            = 6

# Peak GFLOPS using the bench's FMA-as-2-FLOPs convention.
#     single core: 16 ops/cyc x 2 FLOPs/op x freq_GHz
#     all core   : same x NUM_CORES at the lower sustained all-core clock
PEAK_SC_GFLOPS = (PEAK_FMA_OPS_PER_CYCLE_PER_CORE
                  * FLOPS_PER_FMA_OP * FREQ_SINGLE_CORE_GHZ)
PEAK_AC_GFLOPS = (PEAK_FMA_OPS_PER_CYCLE_PER_CORE
                  * FLOPS_PER_FMA_OP * FREQ_ALL_CORE_GHZ * NUM_CORES)


# ---------------------------------------------------------------------
# 3. CSV reading
# ---------------------------------------------------------------------

NUMERIC_KEYS = (
    "m", "median_seconds", "gflops",
    "cycles", "instructions", "ipc",
    "fp_ops", "fp_ops_per_cycle",
    "l1d_miss_rate", "l2_load_hit_rate", "l3_miss_rate",
    "tlb_walk_per_kinst", "dtlb_load_miss_per_kinst",
    "min_mux_pct",
)


def read_csv(path: Path) -> list[dict]:
    """Parse results/metrics.csv into a list of dicts, coercing every
    numeric column to float (m to int). Missing or 'nan' values become
    math.nan so the plotting code can decide what to do.

    Robust to a 'seconds' column header (legacy name): silently aliases
    it to 'median_seconds'.
    """
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            r = dict(raw)
            if "seconds" in r and "median_seconds" not in r:
                r["median_seconds"] = r.pop("seconds")
            for key in NUMERIC_KEYS:
                if key not in r:
                    r[key] = math.nan
                    continue
                v = r[key]
                if v is None or v == "" or v == "nan":
                    r[key] = math.nan if key != "m" else 0
                    continue
                try:
                    r[key] = int(v) if key == "m" else float(v)
                except (TypeError, ValueError):
                    r[key] = math.nan if key != "m" else 0
            rows.append(r)
    return rows


def group_by_variant(rows: list[dict]) -> dict[str, list[dict]]:
    """Return {variant: [row, ...]} with each variant's rows sorted by
    m ascending. Variants not in VARIANTS are still emitted (so new
    variants in the CSV surface, just without a custom style).
    """
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["variant"], []).append(r)
    for variant in out:
        out[variant].sort(key=lambda r: r["m"])
    return out


def style_for(variant: str) -> dict:
    """Look up STYLE with a fallback so unexpected variants do not
    crash the plot."""
    return STYLE.get(variant, dict(color="black", marker="x",
                                   label=variant))


def alpha_for(row: dict) -> float:
    """Attenuate points coming from low-confidence cells (perf event
    multiplexing < 80%). Currently unused by the active plots; kept
    available for future figures."""
    mux = row.get("min_mux_pct", math.nan)
    if isinstance(mux, float) and math.isfinite(mux) and mux < 80.0:
        return 0.35
    return 1.0


# ---------------------------------------------------------------------
# 4. Helpers shared across plots
# ---------------------------------------------------------------------

def _line_per_variant(ax, by_variant, ms, key, ylabel, title,
                      log_y: bool = False, hline=None):
    """Draw one line per variant on `ax`, taking the value of `key`
    from each row. Used by the cache hierarchy multi-panel figure."""
    for variant in VARIANTS:
        rows = by_variant.get(variant, [])
        by_m = {r["m"]: r[key] for r in rows}
        ys = [by_m.get(m, math.nan) for m in ms]
        clean = [(m, y) for m, y in zip(ms, ys)
                 if isinstance(y, float) and math.isfinite(y)]
        if not clean:
            continue
        xs_clean, ys_clean = zip(*clean)
        st = style_for(variant)
        ax.plot(xs_clean, ys_clean,
                color=st["color"], marker=st["marker"],
                linewidth=1.6, markersize=6, label=st["label"])
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    if hline is not None:
        ax.axhline(hline, color="black", linestyle=":",
                   linewidth=1.0, alpha=0.7)
    ax.set_xlabel("m (log2 scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------------
# 5. Individual figures
# ---------------------------------------------------------------------

def plot_gflops_vs_m(by_variant, ms_all, out_path):
    """Figure 1. GFLOPS vs m, one line per variant, log-log axes.
    Big-picture view of throughput across problem sizes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for variant in VARIANTS:
        rows = by_variant.get(variant, [])
        if not rows:
            continue
        xs = [r["m"] for r in rows]
        ys = [r["gflops"] for r in rows]
        st = style_for(variant)
        ax.plot(xs, ys, color=st["color"], marker=st["marker"],
                linewidth=1.6, markersize=7, label=st["label"])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("m (log2 scale)")
    ax.set_ylabel("Throughput (GFLOPS, log scale)")
    ax.set_title(f"Throughput vs problem size  -  {CPU_LABEL}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_per_family(by_variant, ms_all, out_path):
    """Figure 2. Executive plot: only the curated subset of variants
    that represents the optimization arc, plus the theoretical
    single-core and all-core peak GFLOPS ceilings."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for variant in BEST_PER_FAMILY:
        rows = by_variant.get(variant, [])
        if not rows:
            continue
        xs = [r["m"] for r in rows]
        ys = [r["gflops"] for r in rows]
        st = style_for(variant)
        ax.plot(xs, ys, color=st["color"], marker=st["marker"],
                linewidth=2.0, markersize=9, label=st["label"])

    # Reference ceilings: theoretical peak FP32 throughput on the
    # 4600H, computed as (peak FMA uops/cycle) x (FLOPs per FMA) x
    # frequency. Single-core uses the higher sustained AVX2 boost;
    # all-core uses the lower sustained AVX2 boost x NUM_CORES.
    ax.axhline(PEAK_SC_GFLOPS, color="black", linestyle="--",
               linewidth=1.0, alpha=0.6,
               label=f"single-core peak ~{PEAK_SC_GFLOPS:.0f} GFLOPS")
    ax.axhline(PEAK_AC_GFLOPS, color="black", linestyle="-",
               linewidth=1.0, alpha=0.6,
               label=f"all-core peak ~{PEAK_AC_GFLOPS:.0f} GFLOPS")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("m (log2 scale)")
    ax.set_ylabel("Throughput (GFLOPS, log scale)")
    ax.set_title(f"Optimization arc  -  {CPU_LABEL}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cache_hierarchy(by_variant, ms_all, out_path):
    """Figure 3. Three panels covering the cache hierarchy: L1D miss
    rate, L2 load hit rate, L3 miss rate. Each panel has one line per
    variant. Together they show at which level each variant breaks."""
    import matplotlib.pyplot as plt

    ms = sorted(ms_all)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_l1, ax_l2, ax_l3 = axes

    _line_per_variant(ax_l1, by_variant, ms, "l1d_miss_rate",
                      "L1D miss rate (l2_request / loads)",
                      "(a) L1D miss rate")
    _line_per_variant(ax_l2, by_variant, ms, "l2_load_hit_rate",
                      "L2 load hit rate (l2_hits / l2_request)",
                      "(b) L2 load hit rate")
    _line_per_variant(ax_l3, by_variant, ms, "l3_miss_rate",
                      "L3 miss rate (cache_misses / l2_request)",
                      "(c) L3 miss rate")

    handles, labels = ax_l1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(VARIANTS), 7),
               bbox_to_anchor=(0.5, -0.05),
               frameon=False, fontsize=9)
    fig.suptitle(f"Cache hierarchy breakdown  -  {CPU_LABEL}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_omp_scaling(by_variant, ms_all, out_path):
    """Figure 4. Side-by-side bars comparing the single-core AVX2
    variant against its OpenMP counterpart, for both morton and
    tiled_ikj. Speedup vs single-thread is annotated on top of the
    OMP bar; parallel efficiency = speedup / NUM_CORES is shown
    next to it."""
    import numpy as np
    import matplotlib.pyplot as plt

    pairs = (
        ("morton_avx2",    "morton_omp"),
        ("tiled_ikj_avx2", "tiled_ikj_omp"),
    )

    ms = sorted(ms_all)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, (sc_var, omp_var) in zip(axes, pairs):
        sc_by_m  = {r["m"]: r["gflops"]
                    for r in by_variant.get(sc_var, [])}
        omp_by_m = {r["m"]: r["gflops"]
                    for r in by_variant.get(omp_var, [])}
        common = [m for m in ms if m in sc_by_m and m in omp_by_m]
        if not common:
            ax.set_visible(False)
            continue

        x = np.arange(len(common))
        width = 0.38

        sc_y  = [sc_by_m[m]  for m in common]
        omp_y = [omp_by_m[m] for m in common]

        ax.bar(x - width / 2, sc_y, width,
               color=style_for(sc_var)["color"], alpha=0.65,
               label=style_for(sc_var)["label"])
        ax.bar(x + width / 2, omp_y, width,
               color=style_for(omp_var)["color"], alpha=1.0,
               hatch="//", edgecolor="white",
               label=style_for(omp_var)["label"])

        ymax = max(max(sc_y), max(omp_y)) * 1.15
        ax.set_ylim(top=ymax)

        for i, m in enumerate(common):
            speedup = omp_y[i] / sc_y[i] if sc_y[i] > 0 else math.nan
            eff = speedup / NUM_CORES * 100.0
            ax.text(i + width / 2, omp_y[i],
                    f"x{speedup:.2f}\n({eff:.0f}% eff)",
                    ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in common])
        ax.set_xlabel("m")
        ax.set_title(
            f"{sc_var}  vs  {omp_var}  (efficiency = speedup / "
            f"{NUM_CORES} cores)",
            fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    axes[0].set_ylabel("Throughput (GFLOPS)")
    fig.suptitle(f"OpenMP scaling  -  {CPU_LABEL}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 6. Driver
# ---------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path,
                   default=Path("results/metrics.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("plots"))
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"Error: {args.csv} does not exist. "
              "Run 'make results' first.", file=sys.stderr)
        return 1

    rows = read_csv(args.csv)
    if not rows:
        print(f"Error: {args.csv} has no usable rows.", file=sys.stderr)
        return 1

    by_variant = group_by_variant(rows)
    ms_all = sorted({r["m"] for r in rows})
    if not ms_all:
        print("Error: no m values in the CSV.", file=sys.stderr)
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except ModuleNotFoundError:
        print("Error: matplotlib is not installed. "
              "Activate the venv: source ~/venvs/matmul/bin/activate",
              file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    figures = (
        ("gflops_vs_m.png",     plot_gflops_vs_m),
        ("best_per_family.png", plot_best_per_family),
        ("cache_hierarchy.png", plot_cache_hierarchy),
        ("omp_scaling.png",     plot_omp_scaling),
    )

    written = 0
    for filename, fn in figures:
        out_path = args.out_dir / filename
        try:
            fn(by_variant, ms_all, out_path)
            print(f"Wrote {out_path}")
            written += 1
        except Exception as exc:
            print(f"Error rendering {filename}: {exc}", file=sys.stderr)

    print()
    print(f"{written} / {len(figures)} figures written to "
          f"{args.out_dir}/")

    # Compact summary table on stdout for log readers.
    print()
    print(f"{'variant':<16} {'m':>6} {'gflops':>8} {'ipc':>5} "
          f"{'fp/cyc':>7} {'l3_miss':>8} {'mux%':>5}")
    for variant in VARIANTS:
        for r in by_variant.get(variant, []):
            print(f"{variant:<16} {r['m']:>6} "
                  f"{r['gflops']:>8.2f} {r['ipc']:>5.2f} "
                  f"{r['fp_ops_per_cycle']:>7.2f} "
                  f"{r['l3_miss_rate']:>8.4f} "
                  f"{r['min_mux_pct']:>5.0f}")
    return 0 if written == len(figures) else 1


if __name__ == "__main__":
    sys.exit(main())
