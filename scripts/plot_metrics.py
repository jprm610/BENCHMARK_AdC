#!/usr/bin/env python3
"""plot_metrics.py

Render the full set of figures that summarize the matmul benchmark
sweep recorded in results/metrics.csv.

Reads:
    results/metrics.csv     (one row per (variant, m), produced by
                             scripts/consolidate_perf_zen2.py)

Writes (into plots/):
    1.  gflops_vs_m.png             Lines, one per variant; log-log
                                    axes. Shows raw throughput across
                                    problem sizes.
    2.  speedup_vs_naive.png        GFLOPS / GFLOPS_naive(m), per
                                    variant; isolates the algorithmic
                                    gain from the m effect.
    3.  best_per_family.png         Curated subset (naive, loop_ikj,
                                    tiled_ikj, tiled_ikj_avx2,
                                    tiled_ikj_omp, morton, morton_avx2,
                                    morton_omp) telling the optimization
                                    story in one figure.
    4.  efficiency_pct_peak.png     Achieved fp_ops_per_cycle as % of
                                    the Zen 2 peak (16 ops/cycle/core).
                                    Bar grouped, one cluster per m.
    5.  perf_breakdown.png          2x2 panel: IPC, FP ops/cycle, L3
                                    miss rate, TLB walks per kinst.
    6.  cache_hierarchy.png         3 panels: L1D miss rate, L2 load
                                    hit rate, L3 miss rate vs m.
    7.  omp_scaling.png             Side-by-side bars: avx2 vs omp
                                    counterpart for morton and
                                    tiled_ikj; parallel efficiency
                                    label per pair.
    8.  roofline.png                Roofline anchored to the 4600H
                                    (single-core compute peak,
                                    all-core compute peak, DRAM and
                                    cache bandwidth ceilings). Each
                                    variant plotted at its estimated
                                    operational intensity with the
                                    measured GFLOPS.

Hardware model used for the roofline ceilings is the AMD Ryzen 5 4600H
(Renoir, Zen 2; 6 cores; sustained AVX2 boost ~3.3 GHz all-core,
~3.7 GHz single-core; DDR4-3200 dual channel). Operational intensities
per variant are theoretical estimates (algorithmic reuse pattern);
real per-variant measured intensity is not available because the
consolidated CSV stores rates instead of raw load / l2_request counts.

Tolerant to:
    - missing m for any variant: that point is dropped, line continues.
    - low-quality cells (min_mux_pct < 80): the marker is rendered
      with reduced alpha (data still plotted, visually attenuated).
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

# Variant -> color and marker. Color is by family; marker is by
# sub-variant (plain = circle, avx2 = triangle, omp = star).
STYLE = {
    "naive":          dict(color="#444444", marker="o", label="naive"),
    "loop_ijk":       dict(color="#e67e00", marker="o", label="loop ijk"),
    "loop_ikj":       dict(color="#b35900", marker="s", label="loop ikj"),
    "loop_jik":       dict(color="#cc9900", marker="D", label="loop jik"),
    "loop_jki":       dict(color="#997300", marker="v", label="loop jki"),
    "loop_kij":       dict(color="#e6b800", marker="P", label="loop kij"),
    "loop_kji":       dict(color="#b38f00", marker="X", label="loop kji"),
    "tiled_ikj":      dict(color="#1f77b4", marker="o",
                           label="tiled ikj"),
    "tiled_ikj_avx2": dict(color="#1f77b4", marker="^",
                           label="tiled ikj + AVX2"),
    "tiled_ikj_omp":  dict(color="#1f77b4", marker="*",
                           label="tiled ikj + AVX2 + OMP"),
    "morton":         dict(color="#9467bd", marker="o", label="morton"),
    "morton_avx2":    dict(color="#9467bd", marker="^",
                           label="morton + AVX2"),
    "morton_omp":     dict(color="#9467bd", marker="*",
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

# FP32 ops per cycle per core: 2 FMA pipes x 8 SIMD lanes (AVX2) x 2
# (FMA counted as 2 FLOPs). Zen 2 supports 256-bit AVX2.
PEAK_FP_OPS_PER_CYCLE_PER_CORE = 16

# Sustained AVX2 clocks. Empirical figures for the 4600H.
FREQ_SINGLE_CORE_GHZ = 3.7
FREQ_ALL_CORE_GHZ    = 3.3
NUM_CORES            = 6

# Peak compute throughput, derived. Used as horizontal ceilings on the
# roofline.
PEAK_SC_GFLOPS = PEAK_FP_OPS_PER_CYCLE_PER_CORE * FREQ_SINGLE_CORE_GHZ
PEAK_AC_GFLOPS = (PEAK_FP_OPS_PER_CYCLE_PER_CORE
                  * FREQ_ALL_CORE_GHZ * NUM_CORES)

# Bandwidths (GB/s). Sustained estimates, not headline numbers.
# L1D: 32 B/cycle per core (one 256-bit load). L2: similar at L1
# refill. L3: shared, conservative aggregate. DRAM: DDR4-3200 dual
# channel theoretical = 51.2 GB/s.
BW_L1D_SC   = 32 * FREQ_SINGLE_CORE_GHZ          # ~118 GB/s, one core
BW_L2_SC    = 32 * FREQ_SINGLE_CORE_GHZ          # ~118 GB/s, one core
BW_L3_AGG   = 32 * FREQ_ALL_CORE_GHZ             # ~106 GB/s, shared
BW_DRAM     = 51.2                               # DDR4-3200 dual ch.

# Theoretical operational intensity estimates per variant, in
# FP32 FLOPs / byte transferred to/from DRAM. These are derived from
# the algorithmic reuse pattern (not measured from perf), so they are
# documented per family below.
#
# baseline / unblocked ikj / jik:
#   Reads A row and B element / scalar in inner loop. Once matrices
#   exceed L3 the inner loop continuously refills B from DRAM, giving
#   intensity ~ O(1).
# loop_jki / loop_kji:
#   Column-major access on row-major data. TLB and cache thrash
#   dominate; effective DRAM intensity is even worse than naive.
# loop_ikj / loop_kij:
#   B is streamed row by row, A[i][k] hoisted in register. Better
#   reuse than naive but still no blocking.
# tiled_ikj:
#   Block reuse of A and B inside an L2-sized tile (Mc=Kc=256). Each
#   element of A reused Kc times. Intensity ~ Mc/8 in FP32.
# tiled_ikj_avx2:
#   BLIS-style register tile 6x16 inside the L2 macro-tile. Effective
#   intensity grows further because the register tile multiplies the
#   reuse.
# tiled_ikj_omp:
#   Same per-thread intensity as tiled_ikj_avx2 (parallelism across
#   the outer ic loop).
# morton:
#   Cache-oblivious recursion. Reuse depends on subproblem size; in
#   practice between unblocked and tiled.
# morton_avx2, morton_omp:
#   Microkernel 4x16 + Morton block layout brings the effective
#   intensity closer to tiled_ikj_avx2.
INTENSITY_ESTIMATE = {
    "naive":          0.25,
    "loop_ijk":       0.25,
    "loop_jik":       0.25,
    "loop_jki":       0.13,
    "loop_kji":       0.13,
    "loop_ikj":       1.5,
    "loop_kij":       1.5,
    "tiled_ikj":      8.0,
    "tiled_ikj_avx2": 32.0,
    "tiled_ikj_omp":  32.0,
    "morton":         3.0,
    "morton_avx2":    20.0,
    "morton_omp":     20.0,
}


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
    numeric column to float (m to int). Missing or 'nan' values are
    converted to math.nan so the plotting code can decide what to do.

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
    """Return {variant: [row, ...]} with each variant's rows sorted
    by m ascending. Variants not in VARIANTS are still emitted (so
    new variants in the CSV surface, just without a custom style).
    """
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["variant"], []).append(r)
    for variant in out:
        out[variant].sort(key=lambda r: r["m"])
    return out


def style_for(variant: str) -> dict:
    """Look up STYLE with a fallback that still gives the variant a
    distinguishable color so unexpected variants do not crash."""
    return STYLE.get(variant, dict(color="black", marker="x",
                                   label=variant))


def alpha_for(row: dict) -> float:
    """Attenuate points coming from low-confidence cells (perf event
    multiplexing < 80%)."""
    mux = row.get("min_mux_pct", math.nan)
    if isinstance(mux, float) and math.isfinite(mux) and mux < 80.0:
        return 0.35
    return 1.0


# ---------------------------------------------------------------------
# 4. Individual figures
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


def plot_speedup_vs_naive(by_variant, ms_all, out_path):
    """Figure 2. Speedup over naive(m), per variant. Removes the m
    effect from the comparison so the algorithmic gain is the only
    signal."""
    import matplotlib.pyplot as plt

    naive_by_m = {r["m"]: r["gflops"] for r in by_variant.get("naive", [])}

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for variant in VARIANTS:
        if variant == "naive":
            continue
        rows = by_variant.get(variant, [])
        if not rows:
            continue
        xs, ys = [], []
        for r in rows:
            m = r["m"]
            g = r["gflops"]
            base = naive_by_m.get(m)
            if base is None or not math.isfinite(base) or base == 0:
                continue
            if not isinstance(g, float) or not math.isfinite(g):
                continue
            xs.append(m)
            ys.append(g / base)
        if not xs:
            continue
        st = style_for(variant)
        ax.plot(xs, ys, color=st["color"], marker=st["marker"],
                linewidth=1.6, markersize=7, label=st["label"])

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2,
               alpha=0.6, label="naive baseline")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("m (log2 scale)")
    ax.set_ylabel("Speedup over naive (log scale)")
    ax.set_title(f"Speedup vs naive baseline  -  {CPU_LABEL}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_per_family(by_variant, ms_all, out_path):
    """Figure 3. Executive plot: only the curated subset of variants
    that represents the optimization arc."""
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

    # Reference ceilings.
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


def plot_efficiency_pct_peak(by_variant, ms_all, out_path):
    """Figure 4. fp_ops_per_cycle expressed as % of the Zen 2 peak
    (16 ops/cycle/core, single core)."""
    import numpy as np
    import matplotlib.pyplot as plt

    ms = sorted(ms_all)
    n_var = len(VARIANTS)
    width = 0.8 / n_var
    x = np.arange(len(ms))

    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, variant in enumerate(VARIANTS):
        rows = by_variant.get(variant, [])
        by_m = {r["m"]: r["fp_ops_per_cycle"] for r in rows}
        ys = []
        for m in ms:
            v = by_m.get(m, math.nan)
            if isinstance(v, float) and math.isfinite(v):
                # Note: fp_ops_per_cycle from perf counts SSE/AVX ops
                # globally; for OMP variants the counter aggregates
                # across all cores. The 'peak' here is single-core,
                # so OMP variants legitimately can exceed 100%.
                ys.append(100.0 * v / PEAK_FP_OPS_PER_CYCLE_PER_CORE)
            else:
                ys.append(math.nan)
        offset = (idx - (n_var - 1) / 2.0) * width
        st = style_for(variant)
        ax.bar(x + offset, ys, width,
               color=st["color"], label=st["label"])

    ax.axhline(100.0, color="black", linestyle="--", linewidth=1.0,
               alpha=0.7, label="single-core peak (16 ops/cyc)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("m")
    ax.set_ylabel("% of single-core FP32 peak")
    ax.set_title(f"Vector efficiency vs Zen 2 peak  -  {CPU_LABEL}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bar_grouped(ax, by_variant, ms, key, ylabel, title, ylim_top=None):
    """Grouped-bar helper used by plot_perf_breakdown."""
    import numpy as np
    n_var = len(VARIANTS)
    width = 0.8 / n_var
    x = np.arange(len(ms))
    for idx, variant in enumerate(VARIANTS):
        rows = by_variant.get(variant, [])
        by_m = {r["m"]: r[key] for r in rows}
        ys = [by_m.get(m, math.nan) for m in ms]
        offset = (idx - (n_var - 1) / 2.0) * width
        st = style_for(variant)
        ax.bar(x + offset, ys, width,
               color=st["color"], label=st["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("m")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top)


def _line_per_variant(ax, by_variant, ms, key, ylabel, title,
                      log_y: bool = False, hline=None):
    """Line-per-variant helper used by plot_perf_breakdown."""
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


def plot_perf_breakdown(by_variant, ms_all, out_path):
    """Figure 5. 2x2 panel summarizing perf counters: IPC,
    FP ops / cycle, L3 miss rate, TLB walks per kinst."""
    import matplotlib.pyplot as plt

    ms = sorted(ms_all)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_ipc, ax_fp), (ax_l3, ax_tlb) = axes

    _bar_grouped(ax_ipc, by_variant, ms, "ipc",
                 "Instructions per cycle (IPC)",
                 "(a) IPC per variant and m")

    _bar_grouped(ax_fp, by_variant, ms, "fp_ops_per_cycle",
                 "FP ops per cycle",
                 "(b) FMA throughput (FP32 ops / cycle)")
    ax_fp.axhline(PEAK_FP_OPS_PER_CYCLE_PER_CORE,
                  color="black", linestyle=":", linewidth=1.2,
                  alpha=0.8,
                  label=f"peak {PEAK_FP_OPS_PER_CYCLE_PER_CORE} "
                        "(2 FMA x 8 lanes)")
    ax_fp.legend(loc="upper left", fontsize=8)

    _line_per_variant(ax_l3, by_variant, ms, "l3_miss_rate",
                      "L3 miss rate (cache-misses / l2_request)",
                      "(c) L3 miss rate vs m (cliff at m ~ 1024 for naive)")

    _line_per_variant(ax_tlb, by_variant, ms, "tlb_walk_per_kinst",
                      "TLB walks per 1000 instructions",
                      "(d) TLB walks per kinst (log scale)",
                      log_y=True)

    handles, labels = ax_ipc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(VARIANTS), 7),
               bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=9)
    fig.suptitle(f"Perf counter breakdown  -  {CPU_LABEL}", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cache_hierarchy(by_variant, ms_all, out_path):
    """Figure 6. Three panels covering the cache hierarchy: L1D miss
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
    """Figure 7. Side-by-side bars comparing the single-core AVX2
    variant against its OpenMP counterpart, for both morton and
    tiled_ikj. Speedup vs single-thread version is annotated on top
    of the OMP bar; parallel efficiency = speedup / NUM_CORES is
    shown below."""
    import numpy as np
    import matplotlib.pyplot as plt

    pairs = (
        ("morton_avx2",    "morton_omp"),
        ("tiled_ikj_avx2", "tiled_ikj_omp"),
    )

    # Use only m values that have both variants in at least one pair.
    ms = sorted(ms_all)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             sharey=True)

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


def plot_roofline(by_variant, ms_all, out_path):
    """Figure 8. Roofline anchored to the 4600H. Memory ceilings
    (diagonal lines): L1D / L2 / L3 / DRAM bandwidth. Compute
    ceilings (horizontal lines): single-core peak and all-core
    peak.

    Each variant is plotted at one point per m, with:
        x = theoretical operational intensity (see INTENSITY_ESTIMATE,
            documented above; based on algorithmic reuse, not measured
            from perf, because the consolidated CSV stores rates
            instead of raw load / l2_request counts).
        y = measured GFLOPS.

    Marker size encodes m (bigger m = bigger marker).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 7.5))

    # Memory ceilings (diagonals).
    intensity = np.logspace(-1.5, 3.5, 200)
    ceilings = (
        ("L1D BW (1 core)",  BW_L1D_SC,   "tab:cyan",   "-"),
        ("L2  BW (1 core)",  BW_L2_SC,    "tab:green",  "--"),
        ("L3  BW (shared)",  BW_L3_AGG,   "tab:olive",  "-."),
        ("DRAM BW",          BW_DRAM,     "tab:red",    ":"),
    )
    for label, bw, color, ls in ceilings:
        ax.plot(intensity, bw * intensity, color=color, linestyle=ls,
                linewidth=1.2, alpha=0.7,
                label=f"{label} = {bw:.0f} GB/s")

    # Compute ceilings (horizontals).
    ax.axhline(PEAK_SC_GFLOPS, color="black", linestyle="--",
               linewidth=1.0, alpha=0.7,
               label=f"single-core peak ~{PEAK_SC_GFLOPS:.0f} GFLOPS")
    ax.axhline(PEAK_AC_GFLOPS, color="black", linestyle="-",
               linewidth=1.0, alpha=0.7,
               label=f"all-core peak ~{PEAK_AC_GFLOPS:.0f} GFLOPS")

    # Each variant: one marker per m.
    ms_sorted = sorted(ms_all)
    ms_min, ms_max = ms_sorted[0], ms_sorted[-1]

    def marker_size(m):
        if ms_max == ms_min:
            return 90
        t = (math.log2(m) - math.log2(ms_min)) / (
            math.log2(ms_max) - math.log2(ms_min))
        return 50 + t * 250

    for variant in VARIANTS:
        rows = by_variant.get(variant, [])
        if not rows:
            continue
        intensity_x = INTENSITY_ESTIMATE.get(variant)
        if intensity_x is None:
            continue
        st = style_for(variant)
        xs, ys, sizes = [], [], []
        for r in rows:
            g = r["gflops"]
            if not isinstance(g, float) or not math.isfinite(g):
                continue
            xs.append(intensity_x)
            ys.append(g)
            sizes.append(marker_size(r["m"]))
        if not xs:
            continue
        ax.scatter(xs, ys, s=sizes,
                   color=st["color"], marker=st["marker"],
                   edgecolor="black", linewidth=0.6,
                   alpha=0.85, label=st["label"], zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 2000)
    ax.set_ylim(0.05, max(PEAK_AC_GFLOPS * 1.5, 500))
    ax.set_xlabel("Operational intensity (FP32 FLOPs / DRAM byte, "
                  "theoretical)")
    ax.set_ylabel("Measured throughput (GFLOPS)")
    ax.set_title(f"Roofline  -  {CPU_LABEL}\n"
                 "Marker size scales with m. Intensities per variant "
                 "are algorithmic estimates.",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 5. Driver
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
        ("gflops_vs_m.png",         plot_gflops_vs_m),
        ("speedup_vs_naive.png",    plot_speedup_vs_naive),
        ("best_per_family.png",     plot_best_per_family),
        ("efficiency_pct_peak.png", plot_efficiency_pct_peak),
        ("perf_breakdown.png",      plot_perf_breakdown),
        ("cache_hierarchy.png",     plot_cache_hierarchy),
        ("omp_scaling.png",         plot_omp_scaling),
        ("roofline.png",            plot_roofline),
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
