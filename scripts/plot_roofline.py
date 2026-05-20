#!/usr/bin/env python3
"""plot_roofline.py

Sesion 03 / Prompt 8 - construct the final Roofline diagram of the
session, anchored on measured STREAM bandwidth (results/stream_*.txt)
and measured fp ops (results/perf_*_{A,B}.txt).

Inputs:
    results/stream_1t.txt        single-thread STREAM (Copy, Scale, Add, Triad)
    results/stream_6t.txt        6-thread  STREAM   (idem)
    results/perf_<variant>_m<M>_{A,B}.txt
                                 raw perf stat -x , output per cell

Variants discovered automatically from the filenames; the canonical set
is (naive, morton, morton_avx2) at m in {1024, 4096, 8192}.
If perf_morton_omp_m<M>_{A,B}.txt are present (run
'make profile_zen2_omp' first) they are added as the multi-thread
points.

Output:
    plots/roofline_4600h.png

Roofline math:
  achieved_gflops = fp_ret_sse_avx_ops.all / (runtime_ns / 1e9) / 1e9
  effective_AI    = fp_ret_sse_avx_ops.all / (cache-misses * 64)
                    (64 B is the L3 line size on Zen 2; cache-misses on
                     Zen 2 counts LLC misses i.e. requests served from
                     DRAM, so cache-misses * 64 is a conservative lower
                     bound on bytes that crossed the memory bus.)

Compute ceilings:
  single-core peak       =  2 FMA pipes x 8 FP32 lanes x 4.0 GHz turbo
                         =  128 GFLOPS (theoretical, ignored AVX throttle)
  6-core peak (multi)    =  6 x 128 = 768 GFLOPS (ditto)

Memory ceiling = STREAM Triad bandwidth (single-thread or 6-thread).
The ridge point is at intensity = peak_gflops / bandwidth.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path


# Variants and rendering style. morton_omp is included unconditionally;
# the loader silently skips it if no files are present.
VARIANTS = ("naive", "morton", "morton_avx2", "morton_omp")

STYLE = {
    "naive":       dict(color="tab:blue",   marker="o",
                        label="naive (ijk, row-major, -O3 znver2)"),
    "morton":      dict(color="tab:red",    marker="D",
                        label="morton (fino, Z-order de elementos)"),
    "morton_avx2": dict(color="tab:purple", marker="^",
                        label="morton_avx2 (microkernel 4x16, single-thread)"),
    "morton_omp":  dict(color="tab:orange", marker="*",
                        label="morton_omp (OpenMP tasks, multi-thread)"),
}

# Lines per (variant, m) get the same color/marker but different sizes.
MARKER_SIZE_BY_M = {
    1024:  80,
    4096: 140,
    8192: 220,
}
DEFAULT_MARKER_SIZE = 120

L3_LINE_BYTES = 64           # cache line size on Zen 2 for LLC traffic
DEFAULT_PEAK_GHZ = 4.0       # used only as a label; the y values come
                              # from measured fp_ops / runtime_ns


# ------------------------------------------------------------------ #
# Perf parser (mirror of consolidate_perf_zen2.py for self-contained  #
# operation: this script can run standalone without re-consolidating) #
# ------------------------------------------------------------------ #

def parse_perf_file(path: Path) -> tuple[dict[str, float], float | None]:
    """Return ({event: count}, runtime_ns) for one perf -x , output.

    runtime_ns is taken from the first event row (all events agree
    when multiplexing is 100%, which we engineered in Prompt 7).
    Returns ({}, None) if the file is absent.
    """
    counts: dict[str, float] = {}
    runtime_ns: float | None = None
    if not path.is_file():
        return counts, None
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            count_str, _unit, event, rt_str = parts[0], parts[1], parts[2], parts[3]
            if count_str in ("<not counted>", "<not supported>"):
                continue
            try:
                counts[event] = float(count_str)
                if runtime_ns is None and rt_str:
                    runtime_ns = float(rt_str)
            except ValueError:
                continue
    return counts, runtime_ns


def load_cell(results_dir: Path, variant: str, m: int) -> dict | None:
    """Load both A and B perf groups for one (variant, m) cell. Returns
    None if the A file is missing (without A we have no fp_ops)."""
    path_a = results_dir / f"perf_{variant}_m{m}_A.txt"
    path_b = results_dir / f"perf_{variant}_m{m}_B.txt"
    counts_a, runtime_a = parse_perf_file(path_a)
    counts_b, runtime_b = parse_perf_file(path_b)

    if not counts_a:
        return None

    fp_ops    = counts_a.get("fp_ret_sse_avx_ops.all")
    cycles    = counts_a.get("cycles")
    cache_misses = counts_b.get("cache-misses")

    if fp_ops is None or runtime_a is None or runtime_a <= 0:
        return None

    achieved_gflops = fp_ops / runtime_a   # fp_ops / (runtime_ns / 1e9 * 1e9)
                                          # the two 1e9s cancel.

    if cache_misses is not None and cache_misses > 0:
        effective_ai = fp_ops / (cache_misses * L3_LINE_BYTES)
    else:
        effective_ai = math.nan

    # Theoretical AI: matmul reads A once and streams 2 buffers of B
    # plus writes C: bytes = 4 * (m^2 + 2*m*n) per iteration. flops =
    # 2*m^2*n per iteration. n is hardcoded at 128 in the bench drivers.
    n = 128.0
    theoretical_ai = (2.0 * m * m * n) / (4.0 * (m * m + 2.0 * m * n))

    return {
        "variant": variant,
        "m": m,
        "fp_ops": fp_ops,
        "cache_misses": (cache_misses
                         if cache_misses is not None else math.nan),
        "cycles": cycles if cycles is not None else math.nan,
        "runtime_s": runtime_a / 1.0e9,
        "achieved_gflops": achieved_gflops,
        "effective_ai": effective_ai,
        "theoretical_ai": theoretical_ai,
    }


def discover_cells(results_dir: Path) -> list[tuple[str, int]]:
    """Discover (variant, m) cells from filenames perf_<variant>_m<M>_A.txt.
    The match is greedy on variant so that 'morton_avx2' wins over
    'morton' for the same filename prefix.
    """
    pattern = re.compile(r"^perf_(?P<variant>[a-z0-9_]+?)_m(?P<m>\d+)_A\.txt$")
    cells: list[tuple[str, int]] = []
    for path in sorted(results_dir.glob("perf_*_A.txt")):
        match = pattern.match(path.name)
        if match:
            variant = match.group("variant")
            # Map any unknown variant to the closest known one (defensive).
            if variant not in VARIANTS:
                continue
            cells.append((variant, int(match.group("m"))))
    return cells


# ------------------------------------------------------------------ #
# STREAM parser                                                       #
# ------------------------------------------------------------------ #

# STREAM "best rate" lines look like:
#   Function    Best Rate MB/s  Avg time     Min time     Max time
#   Copy:           18438.7     0.069456     0.069416     0.069569
STREAM_LINE = re.compile(
    r"^(Copy|Scale|Add|Triad):\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)


def parse_stream(path: Path) -> dict[str, float] | None:
    """Return {function: best_rate_GBs} (Copy/Scale/Add/Triad). MB/s
    from STREAM is reported in 1024^2 units; we convert to 10^9
    (GB/s) explicitly to be unambiguous: GB/s = MB/s * 1024^2 / 1e9.
    Returns None if the file is missing."""
    if not path.is_file():
        return None
    out: dict[str, float] = {}
    with path.open("r") as fh:
        for line in fh:
            m = STREAM_LINE.match(line.strip())
            if m:
                func = m.group(1)
                rate_mbs = float(m.group(2))
                out[func] = rate_mbs * (1024 ** 2) / 1.0e9
    return out


# ------------------------------------------------------------------ #
# Plot                                                                #
# ------------------------------------------------------------------ #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path,
                   default=Path("plots/roofline_4600h.png"))
    p.add_argument("--cpu-label",
                   default="AMD Ryzen 5 4600H (Renoir, Zen 2)")
    p.add_argument("--peak-single-gflops", type=float, default=128.0,
                   help="theoretical single-core FMA peak in GFLOPS "
                        "(default: %(default)s = 2 FMA x 8 FP32 lanes x 4 GHz)")
    p.add_argument("--peak-multi-gflops", type=float, default=768.0,
                   help="theoretical 6-core FMA peak in GFLOPS "
                        "(default: %(default)s = 6 cores x 128 GFLOPS)")
    p.add_argument("--threads-multi", type=int, default=6,
                   help="suffix of the multi-thread STREAM file "
                        "(default: %(default)s -> stream_6t.txt)")
    p.add_argument("--use-theoretical-ai", action="store_true",
                   help="plot points at theoretical AI (algorithmic) "
                        "instead of effective AI (measured DRAM traffic). "
                        "Useful when cache-misses is missing or noisy.")
    args = p.parse_args()

    cells = discover_cells(args.results_dir)
    if not cells:
        print(f"Error: no perf_*_A.txt files in {args.results_dir}.", file=sys.stderr)
        print("Run 'make profile_zen2' first.", file=sys.stderr)
        return 1

    points = []
    for variant, m in cells:
        cell = load_cell(args.results_dir, variant, m)
        if cell is not None:
            points.append(cell)

    if not points:
        print("Error: no usable perf cells found.", file=sys.stderr)
        return 1

    stream_1t = parse_stream(args.results_dir / "stream_1t.txt")
    stream_mt = parse_stream(args.results_dir / f"stream_{args.threads_multi}t.txt")

    if stream_1t is None:
        print(f"Warning: results/stream_1t.txt missing; "
              "single-thread bandwidth diagonal will be omitted.",
              file=sys.stderr)
    if stream_mt is None:
        print(f"Warning: results/stream_{args.threads_multi}t.txt missing; "
              "multi-thread bandwidth diagonal will be omitted.",
              file=sys.stderr)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        print("Error: matplotlib not installed. "
              "source ~/venvs/matmul/bin/activate", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.0, 7.5))

    # --- Roofline ceilings -------------------------------------------------

    # x-range covers the typical matmul AI region. Effective AIs from
    # perf often land in 1-100 flops/byte for poorly-blocked variants
    # and 100-2000 for well-blocked ones.
    x_min, x_max = 0.5, 5000.0

    # Compute ceilings (horizontal).
    ax.axhline(args.peak_single_gflops, color="black", linestyle=":",
               linewidth=1.4, alpha=0.85,
               label=f"single-core FMA peak ({args.peak_single_gflops:.0f} GFLOPS)")
    ax.axhline(args.peak_multi_gflops,  color="dimgray", linestyle="--",
               linewidth=1.4, alpha=0.85,
               label=f"6-core FMA peak ({args.peak_multi_gflops:.0f} GFLOPS)")

    # Bandwidth ceilings (diagonal: gflops = bw_GBs * AI).
    def add_bw_line(bw_gbs: float, label: str, color: str, ls: str):
        xs = np.geomspace(x_min, x_max, 200)
        ys = bw_gbs * xs
        # Clip the line to the relevant compute ceiling so it does not
        # paint over the whole plot.
        ceiling = max(args.peak_single_gflops, args.peak_multi_gflops) * 1.5
        ys = np.minimum(ys, ceiling)
        ax.plot(xs, ys, linestyle=ls, color=color, linewidth=1.4, alpha=0.85,
                label=label)

    if stream_1t and "Triad" in stream_1t:
        bw_1t = stream_1t["Triad"]
        add_bw_line(bw_1t, f"STREAM Triad 1T ({bw_1t:.1f} GB/s)",
                    color="tab:cyan", ls="-.")
        ridge_1t = args.peak_single_gflops / bw_1t
        ax.scatter([ridge_1t], [args.peak_single_gflops],
                   marker="x", color="tab:cyan", s=90, zorder=5)
        ax.annotate(f"  ridge 1T\n  AI={ridge_1t:.1f}",
                    (ridge_1t, args.peak_single_gflops),
                    textcoords="offset points", xytext=(6, -16),
                    fontsize=8, color="tab:cyan")

    if stream_mt and "Triad" in stream_mt:
        bw_mt = stream_mt["Triad"]
        add_bw_line(bw_mt, f"STREAM Triad {args.threads_multi}T ({bw_mt:.1f} GB/s)",
                    color="tab:brown", ls=(0, (3, 1, 1, 1)))
        ridge_mt = args.peak_multi_gflops / bw_mt
        ax.scatter([ridge_mt], [args.peak_multi_gflops],
                   marker="x", color="tab:brown", s=90, zorder=5)
        ax.annotate(f"  ridge {args.threads_multi}T\n  AI={ridge_mt:.1f}",
                    (ridge_mt, args.peak_multi_gflops),
                    textcoords="offset points", xytext=(6, -16),
                    fontsize=8, color="tab:brown")

    # --- Variant points ----------------------------------------------------

    plotted_variants: set[str] = set()
    for variant in VARIANTS:
        for m in sorted({p["m"] for p in points}):
            cell = next((p for p in points
                         if p["variant"] == variant and p["m"] == m), None)
            if cell is None:
                continue
            ai = cell["theoretical_ai"] if args.use_theoretical_ai \
                                       else cell["effective_ai"]
            if not (isinstance(ai, float) and math.isfinite(ai)) or ai <= 0:
                ai = cell["theoretical_ai"]   # fall back if effective is nan
            gflops = cell["achieved_gflops"]
            if not (isinstance(gflops, float) and math.isfinite(gflops)) \
                    or gflops <= 0:
                continue
            st = STYLE.get(variant, dict(color="gray", marker="o",
                                          label=variant))
            size = MARKER_SIZE_BY_M.get(m, DEFAULT_MARKER_SIZE)
            label = st["label"] if variant not in plotted_variants else None
            ax.scatter([ai], [gflops], s=size,
                       color=st["color"], marker=st["marker"],
                       edgecolors="black", linewidths=0.7,
                       zorder=6, label=label)
            plotted_variants.add(variant)
            ax.annotate(f"  m={m}", (ai, gflops),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7, color=st["color"])

    # --- Axes, legend, title ----------------------------------------------

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.1, max(args.peak_single_gflops, args.peak_multi_gflops) * 1.5)
    ax.set_xlabel("Arithmetic intensity (flops / byte)  ["
                  + ("theoretical" if args.use_theoretical_ai
                     else "effective: fp_ops / (cache-misses * 64 B)")
                  + "]")
    ax.set_ylabel("Achieved GFLOPS (= fp_ops / runtime)")
    ax.set_title("Roofline final - Sesion 03\n" + args.cpu_label,
                 fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}")

    # --- Stdout summary ----------------------------------------------------

    print()
    print(f"{'variant':<14} {'m':>6} {'gflops':>9} "
          f"{'eff_AI':>10} {'theo_AI':>9} {'time_s':>9}")
    for cell in sorted(points, key=lambda c: (VARIANTS.index(c["variant"])
                                              if c["variant"] in VARIANTS
                                              else 99,
                                              c["m"])):
        print(f"{cell['variant']:<14} {cell['m']:>6} "
              f"{cell['achieved_gflops']:>9.2f} "
              f"{cell['effective_ai']:>10.2f} "
              f"{cell['theoretical_ai']:>9.2f} "
              f"{cell['runtime_s']:>9.4f}")
    print()

    if stream_1t:
        print(f"STREAM 1T  : Copy={stream_1t.get('Copy', 0):.1f} GB/s  "
              f"Scale={stream_1t.get('Scale', 0):.1f}  "
              f"Add={stream_1t.get('Add', 0):.1f}  "
              f"Triad={stream_1t.get('Triad', 0):.1f}")
    if stream_mt:
        print(f"STREAM {args.threads_multi}T  : "
              f"Copy={stream_mt.get('Copy', 0):.1f} GB/s  "
              f"Scale={stream_mt.get('Scale', 0):.1f}  "
              f"Add={stream_mt.get('Add', 0):.1f}  "
              f"Triad={stream_mt.get('Triad', 0):.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
