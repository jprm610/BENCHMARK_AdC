#!/usr/bin/env python3
"""plot_sweep_session_03.py

Sesion 03 / Prompt 5 -- visualize the four-variant comparative sweep.

Reads:
    results/session_03_naive.csv
    results/session_03_recursive.csv
    results/session_03_morton.csv
    results/session_03_morton_avx2.csv

Each CSV has the bench-native header
    m,n,num_iters,median_seconds,gflops
i.e. one row per measured (variant, m).

Writes:
    results/session_03_all.csv          consolidated table with leading
                                        'variant' column, used by the
                                        Resultados section in
                                        docs/PLAN_SESION_03.md.
    plots/session_03_gflops_vs_m.png    four curves of GFLOPS vs m,
                                        log2 x axis, cliff annotations
                                        and Roofline guides.
    plots/session_03_speedup_vs_naive.png    three curves (recursive,
                                        morton, morton_avx2) of
                                        gflops_variant / gflops_naive
                                        at the m values common to both.

CLI defaults are calibrated for the Ryzen 5 4600H (Renoir, Zen 2):
    --l3-mib 4         (effective L3 per CCX)
    --peak-fma 128     (single-core FMA peak in GFLOPS)
    --bw-gbs 40        (realistic DDR4-3200 dual-channel bandwidth;
                        STREAM-measured value will replace this in Prompt 8)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


VARIANTS = ("naive", "recursive", "morton", "morton_avx2")

# Visual style per variant. Morton variants use dashed lines because
# their sweep is restricted to powers of two; the dashes signal that
# the curve is not directly comparable point-for-point with naive /
# recursive if those ever pick up non-power-of-two samples.
STYLE = {
    "naive": dict(
        color="tab:blue", marker="o", linestyle="-",
        label="naive (ijk, row-major, -O3 znver2)",
    ),
    "recursive": dict(
        color="tab:green", marker="s", linestyle="-",
        label="recursive (cache-oblivious, row-major)",
    ),
    "morton": dict(
        color="tab:red", marker="D", linestyle="--",
        label="morton (cache-oblivious, Z-order fino)",
    ),
    "morton_avx2": dict(
        color="tab:purple", marker="^", linestyle="-",
        label="morton_avx2 (microkernel 4x16 + Morton de bloques)",
    ),
}


def read_csv(path: Path) -> list[dict] | None:
    """Read one bench CSV into a list of dicts sorted by m. Returns
    None if the file is missing, [] if it exists but has no data rows.
    """
    if not path.is_file():
        return None
    rows: list[dict] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append({
                    "m": int(row["m"]),
                    "n": int(row["n"]),
                    "num_iters": int(row["num_iters"]),
                    "median_seconds": float(row["median_seconds"]),
                    "gflops": float(row["gflops"]),
                })
            except (KeyError, ValueError) as exc:
                print(f"Skipping malformed row {row!r} in {path}: {exc}",
                      file=sys.stderr)
    rows.sort(key=lambda r: r["m"])
    return rows


def write_consolidated_csv(data: dict[str, list[dict]],
                           out_path: Path) -> None:
    """Concatenate the four per-variant CSVs into one with a leading
    'variant' column. Empty/missing variants are skipped silently."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["variant", "m", "n", "num_iters",
                         "median_seconds", "gflops"])
        for variant in VARIANTS:
            for r in data.get(variant) or []:
                writer.writerow([variant, r["m"], r["n"], r["num_iters"],
                                 f"{r['median_seconds']:.6f}",
                                 f"{r['gflops']:.6f}"])
    print(f"Wrote {out_path}")


def m_at_full_a_fits(cache_bytes: float, bytes_per_scalar: int = 4) -> float:
    """Threshold m at which the full m x m matrix A fills the given
    cache level: m = sqrt(cache_bytes / sizeof(scalar))."""
    return math.sqrt(cache_bytes / bytes_per_scalar)


def add_roofline_guides(ax, args, m_min: float, m_max: float) -> None:
    """Draw the two Roofline guides used by Prompt 5: a horizontal line
    at the single-core FMA peak (compute-bound ceiling), and a second
    horizontal line at the memory-bound ceiling estimated from
    bandwidth * arithmetic_intensity. Both are estimates; Prompt 8
    replaces them with STREAM-measured numbers.
    """
    peak = args.peak_fma
    ax.axhline(
        peak, color="black", linestyle=":", linewidth=1, alpha=0.7,
        label=f"single-core FMA peak ({peak:.0f} GFLOPS)",
    )
    # Memory-bound ceiling: gflops <= bandwidth * arithmetic_intensity.
    # For the iterated matmul each iteration does 2*m*m*n flops and
    # transfers approximately bytes_per_scalar * (m*m + 2*m*n) bytes
    # (read A once, stream the two B buffers). At m=n the intensity is
    # ~m/3 flops/byte which is large enough that the memory bound sits
    # well above the compute bound for any m in the sweep; the line
    # gets drawn but is informative only as a sanity reference.
    bw = args.bw_gbs * 1.0e9
    # Use the geometric mean of the swept m as the reference intensity
    # so the drawn line is representative, not the value at the largest m.
    m_ref = math.sqrt(m_min * m_max)
    n = 128.0
    bytes_per_iter = 4.0 * (m_ref * m_ref + 2.0 * m_ref * n)
    flops_per_iter = 2.0 * m_ref * m_ref * n
    intensity = flops_per_iter / bytes_per_iter  # flops / byte
    mem_ceiling = bw * intensity / 1.0e9
    ax.axhline(
        mem_ceiling, color="gray", linestyle="--", linewidth=1, alpha=0.6,
        label=(f"memory-bound estimate "
               f"({args.bw_gbs:.0f} GB/s x {intensity:.1f} flops/B "
               f"= {mem_ceiling:.0f} GFLOPS)"),
    )


def add_cliff_annotations(ax, l3_mib: int) -> None:
    """Vertical guides for the two cliffs that Prompt 5 specifically
    asks the reader to look for: L3 (full A no longer fits) and TLB
    (around m=8192 in practice on the 4600H)."""
    m_l3 = m_at_full_a_fits(l3_mib * 1024 * 1024)
    m_tlb_press = 8192.0
    y_top = ax.get_ylim()[1]

    ax.axvline(m_l3, color="tab:orange", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(
        m_l3, y_top * 0.92,
        f"  cliff L3 ({l3_mib} MiB / 4 B)\n  m ~= {m_l3:.0f}",
        color="tab:orange", rotation=90, ha="left", va="top", fontsize=8,
    )
    ax.axvline(m_tlb_press, color="tab:brown", linestyle=":",
               linewidth=1, alpha=0.7)
    ax.text(
        m_tlb_press, y_top * 0.92,
        f"  TLB pressure\n  m = {int(m_tlb_press)}",
        color="tab:brown", rotation=90, ha="left", va="top", fontsize=8,
    )


def plot_gflops_vs_m(data: dict[str, list[dict]],
                     args, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    all_m: list[int] = []
    for variant in VARIANTS:
        rows = data.get(variant) or []
        if not rows:
            continue
        ms = np.array([r["m"] for r in rows], dtype=float)
        gflops = np.array([r["gflops"] for r in rows], dtype=float)
        ax.plot(ms, gflops, linewidth=1.8, markersize=6, **STYLE[variant])
        all_m.extend(int(m) for m in ms)

    if not all_m:
        print("Error: no data to plot.", file=sys.stderr)
        sys.exit(1)

    m_min, m_max = float(min(all_m)), float(max(all_m))

    ax.set_xscale("log", base=2)
    ax.set_xlabel("m (log2 scale)")
    ax.set_ylabel("Sustained GFLOPS")
    ax.set_title(
        "Sesion 03 sweep: naive vs recursive vs morton vs morton_avx2\n"
        f"{args.cpu_label}",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.3)

    # Add ceilings before annotations so the y limits used by the
    # annotation text are the final ones.
    add_roofline_guides(ax, args, m_min, m_max)
    add_cliff_annotations(ax, args.l3_mib)
    ax.legend(loc="best", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_speedup_vs_naive(data: dict[str, list[dict]],
                          args, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    naive_by_m = {r["m"]: r["gflops"] for r in (data.get("naive") or [])}
    if not naive_by_m:
        print("Warning: naive CSV missing or empty; cannot draw speedup plot.",
              file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    for variant in ("recursive", "morton", "morton_avx2"):
        rows = data.get(variant) or []
        pairs: list[tuple[int, float]] = []
        for r in rows:
            if r["m"] in naive_by_m:
                pairs.append((r["m"], r["gflops"] / naive_by_m[r["m"]]))
        if not pairs:
            print(f"Warning: no m overlap between {variant} and naive; "
                  f"variant skipped in speedup plot.", file=sys.stderr)
            continue
        ms, ratios = zip(*pairs)
        style = dict(STYLE[variant])
        style.pop("label", None)
        ax.plot(ms, ratios, linewidth=1.8, markersize=7,
                label=variant, **style)
        for m_v, r_v in pairs:
            ax.annotate(f"{r_v:.2f}", (m_v, r_v),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7,
               label="parity vs naive (1.0)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m (log2 scale)")
    ax.set_ylabel("Speedup = GFLOPS(variant) / GFLOPS(naive)")
    ax.set_title(
        "Sesion 03 speedup vs naive (same -O3 znver2 flags)\n"
        f"{args.cpu_label}",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=Path("results"),
                   help="directory containing session_03_*.csv "
                        "(default: %(default)s)")
    p.add_argument("--out-dir", type=Path, default=Path("plots"),
                   help="directory for the generated PNGs "
                        "(default: %(default)s)")
    p.add_argument("--consolidated-csv", type=Path, default=None,
                   help="path of the consolidated CSV "
                        "(default: <results-dir>/session_03_all.csv)")
    p.add_argument("--cpu-label", default="AMD Ryzen 5 4600H (Renoir, Zen 2)",
                   help="CPU label shown in plot subtitles")
    p.add_argument("--l3-mib", type=int, default=4,
                   help="effective L3 per CCX in MiB "
                        "(default: %(default)s, valid for Renoir 4600H)")
    p.add_argument("--peak-fma", type=float, default=128.0,
                   help="single-core FMA peak in GFLOPS "
                        "(default: %(default)s for 4 GHz Zen 2)")
    p.add_argument("--bw-gbs", type=float, default=40.0,
                   help="realistic DDR4 bandwidth in GB/s used for the "
                        "memory-bound estimate "
                        "(default: %(default)s; STREAM-measured in Prompt 8)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    paths = {variant: args.results_dir / f"session_03_{variant}.csv"
             for variant in VARIANTS}
    data: dict[str, list[dict]] = {}
    missing_any = False
    for variant in VARIANTS:
        rows = read_csv(paths[variant])
        if rows is None:
            print(f"Warning: {paths[variant]} not found; "
                  f"{variant} curve will be omitted.", file=sys.stderr)
            data[variant] = []
            missing_any = True
        else:
            data[variant] = rows

    if all(not data[v] for v in VARIANTS):
        print("Error: no CSV produced any rows. "
              "Run 'make sweep_session_03' first.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    consolidated = args.consolidated_csv or (
        args.results_dir / "session_03_all.csv"
    )
    write_consolidated_csv(data, consolidated)

    plot_gflops_vs_m(data, args, args.out_dir / "session_03_gflops_vs_m.png")
    plot_speedup_vs_naive(data, args,
                          args.out_dir / "session_03_speedup_vs_naive.png")

    if missing_any:
        print("Note: at least one variant was missing; the plots above "
              "show only the variants present.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
