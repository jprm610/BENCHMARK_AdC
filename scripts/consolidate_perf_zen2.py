#!/usr/bin/env python3
"""consolidate_perf_zen2.py

Sesion 03 / Prompt 7 - parse the perf stat output files produced by
profile_perf_zen2.sh and emit one row per (variant, m) into
results/perf_zen2_summary.csv.

Input layout per cell (written by profile_perf_zen2.sh):
    results/perf_<variant>_m<M>_A.txt    (group A: compute side)
    results/perf_<variant>_m<M>_B.txt    (group B: memory + TLB side)

Each file is the output of `perf stat -x , -e <events> -o <file>`,
which writes one comment line ("# started on ...") followed by one
line per event in CSV:

    <count>,<unit>,<event_name>,<runtime_ns>,<multiplexing_pct>,<metric>,<metric_unit>

The count is the raw counter value, scaled up to the full run length
if multiplexing < 100%. We surface the worst-case multiplexing
percentage per cell so the consumer can flag low-quality cells.

Output CSV columns (12 rows expected for the default sweep):
    variant, m,
    cycles, instructions, ipc,
    fp_ops, fp_ops_per_cycle,
    l1d_miss_rate,                   l2_request / loads
    l2_load_hit_rate,                hits in L2 / l2_requests
    l3_miss_rate,                    cache-misses (LLC misses) / l2_requests
    tlb_walk_per_kinst,              bp_l1_tlb_miss_l2_tlb_miss * 1000 / inst
    dtlb_load_miss_per_kinst,        dTLB-load-misses * 1000 / inst
    min_mux_pct                      lowest multiplexing pct across all
                                     events of this cell (>= 99 means clean)
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

VARIANTS = ("naive", "recursive", "morton", "morton_avx2")
DEFAULT_MS = (1024, 4096, 8192)


def parse_perf_file(path: Path) -> dict[str, tuple[float, float]]:
    """Return {event_name: (count, multiplexing_pct)} for one perf
    output file. Missing or '<not counted>' events are absent from
    the dict; the caller decides whether the absence is fatal.
    """
    out: dict[str, tuple[float, float]] = {}
    if not path.is_file():
        return out
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            count_str, _unit, event = parts[0], parts[1], parts[2]
            mux_str = parts[4]
            if count_str in ("<not counted>", "<not supported>"):
                continue
            try:
                count = float(count_str)
                mux = float(mux_str) if mux_str else 100.0
            except ValueError:
                continue
            out[event] = (count, mux)
    return out


def safe_div(num: float | None, den: float | None) -> float:
    """Return num/den or NaN if any operand is None / non-finite /
    zero denominator."""
    if num is None or den is None:
        return math.nan
    try:
        if not math.isfinite(num) or not math.isfinite(den) or den == 0:
            return math.nan
        return num / den
    except (TypeError, ZeroDivisionError):
        return math.nan


def get(event_map: dict[str, tuple[float, float]],
        event: str) -> tuple[float | None, float | None]:
    if event not in event_map:
        return None, None
    c, m = event_map[event]
    return c, m


def process_cell(variant: str, m: int, results_dir: Path) -> dict | None:
    """Read both group files for one (variant, m) cell, compute the
    derived metrics, and return one row (a dict). Returns None if
    neither file is present (cell was not profiled)."""
    path_a = results_dir / f"perf_{variant}_m{m}_A.txt"
    path_b = results_dir / f"perf_{variant}_m{m}_B.txt"
    events_a = parse_perf_file(path_a)
    events_b = parse_perf_file(path_b)

    if not events_a and not events_b:
        return None

    # cycles and instructions live in both groups; prefer A if present.
    cycles_a, mux_cycles_a = get(events_a, "cycles")
    cycles_b, mux_cycles_b = get(events_b, "cycles")
    cycles = cycles_a if cycles_a is not None else cycles_b

    inst_a, mux_inst_a = get(events_a, "instructions")
    inst_b, mux_inst_b = get(events_b, "instructions")
    instructions = inst_a if inst_a is not None else inst_b

    fp_ops, mux_fp = get(events_a, "fp_ret_sse_avx_ops.all")
    loads, mux_loads = get(events_a, "ls_dispatch.ld_dispatch")
    l2_req, mux_l2_req = get(events_a, "l2_request_g1.all_no_prefetch")

    l2_hit, mux_l2_hit = get(events_b, "l2_cache_req_stat.ls_rd_blk_l_hit_x")
    cache_misses, mux_cm = get(events_b, "cache-misses")
    tlb_walks, mux_tw = get(events_b, "bp_l1_tlb_miss_l2_tlb_miss")
    dtlb_miss, mux_dt = get(events_b, "dTLB-load-misses")

    ipc                      = safe_div(instructions, cycles)
    fp_ops_per_cycle         = safe_div(fp_ops, cycles)
    l1d_miss_rate            = safe_div(l2_req, loads)
    l2_load_hit_rate         = safe_div(l2_hit, l2_req)
    l3_miss_rate             = safe_div(cache_misses, l2_req)
    tlb_walk_per_kinst       = (safe_div(tlb_walks, instructions) * 1000.0
                                if instructions else math.nan)
    dtlb_load_miss_per_kinst = (safe_div(dtlb_miss, instructions) * 1000.0
                                if instructions else math.nan)

    mux_values = [v for v in (mux_cycles_a, mux_cycles_b,
                              mux_inst_a, mux_inst_b,
                              mux_fp, mux_loads, mux_l2_req, mux_l2_hit,
                              mux_cm, mux_tw, mux_dt)
                  if v is not None]
    min_mux_pct = min(mux_values) if mux_values else math.nan

    return {
        "variant": variant,
        "m": m,
        "cycles": cycles if cycles is not None else math.nan,
        "instructions": instructions if instructions is not None else math.nan,
        "ipc": ipc,
        "fp_ops": fp_ops if fp_ops is not None else math.nan,
        "fp_ops_per_cycle": fp_ops_per_cycle,
        "l1d_miss_rate": l1d_miss_rate,
        "l2_load_hit_rate": l2_load_hit_rate,
        "l3_miss_rate": l3_miss_rate,
        "tlb_walk_per_kinst": tlb_walk_per_kinst,
        "dtlb_load_miss_per_kinst": dtlb_load_miss_per_kinst,
        "min_mux_pct": min_mux_pct,
    }


def discover_cells(results_dir: Path) -> list[tuple[str, int]]:
    """Discover (variant, m) cells from filenames matching
    perf_<variant>_m<M>_A.txt. Used as a fallback when --variants /
    --ms are not provided."""
    pattern = re.compile(r"^perf_(?P<variant>[a-z0-9_]+)_m(?P<m>\d+)_A\.txt$")
    cells: list[tuple[str, int]] = []
    for path in sorted(results_dir.glob("perf_*_A.txt")):
        match = pattern.match(path.name)
        if match:
            cells.append((match.group("variant"), int(match.group("m"))))
    return cells


def fmt(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path,
                   default=Path("results/perf_zen2_summary.csv"))
    p.add_argument("--variants", nargs="*", default=None,
                   help="variants to consolidate "
                        "(default: discover from filenames)")
    p.add_argument("--ms", nargs="*", type=int, default=None,
                   help="m values to consolidate "
                        "(default: discover from filenames)")
    args = p.parse_args()

    if args.variants is not None and args.ms is not None:
        cells = [(v, m) for v in args.variants for m in args.ms]
    else:
        cells = discover_cells(args.results_dir)
        if not cells:
            print(f"Error: no perf_*_A.txt files found in {args.results_dir}. "
                  "Run scripts/run_perf_zen2_sweep.sh first.",
                  file=sys.stderr)
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "variant", "m",
        "cycles", "instructions", "ipc",
        "fp_ops", "fp_ops_per_cycle",
        "l1d_miss_rate", "l2_load_hit_rate", "l3_miss_rate",
        "tlb_walk_per_kinst", "dtlb_load_miss_per_kinst",
        "min_mux_pct",
    ]

    written = 0
    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for variant, m in cells:
            row = process_cell(variant, m, args.results_dir)
            if row is None:
                print(f"Warning: no perf files for ({variant}, m={m}); "
                      "skipping.", file=sys.stderr)
                continue
            writer.writerow([fmt(row[col]) for col in columns])
            written += 1

    print(f"Wrote {args.out} ({written} rows)")
    if written == 0:
        return 1

    # Echo the table to stdout so it appears in CI logs.
    print()
    print(args.out.read_text(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
