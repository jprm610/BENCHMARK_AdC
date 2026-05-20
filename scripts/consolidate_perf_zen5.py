#!/usr/bin/env python3
"""consolidate_perf_zen5.py

Parsea los archivos de perf stat producidos por profile_perf_zen5.sh
(servidor AWS c8a.2xlarge / AMD EPYC 9R45 / Zen 5 bajo KVM) y emite
una fila por celda (variant, m) al CSV de salida.

Input por celda:
    results/<variant>/perf_<variant>_m<M>_A.txt   (grupo A: cycles,
        instructions, cache-references, cache-misses)
    results/<variant>/perf_<variant>_m<M>_B.txt   (grupo B: cycles,
        instructions, branch-instructions, branch-misses)
    results/<variant>/bench_<variant>_m<M>.csv     (timing + gflops)

Columnas de salida:
    variant, m,
    median_seconds, gflops,
    cycles, instructions, ipc,
    llc_misses_per_kinst,    cache-misses / instructions * 1000
                             (cache-references no disponible en KVM AMD)
    branch_miss_rate,        branch-misses / branch-instructions
    min_mux_pct
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

CANONICAL_VARIANTS = (
    "naive",
    "morton",
    "morton_avx2",
    "morton_omp",
    "loop_ijk",
    "loop_ikj",
    "loop_jik",
    "loop_jki",
    "loop_kij",
    "loop_kji",
    "tiled_ikj",
    "tiled_ikj_avx2",
    "tiled_ikj_omp",
)


def parse_perf_file(path: Path) -> dict[str, tuple[float, float]]:
    """Retorna {nombre_evento: (count, mux_pct)} para un archivo perf stat -x ,."""
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
            count_str = parts[0]
            event = parts[2].split(":")[0]
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


def parse_bench_file(path: Path) -> tuple[float, float]:
    """Retorna (median_seconds, gflops) del CSV del bench, o (nan, nan)."""
    if not path.is_file():
        return math.nan, math.nan
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            try:
                # tiled_ikj_avx2: variant,m,n,num_iters,bs,median_seconds,gflops (7 cols)
                # loop/tiled_ikj: variant,m,n,num_iters,median_seconds,gflops    (6 cols)
                # naive/morton:   m,n,num_iters,median_seconds,gflops            (5 cols)
                if len(parts) >= 7:
                    return float(parts[5]), float(parts[6])
                elif len(parts) >= 6:
                    return float(parts[4]), float(parts[5])
                else:
                    return float(parts[3]), float(parts[4])
            except (ValueError, IndexError):
                return math.nan, math.nan
    return math.nan, math.nan


def process_cell(variant: str, m: int, results_dir: Path) -> dict | None:
    path_a = results_dir / variant / f"perf_{variant}_m{m}_A.txt"
    path_b = results_dir / variant / f"perf_{variant}_m{m}_B.txt"
    events_a = parse_perf_file(path_a)
    events_b = parse_perf_file(path_b)

    if not events_a and not events_b:
        return None

    bench_path = results_dir / variant / f"bench_{variant}_m{m}.csv"
    median_seconds, gflops = parse_bench_file(bench_path)

    # Grupo A: compute y LLC
    cycles,       mux_cycles = get(events_a, "cycles")
    instructions, mux_inst   = get(events_a, "instructions")
    cache_misses, mux_cm     = get(events_a, "cache-misses")

    # Grupo B: branches (cycles e instructions duplicados para cross-check)
    branch_insts,  mux_bi = get(events_b, "branch-instructions")
    branch_misses, mux_bm = get(events_b, "branch-misses")

    cyc = cycles        if cycles        is not None else None
    ins = instructions  if instructions  is not None else None

    ipc                  = safe_div(ins, cyc)
    llc_misses_per_kinst = (safe_div(cache_misses, ins) * 1000.0
                            if ins else math.nan)
    branch_miss_rate     = safe_div(branch_misses, branch_insts)

    mux_values = [v for v in (mux_cycles, mux_inst, mux_cm, mux_bi, mux_bm)
                  if v is not None]
    min_mux_pct = min(mux_values) if mux_values else math.nan

    return {
        "variant":             variant,
        "m":                   m,
        "median_seconds":      median_seconds,
        "gflops":              gflops,
        "cycles":              cycles       if cycles       is not None else math.nan,
        "instructions":        instructions if instructions is not None else math.nan,
        "ipc":                 ipc,
        "llc_misses_per_kinst": llc_misses_per_kinst,
        "branch_miss_rate":    branch_miss_rate,
        "min_mux_pct":         min_mux_pct,
    }


def discover_cells(results_dir: Path) -> list[tuple[str, int]]:
    pattern = re.compile(r"^perf_(?P<variant>[a-z0-9_]+)_m(?P<m>\d+)_A\.txt$")
    by_variant: dict[str, list[int]] = {}
    for path in results_dir.glob("*/perf_*_A.txt"):
        match = pattern.match(path.name)
        if match:
            variant = match.group("variant")
            m = int(match.group("m"))
            by_variant.setdefault(variant, []).append(m)

    canonical_present = [v for v in CANONICAL_VARIANTS if v in by_variant]
    extras = sorted(v for v in by_variant if v not in CANONICAL_VARIANTS)

    cells: list[tuple[str, int]] = []
    for variant in canonical_present + extras:
        for m in sorted(by_variant[variant]):
            cells.append((variant, m))
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
                   default=Path("results/metrics.csv"))
    p.add_argument("--variants", nargs="*", default=None,
                   help="variantes a consolidar (default: descubrir desde archivos)")
    p.add_argument("--ms", nargs="*", type=int, default=None,
                   help="valores de m a consolidar (default: descubrir desde archivos)")
    args = p.parse_args()

    if args.variants is not None and args.ms is not None:
        cells = [(v, m) for v in args.variants for m in args.ms]
    else:
        cells = discover_cells(args.results_dir)
        if not cells:
            print(f"Error: no se encontraron archivos perf_*_A.txt en {args.results_dir}. "
                  "Ejecutar scripts/run_perf_zen5_sweep.sh primero.",
                  file=sys.stderr)
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "variant", "m",
        "median_seconds", "gflops",
        "cycles", "instructions", "ipc",
        "llc_misses_per_kinst",
        "branch_miss_rate",
        "min_mux_pct",
    ]

    written = 0
    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for variant, m in cells:
            row = process_cell(variant, m, args.results_dir)
            if row is None:
                print(f"Warning: sin archivos perf para ({variant}, m={m}); "
                      "omitiendo.", file=sys.stderr)
                continue
            writer.writerow([fmt(row[col]) for col in columns])
            written += 1

    print(f"Wrote {args.out} ({written} rows)")
    if written == 0:
        return 1

    print()
    print(args.out.read_text(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
