#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from deode_meteo.process import ensure_time_first


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge per-parameter DEODE forcing files for one date")
    ap.add_argument("--run-dir", required=True, help="Run directory with per-parameter files")
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--prefix", default="FORCE_deode")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    date = args.date
    yyyy, mm, dd = date[:4], date[4:6], date[6:8]
    out_name = f"{args.prefix}_y{yyyy}m{mm}d{dd}.nc"
    out_path = run_dir / out_name

    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists (use --overwrite): {out_path}")

    pattern = f"{args.prefix}_*_y{yyyy}m{mm}d{dd}.nc"
    files = sorted(run_dir.glob(pattern))
    files = [p for p in files if p.name != out_name]

    if not files:
        raise SystemExit(f"No parameter files found for {date} in {run_dir}")

    datasets = []
    for path in files:
        ds = xr.open_dataset(path)
        ds = ensure_time_first(ds)
        datasets.append(ds)

    merged = xr.merge(datasets, compat="no_conflicts")
    merged.to_netcdf(out_path)

    for ds in datasets:
        ds.close()

    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
