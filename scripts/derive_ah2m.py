#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import xarray as xr

from deode_meteo.process import compute_absolute_humidity, ensure_time_first


DATE_RE = re.compile(r"_y(\d{4})m(\d{2})d(\d{2})\.nc$")


def parse_date(path: Path) -> str | None:
    match = DATE_RE.search(path.name)
    if not match:
        return None
    return f"{match.group(1)}{match.group(2)}{match.group(3)}"


def select_var(ds: xr.Dataset, preferred: list[str]) -> str:
    for name in preferred:
        if name in ds.data_vars:
            return name
    raise KeyError(f"None of {preferred} found in dataset variables: {list(ds.data_vars)}")


def build_pairs(run_dir: Path) -> dict[str, tuple[Path, Path]]:
    t2_files = {parse_date(p): p for p in run_dir.glob("FORCE_deode_t2_y*.nc")}
    rh_files = {parse_date(p): p for p in run_dir.glob("FORCE_deode_rh2m_y*.nc")}

    pairs = {}
    for date, t2_path in t2_files.items():
        if not date:
            continue
        rh_path = rh_files.get(date)
        if rh_path:
            pairs[date] = (t2_path, rh_path)
    return pairs


def derive_ah2m(t2_path: Path, rh_path: Path, out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        return

    with xr.open_dataset(t2_path) as ds_t2, xr.open_dataset(rh_path) as ds_rh:
        t2_name = select_var(ds_t2, ["t2", "2t"])
        rh_name = select_var(ds_rh, ["rh2m", "2r", "rh"])
        t2, rh = xr.align(ds_t2[t2_name], ds_rh[rh_name], join="exact")
        ah = compute_absolute_humidity(t2, rh)
        ds_out = xr.Dataset({"ah2m": ah})
        ds_out = ensure_time_first(ds_out)
        ds_out.to_netcdf(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive absolute humidity from t2 and rh2m forcing files")
    ap.add_argument("--run-dir", default="forcing/meteo/DEODE_fc/2024/11/19/00")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    pairs = build_pairs(run_dir)
    t2_dates = {parse_date(p) for p in run_dir.glob("FORCE_deode_t2_y*.nc")}
    rh_dates = {parse_date(p) for p in run_dir.glob("FORCE_deode_rh2m_y*.nc")}
    t2_dates.discard(None)
    rh_dates.discard(None)

    missing_rh = sorted(t2_dates - rh_dates)
    missing_t2 = sorted(rh_dates - t2_dates)

    if missing_rh:
        print(f"[WARN] Missing rh2m files for dates: {', '.join(missing_rh)}")
    if missing_t2:
        print(f"[WARN] Missing t2 files for dates: {', '.join(missing_t2)}")

    if not pairs:
        raise SystemExit(f"No matching t2/rh2m pairs found in {run_dir}")

    for date, (t2_path, rh_path) in sorted(pairs.items()):
        out_path = run_dir / f"FORCE_deode_ah2m_y{date[:4]}m{date[4:6]}d{date[6:8]}.nc"
        derive_ah2m(t2_path, rh_path, out_path, overwrite=args.overwrite)
        print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
