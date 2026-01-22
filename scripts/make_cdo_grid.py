#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate a CDO grid description from a NetCDF file")
    ap.add_argument("--source", required=True, help="NetCDF file with target grid")
    ap.add_argument("--out", required=True, help="Output grid file path")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-cdo", action="store_true", help="Skip cdo griddes and use xarray fallback")
    return ap.parse_args()


def write_lonlat_grid(ds: xr.Dataset, out_path: Path) -> None:
    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise SystemExit("No lon/lat coordinates found for lonlat grid")

    lon = ds.coords["lon"]
    lat = ds.coords["lat"]
    if lon.ndim != 1 or lat.ndim != 1:
        raise SystemExit("Fallback only supports 1D lon/lat grids")

    if lon.size < 2 or lat.size < 2:
        raise SystemExit("lon/lat must have at least 2 points")

    xinc = float(np.mean(np.diff(lon.values)))
    yinc = float(np.mean(np.diff(lat.values)))

    content = (
        f"gridtype = lonlat\n"
        f"xsize    = {lon.size}\n"
        f"ysize    = {lat.size}\n"
        f"xfirst   = {float(lon.values[0])}\n"
        f"xinc     = {xinc}\n"
        f"yfirst   = {float(lat.values[0])}\n"
        f"yinc     = {yinc}\n"
    )
    out_path.write_text(content)


def main() -> None:
    args = parse_args()
    src = Path(args.source)
    out = Path(args.out)
    if not src.exists():
        raise SystemExit(f"Source file not found: {src}")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists (use --overwrite): {out}")

    if not args.no_cdo:
        try:
            result = subprocess.run(
                ["cdo", "-s", "griddes", str(src)],
                check=True,
                capture_output=True,
                text=True,
            )
            out.write_text(result.stdout)
            print(out)
            return
        except Exception:
            pass

    ds = xr.open_dataset(src)
    write_lonlat_grid(ds, out)
    print(out)


if __name__ == "__main__":
    main()
