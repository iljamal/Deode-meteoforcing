#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import xarray as xr
import yaml
import matplotlib.pyplot as plt

from deode_meteo.process import ensure_time_first


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sanity plots for DEODE forcing variables")
    ap.add_argument("--run-dir", required=True, help="Run directory with forcing files")
    ap.add_argument(
        "--date",
        action="append",
        default=[],
        help="YYYYMMDD (repeat or comma-separated); default: all dates found",
    )
    ap.add_argument("--locations", default="config/plot_locations.yaml")
    ap.add_argument(
        "--map-steps",
        default="0",
        help="Comma-separated time indices or 'all'",
    )
    ap.add_argument("--vars", default=None, help="Comma-separated variable names to plot")
    ap.add_argument("--maps-only", action="store_true")
    ap.add_argument("--timeseries-only", action="store_true")
    ap.add_argument("--merged-only", action="store_true", help="Require merged daily file")
    ap.add_argument("--no-merged", action="store_true", help="Ignore merged files even if present")
    ap.add_argument("--quiver-stride", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=150)
    return ap.parse_args()


def load_locations(path: Path) -> List[dict]:
    data = yaml.safe_load(path.read_text())
    return list(data.get("locations", []))


def parse_dates(args_dates: List[str], run_dir: Path) -> List[str]:
    dates: List[str] = []
    for item in args_dates:
        dates.extend([d.strip() for d in item.split(",") if d.strip()])
    if dates:
        return sorted(set(dates))

    files = run_dir.glob("FORCE_deode_*_y????????.nc")
    found = set()
    for path in files:
        name = path.name
        if "_y" in name and name.endswith(".nc"):
            date = name.split("_y")[-1].replace("m", "").replace("d", "").replace(".nc", "")
            if len(date) == 8:
                found.add(date)
    return sorted(found)


def find_var_sources(
    run_dir: Path,
    date: str,
    prefer_merged: bool,
    merged_only: bool,
) -> Tuple[Dict[str, Path], Path | None]:
    yyyy, mm, dd = date[:4], date[4:6], date[6:8]
    merged = run_dir / f"FORCE_deode_y{yyyy}m{mm}d{dd}.nc"
    if merged.exists() and prefer_merged:
        return {}, merged
    if merged_only:
        raise SystemExit(f"Merged file missing for {date}: {merged}")

    pattern = f"FORCE_deode_*_y{yyyy}m{mm}d{dd}.nc"
    var_to_file: Dict[str, Path] = {}
    for path in sorted(run_dir.glob(pattern)):
        if path.name == merged.name:
            continue
        with xr.open_dataset(path) as ds:
            for var in ds.data_vars:
                var_to_file[var] = path
    return var_to_file, None


def get_var_da(
    var: str,
    var_to_file: Dict[str, Path],
    merged: xr.Dataset | None,
) -> xr.DataArray:
    if merged is not None:
        return merged[var]

    path = var_to_file[var]
    with xr.open_dataset(path) as ds:
        da = ds[var].load()
    return da


def parse_map_steps(value: str) -> List[int] | None:
    if value.strip().lower() in {"all", "all_steps", "allsteps"}:
        return None
    return [int(x) for x in value.split(",") if x.strip()]


def get_lon_lat(da: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    if "lon" in da.coords and "lat" in da.coords:
        lon = da.coords["lon"].values
        lat = da.coords["lat"].values
        return lon, lat
    raise ValueError("lon/lat coordinates not found in dataset")


def select_point(da: xr.DataArray, lat: float, lon: float) -> xr.DataArray:
    if "lon" not in da.coords or "lat" not in da.coords:
        raise ValueError("lon/lat coordinates not found for point selection")

    lon_c = da.coords["lon"]
    lat_c = da.coords["lat"]

    if lon_c.ndim == 1 and lat_c.ndim == 1:
        return da.sel(lon=lon, lat=lat, method="nearest")

    lon2 = lon_c.values
    lat2 = lat_c.values
    dist2 = (lon2 - lon) ** 2 + (lat2 - lat) ** 2
    idx = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    dim0, dim1 = lon_c.dims
    return da.isel({dim0: idx[0], dim1: idx[1]})


def plot_map(
    da: xr.DataArray,
    out_path: Path,
    title: str,
    step_index: int,
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da_step = da.isel(time=step_index).load()
    lon, lat = get_lon_lat(da_step)

    plt.figure(figsize=(8, 6))
    if da_step.ndim == 1:
        plt.scatter(lon, lat, c=da_step.values, s=1, marker=".", cmap="viridis")
    else:
        plt.pcolormesh(lon, lat, da_step.values, shading="auto")
    plt.colorbar(label=da_step.attrs.get("units", ""))
    plt.title(title)
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_wind_map(
    u: xr.DataArray,
    v: xr.DataArray,
    out_path: Path,
    step_index: int,
    stride: int,
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u_step = u.isel(time=step_index).load()
    v_step = v.isel(time=step_index).load()

    speed = np.sqrt(u_step**2 + v_step**2)
    lon, lat = get_lon_lat(u_step)

    plt.figure(figsize=(8, 6))
    if speed.ndim == 1:
        plt.scatter(lon, lat, c=speed.values, s=1, marker=".", cmap="viridis")
    else:
        plt.pcolormesh(lon, lat, speed.values, shading="auto")
    plt.colorbar(label="m s-1")

    if lon.ndim == 1 and lat.ndim == 1:
        lon2, lat2 = np.meshgrid(lon, lat)
    else:
        lon2, lat2 = lon, lat

    if u_step.ndim == 1:
        plt.quiver(
            lon[::stride],
            lat[::stride],
            u_step.values[::stride],
            v_step.values[::stride],
            scale=500,
        )
    else:
        plt.quiver(
            lon2[::stride, ::stride],
            lat2[::stride, ::stride],
            u_step.values[::stride, ::stride],
            v_step.values[::stride, ::stride],
            scale=500,
        )
    plt.title("Wind speed + vectors")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_timeseries(
    da: xr.DataArray,
    locations: List[dict],
    out_path: Path,
    title: str,
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4))

    for loc in locations:
        name = loc.get("name", "loc")
        lat = float(loc["lat"])
        lon = float(loc["lon"])
        series = select_point(da, lat, lon).load()
        plt.plot(series["time"].values, series.values, label=name)

    plt.title(title)
    plt.xlabel("time")
    plt.ylabel(da.attrs.get("units", ""))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    if args.maps_only and args.timeseries_only:
        raise SystemExit("Choose only one of --maps-only or --timeseries-only")

    map_steps = parse_map_steps(args.map_steps)
    locations = load_locations(Path(args.locations))
    dates = parse_dates(args.date, run_dir)
    if not dates:
        raise SystemExit("No dates found to plot")

    for date in dates:
        var_to_file, merged_path = find_var_sources(
            run_dir,
            date,
            prefer_merged=not args.no_merged,
            merged_only=args.merged_only,
        )
        merged = xr.open_dataset(merged_path) if merged_path else None
        if merged is not None:
            merged = ensure_time_first(merged)
            vars_available = list(merged.data_vars)
        else:
            vars_available = list(var_to_file.keys())

        if args.vars:
            wanted = [v.strip() for v in args.vars.split(",") if v.strip()]
            vars_available = [v for v in vars_available if v in wanted]

        if not vars_available:
            if merged is not None:
                merged.close()
            print(f"[WARN] No variables found for date {date}")
            continue

        yyyy, mm, dd = date[:4], date[4:6], date[6:8]
        plots_dir = run_dir / "plots"
        maps_dir = plots_dir / "maps"
        ts_dir = plots_dir / "timeseries"

        if map_steps is None:
            if merged is not None:
                n_time = merged.sizes.get("time", 0)
            else:
                first_var = vars_available[0]
                with xr.open_dataset(var_to_file[first_var]) as ds_first:
                    n_time = ds_first.sizes.get("time", 0)
            if n_time == 0:
                raise SystemExit(f"No time dimension found for {date}")
            steps_for_date = list(range(n_time))
        else:
            steps_for_date = map_steps

        # scalar maps
        if not args.timeseries_only:
            for var in vars_available:
                da = get_var_da(var, var_to_file, merged)
                for step_index in steps_for_date:
                    out = maps_dir / f"{var}_map_{yyyy}{mm}{dd}_t{step_index:02d}.png"
                    title = f"{var} {yyyy}-{mm}-{dd} t={step_index:02d}"
                    plot_map(da, out, title, step_index, args.dpi)

        # wind vector map if both components exist
        if not args.timeseries_only and "u10" in vars_available and "v10" in vars_available:
            u = get_var_da("u10", var_to_file, merged)
            v = get_var_da("v10", var_to_file, merged)
            for step_index in steps_for_date:
                out = maps_dir / f"wind10_vectors_{yyyy}{mm}{dd}_t{step_index:02d}.png"
                plot_wind_map(u, v, out, step_index, args.quiver_stride, args.dpi)

        # timeseries
        if not args.maps_only:
            for var in vars_available:
                da = get_var_da(var, var_to_file, merged)
                out = ts_dir / f"{var}_ts_{yyyy}{mm}{dd}.png"
                title = f"{var} timeseries {yyyy}-{mm}-{dd}"
                plot_timeseries(da, locations, out, title, args.dpi)

            if "u10" in vars_available and "v10" in vars_available:
                u = get_var_da("u10", var_to_file, merged)
                v = get_var_da("v10", var_to_file, merged)
                speed = np.sqrt(u**2 + v**2)
                speed.name = "wind_speed"
                out = ts_dir / f"wind_speed_ts_{yyyy}{mm}{dd}.png"
                title = f"wind speed timeseries {yyyy}-{mm}-{dd}"
                plot_timeseries(speed, locations, out, title, args.dpi)

        if merged is not None:
            merged.close()


if __name__ == "__main__":
    main()
