from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import xarray as xr

from .config import Config, ParamGroup, group_parameters
from .fetch import fetch_group
from .interpolate import interpolate_cdo, interpolate_python, run_cdo, cdo_griddes
from .process import (
    add_latlon_from_grid_mapping,
    assign_time,
    compute_absolute_humidity,
    convert_accum_to_flux,
    deaccumulate,
    ensure_time_first,
    normalize_time_dim,
    subset_box,
    squeeze_singletons,
    rotate_winds,
)
from .plotting import plot_wind_compare


def run_dirs(cfg: Config, start_dt: datetime) -> Dict[str, Path]:
    output_root = str(cfg.paths.output_root)
    if "{" in output_root and "}" in output_root:
        output_root = output_root.format(
            georef=cfg.dataset.georef,
            date=start_dt.strftime("%Y%m%d"),
            time=start_dt.strftime("%H"),
        )

    if cfg.paths.include_datetime_subdirs:
        run_dir = (
            Path(output_root)
            / f"{start_dt:%Y}"
            / f"{start_dt:%m}"
            / f"{start_dt:%d}"
            / f"{start_dt:%H}"
        )
    else:
        run_dir = Path(output_root)
    return {
        "run": run_dir,
        "grib": run_dir / cfg.paths.grib_subdir,
        "intermediate": run_dir / cfg.paths.intermediate_subdir,
        "weights": run_dir / cfg.paths.weights_subdir,
        "plots": run_dir / cfg.paths.plots_subdir,
    }


def resolve_target_grid(cfg: Config, dirs: Dict[str, Path], grib_path: Path | None = None) -> Path:
    target_spec = cfg.processing.target_grid
    if target_spec is None:
        return cfg.paths.bathy_file

    if isinstance(target_spec, str):
        if target_spec in {"auto", "auto_from_grib"}:
            if grib_path is None:
                raise ValueError("target_grid=auto_from_grib requires a GRIB path")
            out_path = dirs["intermediate"] / "target_grid_from_grib.cdo"
            return cdo_griddes(grib_path, out_path)
        path = Path(target_spec)
        if path.exists():
            return path
        if "gridtype" in target_spec:
            out_path = dirs["intermediate"] / "target_grid.cdo"
            out_path.write_text(target_spec)
            return out_path
        raise ValueError(f"Unknown target_grid string: {target_spec}")

    if isinstance(target_spec, dict):
        if target_spec.get("auto_from_grib") or target_spec.get("auto"):
            if grib_path is None:
                raise ValueError("target_grid=auto_from_grib requires a GRIB path")
            out_path = dirs["intermediate"] / "target_grid_from_grib.cdo"
            return cdo_griddes(grib_path, out_path)
        required = ["gridtype", "xsize", "ysize", "xfirst", "xinc", "yfirst", "yinc"]
        missing = [k for k in required if k not in target_spec]
        if missing:
            raise ValueError(f"target_grid missing keys: {', '.join(missing)}")
        out_path = dirs["intermediate"] / "target_grid.cdo"
        lines = [
            f"gridtype = {target_spec['gridtype']}",
            f"xsize    = {target_spec['xsize']}",
            f"ysize    = {target_spec['ysize']}",
            f"xfirst   = {target_spec['xfirst']}",
            f"xinc     = {target_spec['xinc']}",
            f"yfirst   = {target_spec['yfirst']}",
            f"yinc     = {target_spec['yinc']}",
            "",
        ]
        out_path.write_text("\n".join(lines))
        return out_path

    raise ValueError(f"Unsupported target_grid type: {type(target_spec)}")


def time_hours_for_step(cfg: Config, step_type: str) -> List[int]:
    step = cfg.forecast.step_hours
    max_step = cfg.forecast.max_step
    if step_type == "instant":
        start = 0 if cfg.forecast.include_step0 else step
        return list(range(start, max_step + 1, step))
    if step_type == "accum":
        return list(range(0, max_step + 1, step))
    raise ValueError(f"Unsupported step_type: {step_type}")


def group_contains_wind(group: ParamGroup) -> bool:
    keys = set(group.params.keys())
    return "u10" in keys and "v10" in keys


def rename_to_param_keys(ds: xr.Dataset, group: ParamGroup) -> xr.Dataset:
    mapping = {p.short_name: key for key, p in group.params.items()}
    mapping.update({f"var{p.param_id}": key for key, p in group.params.items()})
    rename = {k: v for k, v in mapping.items() if k in ds}
    return ds.rename(rename) if rename else ds


def process_group_ds(
    ds: xr.Dataset,
    group: ParamGroup,
    cfg: Config,
    start_dt: datetime,
    subset_box_cfg: dict | None = None,
) -> xr.Dataset:
    ds = normalize_time_dim(ds)
    ds = squeeze_singletons(ds)
    ds = rename_to_param_keys(ds, group)
    if "longitude" in ds and "latitude" in ds:
        ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    if subset_box_cfg:
        box = subset_box_cfg
        lon_min = float(box["lon_min"])
        lon_max = float(box["lon_max"])
        lat_min = float(box["lat_min"])
        lat_max = float(box["lat_max"])
        if "lon" not in ds.coords or "lat" not in ds.coords:
            ds = add_latlon_from_grid_mapping(ds)
        ds = subset_box(ds, lon_min, lon_max, lat_min, lat_max)

    if group.step_type == "accum":
        ds = ds.assign_coords(time=np.arange(ds.sizes["time"]))
        if ds.sizes["time"] == cfg.forecast.max_step:
            copy_hour1_to0 = True
        elif ds.sizes["time"] == cfg.forecast.max_step + 1:
            copy_hour1_to0 = False
        else:
            raise ValueError(
                f"Unexpected accum length {ds.sizes['time']} for max_step {cfg.forecast.max_step}"
            )

        updated = {}
        target_len = ds.sizes["time"]
        for key, param in group.params.items():
            if key not in ds:
                continue
            da = deaccumulate(ds[key], copy_hour1_to0=copy_hour1_to0)
            if key in {"ssrd", "strd"}:
                da = da / (cfg.forecast.step_hours * 3600.0)
                da.attrs["units"] = "W m-2"
            da = convert_accum_to_flux(da, cfg.forecast.step_hours, param.units, param.output_units)
            updated[key] = da
            target_len = max(target_len, da.sizes["time"])

        if target_len != ds.sizes["time"]:
            ds = ds.reindex(time=np.arange(target_len))

        for key, da in updated.items():
            ds[key] = da

    hours = time_hours_for_step(cfg, group.step_type)
    ds = assign_time(ds, start_dt, hours)
    ds = ensure_time_first(ds)
    return ds


def load_target_grid(bathy_path: Path) -> xr.Dataset:
    ds = xr.open_dataset(bathy_path)
    if "lon" not in ds or "lat" not in ds:
        raise ValueError("Target grid must have lon/lat coordinates")
    return xr.Dataset({"lon": ds["lon"], "lat": ds["lat"]})


def write_outputs(
    ds: xr.Dataset,
    output_dir: Path,
    prefix: str,
    output_mode: str,
    group_name: str | None = None,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    dates = sorted({str(t)[:10] for t in ds["time"].values.astype("datetime64[ns]")})
    for date in dates:
        yyyy, mm, dd = date.split("-")
        if output_mode == "merged":
            name = f"{prefix}_y{yyyy}m{mm}d{dd}.nc"
        else:
            name = f"{prefix}_{group_name}_y{yyyy}m{mm}d{dd}.nc"

        out_path = output_dir / name
        day_ds = ds.sel(time=date)
        day_ds.to_netcdf(out_path)
        outputs.append(out_path)

    return outputs


def run_pipeline(
    cfg: Config,
    start_dt: datetime,
    live_request: bool = True,
) -> None:
    groups = group_parameters(cfg)
    dirs = run_dirs(cfg, start_dt)

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    date_str = start_dt.strftime("%Y%m%d")
    time_str = start_dt.strftime("%H")

    target_grid = None
    target_grid_path = None
    if cfg.processing.interpolate and cfg.processing.backend == "python":
        target_grid_path = resolve_target_grid(cfg, dirs)
        if target_grid_path.suffix in {".nc", ".nc4", ".cdf"}:
            target_grid = load_target_grid(target_grid_path)
        else:
            target_grid = load_target_grid(cfg.paths.bathy_file)
    processed_groups: Dict[str, xr.Dataset] = {}

    for group in groups.values():
        grib = fetch_group(cfg, group, date_str, time_str, dirs["grib"], live_request=live_request)
        if grib is None:
            print(f"[WARN] Skipping group '{group.name}' (no GRIB)")
            continue
        subset_box_cfg = cfg.processing.subset_box
        if subset_box_cfg and cfg.processing.backend == "cdo":
            lon_min = float(subset_box_cfg["lon_min"])
            lon_max = float(subset_box_cfg["lon_max"])
            lat_min = float(subset_box_cfg["lat_min"])
            lat_max = float(subset_box_cfg["lat_max"])
            subset_grib = dirs["intermediate"] / f"{group.name}_subset_{date_str}T{time_str}.grib2"
            if not subset_grib.exists():
                run_cdo(
                    f"cdo -O sellonlatbox,{lon_min},{lon_max},{lat_min},{lat_max} {grib} {subset_grib}"
                )
            grib = subset_grib
            subset_box_cfg = None

        if not cfg.processing.interpolate:
            ds = xr.open_dataset(grib, engine="cfgrib")
            ds = normalize_time_dim(ds)
            ds = squeeze_singletons(ds)
            ds = rename_to_param_keys(ds, group)

            if cfg.processing.rotate_winds and group_contains_wind(group):
                ds = rotate_winds(ds, "u10", "v10")

        elif cfg.processing.backend == "cdo":
            src_path = grib
            if cfg.processing.rotate_winds and group_contains_wind(group):
                try:
                    ds_wind = xr.open_dataset(grib, engine="cfgrib")
                except Exception as exc:
                    raise RuntimeError("Wind rotation requires cfgrib") from exc
                ds_wind = normalize_time_dim(ds_wind)
                ds_wind = squeeze_singletons(ds_wind)
                ds_wind = rename_to_param_keys(ds_wind, group)
                ds_wind = rotate_winds(ds_wind, "u10", "v10")
                tmp_nc = dirs["intermediate"] / f"{group.name}_rotated.nc"
                ds_wind.to_netcdf(tmp_nc)
                src_path = tmp_nc

            out_nc = dirs["intermediate"] / f"{group.name}_remap.nc"
            if target_grid_path is None:
                target_grid_path = resolve_target_grid(cfg, dirs, grib_path=grib)
            weights_path = dirs["weights"] / f"{group.name}_weights.nc"
            interpolate_cdo(
                src_path,
                target_grid_path,
                out_nc,
                group.interpolation,
                generate_weights=cfg.processing.generate_weights,
                weights_path=weights_path,
            )
            ds = xr.open_dataset(out_nc)

        elif cfg.processing.backend == "python":
            ds = xr.open_dataset(grib, engine="cfgrib")
            ds = normalize_time_dim(ds)
            ds = squeeze_singletons(ds)
            ds = rename_to_param_keys(ds, group)

            if cfg.processing.rotate_winds and group_contains_wind(group):
                ds = rotate_winds(ds, "u10", "v10")

            if "lon" not in ds.coords or "lat" not in ds.coords:
                if "longitude" in ds and "latitude" in ds:
                    ds = ds.rename({"longitude": "lon", "latitude": "lat"})
                else:
                    ds = add_latlon_from_grid_mapping(ds)

            if target_grid is None:
                raise ValueError("Python interpolation requested without a target grid")
            ds = interpolate_python(ds, target_grid, group.interpolation)

        else:
            raise ValueError(f"Unsupported backend: {cfg.processing.backend}")

        ds = process_group_ds(ds, group, cfg, start_dt, subset_box_cfg=subset_box_cfg)
        processed_groups[group.name] = ds

    # merge
    if cfg.processing.output_mode == "merged":
        merged = xr.merge(processed_groups.values())
        if "t2" in merged and "rh2m" in merged:
            merged["ah2m"] = compute_absolute_humidity(merged["t2"], merged["rh2m"])
        merged_out = merged.drop_vars(["u10_grid", "v10_grid"], errors="ignore")
        write_outputs(merged_out, dirs["run"], cfg.processing.output_prefix, "merged")

        if cfg.processing.plots and cfg.processing.rotate_winds and "u10" in merged and "u10_grid" in merged:
            for region in cfg.processing.plot_regions:
                out_path = dirs["plots"] / f"wind_compare_{region.get('name','region')}.png"
                plot_wind_compare(
                    merged,
                    region,
                    out_path,
                    u_name="u10",
                    v_name="v10",
                    step_index=cfg.processing.plot_step_index,
                    stride=cfg.processing.plot_stride,
                )

    else:
        for name, ds in processed_groups.items():
            ds_out = ds.drop_vars(["u10_grid", "v10_grid"], errors="ignore")
            write_outputs(ds_out, dirs["run"], cfg.processing.output_prefix, "per_parameter", group_name=name)

        if cfg.processing.plots and cfg.processing.rotate_winds and "wind10" in processed_groups:
            wind = processed_groups["wind10"]
            for region in cfg.processing.plot_regions:
                out_path = dirs["plots"] / f"wind_compare_{region.get('name','region')}.png"
                plot_wind_compare(
                    wind,
                    region,
                    out_path,
                    u_name="u10",
                    v_name="v10",
                    step_index=cfg.processing.plot_step_index,
                    stride=cfg.processing.plot_stride,
                )
