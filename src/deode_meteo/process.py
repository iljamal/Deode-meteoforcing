from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Optional

import numpy as np
import xarray as xr


def normalize_time_dim(ds: xr.Dataset) -> xr.Dataset:
    if "step" in ds.dims:
        if "time" in ds.dims:
            if ds.sizes["time"] != 1:
                raise ValueError("Both time and step dimensions exist with time size > 1")
            ds = ds.rename({"time": "analysis_time"})
            ds = ds.squeeze("analysis_time", drop=True)
        elif "time" in ds.variables:
            ds = ds.rename({"time": "analysis_time"})
        ds = ds.rename({"step": "time"})
    return ds


def squeeze_singletons(ds: xr.Dataset) -> xr.Dataset:
    for dim in list(ds.dims):
        if ds.sizes.get(dim, 0) == 1:
            ds = ds.squeeze(dim, drop=True)
    return ds


def build_time_index(start_dt: datetime, hours: Iterable[int]) -> np.ndarray:
    return np.array([start_dt + timedelta(hours=h) for h in hours], dtype="datetime64[ns]")


def assign_time(ds: xr.Dataset, start_dt: datetime, hours: Iterable[int]) -> xr.Dataset:
    hours_list = list(hours)
    if "time" not in ds.dims:
        raise ValueError("Dataset has no time dimension")
    if ds.sizes["time"] != len(hours_list):
        raise ValueError(
            f"Time length mismatch: data has {ds.sizes['time']} steps, expected {len(hours_list)}"
        )
    times = build_time_index(start_dt, hours_list)
    return ds.assign_coords(time=("time", times))


def ensure_time_first(ds: xr.Dataset) -> xr.Dataset:
    out = ds.copy()
    for name, da in ds.data_vars.items():
        if "time" not in da.dims:
            continue
        dims = ("time",) + tuple(d for d in da.dims if d != "time")
        out[name] = da.transpose(*dims)
    return out


def _slice_coord(coord: xr.DataArray, vmin: float, vmax: float) -> slice:
    if coord[0] < coord[-1]:
        return slice(vmin, vmax)
    return slice(vmax, vmin)


def subset_box(ds: xr.Dataset, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> xr.Dataset:
    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("subset_box requires lon/lat coordinates")

    lon = ds.coords["lon"]
    lat = ds.coords["lat"]

    def _dims_match(coord: xr.DataArray) -> bool:
        for dim in coord.dims:
            if dim not in ds.dims:
                return False
        if coord.sizes[dim] != ds.sizes[dim]:
            return False
        return True

    if not (_dims_match(lon) and _dims_match(lat)):
        try:
            ds = ds.drop_vars(["lon", "lat"], errors="ignore")
            ds = add_latlon_from_grid_mapping(ds)
            lon = ds.coords["lon"]
            lat = ds.coords["lat"]
        except Exception as exc:
            raise ValueError("lon/lat coordinates are not aligned with data dimensions") from exc

    def _subset_xy() -> xr.Dataset:
        mapping_name = find_grid_mapping_var(ds)
        if mapping_name is None or "x" not in ds.coords or "y" not in ds.coords:
            raise ValueError("x/y coordinates or grid mapping missing for projected subsetting")
        try:
            import pyproj
        except ImportError as exc:
            raise RuntimeError("pyproj is required for projected subsetting") from exc

        mapping = ds[mapping_name]
        crs = pyproj.CRS.from_cf(mapping.attrs)
        transformer = pyproj.Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
        x1, y1 = transformer.transform(lon_min, lat_min)
        x2, y2 = transformer.transform(lon_max, lat_max)
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        return ds.sel(x=_slice_coord(ds.coords["x"], x_min, x_max), y=_slice_coord(ds.coords["y"], y_min, y_max))

    if lon.ndim == 1 and lat.ndim == 1:
        try:
            return ds.sel(
                lon=_slice_coord(lon, lon_min, lon_max),
                lat=_slice_coord(lat, lat_min, lat_max),
            )
        except Exception:
            try:
                return _subset_xy()
            except Exception:
                pass
            lon_mask = (lon >= lon_min) & (lon <= lon_max)
            lat_mask = (lat >= lat_min) & (lat <= lat_max)
            return ds.where(lon_mask, drop=True).where(lat_mask, drop=True)

    mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    return ds.where(mask, drop=True)


def deaccumulate(da: xr.DataArray, copy_hour1_to0: bool = True) -> xr.DataArray:
    if "time" not in da.dims:
        raise ValueError("deaccumulate requires a time dimension")
    first = da.isel(time=0)
    diff = da.diff(dim="time")
    inc = xr.concat([first, diff], dim="time")
    if copy_hour1_to0:
        inc = xr.concat([inc.isel(time=0), inc], dim="time")
    inc = inc.assign_coords(time=np.arange(inc.sizes["time"]))
    return inc


def convert_accum_to_flux(
    da: xr.DataArray,
    step_hours: int,
    units: Optional[str],
    output_units: Optional[str],
) -> xr.DataArray:
    if output_units != "kg m-2 s-1":
        return da

    seconds = float(step_hours) * 3600.0
    if units == "mm":
        scale = 1.0
    elif units == "m":
        scale = 1000.0
    else:
        scale = 1.0
    out = da * (scale / seconds)
    out.attrs["units"] = output_units
    return out


def compute_absolute_humidity(t2: xr.DataArray, rh: xr.DataArray) -> xr.DataArray:
    t_k = t2
    t_c = t_k - 273.15
    e_s = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = (rh / 100.0) * e_s
    ah = 216.7 * e / t_k
    ah.name = "ah2m"
    ah.attrs["units"] = "g m-3"
    ah.attrs["long_name"] = "2 m absolute humidity"
    ah.attrs["derived_from"] = "t2,rh2m"
    return ah


def find_grid_mapping_var(ds: xr.Dataset) -> Optional[str]:
    for name, var in ds.data_vars.items():
        if "grid_mapping_name" in var.attrs:
            return name
    return None


def rotation_angle_from_grid_mapping(ds: xr.Dataset, dx: float = 1000.0) -> xr.DataArray:
    try:
        import pyproj
    except ImportError as exc:
        raise RuntimeError("pyproj is required for wind rotation") from exc

    mapping_name = find_grid_mapping_var(ds)
    if mapping_name is None:
        raise ValueError("No grid_mapping variable found for rotation")

    mapping = ds[mapping_name]
    crs = pyproj.CRS.from_cf(mapping.attrs)
    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("Rotation requires x/y coordinates on the dataset")

    x = ds["x"].values
    y = ds["y"].values
    xx, yy = np.meshgrid(x, y)

    transformer = pyproj.Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    lon_dx, lat_dx = transformer.transform(xx + dx, yy)

    geod = pyproj.Geod(ellps="WGS84")
    az, _, _ = geod.inv(lon, lat, lon_dx, lat_dx)

    angle = np.deg2rad(90.0 - az)
    return xr.DataArray(angle, dims=("y", "x"), coords={"y": ds["y"], "x": ds["x"]})


def add_latlon_from_grid_mapping(ds: xr.Dataset) -> xr.Dataset:
    try:
        import pyproj
    except ImportError as exc:
        raise RuntimeError("pyproj is required to add lat/lon coordinates") from exc

    mapping_name = find_grid_mapping_var(ds)
    if mapping_name is None:
        raise ValueError("No grid_mapping variable found to derive lat/lon")

    mapping = ds[mapping_name]
    crs = pyproj.CRS.from_cf(mapping.attrs)
    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("lat/lon derivation requires x/y coordinates")

    x = ds["x"].values
    y = ds["y"].values
    xx, yy = np.meshgrid(x, y)
    transformer = pyproj.Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    lon, lat = transformer.transform(xx, yy)

    ds = ds.assign_coords(lon=(("y", "x"), lon), lat=(("y", "x"), lat))
    return ds


def rotate_winds(
    ds: xr.Dataset,
    u_name: str,
    v_name: str,
    angle: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    if angle is None:
        angle = rotation_angle_from_grid_mapping(ds)

    u = ds[u_name]
    v = ds[v_name]
    u_rot = u * np.cos(angle) - v * np.sin(angle)
    v_rot = u * np.sin(angle) + v * np.cos(angle)

    out = ds.copy()
    out[f"{u_name}_grid"] = u
    out[f"{v_name}_grid"] = v
    out[u_name] = u_rot
    out[v_name] = v_rot
    out[u_name].attrs["rotation"] = "rotated to true east"
    out[v_name].attrs["rotation"] = "rotated to true north"
    return out
