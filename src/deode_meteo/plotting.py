from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr


def _slice_lat(lat, lat_min, lat_max):
    if lat[0] < lat[-1]:
        return slice(lat_min, lat_max)
    return slice(lat_max, lat_min)


def plot_wind_compare(
    ds: xr.Dataset,
    region: dict,
    out_path: Path,
    u_name: str = "u10",
    v_name: str = "v10",
    step_index: int = 0,
    stride: int = 10,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if u_name not in ds or v_name not in ds:
        raise ValueError("Missing wind components in dataset")
    if f"{u_name}_grid" not in ds or f"{v_name}_grid" not in ds:
        raise ValueError("Missing unrotated wind components for comparison")

    lon = ds["lon"]
    lat = ds["lat"]
    lon_slice = slice(region["lon_min"], region["lon_max"])
    lat_slice = _slice_lat(lat, region["lat_min"], region["lat_max"])

    sub = ds.sel(lon=lon_slice, lat=lat_slice)
    sub = sub.isel(time=step_index)

    u_rot = sub[u_name][::stride, ::stride]
    v_rot = sub[v_name][::stride, ::stride]
    u_raw = sub[f"{u_name}_grid"][::stride, ::stride]
    v_raw = sub[f"{v_name}_grid"][::stride, ::stride]

    lon2d, lat2d = xr.broadcast(sub["lon"], sub["lat"])
    lon2d = lon2d[::stride, ::stride]
    lat2d = lat2d[::stride, ::stride]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].quiver(lon2d, lat2d, u_raw, v_raw, scale=500)
    axes[0].set_title("Wind (grid-relative)")
    axes[0].set_xlabel("lon")
    axes[0].set_ylabel("lat")

    axes[1].quiver(lon2d, lat2d, u_rot, v_rot, scale=500)
    axes[1].set_title("Wind (rotated)")
    axes[1].set_xlabel("lon")
    axes[1].set_ylabel("lat")

    fig.suptitle(region.get("name", "wind_compare"))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
