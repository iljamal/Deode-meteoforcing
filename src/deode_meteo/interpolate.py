from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import xarray as xr


def run_cdo(cmd: str) -> None:
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"CDO command failed: {cmd}")


def cdo_remap_operator(method: str) -> str:
    if method == "remaplin":
        return "remapbil"
    if method == "remapcub":
        return "remapbic"
    if method == "remapmean":
        return "remapcon"
    raise ValueError(f"Unsupported remap method: {method}")


def cdo_gen_operator(method: str) -> str:
    if method == "remaplin":
        return "genbil"
    if method == "remapcub":
        return "gencub"
    if method == "remapmean":
        return "gencon"
    raise ValueError(f"Unsupported remap method for weights: {method}")


def interpolate_cdo(
    src_path: Path,
    target_grid: Path,
    out_path: Path,
    method: str,
    generate_weights: bool = False,
    weights_path: Optional[Path] = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    remap_op = cdo_remap_operator(method)

    if generate_weights and weights_path is not None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        gen_op = cdo_gen_operator(method)
        if not weights_path.exists():
            run_cdo(f"cdo -O {gen_op},{target_grid} {src_path} {weights_path}")
        run_cdo(
            f"cdo -O -f nc4 -z zip_1 {remap_op},{target_grid},{weights_path} {src_path} {out_path}"
        )
    else:
        run_cdo(f"cdo -O -f nc4 -z zip_1 {remap_op},{target_grid} {src_path} {out_path}")

    return out_path


def cdo_griddes(src_path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cdo", "-s", "griddes", str(src_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    out_path.write_text(result.stdout)
    return out_path


def interpolate_python(
    ds: xr.Dataset,
    target_grid: xr.Dataset,
    method: str,
) -> xr.Dataset:
    try:
        import xesmf as xe
    except ImportError as exc:
        raise RuntimeError("xesmf is required for python interpolation") from exc

    method_map = {
        "remaplin": "bilinear",
        "remapcub": "patch",
        "remapmean": "conservative",
    }
    if method not in method_map:
        raise ValueError(f"Unsupported remap method: {method}")

    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("Python interpolation requires lat/lon coords on the source dataset")

    regridder = xe.Regridder(ds, target_grid, method_map[method], reuse_weights=False)
    return regridder(ds)
