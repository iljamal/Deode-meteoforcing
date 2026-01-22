from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import Config, ForecastConfig, ParamGroup


def build_step_list(forecast: ForecastConfig, step_type: str) -> List[str]:
    max_step = forecast.max_step
    step_hours = forecast.step_hours
    if step_type == "instant":
        start = 0 if forecast.include_step0 else step_hours
        return [str(h) for h in range(start, max_step + 1, step_hours)]
    if step_type == "accum":
        return [
            forecast.accum_step_style.format(hour=h)
            for h in range(step_hours, max_step + 1, step_hours)
        ]
    raise ValueError(f"Unsupported step_type: {step_type}")


def build_request(cfg: Config, group: ParamGroup, date_str: str, time_str: str) -> dict:
    step = build_step_list(cfg.forecast, group.step_type)
    return {
        "class": cfg.dataset.class_name,
        "dataset": cfg.dataset.dataset,
        "expver": cfg.dataset.expver,
        "stream": cfg.dataset.stream,
        "date": date_str,
        "time": int(time_str),
        "type": cfg.dataset.type,
        "levtype": cfg.dataset.levtype,
        "georef": cfg.dataset.georef,
        "step": step,
        "param": group.param_ids,
    }


def group_grib_path(out_dir: Path, group: ParamGroup, date_str: str, time_str: str) -> Path:
    stem = f"{group.name}_{group.step_type}_{date_str}T{time_str}"
    return out_dir / f"{stem}.grib2"


def fetch_group(
    cfg: Config,
    group: ParamGroup,
    date_str: str,
    time_str: str,
    out_dir: Path,
    live_request: bool = True,
) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    grib_path = group_grib_path(out_dir, group, date_str, time_str)
    if grib_path.exists():
        return grib_path

    if not live_request:
        raise FileNotFoundError(f"Missing GRIB and live_request disabled: {grib_path}")

    try:
        import earthkit.data
    except ImportError as exc:
        raise RuntimeError("earthkit-data is required for Polytope fetch") from exc

    request = build_request(cfg, group, date_str, time_str)
    try:
        data = earthkit.data.from_source(
            "polytope",
            cfg.dataset.source,
            request,
            address=cfg.dataset.address,
            stream=False,
        )
        data.to_target("file", str(grib_path))
        return grib_path
    except Exception as exc:
        print(f"[WARN] Fetch failed for group '{group.name}': {exc}")
        return None


def date_from_dt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def time_from_dt(dt: datetime) -> str:
    return dt.strftime("%H")
