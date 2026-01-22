from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
import json


@dataclass
class DatasetConfig:
    class_name: str
    dataset: str
    expver: str
    stream: str
    type: str
    levtype: str
    georef: str
    address: str
    source: str
    time: str
    fdb_request: Optional[dict] = None


@dataclass
class ForecastConfig:
    step_hours: int
    max_step: int
    include_step0: bool
    accum_step_style: str


@dataclass
class PathConfig:
    output_root: Path
    bathy_file: Path
    grib_subdir: str
    intermediate_subdir: str
    weights_subdir: str
    plots_subdir: str
    include_datetime_subdirs: bool


@dataclass
class ProcessingConfig:
    backend: str
    interpolate: bool
    generate_weights: bool
    rotate_winds: bool
    output_mode: str
    output_prefix: str
    plots: bool
    plot_step_index: int
    plot_stride: int
    plot_regions: List[dict]
    subset_box: Optional[dict]
    target_grid: Optional[object]


@dataclass
class ParameterConfig:
    short_name: str
    param_id: int
    step_type: str
    interpolation: str
    group: Optional[str] = None
    units: Optional[str] = None
    output_units: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig
    forecast: ForecastConfig
    paths: PathConfig
    processing: ProcessingConfig
    parameters: Dict[str, ParameterConfig]


@dataclass
class ParamGroup:
    name: str
    params: Dict[str, ParameterConfig]
    step_type: str
    interpolation: str

    @property
    def param_ids(self) -> List[int]:
        return [p.param_id for p in self.params.values()]

    @property
    def short_names(self) -> List[str]:
        return [p.short_name for p in self.params.values()]


def load_config(path: Path) -> Config:
    data = yaml.safe_load(Path(path).read_text())

    dataset_raw: Dict[str, Any] = dict(data["dataset"])
    fdb_request = _parse_fdb_request(dataset_raw.get("fdb_request"))
    if fdb_request:
        for key in ("class", "dataset", "expver", "stream", "type", "levtype", "georef", "time"):
            if key in fdb_request:
                dataset_raw[key] = fdb_request[key]

    dataset = DatasetConfig(
        class_name=dataset_raw["class"],
        dataset=dataset_raw["dataset"],
        expver=str(dataset_raw["expver"]),
        stream=dataset_raw["stream"],
        type=dataset_raw["type"],
        levtype=dataset_raw["levtype"],
        georef=dataset_raw["georef"],
        address=dataset_raw["address"],
        source=dataset_raw["source"],
        time=str(dataset_raw.get("time", "00")),
        fdb_request=fdb_request,
    )

    forecast = ForecastConfig(
        step_hours=int(data["forecast"]["step_hours"]),
        max_step=int(data["forecast"]["max_step"]),
        include_step0=bool(data["forecast"]["include_step0"]),
        accum_step_style=str(data["forecast"]["accum_step_style"]),
    )

    paths = PathConfig(
        output_root=Path(data["paths"]["output_root"]),
        bathy_file=Path(data["paths"]["bathy_file"]),
        grib_subdir=data["paths"]["grib_subdir"],
        intermediate_subdir=data["paths"]["intermediate_subdir"],
        weights_subdir=data["paths"]["weights_subdir"],
        plots_subdir=data["paths"]["plots_subdir"],
        include_datetime_subdirs=bool(data["paths"].get("include_datetime_subdirs", True)),
    )

    proc = ProcessingConfig(
        backend=str(data["processing"]["backend"]),
        interpolate=bool(data["processing"].get("interpolate", True)),
        generate_weights=bool(data["processing"]["generate_weights"]),
        rotate_winds=bool(data["processing"]["rotate_winds"]),
        output_mode=str(data["processing"]["output_mode"]),
        output_prefix=str(data["processing"]["output_prefix"]),
        plots=bool(data["processing"]["plots"]),
        plot_step_index=int(data["processing"]["plot_step_index"]),
        plot_stride=int(data["processing"]["plot_stride"]),
        plot_regions=list(data["processing"].get("plot_regions", [])),
        subset_box=data["processing"].get("subset_box"),
        target_grid=data["processing"].get("target_grid"),
    )

    params = {}
    for key, val in data["parameters"].items():
        params[key] = ParameterConfig(
            short_name=str(val["short_name"]),
            param_id=int(val["param_id"]),
            step_type=str(val["step_type"]),
            interpolation=str(val["interpolation"]),
            group=val.get("group"),
            units=val.get("units"),
            output_units=val.get("output_units"),
        )

    return Config(
        dataset=dataset,
        forecast=forecast,
        paths=paths,
        processing=proc,
        parameters=params,
    )


def _parse_fdb_request(value: Any) -> Optional[dict]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = value[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        try:
            return yaml.safe_load(blob)
        except Exception:
            return None


def group_parameters(cfg: Config) -> Dict[str, ParamGroup]:
    grouped: Dict[str, Dict[str, ParameterConfig]] = {}
    for name, param in cfg.parameters.items():
        group_name = param.group or name
        grouped.setdefault(group_name, {})[name] = param

    groups: Dict[str, ParamGroup] = {}
    for gname, params in grouped.items():
        step_types = {p.step_type for p in params.values()}
        if len(step_types) != 1:
            raise ValueError(f"Group {gname} has mixed step types: {step_types}")
        interp = {p.interpolation for p in params.values()}
        if len(interp) != 1:
            raise ValueError(f"Group {gname} has mixed interpolation methods: {interp}")
        groups[gname] = ParamGroup(
            name=gname,
            params=params,
            step_type=next(iter(step_types)),
            interpolation=next(iter(interp)),
        )

    return groups
