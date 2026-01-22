from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DEODE meteo fetch and processing")
    ap.add_argument("--config", default="config/deode_default.yaml")
    ap.add_argument("--date", default=None, help="YYYYMMDD")
    ap.add_argument("--time", default=None, help="HH")
    ap.add_argument("--steps", type=int, default=None, help="Max forecast step in hours")
    ap.add_argument("--backend", choices=["cdo", "python"], default=None)
    ap.add_argument("--no-fetch", action="store_true", help="Do not fetch GRIBs")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    if args.date is None:
        if cfg.dataset.fdb_request and "date" in cfg.dataset.fdb_request:
            args.date = str(cfg.dataset.fdb_request["date"])
        else:
            raise SystemExit("Missing --date and no dataset.fdb_request date found in config")

    if args.time is None:
        if cfg.dataset.fdb_request and "time" in cfg.dataset.fdb_request:
            args.time = str(cfg.dataset.fdb_request["time"])
        else:
            args.time = "00"
    else:
        args.time = str(args.time)

    if len(args.time) == 4:
        args.time = args.time[:2]
    elif len(args.time) == 1:
        args.time = args.time.zfill(2)

    if args.steps is not None:
        cfg.forecast.max_step = int(args.steps)
    if args.backend is not None:
        cfg.processing.backend = args.backend

    start_dt = datetime.strptime(f"{args.date}{args.time}", "%Y%m%d%H")
    run_pipeline(cfg, start_dt, live_request=not args.no_fetch)


if __name__ == "__main__":
    main()
