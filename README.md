## Requirements

- Python 3.10+
- CDO (recommended for interpolation and GRIB subsetting)
- ecCodes (required by cfgrib)
- earthkit-data (Polytope fetch)
- polytope-client (Polytope API client)

### Authentication

For live Polytope requests you need **DESP upgraded user** credentials and the
`desp-authentication.py` flow to obtain a token. Ask DESP Service for access,
then run the authentication script before requesting data.

## Installation

```bash
cd /home/ilja/hpc_atos_perm/deode_wf
/home/eeim/mambaforge/bin/python -m pip install -e .
```

Optional (for Python interpolation):
```bash
/home/eeim/mambaforge/bin/python -m pip install xesmf
```

## Quick start

```bash
PYTHONPATH=src /home/eeim/mambaforge/bin/python -m deode_meteo \
  --config config/deode_default.yaml --steps 48
```

## Output layout

```
forcing/meteo/DEODE_fc_{georef}_{date}/
  grib_raw/
  intermediate/
  plots/
  weights/
  FORCE_deode_yYYYYmMMdDD.nc
  FORCE_deode_<param>_yYYYYmMMdDD.nc
```

## Key options (YAML)

```yaml
processing:
  backend: cdo            # cdo | python
  interpolate: true       # false = no remap
  rotate_winds: true
  output_mode: merged     # merged | per_parameter
  target_grid: auto_from_grib   # or a grid file / grid spec
  subset_box: null        # or lon/lat bounds

paths:
  output_root: forcing/meteo/DEODE_fc_{georef}_{date}
  include_datetime_subdirs: false
```

## Utilities

- **Generate absolute humidity** from existing t2/rh2m files:
  ```bash
  PYTHONPATH=src /home/eeim/mambaforge/bin/python scripts/derive_ah2m.py \
    --run-dir forcing/meteo/DEODE_fc_ud3q9t_20241119
  ```

- **Merge per‑parameter daily files** into one daily file:
  ```bash
  PYTHONPATH=src /home/eeim/mambaforge/bin/python scripts/merge_forcing_day.py \
    --run-dir forcing/meteo/DEODE_fc_ud3q9t_20241119 --date 20241119
  ```

- **Sanity plots** (maps + time series):
  ```bash
  PYTHONPATH=src /home/eeim/mambaforge/bin/python scripts/plot_sanity.py \
    --run-dir forcing/meteo/DEODE_fc_ud3q9t_20241119 --date 20241119 --map-steps all
  ```

## Notes

- CDO is recommended for large grids and robust GRIB subsetting.
- If you want the target grid inferred from GRIB, set:
  ```yaml
  processing:
    target_grid: auto_from_grib
  ```
- Rotation sanity plots are only produced when `rotate_winds: true`.
