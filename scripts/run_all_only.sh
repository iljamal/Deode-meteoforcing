#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/eeim/mambaforge/bin/python}
PYTHONPATH=${PYTHONPATH:-src}
DATE=${1:-20241119}
TIME=${2:-00}
STEPS=${3:-48}
CONFIG_DIR=${CONFIG_DIR:-config}

configs=("${CONFIG_DIR}"/deode_*_only.yaml)
if [ ! -e "${configs[0]}" ]; then
  echo "No config files found in ${CONFIG_DIR}"
  exit 1
fi

for cfg in "${configs[@]}"; do
  base=$(basename "${cfg}")
  if [ "${base}" = "deode_u10_only.yaml" ] || [ "${base}" = "deode_v10_only.yaml" ]; then
    if [ -f "${CONFIG_DIR}/deode_wind10_only.yaml" ]; then
      echo "[SKIP] ${cfg} (use deode_wind10_only.yaml)"
      continue
    fi
  fi
  echo "[RUN] ${cfg}"
  PYTHONPATH="${PYTHONPATH}" "${PYTHON}" -m deode_meteo \
    --config "${cfg}" --date "${DATE}" --time "${TIME}" --steps "${STEPS}"
  echo "[OK]  ${cfg}"
  echo
  
done
