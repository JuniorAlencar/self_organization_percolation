#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOP_ROOT="${SCRIPT_DIR}/../SOP_data"
SOP_ROOT="${SOP_ROOT:-$DEFAULT_SOP_ROOT}"

HEIGHT_STOP_MULTIPLIER=""
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --height-stop-multiplier|--nl-stop)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] $1 requires an integer multiplier, e.g. $1 2" >&2
        exit 1
      fi
      HEIGHT_STOP_MULTIPLIER="$2"
      shift 2
      ;;
    --height-stop-multiplier=*|--nl-stop=*)
      HEIGHT_STOP_MULTIPLIER="${1#*=}"
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

RAW_DIR="raw"
PUBLISHED_DIR="published"
MANIFESTS_DIR="manifests"
MANIFESTS_SIZES_DIR="manifests_sizes"
OUTPUT_SUFFIX=""

if [[ -n "${HEIGHT_STOP_MULTIPLIER}" ]]; then
  if ! [[ "${HEIGHT_STOP_MULTIPLIER}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] height-stop multiplier must be a positive integer: ${HEIGHT_STOP_MULTIPLIER}" >&2
    exit 1
  fi

  RAW_DIR="raw_${HEIGHT_STOP_MULTIPLIER}L_stop"
  PUBLISHED_DIR="published_${HEIGHT_STOP_MULTIPLIER}L_stop"
  MANIFESTS_DIR="manifests_${HEIGHT_STOP_MULTIPLIER}L_stop"
  MANIFESTS_SIZES_DIR="manifests_sizes_${HEIGHT_STOP_MULTIPLIER}L_stop"
  OUTPUT_SUFFIX="_${HEIGHT_STOP_MULTIPLIER}L_stop"
fi

MEANS_EXTRA_ARGS=()
RUN_SIZES=true
if [[ -n "${HEIGHT_STOP_MULTIPLIER}" ]]; then
  MEANS_EXTRA_ARGS+=(--time-series-only)
  RUN_SIZES=false
fi

mkdir -p \
  "${SOP_ROOT}/${RAW_DIR}" \
  "${SOP_ROOT}/${MANIFESTS_DIR}" \
  "${SOP_ROOT}/${MANIFESTS_SIZES_DIR}" \
  "${SOP_ROOT}/${PUBLISHED_DIR}" \
  "${SOP_ROOT}/logs" \
  "${SOP_ROOT}/tmp"

python3 "${SCRIPT_DIR}/index_raw_samples.py" \
  --sop-root "${SOP_ROOT}" \
  --raw-dir "${RAW_DIR}"

python3 "${SCRIPT_DIR}/update_published_means.py" \
  --sop-root "${SOP_ROOT}" \
  --raw-dir "${RAW_DIR}" \
  --published-dir "${PUBLISHED_DIR}" \
  --manifests-dir "${MANIFESTS_DIR}" \
  --output-suffix "${OUTPUT_SUFFIX}" \
  "${MEANS_EXTRA_ARGS[@]}" \
  "${FORWARD_ARGS[@]}"

if [[ "${RUN_SIZES}" == true ]]; then
  python3 "${SCRIPT_DIR}/update_published_sizes.py" \
    --sop-root "${SOP_ROOT}" \
    --raw-dir "${RAW_DIR}" \
    --published-dir "${PUBLISHED_DIR}" \
    --manifests-dir "${MANIFESTS_SIZES_DIR}" \
    --output-suffix "${OUTPUT_SUFFIX}" \
    "${FORWARD_ARGS[@]}"
else
  echo "[INFO] NL-stop mode: skipping size/heavy property calculations; temporal p_mean was saved."
fi
