#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOP_ROOT="${SCRIPT_DIR}/../SOP_data"
SOP_ROOT="${SOP_ROOT:-$DEFAULT_SOP_ROOT}"
if [[ -z "${DYNAMIC_JOBS:-}" ]]; then
  DETECTED_CORES="$(nproc 2>/dev/null || echo 4)"
  if (( DETECTED_CORES > 13 )); then
    DYNAMIC_JOBS=12
  elif (( DETECTED_CORES > 1 )); then
    DYNAMIC_JOBS=$((DETECTED_CORES - 1))
  else
    DYNAMIC_JOBS=1
  fi
fi
DYNAMIC_FINGERPRINT_MODE="${DYNAMIC_FINGERPRINT_MODE:-stat}"
DYNAMIC_DETECT_REPLACED_FILES="${DYNAMIC_DETECT_REPLACED_FILES:-0}"
DYNAMIC_SERIES_MODE="${DYNAMIC_SERIES_MODE:-full}"
DYNAMIC_INCLUDE_LATERALS="${DYNAMIC_INCLUDE_LATERALS:-0}"
DYNAMIC_WRITE_ALL_DATA="${DYNAMIC_WRITE_ALL_DATA:-1}"
DYNAMIC_MIGRATE_PUBLISHED="${DYNAMIC_MIGRATE_PUBLISHED:-0}"

mkdir -p \
  "${SOP_ROOT}/raw_growth_test_dynamic" \
  "${SOP_ROOT}/published_dynamic" \
  "${SOP_ROOT}/manifests_dynamic" \
  "${SOP_ROOT}/logs" \
  "${SOP_ROOT}/tmp"

EXTRA_ARGS=(
  --fingerprint-mode "${DYNAMIC_FINGERPRINT_MODE}"
  --series-mode "${DYNAMIC_SERIES_MODE}"
)

if [[ -n "${DYNAMIC_JOBS}" ]]; then
  EXTRA_ARGS+=(-j "${DYNAMIC_JOBS}")
fi

if [[ "${DYNAMIC_DETECT_REPLACED_FILES}" == "0" || "${DYNAMIC_DETECT_REPLACED_FILES}" == "false" ]]; then
  EXTRA_ARGS+=(--no-detect-replaced-files)
else
  EXTRA_ARGS+=(--detect-replaced-files)
fi

if [[ "${DYNAMIC_INCLUDE_LATERALS}" == "0" || "${DYNAMIC_INCLUDE_LATERALS}" == "false" ]]; then
  EXTRA_ARGS+=(--no-laterals)
else
  EXTRA_ARGS+=(--include-laterals)
fi

if [[ "${DYNAMIC_WRITE_ALL_DATA}" == "0" || "${DYNAMIC_WRITE_ALL_DATA}" == "false" ]]; then
  EXTRA_ARGS+=(--skip-all-data)
else
  EXTRA_ARGS+=(--write-all-data)
fi

if [[ "${DYNAMIC_MIGRATE_PUBLISHED}" == "0" || "${DYNAMIC_MIGRATE_PUBLISHED}" == "false" ]]; then
  EXTRA_ARGS+=(--no-migrate-published)
else
  EXTRA_ARGS+=(--migrate-published)
fi

echo "[update_dynamic] jobs=${DYNAMIC_JOBS} series_mode=${DYNAMIC_SERIES_MODE} laterals=${DYNAMIC_INCLUDE_LATERALS} all_data=${DYNAMIC_WRITE_ALL_DATA} migrate=${DYNAMIC_MIGRATE_PUBLISHED}"

python3 "${SCRIPT_DIR}/process_dynamic_growth.py" \
  --sop-root "${SOP_ROOT}" \
  --manifests-dir "manifests_dynamic" \
  "${EXTRA_ARGS[@]}" \
  "$@"
