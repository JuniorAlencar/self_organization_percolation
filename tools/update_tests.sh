#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOP_ROOT="${SCRIPT_DIR}/../SOP_data"
SOP_ROOT="${SOP_ROOT:-$DEFAULT_SOP_ROOT}"

if [[ -z "${TESTS_JOBS:-}" ]]; then
  DETECTED_CORES="$(nproc 2>/dev/null || echo 4)"
  if (( DETECTED_CORES > 13 )); then
    TESTS_JOBS=12
  elif (( DETECTED_CORES > 1 )); then
    TESTS_JOBS=$((DETECTED_CORES - 1))
  else
    TESTS_JOBS=1
  fi
fi

TESTS_LIST="${TESTS_LIST:-relative}"
TESTS_FINGERPRINT_MODE="${TESTS_FINGERPRINT_MODE:-stat}"
TESTS_SERIES_MODE="${TESTS_SERIES_MODE:-full}"
TESTS_DETECT_REPLACED_FILES="${TESTS_DETECT_REPLACED_FILES:-0}"
TESTS_INCLUDE_LATERALS="${TESTS_INCLUDE_LATERALS:-0}"
TESTS_WRITE_ALL_DATA="${TESTS_WRITE_ALL_DATA:-1}"
TESTS_MIGRATE_PUBLISHED="${TESTS_MIGRATE_PUBLISHED:-0}"
HEIGHT_TESTS_MIN_COUNT="${HEIGHT_TESTS_MIN_COUNT:-1}"

mkdir -p \
  "${SOP_ROOT}/tests_data" \
  "${SOP_ROOT}/processed_tests" \
  "${SOP_ROOT}/manifests_tests" \
  "${SOP_ROOT}/height_tests" \
  "${SOP_ROOT}/logs" \
  "${SOP_ROOT}/tmp"

EXTRA_ARGS=(
  --sop-root "${SOP_ROOT}"
  --jobs "${TESTS_JOBS}"
  --fingerprint-mode "${TESTS_FINGERPRINT_MODE}"
  --series-mode "${TESTS_SERIES_MODE}"
  --height-min-count "${HEIGHT_TESTS_MIN_COUNT}"
)

read -r -a TEST_ARRAY <<< "${TESTS_LIST}"
EXTRA_ARGS+=(--tests "${TEST_ARRAY[@]}")

if [[ "${TESTS_DETECT_REPLACED_FILES}" == "0" || "${TESTS_DETECT_REPLACED_FILES}" == "false" ]]; then
  EXTRA_ARGS+=(--no-detect-replaced-files)
else
  EXTRA_ARGS+=(--detect-replaced-files)
fi

if [[ "${TESTS_INCLUDE_LATERALS}" == "0" || "${TESTS_INCLUDE_LATERALS}" == "false" ]]; then
  EXTRA_ARGS+=(--no-laterals)
else
  EXTRA_ARGS+=(--laterals)
fi

if [[ "${TESTS_WRITE_ALL_DATA}" == "0" || "${TESTS_WRITE_ALL_DATA}" == "false" ]]; then
  EXTRA_ARGS+=(--no-write-all-data)
else
  EXTRA_ARGS+=(--write-all-data)
fi

if [[ "${TESTS_MIGRATE_PUBLISHED}" == "0" || "${TESTS_MIGRATE_PUBLISHED}" == "false" ]]; then
  EXTRA_ARGS+=(--no-migrate-published)
else
  EXTRA_ARGS+=(--migrate-published)
fi

echo "[update_tests] tests=${TESTS_LIST} jobs=${TESTS_JOBS} series_mode=${TESTS_SERIES_MODE}"

python3 "${SCRIPT_DIR}/process_tests.py" "${EXTRA_ARGS[@]}" "$@"
