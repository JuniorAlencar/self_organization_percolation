#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOP_ROOT="${SCRIPT_DIR}/../SOP_data"
SOP_ROOT="${SOP_ROOT:-$DEFAULT_SOP_ROOT}"
DYNAMIC_JOBS="${DYNAMIC_JOBS:-}"
DYNAMIC_FINGERPRINT_MODE="${DYNAMIC_FINGERPRINT_MODE:-stat}"
DYNAMIC_DETECT_REPLACED_FILES="${DYNAMIC_DETECT_REPLACED_FILES:-1}"

mkdir -p \
  "${SOP_ROOT}/raw_growth_test_dynamic" \
  "${SOP_ROOT}/published_dynamic" \
  "${SOP_ROOT}/manifests_dynamic" \
  "${SOP_ROOT}/logs" \
  "${SOP_ROOT}/tmp"

EXTRA_ARGS=(
  --fingerprint-mode "${DYNAMIC_FINGERPRINT_MODE}"
)

if [[ -n "${DYNAMIC_JOBS}" ]]; then
  EXTRA_ARGS+=(-j "${DYNAMIC_JOBS}")
fi

if [[ "${DYNAMIC_DETECT_REPLACED_FILES}" == "0" || "${DYNAMIC_DETECT_REPLACED_FILES}" == "false" ]]; then
  EXTRA_ARGS+=(--no-detect-replaced-files)
else
  EXTRA_ARGS+=(--detect-replaced-files)
fi

python3 "${SCRIPT_DIR}/process_dynamic_growth.py" \
  --sop-root "${SOP_ROOT}" \
  --manifests-dir "manifests_dynamic" \
  --include-laterals \
  "${EXTRA_ARGS[@]}" \
  "$@"
