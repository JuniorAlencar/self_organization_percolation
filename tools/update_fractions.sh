#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOP_ROOT="${SCRIPT_DIR}/../SOP_data"
SOP_ROOT="${SOP_ROOT:-$DEFAULT_SOP_ROOT}"

mkdir -p \
  "${SOP_ROOT}/raw_fractions" \
  "${SOP_ROOT}/published_fractions" \
  "${SOP_ROOT}/manifests_fractions" \
  "${SOP_ROOT}/logs" \
  "${SOP_ROOT}/tmp"

echo "[update_fractions] sop_root=${SOP_ROOT}"

python3 "${SCRIPT_DIR}/process_raw_fractions.py" \
  --sop-root "${SOP_ROOT}" \
  "$@"
