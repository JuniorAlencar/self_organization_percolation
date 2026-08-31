#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOP_ROOT="$(cd "${SCRIPT_DIR}/../SOP_data" && pwd)"
SOP_ROOT="${SOP_ROOT:-$DEFAULT_SOP_ROOT}"

mkdir -p \
  "${SOP_ROOT}/raw_fractions" \
  "${SOP_ROOT}/published_counts" \
  "${SOP_ROOT}/logs" \
  "${SOP_ROOT}/tmp"

echo "[update_counts] sop_root=${SOP_ROOT}"

python3 "${SCRIPT_DIR}/process_counts.py" \
  --sop-root "${SOP_ROOT}" \
  "$@"
