#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOP_ROOT="${SCRIPT_DIR}/../SOP_data"
SOP_ROOT="${SOP_ROOT:-$DEFAULT_SOP_ROOT}"

mkdir -p \
  "${SOP_ROOT}/raw_growth_test_dynamic" \
  "${SOP_ROOT}/published_dynamic" \
  "${SOP_ROOT}/logs" \
  "${SOP_ROOT}/tmp"

python3 "${SCRIPT_DIR}/process_dynamic_growth.py" \
  --sop-root "${SOP_ROOT}" \
  "$@"
