#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOP_ROOT="${SCRIPT_DIR}/../SOP_data"
SOP_ROOT="${SOP_ROOT:-$DEFAULT_SOP_ROOT}"

mkdir -p \
  "${SOP_ROOT}/raw" \
  "${SOP_ROOT}/manifests" \
  "${SOP_ROOT}/published" \
  "${SOP_ROOT}/logs" \
  "${SOP_ROOT}/tmp"

python3 "${SCRIPT_DIR}/index_raw_samples.py"
python3 "${SCRIPT_DIR}/update_published_means.py" --sop-root "${SOP_ROOT}" "$@"