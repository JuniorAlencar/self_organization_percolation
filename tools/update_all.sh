#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Como os scripts estão em tools/ e SOP_data está na raiz do repo,
# o caminho correto é ../SOP_data a partir de tools/
DEFAULT_SOP_ROOT="${SCRIPT_DIR}/../SOP_data"
SOP_ROOT="${SOP_ROOT:-$DEFAULT_SOP_ROOT}"

mkdir -p \
  "${SOP_ROOT}/raw" \
  "${SOP_ROOT}/manifests" \
  "${SOP_ROOT}/published" \
  "${SOP_ROOT}/logs" \
  "${SOP_ROOT}/tmp"

python3 "${SCRIPT_DIR}/update_published_means.py" --sop-root "${SOP_ROOT}" "$@"