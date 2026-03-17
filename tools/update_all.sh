#!/usr/bin/env bash
set -euo pipefail

mkdir -p ../SOP_data/{raw,manifests,reduced_local,published,logs,tmp}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

python3 tools/index_raw_samples.py
python3 tools/update_group_summaries.py
