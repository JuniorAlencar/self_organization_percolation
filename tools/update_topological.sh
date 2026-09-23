#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOP_ROOT="${SCRIPT_DIR}/../SOP_data"
SOP_ROOT="${SOP_ROOT:-$DEFAULT_SOP_ROOT}"

run_counts=1
run_fractions=1
counts_args=()
fractions_args=()
shared_args=()

while (($#)); do
  case "$1" in
    --counts-only)
      run_fractions=0
      shift
      ;;
    --fractions-only)
      run_counts=0
      shift
      ;;
    --counts-arg)
      (($# >= 2)) || { echo "[ERROR] --counts-arg requires a value" >&2; exit 2; }
      counts_args+=("$2")
      shift 2
      ;;
    --fractions-arg)
      (($# >= 2)) || { echo "[ERROR] --fractions-arg requires a value" >&2; exit 2; }
      fractions_args+=("$2")
      shift 2
      ;;
    --sop-root)
      (($# >= 2)) || { echo "[ERROR] --sop-root requires a value" >&2; exit 2; }
      SOP_ROOT="$2"
      shift 2
      ;;
    --raw-dir)
      (($# >= 2)) || { echo "[ERROR] --raw-dir requires a value" >&2; exit 2; }
      shared_args+=(--raw-dir "$2")
      shift 2
      ;;
    --dry-run)
      shared_args+=(--dry-run)
      shift
      ;;
    -h|--help)
      cat <<'HELP'
Usage: update_topological.sh [--sop-root PATH] [--raw-dir DIR] [--dry-run]
                            [--counts-only | --fractions-only]
                            [--counts-arg ARG] [--fractions-arg ARG]

By default, processes both raw topological counts and raw fraction series.
Use repeated --counts-arg/--fractions-arg to pass processor-specific options.
HELP
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p \
  "${SOP_ROOT}/raw_fractions" \
  "${SOP_ROOT}/published_counts" \
  "${SOP_ROOT}/published_fractions" \
  "${SOP_ROOT}/manifests_fractions" \
  "${SOP_ROOT}/logs" \
  "${SOP_ROOT}/tmp"

echo "[update_topological] sop_root=${SOP_ROOT}"

if [[ "$run_counts" -eq 1 ]]; then
  echo "[update_topological] processing topological counts"
  python3 "${SCRIPT_DIR}/process_counts.py" \
    --sop-root "${SOP_ROOT}" \
    "${shared_args[@]}" \
    "${counts_args[@]}"
fi

if [[ "$run_fractions" -eq 1 ]]; then
  echo "[update_topological] processing fractions"
  python3 "${SCRIPT_DIR}/process_raw_fractions.py" \
    --sop-root "${SOP_ROOT}" \
    "${shared_args[@]}" \
    "${fractions_args[@]}"
fi
