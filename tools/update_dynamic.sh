#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOP_ROOT="${SCRIPT_DIR}/../SOP_data"
SOP_ROOT="${SOP_ROOT:-$DEFAULT_SOP_ROOT}"
if [[ -z "${DYNAMIC_JOBS:-}" ]]; then
  DETECTED_CORES="$(nproc 2>/dev/null || echo 4)"
  if (( DETECTED_CORES > 5 )); then
    DYNAMIC_JOBS=4
  elif (( DETECTED_CORES > 1 )); then
    DYNAMIC_JOBS=$((DETECTED_CORES - 1))
  else
    DYNAMIC_JOBS=1
  fi
fi

DYNAMIC_FINGERPRINT_MODE="${DYNAMIC_FINGERPRINT_MODE:-stat}"
DYNAMIC_DETECT_REPLACED_FILES="${DYNAMIC_DETECT_REPLACED_FILES:-0}"
DYNAMIC_SERIES_MODE="${DYNAMIC_SERIES_MODE:-profiles}"
DYNAMIC_INCLUDE_LATERALS="${DYNAMIC_INCLUDE_LATERALS:-0}"
DYNAMIC_WRITE_ALL_DATA="${DYNAMIC_WRITE_ALL_DATA:-1}"
DYNAMIC_MIGRATE_PUBLISHED="${DYNAMIC_MIGRATE_PUBLISHED:-0}"

SKIP_DYNAMIC="${DYNAMIC_SKIP_DYNAMIC_GROWTH:-${SKIP_DYNAMIC:-0}}"
SKIP_HEIGHT_SAMPLES="${DYNAMIC_SKIP_HEIGHT_SAMPLES:-${SKIP_HEIGHT_SAMPLES:-0}}"
SKIP_HEIGHT_ENSEMBLE="${DYNAMIC_SKIP_HEIGHT_ENSEMBLE:-${SKIP_HEIGHT_ENSEMBLE:-0}}"
SKIP_HEIGHT="${DYNAMIC_SKIP_HEIGHT_SERIES:-${SKIP_HEIGHT:-0}}"
HEIGHT_MIN_COUNT="${HEIGHT_MIN_COUNT:-1}"
HEIGHT_MAX_SAMPLES="${HEIGHT_MAX_SAMPLES:-}"

PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-dynamic)
      SKIP_DYNAMIC=1
      shift
      ;;
    --skip-height)
      SKIP_HEIGHT=1
      shift
      ;;
    --skip-height-samples)
      SKIP_HEIGHT_SAMPLES=1
      shift
      ;;
    --skip-height-ensemble)
      SKIP_HEIGHT_ENSEMBLE=1
      shift
      ;;
    --only-dynamic)
      SKIP_DYNAMIC=0
      SKIP_HEIGHT=1
      shift
      ;;
    --only-height)
      SKIP_DYNAMIC=1
      SKIP_HEIGHT=0
      shift
      ;;
    --height-min-count)
      HEIGHT_MIN_COUNT="$2"
      shift 2
      ;;
    --height-max-samples)
      HEIGHT_MAX_SAMPLES="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [options] [process_dynamic_growth options]"
      echo ""
      echo "Orchestrates full dynamic post-processing:"
      echo "  1) process_dynamic_growth.py          (raw JSON -> published & manifests)"
      echo "  2) process_height_timeseries.py       (.yts -> height sample measures)"
      echo "  3) process_height_ensemble_series.py  (.yts -> height ensemble timeseries)"
      echo ""
      echo "Pipeline control:"
      echo "  --skip-dynamic          Skip dynamic growth JSON processing"
      echo "  --skip-height           Skip both height timeseries steps"
      echo "  --skip-height-samples   Skip individual .yts sample measurements"
      echo "  --skip-height-ensemble  Skip .yts ensemble series averaging"
      echo "  --only-dynamic          Run only dynamic growth JSON processing"
      echo "  --only-height           Run only height processing"
      echo "  --height-min-count N    Min sample count threshold for ensemble (default: 1)"
      echo "  --height-max-samples N  Cap samples processed per parameter group"
      echo ""
      echo "All other options are forwarded directly to process_dynamic_growth.py."
      exit 0
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${SKIP_HEIGHT}" == "1" || "${SKIP_HEIGHT}" == "true" ]]; then
  SKIP_HEIGHT_SAMPLES=1
  SKIP_HEIGHT_ENSEMBLE=1
fi

mkdir -p \
  "${SOP_ROOT}/raw_growth_test_dynamic" \
  "${SOP_ROOT}/published_dynamic" \
  "${SOP_ROOT}/manifests_dynamic" \
  "${SOP_ROOT}/processed_height_timeseries" \
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

echo "[update_dynamic] SOP_ROOT=${SOP_ROOT}"
echo "[update_dynamic] Config: jobs=${DYNAMIC_JOBS} series_mode=${DYNAMIC_SERIES_MODE} all_data=${DYNAMIC_WRITE_ALL_DATA} migrate=${DYNAMIC_MIGRATE_PUBLISHED}"
echo "[update_dynamic] Stages: dynamic_growth=$(( 1 - SKIP_DYNAMIC )) height_samples=$(( 1 - SKIP_HEIGHT_SAMPLES )) height_ensemble=$(( 1 - SKIP_HEIGHT_ENSEMBLE ))"

if [[ "${SKIP_DYNAMIC}" == "0" || "${SKIP_DYNAMIC}" == "false" ]]; then
  echo "[update_dynamic] [1/3] Processing dynamic growth samples..."
  python3 "${SCRIPT_DIR}/process_dynamic_growth.py" \
    --sop-root "${SOP_ROOT}" \
    --manifests-dir "manifests_dynamic" \
    "${EXTRA_ARGS[@]}" \
    "${PASSTHROUGH_ARGS[@]}"
fi

HEIGHT_CLI_ARGS=()
if [[ -n "${HEIGHT_MAX_SAMPLES}" ]]; then
  HEIGHT_CLI_ARGS+=(--max-samples-per-group "${HEIGHT_MAX_SAMPLES}")
fi

if [[ "${SKIP_HEIGHT_SAMPLES}" == "0" || "${SKIP_HEIGHT_SAMPLES}" == "false" ]]; then
  echo "[update_dynamic] [2/3] Processing height timeseries (.yts -> sample measures)..."
  python3 "${SCRIPT_DIR}/process_height_timeseries.py" \
    --root "${SOP_ROOT}/raw_growth_test_dynamic" \
    --out-root "${SOP_ROOT}/processed_height_timeseries" \
    "${HEIGHT_CLI_ARGS[@]}"
fi

if [[ "${SKIP_HEIGHT_ENSEMBLE}" == "0" || "${SKIP_HEIGHT_ENSEMBLE}" == "false" ]]; then
  echo "[update_dynamic] [3/3] Processing height ensemble timeseries (.yts -> ensemble averages)..."
  python3 "${SCRIPT_DIR}/process_height_ensemble_series.py" \
    --root "${SOP_ROOT}/raw_growth_test_dynamic" \
    --out-root "${SOP_ROOT}/processed_height_timeseries" \
    --min-count "${HEIGHT_MIN_COUNT}" \
    "${HEIGHT_CLI_ARGS[@]}"
fi

echo "[update_dynamic] All tasks finished successfully."
