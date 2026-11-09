#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOP_ROOT="${SCRIPT_DIR}/../SOP_data"
SOP_ROOT="${SOP_ROOT:-$DEFAULT_SOP_ROOT}"
PUBLISHED_COUNTS_DIR="${PUBLISHED_COUNTS_DIR:-published_counts}"
PUBLISHED_FRACTIONS_DIR="${PUBLISHED_FRACTIONS_DIR:-published_fractions}"
MANIFESTS_FRACTIONS_DIR="${MANIFESTS_FRACTIONS_DIR:-manifests_fractions}"
TOPOLOGICAL_REQUIRE_MANIFESTS="${TOPOLOGICAL_REQUIRE_MANIFESTS:-1}"

if [[ -z "${TOPOLOGICAL_JOBS:-}" ]]; then
  DETECTED_CORES="$(nproc 2>/dev/null || echo 4)"
  if (( DETECTED_CORES > 5 )); then
    TOPOLOGICAL_JOBS=10
  elif (( DETECTED_CORES > 1 )); then
    TOPOLOGICAL_JOBS=$((DETECTED_CORES - 1))
  else
    TOPOLOGICAL_JOBS=1
  fi
fi

run_counts=1
run_fractions=1
counts_args=()
fractions_args=()
shared_args=()

while (($#)); do
  case "$1" in
    -j|--jobs)
      (($# >= 2)) || { echo "[ERROR] --jobs requires a value" >&2; exit 2; }
      TOPOLOGICAL_JOBS="$2"
      shift 2
      ;;
    --refresh-counts)
      counts_args+=(--refresh)
      shift
      ;;
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
    --published-counts-dir)
      (($# >= 2)) || { echo "[ERROR] --published-counts-dir requires a value" >&2; exit 2; }
      PUBLISHED_COUNTS_DIR="$2"
      shift 2
      ;;
    --published-fractions-dir)
      (($# >= 2)) || { echo "[ERROR] --published-fractions-dir requires a value" >&2; exit 2; }
      PUBLISHED_FRACTIONS_DIR="$2"
      shift 2
      ;;
    --manifests-fractions-dir|--manifests-dir)
      (($# >= 2)) || { echo "[ERROR] --manifests-fractions-dir requires a value" >&2; exit 2; }
      MANIFESTS_FRACTIONS_DIR="$2"
      shift 2
      ;;
    --no-require-manifests)
      TOPOLOGICAL_REQUIRE_MANIFESTS=0
      shift
      ;;
    --require-manifests)
      TOPOLOGICAL_REQUIRE_MANIFESTS=1
      shift
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
                            [-j N | --jobs N] [--refresh-counts]
                            [--counts-only | --fractions-only]
                            [--published-counts-dir DIR]
                            [--published-fractions-dir DIR]
                            [--manifests-fractions-dir DIR]
                            [--no-require-manifests] [--require-manifests]
                            [--counts-arg ARG] [--fractions-arg ARG]

By default, processes both raw topological counts and raw fraction series.
Uses worker processes (TOPOLOGICAL_JOBS or --jobs to override).
--jobs 1 runs serially. Counts parallelize by input file; fractions by bundle.
Unchanged counts skip parsing/publishing; --refresh-counts forces reprocessing.
Use --no-require-manifests or TOPOLOGICAL_REQUIRE_MANIFESTS=0 to adopt existing
published_counts and published_fractions without requiring manifests.
Stages run sequentially, sharing the same worker limit to bound memory usage.
Counts outputs in SOP_ROOT/published_counts (one set per parameter group):
  counts_P0_*_p0_*.json          samples with separate spanning_clusters
  counts_P0_*_p0_*.clusters.csv  one row per cluster, with d_bulk/d_min fits
  counts_P0_*_p0_*.samples.csv   spanning count per sample (including zero)
Cluster roles: largest_spanning (index 0) and other_spanning (indices 1+).
Legacy samples retain their properties and are marked spanning_unknown.
C++ fit estimates are preserved; curve-filter options do not refit them.
Example: update_topological.sh --counts-only --sop-root /path/to/SOP_data
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

if [[ "${SOP_ROOT}" = /* ]]; then
  SOP_ROOT="$(cd "${SOP_ROOT}" 2>/dev/null && pwd || echo "${SOP_ROOT}")"
else
  SOP_ROOT="$(cd "${SOP_ROOT}" 2>/dev/null && pwd || realpath "${SOP_ROOT}" 2>/dev/null || echo "${SOP_ROOT}")"
fi

case "$(basename "${SOP_ROOT}")" in
  published_dynamic|published_counts|published_fractions|raw_fractions|raw_growth_test_dynamic|tests_data|processed_height_timeseries|manifests_fractions|manifests_dynamic)
    echo "[ERROR] SOP_ROOT cannot be a subdirectory of SOP_data (${SOP_ROOT}). It must point to SOP_data." >&2
    exit 1
    ;;
esac

[[ "$TOPOLOGICAL_JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[ERROR] jobs must be a positive integer" >&2; exit 2;
}

mkdir -p \
  "${SOP_ROOT}/raw_fractions" \
  "${SOP_ROOT}/${PUBLISHED_COUNTS_DIR}" \
  "${SOP_ROOT}/${PUBLISHED_FRACTIONS_DIR}" \
  "${SOP_ROOT}/${MANIFESTS_FRACTIONS_DIR}" \
  "${SOP_ROOT}/logs" \
  "${SOP_ROOT}/tmp"

if [[ "${TOPOLOGICAL_REQUIRE_MANIFESTS}" == "0" || "${TOPOLOGICAL_REQUIRE_MANIFESTS}" == "false" ]]; then
  counts_args+=(--no-require-manifests)
  fractions_args+=(--no-require-manifests)
else
  counts_args+=(--require-manifests)
  fractions_args+=(--require-manifests)
fi

echo "[update_topological] sop_root=${SOP_ROOT} jobs=${TOPOLOGICAL_JOBS} require_manifests=${TOPOLOGICAL_REQUIRE_MANIFESTS}"

if [[ "$run_counts" -eq 1 ]]; then
  echo "[update_topological] processing counts: largest spanning cluster and each other spanning cluster"
  python3 "${SCRIPT_DIR}/process_counts.py" \
    --sop-root "${SOP_ROOT}" \
    --out-dir "${PUBLISHED_COUNTS_DIR}" \
    --jobs "${TOPOLOGICAL_JOBS}" \
    "${shared_args[@]}" \
    "${counts_args[@]}"
fi

if [[ "$run_fractions" -eq 1 ]]; then
  echo "[update_topological] processing fractions"
  python3 "${SCRIPT_DIR}/process_raw_fractions.py" \
    --sop-root "${SOP_ROOT}" \
    --published-dir "${PUBLISHED_FRACTIONS_DIR}" \
    --manifests-dir "${MANIFESTS_FRACTIONS_DIR}" \
    --jobs "${TOPOLOGICAL_JOBS}" \
    "${shared_args[@]}" \
    "${fractions_args[@]}"
fi
