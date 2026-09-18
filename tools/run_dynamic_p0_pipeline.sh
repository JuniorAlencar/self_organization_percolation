#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_SAMPLES="${PROJECT_ROOT}/python/run_samples.py"
SHELLS_DIR="${PROJECT_ROOT}/shells"
SOP_ROOT="${SOP_ROOT:-${PROJECT_ROOT}/SOP_data}"

P_ZERO_VALUES=(1.0)
P_CAP_VALUES_EXPR="[round(i,2) for i in np.arange(0.1, 1.1, 0.1) if round(i,2) != 0.2]"
SHELL_GLOB="L_1024_*.sh"

RUN_SAMPLES_BACKUP="$(mktemp)"
cp "${RUN_SAMPLES}" "${RUN_SAMPLES_BACKUP}"
SHELL_HELPERS_BACKUP_DIR=""
PIPELINE_SUDO_KEEPALIVE_PID=""

restore_run_samples() {
  cp "${RUN_SAMPLES_BACKUP}" "${RUN_SAMPLES}"
  rm -f "${RUN_SAMPLES_BACKUP}"
}

restore_shell_helpers() {
  if [[ -n "${SHELL_HELPERS_BACKUP_DIR}" && -d "${SHELL_HELPERS_BACKUP_DIR}" ]]; then
    find "${SHELL_HELPERS_BACKUP_DIR}" -maxdepth 1 -type f -name "*.sh" -exec mv -t "${SHELLS_DIR}" {} +
    rmdir "${SHELL_HELPERS_BACKUP_DIR}"
    SHELL_HELPERS_BACKUP_DIR=""
  fi
}

stop_pipeline_sudo_keepalive() {
  if [[ -n "${PIPELINE_SUDO_KEEPALIVE_PID}" ]]; then
    log "Stopping pipeline sudo keepalive (PID=${PIPELINE_SUDO_KEEPALIVE_PID})"
    kill "${PIPELINE_SUDO_KEEPALIVE_PID}" >/dev/null 2>&1 || true
    PIPELINE_SUDO_KEEPALIVE_PID=""
  fi
}

cleanup_on_exit() {
  stop_pipeline_sudo_keepalive
  restore_shell_helpers
  restore_run_samples
}
trap cleanup_on_exit EXIT

log() {
  printf '[run_dynamic_p0_pipeline] %s\n' "$*"
}

start_pipeline_sudo_keepalive() {
  if [[ "${LIMIT_CPU_CLOCK:-1}" != "1" ]]; then
    log "CPU clock limit disabled; sudo keepalive is not needed."
    return 0
  fi

  if ! command -v cpupower >/dev/null 2>&1; then
    log "cpupower not found; run_all.sh will skip CPU clock limiting."
    return 0
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    log "sudo not found; run_all.sh will skip CPU clock limiting."
    return 0
  fi

  log "Validating sudo once for the full p0 pipeline..."
  sudo -v
  log "sudo validated. Keeping credentials alive until the pipeline finishes."

  (
    while true; do
      sudo -n true
      sleep 60
      kill -0 "$$" >/dev/null 2>&1 || exit
    done
  ) >/dev/null 2>&1 &

  PIPELINE_SUDO_KEEPALIVE_PID=$!
}

set_run_samples_parameters() {
  local p0="$1"

  python3 - "${RUN_SAMPLES}" "${p0}" "${P_CAP_VALUES_EXPR}" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
p0 = sys.argv[2]
p_cap_values_expr = sys.argv[3]
text = path.read_text()

replacements = {
    r"^p0\s*=.*$": f"p0 = {p0}",
    r"^P0_lst\s*=.*$": f"P0_lst = {p_cap_values_expr}",
}

for pattern_text, replacement in replacements.items():
    pattern = re.compile(pattern_text, re.MULTILINE)
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise SystemExit(f"Could not find exactly one assignment matching: {pattern_text}")

path.write_text(text)
PY
}

cleanup_execution_shells() {
  find "${SHELLS_DIR}" -maxdepth 1 -type f -name "${SHELL_GLOB}" -delete
}

is_run_all_helper() {
  local base="$1"

  case "${base}" in
    run_all.sh|run_python.sh)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_execution_shell() {
  local base="$1"

  [[ "${base}" == L_1024_*.sh ]]
}

hide_non_execution_shells() {
  restore_shell_helpers

  SHELL_HELPERS_BACKUP_DIR="$(mktemp -d)"

  while IFS= read -r -d '' shell_file; do
    local base
    base="$(basename "${shell_file}")"

    if is_run_all_helper "${base}" || is_execution_shell "${base}"; then
      continue
    fi

    log "Temporarily hiding non-execution shell from run_all.sh: ${base}"
    mv "${shell_file}" "${SHELL_HELPERS_BACKUP_DIR}/"
  done < <(find "${SHELLS_DIR}" -maxdepth 1 -type f -name "*.sh" -print0)
}

cleanup_raw_dynamic_data() {
  local raw_dirs=(
    "${SOP_ROOT}/raw_raw_growth_test_dynamic"
    "${SOP_ROOT}/raw_growth_test_dynamic"
  )

  for raw_dir in "${raw_dirs[@]}"; do
    if [[ -d "${raw_dir}" ]]; then
      log "Removing raw data directory: ${raw_dir}"
      rm -rf "${raw_dir}"
    else
      log "Raw data directory not found, skipping: ${raw_dir}"
    fi
  done
}

log "Project root: ${PROJECT_ROOT}"
log "p0 values: ${P_ZERO_VALUES[*]}"
log "P0 list expression: ${P_CAP_VALUES_EXPR}"

start_pipeline_sudo_keepalive
cleanup_execution_shells

for p0 in "${P_ZERO_VALUES[@]}"; do
  log "Starting cycle for p0=${p0}"

  log "Updating python/run_samples.py to fixed p0=${p0} and the configured P0_lst"
  set_run_samples_parameters "${p0}"

  log "Generating shells with python/run_samples.py"
  (
    cd "${PROJECT_ROOT}/python"
    python3 run_samples.py
  )

  log "Running generated shells with shells/run_all.sh"
  hide_non_execution_shells
  (
    cd "${SHELLS_DIR}"
    ./run_all.sh
  )
  restore_shell_helpers

  log "Removing execution shells: ${SHELLS_DIR}/${SHELL_GLOB}"
  cleanup_execution_shells

  log "Running tools/update_dynamic.sh"
  (
    cd "${SCRIPT_DIR}"
    ./update_dynamic.sh
  )

  log "Cleaning raw dynamic data to free disk space"
  cleanup_raw_dynamic_data

  log "Finished cycle for p0=${p0}"
done

log "All p0 cycles finished successfully."
