#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob

self="$(realpath "$0")"

EXCLUDED_SCRIPTS=(
  "install_python_dependencies.sh"
  "run_python.sh"
)

LIMIT_CPU_CLOCK="${LIMIT_CPU_CLOCK:-1}"
CPU_MAX_FREQ="${CPU_MAX_FREQ:-3.6GHz}"
CPU_GOVERNOR="${CPU_GOVERNOR:-ondemand}"

OLD_GOVERNOR=""
OLD_MAX_FREQ=""
SUDO_KEEPALIVE_PID=""

SCRIPT_START_EPOCH="$(date +%s)"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

elapsed_hms() {
  local total="$1"
  local h=$(( total / 3600 ))
  local m=$(( (total % 3600) / 60 ))
  local s=$(( total % 60 ))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

restore_cpu_settings() {
  [[ "$LIMIT_CPU_CLOCK" -eq 1 ]] || return 0

  command -v cpupower >/dev/null 2>&1 || return 0

  log "Restoring original CPU settings..."

  if [[ -n "$OLD_GOVERNOR" ]]; then
    log "Restoring governor to: $OLD_GOVERNOR"
    sudo -n cpupower frequency-set -g "$OLD_GOVERNOR" >/dev/null 2>&1 || true
  fi

  if [[ -n "$OLD_MAX_FREQ" ]]; then
    log "Restoring max CPU frequency to: $OLD_MAX_FREQ"
    sudo -n cpupower frequency-set -u "$OLD_MAX_FREQ" >/dev/null 2>&1 || true
  fi

  log "CPU settings restoration finished."
}

cleanup() {
  local exit_code=$?
  log "Cleanup started (exit code = $exit_code)"

  if [[ -n "$SUDO_KEEPALIVE_PID" ]]; then
    log "Stopping sudo keepalive (PID=$SUDO_KEEPALIVE_PID)"
    kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
  fi

  restore_cpu_settings

  local script_end_epoch
  script_end_epoch="$(date +%s)"
  local total_elapsed=$(( script_end_epoch - SCRIPT_START_EPOCH ))

  log "run_all.sh finished. Total elapsed: $(elapsed_hms "$total_elapsed")"
}

start_sudo_keepalive() {
  log "Validating sudo permissions..."
  sudo -v
  log "sudo validated successfully."

  (
    while true; do
      sudo -n true
      sleep 60
      kill -0 "$$" >/dev/null 2>&1 || exit
    done
  ) >/dev/null 2>&1 &
  SUDO_KEEPALIVE_PID=$!

  log "sudo keepalive started (PID=$SUDO_KEEPALIVE_PID)"
}

apply_cpu_limit() {
  [[ "$LIMIT_CPU_CLOCK" -eq 1 ]] || {
    log "CPU clock limit disabled."
    return 0
  }

  if ! command -v cpupower >/dev/null 2>&1; then
    log "[WARN] cpupower not found. Running without CPU clock limit."
    return 0
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    log "[WARN] sudo not found. Running without CPU clock limit."
    return 0
  fi

  start_sudo_keepalive

  if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    OLD_GOVERNOR="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true)"
    [[ -n "$OLD_GOVERNOR" ]] && log "Detected current governor: $OLD_GOVERNOR"
  fi

  if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq ]]; then
    OLD_MAX_FREQ="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || true)"
    if [[ -n "$OLD_MAX_FREQ" ]]; then
      OLD_MAX_FREQ="${OLD_MAX_FREQ}KHz"
      log "Detected current max CPU frequency: $OLD_MAX_FREQ"
    fi
  fi

  log "Applying CPU max frequency limit to: $CPU_MAX_FREQ"
  log "Applying CPU governor: $CPU_GOVERNOR"

  sudo -n cpupower frequency-set -g "$CPU_GOVERNOR" >/dev/null 2>&1 || true
  sudo -n cpupower frequency-set -u "$CPU_MAX_FREQ" >/dev/null 2>&1 || {
    log "[WARN] Could not apply CPU clock limit."
    return 0
  }

  log "CPU clock limit applied successfully."
}

is_excluded_script() {
  local name="$1"
  for excluded in "${EXCLUDED_SCRIPTS[@]}"; do
    [[ "$name" == "$excluded" ]] && return 0
  done
  return 1
}

trap cleanup EXIT

log "run_all.sh started"
log "Working directory: $(pwd)"
log "Script path: $self"

apply_cpu_limit

log "Scanning .sh files in current directory..."
mapfile -t scripts < <(printf '%s\n' ./*.sh | sort -V)
log "Found ${#scripts[@]} shell script(s)."

ran_any=0

for f in "${scripts[@]}"; do
  base="$(basename "$f")"

  if [[ "$(realpath "$f")" == "$self" ]]; then
    log "Skipping self: $base"
    continue
  fi

  if is_excluded_script "$base"; then
    log "Skipping excluded script: $base"
    continue
  fi

  ran_any=1
  log "------------------------------------------------------------"
  log "Starting script: $f"

  script_start_epoch="$(date +%s)"

  if "$f"; then
    script_end_epoch="$(date +%s)"
    script_elapsed=$(( script_end_epoch - script_start_epoch ))
    log "Finished successfully: $f"
    log "Elapsed for $base: $(elapsed_hms "$script_elapsed")"
  else
    script_end_epoch="$(date +%s)"
    script_elapsed=$(( script_end_epoch - script_start_epoch ))
    log "[ERROR] Script failed: $f"
    log "Elapsed before failure for $base: $(elapsed_hms "$script_elapsed")"
    # exit 1
  fi
done

if [[ "$ran_any" -eq 0 ]]; then
  log "No runnable scripts found."
fi

log "All scripts were processed."
