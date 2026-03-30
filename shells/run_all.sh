#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob

self="$(realpath "$0")"

EXCLUDED_SCRIPTS=(
  "install_python_dependencies.sh"
)

LIMIT_CPU_CLOCK="${LIMIT_CPU_CLOCK:-1}"
CPU_MAX_FREQ="${CPU_MAX_FREQ:-3.6GHz}"
CPU_GOVERNOR="${CPU_GOVERNOR:-ondemand}"

OLD_GOVERNOR=""
OLD_MAX_FREQ=""
SUDO_KEEPALIVE_PID=""

restore_cpu_settings() {
  [[ "$LIMIT_CPU_CLOCK" -eq 1 ]] || return 0

  command -v cpupower >/dev/null 2>&1 || return 0

  echo
  echo "[INFO] Restoring original CPU settings..."

  if [[ -n "$OLD_GOVERNOR" ]]; then
    sudo -n cpupower frequency-set -g "$OLD_GOVERNOR" >/dev/null 2>&1 || true
  fi

  if [[ -n "$OLD_MAX_FREQ" ]]; then
    sudo -n cpupower frequency-set -u "$OLD_MAX_FREQ" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  if [[ -n "$SUDO_KEEPALIVE_PID" ]]; then
    kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
  fi
  restore_cpu_settings
}

start_sudo_keepalive() {
  echo "[INFO] Validating sudo permissions..."
  sudo -v

  (
    while true; do
      sudo -n true
      sleep 60
      kill -0 "$$" >/dev/null 2>&1 || exit
    done
  ) >/dev/null 2>&1 &
  SUDO_KEEPALIVE_PID=$!
}

apply_cpu_limit() {
  [[ "$LIMIT_CPU_CLOCK" -eq 1 ]] || {
    echo "[INFO] CPU clock limit disabled."
    return 0
  }

  if ! command -v cpupower >/dev/null 2>&1; then
    echo "[WARN] cpupower not found. Running without CPU clock limit."
    return 0
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    echo "[WARN] sudo not found. Running without CPU clock limit."
    return 0
  fi

  start_sudo_keepalive

  if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    OLD_GOVERNOR="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true)"
  fi

  if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq ]]; then
    OLD_MAX_FREQ="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || true)"
    if [[ -n "$OLD_MAX_FREQ" ]]; then
      OLD_MAX_FREQ="${OLD_MAX_FREQ}KHz"
    fi
  fi

  echo "[INFO] Limiting CPU max frequency to $CPU_MAX_FREQ"
  sudo -n cpupower frequency-set -g "$CPU_GOVERNOR" >/dev/null 2>&1 || true
  sudo -n cpupower frequency-set -u "$CPU_MAX_FREQ" >/dev/null 2>&1 || {
    echo "[WARN] Could not apply CPU clock limit."
    return 0
  }

  echo "[INFO] CPU clock limit applied successfully."
}

is_excluded_script() {
  local name="$1"
  for excluded in "${EXCLUDED_SCRIPTS[@]}"; do
    [[ "$name" == "$excluded" ]] && return 0
  done
  return 1
}

trap cleanup EXIT

apply_cpu_limit

mapfile -t scripts < <(printf '%s\n' ./*.sh | sort -V)

for f in "${scripts[@]}"; do
  base="$(basename "$f")"

  [[ "$(realpath "$f")" == "$self" ]] && continue
  is_excluded_script "$base" && continue

  echo ">> Running $f"
  if ! "$f"; then
    echo "[ERROR] $f failed" >&2
    # exit 1
  fi
done

echo
echo "[INFO] All scripts were processed."
