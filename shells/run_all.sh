#!/usr/bin/env bash
set -euo pipefail

# evita que '*.sh' vire string literal se não houver match
shopt -s nullglob

self="$(realpath "$0")"

# =========================
# Configuração de clock
# =========================
LIMIT_CPU_CLOCK="${LIMIT_CPU_CLOCK:-1}"
CPU_MAX_FREQ="${CPU_MAX_FREQ:-4.0GHz}"
CPU_GOVERNOR="${CPU_GOVERNOR:-ondemand}"

OLD_GOVERNOR=""
OLD_MAX_FREQ=""
SUDO_KEEPALIVE_PID=""

restore_cpu_settings() {
  [[ "$LIMIT_CPU_CLOCK" -eq 1 ]] || return 0

  if ! command -v cpupower >/dev/null 2>&1; then
    return 0
  fi

  echo
  echo "[INFO] Restaurando configuração original da CPU..."

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
  echo "[INFO] Validando permissões sudo..."
  sudo -v

  # mantém o ticket do sudo vivo enquanto este script existir
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
    echo "[INFO] Limite de clock desativado."
    return 0
  }

  if ! command -v cpupower >/dev/null 2>&1; then
    echo "[WARN] cpupower não encontrado. Rodando sem limitar clock."
    return 0
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    echo "[WARN] sudo não encontrado. Rodando sem limitar clock."
    return 0
  fi

  start_sudo_keepalive

  if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    OLD_GOVERNOR="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true)"
  fi

  if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq ]]; then
    OLD_MAX_FREQ="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || true)"
    [[ -n "$OLD_MAX_FREQ" ]] && OLD_MAX_FREQ="${OLD_MAX_FREQ}KHz"
  fi

  echo "[INFO] Limitando clock máximo da CPU para $CPU_MAX_FREQ"
  sudo -n cpupower frequency-set -g "$CPU_GOVERNOR" >/dev/null 2>&1 || true
  sudo -n cpupower frequency-set -u "$CPU_MAX_FREQ" >/dev/null 2>&1 || {
    echo "[WARN] Não foi possível aplicar o limite de clock."
    return 0
  }

  echo "[INFO] Limite aplicado com sucesso."
}

trap cleanup EXIT

apply_cpu_limit

mapfile -t scripts < <(printf '%s\n' ./*.sh | sort -V)

for f in "${scripts[@]}"; do
  [[ "$(realpath "$f")" == "$self" ]] && continue

  echo ">> Rodando $f"
  if ! "$f"; then
    echo "[ERRO] $f falhou" >&2
    # exit 1
  fi
done

echo
echo "[INFO] Todos os scripts foram processados."