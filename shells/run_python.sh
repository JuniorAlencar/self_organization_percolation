#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
INSTALL_SYS_PY311="${INSTALL_SYS_PY311:-1}"

log() {
  echo "[run_python] $*"
}

ensure_python311_system() {
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    log "Usando interpretador: $(command -v "$PYTHON_BIN")"
    "$PYTHON_BIN" --version
    return 0
  fi

  if [[ "$INSTALL_SYS_PY311" != "1" ]]; then
    log "Erro: ${PYTHON_BIN} não encontrado e INSTALL_SYS_PY311=${INSTALL_SYS_PY311}."
    exit 1
  fi

  log "${PYTHON_BIN} não encontrado. Instalando Python 3.11 do PPA deadsnakes..."
  sudo apt update
  sudo apt install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install -y python3.11 python3.11-venv python3.11-dev

  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    log "Erro: ${PYTHON_BIN} continua indisponível após a instalação."
    exit 1
  fi

  log "Python 3.11 instalado com sucesso."
  "$PYTHON_BIN" --version
}

main() {
  cd "$SCRIPT_DIR"
  ensure_python311_system

  log "Executando install_python_dependencies.sh com ${PYTHON_BIN} ..."
  PYTHON_BIN="$PYTHON_BIN" INSTALL_VTK=true INSTALL_MAYAVI=true ./install_python_dependencies.sh

  log "Concluído."
}

main "$@"