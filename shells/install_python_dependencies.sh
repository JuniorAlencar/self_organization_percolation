#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Script: install_python_dependencies.sh
# Local esperado:
#   ./shells/install_python_dependencies.sh
#
# Estrutura esperada:
#   ./shells/install_python_dependencies.sh
#   ./requirements.txt   ou   ./requeriments.txt
#
# Ambiente virtual criado em:
#   ./.pyenv
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${PROJECT_ROOT}/.pyenv"
REQ_FILE_1="${PROJECT_ROOT}/requirements.txt"
REQ_FILE_2="${PROJECT_ROOT}/requeriments.txt"

PYTHON_BIN="${PYTHON_BIN:-python3}"

PROJECT_NAME="$(basename "${PROJECT_ROOT}")"
KERNEL_NAME="${PROJECT_NAME}_pyenv"
KERNEL_DISPLAY_NAME="Python (${PROJECT_NAME})"

echo "========================================"
echo "Projeto       : ${PROJECT_ROOT}"
echo "Pasta do venv : ${VENV_DIR}"
echo "Python usado  : ${PYTHON_BIN}"
echo "Kernel Jupyter: ${KERNEL_NAME}"
echo "========================================"

# ---------- verifica python ----------
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Erro: ${PYTHON_BIN} não foi encontrado no sistema."
    exit 1
fi

# ---------- verifica venv do sistema ----------
if ! "${PYTHON_BIN}" -m venv --help >/dev/null 2>&1; then
    echo "Erro: o módulo venv não está disponível para ${PYTHON_BIN}."
    echo "No Ubuntu, instale por exemplo:"
    echo "  sudo apt update"
    echo "  sudo apt install python3-venv"
    exit 1
fi

# ---------- escolhe arquivo de dependências ----------
if [[ -f "${REQ_FILE_1}" ]]; then
    REQUIREMENTS_FILE="${REQ_FILE_1}"
elif [[ -f "${REQ_FILE_2}" ]]; then
    REQUIREMENTS_FILE="${REQ_FILE_2}"
else
    echo "Erro: não encontrei nem:"
    echo "  - ${REQ_FILE_1}"
    echo "  - ${REQ_FILE_2}"
    exit 1
fi

echo "Arquivo de dependências: ${REQUIREMENTS_FILE}"

# ---------- cria/recria venv ----------
create_or_recreate_venv() {
    echo "Criando ambiente virtual em ${VENV_DIR} ..."
    rm -rf "${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
}

if [[ ! -d "${VENV_DIR}" ]]; then
    create_or_recreate_venv
else
    echo "Ambiente virtual já existe. Validando estrutura..."
fi

# se estiver incompleto, recria
if [[ ! -f "${VENV_DIR}/bin/activate" ]] || [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Ambiente virtual incompleto ou corrompido. Recriando..."
    create_or_recreate_venv
fi

# valida novamente após criação
if [[ ! -f "${VENV_DIR}/bin/activate" ]] || [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Erro: falha ao criar ambiente virtual corretamente."
    echo "Esperado:"
    echo "  - ${VENV_DIR}/bin/activate"
    echo "  - ${VENV_DIR}/bin/python"
    exit 1
fi

# ---------- ativa venv ----------
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Python ativo: $(which python)"
echo "Pip ativo   : $(which pip)"

# ---------- garante pip ----------
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel

# ---------- instala dependências ----------
echo "Instalando dependências do projeto ..."
pip install -r "${REQUIREMENTS_FILE}"

# ---------- suporte a Jupyter ----------
echo "Instalando suporte ao Jupyter ..."
pip install jupyter ipykernel

# ---------- registra kernel ----------
echo "Registrando kernel do Jupyter ..."
python -m ipykernel install --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"

echo "========================================"
echo "Instalação concluída com sucesso."
echo
echo "Para ativar o ambiente manualmente:"
echo "  source \"${VENV_DIR}/bin/activate\""
echo
echo "Para abrir o Jupyter:"
echo "  source \"${VENV_DIR}/bin/activate\""
echo "  jupyter notebook"
echo "ou"
echo "  jupyter lab"
echo
echo "Kernel registrado como:"
echo "  ${KERNEL_DISPLAY_NAME}"
echo "========================================"