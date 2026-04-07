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
#
# Comportamento do Mayavi:
#   - INSTALL_MAYAVI=auto   -> tenta instalar só se Python < 3.12
#   - INSTALL_MAYAVI=true   -> força tentativa via pip
#   - INSTALL_MAYAVI=false  -> pula Mayavi
#
# Exemplo:
#   INSTALL_MAYAVI=true ./shells/install_python_dependencies.sh
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

INSTALL_MAYAVI="${INSTALL_MAYAVI:-auto}"      # auto | true | false
MAYAVI_QT_PACKAGE="${MAYAVI_QT_PACKAGE:-PyQt5}"

FILTERED_REQUIREMENTS=""
MAYAVI_STATUS="não instalado"

cleanup() {
    if [[ -n "${FILTERED_REQUIREMENTS}" && -f "${FILTERED_REQUIREMENTS}" ]]; then
        rm -f "${FILTERED_REQUIREMENTS}"
    fi
}
trap cleanup EXIT

echo "========================================"
echo "Projeto       : ${PROJECT_ROOT}"
echo "Pasta do venv : ${VENV_DIR}"
echo "Python usado  : ${PYTHON_BIN}"
echo "Kernel Jupyter: ${KERNEL_NAME}"
echo "INSTALL_MAYAVI: ${INSTALL_MAYAVI}"
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

PYTHON_VERSION_FULL="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
)"

PYTHON_VERSION_MM="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

echo "Versão do Python no venv: ${PYTHON_VERSION_FULL}"

is_python_ge_312() {
    python - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
PY
}

install_mayavi_with_pip() {
    echo "Instalando toolkit Qt para Mayavi (${MAYAVI_QT_PACKAGE}) ..."
    python -m pip install "${MAYAVI_QT_PACKAGE}"

    echo "Tentando instalar Mayavi via pip ..."
    python -m pip install mayavi

    python - <<'PY'
from mayavi import mlab
print("Mayavi importado com sucesso.")
PY
}

# ---------- garante pip ----------
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel

# ---------- instala dependências base ----------
# Remove mayavi / PyQt do requirements para tratar separadamente.
FILTERED_REQUIREMENTS="$(mktemp)"

grep -Eiv '^[[:space:]]*(mayavi|pyqt5|pyqt6)([[:space:]]*([<>=!~]=?|===).*)?$' \
    "${REQUIREMENTS_FILE}" > "${FILTERED_REQUIREMENTS}" || true

echo "Instalando dependências base do projeto ..."
python -m pip install -r "${FILTERED_REQUIREMENTS}"

# ---------- suporte a Jupyter ----------
echo "Instalando suporte ao Jupyter ..."
python -m pip install jupyter jupyterlab ipykernel

# ---------- instala Mayavi separadamente ----------
case "${INSTALL_MAYAVI,,}" in
    auto)
        if is_python_ge_312; then
            echo
            echo "Aviso: Mayavi foi pulado automaticamente no Python ${PYTHON_VERSION_MM}."
            echo "Motivo: builds via pip estão instáveis/quebrando nesse stack."
            echo "Sugestão: use Python 3.11 para o venv, ou um ambiente conda/micromamba separado."
            MAYAVI_STATUS="pulado automaticamente em Python >= 3.12"
        else
            echo "Tentando instalar Mayavi automaticamente ..."
            if install_mayavi_with_pip; then
                MAYAVI_STATUS="instalado com sucesso"
            else
                echo
                echo "Aviso: falha ao instalar Mayavi via pip."
                echo "O restante do ambiente foi instalado com sucesso."
                echo "Sugestão: tente com Python 3.11 ou use conda/micromamba para o Mayavi."
                MAYAVI_STATUS="falhou na instalação via pip"
            fi
        fi
        ;;
    true|1|yes)
        echo "INSTALL_MAYAVI=${INSTALL_MAYAVI}: forçando instalação do Mayavi via pip ..."
        install_mayavi_with_pip
        MAYAVI_STATUS="instalado com sucesso"
        ;;
    false|0|no|skip)
        echo "Mayavi pulado por configuração."
        MAYAVI_STATUS="pulado por configuração"
        ;;
    *)
        echo "Erro: INSTALL_MAYAVI deve ser auto, true ou false."
        exit 1
        ;;
esac

# ---------- registra kernel ----------
echo "Registrando kernel do Jupyter ..."
python -m ipykernel install --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"

echo "========================================"
echo "Instalação concluída."
echo "Python        : ${PYTHON_VERSION_FULL}"
echo "Mayavi        : ${MAYAVI_STATUS}"
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
echo
echo "Exemplos úteis:"
echo "  PYTHON_BIN=python3.11 ./shells/install_python_dependencies.sh"
echo "  INSTALL_MAYAVI=false ./shells/install_python_dependencies.sh"
echo "  INSTALL_MAYAVI=true  ./shells/install_python_dependencies.sh"
echo "========================================"