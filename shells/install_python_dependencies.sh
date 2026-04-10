#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${PROJECT_ROOT}/.pyenv"
REQ_FILE_1="${PROJECT_ROOT}/requirements.txt"
REQ_FILE_2="${PROJECT_ROOT}/requeriments.txt"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_NAME="$(basename "${PROJECT_ROOT}")"
KERNEL_NAME="${PROJECT_NAME}_pyenv"
KERNEL_DISPLAY_NAME="Python (${PROJECT_NAME})"

VTK_VERSION="${VTK_VERSION:-9.4.2}"
QT_PACKAGE="${QT_PACKAGE:-PyQt5}"
INSTALL_VTK="${INSTALL_VTK:-true}"
INSTALL_MAYAVI="${INSTALL_MAYAVI:-true}"

FILTERED_REQUIREMENTS=""
VTK_STATUS="não instalado"
MAYAVI_STATUS="não instalado"

cleanup() {
    if [[ -n "${FILTERED_REQUIREMENTS}" && -f "${FILTERED_REQUIREMENTS}" ]]; then
        rm -f "${FILTERED_REQUIREMENTS}"
    fi
}
trap cleanup EXIT

log() {
    echo "[install_python_dependencies] $*"
}

log "========================================"
log "Projeto        : ${PROJECT_ROOT}"
log "Pasta do venv  : ${VENV_DIR}"
log "Python usado   : ${PYTHON_BIN}"
log "Kernel Jupyter : ${KERNEL_NAME}"
log "Qt package     : ${QT_PACKAGE}"
log "VTK version    : ${VTK_VERSION}"
log "INSTALL_VTK    : ${INSTALL_VTK}"
log "INSTALL_MAYAVI : ${INSTALL_MAYAVI}"
log "========================================"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    log "Erro: ${PYTHON_BIN} não foi encontrado no sistema."
    exit 1
fi

if ! "${PYTHON_BIN}" -m venv --help >/dev/null 2>&1; then
    log "Erro: o módulo venv não está disponível para ${PYTHON_BIN}."
    log "No Ubuntu, instale por exemplo:"
    log "  sudo apt update"
    log "  sudo apt install python3-venv"
    exit 1
fi

if [[ -f "${REQ_FILE_1}" ]]; then
    REQUIREMENTS_FILE="${REQ_FILE_1}"
elif [[ -f "${REQ_FILE_2}" ]]; then
    REQUIREMENTS_FILE="${REQ_FILE_2}"
else
    log "Erro: não encontrei nem:"
    log "  - ${REQ_FILE_1}"
    log "  - ${REQ_FILE_2}"
    exit 1
fi

log "Arquivo de dependências: ${REQUIREMENTS_FILE}"

create_or_recreate_venv() {
    log "Criando ambiente virtual em ${VENV_DIR} ..."
    rm -rf "${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
}

if [[ ! -d "${VENV_DIR}" ]]; then
    create_or_recreate_venv
else
    log "Ambiente virtual já existe. Validando estrutura..."
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]] || [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "Ambiente virtual incompleto ou corrompido. Recriando..."
    create_or_recreate_venv
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]] || [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "Erro: falha ao criar ambiente virtual corretamente."
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

log "Python ativo: $(which python)"
log "Pip ativo   : $(which pip)"

PYTHON_VERSION_FULL="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
)"

log "Versão do Python no venv: ${PYTHON_VERSION_FULL}"

python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel Cython

FILTERED_REQUIREMENTS="$(mktemp)"
grep -Eiv '^[[:space:]]*(mayavi|vtk|pyqt5|pyqt6|pyside2|pyside6)([[:space:]]*([<>=!~]=?|===).*)?$' \
    "${REQUIREMENTS_FILE}" > "${FILTERED_REQUIREMENTS}" || true

log "Instalando dependências base do projeto ..."
python -m pip install --upgrade --force-reinstall -r "${FILTERED_REQUIREMENTS}"

log "Instalando suporte ao Jupyter ..."
python -m pip install --upgrade jupyter jupyterlab ipykernel

if [[ "${INSTALL_VTK,,}" == "true" || "${INSTALL_VTK,,}" == "1" || "${INSTALL_VTK,,}" == "yes" ]]; then
    log "Instalando toolkit Qt (${QT_PACKAGE}) ..."
    python -m pip install --upgrade --force-reinstall "${QT_PACKAGE}"

    log "Instalando vtk==${VTK_VERSION} ..."
    python -m pip install --upgrade --force-reinstall "vtk==${VTK_VERSION}"

    python - <<'PY'
import vtk
print("VTK importado com sucesso:", vtk.vtkVersion.GetVTKVersion())
PY

    VTK_STATUS="instalado com sucesso (vtk==${VTK_VERSION})"
else
    VTK_STATUS="pulado por configuração"
fi

if [[ "${INSTALL_MAYAVI,,}" == "true" || "${INSTALL_MAYAVI,,}" == "1" || "${INSTALL_MAYAVI,,}" == "yes" ]]; then
    log "Instalando mayavi ..."
    python -m pip uninstall -y mayavi >/dev/null 2>&1 || true
    python -m pip install --no-build-isolation --no-cache-dir mayavi

    python - <<'PY'
from mayavi import mlab
print("Mayavi importado com sucesso.")
PY

    MAYAVI_STATUS="instalado com sucesso"
else
    MAYAVI_STATUS="pulado por configuração"
fi

log "Registrando kernel do Jupyter ..."
python -m ipykernel install --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"

log "========================================"
log "Instalação concluída."
log "Python        : ${PYTHON_VERSION_FULL}"
log "VTK           : ${VTK_STATUS}"
log "Mayavi        : ${MAYAVI_STATUS}"
log ""
log "Para ativar o ambiente manualmente:"
log "  source \"${VENV_DIR}/bin/activate\""
log ""
log "Para abrir o Jupyter:"
log "  source \"${VENV_DIR}/bin/activate\""
log "  jupyter notebook"
log "ou"
log "  jupyter lab"
log ""
log "Kernel registrado como:"
log "  ${KERNEL_DISPLAY_NAME}"
log "========================================"