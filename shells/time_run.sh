#!/usr/bin/env bash
set -euo pipefail

cd ..

BIN="./build/SOP"

# ajuda
if [[ ${1-} == "-h" || ${1-} == "--help" || $# -lt 1 ]]; then
  echo "Uso: $0 <args do SOP>"
  echo "Ex.: $0 1024 0.10 123 bond 1.0e-05 2000 2 1 0.001"
  exit 0
fi

# checagens
if [[ ! -x "$BIN" ]]; then
  echo "Erro: binário $BIN não encontrado ou sem permissão de execução." >&2
  exit 2
fi

# comando completo
cmd=("$BIN" "$@")
echo ">> Rodando: ${cmd[*]}"
echo "------------------------------------------------------------"

# escolha do 'time' (GNU time ou gtime no macOS)
TIMEBIN=""
if command -v /usr/bin/time >/dev/null 2>&1; then
  TIMEBIN="/usr/bin/time"
elif command -v gtime >/dev/null 2>&1; then
  TIMEBIN="gtime"
fi

if [[ -n "$TIMEBIN" ]]; then
  # imprime métricas ao final (vai para stderr por padrão)
  "$TIMEBIN" -f $'---\nwall=%E\nuser=%U\nsys=%S\nmaxRSS=%M KB\nexit=%x' "${cmd[@]}"
else
  # fallback sem GNU time: mede apenas wall clock
  start_ns=$(date +%s%N)
  "${cmd[@]}"
  status=$?
  end_ns=$(date +%s%N)
  dt_ms=$(( (end_ns - start_ns)/1000000 ))
  printf -- "---\nwall=%.3f s\nexit=%d\n" "$(bc -l <<< "$dt_ms/1000")" "$status"
fi
