#!/usr/bin/env bash
set -euo pipefail

# evita que '*.sh' vire string literal se não houver match
shopt -s nullglob

self="$(realpath "$0")"   # caminho absoluto deste script

# OPCIONAL: ordem natural (nc_2 antes de nc_10). Comente as 3 linhas de sort se não quiser.
mapfile -t scripts < <(printf '%s\n' ./*.sh | sort -V)
for f in "${scripts[@]}"; do
  # pule a si mesmo
  [[ "$(realpath "$f")" == "$self" ]] && continue

  echo ">> Rodando $f"
  if ! "$f"; then
    echo "[ERRO] $f falhou" >&2
    # se quiser parar no primeiro erro, descomente a linha abaixo:
    # exit 1
  fi
done
