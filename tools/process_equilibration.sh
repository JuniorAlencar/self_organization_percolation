#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOP_ROOT_REAL="${REPO_ROOT}/SOP_data"

RAW_ROOT="${RAW_ROOT:-${SOP_ROOT_REAL}/raw/bond_percolation_equilibration}"
EXE="${EXE:-${REPO_ROOT}/build/EquilibrationReanalysis}"

if command -v nproc >/dev/null 2>&1; then
  DEFAULT_JOBS="$(nproc)"
else
  DEFAULT_JOBS=4
fi
JOBS="${JOBS:-$DEFAULT_JOBS}"

KEEP_WORK="${KEEP_WORK:-0}"
CLEAR_ALL=0
SHOW_HELP=0
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --clear-all) CLEAR_ALL=1 ;;
    -h|--help) SHOW_HELP=1 ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

usage() {
  cat <<'EOF'
Uso:
  JOBS=<num_threads> ./process_equilibration.sh [--clear-all] [args_extras_do_EquilibrationReanalysis]

Descrição:
  O script entra recursivamente em:
    ../SOP_data/raw/bond_percolation_equilibration
  e processa todas as subpastas "data" compatíveis com a estrutura criada pelo create_folders.

Como funciona:
  1) varre recursivamente todas as pastas "data"
  2) detecta automaticamente os parâmetros do grupo pela estrutura de diretórios
  3) extrai seed, P0 e p0 diretamente do nome de cada .json
  4) usa o arquivo:
       <grupo>/process_file.txt
     como manifest dos arquivos já processados
  5) arquivos já presentes no manifest são ignorados
  6) arquivos novos, fora do manifest, são processados em paralelo
  7) cada tarefa chama o EquilibrationReanalysis para um único arquivo (via seed do nome)
  8) ao final de cada sucesso, o manifest é atualizado

Uso mais comum:
  JOBS=8 ./process_equilibration.sh

Reprocessar tudo ignorando o manifest:
  JOBS=8 ./process_equilibration.sh --clear-all

Passando argumentos extras para o EquilibrationReanalysis:
  JOBS=8 ./process_equilibration.sh 10000000 7 20 2.0e-2 1.0e-6 2.0

Reprocessar tudo + argumentos extras:
  JOBS=8 ./process_equilibration.sh --clear-all 10000000 7 20 2.0e-2 1.0e-6 2.0

Variáveis de ambiente opcionais:
  JOBS       número máximo de tarefas paralelas
  RAW_ROOT   raiz da varredura recursiva
             padrão: ../SOP_data/raw/bond_percolation_equilibration
  EXE        caminho do executável
             padrão: ../build/EquilibrationReanalysis
  KEEP_WORK  se 1, mantém diretório temporário de logs/listas
             padrão: 0
EOF
}

if [[ "$SHOW_HELP" == "1" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$RAW_ROOT" ]]; then
  echo "[ERRO] RAW_ROOT não existe: $RAW_ROOT" >&2
  exit 1
fi

if [[ ! -x "$EXE" ]]; then
  echo "[ERRO] Executável não encontrado ou sem permissão: $EXE" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%dT%H%M%S)"
WORK_ROOT="${REPO_ROOT}/.process_equilibration/${RUN_ID}"
LOG_DIR="${WORK_ROOT}/logs"
TASKS_FILE="${WORK_ROOT}/tasks.tsv"

mkdir -p "$WORK_ROOT" "$LOG_DIR"
: > "$TASKS_FILE"

cleanup() {
  if [[ "$KEEP_WORK" != "1" ]]; then
    rm -rf "$WORK_ROOT"
  else
    echo "[INFO] Mantendo WORK_ROOT em: $WORK_ROOT"
  fi
}
trap cleanup EXIT

sanitize_id() {
  local s="$1"
  s="${s//\//__}"
  s="${s// /_}"
  s="${s//:/_}"
  echo "$s"
}

extract_value() {
  local name="$1"
  local prefix="$2"
  if [[ "$name" == ${prefix}* ]]; then
    printf '%s' "${name#${prefix}}"
  else
    return 1
  fi
}

json_in_manifest() {
  local manifest_file="$1"
  local base="$2"

  [[ -f "$manifest_file" ]] || return 1
  grep -Fxq "$base" "$manifest_file"
}

discover_tasks() {
  local task_id=0

  while IFS= read -r -d '' data_dir; do
    local rho_dir k_dir nt_dir nt_mode_dir l_dir dim_dir nc_dir
    rho_dir="$(basename "$(dirname "$data_dir")")"
    k_dir="$(basename "$(dirname "$(dirname "$data_dir")")")"
    nt_dir="$(basename "$(dirname "$(dirname "$(dirname "$data_dir")")")")"
    nt_mode_dir="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$data_dir")")")")")"
    l_dir="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$data_dir")")")")")")"
    dim_dir="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$data_dir")")")")")")")"
    nc_dir="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$data_dir")")")")")")")")"

    local num_colors dim L NT K rho type_perc
    num_colors="$(extract_value "$nc_dir" "num_colors_")"
    dim="$(extract_value "$dim_dir" "dim_")"
    L="$(extract_value "$l_dir" "L_")"
    NT="$(extract_value "$nt_dir" "NT_")"
    K="$(extract_value "$k_dir" "k_")"
    rho="$(extract_value "$rho_dir" "rho_")"

    case "$data_dir" in
      *"/bond_percolation_equilibration/"*) type_perc="bond" ;;
      *"/node_percolation_equilibration/"*) type_perc="node" ;;
      *)
        echo "[WARN] Não consegui inferir type_percolation para: $data_dir" >&2
        continue
        ;;
    esac

    if [[ "$nt_mode_dir" != "NT_constant" ]]; then
      echo "[WARN] Pulando modo NT não suportado: $data_dir" >&2
      continue
    fi

    local group_dir manifest_file group_rel group_id
    group_dir="$(dirname "$data_dir")"
    manifest_file="${group_dir}/process_file.txt"
    group_rel="${group_dir#${SOP_ROOT_REAL}/}"
    group_id="$(sanitize_id "$group_rel")"

    local pending_list="${WORK_ROOT}/pending_${group_id}.lst"
    : > "$pending_list"

    while IFS= read -r -d '' json_file; do
      local base seed P0 PP0

      base="$(basename "$json_file")"

      if [[ "$base" =~ seed_([0-9]+) ]]; then
        seed="${BASH_REMATCH[1]}"
      else
        echo "[WARN] Arquivo sem seed reconhecível, pulando: $json_file" >&2
        continue
      fi

      if [[ "$base" =~ P0_([0-9]+([.][0-9]+)?)_p0_([0-9]+([.][0-9]+)?) ]]; then
        P0="${BASH_REMATCH[1]}"
        PP0="${BASH_REMATCH[3]}"
      else
        echo "[WARN] Arquivo sem padrão P0/p0 reconhecível, pulando: $json_file" >&2
        continue
      fi

      if [[ "$CLEAR_ALL" != "1" ]] && json_in_manifest "$manifest_file" "$base"; then
        continue
      fi

      printf '%s\n' "$json_file" >> "$pending_list"
    done < <(find "$data_dir" -maxdepth 1 -type f -name '*.json' -print0 | sort -z)

    if [[ ! -s "$pending_list" ]]; then
      rm -f "$pending_list"
      continue
    fi

    local total_pending
    total_pending="$(wc -l < "$pending_list" | tr -d ' ')"
    echo "0" > "${WORK_ROOT}/progress_${group_id}.count"
    echo "$total_pending" > "${WORK_ROOT}/progress_${group_id}.total"

    while IFS= read -r json_file; do
      local base seed P0 PP0
      base="$(basename "$json_file")"

      [[ "$base" =~ seed_([0-9]+) ]] || continue
      seed="${BASH_REMATCH[1]}"

      [[ "$base" =~ P0_([0-9]+([.][0-9]+)?)_p0_([0-9]+([.][0-9]+)?) ]] || continue
      P0="${BASH_REMATCH[1]}"
      PP0="${BASH_REMATCH[3]}"

      ((task_id += 1))
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$task_id" \
        "$group_rel" \
        "$group_id" \
        "$group_dir" \
        "$manifest_file" \
        "$json_file" \
        "$base" \
        "$L" \
        "$PP0" \
        "$seed" \
        "$type_perc" \
        "$K" \
        "$NT" \
        "$dim" \
        "$num_colors" \
        "$rho" \
        "$P0" \
        >> "$TASKS_FILE"
    done < "$pending_list"

  done < <(find "$RAW_ROOT" -type d -name data -print0 | sort -z)
}

discover_tasks

if [[ ! -s "$TASKS_FILE" ]]; then
  echo "[INFO] Nenhum arquivo novo para processar."
  echo "[INFO] RAW_ROOT  = $RAW_ROOT"
  echo "[INFO] CLEAR_ALL = $CLEAR_ALL"
  exit 0
fi

TOTAL_TASKS="$(grep -c . "$TASKS_FILE")"
echo "[INFO] REPO_ROOT      = $REPO_ROOT"
echo "[INFO] RAW_ROOT       = $RAW_ROOT"
echo "[INFO] EXE            = $EXE"
echo "[INFO] JOBS           = $JOBS"
echo "[INFO] CLEAR_ALL      = $CLEAR_ALL"
echo "[INFO] TOTAL_TASKS    = $TOTAL_TASKS"
echo "[INFO] WORK_ROOT      = $WORK_ROOT"
echo

run_task() {
  local task_id="$1"
  local group_rel="$2"
  local group_id="$3"
  local group_dir="$4"
  local manifest_file="$5"
  local json_file="$6"
  local json_base="$7"
  local L="$8"
  local PP0="$9"
  local seed="${10}"
  local type_perc="${11}"
  local K="${12}"
  local NT="${13}"
  local dim="${14}"
  local num_colors="${15}"
  local rho="${16}"
  local P0="${17}"

  local log_file="${LOG_DIR}/task_${task_id}_$(sanitize_id "${json_base%.json}").log"
  local progress_file="${WORK_ROOT}/progress_${group_id}.count"
  local total_file="${WORK_ROOT}/progress_${group_id}.total"
  local lock_file="${WORK_ROOT}/progress_${group_id}.lock"

  (
    cd "$REPO_ROOT"
    "$EXE" \
      "$L" "$PP0" "$seed" "$type_perc" "$K" "$NT" "$dim" "$num_colors" "$rho" "$P0" \
      "${EXTRA_ARGS[@]}"
  ) > "$log_file" 2>&1

  {
    flock 200

    touch "$manifest_file"
    if ! grep -Fxq "$json_base" "$manifest_file"; then
      printf '%s\n' "$json_base" >> "$manifest_file"
      sort -u "$manifest_file" -o "$manifest_file"
    fi

    local done total
    done="$(<"$progress_file")"
    total="$(<"$total_file")"
    done=$((done + 1))
    printf '%s\n' "$done" > "$progress_file"

    GREEN_BOLD='\033[1;32m'
    CYAN='\033[0;36m'
    RESET='\033[0m'

    printf "[%b%s/%s%b] Pasta: %s\n" \
    "$GREEN_BOLD" "$done" "$total" "$RESET" "$group_dir"

    printf "           Arquivo: %b%s%b\n" \
    "$CYAN" "$json_base" "$RESET"
  } 200>"$lock_file"
}

status=0
running=0

while IFS=$'\t' read -r \
  task_id group_rel group_id group_dir manifest_file json_file json_base \
  L PP0 seed type_perc K NT dim num_colors rho P0
do
  [[ -n "${task_id:-}" ]] || continue
  [[ -n "${json_file:-}" ]] || continue
  [[ -n "${json_base:-}" ]] || continue

  echo "[QUEUE] Pasta: ${group_dir} | Arquivo: ${json_base}"

  run_task \
  "$task_id" "$group_rel" "$group_id" "$group_dir" "$manifest_file" "$json_file" "$json_base" \
  "$L" "$PP0" "$seed" "$type_perc" "$K" "$NT" "$dim" "$num_colors" "$rho" "$P0" &
	running=$((running + 1))

	if (( running >= JOBS )); then
	  if ! wait -n; then
	    status=1
	  fi
	  running=$((running - 1))
	fi
done < "$TASKS_FILE"

while (( running > 0 )); do
  if ! wait -n; then
    status=1
  fi
  running=$((running - 1))
done

if (( status != 0 )); then
  echo "[ERRO] Uma ou mais tarefas falharam. Veja os logs em: $LOG_DIR" >&2
  exit 1
fi

echo "[DONE] Reprocessamento concluído."
echo "[DONE] Logs em: $LOG_DIR"
