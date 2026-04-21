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

JOBS_WAS_SET=0
if [[ -n "${JOBS+x}" ]]; then
  JOBS_WAS_SET=1
else
  JOBS="$DEFAULT_JOBS"
fi

AUTO_JOBS="${AUTO_JOBS:-1}"
RAM_HEADROOM_PCT="${RAM_HEADROOM_PCT:-20}"
RAM_MIN_HEADROOM_MB="${RAM_MIN_HEADROOM_MB:-2048}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
RAM_DEBUG="${RAM_DEBUG:-0}"

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
  ./process_equilibration.sh [--clear-all] [args_extras_do_EquilibrationReanalysis]

Descrição:
  O script entra recursivamente em:
    ../SOP_data/raw/bond_percolation_equilibration
  e processa todas as subpastas "data" compatíveis com a estrutura criada pelo create_folders.

Nova lógica de execução:
  1) varre recursivamente todas as pastas "data"
  2) detecta automaticamente os parâmetros do grupo pela estrutura de diretórios
  3) extrai seed, P0 e p0 diretamente do nome de cada .json
  4) usa o arquivo:
       <grupo>/process_file.txt
     como manifest dos arquivos já processados
  5) para cada grupo/pasta:
     - monta a lista de arquivos pendentes daquele grupo
     - roda 1 benchmark real em 1 único processo
     - mede o pico de RAM do benchmark
     - lê a RAM disponível do sistema
     - calcula automaticamente quantos JOBS usar para aquele grupo
     - processa imediatamente o restante dos arquivos daquele grupo
  6) só depois passa para o próximo grupo

Uso mais comum:
  ./process_equilibration.sh

Reprocessar tudo ignorando o manifest:
  ./process_equilibration.sh --clear-all

Passando argumentos extras para o EquilibrationReanalysis:
  ./process_equilibration.sh 10000000 7 20 2.0e-2 1.0e-6 2.0

Reprocessar tudo + argumentos extras:
  ./process_equilibration.sh --clear-all 10000000 7 20 2.0e-2 1.0e-6 2.0

Variáveis de ambiente opcionais:
  RAW_ROOT            raiz da varredura recursiva
                      padrão: ../SOP_data/raw/bond_percolation_equilibration

  EXE                 caminho do executável
                      padrão: ../build/EquilibrationReanalysis

  KEEP_WORK           se 1, mantém diretório temporário de logs/listas
                      padrão: 0

  AUTO_JOBS           se 1, calcula JOBS automaticamente por grupo
                      padrão: 1

  JOBS                override manual. Se definido, o cálculo automático é ignorado
                      e esse valor fixo é usado em todos os grupos

  RAM_HEADROOM_PCT    porcentagem de folga sobre MemAvailable
                      padrão: 20

  RAM_MIN_HEADROOM_MB folga mínima absoluta em MiB
                      padrão: 2048

  TIME_BIN            binário usado para medir pico de RAM
                      padrão: /usr/bin/time

  RAM_DEBUG           se 1, imprime detalhes do cálculo de JOBS
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

mkdir -p "$WORK_ROOT" "$LOG_DIR"

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

get_mem_available_kb() {
  awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo 2>/dev/null || true
}

get_mem_total_kb() {
  awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null || true
}

kb_to_mib() {
  local kb="${1:-0}"
  awk -v x="$kb" 'BEGIN { printf "%.1f", x / 1024.0 }'
}

record_task_success() {
  local group_id="$1"
  local group_dir="$2"
  local manifest_file="$3"
  local json_base="$4"
  local tag="${5:-RUN}"

  local progress_file="${WORK_ROOT}/progress_${group_id}.count"
  local total_file="${WORK_ROOT}/progress_${group_id}.total"
  local lock_file="${WORK_ROOT}/progress_${group_id}.lock"

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

    local GREEN_BOLD='\033[1;32m'
    local CYAN='\033[0;36m'
    local YELLOW='\033[1;33m'
    local RESET='\033[0m'

    if [[ "$tag" == "BENCH" ]]; then
      printf "[%b%s/%s%b] %b[BENCH]%b Pasta: %s\n" \
        "$GREEN_BOLD" "$done" "$total" "$RESET" \
        "$YELLOW" "$RESET" "$group_dir"
    else
      printf "[%b%s/%s%b] Pasta: %s\n" \
        "$GREEN_BOLD" "$done" "$total" "$RESET" "$group_dir"
    fi

    printf "           Arquivo: %b%s%b\n" \
      "$CYAN" "$json_base" "$RESET"
  } 200>"$lock_file"
}

compute_jobs_from_peak() {
  local peak_kb="$1"

  local mem_available_kb mem_total_kb
  mem_available_kb="$(get_mem_available_kb)"
  mem_total_kb="$(get_mem_total_kb)"

  if [[ -z "$mem_available_kb" || ! "$mem_available_kb" =~ ^[0-9]+$ ]]; then
    echo "[WARN] Não consegui ler MemAvailable. Usando 1 JOB." >&2
    echo 1
    return 0
  fi

  if [[ -z "$mem_total_kb" || ! "$mem_total_kb" =~ ^[0-9]+$ ]]; then
    mem_total_kb="$mem_available_kb"
  fi

  local reserve_pct_kb reserve_min_kb reserve_kb usable_kb jobs_by_mem
  reserve_pct_kb=$(( mem_available_kb * RAM_HEADROOM_PCT / 100 ))
  reserve_min_kb=$(( RAM_MIN_HEADROOM_MB * 1024 ))

  reserve_kb="$reserve_pct_kb"
  if (( reserve_kb < reserve_min_kb )); then
    reserve_kb="$reserve_min_kb"
  fi

  usable_kb=$(( mem_available_kb - reserve_kb ))
  if (( usable_kb < peak_kb )); then
    jobs_by_mem=1
  else
    jobs_by_mem=$(( usable_kb / peak_kb ))
  fi

  if (( jobs_by_mem < 1 )); then
    jobs_by_mem=1
  fi

  if (( jobs_by_mem > DEFAULT_JOBS )); then
    jobs_by_mem="$DEFAULT_JOBS"
  fi

  if [[ "$RAM_DEBUG" == "1" ]]; then
    echo "[DEBUG RAM] peak_kb          = $peak_kb"
    echo "[DEBUG RAM] mem_available_kb = $mem_available_kb"
    echo "[DEBUG RAM] mem_total_kb     = $mem_total_kb"
    echo "[DEBUG RAM] reserve_pct_kb   = $reserve_pct_kb"
    echo "[DEBUG RAM] reserve_min_kb   = $reserve_min_kb"
    echo "[DEBUG RAM] reserve_kb       = $reserve_kb"
    echo "[DEBUG RAM] usable_kb        = $usable_kb"
    echo "[DEBUG RAM] jobs_by_mem      = $jobs_by_mem"
  fi

  echo "$jobs_by_mem"
}

benchmark_one_file() {
  local json_file="$1"
  local L="$2"
  local PP0="$3"
  local seed="$4"
  local type_perc="$5"
  local K="$6"
  local NT="$7"
  local dim="$8"
  local num_colors="$9"
  local rho="${10}"
  local P0="${11}"

  local rss_file
  rss_file="$(mktemp "${WORK_ROOT}/bench_rss.XXXXXX")"

  local log_file="${LOG_DIR}/bench_$(sanitize_id "$(basename "${json_file%.json}")").log"

  (
    cd "$REPO_ROOT"
    "$TIME_BIN" -f "%M" -o "$rss_file" \
      "$EXE" \
      "$L" "$PP0" "$seed" "$type_perc" "$K" "$NT" "$dim" "$num_colors" "$rho" "$P0" \
      "${EXTRA_ARGS[@]}"
  ) > "$log_file" 2>&1

  local peak_kb
  peak_kb="$(tr -d '[:space:]' < "$rss_file")"
  rm -f "$rss_file"

  if [[ -z "$peak_kb" || ! "$peak_kb" =~ ^[0-9]+$ ]]; then
    echo "[ERRO] Falha ao medir pico de RAM em: $json_file" >&2
    return 1
  fi

  echo "$peak_kb"
}

run_one_file() {
  local json_base="$1"
  local manifest_file="$2"
  local group_id="$3"
  local group_dir="$4"
  local log_file="$5"
  shift 5

  (
    cd "$REPO_ROOT"
    "$EXE" "$@" "${EXTRA_ARGS[@]}"
  ) > "$log_file" 2>&1

  record_task_success "$group_id" "$group_dir" "$manifest_file" "$json_base" "RUN"
}

process_group() {
  local group_rel="$1"
  local group_id="$2"
  local group_dir="$3"
  local data_dir="$4"
  local manifest_file="$5"
  local L="$6"
  local NT="$7"
  local K="$8"
  local rho="$9"
  local dim="${10}"
  local num_colors="${11}"
  local type_perc="${12}"

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
    return 0
  fi

  local total_pending
  total_pending="$(wc -l < "$pending_list" | tr -d ' ')"
  echo "0" > "${WORK_ROOT}/progress_${group_id}.count"
  echo "$total_pending" > "${WORK_ROOT}/progress_${group_id}.total"

  local group_jobs use_auto_jobs
  group_jobs="$JOBS"
  use_auto_jobs=0

  if [[ "$AUTO_JOBS" == "1" && "$JOBS_WAS_SET" != "1" ]]; then
    use_auto_jobs=1
  fi

  local benchmarked_base=""
  if (( use_auto_jobs == 1 )); then
    if [[ ! -x "$TIME_BIN" ]]; then
      echo "[WARN] ${TIME_BIN} não encontrado. Usando 1 JOB neste grupo." >&2
      group_jobs=1
    else
      local first_json first_base seed P0 PP0 peak_kb mem_avail_kb mem_total_kb
      first_json="$(head -n 1 "$pending_list")"
      first_base="$(basename "$first_json")"

      [[ "$first_base" =~ seed_([0-9]+) ]] || {
        echo "[ERRO] Seed inválido no benchmark do grupo: $group_dir" >&2
        return 1
      }
      seed="${BASH_REMATCH[1]}"

      [[ "$first_base" =~ P0_([0-9]+([.][0-9]+)?)_p0_([0-9]+([.][0-9]+)?) ]] || {
        echo "[ERRO] P0/p0 inválido no benchmark do grupo: $group_dir" >&2
        return 1
      }
      P0="${BASH_REMATCH[1]}"
      PP0="${BASH_REMATCH[3]}"

      echo "[BENCH] Pasta: $group_dir"
      echo "        Arquivo: $first_base"

      if ! peak_kb="$(benchmark_one_file "$first_json" "$L" "$PP0" "$seed" "$type_perc" "$K" "$NT" "$dim" "$num_colors" "$rho" "$P0")"; then
        echo "[ERRO] Benchmark falhou para o grupo: $group_dir" >&2
        return 1
      fi

      group_jobs="$(compute_jobs_from_peak "$peak_kb")"
      mem_avail_kb="$(get_mem_available_kb)"
      mem_total_kb="$(get_mem_total_kb)"

      echo "[BENCH] Pico de RAM : $(kb_to_mib "$peak_kb") MiB"
      if [[ -n "$mem_total_kb" ]]; then
        echo "[BENCH] RAM total   : $(kb_to_mib "$mem_total_kb") MiB"
      fi
      if [[ -n "$mem_avail_kb" ]]; then
        echo "[BENCH] RAM livre   : $(kb_to_mib "$mem_avail_kb") MiB"
      fi
      echo "[BENCH] JOBS grupo  : $group_jobs"
      echo

      benchmarked_base="$first_base"
      record_task_success "$group_id" "$group_dir" "$manifest_file" "$first_base" "BENCH"
    fi
  else
    echo "[INFO] Pasta: $group_dir"
    if [[ "$JOBS_WAS_SET" == "1" ]]; then
      echo "[INFO] JOBS manual usado neste grupo: $group_jobs"
    else
      echo "[INFO] AUTO_JOBS desativado. JOBS usado neste grupo: $group_jobs"
    fi
    echo
  fi

  local running=0
  local group_status=0

  while IFS= read -r json_file; do
    local base seed P0 PP0 log_file

    base="$(basename "$json_file")"

    if [[ -n "$benchmarked_base" && "$base" == "$benchmarked_base" ]]; then
      continue
    fi

    [[ "$base" =~ seed_([0-9]+) ]] || {
      echo "[WARN] Arquivo sem seed reconhecível, pulando: $json_file" >&2
      continue
    }
    seed="${BASH_REMATCH[1]}"

    [[ "$base" =~ P0_([0-9]+([.][0-9]+)?)_p0_([0-9]+([.][0-9]+)?) ]] || {
      echo "[WARN] Arquivo sem padrão P0/p0 reconhecível, pulando: $json_file" >&2
      continue
    }
    P0="${BASH_REMATCH[1]}"
    PP0="${BASH_REMATCH[3]}"

    log_file="${LOG_DIR}/task_$(sanitize_id "${base%.json}").log"

    echo "[QUEUE] Pasta: ${group_dir} | Arquivo: ${base}"

    run_one_file \
      "$base" "$manifest_file" "$group_id" "$group_dir" "$log_file" \
      "$L" "$PP0" "$seed" "$type_perc" "$K" "$NT" "$dim" "$num_colors" "$rho" "$P0" &

    running=$((running + 1))

    if (( running >= group_jobs )); then
      if ! wait -n; then
        group_status=1
      fi
      running=$((running - 1))
    fi
  done < "$pending_list"

  while (( running > 0 )); do
    if ! wait -n; then
      group_status=1
    fi
    running=$((running - 1))
  done

  if (( group_status != 0 )); then
    echo "[ERRO] Houve falha em uma ou mais tarefas do grupo: $group_dir" >&2
    return 1
  fi

  return 0
}

echo "[INFO] REPO_ROOT          = $REPO_ROOT"
echo "[INFO] RAW_ROOT           = $RAW_ROOT"
echo "[INFO] EXE                = $EXE"
echo "[INFO] AUTO_JOBS          = $AUTO_JOBS"
echo "[INFO] JOBS_WAS_SET       = $JOBS_WAS_SET"
echo "[INFO] JOBS(default/fixo) = $JOBS"
echo "[INFO] RAM_HEADROOM_PCT   = $RAM_HEADROOM_PCT"
echo "[INFO] RAM_MIN_HEADROOM_MB= $RAM_MIN_HEADROOM_MB"
echo "[INFO] CLEAR_ALL          = $CLEAR_ALL"
echo "[INFO] WORK_ROOT          = $WORK_ROOT"
echo

overall_status=0
groups_seen=0
groups_with_pending=0

while IFS= read -r -d '' data_dir; do
  rho_dir="$(basename "$(dirname "$data_dir")")"
  k_dir="$(basename "$(dirname "$(dirname "$data_dir")")")"
  nt_dir="$(basename "$(dirname "$(dirname "$(dirname "$data_dir")")")")"
  nt_mode_dir="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$data_dir")")")")")"
  l_dir="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$data_dir")")")")")")"
  dim_dir="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$data_dir")")")")")")")"
  nc_dir="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$data_dir")")")")")")")")"

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

  group_dir="$(dirname "$data_dir")"
  manifest_file="${group_dir}/process_file.txt"
  group_rel="${group_dir#${SOP_ROOT_REAL}/}"
  group_id="$(sanitize_id "$group_rel")"

  groups_seen=$((groups_seen + 1))

  if find "$data_dir" -maxdepth 1 -type f -name '*.json' | grep -q .; then
    groups_with_pending=$((groups_with_pending + 1))
  fi

  if ! process_group \
    "$group_rel" \
    "$group_id" \
    "$group_dir" \
    "$data_dir" \
    "$manifest_file" \
    "$L" \
    "$NT" \
    "$K" \
    "$rho" \
    "$dim" \
    "$num_colors" \
    "$type_perc"
  then
    overall_status=1
  fi

done < <(find "$RAW_ROOT" -type d -name data -print0 | sort -z)

echo
echo "[INFO] Grupos encontrados: $groups_seen"
echo "[INFO] Grupos inspecionados com pasta data: $groups_with_pending"

if (( overall_status != 0 )); then
  echo "[ERRO] Uma ou mais tarefas falharam. Veja os logs em: $LOG_DIR" >&2
  exit 1
fi

echo "[DONE] Reprocessamento concluído."
echo "[DONE] Logs em: $LOG_DIR"
