#!/usr/bin/env bash
set -euo pipefail

cd ..

BIN="./build/SOP"
if [[ ! -x "$BIN" ]]; then
  echo "Erro: binário $BIN não encontrado ou sem permissão de execução." >&2
  exit 2
fi

# ====== parâmetros fixados pelo gerador (preenchidos pelo Python) ======
type_perc=node
dim=2
num_colors=1
p0=0.5
rho_val=1
num_execs=1
dsu_flag=True           # deve ser "True" ou "False" para o C++
ram_limit_gb=56.3653
seed_regex=''
seed_group_idx=1

# ====== controle de threads (evita multiplicar consumo)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# ====== saída ======
OUTDIR="./RunningStatisticals"
mkdir -p "$OUTDIR"

outfile_for() {
  local L="$1"
  echo "$OUTDIR/L_${L}_dim_${dim}_num_colors_${num_colors}.txt"
}

init_outfile() {
  local f="$1"
  [[ -f "$f" ]] || echo "Seed Rho MemoryUsed TimeRun" > "$f"
}

detect_time_cmd() {
  if command -v gtime >/dev/null 2>&1; then echo "gtime"; return; fi
  if command -v /usr/bin/time >/dev/null 2>&1 && /usr/bin/time --version 2>&1 | grep -qi gnu; then echo "/usr/bin/time"; return; fi
  if command -v time >/dev/null 2>&1 && time --version 2>&1 | grep -qi gnu; then echo "time"; return; fi
  echo ""
}

# tenta extrair seed do log com várias heurísticas
extract_seed() {
  local log="$1"
  local rgx="$2"
  local grp="$3"
  local s=""

  # 1) regex custom (se usuário passou)
  if [[ -n "$rgx" ]]; then
    s="$(printf "%s" "$log" | perl -0777 -ne 'print $& if m/'"$rgx"'/s' 2>/dev/null || true)"
    if [[ -n "$s" ]]; then
      s="$(printf "%s" "$s" | grep -Eo '(-?0x[0-9A-Fa-f]+|-?[0-9]+)' | head -n1 || true)"
      [[ -n "$s" ]] && { printf "%s" "$s"; return; }
    fi
  fi

  # 2) padrões comuns próximos a "seed"
  s="$(printf "%s" "$log" \
      | grep -Eio '(seed|rng|random(_)?seed|mt19937)[^0-9A-Fa-f-]*(-?0x[0-9A-Fa-f]+|-?[0-9]+)' \
      | head -n1 \
      | grep -Eo '(-?0x[0-9A-Fa-f]+|-?[0-9]+)' || true)"
  if [[ -n "$s" ]]; then printf "%s" "$s"; return; fi

  s="$(printf "%s" "$log" \
      | grep -Eio '(seed|rng|random(_)?seed)\s*[:=]\s*(-?0x[0-9A-Fa-f]+|-?[0-9]+)' \
      | head -n1 \
      | grep -Eo '(-?0x[0-9A-Fa-f]+|-?[0-9]+)' || true)"
  if [[ -n "$s" ]]; then printf "%s" "$s"; return; fi

  # 3) fallback: inteiro "grande"
  s="$(printf "%s" "$log" | grep -Eo '(-?[0-9]{6,})' | head -n1 || true)"
  [[ -z "$s" ]] && s="UNKNOWN"
  printf "%s" "$s"
}

# Roda UMA realização e grava: Seed Rho RAM(GB) Tempo(s)
# Args: L k Nt
run_one() {
  local L="$1"
  local k="$2"
  local Nt="$3"

  local out tf_time tf_log TIME_CMD EXIT_K secs mem_gb
  out="$(outfile_for "$L")"
  init_outfile "$out"

  # seed = -1 (C++ gera internamente)
  cmd=("$BIN" "$L" "$p0" "-1" "$type_perc" "$k" "$Nt" "$dim" "$num_colors" "$rho_val" "$dsu_flag")

  tf_time="$(mktemp)"; tf_log="$(mktemp)"
  trap 'rm -f "$tf_time" "$tf_log"' RETURN

  TIME_CMD="$(detect_time_cmd)"

  # mede wallclock manual também, para o caso do time não escrever nada
  local start_ns end_ns
  start_ns=$(date +%s%N)
  if [[ -n "$TIME_CMD" ]]; then
    LC_ALL=C "$TIME_CMD" -f '%e %M %x' -o "$tf_time" "${cmd[@]}" >"$tf_log" 2>&1 || true
  else
    "${cmd[@]}" >"$tf_log" 2>&1 || true
  fi
  end_ns=$(date +%s%N)

  secs=""
  mem_gb=""

  if [[ -s "$tf_time" ]]; then
    read -r SECS_K RSS_K EXIT_K < "$tf_time" || true
    : "${SECS_K:=0}"; : "${RSS_K:=0}"; : "${EXIT_K:=0}"
    secs="$(awk -v s="$SECS_K" 'BEGIN{printf "%.6f", s+0.0}')"
    mem_gb="$(awk -v k="$RSS_K" 'BEGIN{printf "%.6f", k/1048576}')"
  fi

  # Fallback se time não escreveu (ou escreveu lixo)
  if [[ -z "${secs:-}" || "$secs" = "0" ]]; then
    local dt_ms=$(( (end_ns - start_ns)/1000000 ))
    secs=$(awk -v ms="$dt_ms" 'BEGIN{printf "%.6f", ms/1000.0}')
  fi
  if [[ -z "${mem_gb:-}" ]]; then
    mem_gb="NaN"
  fi

  # tenta extrair seed do log
  seed_detected="$(extract_seed "$(cat "$tf_log")" "$seed_regex" "$seed_group_idx")"

  # se o binário retornou erro, marca FAIL:<seed>
  if grep -qE 'Segmentation fault|Aborted|exception|usage:|invalid|erro|error' "$tf_log"; then
    EXIT_K=1
  fi
  if [[ "${EXIT_K:-0}" != "0" ]]; then
    seed_detected="FAIL:${seed_detected}"
  fi

  echo "$seed_detected $rho_val $mem_gb $secs" >> "$out"
}

# ====== Grupos por memória estimada ======
# Colunas: L k Nt mem_gb
TMP_BASE="$(mktemp)"; trap 'rm -f "$TMP_BASE" $TMP_BASE.*' EXIT

# Bloco JOBS (preenchido pelo Python):
echo 128 1.0e-05 1600 0.5 >> "$TMP_BASE"
echo 256 3.0e-06 6500 1.0 >> "$TMP_BASE"
echo 512 8.0e-07 26000 3.0 >> "$TMP_BASE"
echo 1024 2.0e-07 100000 12.0 >> "$TMP_BASE"

# nproc e chaves de memória
nproc_sys=$(nproc)
mem_keys=$(awk '{print $4}' "$TMP_BASE" | sort -u)

export type_perc dim num_colors p0 rho_val num_execs dsu_flag ram_limit_gb OUTDIR seed_regex seed_group_idx
export -f run_one outfile_for init_outfile detect_time_cmd extract_seed

for mem in $mem_keys; do
  grp="$TMP_BASE.$mem"
  awk -v m="$mem" '{if($4==m) print $0}' "$TMP_BASE" > "$grp"

  # JOBS = min(nproc, floor(ram_limit_gb / mem_gb)), pelo menos 1
  jobs_calc=$(awk -v Rg="$ram_limit_gb" -v mg="$mem" 'BEGIN{ j=int(Rg/mg); if(j<1) j=1; print j; }')
  if [[ $jobs_calc -gt $nproc_sys ]]; then jobs_calc=$nproc_sys; fi

  echo ">> Rodando grupo mem≈${mem}GB com -j $jobs_calc (limite RAM ${ram_limit_gb}GB, nproc ${nproc_sys})"

  if command -v parallel >/dev/null 2>&1; then
    parallel --bar -j "$jobs_calc" --colsep ' ' run_one {1} {2} {3} :::: "$grp"
  else
    # xargs: passa L k Nt (ignora a 4ª coluna mem)
    awk '{print $1, $2, $3}' "$grp" \
    | xargs -P "$jobs_calc" -n 3 -I{} bash -lc 'set -- {}; run_one "$1" "$2" "$3"'
  fi
done

echo "Concluído."
