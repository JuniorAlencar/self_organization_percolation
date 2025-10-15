#!/usr/bin/env bash
set -euo pipefail

cd ..

BIN="./build/SOP"
if [[ ! -x "$BIN" ]]; then
  echo "Erro: binário $BIN não encontrado ou sem permissão de execução." >&2
  exit 2
fi

# ====== parâmetros fixados pelo gerador (preenchidos pelo Python) ======
type_perc=bond
dim=3
num_colors=2
p0=0.1
rho_val=0.5
num_execs=10
dsu_flag=0
ram_limit_gb=27.6664

# ====== controle de threads (evita multiplicar consumo)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# ====== saída ======
# Caminho ABSOLUTO baseado no diretório do próprio .sh
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

# HH:MM:SS[.m] -> segundos
hms_to_seconds() {
  awk -v t="$1" 'BEGIN{n=split(t,a,":"); s=0;
    if(n==3) s=a[1]*3600+a[2]*60+a[3];
    else if(n==2) s=a[1]*60+a[2];
    else s=a[1];
    printf "%.6f", s+0.0
  }'
}

# Extrai um inteiro de "seed" de um log arbitrário
extract_seed() {
  local txt="$1"
  local s
  s="$(printf "%s" "$txt" | grep -Eoi 'seed[^0-9-]*(-?[0-9]+)' | tail -n1 | grep -Eo '(-?[0-9]+)' || true)"
  if [[ -z "$s" ]]; then
    s="$(printf "%s" "$txt" | grep -Eo '[0-9]{4,}' | head -n1 || true)"
  fi
  [[ -z "$s" ]] && s="UNKNOWN"
  printf "%s" "$s"
}

# Roda UMA realização (seed = -1) e grava: Seed Rho RAM(GB) Tempo(s)
run_one() {
  local L="$1"
  local k="$2"
  local Nt="$3"

  local out="$(outfile_for "$L")"
  init_outfile "$out"

  # seed = -1 para gerar internamente
  cmd=("$BIN" "$L" "$p0" "-1" "$type_perc" "$k" "$Nt" "$dim" "$num_colors" "$rho_val" "$dsu_flag")

  local tf_time tf_log
  tf_time="$(mktemp)"; tf_log="$(mktemp)"
  trap 'rm -f "$tf_time" "$tf_log"' RETURN

  if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -f '%E %M %x' -o "$tf_time" "${cmd[@]}" >"$tf_log" 2>&1
    read -r WALL_K RSS_K EXIT_K < "$tf_time"
    secs="$(hms_to_seconds "$WALL_K")"
    mem_gb="$(awk -v k="$RSS_K" 'BEGIN{printf "%.6f", k/1048576}')"
    seed_detected="$(extract_seed "$(cat "$tf_log")")"
    echo "$seed_detected $rho_val $mem_gb $secs" >> "$out"
  else
    start_ns=$(date +%s%N)
    "${cmd[@]}" >"$tf_log" 2>&1
    end_ns=$(date +%s%N)
    dt_ms=$(( (end_ns - start_ns)/1000000 ))
    secs=$(awk -v ms="$dt_ms" 'BEGIN{printf "%.6f", ms/1000.0}')
    seed_detected="$(extract_seed "$(cat "$tf_log")")"
    echo "$seed_detected $rho_val NaN $secs" >> "$out"
  fi
}

# ====== Grupos por memória estimada ======
# Cada linha do bloco abaixo: L k Nt mem_gb
TMP_BASE="$(mktemp)"; trap 'rm -f "$TMP_BASE" $TMP_BASE.*' EXIT

# Bloco JOBS (preenchido pelo Python):
echo 128 1.0e-05 1600 0.5 >> "$TMP_BASE"
echo 128 1.0e-05 1600 0.5 >> "$TMP_BASE"
echo 128 1.0e-05 1600 0.5 >> "$TMP_BASE"
echo 128 1.0e-05 1600 0.5 >> "$TMP_BASE"
echo 128 1.0e-05 1600 0.5 >> "$TMP_BASE"
echo 128 1.0e-05 1600 0.5 >> "$TMP_BASE"
echo 128 1.0e-05 1600 0.5 >> "$TMP_BASE"
echo 128 1.0e-05 1600 0.5 >> "$TMP_BASE"
echo 128 1.0e-05 1600 0.5 >> "$TMP_BASE"
echo 128 1.0e-05 1600 0.5 >> "$TMP_BASE"
echo 256 3.0e-06 6500 1.0 >> "$TMP_BASE"
echo 256 3.0e-06 6500 1.0 >> "$TMP_BASE"
echo 256 3.0e-06 6500 1.0 >> "$TMP_BASE"
echo 256 3.0e-06 6500 1.0 >> "$TMP_BASE"
echo 256 3.0e-06 6500 1.0 >> "$TMP_BASE"
echo 256 3.0e-06 6500 1.0 >> "$TMP_BASE"
echo 256 3.0e-06 6500 1.0 >> "$TMP_BASE"
echo 256 3.0e-06 6500 1.0 >> "$TMP_BASE"
echo 256 3.0e-06 6500 1.0 >> "$TMP_BASE"
echo 256 3.0e-06 6500 1.0 >> "$TMP_BASE"
echo 512 8.0e-07 26000 3.0 >> "$TMP_BASE"
echo 512 8.0e-07 26000 3.0 >> "$TMP_BASE"
echo 512 8.0e-07 26000 3.0 >> "$TMP_BASE"
echo 512 8.0e-07 26000 3.0 >> "$TMP_BASE"
echo 512 8.0e-07 26000 3.0 >> "$TMP_BASE"
echo 512 8.0e-07 26000 3.0 >> "$TMP_BASE"
echo 512 8.0e-07 26000 3.0 >> "$TMP_BASE"
echo 512 8.0e-07 26000 3.0 >> "$TMP_BASE"
echo 512 8.0e-07 26000 3.0 >> "$TMP_BASE"
echo 512 8.0e-07 26000 3.0 >> "$TMP_BASE"
echo 1024 2.0e-07 100000 12.0 >> "$TMP_BASE"
echo 1024 2.0e-07 100000 12.0 >> "$TMP_BASE"
echo 1024 2.0e-07 100000 12.0 >> "$TMP_BASE"
echo 1024 2.0e-07 100000 12.0 >> "$TMP_BASE"
echo 1024 2.0e-07 100000 12.0 >> "$TMP_BASE"
echo 1024 2.0e-07 100000 12.0 >> "$TMP_BASE"
echo 1024 2.0e-07 100000 12.0 >> "$TMP_BASE"
echo 1024 2.0e-07 100000 12.0 >> "$TMP_BASE"
echo 1024 2.0e-07 100000 12.0 >> "$TMP_BASE"
echo 1024 2.0e-07 100000 12.0 >> "$TMP_BASE"

# nproc e chaves de memória
nproc_sys=$(nproc)
mem_keys=$(awk '{print $4}' "$TMP_BASE" | sort -u)

# ====== exporta variáveis para ambientes filhos (parallel/xargs) ======
export type_perc dim num_colors p0 rho_val num_execs dsu_flag ram_limit_gb OUTDIR
export -f run_one hms_to_seconds extract_seed outfile_for init_outfile

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
    xargs -P "$jobs_calc" -n 4 -a "$grp" bash -lc 'run_one "$0" "$1" "$2"'
  fi
done

echo "Concluído."
