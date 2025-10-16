#!/usr/bin/env python3
import argparse, os, shlex, stat, sys, re, textwrap

TEMPLATE = r"""#!/usr/bin/env bash
set -euo pipefail

cd ..

BIN="./build/SOP"
if [[ ! -x "$BIN" ]]; then
  echo "Erro: binário $BIN não encontrado ou sem permissão de execução." >&2
  exit 2
fi

# ====== parâmetros fixados pelo gerador (preenchidos pelo Python) ======
type_perc=__TYPE_PERC__
dim=__DIM__
num_colors=__NUM_COLORS__
p0=__P0__
rho_val=__RHO_VAL__
num_execs=__NUM_EXECS__
dsu_flag=__DSU_FLAG__           # deve ser "True" ou "False" para o C++
ram_limit_gb=__RAM_LIMIT_GB__
seed_regex='__SEED_REGEX__'
seed_group_idx=__SEED_GROUP_IDX__

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
__JOBS_BLOCK__

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
"""

def detect_total_ram_gb():
    try:
        with open("/proc/meminfo","r") as f:
            txt = f.read()
        m = re.search(r"MemTotal:\s+(\d+)\s+kB", txt)
        if m:
            kb = int(m.group(1))
            return kb / (1024**2)  # kB -> GB
    except Exception:
        pass
    return 0.0

def to_bool_str_for_cpp_pascal(val: str) -> str:
    s = str(val).strip().lower()
    if s in ("true","1","yes","y","t"):
        return "True"
    if s in ("false","0","no","n","f"):
        return "False"
    raise ValueError(f"Valor inválido para DSU_calculate: {val} (use True/False)")

def main():
    ap = argparse.ArgumentParser(
        description="Gera *apenas* o .sh em ./shells/ para medir tempo/RAM do SOP mantendo seed=-1 (extraída do log)."
    )
    ap.add_argument("--L", nargs="+", required=True, help="Lista de L (ex.: --L 128 256 512 1024)")
    ap.add_argument("--k_list", nargs="+", required=True, help="Lista de k por L (mesmo tamanho de --L)")
    ap.add_argument("--Nt_list", nargs="+", required=True, help="Lista de Nt por L (mesmo tamanho de --L)")
    ap.add_argument("--mem_list", nargs="*", help="Memória estimada por L (GB), mesmo tamanho de --L)")
    ap.add_argument("--mem_per_job_gb", type=float, default=None, help="Memória estimada uniforme por job (GB) se --mem_list não for fornecida")
    ap.add_argument("--ram_limit_gb", type=float, default=None, help="Limite de RAM global (GB); default = 90%% da RAM total")
    ap.add_argument("--p0", type=float, required=True)
    ap.add_argument("--type_perc", choices=["bond", "node"], required=True)
    ap.add_argument("--dim", type=int, choices=[2,3], required=True)
    ap.add_argument("--num_colors", type=int, required=True)
    ap.add_argument("--rho_val", type=float, required=True)
    ap.add_argument("--DSU_calculate", type=str, required=True, help="True/False (vai virar exatamente True/False no C++)")
    ap.add_argument("--Num_execs", type=int, default=1)
    ap.add_argument("--out", default="run_stats.sh", help="Nome do shell (salvo em ./shells/)")
    ap.add_argument("--seed_regex", type=str, default="", help="Regex custom para extrair seed do log (opcional). Ex.: 'SEED[:=]\\s*([0-9]+)'")
    ap.add_argument("--seed_group_idx", type=int, default=1, help="Índice do grupo capturado na regex custom (se aplicável)")
    args = ap.parse_args()

    if not (len(args.L) == len(args.k_list) == len(args.Nt_list)):
        print("Erro: --L, --k_list e --Nt_list devem ter o MESMO tamanho.", file=sys.stderr)
        sys.exit(2)

    # mem_list
    if args.mem_list:
        if len(args.mem_list) != len(args.L):
            print("Erro: --mem_list deve ter o MESMO tamanho de --L.", file=sys.stderr)
            sys.exit(2)
        mem_list = [float(x) for x in args.mem_list]
    else:
        mem_per = 4.0 if args.mem_per_job_gb is None else float(args.mem_per_job_gb)
        mem_list = [mem_per] * len(args.L)

    # limite RAM
    if args.ram_limit_gb is None:
        total = detect_total_ram_gb()
        ram_limit = max(1.0, 0.9 * total)
    else:
        ram_limit = float(args.ram_limit_gb)

    # DSU flag → "True"/"False" (PascalCase), conforme seu C++
    dsu_flag = to_bool_str_for_cpp_pascal(args.DSU_calculate)

    # bloco de jobs: linhas "L k Nt mem_gb"
    jobs_lines = []
    for L, k, Nt, mem_gb in zip(args.L, args.k_list, args.Nt_list, mem_list):
        for _ in range(int(args.Num_execs)):
            jobs_lines.append(
                f'echo {shlex.quote(str(L))} {shlex.quote(str(k))} {shlex.quote(str(Nt))} {mem_gb} >> "$TMP_BASE"'
            )
    jobs_block = "\n".join(jobs_lines)

    script = TEMPLATE
    script = script.replace("__TYPE_PERC__", shlex.quote(args.type_perc))
    script = script.replace("__DIM__", str(int(args.dim)))
    script = script.replace("__NUM_COLORS__", str(int(args.num_colors)))
    script = script.replace("__P0__", "{:.10g}".format(args.p0))
    script = script.replace("__RHO_VAL__", "{:.10g}".format(args.rho_val))
    script = script.replace("__NUM_EXECS__", str(int(args.Num_execs)))
    script = script.replace("__DSU_FLAG__", dsu_flag)  # exatamente "True"/"False"
    script = script.replace("__RAM_LIMIT_GB__", "{:.6g}".format(ram_limit))
    seed_regex_escaped = args.seed_regex.replace("'", "'\"'\"'")
    script = script.replace("__SEED_REGEX__", seed_regex_escaped)
    script = script.replace("__SEED_GROUP_IDX__", str(int(args.seed_group_idx)))
    script = script.replace("__JOBS_BLOCK__", jobs_block)

    out_name = args.out if args.out.endswith(".sh") else (args.out + ".sh")
    out_path = os.path.join("../shells", out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        f.write(textwrap.dedent(script))
    os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IEXEC)
    print(f"Shell gerado em: {out_path}")
    print(f"(limite de RAM configurado: {ram_limit:.2f} GB)")
    print(f"(DSU flag -> C++): {dsu_flag}")
    if args.seed_regex:
        print(f"(Regex custom de seed): {args.seed_regex} (grupo {args.seed_group_idx})")

if __name__ == "__main__":
    main()
