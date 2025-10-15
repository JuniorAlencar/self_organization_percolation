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
dsu_flag=__DSU_FLAG__
ram_limit_gb=__RAM_LIMIT_GB__

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
__JOBS_BLOCK__

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
"""

def detect_total_ram_gb():
    """Lê /proc/meminfo e retorna RAM total em GB (float)."""
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

def main():
    ap = argparse.ArgumentParser(
        description="Gera *apenas* o .sh em ./shells/ para medir tempo/RAM do SOP (seed=-1) em paralelo, com k/Nt/mem por L."
    )
    ap.add_argument("--L", nargs="+", required=True, help="Lista de L (ex.: --L 128 256 512 1024)")
    ap.add_argument("--k_list", nargs="+", required=True, help="Lista de k por L (mesmo tamanho de --L)")
    ap.add_argument("--Nt_list", nargs="+", required=True, help="Lista de Nt por L (mesmo tamanho de --L)")
    ap.add_argument("--mem_list", nargs="*", help="Memória estimada por L (GB), mesmo tamanho de --L")
    ap.add_argument("--mem_per_job_gb", type=float, default=None, help="Memória estimada uniforme por job (GB) se --mem_list não for fornecida")
    ap.add_argument("--ram_limit_gb", type=float, default=None, help="Limite de RAM global (GB); default = 90% da RAM total")
    ap.add_argument("--p0", type=float, required=True)
    ap.add_argument("--type_perc", choices=["bond", "node"], required=True)
    ap.add_argument("--dim", type=int, choices=[2,3], required=True)
    ap.add_argument("--num_colors", type=int, required=True)
    ap.add_argument("--rho_val", type=float, required=True)
    ap.add_argument("--DSU_calculate", type=str, choices=["True","False","true","false","1","0"], default="False")
    ap.add_argument("--Num_execs", type=int, default=1)
    ap.add_argument("--out", default="run_stats.sh", help="Nome do shell (será salvo em ./shells/)")
    args = ap.parse_args()

    # validações
    if not (len(args.L) == len(args.k_list) == len(args.Nt_list)):
        print("Erro: --L, --k_list e --Nt_list devem ter o MESMO tamanho.", file=sys.stderr)
        sys.exit(2)

    # memória estimada por L
    if args.mem_list:
        if len(args.mem_list) != len(args.L):
            print("Erro: --mem_list deve ter o MESMO tamanho de --L.", file=sys.stderr)
            sys.exit(2)
        mem_list = [float(x) for x in args.mem_list]
    else:
        mem_per = 4.0 if args.mem_per_job_gb is None else float(args.mem_per_job_gb)
        mem_list = [mem_per] * len(args.L)

    # limite de RAM global
    if args.ram_limit_gb is None:
        total = detect_total_ram_gb()
        ram_limit = max(1.0, 0.9 * total)  # 90% do total (mínimo 1GB)
    else:
        ram_limit = float(args.ram_limit_gb)

    dsu_flag = "1" if str(args.DSU_calculate).lower() in ("true","1") else "0"

    # bloco de jobs: linhas "L k Nt mem_gb" repetidas Num_execs vezes
    jobs_lines = []
    for L, k, Nt, mem_gb in zip(args.L, args.k_list, args.Nt_list, mem_list):
        for _ in range(int(args.Num_execs)):
            jobs_lines.append(
                f'echo {shlex.quote(str(L))} {shlex.quote(str(k))} {shlex.quote(str(Nt))} {mem_gb} >> "$TMP_BASE"'
            )
    jobs_block = "\n".join(jobs_lines)

    # substituições seguras no template
    script = TEMPLATE
    script = script.replace("__TYPE_PERC__", shlex.quote(args.type_perc))
    script = script.replace("__DIM__", str(int(args.dim)))
    script = script.replace("__NUM_COLORS__", str(int(args.num_colors)))
    script = script.replace("__P0__", "{:.10g}".format(args.p0))
    script = script.replace("__RHO_VAL__", "{:.10g}".format(args.rho_val))
    script = script.replace("__NUM_EXECS__", str(int(args.Num_execs)))
    script = script.replace("__DSU_FLAG__", "1" if dsu_flag == "1" else "0")
    script = script.replace("__RAM_LIMIT_GB__", "{:.6g}".format(ram_limit))
    script = script.replace("__JOBS_BLOCK__", jobs_block)

    # salva em ./shells/<out>
    out_name = args.out if args.out.endswith(".sh") else (args.out + ".sh")
    out_path = os.path.join("../shells", out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        f.write(textwrap.dedent(script))
    os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IEXEC)
    print(f"Shell gerado em: {out_path}")
    print(f"(limite de RAM configurado: {ram_limit:.2f} GB)")

if __name__ == "__main__":
    main()
