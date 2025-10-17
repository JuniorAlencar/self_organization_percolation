import os
import stat
import textwrap
import numpy as np

# Helper to convert bool variable in python to C++
def as_cli_bool(x, numeric=False) -> str:
    """
    Converte bool vindo do Python/usuário para um token aceito pelo C++ parse_bool.
    numeric=True -> "1"/"0"; numeric=False -> "true"/"false".
    """
    # Python bool/int
    if isinstance(x, bool):
        return ("1" if x else "0") if numeric else ("true" if x else "false")
    if isinstance(x, (int,)):
        return ("1" if x != 0 else "0") if numeric else ("true" if x != 0 else "false")
    # String
    if isinstance(x, str):
        s = x.strip().lower()
        ok = {"1","0","true","false","t","f","yes","no","y","n"}
        if s not in ok:
            raise ValueError(f"Valor inválido para DCU_props: {x!r}")
        # padroniza
        if numeric:
            return "1" if s in {"1","true","t","yes","y"} else "0"
        else:
            return "true" if s in {"1","true","t","yes","y"} else "false"
    raise TypeError(f"Tipo não suportado para DCU_props: {type(x)}")

def shell_data(L:int, type_perc:str, p0:float, 
               seed:int, k:float, NT:int, dim:int, num_colors:int, 
               num_runs:int, rho:list, exec_name:str,
               DCU_props: str = 'false',
               num_threads: int = 4,
               multi: bool = False):
    
    DCU_token = as_cli_bool(DCU_props, numeric=False)  # ou numeric=True para "1"/"0"
    
    """
    Generate a shell script to run SOP multiple times.
    If multi=True, use GNU parallel (with progress bar); otherwise run sequentially with a bash progress bar.
    'num_threads' is baked into the script (no need to pass JOBS at runtime).
    """
    if dim not in (2, 3):
        return "please, enter with dim = 2 or 3"

    os.makedirs("../shells", exist_ok=True)
    print("Creating shell script file in ../shells/")

    rho_list = " ".join(map(str, rho))

    if multi:
        # ---------- parallel version (GNU parallel) ----------
        script = f"""\
#!/usr/bin/env bash
set -euo pipefail

# --- go to project root ---
cd ..

# --- parameters exported for parallel ---
num_runs={num_runs}
rho=({rho_list})
L={L}
p0={p0}
seed={seed}
type="{type_perc}"
k={k}
NT={NT}
dim={dim}
num_colors={num_colors}
DCU={DCU_token}

# --- fixed number of workers (baked into the script) ---
JOBS={num_threads}

# --- check GNU parallel ---
if ! command -v parallel >/dev/null 2>&1; then
  echo "[ERROR] 'parallel' not found. Install it (e.g., sudo apt-get install parallel) or set multi=False."
  exit 1
fi

export L p0 seed type k NT dim num_colors DCU

# --- run Cartesian product: rho × runs, with progress bar ---
parallel -j "$JOBS" --bar --halt soon,fail=1 '
  RHO={{1}}
  RUN={{2}}
  ./build/SOP "$L" "$p0" "$seed" "$type" "$k" "$NT" "$dim" "$num_colors" "$RHO" "$DCU"
' ::: "${{rho[@]}}" ::: $(seq 1 "$num_runs")
"""
    else:
        # ---------- sequential version (custom bash progress bar) ----------
        script = f"""\
#!/usr/bin/env bash
set -euo pipefail

# --- go to project root ---
cd ..

# --- parameters ---
num_runs={num_runs}
rho=({rho_list})
L={L}
p0={p0}
seed={seed}
type="{type_perc}"
k={k}
NT={NT}
dim={dim}
num_colors={num_colors}
DCU={DCU_token}

# --- pretty progress bar (single-line) ---
progress_bar() {{
  local curr="$1" total="$2" width="${{3:-40}}"
  local pct=$(( 100 * curr / total ))
  local filled=$(( width * curr / total ))
  local empty=$(( width - filled ))
  local bar spaces
  printf -v bar    '%*s' "$filled"; bar=${{bar// /#}}
  printf -v spaces '%*s' "$empty"
  printf "\\r[%s%s] %3d%% (%d/%d)" "$bar" "$spaces" "$pct" "$curr" "$total"
}}

TOTAL=$(( num_runs * ${{#rho[@]}} ))
DONE=0

# VERBOSE=1 to echo each command (will break the single-line bar)
VERBOSE=${{VERBOSE:-0}}

for ((run=1; run<=num_runs; run++)); do
  for idx in "${{!rho[@]}}"; do
    RHO=${{rho[$idx]}}

    # update progress bar
    DONE=$((DONE+1))
    progress_bar "$DONE" "$TOTAL" 46

    # run SOP (suppress command echo unless VERBOSE=1)
    if [[ "$VERBOSE" -eq 1 ]]; then
      echo
      echo "./build/SOP $L $p0 $seed $type $k $NT $dim $num_colors $RHO"
      ./build/SOP "$L" "$p0" "$seed" "$type" "$k" "$NT" "$dim" "$num_colors" "$RHO" "$DCU"
    else
      ./build/SOP "$L" "$p0" "$seed" "$type" "$k" "$NT" "$dim" "$num_colors" "$RHO" "$DCU" >/dev/null
    fi
  done
done

# finish line
echo
echo "All runs completed."
"""

    script = textwrap.dedent(script)
    path = os.path.join("../shells", exec_name)
    with open(path, "w") as f:
        f.write(script)

    # make it executable
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)
    print(f"✅ Shell script created at {path}")

def custom_range(start: float, stop: float, n_points: int, ndigits: int = 8):
    if n_points <= 0:
        return []
    if n_points == 1:
        return [round(start, ndigits)]

    xs = np.linspace(start, stop, n_points, endpoint=True, dtype=float)
    xs = np.round(xs, ndigits)         # arredonda tudo
    xs[0]  = round(start, ndigits)     # força extremos exatos
    xs[-1] = round(stop, ndigits)
    return xs.tolist()


import math
from dataclasses import dataclass

@dataclass
class SOPParams:
    L: int
    d: int
    n_c: int
    epsilon0: float
    kappa0: float
    beta: float
    NT_raw: float
    NT: int
    k_raw: float
    k: float
    k_str: str
    notes: str

def _round_sig(x: float, sig: int = 2) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    e = math.floor(math.log10(abs(x)))
    factor = 10 ** (e - sig + 1)
    return round(x / factor) * factor

def _k_round_and_format(x: float, mantissa_decimals: int = 0):
    """
    Arredonda 'x' para mantissa científica inteira e retorna (k_float, k_str).
    Ex.: 1.22e-05 -> (1e-05, '1e-05')    [mantissa_decimals=0]
         8.62e-06 -> (9e-06,  '9e-06')
         9.6e-07  -> (1e-06,  '1e-06')
    Se mantissa_decimals=2, string vira '1.00e-05', '9.00e-06', etc.
    """
    if x == 0 or not math.isfinite(x):
        s = f"{x:.{mantissa_decimals}e}"
        return x, s

    sign = -1 if x < 0 else 1
    a = abs(x)
    exp = math.floor(math.log10(a))
    mant = a / (10 ** exp)               # 1 <= mant < 10
    mant_i = int(round(mant))            # inteiro mais próximo

    # Ajuste do caso 9.5.. -> 10 -> 1e(exp+1)
    if mant_i == 0:
        exp -= 1; mant_i = 1
    if mant_i == 10:
        mant_i = 1; exp += 1

    k_float = sign * mant_i * (10 ** exp)
    if mantissa_decials := mantissa_decimals:  # só para nome curto
        k_str = f"{sign*mant_i:.{mantissa_decials}f}e{exp:+d}".replace("+0", "+").replace("-0", "-")
    else:
        # sem casas na mantissa (“inteira”)
        k_str = f"{sign*mant_i:d}e{exp:+d}".replace("+0", "+").replace("-0", "-")

def sop_choose_params(L: int,
                      n_c: int,
                      d: int = 3,
                      beta: float = 0.5,
                      epsilon0: float = 0.05,
                      kappa0: float = 0.010) -> SOPParams:
    """
    Calcula N_T e k para o modelo SOP multicolor com N_T fixo por L.

    Definições:
      - Frente total por passo (global, indep. de n_c):
          N_T_raw = epsilon0 * L^(d-1)
      - Arredondamento de N_T: 2 algarismos significativos (p.ex., 819->820, 3277->3300)
      - Ganho:
          k_raw = (kappa0 * n_c**beta) / N_T
      - Arredondamento de k: mantissa científica inteira (p.ex., 1.22e-05->1.00e-05)

    Recomendações usuais:
      - epsilon0 ∈ [0.03, 0.06]  (fração da área da frente)
      - kappa0   ∈ [0.007, 0.015] (ganho adimensional alvo κ = k·N_T)
      - beta = 0.0 (sem compensação), 0.5 (**recomendado**, compensação parcial) ou 1.0 (compensação total)

    Parâmetros:
      L       : tamanho linear da rede
      n_c     : número de cores
      d       : dimensão (padrão 3, então N_T ∝ L^(d-1) = L^2)
      beta    : expoente de compensação por número de cores (default 0.5)
      epsilon0: fração da frente (default 0.05)
      kappa0  : ganho adimensional alvo (default 0.010)

    Retorna:
      SOPParams dataclass com NT_raw, NT (arredondado), k_raw, k (arredondado) e string k_str.
    """
    NT_raw = float(epsilon0) * (L ** (d - 1))
    NT = int(_round_sig(NT_raw, sig=2))  # arredonda para 2 sig figs

    # Ganho bruto e seu arredondamento especial
    k_raw = (float(kappa0) * (n_c ** float(beta))) / NT if NT > 0 else float('nan')
    k, k_str = _round_k_scientific(k_raw)

    notes = (
        f"Recomendados: epsilon0 ∈ [0.03, 0.06], kappa0 ∈ [0.007, 0.015], "
        f"beta ∈ {{0.0 (sem compensação), 0.5 (parcial, recomendado), 1.0 (total)}}. "
        f"Usado: epsilon0={epsilon0}, kappa0={kappa0}, beta={beta}. "
        f"N_T é global (não depende de n_c); k cresce com n_c^beta."
    )
    return SOPParams(L=L, d=d, n_c=n_c,
                     epsilon0=epsilon0, kappa0=kappa0, beta=beta,
                     NT_raw=NT_raw, NT=NT,
                     k_raw=k_raw, k=k, k_str=k_str,
                     notes=notes)

def sop_choose_NT_k(L: int,
                    n_c: int,
                    d: int = 3,
                    beta: float = 0.5,
                    epsilon0: float = 0.05,
                    kappa0: float = 0.010,
                    mantissa_decimals: int = 0):
    """
    Retorna (NT, k_float, k_str) com:
      NT  = round_sig(epsilon0 * L^(d-1), 2)  # ex.: 819 -> 820; 3277 -> 3300
      k   = (kappa0 * n_c**beta) / NT
      k_str = k em notação científica com mantissa inteira ('3e-05' por padrão).
    """
    NT_raw = float(epsilon0) * (L ** (d - 1))
    NT = int(_round_sig(NT_raw, sig=2))
    k_raw = (float(kappa0) * (n_c ** float(beta))) / NT if NT > 0 else float('nan')
    k_float, k_str = _k_round_and_format(k_raw, mantissa_decimals=mantissa_decimals)
    return NT, k_float, k_str
# -----------------------------
# Exemplo de uso (comentado):
# p = sop_choose_params(L=512, n_c=4)  # beta=0.5, epsilon0=0.05, kappa0=0.010
# print(p.NT, p.k, p.k_str)
