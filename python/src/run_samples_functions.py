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
