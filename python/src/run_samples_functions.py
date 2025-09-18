import os
import stat
import textwrap
import numpy as np

def shell_data(L:int, type_perc:str, p0:float, 
               seed:int, k:float, NT:int, dim:int, num_colors:int, 
               num_runs:int, rho:list, exec_name:str,
               num_threads: int = 4,
               multi: bool = False):
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

# --- fixed number of workers (baked into the script) ---
JOBS={num_threads}

# --- check GNU parallel ---
if ! command -v parallel >/dev/null 2>&1; then
  echo "[ERROR] 'parallel' not found. Install it (e.g., sudo apt-get install parallel) or set multi=False."
  exit 1
fi

export L p0 seed type k NT dim num_colors

# --- run Cartesian product: rho × runs, with progress bar ---
parallel -j "$JOBS" --bar --halt soon,fail=1 '
  RHO={{1}}
  RUN={{2}}
  ./build/SOP "$L" "$p0" "$seed" "$type" "$k" "$NT" "$dim" "$num_colors" "$RHO"
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
      ./build/SOP "$L" "$p0" "$seed" "$type" "$k" "$NT" "$dim" "$num_colors" "$RHO"
    else
      ./build/SOP "$L" "$p0" "$seed" "$type" "$k" "$NT" "$dim" "$num_colors" "$RHO" >/dev/null
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




def custom_range(start: float, stop: float, n_points: int):
    return [round(i, 8) for i in np.linspace(start, stop, n_points)]