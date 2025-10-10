import os
import stat
import textwrap
import numpy as np
from textwrap import dedent
from pathlib import Path
import subprocess

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

def create_folder(folder_path):
    """
    Creates the folder if it does not already exist.

    Args:
        folder_path (str): Path to the folder to be created.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

import os, stat, textwrap

def create_cluster_cli_shell(exec_name="run_jobs.sh", folder_to_shell="../shells"):
    os.makedirs(folder_to_shell, exist_ok=True)

    script = f"""\
#!/usr/bin/env bash
#SBATCH -J SOP_cli
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# === uso / help ===
usage() {{
  cat <<USAGE
Uso:
  {exec_name} L P0 SEED TYPE_PERC K NT DIM NUM_COLORS RHO NUM_SAMPLES DCU

Exemplo:
  {exec_name} 512 0.7 123 bond 1e-6 200 3 1 0.001 5 true

Notas:
  - TYPE_PERC: "bond" ou "node"
  - DCU     : true/false/yes/no/1/0 (case-insensitive)
USAGE
}}

if [[ "$#" -lt 1 ]]; then
  echo "[ERROR] Nenhum argumento."; usage; exit 1
fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage; exit 0
fi
if [[ "$#" -ne 11 ]]; then
  echo "[ERROR] Número inválido de argumentos ($#)."; usage; exit 2
fi

# --- parâmetros vindos da CLI ---
L="$1"
P0="$2"
SEED="$3"
TYPE_PERC="$4"
K="$5"
NT="$6"
DIM="$7"
NUM_COLORS="$8"
RHO="$9"
NUM_SAMPLES="${{10}}"
DCU="${{11}}"   # passa como string para o parse_bool do C++

# --- volta à raiz do projeto ---
cd ..

EXEC=./build/SOP

echo "=== Received Parameters ==="
echo "L=$L  P0=$P0  SEED=$SEED  TYPE_PERC=$TYPE_PERC"
echo "K=$K  NT=$NT  DIM=$DIM  NUM_COLORS=$NUM_COLORS"
echo "RHO=$RHO  NUM_SAMPLES=$NUM_SAMPLES  DCU=$DCU"
echo "============================"

i=1
while [[ "$i" -le "$NUM_SAMPLES" ]]; do
  srun "$EXEC" "$L" "$P0" "$SEED" "$TYPE_PERC" "$K" "$NT" "$DIM" "$NUM_COLORS" "$RHO" "$DCU"
  i=$(( i + 1 ))
done

echo "Completed successfully."
"""

    path = os.path.join(folder_to_shell, exec_name)
    with open(path, "w") as f:
        f.write(textwrap.dedent(script))

    # tornar executável
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)
    print(f"✅ Shell script criado em {path}")

def run_multi_rho(L, p0, seed, type_perc, k, NT, dim, num_colors, rho_lst, N_samples, DCU_prop):
    # Create .sh if don't exist
    if not os.path.exists("../shells/run_jobs.sh"):
        create_cluster_cli_shell()
    
    # Go to folder ../shells/
    here = Path(__file__).resolve().parent
    shells = here.parent / "shells"   # ../shells relativo ao script
    os.chdir(shells)
    
    for rho in rho_lst:
        cmd = [
        "sbatch", "run_jobs.sh",
        str(L), str(p0), str(seed), str(type_perc),
        str(k), str(NT), str(dim), str(num_colors),
        str(rho), str(N_samples), DCU
        ]
    # Execute the jobs -> 
    subprocess.run(cmd, check=True)