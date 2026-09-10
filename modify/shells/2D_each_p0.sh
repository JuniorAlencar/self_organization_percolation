#!/usr/bin/env bash
set -euo pipefail

cd ..

num_runs=500
rho=1.0
L=2000
SEED=-1
type="node"        # <- agora 'node'
NT=200             # <- NT fixo
K=1.0e-05
dim=2
num_colors=1
JOBS=20

# Lista de p0 a varrer
p0_list=(0.4 0.8 1.0)

command -v parallel >/dev/null 2>&1 || { echo "[ERROR] 'parallel' not found."; exit 1; }
[ -x ./build/SOP ] || { echo "[ERROR] ./build/SOP not found or not executable."; exit 1; }

# Exporta variáveis constantes
export L SEED type K NT dim num_colors rho

for p0 in "${p0_list[@]}"; do
  echo "[INFO] Rodando num_runs=${num_runs} com NT=${NT} e p0=${p0}..."
  export p0  # disponibiliza p0 atual para o bloco do parallel

  # 1 linha de entrada por job; usamos ":" (no-op) para consumir "{}"
  seq 1 "$num_runs" | parallel -j "$JOBS" --bar --halt soon,fail=1 '
    : {};  # consome o argumento do parallel, nada é anexado ao final
    ./build/SOP "$L" "$p0" "$SEED" "$type" "$K" "$NT" "$dim" "$num_colors" "$rho"
  '
done
