#!/usr/bin/env bash
set -euo pipefail

cd ..

num_runs=500
rho=1.0
L=2000
p0=1.0
SEED=-1
type="bond"
NT_LIST=(200 300 400)
K=1.0e-05   # <- lista de NTs
dim=2
num_colors=1
JOBS=20

command -v parallel >/dev/null 2>&1 || { echo "[ERROR] 'parallel' not found."; exit 1; }
[ -x ./build/SOP ] || { echo "[ERROR] ./build/SOP not found or not executable."; exit 1; }

export L p0 SEED type k dim num_colors rho

for NT in "${NT_LIST[@]}"; do
  echo "[INFO] Rodando num_runs=${num_runs} com NT=${NT}..."
  export NT   # exporta o NT atual para o bloco do parallel

  # 1 linha de entrada por job; usamos ":" (no-op) para CONSUMIR "{}"
  seq 1 "$num_runs" | parallel -j "$JOBS" --bar --halt soon,fail=1 '
    : {};  # consome o argumento do parallel, nada é anexado ao final
    ./build/SOP "$L" "$p0" "$SEED" "$type" "$K" "$NT" "$dim" "$num_colors" "$rho"
  '
done
