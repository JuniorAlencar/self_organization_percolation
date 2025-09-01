#!/bin/bash
set -euo pipefail

# Sobe um nível (mesmo comportamento do seu script)
cd ..

# Parâmetros fixos do executável
L=2000
N_SAMPLES=2000
P0=1.0           # p0 fixo para a figura
SEED=-1          # -1 => gera aleatório no código
DIM=2

# Tipos de percolação a rodar
type_percolation=("bond" "node")

# Pares (N_T, k) da figura: (a), (b), (c)
NT_list=(300 200 100)
K_list=(1.0e-03 1.0e-04 1.0e-05)

for tipo in "${type_percolation[@]}"; do
  echo "=== Running for percolation type: $tipo ==="
  for idx in "${!NT_list[@]}"; do
    NT=${NT_list[$idx]}
    K=${K_list[$idx]}
    echo "./build/SOP $L $N_SAMPLES $P0 $SEED $tipo $K $NT $DIM"
    ./build/SOP "$L" "$N_SAMPLES" "$P0" "$SEED" "$tipo" "$K" "$NT" "$DIM"
  done
done
