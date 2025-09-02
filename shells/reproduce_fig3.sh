#!/bin/bash
set -euo pipefail

# Sobe um nível (mesmo comportamento do seu script)
cd ..

# Parâmetros fixos do executável
L=2000
N_SAMPLES=12000
P0=1.0           # p0 fixo para a figura
SEED=-1          # -1 => gera aleatório no código
DIM=2
NUM_COLORS=1
RHO=1
# Tipos de percolação a rodar
type_percolation="bond"

# Pares (N_T, k) da figura: (a), (b), (c)
NT_list=(300 200 100)
K_list=(1.0e-03 1.0e-04 1.0e-05)

for idx in "${!NT_list[@]}"; do
  NT=${NT_list[$idx]}
  K=${K_list[$idx]}
  echo "./build/SOP $L $N_SAMPLES $P0 $SEED $type_percolation $K $NT $DIM $NUM_COLORS $RHO"
  ./build/SOP "$L" "$N_SAMPLES" "$P0" "$SEED" "$type_percolation" "$K" "$NT" "$DIM" "$NUM_COLORS" "$RHO"
done
