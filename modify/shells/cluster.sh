#!/bin/bash
#SBATCH -J L4096_nc_4_rho_0.25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# back to base folder project

cd ..

L=4096
P0=1.0
SEED=-1
TYPE_PERC=bond
K=5e-05
NT=400
DIM=2
NUM_COLORS=4
RHO=0.25
NUM_SAMPLES=500

echo "=== Parâmetros recebidos ==="
echo "L=$L  P0=$P0  SEED=$SEED  TYPE_PERC=$TYPE_PERC"
echo "K=$K  NT=$NT  DIM=$DIM  NUM_COLORS=$NUM_COLORS"
echo "RHO=$RHO  NUM_SAMPLES=$NUM_SAMPLES"
echo "============================"

EXEC=./build/SOP
i=0
while [ "$i" -le "$NUM_SAMPLES" ]; do
  srun "$EXEC" "$L" "$P0" "$SEED" "$TYPE_PERC" "$K" "$NT" "$DIM" "$NUM_COLORS" "$RHO"
  i=$(( i + 1 ))
done

echo "Finalizado com sucesso."
