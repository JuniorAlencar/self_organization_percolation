#!/bin/bash

cd .. # Got o project root

num_runs=5   # number of external repetitions
rho=(0.03 0.035 0.04 0.045 0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09 0.095 0.1 0.105 0.11 0.115 0.12 0.125)  # list of rho values
L=1000
NumSamples=12000
p0=1.0
seed=-1
type="bond"
k=1e-08
NT=12000
dim=3
num_colors=8

for ((run=1; run<=num_runs; run++)); do
  echo "=== Run $run ==="
  for idx in "${!rho[@]}"; do
    RHO=${rho[$idx]}
    echo "./build/SOP $L $NumSamples $p0 $seed $type $k $NT $dim $num_colors $RHO"
    ./build/SOP "$L" "$NumSamples" "$p0" "$seed" "$type" "$k" "$NT" "$dim" "$num_colors" "$RHO"
  done
done
