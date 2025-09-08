#!/bin/bash

cd .. # Got o project root

num_runs=20   # number of external repetitions
rho=(0.003 0.02021053 0.03742105 0.05463158 0.07184211 0.08905263 0.10626316 0.12347368 0.14068421 0.15789474 0.17510526 0.19231579 0.20952632 0.22673684 0.24394737 0.26115789 0.27836842 0.29557895 0.31278947 0.33)  # list of rho values
L=2000
NumSamples=12000
p0=0.5
seed=-1
type="bond"
k=0.0001
NT=200
dim=2
num_colors=3

for ((run=1; run<=num_runs; run++)); do
  echo "=== Run $run ==="
  for idx in "${!rho[@]}"; do
    RHO=${rho[$idx]}
    echo "./build/SOP $L $NumSamples $p0 $seed $type $k $NT $dim $num_colors $RHO"
    ./build/SOP "$L" "$NumSamples" "$p0" "$seed" "$type" "$k" "$NT" "$dim" "$num_colors" "$RHO"
  done
done
