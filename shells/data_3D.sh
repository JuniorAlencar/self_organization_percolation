#!/bin/bash

cd .. # Got o project root

num_runs=10   # number of external repetitions
rho=(0.0001 0.0052 0.0103 0.0154 0.0205 0.0256 0.0307 0.0358 0.0409 0.046 0.0511 0.0562 0.0613 0.0664 0.0715 0.0766 0.0817 0.0868 0.0919 0.097 0.1021 0.1072 0.1123 0.1174 0.1225 0.1276 0.1327 0.1378 0.1429 0.148 0.1531 0.1582 0.1633 0.1684 0.1735 0.1786 0.1837 0.1888 0.1939 0.199 0.2041 0.2092 0.2143 0.2194 0.2245 0.2296 0.2347 0.2398 0.2449 0.25)  # list of rho values
L=128
NumSamples=15000
p0=1.0
seed=-1
type="bond"
k=1e-05
NT=205
dim=3
num_colors=4

for ((run=1; run<=num_runs; run++)); do
  echo "=== Run $run ==="
  for idx in "${!rho[@]}"; do
    RHO=${rho[$idx]}
    echo "./build/SOP $L $NumSamples $p0 $seed $type $k $NT $dim $num_colors $RHO"
    ./build/SOP "$L" "$NumSamples" "$p0" "$seed" "$type" "$k" "$NT" "$dim" "$num_colors" "$RHO"
  done
done
