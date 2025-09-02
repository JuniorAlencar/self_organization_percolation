#!/bin/bash

cd ..

# Parâmetros fixos
L=1000
NumSamples=4000
p0=1.0
seed=-1  # suposição baseada no seu código
type="bond"
k=1.0e-05
NT=200
dim=2
num_colors=1
rho=1.0

echo "Running for size L=$L"
./build/SOP $L $NumSamples $p0 $seed $type $k $NT $dim $num_colors $rho

