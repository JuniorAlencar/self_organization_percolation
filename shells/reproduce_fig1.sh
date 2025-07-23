#!/bin/bash

# Número de amostras a serem geradas
x=1000  # Altere para o número desejado

cd ..

# Parâmetros fixos
L=1000
NumSamples=1000
p0=1.0
seed=-1  # suposição baseada no seu código
type="bond"
k=1.0e-05
NT=200
dim=2

# Loop principal
for ((i=1; i<=x; i++)); do
    echo "Executando amostra $i de $x..."
    ./build/SOP $L $NumSamples $p0 $seed $type $k $NT $dim
done

echo "✅ Finalizado: $x amostras geradas."
