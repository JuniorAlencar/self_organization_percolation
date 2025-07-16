#!/bin/bash

# Voltar um diretório
cd ..

# Tipos de percolação
type_percolation=("bond" "node")

# Valores de p0 para cada tipo
p0_values_bond=("0.3" "0.7" "1.0")
p0_values_node=("0.4" "0.8" "1.0")

# Valores de Nt
Nt_values=(2000 4000 5000 6000 8000 10000)

# Número de amostras para cada (type, p0, Nt)
num_samples=5

# Loop principal
for type in "${type_percolation[@]}"; do
    if [ "$type" == "bond" ]; then
        p0_values=("${p0_values_bond[@]}")
    else
        p0_values=("${p0_values_node[@]}")
    fi

    for p0 in "${p0_values[@]}"; do
        for Nt in "${Nt_values[@]}"; do
            for ((i = 1; i <= num_samples; i++)); do
                echo "Amostra $i - Executando: ./build/SOP 1000 1000 $p0 -1 $type 1.0e-05 $Nt 3"
                ./build/SOP 1000 1000 "$p0" -1 "$type" 1.0e-05 "$Nt" 3
            done
        done
    done
done
