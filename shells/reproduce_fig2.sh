#!/bin/bash

# Go one directory up
cd ..

# Define the types of percolation and the corresponding p0 values
type_percolation=("bond" "node")
p0_values=("0.3 0.7 1.0" "0.4 0.8 1.0")

# Outer loop over percolation types
for i in "${!type_percolation[@]}"; do
  tipo="${type_percolation[$i]}"
  # Read the list of p0 values for the current percolation type
  IFS=' ' read -r -a p0_list <<< "${p0_values[$i]}"

  echo "Running for percolation type: $tipo"

  # Inner loop over the corresponding p0 values
  for p0 in "${p0_list[@]}"; do
    echo "./build/SOP 2000 2000 $p0 -1 $tipo 1.0e-04 200"
    ./build/SOP 2000 2000 "$p0" -1 "$tipo" 1.0e-04 200 2
  done
done