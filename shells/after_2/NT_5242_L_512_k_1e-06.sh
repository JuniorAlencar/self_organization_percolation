#!/usr/bin/env bash
set -euo pipefail

# --- go to project root ---
cd ..

# --- parameters exported for parallel ---
num_runs=20
rho=(0.2796229 0.2829798 0.2863367 0.2896936 0.29305051 0.29640741 0.29976431 0.30312121 0.30647811 0.30983502 0.31319192 0.31654882 0.31990572 0.32326263 0.32661953 0.32997643 0.33333333)
L=512
p0=1.0
seed=-1
type="bond"
k=1e-06
NT=5242
dim=3
num_colors=2
P0=0.1
JOBS=19

# --- check GNU parallel ---
if ! command -v parallel >/dev/null 2>&1; then
  echo "[ERROR] 'parallel' not found. Install it (e.g., sudo apt-get install parallel) or set multi=False."
  exit 1
fi

export L p0 seed type k NT dim num_colors P0

# --- run Cartesian product: rho × runs, with progress bar ---
parallel -j "$JOBS" --bar --halt soon,fail=1 '
  RHO={1}
  RUN={2}
  ./build/SOP "$L" "$p0" "$seed" "$type" "$k" "$NT" "$dim" "$num_colors" "$RHO" "$P0"
' ::: "${rho[@]}" ::: $(seq 1 "$num_runs")
