#!/usr/bin/env bash
set -euo pipefail

# --- go to project root ---
cd ..

# --- parameters exported for parallel ---
num_runs=40
rho=(0.0001 0.0052 0.0103 0.0154 0.0205 0.0256 0.0307 0.0358 0.0409 0.046 0.0511 0.0562 0.0613 0.0664 0.0715 0.0766 0.0817 0.0868 0.0919 0.097 0.1021 0.1072 0.1123 0.1174 0.1225 0.1276 0.1327 0.1378 0.1429 0.148 0.1531 0.1582 0.1633 0.1684 0.1735 0.1786 0.1837 0.1888 0.1939 0.199 0.2041 0.2092 0.2143 0.2194 0.2245 0.2296 0.2347 0.2398 0.2449 0.25)
L=512
p0=1.0
seed=-1
type="bond"
k=1e-06
NT=3500
dim=3
num_colors=4

# --- fixed number of workers (baked into the script) ---
JOBS=12

# --- check GNU parallel ---
if ! command -v parallel >/dev/null 2>&1; then
  echo "[ERROR] 'parallel' not found. Install it (e.g., sudo apt-get install parallel) or set multi=False."
  exit 1
fi

export L p0 seed type k NT dim num_colors

# --- run Cartesian product: rho × runs, with progress bar ---
parallel -j "$JOBS" --bar --halt soon,fail=1 '
  RHO={1}
  RUN={2}
  ./build/SOP "$L" "$p0" "$seed" "$type" "$k" "$NT" "$dim" "$num_colors" "$RHO"
' ::: "${rho[@]}" ::: $(seq 1 "$num_runs")
