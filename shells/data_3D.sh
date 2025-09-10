#!/usr/bin/env bash
set -euo pipefail

# --- go to project root ---
cd ..

# --- parameters exported for parallel ---
num_runs=200
rho=(0.21 0.21061224 0.21122449 0.21183673 0.21244898 0.21306122 0.21367347 0.21428571 0.21489796 0.2155102 0.21612245 0.21673469 0.21734694 0.21795918 0.21857143 0.21918367 0.21979592 0.22040816 0.22102041 0.22163265 0.2222449 0.22285714 0.22346939 0.22408163 0.22469388 0.22530612 0.22591837 0.22653061 0.22714286 0.2277551 0.22836735 0.22897959 0.22959184 0.23020408 0.23081633 0.23142857 0.23204082 0.23265306 0.23326531 0.23387755 0.2344898 0.23510204 0.23571429 0.23632653 0.23693878 0.23755102 0.23816327 0.23877551 0.23938776 0.24)
L=128
p0=1.0
seed=-1
type="bond"
k=1e-05
NT=205
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
