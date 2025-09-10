#!/usr/bin/env bash
set -euo pipefail

# --- go to project root ---
cd ..

# --- parameters exported for parallel ---
num_runs=20
rho=(0.21902 0.21926 0.2195 0.21974 0.21998 0.22022 0.22046 0.2207 0.22094 0.22118 0.22142 0.22166 0.2219 0.22214 0.22238 0.22262 0.22286 0.2231 0.22334 0.22358 0.22382 0.22406 0.2243 0.22454 0.22478 0.22502 0.22526 0.2255 0.22574 0.22598 0.22622 0.22646 0.2267 0.22694 0.22718 0.22742 0.22766 0.2279 0.22814 0.22838 0.22862 0.22886 0.2291 0.22934 0.22958 0.22982 0.23006 0.2303 0.23054 0.23078)
L=256
p0=1.0
seed=-1
type="bond"
k=1e-05
NT=820
dim=3
num_colors=4

# --- fixed number of workers (baked into the script) ---
JOBS=20

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
