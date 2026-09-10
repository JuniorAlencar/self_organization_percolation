#!/usr/bin/env bash
#SBATCH -J SOP_cli
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# === uso / help ===
usage() {
  cat <<USAGE
Uso:
  run_jobs.sh L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS RHO NUM_SAMPLES P0 EQUILIBRATION [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE] [CONTROL_PARAM] [LOG_EPSILON]

Exemplo:
  run_jobs.sh 512 0.7 123 bond 0.03 0.06 3 1 0.001 5 0.1 false
  run_jobs.sh 512 0.7 123 bond 0.03 0.06 3 1 0.001 5 0.1 false false growth_test
  run_jobs.sh 512 0.8 -1 bond 0.01 0.1 2 1 1.0 5 0.2 false false growth_test random false false log_asymmetric 5.0 1e-6

Notas:
  - TYPE_PERC: "bond" ou "node"
  - MODE opcional: "sop" ou "growth_test"
  - CONTROL_RULE opcional: "linear", "log_saturated" ou "log_asymmetric"
  - CONTROL_PARAM: teto positivo em log_saturated; fator X em log_asymmetric
USAGE
}

if [[ "$#" -lt 1 ]]; then
  echo "[ERROR] Nenhum argumento."; usage; exit 1
fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage; exit 0
fi
if [[ "$#" -lt 12 || "$#" -gt 20 ]]; then
  echo "[ERROR] Número inválido de argumentos ($#)."; usage; exit 2
fi

# --- parâmetros vindos da CLI ---
L="$1"
p0="$2"
SEED="$3"
TYPE_PERC="$4"
C="$5"
F_T="$6"
DIM="$7"
NUM_COLORS="$8"
RHO="$9"
NUM_SAMPLES="${10}"
P0="${11}"
EQUILIBRATION="${12}"
PROPERTIES="${13:-}"
RUN_MODE="${14:-}"
INITIAL_LAYOUT="${15:-random}"
SURFACE_OBSERVABLES="${16:-false}"
SAVE_ANIMATION_WINDOW_ONLY="${17:-false}"
CONTROL_RULE="${18:-linear}"
CONTROL_PARAM="${19:-0}"
LOG_EPSILON="${20:-1.0e-12}"

if [[ -n "$PROPERTIES" && -z "$RUN_MODE" ]]; then
  case "$PROPERTIES" in
    sop|growth_test)
      RUN_MODE="$PROPERTIES"
      PROPERTIES="false"
      ;;
  esac
fi

EXTRA_ARGS=()
if [[ "$CONTROL_RULE" != "linear" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT" "$SURFACE_OBSERVABLES" "$SAVE_ANIMATION_WINDOW_ONLY" "$CONTROL_RULE" "$CONTROL_PARAM" "$LOG_EPSILON")
elif [[ "$SAVE_ANIMATION_WINDOW_ONLY" != "false" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT" "$SURFACE_OBSERVABLES" "$SAVE_ANIMATION_WINDOW_ONLY")
elif [[ "$SURFACE_OBSERVABLES" != "false" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT" "$SURFACE_OBSERVABLES")
elif [[ "$INITIAL_LAYOUT" != "random" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT")
elif [[ -n "$RUN_MODE" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "$RUN_MODE")
elif [[ -n "$PROPERTIES" ]]; then
  EXTRA_ARGS=("$PROPERTIES")
fi

# --- volta à raiz do projeto ---
cd ..

EXEC=./build/SOP

echo "=== Received Parameters ==="
echo "L=$L  p0=$p0  SEED=$SEED  TYPE_PERC=$TYPE_PERC"
echo "C=$C  F_T=$F_T  DIM=$DIM  NUM_COLORS=$NUM_COLORS"
echo "RHO=$RHO  NUM_SAMPLES=$NUM_SAMPLES  P0=$P0  EQUILIBRATION=$EQUILIBRATION"
echo "CONTROL_RULE=$CONTROL_RULE  CONTROL_PARAM=$CONTROL_PARAM  LOG_EPSILON=$LOG_EPSILON"
echo "EXTRA_ARGS=${EXTRA_ARGS[*]:-}"
echo "============================"

i=1
while [[ "$i" -le "$NUM_SAMPLES" ]]; do
  srun "$EXEC" "$L" "$p0" "$SEED" "$TYPE_PERC" "$C" "$F_T" "$DIM" "$NUM_COLORS" "$RHO" "$P0" "$EQUILIBRATION" "${EXTRA_ARGS[@]}"
  i=$(( i + 1 ))
done

echo "Completed successfully."
