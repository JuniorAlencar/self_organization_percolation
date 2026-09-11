#!/usr/bin/env bash
#SBATCH -J SOP_cli
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# === uso / help ===
usage() {
  cat <<USAGE
Uso:
  run_jobs.sh L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS RHO NUM_SAMPLES EQUILIBRATION [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE]

Exemplo:
  run_jobs.sh 512 0.7 123 bond 0.03 0.06 3 1 0.001 5 false
  run_jobs.sh 512 0.7 123 bond 0.03 0.06 3 1 0.001 5 false false growth_test
  run_jobs.sh 512 0.8 -1 bond 0.01 0.1 2 1 1.0 5 false false growth_test clustered false false relative

Notas:
  - TYPE_PERC: "bond" ou "node"
  - MODE opcional: "sop" ou "growth_test"
  - CONTROL_RULE opcional: "relative" ou "linear"
USAGE
}

if [[ "$#" -lt 1 ]]; then
  echo "[ERROR] Nenhum argumento."; usage; exit 1
fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage; exit 0
fi
if [[ "$#" -lt 11 || "$#" -gt 17 ]]; then
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
EQUILIBRATION="${11}"
PROPERTIES="${12:-}"
RUN_MODE="${13:-}"
INITIAL_LAYOUT="${14:-clustered}"
SURFACE_OBSERVABLES="${15:-false}"
SAVE_ANIMATION_WINDOW_ONLY="${16:-false}"
CONTROL_RULE="${17:-relative}"

if [[ -n "$PROPERTIES" && -z "$RUN_MODE" ]]; then
  case "$PROPERTIES" in
    sop|growth_test)
      RUN_MODE="$PROPERTIES"
      PROPERTIES="false"
      ;;
  esac
fi

EXTRA_ARGS=()
if [[ "$CONTROL_RULE" != "relative" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT" "$SURFACE_OBSERVABLES" "$SAVE_ANIMATION_WINDOW_ONLY" "$CONTROL_RULE")
elif [[ "$SAVE_ANIMATION_WINDOW_ONLY" != "false" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT" "$SURFACE_OBSERVABLES" "$SAVE_ANIMATION_WINDOW_ONLY")
elif [[ "$SURFACE_OBSERVABLES" != "false" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT" "$SURFACE_OBSERVABLES")
elif [[ "$INITIAL_LAYOUT" != "clustered" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT")
elif [[ -n "$RUN_MODE" && "$RUN_MODE" != "sop" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "$RUN_MODE")
elif [[ -n "$PROPERTIES" && "$PROPERTIES" != "false" ]]; then
  EXTRA_ARGS=("$PROPERTIES")
fi

# --- volta à raiz do projeto ---
cd ..

EXEC=./build/SOP

echo "=== Received Parameters ==="
echo "L=$L  p0=$p0  SEED=$SEED  TYPE_PERC=$TYPE_PERC"
echo "C=$C  F_T=$F_T  DIM=$DIM  NUM_COLORS=$NUM_COLORS"
echo "RHO=$RHO  NUM_SAMPLES=$NUM_SAMPLES  EQUILIBRATION=$EQUILIBRATION"
echo "CONTROL_RULE=$CONTROL_RULE"
echo "EXTRA_ARGS=${EXTRA_ARGS[*]:-}"
echo "============================"

i=1
while [[ "$i" -le "$NUM_SAMPLES" ]]; do
  srun "$EXEC" "$L" "$p0" "$SEED" "$TYPE_PERC" "$C" "$F_T" "$DIM" "$NUM_COLORS" "$RHO" "$EQUILIBRATION" "${EXTRA_ARGS[@]}"
  i=$(( i + 1 ))
done

echo "Completed successfully."
