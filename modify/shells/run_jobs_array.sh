#!/usr/bin/env bash
#SBATCH -J SOP_cli
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G

# Modos:
# 1) fixed:
#    sbatch run_jobs_array.sh fixed L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES rho EQUILIBRATION [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE]
#
# 2) array:
#    sbatch --array=0-(N_RHO-1)%MAX_CONCURRENT run_jobs_array.sh array L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES EQUILIBRATION RHO_FILE [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE]
#
# Chamada do executável SOP:
#    ./SOP L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS rho EQUILIBRATION [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE]

if [[ "$#" -lt 1 ]]; then
  echo "Uso:"
  echo "  fixed: $0 fixed L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES rho EQUILIBRATION [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE]"
  echo "  array: $0 array L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES EQUILIBRATION RHO_FILE [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE]"
  exit 1
fi

MODE="$1"

if [[ "$MODE" == "fixed" ]]; then
  if [[ "$#" -lt 13 || "$#" -gt 19 ]]; then
    echo "Uso: $0 fixed L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES rho EQUILIBRATION [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE]"
    exit 1
  fi

  L="$2"
  p0="$3"
  SEED="$4"
  TYPE_PERC="$5"
  C="$6"
  F_T="$7"
  DIM="$8"
  NUM_COLORS="$9"
  NUM_SAMPLES="${10}"
  TARGET_SAMPLES="${11}"
  RHO="${12}"
  EQUILIBRATION="${13}"
  PROPERTIES="${14:-}"
  RUN_MODE="${15:-}"
  INITIAL_LAYOUT="${16:-clustered}"
  SURFACE_OBSERVABLES="${17:-false}"
  SAVE_ANIMATION_WINDOW_ONLY="${18:-false}"
  CONTROL_RULE="${19:-relative}"

elif [[ "$MODE" == "array" ]]; then
  if [[ "$#" -lt 13 || "$#" -gt 19 ]]; then
    echo "Uso: $0 array L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES EQUILIBRATION RHO_FILE [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE]"
    exit 1
  fi

  L="$2"
  p0="$3"
  SEED="$4"
  TYPE_PERC="$5"
  C="$6"
  F_T="$7"
  DIM="$8"
  NUM_COLORS="$9"
  NUM_SAMPLES="${10}"
  TARGET_SAMPLES="${11}"
  EQUILIBRATION="${12}"
  RHO_FILE="${13}"
  PROPERTIES="${14:-}"
  RUN_MODE="${15:-}"
  INITIAL_LAYOUT="${16:-clustered}"
  SURFACE_OBSERVABLES="${17:-false}"
  SAVE_ANIMATION_WINDOW_ONLY="${18:-false}"
  CONTROL_RULE="${19:-relative}"

  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "[ERROR] MODE=array exige SLURM_ARRAY_TASK_ID."
    exit 1
  fi

  if [[ ! -f "$RHO_FILE" ]]; then
    echo "[ERROR] Arquivo de rho não encontrado: $RHO_FILE"
    exit 1
  fi

  RHO=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RHO_FILE")

  if [[ -z "$RHO" ]]; then
    echo "[ERROR] Não foi possível ler rho para task_id=$SLURM_ARRAY_TASK_ID"
    echo "[ERROR] Arquivo: $RHO_FILE"
    echo "[ERROR] Conteúdo:"
    nl -ba "$RHO_FILE"
    exit 1
  fi
else
  echo "[ERROR] MODE inválido: $MODE"
  echo "Use 'fixed' ou 'array'"
  exit 1
fi

if [[ -n "${PROPERTIES:-}" && -z "${RUN_MODE:-}" ]]; then
  case "$PROPERTIES" in
    sop|growth_test)
      RUN_MODE="$PROPERTIES"
      PROPERTIES="false"
      ;;
  esac
fi

EXTRA_ARGS=()
if [[ "${CONTROL_RULE:-relative}" != "relative" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT" "$SURFACE_OBSERVABLES" "$SAVE_ANIMATION_WINDOW_ONLY" "$CONTROL_RULE")
elif [[ "${SAVE_ANIMATION_WINDOW_ONLY:-false}" != "false" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT" "$SURFACE_OBSERVABLES" "$SAVE_ANIMATION_WINDOW_ONLY")
elif [[ "${SURFACE_OBSERVABLES:-false}" != "false" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT" "$SURFACE_OBSERVABLES")
elif [[ "${INITIAL_LAYOUT:-clustered}" != "clustered" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT")
elif [[ -n "${RUN_MODE:-}" && "$RUN_MODE" != "sop" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "$RUN_MODE")
elif [[ -n "${PROPERTIES:-}" && "$PROPERTIES" != "false" ]]; then
  EXTRA_ARGS=("$PROPERTIES")
fi

if [[ -d "/home/junioralencar/codes/SOP" ]]; then
  WORKDIR="/home/junioralencar/codes/SOP"
  EXEC="/home/junioralencar/codes/SOP/build/SOP"
else
  EXEC="/home/light/Documents/self_organization_percolation/modify/build/SOP"
  WORKDIR="/home/light/Documents/self_organization_percolation/modify"
fi

echo "=== SOP job ==="
echo "MODE=$MODE"
if [[ "$MODE" == "array" ]]; then
  echo "task_id=$SLURM_ARRAY_TASK_ID"
  echo "RHO_FILE=$RHO_FILE"
fi
echo "rho=$RHO"
echo "L=$L p0=$p0 SEED=$SEED TYPE=$TYPE_PERC"
echo "C=$C F_T=$F_T DIM=$DIM NC=$NUM_COLORS"
echo "NSAMPLES=$NUM_SAMPLES TARGET_SAMPLES=$TARGET_SAMPLES EQUILIBRATION=$EQUILIBRATION"
echo "CONTROL_RULE=$CONTROL_RULE"
echo "EXTRA_ARGS=${EXTRA_ARGS[*]:-}"
echo "EXEC=$EXEC"
echo "WORKDIR=$WORKDIR"
echo "==============="

cd "$WORKDIR"

i=1
while [[ "$i" -le "$NUM_SAMPLES" ]]; do
  srun "$EXEC" "$L" "$p0" "$SEED" "$TYPE_PERC" "$C" "$F_T" "$DIM" "$NUM_COLORS" "$RHO" "$EQUILIBRATION" "${EXTRA_ARGS[@]}"
  i=$((i + 1))
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$WORKDIR/python/src:$WORKDIR:${PYTHONPATH:-}"
"$PYTHON_BIN" - "$L" "$p0" "$TYPE_PERC" "$C" "$F_T" "$DIM" "$NUM_COLORS" "$RHO" "${RUN_MODE:-growth_test}" "$TARGET_SAMPLES" "$CONTROL_RULE" <<'PY'
import sys
from run_multi_functions import record_completed_parameter_set

(
    L,
    p0,
    type_perc,
    c,
    f_T,
    dim,
    num_colors,
    rho,
    run_mode,
    target_samples,
    control_rule,
) = sys.argv[1:]

info = record_completed_parameter_set(
    L=int(L),
    p0=float(p0),
    type_perc=type_perc,
    c=float(c),
    f_T=float(f_T),
    dim=int(dim),
    num_colors=int(num_colors),
    rho=float(rho),
    run_mode=run_mode or "growth_test",
    control_rule=control_rule or "relative",
    target_samples=int(target_samples),
)
print(f"Historico atualizado: {info['history_path']} known_count_after={info['known_count_after']}")
PY

echo "Done."
