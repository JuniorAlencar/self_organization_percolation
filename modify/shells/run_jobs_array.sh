#!/usr/bin/env bash
#SBATCH -J SOP_cli
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G

# Modos:
# 1) fixed:
#    sbatch run_jobs_array.sh fixed L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES rho P0 EQUILIBRATION [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE] [CONTROL_PARAM] [LOG_EPSILON] [FRACTION_SAMPLES] [SAMPLE_GAP_OVER_L]
#
# 2) array:
#    sbatch --array=0-(N_RHO-1)%MAX_CONCURRENT run_jobs_array.sh array L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES P0 EQUILIBRATION RHO_FILE [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE] [CONTROL_PARAM] [LOG_EPSILON] [FRACTION_SAMPLES] [SAMPLE_GAP_OVER_L]
#
# Chamada do executável SOP:
#    ./SOP L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS rho P0 EQUILIBRATION [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE] [CONTROL_PARAM] [LOG_EPSILON]

if [[ "$#" -lt 1 ]]; then
  echo "Uso:"
  echo "  fixed: $0 fixed L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES rho P0 EQUILIBRATION [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE] [CONTROL_PARAM] [LOG_EPSILON] [FRACTION_SAMPLES] [SAMPLE_GAP_OVER_L]"
  echo "  array: $0 array L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES P0 EQUILIBRATION RHO_FILE [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE] [CONTROL_PARAM] [LOG_EPSILON] [FRACTION_SAMPLES] [SAMPLE_GAP_OVER_L]"
  exit 1
fi

MODE="$1"

if [[ "$MODE" == "fixed" ]]; then
  if [[ "$#" -lt 14 || "$#" -gt 24 ]]; then
    echo "Uso: $0 fixed L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES rho P0 EQUILIBRATION [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE] [CONTROL_PARAM] [LOG_EPSILON] [FRACTION_SAMPLES] [SAMPLE_GAP_OVER_L]"
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
  P0="${13}"
  EQUILIBRATION="${14}"
  PROPERTIES="${15:-}"
  RUN_MODE="${16:-}"
  INITIAL_LAYOUT="${17:-random}"
  SURFACE_OBSERVABLES="${18:-false}"
  SAVE_ANIMATION_WINDOW_ONLY="${19:-false}"
  CONTROL_RULE="${20:-linear}"
  CONTROL_PARAM="${21:-0}"
  LOG_EPSILON="${22:-1.0e-12}"
  FRACTION_SAMPLES="${23:-30}"
  SAMPLE_GAP_OVER_L="${24:-1.0}"

elif [[ "$MODE" == "array" ]]; then
  if [[ "$#" -lt 14 || "$#" -gt 24 ]]; then
    echo "Uso: $0 array L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES P0 EQUILIBRATION RHO_FILE [PROPERTIES] [MODE] [INITIAL_LAYOUT] [SURFACE_OBSERVABLES] [SAVE_ANIMATION_WINDOW_ONLY] [CONTROL_RULE] [CONTROL_PARAM] [LOG_EPSILON] [FRACTION_SAMPLES] [SAMPLE_GAP_OVER_L]"
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
  P0="${12}"
  EQUILIBRATION="${13}"
  RHO_FILE="${14}"
  PROPERTIES="${15:-}"
  RUN_MODE="${16:-}"
  INITIAL_LAYOUT="${17:-random}"
  SURFACE_OBSERVABLES="${18:-false}"
  SAVE_ANIMATION_WINDOW_ONLY="${19:-false}"
  CONTROL_RULE="${20:-linear}"
  CONTROL_PARAM="${21:-0}"
  LOG_EPSILON="${22:-1.0e-12}"
  FRACTION_SAMPLES="${23:-30}"
  SAMPLE_GAP_OVER_L="${24:-1.0}"

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
    sop|growth_test|fractions|raw_fractions)
      RUN_MODE="$PROPERTIES"
      PROPERTIES="false"
      ;;
  esac
fi

if [[ "${RUN_MODE:-}" == "raw_fractions" ]]; then
  RUN_MODE="fractions"
fi

EXTRA_ARGS=()
if [[ "${RUN_MODE:-}" == "fractions" ]]; then
  PROPERTIES="true"
  EXTRA_ARGS=("$PROPERTIES" "$RUN_MODE" "$INITIAL_LAYOUT" "$SURFACE_OBSERVABLES" "$SAVE_ANIMATION_WINDOW_ONLY" "$CONTROL_RULE" "$CONTROL_PARAM" "$LOG_EPSILON")
elif [[ "${CONTROL_RULE:-linear}" != "linear" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT" "$SURFACE_OBSERVABLES" "$SAVE_ANIMATION_WINDOW_ONLY" "$CONTROL_RULE" "$CONTROL_PARAM" "$LOG_EPSILON")
elif [[ "${SAVE_ANIMATION_WINDOW_ONLY:-false}" != "false" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT" "$SURFACE_OBSERVABLES" "$SAVE_ANIMATION_WINDOW_ONLY")
elif [[ "${SURFACE_OBSERVABLES:-false}" != "false" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT" "$SURFACE_OBSERVABLES")
elif [[ "${INITIAL_LAYOUT:-random}" != "random" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "${RUN_MODE:-sop}" "$INITIAL_LAYOUT")
elif [[ -n "${RUN_MODE:-}" ]]; then
  EXTRA_ARGS=("${PROPERTIES:-false}" "$RUN_MODE")
elif [[ -n "${PROPERTIES:-}" ]]; then
  EXTRA_ARGS=("$PROPERTIES")
fi

EXEC="/home/junioralencar/codes/SOP/build/SOP"
WORKDIR="/home/junioralencar/codes/SOP"

echo "=== SOP job ==="
echo "MODE=$MODE"
if [[ "$MODE" == "array" ]]; then
  echo "task_id=$SLURM_ARRAY_TASK_ID"
  echo "RHO_FILE=$RHO_FILE"
fi
echo "rho=$RHO"
echo "L=$L p0=$p0 SEED=$SEED TYPE=$TYPE_PERC"
echo "C=$C F_T=$F_T DIM=$DIM NC=$NUM_COLORS"
echo "NSAMPLES=$NUM_SAMPLES TARGET_SAMPLES=$TARGET_SAMPLES P0=$P0 EQUILIBRATION=$EQUILIBRATION"
echo "INITIAL_LAYOUT=$INITIAL_LAYOUT SURFACE_OBSERVABLES=$SURFACE_OBSERVABLES SAVE_ANIMATION_WINDOW_ONLY=$SAVE_ANIMATION_WINDOW_ONLY"
echo "CONTROL_RULE=$CONTROL_RULE CONTROL_PARAM=$CONTROL_PARAM LOG_EPSILON=$LOG_EPSILON"
echo "FRACTION_SAMPLES=$FRACTION_SAMPLES"
echo "SAMPLE_GAP_OVER_L=$SAMPLE_GAP_OVER_L"
echo "EXTRA_ARGS=${EXTRA_ARGS[*]:-}"
echo "EXEC=$EXEC"
echo "WORKDIR=$WORKDIR"
echo "==============="

cd "$WORKDIR"

i=1
while [[ "$i" -le "$NUM_SAMPLES" ]]; do
  echo "$EXEC $L $p0 $SEED $TYPE_PERC $C $F_T $DIM $NUM_COLORS $RHO $P0 $EQUILIBRATION ${EXTRA_ARGS[*]}"
  srun "$EXEC" "$L" "$p0" "$SEED" "$TYPE_PERC" "$C" "$F_T" "$DIM" "$NUM_COLORS" "$RHO" "$P0" "$EQUILIBRATION" "${EXTRA_ARGS[@]}"
  i=$((i + 1))
done

if [[ "${RUN_MODE:-}" == "fractions" ]]; then
  echo "[INFO] fractions: pulando registro de historico Python antigo."
  echo "Done."
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$WORKDIR/python/src:$WORKDIR:${PYTHONPATH:-}"
"$PYTHON_BIN" - "$L" "$p0" "$TYPE_PERC" "$C" "$F_T" "$DIM" "$NUM_COLORS" "$RHO" "$P0" "${RUN_MODE:-sop}" "$TARGET_SAMPLES" "$FRACTION_SAMPLES" "$CONTROL_RULE" "$CONTROL_PARAM" "$LOG_EPSILON" <<'PY'
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
    P0,
    run_mode,
    target_samples,
    fraction_samples,
    control_rule,
    control_param,
    log_epsilon,
) = sys.argv[1:]

kwargs = dict(
    L=int(L), p0=float(p0), type_perc=type_perc, c=float(c),
    f_T=float(f_T), dim=int(dim), num_colors=int(num_colors),
    rho=float(rho), P0=float(P0), run_mode=run_mode or "sop",
    target_samples=int(target_samples),
    control_rule=control_rule or "linear",
    control_param=float(control_param),
    log_epsilon=float(log_epsilon),
)
try:
    info = record_completed_parameter_set(**kwargs, fraction_samples=int(fraction_samples))
except TypeError:
    info = record_completed_parameter_set(**kwargs)
print(f"Historico atualizado: {info['history_path']} known_count_after={info['known_count_after']}")
PY

echo "Done."
