import os
import stat
import textwrap
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


def shell_data(
    L: int,
    type_perc: str,
    p0: float,
    seed: int,
    c: float,
    f_T: float,
    dim: int,
    num_colors: int,
    num_runs: int,
    rho: list,
    exec_name: str,
    P0: float | None = None,
    equlibration="false",
    multi: bool = False,
    properties=False,
    mode: str = "sop",
    initial_layout: str = "clustered",
    surface_observables=False,
    save_animation_window_only=False,
    control_rule: str = "relative",
    fraction_samples: int | None = None,
    fraction_gap_over_L: float = 1.0,
    force_topological_counts: bool = False,
    snapshot_dir: str | None = None,
    base_snapshot: str | None = None,
    gap_snapshot: str | None = None,
    **kwargs,
):
    """
    Generate a shell script to run SOP multiple times.

    New SOP executable signature:
        ./build/SOP <L> <p0> <P0> <seed> <type_percolation> <c> <f_T> <dim> <num_colors> <rho_val> <Equilibration> [Properties] [Mode] [InitialLayout] [SurfaceObservables] [SaveAnimationWindowOnly] [ControlRule]

    The update rule is:
        relative (default): p_i(t+1) = p_i(t) + c * (1 - f_i(t)/f_T) with headroom damping
        linear:             p_i(t+1) = p_i(t) + c * (f_T - f_i(t))
    
    Base seeding layout:
        clustered (default): natural contiguous seed cluster on base
        random, blocks, alternating

    If P0 is omitted, it defaults to min(1.0, 1.2 * f_T).
    """
    # Backward compatibility for calls that predate the explicit P0 argument:
    # shell_data(..., exec_name, Equilibration, multi, ...)
    if isinstance(P0, (str, bool)):
        old_equilibration = P0
        old_multi = equlibration
        P0 = None
        equlibration = old_equilibration
        if isinstance(old_multi, bool):
            multi = old_multi

    # Backward compatibility for older calls:
    # shell_data(..., exec_name, P0, num_threads, multi, ...)
    if isinstance(equlibration, (float, int)) and not isinstance(equlibration, bool):
        if "Equilibration" in kwargs:
            equlibration = kwargs["Equilibration"]
        elif isinstance(multi, str):
            equlibration = multi
            multi = False
        else:
            equlibration = "false"

    if P0 is None:
        P0 = min(1.0, 1.2 * f_T)

    if dim not in (2, 3):
        return "please, enter with dim = 2 or 3"

    if isinstance(num_runs, (list, tuple, np.ndarray)):
        if len(num_runs) != 1:
            raise ValueError(
                "num_runs must be a single integer for each generated shell script. "
                "Pass num_runs[idx] when iterating over L values."
            )
        num_runs = num_runs[0]

    num_runs = int(num_runs)

    mode = str(mode).strip()
    if mode not in ("sop", "growth_test", "fractions", "raw_fractions"):
        raise ValueError("mode must be 'sop', 'growth_test', or 'fractions'")

    if fraction_samples is not None:
        fraction_samples = int(fraction_samples)
        if fraction_samples <= 0:
            raise ValueError("fraction_samples must be positive")
    fraction_gap_over_L = float(fraction_gap_over_L)
    if fraction_gap_over_L < 0:
        raise ValueError("fraction_gap_over_L must be non-negative")
    if isinstance(force_topological_counts, bool):
        force_topological_counts = "true" if force_topological_counts else "false"
    else:
        force_topological_counts = str(force_topological_counts).strip().lower()
    if force_topological_counts not in ("true", "false"):
        raise ValueError("force_topological_counts must be true/false")

    if isinstance(properties, bool):
        properties = "true" if properties else "false"
    else:
        properties = str(properties).strip().lower()

    if properties not in ("true", "false"):
        raise ValueError("properties must be true/false")

    if isinstance(surface_observables, bool):
        surface_observables = "true" if surface_observables else "false"
    else:
        surface_observables = str(surface_observables).strip().lower()
    if surface_observables not in ("true", "false"):
        raise ValueError("surface_observables must be true/false")

    if isinstance(save_animation_window_only, bool):
        save_animation_window_only = "true" if save_animation_window_only else "false"
    else:
        save_animation_window_only = str(save_animation_window_only).strip().lower()
    if save_animation_window_only not in ("true", "false"):
        raise ValueError("save_animation_window_only must be true/false")

    control_rule = str(control_rule).strip().lower()
    valid_control_rules = {"relative", "linear"}
    if control_rule not in valid_control_rules:
        raise ValueError("control_rule must be 'relative' or 'linear'")

    initial_layout = str(initial_layout).strip().lower()
    valid_layouts = {"clustered", "cluster", "aglomerada", "random", "blocks", "quadrants", "quadrantes", "alternating", "alternado"}
    if initial_layout not in valid_layouts:
        raise ValueError("initial_layout must be 'clustered', 'random', 'blocks', or 'alternating'")

    os.makedirs("../shells", exist_ok=True)
    print("Creating shell script file in ../shells/")

    rho_list = " ".join(map(str, rho))

    snapshot_exports = []
    if snapshot_dir:
        snapshot_exports.append(f'export SOP_FRACTION_SNAPSHOT_DIR="${{SOP_FRACTION_SNAPSHOT_DIR:-{snapshot_dir}}}"')
    if base_snapshot:
        snapshot_exports.append(f'export SOP_FRACTION_BASE_SNAPSHOT="${{SOP_FRACTION_BASE_SNAPSHOT:-{base_snapshot}}}"')
    if gap_snapshot:
        snapshot_exports.append(f'export SOP_FRACTION_GAP_SNAPSHOT="${{SOP_FRACTION_GAP_SNAPSHOT:-{gap_snapshot}}}"')
    snapshot_export_str = "\n".join(snapshot_exports)
    if snapshot_export_str:
        snapshot_export_str = "\n" + snapshot_export_str

    if multi:
        script = f"""\
#!/usr/bin/env bash
set -euo pipefail

# --- go to project root ---
cd ..

# --- parameters exported for parallel ---
num_runs={num_runs}
rho=({rho_list})
L={L}
p0={p0}
P0={P0}
seed={seed}
type="{type_perc}"
c={c}
f_T={f_T}
dim={dim}
num_colors={num_colors}
Equilibration={equlibration}
Properties={properties}
Mode="{mode}"
InitialLayout="{initial_layout}"
SurfaceObservables={surface_observables}
SaveAnimationWindowOnly={save_animation_window_only}
ControlRule="{control_rule}"
FractionSamples={fraction_samples if fraction_samples is not None else 30}
FractionGapOverL={fraction_gap_over_L}
export SOP_FRACTION_SAMPLES="${{SOP_FRACTION_SAMPLES:-$FractionSamples}}"
export SOP_FRACTION_GAP_OVER_L="${{SOP_FRACTION_GAP_OVER_L:-$FractionGapOverL}}"
ForceTopologicalCounts={force_topological_counts}
if [[ "$ForceTopologicalCounts" == "true" ]]; then
  export SOP_FRACTAL_FORCE_COUNTS=1
fi{snapshot_export_str}

extra_args=()
if [[ "$ControlRule" != "relative" ]]; then
  extra_args=("$Properties" "$Mode" "$InitialLayout" "$SurfaceObservables" "$SaveAnimationWindowOnly" "$ControlRule")
elif [[ "$SaveAnimationWindowOnly" != "false" ]]; then
  extra_args=("$Properties" "$Mode" "$InitialLayout" "$SurfaceObservables" "$SaveAnimationWindowOnly")
elif [[ "$SurfaceObservables" != "false" ]]; then
  extra_args=("$Properties" "$Mode" "$InitialLayout" "$SurfaceObservables")
elif [[ "$InitialLayout" != "clustered" ]]; then
  extra_args=("$Properties" "$Mode" "$InitialLayout")
elif [[ "$Mode" != "sop" ]]; then
  extra_args=("$Properties" "$Mode")
elif [[ "$Properties" == "true" ]]; then
  extra_args=("$Properties")
fi

# --- safety knobs (can be overridden at runtime) ---
MEMORY_SAFETY_FRACTION=${{MEMORY_SAFETY_FRACTION:-0.85}}
RAM_MULTIPLIER=${{RAM_MULTIPLIER:-1.20}}
RESERVE_THREADS=${{RESERVE_THREADS:-1}}
MAX_THREADS=${{MAX_THREADS:-}}

# --- check dependencies ---
if ! command -v parallel >/dev/null 2>&1; then
  echo "[ERROR] 'parallel' not found. Install it (e.g., sudo apt-get install parallel) or set multi=False."
  exit 1
fi

if ! command -v /usr/bin/time >/dev/null 2>&1; then
  echo "[ERROR] '/usr/bin/time' not found. It is required to measure peak RAM."
  exit 1
fi

export L p0 P0 seed type c f_T dim num_colors Equilibration Properties Mode InitialLayout SurfaceObservables SaveAnimationWindowOnly ControlRule

TOTAL=$(( num_runs * ${{#rho[@]}} ))
if [[ "$TOTAL" -le 0 ]]; then
  echo "[ERROR] No jobs to run."
  exit 1
fi

# --- detect machine resources ---
EXTERNAL_JOBS="${{RUN_ALL_JOBS:-${{PARALLEL_JOBS:-${{NUM_JOBS:-${{JOBS:-}}}}}}}}"

if [[ -n "$EXTERNAL_JOBS" && "$EXTERNAL_JOBS" -gt 0 ]]; then
  JOBS="$EXTERNAL_JOBS"
  SKIP_FIRST=0
  echo "[INFO] Using external parallel jobs setting: JOBS=$JOBS"
else
  # --- detect machine resources ---
  CPU_THREADS_TOTAL=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)
  CPU_THREADS_LIMIT=$(( CPU_THREADS_TOTAL - RESERVE_THREADS ))
  if [[ "$CPU_THREADS_LIMIT" -lt 1 ]]; then
    CPU_THREADS_LIMIT=1
  fi

  MEM_TOTAL_KB=$(awk '/MemTotal:/ {{print int($2)}}' /proc/meminfo)
  MEM_AVAILABLE_KB=$(awk '/MemAvailable:/ {{print int($2)}}' /proc/meminfo)

  RAM_BUDGET_KB=$(awk -v total="$MEM_TOTAL_KB" -v avail="$MEM_AVAILABLE_KB" -v frac="$MEMORY_SAFETY_FRACTION" '
    BEGIN {{
      budget = int(total * frac)
      if (avail < budget) budget = avail
      if (budget < 1) budget = 1
      print budget
    }}'
  )

  # --- benchmark exactly one simulation to estimate RAM per job ---
  BENCH_RHO=${{rho[0]}}
  BENCH_LOG=$(mktemp)

  echo "[INFO] Running RAM benchmark for this parameter set..."
  echo "[INFO] Benchmark command: ./build/SOP $L $p0 $P0 $seed $type $c $f_T $dim $num_colors $BENCH_RHO $Equilibration ${{extra_args[*]}}"

  if [[ "$Mode" == "fractions" || "$Mode" == "raw_fractions" ]]; then
    SOP_FRACTION_SAMPLES=1 /usr/bin/time -f "%M" -o "$BENCH_LOG" \
      ./build/SOP "$L" "$p0" "$P0" "$seed" "$type" "$c" "$f_T" "$dim" "$num_colors" "$BENCH_RHO" "$Equilibration" "${{extra_args[@]}}" >/dev/null
    SKIP_FIRST=0
  else
    /usr/bin/time -f "%M" -o "$BENCH_LOG" \
      ./build/SOP "$L" "$p0" "$P0" "$seed" "$type" "$c" "$f_T" "$dim" "$num_colors" "$BENCH_RHO" "$Equilibration" "${{extra_args[@]}}" >/dev/null
    SKIP_FIRST=1
  fi

  PEAK_RAM_KB=$(tr -dc '0-9' < "$BENCH_LOG")
  rm -f "$BENCH_LOG"

  if [[ -z "$PEAK_RAM_KB" || "$PEAK_RAM_KB" -le 0 ]]; then
    echo "[ERROR] Could not determine peak RAM from benchmark."
    exit 1
  fi

  PER_JOB_RAM_KB=$(awk -v peak="$PEAK_RAM_KB" -v mult="$RAM_MULTIPLIER" '
    BEGIN {{
      x = int(peak * mult)
      if (x < 1) x = 1
      print x
    }}'
  )

  THREADS_BY_RAM=$(awk -v budget="$RAM_BUDGET_KB" -v per_job="$PER_JOB_RAM_KB" '
    BEGIN {{
      jobs = int(budget / per_job)
      if (jobs < 1) jobs = 1
      print jobs
    }}'
  )

  JOBS=$THREADS_BY_RAM
  if [[ "$JOBS" -gt "$CPU_THREADS_LIMIT" ]]; then
    JOBS=$CPU_THREADS_LIMIT
  fi

  if [[ -n "$MAX_THREADS" && "$MAX_THREADS" -lt "$JOBS" ]]; then
    JOBS=$MAX_THREADS
  fi

  if [[ "$JOBS" -lt 1 ]]; then
    JOBS=1
  fi

  PEAK_RAM_GB=$(awk -v kb="$PEAK_RAM_KB" 'BEGIN {{printf "%.2f", kb/1024/1024}}')
  PER_JOB_RAM_GB=$(awk -v kb="$PER_JOB_RAM_KB" 'BEGIN {{printf "%.2f", kb/1024/1024}}')
  RAM_BUDGET_GB=$(awk -v kb="$RAM_BUDGET_KB" 'BEGIN {{printf "%.2f", kb/1024/1024}}')
  MEM_TOTAL_GB=$(awk -v kb="$MEM_TOTAL_KB" 'BEGIN {{printf "%.2f", kb/1024/1024}}')
  MEM_AVAILABLE_GB=$(awk -v kb="$MEM_AVAILABLE_KB" 'BEGIN {{printf "%.2f", kb/1024/1024}}')

  echo "[INFO] Peak RAM from benchmark   : $PEAK_RAM_GB GB"
  echo "[INFO] RAM per parallel job      : $PER_JOB_RAM_GB GB (with RAM_MULTIPLIER=$RAM_MULTIPLIER)"
  echo "[INFO] Total RAM                 : $MEM_TOTAL_GB GB"
  echo "[INFO] Available RAM             : $MEM_AVAILABLE_GB GB"
  echo "[INFO] Usable RAM budget         : $RAM_BUDGET_GB GB (MEMORY_SAFETY_FRACTION=$MEMORY_SAFETY_FRACTION)"
  echo "[INFO] CPU threads total         : $CPU_THREADS_TOTAL"
  echo "[INFO] CPU threads usable        : $CPU_THREADS_LIMIT (RESERVE_THREADS=$RESERVE_THREADS)"
  echo "[INFO] Selected GNU parallel jobs: $JOBS"
fi

if [[ "$SKIP_FIRST" -eq 1 && "$TOTAL" -eq 1 ]]; then
  echo "[INFO] Only one simulation requested. Benchmark run already completed it."
  exit 0
fi

# --- build task list ---
TASK_FILE=$(mktemp)
trap 'rm -f "$TASK_FILE"' EXIT
skip_flag="$SKIP_FIRST"

for ((run=1; run<=num_runs; run++)); do
  for idx in "${{!rho[@]}}"; do
    RHO=${{rho[$idx]}}

    if [[ "$skip_flag" -eq 1 ]]; then
      skip_flag=0
      continue
    fi

    printf '%s\t%s\n' "$RHO" "$run" >> "$TASK_FILE"
  done
done

REMAINING=$(wc -l < "$TASK_FILE")
if [[ "$SKIP_FIRST" -eq 1 ]]; then
  echo "[INFO] Benchmark counted as the first completed simulation (1/$TOTAL)."
fi
echo "[INFO] Running $REMAINING simulation(s) with -j $JOBS ..."

parallel -j "$JOBS" --bar --halt soon,fail=1 --colsep '\t' '
  RHO={{1}}
  RUN={{2}}
  extra_args=()
  if [[ "$ControlRule" != "relative" ]]; then
    extra_args=("$Properties" "$Mode" "$InitialLayout" "$SurfaceObservables" "$SaveAnimationWindowOnly" "$ControlRule")
  elif [[ "$SaveAnimationWindowOnly" != "false" ]]; then
    extra_args=("$Properties" "$Mode" "$InitialLayout" "$SurfaceObservables" "$SaveAnimationWindowOnly")
  elif [[ "$SurfaceObservables" != "false" ]]; then
    extra_args=("$Properties" "$Mode" "$InitialLayout" "$SurfaceObservables")
  elif [[ "$InitialLayout" != "clustered" ]]; then
    extra_args=("$Properties" "$Mode" "$InitialLayout")
  elif [[ "$Mode" != "sop" ]]; then
    extra_args=("$Properties" "$Mode")
  elif [[ "$Properties" == "true" ]]; then
    extra_args=("$Properties")
  fi
  ./build/SOP "$L" "$p0" "$P0" "$seed" "$type" "$c" "$f_T" "$dim" "$num_colors" "$RHO" "$Equilibration" "${{extra_args[@]}}"
' :::: "$TASK_FILE"

echo "All runs completed."
"""
    else:
        script = f"""\
#!/usr/bin/env bash
set -euo pipefail

# --- go to project root ---
cd ..

# --- parameters ---
num_runs={num_runs}
rho=({rho_list})
L={L}
p0={p0}
P0={P0}
seed={seed}
type="{type_perc}"
c={c}
f_T={f_T}
dim={dim}
num_colors={num_colors}
Equilibration={equlibration}
Properties={properties}
Mode="{mode}"
InitialLayout="{initial_layout}"
SurfaceObservables={surface_observables}
SaveAnimationWindowOnly={save_animation_window_only}
ControlRule="{control_rule}"
FractionSamples={fraction_samples if fraction_samples is not None else 30}
FractionGapOverL={fraction_gap_over_L}
export SOP_FRACTION_SAMPLES="${{SOP_FRACTION_SAMPLES:-$FractionSamples}}"
export SOP_FRACTION_GAP_OVER_L="${{SOP_FRACTION_GAP_OVER_L:-$FractionGapOverL}}"
ForceTopologicalCounts={force_topological_counts}
if [[ "$ForceTopologicalCounts" == "true" ]]; then
  export SOP_FRACTAL_FORCE_COUNTS=1
fi{snapshot_export_str}

extra_args=()
if [[ "$ControlRule" != "relative" ]]; then
  extra_args=("$Properties" "$Mode" "$InitialLayout" "$SurfaceObservables" "$SaveAnimationWindowOnly" "$ControlRule")
elif [[ "$SaveAnimationWindowOnly" != "false" ]]; then
  extra_args=("$Properties" "$Mode" "$InitialLayout" "$SurfaceObservables" "$SaveAnimationWindowOnly")
elif [[ "$SurfaceObservables" != "false" ]]; then
  extra_args=("$Properties" "$Mode" "$InitialLayout" "$SurfaceObservables")
elif [[ "$InitialLayout" != "clustered" ]]; then
  extra_args=("$Properties" "$Mode" "$InitialLayout")
elif [[ "$Mode" != "sop" ]]; then
  extra_args=("$Properties" "$Mode")
elif [[ "$Properties" == "true" ]]; then
  extra_args=("$Properties")
fi

# --- pretty progress bar (single-line) ---
progress_bar() {{
  local curr="$1" total="$2" width="${{3:-40}}"
  local pct=$(( 100 * curr / total ))
  local filled=$(( width * curr / total ))
  local empty=$(( width - filled ))
  local bar spaces
  printf -v bar '%*s' "$filled"
  bar=${{bar// /#}}
  printf -v spaces '%*s' "$empty"
  printf "\r[%s%s] %3d%% (%d/%d)" "$bar" "$spaces" "$pct" "$curr" "$total"
}}

TOTAL=$(( num_runs * ${{#rho[@]}} ))
DONE=0

if [[ -z "${{VERBOSE+x}}" ]]; then
  if [[ "$Mode" == "fractions" || "$Mode" == "raw_fractions" ]]; then
    VERBOSE=1
  else
    VERBOSE=0
  fi
fi

for ((run=1; run<=num_runs; run++)); do
  for idx in "${{!rho[@]}}"; do
    RHO=${{rho[$idx]}}

    DONE=$((DONE+1))
    progress_bar "$DONE" "$TOTAL" 46

    if [[ "$VERBOSE" -eq 1 ]]; then
      echo
      echo "./build/SOP $L $p0 $P0 $seed $type $c $f_T $dim $num_colors $RHO $Equilibration ${{extra_args[*]}}"
      ./build/SOP "$L" "$p0" "$P0" "$seed" "$type" "$c" "$f_T" "$dim" "$num_colors" "$RHO" "$Equilibration" "${{extra_args[@]}}"
    else
      ./build/SOP "$L" "$p0" "$P0" "$seed" "$type" "$c" "$f_T" "$dim" "$num_colors" "$RHO" "$Equilibration" "${{extra_args[@]}}" >/dev/null
    fi
  done
done

echo
echo "All runs completed."
"""

    script = textwrap.dedent(script)
    path = os.path.join("../shells", exec_name)

    with open(path, "w", encoding="utf-8") as f:
        f.write(script)

    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)

    print(f"✅ Shell script created at {path}")


def custom_range(start: float, stop: float, n_points: int, ndigits: int = 8):
    if n_points <= 0:
        return []
    if n_points == 1:
        return [round(start, ndigits)]

    xs = np.linspace(start, stop, n_points, endpoint=True, dtype=float)
    xs = np.round(xs, ndigits)
    xs[0] = round(start, ndigits)
    xs[-1] = round(stop, ndigits)
    return xs.tolist()


@dataclass
class SOPParams:
    L: int
    d: int
    n_c: int
    epsilon0: float
    kappa0: float
    beta: float
    area_front: int
    f_T_raw: float
    f_T: float
    c_raw: float
    c: float
    c_str: str
    notes: str


def _round_sig(x: float, sig: int = 2) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    e = math.floor(math.log10(abs(x)))
    factor = 10 ** (e - sig + 1)
    return round(x / factor) * factor


def _scientific_round_and_format(x: float, mantissa_decimals: int = 0):
    """
    Round x to a scientific-notation mantissa and return (float_value, str_value).

    Examples with mantissa_decimals=0:
        1.22e-05 -> (1e-05, '1e-5')
        8.62e-06 -> (9e-06, '9e-6')
        9.60e-07 -> (1e-06, '1e-6')
    """
    if x == 0 or not math.isfinite(x):
        s = f"{x:.{mantissa_decimals}e}"
        return x, s

    sign = -1 if x < 0 else 1
    a = abs(x)
    exp = math.floor(math.log10(a))
    mant = a / (10 ** exp)

    if mantissa_decimals == 0:
        mant_rounded = int(round(mant))
    else:
        mant_rounded = round(mant, mantissa_decimals)

    if mant_rounded == 0:
        exp -= 1
        mant_rounded = 1 if mantissa_decimals == 0 else 1.0
    if mant_rounded >= 10:
        mant_rounded = 1 if mantissa_decimals == 0 else 1.0
        exp += 1

    value = sign * float(mant_rounded) * (10 ** exp)

    if mantissa_decimals == 0:
        mant_str = f"{sign * int(mant_rounded):d}"
    else:
        mant_str = f"{sign * float(mant_rounded):.{mantissa_decimals}f}"

    str_value = f"{mant_str}e{exp:+d}".replace("+0", "+").replace("-0", "-")
    return value, str_value


def _round_and_format_decimal(x: float, ndigits: int = 8):
    value = round(float(x), ndigits)
    text = f"{value:.{ndigits}f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return value, text


def sop_choose_params(
    L: int,
    n_c: int,
    d: int = 3,
    beta: float = 0.5,
    epsilon0: float = 0.05,
    kappa0: float = 0.010,
    fT_ndigits: int = 8,
    c_mantissa_decimals: int = 2,
) -> SOPParams:
    """
    Suggest c and f_T for the rescaled SOP model.

    New model:
        p_i(t+1) = p_i(t) + c * (f_T - f_i(t))

    Interpretation used here:
      - f_T = epsilon0 is the target active-front fraction.
      - kappa0 is the target magnitude for c * f_T.
      - beta optionally compensates the number of colors through n_c**beta.

    Therefore:
        c_raw = (kappa0 * n_c**beta) / f_T

    This helper no longer returns N_T or k because those are no longer SOP inputs.
    """
    if d not in (2, 3):
        raise ValueError("please, enter with d = 2 or 3")

    area_front = int(L ** (d - 1))
    f_T_raw = float(epsilon0)
    f_T, _ = _round_and_format_decimal(f_T_raw, ndigits=fT_ndigits)

    c_raw = (float(kappa0) * (n_c ** float(beta))) / f_T if f_T > 0 else float("nan")
    c, c_str = _scientific_round_and_format(c_raw, mantissa_decimals=c_mantissa_decimals)

    notes = (
        f"Novo modelo reescalado: p_i(t+1) = p_i(t) + c [f_T - f_i(t)]. "
        f"Usado: f_T=epsilon0={epsilon0}, kappa0={kappa0}, beta={beta}. "
        f"Neste helper, kappa0 controla aproximadamente c*f_T, com compensação n_c^beta. "
        f"Os antigos parâmetros N_T e k não são mais entradas do executável."
    )

    return SOPParams(
        L=L,
        d=d,
        n_c=n_c,
        epsilon0=epsilon0,
        kappa0=kappa0,
        beta=beta,
        area_front=area_front,
        f_T_raw=f_T_raw,
        f_T=f_T,
        c_raw=c_raw,
        c=c,
        c_str=c_str,
        notes=notes,
    )


def sop_choose_c_fT(
    L: int,
    n_c: int,
    d: int = 3,
    beta: float = 0.5,
    epsilon0: float = 0.05,
    kappa0: float = 0.010,
    fT_ndigits: int = 8,
    c_mantissa_decimals: int = 2,
):
    """
    Return (c_float, f_T_float, c_str, f_T_str) for the new SOP executable.

    Use directly in shell_data as:
        c, f_T, c_str, f_T_str = sop_choose_c_fT(...)
        shell_data(..., c=c, f_T=f_T, ...)
    """
    params = sop_choose_params(
        L=L,
        n_c=n_c,
        d=d,
        beta=beta,
        epsilon0=epsilon0,
        kappa0=kappa0,
        fT_ndigits=fT_ndigits,
        c_mantissa_decimals=c_mantissa_decimals,
    )
    _, f_T_str = _round_and_format_decimal(params.f_T, ndigits=fT_ndigits)
    return params.c, params.f_T, params.c_str, f_T_str


# Backward-compatible alias name, but now returning the new inputs.
# Prefer using sop_choose_c_fT in new notebooks/scripts.
def sop_choose_NT_k(
    L: int,
    n_c: int,
    d: int = 3,
    beta: float = 0.5,
    epsilon0: float = 0.05,
    kappa0: float = 0.010,
    mantissa_decimals: int = 2,
):
    """
    Deprecated compatibility wrapper.

    The old name is kept only to avoid import errors in existing notebooks.
    It now returns the new executable inputs:
        (c_float, f_T_float, c_str, f_T_str)
    """
    return sop_choose_c_fT(
        L=L,
        n_c=n_c,
        d=d,
        beta=beta,
        epsilon0=epsilon0,
        kappa0=kappa0,
        c_mantissa_decimals=mantissa_decimals,
    )


# -----------------------------
# Exemplo de uso:
# p = sop_choose_params(L=512, n_c=4)
# print(p.c, p.f_T, p.c_str)
# shell_data(
#     L=512,
#     type_perc="bond",
#     p0=1.0,
#     seed=-1,
#     c=p.c,
#     f_T=p.f_T,
#     dim=3,
#     num_colors=4,
#     num_runs=10,
#     rho=[0.25],
#     exec_name="run_L512_nc4.sh",
#     P0=0.1,
#     equlibration=1,
#     multi=True,
# )


# =============================================================================
# CONVERGENCE ANALYSIS AND SAMPLE EQUALIZATION FUNCTIONS
# =============================================================================

CONST_FT_GRID = {
    'node': [
        0.3000000,
        0.2847368,
        0.2694737,
        0.2542105,
        0.2389474,
        0.2236842,
        0.2084211,
        0.1931579,
        0.1778947,
        0.1626316,
    ],
    'bond': [
        0.3000000,
        0.2847368,
        0.2694737,
        0.2542105,
        0.2389474,
        0.2236842,
        0.2084211,
        0.1931579,
    ],
}

DEFAULT_TARGET_SAMPLES_POR_L = {
    1024: 1000,
    1448: 900,
    2048: 800,
    2896: 600,
    4096: 500,
    5793: 400,
    8192: 300,
    11585: 200,
    16384: 100,
}


def find_dynamic_file(script_dir: Path | None = None) -> Path:
    if script_dir is None:
        script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / 'SOP_data/all_data_dynamic.dat',
        script_dir.parent.parent / 'SOP_data/all_data_dynamic.dat',
        Path('../SOP_data/all_data_dynamic.dat'),
        Path('SOP_data/all_data_dynamic.dat'),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Nao foi possivel encontrar 'SOP_data/all_data_dynamic.dat'.")


def load_filtered_dynamic(
    dim: int = 2,
    nc: int = 1,
    control_rule: str = 'linear',
    c: float = 0.05,
    p0: float = 0.8,
    P0: float = 0.2,
    l_min: int = 1024,
    script_dir: Path | None = None,
) -> pd.DataFrame:
    p_data = find_dynamic_file(script_dir)
    raw = pd.read_csv(p_data, sep=r'\s+')
    mask = (
        (raw['dim'] == dim)
        & (raw['nc'] == nc)
        & (raw['order'] == 0)
        & (raw['stat_window'] == 0)
        & (raw['control_rule'] == control_rule)
        & np.isclose(raw['c'], c)
        & np.isclose(raw['p0'], p0)
        & np.isclose(raw['P0'], P0)
        & (raw['L'] >= l_min)
    )
    sub = raw.loc[mask].copy()
    sub['stab'] = sub['N_samples_perc'] / sub['N_samples']
    return sub


def resolve_existing_ft_and_samples(
    df_sub: pd.DataFrame, type_perc: str, L: int, ft_target: float, atol: float = 1e-4
) -> tuple[float, int]:
    """
    Procura em all_data_dynamic.dat se ja existe uma serie em (type_perc, L) com f_T
    dentro de `atol` de `ft_target`. Se existir, retorna o f_T exato gravado e seu N_samples.
    Caso contrario, retorna (round(ft_target, 7), 0).
    """
    cand = df_sub[(df_sub['type_perc'] == type_perc) & (df_sub['L'] == L)].copy()
    if cand.empty:
        return round(float(ft_target), 7), 0
    cand['diff'] = (cand['f_T'] - ft_target).abs()
    near = cand[cand['diff'] <= atol].sort_values('diff')
    if near.empty:
        return round(float(ft_target), 7), 0
    best = near.iloc[0]
    return float(best['f_T']), int(best['N_samples'])


def build_all_convergence_jobs(
    df_sub: pd.DataFrame,
    type_lst: list[str] | None = None,
    target_samples_por_L: dict[int, int] | None = None,
    const_ft_grid: dict[str, list[float]] | None = None,
    l_min: int = 1024,
    script_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Monta todos os pontos (prefixo, descricao_grupo, type_perc, L, f_T, curr_samples, target_samples, needed)
    truncando sempre L >= l_min para:
      1) minimum_power: pontos de f_{T,min}(L) (L >= l_min)
      2) conv_const:    cortes de f_T constante decrescente (Tabela 1, L >= l_min)
      3) conv_scaled:   escalonamento proporcional f_T(L) = lambda * f_{T,min}(L) (Tabela 2, L >= l_min)
    evitando duplicatas de (type_perc, L, f_T).
    """
    if type_lst is None:
        type_lst = ['node', 'bond']
    if target_samples_por_L is None:
        target_samples_por_L = DEFAULT_TARGET_SAMPLES_POR_L
    if const_ft_grid is None:
        const_ft_grid = CONST_FT_GRID
    if script_dir is None:
        script_dir = Path(__file__).resolve().parent

    sop_data_dir = None
    for candidate in [
        script_dir.parent / 'SOP_data',
        script_dir.parent.parent / 'SOP_data',
        Path('../SOP_data'),
        Path('SOP_data'),
    ]:
        if candidate.exists() and candidate.is_dir():
            sop_data_dir = candidate
            break

    seen_keys = set()
    all_records = []

    def add_point(prefix: str, group_desc: str, type_perc: str, L: int, ft_req: float, atol: float = 1e-4):
        if int(L) < l_min or int(L) not in target_samples_por_L:
            return
        ft_exact, curr = resolve_existing_ft_and_samples(df_sub, type_perc, L, ft_req, atol=atol)
        key = (type_perc, int(L), f'{ft_exact:.6e}')
        if key in seen_keys:
            return
        seen_keys.add(key)
        target = target_samples_por_L[int(L)]
        needed = max(0, target - curr)
        all_records.append({
            'prefix': prefix,
            'group': group_desc,
            'type_perc': type_perc,
            'L': int(L),
            'f_T': ft_exact,
            'curr': curr,
            'target': target,
            'needed': needed,
        })

    # 1. Pontos de f_{T,min}(L) (smooth_refined.csv, L >= l_min)
    for tp in type_lst:
        if sop_data_dir is not None:
            ref_path = sop_data_dir / f'ft_min_max_2D_{tp}_smooth_refined.csv'
            if ref_path.exists():
                df_ref = pd.read_csv(ref_path)
                df_ref = df_ref[df_ref['L'] >= l_min].sort_values('L')
                for _, r in df_ref.iterrows():
                    add_point('minimum_power', 'f_T,min(L)', tp, int(r['L']), float(r['f_T']), atol=2e-5)

    # 2. Pontos de f_T constante (Tabela 1, L >= l_min)
    df_stab = df_sub[(df_sub['stab'] >= 0.98) & (df_sub['N_samples'] >= 5) & (df_sub['p_mean'] < 0.9)]
    for tp in type_lst:
        if tp in const_ft_grid:
            for ft_c in const_ft_grid[tp]:
                s_ft = df_stab[
                    (df_stab['type_perc'] == tp)
                    & (df_stab['L'] >= l_min)
                    & np.isclose(df_stab['f_T'], ft_c, atol=2e-5)
                ].sort_values('L')
                for _, r in s_ft.iterrows():
                    add_point('conv_const', f'const f_T={ft_c:.4f}', tp, int(r['L']), float(r['f_T']), atol=2e-5)

    # 3. Pontos de f_T(L) = lambda * f_{T,min}(L) (Tabela 2, L >= l_min)
    for tp in type_lst:
        if sop_data_dir is not None:
            ref_path = sop_data_dir / f'ft_min_max_2D_{tp}_smooth_refined.csv'
            if ref_path.exists():
                df_ref = pd.read_csv(ref_path)
                df_ref = df_ref[df_ref['L'] >= l_min].sort_values('L')
                max_lam = 0.30 / df_ref['f_T'].max()
                lams = np.linspace(max_lam, 1.0, 6)[:-1]  # lambda = 1.0 ja esta em minimum_power
                for lam in lams:
                    for _, r in df_ref.iterrows():
                        L_val = int(r['L'])
                        ft_lam = float(lam * r['f_T'])
                        add_point('conv_scaled', f'lambda={lam:.2f}*f_T,min', tp, L_val, ft_lam, atol=1e-4)

    return pd.DataFrame(all_records)


def generate_convergence_shells(
    dim: int = 2,
    nc: int = 1,
    c: float = 0.05,
    p0: float = 0.8,
    P0: float = 0.2,
    seed: int = -1,
    multi: bool = True,
    Equilibration: str = 'false',
    Properties: str = 'false',
    Mode: str = 'growth_test',
    InitialLayout: str = 'clustered',
    ControlRule: str = 'linear',
    type_lst: list[str] | None = None,
    l_min: int = 1024,
    target_samples_por_L: dict[int, int] | None = None,
    const_ft_grid: dict[str, list[float]] | None = None,
    script_dir: Path | None = None,
):
    """
    Gera scripts shell para equalizar N_samples de convergencia.
    """
    if type_lst is None:
        type_lst = ['node', 'bond']
    if target_samples_por_L is None:
        target_samples_por_L = DEFAULT_TARGET_SAMPLES_POR_L
    if script_dir is None:
        script_dir = Path(__file__).resolve().parent

    df_sub = load_filtered_dynamic(
        dim=dim, nc=nc, control_rule=ControlRule, c=c, p0=p0, P0=P0,
        l_min=l_min, script_dir=script_dir
    )
    df_jobs = build_all_convergence_jobs(
        df_sub, type_lst=type_lst, target_samples_por_L=target_samples_por_L,
        const_ft_grid=const_ft_grid, l_min=l_min, script_dir=script_dir
    )

    print('=' * 105)
    print(f'VERIFICACAO DE AMOSTRAS E GERACAO DE SHELLS PARA L >= {l_min} (TODOS OS CASOS)')
    print('=' * 105)

    shells_dir = None
    for cand in [
        script_dir.parent / 'shells',
        script_dir.parent.parent / 'shells',
        Path('../shells'),
        Path('shells'),
    ]:
        if cand.parent.exists():
            shells_dir = cand
            break
    if shells_dir is None:
        shells_dir = script_dir.parent / 'shells'
    shells_dir.mkdir(parents=True, exist_ok=True)

    for pat in ['minimum_power_*.sh', 'conv_const_*.sh', 'conv_scaled_*.sh']:
        for old_f in shells_dir.glob(pat):
            old_f.unlink()

    active_jobs = []
    for _, row in df_jobs.iterrows():
        prefix = row['prefix']
        grp = row['group']
        tp = row['type_perc']
        L = int(row['L'])
        ft = float(row['f_T'])
        curr = int(row['curr'])
        target = int(row['target'])
        needed = int(row['needed'])

        exec_name = (
            f'{prefix}_{tp}_L_{L}_ft_{ft:.6e}_c_{c:g}_nc_{nc}_dim_{dim}'
            f'_p0_{p0:g}_P0_{P0:g}_{ControlRule}_{Mode}_{InitialLayout}.sh'
        )
        if needed > 0:
            active_jobs.append((prefix, grp, tp, L, ft, curr, target, needed, exec_name))

    for prefix, title in [
        ('minimum_power', f'1) PONTOS DE f_{{T,min}}(L) (L >= {l_min})'),
        ('conv_const', f'2) TABELA 1: PONTOS DE f_T CONSTANTE DECRESCENTE (L >= {l_min})'),
        ('conv_scaled', f'3) TABELA 2: PONTOS DE f_T(L) = lambda * f_{{T,min}}(L) (L >= {l_min})'),
    ]:
        sub_df = df_jobs[df_jobs['prefix'] == prefix]
        pend_df = sub_df[sub_df['needed'] > 0]
        print(f'\n{title}')
        print(f'  Total de pontos verificados: {len(sub_df)} | Com amostras faltantes: {len(pend_df)} | Amostras a rodar: {pend_df["needed"].sum()}')
        if not pend_df.empty:
            print('-' * 105)
            print(f"{'Grupo':>22} | {'Tipo':>5} | {'L':>6} | {'f_T':>12} | {'Atual':>6} | {'Alvo':>6} | {'Diferenca (+N)':>14}")
            print('-' * 105)
            for _, r in pend_df.iterrows():
                print(
                    f"{r['group']:>22} | {r['type_perc']:>5} | {int(r['L']):6d} | "
                    f"{r['f_T']:12.7f} | {int(r['curr']):6d} | {int(r['target']):6d} | +{int(r['needed']):13d}"
                )

    previous_dir = Path.cwd()
    try:
        working_dir = script_dir.parent if script_dir.name == 'src' else script_dir
        os.chdir(working_dir)
        for prefix, grp, tp, L, ft, curr, target, num_runs, exec_name in active_jobs:
            shell_data(
                L, tp, p0, seed, c, ft, dim, nc, num_runs, [1 / nc],
                exec_name, P0, Equilibration, multi,
                properties=Properties, mode=Mode,
                initial_layout=InitialLayout, control_rule=ControlRule,
            )
    finally:
        os.chdir(previous_dir)

    print('\n' + '=' * 105)
    for prefix in ['minimum_power', 'conv_const', 'conv_scaled']:
        cnt = sum(1 for j in active_jobs if j[0] == prefix)
        runs = sum(j[7] for j in active_jobs if j[0] == prefix)
        print(f"  Prefixo '{prefix}_*.sh': {cnt:3d} arquivos gerados ({runs:6d} amostras faltantes)")
    print(f"  TOTAL GERAL (L >= {l_min}): {len(active_jobs)} arquivos shell gerados em shells/ ({sum(j[7] for j in active_jobs)} amostras)")
    print('=' * 105)
