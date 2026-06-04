import os
import stat
import textwrap
import numpy as np
import math
from dataclasses import dataclass


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
    P0: float,
    equlibration,
    multi: bool = False,
    properties=False,
    mode: str = "sop",
):
    """
    Generate a shell script to run SOP multiple times.

    New SOP executable signature:
        ./build/SOP <L> <p0> <seed> <type_percolation> <c> <f_T> <dim> <num_colors> <rho_val> <P0> <Equilibration> [Properties] [Mode]

    The old inputs k and N_T were removed. The update rule is now:
        p_i(t+1) = p_i(t) + c * (f_T - f_i(t))

    If multi=True, the generated shell script:
      1. Runs a single benchmark simulation.
      2. Measures peak RAM with /usr/bin/time.
      3. Chooses GNU parallel jobs automatically based on:
         - current available RAM,
         - total machine RAM,
         - total CPU threads,
         - configurable safety margins.

    Runtime knobs (environment variables in the generated shell):
      - MEMORY_SAFETY_FRACTION (default: 0.85)
      - RAM_MULTIPLIER         (default: 1.20)
      - RESERVE_THREADS        (default: 1)
      - MAX_THREADS            (optional hard cap)

    Important:
    - CPU clock limiting is NOT handled here.
    - It should be handled centrally by run_all.sh.
    """

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
    if mode not in ("sop", "growth_test"):
        raise ValueError("mode must be 'sop' or 'growth_test'")

    if isinstance(properties, bool):
        properties = "true" if properties else "false"
    else:
        properties = str(properties).strip().lower()

    if properties not in ("true", "false"):
        raise ValueError("properties must be true/false")

    if mode == "growth_test" and properties == "true":
        raise ValueError("growth_test currently requires properties=false")

    os.makedirs("../shells", exist_ok=True)
    print("Creating shell script file in ../shells/")

    rho_list = " ".join(map(str, rho))

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
seed={seed}
type="{type_perc}"
c={c}
f_T={f_T}
dim={dim}
num_colors={num_colors}
P0={P0}
Equilibration={equlibration}
Properties={properties}
Mode="{mode}"

extra_args=()
if [[ "$Mode" != "sop" ]]; then
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

export L p0 seed type c f_T dim num_colors P0 Equilibration Properties Mode

TOTAL=$(( num_runs * ${{#rho[@]}} ))
if [[ "$TOTAL" -le 0 ]]; then
  echo "[ERROR] No jobs to run."
  exit 1
fi

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
echo "[INFO] Benchmark command: ./build/SOP $L $p0 $seed $type $c $f_T $dim $num_colors $BENCH_RHO $P0 $Equilibration ${{extra_args[*]}}"

/usr/bin/time -f "%M" -o "$BENCH_LOG" \
  ./build/SOP "$L" "$p0" "$seed" "$type" "$c" "$f_T" "$dim" "$num_colors" "$BENCH_RHO" "$P0" "$Equilibration" "${{extra_args[@]}}" >/dev/null

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

if [[ "$TOTAL" -eq 1 ]]; then
  echo "[INFO] Only one simulation requested. Benchmark run already completed it."
  exit 0
fi

# --- build remaining task list, skipping the benchmarked first task ---
TASK_FILE=$(mktemp)
trap 'rm -f "$TASK_FILE"' EXIT
SKIP_FIRST=1

for ((run=1; run<=num_runs; run++)); do
  for idx in "${{!rho[@]}}"; do
    RHO=${{rho[$idx]}}

    if [[ "$SKIP_FIRST" -eq 1 ]]; then
      SKIP_FIRST=0
      continue
    fi

    printf '%s\t%s\n' "$RHO" "$run" >> "$TASK_FILE"
  done
done

REMAINING=$(( TOTAL - 1 ))
echo "[INFO] Benchmark counted as the first completed simulation (1/$TOTAL)."
echo "[INFO] Running remaining $REMAINING simulation(s) with -j $JOBS ..."

parallel -j "$JOBS" --bar --halt soon,fail=1 --colsep '\t' '
  RHO={{1}}
  RUN={{2}}
  extra_args=()
  if [[ "$Mode" != "sop" ]]; then
    extra_args=("$Properties" "$Mode")
  elif [[ "$Properties" == "true" ]]; then
    extra_args=("$Properties")
  fi
  ./build/SOP "$L" "$p0" "$seed" "$type" "$c" "$f_T" "$dim" "$num_colors" "$RHO" "$P0" "$Equilibration" "${{extra_args[@]}}"
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
seed={seed}
type="{type_perc}"
c={c}
f_T={f_T}
dim={dim}
num_colors={num_colors}
Equilibration={equlibration}
P0={P0}
Properties={properties}
Mode="{mode}"

extra_args=()
if [[ "$Mode" != "sop" ]]; then
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

VERBOSE=${{VERBOSE:-0}}

for ((run=1; run<=num_runs; run++)); do
  for idx in "${{!rho[@]}}"; do
    RHO=${{rho[$idx]}}

    DONE=$((DONE+1))
    progress_bar "$DONE" "$TOTAL" 46

    if [[ "$VERBOSE" -eq 1 ]]; then
      echo
      echo "./build/SOP $L $p0 $seed $type $c $f_T $dim $num_colors $RHO $P0 $Equilibration ${{extra_args[*]}}"
      ./build/SOP "$L" "$p0" "$seed" "$type" "$c" "$f_T" "$dim" "$num_colors" "$RHO" "$P0" "$Equilibration" "${{extra_args[@]}}"
    else
      ./build/SOP "$L" "$p0" "$seed" "$type" "$c" "$f_T" "$dim" "$num_colors" "$RHO" "$P0" "$Equilibration" "${{extra_args[@]}}" >/dev/null
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
