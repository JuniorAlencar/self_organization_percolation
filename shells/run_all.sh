#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob

self="$(realpath "$0")"

EXCLUDED_SCRIPTS=(
  "install_python_dependencies.sh"
  "run_python.sh"
)

# -----------------------------------------------------------------------------
# CPU throttling options
# -----------------------------------------------------------------------------
LIMIT_CPU_CLOCK="${LIMIT_CPU_CLOCK:-1}"
CPU_MAX_FREQ="${CPU_MAX_FREQ:-4.0GHz}"
CPU_GOVERNOR="${CPU_GOVERNOR:-ondemand}"

# -----------------------------------------------------------------------------
# Automatic RAM-based job tuning
# -----------------------------------------------------------------------------
# AUTO_RAM_JOBS=1:
#   For each script, run a probe using 1 job, measure peak RSS, estimate how many
#   parallel jobs fit in RAM, then rerun the script with that job count.
#
# IMPORTANT:
#   The child .sh files should use one of these variables in GNU parallel or loops:
#     RUN_ALL_JOBS, NUM_JOBS, N_JOBS, JOBS, NUM_THREADS, THREADS, PARALLEL_JOBS
#   If they call GNU parallel directly, this file installs a temporary wrapper that
#   forces the desired --jobs value.
AUTO_RAM_JOBS="${AUTO_RAM_JOBS:-1}"

# Use only this fraction of the selected RAM basis. With 30 GB and a 10 GB probe,
# 0.80 gives floor(30*0.80/10)=2 jobs.
RAM_USE_FRACTION="${RAM_USE_FRACTION:-0.95}"

# Extra fixed reserve, in GB, subtracted after RAM_USE_FRACTION.
RAM_RESERVE_GB="${RAM_RESERVE_GB:-0}"

# RAM_BASIS=total     -> base calculation on MemTotal
# RAM_BASIS=available -> base calculation on current MemAvailable
RAM_BASIS="${RAM_BASIS:-total}"

# Optional multiplier applied to measured peak RSS. Use >1.0 for more safety.
RAM_PEAK_MULTIPLIER="${RAM_PEAK_MULTIPLIER:-1.00}"

RAM_MIN_JOBS="${RAM_MIN_JOBS:-1}"
RAM_MAX_JOBS="${RAM_MAX_JOBS:-$(nproc --all 2>/dev/null || echo 1)}"

# Where probe/run logs are stored.
RAM_LOG_DIR="${RAM_LOG_DIR:-.run_all_ram_logs}"

# If GNU parallel is used by child scripts during probe, this wrapper tries to stop
# after the first successful job. Set to 0 if your GNU parallel version rejects it.
RUN_ALL_PROBE_ONE_PARALLEL_JOB="${RUN_ALL_PROBE_ONE_PARALLEL_JOB:-1}"

# Optional timeout for the probe, e.g. RAM_PROBE_TIMEOUT=20m. Use 0 to disable.
RAM_PROBE_TIMEOUT="${RAM_PROBE_TIMEOUT:-0}"

OLD_GOVERNOR=""
OLD_MAX_FREQ=""
SUDO_KEEPALIVE_PID=""
PARALLEL_WRAPPER_DIR=""
REAL_PARALLEL=""

SCRIPT_START_EPOCH="$(date +%s)"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

elapsed_hms() {
  local total="$1"
  local h=$(( total / 3600 ))
  local m=$(( (total % 3600) / 60 ))
  local s=$(( total % 60 ))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

kb_to_gb() {
  local kb="$1"
  awk -v kb="$kb" 'BEGIN { printf "%.2f", kb/1024/1024 }'
}

safe_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  echo "$s"
}

mem_kb() {
  local field="MemTotal:"
  case "$RAM_BASIS" in
    total)
      field="MemTotal:"
      ;;
    available)
      field="MemAvailable:"
      ;;
    *)
      log "[WARN] Invalid RAM_BASIS='$RAM_BASIS'. Falling back to total."
      field="MemTotal:"
      ;;
  esac

  awk -v field="$field" '$1 == field { print $2; exit }' /proc/meminfo
}

calculate_jobs_from_ram() {
  local peak_kb_raw="$1"
  local basis_kb="$2"

  awk \
    -v peak_kb_raw="$peak_kb_raw" \
    -v basis_kb="$basis_kb" \
    -v fraction="$RAM_USE_FRACTION" \
    -v reserve_gb="$RAM_RESERVE_GB" \
    -v peak_multiplier="$RAM_PEAK_MULTIPLIER" \
    -v min_jobs="$RAM_MIN_JOBS" \
    -v max_jobs="$RAM_MAX_JOBS" \
    'BEGIN {
      peak_kb = peak_kb_raw * peak_multiplier
      reserve_kb = reserve_gb * 1024 * 1024
      budget_kb = basis_kb * fraction - reserve_kb

      if (peak_kb <= 0 || budget_kb <= 0) {
        jobs = min_jobs
      } else {
        jobs = int(budget_kb / peak_kb)
      }

      if (jobs < min_jobs) jobs = min_jobs
      if (jobs > max_jobs) jobs = max_jobs
      printf "%d", jobs
    }'
}

extract_peak_rss_kb() {
  local time_log="$1"
  awk -F: '/Maximum resident set size/ {
    gsub(/^[ \t]+|[ \t]+$/, "", $2)
    value=$2
  }
  END {
    if (value != "") print value
  }' "$time_log"
}

restore_cpu_settings() {
  [[ "$LIMIT_CPU_CLOCK" -eq 1 ]] || return 0

  command -v cpupower >/dev/null 2>&1 || return 0

  log "Restoring original CPU settings..."

  if [[ -n "$OLD_GOVERNOR" ]]; then
    log "Restoring governor to: $OLD_GOVERNOR"
    sudo -n cpupower frequency-set -g "$OLD_GOVERNOR" >/dev/null 2>&1 || true
  fi

  if [[ -n "$OLD_MAX_FREQ" ]]; then
    log "Restoring max CPU frequency to: $OLD_MAX_FREQ"
    sudo -n cpupower frequency-set -u "$OLD_MAX_FREQ" >/dev/null 2>&1 || true
  fi

  log "CPU settings restoration finished."
}

cleanup() {
  local exit_code=$?
  log "Cleanup started (exit code = $exit_code)"

  if [[ -n "$SUDO_KEEPALIVE_PID" ]]; then
    log "Stopping sudo keepalive (PID=$SUDO_KEEPALIVE_PID)"
    kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
  fi

  restore_cpu_settings

  if [[ -n "$PARALLEL_WRAPPER_DIR" && -d "$PARALLEL_WRAPPER_DIR" ]]; then
    rm -rf "$PARALLEL_WRAPPER_DIR" || true
  fi

  local script_end_epoch
  script_end_epoch="$(date +%s)"
  local total_elapsed=$(( script_end_epoch - SCRIPT_START_EPOCH ))

  log "run_all.sh finished. Total elapsed: $(elapsed_hms "$total_elapsed")"
}

start_sudo_keepalive() {
  log "Validating sudo permissions..."
  sudo -v
  log "sudo validated successfully."

  (
    while true; do
      sudo -n true
      sleep 60
      kill -0 "$$" >/dev/null 2>&1 || exit
    done
  ) >/dev/null 2>&1 &
  SUDO_KEEPALIVE_PID=$!

  log "sudo keepalive started (PID=$SUDO_KEEPALIVE_PID)"
}

apply_cpu_limit() {
  [[ "$LIMIT_CPU_CLOCK" -eq 1 ]] || {
    log "CPU clock limit disabled."
    return 0
  }

  if ! command -v cpupower >/dev/null 2>&1; then
    log "[WARN] cpupower not found. Running without CPU clock limit."
    return 0
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    log "[WARN] sudo not found. Running without CPU clock limit."
    return 0
  fi

  start_sudo_keepalive

  if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    OLD_GOVERNOR="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true)"
    [[ -n "$OLD_GOVERNOR" ]] && log "Detected current governor: $OLD_GOVERNOR"
  fi

  if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq ]]; then
    OLD_MAX_FREQ="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || true)"
    if [[ -n "$OLD_MAX_FREQ" ]]; then
      OLD_MAX_FREQ="${OLD_MAX_FREQ}KHz"
      log "Detected current max CPU frequency: $OLD_MAX_FREQ"
    fi
  fi

  log "Applying CPU max frequency limit to: $CPU_MAX_FREQ"
  log "Applying CPU governor: $CPU_GOVERNOR"

  sudo -n cpupower frequency-set -g "$CPU_GOVERNOR" >/dev/null 2>&1 || true
  sudo -n cpupower frequency-set -u "$CPU_MAX_FREQ" >/dev/null 2>&1 || {
    log "[WARN] Could not apply CPU clock limit."
    return 0
  }

  log "CPU clock limit applied successfully."
}

setup_parallel_wrapper() {
  REAL_PARALLEL="$(command -v parallel 2>/dev/null || true)"
  [[ -n "$REAL_PARALLEL" ]] || {
    log "GNU parallel not found in PATH. Child scripts must use RUN_ALL_JOBS/NUM_JOBS themselves."
    return 0
  }

  PARALLEL_WRAPPER_DIR="$(mktemp -d)"

  cat > "$PARALLEL_WRAPPER_DIR/parallel" <<'WRAPPER_EOF'
#!/usr/bin/env bash
set -euo pipefail

real_parallel="${RUN_ALL_REAL_PARALLEL:?RUN_ALL_REAL_PARALLEL is not set}"
jobs="${RUN_ALL_JOBS:-1}"

args=()
skip_next=0

for arg in "$@"; do
  if [[ "$skip_next" -eq 1 ]]; then
    skip_next=0
    continue
  fi

  case "$arg" in
    -j|--jobs)
      skip_next=1
      continue
      ;;
    -j*|--jobs=*)
      continue
      ;;
    --halt)
      if [[ "${RUN_ALL_PROBE:-0}" == "1" && "${RUN_ALL_PROBE_ONE_PARALLEL_JOB:-1}" == "1" ]]; then
        skip_next=1
        continue
      fi
      args+=("$arg")
      ;;
    --halt=*)
      if [[ "${RUN_ALL_PROBE:-0}" == "1" && "${RUN_ALL_PROBE_ONE_PARALLEL_JOB:-1}" == "1" ]]; then
        continue
      fi
      args+=("$arg")
      ;;
    *)
      args+=("$arg")
      ;;
  esac
done

if [[ "${RUN_ALL_PROBE:-0}" == "1" && "${RUN_ALL_PROBE_ONE_PARALLEL_JOB:-1}" == "1" ]]; then
  exec "$real_parallel" --jobs 1 --halt soon,success=1 "${args[@]}"
else
  exec "$real_parallel" --jobs "$jobs" "${args[@]}"
fi
WRAPPER_EOF

  chmod +x "$PARALLEL_WRAPPER_DIR/parallel"

  export RUN_ALL_REAL_PARALLEL="$REAL_PARALLEL"
  export PATH="$PARALLEL_WRAPPER_DIR:$PATH"

  log "GNU parallel wrapper enabled. Original parallel: $REAL_PARALLEL"
}

is_excluded_script() {
  local name="$1"
  for excluded in "${EXCLUDED_SCRIPTS[@]}"; do
    [[ "$name" == "$excluded" ]] && return 0
  done
  return 1
}

run_script_measured() {
  local script="$1"
  local jobs="$2"
  local mode="$3"
  local out_log="$4"
  local time_log="$5"

  local probe=0
  [[ "$mode" == "probe" ]] && probe=1

  local -a cmd=(
    env
    RUN_ALL_JOBS="$jobs"
    NUM_JOBS="$jobs"
    N_JOBS="$jobs"
    JOBS="$jobs"
    NUM_THREADS="$jobs"
    THREADS="$jobs"
    PARALLEL_JOBS="$jobs"
    OMP_NUM_THREADS="1"
    OPENBLAS_NUM_THREADS="1"
    MKL_NUM_THREADS="1"
    RUN_ALL_PROBE="$probe"
    RUN_ALL_PROBE_ONE_PARALLEL_JOB="$RUN_ALL_PROBE_ONE_PARALLEL_JOB"
    "$script"
  )

  mkdir -p "$(dirname "$out_log")"

  set +e
  if [[ "$mode" == "probe" && "$RAM_PROBE_TIMEOUT" != "0" ]]; then
    /usr/bin/time -v timeout --preserve-status "$RAM_PROBE_TIMEOUT" "${cmd[@]}" >"$out_log" 2>"$time_log"
  elif [[ "$mode" == "run" ]]; then
    /usr/bin/time -v "${cmd[@]}" > >(tee "$out_log") 2> >(tee "$time_log" >&2)
  else
    /usr/bin/time -v "${cmd[@]}" >"$out_log" 2>"$time_log"
  fi
  local status=$?
  set -e

  if [[ "$mode" != "run" && -s "$out_log" ]]; then
    cat "$out_log"
  fi

  if [[ "$mode" != "run" && -s "$time_log" ]]; then
    cat "$time_log" >&2
  fi

  return "$status"
}

run_with_auto_ram_jobs() {
  local f="$1"
  local base="$2"

  local safe_base
  safe_base="$(safe_name "$base")"

  local probe_out="$RAM_LOG_DIR/${safe_base}.probe.out.log"
  local probe_time="$RAM_LOG_DIR/${safe_base}.probe.time.log"
  local run_out="$RAM_LOG_DIR/${safe_base}.run.out.log"
  local run_time="$RAM_LOG_DIR/${safe_base}.run.time.log"

  log "RAM auto-tuning probe: $f"
  log "Probe will run with 1 job. Logs: $probe_out / $probe_time"

  local probe_start_epoch
  probe_start_epoch="$(date +%s)"

  if ! run_script_measured "$f" 1 "probe" "$probe_out" "$probe_time"; then
    local probe_end_epoch
    probe_end_epoch="$(date +%s)"
    log "[ERROR] Probe failed for $f"
    log "Elapsed before probe failure for $base: $(elapsed_hms "$(( probe_end_epoch - probe_start_epoch ))")"
    return 1
  fi

  local probe_end_epoch
  probe_end_epoch="$(date +%s)"
  local probe_elapsed=$(( probe_end_epoch - probe_start_epoch ))

  local peak_kb
  peak_kb="$(extract_peak_rss_kb "$probe_time" || true)"

  if [[ -z "$peak_kb" || ! "$peak_kb" =~ ^[0-9]+$ || "$peak_kb" -le 0 ]]; then
    log "[WARN] Could not read peak RAM from probe. Running $f with 1 job."
    peak_kb=0
  fi

  local basis_kb
  basis_kb="$(mem_kb)"

  local jobs=1
  if [[ "$peak_kb" -gt 0 ]]; then
    jobs="$(calculate_jobs_from_ram "$peak_kb" "$basis_kb")"
  fi

  log "Probe finished: $base"
  log "Probe elapsed: $(elapsed_hms "$probe_elapsed")"
  log "Measured peak RAM per job: $(kb_to_gb "$peak_kb") GB"
  log "RAM basis ($RAM_BASIS): $(kb_to_gb "$basis_kb") GB"
  log "RAM_USE_FRACTION=$RAM_USE_FRACTION | RAM_RESERVE_GB=$RAM_RESERVE_GB | RAM_PEAK_MULTIPLIER=$RAM_PEAK_MULTIPLIER"
  log "Selected jobs for this parameter set: $jobs"

  log "Starting full run: $f with $jobs job(s)"
  local run_start_epoch
  run_start_epoch="$(date +%s)"

  if run_script_measured "$f" "$jobs" "run" "$run_out" "$run_time"; then
    local run_end_epoch
    run_end_epoch="$(date +%s)"
    log "Finished successfully: $f"
    log "Elapsed for $base full run: $(elapsed_hms "$(( run_end_epoch - run_start_epoch ))")"
  else
    local run_end_epoch
    run_end_epoch="$(date +%s)"
    log "[ERROR] Script failed: $f"
    log "Elapsed before failure for $base full run: $(elapsed_hms "$(( run_end_epoch - run_start_epoch ))")"
    return 1
  fi
}

run_without_auto_ram_jobs() {
  local f="$1"
  local base="$2"

  log "Starting script without RAM auto-tuning: $f"
  local script_start_epoch
  script_start_epoch="$(date +%s)"

  if "$f"; then
    local script_end_epoch
    script_end_epoch="$(date +%s)"
    log "Finished successfully: $f"
    log "Elapsed for $base: $(elapsed_hms "$(( script_end_epoch - script_start_epoch ))")"
  else
    local script_end_epoch
    script_end_epoch="$(date +%s)"
    log "[ERROR] Script failed: $f"
    log "Elapsed before failure for $base: $(elapsed_hms "$(( script_end_epoch - script_start_epoch ))")"
    return 1
  fi
}

trap cleanup EXIT

log "run_all.sh started"
log "Working directory: $(pwd)"
log "Script path: $self"

apply_cpu_limit

if [[ "$AUTO_RAM_JOBS" -eq 1 ]]; then
  if [[ ! -x /usr/bin/time ]]; then
    log "[WARN] /usr/bin/time not found. Disabling AUTO_RAM_JOBS."
    AUTO_RAM_JOBS=0
  else
    mkdir -p "$RAM_LOG_DIR"
    setup_parallel_wrapper
    log "RAM auto-tuning enabled."
  fi
else
  log "RAM auto-tuning disabled."
fi

log "Scanning .sh files in current directory..."
mapfile -t scripts < <(printf '%s\n' ./*.sh | sort -V)
log "Found ${#scripts[@]} shell script(s)."

ran_any=0
failed_any=0

for f in "${scripts[@]}"; do
  base="$(basename "$f")"

  if [[ "$(realpath "$f")" == "$self" ]]; then
    log "Skipping self: $base"
    continue
  fi

  if is_excluded_script "$base"; then
    log "Skipping excluded script: $base"
    continue
  fi

  ran_any=1
  log "------------------------------------------------------------"

  if [[ "$AUTO_RAM_JOBS" -eq 1 ]]; then
    if ! run_with_auto_ram_jobs "$f" "$base"; then
      failed_any=1
    fi
  else
    if ! run_without_auto_ram_jobs "$f" "$base"; then
      failed_any=1
    fi
  fi
done

if [[ "$ran_any" -eq 0 ]]; then
  log "No runnable scripts found."
fi

if [[ "$failed_any" -ne 0 ]]; then
  log "[ERROR] One or more scripts failed."
  exit 1
fi

log "All scripts were processed."
