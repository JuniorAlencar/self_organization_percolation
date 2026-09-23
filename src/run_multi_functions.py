import os
import json
import shutil
import stat
import textwrap
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Iterable
from collections import defaultdict


# =========================
# Utils
# =========================

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


def create_folder(folder_path: Path):
    folder_path.mkdir(parents=True, exist_ok=True)


# =========================
# Project paths
# =========================


def _module_project_root() -> Path:
    module_dir = Path(__file__).resolve().parent
    if module_dir.name == "src":
        return module_dir.parent
    return module_dir


PROJECT_ROOT = _module_project_root()
CLUSTER_ROOT = Path("/home/junioralencar/codes/SOP")

# Mantém a configuração absoluta usada no cluster, mas permite importar/testar
# o módulo em outro checkout sem tentar escrever fora do workspace.
ACTIVE_ROOT = CLUSTER_ROOT if CLUSTER_ROOT.exists() else PROJECT_ROOT
SHELLS_DIR = ACTIVE_ROOT / "shells"
BUILD_DIR = ACTIVE_ROOT / "build"

SHELLS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Existing sample detection
# =========================

# O executavel C++ usa FolderCreator("./SOP_data"). Por isso, o script SLURM
# abaixo faz cd para ACTIVE_ROOT antes de chamar o binario, e o Python tambem
# procura as saidas nesse mesmo lugar. SOP_DATA_ROOT permite sobrescrever isso
# explicitamente quando necessario.
DATA_ROOT = Path(os.environ.get("SOP_DATA_ROOT", ACTIVE_ROOT / "SOP_data")).expanduser().resolve()
RUN_HISTORY_PATH = DATA_ROOT / "run_history" / "executed_samples.jsonl"
GROWTH_TEST_DYNAMICS_WINDOW_STEPS = 300


def _unique_paths(paths) -> list[Path]:
    unique = []
    seen = set()
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


# Leituras consideram tambem raizes antigas que podem ter sido criadas quando
# DATA_ROOT dependia do cwd. Escritas novas vao sempre para DATA_ROOT.
READ_DATA_ROOTS = _unique_paths([
    DATA_ROOT,
    ACTIVE_ROOT / "SOP_data",
    PROJECT_ROOT / "SOP_data",
    Path.cwd() / "SOP_data",
])


def _raw_folder_for_mode(run_mode: str) -> str:
    if run_mode == "growth_test":
        return "raw_growth_test_dynamic"
    return "raw"


def _sample_data_dir(
    L,
    type_perc,
    c,
    f_T,
    dim,
    num_colors,
    rho,
    run_mode,
    data_root=DATA_ROOT,
) -> Path:
    path = (
        Path(data_root)
        / _raw_folder_for_mode(run_mode)
        / f"{type_perc}_percolation"
        / f"num_colors_{int(num_colors)}"
        / f"dim_{int(dim)}"
        / f"L_{int(L)}"
        / "fT_constant"
        / f"fT_{float(f_T):.6e}"
        / f"c_{float(c):.6e}"
        / f"rho_{float(rho):.4e}"
    )

    if run_mode == "growth_test":
        path = path / f"stationary_window_{GROWTH_TEST_DYNAMICS_WINDOW_STEPS}"

    return path / "data"


def _sample_data_dirs(
    L,
    type_perc,
    c,
    f_T,
    dim,
    num_colors,
    rho,
    run_mode,
) -> list[Path]:
    data_dirs = [
        _sample_data_dir(
            L=L,
            type_perc=type_perc,
            c=c,
            f_T=f_T,
            dim=dim,
            num_colors=num_colors,
            rho=rho,
            run_mode=run_mode,
            data_root=data_root,
        )
        for data_root in READ_DATA_ROOTS
    ]
    if run_mode == "growth_test":
        legacy_dirs = [
            data_dir.parent.parent / "data"
            for data_dir in data_dirs
            if data_dir.parent.name.startswith("stationary_window_")
        ]
        data_dirs.extend(legacy_dirs)
    return _unique_paths(data_dirs)


def _count_existing_samples(data_dir: Path, P0, p0) -> int:
    if not data_dir.is_dir():
        return 0

    token = f"_P0_{float(P0):.2f}_p0_{float(p0):.2f}"
    return sum(
        1
        for path in data_dir.glob("*.json")
        if path.is_file() and token in path.stem
    )


def _parameter_key(
    *,
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
) -> str:
    payload = {
        "L": int(L),
        "p0": f"{float(p0):.2f}",
        "type_perc": str(type_perc),
        "c": f"{float(c):.6e}",
        "f_T": f"{float(f_T):.6e}",
        "dim": int(dim),
        "num_colors": int(num_colors),
        "rho": f"{float(rho):.4e}",
        "P0": f"{float(P0):.2f}",
        "run_mode": str(run_mode),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _iter_history_records():
    history_paths = _unique_paths(
        [RUN_HISTORY_PATH, *(root / "run_history" / "executed_samples.jsonl" for root in READ_DATA_ROOTS)]
    )
    for history_path in history_paths:
        if not history_path.is_file():
            continue

        with history_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _history_state(parameter_key: str) -> tuple[set[str], int]:
    sample_ids = set()
    known_count = 0
    for record in _iter_history_records() or ():
        if record.get("parameter_key") == parameter_key:
            sample_id = record.get("sample_id")
            if sample_id:
                sample_ids.add(str(sample_id))
            known_count = max(known_count, int(record.get("known_count_after", 0) or 0))
    return sample_ids, max(known_count, len(sample_ids))


def _history_sample_ids(parameter_key: str) -> set[str]:
    sample_ids, _ = _history_state(parameter_key)
    return sample_ids


def _history_known_count(parameter_key: str) -> int:
    _, known_count = _history_state(parameter_key)
    return known_count


def _sample_json_paths(data_dir: Path, P0, p0) -> list[Path]:
    if not data_dir.is_dir():
        return []

    token = f"_P0_{float(P0):.2f}_p0_{float(p0):.2f}"
    return sorted(
        path
        for path in data_dir.glob("*.json")
        if path.is_file() and token in path.stem
    )


def _sample_id_from_json_path(path: Path) -> str:
    return path.stem


def _rho_values(rho):
    _, rho_data = _normalize_rho_input(rho)
    return rho_data if _is_sequence_but_not_string(rho_data) else [rho_data]


def _remove_empty_parents(path: Path, stop_at: Path):
    current = path.parent
    stop_at = stop_at.resolve()
    while current.exists() and current.resolve() != stop_at:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _count_known_samples(data_dir: Path, P0, p0, parameter_key: str) -> int:
    sample_ids, history_count = _history_state(parameter_key)
    data_dirs = _unique_paths([data_dir, *_sample_data_dirs_from_primary(data_dir)])
    for candidate_dir in data_dirs:
        sample_ids.update(
            _sample_id_from_json_path(path)
            for path in _sample_json_paths(candidate_dir, P0, p0)
        )
    return max(history_count, len(sample_ids))


def _sample_data_dirs_from_primary(primary_data_dir: Path) -> list[Path]:
    try:
        relative_data_dir = primary_data_dir.resolve().relative_to(DATA_ROOT)
    except ValueError:
        return []
    return [root / relative_data_dir for root in READ_DATA_ROOTS]


def snapshot_existing_outputs_for_parameters(
    *,
    L,
    p0,
    seed,
    type_perc,
    c,
    f_T,
    dim,
    num_colors,
    rho,
    N_samples,
    P0,
    Equilibration=None,
    equilibration=True,
    properties=False,
    run_mode="sop",
    max_concurrent=3,
    shell_name="run_jobs_array.sh",
) -> list[dict]:
    """Lê o estado atual do disco/historico sem escrever nada."""
    del seed, N_samples, Equilibration, equilibration, properties, max_concurrent, shell_name

    run_mode_arg = _normalize_run_mode(run_mode)
    snapshots = []
    for rho_value in _rho_values(rho):
        data_dir = _sample_data_dir(
            L=L,
            type_perc=type_perc,
            c=c,
            f_T=f_T,
            dim=dim,
            num_colors=num_colors,
            rho=rho_value,
            run_mode=run_mode_arg,
        )
        data_dirs = _sample_data_dirs(
            L=L,
            type_perc=type_perc,
            c=c,
            f_T=f_T,
            dim=dim,
            num_colors=num_colors,
            rho=rho_value,
            run_mode=run_mode_arg,
        )
        parameter_key = _parameter_key(
            L=L,
            p0=p0,
            type_perc=type_perc,
            c=c,
            f_T=f_T,
            dim=dim,
            num_colors=num_colors,
            rho=rho_value,
            P0=P0,
            run_mode=run_mode_arg,
        )
        disk_sample_ids = {
            _sample_id_from_json_path(path)
            for candidate_dir in data_dirs
            for path in _sample_json_paths(candidate_dir, P0=P0, p0=p0)
        }
        disk_count = len(disk_sample_ids)
        known_count = max(_history_known_count(parameter_key), disk_count)
        snapshots.append({
            "parameter_key": parameter_key,
            "known_count_after": known_count,
            "disk_count": disk_count,
            "data_dir": str(data_dir),
            "output_dir": str(data_dir.parent),
            "params": {
                "L": L,
                "p0": p0,
                "type_perc": type_perc,
                "c": c,
                "f_T": float(f_T),
                "dim": dim,
                "num_colors": num_colors,
                "rho": rho_value,
                "P0": P0,
                "run_mode": run_mode_arg,
            },
        })
    return snapshots


def write_existing_output_snapshots(snapshots: list[dict]) -> dict:
    """Grava um historico compacto depois que a limpeza liberou espaço."""
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with RUN_HISTORY_PATH.open("a", encoding="utf-8") as fh:
        for snapshot in snapshots:
            if int(snapshot.get("known_count_after", 0) or 0) <= 0:
                continue
            record = {
                "event": "parameter_count",
                "logged_at": datetime.now(timezone.utc).isoformat(),
                **snapshot,
            }
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            written += 1

    return {
        "history_path": str(RUN_HISTORY_PATH),
        "snapshots_written": written,
    }


def _append_history_record(record: dict):
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    fd = os.open(RUN_HISTORY_PATH, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def record_completed_parameter_set(
    *,
    L,
    p0,
    type_perc,
    c,
    f_T,
    dim,
    num_colors,
    rho,
    P0,
    run_mode="sop",
    target_samples=None,
) -> dict:
    """Registra que um conjunto de parametros terminou com sucesso no SLURM."""
    if target_samples is None:
        raise ValueError("target_samples e obrigatorio para registrar conclusao.")
    run_mode_arg = _normalize_run_mode(run_mode)
    parameter_key = _parameter_key(
        L=L,
        p0=p0,
        type_perc=type_perc,
        c=c,
        f_T=f_T,
        dim=dim,
        num_colors=num_colors,
        rho=rho,
        P0=P0,
        run_mode=run_mode_arg,
    )
    data_dir = _sample_data_dir(
        L=L,
        type_perc=type_perc,
        c=c,
        f_T=f_T,
        dim=dim,
        num_colors=num_colors,
        rho=rho,
        run_mode=run_mode_arg,
    )
    disk_count = sum(
        len(_sample_json_paths(candidate_dir, P0=P0, p0=p0))
        for candidate_dir in _sample_data_dirs(
            L=L,
            type_perc=type_perc,
            c=c,
            f_T=f_T,
            dim=dim,
            num_colors=num_colors,
            rho=rho,
            run_mode=run_mode_arg,
        )
    )
    known_count_after = max(_history_known_count(parameter_key), int(target_samples), disk_count)
    record = {
        "event": "parameter_completed",
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "parameter_key": parameter_key,
        "known_count_after": known_count_after,
        "target_samples": int(target_samples),
        "disk_count": disk_count,
        "data_dir": str(data_dir),
        "output_dir": str(data_dir.parent),
        "params": {
            "L": int(L),
            "p0": float(p0),
            "type_perc": str(type_perc),
            "c": float(c),
            "f_T": float(f_T),
            "dim": int(dim),
            "num_colors": int(num_colors),
            "rho": float(rho),
            "P0": float(P0),
            "run_mode": run_mode_arg,
        },
    }
    _append_history_record(record)
    return {
        "history_path": str(RUN_HISTORY_PATH),
        "parameter_key": parameter_key,
        "known_count_after": known_count_after,
    }


def log_existing_samples_for_parameters(
    *,
    L,
    p0,
    seed,
    type_perc,
    c,
    f_T,
    dim,
    num_colors,
    rho,
    N_samples,
    P0,
    Equilibration=None,
    equilibration=True,
    properties=False,
    run_mode="sop",
    max_concurrent=3,
    shell_name="run_jobs_array.sh",
) -> dict:
    """Registra no historico os samples existentes no disco para estes parametros."""
    del seed, N_samples, Equilibration, equilibration, properties, max_concurrent, shell_name

    run_mode_arg = _normalize_run_mode(run_mode)
    _, rho_data = _normalize_rho_input(rho)
    rho_values = rho_data if _is_sequence_but_not_string(rho_data) else [rho_data]

    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    seen_on_disk = 0

    with RUN_HISTORY_PATH.open("a", encoding="utf-8") as fh:
        for rho_value in rho_values:
            data_dir = _sample_data_dir(
                L=L,
                type_perc=type_perc,
                c=c,
                f_T=f_T,
                dim=dim,
                num_colors=num_colors,
                rho=rho_value,
                run_mode=run_mode_arg,
            )
            data_dirs = _sample_data_dirs(
                L=L,
                type_perc=type_perc,
                c=c,
                f_T=f_T,
                dim=dim,
                num_colors=num_colors,
                rho=rho_value,
                run_mode=run_mode_arg,
            )
            parameter_key = _parameter_key(
                L=L,
                p0=p0,
                type_perc=type_perc,
                c=c,
                f_T=f_T,
                dim=dim,
                num_colors=num_colors,
                rho=rho_value,
                P0=P0,
                run_mode=run_mode_arg,
            )
            history_ids, history_count = _history_state(parameter_key)
            new_sample_records = 0

            for candidate_dir in data_dirs:
                for sample_path in _sample_json_paths(candidate_dir, P0=P0, p0=p0):
                    seen_on_disk += 1
                    sample_id = _sample_id_from_json_path(sample_path)
                    if sample_id in history_ids:
                        continue

                    new_sample_records += 1
                    record = {
                        "event": "sample_seen",
                        "logged_at": datetime.now(timezone.utc).isoformat(),
                        "parameter_key": parameter_key,
                        "sample_id": sample_id,
                        "sample_file": sample_path.name,
                        "data_dir": str(candidate_dir),
                        "params": {
                            "L": L,
                            "p0": p0,
                            "type_perc": type_perc,
                            "c": c,
                            "f_T": float(f_T),
                            "dim": dim,
                            "num_colors": num_colors,
                            "rho": rho_value,
                            "P0": P0,
                            "run_mode": run_mode_arg,
                        },
                    }
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
                    history_ids.add(sample_id)
                    added += 1

            known_count_after = max(history_count, len(history_ids))
            if new_sample_records:
                summary = {
                    "event": "parameter_count",
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                    "parameter_key": parameter_key,
                    "known_count_after": known_count_after,
                    "new_samples_logged": new_sample_records,
                    "data_dir": str(data_dir),
                    "params": {
                        "L": L,
                        "p0": p0,
                        "type_perc": type_perc,
                        "c": c,
                        "f_T": float(f_T),
                        "dim": dim,
                        "num_colors": num_colors,
                        "rho": rho_value,
                        "P0": P0,
                        "run_mode": run_mode_arg,
                    },
                }
                fh.write(json.dumps(summary, sort_keys=True) + "\n")

    return {
        "history_path": str(RUN_HISTORY_PATH),
        "seen_on_disk": seen_on_disk,
        "added_to_history": added,
    }


def cleanup_logged_output_for_parameters(
    *,
    L,
    p0,
    seed,
    type_perc,
    c,
    f_T,
    dim,
    num_colors,
    rho,
    N_samples,
    P0,
    Equilibration=None,
    equilibration=True,
    properties=False,
    run_mode="sop",
    max_concurrent=3,
    shell_name="run_jobs_array.sh",
) -> list[str]:
    """Apaga as pastas de saida destes parametros depois que os samples foram logados."""
    del seed, N_samples, Equilibration, equilibration, properties, max_concurrent, shell_name

    run_mode_arg = _normalize_run_mode(run_mode)
    _, rho_data = _normalize_rho_input(rho)
    rho_values = rho_data if _is_sequence_but_not_string(rho_data) else [rho_data]
    deleted = []

    data_roots = [root.resolve() for root in READ_DATA_ROOTS]
    for rho_value in rho_values:
        data_dirs = _sample_data_dirs(
            L=L,
            type_perc=type_perc,
            c=c,
            f_T=f_T,
            dim=dim,
            num_colors=num_colors,
            rho=rho_value,
            run_mode=run_mode_arg,
        )
        for data_dir in data_dirs:
            output_dir = data_dir.parent
            if not output_dir.exists():
                print(f"🧹 Pasta já ausente: {output_dir}")
                continue

            resolved_output = output_dir.resolve()
            matched_root = next(
                (root for root in data_roots if root == resolved_output or root in resolved_output.parents),
                None,
            )
            if matched_root is None:
                raise RuntimeError(f"Recusando apagar pasta fora das raizes de dados: {output_dir}")

            shutil.rmtree(output_dir)
            _remove_empty_parents(output_dir, matched_root)
            print(f"🧹 Pasta antiga apagada antes da submissão: {output_dir}")
            deleted.append(str(output_dir))

    return deleted


def _remaining_samples_for_rho(
    *,
    L,
    p0,
    type_perc,
    c,
    f_T,
    dim,
    num_colors,
    rho,
    N_samples,
    P0,
    run_mode,
) -> tuple[int, int, Path]:
    data_dir = _sample_data_dir(
        L=L,
        type_perc=type_perc,
        c=c,
        f_T=f_T,
        dim=dim,
        num_colors=num_colors,
        rho=rho,
        run_mode=run_mode,
    )
    parameter_key = _parameter_key(
        L=L,
        p0=p0,
        type_perc=type_perc,
        c=c,
        f_T=f_T,
        dim=dim,
        num_colors=num_colors,
        rho=rho,
        P0=P0,
        run_mode=run_mode,
    )
    existing = _count_known_samples(data_dir, P0=P0, p0=p0, parameter_key=parameter_key)
    remaining = max(0, int(N_samples) - existing)
    return existing, remaining, data_dir


# =========================
# Shell script generator
# =========================

def create_cluster_cli_shell(exec_name: str = "run_jobs_array.sh"):
    shell_path = SHELLS_DIR / exec_name

    script = f"""\
#!/usr/bin/env bash
#SBATCH -J SOP_cli
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G

# Modos:
# 1) fixed:
#    sbatch run_jobs_array.sh fixed L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES rho P0 EQUILIBRATION [PROPERTIES] [MODE]
#
# 2) array:
#    sbatch --array=0-(N_RHO-1)%MAX_CONCURRENT run_jobs_array.sh array L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES P0 EQUILIBRATION RHO_FILE [PROPERTIES] [MODE]
#
# Chamada do executável SOP:
#    ./SOP L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS rho P0 EQUILIBRATION [PROPERTIES] [MODE]

if [[ "$#" -lt 1 ]]; then
  echo "Uso:"
  echo "  fixed: $0 fixed L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES rho P0 EQUILIBRATION [PROPERTIES] [MODE]"
  echo "  array: $0 array L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES P0 EQUILIBRATION RHO_FILE [PROPERTIES] [MODE]"
  exit 1
fi

MODE="$1"

if [[ "$MODE" == "fixed" ]]; then
  if [[ "$#" -lt 14 || "$#" -gt 16 ]]; then
    echo "Uso: $0 fixed L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES rho P0 EQUILIBRATION [PROPERTIES] [MODE]"
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
  NUM_SAMPLES="${{10}}"
  TARGET_SAMPLES="${{11}}"
  RHO="${{12}}"
  P0="${{13}}"
  EQUILIBRATION="${{14}}"
  PROPERTIES="${{15:-}}"
  RUN_MODE="${{16:-}}"

elif [[ "$MODE" == "array" ]]; then
  if [[ "$#" -lt 14 || "$#" -gt 16 ]]; then
    echo "Uso: $0 array L p0 SEED TYPE_PERC C F_T DIM NUM_COLORS NUM_SAMPLES TARGET_SAMPLES P0 EQUILIBRATION RHO_FILE [PROPERTIES] [MODE]"
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
  NUM_SAMPLES="${{10}}"
  TARGET_SAMPLES="${{11}}"
  P0="${{12}}"
  EQUILIBRATION="${{13}}"
  RHO_FILE="${{14}}"
  PROPERTIES="${{15:-}}"
  RUN_MODE="${{16:-}}"

  if [[ -z "${{SLURM_ARRAY_TASK_ID:-}}" ]]; then
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

if [[ -n "${{PROPERTIES:-}}" && -z "${{RUN_MODE:-}}" ]]; then
  case "$PROPERTIES" in
    sop|growth_test)
      RUN_MODE="$PROPERTIES"
      PROPERTIES="false"
      ;;
  esac
fi

EXTRA_ARGS=()
if [[ -n "${{RUN_MODE:-}}" ]]; then
  EXTRA_ARGS=("${{PROPERTIES:-false}}" "$RUN_MODE")
elif [[ -n "${{PROPERTIES:-}}" ]]; then
  EXTRA_ARGS=("$PROPERTIES")
fi

EXEC="{BUILD_DIR}/SOP"
WORKDIR="{ACTIVE_ROOT}"

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
echo "EXTRA_ARGS=${{EXTRA_ARGS[*]:-}}"
echo "EXEC=$EXEC"
echo "WORKDIR=$WORKDIR"
echo "==============="

cd "$WORKDIR"

i=1
while [[ "$i" -le "$NUM_SAMPLES" ]]; do
  srun "$EXEC" "$L" "$p0" "$SEED" "$TYPE_PERC" "$C" "$F_T" "$DIM" "$NUM_COLORS" "$RHO" "$P0" "$EQUILIBRATION" "${{EXTRA_ARGS[@]}}"
  i=$((i + 1))
done

PYTHON_BIN="${{PYTHON_BIN:-python3}}"
export PYTHONPATH="$WORKDIR/python/src:$WORKDIR:${{PYTHONPATH:-}}"
"$PYTHON_BIN" - "$L" "$p0" "$TYPE_PERC" "$C" "$F_T" "$DIM" "$NUM_COLORS" "$RHO" "$P0" "${{RUN_MODE:-sop}}" "$TARGET_SAMPLES" <<'PY'
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
    P0=float(P0),
    run_mode=run_mode or "sop",
    target_samples=int(target_samples),
)
print(f"Historico atualizado: {{info['history_path']}} known_count_after={{info['known_count_after']}}")
PY

echo "Done."
"""

    script_text = textwrap.dedent(script)
    if shell_path.exists():
        try:
            if shell_path.read_text() == script_text:
                shell_path.chmod(shell_path.stat().st_mode | stat.S_IEXEC)
                return
        except OSError:
            pass

    shell_path.write_text(script_text)
    shell_path.chmod(shell_path.stat().st_mode | stat.S_IEXEC)
    print(f"✅ Shell script criado em {shell_path}")


def _is_sequence_but_not_string(x):
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))


def _normalize_rho_input(rho):
    if _is_sequence_but_not_string(rho):
        rho_lst = list(rho)
        if len(rho_lst) == 0:
            raise ValueError("rho lista vazia.")
        return "array", rho_lst
    return "fixed", float(rho)


def _float_tag(value, ndigits: int = 6) -> str:
    """Tag estável para nomes de arquivos auxiliares do SLURM."""
    try:
        return f"{float(value):.{ndigits}e}"
    except Exception:
        return str(value)


def _normalize_equilibration_input(value) -> str:
    """Normaliza o parâmetro de equilíbrio para o formato aceito pelo executável C++.

    Aceita bool, 0/1 ou strings como "true", "false", "yes", "no".
    Retorna sempre "true" ou "false".
    """
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"

    if isinstance(value, (int, np.integer)):
        return "true" if int(value) != 0 else "false"

    if isinstance(value, (float, np.floating)):
        if float(value) in (0.0, 1.0):
            return "true" if float(value) == 1.0 else "false"
        raise ValueError(f"equilibration numérico deve ser 0 ou 1; recebido: {value}")

    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y", "sim", "s", "on"}:
        return "true"
    if text in {"false", "f", "0", "no", "n", "nao", "não", "off"}:
        return "false"

    raise ValueError(
        "equilibration deve ser bool, 0/1, ou string true/false. "
        f"Valor recebido: {value!r}"
    )


def _normalize_bool_input(value, name: str) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y", "sim", "s", "on"}:
        return "true"
    if text in {"false", "f", "0", "no", "n", "nao", "não", "off"}:
        return "false"
    raise ValueError(f"{name} deve ser true/false; recebido: {value!r}")


def _normalize_run_mode(value) -> str:
    mode = str(value).strip()
    if mode not in {"sop", "growth_test"}:
        raise ValueError("run_mode deve ser 'sop' ou 'growth_test'")
    return mode


def get_missing_run_parameters(
    *,
    L,
    p0,
    seed,
    type_perc,
    c,
    f_T,
    dim,
    num_colors,
    rho,
    N_samples,
    P0,
    Equilibration=None,
    equilibration=True,
    properties=False,
    run_mode="sop",
    max_concurrent=3,
    shell_name="run_jobs_array.sh",
):
    """Retorna os parametros que ainda faltam rodar, sem submeter jobs.

    O retorno e uma lista de dicionarios. Cada item tem os mesmos parametros
    necessarios para chamar run_multi_rho_array depois, mas com N_samples igual
    ao numero de amostras restantes.
    """
    if Equilibration is not None:
        equilibration = Equilibration

    run_mode_arg = _normalize_run_mode(run_mode)
    properties_arg = _normalize_bool_input(properties, "properties")
    if run_mode_arg == "growth_test" and properties_arg == "true":
        raise ValueError("growth_test deve ser executado com properties=False")

    _, rho_data = _normalize_rho_input(rho)
    rho_values = rho_data if _is_sequence_but_not_string(rho_data) else [rho_data]

    missing = []
    for rho_value in rho_values:
        existing, remaining, data_dir = _remaining_samples_for_rho(
            L=L,
            p0=p0,
            type_perc=type_perc,
            c=c,
            f_T=f_T,
            dim=dim,
            num_colors=num_colors,
            rho=rho_value,
            N_samples=N_samples,
            P0=P0,
            run_mode=run_mode_arg,
        )
        if remaining <= 0:
            continue

        missing.append({
            "L": L,
            "p0": p0,
            "seed": seed,
            "type_perc": type_perc,
            "c": c,
            "f_T": f_T,
            "dim": dim,
            "num_colors": num_colors,
            "rho": rho_value,
            "N_samples": remaining,
            "P0": P0,
            "equilibration": equilibration,
            "properties": properties,
            "run_mode": run_mode_arg,
            "max_concurrent": max_concurrent,
            "shell_name": shell_name,
            "force_submit_exact": True,
            "target_samples": int(N_samples),
        })

    return missing


def run_multi_rho_array(
    L,
    p0,
    seed,
    type_perc,
    c,
    f_T,
    dim,
    num_colors,
    rho,              # pode ser float ou lista
    N_samples,
    P0,
    Equilibration=None,
    equilibration=True,
    properties=False,
    run_mode="sop",
    max_concurrent=3,
    shell_name="run_jobs_array.sh",
    force_submit_exact=False,
    target_samples=None,
):
    """
    Submete jobs SOP no cluster usando os parâmetros reescalados do modelo:

        p_i(t+1) = p_i(t) + c * [f_T - f_i(t)]

    O executável é chamado como:

        SOP L p0 seed type_perc c f_T dim num_colors rho P0 equilibration [properties] [run_mode]

    Se rho for escalar, submete um job fixed.
    Se rho for lista/array, cria um arquivo de rho e submete um SLURM array.

    equilibration controla o último argumento do executável SOP. Use:
        equilibration=True   para gerar cortes/partições de equilíbrio;
        equilibration=False  para desativar essa etapa.

    Equilibration é mantido apenas como alias retrocompatível.
    """
    # Compatibilidade: chamadas antigas podem usar Equilibration=0/1.
    # Chamadas novas devem preferir equilibration=False/True.
    if Equilibration is not None:
        equilibration = Equilibration

    equilibration_arg = _normalize_equilibration_input(equilibration)
    properties_arg = _normalize_bool_input(properties, "properties")
    run_mode_arg = _normalize_run_mode(run_mode)

    if run_mode_arg == "growth_test" and properties_arg == "true":
        raise ValueError("growth_test deve ser executado com properties=False")

    extra_submit_args = []
    if run_mode_arg != "sop" or properties_arg == "true":
        extra_submit_args = [properties_arg, run_mode_arg]

    shell_script = SHELLS_DIR / shell_name

    # recria sempre para garantir versão atualizada
    create_cluster_cli_shell(exec_name=shell_name)

    mode, rho_data = _normalize_rho_input(rho)
    target_samples = int(target_samples if target_samples is not None else N_samples)

    if mode == "fixed":
        if force_submit_exact:
            existing = 0
            remaining = int(N_samples)
            data_dir = _sample_data_dir(
                L=L,
                type_perc=type_perc,
                c=c,
                f_T=f_T,
                dim=dim,
                num_colors=num_colors,
                rho=rho_data,
                run_mode=run_mode_arg,
            )
        else:
            existing, remaining, data_dir = _remaining_samples_for_rho(
                L=L,
                p0=p0,
                type_perc=type_perc,
                c=c,
                f_T=f_T,
                dim=dim,
                num_colors=num_colors,
                rho=rho_data,
                N_samples=N_samples,
                P0=P0,
                run_mode=run_mode_arg,
            )
        print(
            "📦 Amostras:",
            f"L={L}",
            f"nc={num_colors}",
            f"rho={rho_data}",
            f"c={c}",
            f"f_T={f_T}",
            f"existentes={existing}",
            f"alvo={int(N_samples)}",
            f"restantes={remaining}",
        )
        if remaining <= 0:
            print(f"⏭️  Já completo; ignorando submissão. Pasta: {data_dir}")
            return

        cmd = [
            "sbatch",
            str(shell_script),
            "fixed",
            str(L),
            str(p0),
            str(seed),
            str(type_perc),
            str(c),
            str(f_T),
            str(dim),
            str(num_colors),
            str(remaining),
            str(target_samples),
            str(rho_data),
            str(P0),
            equilibration_arg,
            *extra_submit_args,
        ]
        print("🚀 Submetendo:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return

    # mode == "array"
    rho_lst = rho_data
    remaining_by_count = defaultdict(list)
    skipped = 0
    if force_submit_exact:
        remaining_by_count[int(N_samples)] = list(rho_lst)
    else:
        for rho_value in rho_lst:
            existing, remaining, data_dir = _remaining_samples_for_rho(
                L=L,
                p0=p0,
                type_perc=type_perc,
                c=c,
                f_T=f_T,
                dim=dim,
                num_colors=num_colors,
                rho=rho_value,
                N_samples=N_samples,
                P0=P0,
                run_mode=run_mode_arg,
            )
            print(
                "📦 Amostras:",
                f"L={L}",
                f"nc={num_colors}",
                f"rho={rho_value}",
                f"c={c}",
                f"f_T={f_T}",
                f"existentes={existing}",
                f"alvo={int(N_samples)}",
                f"restantes={remaining}",
            )
            if remaining <= 0:
                skipped += 1
                print(f"⏭️  Já completo; ignorando rho. Pasta: {data_dir}")
                continue
            remaining_by_count[remaining].append(rho_value)

    if not remaining_by_count:
        print(f"✅ Todos os {len(rho_lst)} valores de rho já estão completos.")
        return

    if skipped:
        print(f"⏭️  Rhos já completos ignorados: {skipped}/{len(rho_lst)}")

    for remaining, rho_group in sorted(remaining_by_count.items(), reverse=True):
        rho_file = SHELLS_DIR / (
            f"rho_L{L}_c{_float_tag(c)}_fT{_float_tag(f_T)}_dim{dim}_nc{num_colors}_"
            f"p0{p0}_P0{P0}_seed{seed}_remaining{remaining}.txt"
        )
        rho_file.write_text("\n".join(map(str, rho_group)) + "\n")

        n = len(rho_group)
        if (max_concurrent is None) or (int(max_concurrent) < 1):
            array_spec = f"0-{n-1}"
        else:
            array_spec = f"0-{n-1}%{int(max_concurrent)}"

        cmd = [
            "sbatch",
            f"--array={array_spec}",
            str(shell_script),
            "array",
            str(L),
            str(p0),
            str(seed),
            str(type_perc),
            str(c),
            str(f_T),
            str(dim),
            str(num_colors),
            str(remaining),
            str(target_samples),
            str(P0),
            equilibration_arg,
            str(rho_file),
            *extra_submit_args,
        ]

        print(
            "🚀 Submetendo:",
            " ".join(cmd),
            f"(rhos={n}, amostras_por_rho={remaining})",
        )
        subprocess.run(cmd, check=True)
