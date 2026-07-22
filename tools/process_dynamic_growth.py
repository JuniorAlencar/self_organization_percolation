#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import lzma
import math
import os
import re
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

XZ_BIN = shutil.which("xz")


DYNAMIC_PROCESSING_VERSION = 12
LATERAL_PROCESSING_VERSION = 4
SERIES_ENCODING_KEY = "__encoding__"
DEFAULT_MIN_SUPPORT_FRACTION = 0.8

FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

PARAM_RE = re.compile(
    rf"""
    (?P<type>[A-Za-z0-9]+)_percolation
    /num_colors_(?P<nc>\d+)
    /dim_(?P<dim>\d+)
    /L_(?P<L>\d+)
    /fT_constant
    /fT_(?P<fT>{FLOAT})
    /c_(?P<c>{FLOAT})
    /rho_(?P<rho>{FLOAT})
    (?:/stationary_window_(?P<stat_window>\d+))?
    /data$
    """,
    re.X,
)

RE_P0 = re.compile(rf"(?:^|_)P0_(?P<P0>{FLOAT})(?:_|\.json$)")
RE_p0 = re.compile(rf"(?:^|_)p0_(?P<p0>{FLOAT})(?:_|\.json$)")

ALL_DATA_COLUMNS = [
    "type_perc", "dim", "L", "f_T", "c", "nc", "rho", "p0", "P0",
    "order", "N_samples", "N_samples_perc",
    "p_mean", "p_err", "f_mean", "f_err", "z_max_mean", "z_max_err",
    "z_stat_mean", "z_stat_err", "stat_window",
    "stop_criterion", "t_eq_validation", "t_eq_s_prime_threshold",
    "equilibrium_effective_rel_tol", "post_equilibrium_extra_steps",
]

ALL_COLORS_COLUMNS = [
    "type_perc", "dim", "L", "f_T", "c", "num_colors", "P0", "p0",
    "N_samples", "rho", "nc", "nc_err", "nc_std", "stat_window",
    "stop_criterion", "t_eq_validation", "t_eq_s_prime_threshold",
    "equilibrium_effective_rel_tol", "post_equilibrium_extra_steps",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def manifest_path(manifests_root: Path, rel_group: Path) -> Path:
    return manifests_root / rel_group / "manifest.json"


def load_manifest(manifests_root: Path, rel_group: Path) -> dict[str, Any]:
    path = manifest_path(manifests_root, rel_group)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "processed_json_files": [],
        "n_processed_json_files": 0,
        "summary_file": None,
        "last_update": None,
    }


def rows_from_manifest_cache(
    manifest: dict[str, Any],
    series_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    if int(manifest.get("dynamic_processing_version", 0) or 0) != DYNAMIC_PROCESSING_VERSION:
        return None
    if str(manifest.get("series_mode", "full") or "full") != series_mode:
        return None
    rows = manifest.get("all_rows_cache")
    color_rows = manifest.get("all_color_rows_cache")
    if not isinstance(rows, list) or not isinstance(color_rows, list):
        return None
    return list(rows), list(color_rows)


def manifest_rows_cache_payload(
    all_rows: list[dict[str, Any]],
    all_color_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "all_rows_cache": json_safe(all_rows),
        "all_color_rows_cache": json_safe(all_color_rows),
    }


def save_manifest(manifests_root: Path,
                  rel_group: Path,
                  manifest: dict[str, Any]) -> Path:
    path = manifest_path(manifests_root, rel_group)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(manifest), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    return path


def open_text_auto(path: Path, mode: str, compresslevel: int = 6):
    if path.suffix == ".gz":
        if any(flag in mode for flag in ("w", "a", "x")):
            return gzip.open(path, mode, encoding="utf-8", compresslevel=compresslevel)
        return gzip.open(path, mode, encoding="utf-8")
    if path.suffix == ".xz":
        if any(flag in mode for flag in ("w", "a", "x")):
            return lzma.open(path, mode, encoding="utf-8", preset=compresslevel)
        return lzma.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def open_binary_auto(path: Path, mode: str, compresslevel: int = 6):
    if path.suffix == ".gz":
        if any(flag in mode for flag in ("w", "a", "x")):
            return gzip.open(path, mode, compresslevel=compresslevel)
        return gzip.open(path, mode)
    if path.suffix == ".xz":
        if any(flag in mode for flag in ("w", "a", "x")):
            return lzma.open(path, mode, preset=compresslevel)
        return lzma.open(path, mode)
    return path.open(mode)


def load_json_bundle(path: Path) -> dict[str, Any]:
    with open_text_auto(path, "rt") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON bundle: {path}")
    return data


def write_json_bundle(path: Path, data: dict[str, Any], pretty: bool = False, compresslevel: int = 6) -> None:
    with open_text_auto(path, "wt", compresslevel=compresslevel) as f:
        if pretty:
            json.dump(json_safe(data), f, ensure_ascii=False, indent=2)
        else:
            json.dump(json_safe(data), f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


def compress_json_to_xz(
    source_path: Path,
    target_path: Path,
    *,
    compresslevel: int = 1,
    threads: int = 0,
) -> None:
    tmp_path = target_path.with_name(target_path.name + ".tmp")
    if XZ_BIN:
        with open_binary_auto(source_path, "rb") as source, tmp_path.open("wb") as target:
            proc = subprocess.Popen(
                [XZ_BIN, f"-{int(compresslevel)}", f"-T{int(threads)}", "-c"],
                stdin=subprocess.PIPE,
                stdout=target,
            )
            assert proc.stdin is not None
            try:
                shutil.copyfileobj(source, proc.stdin, length=1024 * 1024)
                proc.stdin.close()
                rc = proc.wait()
            except Exception:
                proc.kill()
                proc.wait()
                raise
            if rc != 0:
                raise RuntimeError(f"xz failed with exit code {rc}: {source_path}")
    else:
        with open_binary_auto(source_path, "rb") as source, open_binary_auto(tmp_path, "wb", compresslevel=compresslevel) as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    tmp_path.replace(target_path)


def finite_float(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_stat_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"stat:{stat.st_size}:{stat.st_mtime_ns}"


def file_fingerprint_for_mode(path: Path, mode: str) -> str:
    if mode == "hash":
        return file_fingerprint(path)
    if mode == "stat":
        return file_stat_fingerprint(path)
    raise ValueError(f"Unknown fingerprint mode: {mode}")


def fingerprint_matches_mode(fingerprint: str | None, mode: str) -> bool:
    if not fingerprint:
        return False
    if mode == "stat":
        return fingerprint.startswith("stat:")
    if mode == "hash":
        return not fingerprint.startswith("stat:")
    return False


def parse_sample_name(path: Path) -> tuple[float, float] | None:
    name = path.name
    m_p0 = RE_p0.search(name)
    m_P0 = RE_P0.search(name)
    if not m_p0 or not m_P0:
        return None
    return float(m_P0.group("P0")), float(m_p0.group("p0"))


def parse_data_dir(path: Path) -> dict[str, Any] | None:
    m = PARAM_RE.search(path.as_posix())
    if not m:
        return None
    g = m.groupdict()
    return {
        "type_perc": g["type"],
        "dim": int(g["dim"]),
        "L": int(g["L"]),
        "f_T": float(g["fT"]),
        "c": float(g["c"]),
        "nc": int(g["nc"]),
        "rho": float(g["rho"]),
        "stat_window": int(g["stat_window"]) if g.get("stat_window") else 0,
    }


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def mean_sem(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return math.nan, math.nan, 0
    mean = float(arr.mean())
    if n == 1:
        return mean, 0.0, 1
    return mean, float(arr.std(ddof=1) / math.sqrt(n)), n


def mean_sem_std(values: list[float]) -> tuple[float, float, float, int]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(arr.mean())
    if n == 1:
        return mean, 0.0, 0.0, 1
    std = float(arr.std(ddof=1))
    return mean, float(std / math.sqrt(n)), std, n


def dat_value(value: Any) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.12g}"
    return str(value)


def parse_order_key(key: str) -> int | None:
    m = re.search(r"order_percolation\s+(\d+)", key)
    if not m:
        return None
    return int(m.group(1))


def dynamic_criterion_metadata_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "stop_criterion": meta.get("growth_test_stop_criterion"),
        "t_eq_validation": meta.get("growth_test_t_eq_validation"),
        "t_eq_s_prime_threshold": finite_float(meta.get("growth_test_t_eq_s_prime_threshold")),
        "equilibrium_effective_rel_tol": finite_float(meta.get("growth_test_equilibrium_effective_rel_tol")),
        "post_equilibrium_extra_steps": meta.get("growth_test_post_equilibrium_extra_steps"),
        "equilibrium_rel_tol_scaling": meta.get("growth_test_equilibrium_rel_tol_scaling"),
    }


def common_dynamic_criterion_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "stop_criterion",
        "t_eq_validation",
        "t_eq_s_prime_threshold",
        "equilibrium_effective_rel_tol",
        "post_equilibrium_extra_steps",
        "equilibrium_rel_tol_scaling",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        values: list[Any] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            if value not in values:
                values.append(value)
        if len(values) == 1:
            out[key] = values[0]
        elif len(values) > 1:
            out[key] = values
    return out


def tail_mean_after_t_eq(time: Any, values: Any, t_eq: float) -> float | None:
    if time is None or values is None:
        return None
    try:
        t = np.asarray(time, dtype=float)
        y = np.asarray(values, dtype=float)
    except Exception:
        return None
    n = min(t.size, y.size)
    if n == 0:
        return None
    t = t[:n]
    y = y[:n]
    mask = (t > float(t_eq)) & np.isfinite(y)
    if not np.any(mask):
        return None
    return float(np.mean(y[mask]))


def min_support_count(n_seeds: int, fraction: float = DEFAULT_MIN_SUPPORT_FRACTION) -> int:
    if n_seeds <= 0:
        return 0
    return max(1, int(math.ceil(float(fraction) * n_seeds)))


def support_indices(counts: list[int], n_seeds: int,
                    fraction: float = DEFAULT_MIN_SUPPORT_FRACTION) -> list[int]:
    threshold = min_support_count(n_seeds, fraction)
    return [idx for idx, n in enumerate(counts) if n >= threshold]


def regular_time_signature(t: np.ndarray) -> tuple[float, float] | None:
    if t.size == 0:
        return None
    start = float(t[0])
    if t.size == 1:
        return start, 1.0
    step = float(t[1] - t[0])
    if not math.isfinite(step) or step <= 0.0:
        return None
    expected = start + step * np.arange(t.size, dtype=float)
    if not np.allclose(t, expected, rtol=0.0, atol=1e-10):
        return None
    return start, step


def mean_regular_prefix_series(cleaned: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any] | None:
    if not cleaned:
        return None
    first_sig = regular_time_signature(cleaned[0][0])
    if first_sig is None:
        return None
    start, step = first_sig
    max_len = max(int(t.size) for t, _ in cleaned)
    sums = np.zeros(max_len, dtype=float)
    sums_sq = np.zeros(max_len, dtype=float)
    counts = np.zeros(max_len, dtype=np.int64)

    for t, y in cleaned:
        sig = regular_time_signature(t)
        if sig is None:
            return None
        t_start, t_step = sig
        if not math.isclose(t_start, start, rel_tol=0.0, abs_tol=1e-10):
            return None
        if not math.isclose(t_step, step, rel_tol=1e-12, abs_tol=1e-10):
            return None
        n = int(min(t.size, y.size))
        if n <= 0:
            continue
        vals = y[:n]
        mask = np.isfinite(vals)
        if not np.any(mask):
            continue
        idx = np.flatnonzero(mask)
        finite_vals = vals[idx]
        sums[idx] += finite_vals
        sums_sq[idx] += finite_vals * finite_vals
        counts[idx] += 1

    valid = counts > 0
    if not np.any(valid):
        return None
    last = int(np.flatnonzero(valid)[-1]) + 1
    counts = counts[:last]
    sums = sums[:last]
    sums_sq = sums_sq[:last]
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variance = np.zeros_like(sums)
    multi = counts > 1
    variance[multi] = np.maximum(
        (sums_sq[multi] - sums[multi] * sums[multi] / counts[multi]) / (counts[multi] - 1),
        0.0,
    )
    std = np.sqrt(variance)
    sem = np.divide(std, np.sqrt(counts), out=np.zeros_like(std), where=counts > 1)
    n_per_t = counts.astype(int).tolist()
    n_seeds = int(len(cleaned))
    common_idx = [idx for idx, n in enumerate(n_per_t) if n == n_seeds]
    supported_idx = support_indices(n_per_t, n_seeds)
    support_count = min_support_count(n_seeds)
    t_grid = start + step * np.arange(last, dtype=float)

    return {
        "time": t_grid.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "sem": sem.tolist(),
        "N_per_t": n_per_t,
        "n_seeds": n_seeds,
        "common_time": t_grid[common_idx].tolist(),
        "common_mean": mean[common_idx].tolist(),
        "common_std": std[common_idx].tolist(),
        "common_sem": sem[common_idx].tolist(),
        "common_N_per_t": [n_per_t[idx] for idx in common_idx],
        "supported_time": t_grid[supported_idx].tolist(),
        "supported_mean": mean[supported_idx].tolist(),
        "supported_std": std[supported_idx].tolist(),
        "supported_sem": sem[supported_idx].tolist(),
        "supported_N_per_t": [n_per_t[idx] for idx in supported_idx],
        "min_support_fraction": DEFAULT_MIN_SUPPORT_FRACTION,
        "min_support_count": support_count,
    }


def mean_series_on_union_grid(series: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    if not series:
        return {
            "time": [],
            "mean": [],
            "std": [],
            "sem": [],
            "N_per_t": [],
            "n_seeds": 0,
            "common_time": [],
            "common_mean": [],
            "common_std": [],
            "common_sem": [],
            "common_N_per_t": [],
            "supported_time": [],
            "supported_mean": [],
            "supported_std": [],
            "supported_sem": [],
            "supported_N_per_t": [],
            "min_support_fraction": DEFAULT_MIN_SUPPORT_FRACTION,
            "min_support_count": 0,
        }

    cleaned: list[tuple[np.ndarray, np.ndarray]] = []
    for t, y in series:
        n = min(t.size, y.size)
        if n <= 0:
            continue
        t_i = np.asarray(t[:n], dtype=float)
        y_i = np.asarray(y[:n], dtype=float)
        mask = np.isfinite(t_i) & np.isfinite(y_i)
        if not np.any(mask):
            continue
        t_i = t_i[mask]
        y_i = y_i[mask]
        order = np.argsort(t_i)
        cleaned.append((t_i[order], y_i[order]))

    if not cleaned:
        return {
            "time": [],
            "mean": [],
            "std": [],
            "sem": [],
            "N_per_t": [],
            "n_seeds": 0,
            "common_time": [],
            "common_mean": [],
            "common_std": [],
            "common_sem": [],
            "common_N_per_t": [],
            "supported_time": [],
            "supported_mean": [],
            "supported_std": [],
            "supported_sem": [],
            "supported_N_per_t": [],
            "min_support_fraction": DEFAULT_MIN_SUPPORT_FRACTION,
            "min_support_count": 0,
        }

    regular_stats = mean_regular_prefix_series(cleaned)
    if regular_stats is not None:
        return regular_stats

    accum: dict[float, list[float]] = {}
    for t, y in cleaned:
        per_series: dict[float, float] = {}
        for tt, yy in zip(t, y):
            per_series[float(tt)] = float(yy)
        for tt, yy in per_series.items():
            stats = accum.setdefault(tt, [0.0, 0.0, 0.0])
            stats[0] += 1.0
            stats[1] += yy
            stats[2] += yy * yy

    t_grid = sorted(accum)
    mean: list[float] = []
    std: list[float] = []
    sem: list[float] = []
    n_per_t: list[int] = []
    for tt in t_grid:
        n = int(accum[tt][0])
        total = accum[tt][1]
        sumsq = accum[tt][2]
        n_per_t.append(n)
        m = total / n
        mean.append(float(m))
        if n > 1:
            variance = max((sumsq - total * total / n) / (n - 1), 0.0)
            s = math.sqrt(variance)
            std.append(s)
            sem.append(float(s / math.sqrt(n)))
        else:
            std.append(0.0)
            sem.append(0.0)

    n_seeds = int(len(cleaned))
    common_idx = [idx for idx, n in enumerate(n_per_t) if n == n_seeds]
    supported_idx = support_indices(n_per_t, n_seeds)
    support_count = min_support_count(n_seeds)

    return {
        "time": [float(t) for t in t_grid],
        "mean": mean,
        "std": std,
        "sem": sem,
        "N_per_t": n_per_t,
        "n_seeds": n_seeds,
        "common_time": [float(t_grid[idx]) for idx in common_idx],
        "common_mean": [mean[idx] for idx in common_idx],
        "common_std": [std[idx] for idx in common_idx],
        "common_sem": [sem[idx] for idx in common_idx],
        "common_N_per_t": [n_per_t[idx] for idx in common_idx],
        "supported_time": [float(t_grid[idx]) for idx in supported_idx],
        "supported_mean": [mean[idx] for idx in supported_idx],
        "supported_std": [std[idx] for idx in supported_idx],
        "supported_sem": [sem[idx] for idx in supported_idx],
        "supported_N_per_t": [n_per_t[idx] for idx in supported_idx],
        "min_support_fraction": DEFAULT_MIN_SUPPORT_FRACTION,
        "min_support_count": support_count,
    }


def clean_numeric_vector(values: Any) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        return np.asarray([], dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.asarray([], dtype=float)
    return arr


def mean_indexed_series(series: list[np.ndarray]) -> dict[str, Any]:
    cleaned = [clean_numeric_vector(values) for values in series]
    cleaned = [values for values in cleaned if values.size > 0]
    if not cleaned:
        return {
            "z": [],
            "mean": [],
            "std": [],
            "sem": [],
            "N_per_z": [],
            "n_seeds": 0,
            "common_z": [],
            "common_mean": [],
            "common_std": [],
            "common_sem": [],
            "common_N_per_z": [],
            "supported_z": [],
            "supported_mean": [],
            "supported_std": [],
            "supported_sem": [],
            "supported_N_per_z": [],
            "min_support_fraction": DEFAULT_MIN_SUPPORT_FRACTION,
            "min_support_count": 0,
        }

    max_len = max(int(values.size) for values in cleaned)
    sums = np.zeros(max_len, dtype=float)
    sums_sq = np.zeros(max_len, dtype=float)
    counts = np.zeros(max_len, dtype=np.int64)

    for values in cleaned:
        mask = np.isfinite(values)
        if not np.any(mask):
            continue
        idx = np.flatnonzero(mask)
        vals = values[idx]
        sums[idx] += vals
        sums_sq[idx] += vals * vals
        counts[idx] += 1

    valid = counts > 0
    if not np.any(valid):
        return {
            "z": [],
            "mean": [],
            "std": [],
            "sem": [],
            "N_per_z": [],
            "n_seeds": 0,
            "common_z": [],
            "common_mean": [],
            "common_std": [],
            "common_sem": [],
            "common_N_per_z": [],
            "supported_z": [],
            "supported_mean": [],
            "supported_std": [],
            "supported_sem": [],
            "supported_N_per_z": [],
            "min_support_fraction": DEFAULT_MIN_SUPPORT_FRACTION,
            "min_support_count": 0,
        }

    last = int(np.flatnonzero(valid)[-1]) + 1
    sums = sums[:last]
    sums_sq = sums_sq[:last]
    counts = counts[:last]
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variance = np.zeros_like(sums)
    multi = counts > 1
    variance[multi] = np.maximum(
        (sums_sq[multi] - sums[multi] * sums[multi] / counts[multi]) / (counts[multi] - 1),
        0.0,
    )
    std = np.sqrt(variance)
    sem = np.divide(std, np.sqrt(counts), out=np.zeros_like(std), where=counts > 1)
    n_seeds = int(len(cleaned))
    common = counts == n_seeds
    support_count = min_support_count(n_seeds)
    supported = counts >= support_count
    z = np.arange(last, dtype=np.int64)

    return {
        "z": z.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "sem": sem.tolist(),
        "N_per_z": counts.astype(int).tolist(),
        "n_seeds": n_seeds,
        "common_z": z[common].tolist(),
        "common_mean": mean[common].tolist(),
        "common_std": std[common].tolist(),
        "common_sem": sem[common].tolist(),
        "common_N_per_z": counts[common].astype(int).tolist(),
        "supported_z": z[supported].tolist(),
        "supported_mean": mean[supported].tolist(),
        "supported_std": std[supported].tolist(),
        "supported_sem": sem[supported].tolist(),
        "supported_N_per_z": counts[supported].astype(int).tolist(),
        "min_support_fraction": DEFAULT_MIN_SUPPORT_FRACTION,
        "min_support_count": support_count,
    }


def average_dynamic_time_series(items: list[dict[str, Any]]) -> dict[str, Any]:
    series_pt: list[tuple[np.ndarray, np.ndarray]] = []
    series_ft: list[tuple[np.ndarray, np.ndarray]] = []
    series_flz: list[np.ndarray] = []
    t_eq_vals: list[float] = []

    for item in items:
        t_eq = finite_float(item.get("t_eq_species"))
        if t_eq is not None:
            t_eq_vals.append(t_eq)

        flz = clean_numeric_vector(item.get("fL_z"))
        if flz.size > 0:
            series_flz.append(flz)

        time = item.get("time")
        pt = item.get("pt")
        if time is None or pt is None:
            continue
        try:
            t = np.asarray(time, dtype=float)
            p = np.asarray(pt, dtype=float)
        except Exception:
            continue
        n_pt = min(t.size, p.size)
        if n_pt <= 1:
            continue
        series_pt.append((t[:n_pt], p[:n_pt]))

        ft = item.get("ft")
        if ft is not None:
            try:
                f = np.asarray(ft, dtype=float)
            except Exception:
                f = np.asarray([], dtype=float)
            n_ft = min(n_pt, f.size)
            if n_ft > 1:
                series_ft.append((t[:n_ft], f[:n_ft]))

    out: dict[str, Any] = {}
    if t_eq_vals:
        arr = np.asarray(t_eq_vals, dtype=float)
        out["t_eq"] = float(np.max(arr))
        out["t_eq_mean"] = float(np.mean(arr))
        out["t_eq_min"] = float(np.min(arr))
        out["t_eq_max"] = float(np.max(arr))
        out["n_t_eq"] = int(arr.size)
    else:
        out["t_eq"] = math.nan
        out["t_eq_mean"] = math.nan
        out["t_eq_min"] = math.nan
        out["t_eq_max"] = math.nan
        out["n_t_eq"] = 0

    if series_pt:
        pt_stats = mean_series_on_union_grid(series_pt)
        out["time"] = pt_stats["time"]
        out["pt_mean"] = pt_stats["mean"]
        out["pt_std"] = pt_stats["std"]
        out["pt_sem"] = pt_stats["sem"]
        out["pt_N_per_t"] = pt_stats["N_per_t"]
        out["n_seeds_pt"] = pt_stats["n_seeds"]
        out["pt_common_time"] = pt_stats["common_time"]
        out["pt_common_mean"] = pt_stats["common_mean"]
        out["pt_common_std"] = pt_stats["common_std"]
        out["pt_common_sem"] = pt_stats["common_sem"]
        out["pt_common_N_per_t"] = pt_stats["common_N_per_t"]
        out["pt_supported_time"] = pt_stats["supported_time"]
        out["pt_supported_mean"] = pt_stats["supported_mean"]
        out["pt_supported_std"] = pt_stats["supported_std"]
        out["pt_supported_sem"] = pt_stats["supported_sem"]
        out["pt_supported_N_per_t"] = pt_stats["supported_N_per_t"]
        out["pt_support_policy"] = "union_observed_times"
        out["pt_common_support_policy"] = "all_samples_present"
        out["pt_supported_support_policy"] = "min_fraction_of_samples_present"
        out["pt_min_support_count"] = pt_stats["min_support_count"]
    else:
        out["time"] = []
        out["pt_mean"] = []
        out["pt_std"] = []
        out["pt_sem"] = []
        out["pt_N_per_t"] = []
        out["n_seeds_pt"] = 0
        out["pt_common_time"] = []
        out["pt_common_mean"] = []
        out["pt_common_std"] = []
        out["pt_common_sem"] = []
        out["pt_common_N_per_t"] = []
        out["pt_supported_time"] = []
        out["pt_supported_mean"] = []
        out["pt_supported_std"] = []
        out["pt_supported_sem"] = []
        out["pt_supported_N_per_t"] = []
        out["pt_min_support_count"] = 0
    out["min_support_fraction"] = DEFAULT_MIN_SUPPORT_FRACTION

    if series_ft:
        ft_stats = mean_series_on_union_grid(series_ft)
        out["ft_time"] = ft_stats["time"]
        out["ft_mean"] = ft_stats["mean"]
        out["ft_std"] = ft_stats["std"]
        out["ft_sem"] = ft_stats["sem"]
        out["ft_N_per_t"] = ft_stats["N_per_t"]
        out["n_seeds_ft"] = ft_stats["n_seeds"]
        out["ft_common_time"] = ft_stats["common_time"]
        out["ft_common_mean"] = ft_stats["common_mean"]
        out["ft_common_std"] = ft_stats["common_std"]
        out["ft_common_sem"] = ft_stats["common_sem"]
        out["ft_common_N_per_t"] = ft_stats["common_N_per_t"]
        out["ft_supported_time"] = ft_stats["supported_time"]
        out["ft_supported_mean"] = ft_stats["supported_mean"]
        out["ft_supported_std"] = ft_stats["supported_std"]
        out["ft_supported_sem"] = ft_stats["supported_sem"]
        out["ft_supported_N_per_t"] = ft_stats["supported_N_per_t"]
        out["ft_support_policy"] = "union_observed_times"
        out["ft_common_support_policy"] = "all_samples_present"
        out["ft_supported_support_policy"] = "min_fraction_of_samples_present"
        out["ft_min_support_count"] = ft_stats["min_support_count"]
    else:
        out["ft_mean"] = []
        out["ft_std"] = []
        out["ft_sem"] = []
        out["ft_N_per_t"] = []
        out["n_seeds_ft"] = 0
        out["ft_supported_time"] = []
        out["ft_supported_mean"] = []
        out["ft_supported_std"] = []
        out["ft_supported_sem"] = []
        out["ft_supported_N_per_t"] = []
        out["ft_min_support_count"] = 0

    flz_stats = mean_indexed_series(series_flz)
    out["fL_z_z"] = flz_stats["z"]
    out["fL_z_mean"] = flz_stats["mean"]
    out["fL_z_std"] = flz_stats["std"]
    out["fL_z_sem"] = flz_stats["sem"]
    out["fL_z_N_per_z"] = flz_stats["N_per_z"]
    out["n_seeds_fL_z"] = flz_stats["n_seeds"]
    out["fL_z_common_z"] = flz_stats["common_z"]
    out["fL_z_common_mean"] = flz_stats["common_mean"]
    out["fL_z_common_std"] = flz_stats["common_std"]
    out["fL_z_common_sem"] = flz_stats["common_sem"]
    out["fL_z_common_N_per_z"] = flz_stats["common_N_per_z"]
    out["fL_z_supported_z"] = flz_stats["supported_z"]
    out["fL_z_supported_mean"] = flz_stats["supported_mean"]
    out["fL_z_supported_std"] = flz_stats["supported_std"]
    out["fL_z_supported_sem"] = flz_stats["supported_sem"]
    out["fL_z_supported_N_per_z"] = flz_stats["supported_N_per_z"]
    out["fL_z_support_policy"] = "union_observed_heights"
    out["fL_z_common_support_policy"] = "all_samples_present"
    out["fL_z_supported_support_policy"] = "min_fraction_of_samples_present"
    out["fL_z_min_support_count"] = flz_stats["min_support_count"]

    return out


def zmax_for_order(
    *,
    order_pos: int,
    color_1b: int | None,
    t_eq_species: float,
    result_t_eq_order: list[float],
    meta_t_eq: list[Any],
    meta_zmax: list[Any],
) -> float | None:
    if not meta_zmax:
        return None

    # New/fixed raw format: meta.t_eq_by_species is already ordered by
    # stabilization and nulls are trailing; z_max is aligned to that order.
    ordered_prefix = True
    if len(meta_t_eq) >= len(result_t_eq_order):
        for a, b in zip(meta_t_eq[:len(result_t_eq_order)], result_t_eq_order):
            fa = finite_float(a)
            if fa is None or not math.isclose(fa, b, rel_tol=1e-12, abs_tol=1e-12):
                ordered_prefix = False
                break
    else:
        ordered_prefix = False

    if ordered_prefix and order_pos < len(meta_zmax):
        return finite_float(meta_zmax[order_pos])

    # Older raw files were species-indexed. Fall back to color when available.
    if color_1b is not None:
        idx = int(color_1b) - 1
        if 0 <= idx < len(meta_zmax):
            return finite_float(meta_zmax[idx])

    # Last resort: pair the finite t_eq entries with z_max by position, skipping
    # null t_eq values. This implements the dynamic-order convention requested.
    finite_positions = [
        i for i, v in enumerate(meta_t_eq)
        if finite_float(v) is not None and i < len(meta_zmax)
    ]
    if order_pos < len(finite_positions):
        return finite_float(meta_zmax[finite_positions[order_pos]])

    if order_pos < len(meta_zmax):
        return finite_float(meta_zmax[order_pos])
    return None


def load_dynamic_sample(
    path: Path,
    *,
    include_time_series: bool = True,
    include_flz: bool = True,
) -> list[dict[str, Any]] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            js = json.load(f)
    except Exception as exc:
        print(f"[warn] ignorando JSON inválido {path}: {exc}")
        return None

    meta = js.get("meta", {}) if isinstance(js.get("meta", {}), dict) else {}
    results = js.get("results", {}) if isinstance(js.get("results", {}), dict) else {}
    if not results:
        return []
    meta_t_eq_for_fallback = meta.get("t_eq_by_species", [])
    if not isinstance(meta_t_eq_for_fallback, list):
        meta_t_eq_for_fallback = []
    criterion_meta = dynamic_criterion_metadata_from_meta(meta)

    candidate_blocks: list[tuple[int, dict[str, Any]]] = []
    for key, block in results.items():
        raw_order = parse_order_key(str(key))
        if raw_order is None:
            continue
        data = (block or {}).get("data", {})
        if not isinstance(data, dict):
            continue
        candidate_blocks.append((raw_order, dict(data)))

    candidate_blocks.sort(key=lambda x: x[0])

    parsed_blocks: list[tuple[int, dict[str, Any]]] = []
    for order_pos, (raw_order, data) in enumerate(candidate_blocks):
        t_eq = finite_float(data.get("t_eq_species"))
        if t_eq is None:
            t_eq = finite_float(data.get("t_eq"))
        if t_eq is None and order_pos < len(meta_t_eq_for_fallback):
            t_eq = finite_float(meta_t_eq_for_fallback[order_pos])
        if t_eq is None:
            continue
        data["t_eq_species"] = t_eq
        parsed_blocks.append((raw_order, data))

    if not parsed_blocks:
        return []

    result_t_eq_order = [float(d.get("t_eq_species")) for _, d in parsed_blocks]
    meta_t_eq = meta.get("t_eq_by_species", [])
    meta_zmax = meta.get("z_max", meta.get("z_max_final", []))
    meta_zstat = meta.get("z_stat", [])
    if not isinstance(meta_t_eq, list):
        meta_t_eq = []
    if not isinstance(meta_zmax, list):
        meta_zmax = []
    if not isinstance(meta_zstat, list):
        meta_zstat = []

    out: list[dict[str, Any]] = []
    for order_pos, (_, data) in enumerate(parsed_blocks):
        t_eq = float(data["t_eq_species"])
        color = data.get("color")
        try:
            color_1b = int(color)
        except Exception:
            color_1b = None

        p_mean = tail_mean_after_t_eq(data.get("time"), data.get("pt"), t_eq)
        f_mean = tail_mean_after_t_eq(data.get("time"), data.get("nt"), t_eq)
        z_max = zmax_for_order(
            order_pos=order_pos,
            color_1b=color_1b,
            t_eq_species=t_eq,
            result_t_eq_order=result_t_eq_order,
            meta_t_eq=meta_t_eq,
            meta_zmax=meta_zmax,
        )
        z_stat = zmax_for_order(
            order_pos=order_pos,
            color_1b=color_1b,
            t_eq_species=t_eq,
            result_t_eq_order=result_t_eq_order,
            meta_t_eq=meta_t_eq,
            meta_zmax=meta_zstat,
        )

        row = {
            "order": order_pos,
            "color": color_1b,
            "t_eq_species": t_eq,
            "p_sample_mean": p_mean,
            "f_sample_mean": f_mean,
            "z_max": z_max,
            "z_stat": z_stat,
            **criterion_meta,
        }
        if include_time_series:
            row["time"] = data.get("time")
            row["pt"] = data.get("pt")
            row["ft"] = data.get("nt")
        if include_flz:
            row["fL_z"] = data.get("fL_z")
        out.append(row)

    return out


def process_one_sample_file(
    sample_path: Path,
    include_time_series: bool = True,
    include_flz: bool = True,
) -> tuple[list[dict[str, Any]], float | None]:
    sample_orders = load_dynamic_sample(
        sample_path,
        include_time_series=include_time_series,
        include_flz=include_flz,
    )
    if sample_orders is None:
        return [], None

    rows: list[dict[str, Any]] = []
    for item in sample_orders:
        rows.append({
            "filename": sample_path.name,
            "order": int(item["order"]),
            "color": item.get("color"),
            "t_eq_species": item.get("t_eq_species"),
            "time": item.get("time"),
            "pt": item.get("pt"),
            "ft": item.get("ft"),
            "fL_z": item.get("fL_z"),
            "p_sample_mean": item.get("p_sample_mean"),
            "f_sample_mean": item.get("f_sample_mean"),
            "z_max": item.get("z_max"),
            "z_stat": item.get("z_stat"),
            "stop_criterion": item.get("stop_criterion"),
            "t_eq_validation": item.get("t_eq_validation"),
            "t_eq_s_prime_threshold": item.get("t_eq_s_prime_threshold"),
            "equilibrium_effective_rel_tol": item.get("equilibrium_effective_rel_tol"),
            "post_equilibrium_extra_steps": item.get("post_equilibrium_extra_steps"),
            "equilibrium_rel_tol_scaling": item.get("equilibrium_rel_tol_scaling"),
        })
    return rows, float(len(sample_orders))


def process_one_sample_file_for_pool(args: tuple[Path, bool, bool]) -> tuple[list[dict[str, Any]], float | None]:
    sample_path, include_time_series, include_flz = args
    return process_one_sample_file(
        sample_path,
        include_time_series=include_time_series,
        include_flz=include_flz,
    )


def sample_cache_path(cache_dir: Path, sample_path: Path) -> Path:
    digest = hashlib.sha1(sample_path.name.encode("utf-8")).hexdigest()
    return cache_dir / f"{sample_path.stem}_{digest[:12]}.summary.json"


def process_one_sample_file_cached(
    args: tuple[Path, bool, bool, str, str | None],
) -> tuple[list[dict[str, Any]], float | None]:
    sample_path, include_time_series, include_flz, fingerprint_mode, cache_dir_raw = args
    if include_time_series or cache_dir_raw is None:
        return process_one_sample_file(
            sample_path,
            include_time_series=include_time_series,
            include_flz=include_flz,
        )

    cache_dir = Path(cache_dir_raw)
    fingerprint = file_fingerprint_for_mode(sample_path, fingerprint_mode)
    cache_path = sample_cache_path(cache_dir, sample_path)
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if (
            isinstance(cached, dict)
            and cached.get("fingerprint") == fingerprint
            and cached.get("fingerprint_mode") == fingerprint_mode
            and bool(cached.get("include_flz")) == include_flz
            and cached.get("format_version") == DYNAMIC_PROCESSING_VERSION
        ):
            rows = cached.get("rows", [])
            stabilized_count = cached.get("stabilized_count")
            if isinstance(rows, list):
                return rows, finite_float(stabilized_count)
    except Exception:
        pass

    rows, stabilized_count = process_one_sample_file(
        sample_path,
        include_time_series=False,
        include_flz=include_flz,
    )
    try:
        ensure_dir(cache_dir)
        payload = {
            "format_version": DYNAMIC_PROCESSING_VERSION,
            "fingerprint": fingerprint,
            "fingerprint_mode": fingerprint_mode,
            "include_flz": include_flz,
            "stabilized_count": stabilized_count,
            "rows": rows,
        }
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
    except Exception:
        pass
    return rows, stabilized_count


def process_sample_files(
    sample_paths: list[Path],
    jobs: int = 1,
    include_time_series: bool = True,
    include_flz: bool = True,
    cache_dir: Path | None = None,
    fingerprint_mode: str = "stat",
) -> tuple[list[dict[str, Any]], list[float]]:
    """Process a list of dynamic sample JSON files and return parsed rows plus per-file stabilization counts."""
    rows: list[dict[str, Any]] = []
    stabilized_counts: list[float] = []

    if jobs <= 1 or len(sample_paths) < 8:
        results = (
            process_one_sample_file_cached(
                (
                    path,
                    include_time_series,
                    include_flz,
                    fingerprint_mode,
                    cache_dir.as_posix() if cache_dir is not None else None,
                )
            )
            for path in sample_paths
        )
        for sample_rows, stabilized_count in results:
            if stabilized_count is None:
                continue
            stabilized_counts.append(stabilized_count)
            rows.extend(sample_rows)
    else:
        workers = min(jobs, len(sample_paths))
        chunksize = max(1, min(32, len(sample_paths) // max(1, workers * 4)))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(
                process_one_sample_file_cached,
                [
                    (
                        path,
                        include_time_series,
                        include_flz,
                        fingerprint_mode,
                        cache_dir.as_posix() if cache_dir is not None else None,
                    )
                    for path in sample_paths
                ],
                chunksize=chunksize,
            )
            for sample_rows, stabilized_count in results:
                if stabilized_count is None:
                    continue
                stabilized_counts.append(stabilized_count)
                rows.extend(sample_rows)

    return rows, stabilized_counts


def compact_series_column(key: str, values: list[Any]) -> Any:
    if not values:
        return []
    first = values[0]
    if all(value == first for value in values):
        return first
    if key == "t" and len(values) > 1:
        numeric = all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)
        if numeric:
            step = float(values[1]) - float(values[0])
            if all(
                math.isclose(float(values[idx]) - float(values[idx - 1]), step, rel_tol=0.0, abs_tol=1e-12)
                for idx in range(1, len(values))
            ):
                return {
                    SERIES_ENCODING_KEY: "range",
                    "start": float(values[0]),
                    "step": step,
                    "n": len(values),
                }
    return values


def compact_series_columns(columns: dict[str, Any]) -> dict[str, Any]:
    return {
        key: compact_series_column(key, values) if isinstance(values, list) else values
        for key, values in columns.items()
    }


def encoded_series_column_length(values: Any) -> int:
    if isinstance(values, list):
        return len(values)
    if isinstance(values, dict) and values.get(SERIES_ENCODING_KEY) == "range":
        return int(values.get("n", 0) or 0)
    return 0


def expand_series_column(values: Any, n_rows: int) -> list[Any]:
    if isinstance(values, list):
        if len(values) >= n_rows:
            return values[:n_rows]
        return values + [None] * (n_rows - len(values))
    if isinstance(values, dict) and values.get(SERIES_ENCODING_KEY) == "range":
        start = float(values.get("start", 0.0) or 0.0)
        step = float(values.get("step", 0.0) or 0.0)
        n = min(n_rows, int(values.get("n", 0) or 0))
        out = [start + step * idx for idx in range(n)]
        return out + [None] * (n_rows - n)
    return [values] * n_rows


def infer_series_length(series: Any) -> int:
    if isinstance(series, list):
        return len([row for row in series if isinstance(row, dict)])
    if isinstance(series, dict):
        return max((encoded_series_column_length(values) for values in series.values()), default=0)
    return 0


def series_to_columns(series: Any, series_length: int | None = None) -> dict[str, list[Any]]:
    if isinstance(series, list):
        rows = [row for row in series if isinstance(row, dict)]
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        return {key: [row.get(key) for row in rows] for key in keys}
    if not isinstance(series, dict):
        return {}
    n_rows = int(series_length or 0) or infer_series_length(series)
    return {key: expand_series_column(values, n_rows) for key, values in series.items()}


def series_rows_to_columns(series: list[dict[str, Any]]) -> dict[str, Any]:
    """Store a time series as aligned property lists instead of per-time dicts."""
    keys: list[str] = []
    seen: set[str] = set()
    for row in series:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return compact_series_columns({key: [row.get(key) for row in series] for key in keys})


def lateral_group_key(sample: dict[str, Any]) -> tuple[Any, ...]:
    return (
        sample.get("obs_type"),
        sample.get("P0"),
        sample.get("p0"),
        sample.get("c"),
        sample.get("f_T"),
        sample.get("t_stat"),
    )


def weighted_mean(values: list[tuple[Any, float]]) -> float | None:
    total_weight = 0.0
    total = 0.0
    for value, weight in values:
        value_float = finite_float(value)
        if value_float is None or weight <= 0:
            continue
        total += value_float * weight
        total_weight += weight
    return total / total_weight if total_weight > 0 else None


def aggregate_lateral_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if isinstance(sample, dict):
            groups[lateral_group_key(sample)].append(sample)

    aggregated: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda x: tuple("" if v is None else str(v) for v in x)):
        items = groups[key]
        expanded_items: list[tuple[dict[str, Any], dict[str, list[Any]], int, float]] = []
        for item in items:
            n_samples = int(item.get("N_samples", item.get("n_samples", 1)) or 1)
            columns = series_to_columns(item.get("series"), item.get("series_length"))
            n_rows = infer_series_length(columns)
            if columns and n_rows > 0 and n_samples > 0:
                expanded_items.append((item, columns, n_rows, float(n_samples)))
        if not expanded_items:
            continue

        common_len = min(n_rows for _, _, n_rows, _ in expanded_items)
        series_keys: list[str] = []
        seen: set[str] = set()
        for _, columns, _, _ in expanded_items:
            for series_key in columns:
                if series_key not in seen:
                    series_keys.append(series_key)
                    seen.add(series_key)

        averaged_columns: dict[str, list[Any]] = {}
        for series_key in series_keys:
            if series_key == "t":
                averaged_columns[series_key] = expanded_items[0][1].get(series_key, [])[:common_len]
                continue
            values_out: list[Any] = []
            for idx in range(common_len):
                values_out.append(weighted_mean([
                    (columns.get(series_key, [None] * common_len)[idx], weight)
                    for _, columns, _, weight in expanded_items
                    if idx < len(columns.get(series_key, []))
                ]))
            averaged_columns[series_key] = values_out

        first = expanded_items[0][0]
        n_samples_total = int(sum(weight for _, _, _, weight in expanded_items))
        n_rows_total = int(sum(int(item.get("n_rows", 0) or 0) for item, _, _, _ in expanded_items))
        obs_type = first.get("obs_type")
        aggregated_sample = {
            "obs_type": obs_type,
            "P0": first.get("P0"),
            "p0": first.get("p0"),
            "c": first.get("c"),
            "f_T": first.get("f_T"),
            "t_stat": first.get("t_stat"),
            "N_samples": n_samples_total,
            "n_rows": n_rows_total,
            "series": compact_series_columns(averaged_columns),
            "series_length": common_len,
            "series_kind": f"{obs_type}_mean" if obs_type else "lateral_mean",
        }
        if n_samples_total == 1:
            for meta_key in ("filename", "sample_id", "seed"):
                if first.get(meta_key) is not None:
                    aggregated_sample[meta_key] = first.get(meta_key)
        aggregated.append(aggregated_sample)
    return aggregated


def merge_lateral_bundles(existing_bundle: dict[str, Any] | None, new_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_samples: list[dict[str, Any]] = []
    if isinstance(existing_bundle, dict):
        raw_samples = existing_bundle.get("samples", [])
        if isinstance(raw_samples, list):
            existing_samples = [sample for sample in raw_samples if isinstance(sample, dict)]
    return aggregate_lateral_samples(existing_samples + new_samples)


def convert_lateral_bundle_to_columnar(bundle: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    changed = False
    samples = bundle.get("samples", [])
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            series = sample.get("series")
            if isinstance(series, list):
                series_rows = [
                    row for row in series if isinstance(row, dict)
                ]
                sample["series"] = series_rows_to_columns(series_rows)
                if sample.get("series_length") != len(series_rows):
                    sample["series_length"] = len(series_rows)
                changed = True
            elif isinstance(series, dict):
                compacted = compact_series_columns(series)
                if compacted != series:
                    sample["series"] = compacted
                    changed = True
            inferred_length = infer_series_length(sample.get("series"))
            if inferred_length and sample.get("series_length") != inferred_length:
                sample["series_length"] = inferred_length
                changed = True
        aggregated_samples = aggregate_lateral_samples([
            sample for sample in samples if isinstance(sample, dict)
        ])
        if aggregated_samples and aggregated_samples != samples:
            bundle["samples"] = aggregated_samples
            changed = True
    meta = bundle.setdefault("meta", {})
    if isinstance(meta, dict) and meta.get("format") != "compact_summary_columnar":
        meta["format"] = "compact_summary_columnar"
        changed = True
    if isinstance(meta, dict) and meta.get("aggregation") != "mean_by_parameter":
        meta["aggregation"] = "mean_by_parameter"
        changed = True
    if isinstance(meta, dict) and meta.get("lateral_processing_version") != LATERAL_PROCESSING_VERSION:
        meta["lateral_processing_version"] = LATERAL_PROCESSING_VERSION
        changed = True
    return bundle, changed


def process_correlation_files(sample_paths: list[Path]) -> tuple[list[dict[str, Any]], list[float]]:
    """Process CSV files from a correlations/ directory into compact summaries for analysis."""
    rows: list[dict[str, Any]] = []
    counts: list[float] = []

    for sample_path in sample_paths:
        try:
            with sample_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                records = list(reader)
        except Exception as exc:
            print(f"[warn] ignorando CSV inválido {sample_path}: {exc}")
            continue

        if not records:
            continue

        obs_type = "correlation" if "lateral_correlation_time" in sample_path.name else "susceptibility"
        counts.append(float(len(records)))

        summary: dict[str, Any] = {
            "filename": sample_path.name,
            "obs_type": obs_type,
            "n_rows": len(records),
            "sample_id": None,
            "t_stat": None,
            "p0": None,
            "P0": None,
            "c": None,
            "f_T": None,
            "seed": None,
        }

        by_t: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for item in records:
            sample_id = item.get("sample_id")
            if summary["sample_id"] is None and sample_id:
                summary["sample_id"] = sample_id
            if summary["t_stat"] is None:
                summary["t_stat"] = finite_float(item.get("t_stat"))
            if summary["p0"] is None:
                summary["p0"] = finite_float(item.get("p0"))
            if summary["P0"] is None:
                summary["P0"] = finite_float(item.get("P0"))
            if summary["c"] is None:
                summary["c"] = finite_float(item.get("c"))
            if summary["f_T"] is None:
                summary["f_T"] = finite_float(item.get("f_T"))
            if summary["seed"] is None:
                summary["seed"] = finite_float(item.get("seed"))

            t_value = finite_float(item.get("t"))
            if t_value is None:
                continue
            by_t[t_value].append(item)

        if obs_type == "correlation":
            series: list[dict[str, Any]] = []
            if records and "C_norm_mean" in records[0]:
                for item in records:
                    t_value = finite_float(item.get("t"))
                    if t_value is None:
                        continue
                    series.append({
                        "t": float(t_value),
                        "n_rows": int(finite_float(item.get("n_rows")) or 0),
                        "C_norm_mean": finite_float(item.get("C_norm_mean")),
                        "C_norm_std": finite_float(item.get("C_norm_std")),
                        "C_norm_absmax": finite_float(item.get("C_norm_absmax")),
                        "r_at_absmax": finite_float(item.get("r_at_absmax")),
                        "valid_norm_mean": finite_float(item.get("valid_norm_mean")),
                        "pair_count_mean": finite_float(item.get("pair_count_mean")),
                    })
            else:
                for t_value in sorted(by_t):
                    chunk = by_t[t_value]
                    c_values = [finite_float(x.get("C_norm")) for x in chunk]
                    c_values = [v for v in c_values if v is not None]
                    r_values = [finite_float(x.get("r")) for x in chunk]
                    r_values = [v for v in r_values if v is not None]
                    valid_values = [finite_float(x.get("valid_norm")) for x in chunk]
                    valid_values = [v for v in valid_values if v is not None]
                    pair_values = [finite_float(x.get("pair_count")) for x in chunk]
                    pair_values = [v for v in pair_values if v is not None]
                    if not c_values:
                        continue
                    abs_vals = [abs(v) for v in c_values]
                    best_idx = int(np.argmax(abs_vals)) if abs_vals else 0
                    series.append({
                        "t": float(t_value),
                        "n_rows": len(chunk),
                        "C_norm_mean": float(np.mean(c_values)) if c_values else None,
                        "C_norm_std": float(np.std(c_values, ddof=1)) if len(c_values) > 1 else 0.0,
                        "C_norm_absmax": float(np.max(abs_vals)) if abs_vals else None,
                        "r_at_absmax": float(r_values[best_idx]) if best_idx < len(r_values) else None,
                        "valid_norm_mean": float(np.mean(valid_values)) if valid_values else None,
                        "pair_count_mean": float(np.mean(pair_values)) if pair_values else None,
                    })
            summary["series"] = series_rows_to_columns(series)
            summary["series_length"] = len(series)
            summary["series_kind"] = "correlation_summary"
        else:
            series: list[dict[str, Any]] = []
            for t_value in sorted(by_t):
                chunk = by_t[t_value]
                chi_incl = [finite_float(x.get("chi_norm_incl0")) for x in chunk]
                chi_incl = [v for v in chi_incl if v is not None]
                chi_excl = [finite_float(x.get("chi_norm_excl0")) for x in chunk]
                chi_excl = [v for v in chi_excl if v is not None]
                r_values = [finite_float(x.get("r_max")) for x in chunk]
                r_values = [v for v in r_values if v is not None]
                valid_values = [finite_float(x.get("n_valid_norm")) for x in chunk]
                valid_values = [v for v in valid_values if v is not None]
                series.append({
                    "t": float(t_value),
                    "n_rows": len(chunk),
                    "chi_norm_incl0_mean": float(np.mean(chi_incl)) if chi_incl else None,
                    "chi_norm_excl0_mean": float(np.mean(chi_excl)) if chi_excl else None,
                    "r_max_mean": float(np.mean(r_values)) if r_values else None,
                    "n_valid_norm_mean": float(np.mean(valid_values)) if valid_values else None,
                })
            summary["series"] = series_rows_to_columns(series)
            summary["series_length"] = len(series)
            summary["series_kind"] = "susceptibility_summary"

        rows.append(summary)

    return rows, counts


def process_lateral_correlations(
    data_dir: Path,
    out_dir: Path,
    sample_paths: list[Path] | None = None,
    existing_bundle: dict[str, Any] | None = None,
) -> Path | None:
    """Disabled: lateral correlation/susceptibility processing is no longer used."""
    del data_dir, out_dir, sample_paths, existing_bundle
    return None


def lateral_correlations_dir(data_dir: Path) -> Path:
    return data_dir.parent / "correlations"


def lateral_bundle_path(out_dir: Path) -> Path:
    return out_dir / "lateral_correlations_bundle.json.xz"


def gzip_lateral_bundle_path(out_dir: Path) -> Path:
    return out_dir / "lateral_correlations_bundle.json.gz"


def legacy_lateral_bundle_path(out_dir: Path) -> Path:
    return out_dir / "lateral_correlations_bundle.json"


def existing_lateral_bundle_path(out_dir: Path) -> Path | None:
    compressed_path = lateral_bundle_path(out_dir)
    if compressed_path.exists():
        return compressed_path
    gzip_path = gzip_lateral_bundle_path(out_dir)
    if gzip_path.exists():
        return gzip_path
    legacy_path = legacy_lateral_bundle_path(out_dir)
    if legacy_path.exists():
        return legacy_path
    return None


def remove_legacy_lateral_bundles(out_dir: Path, keep: Path) -> None:
    for path in (gzip_lateral_bundle_path(out_dir), legacy_lateral_bundle_path(out_dir)):
        if path != keep and path.exists():
            path.unlink()


def ensure_lateral_bundle_compressed(
    out_dir: Path,
    *,
    compresslevel: int = 6,
    threads: int = 1,
) -> Path | None:
    compressed_path = lateral_bundle_path(out_dir)
    if compressed_path.exists():
        return compressed_path
    source_path = gzip_lateral_bundle_path(out_dir)
    if not source_path.exists():
        source_path = legacy_lateral_bundle_path(out_dir)
    if not source_path.exists():
        return None
    compress_json_to_xz(source_path, compressed_path, compresslevel=compresslevel, threads=threads)
    remove_legacy_lateral_bundles(out_dir, keep=compressed_path)
    return compressed_path


def load_lateral_bundle_file(path: Path) -> dict[str, Any] | None:
    try:
        data = load_json_bundle(path)
    except Exception as exc:
        print(f"[warn] não consegui ler bundle lateral {path}: {exc}")
        return None
    return data if isinstance(data, dict) else None


def write_lateral_bundle_file(bundle: dict[str, Any], out_dir: Path) -> Path:
    compressed_path = lateral_bundle_path(out_dir)
    with open_text_auto(compressed_path, "wt", compresslevel=6) as handle:
        json.dump(json_safe(bundle), handle, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
    remove_legacy_lateral_bundles(out_dir, keep=compressed_path)
    return compressed_path


def ensure_lateral_bundle_columnar(out_dir: Path) -> tuple[Path | None, bool]:
    bundle_path = ensure_lateral_bundle_compressed(out_dir) or existing_lateral_bundle_path(out_dir)
    if bundle_path is None or not bundle_path.exists():
        return bundle_path, False
    bundle = load_lateral_bundle_file(bundle_path)
    if bundle is None:
        return bundle_path, False
    bundle, changed = convert_lateral_bundle_to_columnar(bundle)
    if changed or bundle_path.suffix != ".gz":
        return write_lateral_bundle_file(bundle, out_dir), changed
    return bundle_path, False


def collect_lateral_correlation_files(data_dir: Path) -> list[Path]:
    correlations_dir = lateral_correlations_dir(data_dir)
    if not correlations_dir.is_dir():
        return []
    return sorted(path for path in correlations_dir.glob("*.csv") if path.is_file())


def lateral_csv_fingerprints(data_dir: Path) -> dict[str, str]:
    correlations_dir = lateral_correlations_dir(data_dir)
    out: dict[str, str] = {}
    for path in collect_lateral_correlation_files(data_dir):
        out[path.relative_to(correlations_dir).as_posix()] = file_stat_fingerprint(path)
    return dict(sorted(out.items()))


def discover_data_dirs(raw_root: Path) -> list[Path]:
    return sorted(
        p for p in raw_root.rglob("data")
        if p.is_dir() and parse_data_dir(p) is not None
    )


def collect_group_json_files(data_dir: Path) -> list[Path]:
    return sorted(
        path for path in data_dir.glob("*.json")
        if path.is_file() and parse_sample_name(path) is not None
    )


def rows_from_bundle(bundle: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    meta = bundle.get("meta", {}) if isinstance(bundle.get("meta", {}), dict) else {}
    params = {
        "type_perc": meta.get("type_perc"),
        "dim": meta.get("dim"),
        "L": meta.get("L"),
        "f_T": meta.get("f_T"),
        "c": meta.get("c"),
        "nc": meta.get("nc"),
        "rho": meta.get("rho"),
        "stat_window": meta.get("stat_window", -1),
        "stop_criterion": meta.get("stop_criterion"),
        "t_eq_validation": meta.get("t_eq_validation"),
        "t_eq_s_prime_threshold": meta.get("t_eq_s_prime_threshold"),
        "equilibrium_effective_rel_tol": meta.get("equilibrium_effective_rel_tol"),
        "post_equilibrium_extra_steps": meta.get("post_equilibrium_extra_steps"),
    }

    all_rows: list[dict[str, Any]] = []
    all_color_rows: list[dict[str, Any]] = []
    for p0_group in bundle.get("p0_groups", []):
        if not isinstance(p0_group, dict):
            continue
        P0 = p0_group.get("P0_value")
        p0 = p0_group.get("p0_value")
        processed = p0_group.get("num_samples_total", 0)
        colors = p0_group.get("colors", {})
        if not isinstance(colors, dict):
            colors = {}

        all_color_rows.append({
            **params,
            "num_colors": params["nc"],
            "p0": p0,
            "P0": P0,
            "N_samples": processed,
            "nc": colors.get("nc"),
            "nc_err": colors.get("nc_err"),
            "nc_std": colors.get("nc_std"),
        })

        for order_block in p0_group.get("orders", []):
            if not isinstance(order_block, dict):
                continue
            p_stats = order_block.get("p", {})
            f_stats = order_block.get("f", {})
            z_stats = order_block.get("z_max", {})
            z_stat_stats = order_block.get("z_stat", {})
            if not isinstance(p_stats, dict):
                p_stats = {}
            if not isinstance(f_stats, dict):
                f_stats = {}
            if not isinstance(z_stats, dict):
                z_stats = {}
            if not isinstance(z_stat_stats, dict):
                z_stat_stats = {}

            all_rows.append({
                **params,
                "p0": p0,
                "P0": P0,
                "order": order_block.get("order"),
                "N_samples": order_block.get("N_samples", processed),
                "N_samples_perc": order_block.get("N_samples_perc"),
                "p_mean": p_stats.get("mean"),
                "p_err": p_stats.get("err"),
                "f_mean": f_stats.get("mean"),
                "f_err": f_stats.get("err"),
                "z_max_mean": z_stats.get("mean"),
                "z_max_err": z_stats.get("err"),
                "z_stat_mean": z_stat_stats.get("mean"),
                "z_stat_err": z_stat_stats.get("err"),
            })

    return all_rows, all_color_rows


def rows_from_existing_bundle(bundle_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bundle = load_json_bundle(bundle_path)
    return rows_from_bundle(bundle)


def dynamic_bundle_path(out_dir: Path) -> Path:
    return out_dir / "properties_dynamic_bundle.json.xz"


def gzip_dynamic_bundle_path(out_dir: Path) -> Path:
    return out_dir / "properties_dynamic_bundle.json.gz"


def legacy_dynamic_bundle_path(out_dir: Path) -> Path:
    return out_dir / "properties_dynamic_bundle.json"


def existing_dynamic_bundle_path(out_dir: Path) -> Path | None:
    for path in (
        dynamic_bundle_path(out_dir),
        gzip_dynamic_bundle_path(out_dir),
        legacy_dynamic_bundle_path(out_dir),
    ):
        if path.exists():
            return path
    return None


def remove_legacy_dynamic_bundles(out_dir: Path, keep: Path) -> None:
    for path in (gzip_dynamic_bundle_path(out_dir), legacy_dynamic_bundle_path(out_dir)):
        if path != keep and path.exists():
            path.unlink()


def ensure_dynamic_bundle_compressed(
    out_dir: Path,
    *,
    compresslevel: int = 6,
    threads: int = 1,
) -> Path | None:
    compressed_path = dynamic_bundle_path(out_dir)
    if compressed_path.exists():
        return compressed_path
    source_path = gzip_dynamic_bundle_path(out_dir)
    if not source_path.exists():
        source_path = legacy_dynamic_bundle_path(out_dir)
    if not source_path.exists():
        return None
    compress_json_to_xz(source_path, compressed_path, compresslevel=compresslevel, threads=threads)
    remove_legacy_dynamic_bundles(out_dir, keep=compressed_path)
    return compressed_path


def bundle_has_missing_dynamic_series(bundle: dict[str, Any]) -> bool:
    meta = bundle.get("meta", {}) if isinstance(bundle.get("meta", {}), dict) else {}
    if meta.get("series_mode") not in (None, "full"):
        return False
    p0_groups = bundle.get("p0_groups", [])
    if not isinstance(p0_groups, list):
        return False
    for p0_group in p0_groups:
        if not isinstance(p0_group, dict):
            continue
        orders = p0_group.get("orders", [])
        if not isinstance(orders, list):
            continue
        for order in orders:
            if not isinstance(order, dict):
                continue
            data = order.get("data", {}) if isinstance(order.get("data", {}), dict) else {}
            n_samples = int(order.get("N_samples_perc", data.get("n_samples_perc", 0)) or 0)
            if n_samples <= 0:
                continue
            if not data.get("time") or not data.get("pt_mean"):
                return True
            if "pt_N_per_t" not in data:
                return True
    return False


def summary_from_values(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if finite_float(v) is not None]
    mean, err, std, n = mean_sem_std(clean)
    arr = np.asarray(clean, dtype=float)
    if n > 0:
        q25, median, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
        v_min = float(np.min(arr))
        v_max = float(np.max(arr))
    else:
        q25 = median = q75 = v_min = v_max = math.nan
    return {
        "mean": mean,
        "err": err,
        "std": std,
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "min": v_min,
        "max": v_max,
        "n": n,
        "sum": float(sum(clean)) if clean else 0.0,
        "sumsq": float(sum(v * v for v in clean)) if clean else 0.0,
    }


def summary_with_values(values: list[float]) -> dict[str, Any]:
    summary = summary_from_values(values)
    summary["values"] = list(values)
    return summary


def summary_sum(summary: dict[str, Any]) -> float:
    value = finite_float(summary.get("sum"))
    if value is not None:
        return value
    n = int(summary.get("n", 0) or 0)
    mean = finite_float(summary.get("mean"))
    return float(n * mean) if mean is not None else 0.0


def summary_sumsq(summary: dict[str, Any]) -> float:
    value = finite_float(summary.get("sumsq"))
    if value is not None:
        return value
    n = int(summary.get("n", 0) or 0)
    mean = finite_float(summary.get("mean"))
    err = finite_float(summary.get("err"))
    if n <= 0 or mean is None:
        return 0.0
    if n <= 1:
        return float(mean * mean)
    std = finite_float(summary.get("std"))
    variance = float(std * std) if std is not None else (0.0 if err is None else float(err * err * n))
    return float((n - 1) * variance + n * mean * mean)


def finalize_summary(n: int, total: float, sumsq: float) -> dict[str, Any]:
    if n <= 0:
        return {"mean": None, "err": None, "std": None, "n": 0, "sum": 0.0, "sumsq": 0.0}
    mean = total / n
    if n <= 1:
        std = 0.0
        err = 0.0
    else:
        variance = max((sumsq - total * total / n) / (n - 1), 0.0)
        std = math.sqrt(variance)
        err = std / math.sqrt(n)
    return {
        "mean": float(mean),
        "err": float(err),
        "std": float(std),
        "n": int(n),
        "sum": float(total),
        "sumsq": float(sumsq),
    }


def combine_summary_dicts(old_summary: dict[str, Any], new_summary: dict[str, Any]) -> dict[str, Any]:
    old_n = int(old_summary.get("n", 0) or 0)
    new_n = int(new_summary.get("n", 0) or 0)
    combined = finalize_summary(
        old_n + new_n,
        summary_sum(old_summary) + summary_sum(new_summary),
        summary_sumsq(old_summary) + summary_sumsq(new_summary),
    )
    old_values = old_summary.get("values", [])
    new_values = new_summary.get("values", [])
    values: list[Any] = []
    if isinstance(old_values, list):
        values.extend(old_values)
    if isinstance(new_values, list):
        values.extend(new_values)
    if values:
        combined["values"] = values
    return combined


def colors_as_summary(colors: dict[str, Any]) -> dict[str, Any]:
    values = colors.get("values", []) if isinstance(colors.get("values", []), list) else []
    if values:
        return summary_from_values(values)
    return {
        "mean": colors.get("nc"),
        "err": colors.get("nc_err"),
        "std": colors.get("nc_std"),
        "n": int(colors.get("n", colors.get("Nsamples", 0)) or 0),
        "sum": colors.get("sum"),
        "sumsq": colors.get("sumsq"),
    }


def summary_as_colors(summary: dict[str, Any], n_samples_total: int) -> dict[str, Any]:
    return {
        "Nsamples": n_samples_total,
        "nc": summary.get("mean"),
        "nc_err": summary.get("err"),
        "nc_std": summary.get("std"),
        "n": summary.get("n", 0),
        "sum": summary.get("sum", 0.0),
        "sumsq": summary.get("sumsq", 0.0),
    }


def combine_scalar_summary(
    old_mean: float | None,
    old_err: float | None,
    old_n: int,
    new_mean: float | None,
    new_err: float | None,
    new_n: int,
) -> tuple[float | None, float | None, int]:
    if new_mean is None and old_mean is None:
        return None, None, 0
    if old_mean is None:
        return new_mean, new_err, new_n
    if new_mean is None:
        return old_mean, old_err, old_n
    total_n = old_n + new_n
    if total_n <= 0:
        return None, None, 0
    combined = combine_summary_dicts(
        {"mean": old_mean, "err": old_err, "n": old_n},
        {"mean": new_mean, "err": new_err, "n": new_n},
    )
    return combined["mean"], combined["err"], int(combined["n"])


def series_counts(counts: Any, fallback_n: int, length: int) -> list[int]:
    if isinstance(counts, list) and counts:
        out: list[int] = []
        for value in counts[:length]:
            try:
                n = int(value)
            except Exception:
                n = 0
            out.append(max(n, 0))
        if len(out) < length:
            out.extend([0] * (length - len(out)))
        return out
    return [max(int(fallback_n), 0)] * length


def combine_series_arrays(
    old_mean: list[float] | None,
    old_std: list[float] | None,
    old_counts: list[int] | None,
    old_n: int,
    new_mean: list[float] | None,
    new_std: list[float] | None,
    new_counts: list[int] | None,
    new_n: int,
) -> tuple[list[float], list[float], list[float], list[int]]:
    old_mean = list(old_mean or [])
    old_std = list(old_std or [])
    new_mean = list(new_mean or [])
    new_std = list(new_std or [])
    length = max(len(old_mean), len(new_mean))
    if length == 0:
        return [], [], [], []

    old_counts_clean = series_counts(old_counts, old_n, len(old_mean))
    new_counts_clean = series_counts(new_counts, new_n, len(new_mean))
    combined_mean: list[float] = []
    combined_std: list[float] = []
    combined_sem: list[float] = []
    combined_counts: list[int] = []

    for idx in range(length):
        total = 0.0
        sumsq = 0.0
        count = 0
        for means, stds, counts in (
            (old_mean, old_std, old_counts_clean),
            (new_mean, new_std, new_counts_clean),
        ):
            if idx >= len(means) or idx >= len(counts):
                continue
            n = int(counts[idx])
            value = finite_float(means[idx])
            if n <= 0 or value is None:
                continue
            std_value = finite_float(stds[idx]) if idx < len(stds) else 0.0
            std_value = 0.0 if std_value is None else std_value
            total += n * value
            sumsq += n * value * value
            if n > 1:
                sumsq += (n - 1) * std_value * std_value
            count += n

        combined_counts.append(count)
        if count <= 0:
            combined_mean.append(math.nan)
            combined_std.append(math.nan)
            combined_sem.append(math.nan)
            continue
        mean = total / count
        if count <= 1:
            std = 0.0
            sem = 0.0
        else:
            variance = max((sumsq - total * total / count) / (count - 1), 0.0)
            std = math.sqrt(variance)
            sem = std / math.sqrt(count)
        combined_mean.append(float(mean))
        combined_std.append(float(std))
        combined_sem.append(float(sem))

    return combined_mean, combined_std, combined_sem, combined_counts


def choose_axis(old_axis: Any, new_axis: Any, length: int) -> list[Any]:
    old_list = list(old_axis or [])
    new_list = list(new_axis or [])
    axis = new_list if len(new_list) > len(old_list) else old_list
    if len(axis) >= length:
        return axis[:length]
    if axis and all(finite_float(v) is not None for v in axis):
        step = float(axis[-1]) - float(axis[-2]) if len(axis) > 1 else 1.0
        while len(axis) < length:
            axis.append(float(axis[-1]) + step)
        return axis
    return list(range(length))


def series_common_fields(axis: list[Any], mean: list[float], std: list[float], sem: list[float],
                         counts: list[int], total_n: int) -> tuple[list[Any], list[float], list[float], list[float], list[int]]:
    idx = [i for i, n in enumerate(counts) if n == total_n and i < len(axis)]
    return (
        [axis[i] for i in idx],
        [mean[i] for i in idx],
        [std[i] for i in idx],
        [sem[i] for i in idx],
        [counts[i] for i in idx],
    )


def series_supported_fields(axis: list[Any], mean: list[float], std: list[float], sem: list[float],
                            counts: list[int], total_n: int) -> tuple[list[Any], list[float], list[float], list[float], list[int], int]:
    threshold = min_support_count(total_n)
    idx = [i for i, n in enumerate(counts) if n >= threshold and i < len(axis)]
    return (
        [axis[i] for i in idx],
        [mean[i] for i in idx],
        [std[i] for i in idx],
        [sem[i] for i in idx],
        [counts[i] for i in idx],
        threshold,
    )


def merge_order_block(existing_order: dict[str, Any], new_order: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing_order)
    merged["N_samples"] = int(existing_order.get("N_samples", 0) or 0) + int(new_order.get("N_samples", 0) or 0)
    merged["N_samples_perc"] = int(existing_order.get("N_samples_perc", 0) or 0) + int(new_order.get("N_samples_perc", 0) or 0)

    existing_data = existing_order.get("data", {}) if isinstance(existing_order.get("data", {}), dict) else {}
    new_data = new_order.get("data", {}) if isinstance(new_order.get("data", {}), dict) else {}
    merged_data = dict(existing_data)

    old_n_pt = int(existing_data.get("n_seeds_pt", 0) or 0)
    new_n_pt = int(new_data.get("n_seeds_pt", 0) or 0)
    pt_mean, pt_std, pt_sem, pt_counts = combine_series_arrays(
        existing_data.get("pt_mean"),
        existing_data.get("pt_std"),
        existing_data.get("pt_N_per_t"),
        old_n_pt,
        new_data.get("pt_mean"),
        new_data.get("pt_std"),
        new_data.get("pt_N_per_t"),
        new_n_pt,
    )
    time = choose_axis(existing_data.get("time"), new_data.get("time"), len(pt_mean))
    merged_data["pt_mean"] = pt_mean
    merged_data["pt_std"] = pt_std
    merged_data["pt_sem"] = pt_sem
    merged_data["pt_N_per_t"] = pt_counts
    merged_data["n_seeds_pt"] = old_n_pt + new_n_pt
    merged_data["time"] = time
    common_time, common_mean, common_std, common_sem, common_counts = series_common_fields(
        time, pt_mean, pt_std, pt_sem, pt_counts, old_n_pt + new_n_pt
    )
    supported_time, supported_mean, supported_std, supported_sem, supported_counts, supported_threshold = series_supported_fields(
        time, pt_mean, pt_std, pt_sem, pt_counts, old_n_pt + new_n_pt
    )
    merged_data["pt_common_time"] = common_time
    merged_data["pt_common_mean"] = common_mean
    merged_data["pt_common_std"] = common_std
    merged_data["pt_common_sem"] = common_sem
    merged_data["pt_common_N_per_t"] = common_counts
    merged_data["pt_supported_time"] = supported_time
    merged_data["pt_supported_mean"] = supported_mean
    merged_data["pt_supported_std"] = supported_std
    merged_data["pt_supported_sem"] = supported_sem
    merged_data["pt_supported_N_per_t"] = supported_counts
    merged_data["pt_support_policy"] = "union_observed_times"
    merged_data["pt_common_support_policy"] = "all_samples_present"
    merged_data["pt_supported_support_policy"] = "min_fraction_of_samples_present"
    merged_data["pt_min_support_count"] = supported_threshold
    merged_data["min_support_fraction"] = DEFAULT_MIN_SUPPORT_FRACTION

    old_n_ft = int(existing_data.get("n_seeds_ft", 0) or 0)
    new_n_ft = int(new_data.get("n_seeds_ft", 0) or 0)
    if old_n_ft > 0 or new_n_ft > 0:
        ft_mean, ft_std, ft_sem, ft_counts = combine_series_arrays(
            existing_data.get("ft_mean"),
            existing_data.get("ft_std"),
            existing_data.get("ft_N_per_t"),
            old_n_ft,
            new_data.get("ft_mean"),
            new_data.get("ft_std"),
            new_data.get("ft_N_per_t"),
            new_n_ft,
        )
        ft_time = choose_axis(existing_data.get("ft_time") or existing_data.get("time"),
                              new_data.get("ft_time") or new_data.get("time"),
                              len(ft_mean))
        common_time, common_mean, common_std, common_sem, common_counts = series_common_fields(
            ft_time, ft_mean, ft_std, ft_sem, ft_counts, old_n_ft + new_n_ft
        )
        supported_time, supported_mean, supported_std, supported_sem, supported_counts, supported_threshold = series_supported_fields(
            ft_time, ft_mean, ft_std, ft_sem, ft_counts, old_n_ft + new_n_ft
        )
        merged_data["ft_time"] = ft_time
        merged_data["ft_mean"] = ft_mean
        merged_data["ft_std"] = ft_std
        merged_data["ft_sem"] = ft_sem
        merged_data["ft_N_per_t"] = ft_counts
        merged_data["n_seeds_ft"] = old_n_ft + new_n_ft
        merged_data["ft_common_time"] = common_time
        merged_data["ft_common_mean"] = common_mean
        merged_data["ft_common_std"] = common_std
        merged_data["ft_common_sem"] = common_sem
        merged_data["ft_common_N_per_t"] = common_counts
        merged_data["ft_supported_time"] = supported_time
        merged_data["ft_supported_mean"] = supported_mean
        merged_data["ft_supported_std"] = supported_std
        merged_data["ft_supported_sem"] = supported_sem
        merged_data["ft_supported_N_per_t"] = supported_counts
        merged_data["ft_support_policy"] = "union_observed_times"
        merged_data["ft_common_support_policy"] = "all_samples_present"
        merged_data["ft_supported_support_policy"] = "min_fraction_of_samples_present"
        merged_data["ft_min_support_count"] = supported_threshold

    old_n_flz = int(existing_data.get("n_seeds_fL_z", 0) or 0)
    new_n_flz = int(new_data.get("n_seeds_fL_z", 0) or 0)
    if old_n_flz > 0 or new_n_flz > 0:
        flz_mean, flz_std, flz_sem, flz_counts = combine_series_arrays(
            existing_data.get("fL_z_mean"),
            existing_data.get("fL_z_std"),
            existing_data.get("fL_z_N_per_z"),
            old_n_flz,
            new_data.get("fL_z_mean"),
            new_data.get("fL_z_std"),
            new_data.get("fL_z_N_per_z"),
            new_n_flz,
        )
        z_axis = choose_axis(existing_data.get("fL_z_z"), new_data.get("fL_z_z"), len(flz_mean))
        common_z, common_mean, common_std, common_sem, common_counts = series_common_fields(
            z_axis, flz_mean, flz_std, flz_sem, flz_counts, old_n_flz + new_n_flz
        )
        supported_z, supported_mean, supported_std, supported_sem, supported_counts, supported_threshold = series_supported_fields(
            z_axis, flz_mean, flz_std, flz_sem, flz_counts, old_n_flz + new_n_flz
        )
        merged_data["fL_z_z"] = z_axis
        merged_data["fL_z_mean"] = flz_mean
        merged_data["fL_z_std"] = flz_std
        merged_data["fL_z_sem"] = flz_sem
        merged_data["fL_z_N_per_z"] = flz_counts
        merged_data["n_seeds_fL_z"] = old_n_flz + new_n_flz
        merged_data["fL_z_common_z"] = common_z
        merged_data["fL_z_common_mean"] = common_mean
        merged_data["fL_z_common_std"] = common_std
        merged_data["fL_z_common_sem"] = common_sem
        merged_data["fL_z_common_N_per_z"] = common_counts
        merged_data["fL_z_supported_z"] = supported_z
        merged_data["fL_z_supported_mean"] = supported_mean
        merged_data["fL_z_supported_std"] = supported_std
        merged_data["fL_z_supported_sem"] = supported_sem
        merged_data["fL_z_supported_N_per_z"] = supported_counts
        merged_data["fL_z_support_policy"] = "union_observed_heights"
        merged_data["fL_z_common_support_policy"] = "all_samples_present"
        merged_data["fL_z_supported_support_policy"] = "min_fraction_of_samples_present"
        merged_data["fL_z_min_support_count"] = supported_threshold

    p_old = existing_order.get("p", {}) if isinstance(existing_order.get("p", {}), dict) else {}
    p_new = new_order.get("p", {}) if isinstance(new_order.get("p", {}), dict) else {}
    merged["p"] = combine_summary_dicts(p_old, p_new)
    p_mean = merged["p"]["mean"]
    p_err = merged["p"]["err"]
    merged_data["p_tail_mean"] = p_mean
    merged_data["p_tail_err"] = p_err
    merged_data["p_tail_sample_values"] = merged["p"].get("values", [])
    merged_data["p_tail_estimator"] = "mean_of_per_sample_tail_means_after_each_sample_t_eq"

    f_old = existing_order.get("f", {}) if isinstance(existing_order.get("f", {}), dict) else {}
    f_new = new_order.get("f", {}) if isinstance(new_order.get("f", {}), dict) else {}
    merged["f"] = combine_summary_dicts(f_old, f_new)
    f_mean = merged["f"]["mean"]
    f_err = merged["f"]["err"]
    merged_data["f_tail_mean"] = f_mean
    merged_data["f_tail_err"] = f_err
    merged_data["f_tail_sample_values"] = merged["f"].get("values", [])
    merged_data["f_tail_estimator"] = "mean_of_per_sample_tail_means_after_each_sample_t_eq"

    teq_old = existing_order.get("t_eq_species", {}) if isinstance(existing_order.get("t_eq_species", {}), dict) else {}
    teq_new = new_order.get("t_eq_species", {}) if isinstance(new_order.get("t_eq_species", {}), dict) else {}
    merged["t_eq_species"] = combine_summary_dicts(teq_old, teq_new)
    merged_data["t_eq_mean"] = merged["t_eq_species"]["mean"]
    merged_data["n_t_eq"] = merged["t_eq_species"]["n"]
    old_min = finite_float(existing_data.get("t_eq_min"))
    new_min = finite_float(new_data.get("t_eq_min"))
    old_max = finite_float(existing_data.get("t_eq_max"))
    new_max = finite_float(new_data.get("t_eq_max"))
    mins = [v for v in (old_min, new_min) if v is not None]
    maxs = [v for v in (old_max, new_max) if v is not None]
    if mins:
        merged_data["t_eq_min"] = min(mins)
    if maxs:
        merged_data["t_eq_max"] = max(maxs)
        merged_data["t_eq"] = max(maxs)

    z_old = existing_order.get("z_max", {}) if isinstance(existing_order.get("z_max", {}), dict) else {}
    z_new = new_order.get("z_max", {}) if isinstance(new_order.get("z_max", {}), dict) else {}
    merged["z_max"] = combine_summary_dicts(z_old, z_new)
    z_mean = merged["z_max"]["mean"]
    z_err = merged["z_max"]["err"]
    merged_data["z_max_mean"] = z_mean
    merged_data["z_max_err"] = z_err
    merged_data["z_max_std"] = merged["z_max"].get("std")
    merged_data["z_max_values"] = merged["z_max"].get("values", [])

    z_stat_old = existing_order.get("z_stat", {}) if isinstance(existing_order.get("z_stat", {}), dict) else {}
    z_stat_new = new_order.get("z_stat", {}) if isinstance(new_order.get("z_stat", {}), dict) else {}
    merged["z_stat"] = combine_summary_dicts(z_stat_old, z_stat_new)
    z_stat_mean = merged["z_stat"]["mean"]
    z_stat_err = merged["z_stat"]["err"]
    merged_data["z_stat_mean"] = z_stat_mean
    merged_data["z_stat_err"] = z_stat_err
    merged_data["n_samples_perc"] = merged["N_samples_perc"]
    merged_data["n_samples_total"] = merged["N_samples"]

    merged["samples"] = []
    for sample in existing_order.get("samples", []) if isinstance(existing_order.get("samples", []), list) else []:
        merged["samples"].append(sample)
    for sample in new_order.get("samples", []) if isinstance(new_order.get("samples", []), list) else []:
        merged["samples"].append(sample)
    merged["data"] = merged_data
    return merged


def merge_p0_group(existing_group: dict[str, Any], new_group: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing_group)
    existing_orders = {int(order.get("order")): order for order in existing_group.get("orders", []) if isinstance(order, dict) and order.get("order") is not None}
    new_orders = {int(order.get("order")): order for order in new_group.get("orders", []) if isinstance(order, dict) and order.get("order") is not None}
    merged_orders: list[dict[str, Any]] = []
    for order in sorted(set(existing_orders) | set(new_orders)):
        if order in existing_orders and order in new_orders:
            merged_orders.append(merge_order_block(existing_orders[order], new_orders[order]))
        elif order in existing_orders:
            merged_orders.append(dict(existing_orders[order]))
        else:
            merged_orders.append(dict(new_orders[order]))

    merged["orders"] = merged_orders
    merged["num_samples_total"] = int(existing_group.get("num_samples_total", 0) or 0) + int(new_group.get("num_samples_total", 0) or 0)

    old_colors = existing_group.get("colors", {}) if isinstance(existing_group.get("colors", {}), dict) else {}
    new_colors = new_group.get("colors", {}) if isinstance(new_group.get("colors", {}), dict) else {}
    merged["colors"] = summary_as_colors(
        combine_summary_dicts(colors_as_summary(old_colors), colors_as_summary(new_colors)),
        merged["num_samples_total"],
    )
    return merged


def build_bundle_for_files(
    params: dict[str, Any],
    rel_group: Path,
    files: list[Path],
    jobs: int = 1,
    series_mode: str = "full",
    sample_cache_dir: Path | None = None,
    fingerprint_mode: str = "stat",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    include_time_series = series_mode == "full"
    include_flz = series_mode in ("full", "profiles")
    groups: dict[tuple[float, float], list[Path]] = defaultdict(list)
    for fp in files:
        parsed_name = parse_sample_name(fp)
        if parsed_name is None:
            continue
        groups[parsed_name].append(fp)

    bundle: dict[str, Any] = {
        "meta": {
            **params,
            "raw_group": rel_group.as_posix(),
            "num_json_files": len(files),
            "num_parseable_json_files": len({fp.name for files_group in groups.values() for fp in files_group}),
            "dynamic_processing_version": DYNAMIC_PROCESSING_VERSION,
            "series_mode": series_mode,
        },
        "p0_groups": [],
    }

    all_rows: list[dict[str, Any]] = []
    all_color_rows: list[dict[str, Any]] = []

    for (P0, p0), group_files in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        by_order: dict[int, list[dict[str, Any]]] = defaultdict(list)
        sample_rows, stabilized_counts = process_sample_files(
            group_files,
            jobs=jobs,
            include_time_series=include_time_series,
            include_flz=include_flz,
            cache_dir=sample_cache_dir,
            fingerprint_mode=fingerprint_mode,
        )
        bundle["meta"].update(common_dynamic_criterion_metadata(sample_rows))
        processed = len(stabilized_counts)
        for row in sample_rows:
            by_order[int(row["order"])].append(row)

        nc_summary = summary_from_values(stabilized_counts)
        nc_mean = nc_summary["mean"]
        nc_err = nc_summary["err"]
        nc_std = nc_summary["std"]

        p0_group = {
            "P0_value": P0,
            "p0_value": p0,
            "num_samples_total": processed,
            "colors": summary_as_colors(nc_summary, processed),
            "orders": [],
        }

        all_color_rows.append({
            **params,
            "num_colors": params["nc"],
            "p0": p0,
            "P0": P0,
            "N_samples": processed,
            "nc": nc_mean,
            "nc_err": nc_err,
            "nc_std": nc_std,
        })

        for order in sorted(by_order):
            items = by_order[order]
            p_vals = [x["p_sample_mean"] for x in items if x["p_sample_mean"] is not None]
            f_vals = [x["f_sample_mean"] for x in items if x["f_sample_mean"] is not None]
            z_vals = [x["z_max"] for x in items if x["z_max"] is not None]
            z_stat_vals = [x["z_stat"] for x in items if x["z_stat"] is not None]
            teq_vals = [x["t_eq_species"] for x in items if x["t_eq_species"] is not None]

            p_summary = summary_with_values(p_vals)
            f_summary = summary_with_values(f_vals)
            z_summary = summary_with_values(z_vals)
            z_stat_summary = summary_from_values(z_stat_vals)
            teq_summary = summary_from_values(teq_vals)
            p_mean = p_summary["mean"]
            p_err = p_summary["err"]
            f_mean = f_summary["mean"]
            f_err = f_summary["err"]
            z_mean = z_summary["mean"]
            z_err = z_summary["err"]
            z_stat_mean = z_stat_summary["mean"]
            z_stat_err = z_stat_summary["err"]
            series_data = average_dynamic_time_series(items)

            N_samples_perc = len(items)
            order_block = {
                "order": order,
                "N_samples": processed,
                "N_samples_perc": N_samples_perc,
                "data": {
                    **series_data,
                    "p_tail_mean": p_mean,
                    "p_tail_err": p_err,
                    "p_tail_sample_values": p_summary.get("values", []),
                    "p_tail_estimator": "mean_of_per_sample_tail_means_after_each_sample_t_eq",
                    "f_tail_mean": f_mean,
                    "f_tail_err": f_err,
                    "f_tail_sample_values": f_summary.get("values", []),
                    "f_tail_estimator": "mean_of_per_sample_tail_means_after_each_sample_t_eq",
                    "z_max_mean": z_mean,
                    "z_max_err": z_err,
                    "z_max_std": z_summary["std"],
                    "z_max_values": z_summary.get("values", []),
                    "z_stat_mean": z_stat_mean,
                    "z_stat_err": z_stat_err,
                    "n_samples_perc": N_samples_perc,
                    "n_samples_total": processed,
                },
                "t_eq_species": teq_summary,
                "p": p_summary,
                "f": f_summary,
                "z_max": z_summary,
                "z_stat": z_stat_summary,
                "samples": [
                    {
                        "filename": x["filename"],
                        "color": x["color"],
                        "t_eq_species": x["t_eq_species"],
                        "p_sample_mean": x["p_sample_mean"],
                        "f_sample_mean": x["f_sample_mean"],
                        "z_max": x["z_max"],
                        "z_stat": x["z_stat"],
                        "fL_z_len": len(x.get("fL_z") or []),
                        "stop_criterion": x.get("stop_criterion"),
                        "t_eq_validation": x.get("t_eq_validation"),
                        "t_eq_s_prime_threshold": x.get("t_eq_s_prime_threshold"),
                        "equilibrium_effective_rel_tol": x.get("equilibrium_effective_rel_tol"),
                        "post_equilibrium_extra_steps": x.get("post_equilibrium_extra_steps"),
                    }
                    for x in items
                ],
            }
            p0_group["orders"].append(order_block)

            all_rows.append({
                **params,
                "p0": p0,
                "P0": P0,
                "order": order,
                "N_samples": processed,
                "N_samples_perc": N_samples_perc,
                "p_mean": p_mean,
                "p_err": p_err,
                "f_mean": f_mean,
                "f_err": f_err,
                "z_max_mean": z_mean,
                "z_max_err": z_err,
                "z_stat_mean": z_stat_mean,
                "z_stat_err": z_stat_err,
            })

        bundle["p0_groups"].append(p0_group)

    return bundle, all_rows, all_color_rows


def process_group(
    data_dir: Path,
    raw_root: Path,
    published_root: Path,
    manifests_root: Path,
    clear: bool = False,
    include_laterals: bool = False,
    jobs: int = 1,
    fingerprint_mode: str = "stat",
    detect_replaced_files: bool = False,
    pretty_json: bool = False,
    series_mode: str = "full",
    collect_rows: bool = True,
    migrate_published: bool = True,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    if series_mode not in ("full", "profiles", "scalars"):
        raise ValueError(f"Unknown series_mode: {series_mode}")
    params = parse_data_dir(data_dir)
    if params is None:
        raise ValueError(f"Could not parse dynamic data dir: {data_dir}")

    rel_group = data_dir.parent.relative_to(raw_root)
    out_dir = published_root / rel_group
    sample_cache_dir = manifests_root / rel_group / "sample_cache" / series_mode
    ensure_dir(out_dir)
    out_path = existing_dynamic_bundle_path(out_dir) or dynamic_bundle_path(out_dir)

    json_files = collect_group_json_files(data_dir)
    current_json_files = sorted({fp.name for fp in json_files})
    files_by_name = {fp.name: fp for fp in json_files}

    manifest = load_manifest(manifests_root, rel_group)
    manifest_files = set(map(str, manifest.get("processed_json_files", [])))
    manifest_fingerprints_raw = manifest.get("processed_json_file_fingerprints", {})
    manifest_fingerprints = (
        {str(k): str(v) for k, v in manifest_fingerprints_raw.items()}
        if isinstance(manifest_fingerprints_raw, dict)
        else {}
    )
    manifest_version = int(manifest.get("dynamic_processing_version", 0) or 0)
    manifest_series_mode = str(manifest.get("series_mode", "full") or "full")
    include_laterals = False
    names_new_to_manifest = sorted(set(current_json_files) - manifest_files)
    current_file_fingerprints: dict[str, str] = {}
    if manifest_fingerprints:
        names_to_fingerprint = set(names_new_to_manifest)
        if detect_replaced_files:
            names_to_fingerprint.update(
                name for name in current_json_files
                if name in manifest_fingerprints
            )
        current_file_fingerprints = {
            name: file_fingerprint_for_mode(files_by_name[name], fingerprint_mode)
            for name in sorted(names_to_fingerprint)
        }
        replaced_files = (
            [
                name for name in current_json_files
                if name in current_file_fingerprints
                and fingerprint_matches_mode(manifest_fingerprints.get(name), fingerprint_mode)
                and manifest_fingerprints.get(name) != current_file_fingerprints.get(name)
            ]
            if detect_replaced_files
            else []
        )
        new_sample_files = sorted(set(names_new_to_manifest) | set(replaced_files))
    else:
        missing_manifest_files = bool(manifest_files - set(current_json_files))
        if missing_manifest_files and out_path.exists() and not clear:
            new_sample_files = current_json_files
        else:
            new_sample_files = names_new_to_manifest
        current_file_fingerprints = {
            name: file_fingerprint_for_mode(files_by_name[name], fingerprint_mode)
            for name in sorted(new_sample_files)
        }
    existing_bundle_for_validation: dict[str, Any] | None = None
    should_repair_missing_series = False
    should_validate_existing_bundle = (
        out_path.exists()
        and not clear
        and manifest_version != DYNAMIC_PROCESSING_VERSION
    )
    if should_validate_existing_bundle:
        try:
            maybe_bundle = load_json_bundle(out_path)
            if isinstance(maybe_bundle, dict):
                existing_bundle_for_validation = maybe_bundle
                should_repair_missing_series = bundle_has_missing_dynamic_series(maybe_bundle)
        except Exception:
            existing_bundle_for_validation = None
    if should_repair_missing_series and current_json_files:
        print(f"[repair] detected empty dynamic time series in {out_path}; rebuilding from raw samples")
    should_rebuild = (
        clear
        or not out_path.exists()
        or (should_repair_missing_series and bool(current_json_files))
        or (manifest_version != DYNAMIC_PROCESSING_VERSION and bool(current_json_files))
        or (manifest_series_mode != series_mode and bool(current_json_files))
    )
    if not should_rebuild and not new_sample_files:
        if migrate_published:
            migrated_out_path = ensure_dynamic_bundle_compressed(out_dir)
            if migrated_out_path is not None:
                out_path = migrated_out_path
                existing_bundle_for_validation = None
        cached_rows = rows_from_manifest_cache(manifest, series_mode) if collect_rows else None
        rows_cache_missing = collect_rows and cached_rows is None
        if not collect_rows:
            all_rows, all_color_rows = [], []
        elif cached_rows is not None:
            all_rows, all_color_rows = cached_rows
        elif existing_bundle_for_validation is not None:
            all_rows, all_color_rows = rows_from_bundle(existing_bundle_for_validation)
        else:
            all_rows, all_color_rows = rows_from_existing_bundle(out_path)
        if (
            not manifest_fingerprints
            or manifest_version != DYNAMIC_PROCESSING_VERSION
            or manifest.get("fingerprint_mode") != fingerprint_mode
            or rows_cache_missing
        ):
            if detect_replaced_files or not manifest_fingerprints:
                for name in current_json_files:
                    if name not in current_file_fingerprints:
                        current_file_fingerprints[name] = file_fingerprint_for_mode(files_by_name[name], fingerprint_mode)
            fingerprints_out = dict(manifest_fingerprints)
            fingerprints_out.update(current_file_fingerprints)
            manifest.update({
                "group_relpath": rel_group.as_posix(),
                "data_dir": data_dir.as_posix(),
                "processed_json_files": sorted(set(manifest_files) | set(current_json_files)),
                "n_processed_json_files": len(set(manifest_files) | set(current_json_files)),
                "processed_json_file_fingerprints": dict(sorted(fingerprints_out.items())),
                "fingerprint_mode": fingerprint_mode,
                "summary_file": out_path.as_posix(),
                "dynamic_processing_version": DYNAMIC_PROCESSING_VERSION,
                "series_mode": series_mode,
                "last_update": datetime.now(timezone.utc).isoformat(),
            })
            if collect_rows:
                manifest.update(manifest_rows_cache_payload(all_rows, all_color_rows))
            save_manifest(manifests_root, rel_group, manifest)
        print(f"[skip] {out_path} ({len(all_rows)} rows)")
        return out_path, all_rows, all_color_rows

    if new_sample_files:
        preview = ", ".join(new_sample_files[:5])
        suffix = "" if len(new_sample_files) <= 5 else f", ... (+{len(new_sample_files) - 5} more)"
        print(f"[update] detected {len(new_sample_files)} new sample files for {rel_group}: {preview}{suffix}")

    existing_bundle: dict[str, Any] | None = existing_bundle_for_validation
    if out_path.exists() and not clear:
        if existing_bundle is None:
            try:
                existing_bundle = load_json_bundle(out_path)
            except Exception:
                existing_bundle = None

    if new_sample_files and not should_rebuild and existing_bundle is not None and isinstance(existing_bundle, dict):
        batch_bundle, _, _ = build_bundle_for_files(
            params,
            rel_group,
            [path for path in json_files if path.name in set(new_sample_files)],
            jobs=jobs,
            series_mode=series_mode,
            sample_cache_dir=sample_cache_dir,
            fingerprint_mode=fingerprint_mode,
        )
        merged_p0_groups: dict[tuple[float, float], dict[str, Any]] = {}
        for p0_group in existing_bundle.get("p0_groups", []) if isinstance(existing_bundle.get("p0_groups", []), list) else []:
            if not isinstance(p0_group, dict):
                continue
            key = (float(p0_group.get("P0_value", 0.0) or 0.0), float(p0_group.get("p0_value", 0.0) or 0.0))
            merged_p0_groups[key] = dict(p0_group)

        for p0_group in batch_bundle.get("p0_groups", []) if isinstance(batch_bundle.get("p0_groups", []), list) else []:
            if not isinstance(p0_group, dict):
                continue
            key = (float(p0_group.get("P0_value", 0.0) or 0.0), float(p0_group.get("p0_value", 0.0) or 0.0))
            if key in merged_p0_groups:
                merged_p0_groups[key] = merge_p0_group(merged_p0_groups[key], p0_group)
            else:
                merged_p0_groups[key] = dict(p0_group)

        merged_bundle = {
            "meta": {
                **params,
                **{
                    key: batch_bundle.get("meta", {}).get(key)
                    for key in (
                        "stop_criterion",
                        "t_eq_validation",
                        "t_eq_s_prime_threshold",
                        "equilibrium_effective_rel_tol",
                        "post_equilibrium_extra_steps",
                        "equilibrium_rel_tol_scaling",
                    )
                    if isinstance(batch_bundle.get("meta", {}), dict) and
                    batch_bundle.get("meta", {}).get(key) is not None
                },
                "raw_group": rel_group.as_posix(),
                "num_json_files": len(json_files),
                "num_parseable_json_files": len(current_json_files),
                "dynamic_processing_version": DYNAMIC_PROCESSING_VERSION,
                "series_mode": series_mode,
            },
            "p0_groups": [merged_p0_groups[key] for key in sorted(merged_p0_groups)],
        }
        bundle = merged_bundle
    else:
        bundle, _, _ = build_bundle_for_files(
            params,
            rel_group,
            json_files,
            jobs=jobs,
            series_mode=series_mode,
            sample_cache_dir=sample_cache_dir,
            fingerprint_mode=fingerprint_mode,
        )

    all_rows, all_color_rows = rows_from_bundle(bundle) if collect_rows else ([], [])

    out_path = dynamic_bundle_path(out_dir)
    write_json_bundle(out_path, bundle, pretty=pretty_json)
    remove_legacy_dynamic_bundles(out_dir, keep=out_path)

    if clear:
        processed_files_out = set(current_json_files)
        fingerprints_out = {
            name: file_fingerprint_for_mode(files_by_name[name], fingerprint_mode)
            for name in current_json_files
        }
    else:
        processed_files_out = set(manifest_files) | set(current_json_files)
        fingerprints_out = dict(manifest_fingerprints)
        missing_fingerprints = sorted(set(current_json_files) - set(fingerprints_out))
        for name in missing_fingerprints:
            current_file_fingerprints[name] = file_fingerprint_for_mode(files_by_name[name], fingerprint_mode)
        fingerprints_out.update(current_file_fingerprints)

    manifest.update({
        "group_relpath": rel_group.as_posix(),
        "data_dir": data_dir.as_posix(),
        "processed_json_files": sorted(processed_files_out),
        "n_processed_json_files": len(processed_files_out),
        "processed_json_file_fingerprints": dict(sorted(fingerprints_out.items())),
        "fingerprint_mode": fingerprint_mode,
        "summary_file": out_path.as_posix(),
        "dynamic_processing_version": DYNAMIC_PROCESSING_VERSION,
        "series_mode": series_mode,
        "last_update": datetime.now(timezone.utc).isoformat(),
    })
    if collect_rows:
        manifest.update(manifest_rows_cache_payload(all_rows, all_color_rows))
    save_manifest(manifests_root, rel_group, manifest)

    return out_path, all_rows, all_color_rows


def write_all_data(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    rows = sorted(
        rows,
        key=lambda r: (
            r["type_perc"], r["dim"], r["nc"], r["rho"], r["c"], r["f_T"],
            r["L"], r["P0"], r["p0"], r["stat_window"], r["order"],
        ),
    )
    with output_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(ALL_DATA_COLUMNS) + "\n")
        for r in rows:
            f.write(" ".join(dat_value(r.get(col)) for col in ALL_DATA_COLUMNS) + "\n")


def write_all_colors(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    rows = sorted(
        rows,
        key=lambda r: (
            r["type_perc"], r["dim"], r["num_colors"], r["rho"], r["c"],
            r["f_T"], r["L"], r["P0"], r["p0"], r["stat_window"],
        ),
    )
    with output_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(ALL_COLORS_COLUMNS) + "\n")
        for r in rows:
            f.write(" ".join(dat_value(r.get(col)) for col in ALL_COLORS_COLUMNS) + "\n")


def compress_published_only(
    published_root: Path,
    sop_root: Path,
    all_data_name: str,
    all_colors_name: str,
    *,
    compresslevel: int = 1,
    threads: int = 0,
    rebuild_all_data: bool = False,
) -> tuple[int, int, int]:
    ensure_dir(published_root)
    dynamic_converted = 0
    skipped = 0

    candidate_dirs: set[Path] = set()
    for pattern in (
        "properties_dynamic_bundle.json",
        "properties_dynamic_bundle.json.gz",
        "properties_dynamic_bundle.json.xz",
    ):
        candidate_dirs.update(path.parent for path in published_root.rglob(pattern))

    for out_dir in sorted(candidate_dirs):
        before_dynamic = existing_dynamic_bundle_path(out_dir)
        after_dynamic = ensure_dynamic_bundle_compressed(
            out_dir,
            compresslevel=compresslevel,
            threads=threads,
        )
        if after_dynamic is not None:
            if before_dynamic is not None and before_dynamic != after_dynamic:
                dynamic_converted += 1
                print(f"[compress] {before_dynamic} -> {after_dynamic}")
            else:
                skipped += 1

    if rebuild_all_data:
        all_rows: list[dict[str, Any]] = []
        all_color_rows: list[dict[str, Any]] = []
        bundle_paths = sorted(published_root.rglob("properties_dynamic_bundle.json.xz"))
        for bundle_path in bundle_paths:
            try:
                rows, color_rows = rows_from_existing_bundle(bundle_path)
                all_rows.extend(rows)
                all_color_rows.extend(color_rows)
            except Exception as exc:
                print(f"[warn] failed to import {bundle_path}: {exc}")

        if all_rows:
            all_data_path = sop_root / all_data_name
            write_all_data(all_rows, all_data_path)
            print(f"[write] {all_data_path} ({len(all_rows)} rows)")
        if all_color_rows:
            all_colors_path = sop_root / all_colors_name
            write_all_colors(all_color_rows, all_colors_path)
            print(f"[write] {all_colors_path} ({len(all_color_rows)} rows)")

    return dynamic_converted, 0, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process raw_growth_test_dynamic into published_dynamic and all_data_dynamic.dat."
    )
    parser.add_argument("--sop-root", default=str(Path(__file__).resolve().parents[1] / "SOP_data"))
    parser.add_argument("--raw-dir", default="raw_growth_test_dynamic")
    parser.add_argument("--published-dir", default="published_dynamic")
    parser.add_argument("--manifests-dir", default="manifests_dynamic")
    parser.add_argument("--all-data-name", default="all_data_dynamic.dat")
    parser.add_argument("--all-colors-name", default="all_colors_dynamic.dat")
    parser.add_argument(
        "--compress-published-only",
        "-compress-published-only",
        action="store_true",
        help="Only convert existing published dynamic/lateral bundles to .json.xz; does not require raw files.",
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        default=1,
        help="XZ compression level for --compress-published-only. 1 is fastest; 6 is smaller but much slower.",
    )
    parser.add_argument(
        "--compress-threads",
        type=int,
        default=0,
        help="Threads passed to xz for --compress-published-only. 0 means all cores when the xz binary is available.",
    )
    parser.add_argument(
        "--rebuild-all-data",
        action="store_true",
        help="With --compress-published-only, also rebuild all_data/all_colors from compressed properties bundles.",
    )
    parser.add_argument(
        "--write-all-data",
        dest="write_all_data_outputs",
        action="store_true",
        default=True,
        help="Write all_data_dynamic.dat and all_colors_dynamic.dat after processing.",
    )
    parser.add_argument(
        "--skip-all-data",
        dest="write_all_data_outputs",
        action="store_false",
        help="Skip rebuilding all_data_dynamic.dat/all_colors_dynamic.dat. Faster for incremental updates.",
    )
    parser.add_argument("--clear", action="store_true", help="Ignore manifest cache and rebuild dynamic bundles.")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="Number of worker processes used to parse new JSON samples. Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--fingerprint-mode",
        choices=("stat", "hash"),
        default="stat",
        help="How to fingerprint newly processed files in the manifest. stat is much faster; hash is stricter.",
    )
    parser.add_argument(
        "--detect-replaced-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also fingerprint already-known filenames to detect files recreated with the same name. Slower on large datasets.",
    )
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="Write published dynamic bundles with indentation. Slower and larger; compact JSON is the default.",
    )
    parser.add_argument(
        "--migrate-published",
        dest="migrate_published",
        action="store_true",
        default=True,
        help="Migrate legacy published bundles to .json.xz during normal processing.",
    )
    parser.add_argument(
        "--no-migrate-published",
        dest="migrate_published",
        action="store_false",
        help="Do not compress/migrate legacy published bundles during normal processing. Faster for incremental updates.",
    )
    parser.add_argument(
        "--series-mode",
        choices=("full", "profiles", "scalars"),
        default="profiles",
        help=(
            "full stores aggregated pt/ft time series and fL_z profiles; "
            "profiles skips pt/ft time-series aggregation but keeps fL_z; "
            "scalars stores only scalar summaries. profiles is much faster for large datasets."
        ),
    )
    parser.add_argument(
        "--include-laterals",
        dest="include_laterals",
        action="store_true",
        default=False,
        help="Deprecated; lateral correlation/susceptibility processing is disabled.",
    )
    parser.add_argument(
        "--no-laterals",
        dest="include_laterals",
        action="store_false",
        help="Deprecated; lateral correlation/susceptibility processing is always skipped.",
    )
    args = parser.parse_args()

    sop_root = Path(args.sop_root).expanduser().resolve()
    jobs = max(1, int(args.jobs))
    raw_root = sop_root / args.raw_dir
    published_root = sop_root / args.published_dir
    manifests_root = sop_root / args.manifests_dir
    ensure_dir(published_root)
    ensure_dir(manifests_root)

    if args.compress_published_only:
        dynamic_n, lateral_n, skipped_n = compress_published_only(
            published_root,
            sop_root,
            args.all_data_name,
            args.all_colors_name,
            compresslevel=max(0, min(9, int(args.compress_level))),
            threads=max(0, int(args.compress_threads)),
            rebuild_all_data=args.rebuild_all_data,
        )
        print(
            f"[compress] done: dynamic={dynamic_n}, lateral={lateral_n}, "
            f"already_xz_or_missing={skipped_n}"
        )
        return 0

    data_dirs = discover_data_dirs(raw_root)
    print(f"[dynamic] data dirs found: {len(data_dirs)}")

    all_rows: list[dict[str, Any]] = []
    all_color_rows: list[dict[str, Any]] = []
    processed_bundle_paths: set[Path] = set()
    # First, ensure all raw data groups are processed and published bundles are up-to-date.
    for data_dir in data_dirs:
        out_path, rows, color_rows = process_group(
            data_dir,
            raw_root,
            published_root,
            manifests_root,
            clear=args.clear,
            include_laterals=args.include_laterals,
            jobs=jobs,
            fingerprint_mode=args.fingerprint_mode,
            detect_replaced_files=args.detect_replaced_files,
            pretty_json=args.pretty_json,
            series_mode=args.series_mode,
            collect_rows=args.write_all_data_outputs,
            migrate_published=args.migrate_published,
        )
        processed_bundle_paths.add(out_path.resolve())
        all_rows.extend(rows)
        all_color_rows.extend(color_rows)
        print(f"[published] ensured {out_path}")

    if args.write_all_data_outputs:
        # Regardless of raw presence, build the final all_data and all_colors from published bundles.
        bundle_paths = sorted(
            set(published_root.rglob("properties_dynamic_bundle.json.xz"))
            | set(published_root.rglob("properties_dynamic_bundle.json.gz"))
            | set(published_root.rglob("properties_dynamic_bundle.json"))
        )
        remaining_bundle_paths = [
            path for path in bundle_paths
            if path.resolve() not in processed_bundle_paths
        ]
        print(
            f"[dynamic] building all-data from {len(bundle_paths)} published bundles "
            f"({len(remaining_bundle_paths)} imported from disk)"
        )
        for bundle_path in remaining_bundle_paths:
            try:
                rows, color_rows = rows_from_existing_bundle(bundle_path)
                if rows:
                    all_rows.extend(rows)
                    print(f"[import] {bundle_path} ({len(rows)} rows)")
                if color_rows:
                    all_color_rows.extend(color_rows)
            except Exception as exc:
                print(f"[warn] failed to import {bundle_path}: {exc}")

        all_data_path = sop_root / args.all_data_name
        write_all_data(all_rows, all_data_path)
        print(f"[write] {all_data_path} ({len(all_rows)} rows)")

        all_colors_path = sop_root / args.all_colors_name
        write_all_colors(all_color_rows, all_colors_path)
        print(f"[write] {all_colors_path} ({len(all_color_rows)} rows)")
    else:
        print("[dynamic] skipped all_data/all_colors rebuild")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
