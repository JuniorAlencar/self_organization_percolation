#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


DYNAMIC_PROCESSING_VERSION = 7
LATERAL_PROCESSING_VERSION = 1

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
]

ALL_COLORS_COLUMNS = [
    "type_perc", "dim", "L", "f_T", "c", "num_colors", "P0", "p0",
    "N_samples", "rho", "nc", "nc_err", "nc_std", "stat_window",
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


def save_manifest(manifests_root: Path,
                  rel_group: Path,
                  manifest: dict[str, Any]) -> Path:
    path = manifest_path(manifests_root, rel_group)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(manifest), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    return path


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
        "stat_window": int(g["stat_window"]) if g.get("stat_window") else -1,
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


def average_dynamic_time_series(items: list[dict[str, Any]]) -> dict[str, Any]:
    series_pt: list[tuple[np.ndarray, np.ndarray]] = []
    series_ft: list[np.ndarray] = []
    t_eq_vals: list[float] = []

    for item in items:
        t_eq = finite_float(item.get("t_eq_species"))
        if t_eq is not None:
            t_eq_vals.append(t_eq)

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
                series_ft.append(f[:n_ft])

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

    if not series_pt:
        out.update({
            "time": [],
            "pt_mean": [],
            "pt_std": [],
            "pt_sem": [],
            "ft_mean": [],
            "ft_std": [],
            "ft_sem": [],
            "n_seeds_pt": 0,
            "n_seeds_ft": 0,
        })
        return out

    min_len_pt = min(p.size for _, p in series_pt)
    t_common = series_pt[0][0][:min_len_pt]
    pts = np.stack([p[:min_len_pt] for _, p in series_pt], axis=0)
    nseed_pt = int(pts.shape[0])
    pt_mean = np.mean(pts, axis=0)
    if nseed_pt > 1:
        pt_std = np.std(pts, axis=0, ddof=1)
        pt_sem = pt_std / math.sqrt(nseed_pt)
    else:
        pt_std = np.zeros_like(pt_mean)
        pt_sem = np.zeros_like(pt_mean)

    out["time"] = t_common.tolist()
    out["pt_mean"] = pt_mean.tolist()
    out["pt_std"] = pt_std.tolist()
    out["pt_sem"] = pt_sem.tolist()
    out["n_seeds_pt"] = nseed_pt

    if series_ft:
        min_len_ft = min(f.size for f in series_ft)
        min_len = min(min_len_pt, min_len_ft)
        fts = np.stack([f[:min_len] for f in series_ft], axis=0)
        nseed_ft = int(fts.shape[0])
        ft_mean = np.mean(fts, axis=0)
        if nseed_ft > 1:
            ft_std = np.std(fts, axis=0, ddof=1)
            ft_sem = ft_std / math.sqrt(nseed_ft)
        else:
            ft_std = np.zeros_like(ft_mean)
            ft_sem = np.zeros_like(ft_mean)

        pt_mean = np.mean(pts[:, :min_len], axis=0)
        if nseed_pt > 1:
            pt_std = np.std(pts[:, :min_len], axis=0, ddof=1)
            pt_sem = pt_std / math.sqrt(nseed_pt)
        else:
            pt_std = np.zeros_like(pt_mean)
            pt_sem = np.zeros_like(pt_mean)

        out["time"] = t_common[:min_len].tolist()
        out["pt_mean"] = pt_mean.tolist()
        out["pt_std"] = pt_std.tolist()
        out["pt_sem"] = pt_sem.tolist()
        out["ft_mean"] = ft_mean.tolist()
        out["ft_std"] = ft_std.tolist()
        out["ft_sem"] = ft_sem.tolist()
        out["n_seeds_ft"] = nseed_ft
    else:
        out["ft_mean"] = []
        out["ft_std"] = []
        out["ft_sem"] = []
        out["n_seeds_ft"] = 0

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


def load_dynamic_sample(path: Path) -> list[dict[str, Any]] | None:
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

        out.append({
            "order": order_pos,
            "color": color_1b,
            "t_eq_species": t_eq,
            "time": data.get("time"),
            "pt": data.get("pt"),
            "ft": data.get("nt"),
            "p_sample_mean": p_mean,
            "f_sample_mean": f_mean,
            "z_max": z_max,
            "z_stat": z_stat,
        })

    return out


def process_one_sample_file(sample_path: Path) -> tuple[list[dict[str, Any]], float | None]:
    sample_orders = load_dynamic_sample(sample_path)
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
            "p_sample_mean": item.get("p_sample_mean"),
            "f_sample_mean": item.get("f_sample_mean"),
            "z_max": item.get("z_max"),
            "z_stat": item.get("z_stat"),
        })
    return rows, float(len(sample_orders))


def process_sample_files(sample_paths: list[Path], jobs: int = 1) -> tuple[list[dict[str, Any]], list[float]]:
    """Process a list of dynamic sample JSON files and return parsed rows plus per-file stabilization counts."""
    rows: list[dict[str, Any]] = []
    stabilized_counts: list[float] = []

    if jobs <= 1 or len(sample_paths) <= 1:
        results = map(process_one_sample_file, sample_paths)
        for sample_rows, stabilized_count in results:
            if stabilized_count is None:
                continue
            stabilized_counts.append(stabilized_count)
            rows.extend(sample_rows)
    else:
        chunksize = max(1, min(32, len(sample_paths) // (jobs * 4) or 1))
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = executor.map(process_one_sample_file, sample_paths, chunksize=chunksize)
            for sample_rows, stabilized_count in results:
                if stabilized_count is None:
                    continue
                stabilized_counts.append(stabilized_count)
                rows.extend(sample_rows)

    return rows, stabilized_counts


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
            summary["series"] = series
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
            summary["series"] = series
            summary["series_kind"] = "susceptibility_summary"

        rows.append(summary)

    return rows, counts


def process_lateral_correlations(data_dir: Path, out_dir: Path) -> Path | None:
    """Process lateral correlation/susceptibility CSVs in a sibling correlations/ directory when present."""
    correlations_dir = lateral_correlations_dir(data_dir)
    csv_files = collect_lateral_correlation_files(data_dir)
    if not csv_files:
        return None

    rows, counts = process_correlation_files(csv_files)
    bundle = {
        "meta": {
            "data_dir": data_dir.as_posix(),
            "correlations_dir": correlations_dir.as_posix(),
            "n_files": len(csv_files),
            "n_rows": len(rows),
            "row_counts": counts,
            "format": "compact_summary",
        },
        "samples": rows,
    }
    lateral_out_path = lateral_bundle_path(out_dir)
    with gzip.open(lateral_out_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(json_safe(bundle), handle, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
    legacy_path = legacy_lateral_bundle_path(out_dir)
    if legacy_path.exists():
        legacy_path.unlink()
    return lateral_out_path


def lateral_correlations_dir(data_dir: Path) -> Path:
    return data_dir.parent / "correlations"


def lateral_bundle_path(out_dir: Path) -> Path:
    return out_dir / "lateral_correlations_bundle.json.gz"


def legacy_lateral_bundle_path(out_dir: Path) -> Path:
    return out_dir / "lateral_correlations_bundle.json"


def existing_lateral_bundle_path(out_dir: Path) -> Path | None:
    compressed_path = lateral_bundle_path(out_dir)
    if compressed_path.exists():
        return compressed_path
    legacy_path = legacy_lateral_bundle_path(out_dir)
    if legacy_path.exists():
        return legacy_path
    return None


def ensure_lateral_bundle_compressed(out_dir: Path) -> Path | None:
    compressed_path = lateral_bundle_path(out_dir)
    if compressed_path.exists():
        return compressed_path
    legacy_path = legacy_lateral_bundle_path(out_dir)
    if not legacy_path.exists():
        return None
    with legacy_path.open("rb") as source, gzip.open(compressed_path, "wb", compresslevel=6) as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    legacy_path.unlink()
    return compressed_path


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
    with bundle_path.open("r", encoding="utf-8") as f:
        bundle = json.load(f)
    return rows_from_bundle(bundle)


def summary_from_values(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if finite_float(v) is not None]
    mean, err, std, n = mean_sem_std(clean)
    return {
        "mean": mean,
        "err": err,
        "std": std,
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
        return {"mean": None, "err": None, "n": 0, "sum": 0.0, "sumsq": 0.0}
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
    return finalize_summary(
        old_n + new_n,
        summary_sum(old_summary) + summary_sum(new_summary),
        summary_sumsq(old_summary) + summary_sumsq(new_summary),
    )


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


def combine_series_arrays(
    old_mean: list[float] | None,
    old_std: list[float] | None,
    old_n: int,
    new_mean: list[float] | None,
    new_std: list[float] | None,
    new_n: int,
) -> tuple[list[float], list[float], list[float], int]:
    if old_mean is None and new_mean is None:
        return [], [], [], 0
    if old_mean is None:
        return list(new_mean or []), list(new_std or []), [0.0] * len(new_mean or []), new_n
    if new_mean is None:
        return list(old_mean), list(old_std or []), [0.0] * len(old_mean), old_n

    length = min(len(old_mean), len(new_mean))
    combined_mean: list[float] = []
    combined_std: list[float] = []
    combined_sem: list[float] = []
    total_n = old_n + new_n
    for idx in range(length):
        old_val = old_mean[idx] if idx < len(old_mean) else None
        new_val = new_mean[idx] if idx < len(new_mean) else None
        old_std_val = old_std[idx] if old_std is not None and idx < len(old_std) else None
        new_std_val = new_std[idx] if new_std is not None and idx < len(new_std) else None

        if old_val is None and new_val is None:
            combined_mean.append(math.nan)
            combined_std.append(math.nan)
            combined_sem.append(math.nan)
            continue
        if old_val is None:
            combined_mean.append(float(new_val))
            combined_std.append(float(new_std_val) if new_std_val is not None else 0.0)
            combined_sem.append(float(new_std_val) / math.sqrt(new_n) if new_n > 1 and new_std_val is not None else 0.0)
            continue
        if new_val is None:
            combined_mean.append(float(old_val))
            combined_std.append(float(old_std_val) if old_std_val is not None else 0.0)
            combined_sem.append(float(old_std_val) / math.sqrt(old_n) if old_n > 1 and old_std_val is not None else 0.0)
            continue

        mean = (old_n * float(old_val) + new_n * float(new_val)) / total_n
        m2_old = 0.0 if old_n <= 1 else (old_n - 1) * (float(old_std_val) ** 2 if old_std_val is not None else 0.0)
        m2_new = 0.0 if new_n <= 1 else (new_n - 1) * (float(new_std_val) ** 2 if new_std_val is not None else 0.0)
        delta = float(old_val) - float(new_val)
        m2 = m2_old + m2_new + (delta * delta * old_n * new_n / total_n)
        std = 0.0 if total_n <= 1 else math.sqrt(m2 / (total_n - 1))
        sem = 0.0 if total_n <= 1 else std / math.sqrt(total_n)
        combined_mean.append(float(mean))
        combined_std.append(float(std))
        combined_sem.append(float(sem))

    return combined_mean, combined_std, combined_sem, total_n


def merge_order_block(existing_order: dict[str, Any], new_order: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing_order)
    merged["N_samples"] = int(existing_order.get("N_samples", 0) or 0) + int(new_order.get("N_samples", 0) or 0)
    merged["N_samples_perc"] = int(existing_order.get("N_samples_perc", 0) or 0) + int(new_order.get("N_samples_perc", 0) or 0)

    existing_data = existing_order.get("data", {}) if isinstance(existing_order.get("data", {}), dict) else {}
    new_data = new_order.get("data", {}) if isinstance(new_order.get("data", {}), dict) else {}
    merged_data = dict(existing_data)
    merged_data["time"] = existing_data.get("time", new_data.get("time"))

    old_n_pt = int(existing_data.get("n_seeds_pt", 0) or 0)
    new_n_pt = int(new_data.get("n_seeds_pt", 0) or 0)
    pt_mean, pt_std, pt_sem, _ = combine_series_arrays(
        existing_data.get("pt_mean"),
        existing_data.get("pt_std"),
        old_n_pt,
        new_data.get("pt_mean"),
        new_data.get("pt_std"),
        new_n_pt,
    )
    merged_data["pt_mean"] = pt_mean
    merged_data["pt_std"] = pt_std
    merged_data["pt_sem"] = pt_sem
    merged_data["n_seeds_pt"] = old_n_pt + new_n_pt
    merged_data["time"] = list(merged_data.get("time") or [])[:len(pt_mean)]

    old_n_ft = int(existing_data.get("n_seeds_ft", 0) or 0)
    new_n_ft = int(new_data.get("n_seeds_ft", 0) or 0)
    if old_n_ft > 0 or new_n_ft > 0:
        ft_mean, ft_std, ft_sem, _ = combine_series_arrays(
            existing_data.get("ft_mean"),
            existing_data.get("ft_std"),
            old_n_ft,
            new_data.get("ft_mean"),
            new_data.get("ft_std"),
            new_n_ft,
        )
        merged_data["ft_mean"] = ft_mean
        merged_data["ft_std"] = ft_std
        merged_data["ft_sem"] = ft_sem
        merged_data["n_seeds_ft"] = old_n_ft + new_n_ft
        common_len = min(len(merged_data.get("time") or []), len(ft_mean), len(pt_mean))
        merged_data["time"] = list(merged_data.get("time") or [])[:common_len]
        merged_data["pt_mean"] = merged_data["pt_mean"][:common_len]
        merged_data["pt_std"] = merged_data["pt_std"][:common_len]
        merged_data["pt_sem"] = merged_data["pt_sem"][:common_len]
        merged_data["ft_mean"] = merged_data["ft_mean"][:common_len]
        merged_data["ft_std"] = merged_data["ft_std"][:common_len]
        merged_data["ft_sem"] = merged_data["ft_sem"][:common_len]

    p_old = existing_order.get("p", {}) if isinstance(existing_order.get("p", {}), dict) else {}
    p_new = new_order.get("p", {}) if isinstance(new_order.get("p", {}), dict) else {}
    merged["p"] = combine_summary_dicts(p_old, p_new)
    p_mean = merged["p"]["mean"]
    p_err = merged["p"]["err"]
    merged_data["p_tail_mean"] = p_mean
    merged_data["p_tail_err"] = p_err

    f_old = existing_order.get("f", {}) if isinstance(existing_order.get("f", {}), dict) else {}
    f_new = new_order.get("f", {}) if isinstance(new_order.get("f", {}), dict) else {}
    merged["f"] = combine_summary_dicts(f_old, f_new)
    f_mean = merged["f"]["mean"]
    f_err = merged["f"]["err"]
    merged_data["f_tail_mean"] = f_mean
    merged_data["f_tail_err"] = f_err

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
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
        },
        "p0_groups": [],
    }

    all_rows: list[dict[str, Any]] = []
    all_color_rows: list[dict[str, Any]] = []

    for (P0, p0), group_files in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        by_order: dict[int, list[dict[str, Any]]] = defaultdict(list)
        sample_rows, stabilized_counts = process_sample_files(group_files, jobs=jobs)
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

            p_summary = summary_from_values(p_vals)
            f_summary = summary_from_values(f_vals)
            z_summary = summary_from_values(z_vals)
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
                    "f_tail_mean": f_mean,
                    "f_tail_err": f_err,
                    "z_max_mean": z_mean,
                    "z_max_err": z_err,
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
    include_laterals: bool = True,
    jobs: int = 1,
    fingerprint_mode: str = "stat",
    detect_replaced_files: bool = True,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    params = parse_data_dir(data_dir)
    if params is None:
        raise ValueError(f"Could not parse dynamic data dir: {data_dir}")

    rel_group = data_dir.parent.relative_to(raw_root)
    out_dir = published_root / rel_group
    ensure_dir(out_dir)
    out_path = out_dir / "properties_dynamic_bundle.json"

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
    manifest_lateral_fingerprints_raw = manifest.get("lateral_csv_fingerprints", {})
    manifest_lateral_fingerprints = (
        {str(k): str(v) for k, v in manifest_lateral_fingerprints_raw.items()}
        if isinstance(manifest_lateral_fingerprints_raw, dict)
        else {}
    )
    current_lateral_fingerprints = lateral_csv_fingerprints(data_dir) if include_laterals else {}
    lateral_out_path: Path | None = (
        existing_lateral_bundle_path(out_dir) or lateral_bundle_path(out_dir)
        if current_lateral_fingerprints
        else None
    )
    lateral_bundle_exists = lateral_out_path is not None and lateral_out_path.exists()
    lateral_should_process = (
        include_laterals
        and bool(current_lateral_fingerprints)
        and (
            lateral_out_path is None
            or not lateral_bundle_exists
            or (
                bool(manifest_lateral_fingerprints)
                and (
                    int(manifest.get("lateral_processing_version", 0) or 0) != LATERAL_PROCESSING_VERSION
                    or manifest_lateral_fingerprints != current_lateral_fingerprints
                )
            )
        )
    )
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
    should_rebuild = (
        clear
        or not out_path.exists()
    )
    if not should_rebuild and not new_sample_files:
        all_rows, all_color_rows = rows_from_existing_bundle(out_path)
        if lateral_should_process:
            lateral_out_path = process_lateral_correlations(data_dir, out_dir)
            print(f"[write] {lateral_out_path}")
        elif include_laterals and current_lateral_fingerprints:
            lateral_out_path = ensure_lateral_bundle_compressed(out_dir) or lateral_out_path
            print(f"[skip] {lateral_out_path}")
        if (
            not manifest_fingerprints
            or manifest_version != DYNAMIC_PROCESSING_VERSION
            or manifest.get("fingerprint_mode") != fingerprint_mode
            or lateral_should_process
            or manifest_lateral_fingerprints != current_lateral_fingerprints
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
                "lateral_summary_file": lateral_out_path.as_posix() if lateral_out_path is not None else manifest.get("lateral_summary_file"),
                "lateral_csv_fingerprints": current_lateral_fingerprints,
                "lateral_processing_version": LATERAL_PROCESSING_VERSION if current_lateral_fingerprints else manifest.get("lateral_processing_version"),
                "dynamic_processing_version": DYNAMIC_PROCESSING_VERSION,
                "last_update": datetime.now(timezone.utc).isoformat(),
            })
            save_manifest(manifests_root, rel_group, manifest)
        print(f"[skip] {out_path} ({len(all_rows)} rows)")
        return out_path, all_rows, all_color_rows

    if new_sample_files:
        preview = ", ".join(new_sample_files[:5])
        suffix = "" if len(new_sample_files) <= 5 else f", ... (+{len(new_sample_files) - 5} more)"
        print(f"[update] detected {len(new_sample_files)} new sample files for {rel_group}: {preview}{suffix}")

    existing_bundle: dict[str, Any] | None = None
    if out_path.exists() and not clear:
        try:
            with out_path.open("r", encoding="utf-8") as handle:
                existing_bundle = json.load(handle)
        except Exception:
            existing_bundle = None

    if new_sample_files and existing_bundle is not None and isinstance(existing_bundle, dict):
        batch_bundle, _, _ = build_bundle_for_files(
            params,
            rel_group,
            [path for path in json_files if path.name in set(new_sample_files)],
            jobs=jobs,
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
                "raw_group": rel_group.as_posix(),
                "num_json_files": len(json_files),
                "num_parseable_json_files": len(current_json_files),
                "dynamic_processing_version": DYNAMIC_PROCESSING_VERSION,
            },
            "p0_groups": [merged_p0_groups[key] for key in sorted(merged_p0_groups)],
        }
        bundle = merged_bundle
    else:
        bundle, _, _ = build_bundle_for_files(params, rel_group, json_files, jobs=jobs)

    all_rows, all_color_rows = rows_from_bundle(bundle)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(bundle), f, ensure_ascii=False, indent=2)
        f.write("\n")

    if lateral_should_process:
        lateral_out_path = process_lateral_correlations(data_dir, out_dir)
        print(f"[write] {lateral_out_path}")
    elif include_laterals and current_lateral_fingerprints:
        lateral_out_path = ensure_lateral_bundle_compressed(out_dir) or lateral_out_path
        print(f"[skip] {lateral_out_path}")

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
        "lateral_summary_file": lateral_out_path.as_posix() if lateral_out_path is not None else None,
        "lateral_csv_fingerprints": current_lateral_fingerprints,
        "lateral_processing_version": LATERAL_PROCESSING_VERSION if current_lateral_fingerprints else None,
        "dynamic_processing_version": DYNAMIC_PROCESSING_VERSION,
        "last_update": datetime.now(timezone.utc).isoformat(),
    })
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
    parser.add_argument("--clear", action="store_true", help="Ignore manifest cache and rebuild dynamic bundles.")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
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
        default=True,
        help="Also fingerprint already-known filenames to detect files recreated with the same name. Slower on large datasets.",
    )
    parser.add_argument(
        "--include-laterals",
        dest="include_laterals",
        action="store_true",
        default=True,
        help="Process lateral correlation/susceptibility CSVs when present.",
    )
    parser.add_argument(
        "--no-laterals",
        dest="include_laterals",
        action="store_false",
        help="Skip lateral correlation/susceptibility CSV processing.",
    )
    args = parser.parse_args()

    sop_root = Path(args.sop_root).expanduser().resolve()
    jobs = max(1, int(args.jobs))
    raw_root = sop_root / args.raw_dir
    published_root = sop_root / args.published_dir
    manifests_root = sop_root / args.manifests_dir
    ensure_dir(published_root)
    ensure_dir(manifests_root)

    data_dirs = discover_data_dirs(raw_root)
    print(f"[dynamic] data dirs found: {len(data_dirs)}")

    all_rows: list[dict[str, Any]] = []
    all_color_rows: list[dict[str, Any]] = []
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
        )
        all_rows.extend(rows)
        all_color_rows.extend(color_rows)
        print(f"[rows] {out_path} ({len(rows)} rows)")

    all_data_path = sop_root / args.all_data_name
    write_all_data(all_rows, all_data_path)
    print(f"[write] {all_data_path} ({len(all_rows)} rows)")

    all_colors_path = sop_root / args.all_colors_name
    write_all_colors(all_color_rows, all_colors_path)
    print(f"[write] {all_colors_path} ({len(all_color_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
