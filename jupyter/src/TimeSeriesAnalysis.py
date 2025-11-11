# TimeSeriesAnalysis.py
# ------------------------------------------------------------
# Utilities to read SOP JSON outputs, build ensemble statistics,
# compute plateau estimators, and cache mean curves to JSON.
# The cache lives one directory above the raw /data folder.
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import json
import glob
import math
from datetime import datetime
from pathlib import Path
from scipy.interpolate import interp1d
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import random

# ============================================================
# Paths / Regex helpers
# ============================================================

def create_folder(folder_path):
    """
    Creates the folder if it does not already exist.

    Args:
        folder_path (str): Path to the folder to be created.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

# Accepts k/rho in normal float or scientific notation (e.g., 1.0e-04)
PARAMS_RE = re.compile(
    r"""
    (?P<type_perc>[A-Za-z]+)_percolation
    /num_colors_(?P<num_colors>\d+)
    /dim_(?P<dim>\d+)
    /L_(?P<L>\d+)
    /NT_constant/NT_(?P<Nt>\d+)
    /k_(?P<k>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)
    /rho_(?P<rho>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)
    /data
    """,
    re.X,
)

def parse_params_from_path(path: str) -> dict | None:
    """
    Extract type_perc, num_colors, dim, L, Nt, k, rho from a folder path.
    Returns a typed dict or None if it does not match.
    """
    p = path.replace("\\", "/")
    m = PARAMS_RE.search(p)
    if not m:
        return None
    gd = m.groupdict()
    return {
        "type_perc": gd["type_perc"],
        "num_colors": int(gd["num_colors"]),
        "dim": int(gd["dim"]),
        "L": int(gd["L"]),
        "Nt": int(gd["Nt"]),
        "k": float(gd["k"]),
        "rho": float(gd["rho"]),
    }

def create_folder(folder_path: str) -> None:
    """Create folder if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Filename pattern seen in your data
_FNAME_RE = re.compile(
    r"P0_([0-9]*\.?[0-9]+(?:e[+\-]?[0-9]+)?)_p0_([0-9]*\.?[0-9]+(?:e[+\-]?[0-9]+)?)_seed_(\d+)\.json$",
    re.IGNORECASE,
)

def parse_p0_from_filename(path: str) -> float | None:
    """Extract p0 from your file naming pattern."""
    m = _FNAME_RE.search(os.path.basename(path))
    if not m:
        return None
    _, p0_str, _ = m.groups()
    try:
        return float(p0_str)
    except Exception:
        return None


# ============================================================
# Reading raw JSON (robust to empty/corrupted files)
# ============================================================

def read_orders_one_file(file_path: str) -> list[tuple[int | None, np.ndarray, np.ndarray | None]]:
    """
    Read one SOP result JSON and return a list of tuples:
      (order_percolation, p_t_array, N_t_array_or_None)

    - Ignores empty/corrupted files and malformed entries.
    - Truncates series to the shortest length if pt/nt differ.
    """
    try:
        with open(file_path, "r") as f:
            obj = json.load(f)
    except Exception:
        return []

    out = []
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        for item in obj["results"]:
            order = item.get("order_percolation", None)
            d = item.get("data", {})
            if order is None or "time" not in d or "pt" not in d:
                continue

            p = np.asarray(d["pt"], float)
            n_arr = np.asarray(d["nt"], float) if "nt" in d else None
            n = min(len(p), len(n_arr)) if n_arr is not None else len(p)
            if n <= 0:
                continue
            p = p[:n]
            n_arr = n_arr[:n] if n_arr is not None else None
            # store int(order) only if it's integer-like
            order_val = int(order) if isinstance(order, (int, np.integer)) else None
            out.append((order_val, p, n_arr))
    return out  # possibly []

def read_orders_one_file_fix(file_path: str) -> dict:
    """
    Read one SOP result JSON and return a dictionary with all properties of the JSON file.
    """
    try:
        with open(file_path, "r") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return {}

    # Verifica se o arquivo contém a chave 'results' e, se sim, processa
    if isinstance(obj, dict) and "results" in obj:
        results = obj["results"]    
        
        # Retorna o dicionário completo para que você possa acessar todas as propriedades
        return obj
    else:
        print("Estrutura do arquivo inválida")
        return {}


# ============================================================
# Ensemble building
# ============================================================

def interp_to_grid(t_src: np.ndarray, x_src: np.ndarray, t_grid: np.ndarray,
                   kind: str = "linear", fill_value: float = np.nan) -> np.ndarray:
    """Interpolate a single run onto a common time grid."""
    f = interp1d(t_src, x_src, kind=kind, bounds_error=False, fill_value=fill_value)
    return f(t_grid)

def ensemble_stats(times_list: list[np.ndarray],
                   values_list: list[np.ndarray],
                   t_grid: np.ndarray | None = None,
                   burn_in: float | None = None) -> dict:
    """
    Build ensemble statistics over runs on a common time grid.

    Returns a dict with:
      - t_grid (T,)
      - mean (T,), std (T,), sem (T,)
      - ci95 = (low (T,), high (T,))
      - N_per_t (T,)  number of valid runs at each time
      - matrix (M, T) interpolated runs (NaN outside support)
      - valid_mask (M, T) boolean mask of valid entries

    If there are no valid series, returns empty arrays (not None).
    """
    assert len(times_list) == len(values_list)
    M = len(times_list)

    def _empty():
        e = np.array([], float)
        return {
            "t_grid": e, "mean": e, "std": e, "sem": e,
            "ci95": (e, e), "N_per_t": e,
            "matrix": np.empty((0, 0), float),
            "valid_mask": np.empty((0, 0), bool),
        }

    if M == 0:
        return _empty()

    # burn-in and sorting
    t_proc, x_proc = [], []
    for t, x in zip(times_list, values_list):
        t = np.asarray(t, float)
        x = np.asarray(x, float)
        if t.size == 0 or x.size == 0:
            continue
        mask = (t >= burn_in) if burn_in is not None else np.ones_like(t, bool)
        t, x = t[mask], x[mask]
        if t.size == 0 or x.size == 0:
            continue
        idx = np.argsort(t)
        t_proc.append(t[idx]); x_proc.append(x[idx])

    if len(t_proc) == 0:
        return _empty()

    # pick common grid if not provided
    if t_grid is None:
        t0 = max(t[0] for t in t_proc)
        t1 = min(t[-1] for t in t_proc)
        if not (t1 > t0):
            return _empty()
        ngrid = min(int(np.median([len(t) for t in t_proc])), 2000)
        ngrid = max(ngrid, 50)  # ensure a minimum resolution
        t_grid = np.linspace(t0, t1, ngrid)

    # interpolate each run
    X = np.vstack([interp_to_grid(t, x, t_grid) for t, x in zip(t_proc, x_proc)])
    valid = ~np.isnan(X)

    # pointwise stats (sample std/sem ignoring NaNs)
    N = valid.sum(axis=0).astype(float)
    mean = np.nanmean(X, axis=0)

    def nanstd_sample(arr, axis=0):
        m = np.nanmean(arr, axis=axis)
        diff2 = (arr - np.expand_dims(m, axis=axis)) ** 2
        sse = np.nansum(diff2, axis=axis)
        n = np.sum(~np.isnan(arr), axis=axis)
        out = np.full_like(m, np.nan, float)
        ok = n > 1
        out[ok] = np.sqrt(sse[ok] / (n[ok] - 1))
        return out

    std = nanstd_sample(X, axis=0)

    sem = np.full_like(mean, np.nan)
    ok = N > 0
    sem[ok] = std[ok] / np.sqrt(N[ok])

    ci95_low  = mean - 1.96 * sem
    ci95_high = mean + 1.96 * sem

    return {
        "t_grid": t_grid,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95": (ci95_low, ci95_high),
        "N_per_t": N,
        "matrix": X,
        "valid_mask": valid,
    }


# ============================================================
# Basic estimators / formatting
# ============================================================

def tail_mean(x: np.ndarray, tail_frac: float = 0.30) -> float:
    """Mean over the last tail_frac of the series (ignores NaN)."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return np.nan
    start = int((1.0 - tail_frac) * n)
    start = max(0, min(start, n - 1))
    seg = x[start:]
    return np.nan if seg.size == 0 else np.nanmean(seg)

def mean_and_se_across_runs(values: np.ndarray) -> tuple[float, float, int]:
    """Mean and standard error over runs (unweighted)."""
    x = np.asarray(values, float)
    M = x.size
    if M == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(x))
    se   = float(np.std(x, ddof=1) / np.sqrt(M)) if M > 1 else np.nan
    return mean, se, M

def tail_mean_per_run(data_block: dict, t_from: float = 300) -> np.ndarray:
    """
    Given an ensemble dict (from `ensemble_stats`), returns a vector (M,)
    with per-run temporal means over the plateau region t >= t_from.
    """
    t = data_block["t_grid"]
    X = data_block["matrix"]  # shape (M_runs, T_grid)
    if t.size == 0 or X.size == 0:
        return np.array([])
    mask = (t >= t_from)
    if not np.any(mask):
        mask = slice(None)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(X[:, mask], axis=1)
    return m[~np.isnan(m)]

def lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation (ignores NaN)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    x0, x1 = x[:-1], x[1:]
    x0 = x0 - x0.mean()
    x1 = x1 - x1.mean()
    denom = np.sqrt(np.sum(x0**2) * np.sum(x1**2))
    return float(np.sum(x0 * x1) / denom) if denom > 0 else 0.0

def weighted_tail_from_sem(data_block: dict, t_from: float = 300, corr_correct: bool = True
                           ) -> tuple[float, float, int, int]:
    """
    Weighted mean over time using ensemble SEM(t) as weights:
      w_t = 1 / SEM(t)^2
    Optionally applies a simple AR(1) correction using the lag-1
    autocorrelation estimated on the weighted-mean series.

    Returns: (mean, corrected_SE, N_eff, N_points_used)
    """
    t    = data_block["t_grid"]
    mean = data_block["mean"]
    sem  = data_block["sem"]

    if t.size == 0:
        return np.nan, np.nan, 0, 0

    mask = (t >= t_from) & np.isfinite(mean) & np.isfinite(sem) & (sem > 0)
    if not np.any(mask):
        return np.nan, np.nan, 0, 0

    m = mean[mask]
    s = sem[mask]
    w = 1.0 / (s ** 2)

    m_w     = float(np.sum(w * m) / np.sum(w))
    se_naiv = float(1.0 / np.sqrt(np.sum(w)))

    if not corr_correct:
        return m_w, se_naiv, int(len(m)), int(len(m))

    rho1  = lag1_autocorr(m)
    N     = len(m)
    Neff  = int(np.clip(N * (1 - rho1) / (1 + rho1), 1, N))
    se_cor = float(se_naiv * np.sqrt(N / max(Neff, 1)))
    return m_w, se_cor, Neff, N


# ============================================================
# Optional helpers used elsewhere
# ============================================================

def wilson_ci(M: int, N: int, z: float = 1.96) -> tuple[float, float, float]:
    """
    Wilson binomial proportion confidence interval (centered as Wilson score).
    Returns (p_hat, low, high). If N==0 returns (nan, nan, nan).
    """
    if N <= 0:
        return (np.nan, np.nan, np.nan)
    phat = M / N
    denom = 1 + z * z / N
    center = (phat + z * z / (2 * N)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / N + z * z / (4 * N * N)) / denom
    return phat, max(0.0, center - margin), min(1.0, center + margin)

def list_rho_values(
    type_perc: str,
    num_colors: int,
    dim: int,
    L: int,
    Nt: int,
    k: float,
    base_root: str = "../Data",
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-15,
) -> list[float]:
    """
    List all rho values that exist under:
      ../Data/{type_perc}_percolation/num_colors_{num_colors}/dim_{dim}/L_{L}/
      NT_constant/NT_{Nt}/k_*/rho_*/data
    and match the fixed parameters (including k≈ given k).
    """
    base = (
        Path(base_root)
        / f"{type_perc}_percolation"
        / f"num_colors_{num_colors}"
        / f"dim_{dim}"
        / f"L_{L}"
        / "NT_constant"
        / f"NT_{Nt}"
    )
    if not base.exists():
        return []

    rhos = []
    for data_dir in base.glob("k_*/rho_*/data"):
        if not data_dir.is_dir():
            continue
        m = PARAMS_RE.search(str(data_dir.as_posix()))
        if not m:
            continue
        gd = m.groupdict()
        if gd["type_perc"] != type_perc:
            continue
        if int(gd["num_colors"]) != num_colors:
            continue
        if int(gd["dim"]) != dim:
            continue
        if int(gd["L"]) != L:
            continue
        if int(gd["Nt"]) != Nt:
            continue

        k_here = float(gd["k"])
        if not math.isclose(k_here, float(k), rel_tol=rel_tol, abs_tol=abs_tol):
            continue
        rhos.append(float(gd["rho"]))
    return sorted(set(rhos))


# ============================================================
# Filename filtering (match p0 in multiple formats)
# ============================================================

def _filename_matches_p0(fname: str, p0: float) -> bool:
    """
    Try to match a given p0 inside a filename in common formats:
      1.0 / 1.00 / 1.000, with '.' or '_' separators.
    """
    base = os.path.basename(fname)
    reps = {f"{p0:.0f}", f"{p0:.1f}", f"{p0:.2f}", f"{p0:.3f}", f"{p0}"}
    patterns = []
    for r in reps:
        patterns.append(re.escape(r))
        patterns.append(re.escape(r.replace(".", "_")))
    rx = re.compile(r"(?:^|[^0-9])(" + "|".join(patterns) + r")(?:[^0-9]|$)")
    return rx.search(base) is not None

def _safe_p0_tag(p0: float) -> str:
    """Tag for filenames: 1.000 -> '1', 0.700 -> '0.7', 0.333 -> '0.333' then replace '.' with '_'."""
    s = f"{p0:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "_")


# ============================================================
# Caching — (A) per-p0 cache and (B) bundle of multiple p0s
# ============================================================

_CACHE_VERSION = 1

def _cache_dir(type_perc: str, num_colors: int, dim: int, L: int, NT: int, k: float, rho: float) -> str:
    """Directory one level above /data that holds cache files."""
    return (
        f"../Data/{type_perc}_percolation/num_colors_{num_colors}/dim_{dim}/L_{L}/"
        f"NT_constant/NT_{NT}/k_{k:.1e}/rho_{rho:.4e}"
    )

def _cache_file_path(type_perc: str, num_colors: int, dim: int, L: int, NT: int, k: float, rho: float, p0: float) -> str:
    """Cache file for a single p0."""
    return os.path.join(_cache_dir(type_perc, num_colors, dim, L, NT, k, rho),
                        f"mean_properties_p0_{_safe_p0_tag(p0)}.json")

def _bundle_file_path(type_perc: str, num_colors: int, dim: int, L: int, NT: int, k: float, rho: float) -> str:
    """Single bundle cache containing many p0 entries."""
    return os.path.join(_cache_dir(type_perc, num_colors, dim, L, NT, k, rho),
                        "mean_properties_bundle.json")

def _latest_mtime(folder: str) -> float:
    files = glob.glob(os.path.join(folder, "*.json"))
    return max((os.path.getmtime(f) for f in files), default=0.0)

def _stats_to_jsonable(stats: dict) -> dict:
    """Convert numpy arrays to lists before saving."""
    return {
        "t_grid": stats["t_grid"].tolist(),
        "mean":   stats["mean"].tolist(),
        "std":    stats["std"].tolist(),
        "sem":    stats["sem"].tolist(),
        "ci95":   [stats["ci95"][0].tolist(), stats["ci95"][1].tolist()],
        "N_per_t": stats["N_per_t"].tolist(),
    }

def _stats_from_json(obj: dict) -> dict:
    """Convert lists back to numpy arrays when loading."""
    t_grid = np.asarray(obj.get("t_grid", []), float)
    return {
        "t_grid": t_grid,
        "mean":   np.asarray(obj.get("mean", []), float),
        "std":    np.asarray(obj.get("std", []), float),
        "sem":    np.asarray(obj.get("sem", []), float),
        "ci95":  (np.asarray(obj.get("ci95", [[], []])[0], float),
                  np.asarray(obj.get("ci95", [[], []])[1], float)),
        "N_per_t": np.asarray(obj.get("N_per_t", []), float),
        # we do not store the full matrix/valid_mask to keep files small:
        "matrix": np.empty((0, t_grid.size), float),
        "valid_mask": np.empty((0, t_grid.size), bool),
    }


# ------------------------ (A) per-p0 cache ------------------------

def mean_properties(type_perc: str, num_colors: int, dim: int, L: int, NT: int,
                    k: float, rho: float, p0: float) -> tuple[dict, dict]:
    """
    Compute (or load from cache) ensemble stats for a SINGLE p0.

    Returns (p_stats, N_stats) dicts with keys like 't_grid', 'mean', 'std', etc.
    Cache file: mean_properties_p0_<p0_tag>.json (one dir above /data).
    """
    path_files = (
        f"../Data/{type_perc}_percolation/num_colors_{num_colors}/dim_{dim}/L_{L}/"
        f"NT_constant/NT_{NT}/k_{k:.1e}/rho_{rho:.4e}/data"
    )
    save_dir = _cache_dir(type_perc, num_colors, dim, L, NT, k, rho)
    cache_fp = _cache_file_path(type_perc, num_colors, dim, L, NT, k, rho, p0)

    # try reading cache
    if os.path.isfile(cache_fp):
        try:
            with open(cache_fp, "r") as f:
                cached = json.load(f)
            meta = cached.get("_meta", {})
            if (
                meta.get("version") == _CACHE_VERSION
                and meta.get("type_perc") == type_perc
                and meta.get("num_colors") == num_colors
                and meta.get("dim") == dim
                and meta.get("L") == L
                and meta.get("NT") == NT
                and float(meta.get("k")) == float(k)
                and float(meta.get("rho")) == float(rho)
                and float(meta.get("p0")) == float(p0)
            ):
                return _stats_from_json(cached.get("stats_p", {})), _stats_from_json(cached.get("stats_N", {}))
        except Exception:
            pass  # ignore corrupted cache and recompute

    # compute from raw files
    all_files = glob.glob(os.path.join(path_files, "*.json"))
    times_list_p, p_values_list = [], []
    times_list_N, N_values_list = [], []

    for file in all_files:
        if not _filename_matches_p0(file, p0):
            continue
        records = read_orders_one_file(file)
        if not records:
            continue

        order, p, nt = records[0]
        p = np.asarray(p, float)
        if p.size == 0:
            continue

        if order is None:
            t = np.arange(p.shape[0], dtype=float)
        else:
            t = np.asarray(order)
            if t.ndim == 0 or t.size <= 1:
                t = np.arange(p.shape[0], dtype=float)

        # p(t)
        m_p = min(t.size, p.size)
        if m_p > 0:
            t_p = t[:m_p].astype(float, copy=False)
            p_p = p[:m_p]
            if m_p > 1 and not np.all(np.diff(t_p) >= 0):
                idx = np.argsort(t_p)
                t_p, p_p = t_p[idx], p_p[idx]
            times_list_p.append(t_p)
            p_values_list.append(p_p)

        # N(t)
        if nt is not None:
            nt = np.asarray(nt, float)
            m_n = min(t.size, nt.size)
            if m_n > 0:
                t_n = t[:m_n].astype(float, copy=False)
                nt_n = nt[:m_n]
                if m_n > 1 and not np.all(np.diff(t_n) >= 0):
                    idx = np.argsort(t_n)
                    t_n, nt_n = t_n[idx], nt_n[idx]
                times_list_N.append(t_n)
                N_values_list.append(nt_n)

    stats_p = ensemble_stats(times_list_p, p_values_list)
    stats_N = ensemble_stats(times_list_N, N_values_list)

    # save cache
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "_meta": {
            "version": _CACHE_VERSION,
            "type_perc": type_perc,
            "num_colors": num_colors,
            "dim": dim,
            "L": L,
            "NT": NT,
            "k": float(k),
            "rho": float(rho),
            "p0": float(p0),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        "stats_p": _stats_to_jsonable(stats_p),
        "stats_N": _stats_to_jsonable(stats_N),
    }
    try:
        with open(cache_fp, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save cache at {cache_fp}: {e}")

    return stats_p, stats_N

# --------- helpers ---------
FNAME_RE = re.compile(r"P0_(?P<P0>\d+\.\d+)_p0_(?P<p0>\d+\.\d+)_seed_(?P<seed>\d+)\.json$")

def parse_fname(fname: str) -> Tuple[float, float, int] | None:
    m = FNAME_RE.search(os.path.basename(fname))
    if not m:
        return None
    return float(m.group("P0")), float(m.group("p0")), int(m.group("seed"))

def load_orders(path: str) -> Dict[int, Dict[str, Any]]:
    with open(path, "r") as f:
        obj = json.load(f)

    # detectar a lista de blocos
    if isinstance(obj, list):
        blocks = obj
    elif isinstance(obj, dict):
        if "orders" in obj and isinstance(obj["orders"], list):
            blocks = obj["orders"]
        elif "blocks" in obj and isinstance(obj["blocks"], list):
            blocks = obj["blocks"]
        else:
            # fallback: pegar a primeira lista de dicts que tenha 'order_percolation'
            blocks = []
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict) and "order_percolation" in v[0]:
                    blocks = v
                    break
    else:
        blocks = []

    out: Dict[int, Dict[str, Any]] = {}
    for b in blocks:
        if not isinstance(b, dict):
            continue
        ordk = b.get("order_percolation")
        if ordk is None:
            continue
        data = b.get("data", {})
        if not data:
            data = {k: v for k, v in b.items() if k not in ("order_percolation",)}
        out[int(ordk)] = data
    return out

def _mean_series(list_of_lists: List[List[float]]) -> List[float]:
    if not list_of_lists:
        return []
    min_len = min(len(x) for x in list_of_lists)
    if min_len == 0:
        return []
    arr = np.vstack([np.asarray(x[:min_len], dtype=float) for x in list_of_lists])
    return arr.mean(axis=0).tolist()

def _mean_sem(vals: List[float]) -> Dict[str, float]:
    """média e erro-padrão da média (SEM = s/√n, com ddof=1 se n>1)."""
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "sem": float("nan")}
    if n == 1:
        return {"mean": float(vals[0]), "sem": 0.0}
    arr = np.asarray(vals, dtype=float)
    return {"mean": float(arr.mean()), "sem": float(arr.std(ddof=1) / np.sqrt(n))}

def average_by_order(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys_series = ("time", "pt", "nt", "Mt", "Smax", "Ni", "chi")
    # escalares para reportar média + SEM
    keys_scalar_sem = ("time_percolation", "shortest_path_lin")

    out: Dict[str, Any] = {}

    # séries (média elemento-a-elemento)
    for k in keys_series:
        series = [d.get(k) for d in dicts if isinstance(d.get(k), list)]
        out[k] = _mean_series(series)

    # escalares com média + SEM
    stats = {}
    for k in keys_scalar_sem:
        vals = [d.get(k) for d in dicts if isinstance(d.get(k), (int, float))]
        if vals:
            stats[k] = _mean_sem(vals)
    if stats:
        out["stats"] = stats

    # NADA de 'color' e 'rho' no bloco data
    return out


# --------- pipeline principal ---------
def compute_means_for_folder(
    type_perc: str,
    num_colors: int,
    dim: int,
    L: int,
    NT: int,
    k: float,
    rho: float,
    p0_list: List[float],
) -> str:
    """
    Agrega todas as seeds para cada p0 em p0_list, alinha por order_percolation,
    calcula médias e SEM (para time_percolation e shortest_path_lin),
    e salva UM arquivo final 'properties_mean_bundle.json' uma pasta acima de 'data/'.
    Retorna o caminho do JSON salvo.
    """
    base_dir = "../Data"
    data_dir = os.path.join(
        base_dir,
        f"{type_perc}_percolation",
        f"num_colors_{num_colors}",
        f"dim_{dim}",
        f"L_{L}",
        "NT_constant",
        f"NT_{NT}",
        f"k_{k:.1e}",
        f"rho_{rho:.4e}",
        "data",
    )
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Pasta de dados não encontrada: {data_dir}")

    bundle = {
        "meta": {
            "type_perc": type_perc,
            "num_colors": num_colors,
            "dim": dim,
            "L": L,
            "NT": NT,
            "k": float(k),
            "rho": float(rho),
            "base_dir": os.path.dirname(data_dir),  # uma pasta acima de 'data'
        },
        "p0_groups": []
    }

    for p0 in p0_list:
        pattern = os.path.join(data_dir, f"P0_*_p0_{p0:.2f}_seed_*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[aviso] Sem arquivos para p0={p0:.2f} em {data_dir}")
            continue

        per_order: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        seeds = []

        for fp in files:
            parsed = parse_fname(fp)
            if not parsed:
                continue
            _, p0_val, seed = parsed
            seeds.append(seed)

            orders = load_orders(fp)
            for ordk, data in orders.items():
                per_order[ordk].append(data)

        # calcula médias por ordem
        mean_by_order: Dict[int, Dict[str, Any]] = {}
        for ordk, lst in per_order.items():
            mean_by_order[ordk] = average_by_order(lst)

        # montar bloco do p0
        orders_blocks = [
            {"order_percolation": int(ordk), "data": mean_by_order[ordk]}
            for ordk in sorted(mean_by_order.keys())
        ]
        bundle["p0_groups"].append({
            "p0_value": float(p0),
            "num_seeds": len(set(seeds)),
            "seeds": sorted(set(seeds)),
            "orders": orders_blocks
        })

        print(f"[ok] p0={p0:.2f}: {len(files)} arquivos agregados")

    # salvar UMA pasta acima de 'data/'
    out_dir = os.path.dirname(data_dir)
    out_path = os.path.join(out_dir, "properties_mean_bundle.json")
    with open(out_path, "w") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f"[salvo] {out_path}")
    return out_path

# LOAD ONE FILE JSON

def _safe_series(d: dict) -> dict:
    """Garante presença de chaves opcionais como listas vazias."""
    out = dict(d)
    out.setdefault("Smax", [])
    out.setdefault("Ni",   [])
    out.setdefault("chi",  [])
    return out

def load_perc_json(path_json: str | Path) -> Tuple[dict, Dict[int, dict]]:
    """
    Lê o JSON gerado pelo save_percolation_json.
    Retorna:
      - bundle: dict completo lido
      - orders: dict {order_percolation:int -> data:dict} com séries normalizadas
    Suporta:
      - layout 'flat' com "results"
      - layout antigo com "p0_groups"
    """
    path_json = Path(path_json)
    with path_json.open("r") as f:
        bundle = json.load(f)

    orders: Dict[int, dict] = {}

    if "results" in bundle:  # layout atual (um arquivo por execução)
        for item in bundle["results"]:
            order = int(item["order_percolation"])
            data  = _safe_series(item["data"])
            orders[order] = data

    elif "p0_groups" in bundle:  # layout antigo (vários p0 em um arquivo)
        # achata todos os grupos em um único índice por ordem
        for grp in bundle["p0_groups"]:
            for o in grp.get("orders", []):
                order = int(o["order_percolation"])
                data  = _safe_series(o["data"])
                orders[order] = data
    else:
        raise ValueError("JSON não possui 'results' nem 'p0_groups'.")

    return bundle, orders

# (opcional) helper para inferir p0 do nome do arquivo, se você quiser
def infer_p0_from_filename(path_json: str | Path) -> float | None:
    """
    Tenta extrair 'p0' do nome do arquivo (ex.: '..._p0_0.30_...').
    Retorna float ou None se não encontrar.
    """
    m = re.search(r"_p0_([0-9]*\.?[0-9]+)", Path(path_json).name)
    return float(m.group(1)) if m else None

# HOW TO USE
# bundle_path = f"../Data/bond_percolation/num_colors_{NUM_COLORS}/dim_{DIM}/L_{L}/NT_constant/NT_{NT}/k_{K:.1e}/rho_{RHO:.4e}/data/"
# filename = ex: "P0_0.10_p0_0.30_seed_27324716.json"
# bundle, p0_index_like = load_perc_json(bundle_path + filename)
# orders = sorted(p0_index_like.keys())  # ex.: [1,2,3,4]
# p0_index[int(p0_value)][int(order)]

def load_bundle(path_json: str | Path):
    path_json = Path(path_json)
    with path_json.open("r") as f:
        bundle = json.load(f)

    # índice rápido: p0_value -> (order -> data_dict)
    p0_index = {}
    for g in bundle["p0_groups"]:
        p0 = float(g["p0_value"])
        orders = { int(o["order_percolation"]): o["data"] for o in g["orders"] }
        p0_index[p0] = orders
    return bundle, p0_index

def load_bundle_old(
    filepath: str | Path,
    *,
    include_nt: bool = True
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Lê um arquivo .json de simulação e retorna:
      - metas: lista de dicts com metadados por 'order_percolation'
      - df: DataFrame longo com colunas ['order','t','pt','nt'] (nt pode vir NaN)

    Parâmetros
    ----------
    filepath : str | Path
        Caminho do arquivo JSON.
    include_nt : bool
        Se True, tenta carregar a série 'nt' quando existir; caso contrário, preenche NaN.

    Retorno
    -------
    metas : list[dict]
        Para cada entrada em 'results', inclui:
        {'order', 'color', 'rho', 'M_size', 'time_percolation', 'n_time', 'has_nt'}.
    df : pandas.DataFrame
        Linhas por tempo: colunas ['order','t','pt','nt'] (nt pode ser NaN).
    """
    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict) or "results" not in obj or not isinstance(obj["results"], list):
        raise ValueError("JSON inesperado: não achei lista 'results'.")

    metas: List[Dict[str, Any]] = []
    rows = []

    for item in obj["results"]:
        order = item.get("order_percolation")
        d = item.get("data", {})
        if order is None or not isinstance(d, dict):
            continue

        time = d.get("time", None)
        pt   = d.get("pt", None)
        nt   = d.get("nt", None) if include_nt else None

        # Converte para arrays, se existirem
        time = np.asarray(time) if time is not None else None
        pt   = np.asarray(pt)   if pt   is not None else None
        nt   = np.asarray(nt)   if nt   is not None else None

        # Valida tamanhos mínimos
        if time is None or pt is None:
            # sem séries, apenas meta
            n_time = 0
        else:
            n_time = int(min(len(time), len(pt)))
            if nt is not None:
                n_time = int(min(n_time, len(nt)))
            # recorta para o mesmo comprimento
            time = time[:n_time]
            pt   = pt[:n_time]
            if nt is not None:
                nt = nt[:n_time]

            # Empilha linhas
            if n_time > 0:
                if nt is None:
                    # sem nt: preenche NaN
                    for t, p in zip(time, pt):
                        rows.append((int(order), int(t), float(p), np.nan))
                else:
                    for t, p, n in zip(time, pt, nt):
                        rows.append((int(order), int(t), float(p), float(n)))

        metas.append({
            "order": int(order),
            "color": d.get("color", None),
            "rho": d.get("rho", None),
            "M_size": d.get("M_size", None),
            "time_percolation": d.get("time_percolation", None),
            "n_time": n_time,
            "has_nt": (nt is not None)
        })

    df = pd.DataFrame(rows, columns=["order", "t", "pt", "nt"]) if rows else \
         pd.DataFrame(columns=["order", "t", "pt", "nt"])

    return metas, df

# AVARAGE TO OLD STRUCT JSON
# -----------------------------
# Helpers para parsing/carregamento
# -----------------------------

_FNAME_RE = re.compile(
    r"P0_(?P<P0>[-+]?\d+(?:\.\d+)?)_p0_(?P<p0>[-+]?\d+(?:\.\d+)?)_seed_(?P<seed>\d+)\.json$",
    re.IGNORECASE
)

def _parse_fname(filepath: str) -> Optional[Tuple[float, float, int]]:
    """Extrai (P0, p0, seed) do nome do arquivo."""
    m = _FNAME_RE.search(os.path.basename(filepath))
    if not m:
        return None
    try:
        P0 = float(m.group("P0"))
        p0 = float(m.group("p0"))
        seed = int(m.group("seed"))
        return (P0, p0, seed)
    except Exception:
        return None


def _load_orders_new(filepath: str) -> Dict[int, Dict[str, Any]]:
    """
    Lê um JSON no novo formato e devolve:
       { order: {"time": [...], "pt":[...], "nt":[...]?, "M_size": escalar?, "time_percolation": escalar?} }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict) or "results" not in obj or not isinstance(obj["results"], list):
        raise ValueError(f"JSON inesperado (sem 'results'): {filepath}")

    out: Dict[int, Dict[str, Any]] = {}
    for item in obj["results"]:
        ordk = item.get("order_percolation", None)
        data = item.get("data", {})
        if ordk is None or not isinstance(data, dict):
            continue

        # Normaliza campos
        time = data.get("time", [])
        pt   = data.get("pt", [])
        nt   = data.get("nt", None)  # opcional
        Msz  = data.get("M_size", None)
        tperc= data.get("time_percolation", None)

        out[int(ordk)] = {
            "time": list(time) if time is not None else [],
            "pt":   list(pt)   if pt   is not None else [],
            "nt":   (list(nt)  if nt   is not None else None),
            "M_size": Msz,
            "time_percolation": tperc,
        }
    return out


def _mean_sem_1d(values: List[float]) -> Tuple[float, float, int]:
    """média e SEM de uma lista (ignora None/NaN)."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    if n == 1:
        return (float(arr.mean()), 0.0, 1)
    return (float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(n)), n)


def _avg_series_across_seeds(items: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """
    Média/SEM ao longo das seeds para uma SÉRIE (pt ou nt).
    Alinha por índice (usa comprimento mínimo comum).
    Retorna dict com time, <key>_mean, <key>_sem.
    """
    series = []
    times  = []
    for d in items:
        s = d.get(key, None)
        t = d.get("time", None)
        if s is None or t is None:
            continue
        s = np.asarray(s, dtype=float)
        t = np.asarray(t, dtype=int)
        n = min(len(s), len(t))
        if n > 0:
            series.append(s[:n])
            times.append(t[:n])

    if not series:
        return {"time": [], f"{key}_mean": [], f"{key}_sem": [], "n_seeds_series": 0}

    # comprimentos podem variar; usa o mínimo comum
    min_n = min(s.shape[0] for s in series)
    series = [s[:min_n] for s in series]
    times  = [tt[:min_n] for tt in times]

    # assume tempos coerentes; se não, escolhe a primeira sequência
    time_ref = times[0]
    mat = np.stack(series, axis=0)  # (S, T)
    mean = np.nanmean(mat, axis=0)
    if mat.shape[0] > 1:
        sem  = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
    else:
        sem  = np.zeros_like(mean)

    return {
        "time": time_ref.tolist(),
        f"{key}_mean": mean.astype(float).tolist(),
        f"{key}_sem":  sem.astype(float).tolist(),
        "n_seeds_series": int(mat.shape[0])
    }


def _average_by_order_new(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Faz a média para um conjunto de dicionários (uma por seed) de UMA dada ordem.
    - Séries: pt (sempre), nt (se existir).
    - Escalares: time_percolation e M_size (média e SEM).
    """
    # séries
    pt_blk = _avg_series_across_seeds(lst, "pt")
    nt_blk = _avg_series_across_seeds([d for d in lst if d.get("nt", None) is not None], "nt")

    # escalares
    m_tp, sem_tp, n_tp = _mean_sem_1d([d.get("time_percolation", None) for d in lst])
    m_ms, sem_ms, n_ms = _mean_sem_1d([d.get("M_size", None) for d in lst])

    out = {
        "time": pt_blk["time"],

        "pt_mean": pt_blk["pt_mean"],
        "pt_sem":  pt_blk["pt_sem"],
        "n_seeds_pt": pt_blk["n_seeds_series"],

        # nt pode não existir
        "nt_mean": nt_blk.get("nt_mean", []),
        "nt_sem":  nt_blk.get("nt_sem", []),
        "n_seeds_nt": nt_blk.get("n_seeds_series", 0),

        "time_percolation_mean": m_tp,
        "time_percolation_sem":  sem_tp,
        "n_seeds_time_perc":     n_tp,

        "M_size_mean": m_ms,
        "M_size_sem":  sem_ms,
        "n_seeds_M_size": n_ms,
    }
    return out


# -------------------------------------------------------
# Função principal: versão análoga para o novo JSON
# -------------------------------------------------------

# --- helpers que você já tem em outro lugar ---
# _parse_fname(fp) -> Optional[Tuple[float p0, float p0_val, int seed]]
# _load_orders_new(fp) -> Dict[int, Dict[str, Any]]  # por ordem: {"t":..., "pt":..., "nt":...}
# _average_by_order_new(lst) -> Dict[str, Any]       # média/SEM por ordem

import os, glob, json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any

# ----------------------------------------------------------
# 1) helpers já existentes que você mencionou
# ----------------------------------------------------------
def _bootstrap_mean_across_samples(sample_means: np.ndarray, n_boot: int = 20000, rng_seed: int = 12345):
    """Bootstrap ENTRE-SEEDS: amostra means com reposição e devolve (mu_boot, sd_boot)."""
    sample_means = np.asarray(sample_means, dtype=float)
    n = sample_means.size
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = sample_means[idx].mean(axis=1)
    return float(boot_means.mean()), float(boot_means.std(ddof=1))

# ----------------------------------------------------------
# 2) janela deslizante (como você forneceu)
# ----------------------------------------------------------
def rolling_mean_std(t, y, window: int):
    """
    Retorna (t_centrado, media, desvio_padrao) para uma janela deslizante de tamanho 'window'.
    A série resultante fica centrada na janela.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if window < 1 or window > len(y):
        raise ValueError("window fora do intervalo válido")

    c  = np.cumsum(np.insert(y, 0, 0.0))
    c2 = np.cumsum(np.insert(y*y, 0, 0.0))

    mean = (c[window:] - c[:-window]) / window
    var  = (c2[window:] - c2[:-window]) / window - mean**2
    std  = np.sqrt(np.clip(var, 0, None))

    # centraliza no tempo (para janela par funciona como 'centered' padrão)
    t_center = t[(window-1)//2 : len(t) - window//2]
    return t_center, mean, std

# ----------------------------------------------------------
# 3) novas rotinas auxiliares para o modo "entre-seeds (concatenado)"
# ----------------------------------------------------------
def _concat_seed_series_for_pc(pt_by_seed: Dict[int, List[np.ndarray]],
                               t_by_seed: Dict[int, List[np.ndarray]],
                               x_max: float | None) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Concatena, por seed, as séries pt e t (já com x_max aplicado no loop de leitura),
    retornando {seed: {"t": t_concat, "pt": pt_concat}} apenas para seeds com dados.
    """
    out = {}
    for s, pts in pt_by_seed.items():
        if len(pts) == 0: 
            continue
        ts = t_by_seed.get(s, [])
        if len(ts) != len(pts):
            # segurança extra
            continue
        t_concat  = np.concatenate(ts)
        pt_concat = np.concatenate(pts)
        if t_concat.size > 0 and pt_concat.size == t_concat.size:
            out[s] = {"t": t_concat, "pt": pt_concat}
    return out

def _bootstrap_series_across_seeds(series_stack: np.ndarray,
                                   n_boot: int,
                                   rng_seed: int,
                                   batch_size: int = 128,
                                   use_float32: bool = True) -> (np.ndarray, np.ndarray):
    """
    Bootstrap ENTRE-SEEDS por ponto temporal, em streaming (sem alocar (n_boot, n_seeds, T)).
    - series_stack: shape (n_seeds, T)
    Retorna (mean_boot[T], std_boot[T]).
    """
    X = np.asarray(series_stack, dtype=np.float32 if use_float32 else np.float64)  # (n_seeds, T)
    n_seeds, T = X.shape
    rng = np.random.default_rng(rng_seed)

    # estatísticas online (Welford) ao longo das réplicas de bootstrap
    mean = np.zeros(T, dtype=np.float64)
    M2   = np.zeros(T, dtype=np.float64)
    count = 0

    # probabilidades uniformes para a Multinomial (bootstrap clássico com reposição)
    p = np.full(n_seeds, 1.0 / n_seeds, dtype=np.float64)

    # processa em lotes pequenos para não estourar RAM
    for start in range(0, n_boot, batch_size):
        b = min(batch_size, n_boot - start)
        # counts ~ Multinomial(n_seeds, p) para cada réplica (b, n_seeds)
        counts = rng.multinomial(n_seeds, pvals=p, size=b)   # int32
        # média bootstrap do tempo para cada réplica = (counts @ X) / n_seeds
        # (b, n_seeds) @ (n_seeds, T) -> (b, T)
        batch_means = (counts @ X) / float(n_seeds)

        # atualiza estatísticas online por réplica
        for i in range(b):
            count += 1
            delta = batch_means[i] - mean
            mean += delta / count
            M2   += delta * (batch_means[i] - mean)

        # libera rápido memória temporária
        del counts, batch_means

    std = np.sqrt(M2 / (count - 1)) if count > 1 else np.zeros_like(mean)
    return mean, std

def _bootstrap_tail_mean_across_seeds(series_stack: np.ndarray,
                                      t_center: np.ndarray,
                                      *,
                                      n_boot: int,
                                      rng_seed: int,
                                      tail_tmin: float | None = None,   # se dado, usa t >= tail_tmin
                                      tail_frac: float | None = 0.2,    # senão, usa última fração (ex.: 20%)
                                      batch_size: int = 128,
                                      use_float32: bool = True) -> tuple[float, float, int]:
    """
    Bootstrap ENTRE-SEEDS para a MÉDIA DA CAUDA (escalar).
    - series_stack: (n_seeds, T_comum) com as séries roladas por seed alinhadas
    - t_center: (T_comum,) tempos centrais já alinhados
    Retorna (mean_tail_boot, std_tail_boot, T_tail) onde T_tail é o nº de pontos usados na cauda.
    """
    X = np.asarray(series_stack, dtype=np.float32 if use_float32 else np.float64)
    t_center = np.asarray(t_center, dtype=np.float64)
    n_seeds, T = X.shape
    assert t_center.shape[0] == T

    # --- máscara de cauda ---
    if tail_tmin is not None:
        mask = (t_center >= float(tail_tmin))
    else:
        frac = 0.2 if tail_frac is None else float(tail_frac)
        frac = min(max(frac, 0.0), 1.0)
        start = int(round((1.0 - frac) * T))
        mask = np.zeros(T, dtype=bool)
        mask[start:] = True

    T_tail = int(mask.sum())
    if T_tail == 0:
        return float('nan'), float('nan'), 0

    rng = np.random.default_rng(rng_seed)
    p = np.full(n_seeds, 1.0 / n_seeds, dtype=np.float64)

    # estatísticas online (escalar por réplica)
    mean = 0.0
    M2 = 0.0
    count = 0

    for start in range(0, n_boot, batch_size):
        b = min(batch_size, n_boot - start)
        counts = rng.multinomial(n_seeds, pvals=p, size=b)  # (b, n_seeds)

        # (b, n_seeds) @ (n_seeds, T) -> (b, T); média por seed
        batch_means = (counts @ X) / float(n_seeds)

        # média temporal RESTRITA à cauda -> vetor (b,)
        tail_vals = batch_means[:, mask].mean(axis=1)

        # Welford online
        for v in tail_vals:
            count += 1
            delta = v - mean
            mean += delta / count
            M2   += delta * (v - mean)

        del counts, batch_means, tail_vals

    std = np.sqrt(M2 / (count - 1)) if count > 1 else 0.0
    return float(mean), float(std), T_tail


# ----------------------------------------------------------
# 4) sua função principal, agora com suporte a janela deslizante
#    - window_roll: int | None
# ----------------------------------------------------------
def compute_means_for_folder_new(
    type_perc: str,
    num_colors: int,
    dim: int,
    L: int,
    NT: int,
    k: float,
    rho: float,
    p0_list: List[float],
    *,
    x_max: float | None = None,     # limite temporal opcional (ex.: 5000)
    n_boot: int = 20000,            # iterações de bootstrap
    rng_seed: int = 12345,          # semente do bootstrap
    window_roll: int | None = None  # <- NOVO: tamanho da janela deslizante (ex.: 50). Se None, não calcula
) -> str:
    """
    Agrega todas as seeds para cada p0 em p0_list, alinha por order_percolation,
    calcula médias e SEM:
      - séries: pt (e nt se existir);
      - escalares: time_percolation e M_size;
    Além disso, calcula e salva no JSON:
      - p_{c,SOP} (bootstrap ENTRE-SEEDS, escalar)
      - (NOVO) p_{c,SOP}^{roll}(t): série rolada com incerteza bootstrap ENTRE-SEEDS por ponto
        em 'pc_sop_roll': {"t_center":[...], "mean":[...], "std_boot":[...], "n_seeds":..., "n_boot":..., "window":...}
      - (NOVO) por ordem, 'orders -> data -> pt_mean_roll' calculado da série média da ordem.
    """
    base_dir = "../Data"
    data_dir = os.path.join(
        base_dir,
        f"{type_perc}_percolation",
        f"num_colors_{num_colors}",
        f"dim_{dim}",
        f"L_{L}",
        "NT_constant",
        f"NT_{NT}",
        f"k_{k:.1e}",
        f"rho_{rho:.4e}",
        "data",
    )
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Pasta de dados não encontrada: {data_dir}")

    bundle: Dict[str, Any] = {
        "meta": {
            "type_perc": type_perc,
            "num_colors": num_colors,
            "dim": dim,
            "L": L,
            "NT": NT,
            "k": float(k),
            "rho": float(rho),
            "base_dir": os.path.dirname(data_dir),  # uma pasta acima de 'data'
            "x_max_used": None if x_max is None else float(x_max),
            "bootstrap": {"n_boot": int(n_boot), "rng_seed": int(rng_seed)},
            "rolling": {"window": None if window_roll is None else int(window_roll)}
        },
        "p0_groups": []
    }

    for p0 in p0_list:
        # padrão principal (2 casas) + fallback (1 casa)
        pattern = os.path.join(data_dir, f"P0_*_p0_{p0:.2f}_seed_*.json")
        files = sorted(glob.glob(pattern)) or sorted(
            glob.glob(os.path.join(data_dir, f"P0_*_p0_{p0:.1f}_seed_*.json"))
        )

        if not files:
            print(f"[aviso] Sem arquivos para p0={p0:.2f} em {data_dir}")
            continue

        per_order: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        seeds: List[int] = []

        # para p_c,SOP e p_c,SOP^roll (concatenado por seed)
        pt_by_seed: Dict[int, List[np.ndarray]] = defaultdict(list)
        t_by_seed:  Dict[int, List[np.ndarray]] = defaultdict(list)

        for fp in files:
            parsed = _parse_fname(fp)
            if not parsed:
                continue
            _, p0_val, seed = parsed
            seeds.append(seed)

            orders = _load_orders_new(fp)  # {ordk: {"t":..., "pt":..., "nt":...}}
            for ordk, data in orders.items():
                per_order[ordk].append(data)

                # acumular por seed (para pc_sop e pc_sop_roll)
                if "pt" in data:
                    pt_arr = np.asarray(data["pt"], dtype=float)
                    if "t" in data and data["t"] is not None:
                        t_arr = np.asarray(data["t"], dtype=float)
                    else:
                        t_arr = np.arange(len(pt_arr), dtype=float)
                    if x_max is not None:
                        msk = (t_arr <= x_max)
                        pt_arr = pt_arr[msk]
                        t_arr  = t_arr[msk]
                    if pt_arr.size > 0:
                        pt_by_seed[seed].append(pt_arr)
                        t_by_seed[seed].append(t_arr)

        # ---- médias por ordem (como antes) ----
        mean_by_order: Dict[int, Dict[str, Any]] = {}
        for ordk, lst in per_order.items():
            mean_by_order[ordk] = _average_by_order_new(lst)

            # ---- NOVO: janela deslizante sobre a série média da ORDEM ----
            if window_roll is not None:
                data_ord = mean_by_order[ordk]
                if "pt" in data_ord and "t" in data_ord and data_ord["pt"] is not None and data_ord["t"] is not None:
                    t_ord = np.asarray(data_ord["t"], dtype=float)
                    pt_ord_mean = np.asarray(data_ord["pt"], dtype=float)  # série média (já agregada entre arquivos)
                    if len(pt_ord_mean) >= window_roll:
                        t_c, mu_r, sd_r = rolling_mean_std(t_ord, pt_ord_mean, window_roll)
                        # salva dentro do "data" da ordem
                        mean_by_order[ordk]["pt_mean_roll"] = {
                            "t_center": t_c.tolist(),
                            "mean":     mu_r.tolist(),
                            "std":      sd_r.tolist(),
                            "window":   int(window_roll)
                        }

        orders_blocks = [
            {"order_percolation": int(ordk), "data": mean_by_order[ordk]}
            for ordk in sorted(mean_by_order.keys())
        ]

        # -------- p_{c,SOP} (igual ao seu) --------
        # para cada seed: concatena ordens e calcula <pt>_tempo (seed)
        seed_means: list[float] = []
        for s in sorted(set(seeds)):
            pts = pt_by_seed.get(s, [])
            if len(pts) == 0: 
                continue
            pt_concat = np.concatenate(pts)
            if pt_concat.size > 0:
                seed_means.append(float(np.mean(pt_concat)))

        mu_boot, sd_boot = _bootstrap_mean_across_samples(np.array(seed_means, dtype=float),
                                                          n_boot=n_boot, rng_seed=rng_seed)

        # -------- NOVO: p_{c,SOP}^{roll}(t) ENTRE-SEEDS --------
        pc_sop_roll_block = None
        if window_roll is not None:
            seed_series = _concat_seed_series_for_pc(pt_by_seed, t_by_seed, x_max)
            # calcula série rolada por seed
            rolled_means = []
            rolled_times = []
            for s in sorted(seed_series.keys()):
                t_concat  = seed_series[s]["t"]
                pt_concat = seed_series[s]["pt"]
                if len(pt_concat) >= window_roll:
                    t_c, mu_r, _ = rolling_mean_std(t_concat, pt_concat, window_roll)
                    rolled_means.append(mu_r)    # (T_s,)
                    rolled_times.append(t_c)     # (T_s,)

            if len(rolled_means) > 0:
                min_len = min(len(x) for x in rolled_means)
                if min_len >= 1:
                    M = np.stack([m[:min_len] for m in rolled_means], axis=0).astype(np.float32)   # (n_seeds_valid, T)
                    Tstack   = np.stack([tc[:min_len] for tc in rolled_times], axis=0)
                    t_center = np.median(Tstack, axis=0)

                    mean_boot, std_boot_series = _bootstrap_series_across_seeds(
                        M, n_boot=n_boot, rng_seed=rng_seed, batch_size=128, use_float32=True
                    )

                    # --- NOVO: média da cauda com erro bootstrap ENTRE-SEEDS (escalar) ---
                    mean_tail, std_tail, T_tail = _bootstrap_tail_mean_across_seeds(
                        M, t_center,
                        n_boot=n_boot, rng_seed=rng_seed,
                        tail_tmin=None,      # opcional: defina um t mínimo absoluto
                        tail_frac=0.2,       # ex.: última 20% da série; ajuste a gosto
                        batch_size=128, use_float32=True
                    )

                    pc_sop_roll_block = {
                        "t_center": t_center.tolist(),
                        "mean":     mean_boot.tolist(),
                        "std_boot": std_boot_series.tolist(),
                        "n_seeds":  int(M.shape[0]),
                        "n_boot":   int(n_boot),
                        "window":   int(window_roll),
                        # ---- NOVO bloco resumido da cauda ----
                        "tail_summary": {
                            "mode": "frac",
                            "tail_frac": 0.2,           # ou inclua "tail_tmin" se usar limite absoluto
                            "T_used": T_tail,           # quantos pontos entraram na média da cauda
                            "mean": mean_tail,          # média bootstrap ENTRE-SEEDS na cauda
                            "std_boot": std_tail        # desvio bootstrap ENTRE-SEEDS dessa média
                        }
                    }


        p0_group = {
            "p0_value": float(p0),
            "num_seeds": len(set(seeds)),
            "seeds": sorted(set(seeds)),
            "orders": orders_blocks,
            "pc_sop": {
                "mean": mu_boot,          # <pt> médio entre-seeds (bootstrap)
                "std_boot": sd_boot,      # desvio dos meios bootstrap (incerteza entre-seeds)
                "n_seeds": len(seed_means),
                "n_boot": int(n_boot)
            }
        }
        if pc_sop_roll_block is not None:
            p0_group["pc_sop_roll"] = pc_sop_roll_block

        bundle["p0_groups"].append(p0_group)
        msg_roll = ""
        if pc_sop_roll_block is not None:
            msg_roll = f" | pc_SOP^roll: T={len(pc_sop_roll_block['t_center'])}, seeds_valid={pc_sop_roll_block['n_seeds']}, win={pc_sop_roll_block['window']}"
        print(f"[ok] p0={p0:.2f}: {len(files)} arquivos | seeds={len(set(seeds))} | pc_SOP={mu_boot:.6f}±{sd_boot:.6f}{msg_roll}")

    # salvar UMA pasta acima de 'data/'
    out_dir = os.path.dirname(data_dir)
    out_path = os.path.join(out_dir, "properties_mean_bundle.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f"[salvo] {out_path}")
    return out_path


# ------------------ Regex / parsing de arquivo ------------------
FLOAT = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
RGX_FLEX = re.compile(
    rf'^P0_(?P<P0>{FLOAT})_p0_(?P<p0>{FLOAT})_seed_(?P<seed>\d+)\.json$'
)

def parse_filename(path):
    m = RGX_FLEX.match(Path(path).name)
    if not m:
        raise ValueError(f"Nome inválido: {path}")
    return {"P0": float(m.group("P0")), "p0": float(m.group("p0")), "seed": int(m.group("seed"))}

# ------------------ Leitura do JSON do experimento ------------------
def read_experiment_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    meta = raw.get("meta", {}) or {}
    results = raw.get("results", {}) or {}
    fixed_results = {k: (v if isinstance(v, dict) else {}) for k, v in results.items()}
    return {"meta": meta, "results": fixed_results}

# ------------------ Helpers ------------------
def orders_for(num_colors:int):
    # nomes no JSON de entrada têm espaço; no bundle de saída usar underscore
    return [f"order_percolation {i}" for i in range(1, num_colors+1)]

def group_files_by_p0(file_list):
    groups = {}
    for fp in file_list:
        p0 = parse_filename(fp)["p0"]
        groups.setdefault(p0, []).append(fp)
    # ordenar para reprodutibilidade
    for p0 in groups:
        groups[p0] = sorted(groups[p0])
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))

def stack_by_order(files, num_colors):
    """
    Empilha séries por ordem para um conjunto de arquivos (mesmo p0).
    Trunca cada ordem ao menor comprimento comum.
    Retorna dict {ord_label_space: {"t": np.array(T), "Y": np.ndarray(n_files, T)}}
    """
    ords = orders_for(num_colors)
    # comprimento mínimo por ordem
    tmin = {o: np.inf for o in ords}
    for fp in files:
        data = read_experiment_json(fp)["results"]
        for o in ords:
            tlen = len(data[o]["data"]["time"])
            if tlen < tmin[o]:
                tmin[o] = tlen

    out = {}
    for o in ords:
        series = []
        t_keep = None
        T = int(tmin[o])
        for fp in files:
            d = read_experiment_json(fp)["results"][o]["data"]
            if t_keep is None:
                t_keep = np.array(d["time"][:T], dtype=float)
            series.append(np.array(d["pt"][:T], dtype=float))
        out[o] = {"t": t_keep, "Y": np.vstack(series)}  # (n_files, T)
    return out

def summarize(stacked):
    """
    Recebe saída de stack_by_order e devolve dict por ordem com:
    n, t, mean, sem
    """
    sums = {}
    for o, d in stacked.items():
        Y = d["Y"]                # (n, T)
        n = Y.shape[0]
        mean = Y.mean(axis=0)
        if n > 1:
            std = Y.std(axis=0, ddof=1)
            sem = std / np.sqrt(n)
        else:
            sem = np.zeros_like(mean)
        sums[o] = {"n": int(n), "t": d["t"], "mean": mean, "sem": sem}
    return sums

def to_serializable(a):
    # garante listas JSON-serializáveis
    if isinstance(a, np.ndarray):
        return a.tolist()
    if isinstance(a, (np.floating, np.integer)):
        return a.item()
    return a

# ------------------ Parâmetros do caminho ------------------
base_root = "../Data/bond_percolation"


def compute_means_for_folder(L: int, DIM: int, NT:int, K : float, NC_list : list[int], RHO_list : list[float]) -> dict:
    """
       L(int) : Lenght of network  
       DIM(int): dimension of network
       NT(int): parameter of model
       K(float): parameter of model
       NC_list(list): numbers of colors
       RHO_list(list): density of each color
    """
    for num_colors, rho in zip(NC_list, RHO_list):
        data_dir = os.path.join(
            base_root,
            f"num_colors_{num_colors}",
            f"dim_{DIM}",
            f"L_{L}",
            "NT_constant",
            f"NT_{NT}",
            f"k_{K:.1e}",
            f"rho_{rho:.4e}",
            "data",
        )
        parent_dir = os.path.dirname(data_dir)  # um nível acima de 'data'
        os.makedirs(parent_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        if not files:
            print(f"[AVISO] Sem arquivos em: {data_dir}")
            continue

        # 1) agrupar por p0 (nome do arquivo)
        groups = group_files_by_p0(files)

        # 2) montar o bundle
        bundle = {"meta": {"num_colors": num_colors, "rho": rho}}

        for p0_val, flist in groups.items():
            # empilhar e resumir por ordem
            stacked = stack_by_order(flist, num_colors)
            sums = summarize(stacked)
            # montar nó "p0 = {valor}"
            pkey = f"p0 = {p0_val}"
            bundle[pkey] = {}
            for o_space, agg in sums.items():
                # saída deve usar underscore no nome
                idx = int(o_space.split()[-1])  # pega o número da ordem
                o_key = f"order_percolation_{idx}"
                bundle[pkey][o_key] = {
                    "num_samples": int(agg["n"]),
                    "t": to_serializable(agg["t"]),
                    "pt": to_serializable(agg["mean"]),
                    "pt_err": to_serializable(agg["sem"]),
                }

        # 3) salvar JSON
        out_path = os.path.join(parent_dir, "all_data_bundle.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)

        print(f"[OK] Salvo: {out_path}")

def read_all_data_bundle(path, as_dataframe=False):
    """
    Lê um 'all_data_bundle.json' e retorna os dados em um dicionário conveniente.
    Opcionalmente, retorna também um DataFrame "longo" (uma linha por tempo).

    Parâmetros
    ----------
    path : str | Path
        Caminho para o arquivo all_data_bundle.json.
    as_dataframe : bool, padrão False
        Se True, retorna também um DataFrame com colunas:
        [num_colors, rho, p0, order, num_samples, t, pt, pt_err].

    Retorno
    -------
    bundle : dict
        {
          "meta": {"num_colors": int, "rho": float},
          p0_value(float): {
            "order_percolation_1": {"num_samples": int, "t": np.array, "pt": np.array, "pt_err": np.array},
            "order_percolation_2": {...},
            ...
          },
          ...
        }
    df (opcional) : pandas.DataFrame
        Apenas se as_dataframe=True.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if "meta" not in raw or not isinstance(raw["meta"], dict):
        raise ValueError("JSON inválido: campo 'meta' ausente ou malformado.")

    meta = raw["meta"]
    num_colors = meta.get("num_colors", None)
    rho = meta.get("rho", None)

    # Converter chaves "p0 = X" -> float X
    p0_pat = re.compile(r"^\s*p0\s*=\s*(.+?)\s*$", re.IGNORECASE)

    bundle = {"meta": {"num_colors": num_colors, "rho": rho}}
    for k, v in raw.items():
        if k == "meta":
            continue
        m = p0_pat.match(k)
        if not m:
            # ignora chaves inesperadas
            continue
        p0_val = float(m.group(1))

        # Converte listas para np.array
        orders_dict = {}
        for ord_key, payload in v.items():
            t = np.asarray(payload.get("t", []), dtype=float)
            pt = np.asarray(payload.get("pt", []), dtype=float)
            pt_err = np.asarray(payload.get("pt_err", []), dtype=float)
            n = int(payload.get("num_samples", 0))
            orders_dict[ord_key] = {
                "num_samples": n,
                "t": t,
                "pt": pt,
                "pt_err": pt_err
            }
        bundle[p0_val] = orders_dict

    if not as_dataframe:
        return bundle

    # Construir DataFrame longo
    rows = []
    for p0_val, orders in bundle.items():
        if p0_val == "meta":
            continue
        for ord_key, d in orders.items():
            n = d["num_samples"]
            t = d["t"]
            pt = d["pt"]
            pt_err = d["pt_err"]
            if len(t) != len(pt) or len(t) != len(pt_err):
                raise ValueError(f"Tamanhos inconsistentes em {ord_key} para p0={p0_val}.")
            for ti, vi, ei in zip(t, pt, pt_err):
                rows.append({
                    "num_colors": num_colors,
                    "rho": rho,
                    "p0": p0_val,
                    "order": ord_key,
                    "num_samples": n,
                    "t": ti,
                    "pt": vi,
                    "pt_err": ei
                })
    df = pd.DataFrame(rows, columns=["num_colors","rho","p0","order","num_samples","t","pt","pt_err"])
    return bundle, df

def select_random_json(directory: str) -> str:
    """
    Seleciona aleatoriamente um arquivo .json dentro de um diretório.
    
    :param directory: Caminho do diretório onde os arquivos .json estão localizados
    :return: Caminho do arquivo .json selecionado aleatoriamente
    """
    # Lista todos os arquivos no diretório
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    # Se não houver arquivos .json no diretório, retorna None
    if not files:
        print("Nenhum arquivo .json encontrado no diretório.")
        return None
    
    # Seleciona aleatoriamente um arquivo da lista
    random_file = random.choice(files)
    
    # Retorna o caminho completo do arquivo selecionado
    return os.path.join(directory, random_file)