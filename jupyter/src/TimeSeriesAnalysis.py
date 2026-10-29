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
import gzip
import lzma
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

SERIES_ENCODING_KEY = "__encoding__"

# ============================================================
# Paths / Regex helpers
# ============================================================

def lateral_bundle_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "lateral_correlations_bundle.json.xz"


def gzip_lateral_bundle_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "lateral_correlations_bundle.json.gz"


def legacy_lateral_bundle_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "lateral_correlations_bundle.json"


def existing_lateral_bundle_path(out_dir: str | Path) -> Path | None:
    out_dir = Path(out_dir)
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


def resolve_lateral_bundle_path(path: str | Path) -> Path:
    """
    Resolve a lateral correlations bundle path or a published group directory.

    Accepts either:
      - .../lateral_correlations_bundle.json.gz
      - .../lateral_correlations_bundle.json
      - the directory containing one of those files
    """
    bundle_path = Path(path)
    if bundle_path.is_dir():
        found = existing_lateral_bundle_path(bundle_path)
        if found is None:
            raise FileNotFoundError(f"No lateral correlations bundle found in {bundle_path}")
        return found
    return bundle_path


def load_lateral_correlations_bundle(path: str | Path) -> dict[str, Any]:
    """Load a lateral correlations bundle, compressed or legacy uncompressed."""
    bundle_path = resolve_lateral_bundle_path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    if bundle_path.suffix == ".xz":
        with lzma.open(bundle_path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    elif bundle_path.suffix == ".gz":
        with gzip.open(bundle_path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        with bundle_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid lateral correlations bundle: {bundle_path}")
    return data


def _lateral_column_length(values: Any) -> int:
    if isinstance(values, list):
        return len(values)
    if isinstance(values, dict) and values.get(SERIES_ENCODING_KEY) == "range":
        return int(values.get("n", 0) or 0)
    return 0


def _lateral_column_value(values: Any, idx: int) -> Any:
    if isinstance(values, list):
        return values[idx] if idx < len(values) else None
    if isinstance(values, dict) and values.get(SERIES_ENCODING_KEY) == "range":
        start = float(values.get("start", 0.0) or 0.0)
        step = float(values.get("step", 0.0) or 0.0)
        return start + step * idx
    return values


def expand_lateral_series_columns(series: Any, series_length: int | None = None) -> dict[str, list[Any]]:
    """
    Expand a compact lateral time series into aligned property lists.

    Supports:
      - new compact columns with scalar constants and encoded ranges;
      - previous columnar lists;
      - legacy list-of-row dictionaries.
    """
    if isinstance(series, list):
        rows = [item for item in series if isinstance(item, dict)]
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

    keys = list(series)
    lengths = [_lateral_column_length(values) for values in series.values()]
    if series_length:
        lengths.append(int(series_length))
    n_rows = max(lengths, default=0)
    return {
        key: [_lateral_column_value(series.get(key), idx) for idx in range(n_rows)]
        for key in keys
    }


def _lateral_series_to_rows(series: Any, series_length: int | None = None) -> list[dict[str, Any]]:
    columns = expand_lateral_series_columns(series, series_length)
    if not columns:
        return []
    n_rows = max((len(values) for values in columns.values()), default=0)
    return [
        {key: values[idx] if idx < len(values) else None for key, values in columns.items()}
        for idx in range(n_rows)
    ]


def iter_lateral_series_blocks(
    bundle_or_path: dict[str, Any] | str | Path,
    obs_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Read lateral bundle blocks in the native aggregated format.

    Each block contains metadata, N_samples, and expanded aligned series
    columns. Use obs_type="correlation" or "susceptibility" to filter.
    """
    bundle = (
        bundle_or_path
        if isinstance(bundle_or_path, dict)
        else load_lateral_correlations_bundle(bundle_or_path)
    )
    blocks: list[dict[str, Any]] = []
    for sample in bundle.get("samples", []):
        if not isinstance(sample, dict):
            continue
        sample_obs_type = sample.get("obs_type")
        if obs_type is not None and sample_obs_type != obs_type:
            continue
        meta = {
            key: sample.get(key)
            for key in (
                "filename", "sample_id", "obs_type", "series_kind", "t_stat",
                "p0", "P0", "c", "f_T", "seed", "N_samples", "n_rows",
            )
        }
        blocks.append({
            **meta,
            "series_length": sample.get("series_length"),
            "series": expand_lateral_series_columns(sample.get("series"), sample.get("series_length")),
        })
    return blocks


def iter_lateral_series_rows(
    bundle_or_path: dict[str, Any] | str | Path,
    obs_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Flatten lateral bundle samples into per-time rows.

    Each returned row includes sample metadata plus one entry from the stored
    time series. Use obs_type="correlation" or "susceptibility" to filter.
    """
    bundle = (
        bundle_or_path
        if isinstance(bundle_or_path, dict)
        else load_lateral_correlations_bundle(bundle_or_path)
    )
    rows: list[dict[str, Any]] = []
    for sample in bundle.get("samples", []):
        if not isinstance(sample, dict):
            continue
        sample_obs_type = sample.get("obs_type")
        if obs_type is not None and sample_obs_type != obs_type:
            continue
        meta = {
            key: sample.get(key)
            for key in ("filename", "sample_id", "obs_type", "t_stat", "p0", "P0", "c", "f_T", "seed", "N_samples")
        }
        for item in _lateral_series_to_rows(sample.get("series"), sample.get("series_length")):
            rows.append({**meta, **item})
    return rows


def load_lateral_correlations_dataframe(
    path: str | Path,
    obs_type: str | None = None,
) -> pd.DataFrame:
    """
    Load a lateral correlations bundle as a tidy DataFrame.

    The new aggregated bundle stores one block per parameter/observable with
    N_samples and compact columns. This function expands those columns into
    one row per time point while preserving the aggregation metadata.
    """
    return pd.DataFrame(iter_lateral_series_rows(path, obs_type=obs_type))

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

# Accepts c/f_T/rho in normal float or scientific notation (e.g., 5.0e-01)
FLOAT = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
PARAMS_RE = re.compile(
    rf"""
    (?P<type_perc>[A-Za-z]+)_percolation
    /num_colors_(?P<num_colors>\d+)
    /dim_(?P<dim>\d+)
    /L_(?P<L>\d+)
    /fT_constant/fT_(?P<f_T>{FLOAT})
    /c_(?P<c>{FLOAT})
    /rho_(?P<rho>{FLOAT})
    /data
    """,
    re.X,
)

def parse_params_from_path(path: str) -> dict | None:
    """
    Extract type_perc, num_colors, dim, L, f_T, c, rho from a folder path.
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
        "f_T": float(gd["f_T"]),
        "c": float(gd["c"]),
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
      (order_percolation, p_t_array, f_t_array_or_None)

    Supports both result layouts:
      - list: [{"order_percolation": i, "data": {...}}, ...]
      - dict: {"order_percolation i": {"data": {...}}, ...}
    """
    try:
        with open(file_path, "r") as f:
            obj = json.load(f)
    except Exception:
        return []

    if not isinstance(obj, dict):
        return []

    results = obj.get("results", {})
    if isinstance(results, list):
        items = [(str(item.get("order_percolation", "")), item) for item in results if isinstance(item, dict)]
    elif isinstance(results, dict):
        items = list(results.items())
    else:
        return []

    out = []
    for key, item in items:
        if not isinstance(item, dict):
            continue
        order = item.get("order_percolation", None)
        if order is None:
            m = re.search(r"(\d+)", str(key))
            order = int(m.group(1)) if m else None
        d = item.get("data", item)
        if order is None or not isinstance(d, dict) or "pt" not in d:
            continue

        p = np.asarray(d["pt"], float)
        f_arr = np.asarray(d.get("ft", d.get("f_t", [])), float) if ("ft" in d or "f_t" in d) else None
        n = min(len(p), len(f_arr)) if f_arr is not None else len(p)
        if n <= 0:
            continue
        p = p[:n]
        f_arr = f_arr[:n] if f_arr is not None else None
        out.append((int(order), p, f_arr))

    return out

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
    f_T: float,
    c: float,
    base_root: str = "../Data",
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-15,
) -> list[float]:
    """
    List all rho values that exist under:
      ../Data/{type_perc}_percolation/num_colors_{num_colors}/dim_{dim}/L_{L}/fT_constant/fT_*/c_*/rho_*/data
    and match the fixed parameters c and f_T.
    """
    base = (
        Path(base_root)
        / f"{type_perc}_percolation"
        / f"num_colors_{num_colors}"
        / f"dim_{dim}"
        / f"L_{L}"
        / "fT_constant"
    )
    if not base.exists():
        return []

    rhos = []
    for data_dir in base.glob("fT_*/c_*/rho_*/data"):
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

        f_T_here = float(gd["f_T"])
        c_here = float(gd["c"])
        if not math.isclose(f_T_here, float(f_T), rel_tol=rel_tol, abs_tol=abs_tol):
            continue
        if not math.isclose(c_here, float(c), rel_tol=rel_tol, abs_tol=abs_tol):
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

def _cache_dir(type_perc: str, num_colors: int, dim: int, L: int, f_T: float, c: float, rho: float) -> str:
    """Directory one level above /data that holds cache files."""
    return (
        f"../Data/{type_perc}_percolation/num_colors_{num_colors}/dim_{dim}/L_{L}/"
        f"fT_constant/fT_{f_T:.6e}/c_{c:.6e}/rho_{rho:.4e}"
    )

def _cache_file_path(type_perc: str, num_colors: int, dim: int, L: int, f_T: float, c: float, rho: float, p0: float) -> str:
    """Cache file for a single p0."""
    return os.path.join(_cache_dir(type_perc, num_colors, dim, L, f_T, c, rho),
                        f"mean_properties_p0_{_safe_p0_tag(p0)}.json")

def _bundle_file_path(type_perc: str, num_colors: int, dim: int, L: int, f_T: float, c: float, rho: float) -> str:
    """Single bundle cache containing many p0 entries."""
    return os.path.join(_cache_dir(type_perc, num_colors, dim, L, f_T, c, rho),
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

def mean_properties(type_perc: str, num_colors: int, dim: int, L: int, f_T: float,
                    c: float, rho: float, p0: float) -> tuple[dict, dict]:
    """
    Compute (or load from cache) ensemble stats for a SINGLE p0.

    Returns (p_stats, f_stats) dicts with keys like 't_grid', 'mean', 'std', etc.
    Cache file: mean_properties_p0_<p0_tag>.json (one dir above /data).
    """
    path_files = (
        f"../Data/{type_perc}_percolation/num_colors_{num_colors}/dim_{dim}/L_{L}/"
        f"fT_constant/fT_{f_T:.6e}/c_{c:.6e}/rho_{rho:.4e}/data"
    )
    save_dir = _cache_dir(type_perc, num_colors, dim, L, f_T, c, rho)
    cache_fp = _cache_file_path(type_perc, num_colors, dim, L, f_T, c, rho, p0)

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
                and float(meta.get("f_T")) == float(f_T)
                and float(meta.get("c")) == float(c)
                and float(meta.get("rho")) == float(rho)
                and float(meta.get("p0")) == float(p0)
            ):
                return _stats_from_json(cached.get("stats_p", {})), _stats_from_json(cached.get("stats_f", cached.get("stats_N", {})))
        except Exception:
            pass  # ignore corrupted cache and recompute

    # compute from raw files
    all_files = glob.glob(os.path.join(path_files, "*.json"))
    times_list_p, p_values_list = [], []
    times_list_f, f_values_list = [], []

    for file in all_files:
        if not _filename_matches_p0(file, p0):
            continue
        records = read_orders_one_file(file)
        if not records:
            continue

        order, p, ft = records[0]
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

        # f(t)
        if ft is not None:
            ft = np.asarray(ft, float)
            m_f = min(t.size, ft.size)
            if m_f > 0:
                t_f = t[:m_f].astype(float, copy=False)
                ft_f = ft[:m_f]
                if m_f > 1 and not np.all(np.diff(t_f) >= 0):
                    idx = np.argsort(t_f)
                    t_f, ft_f = t_f[idx], ft_f[idx]
                times_list_f.append(t_f)
                f_values_list.append(ft_f)

    stats_p = ensemble_stats(times_list_p, p_values_list)
    stats_f = ensemble_stats(times_list_f, f_values_list)

    # save cache
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "_meta": {
            "version": _CACHE_VERSION,
            "type_perc": type_perc,
            "num_colors": num_colors,
            "dim": dim,
            "L": L,
            "f_T": float(f_T),
            "c": float(c),
            "rho": float(rho),
            "p0": float(p0),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        "stats_p": _stats_to_jsonable(stats_p),
        "stats_f": _stats_to_jsonable(stats_f),
    }
    try:
        with open(cache_fp, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save cache at {cache_fp}: {e}")

    return stats_p, stats_f

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
    keys_series = ("time", "pt", "ft", "Mt", "Smax", "Ni", "chi")
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
# bundle_path = f"../Data/bond_percolation/num_colors_{NUM_COLORS}/dim_{DIM}/L_{L}/fT_constant/fT_{f_T:.6e}/c_{C:.6e}/rho_{RHO:.4e}/data/"
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
    include_ft: bool = True
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Lê um arquivo .json de simulação e retorna:
      - metas: lista de dicts com metadados por 'order_percolation'
      - df: DataFrame longo com colunas ['order','t','pt','ft'] (ft pode vir NaN)

    Parâmetros
    ----------
    filepath : str | Path
        Caminho do arquivo JSON.
    include_ft : bool
        Se True, tenta carregar a série 'ft' quando existir; caso contrário, preenche NaN.

    Retorno
    -------
    metas : list[dict]
        Para cada entrada em 'results', inclui:
        {'order', 'color', 'rho', 'M_size', 'time_percolation', 'n_time', 'has_ft'}.
    df : pandas.DataFrame
        Linhas por tempo: colunas ['order','t','pt','ft'] (ft pode ser NaN).
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
        ft   = d.get("ft", None) if include_ft else None

        # Converte para arrays, se existirem
        time = np.asarray(time) if time is not None else None
        pt   = np.asarray(pt)   if pt   is not None else None
        ft   = np.asarray(ft)   if ft   is not None else None

        # Valida tamanhos mínimos
        if time is None or pt is None:
            # sem séries, apenas meta
            n_time = 0
        else:
            n_time = int(min(len(time), len(pt)))
            if ft is not None:
                n_time = int(min(n_time, len(ft)))
            # recorta para o mesmo comprimento
            time = time[:n_time]
            pt   = pt[:n_time]
            if ft is not None:
                ft = ft[:n_time]

            # Empilha linhas
            if n_time > 0:
                if ft is None:
                    # sem ft: preenche NaN
                    for t, p in zip(time, pt):
                        rows.append((int(order), int(t), float(p), np.nan))
                else:
                    for t, p, n in zip(time, pt, ft):
                        rows.append((int(order), int(t), float(p), float(n)))

        metas.append({
            "order": int(order),
            "color": d.get("color", None),
            "rho": d.get("rho", None),
            "M_size": d.get("M_size", None),
            "time_percolation": d.get("time_percolation", None),
            "n_time": n_time,
            "has_ft": (ft is not None)
        })

    df = pd.DataFrame(rows, columns=["order", "t", "pt", "ft"]) if rows else \
         pd.DataFrame(columns=["order", "t", "pt", "ft"])

    return metas, df

# AVARAGE TO OLD STRUCT JSON
# -----------------------------
# Helpers para parsing/carregamento
# -----------------------------

_FNAME_RE = re.compile(
    r"P0_(?P<P0>[-+]?\d+(?:\.\d+)?)_p0_(?P<p0>[-+]?\d+(?:\.\d+)?)_seed_(?P<seed>\d+)\.json$",
    re.IGNORECASE
)


import json
import re
from typing import Dict, Any

def _load_orders_new(filepath: str) -> Dict[int, Dict[str, Any]]:
    """
    Lê um JSON de time series e devolve:
        {ordem_percolacao (int): data_dict}

    Aceita dois formatos de 'results':

    1) Formato novo (lista):
       "results": [
         {"order_percolation": 1, "data": {...}},
         {"order_percolation": 2, "data": {...}},
         ...
       ]

    2) Formato legado (dict):
       "results": {
         "order_percolation 1": {"data": {...}},
         "order_percolation 2": {"data": {...}}
       }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict) or "results" not in obj:
        raise ValueError(f"JSON inesperado (sem 'results'): {filepath}")

    results = obj["results"]
    out: Dict[int, Dict[str, Any]] = {}

    # --- caso 1: results é uma lista (formato novo) ---
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            ordk_val = item.get("order_percolation") or item.get("order")
            if ordk_val is None:
                continue
            ordk = int(ordk_val)
            data_block = item.get("data", item)
            out[ordk] = data_block

    # --- caso 2: results é um dict (formato antigo, como o da screenshot) ---
    elif isinstance(results, dict):
        for key, block in results.items():
            if not isinstance(block, dict):
                continue

            # tenta pegar a ordem de dentro do bloco…
            ordk_val = block.get("order_percolation")
            # …senão extrai número do próprio nome da chave, ex: "order_percolation 1"
            if ordk_val is None:
                m = re.search(r"(\d+)", str(key))
                if not m:
                    continue
                ordk_val = int(m.group(1))

            ordk = int(ordk_val)
            data_block = block.get("data", block)
            out[ordk] = data_block

    else:
        raise ValueError(
            f"JSON inesperado: 'results' é do tipo {type(results)} em {filepath}"
        )

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
    Média/SEM ao longo das seeds para uma SÉRIE (pt ou ft).
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
    - Séries: pt (sempre), ft (se existir).
    - Escalares: time_percolation e M_size (média e SEM).
    """
    # séries
    pt_blk = _avg_series_across_seeds(lst, "pt")
    ft_blk = _avg_series_across_seeds([d for d in lst if d.get("ft", None) is not None], "ft")

    # escalares
    m_ms, sem_ms, n_ms = _mean_sem_1d([d.get("M_size", None) for d in lst])

    out = {
        "time": pt_blk["time"],

        "pt_mean": pt_blk["pt_mean"],
        "pt_sem":  pt_blk["pt_sem"],
        "n_seeds_pt": pt_blk["n_seeds_series"],

        # ft pode não existir
        "ft_mean": ft_blk.get("ft_mean", []),
        "ft_sem":  ft_blk.get("ft_sem", []),
        "n_seeds_ft": ft_blk.get("n_seeds_series", 0),

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
# _load_orders_new(fp) -> Dict[int, Dict[str, Any]]  # por ordem: {"t":..., "pt":..., "ft":...}
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
    f_T: float,
    c: float,
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
      - séries: pt (e ft se existir);
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
        "fT_constant",
        f"fT_{f_T:.6e}",
        f"c_{c:.6e}",
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
            "f_T": float(f_T),
            "c": float(c),
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

            orders = _load_orders_new(fp)  # {ordk: {"t":..., "pt":..., "ft":...}}
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


def compute_means_for_folder_tests(
    type_perc: str,
    num_colors: int,
    dim: int,
    L: int,
    f_T: float,
    c: float,
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
      - séries: pt (e ft se existir);
      - escalares: time_percolation e M_size;
    Além disso, calcula e salva no JSON:
      - p_{c,SOP} (bootstrap ENTRE-SEEDS, escalar)
      - (NOVO) p_{c,SOP}^{roll}(t): série rolada com incerteza bootstrap ENTRE-SEEDS por ponto
        em 'pc_sop_roll': {"t_center":[...], "mean":[...], "std_boot":[...], "n_seeds":..., "n_boot":..., "window":...}
      - (NOVO) por ordem, 'orders -> data -> pt_mean_roll' calculado da série média da ordem.
    """
    base_dir = "../Data_tests"
    data_dir = os.path.join(
        base_dir,
        f"{type_perc}_percolation",
        f"num_colors_{num_colors}",
        f"dim_{dim}",
        f"L_{L}",
        "fT_constant",
        f"fT_{f_T:.6e}",
        f"c_{c:.6e}",
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
            "f_T": float(f_T),
            "c": float(c),
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

            orders = _load_orders_new(fp)  # {ordk: {"t":..., "pt":..., "ft":...}}
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

def resolve_json_bundle_path(filepath):
    path = Path(filepath)
    if path.exists():
        return path
    if path.name.endswith(".json"):
        for suffix in (".xz", ".gz"):
            candidate = Path(str(path) + suffix)
            if candidate.exists():
                return candidate
    return path


def load_json_bundle_auto(filepath):
    path = resolve_json_bundle_path(filepath)
    if path.suffix == ".xz":
        with lzma.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_properties_bundle(filepath):
    """
    Lê properties_mean_bundle.json no formato atual do pipeline.

    Retorna um dict indexado por p0:
        result[p0] = {
            "num_seeds": int,
            "pc_sop": dict,
            "colors": dict,
            "orders": {
                order_percolation: {
                    ... dados de 'data' ...
                }
            }
        }
    """
    js = load_json_bundle_auto(filepath)

    result = {}

    for group in js.get("p0_groups", []):
        p0 = float(group["p0_value"])

        orders_dict = {}
        for ob in group.get("orders", []):
            order = int(ob["order_percolation"])
            orders_dict[order] = ob.get("data", {})

        result[p0] = {
            "num_seeds": int(group.get("num_seeds", 0)),
            "pc_sop": group.get("pc_sop", {}),
            "colors": group.get("colors", {}),
            "orders": orders_dict,
        }

    return result


import os
import glob
import random
import re
from typing import Optional

FLOAT = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
RE_P0 = re.compile(rf'P0_(?P<P0>{FLOAT})')
RE_p0 = re.compile(rf'p0_(?P<p0>{FLOAT})')
RE_seed = re.compile(r'seed_(?P<seed>\d+)', re.IGNORECASE)


import os
import glob
import random
from typing import Optional, Set

def select_random_json(
    directory: str,
    p0: float,
    P0: float,
    used_files: Optional[Set[str]] = None
) -> Optional[str]:
    """
    Seleciona aleatoriamente um arquivo .json dentro de 'directory' cujo nome
    contenha simultaneamente os valores especificados de P0 e p0, aceitando
    tanto o formato antigo quanto o novo, e evitando repetir arquivos já
    sorteados em 'used_files'.

    Exemplos aceitos:
      P0_0.10_p0_1.00_seed_123.json
      light_seed_123_ts_20260319T135046_P0_0.10_p0_1.00.json
      apolo1_seed_777_ts_..._P0_0.10_p0_1.00.json

    :param directory: Diretório onde estão os .json
    :param p0: Valor de p0 a filtrar
    :param P0: Valor de P0 a filtrar
    :param used_files: Conjunto de arquivos já usados; esses serão ignorados
    :return: Caminho absoluto do .json selecionado, ou None se não encontrar
    """
    directory_abs = os.path.abspath(os.path.expanduser(directory))
    if not os.path.isdir(directory_abs):
        raise NotADirectoryError(f"Diretório inválido: {directory_abs}")

    if used_files is None:
        used_files = set()

    target_p0 = float(p0)
    target_P0 = float(P0)
    candidates = []

    for f in glob.glob(os.path.join(directory_abs, "*.json")):
        if not os.path.isfile(f):
            continue

        f_abs = os.path.abspath(f)

        # ignora arquivos já usados
        if f_abs in used_files:
            continue

        name = os.path.basename(f_abs)

        mP0 = RE_P0.search(name)
        mp0 = RE_p0.search(name)
        ms = RE_seed.search(name)

        # exige que o arquivo tenha os três campos
        if not (mP0 and mp0 and ms):
            continue

        try:
            P0_file = float(mP0.group("P0"))
            p0_file = float(mp0.group("p0"))
        except Exception:
            continue

        if abs(P0_file - target_P0) < 1e-12 and abs(p0_file - target_p0) < 1e-12:
            candidates.append(f_abs)

    if not candidates:
        print(
            f"Nenhum arquivo novo encontrado com P0={target_P0:.2f} "
            f"e p0={target_p0:.2f} em {directory_abs}."
        )
        return None

    chosen = random.choice(candidates)
    used_files.add(chosen)

    return chosen

def rolling_weighted_mean(y, sem, w):
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)

    n = y.size
    m = n - w + 1
    if m <= 0:
        return np.array([]), np.array([]), np.array([])

    mu = np.empty(m, dtype=float)
    se = np.empty(m, dtype=float)
    chi2r = np.empty(m, dtype=float)

    eps = 1e-15
    for i in range(m):
        yw = y[i:i+w]
        sw = np.maximum(sem[i:i+w], eps)

        wgt = 1.0 / (sw * sw)
        W = np.sum(wgt)

        mui = np.sum(wgt * yw) / W
        sei = np.sqrt(1.0 / W)

        dof = max(w - 1, 1)
        chi2 = np.sum(((yw - mui) / sw) ** 2)
        chi2ri = chi2 / dof

        mu[i] = mui
        se[i] = sei
        chi2r[i] = chi2ri

    return mu, se, chi2r

def read_mean_json(N_COLORS:int, DIM:int, L:int, f_T:float, C:float, RHO:float):
    filename = (
        f"../SOP_data/published/bond_percolation/num_colors_{N_COLORS}/dim_{DIM}/"
        f"L_{L}/fT_constant/fT_{f_T:.6e}/c_{C:.6e}/rho_{RHO:.4e}/properties_mean_bundle.json"
    )
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    group0 = data["p0_groups"][0]
    num_seeds = group0["num_seeds"]

    orders_raw = group0["orders"]  # dict com chaves "0","1",...

    if isinstance(orders_raw, dict) and "0" in orders_raw:
        orders = [orders_raw[str(i)] for i in range(N_COLORS)]
    elif isinstance(orders_raw, dict):
        orders = [orders_raw[i] for i in range(N_COLORS)]
    else:
        orders = orders_raw  # caso já seja lista

    return orders, num_seeds

def rolling_weighted_mean(y, sem, w):
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)

    n = y.size
    m = n - w + 1
    if m <= 0:
        return np.array([]), np.array([]), np.array([])

    mu = np.empty(m, dtype=float)
    se = np.empty(m, dtype=float)
    chi2r = np.empty(m, dtype=float)

    eps = 1e-15
    for i in range(m):
        yw = y[i:i+w]
        sw = np.maximum(sem[i:i+w], eps)

        wgt = 1.0 / (sw * sw)
        W = np.sum(wgt)

        mui = np.sum(wgt * yw) / W
        sei = np.sqrt(1.0 / W)

        dof = max(w - 1, 1)
        chi2 = np.sum(((yw - mui) / sw) ** 2)
        chi2ri = chi2 / dof

        mu[i] = mui
        se[i] = sei
        chi2r[i] = chi2ri

    return mu, se, chi2r


def detect_equilibrium_start_with_errors(t, y, sem, w=40, lag=None, consec=6, z=2.0, chi2r_max=2.0,
                                         tail_frac=0.20, min_start_frac=0.05):
    """
    Retorna o ÍNDICE em t (não em mu/ok) correspondente ao início do equilíbrio.
    Se não encontrar um 'run' válido, usa fallback baseado no platô final (tail).
    """
    if lag is None:
        lag = w

    t = np.asarray(t)
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)

    n = y.size
    if n < (w + lag + 5):
        return 0

    # rolling_weighted_mean deve devolver mu e se_mu alinhados com janelas do sinal original.
    # Assumimos aqui que mu[k] corresponde à janela que termina em y[k+w-1].
    mu, se_mu, chi2r = rolling_weighted_mean(y, sem, w)
    m = mu.size
    if m <= lag:
        return 0

    dm = np.abs(mu[lag:] - mu[:-lag])
    se_comb = np.sqrt(se_mu[lag:]**2 + se_mu[:-lag]**2)

    ok_change = dm <= (z * se_comb)
    ok_chi = (chi2r[lag:] <= chi2r_max) & (chi2r[:-lag] <= chi2r_max)
    ok = ok_change & ok_chi

    # evita detectar "equilíbrio" cedo demais por acaso:
    # força começar a procurar após uma fração inicial do tempo
    min_mu_idx = int(np.floor(min_start_frac * m))
    j_start = max(0, min_mu_idx - lag)

    run = 0
    for j in range(j_start, ok.size):
        run = run + 1 if ok[j] else 0
        if run >= consec:
            # j refere-se ao vetor ok. O par comparado é (mu[j], mu[j+lag]).
            # O "ponto atual" (mais tardio) é mu_idx = j + lag.
            mu_idx = j + lag

            # mu_idx -> índice em y/t: se mu[k] é janela que termina em y[k+w-1]
            idx_t = mu_idx + (w - 1)
            idx_t = int(np.clip(idx_t, 0, n - 1))
            return idx_t

    # -------------------------
    # FALLBACK (não retorna 0)
    # -------------------------
    # Estima platô final (últimos tail_frac) e acha o primeiro ponto em que
    # o rolling_mean entra e permanece próximo desse platô.
    tail_start = int(np.floor((1.0 - tail_frac) * n))
    tail_start = np.clip(tail_start, 0, n - 1)

    mu_tail, se_tail = weighted_mean_and_sem(y[tail_start:], sem[tail_start:])
    # tolerância: z * erro combinado do ponto vs platô
    # usa erro do platô (se_tail) e erro do ponto (sem[i])
    tol = z * np.sqrt(se_tail**2 + np.maximum(sem, 1e-15)**2)

    # procura o primeiro i (após min_start_frac) tal que |y - mu_tail| <= tol por 'consec' passos
    i0 = int(np.floor(min_start_frac * n))
    run = 0
    for i in range(i0, n):
        if abs(y[i] - mu_tail) <= tol[i]:
            run += 1
            if run >= consec:
                return int(i - consec + 1)
        else:
            run = 0

    # último fallback: começa no tail_start
    return int(tail_start)


def idx_from_t0(t, t0):
    t = np.asarray(t)
    return int(np.searchsorted(t, t0, side="left"))

def weighted_mean_and_sem(y, sem):
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)
    eps = 1e-15
    sem = np.maximum(sem, eps)

    wgt = 1.0 / (sem * sem)
    W = np.sum(wgt)

    mu = float(np.sum(wgt * y) / W)
    se = float(np.sqrt(1.0 / W))
    return mu, se

# ============================================================
# Single growth_test sample: return series and plot the network only
# ============================================================

def _centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average matching centered_moving_average_tail in C++."""
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    half_left = window // 2
    half_right = window - half_left - 1
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    i = np.arange(values.size)
    left = np.maximum(0, i - half_left)
    right = np.minimum(values.size, i + half_right + 1)
    return (prefix[right] - prefix[left]) / (right - left)


def load_growth_test_stabilization_series(
    json_path: str | Path,
    order: int | None = None,
) -> dict[str, list[float]]:
    """Return p(t), its centered moving average, block means, and s'(t) as lists.

    Smoothing, block averaging, and finite differences follow the growth_test
    estimator in src/network.cpp. The dictionary contains only numeric lists.
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta", {})
    results = payload.get("results", {})
    if isinstance(results, list):
        entries = [(item.get("order_percolation"), item) for item in results]
    elif isinstance(results, dict):
        entries = list(results.items())
    else:
        raise ValueError(f"Unsupported results layout in {json_path}")
    selected = None
    for key, item in entries:
        raw_order = item.get("order_percolation") if isinstance(item, dict) else None
        if raw_order is None:
            match = re.search(r"(\d+)", str(key))
            raw_order = int(match.group(1)) if match else None
        if order is None or raw_order == int(order):
            selected = item
            break
    if selected is None:
        raise ValueError(f"No percolation order {order} found in {json_path}")
    data = selected.get("data", selected)
    time = np.asarray(data.get("time", []), dtype=float)
    p_t = np.asarray(data.get("pt", data.get("p_t", [])), dtype=float)
    if time.size == 0 or p_t.size != time.size:
        raise ValueError(f"Invalid or misaligned time/p(t) arrays in {json_path}")

    smooth_window = int(meta.get("growth_test_t_eq_smoothing_window", 15))
    block_window = int(meta.get("growth_test_t_eq_window_block", 10))
    moving_average = _centered_moving_average(p_t, smooth_window)
    n_blocks = time.size // block_window
    if n_blocks < 3:
        raise ValueError("At least three complete time blocks are required to calculate the derivative")
    block_time = np.array([
        np.mean(time[k * block_window:(k + 1) * block_window])
        for k in range(n_blocks)
    ])
    block_mean_pt = np.array([
        np.mean(moving_average[k * block_window:(k + 1) * block_window])
        for k in range(n_blocks)
    ])
    variation = np.abs(np.diff(block_mean_pt))
    derivative_time = 0.5 * (block_time[:-1] + block_time[1:])
    derivative = np.empty_like(variation)
    for i in range(variation.size):
        if i == 0:
            derivative[i] = (variation[1] - variation[0]) / (derivative_time[1] - derivative_time[0])
        elif i == variation.size - 1:
            derivative[i] = (variation[i] - variation[i - 1]) / (derivative_time[i] - derivative_time[i - 1])
        else:
            derivative[i] = (variation[i + 1] - variation[i - 1]) / (derivative_time[i + 1] - derivative_time[i - 1])

    return {
        "time": time.tolist(),
        "p_t": p_t.tolist(),
        "moving_average": moving_average.tolist(),
        "block_time": block_time.tolist(),
        "block_mean_pt": block_mean_pt.tolist(),
        "variation": variation.tolist(),
        "derivative_time": derivative_time.tolist(),
        "derivative": derivative.tolist(),
    }


def _read_compact_network_2d(
    path: str | Path,
    L: int,
    return_activation_time: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read occupied coordinates and activation times from NetworkCompact .bin without loading edges."""
    import struct
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"Compact network file is too short: {path}")
        magic, n = struct.unpack("<II", header[:8])
        if magic != 0x4E455447:
            raise ValueError(f"Invalid compact network magic in {path}")
        pos = np.fromfile(handle, dtype="<u4", count=n).astype(np.int64)
        species = np.fromfile(handle, dtype="u1", count=n)
        activation_time = np.fromfile(handle, dtype="<u4", count=n)
    active = species > 0
    flat = pos[active]
    x = flat % L
    z = flat // L
    if return_activation_time:
        return x, z, activation_time[active]
    return x, z


def get_growth_test_network_partition(
    json_path: str | Path,
    *,
    network_path: str | Path | None = None,
    z_stab: int | None = None,
    t_stab: float | int | None = None,
    L: int | None = None,
) -> dict:
    """
    Determina exatamente todos os sítios da rede compacta 2D particionados pela
    fronteira física ruidosa no momento da estabilização:
      - fora da cauda (antes da estabilização): z < h_front[x] (laranja)
      - no limiar / fronteira física: z == h_front[x] (vermelho)
      - na cauda (pós-estabilização): z > h_front[x] (roxo)

    Retorna um dicionário com:
      - 'L', 'height', 'z_stab', 't_stab'
      - 'h_front': array 1D com a altura da fronteira h(x) para cada coluna x in [0, L)
      - 'x', 'z', 'activation_time'
      - 'before_mask', 'frontier_mask', 'after_mask'
      - 'x_before', 'z_before'
      - 'x_frontier', 'z_frontier'
      - 'x_after', 'z_after'
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta", {})
    params = parse_params_from_path(str(json_path.parent)) or {}
    resolved_L = params.get("L", meta.get("L", L))
    if not resolved_L:
        match_L = re.search(r"(?:^|[/\\])L_(\d+)(?:_|[/\\])", str(json_path))
        if match_L:
            resolved_L = int(match_L.group(1))
    L = int(resolved_L or 0)
    if L <= 0:
        raise ValueError(f"Could not determine L from {json_path}; pass L explicitly")

    if z_stab is None:
        z_stab = meta.get("y_stat", meta.get("growth_test_z_stat_by_color"))
    if isinstance(z_stab, (list, tuple, np.ndarray)):
        z_stab = z_stab[0] if len(z_stab) else None
    if z_stab is not None:
        z_stab = int(z_stab)

    if network_path is None:
        network_path = json_path.parent.parent / "network" / (json_path.stem + ".bin")

    x, z, act_time = _read_compact_network_2d(network_path, L, return_activation_time=True)
    if x.size == 0:
        raise ValueError(f"No occupied sites in compact network: {network_path}")

    height = int(z.max()) + 1

    # Determina o momento t_stab_moment em que a rede atinge a estabilização
    if t_stab is not None:
        t_stab_moment = float(t_stab)
    elif z_stab is not None:
        mask_z = z >= z_stab
        if np.any(mask_z):
            t_stab_moment = int(act_time[mask_z].min())
        else:
            t_stab_moment = int(round(float(meta.get("t_eq", 0))))
    else:
        t_stab_moment = int(round(float(meta.get("t_eq", 0))))

    # Perfil da fronteira superior h_front(x) no momento da estabilização
    h_front = np.full(L, -1, dtype=np.int64)
    mask_t = act_time <= t_stab_moment
    if np.any(mask_t):
        np.maximum.at(h_front, x[mask_t], z[mask_t])
    if np.any(h_front < 0):
        valid_idx = np.where(h_front >= 0)[0]
        if len(valid_idx) > 0:
            h_front[h_front < 0] = np.interp(np.where(h_front < 0)[0], valid_idx, h_front[valid_idx]).astype(int)
        else:
            h_front[:] = 0

    hf_per_site = h_front[x]
    before_mask = z < hf_per_site
    frontier_mask = z == hf_per_site
    after_mask = z >= hf_per_site

    return {
        "L": L,
        "height": height,
        "z_stab": z_stab if z_stab is not None else int(h_front.max()),
        "t_stab": t_stab_moment,
        "h_front": h_front,
        "x": x,
        "z": z,
        "activation_time": act_time,
        "before_mask": before_mask,
        "frontier_mask": frontier_mask,
        "after_mask": after_mask,
        "x_before": x[before_mask],
        "z_before": z[before_mask],
        "x_frontier": x[frontier_mask],
        "z_frontier": z[frontier_mask],
        "x_after": x[after_mask],
        "z_after": z[after_mask],
    }


def _patch_pdf_writer_safe():
    """Garante que o backend PDF do matplotlib use FlateDecode seguro sem compressão redundante."""
    try:
        import matplotlib.backends.backend_pdf as bpdf
        from matplotlib.backends.backend_pdf import Name

        if getattr(bpdf.PdfFile, "_writeImg_poppler_safe", False):
            return

        def safe_writeImg(self, data, id, smask=None):
            height, width, color_channels = data.shape
            obj = {
                'Type': Name('XObject'),
                'Subtype': Name('Image'),
                'Width': width,
                'Height': height,
                'ColorSpace': Name({1: 'DeviceGray', 3: 'DeviceRGB'}[color_channels]),
                'BitsPerComponent': 8,
            }
            if smask:
                obj['SMask'] = smask

            self.beginStream(id, self.reserveObject('length of image stream'), obj)
            self.currentstream.write(data.tobytes())
            self.endStream()

        bpdf.PdfFile._writeImg = safe_writeImg
        bpdf.PdfFile._writeImg_poppler_safe = True
    except Exception:
        pass


def plot_growth_test_network(
    json_path: str | Path,
    *,
    network_path: str | Path | None = None,
    x_label: str = "x",
    y_label: str = "y",
    fs_labels: float = 20.0,
    fs_zstab: float = 15,
    L: float = 1024,
    legend_label_stab: str | None = None,
    legend_label_frontier: str | None = None,
    legend_label_posstab: str | None = None,
    fs_legend: float = 15,
    z_stab: int | None = None,
    t_stab: float | int | None = None,
    frontier: str | None = None,
    use_frontier_sites: bool = False,
    frontier_color: str = "#d1495b",
    frontier_linewidth: float = 1.35,
    show_frontier_line: bool = True,
    show_frontier_text: bool = True,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (4.2, 9.0),
    ax=None,
    title: str | None = None,
    fs_title: float = 20,
    return_partition: bool = False,
):
    """
    Plota a rede 2D compacta:
      - frontier='frontier_y' (ou use_frontier_sites=True): particiona exatamente pelos sítios no
        limiar da estabilização t = t_stab: z < h_front[x] (laranja) e z >= h_front[x] (roxo),
        traçando a curva ruidosa y = y_stab(x) nos sítios da fronteira física.
      - frontier='max_y': particiona em z < max(y_stab) (laranja) e z >= max(y_stab) (roxo),
        traçando a reta horizontal no valor máximo da altura da fronteira, com captions
        adaptados para y < max(y_stab), y = max(y_stab) e y >= max(y_stab).
      - frontier=None (padrão legado se use_frontier_sites=False): particiona horizontalmente
        em z < z_stab (laranja) e z >= z_stab (roxo), com reta horizontal z = z_stab.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _patch_pdf_writer_safe()

    if frontier is not None and frontier not in ("max_y", "frontier_y"):
        raise ValueError(f"frontier must be 'max_y', 'frontier_y', or None; got {frontier!r}")

    # Determina o modo
    if frontier is not None:
        mode = frontier
    elif use_frontier_sites or (legend_label_frontier is not None):
        mode = "frontier_y"
    else:
        mode = "legacy"

    # Define legendas padrão de acordo com o modo
    if mode == "frontier_y":
        if legend_label_stab is None:
            legend_label_stab = r"$y < y_{\mathrm{stab}}(x)$"
        if legend_label_frontier is None:
            legend_label_frontier = r"$y = y_{\mathrm{stab}}(x)$"
        if legend_label_posstab is None:
            legend_label_posstab = r"$y \geq y_{\mathrm{stab}}(x)$"
    elif mode == "max_y":
        if legend_label_stab is None:
            legend_label_stab = r"$y < \max(y_{\mathrm{stab}})$"
        if legend_label_frontier is None:
            legend_label_frontier = r"$y = \max(y_{\mathrm{stab}})$"
        if legend_label_posstab is None:
            legend_label_posstab = r"$y \geq \max(y_{\mathrm{stab}})$"
    else:
        if legend_label_stab is None:
            legend_label_stab = "stab"
        if legend_label_posstab is None:
            legend_label_posstab = "posstab"

    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta", {})
    params = parse_params_from_path(str(json_path.parent)) or {}
    resolved_L = params.get("L", meta.get("L", L))
    if not resolved_L:
        match_L = re.search(r"(?:^|[/\\])L_(\d+)(?:_|[/\\])", str(json_path))
        if match_L:
            resolved_L = int(match_L.group(1))
    L = int(resolved_L or 0)
    if L <= 0:
        raise ValueError(f"Could not determine L from {json_path}; pass L explicitly")

    if network_path is None:
        network_path = json_path.parent.parent / "network" / (json_path.stem + ".bin")

    if mode in ("frontier_y", "max_y"):
        part = get_growth_test_network_partition(
            json_path,
            network_path=network_path,
            z_stab=z_stab,
            t_stab=t_stab,
            L=L,
        )
        L = part["L"]
        height = part["height"]
        z_stab_val = part["z_stab"]
        h_front = part["h_front"]
        max_y = int(np.max(h_front))
        x = part["x"]
        z = part["z"]

        if mode == "frontier_y":
            before = part["before_mask"]
            after = part["after_mask"]
        else:
            before = z < max_y
            after = z >= max_y

        image = np.zeros((height, L), dtype=np.uint8)
        image[z[before], x[before]] = 1
        image[z[after], x[after]] = 2
    else:
        if z_stab is None:
            z_stab = meta.get("y_stat", meta.get("growth_test_z_stat_by_color"))
        if isinstance(z_stab, (list, tuple, np.ndarray)):
            z_stab = z_stab[0] if len(z_stab) else None
        if z_stab is None:
            raise ValueError("y_stab is missing from the sample; pass it explicitly")
        z_stab = int(z_stab)
        z_stab_val = z_stab
        max_y = z_stab

        x, z = _read_compact_network_2d(network_path, L)
        if x.size == 0:
            raise ValueError(f"No occupied sites in compact network: {network_path}")
        height = int(z.max()) + 1
        if z_stab < 0 or z_stab >= height:
            raise ValueError(f"y_stab={z_stab} lies outside network height [0, {height - 1}]")

        image = np.zeros((height, L), dtype=np.uint8)
        before = z < z_stab
        image[z[before], x[before]] = 1
        image[z[~before], x[~before]] = 2
        part = None

    cmap = ListedColormap(["#fff7ef", "#f28e2b", "#8a5cf6"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white", constrained_layout=True)
    else:
        fig = ax.figure

    ax.set_facecolor("#fff7ef")
    ax.imshow(
        image,
        origin="lower",
        interpolation="none",
        cmap=cmap,
        norm=norm,
        extent=(-0.5, L - 0.5, -0.5, height - 0.5),
        rasterized=False,
        aspect="equal",
    )

    if mode == "frontier_y":
        if show_frontier_line:
            x_line = np.arange(L)
            ax.plot(x_line, h_front, color=frontier_color, linewidth=frontier_linewidth, zorder=4)

        if show_frontier_text and z_stab_val is not None:
            ax.text(
                0.04 * L,
                z_stab_val + max(10, 0.006 * height),
                fr"$y_{{\mathrm{{stab}}}}={z_stab_val}$",
                ha="left",
                va="bottom",
                color="#000000",
                fontsize=fs_zstab,
                bbox={"facecolor": "#fff7ef", "edgecolor": "none", "alpha": 0.92, "pad": 1.5},
            )

        legend_handles = [
            Patch(facecolor="#f28e2b", edgecolor="none", label=legend_label_stab),
        ]
        if show_frontier_line and legend_label_frontier:
            legend_handles.append(
                Line2D([0], [0], color=frontier_color, linewidth=max(1.8, frontier_linewidth), label=legend_label_frontier)
            )
        legend_handles.append(
            Patch(facecolor="#8a5cf6", edgecolor="none", label=legend_label_posstab)
        )
    elif mode == "max_y":
        if show_frontier_line:
            ax.axhline(max_y, color=frontier_color, linewidth=frontier_linewidth, zorder=4)

        if show_frontier_text:
            ax.text(
                0.04 * L,
                max_y + max(10, 0.006 * height),
                fr"$\max(y_{{\mathrm{{stab}}}})={max_y}$",
                ha="left",
                va="bottom",
                color="#000000",
                fontsize=fs_zstab,
                bbox={"facecolor": "#fff7ef", "edgecolor": "none", "alpha": 0.92, "pad": 1.5},
            )

        legend_handles = [
            Patch(facecolor="#f28e2b", edgecolor="none", label=legend_label_stab),
        ]
        if show_frontier_line and legend_label_frontier:
            legend_handles.append(
                Line2D([0], [0], color=frontier_color, linewidth=max(1.8, frontier_linewidth), label=legend_label_frontier)
            )
        legend_handles.append(
            Patch(facecolor="#8a5cf6", edgecolor="none", label=legend_label_posstab)
        )
    else:  # legacy mode
        if show_frontier_line:
            ax.axhline(z_stab, color=frontier_color, linewidth=frontier_linewidth, zorder=4)

        if show_frontier_text:
            ax.text(
                0.04 * L,
                z_stab + max(10, 0.006 * height),
                fr"$y_{{\mathrm{{stab}}}}={z_stab}$",
                ha="left",
                va="bottom",
                color="#000000",
                fontsize=fs_zstab,
                bbox={"facecolor": "#fff7ef", "edgecolor": "none", "alpha": 0.92, "pad": 1.5},
            )

        legend_handles = [
            Patch(facecolor="#f28e2b", edgecolor="none", label=f"{legend_label_stab}"),
        ]
        if show_frontier_line and legend_label_frontier:
            legend_handles.append(
                Line2D([0], [0], color=frontier_color, linewidth=max(1.8, frontier_linewidth), label=legend_label_frontier)
            )
        legend_handles.append(
            Patch(facecolor="#8a5cf6", edgecolor="none", label=f"{legend_label_posstab}"),
        )

    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_xlabel(f"{x_label}", fontsize=fs_labels, color="#263238")
    ax.set_ylabel(f"{y_label}", fontsize=fs_labels, color="#263238")
    if title is not None:
        ax.set_title(title, fontsize=fs_title, color="#263238", pad=8)
    for spine in ax.spines.values():
        spine.set_color("#263238")
        spine.set_linewidth(0.9)

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        fontsize=fs_legend,
    )
    ax.set_xticks(np.arange(0, L + 1, 250))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")

    if return_partition:
        return fig, ax, part
    return fig, ax


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.collections as mcoll
from matplotlib.legend_handler import HandlerErrorbar
from pathlib import Path
from scipy.optimize import curve_fit


# ============================================================
# Funções Auxiliares de Arredondamento e Modelo
# ============================================================

def round_value_error(value, error):
    r"""
    Arredonda valor e erro de forma consistente.

    Regra:
    - 2 algarismos significativos no erro se o primeiro
      algarismo for 1 ou 2;
    - 1 algarismo significativo caso contrário.
    """
    if (
        not np.isfinite(value)
        or not np.isfinite(error)
        or error <= 0
    ):
        return str(value), str(error)

    exponent = int(np.floor(np.log10(abs(error))))
    first_digit = int(np.floor(error / 10**exponent))

    if first_digit in [1, 2]:
        n_sig = 2
    else:
        n_sig = 1

    decimals = -(exponent - (n_sig - 1))
    decimals = max(decimals, 0)

    error_rounded = round(error, decimals)
    value_rounded = round(value, decimals)

    value_str = f"{value_rounded:.{decimals}f}"
    error_str = f"{error_rounded:.{decimals}f}"

    return value_str, error_str


def finite_size_model(L, pc, A, nu):
    r"""
    Modelo de finite-size scaling:
    p*(L) = pc [1 - A L^(-1/nu)]
    """
    return pc * (1.0 - A * L**(-1.0 / nu))


def _get_aligned_data(type_perc, c_val=0.05, thresh=0.98):
    r"""
    Gera DataFrame de dados alinhados à reta suave de f_T com taxa de estabilização >= thresh.
    """
    candidates = [
        Path('../SOP_data/all_data_dynamic.dat'),
        Path('SOP_data/all_data_dynamic.dat'),
    ]
    data_path = None
    for p in candidates:
        if p.exists():
            data_path = p
            break
    if data_path is None:
        raise FileNotFoundError("Não foi possível encontrar 'all_data_dynamic.dat'.")

    ma_raw = pd.read_csv(data_path, sep=r'\s+')
    ma_mask = (
        (ma_raw['type_perc'] == type_perc)
        & (ma_raw['dim'] == 2) & (ma_raw['nc'] == 1)
        & (ma_raw['order'] == 0) & (ma_raw['stat_window'] == 0)
        & (ma_raw['control_rule'] == 'linear')
    )
    for col, val in {'c': c_val, 'p0': 0.8, 'P0': 0.2, 'rho': 1.0}.items():
        if col in ma_raw.columns:
            ma_mask &= np.isclose(ma_raw[col], val, rtol=0, atol=1e-12)

    ma_data = ma_raw.loc[ma_mask].copy()
    ma_data['all_ok'] = ma_data['N_samples'].eq(ma_data['N_samples_perc']) & ma_data['N_samples'].ge(5)

    # 1. Encontra o menor f_T com 100% de estabilização para cada L
    rows_min = []
    for m_L, m_g in ma_data.groupby('L'):
        m_g = m_g.sort_values('f_T')
        m_ok = m_g.loc[m_g['all_ok'] & m_g['p_mean'].between(0, 0.9, inclusive='left') & m_g['f_T'].le(0.4)]
        if m_ok.empty:
            continue
        m_r = m_ok.iloc[0]
        rows_min.append({
            'L': int(m_L),
            'ft_observado': m_r['f_T'],
            'p_at_min': m_r['p_mean'],
            'p_err': m_r['p_err']
        })
    df_min = pd.DataFrame(rows_min).sort_values('L').reset_index(drop=True)

    # 2. Ajuste da reta suave de f_T em log-log: log(f_T) = a * log(L/512) + b
    L_vals = df_min['L'].to_numpy(float)
    x = np.log(L_vals / 512.0)
    logf = np.log(df_min['ft_observado'].to_numpy())
    coef = np.polyfit(x, logf, 1)
    f_pred = np.exp(np.polyval(coef, x))

    # 3. Para cada L, seleciona o ponto com (N_samples_perc/N_samples) >= thresh mais próximo da reta suave
    chosen = []
    for l_val, ft_target in zip(L_vals, f_pred):
        pool = ma_data.loc[
            (ma_data['L'] == l_val)
            & ((ma_data['N_samples_perc'] / ma_data['N_samples']) >= thresh)
            & ma_data['N_samples'].ge(5)
            & ma_data['p_mean'].between(0, 0.9, inclusive='left')
            & ma_data['f_T'].le(0.4)
        ].copy()
        if pool.empty:
            continue
        pool['diff'] = (pool['f_T'] - ft_target).abs()
        best = pool.sort_values('diff').iloc[0]
        chosen.append({
            'L': int(l_val),
            'p_usado': best['p_mean'],
            'p_err_usado': best['p_err'],
            'ft_usado': best['f_T'],
            'ft_pred_reta': ft_target,
            'N_samples': int(best['N_samples']),
            'N_samples_perc': int(best['N_samples_perc'])
        })
    return pd.DataFrame(chosen)


def _load_or_resolve_data(
    data,
    type_perc,
    c_val=0.05,
    use_aligned=False,
    thresh=0.98,
    L_min=None,
):
    r"""
    Carrega dados do arquivo CSV padrão, extrai colunas de um DataFrame fornecido,
    ou resolve automaticamente os dados alinhados à reta suave (use_aligned=True).
    Aplica filtro opcional L >= L_min.
    """
    if data is None and use_aligned:
        # Se houver dados refinados de f_T contínuo gerados pelo run_samples, prioriza-os
        refined_candidates = [
            Path(f'../SOP_data/ft_min_max_2D_{type_perc}_smooth_refined.csv'),
            Path(f'SOP_data/ft_min_max_2D_{type_perc}_smooth_refined.csv'),
        ]
        found_refined = None
        for p in refined_candidates:
            if p.exists():
                found_refined = pd.read_csv(p)
                break
        if found_refined is not None:
            df = found_refined
        else:
            df = _get_aligned_data(type_perc, c_val=c_val, thresh=thresh)
    elif data is None:
        candidates = [
            Path(f'../SOP_data/ft_min_max_2D_{type_perc}_smooth_refined.csv'),
            Path(f'SOP_data/ft_min_max_2D_{type_perc}_smooth_refined.csv'),
            Path(f'../SOP_data/ft_min_max_2D_{type_perc}_linear.csv'),
            Path(f'SOP_data/ft_min_max_2D_{type_perc}_linear.csv'),
        ]
        found = None
        for p in candidates:
            if p.exists():
                found = pd.read_csv(p)
                break
        if found is None:
            raise FileNotFoundError(f'Não foi possível encontrar os dados padrão para {type_perc}.')
        df = found
    elif isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    if 'c' in df.columns and c_val is not None:
        df = df[np.isclose(df['c'], c_val)]

    # Detecção automática de colunas compatíveis
    l_col = 'L' if 'L' in df.columns else df.columns[0]
    if L_min is not None:
        df = df[df[l_col] >= L_min]

    y_col = next((c for c in ['p_min', 'p_usado', 'p_mean', 'p_at_min', 'y'] if c in df.columns), None)
    err_col = next((c for c in ['p_min_err', 'p_err_usado', 'p_err', 'y_err', 'sigma'] if c in df.columns), None)

    if y_col is None or err_col is None:
        raise ValueError(f'Colunas de valor ou erro não encontradas em {df.columns}.')

    df = df.sort_values(l_col)
    L = df[l_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    y_err = df[err_col].to_numpy(dtype=float)

    valid = (
        np.isfinite(L)
        & np.isfinite(y)
        & np.isfinite(y_err)
        & (L > 0)
        & (y_err > 0)
    )
    return L[valid], y[valid], y_err[valid]


# ============================================================
# Função Principal de Plotagem para o Artigo
# ============================================================

def _get_ft_min_observed_data(type_perc, c_val=0.05, L_min=1024):
    r"""
    Retorna os menores valores observados de f_T (com estabilização >= 98%) para cada L >= L_min,
    lendo prioritariamente de ft_min_max_2D_{type_perc}_smooth_refined.csv.
    """
    for p in [
        Path(f'../SOP_data/ft_min_max_2D_{type_perc}_smooth_refined.csv'),
        Path(f'SOP_data/ft_min_max_2D_{type_perc}_smooth_refined.csv'),
    ]:
        if p.exists():
            df_csv = pd.read_csv(p)
            if 'f_T_min_obs' in df_csv.columns:
                if L_min is not None:
                    df_csv = df_csv[df_csv['L'] >= L_min]
                df_csv = df_csv.sort_values('L')
                return df_csv['L'].to_numpy(dtype=float), df_csv['f_T_min_obs'].to_numpy(dtype=float)

    # Mínimos discretos de referência (estabilização >= 98%) para c = 0.05
    baseline_minima = {
        'node': {
            512: 0.1931579, 645: 0.1778947, 813: 0.1626316,
            1024: 0.1525225, 1448: 0.1464674, 2048: 0.1321053,
            2896: 0.1168421, 4096: 0.1015789, 5793: 0.09328492,
            8192: 0.0870912, 11585: 0.08219182, 16384: 0.07230416,
        },
        'bond': {
            512: 0.2381803, 645: 0.2236842, 813: 0.2084211,
            1024: 0.1931579, 1448: 0.1778947, 2048: 0.1626316,
            2896: 0.1473684, 4096: 0.1325766, 5793: 0.1200485,
            8192: 0.1072443, 11585: 0.09604457, 16384: 0.0852577,
        },
    }
    d = baseline_minima.get(type_perc, {})
    items = sorted((k, v) for k, v in d.items() if (L_min is None or k >= L_min))
    L_arr = np.array([k for k, _ in items], dtype=float)
    ft_arr = np.array([v for _, v in items], dtype=float)
    return L_arr, ft_arr


def plot_article_fss(
    # --- Dados de Entrada ---
    df_node=None,
    df_bond=None,
    c_val=0.05,
    p0_node=None,
    p0_bond=None,
    use_aligned=False,
    thresh_aligned=0.98,
    L_min=1024,
    # --- Modelo e Expoente Nu ---
    fix_nu=False,
    nu_theoretical=4.0 / 3.0,
    display_nu_mode='nu',  # 'nu', 'both', 'inv_nu'
    # --- Dimensões da Figura ---
    figsize=(14.0, 11.2),
    dpi=300,
    # --- Títulos e Rótulos dos Painéis ---
    panel_titles=None,
    panel_label_ft_node=r'\textbf{(a) Site percolation}',
    panel_label_ft_bond=r'\textbf{(b) Bond percolation}',
    panel_label_node=r'\textbf{(c) Site percolation}',
    panel_label_bond=r'\textbf{(d) Bond percolation}',
    panel_label_pos_ft_node=(0.05, 0.19),
    panel_label_pos_ft_bond=(0.05, 0.19),
    panel_label_pos_node=(0.95, 0.19),
    panel_label_pos_bond=(0.95, 0.19),
    fontsize_panel_label=17,
    panel_label_ha='right',
    panel_label_va='bottom',
    xlabel_ft=r'$L$',
    ylabel_ft=r'$f_{\mathrm{T,min}}(L)$',
    xlabel=None,
    ylabel=r'$p^*(L; f_{\mathrm{T,min}})$',
    ylabel_node=None,
    ylabel_bond=None,
    fontsize_titles=16,
    fontsize_labels=18,
    # --- Ticks dos Eixos ---
    fontsize_ticks=14,
    tick_direction='in',
    tick_width_major=1.2,
    tick_length_major=6.0,
    tick_width_minor=0.8,
    tick_length_minor=3.5,
    top_ticks=True,
    right_ticks=True,
    # --- Limites do Eixo X ---
    xlim=None,
    xlim_node=None,
    xlim_bond=None,
    xlim_margin=0.00045,
    # --- Estilo dos Dados Experimentais ---
    color_data_node='red',
    color_data_bond='red',
    marker_node='o',
    marker_bond='s',
    marker_size=9.0,
    marker_facecolor='none',
    markeredgewidth=1.3,
    capsize=3.5,
    elinewidth=1.2,
    data_label=None,
    # --- Estilo das Linhas de Ajuste ---
    color_fit_node='black',
    color_fit_bond='black',
    linestyle_fit='-',
    linewidth_fit=2.2,
    # --- Legenda ---
    show_legend=True,
    legend_loc_ft_node='upper right',
    legend_loc_ft_bond='upper right',
    legend_loc_node='upper left',
    legend_loc_bond='upper left',
    fontsize_legend=15,
    legend_frameon=False,
    legend_handlelength=2.5,
    legend_borderaxespad=1.2,
    show_chi2=False,
    # --- Caixa de Texto com a Fórmula do Modelo ---
    show_formula=True,
    formula_text_ft=r'$f_{\mathrm{T,min}}(L) = A L^{-b}$',
    formula_pos_ft_node=(0.05, 0.07),
    formula_pos_ft_bond=(0.05, 0.07),
    formula_text=None,
    formula_pos_node=(0.95, 0.07),
    formula_pos_bond=(0.95, 0.07),
    fontsize_formula=16,
    formula_ha='right',
    formula_va='bottom',
    # --- Saída / Salvamento ---
    save_path=None,
    show_plot=True,
    verbose=True,
):
    r"""
    Constrói a figura de publicação em layout subplots 2x2:
      - Linha 1: f_{T,min}(L) vs L com ajuste Power-law (Site e Bond percolation)
      - Linha 2: p*(L; f_{T,min}) vs L^{-1/\nu*} (Site e Bond percolation)
    """
    if xlabel is None:
        xlabel = r'$L^{-3/4}$' if fix_nu else r'$L^{-1/\nu^\ast}$'

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    configs = [
        {
            'name': 'node',
            'ax_ft': axes[0, 0],
            'ax_p': axes[1, 0],
            'data': df_node,
            'title': panel_titles[0] if panel_titles else None,
            'panel_label_ft': panel_label_ft_node,
            'panel_label_pos_ft': panel_label_pos_ft_node,
            'legend_loc_ft': legend_loc_ft_node,
            'formula_pos_ft': formula_pos_ft_node,
            'panel_label': panel_label_node,
            'panel_label_pos': panel_label_pos_node,
            'ylabel': ylabel_node or ylabel,
            'color_data': color_data_node,
            'marker': marker_node,
            'color_fit': color_fit_node,
            'legend_loc': legend_loc_node,
            'formula_pos': formula_pos_node,
            'xlim': xlim_node or xlim,
            'p0': p0_node,
        },
        {
            'name': 'bond',
            'ax_ft': axes[0, 1],
            'ax_p': axes[1, 1],
            'data': df_bond,
            'title': panel_titles[1] if panel_titles else None,
            'panel_label_ft': panel_label_ft_bond,
            'panel_label_pos_ft': panel_label_pos_ft_bond,
            'legend_loc_ft': legend_loc_ft_bond,
            'formula_pos_ft': formula_pos_ft_bond,
            'panel_label': panel_label_bond,
            'panel_label_pos': panel_label_pos_bond,
            'ylabel': ylabel_bond or ylabel,
            'color_data': color_data_bond,
            'marker': marker_bond,
            'color_fit': color_fit_bond,
            'legend_loc': legend_loc_bond,
            'formula_pos': formula_pos_bond,
            'xlim': xlim_bond or xlim,
            'p0': p0_bond,
        },
    ]

    results = {}

    for cfg in configs:
        ax_ft = cfg['ax_ft']
        ax = cfg['ax_p']
        t_name = cfg['name']

        # =====================================================================
        # LINHA 1: f_{T,min}(L) versus L com ajuste Power-law
        # =====================================================================
        L_ft, ft_min = _get_ft_min_observed_data(t_name, c_val=c_val, L_min=L_min)

        p_pow, cov_pow = np.polyfit(np.log(L_ft), np.log(ft_min), 1, cov=True)
        b_ft = float(-p_pow[0])
        b_ft_err = float(np.sqrt(cov_pow[0, 0]))
        A_ft = float(np.exp(p_pow[1]))
        A_ft_err = float(A_ft * np.sqrt(cov_pow[1, 1]))

        ft_pred_pts = A_ft * (L_ft ** (-b_ft))
        r2_ft = float(1.0 - np.sum((ft_min - ft_pred_pts)**2) / np.sum((ft_min - np.mean(ft_min))**2))

        A_ft_s, A_ft_es = round_value_error(A_ft, A_ft_err)
        b_ft_s, b_ft_es = round_value_error(b_ft, b_ft_err)

        L_ft_dense = np.geomspace(np.min(L_ft) * 0.92, np.max(L_ft) * 1.08, 400)
        ft_fit_dense = A_ft * (L_ft_dense ** (-b_ft))

        ft_fit_label = (
            fr'$A = {A_ft_s} \pm {A_ft_es}$'
            '\n'
            fr'$b = {b_ft_s} \pm {b_ft_es}$'
            '\n'
            fr'$R^2 = {r2_ft:.4f}$'
        )

        # Curva Power-law preta
        ax_ft.plot(
            L_ft_dense, ft_fit_dense,
            color=cfg['color_fit'], linestyle=linestyle_fit,
            linewidth=linewidth_fit, label=ft_fit_label, zorder=2
        )

        # Pontos apenas dos menores valores (f_{T,min})
        ax_ft.plot(
            L_ft, ft_min,
            linestyle='none', marker=cfg['marker'], color=cfg['color_data'],
            markerfacecolor=marker_facecolor, markeredgecolor=cfg['color_data'],
            markeredgewidth=markeredgewidth, markersize=marker_size,
            label=data_label, zorder=3
        )

        ax_ft.set_xscale('log', base=2)
        ax_ft.set_xlim(np.min(L_ft) * 0.88, np.max(L_ft) * 1.14)

        if cfg['panel_label_ft']:
            ax_ft.text(
                cfg['panel_label_pos_ft'][0], cfg['panel_label_pos_ft'][1],
                cfg['panel_label_ft'],
                transform=ax_ft.transAxes, fontsize=fontsize_panel_label,
                horizontalalignment='left', verticalalignment=panel_label_va
            )

        if show_formula and formula_text_ft:
            ax_ft.text(
                cfg['formula_pos_ft'][0], cfg['formula_pos_ft'][1],
                formula_text_ft,
                transform=ax_ft.transAxes, fontsize=fontsize_formula,
                horizontalalignment='left', verticalalignment=formula_va
            )

        if cfg['title']:
            ax_ft.set_title(cfg['title'], fontsize=fontsize_titles)
        ax_ft.set_xlabel(xlabel_ft, fontsize=fontsize_labels)
        ax_ft.set_ylabel(ylabel_ft, fontsize=fontsize_labels)

        ax_ft.tick_params(
            axis='both', which='major',
            direction=tick_direction, length=tick_length_major, width=tick_width_major,
            labelsize=fontsize_ticks, top=top_ticks, right=right_ticks
        )
        ax_ft.tick_params(
            axis='both', which='minor',
            direction=tick_direction, length=tick_length_minor, width=tick_width_minor,
            top=top_ticks, right=right_ticks
        )
        ax_ft.minorticks_on()

        if show_legend:
            ax_ft.legend(
                loc=cfg['legend_loc_ft'],
                fontsize=fontsize_legend,
                frameon=legend_frameon,
                handlelength=legend_handlelength,
                borderaxespad=legend_borderaxespad
            )

        # =====================================================================
        # LINHA 2: Finite-Size Scaling de p*(L; f_{T,min})
        # =====================================================================
        L, y, y_err = _load_or_resolve_data(
            cfg['data'], t_name, c_val=c_val,
            use_aligned=use_aligned, thresh=thresh_aligned, L_min=L_min
        )
        min_pts = 3 if fix_nu else 4
        if len(L) < min_pts:
            raise ValueError(f'São necessários pelo menos {min_pts} pontos válidos para {t_name}. Encontrados: {len(L)}.')

        if fix_nu:
            def model_fixed(L_arr, pc, A):
                return pc * (1.0 - A * L_arr**(-1.0 / nu_theoretical))

            p0 = cfg['p0']
            if p0 is None or len(p0) < 2:
                p0_fit = [y[-1], -1.0]
            else:
                p0_fit = [p0[0], p0[1]]

            popt, pcov = curve_fit(
                model_fixed, L, y,
                sigma=y_err, absolute_sigma=True,
                p0=p0_fit, maxfev=100000
            )
            perr = np.sqrt(np.diag(pcov))
            pc_b, A_b = popt
            pc_e, A_e = perr
            nu_b, nu_e = nu_theoretical, 0.0
            inv_nu_b, inv_nu_e = 1.0 / nu_theoretical, 0.0

            y_fit_pts = model_fixed(L, *popt)
            res_sigma = (y - y_fit_pts) / y_err
            chi2 = float(np.sum(res_sigma**2))
            dof = len(y) - len(popt)
            chi2_red = chi2 / dof if dof > 0 else np.nan

            pc_s, pc_es = round_value_error(pc_b, pc_e)
            A_s, A_es = round_value_error(A_b, A_e)
            nu_s, nu_es = "4/3", "0"
            inv_nu_s, inv_nu_es = "3/4", "0"

            results[t_name] = {
                'L': L, 'y': y, 'y_err': y_err,
                'popt': popt, 'pcov': pcov, 'perr': perr,
                'pc': (pc_b, pc_e, pc_s, pc_es),
                'A': (A_b, A_e, A_s, A_es),
                'nu': (nu_b, nu_e, nu_s, nu_es),
                'inv_nu': (inv_nu_b, inv_nu_e, inv_nu_s, inv_nu_es),
                'chi2': chi2, 'dof': dof, 'chi2_red': chi2_red,
                'residuals_sigma': res_sigma,
                'fix_nu': True,
                'ft_fit': {'A': (A_ft, A_ft_err, A_ft_s, A_ft_es), 'b': (b_ft, b_ft_err, b_ft_s, b_ft_es), 'r2': r2_ft},
            }

            if display_nu_mode == 'both':
                nu_line = fr'$\nu^\ast = 4/3 \;\mathrm{{(fixo)}}$' + '\n' + fr'$1/\nu^\ast = 3/4 \;\mathrm{{(fixo)}}$'
            elif display_nu_mode == 'nu':
                nu_line = fr'$\nu^\ast = 4/3 \;\mathrm{{(fixo)}}$'
            else:
                nu_line = fr'$1/\nu^\ast = 3/4 \;\mathrm{{(fixo)}}$'

        else:
            p0 = cfg['p0']
            if p0 is None:
                p0 = [y[-1], -1.0, 4.0 / 3.0]

            popt, pcov = curve_fit(
                finite_size_model, L, y,
                sigma=y_err, absolute_sigma=True,
                p0=p0, maxfev=100000
            )
            perr = np.sqrt(np.diag(pcov))
            pc_b, A_b, nu_b = popt
            pc_e, A_e, nu_e = perr
            inv_nu_b = 1.0 / nu_b
            inv_nu_e = nu_e / (nu_b**2)

            y_fit_pts = finite_size_model(L, *popt)
            res_sigma = (y - y_fit_pts) / y_err
            chi2 = float(np.sum(res_sigma**2))
            dof = len(y) - len(popt)
            chi2_red = chi2 / dof if dof > 0 else np.nan

            pc_s, pc_es = round_value_error(pc_b, pc_e)
            A_s, A_es = round_value_error(A_b, A_e)
            nu_s, nu_es = round_value_error(nu_b, nu_e)
            inv_nu_s, inv_nu_es = round_value_error(inv_nu_b, inv_nu_e)

            results[t_name] = {
                'L': L, 'y': y, 'y_err': y_err,
                'popt': popt, 'pcov': pcov, 'perr': perr,
                'pc': (pc_b, pc_e, pc_s, pc_es),
                'A': (A_b, A_e, A_s, A_es),
                'nu': (nu_b, nu_e, nu_s, nu_es),
                'inv_nu': (inv_nu_b, inv_nu_e, inv_nu_s, inv_nu_es),
                'chi2': chi2, 'dof': dof, 'chi2_red': chi2_red,
                'residuals_sigma': res_sigma,
                'fix_nu': False,
                'ft_fit': {'A': (A_ft, A_ft_err, A_ft_s, A_ft_es), 'b': (b_ft, b_ft_err, b_ft_s, b_ft_es), 'r2': r2_ft},
            }

            if display_nu_mode == 'both':
                nu_line = fr'$\nu^\ast = {nu_s} \pm {nu_es}$' + '\n' + fr'$1/\nu^\ast = {inv_nu_s} \pm {inv_nu_es}$'
            elif display_nu_mode == 'nu':
                nu_line = fr'$\nu^\ast = {nu_s} \pm {nu_es}$'
            else:
                nu_line = fr'$1/\nu^\ast = {inv_nu_s} \pm {inv_nu_es}$'

        nu_plot = nu_theoretical if fix_nu else nu_b
        x = L**(-1.0 / nu_plot)

        x_max_val = float(np.max(x))
        if cfg['xlim'] is not None:
            x_lim_left, x_lim_right = cfg['xlim']
        else:
            x_lim_left = 0.0
            x_lim_right = x_max_val + xlim_margin

        x_fit = np.linspace(x_lim_left, x_lim_right, 500)
        y_fit = pc_b * (1.0 - A_b * x_fit)

        fit_label = (
            fr'$p_c^* = {pc_s} \pm {pc_es}$'
            '\n'
            + nu_line
        )
        if show_chi2:
            fit_label += '\n' + fr'$\chi^2_\mathrm{{red}} = {chi2_red:.2f}$'

        ax.plot(
            x_fit, y_fit,
            color=cfg['color_fit'], linestyle=linestyle_fit,
            linewidth=linewidth_fit, label=fit_label, zorder=2
        )

        ax.errorbar(
            x, y, yerr=y_err,
            fmt=cfg['marker'], color=cfg['color_data'],
            markerfacecolor=marker_facecolor, markeredgecolor=cfg['color_data'],
            markeredgewidth=markeredgewidth,
            markersize=marker_size, capsize=capsize, elinewidth=elinewidth,
            label=data_label, zorder=3
        )

        if cfg['panel_label']:
            ax.text(
                cfg['panel_label_pos'][0], cfg['panel_label_pos'][1],
                cfg['panel_label'],
                transform=ax.transAxes, fontsize=fontsize_panel_label,
                horizontalalignment=panel_label_ha, verticalalignment=panel_label_va
            )

        if show_formula:
            if formula_text is not None:
                f_txt = formula_text
            elif fix_nu:
                f_txt = r'$p^*(L; f_{\mathrm{T,min}}) = p_c^* (1 - A L^{-4/3})$'
            else:
                f_txt = r'$p^*(L; f_{\mathrm{T,min}}) = p_c^* (1 - A L^{-1/\nu^*})$'

            if f_txt:
                ax.text(
                    cfg['formula_pos'][0], cfg['formula_pos'][1], f_txt,
                    transform=ax.transAxes, fontsize=fontsize_formula,
                    horizontalalignment=formula_ha, verticalalignment=formula_va
                )

        ax.set_xlim(x_lim_left, x_lim_right)
        ax.set_xlabel(xlabel, fontsize=fontsize_labels)
        ax.set_ylabel(cfg['ylabel'], fontsize=fontsize_labels)

        ax.tick_params(
            axis='both', which='major',
            direction=tick_direction, length=tick_length_major, width=tick_width_major,
            labelsize=fontsize_ticks, top=top_ticks, right=right_ticks
        )
        ax.tick_params(
            axis='both', which='minor',
            direction=tick_direction, length=tick_length_minor, width=tick_width_minor,
            top=top_ticks, right=right_ticks
        )
        ax.minorticks_on()

        if show_legend:
            ax.legend(
                loc=cfg['legend_loc'],
                fontsize=fontsize_legend,
                frameon=legend_frameon,
                handlelength=legend_handlelength,
                borderaxespad=legend_borderaxespad
            )

    fig.tight_layout()

    if verbose:
        print('=' * 75)
        title_summary = "COM NU FIXO = 4/3 (THEORETICAL)" if fix_nu else "COM NU ESTIMADO LIVRE"
        print(f'RESUMO DOS AJUSTES FINITE-SIZE SCALING: {title_summary}')
        print('=' * 75)
        for t_name in ['node', 'bond']:
            res = results[t_name]
            _, _, pc_s, pc_es = res['pc']
            _, _, A_s, A_es = res['A']
            _, _, nu_s, nu_es = res['nu']
            _, _, inv_nu_s, inv_nu_es = res['inv_nu']
            ft_info = res['ft_fit']
            print(f'[{t_name.upper()} PERCOLATION]')
            print(f'  f_T,min(L): A = {ft_info["A"][2]} +/- {ft_info["A"][3]}, b = {ft_info["b"][2]} +/- {ft_info["b"][3]} (R^2 = {ft_info["r2"]:.4f})')
            print(f'  pc*       = {pc_s} +/- {pc_es}')
            print(f'  A         = {A_s} +/- {A_es}')
            if fix_nu:
                print(f'  nu*       = 4/3 (fixo)')
                print(f'  1/nu*     = 3/4 (fixo)')
            else:
                print(f'  nu*       = {nu_s} +/- {nu_es}')
                print(f'  1/nu*     = {inv_nu_s} +/- {inv_nu_es}')
            print(f'  chi2      = {res["chi2"]:.4f}  (dof = {res["dof"]})')
            print(f'  chi2_red  = {res["chi2_red"]:.4f}')
            print(f'  max |res| = {np.max(np.abs(res["residuals_sigma"])):.2f} sigma')
            print('-' * 75)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig, axes, results


# ============================================================
# Finite-Size Scaling para um f_T fixo (ex: f_T = 0.30)
# ============================================================

def _fit_fss_robust(L, y, y_err):
    r"""
    Obtém um chute inicial ótimo para finite_size_model varrendo nu em uma grade fina
    (onde p*(L) = p_c* - B * L^{-1/nu} é estritamente linear em L^{-1/nu}) e em seguida
    refina via curve_fit não-linear com pesos exatos 1/y_err^2.
    """
    w = 1.0 / (y_err ** 2)
    W = np.sum(w)
    yb = np.sum(w * y) / W
    best_chi2 = np.inf
    best_p0 = [float(y[-1]), -1.0, 4.0 / 3.0]
    for nu_try in np.linspace(0.25, 2.5, 500):
        x = L ** (-1.0 / nu_try)
        xb = np.sum(w * x) / W
        denom = np.sum(w * (x - xb) ** 2)
        if denom <= 0:
            continue
        c1 = np.sum(w * (x - xb) * (y - yb)) / denom
        c0 = yb - c1 * xb
        if c0 <= 0:
            continue
        chi2 = float(np.sum(w * (y - (c0 + c1 * x)) ** 2))
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_p0 = [float(c0), float(-c1 / c0), float(nu_try)]
    popt, pcov = curve_fit(
        finite_size_model, L, y,
        sigma=y_err, absolute_sigma=True,
        p0=best_p0, maxfev=100000
    )
    return popt, pcov


def get_fixed_ft_data(
    type_perc,
    ft_val=0.3,
    c_val=0.05,
    L_min=1024,
    thresh=0.98,
    ft_tol=1e-3,
    data_path=None,
):
    r"""
    Extrai de all_data_dynamic.dat todos os tamanhos L >= L_min disponíveis para um valor fixo de f_T
    (dentro da tolerância ft_tol) e retorna (L, p_star, p_err, df_filtrado).
    """
    if data_path is not None:
        p_found = Path(data_path)
    else:
        p_found = None
        for p in [
            Path('../SOP_data/all_data_dynamic.dat'),
            Path('SOP_data/all_data_dynamic.dat'),
        ]:
            if p.exists():
                p_found = p
                break
    if p_found is None or not p_found.exists():
        raise FileNotFoundError("Não foi possível encontrar 'all_data_dynamic.dat'.")

    raw = pd.read_csv(p_found, sep=r'\s+')
    mask = (
        (raw['type_perc'] == type_perc)
        & (raw['dim'] == 2)
        & (raw['nc'] == 1)
        & (raw['order'] == 0)
        & (raw['stat_window'] == 0)
        & (raw['control_rule'] == 'linear')
    )
    for col, val in {'c': c_val, 'p0': 0.8, 'P0': 0.2, 'rho': 1.0}.items():
        if col in raw.columns:
            mask &= np.isclose(raw[col], val, rtol=0, atol=1e-12)

    df = raw.loc[mask].copy()
    if L_min is not None:
        df = df[df['L'] >= L_min]
    if thresh is not None and 'N_samples_perc' in df.columns and 'N_samples' in df.columns:
        df = df[(df['N_samples_perc'] / df['N_samples']) >= thresh]

    df['ft_diff'] = (df['f_T'] - ft_val).abs()
    df = df[df['ft_diff'] <= ft_tol]
    if df.empty:
        raise ValueError(
            f'Nenhum ponto encontrado para {type_perc} com f_T = {ft_val} (tolerância={ft_tol}).'
        )

    # Seleciona para cada L a linha mais próxima do f_T solicitado
    df = (
        df.sort_values(['L', 'ft_diff'])
        .groupby('L', as_index=False)
        .first()
        .sort_values('L')
        .reset_index(drop=True)
    )

    L = df['L'].to_numpy(dtype=float)
    y = df['p_mean'].to_numpy(dtype=float)
    y_err = df['p_err'].to_numpy(dtype=float)
    return L, y, y_err, df


def plot_fss_fixed_ft(
    ft_val=0.3,
    c_val=0.05,
    L_min=1024,
    thresh=0.98,
    ft_tol=1e-3,
    fix_nu=False,
    nu_theoretical=4.0 / 3.0,
    x_mode='both',  # 'both' (2x2: vs L e vs L^{-1/nu*}), 'scaled' (1x2: vs L^{-1/nu*}), 'L' (1x2: vs L)
    show_A_in_legend=False,
    figsize=None,
    dpi=300,
    save_path=None,
    show_plot=True,
    verbose=True,
):
    r"""
    Realiza o ajuste de Finite-Size Scaling de p*(L; f_T) para um f_T fixo (truncando L >= L_min):
        p*(L; f_T) = p_c* (1 - A L^{-1/\nu*})
    para Site (node) e Bond percolation.
    """
    if figsize is None:
        figsize = (14.0, 11.2) if x_mode == 'both' else (14.0, 5.6)

    if x_mode == 'both':
        fig, axes = plt.subplots(2, 2, figsize=figsize)
    else:
        fig, axes_1d = plt.subplots(1, 2, figsize=figsize)
        axes = np.array([axes_1d])

    ft_str = f'{ft_val:.2f}' if abs(ft_val * 100 - round(ft_val * 100)) < 1e-3 else f'{ft_val:.4f}'
    ylabel_str = fr'$p^*(L; f_{{\mathrm{{T}}}}={ft_str})$'
    xlabel_scaled = r'$L^{-3/4}$' if fix_nu else r'$L^{-1/\nu^\ast}$'
    f_txt = (
        r'$p^*(L; f_{\mathrm{T}}) = p_c^* (1 - A L^{-4/3})$'
        if fix_nu
        else r'$p^*(L; f_{\mathrm{T}}) = p_c^* (1 - A L^{-1/\nu^*})$'
    )

    configs = [
        {
            'name': 'node',
            'marker': 'o',
            'color_data': 'red',
            'color_fit': 'black',
            'label_top': r'\textbf{(a) Site percolation}',
            'label_bot': r'\textbf{(c) Site percolation}' if x_mode == 'both' else r'\textbf{(a) Site percolation}',
        },
        {
            'name': 'bond',
            'marker': 's',
            'color_data': 'red',
            'color_fit': 'black',
            'label_top': r'\textbf{(b) Bond percolation}',
            'label_bot': r'\textbf{(d) Bond percolation}' if x_mode == 'both' else r'\textbf{(b) Bond percolation}',
        },
    ]

    results = {}

    for j, cfg in enumerate(configs):
        t_name = cfg['name']
        L, y, y_err, df_sub = get_fixed_ft_data(
            t_name, ft_val=ft_val, c_val=c_val, L_min=L_min, thresh=thresh, ft_tol=ft_tol
        )

        if fix_nu:
            def model_fixed(L_arr, pc, A):
                return pc * (1.0 - A * L_arr**(-1.0 / nu_theoretical))

            popt, pcov = curve_fit(
                model_fixed, L, y,
                sigma=y_err, absolute_sigma=True,
                p0=[y[-1], -1.0], maxfev=100000
            )
            perr = np.sqrt(np.diag(pcov))
            pc_b, A_b = popt
            pc_e, A_e = perr
            nu_b, nu_e = nu_theoretical, 0.0
            inv_nu_b, inv_nu_e = 1.0 / nu_theoretical, 0.0
            y_fit_pts = model_fixed(L, *popt)
            pc_s, pc_es = round_value_error(pc_b, pc_e)
            A_s, A_es = round_value_error(A_b, A_e)
            nu_s, nu_es = '4/3', '0'
            inv_nu_s, inv_nu_es = '3/4', '0'
            nu_line = r'$\nu^\ast = 4/3 \;\mathrm{(fixo)}$'
        else:
            popt, pcov = _fit_fss_robust(L, y, y_err)
            perr = np.sqrt(np.diag(pcov))
            pc_b, A_b, nu_b = popt
            pc_e, A_e, nu_e = perr
            inv_nu_b = 1.0 / nu_b
            inv_nu_e = nu_e / (nu_b**2)
            y_fit_pts = finite_size_model(L, *popt)
            pc_s, pc_es = round_value_error(pc_b, pc_e)
            A_s, A_es = round_value_error(A_b, A_e)
            nu_s, nu_es = round_value_error(nu_b, nu_e)
            inv_nu_s, inv_nu_es = round_value_error(inv_nu_b, inv_nu_e)
            nu_line = fr'$\nu^\ast = {nu_s} \pm {nu_es}$'

        res_sigma = (y - y_fit_pts) / y_err
        chi2 = float(np.sum(res_sigma**2))
        dof = len(y) - len(popt)
        chi2_red = chi2 / dof if dof > 0 else np.nan
        ss_res = float(np.sum((y - y_fit_pts)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        results[t_name] = {
            'L': L, 'y': y, 'y_err': y_err, 'df': df_sub,
            'popt': popt, 'pcov': pcov, 'perr': perr,
            'pc': (pc_b, pc_e, pc_s, pc_es),
            'A': (A_b, A_e, A_s, A_es),
            'nu': (nu_b, nu_e, nu_s, nu_es),
            'inv_nu': (inv_nu_b, inv_nu_e, inv_nu_s, inv_nu_es),
            'chi2': chi2, 'dof': dof, 'chi2_red': chi2_red, 'r2': r2,
            'residuals_sigma': res_sigma,
        }

        if show_A_in_legend:
            fit_label = (
                fr'$p_c^* = {pc_s} \pm {pc_es}$'
                '\n'
                fr'$A = {A_s} \pm {A_es}$'
                '\n'
                + nu_line
            )
        else:
            fit_label = fr'$p_c^* = {pc_s} \pm {pc_es}$' + '\n' + nu_line

        # Painel p*(L; f_T) versus L
        if x_mode in ('both', 'L'):
            ax_l = axes[0, j]
            L_dense = np.geomspace(np.min(L) * 0.92, np.max(L) * 1.08, 500)
            y_dense = pc_b * (1.0 - A_b * L_dense**(-1.0 / nu_b))
            ax_l.plot(
                L_dense, y_dense,
                color=cfg['color_fit'], linestyle='-', linewidth=2.2,
                label=fit_label, zorder=2
            )
            ax_l.errorbar(
                L, y, yerr=y_err,
                fmt=cfg['marker'], color=cfg['color_data'],
                markerfacecolor='none', markeredgecolor=cfg['color_data'],
                markeredgewidth=1.3, markersize=9.0, capsize=3.5, elinewidth=1.2,
                zorder=3
            )
            ax_l.set_xscale('log', base=2)
            ax_l.set_xlim(np.min(L) * 0.88, np.max(L) * 1.14)
            ax_l.set_xlabel(r'$L$', fontsize=18)
            ax_l.set_ylabel(ylabel_str, fontsize=18)
            ax_l.text(
                0.95, 0.50, cfg['label_top'],
                transform=ax_l.transAxes, fontsize=17,
                horizontalalignment='right', verticalalignment='bottom'
            )
            ax_l.text(
                0.95, 0.37, f_txt,
                transform=ax_l.transAxes, fontsize=16,
                horizontalalignment='right', verticalalignment='bottom'
            )
            ax_l.legend(
                loc='upper right', fontsize=15, frameon=False,
                handlelength=2.5, borderaxespad=1.2
            )
            ax_l.tick_params(
                axis='both', which='major', direction='in',
                length=6.0, width=1.2, labelsize=14, top=True, right=True
            )
            ax_l.tick_params(
                axis='both', which='minor', direction='in',
                length=3.5, width=0.8, top=True, right=True
            )
            ax_l.minorticks_on()

        # Painel p*(L; f_T) versus L^{-1/nu*}
        if x_mode in ('both', 'scaled'):
            ax_s = axes[1, j] if x_mode == 'both' else axes[0, j]
            x = L**(-1.0 / nu_b)
            x_max = float(np.max(x)) * 1.08
            x_fit = np.linspace(0.0, x_max, 500)
            y_fit = pc_b * (1.0 - A_b * x_fit)
            ax_s.plot(
                x_fit, y_fit,
                color=cfg['color_fit'], linestyle='-', linewidth=2.2,
                label=fit_label, zorder=2
            )
            ax_s.errorbar(
                x, y, yerr=y_err,
                fmt=cfg['marker'], color=cfg['color_data'],
                markerfacecolor='none', markeredgecolor=cfg['color_data'],
                markeredgewidth=1.3, markersize=9.0, capsize=3.5, elinewidth=1.2,
                zorder=3
            )
            ax_s.set_xlim(0.0, x_max)
            ax_s.set_xlabel(xlabel_scaled, fontsize=18)
            ax_s.set_ylabel(ylabel_str, fontsize=18)
            ax_s.text(
                0.95, 0.19, cfg['label_bot'],
                transform=ax_s.transAxes, fontsize=17,
                horizontalalignment='right', verticalalignment='bottom'
            )
            ax_s.text(
                0.95, 0.07, f_txt,
                transform=ax_s.transAxes, fontsize=16,
                horizontalalignment='right', verticalalignment='bottom'
            )
            ax_s.legend(
                loc='upper left', fontsize=15, frameon=False,
                handlelength=2.5, borderaxespad=1.2
            )
            ax_s.tick_params(
                axis='both', which='major', direction='in',
                length=6.0, width=1.2, labelsize=14, top=True, right=True
            )
            ax_s.tick_params(
                axis='both', which='minor', direction='in',
                length=3.5, width=0.8, top=True, right=True
            )
            ax_s.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2), useMathText=True)
            ax_s.minorticks_on()

    fig.tight_layout()

    if verbose:
        print('=' * 75)
        mode_str = 'NU FIXO = 4/3' if fix_nu else 'NU LIVRE'
        print(f'AJUSTE FSS PARA f_T FIXO = {ft_val} ({mode_str}, L >= {L_min})')
        print('=' * 75)
        for t_name in ['node', 'bond']:
            res = results[t_name]
            _, _, pc_s, pc_es = res['pc']
            _, _, A_s, A_es = res['A']
            _, _, nu_s, nu_es = res['nu']
            _, _, inv_nu_s, inv_nu_es = res['inv_nu']
            print(f'[{t_name.upper()} PERCOLATION] (N_pontos = {len(res["L"])}, L in [{int(min(res["L"]))}, {int(max(res["L"]))}])')
            print(f'  pc*       = {pc_s} +/- {pc_es}')
            print(f'  A         = {A_s} +/- {A_es}')
            if fix_nu:
                print('  nu*       = 4/3 (fixo)')
                print('  1/nu*     = 3/4 (fixo)')
            else:
                print(f'  nu*       = {nu_s} +/- {nu_es}')
                print(f'  1/nu*     = {inv_nu_s} +/- {inv_nu_es}')
            print(f'  chi2_red  = {res["chi2_red"]:.4f} (chi2 = {res["chi2"]:.2f}, dof = {res["dof"]}, R2 = {res["r2"]:.6f})')
            print('-' * 75)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig, axes, results


# ============================================================
# Estudo de convergência de p_c* e nu_eff* conforme variamos f_T (L >= 1024)
# ============================================================

def analyze_ft_convergence(c_val=0.05, thresh=0.98, L_min=1024):
    r"""
    Avalia o comportamento de p_c* e nu_eff* usando duas estratégias complementares,
    truncando sempre L >= L_min (padrão L_min = 1024) em todos os casos:
      1) const_ft: valores fixos decrescentes de f_T na grade (mantendo os 9 tamanhos L >= 1024).
      2) scaled_ft_min: escalonamento proporcional f_T(L) = \lambda f_{T,min}(L) (\lambda -> 1.0),
         mantendo os mesmos 9 tamanhos L >= 1024 ativos em uma mesma lei de potência.
    """
    p_found = None
    for p in [Path('../SOP_data/all_data_dynamic.dat'), Path('SOP_data/all_data_dynamic.dat')]:
        if p.exists():
            p_found = p
            break
    if p_found is None:
        raise FileNotFoundError("Não foi possível encontrar 'all_data_dynamic.dat'.")

    raw = pd.read_csv(p_found, sep=r'\s+')
    mask = (
        (raw['dim'] == 2)
        & (raw['nc'] == 1)
        & (raw['order'] == 0)
        & (raw['stat_window'] == 0)
        & (raw['control_rule'] == 'linear')
        & np.isclose(raw['c'], c_val)
        & np.isclose(raw['p0'], 0.8)
        & np.isclose(raw['P0'], 0.2)
        & np.isclose(raw['rho'], 1.0)
    )
    sub = raw.loc[mask].copy()
    sub['stab'] = sub['N_samples_perc'] / sub['N_samples']
    sub_ok = sub[
        (sub['stab'] >= thresh)
        & (sub['N_samples'] >= 5)
        & (sub['p_mean'] < 0.9)
        & (sub['f_T'] <= 0.305)
        & (sub['L'] >= L_min)
    ].copy()

    def interp_p(df_L, ft_target):
        df_L = df_L.sort_values('f_T')
        fts = df_L['f_T'].to_numpy(float)
        ps = df_L['p_mean'].to_numpy(float)
        errs = df_L['p_err'].to_numpy(float)
        return float(np.interp(ft_target, fts, ps)), float(np.interp(ft_target, fts, errs))

    out = {'const_ft': {}, 'scaled_ft_min': {}}

    # Valores da grade onde todos os 9 tamanhos L >= 1024 estão presentes
    ft_const_list_node = [
        0.3000000, 0.2847368, 0.2694737, 0.2542105, 0.2389474,
        0.2236842, 0.2084211, 0.1931579, 0.1778947, 0.1626316,
    ]
    ft_const_list_bond = [
        0.3000000, 0.2847368, 0.2694737, 0.2542105, 0.2389474,
        0.2236842, 0.2084211, 0.1931579,
    ]

    for tp, ft_list in [('node', ft_const_list_node), ('bond', ft_const_list_bond)]:
        s_tp = sub_ok[sub_ok['type_perc'] == tp].copy()
        rows_res = []
        for ft_val in ft_list:
            pts = []
            for l_val, g in s_tp.groupby('L'):
                g_near = g[np.abs(g['f_T'] - ft_val) <= 1e-3].sort_values(
                    by='f_T', key=lambda s: np.abs(s - ft_val)
                )
                if not g_near.empty:
                    r0 = g_near.iloc[0]
                    pts.append((l_val, float(r0['p_mean']), float(r0['p_err'])))
            if len(pts) < 9:
                continue
            L = np.array([x[0] for x in pts], float)
            y = np.array([x[1] for x in pts], float)
            y_err = np.array([x[2] for x in pts], float)
            popt, pcov = _fit_fss_robust(L, y, y_err)
            perr = np.sqrt(np.diag(pcov))
            y_pred = finite_size_model(L, *popt)
            chi2_r = float(np.sum(((y - y_pred) / y_err)**2) / (len(y) - 3))
            ss_res = float(np.sum((y - y_pred)**2))
            ss_tot = float(np.sum((y - np.mean(y))**2))
            r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
            pc_s, pc_es = round_value_error(popt[0], perr[0])
            nu_s, nu_es = round_value_error(popt[2], perr[2])
            rows_res.append({
                'Regime': f'{ft_val:.4f}',
                'ft_ref': ft_val,
                'N_L': len(L),
                'L_min': int(min(L)),
                'pc': popt[0], 'pc_err': perr[0], 'pc_str': f'{pc_s} +/- {pc_es}',
                'nu': popt[2], 'nu_err': perr[2], 'nu_str': f'{nu_s} +/- {nu_es}',
                'chi2_red': chi2_r,
                'r2': r2_val,
            })

        for p_ref in [
            Path(f'../SOP_data/ft_min_max_2D_{tp}_smooth_refined.csv'),
            Path(f'SOP_data/ft_min_max_2D_{tp}_smooth_refined.csv'),
        ]:
            if p_ref.exists():
                df_ref = pd.read_csv(p_ref)
                break
        df_ref = df_ref[df_ref['L'] >= L_min].sort_values('L')
        L_r = df_ref['L'].to_numpy(float)
        y_r = df_ref['p_min'].to_numpy(float)
        err_r = df_ref['p_min_err'].to_numpy(float)
        popt, pcov = _fit_fss_robust(L_r, y_r, err_r)
        perr = np.sqrt(np.diag(pcov))
        y_r_pred = finite_size_model(L_r, *popt)
        chi2_r = float(np.sum(((y_r - y_r_pred) / err_r)**2) / (len(y_r) - 3))
        ss_res = float(np.sum((y_r - y_r_pred)**2))
        ss_tot = float(np.sum((y_r - np.mean(y_r))**2))
        r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        pc_s, pc_es = round_value_error(popt[0], perr[0])
        nu_s, nu_es = round_value_error(popt[2], perr[2])
        rows_res.append({
            'Regime': 'f_T,min(L)',
            'ft_ref': float(df_ref['f_T'].mean()),
            'N_L': len(L_r),
            'L_min': int(min(L_r)),
            'pc': popt[0], 'pc_err': perr[0], 'pc_str': f'{pc_s} +/- {pc_es}',
            'nu': popt[2], 'nu_err': perr[2], 'nu_str': f'{nu_s} +/- {nu_es}',
            'chi2_red': chi2_r,
            'r2': r2_val,
        })
        out['const_ft'][tp] = pd.DataFrame(rows_res)

    for tp in ['node', 'bond']:
        for p_ref in [
            Path(f'../SOP_data/ft_min_max_2D_{tp}_smooth_refined.csv'),
            Path(f'SOP_data/ft_min_max_2D_{tp}_smooth_refined.csv'),
        ]:
            if p_ref.exists():
                df_ref = pd.read_csv(p_ref)
                break
        df_ref = df_ref[df_ref['L'] >= L_min].sort_values('L')
        s_tp = sub_ok[sub_ok['type_perc'] == tp].copy()
        max_lam = 0.30 / df_ref['f_T'].max()
        lams = np.linspace(max_lam, 1.0, 6)
        rows_lam = []
        for idx_l, lam in enumerate(lams):
            if idx_l == len(lams) - 1:
                L_r = df_ref['L'].to_numpy(float)
                y_r = df_ref['p_min'].to_numpy(float)
                err_r = df_ref['p_min_err'].to_numpy(float)
                ft_mean = float(df_ref['f_T'].mean())
            else:
                pts = []
                for _, r_ref in df_ref.iterrows():
                    l_val = int(r_ref['L'])
                    ft_t = lam * r_ref['f_T']
                    g = s_tp[s_tp['L'] == l_val]
                    pv, ev = interp_p(g, ft_t)
                    pts.append((l_val, ft_t, pv, ev))
                L_r = np.array([x[0] for x in pts], float)
                ft_mean = float(np.mean([x[1] for x in pts]))
                y_r = np.array([x[2] for x in pts], float)
                err_r = np.array([x[3] for x in pts], float)
            popt, pcov = _fit_fss_robust(L_r, y_r, err_r)
            perr = np.sqrt(np.diag(pcov))
            y_r_pred = finite_size_model(L_r, *popt)
            chi2_r = float(np.sum(((y_r - y_r_pred) / err_r)**2) / (len(y_r) - 3))
            ss_res = float(np.sum((y_r - y_r_pred)**2))
            ss_tot = float(np.sum((y_r - np.mean(y_r))**2))
            r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
            pc_s, pc_es = round_value_error(popt[0], perr[0])
            nu_s, nu_es = round_value_error(popt[2], perr[2])
            rows_lam.append({
                'lambda': round(float(lam), 2),
                'ft_mean': round(ft_mean, 4),
                'N_L': len(L_r),
                'L_min': int(min(L_r)),
                'pc': popt[0], 'pc_err': perr[0], 'pc_str': f'{pc_s} +/- {pc_es}',
                'nu': popt[2], 'nu_err': perr[2], 'nu_str': f'{nu_s} +/- {nu_es}',
                'chi2_red': chi2_r,
                'r2': r2_val,
            })
        out['scaled_ft_min'][tp] = pd.DataFrame(rows_lam)

    return out


class _CleanErrorbarHandler(HandlerErrorbar):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        plotlines, caplines, barlinecols = orig_handle
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        ydata = np.full_like(xdata, (height - ydescent) / 2)
        xm = float(xdata_marker[0])
        ym = float(ydata[0])

        ms = plotlines.get_markersize()
        # 1. Linha guia no fundo
        legline = Line2D(xdata, ydata)
        self.update_prop(legline, plotlines, legend)
        legline.set_drawstyle('default')
        legline.set_marker('none')

        # 2. Marcador com fundo branco para que a linha guia não passe por dentro
        legline_marker = Line2D([xm], [ym])
        self.update_prop(legline_marker, plotlines, legend)
        legline_marker.set_linestyle('None')
        legline_marker.set_markerfacecolor('white')
        legline_marker.set_markeredgewidth(1.4)
        legline_marker.set_markersize(ms)

        # 3. Barra de erro vertical perfeitamente centrada no marcador
        yerr_size = ms * 0.75
        verts = [((xm, ym - yerr_size), (xm, ym + yerr_size))]
        coll = mcoll.LineCollection(verts)
        self.update_prop(coll, barlinecols[0] if barlinecols else plotlines, legend)
        coll.set_color(plotlines.get_color())
        coll.set_linewidth(1.8)

        cap_w = 3.5
        cap_top = Line2D([xm - cap_w, xm + cap_w], [ym + yerr_size, ym + yerr_size],
                         color=plotlines.get_color(), lw=1.4)
        cap_bot = Line2D([xm - cap_w, xm + cap_w], [ym - yerr_size, ym - yerr_size],
                         color=plotlines.get_color(), lw=1.4)

        artists = [legline, legline_marker, coll, cap_top, cap_bot]
        for a in artists:
            a.set_transform(trans)
        return artists


def plot_ft_convergence_study(c_val=0.05, thresh=0.98, L_min=1024, save_path=None, show_plot=True, verbose=True):
    r"""
    Plota a evolução de p_c* e \nu_eff* em função de f_T (Método 1: f_T constante;
    Método 2: f_T(L) = \lambda f_{T,min}(L)), truncando L >= L_min em todos os casos e
    destacando a convergência para os valores críticos teóricos (p_c e \nu = 4/3).
    """
    out = analyze_ft_convergence(c_val=c_val, thresh=thresh, L_min=L_min)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.6), sharex='col')

    configs = [
        {
            'tp': 'node',
            'col': 0,
            'pc_theo': 0.592746,
            'pc_theo_lbl': r'$p_c \approx 0.59275$',
            'title_pc': '(a) Site Percolation',
            'title_nu': '(c) Site Percolation',
            'marker': 'o',
        },
        {
            'tp': 'bond',
            'col': 1,
            'pc_theo': 0.500000,
            'pc_theo_lbl': r'$p_c = 0.50000$',
            'title_pc': '(b) Bond Percolation',
            'title_nu': '(d) Bond Percolation',
            'marker': 's',
        },
    ]

    for cfg in configs:
        tp = cfg['tp']
        j = cfg['col']
        df_c = out['const_ft'][tp]
        df_s = out['scaled_ft_min'][tp]

        # Separa os pontos de f_T constante do ponto limite f_{T,min}(L)
        df_c_only = df_c[df_c['Regime'] != 'f_T,min(L)']
        df_lim = df_c[df_c['Regime'] == 'f_T,min(L)'].iloc[0]

        ax_pc = axes[0, j]
        ax_nu = axes[1, j]

        # Inverte o eixo x da coluna no início para posicionamento correto da legenda
        ax_nu.invert_xaxis()

        # --- Painel Superior: p_{c,eff}* vs f_T ---
        # Somente marcadores, sem linhas guias e sem barras de erro
        ax_pc.plot(
            df_c_only['ft_ref'], df_c_only['pc'],
            marker=cfg['marker'], color='tab:blue', markerfacecolor='none',
            markeredgewidth=1.4, markersize=8.5, linestyle='none',
            label=r'$\mathrm{Constant}\;f_{\mathrm{T}}$'
        )
        ax_pc.plot(
            df_s['ft_mean'], df_s['pc'],
            marker=cfg['marker'], color='red', markerfacecolor='none',
            markeredgewidth=1.4, markersize=8.5, linestyle='none',
            label=r'$f_{\mathrm{T}}(L) = \lambda\, f_{\mathrm{T,min}}(L)$'
        )
        ax_pc.plot(
            df_lim['ft_ref'], df_lim['pc'],
            marker='*', color='black', markersize=14, linestyle='none',
            label=r'$\mathrm{Limit}\;f_{\mathrm{T,min}}(L)$', zorder=5
        )
        ax_pc.axhline(cfg['pc_theo'], color='black', linestyle=':', linewidth=1.6, label=cfg['pc_theo_lbl'])
        if j == 0:
            ax_pc.set_ylabel(r'$p_{\mathrm{c,eff}}^*$', fontsize=18)
        ax_pc.text(0.05, 0.12, cfg['title_pc'], transform=ax_pc.transAxes, fontsize=16, fontweight='bold', ha='left', va='bottom')
        ax_pc.legend(loc='upper right', fontsize=13.5, frameon=False)
        ax_pc.tick_params(
            axis='both', which='major', direction='in',
            length=6.0, width=1.2, labelsize=14, top=True, right=True, labelbottom=False
        )
        ax_pc.tick_params(axis='both', which='minor', direction='in', length=3.5, width=0.8, top=True, right=True)
        ax_pc.minorticks_on()

        # --- Painel Inferior: nu_eff* vs f_T ---
        # 1. Linhas guias conectando os marcadores (zorder=2)
        ax_nu.plot(df_c_only['ft_ref'], df_c_only['nu'], '-', color='tab:blue', lw=1.8, zorder=2)
        ax_nu.plot(df_s['ft_mean'], df_s['nu'], '--', color='red', lw=1.8, zorder=2)

        # 2. Marcadores com fundo branco para que a linha guia não passe por dentro (zorder=3)
        ax_nu.plot(
            df_c_only['ft_ref'], df_c_only['nu'], linestyle='none', marker=cfg['marker'],
            color='tab:blue', markerfacecolor='white', markeredgewidth=1.4, markersize=8.5, zorder=3
        )
        ax_nu.plot(
            df_s['ft_mean'], df_s['nu'], linestyle='none', marker=cfg['marker'],
            color='red', markerfacecolor='white', markeredgewidth=1.4, markersize=8.5, zorder=3
        )

        # 3. Barras de erro por cima do marcador (zorder=4)
        ax_nu.errorbar(
            df_c_only['ft_ref'], df_c_only['nu'], yerr=df_c_only['nu_err'],
            fmt='none', ecolor='tab:blue', capsize=3.5, elinewidth=1.8, capthick=1.4, zorder=4
        )
        ax_nu.errorbar(
            df_s['ft_mean'], df_s['nu'], yerr=df_s['nu_err'],
            fmt='none', ecolor='red', capsize=3.5, elinewidth=1.8, capthick=1.4, zorder=4
        )

        h_lim = ax_nu.plot(
            df_lim['ft_ref'], df_lim['nu'],
            marker='*', color='black', markersize=14, linestyle='none',
            label=r'$\mathrm{Limit}\;f_{\mathrm{T,min}}(L)$', zorder=5
        )[0]
        h_nu_theo = ax_nu.axhline(4.0 / 3.0, color='black', linestyle=':', linewidth=1.6, label=r'$\nu = 4/3 \approx 1.333$')

        # Handles para a legenda de nu
        h_c_dummy = ax_nu.errorbar(
            [], [], yerr=[], fmt=cfg['marker'] + '-', color='tab:blue',
            markerfacecolor='white', markeredgewidth=1.4, markersize=8.5, capsize=3.5, lw=1.8
        )
        h_s_dummy = ax_nu.errorbar(
            [], [], yerr=[], fmt=cfg['marker'] + '--', color='red',
            markerfacecolor='white', markeredgewidth=1.4, markersize=8.5, capsize=3.5, lw=1.8
        )

        y_min_nu = float(min(df_c_only['nu'].min(), df_s['nu'].min())) - 0.06
        ax_nu.set_ylim(y_min_nu, 1.46)
        ax_nu.set_xlabel(r'$f_{\mathrm{T}}\;\;(\mathrm{or}\;\langle f_{\mathrm{T}}(L)\rangle)$', fontsize=18)
        if j == 0:
            ax_nu.set_ylabel(r'$\nu_{\mathrm{eff}}^*$', fontsize=18)
        ax_nu.text(0.05, 0.90, cfg['title_nu'], transform=ax_nu.transAxes, fontsize=16, fontweight='bold', ha='left', va='bottom')

        handles_nu = [h_c_dummy, h_s_dummy, h_lim, h_nu_theo]
        labels_nu = [
            r'$\mathrm{Constant}\;f_{\mathrm{T}}$',
            r'$f_{\mathrm{T}}(L) = \lambda\, f_{\mathrm{T,min}}(L)$',
            r'$\mathrm{Limit}\;f_{\mathrm{T,min}}(L)$',
            r'$\nu = 4/3 \approx 1.333$',
        ]
        ax_nu.legend(
            handles=handles_nu, labels=labels_nu,
            handler_map={type(h_c_dummy): _CleanErrorbarHandler()},
            loc='lower right', fontsize=13.5, frameon=False
        )
        ax_nu.tick_params(
            axis='both', which='major', direction='in',
            length=6.0, width=1.2, labelsize=14, top=True, right=True, labelbottom=True
        )
        ax_nu.tick_params(axis='both', which='minor', direction='in', length=3.5, width=0.8, top=True, right=True)
        ax_nu.minorticks_on()

    fig.tight_layout()

    if verbose:
        for tp in ['node', 'bond']:
            print('=' * 85)
            print(f'TABELA 1 ({tp.upper()}): f_T CONSTANTE DECRESCENTE -> LIMITE f_T,min(L) (L >= {L_min})')
            print('=' * 85)
            print(out['const_ft'][tp][['Regime', 'N_L', 'L_min', 'pc_str', 'nu_str', 'chi2_red', 'r2']].to_string(index=False))
            print()
            print(f'TABELA 2 ({tp.upper()}): ESCALONAMENTO PROPORCIONAL f_T(L) = lambda * f_T,min(L) (L >= {L_min})')
            print('-' * 85)
            print(out['scaled_ft_min'][tp][['lambda', 'ft_mean', 'N_L', 'L_min', 'pc_str', 'nu_str', 'chi2_red', 'r2']].to_string(index=False))
            print()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig, axes, out


# ============================================================
# Exportações e aliases para uso interativo
# ============================================================
load_or_resolve_data = _load_or_resolve_data
get_aligned_data = _get_aligned_data

# Garante que 'from src.TimeSeriesAnalysis import *' também disponibilize os
# utilitários internos de FSS caso a função plot_article_fss seja copiada para um notebook:
__all__ = [name for name in dir() if not name.startswith('__')] + [
    '_load_or_resolve_data',
    '_get_aligned_data',
    'get_fixed_ft_data',
    'plot_fss_fixed_ft',
    'analyze_ft_convergence',
    'plot_ft_convergence_study',
]