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

import numpy as np
from scipy.interpolate import interp1d


# ============================================================
# Paths / Regex helpers
# ============================================================

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


# ------------------------ (B) bundle cache (many p0) ------------------------

def _compute_stats_for_p0(path_files: str, p0: float) -> tuple[dict, dict, dict]:
    """Compute ensemble stats for a single p0 reading all matching raw JSON files.
       Returns (stats_p, stats_N, counts_dict)."""
    times_list_p, p_values_list = [], []
    times_list_N, N_values_list = [], []
    files_used = 0

    for file in glob.glob(os.path.join(path_files, "*.json")):
        if not _filename_matches_p0(file, p0):
            continue
        recs = read_orders_one_file(file)
        if not recs:
            continue

        order, p, nt = recs[0]
        p = np.asarray(p, float)
        if p.size == 0:
            continue

        # build time vector
        if order is None:
            t = np.arange(p.shape[0], dtype=float)
        else:
            t = np.asarray(order)
            if t.ndim == 0 or t.size <= 1:
                t = np.arange(p.shape[0], dtype=float)

        used_any = False

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
            used_any = True

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
                used_any = True

        if used_any:
            files_used += 1

    stats_p = ensemble_stats(times_list_p, p_values_list)
    stats_N = ensemble_stats(times_list_N, N_values_list)
    counts = {
        "files_used": int(files_used),
        "runs_p": int(len(times_list_p)),
        "runs_N": int(len(times_list_N)),
    }
    return stats_p, stats_N, counts


def mean_properties_bundle(type_perc: str, num_colors: int, dim: int, L: int, NT: int,
                           k: float, rho: float, p0_list: list[float],
                           force_recompute: bool = False,
                           with_counts: bool = False
                           ) -> tuple[dict[float, dict], dict[float, dict]] | tuple[dict[float, dict], dict[float, dict], dict[float, dict]]:
    """
    Read (or create/update) a single JSON cache one level above /data with:
      {
        "meta": {...},
        "p_stats": { "<p0_tag>": <stats> },
        "N_stats": { "<p0_tag>": <stats> },
        "samples": { "<p0_tag>": {"files_used": int, "runs_p": int, "runs_N": int} }
      }

    Returns:
      p_stats_by_p0, N_stats_by_p0   (and optionally counts_by_p0 if with_counts=True)
    """
    base_path  = _cache_dir(type_perc, num_colors, dim, L, NT, k, rho)
    path_files = os.path.join(base_path, "data")
    bundle_fp  = _bundle_file_path(type_perc, num_colors, dim, L, NT, k, rho)

    os.makedirs(base_path, exist_ok=True)

    # try to load existing bundle
    bundle = None
    if os.path.exists(bundle_fp) and not force_recompute:
        try:
            with open(bundle_fp, "r") as f:
                bundle = json.load(f)
        except Exception:
            bundle = None

    data_mtime   = _latest_mtime(path_files)
    bundle_mtime = os.path.getmtime(bundle_fp) if bundle and os.path.exists(bundle_fp) else 0.0

    need_recompute_all = force_recompute or (bundle is None) or (bundle_mtime < data_mtime)
    if need_recompute_all:
        bundle = {
            "meta": {
                "version": _CACHE_VERSION,
                "type_perc": type_perc,
                "num_colors": num_colors,
                "dim": dim,
                "L": L,
                "NT": NT,
                "k": float(k),
                "rho": float(rho),
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "source_dir": path_files,
            },
            "p_stats": {},
            "N_stats": {},
            "samples": {},   # <-- new block
        }
    else:
        # ensure 'samples' exists for older bundles
        if "samples" not in bundle:
            bundle["samples"] = {}

    # ensure every requested p0 is present
    for p0 in p0_list:
        key = _safe_p0_tag(p0)
        need_this = (
            need_recompute_all
            or (key not in bundle.get("p_stats", {}))
            or (key not in bundle.get("N_stats", {}))
            or (key not in bundle.get("samples", {}))
        )
        if need_this:
            stats_p, stats_N, counts = _compute_stats_for_p0(path_files, p0)
            bundle["p_stats"][key]  = _stats_to_jsonable(stats_p)
            bundle["N_stats"][key]  = _stats_to_jsonable(stats_N)
            bundle["samples"][key]  = counts

    # save (updated) bundle
    try:
        with open(bundle_fp, "w") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save bundle at {bundle_fp}: {e}")

    # return unpacked dicts keyed by *float* p0
    p_stats_by_p0, N_stats_by_p0 = {}, {}
    counts_by_p0 = {}
    for p0 in p0_list:
        key = _safe_p0_tag(p0)
        p_stats_by_p0[float(p0)] = _stats_from_json(bundle["p_stats"][key])
        N_stats_by_p0[float(p0)] = _stats_from_json(bundle["N_stats"][key])
        c = bundle["samples"].get(key, {"files_used": 0, "runs_p": 0, "runs_N": 0})
        counts_by_p0[float(p0)] = {"files_used": int(c.get("files_used", 0)),
                                   "runs_p": int(c.get("runs_p", 0)),
                                   "runs_N": int(c.get("runs_N", 0))}

    if with_counts:
        return p_stats_by_p0, N_stats_by_p0, counts_by_p0
    return p_stats_by_p0, N_stats_by_p0
