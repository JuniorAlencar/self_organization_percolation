import re, os, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import math
from collections.abc import Iterable

# Aceita k/rho em float normal ou notação científica (ex.: 1.0e-04, 8.9e-02)
PARAMS_RE = re.compile(r"""
    (?P<type_perc>[A-Za-z]+)_percolation
    /num_colors_(?P<num_colors>\d+)
    /dim_(?P<dim>\d+)
    /L_(?P<L>\d+)
    /NT_constant/NT_(?P<Nt>\d+)
    /k_(?P<k>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)
    /rho_(?P<rho>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)
    /data
""", re.X)

def mean_nc_by_rho(df, num_colors=4, p0_filter=None, L_filter=None, k_filter=None):
    """
    Compute <n_c> (mean number of percolated colors per seed) as a function of rho,
    optionally filtering by p0, L, and k.

    k_filter supports:
      - scalar: exact match with np.isclose
      - 2-tuple/list: interval [k_min, k_max]
      - iterable of scalars: union of np.isclose matches
    Returns columns: ['L','rho','N','nc_mean','nc_sem'].
    """
    d = df.copy()

    # Ensure numeric types (robust if some columns are absent)
    for c in ["L","rho","order","num_samples","num_sample_perc","p0","k"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # p0 filter (keep original rounding-by-2-decimals behavior)
    if p0_filter is not None and "p0" in d.columns:
        d = d[d["p0"].round(2).eq(round(float(p0_filter), 2))]

    # L filter
    if L_filter is not None and "L" in d.columns:
        d = d[d["L"].eq(L_filter)]

    # k filter (scalar / interval / list)
    if k_filter is not None and "k" in d.columns:
        kseries = d["k"].astype(float).values
        if np.isscalar(k_filter):
            mask = np.isclose(kseries, float(k_filter), rtol=0.0, atol=1e-12)
        elif isinstance(k_filter, (list, tuple, np.ndarray)) and len(k_filter) == 2 and np.all(np.isfinite(k_filter)):
            kmin, kmax = float(min(k_filter)), float(max(k_filter))
            mask = (kseries >= kmin) & (kseries <= kmax)
        elif isinstance(k_filter, Iterable):
            vals = np.array(list(k_filter), dtype=float)
            mask = np.zeros(len(d), dtype=bool)
            for kv in vals:
                mask |= np.isclose(kseries, float(kv), rtol=0.0, atol=1e-12)
        else:
            raise ValueError("k_filter must be a scalar, a 2-tuple/list (range), or an iterable of scalars.")
        d = d[mask]

    # If nothing left, return empty frame with the expected schema (avoid KeyError on sort)
    if d.empty:
        # Optional: helpful hint about available k's
        if "k" in df.columns and k_filter is not None:
            avail_k = np.sort(pd.to_numeric(df["k"], errors="coerce").dropna().unique())
            print(f"[WARN] No rows after k_filter={k_filter}. Available k values (first 10): {avail_k[:10]}")
        return pd.DataFrame(columns=["L","rho","N","nc_mean","nc_sem"])

    rows = []
    # Group by rho and aggregate
    for rho, grp in d.groupby("rho", dropna=True):
        # N = total number of seeds; keep your original (max) logic
        N = int(np.nanmax(grp["num_samples"])) if len(grp) else 0
        if N <= 0:
            rows.append({"L": (L_filter if L_filter is not None else np.nan),
                         "rho": rho, "N": 0, "nc_mean": np.nan, "nc_sem": np.nan})
            continue

        # Build M_k: seeds with >= k colors percolated
        M = np.zeros(num_colors + 2, dtype=float)  # indices 1..num_colors; M[C+1]=0
        for _, r in grp.iterrows():
            k = int(r["order"])
            if 1 <= k <= num_colors:
                M[k] = float(r["num_sample_perc"])
        M = np.clip(M, 0, N)

        # Tail-sum mean
        nc_mean = np.sum(M[1:num_colors+1]) / N

        # Exact variance from x_k = M_k - M_{k+1}
        X = np.zeros(num_colors + 1, dtype=float)
        for k in range(1, num_colors + 1):
            X[k] = max(M[k] - M[k + 1], 0.0)
        p = X[1:num_colors + 1] / N
        ks = np.arange(1, num_colors + 1, dtype=float)
        var_nc = np.sum((ks - nc_mean) ** 2 * p)
        nc_sem = np.sqrt(var_nc / N) if N > 0 else np.nan

        rows.append({"L": (L_filter if L_filter is not None else np.nan),
                     "rho": rho, "N": N, "nc_mean": nc_mean, "nc_sem": nc_sem})

    out = pd.DataFrame(rows)
    if not out.empty and "rho" in out.columns:
        out = out.sort_values("rho").reset_index(drop=True)
    return out

def find_drop_interval(res_df, num_colors=None, frac_hi=0.98, frac_lo=0.02,
                       smooth_window=5, pad=0.002):
    """
    Find the rho interval where <n_c> drops from the upper plateau (~num_colors) to ~0.

    Parameters
    ----------
    res_df : DataFrame
        Must contain columns ['rho','nc_mean']; already filtered for one L (and p0, if desired).
    num_colors : int or None
        If None, inferred as the max of nc_mean (useful when the plateau is ~C).
    frac_hi : float
        Upper threshold as a fraction of num_colors (e.g., 0.98 -> 98% of C).
    frac_lo : float
        Lower threshold as a fraction of num_colors (e.g., 0.02 -> 2% of C).
    smooth_window : int
        Rolling median window (odd recommended) to reduce noise before thresholding.
    pad : float
        Margin added to the detected interval on both sides.

    Returns
    -------
    (rho_min, rho_max, df_used) or (None, None, df_used) if nothing found.
    """
    r = res_df[["rho", "nc_mean"]].dropna().sort_values("rho").reset_index(drop=True).copy()
    if r.empty:
        return None, None, r

    # infer plateau (C) if not provided
    C = float(np.nanmax(r["nc_mean"])) if num_colors is None else float(num_colors)

    # (1) smooth with rolling median to mitigate outliers/noise
    if smooth_window and smooth_window > 1:
        r["nc_smooth"] = r["nc_mean"].rolling(smooth_window, center=True, min_periods=1).median()
    else:
        r["nc_smooth"] = r["nc_mean"]

    hi_thr = C * frac_hi   # near the upper plateau
    lo_thr = C * frac_lo   # near zero

    # Core mask: points between hi and lo thresholds (transition band)
    mask = (r["nc_smooth"] < hi_thr) & (r["nc_smooth"] > lo_thr)

    # Fallback: if mask is empty (too sharp/noisy), use large-slope region
    if not mask.any():
        slope = np.abs(np.gradient(r["nc_smooth"].values, r["rho"].values))
        thr = np.nanpercentile(slope, 90)  # top 10% steepest points
        mask = slope >= thr

    sel = r.loc[mask, "rho"]
    if sel.empty:
        return None, None, r

    rho_min = max(r["rho"].min(), sel.min() - pad)
    rho_max = min(r["rho"].max(), sel.max() + pad)
    return float(rho_min), float(rho_max), r


def make_refined_rho_grid(rho_min, rho_max, step=None, n_points=40):
    """
    Build a refined rho grid inside [rho_min, rho_max] to run more samples.

    If 'step' is provided, use np.arange; otherwise use 'n_points' with linspace.

    Returns
    -------
    np.ndarray of rho values (float).
    """
    if rho_min is None or rho_max is None or rho_min >= rho_max:
        return np.array([])
    if step is not None and step > 0:
        # ensure inclusiveness of the right edge
        return np.arange(rho_min, rho_max + step/2, step)
    # default: fixed number of points
    return np.linspace(rho_min, rho_max, int(n_points))


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
):
    """
    Retorna todos os rho (float) existentes em:
      ../Data/{type_perc}_percolation/num_colors_{num_colors}/dim_{dim}/L_{L}/NT_constant/NT_{Nt}/k_*/rho_*/data
    que coincidam com os parâmetros fixos e com k (~=) ao informado.
    """
    base = (Path(base_root)
            / f"{type_perc}_percolation"
            / f"num_colors_{num_colors}"
            / f"dim_{dim}"
            / f"L_{L}"
            / "NT_constant"
            / f"NT_{Nt}")

    if not base.exists():
        return []

    rhos = []
    # Procurar todos caminhos .../k_*/rho_*/data
    for data_dir in base.glob("k_*/rho_*/data"):
        if not data_dir.is_dir():
            continue
        m = PARAMS_RE.search(str(data_dir.as_posix()))
        if not m:
            continue
        gd = m.groupdict()
        # Verificar os fixos
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

    # Deixar únicos e ordenados
    rhos = sorted(set(rhos))
    return rhos