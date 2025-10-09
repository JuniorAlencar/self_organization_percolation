import re, os, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import math
from collections.abc import Iterable

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

# ---------- utilidades ----------
def read_table_auto(path: Path) -> pd.DataFrame:
    """Lê tabela auto-separada, tratando 'Null' como NaN."""
    try:
        return pd.read_csv(path, sep=None, engine="python", na_values=["Null"])
    except Exception:
        return pd.read_csv(path, sep=r"\t+", engine="python", na_values=["Null"])

def compute_nc_from_df(df: pd.DataFrame):
    """
    n_c: média da qtde de pt_mean não-NaN por filename,
    n_c_err: erro padrão da média (SEM),
    Nsamples: nº de filenames.
    """
    req = {"filename", "order", "pt_mean"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"Colunas ausentes: {missing}")

    df = df.copy()
    df["pt_mean"] = pd.to_numeric(df["pt_mean"], errors="coerce")

    counts = (
        df.groupby("filename")["pt_mean"]
          .apply(lambda s: int(s.notna().sum()))
          .astype(int)
    )
    vals = counts.to_numpy()
    N = len(vals)
    if N == 0:
        return float("nan"), float("nan"), 0
    n_c = float(np.mean(vals))
    n_c_err = 0.0 if N == 1 else float(np.std(vals, ddof=1) / np.sqrt(N))
    return n_c, n_c_err, N

def upsert_summary(summary_path: Path, rows: list[dict]) -> pd.DataFrame:
    """
    Upsert por chave (L, n_colors, NT, k, rho).
    Salva sem a coluna 'source' (mudança 2).
    """
    key_cols = ["L", "n_colors", "NT", "k", "rho"]
    cols_all = key_cols + ["n_c", "n_c_err", "Nsamples"]  # <- sem 'source'
    new_df = pd.DataFrame(rows, columns=cols_all)

    if summary_path.exists():
        old_df = pd.read_csv(summary_path)
        # Garante colunas esperadas (se faltar, cria vazias)
        for c in cols_all:
            if c not in old_df.columns:
                old_df[c] = np.nan
        # Remove 'source' se por acaso existir de versões antigas:
        if "source" in old_df.columns:
            old_df = old_df.drop(columns=["source"])
        # Reordena
        old_df = old_df[cols_all]
    else:
        old_df = pd.DataFrame(columns=cols_all)

    # Tipos consistentes
    for c in ["L", "n_colors", "NT"]:
        new_df[c] = pd.to_numeric(new_df[c], errors="coerce").astype("Int64")
        old_df[c] = pd.to_numeric(old_df[c], errors="coerce").astype("Int64")
    for c in ["k", "rho", "n_c", "n_c_err"]:
        new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
        old_df[c] = pd.to_numeric(old_df[c], errors="coerce")
    old_df["Nsamples"] = pd.to_numeric(old_df["Nsamples"], errors="coerce").astype("Int64")
    new_df["Nsamples"] = pd.to_numeric(new_df["Nsamples"], errors="coerce").astype("Int64")

    # Upsert por índice composto
    old_idx = old_df.set_index(key_cols)
    new_idx = new_df.set_index(key_cols)
    old_idx.update(new_idx)
    merged = pd.concat([old_idx[~old_idx.index.isin(new_idx.index)], new_idx]).reset_index()

    # Ordena e salva
    merged = merged[["L","n_colors","NT","n_c","n_c_err","rho","k","Nsamples"]]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(summary_path, index=False)
    return merged

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

def processing_data_nc(L_lst, Nt_lst, k_lst, num_colors, dim, type_perc):
    base_root  = "../Data"
    output_dir = "../Data/bond_percolation/"   # <- mudança 1
    out_csv    = None  # definido abaixo com base no dim

    # ---------- processamento principal ----------
    rows = []
    root = Path(base_root)

    for L in L_lst:
        for NT in Nt_lst:
            for k in k_lst:
                # Lista rhos existentes para este (L, NT, k)
                rho_values = list_rho_values(type_perc, num_colors, dim, L, NT, k, base_root=base_root)
                for rho in rho_values:
                    p = root / f"{type_perc}_percolation" / f"num_colors_{num_colors}" / f"dim_{dim}" / \
                        f"L_{L}" / "NT_constant" / f"NT_{NT}" / f"k_{k:.1e}" / f"rho_{rho:.4e}" / "data" / "process_names.txt"
                    if not p.exists():
                        continue
                    try:
                        df = read_table_auto(p)
                        n_c, n_c_err, N = compute_nc_from_df(df)
                        rows.append({
                            "L": L,
                            "n_colors": num_colors,
                            "NT": NT,
                            "k": k,
                            "rho": float(rho),
                            "n_c": n_c,
                            "n_c_err": n_c_err,
                            "Nsamples": N
                        })
                    except Exception as e:
                        print(f"[WARN] Falha em {p}: {e}")

    # Define caminho final conforme solicitado
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_csv_path = out_dir_path / f"nc_dim_{dim}.csv"   # <- mudança 1

    merged = upsert_summary(out_csv_path, rows)

    # Log curto
    print("\nResumo atualizado:")
    print(merged.sort_values(["L","NT","k","rho"]).to_string(index=False))
    print(f"\nArquivo salvo/atualizado: {out_csv_path}")

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