import re, os, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# Aceita c/f_T/rho em float normal ou notação científica (ex.: 5.0e-01, 5.0e-02)
FLOAT = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
PARAMS_RE = re.compile(rf"""
    (?P<type_perc>[A-Za-z]+)_percolation
    /num_colors_(?P<num_colors>\d+)
    /dim_(?P<dim>\d+)
    /L_(?P<L>\d+)
    /fT_constant/fT_(?P<f_T>{FLOAT})
    /c_(?P<c>{FLOAT})
    /rho_(?P<rho>{FLOAT})
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
    Upsert por chave (L, n_colors, f_T, c, rho).
    Salva sem a coluna 'source' (mudança 2).
    """
    key_cols = ["L", "n_colors", "f_T", "c", "rho"]
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
    for c in ["L", "n_colors"]:
        new_df[c] = pd.to_numeric(new_df[c], errors="coerce").astype("Int64")
        old_df[c] = pd.to_numeric(old_df[c], errors="coerce").astype("Int64")
    for c in ["f_T", "c", "rho", "n_c", "n_c_err"]:
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
    merged = merged[["L","n_colors","f_T","c","n_c","n_c_err","rho","Nsamples"]]
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
    f_T: float,
    c: float,
    base_root: str = "../Data",
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-15,
):
    """
    Retorna todos os rho (float) existentes em:
      ../Data/{type_perc}_percolation/num_colors_{num_colors}/dim_{dim}/L_{L}/fT_constant/fT_*/c_*/rho_*/data
    que coincidam com os parâmetros fixos e com c/f_T informados.
    """
    base = (Path(base_root)
            / f"{type_perc}_percolation"
            / f"num_colors_{num_colors}"
            / f"dim_{dim}"
            / f"L_{L}"
            / "fT_constant")

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

def processing_data_nc(L_lst, f_T_lst, c_lst, num_colors, dim, type_perc):
    base_root = "../Data"
    output_dir = "../Data/bond_percolation/"

    rows = []
    root = Path(base_root)

    for L in L_lst:
        for f_T in f_T_lst:
            for c_val in c_lst:
                rho_values = list_rho_values(type_perc, num_colors, dim, L, f_T, c_val, base_root=base_root)
                for rho in rho_values:
                    p = (root / f"{type_perc}_percolation" / f"num_colors_{num_colors}" / f"dim_{dim}" /
                         f"L_{L}" / "fT_constant" / f"fT_{f_T:.6e}" / f"c_{c_val:.6e}" /
                         f"rho_{rho:.4e}" / "data" / "process_names.txt")
                    if not p.exists():
                        continue
                    try:
                        df = read_table_auto(p)
                        n_c, n_c_err, N = compute_nc_from_df(df)
                        rows.append({
                            "L": L,
                            "n_colors": num_colors,
                            "f_T": float(f_T),
                            "c": float(c_val),
                            "rho": float(rho),
                            "n_c": n_c,
                            "n_c_err": n_c_err,
                            "Nsamples": N,
                        })
                    except Exception as e:
                        print(f"[WARN] Falha em {p}: {e}")

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_csv_path = out_dir_path / f"nc_dim_{dim}.csv"

    merged = upsert_summary(out_csv_path, rows)

    print("\nResumo atualizado:")
    if not merged.empty:
        print(merged.sort_values(["L", "f_T", "c", "rho"]).to_string(index=False))
    else:
        print("[vazio] nenhum dado encontrado")
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

def latex_one_decimal(x, pos):
    if abs(x) < 1e-12:
        x = 0
    return rf'${x:.1f}$'

def panel_label_from_index(i):
    letters = ""
    while True:
        letters = chr(ord('a') + i % 26) + letters
        i = i // 26 - 1
        if i < 0:
            break
    return letters

def plot_nc_dynamic_grid(
    df_dynamic,
    df_series,
    L_lst,
    ns_lst,
    c_lst,
    window=300,
    base=5,
    dim=2,
    order=0,
    p0=0.8,
    p_cut=0.95,
    x_min=0,
    tick_fs=16,
    label_fs=18,
    title_fs=18,
    tick_len=8,
    tick_width=1.4,
    fs_legend=15,
    markers=('o', 's', 'D', '^'),
    ms=7,
    mew=1.4,
    print_bounds=True,
    savepath=None,

    # novos argumentos
    wspace=0.02,
    hspace=0.08,
    ns_text_x=0.06,
    ns_text_y=0.90,
    ns_text_fs=18,
    legend_loc='lower right',
    panel_label_x=0.06,
    panel_label_y=0.97,
    panel_label_fs=18,
):
    """
    Plota <nc> versus f_T usando df_dynamic, truncando superiormente por ft_max.

    O ft_max é calculado a partir do df_series, depois do truncamento p_mean <= p_cut.
    O ft_min é calculado a partir do df_dynamic.

    Retorna:
        fig, axes, parms, ft_bounds
    """

    if len(markers) < len(L_lst):
        raise ValueError(
            f"Número de markers insuficiente: len(markers)={len(markers)}, "
            f"mas len(L_lst)={len(L_lst)}"
        )

    parms = {
        'c': [],
        'ft_min': [],
        'ft_max': [],
        'ns': [],
        'dim': [],
        'p0': [],
    }

    ft_bounds = {}

    # ============================================================
    # 1) Calcula ft_min e ft_max para cada par (ns, c)
    # ============================================================

    for ns in ns_lst:
        rho = round(1 / ns, 5)

        for c in c_lst:
            ft_min_per_L = []
            ft_max_per_L = []

            # ----------------------------
            # ft_min vem do df_dynamic
            # ----------------------------
            for L in L_lst:
                df_d = df_dynamic[
                    (np.isclose(df_dynamic['c'], c)) &
                    (df_dynamic['num_colors'] == ns) &
                    (np.isclose(df_dynamic['p0'], p0)) &
                    (df_dynamic['dim'] == dim) &
                    (df_dynamic['L'] == L) &
                    (df_dynamic['stat_window'] == window)
                ]

                ft_valid = df_d.loc[
                    np.isclose(df_d['nc'], ns),
                    'f_T'
                ].dropna()

                if ft_valid.empty:
                    raise ValueError(
                        f"Nenhum ponto encontrado para calcular ft_min: "
                        f"ns={ns}, c={c}, L={L}, p0={p0}, dim={dim}, window={window}"
                    )

                ft_min_per_L.append(ft_valid.min())

            ft_min = max(ft_min_per_L)

            # ----------------------------
            # ft_max vem do df_series truncado
            # ----------------------------
            for L in L_lst:
                df_s = df_series[
                    (np.isclose(df_series['c'], c)) &
                    (df_series['nc'] == ns) &
                    (np.isclose(df_series['p0'], p0)) &
                    (df_series['dim'] == dim) &
                    (df_series['L'] == L) &
                    (df_series['stat_window'] == window) &
                    (df_series['order'] == order) &
                    (np.isclose(df_series['rho'], rho))
                ]

                df_filter = df_s[
                    df_s['f_T'] >= ft_min
                ].dropna(subset=['f_T', 'p_mean'])

                df_trunced = df_filter[
                    df_filter['p_mean'] <= p_cut
                ]

                if df_trunced.empty:
                    raise ValueError(
                        f"Nenhum ponto encontrado para calcular ft_max: "
                        f"ns={ns}, c={c}, L={L}, ft_min={ft_min}, "
                        f"rho={rho}, p_cut={p_cut}"
                    )

                ft_max_per_L.append(df_trunced['f_T'].max())

            ft_max = min(ft_max_per_L)

            ft_bounds[(ns, c)] = {
                'ft_min': ft_min,
                'ft_max': ft_max,
                'rho': rho,
            }

            parms['c'].append(c)
            parms['ft_min'].append(ft_min)
            parms['ft_max'].append(ft_max)
            parms['ns'].append(ns)
            parms['dim'].append(dim)
            parms['p0'].append(p0)

            if print_bounds:
                print(
                    f"ns={ns} | c={c:.2f} | rho={rho:.5f} | "
                    f"ft_min={ft_min:.6f} | ft_max={ft_max:.6f}"
                )

    # ============================================================
    # 2) Plota grade: linhas = ns, colunas = c
    # ============================================================

    nrows = len(ns_lst)
    ncols = len(c_lst)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(base * ncols, base * nrows),
        sharey='row',
        squeeze=False
    )

    for ax in axes.flat:
        ax.set_box_aspect(1)

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=tick_fs,
            length=tick_len,
            width=tick_width,
            direction='in',
            top=True,
            right=True
        )

        ax.tick_params(
            axis='both',
            which='minor',
            length=5,
            width=1.4,
            direction='in',
            top=True,
            right=True
        )

        ax.minorticks_on()

        # força todos os ticks do eixo y a terem uma casa decimal
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(latex_one_decimal))

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.08,
        top=0.93,
        wspace=wspace,
        hspace=hspace
    )

    for row, ns in enumerate(ns_lst):
        for col, c in enumerate(c_lst):
            ax = axes[row, col]

            ft_min = ft_bounds[(ns, c)]['ft_min']
            ft_max = ft_bounds[(ns, c)]['ft_max']

            for idx_L, L in enumerate(L_lst):
                df_d = df_dynamic[
                    (np.isclose(df_dynamic['c'], c)) &
                    (df_dynamic['num_colors'] == ns) &
                    (np.isclose(df_dynamic['p0'], p0)) &
                    (df_dynamic['dim'] == dim) &
                    (df_dynamic['L'] == L) &
                    (df_dynamic['stat_window'] == window)
                ]

                # Truncamento superior obtido a partir do df_series
                df_plot = df_d[
                    df_d['f_T'] <= ft_max
                ].dropna(subset=['f_T', 'nc'])

                df_plot = df_plot.sort_values('f_T')

                ft = df_plot['f_T']
                nc_plot = df_plot['nc']

                ax.plot(
                    ft,
                    nc_plot,
                    mew=mew,
                    marker=markers[idx_L],
                    ms=ms,
                    ls='None',
                    label=f'$L = {L}$',
                    clip_on=False
                )
                panel_idx = row * ncols + col
                panel_label = panel_label_from_index(panel_idx)

                ax.text(
                    panel_label_x,
                    panel_label_y,
                    rf'$({panel_label})$',
                    transform=ax.transAxes,
                    fontsize=panel_label_fs,
                    ha='left',
                    va='top'
                )
            ax.set_xlim(x_min, ft_max)

            if row == 0:
                ax.set_title(rf'$c = {c}$', fontsize=title_fs)

            if col == 0:
                ax.set_ylabel(r'$\langle n_c \rangle$', fontsize=label_fs)

                ax.text(
                    ns_text_x,
                    ns_text_y,
                    rf'$n_s = {ns}$',
                    transform=ax.transAxes,
                    fontsize=ns_text_fs,
                    ha='left',
                    va='top'
                )

                ax.legend(
                    fontsize=fs_legend,
                    loc=legend_loc,
                    frameon=False
                )

            if row == nrows - 1:
                ax.set_xlabel(r'$f_T$', fontsize=label_fs)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')

    return fig, axes, parms, ft_bounds

def latex_two_decimal(x, pos):
    if abs(x) < 1e-12:
        x = 0
    return rf'${x:.2f}$'

def plot_pmean_series_grid(
    df_dynamic,
    df_series,
    L_lst,
    ns_lst,
    c_lst,
    window=300,
    base=5,
    dim=2,
    order=0,
    p0=0.8,
    p_cut=0.95,
    pc=0.5,
    tick_fs=16,
    label_fs=18,
    title_fs=18,
    tick_len=8,
    tick_width=1.4,
    fs_legend=15,
    markers=('o', 's', 'D', '^'),
    ms=7,
    mew=1.4,
    print_bounds=True,
    savepath=None,

    # espaçamento
    wspace=0.02,
    hspace=0.08,

    # texto de ns
    ns_text_x=0.06,
    ns_text_y=0.82,
    ns_text_fs=18,

    # legenda
    legend_loc='lower right',

    # labels (a), (b), ...
    panel_label_x=0.06,
    panel_label_y=0.97,
    panel_label_fs=18,

    # linha horizontal em pc
    draw_pc_line=True,
    pc_lw=2.0,
    pc_color='k',
):
    """
    Plota p_mean versus f_T usando df_series.

    A grade possui:
        linhas  -> ns_lst
        colunas -> c_lst

    O intervalo [ft_min, ft_max] é calculado da mesma forma do código original:

        ft_min:
            vem do df_dynamic, pegando o menor f_T em que nc == ns
            para cada L, e depois tomando o maior desses mínimos.

        ft_max:
            vem do df_series, depois do truncamento p_mean <= p_cut,
            para cada L, e depois tomando o menor desses máximos.

    Retorna:
        fig, axes, parms, ft_bounds
    """

    if len(markers) < len(L_lst):
        raise ValueError(
            f"Número de markers insuficiente: len(markers)={len(markers)}, "
            f"mas len(L_lst)={len(L_lst)}"
        )

    parms = {
        'c': [],
        'ft_min': [],
        'ft_max': [],
        'ns': [],
        'dim': [],
        'p0': [],
    }

    ft_bounds = {}

    # ============================================================
    # 1) Calcula ft_min e ft_max para cada par (ns, c)
    # ============================================================

    for ns in ns_lst:
        rho = round(1 / ns, 5)

        for c in c_lst:
            ft_min_per_L = []
            ft_max_per_L = []

            # ----------------------------
            # ft_min vem do df_dynamic
            # ----------------------------
            for L in L_lst:
                df_d = df_dynamic[
                    (np.isclose(df_dynamic['c'], c)) &
                    (df_dynamic['num_colors'] == ns) &
                    (np.isclose(df_dynamic['p0'], p0)) &
                    (df_dynamic['dim'] == dim) &
                    (df_dynamic['L'] == L) &
                    (df_dynamic['stat_window'] == window)
                ]

                ft_valid = df_d.loc[
                    np.isclose(df_d['nc'], ns),
                    'f_T'
                ].dropna()

                if ft_valid.empty:
                    raise ValueError(
                        f"Nenhum ponto encontrado para calcular ft_min: "
                        f"ns={ns}, c={c}, L={L}, p0={p0}, dim={dim}, window={window}"
                    )

                ft_min_per_L.append(ft_valid.min())

            ft_min = max(ft_min_per_L)

            # ----------------------------
            # ft_max vem do df_series truncado
            # ----------------------------
            for L in L_lst:
                df_s = df_series[
                    (np.isclose(df_series['c'], c)) &
                    (df_series['nc'] == ns) &
                    (np.isclose(df_series['p0'], p0)) &
                    (df_series['dim'] == dim) &
                    (df_series['L'] == L) &
                    (df_series['stat_window'] == window) &
                    (df_series['order'] == order) &
                    (np.isclose(df_series['rho'], rho))
                ]

                df_filter = df_s[
                    df_s['f_T'] >= ft_min
                ].dropna(subset=['f_T', 'p_mean'])

                df_trunced = df_filter[
                    df_filter['p_mean'] <= p_cut
                ]

                if df_trunced.empty:
                    raise ValueError(
                        f"Nenhum ponto encontrado para calcular ft_max: "
                        f"ns={ns}, c={c}, L={L}, ft_min={ft_min}, "
                        f"rho={rho}, p_cut={p_cut}"
                    )

                ft_max_per_L.append(df_trunced['f_T'].max())

            ft_max = min(ft_max_per_L)

            ft_bounds[(ns, c)] = {
                'ft_min': ft_min,
                'ft_max': ft_max,
                'rho': rho,
            }

            parms['c'].append(c)
            parms['ft_min'].append(ft_min)
            parms['ft_max'].append(ft_max)
            parms['ns'].append(ns)
            parms['dim'].append(dim)
            parms['p0'].append(p0)

            if print_bounds:
                print(
                    f"ns={ns} | c={c:.2f} | rho={rho:.5f} | "
                    f"ft_min={ft_min:.6f} | ft_max={ft_max:.6f}"
                )

    # ============================================================
    # 2) Plota grade: linhas = ns, colunas = c
    # ============================================================

    nrows = len(ns_lst)
    ncols = len(c_lst)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(base * ncols, base * nrows),
        sharey='row',
        squeeze=False
    )

    for ax in axes.flat:
        ax.set_box_aspect(1)

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=tick_fs,
            length=tick_len,
            width=tick_width,
            direction='in',
            top=True,
            right=True
        )

        ax.tick_params(
            axis='both',
            which='minor',
            length=5,
            width=1.4,
            direction='in',
            top=True,
            right=True
        )

        ax.minorticks_on()

        # Mantém formato com uma casa decimal e fonte LaTeX/mathtext
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(latex_two_decimal)
        )

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.08,
        top=0.93,
        wspace=wspace,
        hspace=hspace
    )

    for row, ns in enumerate(ns_lst):
        rho = round(1 / ns, 5)

        for col, c in enumerate(c_lst):
            ax = axes[row, col]

            # ----------------------------
            # Label (a), (b), (c), ...
            # ----------------------------
            panel_idx = row * ncols + col
            panel_label = panel_label_from_index(panel_idx)

            ax.text(
                panel_label_x,
                panel_label_y,
                rf'$({panel_label})$',
                transform=ax.transAxes,
                fontsize=panel_label_fs,
                ha='left',
                va='top'
            )

            ft_min = ft_bounds[(ns, c)]['ft_min']
            ft_max = ft_bounds[(ns, c)]['ft_max']

            for idx_L, L in enumerate(L_lst):
                df_s = df_series[
                    (np.isclose(df_series['c'], c)) &
                    (df_series['nc'] == ns) &
                    (np.isclose(df_series['p0'], p0)) &
                    (df_series['dim'] == dim) &
                    (df_series['L'] == L) &
                    (df_series['stat_window'] == window) &
                    (df_series['order'] == order) &
                    (np.isclose(df_series['rho'], rho))
                ]

                df_filter = df_s[
                    (df_s['f_T'] >= ft_min) &
                    (df_s['f_T'] <= ft_max)
                ].dropna(subset=['f_T', 'p_mean'])

                df_trunced = df_filter[
                    df_filter['p_mean'] <= p_cut
                ]

                df_trunced = df_trunced.sort_values('f_T')

                ft = df_trunced['f_T']
                pmean = df_trunced['p_mean']

                ax.plot(
                    ft,
                    pmean,
                    marker=markers[idx_L],
                    mew=mew,
                    ms=ms,
                    ls='None',
                    label=f'$L = {L}$',
                    clip_on=False
                )

            if draw_pc_line:
                ax.axhline(
                    y=pc,
                    color=pc_color,
                    lw=pc_lw
                )

            ax.set_xlim(ft_min, ft_max)

            if row == 0:
                ax.set_title(rf'$c = {c}$', fontsize=title_fs)

            if col == 0:
                ax.set_ylabel(r'$p^*$', fontsize=label_fs)

                ax.text(
                    ns_text_x,
                    ns_text_y,
                    rf'$n_s = {ns}$',
                    transform=ax.transAxes,
                    fontsize=ns_text_fs,
                    ha='left',
                    va='top'
                )

                ax.legend(
                    fontsize=fs_legend,
                    loc=legend_loc,
                    frameon=False
                )

            if row == nrows - 1:
                ax.set_xlabel(r'$f_T$', fontsize=label_fs)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')

    return fig, axes, parms, ft_bounds