r"""
BetaScalingAnalysis.py
======================
Módulo para o cálculo do expoente crítico beta (e dos expoentes dos segundos spanning clusters)
a partir de simulações estabilizadas de Percolação Auto-Organizada (SOP).

Baseado no procedimento de escala sem mistura com nu:
    P_\infty(p) \propto (p - p_c)^\beta,   p > p_c
onde:
    p = p^*(L, f_T) (fração de ocupação estacionária)
    \Delta p(L, f_T) = p^*(L, f_T) - p_c
    P_L(\Delta p) = fração de ocupação dos sítios após estabilização (do maior cluster).

Inclui:
  - Carregamento e consolidação de dados de SOP_data/published_fractions
  - Função rigorosa para checagem e identificação da região de linearidade
  - Cálculo de beta(L) para cada tamanho de rede L
  - Cálculo dos expoentes críticos do segundo spanning cluster (P_2)
  - Plots de alta qualidade para publicações científicas
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import linregress

__all__ = [
    "load_beta_dataset",
    "get_dataset_summary",
    "check_linearity_and_find_scaling_region",
    "fit_power_law_beta",
    "scan_optimal_pc",
    "compute_beta_for_all_sizes",
    "analyze_second_spanning_cluster",
    "plot_family_curves",
    "plot_linearity_diagnostics",
    "plot_beta_vs_L",
    "plot_second_spanning_cluster_analysis",
]


def load_beta_dataset(
    parquet_path: str | Path = "../SOP_data/published_fractions_consolidated.parquet",
    fractions_dir: str | Path = "../SOP_data/published_fractions",
) -> pd.DataFrame:
    """
    Carrega o DataFrame consolidado de frações com cache em formato Parquet para alta performance.
    """
    p_pq = Path(parquet_path)
    if not p_pq.exists():
        # Tenta no diretório atual ou raiz do repositório
        alt_pq = Path("SOP_data/published_fractions_consolidated.parquet")
        if alt_pq.exists():
            p_pq = alt_pq

    if p_pq.exists():
        return pd.read_parquet(p_pq)

    # Se não existir o parquet, carrega via FractionsClusterAnalysis e salva cache
    from src.FractionsClusterAnalysis import load_published_fractions_df

    p_frac = Path(fractions_dir)
    if not p_frac.exists():
        alt_frac = Path("SOP_data/published_fractions")
        if alt_frac.exists():
            p_frac = alt_frac

    df = load_published_fractions_df(p_frac)
    try:
        p_pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p_pq, index=False)
    except Exception as e:
        print(f"Aviso: Não foi possível salvar cache Parquet: {e}")

    return df


def get_dataset_summary(df: pd.DataFrame, perc_type: str = "node") -> pd.DataFrame:
    """
    Retorna uma tabela resumo contendo os tamanhos L disponíveis,
    número de valores de f_T, contagem total de amostras e intervalos de p* e P_1.
    """
    sub = df[df["type"] == perc_type]
    col_p = "p_stab_node" if perc_type == "node" else "p_stab_bond"

    records = []
    for L in sorted(sub["L"].unique()):
        sub_L = sub[sub["L"] == L]
        fts = sorted(sub_L["f_T"].unique())
        n_samples = len(sub_L)
        p_min = sub_L[col_p].min()
        p_max = sub_L[col_p].max()
        p1_min = sub_L["c1_frac_site"].min()
        p1_max = sub_L["c1_frac_site"].max()
        k2_count = (sub_L["k"] >= 2).sum()
        k2_pct = (k2_count / n_samples) * 100.0 if n_samples > 0 else 0.0

        records.append({
            "L": int(L),
            "N_fT": len(fts),
            "N_amostras": n_samples,
            "p*_min": p_min,
            "p*_max": p_max,
            "P1_min": p1_min,
            "P1_max": p1_max,
            "Amostras_k>=2": k2_count,
            "% k>=2": k2_pct,
        })

    return pd.DataFrame(records)


def check_linearity_and_find_scaling_region(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    y_err: Optional[np.ndarray | list[float]] = None,
    min_points: int = 3,
    max_points: Optional[int] = None,
    criterion: str = "adj_r2",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    r"""
    Verifica a linearidade em todas as sub-regiões contíguas e identifica a melhor região de escala.

    Métricas calculadas para cada janela:
      - beta (inclinação no espaço log-log: d ln y / d ln x)
      - beta_stderr (erro padrão da inclinação)
      - ln_A (intercepto)
      - r2 (coeficiente de determinação R²)
      - adj_r2 (R² ajustado ao número de graus de liberdade)
      - rmse (raiz do erro quadrático médio dos resíduos logarítmicos)
      - curvature (coeficiente quadrático do polinômio de ordem 2; curvatura residual)
      - span_ratio (razão x_max / x_min da janela no espaço linear)

    Parâmetros:
      x: array com valores de \Delta p (deve ser estritamente positivo)
      y: array com valores de P_L (deve ser estritamente positivo)
      y_err: incertezas opcionais de y
      min_points: número mínimo de pontos contíguos na janela (padrão: 3)
      max_points: número máximo opcional de pontos contíguos
      criterion: critério de ordenação ('adj_r2', 'r2', 'rmse' ou 'curvature')

    Retorna:
      (windows_df, best_region_dict)
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # Filtra valores válidos estritamente positivos
    valid = (x_arr > 0) & (y_arr > 0) & np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]

    n = len(x_arr)
    if n < min_points:
        raise ValueError(
            f"Número de pontos válidos ({n}) menor que min_points ({min_points})."
        )

    log_x = np.log(x_arr)
    log_y = np.log(y_arr)

    max_p = n if max_points is None else min(n, max_points)

    records: List[Dict[str, Any]] = []

    for i in range(n):
        for j in range(i + min_points - 1, min(n, i + max_p)):
            sub_x = log_x[i : j + 1]
            sub_y = log_y[i : j + 1]
            k_pts = len(sub_x)

            res = linregress(sub_x, sub_y)
            r2 = float(res.rvalue**2)
            adj_r2 = float(1.0 - (1.0 - r2) * (k_pts - 1) / (k_pts - 2)) if k_pts > 2 else r2

            y_pred = res.intercept + res.slope * sub_x
            resids = sub_y - y_pred
            rmse = float(np.sqrt(np.mean(resids**2)))

            # Teste de curvatura (ajuste parabólico nos resíduos)
            if k_pts >= 4:
                p_poly = np.polyfit(sub_x, sub_y, 2)
                curvature = float(abs(p_poly[0]))
            else:
                curvature = 0.0

            records.append({
                "start_idx": i,
                "end_idx": j,
                "n_pts": k_pts,
                "x_min": float(x_arr[i]),
                "x_max": float(x_arr[j]),
                "span_ratio": float(x_arr[j] / x_arr[i]) if x_arr[i] > 0 else np.nan,
                "beta": float(res.slope),
                "beta_err": float(res.stderr) if res.stderr is not None else 0.0,
                "ln_A": float(res.intercept),
                "A": float(np.exp(res.intercept)),
                "r2": r2,
                "adj_r2": adj_r2,
                "rmse": rmse,
                "curvature": curvature,
                "p_value": float(res.pvalue),
            })

    df_windows = pd.DataFrame(records)

    # Ordenação conforme o critério
    if criterion == "adj_r2":
        df_windows.sort_values(
            by=["adj_r2", "n_pts", "r2"], ascending=[False, False, False], inplace=True
        )
    elif criterion == "r2":
        df_windows.sort_values(
            by=["r2", "n_pts"], ascending=[False, False], inplace=True
        )
    elif criterion == "rmse":
        df_windows.sort_values(
            by=["rmse", "n_pts"], ascending=[True, False], inplace=True
        )
    elif criterion == "curvature":
        df_windows.sort_values(
            by=["curvature", "adj_r2"], ascending=[True, False], inplace=True
        )
    else:
        df_windows.sort_values(
            by=["adj_r2", "r2"], ascending=[False, False], inplace=True
        )

    df_windows.reset_index(drop=True, inplace=True)
    best = df_windows.iloc[0].to_dict()

    return df_windows, best


def fit_power_law_beta(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    window: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    r"""
    Ajusta a lei de potência P = A * (\Delta p)^\beta via regressão linear em log-log.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if window is not None:
        i, j = window
        x_sub = x_arr[i : j + 1]
        y_sub = y_arr[i : j + 1]
    else:
        x_sub = x_arr
        y_sub = y_arr

    log_x = np.log(x_sub)
    log_y = np.log(y_sub)

    res = linregress(log_x, log_y)
    r2 = float(res.rvalue**2)
    n_pts = len(x_sub)
    adj_r2 = float(1.0 - (1.0 - r2) * (n_pts - 1) / (n_pts - 2)) if n_pts > 2 else r2

    y_fit = np.exp(res.intercept) * (x_sub**res.slope)
    resids = log_y - (res.intercept + res.slope * log_x)
    rmse = float(np.sqrt(np.mean(resids**2)))

    return {
        "beta": float(res.slope),
        "beta_err": float(res.stderr) if res.stderr is not None else 0.0,
        "A": float(np.exp(res.intercept)),
        "ln_A": float(res.intercept),
        "intercept_err": float(res.intercept_stderr) if hasattr(res, "intercept_stderr") else 0.0,
        "r2": r2,
        "adj_r2": adj_r2,
        "rmse": rmse,
        "n_pts": n_pts,
        "x_sub": x_sub,
        "y_sub": y_sub,
        "y_fit": y_fit,
        "residuals": resids,
    }


def scan_optimal_pc(
    p_values: np.ndarray,
    P_values: np.ndarray,
    bounds: Optional[Tuple[float, float]] = None,
    min_points: int = 3,
) -> Tuple[float, float, float]:
    """
    Realiza uma varredura para encontrar o ponto crítico pc que maximiza a linearidade R²
    de ln P vs ln(p* - pc).

    Retorna: (best_pc, best_beta, best_r2)
    """
    p_arr = np.asarray(p_values, dtype=float)
    P_arr = np.asarray(P_values, dtype=float)
    p_min = float(np.min(p_arr))

    if bounds is None:
        bounds = (0.0, p_min - 1e-4)

    def objective(pc_cand: float) -> float:
        dp = p_arr - pc_cand
        if np.any(dp <= 1e-9):
            return 1e6
        res = linregress(np.log(dp), np.log(P_arr))
        return -float(res.rvalue**2)

    res_opt = minimize_scalar(objective, bounds=bounds, method="bounded")
    best_pc = float(res_opt.x)
    dp_opt = p_arr - best_pc
    fit_opt = linregress(np.log(dp_opt), np.log(P_arr))

    return best_pc, float(fit_opt.slope), float(fit_opt.rvalue**2)


def compute_beta_for_all_sizes(
    df: pd.DataFrame,
    perc_type: str = "node",
    observable: str = "c1_frac_site",
    pc_mode: str = "optimal_scan",
    pc_fixed_val: Optional[float] = None,
    min_points: int = 3,
) -> pd.DataFrame:
    r"""
    Calcula o expoente crítico beta para cada tamanho L disponível na base de dados.

    Observáveis suportados:
      - 'c1_frac_site': fração de sítios do maior cluster S_1 / L^2 (padrão)
      - 'c1_plus_c2': fração total de sítios percolantes (S_1 + S_2) / L^2
      - 'c1_frac_bond': fração de ligações do maior cluster E_1 / E_total

    Modos de p_c suportados:
      - 'optimal_scan': varre pc para maximizar R² de ln P vs ln(p* - pc) em cada L
      - 'fixed': usa pc fixo fornecido em pc_fixed_val
      - 'relative_min': usa pc(L) = p*(L, f_{T,min}) - delta (offset para viabilizar ln)
      - 'linear_extrap': usa pc(L) como a raiz extrapolada P(p*) = 0 da reta

    Retorna DataFrame com as estatísticas de cada tamanho.
    """
    sub = df[df["type"] == perc_type].copy()
    col_p = "p_stab_node" if perc_type == "node" else "p_stab_bond"
    L_list = sorted(sub["L"].unique())

    if observable == "c1_plus_c2":
        sub["_target_obs"] = sub["c1_frac_site"] + sub["c2_frac_site"].fillna(0.0)
    elif observable in sub.columns:
        sub["_target_obs"] = sub[observable]
    else:
        sub["_target_obs"] = sub["c1_frac_site"]

    results = []

    for L in L_list:
        sub_L = sub[sub["L"] == L]
        grp = (
            sub_L.groupby("f_T")
            .agg(
                p_mean=(col_p, "mean"),
                obs_mean=("_target_obs", "mean"),
                obs_sem=("_target_obs", lambda s: s.std() / np.sqrt(len(s)) if len(s) > 1 else 0.0),
                n_samples=("_target_obs", "count"),
            )
            .reset_index()
            .sort_values("f_T")
        )

        p = grp["p_mean"].values
        P1 = grp["obs_mean"].values

        # Determina pc
        if pc_mode == "optimal_scan":
            pc, _, _ = scan_optimal_pc(p, P1, min_points=min_points)
        elif pc_mode == "fixed":
            if pc_fixed_val is None:
                pc = 0.5000 if perc_type == "bond" else 0.592746
            else:
                pc = pc_fixed_val
        elif pc_mode == "relative_min":
            # delta = 1% do span
            delta = 0.01 * (p.max() - p.min())
            pc = p.min() - delta
        elif pc_mode == "linear_extrap":
            p_fit = np.polyfit(p, P1, 1)
            pc = float(-p_fit[1] / p_fit[0])
            if pc >= p.min():
                pc = p.min() - 0.01
        else:
            pc, _, _ = scan_optimal_pc(p, P1, min_points=min_points)

        dp = p - pc
        if np.any(dp <= 0):
            # Fallback seguro para viabilizar log
            pc = p.min() - 0.005
            dp = p - pc

        # Checa linearidade e seleciona a melhor região
        df_win, best = check_linearity_and_find_scaling_region(
            dp, P1, min_points=min_points
        )

        # Ajuste no intervalo completo para comparação
        fit_full = fit_power_law_beta(dp, P1)

        results.append({
            "L": int(L),
            "N_pontos": len(p),
            "p_c": pc,
            "p_min": float(p.min()),
            "p_max": float(p.max()),
            "beta_best": best["beta"],
            "beta_err_best": best["beta_err"],
            "r2_best": best["r2"],
            "adj_r2_best": best["adj_r2"],
            "rmse_best": best["rmse"],
            "start_idx": int(best["start_idx"]),
            "end_idx": int(best["end_idx"]),
            "n_pts_best": int(best["n_pts"]),
            "dp_min_best": best["x_min"],
            "dp_max_best": best["x_max"],
            "beta_full": fit_full["beta"],
            "beta_err_full": fit_full["beta_err"],
            "r2_full": fit_full["r2"],
        })

    return pd.DataFrame(results)


def analyze_second_spanning_cluster(
    df: pd.DataFrame, perc_type: str = "bond"
) -> Dict[str, Any]:
    """
    Analisa os expoentes críticos e a escala de tamanho finito do segundo spanning cluster (P_2).

    Retorna dicionário contendo:
      - 'fss_summary': tabela de escala com L (P_2 médio condicional, P_1, razão P_2/P_1, contagem)
      - 'fss_fit_P2': ajuste de lei de potência P_2(L) ~ L^{-beta_2 / nu}
      - 'fss_fit_P1': ajuste de lei de potência P_1(L) ~ L^{-beta_1 / nu}
      - 'dp_scaling': tabela de P_2(Delta p) para cada tamanho de rede
      - 'multiplicity_scaling': taxas de ocorrência P(k >= 2) em função de p* e L
    """
    sub = df[df["type"] == perc_type]
    col_p = "p_stab_node" if perc_type == "node" else "p_stab_bond"

    # 1. Escala com o tamanho de rede L (Finite-Size Scaling no regime k >= 2)
    sub_k2 = sub[sub["k"] >= 2]
    L_list = sorted(sub["L"].unique())

    fss_records = []
    for L in L_list:
        sub_L_all = sub[sub["L"] == L]
        sub_L_k2 = sub_k2[sub_k2["L"] == L]

        n_tot = len(sub_L_all)
        n_k2 = len(sub_L_k2)
        freq_k2 = (n_k2 / n_tot) * 100.0 if n_tot > 0 else 0.0

        if n_k2 > 0:
            c2_s = sub_L_k2["c2_frac_site"].dropna()
            c1_s = sub_L_k2["c1_frac_site"].dropna()
            p2_mean = float(c2_s.mean())
            p2_sem = float(c2_s.std() / np.sqrt(len(c2_s))) if len(c2_s) > 1 else 0.0
            p1_mean = float(c1_s.mean())
            p1_sem = float(c1_s.std() / np.sqrt(len(c1_s))) if len(c1_s) > 1 else 0.0
            ratio_mean = float((c2_s / c1_s).mean())
            ratio_sem = float((c2_s / c1_s).std() / np.sqrt(len(c2_s))) if len(c2_s) > 1 else 0.0
            p2_uncond = float(sub_L_all["c2_frac_site"].fillna(0.0).mean())
        else:
            p2_mean = np.nan
            p2_sem = np.nan
            p1_mean = np.nan
            p1_sem = np.nan
            ratio_mean = np.nan
            ratio_sem = np.nan
            p2_uncond = 0.0

        fss_records.append({
            "L": int(L),
            "N_amostras": n_tot,
            "N_k2": n_k2,
            "Freq_k2_%": freq_k2,
            "P2_cond_mean": p2_mean,
            "P2_cond_sem": p2_sem,
            "P1_k2_mean": p1_mean,
            "P1_k2_sem": p1_sem,
            "Ratio_P2_P1": ratio_mean,
            "Ratio_P2_P1_sem": ratio_sem,
            "P2_uncond_mean": p2_uncond,
        })

    fss_df = pd.DataFrame(fss_records)

    # Ajuste da lei de potência P_2 ~ L^{-d_ex} (onde d_ex = beta_2 / nu)
    valid_fss = fss_df[fss_df["P2_cond_mean"].notna() & (fss_df["N_k2"] >= 50)]
    if len(valid_fss) >= 3:
        fit_p2 = linregress(np.log(valid_fss["L"]), np.log(valid_fss["P2_cond_mean"]))
        fit_p1 = linregress(np.log(valid_fss["L"]), np.log(valid_fss["P1_k2_mean"]))
        fss_p2_res = {
            "exponent": float(-fit_p2.slope),
            "exponent_err": float(fit_p2.stderr),
            "fractal_dim": float(2.0 + fit_p2.slope),
            "A": float(np.exp(fit_p2.intercept)),
            "r2": float(fit_p2.rvalue**2),
            "L_vals": valid_fss["L"].values,
            "P2_vals": valid_fss["P2_cond_mean"].values,
        }
        fss_p1_res = {
            "exponent": float(-fit_p1.slope),
            "exponent_err": float(fit_p1.stderr),
            "fractal_dim": float(2.0 + fit_p1.slope),
            "A": float(np.exp(fit_p1.intercept)),
            "r2": float(fit_p1.rvalue**2),
            "L_vals": valid_fss["L"].values,
            "P1_vals": valid_fss["P1_k2_mean"].values,
        }
    else:
        fss_p2_res = {}
        fss_p1_res = {}

    # 2. Variação com f_T e \Delta p para cada tamanho de rede
    dp_records = []
    for L in [512, 1024, 2048, 4096, 8192, 16384]:
        sub_L = sub[sub["L"] == L]
        if sub_L.empty:
            continue
        grp = (
            sub_L.groupby("f_T")
            .agg(
                p_mean=(col_p, "mean"),
                p2_cond=("c2_frac_site", lambda s: s.dropna().mean() if len(s.dropna()) > 0 else np.nan),
                p2_all=("c2_frac_site", lambda s: s.fillna(0.0).mean()),
                freq_k2=("k", lambda s: (s >= 2).mean()),
                n_pts=("k", "count"),
            )
            .reset_index()
            .sort_values("f_T")
        )

        p = grp["p_mean"].values
        p2_all = grp["p2_all"].values
        if len(p2_all) >= 3 and not np.all(np.isnan(p2_all)):
            # Ajuste de crescimento incondicional de P_2 com delta p
            pc_scan, beta_all, r2_all = scan_optimal_pc(p, p2_all)
            dp_records.append({
                "L": int(L),
                "pc_opt": pc_scan,
                "beta2_uncond": beta_all,
                "r2_uncond": r2_all,
            })

    return {
        "fss_summary": fss_df,
        "fss_fit_P2": fss_p2_res,
        "fss_fit_P1": fss_p1_res,
        "dp_scaling": pd.DataFrame(dp_records),
    }


# =====================================================================
# FUNÇÕES DE PLOTAGEM DE ALTA QUALIDADE
# =====================================================================

def plot_family_curves(
    df: pd.DataFrame,
    perc_type: str = "node",
    observable: str = "c1_frac_site",
    L_subset: Optional[List[int]] = None,
    figsize: Tuple[float, float] = (15.0, 6.2),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    r"""
    Plota a família de curvas P_L(\Delta p) conforme proposto pelo orientador:
      - Painel Esquerdo (a): Escala linear P_L vs \Delta p
      - Painel Direito (b): Escala log-log ln P_L vs ln \Delta p com retas de ajuste
    """
    if L_subset is None:
        L_subset = [512, 1024, 2048, 4096, 8192, 16384]

    col_p = "p_stab_node" if perc_type == "node" else "p_stab_bond"
    type_name = "Site (Node)" if perc_type == "node" else "Bond"
    obs_label = r"P_{\mathrm{tot}} = (S_1 + S_2)/L^2" if observable == "c1_plus_c2" else r"P_L(\Delta p) = S_1/L^2"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(L_subset)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for idx, L in enumerate(L_subset):
        sub_L = df[(df["type"] == perc_type) & (df["L"] == L)].copy()
        if sub_L.empty:
            continue

        if observable == "c1_plus_c2":
            sub_L["_target_obs"] = sub_L["c1_frac_site"] + sub_L["c2_frac_site"].fillna(0.0)
        elif observable in sub_L.columns:
            sub_L["_target_obs"] = sub_L[observable]
        else:
            sub_L["_target_obs"] = sub_L["c1_frac_site"]

        grp = (
            sub_L.groupby("f_T")
            .agg(
                p_mean=(col_p, "mean"),
                obs_mean=("_target_obs", "mean"),
                obs_sem=("_target_obs", lambda s: s.std() / np.sqrt(len(s)) if len(s) > 1 else 0.0),
            )
            .reset_index()
            .sort_values("f_T")
        )

        p = grp["p_mean"].values
        P = grp["obs_mean"].values
        P_err = grp["obs_sem"].values

        # Obtém melhor pc para escala logarítmica
        pc, beta_opt, r2_opt = scan_optimal_pc(p, P)
        dp = p - pc

        # Checa linearidade para encontrar a região ideal
        _, best_win = check_linearity_and_find_scaling_region(dp, P)
        si = int(best_win["start_idx"])
        ei = int(best_win["end_idx"])

        c = colors[idx]
        m = markers[idx % len(markers)]
        lbl = rf"$L = {L}\ (\beta = {best_win['beta']:.2f})$"

        # Painel Linear
        ax1.errorbar(
            dp,
            P,
            yerr=P_err,
            fmt=m,
            color=c,
            ms=8,
            mfc="white",
            mew=1.8,
            capsize=3,
            label=lbl,
        )
        ax1.plot(dp, P, "-", color=c, alpha=0.5, linewidth=1.2)

        # Painel Log-Log
        ax2.plot(
            dp,
            P,
            m,
            color=c,
            ms=8,
            mfc="white",
            mew=1.8,
            label=lbl,
        )
        # Reta de ajuste na região linear identificada
        dp_fit_line = np.geomspace(dp[si], dp[ei], 50)
        p_fit_line = best_win["A"] * (dp_fit_line ** best_win["beta"])
        ax2.plot(dp_fit_line, p_fit_line, "--", color=c, linewidth=2.0)

    # Configurações do Painel (a)
    ax1.set_xlabel(r"$\Delta p(L, f_{\mathrm{T}}) = p^* - p_c$", fontsize=20)
    ax1.set_ylabel(rf"${obs_label}$", fontsize=20)
    ax1.set_title(r"\textbf{(a)} Família de Curvas Lineares", loc="left", fontsize=18)
    ax1.set_title(rf"{type_name} Percolation", loc="right", fontsize=16)
    ax1.legend(loc="lower right", fontsize=13, frameon=False)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Configurações do Painel (b)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\Delta p = p^* - p_c$", fontsize=20)
    ax2.set_ylabel(rf"${obs_label}$", fontsize=20)
    ax2.set_title(r"\textbf{(b)} Regimes de Escala Log-Log", loc="left", fontsize=18)
    ax2.set_title(rf"$P_L \propto (\Delta p)^\beta$", loc="right", fontsize=16)
    ax2.legend(loc="lower right", fontsize=13, frameon=False)
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)

    return fig, (ax1, ax2)


def plot_linearity_diagnostics(
    dp: np.ndarray,
    P: np.ndarray,
    L: int,
    windows_df: pd.DataFrame,
    best_fit: Dict[str, Any],
    perc_type: str = "node",
    figsize: Tuple[float, float] = (15.0, 5.0),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plota diagnóstico de linearidade para um dado L:
      - Painel (a): Pontos em log-log com a região linear selecionada destacada
      - Painel (b): Resíduos do ajuste logarítmico (ln P_obs - ln P_pred)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    si = int(best_fit["start_idx"])
    ei = int(best_fit["end_idx"])

    # Painel 1: Ajuste na região selecionada
    ax1.plot(dp, P, "o", color="#1f77b4", ms=9, mfc="white", mew=2.0, label="Todos os pontos")
    ax1.plot(
        dp[si : ei + 1],
        P[si : ei + 1],
        "o",
        color="#d62728",
        ms=10,
        label=rf"Região linear [{si}:{ei}] ($N={ei-si+1}$)",
    )

    dp_dense = np.geomspace(dp[si], dp[ei], 100)
    p_pred_dense = best_fit["A"] * (dp_dense ** best_fit["beta"])
    ax1.plot(
        dp_dense,
        p_pred_dense,
        "-",
        color="#d62728",
        linewidth=2.2,
        label=rf"$\beta = {best_fit['beta']:.4f} \pm {best_fit['beta_err']:.4f}$ ($R^2 = {best_fit['r2']:.4f}$)",
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\Delta p$", fontsize=18)
    ax1.set_ylabel(r"$P_L(\Delta p)$", fontsize=18)
    ax1.set_title(rf"\textbf{{(a)}} Diagnóstico de Escala ($L = {L}$)", loc="left", fontsize=16)
    ax1.legend(loc="lower right", fontsize=13, frameon=False)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)

    # Painel 2: Resíduos
    log_dp_sub = np.log(dp[si : ei + 1])
    log_P_sub = np.log(P[si : ei + 1])
    log_P_pred = best_fit["ln_A"] + best_fit["beta"] * log_dp_sub
    resids = log_P_sub - log_P_pred

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax2.plot(dp[si : ei + 1], resids, "s-", color="#2ca02c", ms=8, linewidth=1.8, label="Resíduos")
    ax2.fill_between(
        dp[si : ei + 1],
        -best_fit["rmse"],
        best_fit["rmse"],
        color="#2ca02c",
        alpha=0.2,
        label=rf"Banda RMSE ($\pm {best_fit['rmse']:.4f}$)",
    )

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\Delta p$", fontsize=18)
    ax2.set_ylabel(r"$\ln P_{\mathrm{obs}} - \ln P_{\mathrm{fit}}$", fontsize=18)
    ax2.set_title(r"\textbf{(b)} Resíduos Logarítmicos", loc="left", fontsize=16)
    ax2.legend(loc="upper right", fontsize=13, frameon=False)
    ax2.grid(True, linestyle=":", alpha=0.5)

    return fig, (ax1, ax2)


def plot_beta_vs_L(
    beta_df: pd.DataFrame,
    perc_type: str = "node",
    figsize: Tuple[float, float] = (15.0, 5.5),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plota a evolução de beta(L) em função de 1/L e de log2(L).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    L = beta_df["L"].values
    inv_L = 1.0 / L
    beta = beta_df["beta_best"].values
    beta_err = beta_df["beta_err_best"].values

    type_name = "Site (Node)" if perc_type == "node" else "Bond"

    # Painel 1: beta vs 1/L
    ax1.errorbar(
        inv_L,
        beta,
        yerr=beta_err,
        fmt="o",
        color="#1f77b4",
        ms=8,
        mfc="white",
        mew=1.8,
        capsize=3,
        label=r"$\beta(L)$",
    )

    # Ajuste de extrapolação termodinâmica para tamanhos grandes (ex: L >= 1024)
    mask_large = L >= 1024
    if np.sum(mask_large) >= 3:
        fit_extrap = linregress(inv_L[mask_large], beta[mask_large])
        inv_L_dense = np.linspace(0, inv_L.max(), 100)
        ax1.plot(
            inv_L_dense,
            fit_extrap.intercept + fit_extrap.slope * inv_L_dense,
            "--",
            color="#d62728",
            linewidth=1.8,
            label=rf"Extrapolação $L \to \infty$: $\beta_\infty \approx {fit_extrap.intercept:.3f}$",
        )
        ax1.scatter(
            0,
            fit_extrap.intercept,
            color="#d62728",
            s=90,
            zorder=10,
            marker="*",
            label=rf"$\beta_\infty = {fit_extrap.intercept:.3f}$",
        )

    ax1.set_xlabel(r"$1/L$", fontsize=18)
    ax1.set_ylabel(r"$\beta(L)$", fontsize=18)
    ax1.set_title(rf"\textbf{{(a)}} Extrapolação $L \to \infty$ ({type_name})", loc="left", fontsize=16)
    ax1.legend(loc="upper left", fontsize=13, frameon=False)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Painel 2: beta vs L (log2)
    ax2.errorbar(
        L,
        beta,
        yerr=beta_err,
        fmt="s-",
        color="#2ca02c",
        ms=8,
        mfc="white",
        mew=1.8,
        capsize=3,
        label=r"$\beta(L)$",
    )
    ax2.set_xscale("log", base=2)
    ticks_2 = [512, 1024, 2048, 4096, 8192, 16384]
    ax2.set_xticks(ticks_2)
    ax2.set_xticklabels([str(t) for t in ticks_2], fontsize=13)
    ax2.set_xlabel(r"$L$", fontsize=18)
    ax2.set_ylabel(r"$\beta(L)$", fontsize=18)
    ax2.set_title(r"\textbf{(b)} Dependência com Tamanho $L$", loc="left", fontsize=16)
    ax2.legend(loc="upper right", fontsize=13, frameon=False)
    ax2.grid(True, linestyle=":", alpha=0.5)

    return fig, (ax1, ax2)


def plot_second_spanning_cluster_analysis(
    fss_data: Dict[str, Any],
    df: pd.DataFrame,
    perc_type: str = "bond",
    figsize: Tuple[float, float] = (15.0, 11.0),
) -> Tuple[plt.Figure, Any]:
    r"""
    Plota figura abrangente 2x2 para a física dos segundos spanning clusters (k >= 2):
      - (a): Escala de tamanho finito P_2(L) ~ L^{-\beta_2/\nu}
      - (b): Razão de massas universal P_2 / P_1 vs L
      - (c): Probabilidade de múltiplos spanning clusters P(k >= 2) vs p*
      - (d): Crescimento incondicional de \langle P_2 \rangle em função de p*
    """
    fss_df = fss_data["fss_summary"]
    p2_fit = fss_data.get("fss_fit_P2", {})
    sub = df[df["type"] == perc_type]
    col_p = "p_stab_node" if perc_type == "node" else "p_stab_bond"

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # (a) Finite Size Scaling de P_2 com L
    valid_fss = fss_df[fss_df["P2_cond_mean"].notna() & (fss_df["N_k2"] >= 50)]
    ax1.errorbar(
        valid_fss["L"],
        valid_fss["P2_cond_mean"],
        yerr=valid_fss["P2_cond_sem"],
        fmt="o",
        color="#d62728",
        ms=8,
        mfc="white",
        mew=1.8,
        capsize=3,
        label=r"Dados $P_2(L) = S_2 / L^2$",
    )

    if p2_fit and "exponent" in p2_fit:
        L_dense = np.geomspace(valid_fss["L"].min(), valid_fss["L"].max(), 100)
        p2_model = p2_fit["A"] * (L_dense ** (-p2_fit["exponent"]))
        ex_lbl = (
            rf"$P_2 \sim L^{{-{p2_fit['exponent']:.3f}}}$ "
            rf"($\beta_2/\nu \approx 5/48 \approx 0.104$)"
        )
        ax1.plot(L_dense, p2_model, "--", color="black", linewidth=2.0, label=ex_lbl)

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ticks_2 = [512, 1024, 2048, 4096, 8192, 16384]
    ax1.set_xticks(ticks_2)
    ax1.set_xticklabels([str(t) for t in ticks_2], fontsize=13)
    ax1.set_xlabel(r"$L$", fontsize=18)
    ax1.set_ylabel(r"$P_2 = S_2 / L^2$", fontsize=18)
    ax1.set_title(r"\textbf{(a)} Escala de Tamanho Finito: $P_2(L)$", loc="left", fontsize=16)
    ax1.legend(loc="upper right", fontsize=13, frameon=False)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)

    # (b) Razão de massas P_2 / P_1
    ax2.errorbar(
        valid_fss["L"],
        valid_fss["Ratio_P2_P1"],
        yerr=valid_fss["Ratio_P2_P1_sem"],
        fmt="s-",
        color="#1f77b4",
        ms=8,
        mfc="white",
        mew=1.8,
        capsize=3,
        label=r"$\langle S_2 / S_1 \rangle$",
    )
    mean_ratio = float(valid_fss["Ratio_P2_P1"].mean())
    ax2.axhline(
        mean_ratio,
        color="#1f77b4",
        linestyle=":",
        linewidth=1.8,
        label=rf"Média Universal: $\approx {mean_ratio:.3f}$",
    )
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ticks_2)
    ax2.set_xticklabels([str(t) for t in ticks_2], fontsize=13)
    ax2.set_xlabel(r"$L$", fontsize=18)
    ax2.set_ylabel(r"$S_2 / S_1$", fontsize=18)
    ax2.set_title(r"\textbf{(b)} Partição de Massa: $P_2 / P_1$", loc="left", fontsize=16)
    ax2.legend(loc="upper right", fontsize=13, frameon=False)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # (c) Probabilidade de bifurcação P(k >= 2) vs p*
    L_sub_list = [512, 1024, 2048, 4096, 8192, 16384]
    colors_c = plt.cm.plasma(np.linspace(0.1, 0.9, len(L_sub_list)))

    for idx, L in enumerate(L_sub_list):
        sub_L = sub[sub["L"] == L]
        if sub_L.empty:
            continue
        grp = (
            sub_L.groupby("f_T")
            .agg(
                p_mean=(col_p, "mean"),
                freq_k2=("k", lambda s: (s >= 2).mean() * 100.0),
                p2_all=("c2_frac_site", lambda s: s.fillna(0.0).mean()),
            )
            .reset_index()
            .sort_values("f_T")
        )

        ax3.plot(
            grp["p_mean"],
            grp["freq_k2"],
            "o-",
            color=colors_c[idx],
            ms=7,
            label=rf"$L = {L}$",
        )
        ax4.plot(
            grp["p_mean"],
            grp["p2_all"],
            "s-",
            color=colors_c[idx],
            ms=7,
            label=rf"$L = {L}$",
        )

    ax3.set_xlabel(r"$p^*$ (Fração Estacionária)", fontsize=18)
    ax3.set_ylabel(r"$P(k \ge 2\mid L)\ [\%]$", fontsize=18)
    ax3.set_title(r"\textbf{(c)} Ocorrência de $k \ge 2$ vs $p^*$", loc="left", fontsize=16)
    ax3.legend(loc="upper left", fontsize=12, frameon=False)
    ax3.grid(True, linestyle=":", alpha=0.5)

    # (d) Crescimento incondicional de P_2
    ax4.set_xlabel(r"$p^*$ (Fração Estacionária)", fontsize=18)
    ax4.set_ylabel(r"$\langle P_2 \rangle_{\mathrm{all}} = \frac{1}{N}\sum S_{2,i}/L^2$", fontsize=18)
    ax4.set_title(r"\textbf{(d)} Densidade Global do $2^\circ$ Spanning Cluster", loc="left", fontsize=16)
    ax4.legend(loc="upper left", fontsize=12, frameon=False)
    ax4.grid(True, linestyle=":", alpha=0.5)

    return fig, axes

