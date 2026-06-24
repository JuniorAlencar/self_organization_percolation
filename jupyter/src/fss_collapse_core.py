# ============================================================
# fss_collapse_core.py
# Funções para colapso:
#
# p_L(f_T) = F_inf(f_T - A L^{-lambda}) + B L^{-lambda}
#
# Variáveis colapsadas:
#
# x_col = f_T - A L^{-lambda}
# y_col = p_mean - B L^{-lambda}
# ============================================================

import numpy as np
import pandas as pd

from scipy.optimize import least_squares


# ============================================================
# 1. Funções básicas
# ============================================================

def poly_predict(x, theta, x_ref, x_scale):
    """
    Avalia F_inf(x), modelada como polinômio em variável reescalada.

    z = (x - x_ref)/x_scale
    F_inf(x) = theta_0 + theta_1 z + theta_2 z^2 + ...
    """
    x = np.asarray(x, dtype=float)
    z = (x - x_ref) / x_scale
    return np.polynomial.polynomial.polyval(z, theta)


def p_mean_infinity(f_T, result):
    """
    Retorna p_mean no limite L -> infinito.

    No modelo:
        p_L(f_T) = F_inf(f_T - A L^{-lambda}) + B L^{-lambda}

    Quando L -> infinito:
        p_inf(f_T) = F_inf(f_T)
    """
    theta_inf = result["theta_inf"]
    x_ref = result["x_ref"]
    x_scale = result["x_scale"]

    return poly_predict(f_T, theta_inf, x_ref, x_scale)


# ============================================================
# 2. Filtro dos dados para um dado c e L
# ============================================================

def get_filtered_curve(
    df_series,
    L,
    c,
    nc,
    p0,
    P0,
    order,
    dim,
    type_perc="bond",
    min_samples_perc=5,
    p_col="p_mean",
    f_col="f_T",
    p_cut=0.90,
    p_cut_mode="remove",
):
    """
    Filtra os dados para um dado par (c, L).

    p_cut_mode:
        "remove":
            remove todos os pontos com p_mean >= p_cut.
            Reproduz seu bloco atual:
                df_plot = df_plot[df_plot["p_mean"] < 0.90]

        "include_first":
            mantém os pontos até o primeiro p_mean >= p_cut,
            incluindo esse primeiro ponto.

        None ou "none":
            não aplica corte em p_mean.
    """

    mask = (
        (df_series["type_perc"] == type_perc)
        & np.isclose(df_series["p0"].astype(float), p0)
        & np.isclose(df_series["P0"].astype(float), P0)
        & (df_series["order"].astype(int) == int(order))
        & (df_series["nc"].astype(int) == int(nc))
        & (df_series["L"].astype(int) == int(L))
        & np.isclose(df_series["c"].astype(float), c)
        & (df_series["dim"].astype(int) == int(dim))
    )

    df = df_series[mask].copy()

    if len(df) == 0:
        raise ValueError(f"Nenhum dado encontrado para c={c}, L={L}.")

    df = df.dropna(subset=[f_col, p_col])
    df = df.sort_values(f_col).reset_index(drop=True)

    df = df[(df[p_col] >= 0) & (df[p_col] <= 1)]

    if "N_samples" in df.columns and "N_samples_perc" in df.columns:
        df = df[df["N_samples"] == df["N_samples_perc"]]

    if "N_samples_perc" in df.columns:
        df = df[df["N_samples_perc"] >= min_samples_perc]

    df = df.sort_values(f_col).reset_index(drop=True)

    # Se houver f_T repetido, tira média de p_mean.
    df = (
        df
        .groupby(f_col, as_index=False)
        .agg({p_col: "mean"})
        .sort_values(f_col)
        .reset_index(drop=True)
    )

    if p_cut is not None and p_cut_mode is not None:
        if p_cut_mode == "remove":
            df = df[df[p_col] < p_cut].copy()

        elif p_cut_mode == "include_first":
            idx_cut = np.where(df[p_col].to_numpy() >= p_cut)[0]
            if len(idx_cut) > 0:
                df = df.iloc[:idx_cut[0] + 1].copy()

        elif p_cut_mode == "none":
            pass

        else:
            raise ValueError(
                "p_cut_mode deve ser 'remove', 'include_first', 'none' ou None."
            )

    df = df.sort_values(f_col).reset_index(drop=True)

    return df


def build_curves_for_c(
    df_series,
    c,
    L_lst,
    nc,
    p0,
    P0,
    order,
    dim,
    type_perc="bond",
    min_samples_perc=5,
    p_col="p_mean",
    f_col="f_T",
    p_cut=0.90,
    p_cut_mode="remove",
    verbose=False,
):
    """
    Monta um dicionário:

        curves[L] = dataframe filtrado para aquele L

    para um dado valor de c.
    """

    curves = {}

    for L in L_lst:
        df_L = get_filtered_curve(
            df_series=df_series,
            L=L,
            c=c,
            nc=nc,
            p0=p0,
            P0=P0,
            order=order,
            dim=dim,
            type_perc=type_perc,
            min_samples_perc=min_samples_perc,
            p_col=p_col,
            f_col=f_col,
            p_cut=p_cut,
            p_cut_mode=p_cut_mode,
        )

        if len(df_L) < 4:
            raise ValueError(
                f"Poucos pontos para c={c}, L={L}: {len(df_L)} pontos."
            )

        curves[L] = df_L

        if verbose:
            print(
                f"c={c}, L={L}: "
                f"{f_col}_min={df_L[f_col].min():.8f}, "
                f"{f_col}_max={df_L[f_col].max():.8f}, "
                f"N={len(df_L)}"
            )

    return curves


# ============================================================
# 3. Domínio comum ou completo
# ============================================================

def restrict_to_common_domain(
    curves,
    L_lst,
    f_col="f_T",
    min_points=4,
):
    """
    Restringe todas as curvas ao intervalo comum de f_T.
    """

    f_min_common = max(curves[L][f_col].min() for L in L_lst)
    f_max_common = min(curves[L][f_col].max() for L in L_lst)

    if f_max_common <= f_min_common:
        raise ValueError("Não existe domínio comum de f_T entre todos os L.")

    curves_common = {}

    for L in L_lst:
        df = curves[L]
        df_aux = df[
            (df[f_col] >= f_min_common)
            & (df[f_col] <= f_max_common)
        ].copy()

        df_aux = df_aux.sort_values(f_col).reset_index(drop=True)

        if len(df_aux) < min_points:
            raise ValueError(
                f"L={L} ficou com poucos pontos no domínio comum: {len(df_aux)}."
            )

        curves_common[L] = df_aux

    return curves_common


def get_scale_from_curves(
    curves,
    L_lst,
    f_col="f_T",
):
    """
    Define escala numérica para o polinômio F_inf.
    """

    f_min = min(curves[L][f_col].min() for L in L_lst)
    f_max = max(curves[L][f_col].max() for L in L_lst)

    x_ref = 0.5 * (f_min + f_max)
    x_scale = 0.5 * (f_max - f_min)

    if x_scale == 0:
        raise ValueError("x_scale ficou zero.")

    return x_ref, x_scale, f_min, f_max


# ============================================================
# 4. Fit do colapso para lambda fixo
# ============================================================

def fit_collapse_for_lambda(
    curves,
    L_lst,
    lamb,
    degree=3,
    p_col="p_mean",
    f_col="f_T",
    initial_params=None,
):
    """
    Ajusta, para lambda fixo:

        p_L(f_T) = F_inf(f_T - A L^{-lambda}) + B L^{-lambda}

    Parâmetros ajustados:
        theta_inf : coeficientes de F_inf
        A_shift   : deslocamento horizontal
        B_vert    : correção vertical
    """

    x_ref, x_scale, f_min, f_max = get_scale_from_curves(
        curves=curves,
        L_lst=L_lst,
        f_col=f_col,
    )

    x_all = []
    y_all = []

    for L in L_lst:
        df = curves[L]
        x_all.append(df[f_col].to_numpy(dtype=float))
        y_all.append(df[p_col].to_numpy(dtype=float))

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)

    if initial_params is None:
        z_all = (x_all - x_ref) / x_scale
        theta0 = np.polynomial.polynomial.polyfit(
            z_all,
            y_all,
            deg=degree,
        )

        A0 = 0.0
        B0 = 0.0

        params0 = np.r_[theta0, A0, B0]
    else:
        params0 = np.asarray(initial_params, dtype=float)

    def residual(params):
        theta = params[:degree + 1]
        A_shift = params[degree + 1]
        B_vert = params[degree + 2]

        res = []

        for L in L_lst:
            df = curves[L]

            x = df[f_col].to_numpy(dtype=float)
            y = df[p_col].to_numpy(dtype=float)

            correction = float(L) ** (-lamb)

            x_eff = x - A_shift * correction
            y_pred = poly_predict(x_eff, theta, x_ref, x_scale) + B_vert * correction

            res.append(y_pred - y)

        return np.concatenate(res)

    opt = least_squares(
        residual,
        params0,
        max_nfev=30000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    params = opt.x
    res = residual(params)

    sse = np.sum(res**2)
    sst = np.sum((y_all - np.mean(y_all))**2)

    r2 = 1.0 - sse / sst
    rmse = np.sqrt(np.mean(res**2))

    result = {
        "lambda": lamb,
        "params": params,
        "theta_inf": params[:degree + 1],
        "A_shift": params[degree + 1],
        "B_vert": params[degree + 2],
        "R2": r2,
        "RMSE": rmse,
        "success": opt.success,
        "message": opt.message,
        "x_ref": x_ref,
        "x_scale": x_scale,
        "f_min": f_min,
        "f_max": f_max,
        "degree": degree,
    }

    return result


# ============================================================
# 5. Varredura em lambda
# ============================================================

def scan_lambda_collapse(
    curves,
    L_lst,
    lambda_min=0.4,
    lambda_max=1.4,
    n_lambda=101,
    degree=3,
    p_col="p_mean",
    f_col="f_T",
    warm_start=True,
):
    """
    Varre lambda em [lambda_min, lambda_max],
    calcula R² para cada lambda e retorna:

        df_scan
        best_result
        all_results
    """

    lambda_grid = np.linspace(lambda_min, lambda_max, n_lambda)

    all_results = []
    initial_params = None

    for lamb in lambda_grid:
        result_lamb = fit_collapse_for_lambda(
            curves=curves,
            L_lst=L_lst,
            lamb=lamb,
            degree=degree,
            p_col=p_col,
            f_col=f_col,
            initial_params=initial_params,
        )

        all_results.append(result_lamb)

        if warm_start:
            initial_params = result_lamb["params"]

    df_scan = pd.DataFrame({
        "lambda": [r["lambda"] for r in all_results],
        "R2": [r["R2"] for r in all_results],
        "RMSE": [r["RMSE"] for r in all_results],
        "A_shift": [r["A_shift"] for r in all_results],
        "B_vert": [r["B_vert"] for r in all_results],
        "success": [r["success"] for r in all_results],
    })

    idx_best = df_scan["R2"].idxmax()
    best_result = all_results[idx_best]

    return df_scan, best_result, all_results


# ============================================================
# 6. Dados colapsados e curva ajustada
# ============================================================

def make_collapse_data(
    curves,
    L_lst,
    result,
    p_col="p_mean",
    f_col="f_T",
):
    """
    Retorna os dados colapsados para cada L:

        x_col = f_T - A L^{-lambda}
        y_col = p_mean - B L^{-lambda}
    """

    lamb = result["lambda"]
    A_shift = result["A_shift"]
    B_vert = result["B_vert"]

    collapse_data = {}

    for L in L_lst:
        df = curves[L].copy()

        correction = float(L) ** (-lamb)

        df["x_col"] = df[f_col].to_numpy(dtype=float) - A_shift * correction
        df["y_col"] = df[p_col].to_numpy(dtype=float) - B_vert * correction
        df["L"] = L

        collapse_data[L] = df

    return collapse_data


def make_fit_curve_from_collapse(
    collapse_data,
    L_lst,
    result,
    n_points=600,
):
    """
    Retorna a curva F_inf no intervalo dos dados colapsados.
    """

    x_min = min(collapse_data[L]["x_col"].min() for L in L_lst)
    x_max = max(collapse_data[L]["x_col"].max() for L in L_lst)

    x_fit = np.linspace(x_min, x_max, n_points)

    y_fit = poly_predict(
        x_fit,
        result["theta_inf"],
        result["x_ref"],
        result["x_scale"],
    )

    return pd.DataFrame({
        "x": x_fit,
        "y": y_fit,
    })


# ============================================================
# 7. Função principal: calcula tudo para um único c
# ============================================================

def collapse_for_c(
    df_series,
    c,
    L_lst,
    nc,
    p0,
    P0,
    order,
    dim,
    type_perc="bond",
    min_samples_perc=5,
    p_col="p_mean",
    f_col="f_T",
    p_cut=0.90,
    p_cut_mode="remove",
    domain_mode="common",
    degree=3,
    lambda_min=0.4,
    lambda_max=1.4,
    n_lambda=101,
    verbose=False,
):
    """
    Calcula tudo que você precisa para um dado valor de c.

    Retorna um dicionário com:

        result["curves"]
        result["collapse_data"]
        result["fit_curve"]
        result["scan"]
        result["best_result"]
        result["lambda_best"]
        result["R2_best"]
        result["RMSE_best"]
    """

    curves_all = build_curves_for_c(
        df_series=df_series,
        c=c,
        L_lst=L_lst,
        nc=nc,
        p0=p0,
        P0=P0,
        order=order,
        dim=dim,
        type_perc=type_perc,
        min_samples_perc=min_samples_perc,
        p_col=p_col,
        f_col=f_col,
        p_cut=p_cut,
        p_cut_mode=p_cut_mode,
        verbose=verbose,
    )

    if domain_mode == "common":
        curves_used = restrict_to_common_domain(
            curves=curves_all,
            L_lst=L_lst,
            f_col=f_col,
            min_points=degree + 2,
        )

    elif domain_mode == "full":
        curves_used = curves_all

    else:
        raise ValueError("domain_mode deve ser 'common' ou 'full'.")

    df_scan, best_result, all_results = scan_lambda_collapse(
        curves=curves_used,
        L_lst=L_lst,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        n_lambda=n_lambda,
        degree=degree,
        p_col=p_col,
        f_col=f_col,
        warm_start=True,
    )

    collapse_data = make_collapse_data(
        curves=curves_used,
        L_lst=L_lst,
        result=best_result,
        p_col=p_col,
        f_col=f_col,
    )

    fit_curve = make_fit_curve_from_collapse(
        collapse_data=collapse_data,
        L_lst=L_lst,
        result=best_result,
        n_points=600,
    )

    out = {
        "c": c,
        "curves_all": curves_all,
        "curves": curves_used,
        "collapse_data": collapse_data,
        "fit_curve": fit_curve,
        "scan": df_scan,
        "all_results": all_results,
        "best_result": best_result,
        "lambda_best": best_result["lambda"],
        "R2_best": best_result["R2"],
        "RMSE_best": best_result["RMSE"],
        "A_shift": best_result["A_shift"],
        "B_vert": best_result["B_vert"],
    }

    if verbose:
        print("\n===== Melhor colapso =====")
        print(f"c           = {c}")
        print(f"lambda_best = {out['lambda_best']:.6f}")
        print(f"R2_best     = {out['R2_best']:.10f}")
        print(f"RMSE_best   = {out['RMSE_best']:.10e}")
        print(f"A_shift     = {out['A_shift']:.10e}")
        print(f"B_vert      = {out['B_vert']:.10e}")

    return out

def fit_linear_feedback_model(
    t,
    p,
    f,
    fT,
    c,
    t_stat=None,
    p_star=None,
    tail_fraction=0.3,
    u_max=None,
    eta_max=None,
    fit_after_t_stat=False,
    min_points=10
):
    """
    Ajusta o modelo linear efetivo:

        u_{t+1} = u_t - c eta_t
        eta_{t+1} = a u_t + b eta_t

    com

        u_t = p_t - p*
        eta_t = f(t) - f_T

    Aqui c NÃO é estimado. Ele é fornecido como parâmetro de controle.

    Parameters
    ----------
    t : array
        Vetor de tempos.

    p : array
        Série temporal p(t).

    f : array
        Série temporal f(t) = N(t)/L^(d-1).

    fT : float
        Valor alvo f_T.

    c : float
        Parâmetro de controle da regra de atualização do SOP.

    t_stat : float, optional
        Tempo a partir do qual o regime estacionário é considerado.
        Usado para estimar p* se p_star não for fornecido.

    p_star : float, optional
        Valor estacionário de p. Se None, será estimado.

    tail_fraction : float
        Fração final da série usada para estimar p*, caso t_stat=None.

    u_max : float, optional
        Janela local em |u_t|.

    eta_max : float, optional
        Janela local em |eta_t|.

    fit_after_t_stat : bool
        Se True, usa apenas pontos com t >= t_stat no ajuste.

    min_points : int
        Número mínimo de pontos necessários para ajustar.

    Returns
    -------
    result : dict
        Dicionário com p_star, a, b, c, lambdas, Jacobiano e dados usados.
    """

    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    f = np.asarray(f, dtype=float)

    if not (len(t) == len(p) == len(f)):
        raise ValueError("t, p e f devem ter o mesmo tamanho.")

    if len(t) < 3:
        raise ValueError("A série precisa ter pelo menos 3 pontos.")

    # ------------------------------------------------------------
    # 1. Estima p*
    # ------------------------------------------------------------
    if p_star is None:
        if t_stat is not None:
            mask_stat = t >= t_stat

            if np.sum(mask_stat) < min_points:
                raise ValueError(
                    "Poucos pontos para estimar p_star com esse t_stat."
                )

            p_star = np.mean(p[mask_stat])

        else:
            n_tail = int(tail_fraction * len(p))
            n_tail = max(n_tail, min_points)

            p_star = np.mean(p[-n_tail:])

    # ------------------------------------------------------------
    # 2. Define u_t e eta_t
    # ------------------------------------------------------------
    u = p - p_star
    eta = f - fT

    # Variáveis no tempo t
    u_t = u[:-1]
    eta_t = eta[:-1]
    t_t = t[:-1]

    # Variáveis no tempo t+1
    u_next = u[1:]
    eta_next = eta[1:]

    # ------------------------------------------------------------
    # 3. Máscara para ajuste local
    # ------------------------------------------------------------
    mask = np.ones_like(u_t, dtype=bool)

    if u_max is not None:
        mask &= np.abs(u_t) <= u_max

    if eta_max is not None:
        mask &= np.abs(eta_t) <= eta_max

    if fit_after_t_stat and t_stat is not None:
        mask &= t_t >= t_stat

    n_points_fit = np.sum(mask)

    if n_points_fit < min_points:
        raise ValueError(
            f"Poucos pontos para o ajuste: {n_points_fit} pontos. "
            "Tente aumentar u_max, eta_max ou desativar fit_after_t_stat."
        )

    u_fit = u_t[mask]
    eta_fit = eta_t[mask]

    u_next_fit = u_next[mask]
    eta_next_fit = eta_next[mask]

    # ------------------------------------------------------------
    # 4. Ajusta apenas a e b:
    #
    #       eta_{t+1} = a u_t + b eta_t
    # ------------------------------------------------------------
    X_ab = np.column_stack([u_fit, eta_fit])
    y_ab = eta_next_fit

    coef_ab, *_ = np.linalg.lstsq(X_ab, y_ab, rcond=None)

    a_est = coef_ab[0]
    b_est = coef_ab[1]

    # ------------------------------------------------------------
    # 5. Usa c fornecido pelo usuário
    # ------------------------------------------------------------
    c_used = float(c)

    # ------------------------------------------------------------
    # 6. Monta o Jacobiano
    # ------------------------------------------------------------
    J = np.array([
        [1.0, -c_used],
        [a_est, b_est]
    ])

    lambdas = np.linalg.eigvals(J)

    lambda_1 = lambdas[0]
    lambda_2 = lambdas[1]

    spectral_radius = np.max(np.abs(lambdas))

    if 0 < spectral_radius < 1:
        tau = -1.0 / np.log(spectral_radius)
    else:
        tau = np.nan

    # ------------------------------------------------------------
    # 7. Predições do modelo
    # ------------------------------------------------------------
    u_next_pred = u_fit - c_used * eta_fit
    eta_next_pred = a_est * u_fit + b_est * eta_fit

    mse_u = np.mean((u_next_fit - u_next_pred)**2)
    mse_eta = np.mean((eta_next_fit - eta_next_pred)**2)

    return {
        "p_star": p_star,
        "a": a_est,
        "b": b_est,
        "c": c_used,
        "J": J,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "spectral_radius": spectral_radius,
        "tau": tau,
        "mse_u": mse_u,
        "mse_eta": mse_eta,
        "n_points_fit": n_points_fit,
        "u_fit": u_fit,
        "eta_fit": eta_fit,
        "u_next_fit": u_next_fit,
        "eta_next_fit": eta_next_fit,
        "u_next_pred": u_next_pred,
        "eta_next_pred": eta_next_pred,
    }

def label(i):
    """
    Retorna labels no formato LaTeX:
        0 -> r"$(a)$"
        1 -> r"$(b)$"
        ...
        25 -> r"$(z)$"
        26 -> r"$(aa)$"
        27 -> r"$(ab)$"
    """
    letters = ""
    i = int(i)

    while True:
        i, r = divmod(i, 26)
        letters = chr(ord("a") + r) + letters
        if i == 0:
            break
        i -= 1

    return fr"$({letters})$"