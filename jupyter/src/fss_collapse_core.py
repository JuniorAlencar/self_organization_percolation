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

import numpy as np


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
    min_points=10,
    memory_mode=None,
    include_intercept=False,
):
    """
    Ajusta o modelo linear efetivo com ou sem memória.

    Modelo original:

        u_{t+1} = u_t - c eta_t
        eta_{t+1} = a u_t + b eta_t

    Modelo com atraso explícito:

        eta_{t+1} = h0 + a u_t + b eta_t + d eta_{t-1}

    Modelo com diferença temporal:

        eta_{t+1} = h0 + a u_t + b eta_t + d(eta_t - eta_{t-1})

    com

        u_t = p_t - p*
        eta_t = f(t) - f_T

    Parameters
    ----------
    memory_mode : None, "eta_lag" ou "delta_eta"
        None:
            usa o modelo original.
        "eta_lag":
            usa eta_{t-1} como variável adicional.
        "delta_eta":
            usa eta_t - eta_{t-1} como variável adicional.

    include_intercept : bool
        Se True, inclui termo constante h0 no ajuste de eta_{t+1}.

    Returns
    -------
    result : dict
        Dicionário com parâmetros ajustados, matriz dinâmica, autovalores,
        métricas e dados usados no ajuste.
    """

    valid_memory_modes = [None, "eta_lag", "delta_eta"]
    if memory_mode not in valid_memory_modes:
        raise ValueError(
            f"memory_mode deve ser um destes valores: {valid_memory_modes}"
        )

    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    f = np.asarray(f, dtype=float)

    if not (len(t) == len(p) == len(f)):
        raise ValueError("t, p e f devem ter o mesmo tamanho.")

    if memory_mode is None:
        if len(t) < 3:
            raise ValueError("A série precisa ter pelo menos 3 pontos.")
    else:
        if len(t) < 4:
            raise ValueError(
                "A série precisa ter pelo menos 4 pontos para usar memória."
            )

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

    # ------------------------------------------------------------
    # 3. Monta as variáveis alinhadas no tempo
    # ------------------------------------------------------------
    if memory_mode is None:
        # Transições:
        # t -> t+1
        u_t = u[:-1]
        eta_t = eta[:-1]
        t_t = t[:-1]

        u_next = u[1:]
        eta_next = eta[1:]

        eta_lag = None
        delta_eta = None

    else:
        # Transições:
        # t -> t+1, mas agora usando eta_{t-1}
        #
        # índices:
        # eta_lag = eta[t-1] = eta[:-2]
        # eta_t   = eta[t]   = eta[1:-1]
        # eta_next= eta[t+1] = eta[2:]
        u_t = u[1:-1]
        eta_t = eta[1:-1]
        eta_lag = eta[:-2]
        delta_eta = eta_t - eta_lag
        t_t = t[1:-1]

        u_next = u[2:]
        eta_next = eta[2:]

    # ------------------------------------------------------------
    # 4. Máscara para ajuste local
    # ------------------------------------------------------------
    mask = np.ones_like(u_t, dtype=bool)

    mask &= np.isfinite(u_t)
    mask &= np.isfinite(eta_t)
    mask &= np.isfinite(u_next)
    mask &= np.isfinite(eta_next)

    if memory_mode is not None:
        mask &= np.isfinite(eta_lag)
        mask &= np.isfinite(delta_eta)

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

    if memory_mode is not None:
        eta_lag_fit = eta_lag[mask]
        delta_eta_fit = delta_eta[mask]
    else:
        eta_lag_fit = None
        delta_eta_fit = None

    # ------------------------------------------------------------
    # 5. Monta matriz de regressão para eta_{t+1}
    # ------------------------------------------------------------
    X_cols = []
    param_names = []

    if include_intercept:
        X_cols.append(np.ones_like(u_fit))
        param_names.append("h0")

    X_cols.append(u_fit)
    param_names.append("a")

    X_cols.append(eta_fit)
    param_names.append("b")

    if memory_mode == "eta_lag":
        X_cols.append(eta_lag_fit)
        param_names.append("d")

    elif memory_mode == "delta_eta":
        X_cols.append(delta_eta_fit)
        param_names.append("d")

    X = np.column_stack(X_cols)
    y = eta_next_fit

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    coef_dict = {
        name: value for name, value in zip(param_names, coef)
    }

    h0_est = coef_dict.get("h0", 0.0)
    a_est = coef_dict["a"]
    b_est = coef_dict["b"]
    d_est = coef_dict.get("d", 0.0)

    # ------------------------------------------------------------
    # 6. Usa c fornecido pelo usuário
    # ------------------------------------------------------------
    c_used = float(c)

    # ------------------------------------------------------------
    # 7. Monta matriz dinâmica
    # ------------------------------------------------------------
    if memory_mode is None:
        # Estado:
        # X_t = (u_t, eta_t)
        #
        # X_{t+1} = J X_t
        J = np.array([
            [1.0, -c_used],
            [a_est, b_est],
        ])

    elif memory_mode == "eta_lag":
        # Estado aumentado:
        # X_t = (u_t, eta_t, eta_{t-1})
        #
        # u_{t+1}     = u_t - c eta_t
        # eta_{t+1}   = a u_t + b eta_t + d eta_{t-1}
        # eta_t       = eta_t
        J = np.array([
            [1.0, -c_used, 0.0],
            [a_est, b_est, d_est],
            [0.0, 1.0, 0.0],
        ])

    elif memory_mode == "delta_eta":
        # Modelo ajustado:
        # eta_{t+1} = a u_t + b eta_t + d(eta_t - eta_{t-1})
        #
        # Forma equivalente:
        # eta_{t+1} = a u_t + (b+d) eta_t - d eta_{t-1}
        b_eff = b_est + d_est
        d_eff = -d_est

        J = np.array([
            [1.0, -c_used, 0.0],
            [a_est, b_eff, d_eff],
            [0.0, 1.0, 0.0],
        ])

    lambdas = np.linalg.eigvals(J)
    spectral_radius = np.max(np.abs(lambdas))

    if 0 < spectral_radius < 1:
        tau = -1.0 / np.log(spectral_radius)
    else:
        tau = np.nan

    # ------------------------------------------------------------
    # 8. Predições do modelo
    # ------------------------------------------------------------
    u_next_pred = u_fit - c_used * eta_fit
    eta_next_pred = X @ coef

    residual_eta = eta_next_fit - eta_next_pred
    residual_u = u_next_fit - u_next_pred

    mse_u = np.mean(residual_u**2)
    mse_eta = np.mean(residual_eta**2)

    rmse_u = np.sqrt(mse_u)
    rmse_eta = np.sqrt(mse_eta)

    mae_eta = np.mean(np.abs(residual_eta))

    ss_res = np.sum(residual_eta**2)
    ss_tot = np.sum((eta_next_fit - np.mean(eta_next_fit))**2)

    if ss_tot > 0:
        r2_eta = 1.0 - ss_res / ss_tot
    else:
        r2_eta = np.nan

    n = len(eta_next_fit)
    k = X.shape[1]
    eps = 1e-300

    aic_eta = n * np.log(ss_res / n + eps) + 2 * k
    bic_eta = n * np.log(ss_res / n + eps) + k * np.log(n)

    # ------------------------------------------------------------
    # 9. Mede contribuição da memória
    # ------------------------------------------------------------
    linear_no_memory_part = a_est * u_fit + b_est * eta_fit

    if memory_mode == "eta_lag":
        memory_part = d_est * eta_lag_fit

    elif memory_mode == "delta_eta":
        memory_part = d_est * delta_eta_fit

    else:
        memory_part = np.zeros_like(eta_fit)

    linear_rms = np.sqrt(np.mean(linear_no_memory_part**2))
    memory_rms = np.sqrt(np.mean(memory_part**2))

    if linear_rms > 0:
        R_memory = memory_rms / linear_rms
    else:
        R_memory = np.nan

    # ------------------------------------------------------------
    # 10. Retorno
    # ------------------------------------------------------------
    result = {
        "memory_mode": memory_mode,
        "include_intercept": include_intercept,

        "p_star": p_star,
        "h0": h0_est,
        "a": a_est,
        "b": b_est,
        "d": d_est,
        "c": c_used,

        "J": J,
        "lambdas": lambdas,
        "spectral_radius": spectral_radius,
        "tau": tau,

        "mse_u": mse_u,
        "mse_eta": mse_eta,
        "rmse_u": rmse_u,
        "rmse_eta": rmse_eta,
        "mae_eta": mae_eta,
        "r2_eta": r2_eta,
        "aic_eta": aic_eta,
        "bic_eta": bic_eta,
        "R_memory": R_memory,

        "n_points_fit": n_points_fit,
        "param_names": param_names,
        "coef": coef,
        "coef_dict": coef_dict,

        "u_fit": u_fit,
        "eta_fit": eta_fit,
        "eta_lag_fit": eta_lag_fit,
        "delta_eta_fit": delta_eta_fit,

        "u_next_fit": u_next_fit,
        "eta_next_fit": eta_next_fit,

        "u_next_pred": u_next_pred,
        "eta_next_pred": eta_next_pred,

        "residual_u": residual_u,
        "residual_eta": residual_eta,
    }

    # Mantém compatibilidade parcial com o caso 2D antigo
    if len(lambdas) >= 1:
        result["lambda_1"] = lambdas[0]
    if len(lambdas) >= 2:
        result["lambda_2"] = lambdas[1]
    if len(lambdas) >= 3:
        result["lambda_3"] = lambdas[2]

    # Coeficientes efetivos úteis para interpretar delta_eta
    if memory_mode == "delta_eta":
        result["b_eff"] = b_est + d_est
        result["d_eff"] = -d_est

    return result

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