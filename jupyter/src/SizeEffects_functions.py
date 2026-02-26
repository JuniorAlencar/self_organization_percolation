import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

def fit_pc_finite_size(
    L,
    pc,
    pc_err=None,
    omega=1.0,
    L_min=None,
    plot=False,
    ax=None
):
    """
    Ajuste: p_c(L) = p_inf + a * L^{-omega}

    Parâmetros
    ----------
    L : array-like
        Tamanhos do sistema.
    pc : array-like
        Valores de p_c(L).
    pc_err : array-like ou None
        Erros associados a p_c. Se None, ajuste não ponderado.
    omega : float
        Expoente desejado.
    L_min : float ou None
        Se definido, usa apenas L >= L_min.
    plot : bool
        Se True, faz o plot do ajuste.
    ax : matplotlib axis ou None
        Eixo onde plotar.

    Retorna
    -------
    dict com parâmetros do ajuste.
    """

    L = np.asarray(L, dtype=float)
    pc = np.asarray(pc, dtype=float)

    if pc_err is not None:
        pc_err = np.asarray(pc_err, dtype=float)

    # corte opcional
    if L_min is not None:
        mask = (L >= L_min)
        L = L[mask]
        pc = pc[mask]
        if pc_err is not None:
            pc_err = pc_err[mask]

    x = L ** (-omega)

    def model(x, p_inf, a):
        return p_inf + a * x

    if pc_err is not None:
        popt, pcov = curve_fit(
            model, x, pc,
            sigma=pc_err,
            absolute_sigma=True
        )
    else:
        popt, pcov = curve_fit(model, x, pc)

    p_inf, a = popt
    sigma_pinf, sigma_a = np.sqrt(np.diag(pcov))

    # chi² reduzido (se houver erro)
    chi2_red = None
    if pc_err is not None:
        residuals = pc - model(x, p_inf, a)
        chi2 = np.sum((residuals / pc_err) ** 2)
        ndof = len(pc) - 2
        if ndof > 0:
            chi2_red = chi2 / ndof

    # Plot opcional
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7,6))

        ax.errorbar(
            1/L, pc,
            yerr=pc_err if pc_err is not None else None,
            fmt='o',
            capsize=4
        )

        L_fit = np.linspace(min(L), max(L), 400)
        ax.plot(
            1/L_fit,
            p_inf + a * L_fit**(-omega),
            '-',
            lw=2.5,
            label=rf"$\omega={omega}$" "\n"
                  rf"$p_\infty={p_inf:.6f}\pm{sigma_pinf:.6f}$"
        )

        ax.set_xlabel(r"$1/L$")
        ax.set_ylabel(r"$p_c(L)$")
        ax.legend()
        plt.tight_layout()

    return {
        "omega": omega,
        "p_inf": p_inf,
        "sigma_p_inf": sigma_pinf,
        "a": a,
        "sigma_a": sigma_a,
        "chi2_red": chi2_red
    }

def format_param_parenthesis(value, error):
    """
    Formato: valor(erro)
    Erro com 1 dígito significativo.
    Nunca usa notação científica.
    """

    if error == 0 or np.isnan(error):
        return f"{value}"

    exponent = int(math.floor(math.log10(abs(error))))
    
    # número de casas decimais necessárias
    decimals = -exponent if exponent < 0 else 0

    # erro com 1 dígito significativo
    error_rounded = round(error, decimals)
    value_rounded = round(value, decimals)

    # converte erro para inteiro na casa correta
    error_int = int(round(error_rounded * 10**decimals))

    # formata valor com número fixo de casas
    value_str = f"{value_rounded:.{decimals}f}"

    return f"{value_str}({error_int})"

def format_legend(A, sigma_A, B, sigma_B, nu):

    A_str = format_param_parenthesis(A, sigma_A)
    B_str = format_param_parenthesis(B, sigma_B)

    return (
        rf"$\nu = {nu:.3f}$" + "\n" +
        rf"$A = {A_str}$" + "\n" +
        rf"$B = {B_str}$"
    )

def linear_regression_weighted(x, y, y_err, scale_by_chi2=True, eps=1e-15):
    """
    Regressão linear ponderada.
    Modelo: y = A x + B

    Se scale_by_chi2=True:
      - multiplica as incertezas dos parâmetros por sqrt(chi2_red) quando dof>0,
        o que corrige subestimação quando y_err não bate com a dispersão real.

    Retorna:
    A, B, sigma_A, sigma_B, chi2, chi2_red, R2, y_fit, cov
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)

    # proteção
    y_err = np.maximum(y_err, eps)

    w = 1.0 / (y_err**2)

    S   = np.sum(w)
    Sx  = np.sum(w * x)
    Sy  = np.sum(w * y)
    Sxx = np.sum(w * x * x)
    Sxy = np.sum(w * x * y)

    Delta = S * Sxx - Sx**2
    if not np.isfinite(Delta) or Delta <= 0:
        raise ValueError("Delta inválido (problema numérico/degenerescência).")

    A = (S * Sxy - Sx * Sy) / Delta
    B = (Sxx * Sy - Sx * Sxy) / Delta

    y_fit = A * x + B

    # χ²
    chi2 = np.sum(w * (y - y_fit)**2)
    dof = len(x) - 2
    chi2_red = chi2 / dof if dof > 0 else np.nan

    # covariância (X^T W X)^-1 para [A, B]
    cov = np.array([[ S / Delta,   -Sx / Delta],
                    [-Sx / Delta,  Sxx / Delta]], dtype=float)

    # escala por chi2_red (se desejado e dof válido)
    if scale_by_chi2 and (dof > 0) and np.isfinite(chi2_red) and (chi2_red > 0):
        cov = cov * chi2_red

    sigma_A = float(np.sqrt(cov[0, 0]))
    sigma_B = float(np.sqrt(cov[1, 1]))

    # R² ponderado
    y_mean_w = Sy / S
    chi2_tot = np.sum(w * (y - y_mean_w)**2)
    R2 = 1.0 - chi2 / chi2_tot if chi2_tot > 0 else np.nan

    return A, B, sigma_A, sigma_B, chi2, chi2_red, R2, y_fit, cov


import numpy as np
import math

def format_param_parenthesis(value, error):
    """
    Formato: valor(erro)
    Erro com 1 dígito significativo.
    Nunca usa notação científica.
    """

    if error == 0 or np.isnan(error):
        return f"{value}"

    exponent = int(math.floor(math.log10(abs(error))))
    
    # número de casas decimais necessárias
    decimals = -exponent if exponent < 0 else 0

    # erro com 1 dígito significativo
    error_rounded = round(error, decimals)
    value_rounded = round(value, decimals)

    # converte erro para inteiro na casa correta
    error_int = int(round(error_rounded * 10**decimals))

    # formata valor com número fixo de casas
    value_str = f"{value_rounded:.{decimals}f}"

    return f"{value_str}({error_int})"

def format_legend(A, sigma_A, B, sigma_B, nu):

    A_str = format_param_parenthesis(A, sigma_A)
    B_str = format_param_parenthesis(B, sigma_B)

    return (
        rf"$\nu = {nu:.3f}$" + "\n" +
        rf"$A = {A_str}$" + "\n" +
        rf"$B = {B_str}$"
    )