import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import pandas as pd
from matplotlib.lines import Line2D


df_parameters = pd.read_csv("../SOP_data/best_parameters.csv", sep=',')
df = pd.read_csv("../SOP_data/all_data.dat", sep=' ')
def shortest_path_best(L_lst_filter, L_lst, nc, k, rho, P0, p0):
    f_common = df_parameters[(df_parameters['k']==k) & (df_parameters['P0']==P0) & 
                                (df_parameters['L'].isin(L_lst_filter)) & (df_parameters['n_s']==nc)]['fmin'].max()
    
    NT_lst = [int(L**2*f_common) for L in L_lst]
    
    print(f"fmin = {f_common} to n_s = {nc}")
    
    df_base = df[
        (df['k'] == 1.0e-06) &
        (df['rho'] == rho) &
        (df['nc'] == nc) &
        (df['P0'] == P0) &
        (df['p0'] == p0)
    ].copy()

    rows_nt = []

    for L, Nt_target in zip(L_lst, NT_lst):
        df_L = df_base[df_base['L'] == L].copy()

        if df_L.empty:
            continue

        # pega apenas Nt únicos disponíveis para esse L
        nts_disponiveis = df_L['Nt'].drop_duplicates()

        # índice do Nt mais próximo
        idx_min = (nts_disponiveis - Nt_target).abs().idxmin()
        Nt_escolhido = nts_disponiveis.loc[idx_min]

        rows_nt.append({
            'L': L,
            'Nt_target': Nt_target,
            'Nt_escolhido': Nt_escolhido
        })

    df_nt_match = pd.DataFrame(rows_nt)

    df_filter = df_base.merge(
        df_nt_match[['L', 'Nt_escolhido']],
        left_on=['L', 'Nt'],
        right_on=['L', 'Nt_escolhido'],
        how='inner'
    ).drop(columns='Nt_escolhido')
    
    return df_filter



def linear(X, a, b):
    return a * X + b

def weighted_linear_regression_log2(x, y, y_err):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)

    mask = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(y_err) &
        (x > 0) & (y > 0) & (y_err > 0)
    )

    x = x[mask]
    y = y[mask]
    y_err = y_err[mask]

    X = np.log2(x)
    Y = np.log2(y)
    Y_err = y_err / (y * np.log(2))   # propagação do erro

    popt, pcov = curve_fit(
        linear, X, Y,
        sigma=Y_err,
        absolute_sigma=True
    )

    a, b = popt
    a_err, b_err = np.sqrt(np.diag(pcov))

    Y_fit = linear(X, a, b)
    chi2 = np.sum(((Y - Y_fit) / Y_err)**2)
    dof = len(Y) - 2
    chi2_red = chi2 / dof if dof > 0 else np.nan

    return {
        "a": a,
        "a_err": a_err,
        "b": b,
        "b_err": b_err,
        "X": X,
        "Y": Y,
        "Y_err": Y_err,
        "chi2_red": chi2_red
    }

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

def log2_with_error(val, err):
    val = np.asarray(val, dtype=float)
    err = np.asarray(err, dtype=float)
    y = np.log2(val)
    y_err = err / (val * np.log(2))
    return y, y_err

def fit_linear_weighted(x, y, y_err):
    def linear(x, a, b):
        return a * x + b
    popt, pcov = curve_fit(linear, x, y, sigma=y_err, absolute_sigma=True)
    a, b = popt
    sigma_a, sigma_b = np.sqrt(np.diag(pcov))
    return a, b, sigma_a, sigma_b

def extract_shortest_path_data(df, L_lst, nc, ord, rho=None):
    if rho is None:
        rho = 1 / nc

    short = []
    short_err = []
    for L in L_lst:
        df_sub = df[
            (df["L"] == L) &
            (df["nc"] == nc) &
            (df["order"] == ord) &
            (df["rho"] == rho)
        ]
        if df_sub.empty:
            raise ValueError(f"Sem dados para L={L}, nc={nc}, ord={ord}, rho={rho}")

        short.append(df_sub["shortest_path"].values[0])
        short_err.append(df_sub["shortest_path_err"].values[0])

    return np.asarray(L_lst, float), np.asarray(short, float), np.asarray(short_err, float)

def plot_all_orders_together_two_legends(
    df, L_lst, nc,
    orders=None, rho=None,
    figsize=(9, 7),
    fs_ticks=25, fs_labels=30,
    fs_leg_out=22, fs_leg_in=22,
    lw_data=0.0,   # marcador-only (como seu exemplo)
    lw_fit=3.0,
    capsize=4
):
    if orders is None:
        orders = [i for i in range(1, nc+1)]
    if rho is None:
        rho = 1 / nc

    # cores (se quiser tudo preto, troque por ['k']*len(orders))
    colors = [
        (0.85, 0.1, 0.1),
        (0.1, 0.2, 0.85),
        (0.1, 0.7, 0.2),
        (0.75, 0.2, 0.75),
    ]

    # símbolos diferentes para cada i (como no seu exemplo 2)
    markers = ['o', 's', 'd', '^']  # círculo, quadrado, losango, triângulo

    fig, ax = plt.subplots(figsize=figsize)

    x = np.log2(np.asarray(L_lst, dtype=float))
    x_fit = np.linspace(x.min(), x.max(), 400)

    # --- handles/labels para as duas legendas ---
    handles_out, labels_out = [], []  # fora: i = ...
    handles_in, labels_in = [], []    # dentro: a,b...

    for idx, ord in enumerate(orders):
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]

        L, short, short_err = extract_shortest_path_data(df, L_lst, nc, ord, rho=rho)
        y, y_err = log2_with_error(short, short_err)

        a, b, sigma_a, sigma_b = fit_linear_weighted(x, y, y_err)

        # pontos (marcador vazio com borda colorida, como exemplo)
        ax.errorbar(
            x, y, yerr=y_err,
            fmt=m,
            ms=10,
            mfc='none',
            mec=c,
            mew=2.0,
            ecolor=c,
            elinewidth=1.8,
            capsize=capsize,
            lw=lw_data
        )

        # reta do fit
        ax.plot(x_fit, a*x_fit + b, color=c, lw=lw_fit)

        # -------- legenda fora (somente símbolos + i) --------
        handles_out.append(Line2D([0], [0], marker=m, linestyle='None',
                                  markerfacecolor='none', markeredgecolor=c,
                                  markeredgewidth=2.0, markersize=12))
        labels_out.append(rf"$i= {ord}$")

        # -------- legenda dentro (somente a e b, com a mesma cor da reta) --------
        handles_in.append(Line2D([0], [0], color=c, lw=lw_fit))
        labels_in.append(
            rf"$a={a:.3f}\pm{sigma_a:.3f}$" "\n"
            rf"$b={b:.3f}\pm{sigma_b:.3f}$"
        )

    ax.set_xlabel(r"$\log_2 L$", fontsize=fs_labels)
    ax.set_ylabel(r"$\log_2 \ell$", fontsize=fs_labels)
    ax.tick_params(axis='both', which='major', labelsize=fs_ticks)

    # --- legenda interna: parâmetros (dentro) ---
    leg_in = ax.legend(
        handles_in, labels_in,
        loc="upper left",
        fontsize=fs_leg_in,
        frameon=False,
        handlelength=3.0,
        labelspacing=1.0,
        borderpad=0.2
    )
    ax.add_artist(leg_in)

    # --- legenda externa: i (fora) ---
    ax.legend(
        handles_out, labels_out,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=fs_leg_out,
        frameon=False,
        handletextpad=0.6
    )

    fig.tight_layout()
    return fig, ax
def format_legend(A, sigma_A, B, sigma_B, nu):

    A_str = format_param_parenthesis(A, sigma_A)
    B_str = format_param_parenthesis(B, sigma_B)

    return (
        rf"$\nu = {nu:.3f}$" + "\n" +
        rf"$A = {A_str}$" + "\n" +
        rf"$B = {B_str}$"
    )