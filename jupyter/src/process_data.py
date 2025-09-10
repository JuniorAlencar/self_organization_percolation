import re, os, json, glob, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import math

# --- regex para extrair params do caminho ---
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

def parse_params_from_path(path: str):
    """
    Extrai type_perc, num_colors, dim, L, Nt, k, rho do caminho.
    Retorna dict tipado ou None se não casar.
    """
    # normaliza separadores
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

# --- já existente no seu código ---
_fname_re = re.compile(
    r"P0_([0-9]*\.?[0-9]+(?:e[+\-]?[0-9]+)?)_p0_([0-9]*\.?[0-9]+(?:e[+\-]?[0-9]+)?)_seed_(\d+)\.json$",
    re.IGNORECASE
)


def wilson_ci(M, N, z=1.96):
    if N <= 0:
        return (np.nan, np.nan, np.nan)
    phat = M / N
    denom = 1 + z*z/N
    center = (phat + z*z/(2*N)) / denom
    margin = z * math.sqrt(phat*(1-phat)/N + z*z/(4*N*N)) / denom
    return phat, max(0.0, center-margin), min(1.0, center+margin)

def parse_p0_from_filename(path):
    m = _fname_re.search(os.path.basename(path))
    if not m: 
        return None
    _, p0_str, _ = m.groups()
    try:
        return float(p0_str)
    except Exception:
        return None

def read_orders_one_file(file_path):
    """
    Retorna lista de (order, pt_array, nt_array_ou_None) para um .json com 'results'.
    """
    with open(file_path, "r") as f:
        obj = json.load(f)

    out = []
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        for item in obj["results"]:
            order = item.get("order_percolation", None)
            d = item.get("data", {})
            if order is None or "time" not in d or "pt" not in d:
                continue
            p = np.asarray(d["pt"], float)
            n_arr = np.asarray(d["nt"], float) if "nt" in d else None
            # alinhar comprimentos (caso raro de listas de tamanhos distintos)
            n = min(len(p), len(n_arr)) if n_arr is not None else len(p)
            if n <= 0:
                continue
            p = p[:n]
            n_arr = n_arr[:n] if n_arr is not None else None
            out.append((int(order), p, n_arr))
    return out  # pode ser []


def sem_acf(x, max_lag=None):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return (np.nan, np.nan, np.nan, np.nan)
    mean = x.mean()
    y = x - mean
    var = y.var(ddof=1)
    if var == 0:
        return (mean, 0.0, 0.0, float(n))
    if max_lag is None:
        max_lag = int(np.ceil(n**(1/3)))
    acfs = []
    for k in range(1, max_lag+1):
        acf_k = np.dot(y[:-k], y[k:]) / ((n-k) * var)
        acfs.append(acf_k)
    tau_int = 1.0 + 2.0 * sum(a for a in acfs if a > 0)
    n_eff = max(n / tau_int, 1.0)
    std = np.sqrt(var)
    sem = std / math.sqrt(n_eff)
    return (mean, std, sem, n_eff)


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

def data_single_sample(type_perc, num_colors, dim, L, Nt, k, rho, p0, seed):
    path = f"/home/junior/Documents/self_organization_percolation/Data/{type_perc}_percolation/num_colors_{num_colors}/dim_{dim}/L_{L}/NT_constant/NT_{Nt}/k_{k:.1e}/rho_{rho:.1e}/data/"
    filename = f"P0_0.10_p0_{p0:.2f}_seed_{seed}.json"
    file_path = path + filename
    data = read_orders_one_file(file_path)
    try:
        dict_data = {"t":list(range(len(data[0][1])))}
        dict_data.update({f"p_{i+1}":[float(j) for j in data[i][1]] for i in range(0,4)})
        dict_data.update({f"N_{i+1}":[int(j) for j in data[i][2]] for i in range(0,4)})
        return dict_data
    
    except IndexError as e:
        print("file empty, no percolation occurred, please, enter with other seed or p0:", e)
    except FileNotFoundError as e:
        print("File not found, enter with other parameters:", e)

# ---------- helpers ----------
def tail_mean(x, tail_frac=0.30):
    """Média nos últimos tail_frac da série x (ignora NaN)."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return np.nan
    start = int((1.0 - tail_frac) * n)
    start = max(0, min(start, n-1))
    seg = x[start:]
    return np.nan if seg.size == 0 else np.nanmean(seg)

def bootstrap_mean_scalar(vals, prop, n_boot=20000, ci=0.95, rng=None,):
    """
    Bootstrap do valor médio a partir de vals (array 1D).
    Retorna: mean_point, se_boot, (ci_low, ci_high).
    """
    if(prop=="pt"):
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        m = vals.size
    elif(prop=="Nt"):
        vals = np.asarray(vals, int)
        vals = vals[np.isfinite(vals)]
        m = vals.size
    if m == 0:
        return (np.nan, np.nan, (np.nan, np.nan))
    if rng is None:
        rng = np.random.default_rng()
    if m == 1:
        # com 1 curva, SEM entre-curvas é 0 (ou NaN se preferir)
        return (float(vals[0]), 0.0, (float(vals[0]), float(vals[0])))
    idx = rng.integers(0, m, size=(n_boot, m))
    boot_means = vals[idx].mean(axis=1)
    mean_point = float(vals.mean())
    se_boot    = float(boot_means.std(ddof=1))
    alpha = 0.5*(1-ci)
    lo, hi = np.quantile(boot_means, [alpha, 1-alpha])
    return mean_point, se_boot, (float(lo), float(hi))


def fmt_pm(mean, se, n_used, dec=3, thresh=None):
    """
    Formata ⟨p⟩ ± erro.
    - se n_used <= 1: não mostra erro (não há variância entre-curvas)
    - se se < 10^{-dec} ou se < thresh: mostra como limite, ex.: < 1e-3
    """
    if n_used <= 1 or not np.isfinite(se):
        return rf"$\langle p \rangle = {mean:.{dec}f}$"
    if thresh is None:
        thresh = 10.0**(-dec)
    if se <= 0 or se < thresh:
        return rf"$\langle p \rangle = {mean:.{dec}f}\ \pm\ <{thresh:.0e}$"
    return rf"$\langle p \rangle = {mean:.{dec}f}\ \pm\ {se:.{dec}f}$"


def fmt_pm_N(mean, se, n_used, dec=1, thresh=None):
    r"""
    Formata $\langle N \rangle \pm$ erro para séries N(t).

    - Se n_used <= 1: mostra só a média (não há variância entre-curvas).
    - Se o erro é muito pequeno, mostra como limite:  "< Δ",
      onde Δ é a menor unidade exibida (controlada por `dec`).

    Parâmetros:
      mean   : média
      se     : erro (SEM/boot)
      n_used : nº de curvas usadas
      dec    : casas decimais para exibir (default 1)
      thresh : limiar abaixo do qual vira "< Δ".
               Default: Δ/2, onde Δ = 10^{-dec} (para contagens é razoável).
    """

    if thresh is None:
        # metade da menor unidade exibida (ex.: dec=1 -> Δ=0.1 => thresh=0.05; dec=0 -> Δ=1 => thresh=0.5)
        thresh = 0.5 * (10.0 ** (-dec))

    if n_used <= 1 or not np.isfinite(se):
        return rf"$\langle N \rangle = {mean:.{dec}f}$"

    if se <= 0 or se < thresh:
        # mostra limite na menor unidade exibida
        # para dec>=3 também funciona bem com notação científica:
        if dec >= 3:
            lim = f"{10.0**(-dec):.0e}"       # ex.: 1e-3
        else:
            lim = f"{10.0**(-dec):.{dec}f}"   # ex.: 0.1, 1.0
            # remove zeros e ponto finais desnecessários
            lim = lim.rstrip('0').rstrip('.') if '.' in lim else lim
        return rf"$\langle N \rangle = {mean:.{dec}f}\ \pm\ <{lim}$"

    return rf"$\langle N \rangle = {mean:.{dec}f}\ \pm\ {se:.{dec}f}$"


# # ---- helpers (use os mesmos de antes se já tiver) ----
# def tail_mean(x, tail_frac=0.30):
#     x = np.asarray(x, float); n = len(x)
#     if n == 0: return np.nan
#     start = max(0, min(int((1.0 - tail_frac) * n), n-1))
#     seg = x[start:]
#     return np.nan if seg.size == 0 else np.nanmean(seg)

# def bootstrap_mean_scalar(vals, n_boot=20000, ci=0.95, rng=None):
#     vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
#     m = vals.size
#     if m == 0: return (np.nan, np.nan, (np.nan, np.nan))
#     if rng is None: rng = np.random.default_rng()
#     if m == 1: return (float(vals[0]), 0.0, (float(vals[0]), float(vals[0])))
#     idx = rng.integers(0, m, size=(n_boot, m))
#     boot_means = vals[idx].mean(axis=1)
#     mean_point = float(vals.mean())
#     se_boot = float(boot_means.std(ddof=1))
#     alpha = 0.5*(1-ci)
#     lo, hi = np.quantile(boot_means, [alpha, 1-alpha])
#     return mean_point, se_boot, (float(lo), float(hi))
