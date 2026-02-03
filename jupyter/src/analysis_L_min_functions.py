import numpy as np
import json
import os
import pandas as pd
from .TimeSeriesAnalysis import compute_means_for_folder_tests

def calculate_means_L(N_COLORS, DIM, NT, K, RHO, p0):
    base = f"../Data_tests/bond_percolation/num_colors_{N_COLORS}/dim_{DIM}/"
    L_lst = []

    for entry in os.scandir(base):
        if entry.is_dir():
            L_lst.append(entry.path[-3:])
    
    for L in L_lst:
        compute_means_for_folder_tests(type_perc='bond', num_colors=N_COLORS, dim=DIM, L=L, NT=NT, k=K, rho=RHO, p0_list=p0)
    return L_lst

def read_mean_json(N_COLORS:int, DIM:int, L:int, NT:int, K:float, RHO:float):
    filename = (
        f"../Data_tests/bond_percolation/num_colors_{N_COLORS}/dim_{DIM}/"
        f"L_{L}/NT_constant/NT_{NT}/k_{K:.1e}/rho_{RHO:.4e}/properties_mean_bundle.json"
    )
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data['p0_groups'][0]['orders']


def generate_pc_estimate(L_lst, N_COLORS,DIM, NT, K, RHO):
    all_data = {"L":[], "nc":[],"dim":[], "t0":[], "pc":[], "pc_err":[],"err_rel":[],
            "order":[]}
    
    for L in L_lst:
        data = read_mean_json(N_COLORS, DIM, L, NT, K, RHO)

        series = []
        for i in range(N_COLORS):
            d = data[i]["data"]
            t = np.array(d["time"], dtype=float)
            pt = np.array(d["pt_mean"], dtype=float)
            pt_sem = np.array(d["pt_sem"], dtype=float)
            series.append((t, pt, pt_sem))

        # estima t0 individual e pega t0 global (mais tardio)
        t0_ind = []
        for (t, pt, pt_sem) in series:
            idx0_i = detect_equilibrium_start_with_errors(t, pt, pt_sem, w=40, consec=6, z=2.0, chi2r_max=2.0)
            t0_ind.append(t[idx0_i])

        t0_global = float(max(t0_ind))

        print(f"t0 individual: {t0_ind[0]:.2f}, {t0_ind[1]:.2f}")
        print(f"t0 GLOBAL (usado para ambas): {t0_global:.2f}")

        pc = 0.24881182
        # plot e linhas no mean_eq
        for i, (t, pt, pt_sem) in enumerate(series):
            idx0_g = idx_from_t0(t, t0_global)

            mean_eq, sem_eq = weighted_mean_and_sem(pt[idx0_g:], pt_sem[idx0_g:])
            err = (abs(mean_eq - pc)/pc)*100
            
            all_data["L"].append(L)
            all_data["nc"].append(N_COLORS)
            all_data["t0"].append(t0_global)
            all_data["pc"].append(mean_eq)
            all_data['dim'].append(DIM)
            all_data["pc_err"].append(sem_eq)
            all_data["err_rel"].append(err)
            all_data['order'].append(i+1)
    
    df = pd.DataFrame(data=all_data).sort_values(by=['L'])
    
    return df

def rolling_weighted_mean(y, sem, w):
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)

    n = y.size
    m = n - w + 1
    if m <= 0:
        return np.array([]), np.array([]), np.array([])

    mu = np.empty(m, dtype=float)
    se = np.empty(m, dtype=float)
    chi2r = np.empty(m, dtype=float)

    eps = 1e-15
    for i in range(m):
        yw = y[i:i+w]
        sw = np.maximum(sem[i:i+w], eps)

        wgt = 1.0 / (sw * sw)
        W = np.sum(wgt)

        mui = np.sum(wgt * yw) / W
        sei = np.sqrt(1.0 / W)

        dof = max(w - 1, 1)
        chi2 = np.sum(((yw - mui) / sw) ** 2)
        chi2ri = chi2 / dof

        mu[i] = mui
        se[i] = sei
        chi2r[i] = chi2ri

    return mu, se, chi2r


def detect_equilibrium_start_with_errors(t, y, sem, w=40, lag=None, consec=6, z=2.0, chi2r_max=2.0):
    if lag is None:
        lag = w

    t = np.asarray(t)
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)

    n = y.size
    if n < (w + lag + 5):
        return 0

    mu, se_mu, chi2r = rolling_weighted_mean(y, sem, w)
    m = mu.size
    if m <= lag:
        return 0

    dm = np.abs(mu[lag:] - mu[:-lag])
    se_comb = np.sqrt(se_mu[lag:]**2 + se_mu[:-lag]**2)

    ok_change = dm <= (z * se_comb)
    ok_chi = (chi2r[lag:] <= chi2r_max) & (chi2r[:-lag] <= chi2r_max)
    ok = ok_change & ok_chi

    run = 0
    for j, flag in enumerate(ok):
        run = run + 1 if flag else 0
        if run >= consec:
            return int(j)

    return 0


def weighted_mean_and_sem(y, sem):
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)
    eps = 1e-15
    sem = np.maximum(sem, eps)

    wgt = 1.0 / (sem * sem)
    W = np.sum(wgt)

    mu = float(np.sum(wgt * y) / W)
    se = float(np.sqrt(1.0 / W))
    return mu, se


def idx_from_t0(t, t0):
    t = np.asarray(t)
    return int(np.searchsorted(t, t0, side="left"))