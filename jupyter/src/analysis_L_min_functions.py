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

    group0 = data["p0_groups"][0]
    num_seeds = group0["num_seeds"]

    orders_raw = group0["orders"]  # dict com chaves "0","1",...

    if isinstance(orders_raw, dict) and "0" in orders_raw:
        orders = [orders_raw[str(i)] for i in range(N_COLORS)]
    elif isinstance(orders_raw, dict):
        orders = [orders_raw[i] for i in range(N_COLORS)]
    else:
        orders = orders_raw  # caso já seja lista

    return orders, num_seeds

def generate_pc_estimate(L_lst, N_COLORS, DIM, NT, K, RHO, pc_ref=0.24881182):
    fn = "../Data_tests/bond_percolation/pc_estimative.csv"

    # -----------------------------
    # carrega CSV existente (se houver)
    # -----------------------------
    if os.path.exists(fn):
        df_old = pd.read_csv(fn)
    else:
        df_old = pd.DataFrame()

    rows_to_add = []          # linhas novas (recalculadas ou novas)
    keys_to_replace = set()   # chaves (casos) que serão removidos do df_old e substituídos

    for L in L_lst:
        L_int = int(L)

        # caminho do json mean (seu "file_mean")
        file_mean = (
            f"../Data_tests/bond_percolation/num_colors_{N_COLORS}/dim_{DIM}/"
            f"L_{L_int}/NT_constant/NT_{NT}/k_{K:.1e}/rho_{RHO:.4e}/properties_mean_bundle.json"
        )

        # lê o json e pega num_seeds
        with open(file_mean, "r", encoding="utf-8") as f:
            js = json.load(f)

        group0 = js["p0_groups"][0]
        num_seeds = group0["num_seeds"]  # N_samples real do json

        # orders pode vir como dict com chaves "0","1",...
        orders = group0["orders"]
        if isinstance(orders, dict):
            # garante lista ordenada 0..N_COLORS-1
            orders = [orders[str(i)] for i in range(N_COLORS)]

        # -----------------------------
        # decide se precisa recalcular
        # -----------------------------
        need_recalc = True

        if not df_old.empty:
            # filtra as linhas do CSV para esse "caso" (mesmos parâmetros globais)
            mask_case = (
                (df_old.get("L") == L_int) &
                (df_old.get("nc") == N_COLORS) &
                (df_old.get("dim") == DIM) &
                (df_old.get("NT") == NT) &
                (df_old.get("k") == float(K)) &
                (df_old.get("rho") == float(RHO))
            )

            # se já existem linhas para esse caso, checa N_samples
            if mask_case.any():
                # se qualquer linha divergir, recalcula e substitui tudo desse caso
                old_samples = df_old.loc[mask_case, "N_samples"].unique()
                if len(old_samples) == 1 and int(old_samples[0]) == int(num_seeds):
                    need_recalc = False
                else:
                    need_recalc = True

        if not need_recalc:
            # está atualizado, não faz nada
            continue

        # marca esse caso para ser substituído no CSV
        keys_to_replace.add((L_int, N_COLORS, DIM, NT, float(K), float(RHO)))

        # -----------------------------
        # RECALCULA as propriedades
        # -----------------------------
        series = []
        for i in range(N_COLORS):
            d = orders[i]["data"]
            t = np.array(d["time"], dtype=float)
            pt = np.array(d["pt_mean"], dtype=float)
            pt_sem = np.array(d["pt_sem"], dtype=float)
            series.append((t, pt, pt_sem))

        # t0 global (mais tardio)
        t0_ind = []
        for (t, pt, pt_sem) in series:
            idx0_i = detect_equilibrium_start_with_errors(
                t, pt, pt_sem, w=40, consec=6, z=2.0, chi2r_max=2.0
            )
            t0_ind.append(t[idx0_i])
        t0_global = float(max(t0_ind))

        # cria linhas novas (uma por "order")
        for i, (t, pt, pt_sem) in enumerate(series):
            idx0_g = idx_from_t0(t, t0_global)
            mean_eq, sem_eq = weighted_mean_and_sem(pt[idx0_g:], pt_sem[idx0_g:])
            err = (abs(mean_eq - pc_ref) / pc_ref) * 100

            rows_to_add.append({
                "L": L_int,
                "nc": N_COLORS,
                "dim": DIM,
                "NT": NT,
                "k": float(K),
                "rho": float(RHO),

                "t0": t0_global,
                "pc": mean_eq,
                "pc_err": sem_eq,
                "err_rel": err,
                "order": i + 1,

                "N_samples": int(num_seeds),
            })

    # -----------------------------
    # monta df_final: remove casos antigos substituídos e adiciona novos
    # -----------------------------
    if df_old.empty:
        df_final = pd.DataFrame(rows_to_add)
    else:
        # remove todas as linhas dos casos que serão substituídos
        if keys_to_replace:
            mask_keep = np.ones(len(df_old), dtype=bool)

            for (L_int, nc, dim, NT_, k_, rho_) in keys_to_replace:
                m = (
                    (df_old["L"] == L_int) &
                    (df_old["nc"] == nc) &
                    (df_old["dim"] == dim) &
                    (df_old["NT"] == NT_) &
                    (df_old["k"] == k_) &
                    (df_old["rho"] == rho_)
                )
                mask_keep &= ~m

            df_kept = df_old.loc[mask_keep].copy()
        else:
            df_kept = df_old.copy()

        df_new = pd.DataFrame(rows_to_add)
        df_final = pd.concat([df_kept, df_new], ignore_index=True)

    # ordena (opcional)
    if not df_final.empty:
        df_final = df_final.sort_values(by=["nc", "dim", "NT", "k", "rho", "L", "order"]).reset_index(drop=True)

    # salva sem índice
    df_final.to_csv(fn, index=False)


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

# def detect_equilibrium_start_with_errors(t, y, sem, w=40, lag=None, consec=6, z=2.0, chi2r_max=2.0):
#     if lag is None:
#         lag = w

#     t = np.asarray(t)
#     y = np.asarray(y, dtype=float)
#     sem = np.asarray(sem, dtype=float)

#     n = y.size
#     if n < (w + lag + 5):
#         return 0

#     mu, se_mu, chi2r = rolling_weighted_mean(y, sem, w)
#     m = mu.size
#     if m <= lag:
#         return 0

#     dm = np.abs(mu[lag:] - mu[:-lag])
#     se_comb = np.sqrt(se_mu[lag:]**2 + se_mu[:-lag]**2)

#     ok_change = dm <= (z * se_comb)
#     ok_chi = (chi2r[lag:] <= chi2r_max) & (chi2r[:-lag] <= chi2r_max)
#     ok = ok_change & ok_chi

#     run = 0
#     for j, flag in enumerate(ok):
#         run = run + 1 if flag else 0
#         if run >= consec:
#             return int(j)

#     return 0
def detect_equilibrium_start_with_errors(t, y, sem, w=40, lag=None, consec=6, z=2.0, chi2r_max=2.0,
                                         tail_frac=0.20, min_start_frac=0.05):
    """
    Retorna o ÍNDICE em t (não em mu/ok) correspondente ao início do equilíbrio.
    Se não encontrar um 'run' válido, usa fallback baseado no platô final (tail).
    """
    if lag is None:
        lag = w

    t = np.asarray(t)
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)

    n = y.size
    if n < (w + lag + 5):
        return 0

    # rolling_weighted_mean deve devolver mu e se_mu alinhados com janelas do sinal original.
    # Assumimos aqui que mu[k] corresponde à janela que termina em y[k+w-1].
    mu, se_mu, chi2r = rolling_weighted_mean(y, sem, w)
    m = mu.size
    if m <= lag:
        return 0

    dm = np.abs(mu[lag:] - mu[:-lag])
    se_comb = np.sqrt(se_mu[lag:]**2 + se_mu[:-lag]**2)

    ok_change = dm <= (z * se_comb)
    ok_chi = (chi2r[lag:] <= chi2r_max) & (chi2r[:-lag] <= chi2r_max)
    ok = ok_change & ok_chi

    # evita detectar "equilíbrio" cedo demais por acaso:
    # força começar a procurar após uma fração inicial do tempo
    min_mu_idx = int(np.floor(min_start_frac * m))
    j_start = max(0, min_mu_idx - lag)

    run = 0
    for j in range(j_start, ok.size):
        run = run + 1 if ok[j] else 0
        if run >= consec:
            # j refere-se ao vetor ok. O par comparado é (mu[j], mu[j+lag]).
            # O "ponto atual" (mais tardio) é mu_idx = j + lag.
            mu_idx = j + lag

            # mu_idx -> índice em y/t: se mu[k] é janela que termina em y[k+w-1]
            idx_t = mu_idx + (w - 1)
            idx_t = int(np.clip(idx_t, 0, n - 1))
            return idx_t

    # -------------------------
    # FALLBACK (não retorna 0)
    # -------------------------
    # Estima platô final (últimos tail_frac) e acha o primeiro ponto em que
    # o rolling_mean entra e permanece próximo desse platô.
    tail_start = int(np.floor((1.0 - tail_frac) * n))
    tail_start = np.clip(tail_start, 0, n - 1)

    mu_tail, se_tail = weighted_mean_and_sem(y[tail_start:], sem[tail_start:])
    # tolerância: z * erro combinado do ponto vs platô
    # usa erro do platô (se_tail) e erro do ponto (sem[i])
    tol = z * np.sqrt(se_tail**2 + np.maximum(sem, 1e-15)**2)

    # procura o primeiro i (após min_start_frac) tal que |y - mu_tail| <= tol por 'consec' passos
    i0 = int(np.floor(min_start_frac * n))
    run = 0
    for i in range(i0, n):
        if abs(y[i] - mu_tail) <= tol[i]:
            run += 1
            if run >= consec:
                return int(i - consec + 1)
        else:
            run = 0

    # último fallback: começa no tail_start
    return int(tail_start)



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