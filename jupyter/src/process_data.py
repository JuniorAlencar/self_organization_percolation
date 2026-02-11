import re
from pathlib import Path
import os, glob
import pandas as pd
from typing import Sequence, Optional, Literal, Dict, Any, List, Tuple
import numpy as np
import math
from collections import defaultdict
import json
import gc


FLOAT = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

DIR_RE = re.compile(
    rf"""
    ^.*?bond_percolation
    /num_colors_(?P<nc>\d+)
    /dim_(?P<dim>\d+)
    /L_(?P<L>\d+)
    /NT_constant
    /NT_(?P<Nt>\d+)
    /k_(?P<k>{FLOAT})
    /rho_(?P<rho>{FLOAT})
    /data/?$
    """,
    re.X
)


def collect_param_combinations(
    root_dir: str,
    *,
    type_perc: Optional[str] = None,
    dir_re: re.Pattern = DIR_RE,
) -> List[Tuple[str, int, int, int, int, float, float]]:
    """
    Varre root_dir (ex: "../Data/bond_percolation/")
    e retorna lista de tuplas:

    (type_perc, nc, dim, L, NT, k, rho)

    Sem caminho.
    """

    root_dir = os.path.normpath(root_dir)

    # Inferir type_perc automaticamente se não fornecido
    if type_perc is None:
        base = os.path.basename(root_dir)
        type_perc = base.replace("_percolation", "")

    combos = set()

    for dirpath, dirnames, filenames in os.walk(root_dir):

        if os.path.basename(dirpath) != "data":
            continue

        path_norm = os.path.normpath(dirpath).replace(os.sep, "/")
        m = dir_re.match(path_norm)
        if not m:
            continue

        nc = int(m.group("nc"))
        dim = int(m.group("dim"))
        L = int(m.group("L"))
        NT = int(m.group("Nt"))
        k = float(m.group("k"))
        rho = float(m.group("rho"))

        combos.add((type_perc, nc, dim, L, NT, k, rho))

    # ordenar para execução consistente
    combos = sorted(
        combos,
        key=lambda x: (x[0], x[1], x[2], x[4], x[5], x[6], x[3])
    )

    return combos

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

def parse_data_dir(path: str):
    """Extrai nc, dim, L, Nt, k, rho de um path de diretório '.../data'."""
    m = DIR_RE.match(Path(path).as_posix())
    if not m:
        return None
    g = m.groupdict()
    return {
        "nc": int(g["nc"]),
        "dim": int(g["dim"]),
        "L": int(g["L"]),
        "Nt": int(g["Nt"]),
        "k": float(g["k"]),
        "rho": float(g["rho"]),
    }

def _scalar_or_last(x):
    """Converte para escalar; se for lista/array, pega o último valor."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(x[-1]) if len(x) > 0 else np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def read_experiment_json(path):
    """
    Lê um arquivo JSON do experimento e retorna um dicionário
    com a mesma estrutura do JSON: {'meta': ..., 'results': ...}.

    - Mantém todos os tipos e listas como no arquivo original.
    - Se 'meta' ou 'results' não existirem, retorna vazios compatíveis.

    Parâmetros
    ----------
    path : str | Path
        Caminho para o arquivo .json

    Retorno
    -------
    data_dict : dict
        {'meta': dict, 'results': dict[str, dict]}
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    meta = raw.get("meta", {}) or {}
    results = raw.get("results", {}) or {}

    # Garantia leve de formato: cada entrada em results deve ser um dict
    # (não altera conteúdo interno, apenas evita valores None/escalares acidentais)
    fixed_results = {}
    for k, v in results.items():
        fixed_results[k] = v if isinstance(v, dict) else {}

    return {"meta": meta, "results": fixed_results}

def tail_mean(
    x: Sequence[float],
    *,
    tail_len: Optional[int] = None,
    tail_frac: Optional[float] = 0.2,
    method: Literal["iid", "autocorr"] = "iid",
    max_lag: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calcula a média na cauda e o erro associado.

    Parâmetros
    ----------
    x : sequência numérica (list/np.ndarray)
        Série de dados.
    tail_len : int, opcional
        Quantidade de pontos finais usados. Se None, usa tail_frac.
    tail_frac : float in (0,1], opcional
        Fração final usada (por padrão, 20% finais) se tail_len não for fornecido.
    method : {"iid","autocorr"}
        - "iid": assume independência; sem = s/√N.
        - "autocorr": corrige para autocorrelação via τ_int estimado.
    max_lag : int, opcional
        Máximo defasagem para estimar autocorrelação quando method="autocorr".
        Se None, usa min( N_tail//2 , 1000 ).

    Retorna
    -------
    dict com:
        mean : float         -> média na cauda
        sem  : float         -> erro padrão da média
        std  : float         -> desvio-padrão amostral da cauda
        n_tail : int         -> tamanho da cauda usada
        start_idx : int      -> índice inicial da cauda (inclusivo)
        method : str         -> método de erro usado
        n_eff : float        -> tamanho efetivo (apenas em "autocorr"; em "iid" = n_tail)
        tau_int : float      -> tempo de autocorrelação integrado (apenas em "autocorr")
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x deve ser um vetor 1D não vazio.")

    N = x.size
    if tail_len is None:
        if tail_frac is None or not (0 < tail_frac <= 1):
            raise ValueError("Defina tail_len OU tail_frac em (0,1].")
        tail_len = max(1, int(np.floor(tail_frac * N)))
    tail_len = min(tail_len, N)

    start_idx = N - tail_len
    tail = x[start_idx:]

    mean = float(np.mean(tail))
    # Desvio-padrão amostral (Bessel, ddof=1) — se só 1 ponto, std=0
    std = float(np.std(tail, ddof=1)) if tail_len > 1 else 0.0

    if method == "iid" or tail_len <= 1 or std == 0.0:
        sem = std / np.sqrt(tail_len) if tail_len > 0 else np.nan
        return {
            "mean": mean,
            "sem": float(sem),
            "std": std,
            "n_tail": int(tail_len),
            "start_idx": int(start_idx),
            "method": "iid",
            "n_eff": float(tail_len),
            "tau_int": 0.0,
        }

    # --- método com autocorrelação ---
    # Estima a autocorrelação normalizada até max_lag e integra enquanto positiva (regra de Sokal).
    y = tail - mean
    n = y.size
    if max_lag is None:
        max_lag = min(n // 2, 1000)

    # autocorrelação via FFT (O(n log n))
    # corr[k] = sum_{t} y[t]*y[t+k]
    def _acf_fft(v):
        m = int(2 ** np.ceil(np.log2(2 * len(v) - 1)))
        fv = np.fft.rfft(v, n=m)
        acf = np.fft.irfft(fv * np.conj(fv), n=m)[:len(v)]
        return acf

    acf_raw = _acf_fft(y)
    acf_raw = acf_raw / acf_raw[0]  # normaliza: acf[0] = 1

    # τ_int = 0.5 + sum_{k=1..K} acf[k], parando quando acf vira negativa (janela Geyer/Sokal)
    tau_int = 0.5
    for k in range(1, max_lag + 1):
        if k >= len(acf_raw):
            break
        if acf_raw[k] <= 0:
            break
        tau_int += acf_raw[k]

    # N_eff = n / (2 * τ_int); limite inferior 1
    n_eff = max(1.0, n / (2.0 * tau_int))
    sem = std / np.sqrt(n_eff)

    return {
        "mean": mean,
        "sem": float(sem),
        "std": std,
        "n_tail": int(n),
        "start_idx": int(start_idx),
        "method": "autocorr",
        "n_eff": float(n_eff),
        "tau_int": float(tau_int),
    }


# mantém como antes
desired_cols = ['filename','P0','p0','order','p_mean','p_std','p_sem','shortest_path','S_perc']

def process_one_data_dir(data_dir: str, verbose: bool = True):
    info = parse_data_dir(data_dir)
    if info is None:
        if verbose:
            print(f"[skip] não bate o padrão: {data_dir}")
        return

    data_dir = Path(data_dir)
    filename_save = data_dir / "all_data.dat"
    all_files = sorted(glob.glob(str(data_dir / "*.json")))
    if not all_files:
        if verbose:
            print(f"[vazio] sem JSON em: {data_dir}")
        if not filename_save.exists():
            # cria arquivo vazio com cabeçalho
            pd.DataFrame(columns=desired_cols).to_csv(
                filename_save, sep=" ", index=False, na_rep="NaN"
            )
            if verbose: print(f"[created-empty] {filename_save} (sem arquivos JSON)")
        return

    # --- carrega df existente (robusto) ---
    df = None
    known = set()
    if filename_save.exists():
        try:
            df = pd.read_csv(
                filename_save,
                sep=r"\s+",
                engine="python",
                dtype={"filename": str},
                na_values=["nan","NaN","None",""],   # adicione "Null" aqui se usar na_rep="Null"
            )
            if isinstance(df, pd.DataFrame) and ("filename" in df.columns):
                known = set(df["filename"].astype(str).values)
            else:
                if verbose:
                    print(f"[warn] '{filename_save.name}' sem coluna 'filename'; vou recriar.")
                df = None
                known = set()
        except Exception as e:
            if verbose:
                print(f"[warn] Falha ao ler {filename_save}: {e}. Vou recriar.")
            df = None
            known = set()

    dict_new = {c: [] for c in desired_cols}

    # contadores diagnósticos
    n_skipped_no_data = 0
    n_skipped_empty_pt = 0
    n_skipped_bad_order = 0
    n_added_empty_results = 0

    added = 0
    for file in all_files:
        filename = os.path.basename(file)
        if filename in known:
            continue  # já registrado

        file_content = read_experiment_json(file)
        results_block = file_content.get('results', {})

        parms = parse_filename(filename)

        # results vazio -> registra linha com NaN nas colunas pedidas
        if not results_block:
            dict_new['filename'].append(filename)
            dict_new['P0'].append(parms['P0'])
            dict_new['p0'].append(parms['p0'])
            dict_new['order'].append(np.nan)
            dict_new['p_mean'].append(np.nan)
            dict_new['p_std'].append(np.nan)
            dict_new['p_sem'].append(np.nan)
            dict_new['shortest_path'].append(np.nan)
            dict_new['S_perc'].append(np.nan)
            added += 1
            n_added_empty_results += 1
            continue

        # results não-vazio: processa normalmente cada 'order'
        for order_key, node in results_block.items():
            d = (node or {}).get('data', None)
            if not d:
                n_skipped_no_data += 1
                continue

            pt = d.get('pt', [])
            # mesmo se pt vazio, registramos linha com NaNs e (se possível) o order
            if pt is None or len(pt) == 0:
                try:
                    order_num = int(str(order_key).split()[-1])
                except Exception:
                    order_num = np.nan
                    n_skipped_bad_order += 1

                dict_new['filename'].append(filename)
                dict_new['P0'].append(parms['P0'])
                dict_new['p0'].append(parms['p0'])
                dict_new['order'].append(order_num)
                dict_new['p_mean'].append(np.nan)
                dict_new['p_std'].append(np.nan)
                dict_new['p_sem'].append(np.nan)
                dict_new['shortest_path'].append(_scalar_or_last(d.get('shortest_path_lin')))
                dict_new['S_perc'].append(_scalar_or_last(d.get('M_size')))
                added += 1
                n_skipped_empty_pt += 1
                continue

            # estatística da cauda
            s = tail_mean(pt, tail_frac=0.2, method="autocorr")

            try:
                order_num = int(str(order_key).split()[-1])
            except Exception:
                n_skipped_bad_order += 1
                order_num = np.nan

            dict_new['filename'].append(filename)
            dict_new['P0'].append(parms['P0'])
            dict_new['p0'].append(parms['p0'])
            dict_new['order'].append(order_num)
            dict_new['p_mean'].append(s['mean'])
            dict_new['p_std'].append(s['std'])
            dict_new['p_sem'].append(s['sem'])
            dict_new['shortest_path'].append(_scalar_or_last(d.get('shortest_path_lin')))
            dict_new['S_perc'].append(_scalar_or_last(d.get('M_size')))
            added += 1

    # --- escreve/atualiza ---
    if added > 0:
        df_new = pd.DataFrame(dict_new)
        if df is None:
            df = df_new[desired_cols]
        else:
            for c in desired_cols:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[desired_cols]
            df = pd.concat([df, df_new[desired_cols]], ignore_index=True)

        # remove duplicados por (filename, order)
        df = df.drop_duplicates(subset=["filename", "order"], keep="last").reset_index(drop=True)
        df.to_csv(filename_save, sep=" ", index=False, na_rep="NaN")
        if verbose:
            print(f"[ok] {added} linhas → {filename_save} (inclui results vazios: {n_added_empty_results})")
    else:
        # sem novidades; ainda assim garante existência do arquivo com cabeçalho
        if not filename_save.exists():
            pd.DataFrame(columns=desired_cols).to_csv(
                filename_save, sep=" ", index=False, na_rep="NaN"
            )
            if verbose:
                print(f"[created-empty] {filename_save} (sem novidades)")

    if verbose:
        print(f"[info] skipped: no_data={n_skipped_no_data}, empty_pt={n_skipped_empty_pt}, bad_order={n_skipped_bad_order}, added_empty_results={n_added_empty_results}")


def process_all_roots(base_root="../Data/bond_percolation", verbose=True, clean_outputs=False):
    """
    Percorre todas as pastas que batem o padrão:
      ../Data/bond_percolation/num_colors_{nc}/dim_{dim}/L_{L}/NT_constant/NT_{Nt}/k_{k}/rho_{rho}/data
    e chama process_one_data_dir para cada 'data/'.

    clean_outputs=True  -> remove 'all_data.dat' de cada pasta antes de processar.
    """
    base = Path(base_root)
    guess = base.glob("num_colors_*/dim_*/L_*/NT_constant/NT_*/k_*/rho_*/data")

    count = 0
    deleted = 0
    for d in guess:
        dposix = d.as_posix()
        if not DIR_RE.match(dposix):
            if verbose:
                print(f"[ignorado] {dposix} (não bate regex)")
            continue

        if clean_outputs:
            target = Path(d) / "all_data.dat"
            if target.exists():
                try:
                    os.remove(target)
                    deleted += 1
                    if verbose:
                        print(f"[clean] removido: {target}")
                except Exception as e:
                    if verbose:
                        print(f"[warn] não consegui remover {target}: {e}")

        process_one_data_dir(dposix, verbose=verbose)
        count += 1

    if verbose:
        msg = f"[done] pastas processadas: {count}"
        if clean_outputs:
            msg += f" | arquivos removidos: {deleted}"
        print(msg)
    return {"processed_dirs": count, "removed_all_data": deleted if clean_outputs else 0}



FLOAT = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
RGX_FLEX = re.compile(
    rf'^P0_(?P<P0>{FLOAT})_p0_(?P<p0>{FLOAT})_seed_(?P<seed>\d+)\.json$'
)

def parse_filename(path):
    m = RGX_FLEX.match(Path(path).name)
    if not m:
        raise ValueError(f"Nome inválido: {path}")
    return {
        "P0": float(m.group("P0")),
        "p0": float(m.group("p0")),
        "seed": int(m.group("seed")),
    }
def combine_tail_means(run_stats: list[dict], random_effects: bool = True):
    """
    run_stats: lista de dicts com, no mínimo, {'mean': ..., 'sem': ...}
               (ex.: a saída da sua função tail_mean(method="autocorr"))
    random_effects: True -> DerSimonian-Laird; False -> fixed-effect

    Retorna: {'mean': ..., 'se': ..., 'method': 'FE'|'RE', 'tau2': ... , 'R': ...}
    """
    # filtra runs válidos
    stats = [d for d in run_stats if (d is not None and math.isfinite(d.get('mean', float('nan')))
                                      and math.isfinite(d.get('sem', float('nan'))) and d['sem'] > 0)]
    R = len(stats)
    if R == 0:
        return {'mean': float('nan'), 'se': float('nan'), 'method': 'FE', 'tau2': 0.0, 'R': 0}

    means = [d['mean'] for d in stats]
    vars_ = [d['sem']**2 for d in stats]
    w = [1.0/v for v in vars_]

    # fixed-effect
    sumw = sum(w)
    m_fe = sum(wi*mi for wi, mi in zip(w, means)) / sumw
    se_fe = (1.0 / sumw) ** 0.5

    if not random_effects or R == 1:
        return {'mean': m_fe, 'se': se_fe, 'method': 'FE', 'tau2': 0.0, 'R': R}

    # heterogeneidade (DL)
    Q = sum(wi*(mi - m_fe)**2 for wi, mi in zip(w, means))
    c = sumw - sum(wi*wi for wi in w) / sumw
    tau2 = max(0.0, (Q - (R - 1)) / c) if c > 0 else 0.0

    w_star = [1.0/(v + tau2) for v in vars_]
    sumw_star = sum(w_star)
    m_re = sum(wi*mi for wi, mi in zip(w_star, means)) / sumw_star
    se_re = (1.0 / sumw_star) ** 0.5

    return {'mean': m_re, 'se': se_re, 'method': 'RE', 'tau2': tau2, 'R': R}

# ------------------------------------------------------------
# Leitura robusta de all_data.dat
# ------------------------------------------------------------
def _load_all_data(file_path: str) -> pd.DataFrame:
    """
    Lê all_data.dat aceitando múltiplos espaços e 'NaN' como nulos.
    Garante colunas esperadas; se faltarem, cria com NaN.
    Também faz 'backfill' de P0/p0 a partir do filename quando estiverem faltando.
    """
    expected = ['filename','P0','p0','order','p_mean','p_std','p_sem','shortest_path','S_perc']
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=expected)

    try:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            engine="python",
            dtype={"filename": str},
            na_values=["nan","NaN","Null","None",""]
        )
    except Exception:
        df = pd.DataFrame(columns=expected)

    # garante colunas
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # tipos numéricos
    for c in ["P0","p0","order","p_mean","p_std","p_sem","shortest_path","S_perc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["filename"] = df["filename"].astype(str)

    # ---- BACKFILL de P0/p0 quando NaN ----
    mask_missing = df["P0"].isna() | df["p0"].isna()
    if mask_missing.any():
        def _try_parse(row):
            try:
                vals = parse_filename(row["filename"])
                if np.isnan(row["P0"]): row["P0"] = vals["P0"]
                if np.isnan(row["p0"]): row["p0"] = vals["p0"]
            except Exception:
                pass
            return row
        df.loc[mask_missing] = df.loc[mask_missing].apply(_try_parse, axis=1)

    return df[expected]


# ------------------------------------------------------------
# Estatística auxiliar: média e erro padrão (std/sqrt(n))
# ------------------------------------------------------------
def _mean_sem(arr_like):
    v = pd.to_numeric(pd.Series(arr_like), errors="coerce").dropna().to_numpy()
    n = v.size
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n > 0 else np.nan
    return mean, sem, n


# ------------------------------------------------------------
# Agrega um diretório .../data por 'order', produzindo linhas
# ------------------------------------------------------------
def _summarize_one_data_dir(data_dir: str) -> list[dict]:
    """
    Retorna uma lista de dicionários (uma linha por (order, P0, p0)) com:
    L, Nt, k, nc, rho, p0, P0, order, N_samples, p_mean, p_err,
    shortest_path, shortest_path_err, S_perc, S_perc_err
    """
    info = parse_data_dir(data_dir)
    if info is None:
        return []

    all_file = Path(data_dir) / "all_data.dat"
    df = _load_all_data(all_file.as_posix())

    # linhas válidas: precisam ter order e P0/p0 definidos
    df_valid = df.dropna(subset=["order", "P0", "p0"]).copy()
    if df_valid.empty:
        return []

    rows = []
    # agrupa por (order, P0, p0) para não misturar rodadas com parâmetros diferentes
    for (order, P0_val, p0_val), sub in df_valid.groupby(["order", "P0", "p0"]):
        order = int(order)

        # N_samples = nº de seeds (filenames) distintos nesse grupo
        N_samples = int(sub["filename"].nunique())

        # p_mean/p_err combinando por seed (última linha por filename)
        run_stats = []
        for fname, g in sub.groupby("filename"):
            last = g.tail(1).iloc[0]
            m = last.get("p_mean", np.nan)
            s = last.get("p_sem", np.nan)
            if np.isfinite(m) and np.isfinite(s) and s >= 0:
                run_stats.append({"mean": float(m), "sem": float(s)})

        if len(run_stats) > 0:
            combo = combine_tail_means(run_stats, random_effects=False)
            p_mean = float(combo["mean"])
            p_err  = float(combo["se"])
        else:
            p_mean, p_err = np.nan, np.nan

        sp_mean, sp_sem, _ = _mean_sem(sub["shortest_path"])
        sperc_mean, sperc_sem, _ = _mean_sem(sub["S_perc"])

        rows.append({
            "L":   int(info["L"]),
            "Nt":  int(info["Nt"]),
            "k":   float(info["k"]),
            "nc":  int(info["nc"]),
            "rho": float(info["rho"]),
            "p0":  float(p0_val),
            "P0":  float(P0_val),
            "order": int(order),
            "N_samples": N_samples,
            "p_mean": p_mean,
            "p_err":  p_err,
            "shortest_path": sp_mean,
            "shortest_path_err": sp_sem,
            "S_perc": sperc_mean,
            "S_perc_err": sperc_sem,
        })

    return rows



# ------------------------------------------------------------
# Varre TODAS as pastas e escreve 1 arquivo por dimensão:
# ../Data/bond_percolation/all_data_{dim}D.dat
# ------------------------------------------------------------
def summarize_all_dirs(base_root: str = "../Data/bond_percolation",
                       verbose: bool = True) -> dict[int, Path]:
    """
    Percorre todas as pastas que batem o padrão e cria arquivos agregados por dimensão:
      <base_root>/all_data_{dim}D.dat

    Agrega diretamente a partir de cada all_data.dat, criando uma linha por (P0, p0, order).
    """
    base = Path(base_root)
    outputs: dict[int, Path] = {}
    buckets: dict[int, list[dict]] = {}

    for d in base.glob("num_colors_*/dim_*/L_*/NT_constant/NT_*/k_*/rho_*/data"):
        dposix = d.as_posix()
        m_dir = DIR_RE.match(dposix)  # <<< NÃO usar 'm'
        if not m_dir:
            if verbose:
                print(f"[ignorado] {dposix} (não bate regex)")
            continue

        info = parse_data_dir(dposix)
        if info is None:
            if verbose:
                print(f"[ignorado] {dposix} (parse falhou)")
            continue

        all_file = Path(d) / "all_data.dat"
        if not all_file.exists():
            if verbose:
                print(f"[info] sem all_data.dat em: {dposix}")
            continue

        df = _load_all_data(all_file.as_posix())
        if df.empty:
            if verbose:
                print(f"[info] all_data.dat vazio em: {dposix}")
            continue

        rows: list[dict] = []
        for (P0_val, p0_val, order_val), ggrp in df.groupby(["P0", "p0", "order"], dropna=False):
            if pd.isna(order_val):
                continue
            try:
                order_int = int(order_val)
            except Exception:
                continue

            N_samples = int(ggrp["filename"].nunique())

            run_stats = []
            for fname, gfname in ggrp.groupby("filename"):
                last = gfname.tail(1).iloc[0]
                p_mean_val = float(last.get("p_mean", float("nan")))
                p_sem_val  = float(last.get("p_sem",  float("nan")))
                if np.isfinite(p_mean_val) and np.isfinite(p_sem_val) and p_sem_val >= 0:
                    run_stats.append({"mean": p_mean_val, "sem": p_sem_val})

            if run_stats:
                combo = combine_tail_means(run_stats, random_effects=False)
                p_mean = float(combo["mean"])
                p_err  = float(combo["se"])
            else:
                p_mean, p_err = np.nan, np.nan

            sp_mean, sp_sem, _ = _mean_sem(ggrp["shortest_path"])
            sperc_mean, sperc_sem, _ = _mean_sem(ggrp["S_perc"])

            rows.append({
                "L":   int(info["L"]),
                "Nt":  int(info["Nt"]),
                "k":   float(info["k"]),
                "nc":  int(info["nc"]),
                "rho": float(info["rho"]),
                "p0":  float(p0_val) if pd.notna(p0_val) else np.nan,
                "P0":  float(P0_val) if pd.notna(P0_val) else np.nan,
                "order": order_int,
                "N_samples": N_samples,
                "p_mean": p_mean,
                "p_err":  p_err,
                "shortest_path": sp_mean,
                "shortest_path_err": sp_sem,
                "S_perc": sperc_mean,
                "S_perc_err": sperc_sem,
            })

        if not rows:
            if verbose:
                print(f"[info] sem grupos (P0,p0,order) válidos em: {dposix}")
            continue

        dim = int(m_dir.group("dim"))  # <<< agora é o Match
        buckets.setdefault(dim, []).extend(rows)
        if verbose:
            print(f"[ok] {len(rows)} linhas agregadas de: {dposix}")

    # grava um arquivo por dimensão
    for dim, records in buckets.items():
        out_df = pd.DataFrame.from_records(records, columns=[
            "L","Nt","k","nc","rho","p0","P0","order",
            "N_samples","p_mean","p_err",
            "shortest_path","shortest_path_err",
            "S_perc","S_perc_err"
        ])
        # ordem estável opcional
        try:
            out_df = out_df.sort_values(
                by=["L","nc","rho","p0","P0","order","Nt","k"],
                kind="mergesort",
                ignore_index=True
            )
        except Exception:
            pass

        out_path = base / f"all_data_{dim}D.dat"
        out_df.to_csv(out_path.as_posix(), sep=" ", index=False, na_rep="NaN")
        outputs[dim] = out_path
        if verbose:
            print(f"[write] {out_path}  ({len(out_df)} linhas)")

    if verbose and not buckets:
        print("[done] nenhuma pasta com ordens válidas encontrada.")

    return outputs

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

def _get_keys_for_series(d: Dict[str, Any]) -> tuple[str, str, str]:
    # Você está gerando assim no _average_by_order_new
    return "time", "pt_mean", "pt_sem"

_FNAME_RE = re.compile(
    r"P0_([0-9]*\.?[0-9]+(?:e[+\-]?[0-9]+)?)_p0_([0-9]*\.?[0-9]+(?:e[+\-]?[0-9]+)?)_seed_(\d+)\.json$",
    re.IGNORECASE,
)

def _avg_series_across_seeds(items: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """
    Média/SEM ao longo das seeds para uma SÉRIE (pt ou nt).
    Alinha por índice (usa comprimento mínimo comum).
    Retorna dict com time, <key>_mean, <key>_sem.
    """
    series = []
    times  = []
    for d in items:
        s = d.get(key, None)
        t = d.get("time", None)
        if s is None or t is None:
            continue
        s = np.asarray(s, dtype=float)
        t = np.asarray(t, dtype=int)
        n = min(len(s), len(t))
        if n > 0:
            series.append(s[:n])
            times.append(t[:n])

    if not series:
        return {"time": [], f"{key}_mean": [], f"{key}_sem": [], "n_seeds_series": 0}

    # comprimentos podem variar; usa o mínimo comum
    min_n = min(s.shape[0] for s in series)
    series = [s[:min_n] for s in series]
    times  = [tt[:min_n] for tt in times]

    # assume tempos coerentes; se não, escolhe a primeira sequência
    time_ref = times[0]
    mat = np.stack(series, axis=0)  # (S, T)
    mean = np.nanmean(mat, axis=0)
    if mat.shape[0] > 1:
        sem  = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
    else:
        sem  = np.zeros_like(mean)

    return {
        "time": time_ref.tolist(),
        f"{key}_mean": mean.astype(float).tolist(),
        f"{key}_sem":  sem.astype(float).tolist(),
        "n_seeds_series": int(mat.shape[0])
    }

def _mean_sem_1d(values: List[float]) -> Tuple[float, float, int]:
    """média e SEM de uma lista (ignora None/NaN)."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    if n == 1:
        return (float(arr.mean()), 0.0, 1)
    return (float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(n)), n)

def _average_by_order_new(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    lst: lista de amostras (seeds) para UMA ordem, cada uma com:
      {"t": array, "pt": array, ("nt": array)}

    Retorna um dict com:
      - "time" (lista)
      - "pt_mean" (lista)
      - "pt_sem" (lista)
      - se houver nt em todas: "nt_mean", "nt_sem"
    """
    if not lst:
        return {}

    # filtra entradas válidas
    valid = [d for d in lst if isinstance(d, dict) and "t" in d and "pt" in d]
    if not valid:
        return {}

    Tlist = [np.asarray(d["t"], dtype=float) for d in valid]
    Plist = [np.asarray(d["pt"], dtype=float) for d in valid]

    min_len = min(min(len(t) for t in Tlist), min(len(p) for p in Plist))
    if min_len < 2:
        return {}

    t_ref = Tlist[0][:min_len]

    P = np.stack([p[:min_len] for p in Plist], axis=0)  # (n_seeds, T)
    pt_mean = np.mean(P, axis=0)

    if P.shape[0] > 1:
        pt_sem = np.std(P, axis=0, ddof=1) / np.sqrt(P.shape[0])
    else:
        pt_sem = np.zeros(min_len, dtype=float)

    out: Dict[str, Any] = {
        "time": t_ref.tolist(),
        "pt_mean": pt_mean.tolist(),
        "pt_sem": pt_sem.tolist(),
    }

    # nt somente se existir em TODAS
    if all("nt" in d for d in valid):
        Nlist = [np.asarray(d["nt"], dtype=float) for d in valid]
        min_len_nt = min(min_len, min(len(n) for n in Nlist))
        if min_len_nt >= 2:
            N = np.stack([n[:min_len_nt] for n in Nlist], axis=0)
            nt_mean = np.mean(N, axis=0)
            if N.shape[0] > 1:
                nt_sem = np.std(N, axis=0, ddof=1) / np.sqrt(N.shape[0])
            else:
                nt_sem = np.zeros(min_len_nt, dtype=float)

            out["nt_mean"] = nt_mean.tolist()
            out["nt_sem"] = nt_sem.tolist()

    return out

def _sanitize_for_json(obj):
    import numpy as np
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, list):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    return str(obj)


def _load_orders_new(fp: str) -> Dict[int, Dict[str, Any]]:
    """
    Lê um JSON de amostra e retorna:
      {ordk: {"t": np.ndarray, "pt": np.ndarray, "nt": np.ndarray (se existir), ...}}
    Se o JSON estiver corrompido/vazio (JSONDecodeError), remove o arquivo e retorna {}.
    """
    try:
        with open(fp, "r", encoding="utf-8") as f:
            js = json.load(f)
    except json.JSONDecodeError:
        # arquivo inválido -> apaga e segue
        try:
            os.remove(fp)
            print(f"[warn] JSON inválido removido: {fp}")
        except OSError as ex:
            print(f"[warn] JSON inválido, falha ao remover: {fp} | {ex}")
        return {}
    except OSError as ex:
        print(f"[warn] Falha ao abrir arquivo: {fp} | {ex}")
        return {}

    results = js.get("results")
    if not isinstance(results, dict):
        return {}

    out: Dict[int, Dict[str, Any]] = {}

    for key, block in results.items():
        if not isinstance(key, str):
            continue
        m = ORDER_RE.search(key)
        if not m:
            continue

        ord1 = int(m.group(1))
        ordk = ord1 - 1

        data = (block or {}).get("data", None)
        if not isinstance(data, dict):
            continue

        t = data.get("time", data.get("t"))
        pt = data.get("pt")
        if t is None or pt is None:
            continue

        t = np.asarray(t, dtype=float)
        pt = np.asarray(pt, dtype=float)

        nt = data.get("nt", None)
        nt_arr = None
        if nt is not None:
            nt_arr = np.asarray(nt, dtype=float)

        nmin = min(len(t), len(pt))
        if nmin <= 0:
            continue
        t = t[:nmin]
        pt = pt[:nmin]
        if nt_arr is not None:
            nt_arr = nt_arr[:nmin]

        d: Dict[str, Any] = {"t": t, "pt": pt}
        if nt_arr is not None:
            d["nt"] = nt_arr

        # extras (se existirem)
        if "shortest_path_lin" in data:
            d["shortest_path_lin"] = data["shortest_path_lin"]
        if "M_size" in data:
            d["M_size"] = data["M_size"]

        out[ordk] = d

    return out

def _parse_fname(filepath: str) -> Optional[Tuple[float, float, int]]:
    """Extrai (P0, p0, seed) do nome do arquivo."""
    m = _FNAME_RE.search(os.path.basename(filepath))
    if not m:
        return None
    try:
        P0 = float(m.group("P0"))
        p0 = float(m.group("p0"))
        seed = int(m.group("seed"))
        return (P0, p0, seed)
    except Exception:
        return None

SEED_RE = re.compile(r"(?:^|_)seed_(\d+)(?:_|\.json$)")

def _parse_seed(fp: str) -> int | None:
    name = os.path.basename(fp)
    m = SEED_RE.search(name)
    if not m:
        return None
    return int(m.group(1))

ORDER_RE = re.compile(r"order_percolation\s*(\d+)\s*$")

def rolling_mean_std(t, y, window: int):
    """
    Retorna (t_centrado, media, desvio_padrao) para uma janela deslizante de tamanho 'window'.
    A série resultante fica centrada na janela.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if window < 1 or window > len(y):
        raise ValueError("window fora do intervalo válido")

    c  = np.cumsum(np.insert(y, 0, 0.0))
    c2 = np.cumsum(np.insert(y*y, 0, 0.0))

    mean = (c[window:] - c[:-window]) / window
    var  = (c2[window:] - c2[:-window]) / window - mean**2
    std  = np.sqrt(np.clip(var, 0, None))

    # centraliza no tempo (para janela par funciona como 'centered' padrão)
    t_center = t[(window-1)//2 : len(t) - window//2]
    return t_center, mean, std

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

def _concat_seed_series_for_pc(pt_by_seed: Dict[int, List[np.ndarray]],
                               t_by_seed: Dict[int, List[np.ndarray]],
                               x_max: float | None) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Concatena, por seed, as séries pt e t (já com x_max aplicado no loop de leitura),
    retornando {seed: {"t": t_concat, "pt": pt_concat}} apenas para seeds com dados.
    """
    out = {}
    for s, pts in pt_by_seed.items():
        if len(pts) == 0: 
            continue
        ts = t_by_seed.get(s, [])
        if len(ts) != len(pts):
            # segurança extra
            continue
        t_concat  = np.concatenate(ts)
        pt_concat = np.concatenate(pts)
        if t_concat.size > 0 and pt_concat.size == t_concat.size:
            out[s] = {"t": t_concat, "pt": pt_concat}
    return out

def _bootstrap_series_across_seeds(series_stack: np.ndarray,
                                   n_boot: int,
                                   rng_seed: int,
                                   batch_size: int = 128,
                                   use_float32: bool = True) -> (np.ndarray, np.ndarray):
    """
    Bootstrap ENTRE-SEEDS por ponto temporal, em streaming (sem alocar (n_boot, n_seeds, T)).
    - series_stack: shape (n_seeds, T)
    Retorna (mean_boot[T], std_boot[T]).
    """
    X = np.asarray(series_stack, dtype=np.float32 if use_float32 else np.float64)  # (n_seeds, T)
    n_seeds, T = X.shape
    rng = np.random.default_rng(rng_seed)

    # estatísticas online (Welford) ao longo das réplicas de bootstrap
    mean = np.zeros(T, dtype=np.float64)
    M2   = np.zeros(T, dtype=np.float64)
    count = 0

    # probabilidades uniformes para a Multinomial (bootstrap clássico com reposição)
    p = np.full(n_seeds, 1.0 / n_seeds, dtype=np.float64)

    # processa em lotes pequenos para não estourar RAM
    for start in range(0, n_boot, batch_size):
        b = min(batch_size, n_boot - start)
        # counts ~ Multinomial(n_seeds, p) para cada réplica (b, n_seeds)
        counts = rng.multinomial(n_seeds, pvals=p, size=b)   # int32
        # média bootstrap do tempo para cada réplica = (counts @ X) / n_seeds
        # (b, n_seeds) @ (n_seeds, T) -> (b, T)
        batch_means = (counts @ X) / float(n_seeds)

        # atualiza estatísticas online por réplica
        for i in range(b):
            count += 1
            delta = batch_means[i] - mean
            mean += delta / count
            M2   += delta * (batch_means[i] - mean)

        # libera rápido memória temporária
        del counts, batch_means

    std = np.sqrt(M2 / (count - 1)) if count > 1 else np.zeros_like(mean)
    return mean, std

def _bootstrap_tail_mean_across_seeds(series_stack: np.ndarray,
                                      t_center: np.ndarray,
                                      *,
                                      n_boot: int,
                                      rng_seed: int,
                                      tail_tmin: float | None = None,   # se dado, usa t >= tail_tmin
                                      tail_frac: float | None = 0.2,    # senão, usa última fração (ex.: 20%)
                                      batch_size: int = 128,
                                      use_float32: bool = True) -> tuple[float, float, int]:
    """
    Bootstrap ENTRE-SEEDS para a MÉDIA DA CAUDA (escalar).
    - series_stack: (n_seeds, T_comum) com as séries roladas por seed alinhadas
    - t_center: (T_comum,) tempos centrais já alinhados
    Retorna (mean_tail_boot, std_tail_boot, T_tail) onde T_tail é o nº de pontos usados na cauda.
    """
    X = np.asarray(series_stack, dtype=np.float32 if use_float32 else np.float64)
    t_center = np.asarray(t_center, dtype=np.float64)
    n_seeds, T = X.shape
    assert t_center.shape[0] == T

    # --- máscara de cauda ---
    if tail_tmin is not None:
        mask = (t_center >= float(tail_tmin))
    else:
        frac = 0.2 if tail_frac is None else float(tail_frac)
        frac = min(max(frac, 0.0), 1.0)
        start = int(round((1.0 - frac) * T))
        mask = np.zeros(T, dtype=bool)
        mask[start:] = True

    T_tail = int(mask.sum())
    if T_tail == 0:
        return float('nan'), float('nan'), 0

    rng = np.random.default_rng(rng_seed)
    p = np.full(n_seeds, 1.0 / n_seeds, dtype=np.float64)

    # estatísticas online (escalar por réplica)
    mean = 0.0
    M2 = 0.0
    count = 0

    for start in range(0, n_boot, batch_size):
        b = min(batch_size, n_boot - start)
        counts = rng.multinomial(n_seeds, pvals=p, size=b)  # (b, n_seeds)

        # (b, n_seeds) @ (n_seeds, T) -> (b, T); média por seed
        batch_means = (counts @ X) / float(n_seeds)

        # média temporal RESTRITA à cauda -> vetor (b,)
        tail_vals = batch_means[:, mask].mean(axis=1)

        # Welford online
        for v in tail_vals:
            count += 1
            delta = v - mean
            mean += delta / count
            M2   += delta * (v - mean)

        del counts, batch_means, tail_vals

    std = np.sqrt(M2 / (count - 1)) if count > 1 else 0.0
    return float(mean), float(std), T_tail

def _safe_float(x):
    if x is None:
        return None
    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x):
            return None
    return float(x)

def _init_running_stats(T: int, with_nt: bool):
    st = {
        "n": 0,
        "sum_pt": np.zeros(T, dtype=np.float64),
        "sum_pt2": np.zeros(T, dtype=np.float64),
    }
    if with_nt:
        st["sum_nt"] = np.zeros(T, dtype=np.float64)
        st["sum_nt2"] = np.zeros(T, dtype=np.float64)
        st["with_nt"] = True
    else:
        st["with_nt"] = False
    return st
def _truncate_running_stats(st: Dict[str, Any], T: int):
    st["sum_pt"] = st["sum_pt"][:T]
    st["sum_pt2"] = st["sum_pt2"][:T]
    if st.get("with_nt", False):
        st["sum_nt"] = st["sum_nt"][:T]
        st["sum_nt2"] = st["sum_nt2"][:T]

def _update_running_stats(st: Dict[str, Any], pt: np.ndarray, nt: Optional[np.ndarray]):
    st["n"] += 1
    st["sum_pt"] += pt
    st["sum_pt2"] += pt * pt
    if st.get("with_nt", False) and nt is not None:
        st["sum_nt"] += nt
        st["sum_nt2"] += nt * nt


def _finalize_running_stats(t_ref: np.ndarray, st: Dict[str, Any]) -> Dict[str, Any]:
    n = int(st["n"])
    if n <= 0:
        return {}

    pt_mean = st["sum_pt"] / n
    if n > 1:
        pt_var = (st["sum_pt2"] - (st["sum_pt"] * st["sum_pt"]) / n) / (n - 1)
        pt_var = np.maximum(pt_var, 0.0)
        pt_sem = np.sqrt(pt_var) / np.sqrt(n)
    else:
        pt_sem = np.zeros_like(pt_mean)

    out: Dict[str, Any] = {
        "time": t_ref.tolist(),
        "pt_mean": pt_mean.tolist(),
        "pt_sem": pt_sem.tolist(),
    }

    if st.get("with_nt", False):
        nt_mean = st["sum_nt"] / n
        if n > 1:
            nt_var = (st["sum_nt2"] - (st["sum_nt"] * st["sum_nt"]) / n) / (n - 1)
            nt_var = np.maximum(nt_var, 0.0)
            nt_sem = np.sqrt(nt_var) / np.sqrt(n)
        else:
            nt_sem = np.zeros_like(nt_mean)

        out["nt_mean"] = nt_mean.tolist()
        out["nt_sem"] = nt_sem.tolist()

    return out

def compute_means_for_folder_new(
    type_perc: str,
    num_colors: int,
    dim: int,
    L: int,
    NT: int,
    k: float,
    rho: float,
    p0_list: List[float],
    *,
    x_max: float | None = None,
    n_boot: int = 20000,          # mantido (usado em roll, se ativado)
    rng_seed: int = 12345,
    window_roll: int | None = None,
    clear_data: bool = False,
) -> str:
    """
    Versão streaming: não acumula séries por seed/ordem na RAM.
    Mantém estrutura do properties_mean_bundle.json.
    """
    base_dir = "../Data"
    data_dir = os.path.join(
        base_dir,
        f"{type_perc}_percolation",
        f"num_colors_{num_colors}",
        f"dim_{dim}",
        f"L_{L}",
        "NT_constant",
        f"NT_{NT}",
        f"k_{k:.1e}",
        f"rho_{rho:.4e}",
        "data",
    )
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Pasta de dados não encontrada: {data_dir}")

    out_dir = os.path.dirname(data_dir)
    out_path = os.path.join(out_dir, "properties_mean_bundle.json")

    if os.path.isfile(out_path) and clear_data:
        os.remove(out_path)

    bundle: Dict[str, Any] = {
        "meta": {
            "type_perc": type_perc,
            "num_colors": num_colors,
            "dim": dim,
            "L": L,
            "NT": NT,
            "k": float(k),
            "rho": float(rho),
            "base_dir": os.path.dirname(data_dir),
            "x_max_used": None if x_max is None else float(x_max),
            "bootstrap": {"n_boot": int(n_boot), "rng_seed": int(rng_seed)},
            "rolling": {"window": None if window_roll is None else int(window_roll)}
        },
        "p0_groups": []
    }

    # ----------------------------------------------------------
    # loop por p0
    # ----------------------------------------------------------
    for p0 in p0_list:
        pattern = os.path.join(data_dir, f"P0_*_p0_{p0:.2f}_seed_*.json")
        files = sorted(glob.glob(pattern)) or sorted(
            glob.glob(os.path.join(data_dir, f"P0_*_p0_{p0:.1f}_seed_*.json"))
        )

        if not files:
            print(f"[aviso] Sem arquivos para p0={p0:.2f} em {data_dir}")
            continue

        # streaming stats por ordem
        t_ref_by_order: Dict[int, np.ndarray] = {}
        stats_by_order: Dict[int, Dict[str, Any]] = {}

        # contadores
        seeds_set = set()
        n_files_ok = 0

        # (opcional) para pc_sop_roll — pode ser pesado, mas não explode como per_order
        pt_by_seed = defaultdict(list) if window_roll is not None else None
        t_by_seed  = defaultdict(list) if window_roll is not None else None

        for fp in files:
            # carrega ordens (deve retornar arrays)
            orders = _load_orders_new(fp)
            if not orders:
                # arquivo inválido/sem dados
                continue

            n_files_ok += 1

            # seed só se for usar roll / estatística de seeds
            seed = _parse_seed(fp)
            if seed is not None:
                seeds_set.add(seed)

            for ordk, data in orders.items():
                # espera data com "t" e "pt"
                if "t" not in data or "pt" not in data:
                    continue

                t = np.asarray(data["t"], dtype=float)
                pt = np.asarray(data["pt"], dtype=float)
                nt = np.asarray(data["nt"], dtype=float) if "nt" in data and data["nt"] is not None else None

                if x_max is not None:
                    msk = (t <= x_max)
                    t = t[msk]
                    pt = pt[msk]
                    if nt is not None:
                        nt = nt[msk]

                if pt.size < 2:
                    continue

                # inicializa referência do tempo/stats na primeira vez
                if ordk not in t_ref_by_order:
                    t_ref_by_order[ordk] = t.copy()
                    stats_by_order[ordk] = _init_running_stats(len(t), with_nt=(nt is not None))
                else:
                    # alinha no menor tamanho comum
                    T = min(len(t_ref_by_order[ordk]), len(t))
                    if T < 2:
                        continue
                    t_ref_by_order[ordk] = t_ref_by_order[ordk][:T]
                    _truncate_running_stats(stats_by_order[ordk], T)
                    pt = pt[:T]
                    if nt is not None:
                        nt = nt[:T]

                _update_running_stats(stats_by_order[ordk], pt, nt)

                # para roll, acumula apenas se seed foi parseada
                if window_roll is not None and seed is not None and pt_by_seed is not None:
                    # aqui guardamos pt/t por seed; se ficar pesado, rode window_roll=None em batch
                    pt_by_seed[seed].append(pt.astype(np.float32))
                    t_by_seed[seed].append(t_ref_by_order[ordk].astype(np.float32))

        # finaliza médias por ordem
        mean_by_order: Dict[int, Dict[str, Any]] = {}
        for ordk in sorted(stats_by_order.keys()):
            mean_by_order[ordk] = _finalize_running_stats(t_ref_by_order[ordk], stats_by_order[ordk])

            # pt_mean_roll por ordem (barato, usa só série média)
            if window_roll is not None and mean_by_order[ordk]:
                d = mean_by_order[ordk]
                if "time" in d and "pt_mean" in d and d["time"] and d["pt_mean"]:
                    t_ord = np.asarray(d["time"], dtype=float)
                    pt_ord_mean = np.asarray(d["pt_mean"], dtype=float)
                    if len(pt_ord_mean) >= window_roll:
                        t_c, mu_r, sd_r = rolling_mean_std(t_ord, pt_ord_mean, window_roll)
                        mean_by_order[ordk]["pt_mean_roll"] = {
                            "t_center": t_c.tolist(),
                            "mean":     mu_r.tolist(),
                            "std":      sd_r.tolist(),
                            "window":   int(window_roll)
                        }

        orders_blocks = [
            {"order_percolation": int(ordk), "data": mean_by_order[ordk]}
            for ordk in sorted(mean_by_order.keys())
        ]

        # ==========================================================
        # pc_sop (método t0_global + weighted_mean_and_sem)
        # ==========================================================
        series = []
        for ordk in sorted(mean_by_order.keys()):
            d = mean_by_order[ordk]
            if not d:
                continue

            kt, kpt, ksem = _get_keys_for_series(d)
            if kt not in d or kpt not in d or ksem not in d:
                continue

            t = np.asarray(d[kt], dtype=float)
            pt = np.asarray(d[kpt], dtype=float)
            pt_sem = np.asarray(d[ksem], dtype=float)

            if t.size < 2:
                continue
            series.append((t, pt, pt_sem))

        t0_ind = []
        for (t, pt, pt_sem) in series:
            idx0_i = detect_equilibrium_start_with_errors(
                t, pt, pt_sem, w=40, consec=6, z=2.0, chi2r_max=2.0
            )
            idx0_i = int(np.clip(idx0_i, 0, max(len(t) - 1, 0)))
            t0_ind.append(float(t[idx0_i]))

        t0_global = float(max(t0_ind)) if len(t0_ind) > 0 else float("nan")

        mean_eq_list = []
        sem_eq_list = []
        for (t, pt, pt_sem) in series:
            idx0_g = idx_from_t0(t, t0_global)
            if idx0_g >= len(pt):
                continue
            mean_eq, sem_eq = weighted_mean_and_sem(pt[idx0_g:], pt_sem[idx0_g:])
            mean_eq_list.append(mean_eq)
            sem_eq_list.append(sem_eq)

        if len(mean_eq_list) == 0:
            pc_mean = float("nan")
            pc_sem = float("nan")
        else:
            pc_mean, pc_sem = weighted_mean_and_sem(mean_eq_list, sem_eq_list)

        # ==========================================================
        # pc_sop_roll (opcional) — mantém sua lógica original
        # ==========================================================
        pc_sop_roll_block = None
        if window_roll is not None and pt_by_seed is not None and t_by_seed is not None:
            seed_series = _concat_seed_series_for_pc(pt_by_seed, t_by_seed, x_max)

            rolled_means = []
            rolled_times = []
            for s in sorted(seed_series.keys()):
                t_concat  = seed_series[s]["t"]
                pt_concat = seed_series[s]["pt"]
                if len(pt_concat) >= window_roll:
                    t_c, mu_r, _ = rolling_mean_std(t_concat, pt_concat, window_roll)
                    rolled_means.append(mu_r.astype(np.float32))
                    rolled_times.append(t_c.astype(np.float32))

            if len(rolled_means) > 0:
                min_len = min(len(x) for x in rolled_means)
                if min_len >= 1:
                    M = np.stack([m[:min_len] for m in rolled_means], axis=0).astype(np.float32)
                    Tstack   = np.stack([tc[:min_len] for tc in rolled_times], axis=0)
                    t_center = np.median(Tstack, axis=0)

                    mean_boot, std_boot_series = _bootstrap_series_across_seeds(
                        M, n_boot=n_boot, rng_seed=rng_seed, batch_size=128, use_float32=True
                    )

                    mean_tail, std_tail, T_tail = _bootstrap_tail_mean_across_seeds(
                        M, t_center,
                        n_boot=n_boot, rng_seed=rng_seed,
                        tail_tmin=None,
                        tail_frac=0.2,
                        batch_size=128, use_float32=True
                    )

                    pc_sop_roll_block = {
                        "t_center": t_center.tolist(),
                        "mean":     mean_boot.tolist(),
                        "std_boot": std_boot_series.tolist(),
                        "n_seeds":  int(M.shape[0]),
                        "n_boot":   int(n_boot),
                        "window":   int(window_roll),
                        "tail_summary": {
                            "mode": "frac",
                            "tail_frac": 0.2,
                            "T_used": int(T_tail),
                            "mean": _safe_float(mean_tail),
                            "std_boot": _safe_float(std_tail)
                        }
                    }

        # ----------------------------------------------------------
        # monta p0_group (estrutura idêntica)
        # ----------------------------------------------------------
        p0_group = {
            "p0_value": float(p0),
            "num_seeds": int(len(seeds_set)),            # seeds parseadas no nome
            "seeds": sorted(seeds_set),
            "orders": orders_blocks,
            "pc_sop": {
                "mean": _safe_float(pc_mean),
                "std_boot": _safe_float(pc_sem),
                "n_seeds": int(len(seeds_set)) if len(seeds_set) > 0 else int(n_files_ok),
                "n_boot": int(n_boot),
            }
        }
        if pc_sop_roll_block is not None:
            p0_group["pc_sop_roll"] = pc_sop_roll_block

        bundle["p0_groups"].append(p0_group)

        msg_roll = ""
        if pc_sop_roll_block is not None:
            msg_roll = (
                f" | pc_SOP^roll: T={len(pc_sop_roll_block['t_center'])}, "
                f"seeds_valid={pc_sop_roll_block['n_seeds']}, win={pc_sop_roll_block['window']}"
            )

        print(
            f"[ok] p0={p0:.2f}: {len(files)} arquivos | files_ok={n_files_ok} | "
            f"seeds={len(seeds_set)} | pc_SOP={pc_mean:.6f}±{pc_sem:.6f} | t0_global={t0_global:.3f}{msg_roll}"
        )

        # limpeza agressiva por p0 (ajuda em Jupyter)
        del t_ref_by_order, stats_by_order, mean_by_order, orders_blocks, series
        if pt_by_seed is not None:
            pt_by_seed.clear()
        if t_by_seed is not None:
            t_by_seed.clear()
        gc.collect()

    # sanitize UMA vez e salva com JSON válido
    bundle = _sanitize_for_json(bundle)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"[salvo] {out_path}")
    return out_path