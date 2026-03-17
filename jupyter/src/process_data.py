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
from tqdm import tqdm
from IPython.display import clear_output

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

def _sem_scalar(vals: List[float]) -> float:
    a = np.asarray(vals, dtype=float)
    if a.size <= 1:
        return 0.0
    return float(np.std(a, ddof=1) / np.sqrt(a.size))

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



_SEED_RE = re.compile(r"_seed_(\d+)\.json$")

def _extract_seed_from_filename(fp: str) -> int | None:
    m = _SEED_RE.search(os.path.basename(fp))
    if not m:
        return None
    return int(m.group(1))

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


def process_all_roots(base_root="../SOP_data/raw/bond_percolation", verbose=True, clean_outputs=False):
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

# ============================================================
# NOVO helper: SEM robusto
# ============================================================
def _sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(x.size))

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
def summarize_all_dirs(base_root: str = "../SOP_data/raw/bond_percolation",
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

import numpy as np
from typing import Any, Dict, List

# def _average_by_order_new(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     Faz média/STD/SEM entre seeds para uma ordem.
#     Corrige o problema de shapes diferentes truncando todas as séries
#     para o menor comprimento comum entre seeds válidas.

#     Agora também salva:
#       - pt_std (dispersão entre seeds)
#       - nt_std (dispersão entre seeds)
#     """

#     series_pt = []
#     series_nt = []
#     spl_vals: List[float] = []
#     msz_vals: List[float] = []

#     for d in lst:
#         t = d.get("t", None)
#         pt = d.get("pt", None)
#         nt = d.get("nt", None)

#         if t is None or pt is None:
#             continue

#         t = np.asarray(t, dtype=float)
#         pt = np.asarray(pt, dtype=float)

#         n_pt = min(len(t), len(pt))
#         if n_pt <= 0:
#             continue
#         series_pt.append((t[:n_pt], pt[:n_pt]))

#         if nt is not None:
#             nt = np.asarray(nt, dtype=float)
#             n_nt = min(len(t), len(nt), n_pt)
#             if n_nt > 0:
#                 series_nt.append(nt[:n_nt])

#         spl = d.get("shortest_path_lin", None)
#         if spl is not None:
#             try:
#                 spl_vals.append(float(spl))
#             except Exception:
#                 pass

#         msz = d.get("M_size", None)
#         if msz is not None:
#             try:
#                 msz_vals.append(float(msz))
#             except Exception:
#                 pass

#     if not series_pt:
#         return {
#             "time": [],
#             "pt_mean": [],
#             "pt_std": [],
#             "pt_sem": [],
#             "nt_mean": [],
#             "nt_std": [],
#             "nt_sem": [],
#             "n_seeds_pt": 0,
#             "n_seeds_nt": 0,
#         }

#     min_len_pt = min(len(pt) for (_, pt) in series_pt)
#     t_common = series_pt[0][0][:min_len_pt]
#     pts = np.stack([pt[:min_len_pt] for (_, pt) in series_pt], axis=0)  # (nseed, T)

#     nseed_pt = int(pts.shape[0])
#     pt_mean = np.mean(pts, axis=0)
#     if nseed_pt > 1:
#         pt_std = np.std(pts, axis=0, ddof=1)
#         pt_sem = pt_std / np.sqrt(nseed_pt)
#     else:
#         pt_std = np.zeros_like(pt_mean)
#         pt_sem = np.zeros_like(pt_mean)

#     out: Dict[str, Any] = {}
#     out["time"] = t_common.tolist()
#     out["pt_mean"] = pt_mean.tolist()
#     out["pt_std"]  = pt_std.tolist()
#     out["pt_sem"]  = pt_sem.tolist()
#     out["n_seeds_pt"] = nseed_pt

#     # nt
#     if series_nt:
#         min_len_nt = min(len(nt) for nt in series_nt)
#         min_len = min(min_len_pt, min_len_nt)

#         nts = np.stack([nt[:min_len] for nt in series_nt], axis=0)
#         nseed_nt = int(nts.shape[0])

#         nt_mean = np.mean(nts, axis=0)
#         if nseed_nt > 1:
#             nt_std = np.std(nts, axis=0, ddof=1)
#             nt_sem = nt_std / np.sqrt(nseed_nt)
#         else:
#             nt_std = np.zeros_like(nt_mean)
#             nt_sem = np.zeros_like(nt_mean)

#         # re-trunca pt para bater com nt
#         pt_mean2 = np.mean(pts[:, :min_len], axis=0)
#         if nseed_pt > 1:
#             pt_std2 = np.std(pts[:, :min_len], axis=0, ddof=1)
#             pt_sem2 = pt_std2 / np.sqrt(nseed_pt)
#         else:
#             pt_std2 = np.zeros_like(pt_mean2)
#             pt_sem2 = np.zeros_like(pt_mean2)

#         out["time"] = t_common[:min_len].tolist()
#         out["pt_mean"] = pt_mean2.tolist()
#         out["pt_std"]  = pt_std2.tolist()
#         out["pt_sem"]  = pt_sem2.tolist()

#         out["nt_mean"] = nt_mean.tolist()
#         out["nt_std"]  = nt_std.tolist()
#         out["nt_sem"]  = nt_sem.tolist()
#         out["n_seeds_nt"] = nseed_nt
#     else:
#         out["nt_mean"] = []
#         out["nt_std"]  = []
#         out["nt_sem"]  = []
#         out["n_seeds_nt"] = 0

#     # escalares (mantive SEM; se quiser STD também, dá pra adicionar)
#     if spl_vals:
#         a = np.asarray(spl_vals, dtype=float)
#         out["shortest_path_lin_mean"] = float(np.mean(a))
#         out["shortest_path_lin_sem"]  = float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
#         out["n_seeds_shortest_path_lin"] = int(a.size)

#     if msz_vals:
#         a = np.asarray(msz_vals, dtype=float)
#         out["M_size_mean"] = float(np.mean(a))
#         out["M_size_sem"]  = float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
#         out["n_seeds_M_size"] = int(a.size)

#     return out

def _average_by_order_new(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Média/STD/SEM entre seeds para uma ordem.

    Agora salva:
      - pt_std: desvio padrão entre seeds (dispersão)
      - pt_sem: erro padrão da média entre seeds (precisão da média)
      - nt_std, nt_sem análogos
    """
    series_pt = []
    series_nt = []
    spl_vals: List[float] = []
    msz_vals: List[float] = []

    for d in lst:
        t = d.get("t", None)
        pt = d.get("pt", None)
        nt = d.get("nt", None)

        if t is None or pt is None:
            continue

        t = np.asarray(t, dtype=float)
        pt = np.asarray(pt, dtype=float)

        n_pt = min(len(t), len(pt))
        if n_pt <= 1:
            continue

        series_pt.append((t[:n_pt], pt[:n_pt]))

        if nt is not None:
            nt = np.asarray(nt, dtype=float)
            n_nt = min(len(t), len(nt), n_pt)
            if n_nt > 1:
                series_nt.append(nt[:n_nt])

        spl = d.get("shortest_path_lin", None)
        if spl is not None:
            try:
                spl_vals.append(float(spl))
            except Exception:
                pass

        msz = d.get("M_size", None)
        if msz is not None:
            try:
                msz_vals.append(float(msz))
            except Exception:
                pass

    if not series_pt:
        return {
            "time": [],
            "pt_mean": [],
            "pt_std": [],
            "pt_sem": [],
            "nt_mean": [],
            "nt_std": [],
            "nt_sem": [],
            "n_seeds_pt": 0,
            "n_seeds_nt": 0,
        }

    min_len_pt = min(len(pt) for (_, pt) in series_pt)
    t_common = series_pt[0][0][:min_len_pt]
    pts = np.stack([pt[:min_len_pt] for (_, pt) in series_pt], axis=0)  # (nseed, T)

    nseed_pt = int(pts.shape[0])
    pt_mean = np.mean(pts, axis=0)
    if nseed_pt > 1:
        pt_std = np.std(pts, axis=0, ddof=1)
        pt_sem = pt_std / np.sqrt(nseed_pt)
    else:
        pt_std = np.zeros_like(pt_mean)
        pt_sem = np.zeros_like(pt_mean)

    out: Dict[str, Any] = {}
    out["time"] = t_common.tolist()
    out["pt_mean"] = pt_mean.tolist()
    out["pt_std"] = pt_std.tolist()
    out["pt_sem"] = pt_sem.tolist()
    out["n_seeds_pt"] = nseed_pt

    # nt
    if series_nt:
        min_len_nt = min(len(nt) for nt in series_nt)
        min_len = min(min_len_pt, min_len_nt)

        nts = np.stack([nt[:min_len] for nt in series_nt], axis=0)
        nseed_nt = int(nts.shape[0])

        nt_mean = np.mean(nts, axis=0)
        if nseed_nt > 1:
            nt_std = np.std(nts, axis=0, ddof=1)
            nt_sem = nt_std / np.sqrt(nseed_nt)
        else:
            nt_std = np.zeros_like(nt_mean)
            nt_sem = np.zeros_like(nt_mean)

        # re-trunca pt para bater com nt
        pt_mean2 = np.mean(pts[:, :min_len], axis=0)
        if nseed_pt > 1:
            pt_std2 = np.std(pts[:, :min_len], axis=0, ddof=1)
            pt_sem2 = pt_std2 / np.sqrt(nseed_pt)
        else:
            pt_std2 = np.zeros_like(pt_mean2)
            pt_sem2 = np.zeros_like(pt_mean2)

        out["time"] = t_common[:min_len].tolist()
        out["pt_mean"] = pt_mean2.tolist()
        out["pt_std"] = pt_std2.tolist()
        out["pt_sem"] = pt_sem2.tolist()

        out["nt_mean"] = nt_mean.tolist()
        out["nt_std"] = nt_std.tolist()
        out["nt_sem"] = nt_sem.tolist()
        out["n_seeds_nt"] = nseed_nt
    else:
        out["nt_mean"] = []
        out["nt_std"] = []
        out["nt_sem"] = []
        out["n_seeds_nt"] = 0

    # escalares (mantive SEM; se quiser STD também, dá pra adicionar)
    if spl_vals:
        a = np.asarray(spl_vals, dtype=float)
        out["shortest_path_lin_mean"] = float(np.mean(a))
        out["shortest_path_lin_sem"] = float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
        out["n_seeds_shortest_path_lin"] = int(a.size)

    if msz_vals:
        a = np.asarray(msz_vals, dtype=float)
        out["M_size_mean"] = float(np.mean(a))
        out["M_size_sem"] = float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
        out["n_seeds_M_size"] = int(a.size)

    return out

def _sanitize_for_json(obj: Any) -> Any:
    """Recursivo: remove NaN/inf, converte numpy types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (not np.isfinite(v)) else v
    if isinstance(obj, float):
        return None if (not np.isfinite(obj)) else obj
    return obj


# ----------------------------
def _load_orders_new(fp: str) -> dict[int, dict]:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            js = json.load(f)
    except json.JSONDecodeError:
        try:
            os.remove(fp)
            print(f"[warn] JSON inválido removido: {fp}")
        except OSError as ex:
            print(f"[warn] JSON inválido, falha ao remover: {fp} | {ex}")
        return {}
    except OSError as ex:
        print(f"[warn] Falha ao abrir {fp}: {ex}")
        return {}

    results = js.get("results", {})
    if not isinstance(results, dict):
        return {}

    out: dict[int, dict] = {}
    for key, block in results.items():
        if not isinstance(key, str) or "order_percolation" not in key:
            continue

        digits = "".join(ch for ch in key if ch.isdigit())
        if not digits:
            continue

        ord1 = int(digits)          # 1-based
        ordk = ord1 - 1             # 0-based

        data = (block or {}).get("data", {})
        if not isinstance(data, dict):
            continue

        t = data.get("time", data.get("t", None))
        pt = data.get("pt", None)
        nt = data.get("nt", None)

        if t is not None:
            t = np.asarray(t, dtype=float)
        if pt is not None:
            pt = np.asarray(pt, dtype=float)
        if nt is not None:
            nt = np.asarray(nt, dtype=float)

        out[ordk] = {
            "t": t,
            "pt": pt,
            "nt": nt,
            "shortest_path_lin": data.get("shortest_path_lin", None),
            "M_size": data.get("M_size", None),
        }

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

# def _bootstrap_series_across_seeds(series_stack: np.ndarray,
#                                    n_boot: int,
#                                    rng_seed: int,
#                                    batch_size: int = 128,
#                                    use_float32: bool = True) -> (np.ndarray, np.ndarray):
#     """
#     Bootstrap ENTRE-SEEDS por ponto temporal, em streaming (sem alocar (n_boot, n_seeds, T)).
#     - series_stack: shape (n_seeds, T)
#     Retorna (mean_boot[T], std_boot[T]).
#     """
#     X = np.asarray(series_stack, dtype=np.float32 if use_float32 else np.float64)  # (n_seeds, T)
#     n_seeds, T = X.shape
#     rng = np.random.default_rng(rng_seed)

#     # estatísticas online (Welford) ao longo das réplicas de bootstrap
#     mean = np.zeros(T, dtype=np.float64)
#     M2   = np.zeros(T, dtype=np.float64)
#     count = 0

#     # probabilidades uniformes para a Multinomial (bootstrap clássico com reposição)
#     p = np.full(n_seeds, 1.0 / n_seeds, dtype=np.float64)

#     # processa em lotes pequenos para não estourar RAM
#     for start in range(0, n_boot, batch_size):
#         b = min(batch_size, n_boot - start)
#         # counts ~ Multinomial(n_seeds, p) para cada réplica (b, n_seeds)
#         counts = rng.multinomial(n_seeds, pvals=p, size=b)   # int32
#         # média bootstrap do tempo para cada réplica = (counts @ X) / n_seeds
#         # (b, n_seeds) @ (n_seeds, T) -> (b, T)
#         batch_means = (counts @ X) / float(n_seeds)

#         # atualiza estatísticas online por réplica
#         for i in range(b):
#             count += 1
#             delta = batch_means[i] - mean
#             mean += delta / count
#             M2   += delta * (batch_means[i] - mean)

#         # libera rápido memória temporária
#         del counts, batch_means

#     std = np.sqrt(M2 / (count - 1)) if count > 1 else np.zeros_like(mean)
#     return mean, std



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

def _safe_float(x: Any) -> Any:
    """Converte float/np.float em float Python; transforma NaN/inf em None p/ JSON."""
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v

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


def _list_seed_files_in_data_dir(data_dir: str, p0_list: List[float]) -> List[str]:
    files_all: List[str] = []
    for p0 in p0_list:
        pat = os.path.join(data_dir, f"P0_*_p0_{p0:.2f}_seed_*.json")
        files_all.extend(sorted(glob.glob(pat)))
    return sorted({os.path.basename(fp) for fp in files_all})

# def compute_means_for_folder(
#     type_perc: str,
#     num_colors: int,
#     dim: int,
#     L: int,
#     NT: int,
#     k: float,
#     rho: float,
#     p0_list: List[float],
#     *,
#     base_dir: str,
#     x_max: float | None = None,
#     n_boot: int = 20000,
#     rng_seed: int = 12345,
#     window_roll: int | None = None,
#     clear_data: bool = False,
# ) -> str:
#     """
#     Versão AJUSTADA para o seu problema de erros do pt:

#     - O que estava acontecendo: _average_by_order_new devolve pt_sem como SEM da MÉDIA
#       ao longo das seeds (std / sqrt(n_seeds)). Se você tem muitas seeds, isso fica
#       naturalmente MUITO pequeno.

#     - O ajuste aqui: preservamos o SEM original em `pt_sem_mean` e passamos a salvar
#       em `pt_sem` o DESVIO-PADRÃO entre seeds (pt_std). Assim:
#         pt_std = pt_sem_mean * sqrt(n_seeds)
#       e `pt_sem` fica na escala das flutuações amostrais entre seeds (bem maior).

#     - Além disso, toda a detecção de equilíbrio e médias ponderadas passam a usar esse
#       `pt_sem` (agora interpretado como pt_std), evitando pesos absurdos.
#     """
#     base_dir = os.path.abspath(base_dir)

#     data_dir = os.path.join(
#         base_dir,
#         f"{type_perc}_percolation",
#         f"num_colors_{num_colors}",
#         f"dim_{dim}",
#         f"L_{L}",
#         "NT_constant",
#         f"NT_{NT}",
#         f"k_{k:.1e}",
#         f"rho_{rho:.4e}",
#         "data",
#     )
#     if not os.path.isdir(data_dir):
#         raise FileNotFoundError(f"Pasta de dados não encontrada: {data_dir}")

#     out_dir = os.path.dirname(data_dir)
#     out_path = os.path.join(out_dir, "properties_mean_bundle.json")

#     if clear_data and os.path.isfile(out_path):
#         os.remove(out_path)
#         print(f"[clear_data] removido: {out_path}")

#     # incremental: se existe e não tem arquivo novo, skip
#     current_seed_files = []
#     for p0 in p0_list:
#         current_seed_files += sorted(
#             glob.glob(os.path.join(data_dir, f"P0_*_p0_{p0:.2f}_seed_*.json"))
#         )
#     current_seed_files = sorted({os.path.basename(x) for x in current_seed_files})

#     if os.path.isfile(out_path) and (not clear_data):
#         try:
#             with open(out_path, "r", encoding="utf-8") as f:
#                 old = json.load(f)
#             old_used = old.get("meta", {}).get("seed_used", [])
#             old_used = old_used if isinstance(old_used, list) else []
#             if set(current_seed_files).issubset(set(map(str, old_used))):
#                 print(f"[skip] atualizado: {out_path}")
#                 return out_path
#         except Exception as e:
#             print(f"[recalc] falha lendo bundle antigo ({e}) -> recalculando")

#     seed_used_set = set()

#     bundle: Dict[str, Any] = {
#         "meta": {
#             "type_perc": type_perc,
#             "num_colors": num_colors,
#             "dim": dim,
#             "L": L,
#             "NT": NT,
#             "k": float(k),
#             "rho": float(rho),
#             "base_dir": out_dir,
#             "x_max_used": None if x_max is None else float(x_max),
#             "bootstrap": {"n_boot": int(n_boot), "rng_seed": int(rng_seed)},
#             "rolling": {"window": None if window_roll is None else int(window_roll)},
#             "seed_used": [],  # <<< AQUI (meta)
#         },
#         "p0_groups": [],
#     }

#     for p0 in p0_list:
#         files = sorted(glob.glob(os.path.join(data_dir, f"P0_*_p0_{p0:.2f}_seed_*.json")))
#         print(f"[debug] data_dir={data_dir} | p0={p0:.2f} | files={len(files)}")

#         if not files:
#             print(f"[aviso] Sem arquivos para p0={p0:.2f}")
#             continue

#         per_order: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
#         seeds_set: set[int] = set()
#         valid_files = 0

#         for fp in files:
#             orders = _load_orders_new(fp)
#             if not orders:
#                 continue

#             valid_files += 1

#             seed = _extract_seed_from_filename(fp)
#             if seed is not None:
#                 seeds_set.add(seed)

#             bn = os.path.basename(fp)
#             seed_used_set.add(bn)

#             for ordk, data in orders.items():
#                 # corta por x_max, se usado
#                 if x_max is not None and data.get("t") is not None:
#                     t = np.asarray(data["t"], dtype=float)
#                     m = (t <= x_max)
#                     data["t"] = t[m]
#                     if data.get("pt") is not None:
#                         data["pt"] = np.asarray(data["pt"], dtype=float)[m]
#                     if data.get("nt") is not None:
#                         data["nt"] = np.asarray(data["nt"], dtype=float)[m]

#                 per_order[ordk].append(data)

#         if valid_files == 0:
#             print(f"[warn] p0={p0:.2f}: nenhum arquivo válido lido (JSON vazio/ruim?) -> pulando grupo")
#             continue

#         # ====== médias por ordem ======
#         mean_by_order: Dict[int, Dict[str, Any]] = {}
#         for ordk, lst in per_order.items():
#             mean_by_order[ordk] = _average_by_order_new(lst)

#             # -------- AJUSTE CRÍTICO DOS ERROS DO pt --------
#             d = mean_by_order[ordk]

#             # _average_by_order_new retorna pt_sem como SEM da média entre seeds
#             # aqui vamos:
#             #   - guardar o SEM original em pt_sem_mean
#             #   - transformar pt_sem em desvio-padrão entre seeds (pt_std)
#             if d.get("pt_sem") is not None:
#                 sem_mean = np.asarray(d["pt_sem"], dtype=float)

#                 n_seeds_pt = int(d.get("n_seeds_pt", 0) or 0)
#                 if n_seeds_pt > 1:
#                     pt_std = sem_mean * np.sqrt(n_seeds_pt)
#                 else:
#                     pt_std = sem_mean * 0.0  # sem seeds suficientes -> std=0

#                 # preserva SEM original
#                 d["pt_sem_mean"] = sem_mean.astype(float).tolist()
#                 # substitui pt_sem pelo std entre seeds (o que você quer como "erro" no pt_mean)
#                 d["pt_sem"] = pt_std.astype(float).tolist()
#                 # opcional: também salvar explicitamente
#                 d["pt_std"] = pt_std.astype(float).tolist()
#             # -----------------------------------------------

#         # ====== pc_sop global e por ordem ======
#         series = []
#         for ordk in sorted(mean_by_order.keys()):
#             d = mean_by_order[ordk]
#             if not d.get("time") or not d.get("pt_mean") or not d.get("pt_sem"):
#                 continue
#             t = np.asarray(d["time"], dtype=float)
#             pt = np.asarray(d["pt_mean"], dtype=float)

#             # agora pt_sem = pt_std (entre seeds), após o ajuste acima
#             pt_sem = np.asarray(d["pt_sem"], dtype=float)

#             if t.size > 0:
#                 series.append((ordk, t, pt, pt_sem))

#         t0_list = []
#         for (_, t, pt, pt_sem) in series:
#             idx0 = detect_equilibrium_start_with_errors(
#                 t, pt, pt_sem, w=40, consec=6, z=2.0, chi2r_max=2.0
#             )
#             idx0 = int(np.clip(idx0, 0, len(t) - 1))
#             t0_list.append(float(t[idx0]))
#         t0_global = float(max(t0_list)) if t0_list else float("nan")

#         mean_eq_list = []
#         sem_eq_list = []
#         for (_, t, pt, pt_sem) in series:
#             idxg = idx_from_t0(t, t0_global)
#             if idxg < len(pt):
#                 mean_eq, sem_eq = weighted_mean_and_sem(pt[idxg:], pt_sem[idxg:])
#                 mean_eq_list.append(mean_eq)
#                 sem_eq_list.append(sem_eq)

#         if mean_eq_list:
#             pc_mean, pc_sem = weighted_mean_and_sem(mean_eq_list, sem_eq_list)
#         else:
#             pc_mean, pc_sem = float("nan"), float("nan")

#         # pc_sop por ordem
#         for (ordk, t, pt, pt_sem) in series:
#             idx0 = detect_equilibrium_start_with_errors(
#                 t, pt, pt_sem, w=40, consec=6, z=2.0, chi2r_max=2.0
#             )
#             idx0 = int(np.clip(idx0, 0, len(t) - 1))
#             t0_i = float(t[idx0])

#             idxi = idx_from_t0(t, t0_i)
#             if idxi < len(pt):
#                 pc_i_mean, pc_i_sem = weighted_mean_and_sem(pt[idxi:], pt_sem[idxi:])
#             else:
#                 pc_i_mean, pc_i_sem = float("nan"), float("nan")

#             mean_by_order[ordk]["pc_sop"] = {
#                 "mean": _safe_float(pc_i_mean),
#                 "std_boot": _safe_float(pc_i_sem),
#                 "n_seeds": int(mean_by_order[ordk].get("n_seeds_pt", 0)),
#                 "n_boot": int(n_boot),
#                 "t0": _safe_float(t0_i),
#             }

#         orders_blocks = [
#             {"order_percolation": int(ordk), "data": mean_by_order[ordk]}
#             for ordk in sorted(mean_by_order.keys())
#         ]

#         p0_group = {
#             "p0_value": float(p0),
#             "num_seeds": len(seeds_set),
#             "orders": orders_blocks,
#             "pc_sop": {
#                 "mean": _safe_float(pc_mean),
#                 "std_boot": _safe_float(pc_sem),
#                 "n_seeds": len(seeds_set),
#                 "n_boot": int(n_boot),
#                 "t0_global": _safe_float(t0_global),
#             },
#         }

#         bundle["p0_groups"].append(p0_group)

#         print(
#             f"[ok] p0={p0:.2f}: files={len(files)} valid={valid_files} | "
#             f"seeds={len(seeds_set)} | pc_SOP={pc_mean:.6f}±{pc_sem:.6f}"
#         )

#         # limpeza por p0 (streaming)
#         per_order.clear()
#         mean_by_order.clear()
#         series.clear()
#         seeds_set.clear()
#         gc.collect()

#     bundle["meta"]["seed_used"] = sorted(seed_used_set)

#     bundle = _sanitize_for_json(bundle)
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(bundle, f, ensure_ascii=False, indent=2, allow_nan=False)

#     print(f"[salvo] {out_path}")
#     return out_path
def compute_means_for_folder(
    type_perc: str,
    num_colors: int,
    dim: int,
    L: int,
    NT: int,
    k: float,
    rho: float,
    p0_list: List[float],
    *,
    base_dir: str,
    x_max: float | None = None,
    n_boot: int = 20000,
    rng_seed: int = 12345,
    window_roll: int | None = None,
    clear_data: bool = False,
) -> str:
    """
    Versão corrigida para não subestimar erros do platô:

    - Mantém pt_mean(t) como média entre seeds.
    - Salva pt_std(t) e pt_sem(t) via _average_by_order_new.
    - Estima pc_sop de forma robusta:
        1) detecta t0_global a partir das curvas médias (por ordem)
        2) para cada seed: calcula média na cauda com tail_mean(..., method="autocorr")
        3) combina seeds com combine_tail_means(..., random_effects=True)
        4) combina ordens por média ponderada
    """

    base_dir = os.path.abspath(base_dir)

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

    if clear_data and os.path.isfile(out_path):
        os.remove(out_path)
        print(f"[clear_data] removido: {out_path}")

    # incremental: se existe e não tem arquivo novo, skip
    current_seed_files = []
    for p0 in p0_list:
        current_seed_files += sorted(
            glob.glob(os.path.join(data_dir, f"P0_*_p0_{p0:.2f}_seed_*.json"))
        )
    current_seed_files = sorted({os.path.basename(x) for x in current_seed_files})

    if os.path.isfile(out_path) and (not clear_data):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            old_used = old.get("meta", {}).get("seed_used", [])
            old_used = old_used if isinstance(old_used, list) else []
            if set(current_seed_files).issubset(set(map(str, old_used))):
                print(f"[skip] atualizado: {out_path}")
                return out_path
        except Exception as e:
            print(f"[recalc] falha lendo bundle antigo ({e}) -> recalculando")

    seed_used_set = set()

    bundle: Dict[str, Any] = {
        "meta": {
            "type_perc": type_perc,
            "num_colors": num_colors,
            "dim": dim,
            "L": L,
            "NT": NT,
            "k": float(k),
            "rho": float(rho),
            "base_dir": out_dir,
            "x_max_used": None if x_max is None else float(x_max),
            "bootstrap": {"n_boot": int(n_boot), "rng_seed": int(rng_seed)},
            "rolling": {"window": None if window_roll is None else int(window_roll)},
            "seed_used": [],
        },
        "p0_groups": [],
    }

    for p0 in p0_list:
        files = sorted(glob.glob(os.path.join(data_dir, f"P0_*_p0_{p0:.2f}_seed_*.json")))
        print(f"[debug] data_dir={data_dir} | p0={p0:.2f} | files={len(files)}")

        if not files:
            print(f"[aviso] Sem arquivos para p0={p0:.2f}")
            continue

        per_order: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        per_order_seed_series: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

        seeds_set: set[int] = set()
        valid_files = 0

        for fp in files:
            orders = _load_orders_new(fp)
            if not orders:
                continue

            valid_files += 1

            seed = _extract_seed_from_filename(fp)
            if seed is not None:
                seeds_set.add(seed)

            bn = os.path.basename(fp)
            seed_used_set.add(bn)

            for ordk, data in orders.items():
                # corta por x_max, se usado
                if x_max is not None and data.get("t") is not None:
                    t = np.asarray(data["t"], dtype=float)
                    m = (t <= x_max)
                    data["t"] = t[m]
                    if data.get("pt") is not None:
                        data["pt"] = np.asarray(data["pt"], dtype=float)[m]
                    if data.get("nt") is not None:
                        data["nt"] = np.asarray(data["nt"], dtype=float)[m]

                # guarda (t,pt) da seed para pc com autocorr
                if data.get("t") is not None and data.get("pt") is not None:
                    t_seed = np.asarray(data["t"], dtype=float)
                    pt_seed = np.asarray(data["pt"], dtype=float)
                    n0 = min(t_seed.size, pt_seed.size)
                    if n0 > 1:
                        per_order_seed_series[ordk].append((t_seed[:n0], pt_seed[:n0]))

                per_order[ordk].append(data)

        if valid_files == 0:
            print(f"[warn] p0={p0:.2f}: nenhum arquivo válido lido (JSON vazio/ruim?) -> pulando grupo")
            continue

        # ====== médias por ordem ======
        mean_by_order: Dict[int, Dict[str, Any]] = {}
        for ordk, lst in per_order.items():
            mean_by_order[ordk] = _average_by_order_new(lst)

        # ====== pc_sop global e por ordem (per-seed + autocorr + RE) ======

        # 1) t0_global: detecta usando curvas médias por ordem (usa pt_sem como erro de pt_mean)
        series_mean = []
        for ordk in sorted(mean_by_order.keys()):
            d = mean_by_order[ordk]
            if not d.get("time") or not d.get("pt_mean") or not d.get("pt_sem"):
                continue
            t = np.asarray(d["time"], dtype=float)
            pt = np.asarray(d["pt_mean"], dtype=float)
            pt_sem = np.asarray(d["pt_sem"], dtype=float)  # SEM da média entre seeds
            if t.size > 0:
                series_mean.append((ordk, t, pt, pt_sem))

        t0_list = []
        for (_, t, pt, pt_sem) in series_mean:
            idx0 = detect_equilibrium_start_with_errors(
                t, pt, pt_sem, w=40, consec=6, z=2.0, chi2r_max=2.0
            )
            idx0 = int(np.clip(idx0, 0, len(t) - 1))
            t0_list.append(float(t[idx0]))
        t0_global = float(max(t0_list)) if t0_list else float("nan")

        # 2) pc por ordem: pc por seed com autocorr e combine_tail_means (random effects)
        pc_by_order: Dict[int, Tuple[float, float, int]] = {}
        for ordk in sorted(mean_by_order.keys()):
            runs = []
            for (t_seed, pt_seed) in per_order_seed_series.get(ordk, []):
                if not np.isfinite(t0_global):
                    continue
                idx = idx_from_t0(t_seed, t0_global)
                if idx >= pt_seed.size:
                    continue

                s = tail_mean(pt_seed[idx:], tail_frac=0.2, method="autocorr")
                if np.isfinite(s["mean"]) and np.isfinite(s["sem"]) and (s["sem"] > 0):
                    runs.append({"mean": float(s["mean"]), "sem": float(s["sem"])})

            combo = combine_tail_means(runs, random_effects=True)
            pc_i_mean = float(combo["mean"])
            pc_i_sem = float(combo["se"])
            n_used = int(combo.get("R", 0))

            pc_by_order[ordk] = (pc_i_mean, pc_i_sem, n_used)

            mean_by_order[ordk]["pc_sop"] = {
                "mean": _safe_float(pc_i_mean),
                "std_boot": _safe_float(pc_i_sem),
                "n_seeds": int(n_used),
                "n_boot": int(n_boot),
                "t0": _safe_float(t0_global),
                "pc_method": "per-seed tail_mean(autocorr) + random-effects",
            }

        # 3) pc global: combina ordens por média ponderada
        mean_eq_list = []
        sem_eq_list = []
        for ordk in sorted(pc_by_order.keys()):
            m, s, _ = pc_by_order[ordk]
            if np.isfinite(m) and np.isfinite(s) and (s > 0):
                mean_eq_list.append(m)
                sem_eq_list.append(s)

        if mean_eq_list:
            pc_mean, pc_sem = weighted_mean_and_sem(mean_eq_list, sem_eq_list)
        else:
            pc_mean, pc_sem = float("nan"), float("nan")

        orders_blocks = [
            {"order_percolation": int(ordk), "data": mean_by_order[ordk]}
            for ordk in sorted(mean_by_order.keys())
        ]

        p0_group = {
            "p0_value": float(p0),
            "num_seeds": len(seeds_set),
            "orders": orders_blocks,
            "pc_sop": {
                "mean": _safe_float(pc_mean),
                "std_boot": _safe_float(pc_sem),
                "n_seeds": len(seeds_set),
                "n_boot": int(n_boot),
                "t0_global": _safe_float(t0_global),
                "pc_method": "combine orders of (per-seed autocorr + random-effects)",
            },
        }

        bundle["p0_groups"].append(p0_group)

        print(
            f"[ok] p0={p0:.2f}: files={len(files)} valid={valid_files} | "
            f"seeds={len(seeds_set)} | pc_SOP={pc_mean:.6f}±{pc_sem:.6f}"
        )

        # limpeza por p0
        per_order.clear()
        per_order_seed_series.clear()
        mean_by_order.clear()
        series_mean.clear()
        seeds_set.clear()
        gc.collect()

    bundle["meta"]["seed_used"] = sorted(seed_used_set)

    bundle = _sanitize_for_json(bundle)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"[salvo] {out_path}")
    return out_path

FLOAT = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

# casa com o caminho da pasta (sem o /data) OU do arquivo properties_mean_bundle.json
PARAM_RE = re.compile(
    rf"""
    /num_colors_(?P<nc>\d+)
    /dim_(?P<dim>\d+)
    /L_(?P<L>\d+)
    /NT_constant
    /NT_(?P<Nt>\d+)
    /k_(?P<k>{FLOAT})
    /rho_(?P<rho>{FLOAT})
    (?:/data)?/?$
    """,
    re.X
)

# parse do seed filename (basename) ex: P0_0.10_p0_1.00_seed_639042854.json
SEEDFILE_RE = re.compile(
    rf"^P0_(?P<P0>{FLOAT})_p0_(?P<p0>{FLOAT})_seed_(?P<seed>\d+)\.json$"
)

def _safe_float_df(x) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        if math.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")
    

def _parse_params_from_path(path: str) -> Optional[Tuple[int, int, float, int, float, int]]:
    """
    Retorna (L, Nt, k, nc, rho, dim) do path.
    """
    m = PARAM_RE.search(path.replace("\\", "/"))
    if not m:
        return None

    L = int(m.group("L"))
    Nt = int(m.group("Nt"))
    k = float(m.group("k"))
    nc = int(m.group("nc"))
    rho = float(m.group("rho"))
    dim = int(m.group("dim"))

    return (L, Nt, k, nc, rho, dim)

def _parse_P0_p0_from_seed_used(seed_used: List[str]) -> Tuple[float, float]:
    """
    Retorna (P0_mean, p0_mean) baseado nos basenames em seed_used.
    Se não der pra extrair, retorna (nan, nan).
    """
    P0_vals = []
    p0_vals = []
    for bn in seed_used:
        bn = os.path.basename(str(bn))
        m = SEEDFILE_RE.match(bn)
        if not m:
            continue
        P0_vals.append(float(m.group("P0")))
        p0_vals.append(float(m.group("p0")))
    if len(P0_vals) == 0:
        return float("nan"), float("nan")
    return float(np.mean(P0_vals)), float(np.mean(p0_vals))

def build_properties_dataframe(root: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    root = os.path.abspath(root)

    bundle_files = []
    for dirpath, _, filenames in os.walk(root):
        if "properties_mean_bundle.json" in filenames:
            bundle_files.append(os.path.join(dirpath, "properties_mean_bundle.json"))

    for bundle_path in sorted(bundle_files):

        parsed = _parse_params_from_path(os.path.dirname(bundle_path))
        if parsed is None:
            parsed = _parse_params_from_path(bundle_path)
        if parsed is None:
            continue

        L, Nt, k, nc, rho, dim = parsed

        try:
            with open(bundle_path, "r", encoding="utf-8") as f:
                js = json.load(f)
        except Exception:
            continue

        meta = js.get("meta", {})
        seed_used = meta.get("seed_used", [])
        if not isinstance(seed_used, list):
            seed_used = []

        p0_groups = js.get("p0_groups", [])
        if not isinstance(p0_groups, list) or len(p0_groups) == 0:
            continue

        P0_mean, _ = _parse_P0_p0_from_seed_used(seed_used)

        for g in p0_groups:
            p0_val = _safe_float(g.get("p0_value", float("nan")))
            N_samples = int(g.get("num_seeds", 0) or 0)

            orders = g.get("orders", [])
            if not isinstance(orders, list) or len(orders) == 0:
                continue

            for ob in orders:
                ordk = ob.get("order_percolation", None)
                if ordk is None:
                    continue

                order = int(ordk) + 1
                d = ob.get("data", {}) or {}

                pc_block = d.get("pc_sop", {}) if isinstance(d.get("pc_sop", {}), dict) else {}
                p_mean = _safe_float(pc_block.get("mean", float("nan")))
                p_err  = _safe_float(pc_block.get("std_boot", float("nan")))

                shortest_path = _safe_float(d.get("shortest_path_lin_mean", float("nan")))
                shortest_path_err = _safe_float(d.get("shortest_path_lin_sem", float("nan")))

                S_perc = _safe_float(d.get("M_size_mean", float("nan")))
                S_perc_err = _safe_float(d.get("M_size_sem", float("nan")))

                rows.append({
                    "dim": dim,
                    "L": L,
                    "Nt": Nt,
                    "k": k,
                    "nc": nc,
                    "rho": rho,
                    "p0": p0_val,
                    "P0": P0_mean,
                    "order": order,
                    "N_samples": N_samples,
                    "p_mean": p_mean,
                    "p_err": p_err,
                    "shortest_path": shortest_path,
                    "shortest_path_err": shortest_path_err,
                    "S_perc": S_perc,
                    "S_perc_err": S_perc_err,
                })

    cols = [
        "dim", "L", "Nt", "k", "nc", "rho", "p0", "P0",
        "order", "N_samples",
        "p_mean", "p_err",
        "shortest_path", "shortest_path_err",
        "S_perc", "S_perc_err",
    ]

    df = pd.DataFrame(rows, columns=cols)

    if not df.empty:
        df = df.sort_values(
            by=["dim", "nc", "rho", "k", "Nt", "L", "p0", "order"]
        ).reset_index(drop=True)
    
    df.to_csv(root + "/all_data.dat", index=False)
    

def process_all_data(clear_data: bool = False):
    p0_lst = [1.0]
    base_root = "../SOP_data/raw"               # <<< IMPORTANTE
    bas = "../SOP_data/raw/bond_percolation"

    all_parms = collect_param_combinations(bas)

    pbar = tqdm(all_parms, desc="Processando conjuntos", ncols=120)
    CLEAR_EVERY = 10

    for i, par in enumerate(pbar, 1):

        if i % CLEAR_EVERY == 0:
            clear_output(wait=True)
            print(f"--- Terminal limpo após {i} execuções ---")
            pbar.refresh()

        tp, nc, DIM, L, NT, K, RHO = par

        compute_means_for_folder(
            type_perc=tp,
            num_colors=nc,
            dim=DIM,
            L=L,
            NT=NT,
            k=K,
            rho=RHO,
            p0_list=p0_lst,
            base_dir=base_root,     # <<< NOVO ARGUMENTO OBRIGATÓRIO
            clear_data=False         # ou False se quiser incremental
        )

    clear_output(wait=True)
    print("Processamento finalizado.")
    
    root = "../Data/bond_percolation"
    build_properties_dataframe(root)