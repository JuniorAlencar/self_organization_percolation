import re
from pathlib import Path
import os, glob
import pandas as pd
from typing import Sequence, Optional, Literal, Dict, Any
import numpy as np
import math
import json


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

    for c in expected:
        if c not in df.columns:
            df[c] = np.nan
    # tipos
    for c in ["P0","p0","order","p_mean","p_std","p_sem","shortest_path","S_perc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["filename"] = df["filename"].astype(str)
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
    Retorna uma lista de dicionários (uma linha por 'order') com:
    L, Nt, k, nc, rho, p0, P0, order, N_samples, p_mean, p_err,
    shortest_path, shortest_path_err, S_perc, S_perc_err
    """
    info = parse_data_dir(data_dir)
    if info is None:
        return []

    all_file = Path(data_dir) / "all_data.dat"
    df = _load_all_data(all_file.as_posix())

    # p0/P0: tentamos pegar único valor (se múltiplos, fica NaN)
    def _unique_or_nan(s: pd.Series):
        vals = s.dropna().unique()
        return float(vals[0]) if vals.size == 1 else np.nan

    p0_uni = _unique_or_nan(df["p0"])
    P0_uni = _unique_or_nan(df["P0"])

    # Agrupar por 'order' (apenas ordens não-NaN)
    rows = []
    orders = sorted([o for o in df["order"].dropna().unique()])
    if len(orders) == 0:
        # não há ordens válidas → não gera linha agregada
        return rows

    for order in orders:
        sub = df.loc[df["order"] == order].copy()

        # N_samples = número de filenames distintos para essa order
        N_samples = int(sub["filename"].nunique())

        # p_mean/p_err com combine_tail_means usando (mean, sem) de cada seed
        run_stats = []
        # Usamos a última linha por filename (se houver repetidos) para evitar duplicação
        for fname, g in sub.groupby("filename"):
            # Uma linha por seed: se houver várias linhas (improvável), pega a última
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

        # shortest_path: média simples e erro padrão (ignorando NaN)
        sp_mean, sp_sem, _ = _mean_sem(sub["shortest_path"])
        # S_perc: média simples e erro padrão
        sperc_mean, sperc_sem, _ = _mean_sem(sub["S_perc"])

        rows.append({
            "L":   int(info["L"]),
            "Nt":  int(info["Nt"]),
            "k":   float(info["k"]),
            "nc":  int(info["nc"]),
            "rho": float(info["rho"]),
            "p0":  p0_uni,
            "P0":  P0_uni,
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

    Retorna: {dim: Path(arquivo_gerado)}
    """
    base = Path(base_root)
    outputs: dict[int, Path] = {}
    buckets: dict[int, list[dict]] = {}

    # percorre apenas diretórios que batem DIR_RE
    for d in base.glob("num_colors_*/dim_*/L_*/NT_constant/NT_*/k_*/rho_*/data"):
        dposix = d.as_posix()
        m = DIR_RE.match(dposix)
        if not m:
            if verbose:
                print(f"[ignorado] {dposix} (não bate regex)")
            continue

        # agrega esse data_dir
        rows = _summarize_one_data_dir(dposix)
        if len(rows) == 0:
            if verbose:
                print(f"[info] sem ordens válidas em: {dposix}")
            continue

        dim = int(m.group("dim"))
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

        out_path = base / f"all_data_{dim}D.dat"
        out_df.to_csv(out_path.as_posix(), sep=" ", index=False, na_rep="NaN")
        outputs[dim] = out_path
        if verbose:
            print(f"[write] {out_path}  ({len(out_df)} linhas)")

    if verbose and not buckets:
        print("[done] nenhuma pasta com ordens válidas encontrada.")

    return outputs
