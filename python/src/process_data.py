import re, os, json, glob, math, textwrap, stat
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# --- regex para extrair params do caminho ---
# Aceita k/rho em float normal ou notação científica (ex.: 1.0e-04, 8.9e-02, 1.0000e-04)
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

# --- regex do nome de arquivo de seed (P0/p0/seed) ---
_fname_re = re.compile(
    r"P0_([0-9]*\.?[0-9]+(?:e[+\-]?[0-9]+)?)_p0_([0-9]*\.?[0-9]+(?:e[+\-]?[0-9]+)?)_seed_(\d+)\.json$",
    re.IGNORECASE
)

# regex para extrair p0 do nome do arquivo (com grupos nomeados)
FNAME_RE = re.compile(
    r"P0_(?P<P0>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)_p0_(?P<p0>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)_seed_(?P<seed>\d+)\.json$",
    re.IGNORECASE,
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
    Parse one JSON with the "results" list and return a list of tuples
    (order, pt_array, nt_array_or_None, M_size_or_None).

    - pt/nt are full time series (floats)
    - M_size is a single scalar stored in each result's "data"
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

            # time series
            p = np.asarray(d["pt"], float)
            n_arr = np.asarray(d["nt"], float) if "nt" in d else None

            # align lengths if needed
            n = min(len(p), len(n_arr)) if n_arr is not None else len(p)
            if n <= 0:
                continue
            p = p[:n]
            n_arr = n_arr[:n] if n_arr is not None else None

            # scalar M_size (may be missing)
            m_size = d.get("M_size", None)
            try:
                m_size = float(m_size) if m_size is not None else None
            except Exception:
                m_size = None

            out.append((int(order), p, n_arr, m_size))
    return out  # may be []


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

# ---------- Helpers: localizar subpastas k_xxx e rho_xxx por aproximação ----------
_FLOAT_DIR_RE = re.compile(r"^(?P<key>k|rho)_(?P<val>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)$", re.I)

def _find_numeric_subdir(base: Path, key: str, target: float,
                         rel_tol: float = 1e-12, abs_tol: float = 1e-15) -> Path | None:
    """
    Procura em 'base' uma subpasta cujo nome seja '{key}_<numero>' (em qualquer formatação),
    retornando aquela cujo valor numérico é ~== target.
    """
    key = key.lower()
    best = None
    best_err = None
    if not base.exists():
        return None
    for d in base.iterdir():
        if not d.is_dir():
            continue
        m = _FLOAT_DIR_RE.match(d.name)
        if not m:
            continue
        if m.group("key").lower() != key:
            continue
        try:
            val = float(m.group("val"))
        except Exception:
            continue
        if math.isclose(val, float(target), rel_tol=rel_tol, abs_tol=abs_tol):
            err = abs(val - float(target))
            if (best is None) or (err < best_err):
                best, best_err = d, err
    return best

def _first_existing(path_candidates):
    for p in path_candidates:
        if p is not None and Path(p).exists():
            return Path(p)
    return None

# --------------- Núcleo estatístico p0->ordem ----------------
def summarize_multi_seed_by_order(files, burn_in_frac=0.2, verbose=False):
    """
    Aggregate multiple seed JSON files by percolation order.

    For each order k we compute:
      - pt_mean, pt_sem_between (mean/SEM of stationary pt across seeds)
      - nt_mean, nt_sem_between (idem for nt, if available)
      - M_size_mean, M_size_sem_between (mean/SEM across seeds for scalar M_size)
      - n_seeds_contributed (how many seeds actually contributed for that order)
    """
    per_order_pt, per_order_nt, per_order_msize = {}, {}, {}
    any_seen = False
    processed_here = set()

    for jf in files:
        # track processed basenames
        if parse_p0_from_filename(jf) is not None:
            processed_here.add(os.path.basename(jf))
        try:
            entries = read_orders_one_file(jf)
        except Exception as e:
            if verbose:
                print(f"[WARN] failed to read {os.path.basename(jf)}: {e}")
            continue
        if entries:
            any_seen = True

        for order, p_arr, n_arr, m_size in entries:
            n = len(p_arr)
            if n < 3:
                continue

            # use tail after burn-in for pt/nt
            start = int(burn_in_frac * n)
            p_stationary = p_arr[start:]
            if p_stationary.size < 1:
                continue

            mean_p, _, _, _ = sem_acf(p_stationary)
            per_order_pt.setdefault(order, []).append(mean_p)

            if n_arr is not None:
                n_stationary = n_arr[start:]
                if n_stationary.size > 0:
                    mean_n, _, _, _ = sem_acf(n_stationary)
                    per_order_nt.setdefault(order, []).append(mean_n)

            # scalar M_size (no burn-in; it's a single number)
            if m_size is not None and np.isfinite(m_size):
                per_order_msize.setdefault(order, []).append(float(m_size))

    if not any_seen and not per_order_pt and not per_order_nt and not per_order_msize:
        return {}, True, processed_here

    summary = {}
    orders = sorted(set(list(per_order_pt.keys()) +
                        list(per_order_nt.keys()) +
                        list(per_order_msize.keys())))
    for order in orders:
        # pt
        mp = np.asarray(per_order_pt.get(order, []), float)
        Sp = len(mp)
        pt_mean = float(mp.mean()) if Sp > 0 else np.nan
        pt_sem  = float(mp.std(ddof=1)/np.sqrt(Sp)) if Sp > 1 else (0.0 if Sp == 1 else np.nan)

        # nt
        mn = np.asarray(per_order_nt.get(order, []), float)
        Sn = len(mn)
        nt_mean = float(mn.mean()) if Sn > 0 else np.nan
        nt_sem  = float(mn.std(ddof=1)/np.sqrt(Sn)) if Sn > 1 else (0.0 if Sn == 1 else np.nan)

        # M_size (between seeds)
        ms = np.asarray(per_order_msize.get(order, []), float)
        Sm = len(ms)
        msize_mean = float(ms.mean()) if Sm > 0 else np.nan
        msize_sem  = float(ms.std(ddof=1)/np.sqrt(Sm)) if Sm > 1 else (0.0 if Sm == 1 else np.nan)

        # prefer pt count as "contributed"; fallback to others if needed
        n_contrib = Sp or Sn or Sm

        summary[order] = {
            "n_seeds_contributed": int(n_contrib),
            "pt_mean": pt_mean,
            "pt_sem_between": pt_sem,
            "nt_mean": nt_mean,
            "nt_sem_between": nt_sem,
            "M_size_mean": msize_mean,
            "M_size_sem_between": msize_sem,
        }
    return summary, False, processed_here


# 3) build_dataframe_by_p0
def build_dataframe_by_p0(all_files, burn_in_frac=0.2, verbose=False, path_hint: str = None):
    """
    Build the per-(p0, order) summary DataFrame for a given parameter set.
    Produces only: pt_mean/pt_erro, nt_mean/nt_erro, M_size_mean/M_size_erro, perc_rate.
    """
    meta_source = path_hint or (all_files[0] if all_files else "")
    meta = parse_params_from_path(meta_source) or \
           (parse_params_from_path(all_files[0]) if all_files else None) or \
           {"type_perc": None, "num_colors": None, "dim": None, "L": None, "Nt": None, "k": None, "rho": None}
    n_orders = meta.get("num_colors") or 3

    # group by rounded p0
    groups = {}
    for f in all_files:
        p0_raw = parse_p0_from_filename(f)
        if p0_raw is None:
            if verbose:
                print(f"[WARN] unexpected filename, skipping: {os.path.basename(f)}")
            continue
        p0_key = round(float(p0_raw), 1)
        groups.setdefault(p0_key, []).append(f)

    cols = [
        "type_perc","num_colors","dim","L","Nt","k","rho","p0","order",
        "num_samples","num_sample_perc",
        "pt_mean","pt_erro","nt_mean","nt_erro",
        "M_size_mean","M_size_erro",
        "perc_rate",
    ]
    rows, processed = [], set()

    if not groups:
        return pd.DataFrame([{c: None for c in cols}])[cols], processed

    for p0_val in sorted(groups.keys()):
        p0_fmt = round(float(p0_val), 1)
        summary, all_empty, processed_here = summarize_multi_seed_by_order(
            groups[p0_val], burn_in_frac=burn_in_frac, verbose=verbose
        )
        processed |= set(processed_here)
        N = int(len(processed_here))  # number of seeds for this (p0, params)

        def append_row(order, M, pt_m, pt_e, nt_m, nt_e, msz_m, msz_e):
            # percolation rate (no CI; just M/N)
            q = (float(M) / float(N)) if N > 0 else np.nan

            rows.append({
                "type_perc": meta["type_perc"], "num_colors": meta["num_colors"], "dim": meta["dim"],
                "L": meta["L"], "Nt": meta["Nt"], "k": meta["k"], "rho": meta["rho"],
                "p0": p0_fmt, "order": order,
                "num_samples": N, "num_sample_perc": int(M),
                "pt_mean": pt_m, "pt_erro": pt_e, "nt_mean": nt_m, "nt_erro": nt_e,
                "M_size_mean": msz_m, "M_size_erro": msz_e,
                "perc_rate": q,
            })

        if all_empty or not summary:
            for order in range(1, int(n_orders) + 1):
                append_row(order, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            continue

        for order in range(1, int(n_orders) + 1):
            if order in summary:
                s = summary[order]
                append_row(order, s["n_seeds_contributed"],
                           s["pt_mean"], s["pt_sem_between"],
                           s["nt_mean"], s["nt_sem_between"],
                           s["M_size_mean"], s["M_size_sem_between"])
            else:
                append_row(order, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    df = pd.DataFrame(rows, columns=cols)

    # drop logical duplicates
    df = df.drop_duplicates(
        subset=["type_perc","num_colors","dim","L","Nt","k","rho","p0","order"],
        keep="last"
    )

    # numeric coercion
    num_cols = [
        "num_colors","dim","L","Nt","k","rho","p0","order",
        "num_samples","num_sample_perc",
        "pt_mean","pt_erro","nt_mean","nt_erro",
        "M_size_mean","M_size_erro",
        "perc_rate",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "p0" in df.columns:
        df["p0"] = df["p0"].round(1)

    # ensure deprecated columns are not present (guard if old code wrote them)
    df = df.drop(columns=[c for c in DEPRECATED_COLS if c in df.columns], errors="ignore")

    return df, processed



def _normalize_names(iterable):
    """Basenames sem espaços/vazios -> set único."""
    out = set()
    for x in iterable:
        b = os.path.basename(x).strip()
        if b:
            out.add(b)
    return out

def _expected_json_basenames(all_files):
    """Somente arquivos que batem com o padrão de simulação (via FNAME_RE)."""
    want = []
    for f in all_files:
        name = os.path.basename(f)
        if FNAME_RE.search(name):   # garante que é um dos seus JSONs de seed
            want.append(name)
    return _normalize_names(want)

def process_with_guard(all_files,
                       out_dat_path: Path,
                       out_txt_path: Path,
                       burn_in_frac=0.2,
                       verbose=False,
                       path_hint: str = None,
                       force_recompute: bool = False):
    """
    Versão estendida: mantém o .dat agregado e transforma o process_names.txt em TSV
    com header 'filename' + colunas das propriedades calculadas por ARQUIVO (seed) e ORDEM.
    """
    # ----------------------- utilidades locais -----------------------
    def _read_prev_names_first_column(path: Path) -> set[str]:
        """Lê o TXT/TSV anterior e retorna apenas a primeira coluna (filenames)."""
        if not path.exists():
            return set()
        lines = path.read_text().splitlines()
        if not lines:
            return set()
        first = lines[0]
        # TSV moderno com header 'filename'
        if "\t" in first and first.split("\t", 1)[0].strip().lower() == "filename":
            out = set()
            for ln in lines[1:]:
                if not ln.strip():
                    continue
                out.add(os.path.basename(ln.split("\t", 1)[0].strip()))
            return out
        # Modo legado: 1 nome por linha
        return _normalize_names(lines)

    def _per_file_rows(file_path: str, meta: dict, n_orders: int, burn_in_frac=0.2):
        """
        Calcula estatísticas por arquivo/ordem.
        Retorna lista de dicts com: filename + BASE_COLS.
        num_samples=1; num_sample_perc=1 se a ordem existe no arquivo, senão 0.
        perc_rate = 1.0 quando a ordem existe; 0.0 quando não.
        """
        rows = []
        base = {
            "type_perc": meta.get("type_perc"),
            "num_colors": meta.get("num_colors"),
            "dim": meta.get("dim"),
            "L": meta.get("L"),
            "Nt": meta.get("Nt"),
            "k": meta.get("k"),
            "rho": meta.get("rho"),
        }
        p0_val = parse_p0_from_filename(file_path)
        try:
            entries = read_orders_one_file(file_path)
        except Exception:
            entries = []

        # index por ordem
        by_order = {}
        for order, p_arr, n_arr, m_size in entries:
            by_order[int(order)] = (
                np.asarray(p_arr, float) if p_arr is not None else np.array([], float),
                None if n_arr is None else np.asarray(n_arr, float),
                m_size if (m_size is None or np.isfinite(m_size)) else None
            )

        for order in range(1, int(n_orders) + 1):
            if order in by_order:
                p_arr, n_arr, m_size = by_order[order]
                n = len(p_arr)
                start = int(burn_in_frac * n) if n > 0 else 0
                # pt
                if n > 0 and start < n:
                    m_pt, _, _, _ = sem_acf(p_arr[start:])
                    pt_mean, pt_sem = float(m_pt), 0.0  # 1 arquivo => erro entre seeds = 0
                else:
                    pt_mean = np.nan
                    pt_sem  = np.nan
                # nt
                if n_arr is not None and n_arr.size > 0:
                    ns = n_arr[start:] if start < n_arr.size else np.array([], float)
                    if ns.size > 0:
                        m_nt, _, _, _ = sem_acf(ns)
                        nt_mean, nt_sem = float(m_nt), 0.0
                    else:
                        nt_mean, nt_sem = (np.nan, np.nan)
                else:
                    nt_mean, nt_sem = (np.nan, np.nan)
                # M_size
                if m_size is not None and np.isfinite(m_size):
                    msize_mean, msize_sem = float(m_size), 0.0
                else:
                    msize_mean, msize_sem = (np.nan, np.nan)

                rows.append({
                    "filename": os.path.basename(file_path),
                    **base,
                    "p0": None if p0_val is None else round(float(p0_val), 1),
                    "order": int(order),
                    "num_samples": 1,
                    "num_sample_perc": 1,
                    "pt_mean": pt_mean, "pt_erro": pt_sem,
                    "nt_mean": nt_mean, "nt_erro": nt_sem,
                    "M_size_mean": msize_mean, "M_size_erro": msize_sem,
                    "perc_rate": 1.0,
                })
            else:
                rows.append({
                    "filename": os.path.basename(file_path),
                    **base,
                    "p0": None if p0_val is None else round(float(p0_val), 1),
                    "order": int(order),
                    "num_samples": 1,
                    "num_sample_perc": 0,
                    "pt_mean": np.nan, "pt_erro": np.nan,
                    "nt_mean": np.nan, "nt_erro": np.nan,
                    "M_size_mean": np.nan, "M_size_erro": np.nan,
                    "perc_rate": 0.0,
                })
        return rows
    # ----------------------- /utilidades locais -----------------------

    # --- Determina o data_dir real ---
    data_dir = None
    if all_files:
        data_dir = Path(all_files[0]).parent
    else:
        p = Path(out_dat_path)
        data_dir = p.parent if p.suffix.lower() == ".dat" else p
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Realinha os caminhos de saída para dentro do data_dir real ---
    out_dat_path = data_dir / Path(out_dat_path).name
    out_txt_path = data_dir / Path(out_txt_path).name

    # --- Se não recebi lista, busco os jsons no data_dir ---
    if not all_files:
        all_files = [str(x) for x in sorted(data_dir.glob("*.json"))]

    out_dat_path.parent.mkdir(parents=True, exist_ok=True)

    # Conjunto esperado (basenames) e anterior (do TXT/TSV)
    expected_set = _expected_json_basenames(all_files)
    if out_txt_path.exists():
        prev_set = _read_prev_names_first_column(out_txt_path)
    else:
        prev_set = set()

    # Atalho: se nada mudou e .dat existe -> reaproveita SEM reescrever .txt
    if (not force_recompute) and expected_set and (expected_set == prev_set) and out_dat_path.exists():
        if verbose:
            print(f"[INFO] Up-to-date: {len(prev_set)} nomes no TXT = {len(expected_set)} JSONs na pasta.")
            print(f"[INFO] Reaproveitando {out_dat_path.name} sem tocar no {out_txt_path.name}.")
        df = pd.read_csv(out_dat_path, sep="\t", na_values=["Null"])
        return df, prev_set

    # (Re)processa .dat
    if verbose:
        print(f"[INFO] Reprocessando: force={force_recompute}, expected={len(expected_set)}, "
              f"prev={len(prev_set)}, dat_existe={out_dat_path.exists()}")
    df, processed_set = build_dataframe_by_p0(
        all_files, burn_in_frac=burn_in_frac, verbose=verbose, path_hint=path_hint
    )
    df.to_csv(out_dat_path, sep="\t", index=False, na_rep="Null")
    if verbose:
        print("[INFO] .dat salvo em:", out_dat_path.resolve())

    # Conjunto final esperado para o TXT
    new_set = expected_set if expected_set else _normalize_names(processed_set)

    # --------- Construção do TSV detalhado por arquivo/ordem ----------
    meta = parse_params_from_path(path_hint or (all_files[0] if all_files else str(out_dat_path.parent)))
    if meta is None:
        meta = {"type_perc": None, "num_colors": None, "dim": None, "L": None, "Nt": None, "k": None, "rho": None}
    n_orders = meta.get("num_colors") or 1

    # mapear basename -> caminho completo
    by_basename = {os.path.basename(p): p for p in all_files}
    names_rows = []
    for bname in sorted(new_set):
        full = by_basename.get(bname)
        if not full or not Path(full).exists():
            names_rows.append({
                "filename": bname,
                "type_perc": meta["type_perc"], "num_colors": meta["num_colors"], "dim": meta["dim"],
                "L": meta["L"], "Nt": meta["Nt"], "k": meta["k"], "rho": meta["rho"],
                "p0": np.nan, "order": np.nan,
                "num_samples": np.nan, "num_sample_perc": np.nan,
                "pt_mean": np.nan, "pt_erro": np.nan, "nt_mean": np.nan, "nt_erro": np.nan,
                "M_size_mean": np.nan, "M_size_erro": np.nan, "perc_rate": np.nan,
            })
            continue
        names_rows.extend(_per_file_rows(full, meta, n_orders, burn_in_frac=burn_in_frac))

    names_cols = (["filename"] + BASE_COLS)
    names_df = pd.DataFrame(names_rows, columns=names_cols)

    # Escreve o TSV apenas se houver alteração no conjunto
    if new_set != prev_set:
        tmp = out_txt_path.with_suffix(".tmp")
        names_df.to_csv(tmp, sep="\t", index=False, na_rep="Null")
        tmp.replace(out_txt_path)
        if verbose:
            print(f"[INFO] {out_txt_path.name} atualizado (formato TSV) com {len(names_df)} linhas.")
    else:
        # Mesmo conjunto: ainda assim garante o formato novo
        names_df.to_csv(out_txt_path, sep="\t", index=False, na_rep="Null")
        if verbose:
            print(f"[INFO] {out_txt_path.name} recriado no formato TSV (mesmo conjunto).")

    return df, new_set


def saving_data(all_data,
                output_data: Path,
                output_names: Path,
                burn_in_frac=0.20,
                verbose=False,
                path_hint: str = None,
                force_recompute: bool = False):
    return process_with_guard(
        all_files=all_data,
        out_dat_path=output_data,
        out_txt_path=output_names,
        burn_in_frac=burn_in_frac,
        verbose=verbose,
        path_hint=path_hint,
        force_recompute=force_recompute,
    )

# ### NEW: leitura robusta do process_names(.txt|.tsv) para extrair só os filenames
def _read_prev_names_first_column(path: Path) -> set[str]:
    """
    Lê o arquivo de nomes (pode ser lista simples ou TSV com header 'filename')
    e retorna um set com os basenames (primeira coluna).
    """
    if not path.exists():
        return set()
    txt = path.read_text().splitlines()
    if not txt:
        return set()
    # Se for TSV com header começando por 'filename'
    if "\t" in txt[0] and txt[0].split("\t", 1)[0].strip().lower() == "filename":
        out = set()
        for line in txt[1:]:
            if not line.strip():
                continue
            first = line.split("\t", 1)[0].strip()
            if first:
                out.add(os.path.basename(first))
        return out
    # Caso antigo: lista simples de nomes
    return _normalize_names(txt)


# ### NEW: computar estatísticas por ARQUIVO (seed) e por ORDEM
def _per_file_rows(file_path: str, meta: dict, n_orders: int, burn_in_frac=0.2):
    """
    Retorna uma lista de dicionários (linhas) com:
      filename + BASE_COLS (preenchidos por arquivo/ordem).
    num_samples=1 sempre; num_sample_perc=1 se a ordem existir no JSON, senão 0.
    perc_rate = 1.0 quando a ordem existe, senão 0.0.
    """
    rows = []
    base = {  # meta fixa desse arquivo
        "type_perc": meta["type_perc"], "num_colors": meta["num_colors"],
        "dim": meta["dim"], "L": meta["L"], "Nt": meta["Nt"],
        "k": meta["k"], "rho": meta["rho"],
    }
    p0_val = parse_p0_from_filename(file_path)
    try:
        entries = read_orders_one_file(file_path)
    except Exception:
        entries = []

    # indexar entries por ordem
    by_order = {}
    for order, p_arr, n_arr, m_size in entries:
        by_order[int(order)] = (np.asarray(p_arr, float),
                                None if n_arr is None else np.asarray(n_arr, float),
                                m_size if (m_size is None or np.isfinite(m_size)) else None)

    for order in range(1, int(n_orders) + 1):
        if order in by_order:
            p_arr, n_arr, m_size = by_order[order]
            n = len(p_arr)
            start = int(burn_in_frac * n)
            p_stationary = p_arr[start:] if n > 0 else np.array([], float)
            pt_mean, pt_sem, = (np.nan, np.nan)
            if p_stationary.size > 0:
                m, _, se, _ = sem_acf(p_stationary)
                pt_mean, pt_sem = float(m), 0.0  # erro entre sementes = 0 para um arquivo

            nt_mean, nt_sem = (np.nan, np.nan)
            if n_arr is not None and n_arr.size > 0:
                n_stationary = n_arr[start:] if n_arr.size > start else np.array([], float)
                if n_stationary.size > 0:
                    m, _, se, _ = sem_acf(n_stationary)
                    nt_mean, nt_sem = float(m), 0.0  # idem

            msize_mean, msize_sem = (np.nan, np.nan)
            if m_size is not None and np.isfinite(m_size):
                msize_mean, msize_sem = float(m_size), 0.0

            rows.append({
                "filename": os.path.basename(file_path),
                **base,
                "p0": None if p0_val is None else round(float(p0_val), 1),
                "order": int(order),
                "num_samples": 1,
                "num_sample_perc": 1,
                "pt_mean": pt_mean, "pt_erro": pt_sem,
                "nt_mean": nt_mean, "nt_erro": nt_sem,
                "M_size_mean": msize_mean, "M_size_erro": msize_sem,
                "perc_rate": 1.0,
            })
        else:
            # ordem ausente neste arquivo
            rows.append({
                "filename": os.path.basename(file_path),
                **base,
                "p0": None if p0_val is None else round(float(p0_val), 1),
                "order": int(order),
                "num_samples": 1,
                "num_sample_perc": 0,
                "pt_mean": np.nan, "pt_erro": np.nan,
                "nt_mean": np.nan, "nt_erro": np.nan,
                "M_size_mean": np.nan, "M_size_erro": np.nan,
                "perc_rate": 0.0,
            })
    return rows


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

# Columns to guarantee across all partial and accumulated datasets
# --- schema we want to preserve in .dat and DataFrames ---
BASE_COLS = [
    "type_perc","num_colors","dim","L","Nt","k","rho","p0","order",
    "num_samples","num_sample_perc",
    "pt_mean","pt_erro","nt_mean","nt_erro",
    "M_size_mean","M_size_erro",
    "perc_rate",
]

# --- columns we want to drop forever (deprecated) ---
DEPRECATED_COLS = [
    "perc_ci_low","perc_ci_high",
    "pt_mean_uncond","pt_erro_uncond",
    "nt_mean_uncond","nt_erro_uncond",
]


# Composite key that uniquely identifies a measurement row
KEY_COLS = ["type_perc","num_colors","dim","L","Nt","k","rho","p0","order"]


def _align_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Ensure 'df' has all 'columns'; add missing ones with NaN and return a view
    with exactly 'columns' in that order (copy for safety).
    """
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df[columns].copy()

# keep your existing BASE_COLS / KEY_COLS / DEPRECATED_COLS
# BASE_COLS = [...]
# KEY_COLS  = [...]
# DEPRECATED_COLS = [...]

def _align_to_union(df: pd.DataFrame, union_cols: list) -> pd.DataFrame:
    """Ensure df has exactly the columns in 'union_cols' (adding missing as NaN, preserving order)."""
    for c in union_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[union_cols].copy()

def join_all_data(type_perc, num_colors, dim, L, Nt, k, base_root="../Data",
                  rel_tol: float = 1e-12, abs_tol: float = 1e-15):
    """
    Aggregate all 'all_data*.dat' across rho under (type_perc, num_colors, dim, L, Nt, k),
    and UPDATE the global accumulated file:

        {base_root}/{type_perc}_percolation/all_data_{dim}D.dat

    - Defines out_dir/out_path at the top (avoids UnboundLocalError).
    - Keeps any extra columns (e.g., M_size_mean/M_size_erro) and drops deprecated ones.
    - De-duplicates by KEY_COLS, keeping the newest rows.
    """

    # --- define output targets FIRST (so they exist in any code path) ---
    out_dir = Path(base_root) / f"{type_perc}_percolation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"all_data_{dim}D.dat"

    # discover rho values
    rho_values = list_rho_values(type_perc, num_colors, dim, L, Nt, k, base_root=base_root)
    if not rho_values:
        print("[WARN] No rho values found.")
        # If nothing to add, return current accumulated (if any) to avoid surprises
        if out_path.exists():
            try:
                acc = pd.read_csv(out_path, sep="\t", na_values=["Null"])
                acc = acc.drop(columns=[c for c in DEPRECATED_COLS if c in acc.columns], errors="ignore")
                return acc
            except Exception:
                pass
        return pd.DataFrame(columns=BASE_COLS)

    base_nt = (Path(base_root)
               / f"{type_perc}_percolation"
               / f"num_colors_{num_colors}"
               / f"dim_{dim}"
               / f"L_{L}"
               / "NT_constant"
               / f"NT_{Nt}")

    # find k_* numerically close to given k
    k_dir = _find_numeric_subdir(base_nt, "k", k, rel_tol=rel_tol, abs_tol=abs_tol)
    if k_dir is None:
        print(f"[WARN] k_* folder not found (k≈{k}):", base_nt)
        # same behavior: return existing accumulated, if any
        if out_path.exists():
            try:
                acc = pd.read_csv(out_path, sep="\t", na_values=["Null"])
                acc = acc.drop(columns=[c for c in DEPRECATED_COLS if c in acc.columns], errors="ignore")
                return acc
            except Exception:
                pass
        return pd.DataFrame(columns=BASE_COLS)

    dfs = []
    for rho in sorted(rho_values):
        rho_dir = _find_numeric_subdir(k_dir, "rho", rho, rel_tol=rel_tol, abs_tol=abs_tol)
        if rho_dir is None:
            print(f"[WARN] rho_* folder not found (rho≈{rho}) under {k_dir}")
            continue

        data_dir = rho_dir / "data"
        # accept both file names
        candidates = [data_dir / "all_data.dat", data_dir / f"all_data_{dim}D.dat"]
        fpath = _first_existing(candidates)
        if fpath is None:
            print(f"[WARN] No all_data*.dat in {data_dir}")
            continue

        try:
            df_rho = pd.read_csv(fpath, sep="\t", na_values=["Null"])
        except Exception as e:
            print(f"[WARN] Failed to read {fpath}: {e}")
            continue

        # drop deprecated columns if they exist
        df_rho = df_rho.drop(columns=[c for c in DEPRECATED_COLS if c in df_rho.columns], errors="ignore")

        # ensure BASE_COLS exist, but DO NOT drop extra columns (keep new fields like M_size_*)
        for c in BASE_COLS:
            if c not in df_rho.columns:
                df_rho[c] = np.nan

        dfs.append(df_rho)

    # If no valid partials found, return accumulated (if exists) else empty
    if not dfs:
        print("[WARN] No valid all_data*.dat found to merge.")
        if out_path.exists():
            try:
                acc = pd.read_csv(out_path, sep="\t", na_values=["Null"])
                acc = acc.drop(columns=[c for c in DEPRECATED_COLS if c in acc.columns], errors="ignore")
                return acc
            except Exception:
                pass
        return pd.DataFrame(columns=BASE_COLS)

    # new block for this parameter set
    new_block = pd.concat(dfs, ignore_index=True, sort=False)

    # numeric coercion & rounding
    num_cols = ["num_colors","dim","L","Nt","k","rho","p0","order",
                "num_samples","num_sample_perc","pt_mean","pt_erro","nt_mean","nt_erro",
                "M_size_mean","M_size_erro","perc_rate"]
    for c in num_cols:
        if c in new_block.columns:
            new_block[c] = pd.to_numeric(new_block[c], errors="coerce")
    if "p0" in new_block.columns:
        new_block["p0"] = new_block["p0"].round(1)

    # drop duplicates within the new block
    new_block = new_block.drop_duplicates(subset=KEY_COLS, keep="last")

    # read accumulated file (if any) and clean deprecated cols
    if out_path.exists():
        try:
            acc = pd.read_csv(out_path, sep="\t", na_values=["Null"])
            acc = acc.drop(columns=[c for c in DEPRECATED_COLS if c in acc.columns], errors="ignore")
        except Exception as e:
            print(f"[WARN] Failed to read accumulated file {out_path}: {e}")
            acc = pd.DataFrame(columns=BASE_COLS)
    else:
        acc = pd.DataFrame(columns=BASE_COLS)

    # union schema: BASE_COLS + any extra columns present in either side
    union_cols = list(dict.fromkeys(list(BASE_COLS) + list(acc.columns) + list(new_block.columns)))

    acc_aligned = _align_to_union(acc, union_cols)
    new_aligned = _align_to_union(new_block, union_cols)

    merged = pd.concat([acc_aligned, new_aligned], ignore_index=True, sort=False)

    # normalize dtypes for key and drop duplicates (newest wins)
    if "type_perc" in merged.columns:
        merged["type_perc"] = merged["type_perc"].astype(str)
    merged = merged.drop_duplicates(subset=KEY_COLS, keep="last")

    # sort for readability
    sort_cols = ["type_perc","num_colors","dim","L","Nt","k","rho","p0","order"]
    sort_cols = [c for c in sort_cols if c in merged.columns]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)

    # persist
    merged.to_csv(out_path, sep="\t", index=False, na_rep="Null")
    print("Updated:", out_path.resolve())

    return merged



def delete_json_with_p0(type_perc, num_colors, dim, L, Nt, k,
                        p0_target=0.30, base_root="../Data",
                        rel_tol=0, abs_tol=5e-04,
                        dry_run=True):
    """
    Apaga (ou só lista, se dry_run=True) todos os .json com p0 == p0_target
    nas pastas ../Data/{type_perc}_percolation/.../k_{k}/rho_*/data

    abs_tol padrão 5e-4 cobre variações de escrita (ex.: 0.300000 vs 0.3).
    """
    base_k = (Path(base_root)
              / f"{type_perc}_percolation"
              / f"num_colors_{num_colors}"
              / f"dim_{dim}"
              / f"L_{L}"
              / "NT_constant"
              / f"NT_{Nt}"
              / f"k_{k:.1e}")

    if not base_k.exists():
        print("[WARN] diretório não encontrado:", base_k)
        return 0, 0

    total, matched = 0, 0
    for data_dir in base_k.glob("rho_*/data"):
        if not data_dir.is_dir():
            continue
        for jf in data_dir.glob("*.json"):
            total += 1
            m = FNAME_RE.search(jf.name)
            if not m:
                continue
            try:
                p0_val = float(m.group("p0"))
            except Exception:
                continue
            if math.isclose(p0_val, float(p0_target), rel_tol=rel_tol, abs_tol=abs_tol):
                matched += 1
                if dry_run:
                    print("[DRY-RUN] apagar:", jf)
                else:
                    try:
                        jf.unlink()
                        print("[OK] apagado:", jf)
                    except Exception as e:
                        print("[ERR] não apagou:", jf, "->", e)
    print(f"\nArquivos verificados: {total} | p0≈{p0_target:.2f} encontrados: {matched} "
          f"| ação: {'listar' if dry_run else 'apagar'}")
    return total, matched

# ---------- helpers de análise temporal usados nos seus plots ----------
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

def bootstrap_mean_scalar(vals, n_boot=20000, ci=0.95, rng=None):
    """
    Bootstrap do valor médio a partir de vals (array 1D).
    Retorna: mean_point, se_boot, (ci_low, ci_high).
    """
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    m = vals.size
    if m == 0:
        return (np.nan, np.nan, (np.nan, np.nan))
    if rng is None:
        rng = np.random.default_rng()
    if m == 1:
        return (float(vals[0]), 0.0, (float(vals[0]), float(vals[0])))
    idx = rng.integers(0, m, size=(n_boot, m))
    boot_means = vals[idx].mean(axis=1)
    mean_point = float(vals.mean())
    se_boot    = float(boot_means.std(ddof=1))
    alpha = 0.5*(1-ci)
    lo, hi = np.quantile(boot_means, [alpha, 1-alpha])
    return mean_point, se_boot, (float(lo), float(hi))

# ---------- função utilitária: carregar uma única amostra robustamente ----------
def data_single_sample(type_perc, num_colors, dim, L, Nt, k, rho, p0, seed,
                       base_root="/home/junior/Documents/self_organization_percolation/Data",
                       rel_tol: float = 1e-12, abs_tol: float = 1e-15):
    """
    Localiza a pasta k_* e rho_* por aproximação numérica e abre um JSON específico
    'P0_0.10_p0_{p0:.2f}_seed_{seed}.json'.
    """
    base_nt = (Path(base_root)
               / f"{type_perc}_percolation"
               / f"num_colors_{num_colors}"
               / f"dim_{dim}"
               / f"L_{L}"
               / "NT_constant"
               / f"NT_{Nt}")

    k_dir = _find_numeric_subdir(base_nt, "k", k, rel_tol=rel_tol, abs_tol=abs_tol)
    if k_dir is None:
        raise FileNotFoundError(f"Não encontrei pasta k≈{k} em {base_nt}")

    rho_dir = _find_numeric_subdir(k_dir, "rho", rho, rel_tol=rel_tol, abs_tol=abs_tol)
    if rho_dir is None:
        raise FileNotFoundError(f"Não encontrei pasta rho≈{rho} em {k_dir}")

    data_dir = rho_dir / "data"
    filename = f"P0_0.10_p0_{p0:.2f}_seed_{seed}.json"
    file_path = data_dir / filename

    entries = read_orders_one_file(str(file_path))
    try:
        # monta dicionário com t, p_1..p_4 e N_1..N_4 (se existirem)
        t_len = len(entries[0][1]) if entries else 0
        dct = {"t": list(range(t_len))}
        for i in range(num_colors):
            # entries[i] pode não existir se não houve percolação daquela cor
            if i < len(entries):
                dct[f"p_{i+1}"] = [float(v) for v in entries[i][1]]
                if entries[i][2] is not None:
                    dct[f"N_{i+1}"] = [int(v) for v in entries[i][2]]
            else:
                dct[f"p_{i+1}"] = [np.nan]*t_len
                dct[f"N_{i+1}"] = [np.nan]*t_len
        return dct
    except IndexError as e:
        raise IndexError("Arquivo vazio: não houve percolação nesta seed/p0") from e
