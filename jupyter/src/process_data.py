import re, os, json, glob, math
import numpy as np
import pandas as pd
from pathlib import Path

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
    with open(file_path, "r") as f:
        obj = json.load(f)
    out = []
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        for item in obj["results"]:
            order = item.get("order_percolation", None)
            d = item.get("data", {})
            if order is None or "time" not in d or "pt" not in d:
                continue
            t = np.asarray(d["time"], float)
            p = np.asarray(d["pt"], float)
            n = min(len(t), len(p))
            if n:
                out.append((int(order), t[:n], p[:n]))
    return out

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

def summarize_multi_seed_by_order(files, burn_in_frac=0.2, verbose=False):
    per_order = {}
    any_seen = False
    processed_here = set()

    for jf in files:
        # marque como "processado" apenas se casa o padrão do filename
        if parse_p0_from_filename(jf) is not None:
            processed_here.add(os.path.basename(jf))

        try:
            entries = read_orders_one_file(jf)
        except Exception as e:
            if verbose: print(f"[WARN] falha lendo {os.path.basename(jf)}: {e}")
            continue
        if entries:
            any_seen = True

        for order, tt, pp in entries:
            n = len(pp)
            if n < 3:
                continue
            start = int(burn_in_frac * n)
            p_stationary = pp[start:]
            mean, std, sem, neff = sem_acf(p_stationary)
            per_order.setdefault(order, []).append(mean)

    if not any_seen:
        return {}, True, processed_here
    if not per_order:
        return {}, False, processed_here

    summary = {}
    for order, means in per_order.items():
        means = np.asarray([m for m in means if np.isfinite(m)], float)
        S = len(means)
        if S == 0:
            continue
        grand_mean = means.mean()
        std_between = means.std(ddof=1) if S > 1 else np.nan
        sem_between = (std_between / math.sqrt(S)) if S > 1 else np.nan
        ci95 = (grand_mean - 1.96*sem_between, grand_mean + 1.96*sem_between) if S > 1 else (np.nan, np.nan)
        summary[order] = {
            "n_seeds": int(S),
            "grand_mean": float(grand_mean),
            "sem_between_seeds": (float(sem_between) if np.isfinite(sem_between) else np.nan),
            "ci95_between_seeds": (
                float(ci95[0]) if np.isfinite(ci95[0]) else np.nan,
                float(ci95[1]) if np.isfinite(ci95[1]) else np.nan
            ),
        }
    return summary, False, processed_here

# ========= build_dataframe_by_p0 ATUALIZADA =========
def build_dataframe_by_p0(all_files, burn_in_frac=0.2, verbose=False, path_hint: str = None):
    """
    Agrupa 'all_files' por p0, roda o resumo por ORDEM para cada p0
    e devolve (DataFrame, processed_set). Os metadados (type_perc, num_colors, ...)
    são extraídos do caminho via regex.
    - path_hint: pode ser o 'path_data'; se None, usa all_files[0] (quando existir).
    """
    # 0) Extrair metaparâmetros do caminho
    meta_source = path_hint or (all_files[0] if all_files else "")
    meta = parse_params_from_path(meta_source)
    # fallback: tente extrair do próprio arquivo se path_hint não casar
    if meta is None and all_files:
        meta = parse_params_from_path(all_files[0])
    if meta is None:
        # não achou nada — vai preencher Null nos metacampos
        meta = {"type_perc": None, "num_colors": None, "dim": None, "L": None, "Nt": None, "k": None, "rho": None}

    # 1) agrupar por p0
    groups = {}
    for f in all_files:
        p0 = parse_p0_from_filename(f)
        if p0 is None:
            if verbose: print(f"[WARN] nome inesperado, ignorando: {os.path.basename(f)}")
            continue
        groups.setdefault(p0, []).append(f)

    cols = ["type_perc","num_colors","dim","L","Nt","k","rho","p0","order","num_samples","p_mean","IC95","erro"]
    rows = []
    processed = set()

    if not groups:
        return pd.DataFrame([{c: None for c in cols}])[cols], processed

    for p0_val in sorted(groups.keys()):
        summary, all_empty, processed_here = summarize_multi_seed_by_order(groups[p0_val], burn_in_frac=burn_in_frac, verbose=verbose)
        processed |= set(processed_here)

        if all_empty or not summary:
            rows.append({c: None for c in cols})
            continue

        for order in sorted(summary.keys()):
            s = summary[order]
            ic_low, ic_high = s["ci95_between_seeds"]
            rows.append({
                "type_perc": meta["type_perc"],
                "num_colors": meta["num_colors"],
                "dim": meta["dim"],
                "L": meta["L"],
                "Nt": meta["Nt"],
                "k": meta["k"],
                "rho": meta["rho"],
                "p0": p0_val,
                "order": order,
                "num_samples": s["n_seeds"],
                "p_mean": s["grand_mean"],
                "IC95": (ic_low, ic_high),
                "erro": s["sem_between_seeds"],
            })

    return pd.DataFrame(rows, columns=cols), processed

def process_with_guard(all_files,
                       out_dat_path: Path,
                       out_txt_path: Path,
                       burn_in_frac=0.2,
                       verbose=False,
                       path_hint: str = None):
    """
    - Compara quantos .json 'válidos' (que casam o regex P0_*_p0_*_seed_*.json) existem
      com o número de nomes já listados no process_names.txt.
    - Se (iguais) e existir o .dat, apenas carrega o .dat.
    - Senão, (re)processa tudo via build_dataframe_by_p0(..., path_hint=path_hint),
      salva o .dat e reescreve o process_names.txt com os nomes processados.
    """
    out_dat_path.parent.mkdir(parents=True, exist_ok=True)

    # Apenas os arquivos que casam o padrão (serão de fato processados)
    expected_total = sum(1 for f in all_files if parse_p0_from_filename(f) is not None)

    # Lidos previamente (se existir)
    if out_txt_path.exists():
        prev = [ln.strip() for ln in out_txt_path.read_text().splitlines() if ln.strip()]
        prev_set = set(prev)
    else:
        prev_set = set()

    # Guarda: se já processou tudo e o .dat existe, reaproveita
    if expected_total > 0 and len(prev_set) == expected_total and out_dat_path.exists():
        print(f"[INFO] {len(prev_set)} arquivos já processados (= {expected_total}). "
              f"Reaproveitando {out_dat_path.name}.")
        df = pd.read_csv(out_dat_path, sep="\t")
        return df, prev_set

    # (Re)processa TUDO (por p0, com meta extraída do path_hint)
    df, processed_set = build_dataframe_by_p0(
        all_files,
        burn_in_frac=burn_in_frac,
        verbose=verbose,
        path_hint=path_hint,   # <<< importante para extrair type_perc, L, Nt, k, rho do caminho
    )

    # Salva .dat (TSV)
    df.to_csv(out_dat_path, sep="\t", index=False, na_rep="Null")
    print("[INFO] .dat salvo em:", out_dat_path.resolve())

    # Atualiza process_names.txt com os nomes realmente processados (ordenados)
    out_txt_path.write_text("\n".join(sorted(processed_set)) + ("\n" if processed_set else ""))
    print(f"[INFO] process_names.txt atualizado com {len(processed_set)} nomes:", out_txt_path.resolve())

    return df, processed_set


def saving_data(all_data,
                output_data: Path,
                output_names: Path,
                burn_in_frac=0.20,
                verbose=False,
                path_hint: str = None):
    """
    Wrapper simples para chamar a guarda com os argumentos desejados.
    - all_data: lista de caminhos .json
    - output_data: Path para salvar o .dat (ex.: Path('../Data/bond_percolation')/'all_data.dat')
    - output_names: Path para salvar o .txt (ex.: Path('../Data/bond_percolation')/'process_names.txt')
    - path_hint: passe o 'path_data' para extrair os metaparâmetros do caminho
    """
    return process_with_guard(
        all_files=all_data,
        out_dat_path=output_data,
        out_txt_path=output_names,
        burn_in_frac=burn_in_frac,
        verbose=verbose,
        path_hint=path_hint,
    )
