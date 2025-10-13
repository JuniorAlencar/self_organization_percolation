import re, os, json, glob, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# =============================================================
#  Regex helpers (stable)
# =============================================================
# Accepts k/rho in normal float or scientific notation (e.g., 1.0e-04)
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

FNAME_RE = re.compile(
    r"P0_(?P<P0>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)_p0_(?P<p0>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)_seed_(?P<seed>\d+)\.json$",
    re.IGNORECASE,
)

_FLOAT_DIR_RE = re.compile(r"^(?P<key>k|rho)_(?P<val>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)$", re.I)

# =============================================================
#  Core parsing utilities
# =============================================================

def parse_params_from_path(path: str):
    """Extract type_perc, num_colors, dim, L, Nt, k, rho from a '.../data' path."""
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


def _find_numeric_subdir(base: Path, key: str, target: float,
                         rel_tol: float = 1e-12, abs_tol: float = 1e-15) -> Path | None:
    """Find subdir named '{key}_<number>' numerically close to target."""
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

# =============================================================
#  I/O + statistics
# =============================================================

def read_orders_one_file(file_path):
    """
    Parse one JSON with a "results" list and return a list of tuples
    (order, pt_array, nt_array_or_None, M_size_or_None).
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

            n = min(len(p), len(n_arr)) if n_arr is not None else len(p)
            if n <= 0:
                continue
            p = p[:n]
            n_arr = n_arr[:n] if n_arr is not None else None

            m_size = d.get("M_size", None)
            try:
                m_size = float(m_size) if m_size is not None else None
            except Exception:
                m_size = None

            out.append((int(order), p, n_arr, m_size))
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

# =============================================================
#  Dataset builders
# =============================================================

BASE_COLS = [
    "type_perc","num_colors","dim","L","Nt","k","rho","p0","order",
    "num_samples","num_sample_perc",
    "pt_mean","pt_erro","nt_mean","nt_erro",
    "M_size_mean","M_size_erro",
    "perc_rate",
]

DEPRECATED_COLS = [
    "perc_ci_low","perc_ci_high",
    "pt_mean_uncond","pt_erro_uncond",
    "nt_mean_uncond","nt_erro_uncond",
]

KEY_COLS = ["type_perc","num_colors","dim","L","Nt","k","rho","p0","order"]


def _normalize_names(iterable):
    out = set()
    for x in iterable:
        b = os.path.basename(x).strip()
        if b:
            out.add(b)
    return out


def parse_p0_from_filename(path):
    m = FNAME_RE.search(os.path.basename(path))
    if not m:
        return None
    try:
        return float(m.group("p0"))
    except Exception:
        return None


def summarize_multi_seed_by_order(files, burn_in_frac=0.2, verbose=False):
    per_order_pt, per_order_nt, per_order_msize = {}, {}, {}
    any_seen = False
    processed_here = set()

    for jf in files:
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

            if m_size is not None and np.isfinite(m_size):
                per_order_msize.setdefault(order, []).append(float(m_size))

    if not any_seen and not per_order_pt and not per_order_nt and not per_order_msize:
        return {}, True, processed_here

    summary = {}
    orders = sorted(set(list(per_order_pt.keys()) + list(per_order_nt.keys()) + list(per_order_msize.keys())))
    for order in orders:
        mp = np.asarray(per_order_pt.get(order, []), float)
        Sp = len(mp)
        pt_mean = float(mp.mean()) if Sp > 0 else np.nan
        pt_sem  = float(mp.std(ddof=1)/np.sqrt(Sp)) if Sp > 1 else (0.0 if Sp == 1 else np.nan)

        mn = np.asarray(per_order_nt.get(order, []), float)
        Sn = len(mn)
        nt_mean = float(mn.mean()) if Sn > 0 else np.nan
        nt_sem  = float(mn.std(ddof=1)/np.sqrt(Sn)) if Sn > 1 else (0.0 if Sn == 1 else np.nan)

        ms = np.asarray(per_order_msize.get(order, []), float)
        Sm = len(ms)
        msize_mean = float(ms.mean()) if Sm > 0 else np.nan
        msize_sem  = float(ms.std(ddof=1)/np.sqrt(Sm)) if Sm > 1 else (0.0 if Sm == 1 else np.nan)

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


def build_dataframe_by_p0(all_files, burn_in_frac=0.2, verbose=False, path_hint: str | None = None):
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

    cols = BASE_COLS
    rows, processed = [], set()

    if not groups:
        return pd.DataFrame([{c: None for c in cols}])[cols], processed

    for p0_val in sorted(groups.keys()):
        p0_fmt = round(float(p0_val), 1)
        summary, all_empty, processed_here = summarize_multi_seed_by_order(
            groups[p0_val], burn_in_frac=burn_in_frac, verbose=verbose
        )
        processed |= set(processed_here)
        N = int(len(processed_here))

        def append_row(order, M, pt_m, pt_e, nt_m, nt_e, msz_m, msz_e):
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
                append_row(order, s["n_seeds_contributed"], s["pt_mean"], s["pt_sem_between"],
                           s["nt_mean"], s["nt_sem_between"], s["M_size_mean"], s["M_size_sem_between"])
            else:
                append_row(order, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    df = pd.DataFrame(rows, columns=cols)

    df = df.drop_duplicates(subset=KEY_COLS, keep="last")

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

    df = df.drop(columns=[c for c in DEPRECATED_COLS if c in df.columns], errors="ignore")
    return df, processed


# =============================================================
#  Public processing APIs
# =============================================================

def process_with_guard(all_files,
                       out_dat_path: Path,
                       out_txt_path: Path,
                       burn_in_frac=0.2,
                       verbose=False,
                       path_hint: str | None = None,
                       force_recompute: bool = False):
    """Process a list of seed JSONs into a per-(p0,order) .dat and a TSV of per-file stats."""
    def _read_prev_names_first_column(path: Path) -> set[str]:
        if not path.exists():
            return set()
        lines = path.read_text().splitlines()
        if not lines:
            return set()
        first = lines[0]
        if "\t" in first and first.split("\t", 1)[0].strip().lower() == "filename":
            out = set()
            for ln in lines[1:]:
                if not ln.strip():
                    continue
                out.add(os.path.basename(ln.split("\t", 1)[0].strip()))
            return out
        return _normalize_names(lines)

    def _per_file_rows(file_path: str, meta: dict, n_orders: int, burn_in_frac=0.2):
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
        by_order = {}
        for order, p_arr, n_arr, m_size in entries:
            by_order[int(order)] = (
                np.asarray(p_arr, float) if p_arr is not None else np.array([], float),
                None if n_arr is None else np.asarray(n_arr, float),
                m_size if (m_size is None or np.isfinite(m_size)) else None,
            )
        for order in range(1, int(n_orders) + 1):
            if order in by_order:
                p_arr, n_arr, m_size = by_order[order]
                n = len(p_arr)
                start = int(burn_in_frac * n) if n > 0 else 0
                if n > 0 and start < n:
                    m_pt, _, _, _ = sem_acf(p_arr[start:])
                    pt_mean, pt_sem = float(m_pt), 0.0
                else:
                    pt_mean, pt_sem = (np.nan, np.nan)
                if n_arr is not None and n_arr.size > 0:
                    ns = n_arr[start:] if start < n_arr.size else np.array([], float)
                    if ns.size > 0:
                        m_nt, _, _, _ = sem_acf(ns)
                        nt_mean, nt_sem = float(m_nt), 0.0
                    else:
                        nt_mean, nt_sem = (np.nan, np.nan)
                else:
                    nt_mean, nt_sem = (np.nan, np.nan)
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

    # Determine real data_dir and normalize outputs inside it
    data_dir = None
    if all_files:
        data_dir = Path(all_files[0]).parent
    else:
        p = Path(out_dat_path)
        data_dir = p.parent if p.suffix.lower() == ".dat" else p
    data_dir.mkdir(parents=True, exist_ok=True)

    out_dat_path = data_dir / Path(out_dat_path).name
    out_txt_path = data_dir / Path(out_txt_path).name

    if not all_files:
        all_files = [str(x) for x in sorted(data_dir.glob("*.json"))]

    out_dat_path.parent.mkdir(parents=True, exist_ok=True)

    expected_set = _normalize_names([f for f in all_files if FNAME_RE.search(os.path.basename(f))])
    prev_set = _read_prev_names_first_column(out_txt_path) if out_txt_path.exists() else set()

    if (not force_recompute) and expected_set and (expected_set == prev_set) and out_dat_path.exists():
        if verbose:
            print(f"[INFO] Up-to-date: {len(prev_set)} names == {len(expected_set)} JSONs: {data_dir}")
        df = pd.read_csv(out_dat_path, sep="\t", na_values=["Null"])
        return df, prev_set

    if verbose:
        print(f"[INFO] Reprocessing: force={force_recompute} dir={data_dir}")
    df, processed_set = build_dataframe_by_p0(
        all_files, burn_in_frac=burn_in_frac, verbose=verbose, path_hint=str(data_dir)
    )
    df.to_csv(out_dat_path, sep="\t", index=False, na_rep="Null")

    # Build TSV per-file/ordem (modern format with header)
    meta = parse_params_from_path(str(data_dir)) or {"type_perc": None, "num_colors": None, "dim": None, "L": None, "Nt": None, "k": None, "rho": None}
    n_orders = meta.get("num_colors") or 1
    by_basename = {os.path.basename(p): p for p in all_files}
    names_rows = []
    new_set = expected_set if expected_set else _normalize_names(processed_set)
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
    names_df.to_csv(out_txt_path, sep="\t", index=False, na_rep="Null")

    return df, new_set


def saving_data(all_data,
                output_data: Path,
                output_names: Path,
                burn_in_frac=0.20,
                verbose=False,
                path_hint: str | None = None,
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

# =============================================================
#  Discovery helpers (used by recursive crawlers)
# =============================================================

def iter_all_data_dirs(base_root: str = "../Data"):
    """Yield every directory matching .../data under base_root that fits our layout."""
    root = Path(base_root)
    if not root.exists():
        return
    for data_dir in root.rglob("data"):
        if not data_dir.is_dir():
            continue
        meta = parse_params_from_path(str(data_dir))
        if meta is None:
            continue
        yield data_dir, meta


def list_rho_values(type_perc: str, num_colors: int, dim: int, L: int, Nt: int, k: float,
                    base_root: str = "../Data", rel_tol: float = 1e-12, abs_tol: float = 1e-15):
    base = (Path(base_root) / f"{type_perc}_percolation" / f"num_colors_{num_colors}" / f"dim_{dim}" /
            f"L_{L}" / "NT_constant" / f"NT_{Nt}")

    if not base.exists():
        return []

    rhos = []
    for data_dir in base.glob("k_*/rho_*/data"):
        if not data_dir.is_dir():
            continue
        m = PARAMS_RE.search(str(data_dir.as_posix()))
        if not m:
            continue
        gd = m.groupdict()
        if gd["type_perc"] != type_perc or int(gd["num_colors"]) != num_colors or int(gd["dim"]) != dim or \
           int(gd["L"]) != L or int(gd["Nt"]) != Nt:
            continue
        k_here = float(gd["k"])
        if not math.isclose(k_here, float(k), rel_tol=rel_tol, abs_tol=abs_tol):
            continue
        rhos.append(float(gd["rho"]))
    return sorted(set(rhos))


def _align_to_union(df: pd.DataFrame, union_cols: list) -> pd.DataFrame:
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
    """
    out_dir = Path(base_root) / f"{type_perc}_percolation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"all_data_{dim}D.dat"

    rho_values = list_rho_values(type_perc, num_colors, dim, L, Nt, k, base_root=base_root)
    if not rho_values:
        if out_path.exists():
            try:
                acc = pd.read_csv(out_path, sep="\t", na_values=["Null"])
                acc = acc.drop(columns=[c for c in DEPRECATED_COLS if c in acc.columns], errors="ignore")
                return acc
            except Exception:
                pass
        return pd.DataFrame(columns=BASE_COLS)

    base_nt = (Path(base_root) / f"{type_perc}_percolation" / f"num_colors_{num_colors}" /
               f"dim_{dim}" / f"L_{L}" / "NT_constant" / f"NT_{Nt}")

    k_dir = _find_numeric_subdir(base_nt, "k", k, rel_tol=rel_tol, abs_tol=abs_tol)
    if k_dir is None:
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
            continue
        data_dir = rho_dir / "data"
        candidates = [data_dir / "all_data.dat", data_dir / f"all_data_{dim}D.dat"]
        fpath = _first_existing(candidates)
        if fpath is None:
            continue
        try:
            df_rho = pd.read_csv(fpath, sep="\t", na_values=["Null"])
        except Exception:
            continue
        df_rho = df_rho.drop(columns=[c for c in DEPRECATED_COLS if c in df_rho.columns], errors="ignore")
        for c in BASE_COLS:
            if c not in df_rho.columns:
                df_rho[c] = np.nan
        dfs.append(df_rho)

    if not dfs:
        if out_path.exists():
            try:
                acc = pd.read_csv(out_path, sep="\t", na_values=["Null"])
                acc = acc.drop(columns=[c for c in DEPRECATED_COLS if c in acc.columns], errors="ignore")
                return acc
            except Exception:
                pass
        return pd.DataFrame(columns=BASE_COLS)

    new_block = pd.concat(dfs, ignore_index=True, sort=False)

    num_cols = ["num_colors","dim","L","Nt","k","rho","p0","order",
                "num_samples","num_sample_perc","pt_mean","pt_erro","nt_mean","nt_erro",
                "M_size_mean","M_size_erro","perc_rate"]
    for c in num_cols:
        if c in new_block.columns:
            new_block[c] = pd.to_numeric(new_block[c], errors="coerce")
    if "p0" in new_block.columns:
        new_block["p0"] = new_block["p0"].round(1)

    new_block = new_block.drop_duplicates(subset=KEY_COLS, keep="last")

    if out_path.exists():
        try:
            acc = pd.read_csv(out_path, sep="\t", na_values=["Null"])
            acc = acc.drop(columns=[c for c in DEPRECATED_COLS if c in acc.columns], errors="ignore")
        except Exception:
            acc = pd.DataFrame(columns=BASE_COLS)
    else:
        acc = pd.DataFrame(columns=BASE_COLS)

    union_cols = list(dict.fromkeys(list(BASE_COLS) + list(acc.columns) + list(new_block.columns)))
    acc_aligned = _align_to_union(acc, union_cols)
    new_aligned = _align_to_union(new_block, union_cols)

    merged = pd.concat([acc_aligned, new_aligned], ignore_index=True, sort=False)
    if "type_perc" in merged.columns:
        merged["type_perc"] = merged["type_perc"].astype(str)
    merged = merged.drop_duplicates(subset=KEY_COLS, keep="last")

    sort_cols = ["type_perc","num_colors","dim","L","Nt","k","rho","p0","order"]
    sort_cols = [c for c in sort_cols if c in merged.columns]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)

    merged.to_csv(out_path, sep="\t", index=False, na_rep="Null")
    print("Updated:", out_path.resolve())
    return merged

# =============================================================
#  NEW: Recursive crawlers
# =============================================================

def _latest_mtime_json(data_dir: Path) -> float:
    latest = 0.0
    for p in data_dir.glob("*.json"):
        try:
            ts = p.stat().st_mtime
            if ts > latest:
                latest = ts
        except Exception:
            pass
    return latest


def crawl_and_process(base_root: str = "../Data", burn_in_frac: float = 0.20,
                      verbose: bool = True, force_recompute: bool = False,
                      workers: int | None = None):
    """
    1) Recursively find EVERY '.../data' folder under base_root;
    2) Run saving_data() inside it (produces all_data.dat + process_names.txt (TSV style));
    3) Return a set of unique (type_perc, num_colors, dim, L, Nt, k) tuples discovered.

    Performance tweaks:
      - Quick-skip by mtime: if all_data.dat is newer than every *.json and force_recompute=False,
        we skip the folder without reading/process_names.txt.
      - Optional parallelism (IO-bound) via ThreadPoolExecutor.
    """
    tuples = set()

    jobs = []
    for data_dir, meta in iter_all_data_dirs(base_root):
        jobs.append((data_dir, meta))

    def _process_dir(job):
        data_dir, meta = job
        try:
            out_dat = data_dir / "all_data.dat"
            out_txt = data_dir / "process_names.txt"
            if (not force_recompute) and out_dat.exists():
                latest_json = _latest_mtime_json(data_dir)
                try:
                    out_mtime = out_dat.stat().st_mtime
                except Exception:
                    out_mtime = 0.0
                # If output newer than all inputs and names file exists -> fast skip
                if out_mtime >= latest_json and out_txt.exists():
                    if verbose:
                        print(f"[SKIP] Up-to-date by mtime: {data_dir}")
                    return (meta["type_perc"], meta["num_colors"], meta["dim"], meta["L"], meta["Nt"], meta["k"])  # still register tuple

            all_json = sorted([str(p) for p in data_dir.glob("*.json")])
            if verbose:
                print("[DIR]", data_dir)
            saving_data(
                all_data=all_json,
                output_data=out_dat,
                output_names=out_txt,
                burn_in_frac=burn_in_frac,
                verbose=verbose,
                path_hint=str(data_dir),
                force_recompute=force_recompute,
            )
            return (meta["type_perc"], meta["num_colors"], meta["dim"], meta["L"], meta["Nt"], meta["k"])  # no rho
        except Exception as e:
            print(f"[WARN] Failed in {data_dir}: {e}")
            return None

    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_dir, j) for j in jobs]
            for fut in as_completed(futs):
                res = fut.result()
                if res:
                    tuples.add(res)
    else:
        for j in jobs:
            res = _process_dir(j)
            if res:
                tuples.add(res)

    return tuples


def _global_join_is_up_to_date(base_root: str, type_perc: str, dim: int, combos: list[tuple]) -> bool:
    out_path = Path(base_root) / f"{type_perc}_percolation" / f"all_data_{dim}D.dat"
    if not out_path.exists():
        return False
    try:
        out_mtime = out_path.stat().st_mtime
    except Exception:
        return False
    # If global is newer than every constituent all_data.dat, we can skip
    latest_constituent = 0.0
    for (_, _nc, _dim, L, Nt, k) in combos:
        if _dim != dim:
            continue
        base = Path(base_root) / f"{type_perc}_percolation" / f"num_colors_{_nc}" / f"dim_{_dim}" / f"L_{L}" / "NT_constant" / f"NT_{Nt}"
        k_dir = _find_numeric_subdir(base, "k", k)
        if not k_dir:
            continue
        for rho_dir in k_dir.glob("rho_*/data"):
            f = rho_dir / "all_data.dat"
            if f.exists():
                try:
                    mt = f.stat().st_mtime
                    if mt > latest_constituent:
                        latest_constituent = mt
                except Exception:
                    pass
    return out_mtime >= latest_constituent and latest_constituent > 0.0


def crawl_join_all_data(base_root: str = "../Data", verbose: bool = True,
                        rel_tol: float = 1e-12, abs_tol: float = 1e-15):
    """
    Discover all (type_perc, num_colors, dim, L, Nt, k) combinations under base_root and
    call join_all_data() for each, updating the global all_data_{dim}D.dat per type_perc.

    Performance tweaks:
      - Fast global skip: if the per-(type_perc, dim) output is newer than all involved
        constituent all_data.dat files, we skip its join entirely.
    """
    combos = set()
    for data_dir, meta in iter_all_data_dirs(base_root):
        combos.add((meta["type_perc"], meta["num_colors"], meta["dim"], meta["L"], meta["Nt"], meta["k"]))

    # group by (type_perc, dim) for the global skip check
    by_td = {}
    for c in combos:
        td = (c[0], c[2])
        by_td.setdefault(td, []).append(c)

    for (type_perc, dim), group in sorted(by_td.items()):
        if _global_join_is_up_to_date(base_root, type_perc, dim, group):
            if verbose:
                print(f"[SKIP] Global join up-to-date: {type_perc} dim={dim}")
            continue
        # run join per combo in this (type_perc, dim)
        for (t, num_colors, d, L, Nt, k) in sorted(group):
            try:
                join_all_data(t, num_colors, d, L, Nt, k,
                              base_root=base_root, rel_tol=rel_tol, abs_tol=abs_tol)
            except Exception as e:
                print(f"[WARN] join failed for {(t, num_colors, d, L, Nt, k)}: {e}")

# =============================================================
#  CLI entry (optional)
# =============================================================
# =============================================================
#  CLI helpers + programmatic entry
# =============================================================

def run_processing_data(base_root: str = "../Data", *, force_recompute: bool = True,
                        burn_in: float = 0.20, verbose: bool = True):
    """Programmatic entry point (safe for notebooks)."""
    print("=== STEP 1: per-folder processing ===")
    _ = crawl_and_process(base_root=base_root, burn_in_frac=burn_in,
                          verbose=verbose, force_recompute=force_recompute)

    print("=== STEP 2: global joins per (type,dim) ===")
    crawl_join_all_data(base_root=base_root, verbose=verbose)

    print("Done.")


def run_processing_data_cli(argv: list[str] | None = None):
    """Command-line entry that plays nice with Jupyter if argv=[] is passed."""
    import argparse
    ap = argparse.ArgumentParser(description="Recursive processing for SOP outputs")
    ap.add_argument("--base_root", default="../Data", help="Root folder with percolation outputs")
    ap.add_argument("--no_force", action="store_true", help="Do not force recompute (default: force)")
    ap.add_argument("--burn_in", type=float, default=0.20, help="Burn-in fraction for pt/nt")
    ap.add_argument("--quiet", action="store_true", help="Less verbose logs")
    # In notebooks, Jupyter passes its own args; use parse_known_args to avoid crashes
    if argv is None:
        args, _unknown = ap.parse_known_args()
    else:
        args = ap.parse_args(argv)

    force = not args.no_force
    verbose = not args.quiet
    run_processing_data(base_root=args.base_root, force_recompute=force,
                        burn_in=args.burn_in, verbose=verbose)


# =============================================================
#  CLI entry (optional)
# =============================================================
if __name__ == "__main__":
    run_processing_data_cli()
