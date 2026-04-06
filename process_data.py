
import re
from pathlib import Path
import os
import glob
import pandas as pd
from typing import Sequence, Optional, Literal, Dict, Any, List, Tuple
import numpy as np
import math
from collections import defaultdict
import json
import gc
from tqdm import tqdm

FLOAT = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

DIR_RE = re.compile(
    rf"""
    ^.*?
    (?P<type>[A-Za-z0-9]+)_percolation
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

PARAM_RE = re.compile(
    rf"""
    /(?P<type>[A-Za-z0-9]+)_percolation
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

# Busca os campos em qualquer posição do basename
# Mantém P0 em maiúsculo
RE_P0 = re.compile(rf'(?:^|_)P0_(?P<P0>{FLOAT})(?:_|\.json$)')

# p0 em minúsculo, sem case-insensitive,
# para NÃO capturar o P0_...
RE_p0 = re.compile(rf'(?:^|_)p0_(?P<p0>{FLOAT})(?:_|\.json$)')

# seed pode continuar flexível
RE_seed = re.compile(r'(?:^|_)seed_(?P<seed>\d+)(?:_|\.json$)', re.IGNORECASE)

DEFAULT_DESIRED_COLS = [
    'filename', 'P0', 'p0', 'order', 'p_mean', 'p_std', 'p_sem', 'shortest_path', 'S_perc'
]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_folder(folder_path):
    ensure_dir(folder_path)


def _sem_scalar(vals: List[float]) -> float:
    a = np.asarray(vals, dtype=float)
    if a.size <= 1:
        return 0.0
    return float(np.std(a, ddof=1) / np.sqrt(a.size))


def parse_data_dir(path: str):
    m = DIR_RE.match(Path(path).as_posix())
    if not m:
        return None
    g = m.groupdict()
    return {
        "type_perc": g["type"],
        "nc": int(g["nc"]),
        "dim": int(g["dim"]),
        "L": int(g["L"]),
        "Nt": int(g["Nt"]),
        "k": float(g["k"]),
        "rho": float(g["rho"]),
    }


def collect_param_combinations(
    root_dir: str,
    *,
    type_perc: Optional[str] = None,
    dir_re: re.Pattern = DIR_RE,
) -> List[Tuple[str, int, int, int, int, float, float]]:
    root_dir = os.path.normpath(root_dir)
    combos = set()

    for dirpath, _, _ in os.walk(root_dir):
        if os.path.basename(dirpath) != "data":
            continue

        path_norm = os.path.normpath(dirpath).replace(os.sep, "/")
        m = dir_re.match(path_norm)
        if not m:
            continue

        tp = m.group("type")
        if type_perc is not None and tp != type_perc:
            continue

        nc = int(m.group("nc"))
        dim = int(m.group("dim"))
        L = int(m.group("L"))
        NT = int(m.group("Nt"))
        k = float(m.group("k"))
        rho = float(m.group("rho"))
        combos.add((tp, nc, dim, L, NT, k, rho))

    combos = sorted(combos, key=lambda x: (x[0], x[1], x[2], x[4], x[5], x[6], x[3]))
    return combos


def _scalar_or_last(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(x[-1]) if len(x) > 0 else np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def _extract_seed_from_filename(fp: str) -> int | None:
    m = RE_seed.search(os.path.basename(fp))
    if not m:
        return None
    return int(m.group("seed"))


def parse_filename(path):
    name = Path(path).name

    mP0 = RE_P0.search(name)
    mp0 = RE_p0.search(name)
    ms = RE_seed.search(name)

    if not (mP0 and mp0 and ms):
        raise ValueError(f"Nome inválido: {path}")

    return {
        "P0": float(mP0.group("P0")),
        "p0": float(mp0.group("p0")),
        "seed": int(ms.group("seed")),
    }


def read_experiment_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    meta = raw.get("meta", {}) or {}
    results = raw.get("results", {}) or {}

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

    y = tail - mean
    n = y.size
    if max_lag is None:
        max_lag = min(n // 2, 1000)

    def _acf_fft(v):
        m = int(2 ** np.ceil(np.log2(2 * len(v) - 1)))
        fv = np.fft.rfft(v, n=m)
        acf = np.fft.irfft(fv * np.conj(fv), n=m)[:len(v)]
        return acf

    acf_raw = _acf_fft(y)
    acf_raw = acf_raw / acf_raw[0]

    tau_int = 0.5
    for k in range(1, max_lag + 1):
        if k >= len(acf_raw):
            break
        if acf_raw[k] <= 0:
            break
        tau_int += acf_raw[k]

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


def combine_tail_means(run_stats: list[dict], random_effects: bool = True):
    stats = [d for d in run_stats if (
        d is not None
        and math.isfinite(d.get('mean', float('nan')))
        and math.isfinite(d.get('sem', float('nan')))
        and d['sem'] > 0
    )]
    R = len(stats)
    if R == 0:
        return {'mean': float('nan'), 'se': float('nan'), 'method': 'FE', 'tau2': 0.0, 'R': 0}

    means = [d['mean'] for d in stats]
    vars_ = [d['sem']**2 for d in stats]
    w = [1.0/v for v in vars_]

    sumw = sum(w)
    m_fe = sum(wi * mi for wi, mi in zip(w, means)) / sumw
    se_fe = (1.0 / sumw) ** 0.5

    if not random_effects or R == 1:
        return {'mean': m_fe, 'se': se_fe, 'method': 'FE', 'tau2': 0.0, 'R': R}

    Q = sum(wi * (mi - m_fe) ** 2 for wi, mi in zip(w, means))
    c = sumw - sum(wi * wi for wi in w) / sumw
    tau2 = max(0.0, (Q - (R - 1)) / c) if c > 0 else 0.0

    w_star = [1.0 / (v + tau2) for v in vars_]
    sumw_star = sum(w_star)
    m_re = sum(wi * mi for wi, mi in zip(w_star, means)) / sumw_star
    se_re = (1.0 / sumw_star) ** 0.5

    return {'mean': m_re, 'se': se_re, 'method': 'RE', 'tau2': tau2, 'R': R}


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


def _mean_sem(arr_like):
    v = pd.to_numeric(pd.Series(arr_like), errors="coerce").dropna().to_numpy()
    n = v.size
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n > 0 else np.nan
    return mean, sem, n


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
        yw = y[i:i + w]
        sw = np.maximum(sem[i:i + w], eps)

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


def detect_equilibrium_start_with_errors(
    t, y, sem, w=40, lag=None, consec=6, z=2.0, chi2r_max=2.0,
    tail_frac=0.20, min_start_frac=0.05
):
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
    se_comb = np.sqrt(se_mu[lag:] ** 2 + se_mu[:-lag] ** 2)

    ok_change = dm <= (z * se_comb)
    ok_chi = (chi2r[lag:] <= chi2r_max) & (chi2r[:-lag] <= chi2r_max)
    ok = ok_change & ok_chi

    min_mu_idx = int(np.floor(min_start_frac * m))
    j_start = max(0, min_mu_idx - lag)

    run = 0
    for j in range(j_start, ok.size):
        run = run + 1 if ok[j] else 0
        if run >= consec:
            mu_idx = j + lag
            idx_t = mu_idx + (w - 1)
            idx_t = int(np.clip(idx_t, 0, n - 1))
            return idx_t

    tail_start = int(np.floor((1.0 - tail_frac) * n))
    tail_start = np.clip(tail_start, 0, n - 1)

    mu_tail, se_tail = weighted_mean_and_sem(y[tail_start:], sem[tail_start:])
    tol = z * np.sqrt(se_tail ** 2 + np.maximum(sem, 1e-15) ** 2)

    i0 = int(np.floor(min_start_frac * n))
    run = 0
    for i in range(i0, n):
        if abs(y[i] - mu_tail) <= tol[i]:
            run += 1
            if run >= consec:
                return int(i - consec + 1)
        else:
            run = 0

    return int(tail_start)


def _safe_float(x: Any) -> Any:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _sanitize_for_json(obj: Any) -> Any:
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


def _load_orders_new(fp: str) -> Optional[dict[int, dict]]:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            js = json.load(f)
    except json.JSONDecodeError:
        print(f"[warn] JSON inválido ignorado: {fp}")
        return None
    except OSError as ex:
        print(f"[warn] Falha ao abrir {fp}: {ex}")
        return None

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

        ord1 = int(digits)
        ordk = ord1 - 1

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


def _average_by_order_new(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    pts = np.stack([pt[:min_len_pt] for (_, pt) in series_pt], axis=0)

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


def _parse_fname(filepath: str) -> Optional[Tuple[float, float, int]]:
    name = os.path.basename(filepath)

    mP0 = RE_P0.search(name)
    mp0 = RE_p0.search(name)
    ms = RE_seed.search(name)

    if not (mP0 and mp0 and ms):
        return None

    try:
        return (
            float(mP0.group("P0")),
            float(mp0.group("p0")),
            int(ms.group("seed")),
        )
    except Exception:
        return None


def _discover_p0_values(all_jsons: List[str]) -> List[float]:
    """Descobre automaticamente os grupos de p0 presentes nos nomes dos arquivos."""
    found: set[float] = set()

    for fp in all_jsons:
        parsed = _parse_fname(fp)
        if parsed is None:
            continue
        _, p0_file, _ = parsed
        found.add(float(p0_file))

    return sorted(found)


def _parse_params_from_path(path: str) -> Optional[Tuple[str, int, int, float, int, float, int]]:
    m = PARAM_RE.search(path.replace("\\", "/"))
    if not m:
        return None

    type_perc = m.group("type")
    L = int(m.group("L"))
    Nt = int(m.group("Nt"))
    k = float(m.group("k"))
    nc = int(m.group("nc"))
    rho = float(m.group("rho"))
    dim = int(m.group("dim"))

    return (type_perc, L, Nt, k, nc, rho, dim)


def _parse_P0_p0_from_seed_used(seed_used: List[str]) -> Tuple[float, float]:
    P0_vals = []
    p0_vals = []

    for bn in seed_used:
        bn = os.path.basename(str(bn))

        mP0 = RE_P0.search(bn)
        mp0 = RE_p0.search(bn)

        if not (mP0 and mp0):
            continue

        try:
            P0_vals.append(float(mP0.group("P0")))
            p0_vals.append(float(mp0.group("p0")))
        except Exception:
            continue

    if len(P0_vals) == 0:
        return float("nan"), float("nan")

    return float(np.mean(P0_vals)), float(np.mean(p0_vals))

def _manifest_path(manifest_root: str | Path, rel_group_dir: str | Path) -> Path:
    return Path(manifest_root) / Path(rel_group_dir) / "manifest.json"


def _load_manifest(manifest_root: str | Path, rel_group_dir: str | Path) -> Dict[str, Any]:
    path = _manifest_path(manifest_root, rel_group_dir)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "processed_json_files": [],
        "n_processed_json_files": 0,
        "summary_file": None,
        "last_update": None,
    }


def _save_manifest(manifest_root: str | Path, rel_group_dir: str | Path, manifest: Dict[str, Any]) -> Path:
    path = _manifest_path(manifest_root, rel_group_dir)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(manifest), f, ensure_ascii=False, indent=2, allow_nan=False)
    return path

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
    raw_root: str,
    published_root: str,
    manifests_root: str,
    x_max: float | None = None,
    n_boot: int = 20000,
    rng_seed: int = 12345,
    window_roll: int | None = None,
    clear_data: bool = False,
    verbose: bool = True,
) -> Optional[str]:
    raw_root = os.path.abspath(raw_root)
    published_root = os.path.abspath(published_root)
    manifests_root = os.path.abspath(manifests_root)

    rel_group = os.path.join(
        f"{type_perc}_percolation",
        f"num_colors_{num_colors}",
        f"dim_{dim}",
        f"L_{L}",
        "NT_constant",
        f"NT_{NT}",
        f"k_{k:.1e}",
        f"rho_{rho:.4e}",
    )

    data_dir = os.path.join(raw_root, rel_group, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Pasta de dados não encontrada: {data_dir}")

    out_dir = os.path.join(published_root, rel_group)
    ensure_dir(out_dir)

    out_path = os.path.join(out_dir, "properties_mean_bundle.json")
    colors_path = os.path.join(out_dir, "colors_percolation.dat")

    manifest = _load_manifest(manifests_root, rel_group)

    if clear_data:
        if os.path.isfile(out_path):
            os.remove(out_path)
        if os.path.isfile(colors_path):
            os.remove(colors_path)
        manifest["processed_json_files"] = []
        manifest["n_processed_json_files"] = 0
        manifest["summary_file"] = None
        manifest["last_update"] = None
        if verbose:
            print(f"[clear_data] removido: {out_path}")
            print(f"[clear_data] removido: {colors_path}")

    all_jsons = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if verbose and all_jsons:
        print("[sample names check]")
        for fp in all_jsons[:5]:
            print("   ", os.path.basename(fp), "->", _parse_fname(fp))

    if not p0_list:
        p0_list = _discover_p0_values(all_jsons)
        if verbose:
            print(f"[auto-p0] grupos detectados em {data_dir}: {p0_list}")
    selected_p0 = [float(p) for p in p0_list]
    bad_name_files = []
    current_seed_files = []
    for fp in all_jsons:
        parsed = _parse_fname(fp)
        if parsed is None:
            bad_name_files.append(os.path.basename(fp))
            continue
        _, p0_file, _ = parsed
        for p0 in selected_p0:
            if abs(float(p0_file) - float(p0)) < 1e-12:
                current_seed_files.append(os.path.basename(fp))
                break

    current_seed_files = sorted(set(current_seed_files))

    if bad_name_files and verbose:
        print(f"[warn] {data_dir}: {len(bad_name_files)} arquivo(s) com nome fora do padrão flexível")
        for bn in bad_name_files[:10]:
            print(f"       - {bn}")
        if len(bad_name_files) > 10:
            print("       ...")

    processed_files = set(map(str, manifest.get("processed_json_files", [])))
    new_files = [bn for bn in current_seed_files if bn not in processed_files]

    if verbose:
        print(
            f"[group] {rel_group} | total_json={len(all_jsons)} "
            f"| parseable={len(current_seed_files)} | new={len(new_files)} | clear_data={clear_data}"
        )

    if (not clear_data) and os.path.isfile(out_path) and len(new_files) == 0:
        if verbose:
            print(f"[skip] atualizado: {out_path}")
        return out_path

    seed_used_set = set(current_seed_files)

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
            "p0_groups_detected": [float(p) for p in selected_p0],
        },
        "p0_groups": [],
    }

    colors_per_sample_all: List[int] = []

    for p0 in selected_p0:
        files = []
        for fp in all_jsons:
            parsed = _parse_fname(fp)
            if parsed is None:
                continue
            _, p0_file, _ = parsed
            if abs(float(p0_file) - float(p0)) < 1e-12:
                files.append(fp)

        if verbose:
            print(f"[debug] data_dir={data_dir} | p0={p0:.2f} | files={len(files)}")

        if not files:
            if verbose:
                print(f"[aviso] Sem arquivos para p0={p0:.2f}")
            continue

        per_order: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        per_order_seed_series: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

        seeds_set: set[int] = set()
        seeds_total_set: set[int] = set()
        seeds_non_perc_set: set[int] = set()
        valid_files = 0
        colors_per_sample_this_p0: List[int] = []

        for fp in files:
            orders = _load_orders_new(fp)
            if orders is None:
                continue

            n_orders = len(orders)
            colors_per_sample_this_p0.append(n_orders)
            colors_per_sample_all.append(n_orders)

            seed = _extract_seed_from_filename(fp)
            if seed is not None:
                seeds_total_set.add(seed)

            if not orders:
                if seed is not None:
                    seeds_non_perc_set.add(seed)
                continue

            valid_files += 1

            if seed is not None:
                seeds_set.add(seed)

            for ordk, data in orders.items():
                data_local = dict(data)

                if x_max is not None and data_local.get("t") is not None:
                    t = np.asarray(data_local["t"], dtype=float)
                    m = (t <= x_max)
                    data_local["t"] = t[m]
                    if data_local.get("pt") is not None:
                        data_local["pt"] = np.asarray(data_local["pt"], dtype=float)[m]
                    if data_local.get("nt") is not None:
                        data_local["nt"] = np.asarray(data_local["nt"], dtype=float)[m]

                if data_local.get("t") is not None and data_local.get("pt") is not None:
                    t_seed = np.asarray(data_local["t"], dtype=float)
                    pt_seed = np.asarray(data_local["pt"], dtype=float)
                    n0 = min(t_seed.size, pt_seed.size)
                    if n0 > 1:
                        per_order_seed_series[ordk].append((t_seed[:n0], pt_seed[:n0]))

                per_order[ordk].append(data_local)

        total_files_this_p0 = len(colors_per_sample_this_p0)

        if verbose:
            print(
                f"[debug] p0={p0:.2f} | total_files={total_files_this_p0} "
                f"| valid_files={valid_files} | non_perc={total_files_this_p0 - valid_files}"
            )

        mean_by_order: Dict[int, Dict[str, Any]] = {}
        for ordk, lst in per_order.items():
            mean_by_order[ordk] = _average_by_order_new(lst)
            mean_by_order[ordk]["n_samples_perc"] = int(len(lst))
            mean_by_order[ordk]["n_samples_total"] = int(total_files_this_p0)
            mean_by_order[ordk]["n_samples_non_perc"] = int(total_files_this_p0 - len(lst))

        series_mean = []
        for ordk in sorted(mean_by_order.keys()):
            d = mean_by_order[ordk]
            if not d.get("time") or not d.get("pt_mean") or not d.get("pt_sem"):
                continue
            t = np.asarray(d["time"], dtype=float)
            pt = np.asarray(d["pt_mean"], dtype=float)
            pt_sem = np.asarray(d["pt_sem"], dtype=float)
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

        orders_blocks = [{"order_percolation": int(ordk), "data": mean_by_order[ordk]} for ordk in sorted(mean_by_order.keys())]

        colors_arr = np.asarray(colors_per_sample_this_p0, dtype=float)
        if colors_arr.size > 0:
            nc_mean = float(np.mean(colors_arr))
            nc_std = float(np.std(colors_arr, ddof=1)) if colors_arr.size > 1 else 0.0
            nc_err = float(nc_std / np.sqrt(colors_arr.size))
        else:
            nc_mean = float("nan")
            nc_std = float("nan")
            nc_err = float("nan")

        p0_group = {
            "p0_value": float(p0),
            "num_seeds": len(seeds_set),
            "num_seeds_total": len(seeds_total_set),
            "num_seeds_non_percolating": len(seeds_non_perc_set),
            "num_samples_total": int(total_files_this_p0),
            "num_samples_percolating_any_order": int(valid_files),
            "num_samples_non_percolating": int(total_files_this_p0 - valid_files),
            "orders": orders_blocks,
            "pc_sop": {
                "mean": _safe_float(pc_mean),
                "std_boot": _safe_float(pc_sem),
                "n_seeds": len(seeds_set),
                "n_boot": int(n_boot),
                "t0_global": _safe_float(t0_global),
                "pc_method": "combine orders of (per-seed autocorr + random-effects)",
            },
            "colors": {
                "Nsamples": int(colors_arr.size),
                "nc": _safe_float(nc_mean),
                "nc_std": _safe_float(nc_std),
                "nc_err": _safe_float(nc_err),
            },
        }

        bundle["p0_groups"].append(p0_group)

    bundle["meta"]["seed_used"] = sorted(seed_used_set)
    bundle = _sanitize_for_json(bundle)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2, allow_nan=False)

    with open(colors_path, "w", encoding="utf-8") as f:
        for val in colors_per_sample_all:
            f.write(f"{int(val)}\n")

    manifest.update({
        "group_relpath": rel_group,
        "data_dir": data_dir,
        "processed_json_files": sorted(current_seed_files),
        "n_processed_json_files": len(current_seed_files),
        "summary_file": out_path,
        "last_update": pd.Timestamp.utcnow().isoformat(),
    })
    _save_manifest(manifests_root, rel_group, manifest)

    return out_path

def build_properties_dataframe(published_root: str, output_file: str | Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    published_root = os.path.abspath(published_root)
    output_file = Path(output_file)

    bundle_files = []
    for dirpath, _, filenames in os.walk(published_root):
        if "properties_mean_bundle.json" in filenames:
            bundle_files.append(os.path.join(dirpath, "properties_mean_bundle.json"))

    for bundle_path in sorted(bundle_files):
        parsed = _parse_params_from_path(os.path.dirname(bundle_path))
        if parsed is None:
            parsed = _parse_params_from_path(bundle_path)
        if parsed is None:
            continue

        type_perc, L, Nt, k, nc, rho, dim = parsed

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
            colors_block = g.get("colors", {}) if isinstance(g.get("colors", {}), dict) else {}
            N_samples = int(
                g.get("num_samples_total", colors_block.get("Nsamples", g.get("num_seeds_total", g.get("num_seeds", 0)))) or 0
            )

            orders = g.get("orders", [])
            if not isinstance(orders, list) or len(orders) == 0:
                continue

            for ob in orders:
                ordk = ob.get("order_percolation", None)
                if ordk is None:
                    continue

                order = int(ordk) + 1
                d = ob.get("data", {}) or {}
                N_samples_perc = int(d.get("n_samples_perc", 0) or 0)

                pc_block = d.get("pc_sop", {}) if isinstance(d.get("pc_sop", {}), dict) else {}
                p_mean = _safe_float(pc_block.get("mean", float("nan")))
                p_err = _safe_float(pc_block.get("std_boot", float("nan")))

                shortest_path = _safe_float(d.get("shortest_path_lin_mean", float("nan")))
                shortest_path_err = _safe_float(d.get("shortest_path_lin_sem", float("nan")))

                S_perc = _safe_float(d.get("M_size_mean", float("nan")))
                S_perc_err = _safe_float(d.get("M_size_sem", float("nan")))

                rows.append({
                    "type_perc": type_perc,
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
                    "N_samples_perc": N_samples_perc,
                    "p_mean": p_mean,
                    "p_err": p_err,
                    "shortest_path": shortest_path,
                    "shortest_path_err": shortest_path_err,
                    "S_perc": S_perc,
                    "S_perc_err": S_perc_err,
                })

    cols = [
        "type_perc", "dim", "L", "Nt", "k", "nc", "rho", "p0", "P0",
        "order", "N_samples", "N_samples_perc",
        "p_mean", "p_err",
        "shortest_path", "shortest_path_err",
        "S_perc", "S_perc_err",
    ]

    df = pd.DataFrame(rows, columns=cols)

    if not df.empty:
        df = df.sort_values(
            by=["type_perc", "dim", "nc", "rho", "k", "Nt", "L", "p0", "order"]
        ).reset_index(drop=True)

    ensure_dir(output_file.parent)
    df.to_csv(output_file, index=False, sep=" ")
    return df


def build_colors_dataframe(published_root: str, output_file: str | Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    published_root = os.path.abspath(published_root)
    output_file = Path(output_file)

    bundle_files = []
    for dirpath, _, filenames in os.walk(published_root):
        if "properties_mean_bundle.json" in filenames:
            bundle_files.append(os.path.join(dirpath, "properties_mean_bundle.json"))

    for bundle_path in sorted(bundle_files):
        parsed = _parse_params_from_path(os.path.dirname(bundle_path))
        if parsed is None:
            parsed = _parse_params_from_path(bundle_path)
        if parsed is None:
            continue

        type_perc, L, Nt, k, nc_model, rho, dim = parsed

        try:
            with open(bundle_path, "r", encoding="utf-8") as f:
                js = json.load(f)
        except Exception:
            continue

        p0_groups = js.get("p0_groups", [])
        if not isinstance(p0_groups, list) or len(p0_groups) == 0:
            continue

        for g in p0_groups:
            p0_val = _safe_float(g.get("p0_value", float("nan")))
            cstats = g.get("colors", {}) if isinstance(g.get("colors", {}), dict) else {}

            rows.append({
                "L": L,
                "dim": dim,
                "Nt": Nt,
                "k": k,
                "num_colors": nc_model,
                "p0": p0_val,
                "Nsamples": int(cstats.get("Nsamples", 0) or 0),
                "rho": rho,
                "nc": _safe_float(cstats.get("nc", float("nan"))),
                "nc_err": _safe_float(cstats.get("nc_err", float("nan"))),
                "nc_std": _safe_float(cstats.get("nc_std", float("nan"))),
            })

    cols = ["L", "dim", "Nt", "k", "num_colors", "p0", "Nsamples", "rho", "nc", "nc_err", "nc_std"]
    df = pd.DataFrame(rows, columns=cols)

    if not df.empty:
        df = df.sort_values(
            by=["dim", "num_colors", "rho", "k", "Nt", "L", "p0"]
        ).reset_index(drop=True)

    ensure_dir(output_file.parent)
    df.to_csv(output_file, index=False, sep=" ")
    return df

def process_all_data(
    clear_data: bool = False,
    *,
    sop_root: str = "../SOP_data",
    p0_lst: Optional[List[float]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    if p0_lst is not None:
        p0_lst = [float(p) for p in p0_lst]
        if len(p0_lst) == 0:
            p0_lst = None

    sop_root = os.path.abspath(sop_root)
    raw_root = os.path.join(sop_root, "raw")
    published_root = os.path.join(sop_root, "published")
    manifests_root = os.path.join(sop_root, "manifests")

    ensure_dir(raw_root)
    ensure_dir(published_root)
    ensure_dir(manifests_root)

    all_parms = collect_param_combinations(raw_root)

    iterator = tqdm(
        all_parms,
        desc="Processando conjuntos",
        ncols=120,
        dynamic_ncols=False,
        leave=True,
    )

    for tp, nc, DIM, L, NT, K, RHO in iterator:
        iterator.set_postfix_str(
            f"{tp} nc={nc} dim={DIM} L={L} NT={NT} k={K:.1e} rho={RHO:.4e}"
        )
        compute_means_for_folder(
            type_perc=tp,
            num_colors=nc,
            dim=DIM,
            L=L,
            NT=NT,
            k=K,
            rho=RHO,
            p0_list=p0_lst,
            raw_root=raw_root,
            published_root=published_root,
            manifests_root=manifests_root,
            clear_data=clear_data,
            verbose=verbose,
        )

    if verbose:
        print("Processamento finalizado. Construindo SOP_data/all_data.dat ...")

    df = build_properties_dataframe(
        published_root=published_root,
        output_file=os.path.join(sop_root, "all_data.dat"),
    )

    if verbose:
        print(f"[write] {os.path.join(sop_root, 'all_data.dat')} ({len(df)} linhas)")
        print("Construindo SOP_data/all_colors.dat ...")

    df_colors = build_colors_dataframe(
        published_root=published_root,
        output_file=os.path.join(sop_root, "all_colors.dat"),
    )

    if verbose:
        print(f"[write] {os.path.join(sop_root, 'all_colors.dat')} ({len(df_colors)} linhas)")
    return df

if __name__ == "__main__":
    process_all_data(clear_data=False)
