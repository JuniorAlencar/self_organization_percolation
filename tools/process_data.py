
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
    /fT_constant
    /fT_(?P<fT>{FLOAT})
    /c_(?P<c>{FLOAT})
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
    /fT_constant
    /fT_(?P<fT>{FLOAT})
    /c_(?P<c>{FLOAT})
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

DEFAULT_SIZE_COLS = [
    "type_perc", "dim", "L", "f_T", "c", "nc", "rho", "p0", "P0",
    "order", "N_samples", "N_samples_perc",
    "shortest_path", "shortest_path_err",
    "S_perc", "S_perc_err",
    "shortest_path_preteq", "shortest_path_preteq_err",
    "S_perc_preteq", "S_perc_preteq_err",
    "shortest_path_posteq", "shortest_path_posteq_err",
    "S_perc_posteq", "S_perc_posteq_err",
]



def _float_close(a: float, b: float, *, rel_tol: float = 1e-12, abs_tol: float = 1e-15) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    except Exception:
        return False


def _build_group_relpath(
    type_perc: str,
    num_colors: int,
    dim: int,
    L: int,
    f_T: float,
    c: float,
    rho: float,
) -> str:
    return os.path.join(
        f"{type_perc}_percolation",
        f"num_colors_{num_colors}",
        f"dim_{dim}",
        f"L_{L}",
        "fT_constant",
        f"fT_{f_T:.6e}",
        f"c_{c:.6e}",
        f"rho_{rho:.4e}",
    )


def _resolve_existing_group_relpath(
    raw_root: str,
    type_perc: str,
    num_colors: int,
    dim: int,
    L: int,
    f_T: float,
    c: float,
    rho: float,
) -> str:
    """Resolve o diretório real do grupo sem depender da formatação exata dos floats."""
    raw_root = os.path.abspath(raw_root)

    formatted = _build_group_relpath(type_perc, num_colors, dim, L, f_T, c, rho)
    formatted_data_dir = os.path.join(raw_root, formatted, "data")
    if os.path.isdir(formatted_data_dir):
        return formatted

    base = os.path.join(
        raw_root,
        f"{type_perc}_percolation",
        f"num_colors_{num_colors}",
        f"dim_{dim}",
        f"L_{L}",
        "fT_constant",
    )

    if not os.path.isdir(base):
        return formatted

    candidates = sorted(glob.glob(os.path.join(base, "fT_*", "c_*", "rho_*", "data")))
    matches: List[str] = []

    for data_dir in candidates:
        parsed = parse_data_dir(data_dir)
        if parsed is None:
            continue

        if (
            parsed["type_perc"] == type_perc
            and int(parsed["nc"]) == int(num_colors)
            and int(parsed["dim"]) == int(dim)
            and int(parsed["L"]) == int(L)
            and _float_close(parsed["f_T"], f_T)
            and _float_close(parsed["c"], c)
            and _float_close(parsed["rho"], rho)
        ):
            group_abs = os.path.dirname(data_dir)
            matches.append(os.path.relpath(group_abs, raw_root))

    if matches:
        return matches[0]

    return formatted


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
        "f_T": float(g["fT"]),
        "c": float(g["c"]),
        "rho": float(g["rho"]),
    }


def _load_orders_sizes_new(fp: str) -> Optional[dict[int, dict]]:
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

    required_keys = [
        "shortest_path_lin",
        "M_size",
        "sp_lin_preteq",
        "sp_lin_posteq",
        "M_size_preteq",
        "M_size_posteq",
    ]

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

        vals: dict[str, float] = {}
        ok = True
        for rk in required_keys:
            v = _safe_float(data.get(rk, None))
            if v is None:
                ok = False
                break
            vals[rk] = float(v)

        if not ok:
            continue

        # Critério pedido: só entra no novo fluxo se os dois SPs temporais existirem
        # e forem diferentes de -1
        if int(vals["sp_lin_preteq"]) == -1 or int(vals["sp_lin_posteq"]) == -1:
            continue

        out[ordk] = vals

    return out


def _average_sizes_by_order(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_map = [
        ("shortest_path_lin", "shortest_path"),
        ("M_size", "S_perc"),
        ("sp_lin_preteq", "shortest_path_preteq"),
        ("M_size_preteq", "S_perc_preteq"),
        ("sp_lin_posteq", "shortest_path_posteq"),
        ("M_size_posteq", "S_perc_posteq"),
    ]

    out: Dict[str, Any] = {}

    for src_key, prefix in metric_map:
        vals = []
        for d in lst:
            v = _safe_float(d.get(src_key, None))
            if v is not None:
                vals.append(v)

        mean, sem, n = _mean_sem(vals)
        out[f"{prefix}_mean"] = mean
        out[f"{prefix}_sem"] = sem
        out[f"n_{prefix}"] = n

    return out

def compute_sizes_for_folder(
    type_perc: str,
    num_colors: int,
    dim: int,
    L: int,
    f_T: float,
    c: float,
    rho: float,
    p0_list: List[float],
    *,
    raw_root: str,
    published_root: str,
    manifests_root: str,
    rel_group: Optional[str] = None,
    clear_data: bool = False,
    verbose: bool = True,
) -> Optional[str]:
    raw_root = os.path.abspath(raw_root)
    published_root = os.path.abspath(published_root)
    manifests_root = os.path.abspath(manifests_root)

    if rel_group is None:
        rel_group = _resolve_existing_group_relpath(
            raw_root,
            type_perc,
            num_colors,
            dim,
            L,
            f_T,
            c,
            rho,
        )
    else:
        rel_group = os.path.normpath(str(rel_group))

    data_dir = os.path.join(raw_root, rel_group, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Pasta de dados não encontrada: {data_dir}")

    out_dir = os.path.join(published_root, rel_group)
    ensure_dir(out_dir)

    out_path = os.path.join(out_dir, "properties_sizes_bundle.json")

    manifest = _load_manifest(manifests_root, rel_group)

    if clear_data:
        if os.path.isfile(out_path):
            os.remove(out_path)
        manifest["processed_json_files"] = []
        manifest["n_processed_json_files"] = 0
        manifest["input_json_files"] = []
        manifest["n_input_json_files"] = 0
        manifest["summary_file"] = None
        manifest["last_update"] = None
        if verbose:
            print(f"[clear_data sizes] removido: {out_path}")

    all_jsons = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    selected_p0_filter = None if not p0_list else [float(p) for p in p0_list]
    selected_groups = _discover_sample_groups(all_jsons, p0_filter=selected_p0_filter)
    current_seed_files = sorted({os.path.basename(fp) for g in selected_groups for fp in g["files"]})

    manifest_input_files = manifest.get("input_json_files", None)
    if manifest_input_files is None:
        manifest_input_files = manifest.get("processed_json_files", [])
    input_files_done = set(map(str, manifest_input_files)) == set(current_seed_files)
    detected_group_keys = _detected_group_keys(selected_groups)
    existing_group_keys = _bundle_group_keys(out_path) if os.path.isfile(out_path) else None
    bundle_groups_match = existing_group_keys == detected_group_keys

    if verbose:
        print(
            f"[sizes group] {rel_group} | total_json={len(all_jsons)} "
            f"| parseable={len(current_seed_files)} | clear_data={clear_data}"
        )
        if os.path.isfile(out_path) and not bundle_groups_match:
            print(
                f"[sizes rebuild] grupos no bundle diferem dos raw detectados: "
                f"bundle={sorted(existing_group_keys or [])} raw={sorted(detected_group_keys)}"
            )

    if (
        (not clear_data)
        and os.path.isfile(out_path)
        and input_files_done
        and bundle_groups_match
    ):
        if verbose:
            print(f"[sizes skip] atualizado: {out_path}")
        return out_path

    bundle: Dict[str, Any] = {
        "meta": {
            "type_perc": type_perc,
            "num_colors": num_colors,
            "dim": dim,
            "L": L,
            "f_T": float(f_T),
            "c": float(c),
            "rho": float(rho),
            "base_dir": out_dir,
            "seed_used": [],
            "p0_groups_detected": sorted({float(g["p0"]) for g in selected_groups}),
            "P0_groups_detected": sorted({float(g["P0"]) for g in selected_groups}),
            "sample_groups_detected": _sample_group_summaries(selected_groups),
        },
        "p0_groups": [],
    }

    seed_used_set = set()
    current_valid_files: set[str] = set()
    valid_selected_groups: List[Dict[str, Any]] = []

    for group in selected_groups:
        P0_value = float(group["P0"])
        p0_value = float(group["p0"])
        files = list(group["files"])

        per_order: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        seeds_valid_set: set[int] = set()
        valid_files = 0

        for fp in files:
            orders = _load_orders_sizes_new(fp)
            if orders is None:
                continue

            if len(orders) == 0:
                continue

            valid_files += 1
            current_valid_files.add(os.path.basename(fp))

            seed = _extract_seed_from_filename(fp)
            if seed is not None:
                seeds_valid_set.add(seed)
                seed_used_set.add(seed)

            for ordk, data in orders.items():
                per_order[ordk].append(dict(data))

        if valid_files == 0:
            if verbose:
                print(f"[sizes] sem samples válidas em {data_dir} para P0={P0_value:.2f}, p0={p0_value:.2f}")
            continue

        valid_selected_groups.append(group)

        mean_by_order: Dict[int, Dict[str, Any]] = {}
        for ordk, lst in per_order.items():
            mean_by_order[ordk] = _average_sizes_by_order(lst)
            mean_by_order[ordk]["n_samples_perc"] = int(len(lst))

        orders_blocks = [
            {"order_percolation": int(ordk), "data": mean_by_order[ordk]}
            for ordk in sorted(mean_by_order.keys())
        ]

        bundle["p0_groups"].append({
            "P0_value": float(P0_value),
            "p0_value": float(p0_value),
            "num_samples_valid": int(valid_files),
            "num_seeds_valid": int(len(seeds_valid_set)),
            "orders": orders_blocks,
        })

    _ensure_bundle_group_P0_values(bundle, valid_selected_groups)
    _ensure_sample_group_summaries(bundle, selected_groups)

    bundle["meta"]["seed_used"] = sorted(seed_used_set)
    bundle = _sanitize_for_json(bundle)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2, allow_nan=False)

    manifest.update({
        "group_relpath": rel_group,
        "data_dir": data_dir,
        "input_json_files": sorted(current_seed_files),
        "n_input_json_files": len(current_seed_files),
        "processed_json_files": sorted(current_valid_files),
        "n_processed_json_files": len(current_valid_files),
        "summary_file": out_path,
        "last_update": pd.Timestamp.utcnow().isoformat(),
    })
    _save_manifest(manifests_root, rel_group, manifest)

    return out_path


def build_sizes_dataframe(published_root: str, output_file: str | Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    published_root = os.path.abspath(published_root)
    output_file = Path(output_file)

    bundle_files = []
    for dirpath, _, filenames in os.walk(published_root):
        if "properties_sizes_bundle.json" in filenames:
            bundle_files.append(os.path.join(dirpath, "properties_sizes_bundle.json"))

    for bundle_path in sorted(bundle_files):
        parsed = _parse_params_from_path(os.path.dirname(bundle_path))
        if parsed is None:
            parsed = _parse_params_from_path(bundle_path)
        if parsed is None:
            continue

        type_perc, L, f_T, c, nc, rho, dim = parsed

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
            P0_val = _safe_float(g.get("P0_value", float("nan")))
            N_samples = int(g.get("num_samples_valid", 0) or 0)

            orders = g.get("orders", [])
            if not isinstance(orders, list) or len(orders) == 0:
                continue

            for ob in orders:
                ordk = ob.get("order_percolation", None)
                if ordk is None:
                    continue

                order = int(ordk) + 1
                d = ob.get("data", {}) or {}

                rows.append({
                    "type_perc": type_perc,
                    "dim": dim,
                    "L": L,
                    "f_T": f_T,
                    "c": c,
                    "nc": nc,
                    "rho": rho,
                    "p0": p0_val,
                    "P0": P0_val,
                    "order": order,
                    "N_samples": N_samples,
                    "N_samples_perc": int(d.get("n_samples_perc", 0) or 0),

                    "shortest_path": _safe_float(d.get("shortest_path_mean", float("nan"))),
                    "shortest_path_err": _safe_float(d.get("shortest_path_sem", float("nan"))),

                    "S_perc": _safe_float(d.get("S_perc_mean", float("nan"))),
                    "S_perc_err": _safe_float(d.get("S_perc_sem", float("nan"))),

                    "shortest_path_preteq": _safe_float(d.get("shortest_path_preteq_mean", float("nan"))),
                    "shortest_path_preteq_err": _safe_float(d.get("shortest_path_preteq_sem", float("nan"))),

                    "S_perc_preteq": _safe_float(d.get("S_perc_preteq_mean", float("nan"))),
                    "S_perc_preteq_err": _safe_float(d.get("S_perc_preteq_sem", float("nan"))),

                    "shortest_path_posteq": _safe_float(d.get("shortest_path_posteq_mean", float("nan"))),
                    "shortest_path_posteq_err": _safe_float(d.get("shortest_path_posteq_sem", float("nan"))),

                    "S_perc_posteq": _safe_float(d.get("S_perc_posteq_mean", float("nan"))),
                    "S_perc_posteq_err": _safe_float(d.get("S_perc_posteq_sem", float("nan"))),
                })

    df = pd.DataFrame(rows, columns=DEFAULT_SIZE_COLS)

    if not df.empty:
        df = df.sort_values(
            by=["type_perc", "dim", "nc", "rho", "c", "f_T", "L", "P0", "p0", "order"]
        ).reset_index(drop=True)

    ensure_dir(output_file.parent)
    df.to_csv(output_file, index=False, sep=" ")
    return df


def process_all_data_sizes(
    clear_data: bool = False,
    *,
    sop_root: str = "../SOP_data",
    raw_dir: str = "raw",
    published_dir: str = "published",
    manifests_dir: str = "manifests_sizes",
    output_suffix: str = "",
    p0_lst: Optional[List[float]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    if p0_lst is not None:
        p0_lst = [float(p) for p in p0_lst]
        if len(p0_lst) == 0:
            p0_lst = None

    sop_root = os.path.abspath(sop_root)
    raw_root = os.path.join(sop_root, raw_dir)
    published_root = os.path.join(sop_root, published_dir)
    manifests_root = os.path.join(sop_root, manifests_dir)
    all_data_sizes_name = f"all_data_sizes{output_suffix}.dat"

    ensure_dir(raw_root)
    ensure_dir(published_root)
    ensure_dir(manifests_root)

    all_parms = collect_param_combinations(raw_root, return_relpath=True)

    iterator = tqdm(
        all_parms,
        desc="Processando conjuntos [sizes]",
        ncols=120,
        dynamic_ncols=False,
        leave=True,
    )

    for tp, nc, DIM, L, FT, C, RHO, REL_GROUP in iterator:
        iterator.set_postfix_str(
            f"{tp} nc={nc} dim={DIM} L={L} f_T={FT:.6e} c={C:.6e} rho={RHO:.4e}"
        )
        compute_sizes_for_folder(
            type_perc=tp,
            num_colors=nc,
            dim=DIM,
            L=L,
            f_T=FT,
            c=C,
            rho=RHO,
            p0_list=p0_lst,
            raw_root=raw_root,
            published_root=published_root,
            manifests_root=manifests_root,
            rel_group=REL_GROUP,
            clear_data=clear_data,
            verbose=verbose,
        )

    if verbose:
        print(f"Processamento [sizes] finalizado. Construindo SOP_data/{all_data_sizes_name} ...")

    df = build_sizes_dataframe(
        published_root=published_root,
        output_file=os.path.join(sop_root, all_data_sizes_name),
    )

    if verbose:
        print(f"[write] {os.path.join(sop_root, all_data_sizes_name)} ({len(df)} linhas)")

    return df

def collect_param_combinations(
    root_dir: str,
    *,
    type_perc: Optional[str] = None,
    dir_re: re.Pattern = DIR_RE,
    return_relpath: bool = False,
) -> List[Tuple]:
    """
    Coleta combinações de parâmetros a partir dos diretórios existentes.

    Quando return_relpath=True, retorna também o caminho relativo real do grupo.
    Isso evita reconstruir nomes de diretório a partir de floats, que pode falhar
    quando as pastas usam, por exemplo, fT_5.000000e-02 em vez de fT_5.0000e-02.
    """
    root_dir = os.path.abspath(os.path.normpath(root_dir))
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
        f_T = float(m.group("fT"))
        c = float(m.group("c"))
        rho = float(m.group("rho"))

        if return_relpath:
            group_abs = os.path.dirname(os.path.normpath(dirpath))
            rel_group = os.path.relpath(group_abs, root_dir).replace(os.sep, "/")
            combos.add((tp, nc, dim, L, f_T, c, rho, rel_group))
        else:
            combos.add((tp, nc, dim, L, f_T, c, rho))

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


def parse_filename(path: str | Path) -> Dict[str, Any]:
    parsed = _parse_fname(str(path))
    if parsed is None:
        raise ValueError(f"Nome inválido: {path}")

    P0_value, p0_value, seed = parsed
    return {"P0": P0_value, "p0": p0_value, "seed": seed}


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

    meta = js.get("meta", {}) if isinstance(js.get("meta", {}), dict) else {}
    t_eq_json = _safe_float(meta.get("t_eq", js.get("t_eq", None)))

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
        ft = data.get("ft", data.get("f_t", None))

        if t is not None:
            t = np.asarray(t, dtype=float)
        if pt is not None:
            pt = np.asarray(pt, dtype=float)
        if ft is not None:
            ft = np.asarray(ft, dtype=float)

        out[ordk] = {
            "t": t,
            "pt": pt,
            "ft": ft,
            "t_eq": t_eq_json,
            "shortest_path_lin": data.get("shortest_path_lin", None),
            "M_size": data.get("M_size", None),
        }

    return out


def _average_by_order_new(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
    series_pt = []
    series_ft = []
    t_eq_vals: List[float] = []
    spl_vals: List[float] = []
    msz_vals: List[float] = []

    for d in lst:
        t = d.get("t", None)
        pt = d.get("pt", None)
        ft = d.get("ft", None)
        t_eq = _safe_float(d.get("t_eq", None))
        if t_eq is not None:
            t_eq_vals.append(float(t_eq))

        if t is None or pt is None:
            continue

        t = np.asarray(t, dtype=float)
        pt = np.asarray(pt, dtype=float)

        n_pt = min(len(t), len(pt))
        if n_pt <= 1:
            continue

        series_pt.append((t[:n_pt], pt[:n_pt]))

        if ft is not None:
            ft = np.asarray(ft, dtype=float)
            n_ft = min(len(t), len(ft), n_pt)
            if n_ft > 1:
                series_ft.append(ft[:n_ft])

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
            "ft_mean": [],
            "ft_std": [],
            "ft_sem": [],
            "n_seeds_pt": 0,
            "n_seeds_ft": 0,
            "t_eq": float(max(t_eq_vals)) if t_eq_vals else float("nan"),
            "t_eq_mean": float(np.mean(t_eq_vals)) if t_eq_vals else float("nan"),
            "t_eq_min": float(np.min(t_eq_vals)) if t_eq_vals else float("nan"),
            "t_eq_max": float(np.max(t_eq_vals)) if t_eq_vals else float("nan"),
            "n_t_eq": int(len(t_eq_vals)),
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

    if t_eq_vals:
        t_eq_arr = np.asarray(t_eq_vals, dtype=float)
        out["t_eq"] = float(np.max(t_eq_arr))
        out["t_eq_mean"] = float(np.mean(t_eq_arr))
        out["t_eq_min"] = float(np.min(t_eq_arr))
        out["t_eq_max"] = float(np.max(t_eq_arr))
        out["n_t_eq"] = int(t_eq_arr.size)
    else:
        out["t_eq"] = float("nan")
        out["t_eq_mean"] = float("nan")
        out["t_eq_min"] = float("nan")
        out["t_eq_max"] = float("nan")
        out["n_t_eq"] = 0

    if series_ft:
        min_len_ft = min(len(ft) for ft in series_ft)
        min_len = min(min_len_pt, min_len_ft)

        fts = np.stack([ft[:min_len] for ft in series_ft], axis=0)
        nseed_ft = int(fts.shape[0])

        ft_mean = np.mean(fts, axis=0)
        if nseed_ft > 1:
            ft_std = np.std(fts, axis=0, ddof=1)
            ft_sem = ft_std / np.sqrt(nseed_ft)
        else:
            ft_std = np.zeros_like(ft_mean)
            ft_sem = np.zeros_like(ft_mean)

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
        out["ft_mean"] = ft_mean.tolist()
        out["ft_std"] = ft_std.tolist()
        out["ft_sem"] = ft_sem.tolist()
        out["n_seeds_ft"] = nseed_ft
    else:
        out["ft_mean"] = []
        out["ft_std"] = []
        out["ft_sem"] = []
        out["n_seeds_ft"] = 0

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


def _discover_sample_groups(
    all_jsons: List[str],
    *,
    p0_filter: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Agrupa os arquivos pelo par exato (P0, p0) lido diretamente do nome do arquivo.

    Isso evita misturar amostras que compartilham o mesmo diretório físico, mas têm
    valores diferentes de P0. O problema observado em all_data.dat vinha justamente
    de colapsar tudo apenas por p0 e depois reconstruir P0 por média dos filenames.
    """
    groups: Dict[Tuple[float, float], List[str]] = defaultdict(list)

    for fp in all_jsons:
        parsed = _parse_fname(fp)
        if parsed is None:
            continue

        P0_file, p0_file, _ = parsed

        if p0_filter is not None:
            keep = any(abs(float(p0_file) - float(p0_sel)) < 1e-12 for p0_sel in p0_filter)
            if not keep:
                continue

        groups[(float(P0_file), float(p0_file))].append(fp)

    out: List[Dict[str, Any]] = []
    for (P0_val, p0_val), files in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.append({
            "P0": float(P0_val),
            "p0": float(p0_val),
            "files": sorted(files),
            "n_files": len(files),
        })

    return out


def _sample_group_summaries(selected_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "P0_value": float(group["P0"]),
            "p0_value": float(group["p0"]),
            "n_files": int(group["n_files"]),
        }
        for group in selected_groups
    ]


def _detected_group_keys(selected_groups: List[Dict[str, Any]]) -> set[Tuple[float, float]]:
    return {
        (float(group["P0"]), float(group["p0"]))
        for group in selected_groups
    }


def _parse_params_from_path(path: str) -> Optional[Tuple[str, int, float, float, int, float, int]]:
    m = PARAM_RE.search(path.replace("\\", "/"))
    if not m:
        return None

    type_perc = m.group("type")
    L = int(m.group("L"))
    f_T = float(m.group("fT"))
    c = float(m.group("c"))
    nc = int(m.group("nc"))
    rho = float(m.group("rho"))
    dim = int(m.group("dim"))

    return (type_perc, L, f_T, c, nc, rho, dim)


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


def _tail_stats_from_mean_series(
    t: np.ndarray,
    y: np.ndarray,
    sem: np.ndarray,
    t0: float,
) -> Tuple[float, float, int]:
    """
    Calcula o valor assintótico a partir da série média y(t) e de seu erro padrão.

    A partir de t0, usa média ponderada por 1/sem^2 quando houver sem finito e
    positivo; caso contrário, cai para a média simples da cauda.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)

    if t.size == 0 or y.size == 0 or sem.size == 0 or not np.isfinite(t0):
        return float("nan"), float("nan"), 0

    n = min(t.size, y.size, sem.size)
    t = t[:n]
    y = y[:n]
    sem = sem[:n]

    idx = idx_from_t0(t, t0)
    if idx >= n:
        return float("nan"), float("nan"), 0

    t_tail = t[idx:]
    y_tail = y[idx:]
    sem_tail = sem[idx:]

    mask = np.isfinite(t_tail) & np.isfinite(y_tail)
    y_tail = y_tail[mask]
    sem_tail = sem_tail[mask]

    if y_tail.size == 0:
        return float("nan"), float("nan"), 0

    sem_mask = np.isfinite(sem_tail) & (sem_tail > 0)
    if np.any(sem_mask):
        mu, se = weighted_mean_and_sem(y_tail[sem_mask], sem_tail[sem_mask])
        return float(mu), float(se), int(np.count_nonzero(sem_mask))

    mu = float(np.mean(y_tail))
    if y_tail.size > 1:
        se = float(np.std(y_tail, ddof=1) / np.sqrt(y_tail.size))
    else:
        se = 0.0
    return mu, se, int(y_tail.size)

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


def _bundle_group_keys(bundle_path: str | Path) -> Optional[set[Tuple[float, float]]]:
    try:
        with Path(bundle_path).open("r", encoding="utf-8") as f:
            js = json.load(f)
    except Exception:
        return None

    p0_groups = js.get("p0_groups", [])
    if not isinstance(p0_groups, list):
        return None

    keys: set[Tuple[float, float]] = set()
    for group in p0_groups:
        if not isinstance(group, dict):
            return None
        P0_value = _safe_float(group.get("P0_value", None))
        p0_value = _safe_float(group.get("p0_value", None))
        if P0_value is None or p0_value is None:
            return None
        keys.add((float(P0_value), float(p0_value)))

    return keys


def _ensure_bundle_group_P0_values(
    bundle: Dict[str, Any],
    selected_groups: List[Dict[str, Any]],
) -> None:
    p0_groups = bundle.get("p0_groups", [])
    if not isinstance(p0_groups, list):
        raise ValueError("Bundle inválido: 'p0_groups' não é uma lista.")

    if len(p0_groups) != len(selected_groups):
        raise ValueError(
            "Bundle inválido: número de p0_groups difere dos grupos detectados "
            f"({len(p0_groups)} != {len(selected_groups)})."
        )

    for idx, (p0_group, detected_group) in enumerate(zip(p0_groups, selected_groups)):
        if not isinstance(p0_group, dict):
            raise ValueError(f"Bundle inválido: p0_groups[{idx}] não é um objeto.")

        detected_P0 = float(detected_group["P0"])
        detected_p0 = float(detected_group["p0"])
        group_p0 = _safe_float(p0_group.get("p0_value", None))

        if group_p0 is None or not _float_close(group_p0, detected_p0):
            raise ValueError(
                f"Bundle inválido: p0_groups[{idx}] tem p0_value={group_p0}, "
                f"mas o grupo detectado tem p0={detected_p0}."
            )

        group_P0 = _safe_float(p0_group.get("P0_value", None))
        if group_P0 is None:
            group_P0 = detected_P0

        if not _float_close(group_P0, detected_P0):
            raise ValueError(
                f"Bundle inválido: p0_groups[{idx}] tem P0_value={group_P0}, "
                f"mas o grupo detectado tem P0={detected_P0}."
            )

        p0_groups[idx] = {
            "P0_value": float(detected_P0),
            **{k: v for k, v in p0_group.items() if k != "P0_value"},
        }


def _ensure_sample_group_summaries(
    bundle: Dict[str, Any],
    selected_groups: List[Dict[str, Any]],
) -> None:
    meta = bundle.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("Bundle inválido: 'meta' não é um objeto.")
    meta["sample_groups_detected"] = _sample_group_summaries(selected_groups)

def compute_means_for_folder(
    type_perc: str,
    num_colors: int,
    dim: int,
    L: int,
    f_T: float,
    c: float,
    rho: float,
    p0_list: List[float],
    *,
    raw_root: str,
    published_root: str,
    manifests_root: str,
    rel_group: Optional[str] = None,
    x_max: float | None = None,
    n_boot: int = 20000,
    rng_seed: int = 12345,
    window_roll: int | None = None,
    time_series_only: bool = False,
    clear_data: bool = False,
    verbose: bool = True,
) -> Optional[str]:
    raw_root = os.path.abspath(raw_root)
    published_root = os.path.abspath(published_root)
    manifests_root = os.path.abspath(manifests_root)

    if rel_group is None:
        rel_group = _resolve_existing_group_relpath(
            raw_root,
            type_perc,
            num_colors,
            dim,
            L,
            f_T,
            c,
            rho,
        )
    else:
        rel_group = os.path.normpath(str(rel_group))

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

    selected_p0_filter = None if not p0_list else [float(p) for p in p0_list]
    selected_groups = _discover_sample_groups(all_jsons, p0_filter=selected_p0_filter)

    if verbose:
        detected_pairs = [(g["P0"], g["p0"]) for g in selected_groups]
        print(f"[auto-groups] grupos detectados em {data_dir}: {detected_pairs}")

    bad_name_files = []
    for fp in all_jsons:
        if _parse_fname(fp) is None:
            bad_name_files.append(os.path.basename(fp))

    current_seed_files = sorted({os.path.basename(fp) for g in selected_groups for fp in g["files"]})

    if bad_name_files and verbose:
        print(f"[warn] {data_dir}: {len(bad_name_files)} arquivo(s) com nome fora do padrão flexível")
        for bn in bad_name_files[:10]:
            print(f"       - {bn}")
        if len(bad_name_files) > 10:
            print("       ...")

    processed_files = set(map(str, manifest.get("processed_json_files", [])))
    new_files = [bn for bn in current_seed_files if bn not in processed_files]
    manifest_time_series_only = bool(manifest.get("time_series_only", False))
    detected_group_keys = _detected_group_keys(selected_groups)
    existing_group_keys = _bundle_group_keys(out_path) if os.path.isfile(out_path) else None
    bundle_groups_match = existing_group_keys == detected_group_keys

    if verbose:
        print(
            f"[group] {rel_group} | total_json={len(all_jsons)} "
            f"| parseable={len(current_seed_files)} | new={len(new_files)} | clear_data={clear_data}"
        )
        if os.path.isfile(out_path) and not bundle_groups_match:
            print(
                f"[rebuild] grupos no bundle diferem dos raw detectados: "
                f"bundle={sorted(existing_group_keys or [])} raw={sorted(detected_group_keys)}"
            )

    if (
        (not clear_data)
        and os.path.isfile(out_path)
        and len(new_files) == 0
        and manifest_time_series_only == bool(time_series_only)
        and bundle_groups_match
    ):
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
            "f_T": float(f_T),
            "c": float(c),
            "rho": float(rho),
            "base_dir": out_dir,
            "x_max_used": None if x_max is None else float(x_max),
            "bootstrap": {"n_boot": int(n_boot), "rng_seed": int(rng_seed)},
            "rolling": {"window": None if window_roll is None else int(window_roll)},
            "seed_used": [],
            "p0_groups_detected": sorted({float(g["p0"]) for g in selected_groups}),
            "P0_groups_detected": sorted({float(g["P0"]) for g in selected_groups}),
            "sample_groups_detected": _sample_group_summaries(selected_groups),
            "time_series_only": bool(time_series_only),
        },
        "p0_groups": [],
    }

    colors_per_sample_all: List[int] = []

    for group in selected_groups:
        P0_value = float(group["P0"])
        p0_value = float(group["p0"])
        files = list(group["files"])

        if verbose:
            print(
                f"[debug] data_dir={data_dir} | P0={P0_value:.2f} | "
                f"p0={p0_value:.2f} | files={len(files)}"
            )

        if not files:
            if verbose:
                print(f"[aviso] Sem arquivos para P0={P0_value:.2f}, p0={p0_value:.2f}")
            continue

        per_order: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        seeds_set: set[int] = set()
        seeds_total_set: set[int] = set()
        seeds_non_perc_set: set[int] = set()
        valid_files = 0
        colors_per_sample_this_group: List[int] = []

        for fp in files:
            orders = _load_orders_new(fp)
            if orders is None:
                continue

            n_orders = len(orders)
            colors_per_sample_this_group.append(n_orders)
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
                    if data_local.get("ft") is not None:
                        data_local["ft"] = np.asarray(data_local["ft"], dtype=float)[m]

                per_order[ordk].append(data_local)

        total_files_this_group = len(colors_per_sample_this_group)

        if verbose:
            print(
                f"[debug] P0={P0_value:.2f} | p0={p0_value:.2f} | total_files={total_files_this_group} "
                f"| valid_files={valid_files} | non_perc={total_files_this_group - valid_files}"
            )

        mean_by_order: Dict[int, Dict[str, Any]] = {}
        for ordk, lst in per_order.items():
            mean_by_order[ordk] = _average_by_order_new(lst)
            mean_by_order[ordk]["n_samples_perc"] = int(len(lst))
            mean_by_order[ordk]["n_samples_total"] = int(total_files_this_group)
            mean_by_order[ordk]["n_samples_non_perc"] = int(total_files_this_group - len(lst))

        t0_by_order: Dict[int, float] = {}
        for ordk in sorted(mean_by_order.keys()):
            d = mean_by_order[ordk]
            t_eq = _safe_float(d.get("t_eq", None))
            if t_eq is None:
                continue
            t0_by_order[ordk] = float(t_eq)
            mean_by_order[ordk]["t_eq"] = float(t_eq)
            mean_by_order[ordk]["t_eq_source"] = "json meta.t_eq; max over samples in this order"

        t0_global = float(max(t0_by_order.values())) if t0_by_order else float("nan")

        pc_by_order: Dict[int, Tuple[float, float, int]] = {}
        for ordk in sorted(mean_by_order.keys()):
            d = mean_by_order[ordk]
            if not d.get("time") or not d.get("pt_mean") or not d.get("pt_sem"):
                pc_by_order[ordk] = (float("nan"), float("nan"), 0)
                continue

            t = np.asarray(d["time"], dtype=float)
            pt = np.asarray(d["pt_mean"], dtype=float)
            pt_sem = np.asarray(d["pt_sem"], dtype=float)
            n0 = min(t.size, pt.size, pt_sem.size)
            if n0 == 0:
                pc_by_order[ordk] = (float("nan"), float("nan"), 0)
                continue

            pc_i_mean, pc_i_sem, n_tail = _tail_stats_from_mean_series(
                t[:n0],
                pt[:n0],
                pt_sem[:n0],
                t0_global,
            )
            pc_by_order[ordk] = (pc_i_mean, pc_i_sem, n_tail)

            mean_by_order[ordk]["pc_sop"] = {
                "mean": _safe_float(pc_i_mean),
                "std_boot": _safe_float(pc_i_sem),
                "n_tail_points": int(n_tail),
                "n_boot": int(n_boot),
                "t0": _safe_float(t0_global),
                "pc_method": "ensemble-mean tail after t_eq read from sample json meta",
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

        orders_blocks = [
            {"order_percolation": int(ordk), "data": mean_by_order[ordk]}
            for ordk in sorted(mean_by_order.keys())
        ]

        colors_arr = np.asarray(colors_per_sample_this_group, dtype=float)
        if colors_arr.size > 0:
            nc_mean = float(np.mean(colors_arr))
            nc_std = float(np.std(colors_arr, ddof=1)) if colors_arr.size > 1 else 0.0
            nc_err = float(nc_std / np.sqrt(colors_arr.size))
        else:
            nc_mean = float("nan")
            nc_std = float("nan")
            nc_err = float("nan")

        p0_group = {
            "P0_value": float(P0_value),
            "p0_value": float(p0_value),
            "num_seeds": len(seeds_set),
            "num_seeds_total": len(seeds_total_set),
            "num_seeds_non_percolating": len(seeds_non_perc_set),
            "num_samples_total": int(total_files_this_group),
            "num_samples_percolating_any_order": int(valid_files),
            "num_samples_non_percolating": int(total_files_this_group - valid_files),
            "orders": orders_blocks,
            "pc_sop": {
                "mean": _safe_float(pc_mean),
                "std_boot": _safe_float(pc_sem),
                "n_seeds": len(seeds_set),
                "n_boot": int(n_boot),
                "t0_global": _safe_float(t0_global),
                "pc_method": "combine orders of ensemble-mean tails after global t_eq read from sample json meta",
            },
        }

        if not time_series_only:
            p0_group["colors"] = {
                "Nsamples": int(colors_arr.size),
                "nc": _safe_float(nc_mean),
                "nc_std": _safe_float(nc_std),
                "nc_err": _safe_float(nc_err),
            }

        bundle["p0_groups"].append(p0_group)

    _ensure_bundle_group_P0_values(bundle, selected_groups)
    _ensure_sample_group_summaries(bundle, selected_groups)

    bundle["meta"]["seed_used"] = sorted(seed_used_set)
    bundle = _sanitize_for_json(bundle)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2, allow_nan=False)

    if not time_series_only:
        with open(colors_path, "w", encoding="utf-8") as f:
            for val in colors_per_sample_all:
                f.write(f"{int(val)}\n")

    manifest.update({
        "group_relpath": rel_group,
        "data_dir": data_dir,
        "processed_json_files": sorted(current_seed_files),
        "n_processed_json_files": len(current_seed_files),
        "summary_file": out_path,
        "time_series_only": bool(time_series_only),
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

        type_perc, L, f_T, c, nc, rho, dim = parsed

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

        for g in p0_groups:
            p0_val = _safe_float(g.get("p0_value", float("nan")))
            P0_val = _safe_float(g.get("P0_value", float("nan")))
            if P0_val is None:
                P0_val, _ = _parse_P0_p0_from_seed_used(seed_used)
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
                    "f_T": f_T,
                    "c": c,
                    "nc": nc,
                    "rho": rho,
                    "p0": p0_val,
                    "P0": P0_val,
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
        "type_perc", "dim", "L", "f_T", "c", "nc", "rho", "p0", "P0",
        "order", "N_samples", "N_samples_perc",
        "p_mean", "p_err",
        "shortest_path", "shortest_path_err",
        "S_perc", "S_perc_err",
    ]

    df = pd.DataFrame(rows, columns=cols)

    if not df.empty:
        df = df.sort_values(
            by=["type_perc", "dim", "nc", "rho", "c", "f_T", "L", "P0", "p0", "order"]
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

        type_perc, L, f_T, c, nc_model, rho, dim = parsed

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
            P0_val = _safe_float(g.get("P0_value", float("nan")))
            cstats = g.get("colors", {}) if isinstance(g.get("colors", {}), dict) else {}

            rows.append({
                "L": L,
                "dim": dim,
                "f_T": f_T,
                "c": c,
                "num_colors": nc_model,
                "P0": P0_val,
                "p0": p0_val,
                "Nsamples": int(cstats.get("Nsamples", 0) or 0),
                "rho": rho,
                "nc": _safe_float(cstats.get("nc", float("nan"))),
                "nc_err": _safe_float(cstats.get("nc_err", float("nan"))),
                "nc_std": _safe_float(cstats.get("nc_std", float("nan"))),
            })

    cols = ["L", "dim", "f_T", "c", "num_colors", "P0", "p0", "Nsamples", "rho", "nc", "nc_err", "nc_std"]
    df = pd.DataFrame(rows, columns=cols)

    if not df.empty:
        df = df.sort_values(
            by=["dim", "num_colors", "rho", "c", "f_T", "L", "P0", "p0"]
        ).reset_index(drop=True)

    ensure_dir(output_file.parent)
    df.to_csv(output_file, index=False, sep=" ")
    return df

def process_all_data(
    clear_data: bool = False,
    *,
    sop_root: str = "../SOP_data",
    raw_dir: str = "raw",
    published_dir: str = "published",
    manifests_dir: str = "manifests",
    output_suffix: str = "",
    time_series_only: bool = False,
    p0_lst: Optional[List[float]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    if p0_lst is not None:
        p0_lst = [float(p) for p in p0_lst]
        if len(p0_lst) == 0:
            p0_lst = None

    sop_root = os.path.abspath(sop_root)
    raw_root = os.path.join(sop_root, raw_dir)
    published_root = os.path.join(sop_root, published_dir)
    manifests_root = os.path.join(sop_root, manifests_dir)
    all_data_name = f"all_data{output_suffix}.dat"
    all_colors_name = f"all_colors{output_suffix}.dat"

    ensure_dir(raw_root)
    ensure_dir(published_root)
    ensure_dir(manifests_root)

    all_parms = collect_param_combinations(raw_root, return_relpath=True)

    iterator = tqdm(
        all_parms,
        desc="Processando conjuntos",
        ncols=120,
        dynamic_ncols=False,
        leave=True,
    )

    for tp, nc, DIM, L, FT, C, RHO, REL_GROUP in iterator:
        iterator.set_postfix_str(
            f"{tp} nc={nc} dim={DIM} L={L} f_T={FT:.6e} c={C:.6e} rho={RHO:.4e}"
        )
        compute_means_for_folder(
            type_perc=tp,
            num_colors=nc,
            dim=DIM,
            L=L,
            f_T=FT,
            c=C,
            rho=RHO,
            p0_list=p0_lst,
            raw_root=raw_root,
            published_root=published_root,
            manifests_root=manifests_root,
            rel_group=REL_GROUP,
            time_series_only=time_series_only,
            clear_data=clear_data,
            verbose=verbose,
        )

    if verbose:
        print(f"Processamento finalizado. Construindo SOP_data/{all_data_name} ...")

    df = build_properties_dataframe(
        published_root=published_root,
        output_file=os.path.join(sop_root, all_data_name),
    )

    if verbose:
        print(f"[write] {os.path.join(sop_root, all_data_name)} ({len(df)} linhas)")

    if time_series_only:
        if verbose:
            print("Modo NL-stop: p_mean/p_err calculados; pulando all_colors.")
        return df

    if verbose:
        print(f"Construindo SOP_data/{all_colors_name} ...")

    df_colors = build_colors_dataframe(
        published_root=published_root,
        output_file=os.path.join(sop_root, all_colors_name),
    )

    if verbose:
        print(f"[write] {os.path.join(sop_root, all_colors_name)} ({len(df_colors)} linhas)")
    return df

if __name__ == "__main__":
    process_all_data(clear_data=False)
    process_all_data_sizes(clear_data=False)
