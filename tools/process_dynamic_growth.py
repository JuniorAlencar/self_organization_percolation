#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

PARAM_RE = re.compile(
    rf"""
    (?P<type>[A-Za-z0-9]+)_percolation
    /num_colors_(?P<nc>\d+)
    /dim_(?P<dim>\d+)
    /L_(?P<L>\d+)
    /fT_constant
    /fT_(?P<fT>{FLOAT})
    /c_(?P<c>{FLOAT})
    /rho_(?P<rho>{FLOAT})
    (?:/stationary_window_(?P<stat_window>\d+))?
    /data$
    """,
    re.X,
)

RE_P0 = re.compile(rf"(?:^|_)P0_(?P<P0>{FLOAT})(?:_|\.json$)")
RE_p0 = re.compile(rf"(?:^|_)p0_(?P<p0>{FLOAT})(?:_|\.json$)")

ALL_DATA_COLUMNS = [
    "type_perc", "dim", "L", "f_T", "c", "nc", "rho", "p0", "P0",
    "order", "N_samples", "N_samples_perc",
    "p_mean", "p_err", "f_mean", "f_err", "z_max_mean", "z_max_err",
    "z_stat_mean", "z_stat_err", "stat_window",
]

ALL_COLORS_COLUMNS = [
    "type_perc", "dim", "L", "f_T", "c", "num_colors", "P0", "p0",
    "N_samples", "rho", "nc", "nc_err", "nc_std", "stat_window",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_float(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def parse_sample_name(path: Path) -> tuple[float, float] | None:
    name = path.name
    m_p0 = RE_p0.search(name)
    m_P0 = RE_P0.search(name)
    if not m_p0 or not m_P0:
        return None
    return float(m_P0.group("P0")), float(m_p0.group("p0"))


def parse_data_dir(path: Path) -> dict[str, Any] | None:
    m = PARAM_RE.search(path.as_posix())
    if not m:
        return None
    g = m.groupdict()
    return {
        "type_perc": g["type"],
        "dim": int(g["dim"]),
        "L": int(g["L"]),
        "f_T": float(g["fT"]),
        "c": float(g["c"]),
        "nc": int(g["nc"]),
        "rho": float(g["rho"]),
        "stat_window": int(g["stat_window"]) if g.get("stat_window") else -1,
    }


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def mean_sem(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return math.nan, math.nan, 0
    mean = float(arr.mean())
    if n == 1:
        return mean, 0.0, 1
    return mean, float(arr.std(ddof=1) / math.sqrt(n)), n


def mean_sem_std(values: list[float]) -> tuple[float, float, float, int]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(arr.mean())
    if n == 1:
        return mean, 0.0, 0.0, 1
    std = float(arr.std(ddof=1))
    return mean, float(std / math.sqrt(n)), std, n


def dat_value(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.12g}"
    return str(value)


def parse_order_key(key: str) -> int | None:
    m = re.search(r"order_percolation\s+(\d+)", key)
    if not m:
        return None
    return int(m.group(1))


def tail_mean_after_t_eq(time: Any, values: Any, t_eq: float) -> float | None:
    if time is None or values is None:
        return None
    try:
        t = np.asarray(time, dtype=float)
        y = np.asarray(values, dtype=float)
    except Exception:
        return None
    n = min(t.size, y.size)
    if n == 0:
        return None
    t = t[:n]
    y = y[:n]
    mask = (t > float(t_eq)) & np.isfinite(y)
    if not np.any(mask):
        return None
    return float(np.mean(y[mask]))


def zmax_for_order(
    *,
    order_pos: int,
    color_1b: int | None,
    t_eq_species: float,
    result_t_eq_order: list[float],
    meta_t_eq: list[Any],
    meta_zmax: list[Any],
) -> float | None:
    if not meta_zmax:
        return None

    # New/fixed raw format: meta.t_eq_by_species is already ordered by
    # stabilization and nulls are trailing; z_max is aligned to that order.
    ordered_prefix = True
    if len(meta_t_eq) >= len(result_t_eq_order):
        for a, b in zip(meta_t_eq[:len(result_t_eq_order)], result_t_eq_order):
            fa = finite_float(a)
            if fa is None or not math.isclose(fa, b, rel_tol=1e-12, abs_tol=1e-12):
                ordered_prefix = False
                break
    else:
        ordered_prefix = False

    if ordered_prefix and order_pos < len(meta_zmax):
        return finite_float(meta_zmax[order_pos])

    # Older raw files were species-indexed. Fall back to color when available.
    if color_1b is not None:
        idx = int(color_1b) - 1
        if 0 <= idx < len(meta_zmax):
            return finite_float(meta_zmax[idx])

    # Last resort: pair the finite t_eq entries with z_max by position, skipping
    # null t_eq values. This implements the dynamic-order convention requested.
    finite_positions = [
        i for i, v in enumerate(meta_t_eq)
        if finite_float(v) is not None and i < len(meta_zmax)
    ]
    if order_pos < len(finite_positions):
        return finite_float(meta_zmax[finite_positions[order_pos]])

    if order_pos < len(meta_zmax):
        return finite_float(meta_zmax[order_pos])
    return None


def load_dynamic_sample(path: Path) -> list[dict[str, Any]] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            js = json.load(f)
    except Exception as exc:
        print(f"[warn] ignorando JSON inválido {path}: {exc}")
        return None

    meta = js.get("meta", {}) if isinstance(js.get("meta", {}), dict) else {}
    results = js.get("results", {}) if isinstance(js.get("results", {}), dict) else {}
    if not results:
        return []

    parsed_blocks: list[tuple[int, dict[str, Any]]] = []
    for key, block in results.items():
        raw_order = parse_order_key(str(key))
        if raw_order is None:
            continue
        data = (block or {}).get("data", {})
        if not isinstance(data, dict):
            continue
        t_eq = finite_float(data.get("t_eq_species"))
        if t_eq is None:
            continue
        parsed_blocks.append((raw_order, data))

    if not parsed_blocks:
        return []

    parsed_blocks.sort(key=lambda x: x[0])
    result_t_eq_order = [float(d.get("t_eq_species")) for _, d in parsed_blocks]
    meta_t_eq = meta.get("t_eq_by_species", [])
    meta_zmax = meta.get("z_max", meta.get("z_max_final", []))
    meta_zstat = meta.get("z_stat", [])
    if not isinstance(meta_t_eq, list):
        meta_t_eq = []
    if not isinstance(meta_zmax, list):
        meta_zmax = []
    if not isinstance(meta_zstat, list):
        meta_zstat = []

    out: list[dict[str, Any]] = []
    for order_pos, (_, data) in enumerate(parsed_blocks):
        t_eq = float(data["t_eq_species"])
        color = data.get("color")
        try:
            color_1b = int(color)
        except Exception:
            color_1b = None

        p_mean = tail_mean_after_t_eq(data.get("time"), data.get("pt"), t_eq)
        f_mean = tail_mean_after_t_eq(data.get("time"), data.get("nt"), t_eq)
        z_max = zmax_for_order(
            order_pos=order_pos,
            color_1b=color_1b,
            t_eq_species=t_eq,
            result_t_eq_order=result_t_eq_order,
            meta_t_eq=meta_t_eq,
            meta_zmax=meta_zmax,
        )
        z_stat = zmax_for_order(
            order_pos=order_pos,
            color_1b=color_1b,
            t_eq_species=t_eq,
            result_t_eq_order=result_t_eq_order,
            meta_t_eq=meta_t_eq,
            meta_zmax=meta_zstat,
        )

        out.append({
            "order": order_pos,
            "color": color_1b,
            "t_eq_species": t_eq,
            "p_sample_mean": p_mean,
            "f_sample_mean": f_mean,
            "z_max": z_max,
            "z_stat": z_stat,
        })

    return out


def discover_data_dirs(raw_root: Path) -> list[Path]:
    return sorted(
        p for p in raw_root.rglob("data")
        if p.is_dir() and parse_data_dir(p) is not None
    )


def process_group(
    data_dir: Path,
    raw_root: Path,
    published_root: Path,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    params = parse_data_dir(data_dir)
    if params is None:
        raise ValueError(f"Could not parse dynamic data dir: {data_dir}")

    rel_group = data_dir.parent.relative_to(raw_root)
    out_dir = published_root / rel_group
    ensure_dir(out_dir)

    json_files = sorted(data_dir.glob("*.json"))
    groups: dict[tuple[float, float], list[Path]] = defaultdict(list)
    for fp in json_files:
        parsed_name = parse_sample_name(fp)
        if parsed_name is None:
            continue
        groups[parsed_name].append(fp)

    bundle: dict[str, Any] = {
        "meta": {
            **params,
            "raw_group": rel_group.as_posix(),
            "num_json_files": len(json_files),
        },
        "p0_groups": [],
    }

    all_rows: list[dict[str, Any]] = []
    all_color_rows: list[dict[str, Any]] = []

    for (P0, p0), files in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        by_order: dict[int, list[dict[str, Any]]] = defaultdict(list)
        stabilized_counts: list[float] = []
        processed = 0
        for fp in files:
            sample_orders = load_dynamic_sample(fp)
            if sample_orders is None:
                continue
            processed += 1
            stabilized_counts.append(float(len(sample_orders)))
            for item in sample_orders:
                by_order[int(item["order"])].append({**item, "filename": fp.name})

        nc_mean, nc_err, nc_std, n_nc = mean_sem_std(stabilized_counts)

        p0_group = {
            "P0_value": P0,
            "p0_value": p0,
            "num_samples_total": processed,
            "colors": {
                "Nsamples": processed,
                "nc": nc_mean,
                "nc_err": nc_err,
                "nc_std": nc_std,
                "n": n_nc,
                "values": stabilized_counts,
            },
            "orders": [],
        }

        all_color_rows.append({
            **params,
            "num_colors": params["nc"],
            "p0": p0,
            "P0": P0,
            "N_samples": processed,
            "nc": nc_mean,
            "nc_err": nc_err,
            "nc_std": nc_std,
        })

        for order in sorted(by_order):
            items = by_order[order]
            p_vals = [x["p_sample_mean"] for x in items if x["p_sample_mean"] is not None]
            f_vals = [x["f_sample_mean"] for x in items if x["f_sample_mean"] is not None]
            z_vals = [x["z_max"] for x in items if x["z_max"] is not None]
            z_stat_vals = [x["z_stat"] for x in items if x["z_stat"] is not None]
            teq_vals = [x["t_eq_species"] for x in items if x["t_eq_species"] is not None]

            p_mean, p_err, n_p = mean_sem(p_vals)
            f_mean, f_err, n_f = mean_sem(f_vals)
            z_mean, z_err, n_z = mean_sem(z_vals)
            z_stat_mean, z_stat_err, n_z_stat = mean_sem(z_stat_vals)
            teq_mean, teq_err, n_teq = mean_sem(teq_vals)

            N_samples_perc = len(items)
            order_block = {
                "order": order,
                "N_samples": processed,
                "N_samples_perc": N_samples_perc,
                "t_eq_species": {
                    "mean": teq_mean,
                    "err": teq_err,
                    "n": n_teq,
                    "values": teq_vals,
                },
                "p": {"mean": p_mean, "err": p_err, "n": n_p},
                "f": {"mean": f_mean, "err": f_err, "n": n_f},
                "z_max": {"mean": z_mean, "err": z_err, "n": n_z, "values": z_vals},
                "z_stat": {
                    "mean": z_stat_mean,
                    "err": z_stat_err,
                    "n": n_z_stat,
                    "values": z_stat_vals,
                },
                "samples": [
                    {
                        "filename": x["filename"],
                        "color": x["color"],
                        "t_eq_species": x["t_eq_species"],
                        "p_sample_mean": x["p_sample_mean"],
                        "f_sample_mean": x["f_sample_mean"],
                        "z_max": x["z_max"],
                        "z_stat": x["z_stat"],
                    }
                    for x in items
                ],
            }
            p0_group["orders"].append(order_block)

            all_rows.append({
                **params,
                "p0": p0,
                "P0": P0,
                "order": order,
                "N_samples": processed,
                "N_samples_perc": N_samples_perc,
                "p_mean": p_mean,
                "p_err": p_err,
                "f_mean": f_mean,
                "f_err": f_err,
                "z_max_mean": z_mean,
                "z_max_err": z_err,
                "z_stat_mean": z_stat_mean,
                "z_stat_err": z_stat_err,
            })

        bundle["p0_groups"].append(p0_group)

    out_path = out_dir / "properties_dynamic_bundle.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(bundle), f, ensure_ascii=False, indent=2)
        f.write("\n")

    return out_path, all_rows, all_color_rows


def write_all_data(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    rows = sorted(
        rows,
        key=lambda r: (
            r["type_perc"], r["dim"], r["nc"], r["rho"], r["c"], r["f_T"],
            r["L"], r["P0"], r["p0"], r["stat_window"], r["order"],
        ),
    )
    with output_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(ALL_DATA_COLUMNS) + "\n")
        for r in rows:
            f.write(" ".join(dat_value(r.get(col)) for col in ALL_DATA_COLUMNS) + "\n")


def write_all_colors(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    rows = sorted(
        rows,
        key=lambda r: (
            r["type_perc"], r["dim"], r["num_colors"], r["rho"], r["c"],
            r["f_T"], r["L"], r["P0"], r["p0"], r["stat_window"],
        ),
    )
    with output_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(ALL_COLORS_COLUMNS) + "\n")
        for r in rows:
            f.write(" ".join(dat_value(r.get(col)) for col in ALL_COLORS_COLUMNS) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process raw_growth_test_dynamic into published_dynamic and all_data_dynamic.dat."
    )
    parser.add_argument("--sop-root", default=str(Path(__file__).resolve().parents[1] / "SOP_data"))
    parser.add_argument("--raw-dir", default="raw_growth_test_dynamic")
    parser.add_argument("--published-dir", default="published_dynamic")
    parser.add_argument("--all-data-name", default="all_data_dynamic.dat")
    parser.add_argument("--all-colors-name", default="all_colors_dynamic.dat")
    parser.add_argument("--clear", action="store_true", help="Rebuild output files. Current implementation always rewrites.")
    args = parser.parse_args()

    sop_root = Path(args.sop_root).expanduser().resolve()
    raw_root = sop_root / args.raw_dir
    published_root = sop_root / args.published_dir
    ensure_dir(published_root)

    data_dirs = discover_data_dirs(raw_root)
    print(f"[dynamic] data dirs found: {len(data_dirs)}")

    all_rows: list[dict[str, Any]] = []
    all_color_rows: list[dict[str, Any]] = []
    for data_dir in data_dirs:
        out_path, rows, color_rows = process_group(data_dir, raw_root, published_root)
        all_rows.extend(rows)
        all_color_rows.extend(color_rows)
        print(f"[write] {out_path} ({len(rows)} rows)")

    all_data_path = sop_root / args.all_data_name
    write_all_data(all_rows, all_data_path)
    print(f"[write] {all_data_path} ({len(all_rows)} rows)")

    all_colors_path = sop_root / args.all_colors_name
    write_all_colors(all_color_rows, all_colors_path)
    print(f"[write] {all_colors_path} ({len(all_color_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
