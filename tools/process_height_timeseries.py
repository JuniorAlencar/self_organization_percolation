#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PATH_RE = re.compile(
    r"(?P<type_perc>bond|node)_percolation/"
    r"num_colors_(?P<num_colors>\d+)/dim_(?P<dim>\d+)/"
    r"L_(?P<L>\d+)/fT_constant/fT_(?P<f_T>[0-9.eE+-]+)/"
    r"c_(?P<c>[0-9.eE+-]+)"
    r"(?:/f0_(?P<f0>[0-9.eE+-]+))?"
    r"(?:/epsilon_(?P<epsilon>[0-9.eE+-]+))?"
    r"/rho_(?P<rho>[0-9.eE+-]+)/data/"
    r"(?P<sample>[^/]+)\.yts$"
)
SAMPLE_RE = re.compile(
    r"seed_(?P<seed>\d+)_ts_(?P<ts>[^_]+)_P0_(?P<P0>[0-9.eE+-]+)_p0_(?P<p0>[0-9.eE+-]+)"
)

MAGIC_YTS = 0x53545059
VERSION_YTS = 1
HEADER_BYTES = 20
PER_COLOR_BYTES = 8 + 8 + 4 + 8 + 8 + 4

MEASURE_FIELDS = [
    "type_perc",
    "num_colors",
    "dim",
    "L",
    "f_T",
    "c",
    "f0",
    "epsilon",
    "rho",
    "sample_id",
    "seed",
    "P0",
    "p0",
    "color",
    "nt",
    "t_min",
    "t_max",
    "y_width_min",
    "y_width_max",
    "y_width_tail_mean",
    "y_width_tail_std",
    "y_width_sat",
    "y_mean_tail",
    "y_max_final",
    "y_front_count_total",
    "beta",
    "beta_intercept",
    "beta_r2",
    "beta_n",
    "beta_t_min",
    "beta_t_max",
    "t_star_50",
    "t_star_90",
    "source_path",
]

SUMMARY_FIELDS = [
    "type_perc",
    "num_colors",
    "dim",
    "L",
    "f_T",
    "c",
    "f0",
    "epsilon",
    "rho",
    "color",
    "n_samples",
    "beta_mean",
    "beta_std",
    "beta_sem",
    "beta_r2_mean",
    "y_width_sat_mean",
    "y_width_sat_std",
    "y_width_sat_sem",
    "t_star_50_mean",
    "t_star_90_mean",
    "nt_min",
    "nt_max",
]


@dataclass(frozen=True)
class PathMeta:
    type_perc: str
    num_colors: int
    dim: int
    L: int
    f_T: float
    c: float
    f0: float | None
    epsilon: float | None
    rho: float
    sample: str
    seed: int | None
    P0: float | None
    p0: float | None


@dataclass
class YtsData:
    time: np.ndarray
    y_mean: list[np.ndarray]
    y_width: list[np.ndarray]
    y_max: list[np.ndarray]
    y_front_mean: list[np.ndarray]
    y_front_width: list[np.ndarray]
    y_front_count: list[np.ndarray]


def parse_path(path: Path) -> PathMeta:
    match = PATH_RE.search(str(path))
    if not match:
        raise ValueError(f"could not parse parameters from path: {path}")
    meta = match.groupdict()
    sample_match = SAMPLE_RE.search(meta["sample"])
    return PathMeta(
        type_perc=meta["type_perc"],
        num_colors=int(meta["num_colors"]),
        dim=int(meta["dim"]),
        L=int(meta["L"]),
        f_T=float(meta["f_T"]),
        c=float(meta["c"]),
        f0=float(meta["f0"]) if meta.get("f0") else None,
        epsilon=float(meta["epsilon"]) if meta.get("epsilon") else None,
        rho=float(meta["rho"]),
        sample=meta["sample"],
        seed=int(sample_match.group("seed")) if sample_match else None,
        P0=float(sample_match.group("P0")) if sample_match else None,
        p0=float(sample_match.group("p0")) if sample_match else None,
    )


def group_key(meta: PathMeta) -> tuple:
    return (
        meta.type_perc,
        meta.num_colors,
        meta.dim,
        meta.L,
        meta.f_T,
        meta.c,
        meta.f0,
        meta.epsilon,
        meta.rho,
    )


def read_yts(path: Path, colors: Iterable[int] | None = None) -> YtsData:
    with path.open("rb") as handle:
        magic, version, num_colors, nt = struct.unpack("<IIIQ", handle.read(HEADER_BYTES))
        if magic != MAGIC_YTS:
            raise ValueError(f"invalid .yts magic in {path}: {hex(magic)}")
        if version != VERSION_YTS:
            raise ValueError(f"unsupported .yts version {version} in {path}")
        selected = list(range(num_colors)) if colors is None else sorted(set(colors))
        bad_colors = [color for color in selected if color < 0 or color >= num_colors]
        if bad_colors:
            raise ValueError(f"colors outside [0,{num_colors}) for {path}: {bad_colors}")

        time = np.frombuffer(handle.read(4 * nt), dtype="<i4").astype(np.float64)
        row_bytes = int(nt) * PER_COLOR_BYTES
        arrays: dict[int, tuple[np.ndarray, ...]] = {}
        for color in range(num_colors):
            if color not in selected:
                handle.seek(row_bytes, 1)
                continue
            y_mean = np.frombuffer(handle.read(8 * nt), dtype="<f8").copy()
            y_width = np.frombuffer(handle.read(8 * nt), dtype="<f8").copy()
            y_max = np.frombuffer(handle.read(4 * nt), dtype="<i4").astype(np.float64)
            y_front_mean = np.frombuffer(handle.read(8 * nt), dtype="<f8").copy()
            y_front_width = np.frombuffer(handle.read(8 * nt), dtype="<f8").copy()
            y_front_count = np.frombuffer(handle.read(4 * nt), dtype="<u4").astype(np.float64)
            arrays[color] = (y_mean, y_width, y_max, y_front_mean, y_front_width, y_front_count)

    return YtsData(
        time=time,
        y_mean=[arrays[color][0] for color in selected],
        y_width=[arrays[color][1] for color in selected],
        y_max=[arrays[color][2] for color in selected],
        y_front_mean=[arrays[color][3] for color in selected],
        y_front_width=[arrays[color][4] for color in selected],
        y_front_count=[arrays[color][5] for color in selected],
    )


def finite_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    return (
        float(finite.min()),
        float(finite.max()),
        float(finite.mean()),
        float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
    )


def tail_values(values: np.ndarray, fraction: float) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return finite
    tail_n = max(1, int(math.ceil(finite.size * fraction)))
    return finite[-tail_n:]


def first_crossing_time(time: np.ndarray, values: np.ndarray, target: float) -> float:
    if not np.isfinite(target):
        return math.nan
    keep = np.isfinite(time) & np.isfinite(values)
    time = time[keep]
    values = values[keep]
    if time.size == 0:
        return math.nan
    hits = np.flatnonzero(values >= target)
    return float(time[hits[0]]) if hits.size else math.nan


def fit_power_law(time: np.ndarray, values: np.ndarray, frac_range: tuple[float, float]) -> dict:
    keep = np.isfinite(time) & np.isfinite(values) & (time > 0) & (values > 0)
    time = time[keep]
    values = values[keep]
    if time.size < 8:
        return {
            "beta": math.nan,
            "intercept": math.nan,
            "r2": math.nan,
            "n": int(time.size),
            "t_min": math.nan,
            "t_max": math.nan,
        }

    lo, hi = frac_range
    log_time = np.log(time)
    t_min = float(np.exp(log_time.min() + lo * (log_time.max() - log_time.min())))
    t_max = float(np.exp(log_time.min() + hi * (log_time.max() - log_time.min())))
    fit_mask = (time >= t_min) & (time <= t_max)
    if int(fit_mask.sum()) < 6:
        fit_mask = np.ones_like(time, dtype=bool)
        t_min = float(time.min())
        t_max = float(time.max())

    x = np.log(time[fit_mask])
    y = np.log(values[fit_mask])
    beta, intercept = np.polyfit(x, y, 1)
    pred = beta * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "beta": float(beta),
        "intercept": float(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan,
        "n": int(fit_mask.sum()),
        "t_min": t_min,
        "t_max": t_max,
    }


def measure_sample(
    path: Path,
    meta: PathMeta,
    colors: list[int] | None,
    fit_frac_range: tuple[float, float],
    tail_fraction: float,
) -> list[dict]:
    data = read_yts(path, colors)
    selected_colors = list(range(len(data.y_width))) if colors is None else sorted(set(colors))
    rows = []
    for idx, color in enumerate(selected_colors):
        width = data.y_width[idx]
        width_min, width_max, _, _ = finite_stats(width)
        tail_width = tail_values(width, tail_fraction)
        tail_mean = float(tail_width.mean()) if tail_width.size else math.nan
        tail_std = float(tail_width.std(ddof=1)) if tail_width.size > 1 else 0.0 if tail_width.size else math.nan
        y_mean_tail = tail_values(data.y_mean[idx], tail_fraction)
        width_sat = tail_mean
        fit = fit_power_law(data.time, width, fit_frac_range)
        row = {
            "type_perc": meta.type_perc,
            "num_colors": meta.num_colors,
            "dim": meta.dim,
            "L": meta.L,
            "f_T": f"{meta.f_T:.17g}",
            "c": f"{meta.c:.17g}",
            "f0": "" if meta.f0 is None else f"{meta.f0:.17g}",
            "epsilon": "" if meta.epsilon is None else f"{meta.epsilon:.17g}",
            "rho": f"{meta.rho:.17g}",
            "sample_id": meta.sample,
            "seed": "" if meta.seed is None else meta.seed,
            "P0": "" if meta.P0 is None else f"{meta.P0:.17g}",
            "p0": "" if meta.p0 is None else f"{meta.p0:.17g}",
            "color": color,
            "nt": int(data.time.size),
            "t_min": float(np.nanmin(data.time)) if data.time.size else math.nan,
            "t_max": float(np.nanmax(data.time)) if data.time.size else math.nan,
            "y_width_min": width_min,
            "y_width_max": width_max,
            "y_width_tail_mean": tail_mean,
            "y_width_tail_std": tail_std,
            "y_width_sat": width_sat,
            "y_mean_tail": float(y_mean_tail.mean()) if y_mean_tail.size else math.nan,
            "y_max_final": float(data.y_max[idx][-1]) if data.y_max[idx].size else math.nan,
            "y_front_count_total": float(np.nansum(data.y_front_count[idx])),
            "beta": fit["beta"],
            "beta_intercept": fit["intercept"],
            "beta_r2": fit["r2"],
            "beta_n": fit["n"],
            "beta_t_min": fit["t_min"],
            "beta_t_max": fit["t_max"],
            "t_star_50": first_crossing_time(data.time, width, 0.5 * width_sat),
            "t_star_90": first_crossing_time(data.time, width, 0.9 * width_sat),
            "source_path": str(path),
        }
        rows.append(row)
    return rows


def write_rows_csv_gz(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def output_group_dir(raw_root: Path, output_root: Path, first_path: Path) -> Path:
    group_dir = first_path.parent.parent
    try:
        rel = group_dir.relative_to(raw_root)
    except ValueError:
        rel = Path(*[part for part in group_dir.parts if part not in {"", "/"}][-8:])
    return output_root / rel


def collect_files(root: Path, args: argparse.Namespace) -> dict[tuple, list[tuple[PathMeta, Path]]]:
    groups: dict[tuple, list[tuple[PathMeta, Path]]] = defaultdict(list)
    for path in sorted(root.rglob("*.yts")):
        try:
            meta = parse_path(path)
        except ValueError:
            continue
        if args.type_perc and meta.type_perc != args.type_perc:
            continue
        if args.lengths and meta.L not in args.lengths:
            continue
        if args.f_T and not any(math.isclose(meta.f_T, value, rel_tol=1e-12, abs_tol=1e-12) for value in args.f_T):
            continue
        groups[group_key(meta)].append((meta, path))
    if args.max_samples_per_group:
        groups = {
            key: values[: args.max_samples_per_group]
            for key, values in groups.items()
        }
    return groups


def summarize_group(rows: list[dict]) -> dict:
    first = rows[0]

    def arr(name: str) -> np.ndarray:
        return np.array([float(row[name]) for row in rows if str(row[name]) != ""], dtype=float)

    beta = arr("beta")
    beta = beta[np.isfinite(beta)]
    beta_r2 = arr("beta_r2")
    beta_r2 = beta_r2[np.isfinite(beta_r2)]
    width_sat = arr("y_width_sat")
    width_sat = width_sat[np.isfinite(width_sat)]
    t50 = arr("t_star_50")
    t50 = t50[np.isfinite(t50)]
    t90 = arr("t_star_90")
    t90 = t90[np.isfinite(t90)]
    nt = arr("nt")

    def mean(values: np.ndarray) -> float:
        return float(values.mean()) if values.size else math.nan

    def std(values: np.ndarray) -> float:
        return float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size else math.nan

    def sem(values: np.ndarray) -> float:
        return std(values) / math.sqrt(values.size) if values.size else math.nan

    return {
        "type_perc": first["type_perc"],
        "num_colors": first["num_colors"],
        "dim": first["dim"],
        "L": first["L"],
        "f_T": first["f_T"],
        "c": first["c"],
        "f0": first["f0"],
        "epsilon": first["epsilon"],
        "rho": first["rho"],
        "color": first["color"],
        "n_samples": len(rows),
        "beta_mean": mean(beta),
        "beta_std": std(beta),
        "beta_sem": sem(beta),
        "beta_r2_mean": mean(beta_r2),
        "y_width_sat_mean": mean(width_sat),
        "y_width_sat_std": std(width_sat),
        "y_width_sat_sem": sem(width_sat),
        "t_star_50_mean": mean(t50),
        "t_star_90_mean": mean(t90),
        "nt_min": int(np.nanmin(nt)) if nt.size else "",
        "nt_max": int(np.nanmax(nt)) if nt.size else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process compact .yts height time series into sample-level measures."
    )
    parser.add_argument("--root", type=Path, default=Path("SOP_data/raw_growth_test_dynamic"))
    parser.add_argument("--out-root", type=Path, default=Path("SOP_data/processed_height_timeseries"))
    parser.add_argument("--type-perc", choices=["bond", "node"], default=None)
    parser.add_argument("--lengths", type=int, nargs="*", default=None)
    parser.add_argument("--f-T", type=float, nargs="*", default=None)
    parser.add_argument("--colors", type=int, nargs="*", default=None)
    parser.add_argument("--fit-frac-range", type=float, nargs=2, default=(0.08, 0.55))
    parser.add_argument("--tail-fraction", type=float, default=0.20)
    parser.add_argument("--max-samples-per-group", type=int, default=None)
    args = parser.parse_args()

    if not 0 < args.tail_fraction <= 1:
        raise SystemExit("--tail-fraction must be in (0, 1].")
    if not 0 <= args.fit_frac_range[0] < args.fit_frac_range[1] <= 1:
        raise SystemExit("--fit-frac-range must satisfy 0 <= lo < hi <= 1.")

    groups = collect_files(args.root, args)
    if not groups:
        raise SystemExit("No .yts files selected.")

    summary_rows: list[dict] = []
    all_sample_rows: list[dict] = []
    total_samples = 0
    for key, items in sorted(groups.items()):
        group_rows: list[dict] = []
        for meta, path in items:
            group_rows.extend(
                measure_sample(
                    path,
                    meta,
                    args.colors,
                    tuple(args.fit_frac_range),
                    args.tail_fraction,
                )
            )
        if not group_rows:
            print(f"[warn] {key}: no rows written; check --colors.")
            continue
        by_color: dict[int, list[dict]] = defaultdict(list)
        for row in group_rows:
            by_color[int(row["color"])].append(row)
        for rows in by_color.values():
            summary_rows.append(summarize_group(rows))

        out_dir = output_group_dir(args.root, args.out_root, items[0][1])
        write_rows_csv_gz(out_dir / "height_sample_measures.csv.gz", group_rows, MEASURE_FIELDS)
        all_sample_rows.extend(group_rows)
        total_samples += len(items)
        print(f"[ok] {key}: {len(items)} samples -> {out_dir / 'height_sample_measures.csv.gz'}")

    if not all_sample_rows:
        raise SystemExit("No sample rows were produced.")

    write_rows_csv_gz(args.out_root / "height_sample_measures_all.csv.gz", all_sample_rows, MEASURE_FIELDS)
    write_rows_csv_gz(args.out_root / "height_group_summary.csv.gz", summary_rows, SUMMARY_FIELDS)
    print(f"[done] processed {total_samples} samples in {len(groups)} groups")
    print(f"[done] samples -> {args.out_root / 'height_sample_measures_all.csv.gz'}")
    print(f"[done] summary -> {args.out_root / 'height_group_summary.csv.gz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
