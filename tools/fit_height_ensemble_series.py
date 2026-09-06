#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUP_FIELDS = ["type_perc", "num_colors", "dim", "L", "f_T", "c", "rho", "color"]
SUMMARY_FIELDS = GROUP_FIELDS + [
    "max_n",
    "fit_n",
    "fit_t_min",
    "fit_t_max",
    "beta",
    "beta_intercept",
    "beta_r2",
    "plot_path",
]


def open_text(path: Path):
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open("rt", newline="")


def load_rows(path: Path, args: argparse.Namespace) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if args.type_perc and row["type_perc"] != args.type_perc:
                continue
            if args.lengths and int(row["L"]) not in args.lengths:
                continue
            if args.f_T and not any(math.isclose(float(row["f_T"]), value, rel_tol=1e-12, abs_tol=1e-12) for value in args.f_T):
                continue
            if args.colors and int(row["color"]) not in args.colors:
                continue
            key = tuple(row[field] for field in GROUP_FIELDS)
            groups[key].append(row)
    return groups


def fit_power_law(time: np.ndarray, values: np.ndarray, frac_range: tuple[float, float]) -> dict:
    keep = np.isfinite(time) & np.isfinite(values) & (time > 0) & (values > 0)
    time = time[keep]
    values = values[keep]
    if time.size < 8:
        return {"beta": math.nan, "intercept": math.nan, "r2": math.nan, "n": int(time.size), "t_min": math.nan, "t_max": math.nan}

    lo, hi = frac_range
    log_time = np.log(time)
    t_min = float(np.exp(log_time.min() + lo * (log_time.max() - log_time.min())))
    t_max = float(np.exp(log_time.min() + hi * (log_time.max() - log_time.min())))
    fit = (time >= t_min) & (time <= t_max)
    if int(fit.sum()) < 6:
        fit = np.ones_like(time, dtype=bool)
        t_min = float(time.min())
        t_max = float(time.max())

    x = np.log(time[fit])
    y = np.log(values[fit])
    beta, intercept = np.polyfit(x, y, 1)
    pred = beta * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "beta": float(beta),
        "intercept": float(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan,
        "n": int(fit.sum()),
        "t_min": t_min,
        "t_max": t_max,
    }


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_group(key: tuple, time: np.ndarray, mean: np.ndarray, sem: np.ndarray, fit: dict, out_dir: Path) -> Path:
    meta = dict(zip(GROUP_FIELDS, key))
    name = (
        f"{meta['type_perc']}_L{meta['L']}_fT{float(meta['f_T']):.8g}_"
        f"c{float(meta['c']):.4g}_rho{float(meta['rho']):.4g}_color{meta['color']}.png"
    )
    out_path = out_dir / name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    order = np.argsort(time)
    time = time[order]
    mean = mean[order]
    sem = sem[order]
    keep = np.isfinite(time) & np.isfinite(mean) & (time > 0) & (mean > 0)
    time = time[keep]
    mean = mean[keep]
    sem = sem[keep]

    step = max(1, int(math.ceil(time.size / 2500)))
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(time, mean, lw=1.4, color="#2f6f9f", label=r"$\langle W(t)\rangle$")
    if np.any(np.isfinite(sem)):
        tt = time[::step]
        yy = mean[::step]
        ee = sem[::step]
        ax.fill_between(tt, np.maximum(yy - ee, 1e-300), yy + ee, color="#2f6f9f", alpha=0.18, linewidth=0)
    if np.isfinite(fit["beta"]):
        tt = np.geomspace(fit["t_min"], fit["t_max"], 200)
        yy = np.exp(fit["intercept"]) * tt ** fit["beta"]
        ax.plot(tt, yy, "--", color="#202020", lw=1.2, label=rf"$\beta={fit['beta']:.4g}$")
        ax.axvspan(fit["t_min"], fit["t_max"], color="#202020", alpha=0.08)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\langle W(t)\rangle$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    ax.set_title(
        f"{meta['type_perc']}, L={meta['L']}, f_T={float(meta['f_T']):.8g}, "
        f"c={float(meta['c']):g}, rho={float(meta['rho']):g}, color={meta['color']}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit and plot ensemble-averaged height time series."
    )
    parser.add_argument("--input", type=Path, default=Path("SOP_data/processed_height_timeseries/height_ensemble_timeseries_all.csv.gz"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/FamilyVicsek/height_ensemble_fits"))
    parser.add_argument("--type-perc", choices=["bond", "node"], default=None)
    parser.add_argument("--lengths", type=int, nargs="*", default=None)
    parser.add_argument("--f-T", type=float, nargs="*", default=None)
    parser.add_argument("--colors", type=int, nargs="*", default=None)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--min-n-frac", type=float, default=0.5)
    parser.add_argument("--fit-frac-range", type=float, nargs=2, default=(0.08, 0.55))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.min_n_frac <= 1:
        raise SystemExit("--min-n-frac must be in [0, 1].")
    if not 0 <= args.fit_frac_range[0] < args.fit_frac_range[1] <= 1:
        raise SystemExit("--fit-frac-range must satisfy 0 <= lo < hi <= 1.")

    groups = load_rows(args.input, args)
    if not groups:
        raise SystemExit("No ensemble rows selected.")

    summary_rows: list[dict] = []
    for key, rows in sorted(groups.items()):
        time = np.array([float(row["t"]) for row in rows], dtype=np.float64)
        mean = np.array([float(row["y_width_mean"]) for row in rows], dtype=np.float64)
        sem = np.array([float(row["y_width_sem"]) for row in rows], dtype=np.float64)
        count = np.array([int(row["n"]) for row in rows], dtype=np.int32)
        max_n = int(count.max()) if count.size else 0
        min_n = max(args.min_count, int(math.ceil(max_n * args.min_n_frac)))
        keep = count >= min_n
        fit = fit_power_law(time[keep], mean[keep], tuple(args.fit_frac_range))
        plot_path = ""
        if not args.no_plots:
            plot_path = str(plot_group(key, time[keep], mean[keep], sem[keep], fit, args.out_dir))
        row = dict(zip(GROUP_FIELDS, key))
        row.update(
            {
                "max_n": max_n,
                "fit_n": fit["n"],
                "fit_t_min": fit["t_min"],
                "fit_t_max": fit["t_max"],
                "beta": fit["beta"],
                "beta_intercept": fit["intercept"],
                "beta_r2": fit["r2"],
                "plot_path": plot_path,
            }
        )
        summary_rows.append(row)
        print(f"[ok] {key}: beta={fit['beta']:.6g}, R2={fit['r2']:.6g}, n={fit['n']}")

    out_summary = args.out_dir / "height_ensemble_fit_summary.csv.gz"
    write_summary(out_summary, summary_rows)
    print(f"[done] summary -> {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
