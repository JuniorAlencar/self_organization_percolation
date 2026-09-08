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
    "fit_method",
    "fit_n",
    "fit_t_min",
    "fit_t_max",
    "beta",
    "beta_intercept",
    "beta_r2",
    "beta_eff_mean",
    "beta_eff_std",
    "window_score",
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


def fit_range(time: np.ndarray, values: np.ndarray, t_min: float, t_max: float) -> dict:
    fit = np.isfinite(time) & np.isfinite(values) & (time >= t_min) & (time <= t_max) & (time > 0) & (values > 0)
    if int(fit.sum()) < 2:
        return {
            "beta": math.nan,
            "intercept": math.nan,
            "r2": math.nan,
            "n": int(fit.sum()),
            "t_min": t_min,
            "t_max": t_max,
            "beta_eff_mean": math.nan,
            "beta_eff_std": math.nan,
            "score": math.nan,
        }
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
        "t_min": float(t_min),
        "t_max": float(t_max),
        "beta_eff_mean": math.nan,
        "beta_eff_std": math.nan,
        "score": math.nan,
    }


def fit_power_law_fraction(time: np.ndarray, values: np.ndarray, frac_range: tuple[float, float]) -> dict:
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
            "beta_eff_mean": math.nan,
            "beta_eff_std": math.nan,
            "score": math.nan,
        }

    lo, hi = frac_range
    log_time = np.log(time)
    t_min = float(np.exp(log_time.min() + lo * (log_time.max() - log_time.min())))
    t_max = float(np.exp(log_time.min() + hi * (log_time.max() - log_time.min())))
    fit = (time >= t_min) & (time <= t_max)
    if int(fit.sum()) < 6:
        fit = np.ones_like(time, dtype=bool)
        t_min = float(time.min())
        t_max = float(time.max())

    return fit_range(time, values, t_min, t_max)


def log_bin_series(time: np.ndarray, values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(time) & np.isfinite(values) & (time > 0) & (values > 0)
    time = time[keep]
    values = values[keep]
    order = np.argsort(time)
    time = time[order]
    values = values[order]
    if time.size <= max_points:
        return time, values

    edges = np.geomspace(time.min(), time.max(), max_points + 1)
    bins = np.searchsorted(edges, time, side="right") - 1
    bins = np.clip(bins, 0, max_points - 1)
    binned_t = []
    binned_y = []
    for bin_idx in np.unique(bins):
        mask = bins == bin_idx
        binned_t.append(float(np.exp(np.mean(np.log(time[mask])))))
        binned_y.append(float(np.mean(values[mask])))
    return np.array(binned_t, dtype=np.float64), np.array(binned_y, dtype=np.float64)


def prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


def segment_sum(prefix: np.ndarray, start: int, end: int) -> float:
    return float(prefix[end] - prefix[start])


def auto_fit_power_law(
    time: np.ndarray,
    values: np.ndarray,
    *,
    max_points: int,
    min_points: int,
    min_decades: float,
    min_beta: float,
    max_beta_std: float,
    beta_std_weight: float,
) -> dict:
    btime, bvalues = log_bin_series(time, values, max_points)
    if btime.size < min_points:
        result = fit_power_law_fraction(time, values, (0.08, 0.55))
        result["score"] = math.nan
        return result

    x = np.log(btime)
    y = np.log(bvalues)
    beta_eff = np.gradient(y, x)
    px = prefix_sum(x)
    py = prefix_sum(y)
    pxx = prefix_sum(x * x)
    pyy = prefix_sum(y * y)
    pxy = prefix_sum(x * y)
    pb = prefix_sum(beta_eff)
    pbb = prefix_sum(beta_eff * beta_eff)

    min_log_width = min_decades * math.log(10.0)
    best: dict | None = None
    n_points = x.size
    for start in range(0, n_points - min_points + 1):
        for end in range(start + min_points, n_points + 1):
            if x[end - 1] - x[start] < min_log_width:
                continue
            n = end - start
            sum_x = segment_sum(px, start, end)
            sum_y = segment_sum(py, start, end)
            sum_xx = segment_sum(pxx, start, end)
            sum_yy = segment_sum(pyy, start, end)
            sum_xy = segment_sum(pxy, start, end)
            denom = n * sum_xx - sum_x * sum_x
            if denom <= 0:
                continue
            beta = (n * sum_xy - sum_x * sum_y) / denom
            if beta < min_beta:
                continue
            intercept = (sum_y - beta * sum_x) / n
            ss_tot = sum_yy - (sum_y * sum_y) / n
            ss_res = sum_yy + beta * beta * sum_xx + n * intercept * intercept
            ss_res += 2.0 * beta * intercept * sum_x - 2.0 * beta * sum_xy - 2.0 * intercept * sum_y
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
            beta_mean = segment_sum(pb, start, end) / n
            beta_var = segment_sum(pbb, start, end) / n - beta_mean * beta_mean
            beta_std = math.sqrt(max(beta_var, 0.0))
            if beta_std > max_beta_std:
                continue
            width_decades = (x[end - 1] - x[start]) / math.log(10.0)
            score = r2 - beta_std_weight * beta_std + 0.015 * math.log1p(width_decades)
            if best is None or score > best["score"]:
                best = {
                    "beta": float(beta),
                    "intercept": float(intercept),
                    "r2": float(r2),
                    "n": int(n),
                    "t_min": float(btime[start]),
                    "t_max": float(btime[end - 1]),
                    "beta_eff_mean": float(beta_mean),
                    "beta_eff_std": float(beta_std),
                    "score": float(score),
                }

    if best is None:
        result = fit_power_law_fraction(time, values, (0.08, 0.55))
        result["score"] = math.nan
        return result
    refined = fit_range(time, values, best["t_min"], best["t_max"])
    refined["beta_eff_mean"] = best["beta_eff_mean"]
    refined["beta_eff_std"] = best["beta_eff_std"]
    refined["score"] = best["score"]
    return refined


def effective_exponent(time: np.ndarray, values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    btime, bvalues = log_bin_series(time, values, max_points)
    if btime.size < 3:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return btime, np.gradient(np.log(bvalues), np.log(btime))


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_group(key: tuple, time: np.ndarray, mean: np.ndarray, sem: np.ndarray, fit: dict, out_dir: Path, max_eff_points: int) -> Path:
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
    fig, (ax, ax_eff) = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True, height_ratios=(2.2, 1.0))
    y_for_limits = [mean[np.isfinite(mean) & (mean > 0)]]
    ax.plot(time, mean, lw=1.4, color="#2f6f9f", label=r"$\langle W(t)\rangle$")
    if np.any(np.isfinite(sem)):
        tt = time[::step]
        yy = mean[::step]
        ee = sem[::step]
        lower = yy - ee
        upper = yy + ee
        valid_err = np.isfinite(tt) & np.isfinite(lower) & np.isfinite(upper) & (tt > 0) & (lower > 0) & (upper > 0)
        if np.any(valid_err):
            ax.fill_between(tt[valid_err], lower[valid_err], upper[valid_err], color="#2f6f9f", alpha=0.18, linewidth=0)
    if np.isfinite(fit["beta"]):
        tt = np.geomspace(fit["t_min"], fit["t_max"], 200)
        yy = np.exp(fit["intercept"]) * tt ** fit["beta"]
        y_for_limits.append(yy[np.isfinite(yy) & (yy > 0)])
        ax.plot(tt, yy, color="#d95f02", lw=3.0, label=rf"$\beta={fit['beta']:.4g}$")
        ax.axvspan(fit["t_min"], fit["t_max"], color="#d95f02", alpha=0.12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    y_positive = np.concatenate([values for values in y_for_limits if values.size])
    if y_positive.size:
        ymin = float(np.nanmin(y_positive))
        ymax = float(np.nanmax(y_positive))
        if ymin > 0 and ymax > ymin:
            pad = math.exp(0.08 * (math.log(ymax) - math.log(ymin)))
            ax.set_ylim(ymin / pad, ymax * pad)
    ax.set_ylabel(r"$\langle W(t)\rangle$")
    ax.legend(frameon=False)
    ax.set_title(
        f"{meta['type_perc']}, L={meta['L']}, f_T={float(meta['f_T']):.8g}, "
        f"c={float(meta['c']):g}, rho={float(meta['rho']):g}, color={meta['color']}"
    )
    eff_t, eff_beta = effective_exponent(time, mean, max_eff_points)
    if eff_t.size:
        ax_eff.plot(eff_t, eff_beta, lw=1.0, color="#9f5f2f", label=r"$\beta_{\mathrm{eff}}(t)$")
    if np.isfinite(fit["t_min"]) and np.isfinite(fit["t_max"]):
        ax_eff.axvspan(fit["t_min"], fit["t_max"], color="#d95f02", alpha=0.12)
        if np.isfinite(fit.get("beta", math.nan)):
            ax_eff.axhline(fit["beta"], color="#d95f02", lw=2.0)
    ax_eff.set_xscale("log")
    ax_eff.set_xlabel("t")
    ax_eff.set_ylabel(r"$d\log W/d\log t$")
    ax_eff.legend(frameon=False)
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
    parser.add_argument("--fit-method", choices=["auto", "fraction"], default="auto")
    parser.add_argument("--fit-frac-range", type=float, nargs=2, default=(0.08, 0.55))
    parser.add_argument("--auto-max-points", type=int, default=500)
    parser.add_argument("--auto-min-points", type=int, default=24)
    parser.add_argument("--auto-min-decades", type=float, default=0.35)
    parser.add_argument("--auto-min-beta", type=float, default=0.03)
    parser.add_argument("--auto-max-beta-std", type=float, default=0.20)
    parser.add_argument("--auto-beta-std-weight", type=float, default=0.35)
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
        if args.fit_method == "auto":
            fit = auto_fit_power_law(
                time[keep],
                mean[keep],
                max_points=args.auto_max_points,
                min_points=args.auto_min_points,
                min_decades=args.auto_min_decades,
                min_beta=args.auto_min_beta,
                max_beta_std=args.auto_max_beta_std,
                beta_std_weight=args.auto_beta_std_weight,
            )
        else:
            fit = fit_power_law_fraction(time[keep], mean[keep], tuple(args.fit_frac_range))
        plot_path = ""
        if not args.no_plots:
            plot_path = str(plot_group(key, time[keep], mean[keep], sem[keep], fit, args.out_dir, args.auto_max_points))
        row = dict(zip(GROUP_FIELDS, key))
        row.update(
            {
                "max_n": max_n,
                "fit_method": args.fit_method,
                "fit_n": fit["n"],
                "fit_t_min": fit["t_min"],
                "fit_t_max": fit["t_max"],
                "beta": fit["beta"],
                "beta_intercept": fit["intercept"],
                "beta_r2": fit["r2"],
                "beta_eff_mean": fit["beta_eff_mean"],
                "beta_eff_std": fit["beta_eff_std"],
                "window_score": fit["score"],
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
