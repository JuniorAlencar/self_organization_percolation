#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import lzma
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize


@dataclass
class Curve:
    path: Path
    L: float
    t: np.ndarray
    p: np.ndarray
    p_sem: np.ndarray | None
    label: str


def load_json(path: Path) -> dict:
    if path.suffix == ".xz":
        with lzma.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_bundle_curves(path: Path, p0: float | None, P0: float | None, order: int | None) -> list[Curve]:
    bundle = load_json(path)
    meta = bundle.get("meta", {})
    L = float(meta["L"])
    curves: list[Curve] = []

    for group in bundle.get("p0_groups", []):
        group_p0 = float(group.get("p0_value", np.nan))
        group_P0 = float(group.get("P0_value", np.nan))
        if p0 is not None and not np.isclose(group_p0, p0):
            continue
        if P0 is not None and not np.isclose(group_P0, P0):
            continue

        for order_block in group.get("orders", []):
            block_order = int(order_block.get("order", -1))
            if order is not None and block_order != order:
                continue

            data = order_block.get("data", {})
            t = np.asarray(data.get("pt_supported_time", data.get("time", [])), dtype=float)
            p = np.asarray(data.get("pt_supported_mean", data.get("pt_mean", [])), dtype=float)
            sem_raw = data.get("pt_supported_sem", data.get("pt_sem"))
            p_sem = np.asarray(sem_raw, dtype=float) if sem_raw is not None else None

            n = min(t.size, p.size, p_sem.size if p_sem is not None else t.size)
            if n < 4:
                continue

            t = t[:n]
            p = p[:n]
            p_sem = p_sem[:n] if p_sem is not None else None
            finite = np.isfinite(t) & np.isfinite(p)
            if p_sem is not None:
                finite &= np.isfinite(p_sem)
            t = t[finite]
            p = p[finite]
            p_sem = p_sem[finite] if p_sem is not None else None
            if t.size < 4:
                continue

            curves.append(
                Curve(
                    path=path,
                    L=L,
                    t=t,
                    p=p,
                    p_sem=p_sem,
                    label=f"L={int(L):g}, p0={group_p0:g}, P0={group_P0:g}, order={block_order}",
                )
            )
    return curves


def observable(curve: Curve, mode: str, tail_fraction: float) -> np.ndarray:
    p = curve.p
    if mode == "p":
        return p.copy()

    tail_start = max(0, min(p.size - 1, int((1.0 - tail_fraction) * p.size)))
    p_sat = float(np.nanmean(p[tail_start:]))

    if mode == "growth":
        return np.abs(p[0] - p)
    if mode == "relaxation":
        return np.abs(p - p_sat)
    if mode == "centered":
        return p - p_sat
    raise ValueError(f"Unknown observable mode: {mode}")


def downsample_log(t: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if t.size <= max_points:
        return t, y
    positive = t > 0
    if positive.sum() < max_points // 2:
        idx = np.linspace(0, t.size - 1, max_points).astype(int)
    else:
        log_idx = np.geomspace(1, t.size, max_points).astype(int) - 1
        idx = np.unique(np.clip(log_idx, 0, t.size - 1))
    return t[idx], y[idx]


def prepare_curves(
    curves: list[Curve],
    mode: str,
    tail_fraction: float,
    max_points: int,
    t_min: float | None,
    t_max: float | None,
) -> list[tuple[Curve, np.ndarray, np.ndarray]]:
    out = []
    for curve in curves:
        y = observable(curve, mode, tail_fraction)
        t = curve.t.copy()
        keep = np.isfinite(t) & np.isfinite(y)
        keep &= t > 0
        if mode in {"growth", "relaxation"}:
            keep &= y > 0
        if t_min is not None:
            keep &= t >= t_min
        if t_max is not None:
            keep &= t <= t_max
        t = t[keep]
        y = y[keep]
        if t.size >= 4:
            t, y = downsample_log(t, y, max_points)
            out.append((curve, t, y))
    return out


def estimate_beta(prepared, fit_range: tuple[float, float]) -> list[tuple[float, float, float]]:
    lo, hi = fit_range
    estimates = []
    for curve, t, y in prepared:
        t0 = t.min()
        t1 = t.max()
        fit_lo = t0 * (t1 / t0) ** lo
        fit_hi = t0 * (t1 / t0) ** hi
        keep = (t >= fit_lo) & (t <= fit_hi) & (y > 0)
        if keep.sum() < 4:
            continue
        slope, intercept = np.polyfit(np.log(t[keep]), np.log(y[keep]), 1)
        yhat = slope * np.log(t[keep]) + intercept
        ss_res = float(np.sum((np.log(y[keep]) - yhat) ** 2))
        ss_tot = float(np.sum((np.log(y[keep]) - np.mean(np.log(y[keep]))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        estimates.append((curve.L, float(slope), r2))
    return estimates


def collapse_score(params: np.ndarray, prepared, n_grid: int) -> float:
    alpha, z = params
    xs = []
    ys = []
    for curve, t, y in prepared:
        xs.append(t / curve.L**z)
        ys.append(y / curve.L**alpha)

    x_min = max(np.min(x) for x in xs)
    x_max = min(np.max(x) for x in xs)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return 1e30

    grid = np.geomspace(x_min, x_max, n_grid)
    interp = []
    for x, y in zip(xs, ys):
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        keep = np.isfinite(x_sorted) & np.isfinite(y_sorted)
        x_sorted = x_sorted[keep]
        y_sorted = y_sorted[keep]
        unique_x, unique_idx = np.unique(x_sorted, return_index=True)
        unique_y = y_sorted[unique_idx]
        if unique_x.size < 4:
            return 1e30
        interp.append(np.interp(grid, unique_x, unique_y))

    mat = np.vstack(interp)
    mean = np.mean(mat, axis=0)
    scale = np.maximum(np.abs(mean), np.nanmedian(np.abs(mat)) * 1e-6)
    return float(np.mean(((mat - mean) / scale) ** 2))


def fit_collapse(prepared, alpha_bounds, z_bounds, n_grid: int, seed: int) -> tuple[float, float, float]:
    bounds = [alpha_bounds, z_bounds]
    result = differential_evolution(
        lambda x: collapse_score(x, prepared, n_grid),
        bounds=bounds,
        seed=seed,
        polish=False,
        tol=1e-4,
    )
    polished = minimize(
        lambda x: collapse_score(x, prepared, n_grid),
        result.x,
        bounds=bounds,
        method="Nelder-Mead",
    )
    x = polished.x if polished.fun <= result.fun else result.x
    score = min(float(polished.fun), float(result.fun))
    return float(x[0]), float(x[1]), score


def plot_raw(prepared, out_path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for curve, t, y in prepared:
        ax.plot(t, y, lw=1.6, label=curve.label)
    ax.set_xscale("log")
    if np.all([np.all(y > 0) for _, _, y in prepared]):
        ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_collapse(prepared, alpha: float, z: float, out_path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for curve, t, y in prepared:
        ax.plot(t / curve.L**z, y / curve.L**alpha, lw=1.6, label=curve.label)
    ax.set_xscale("log")
    if np.all([np.all(y > 0) for _, _, y in prepared]):
        ax.set_yscale("log")
    ax.set_xlabel(r"$t / L^z$")
    ax.set_ylabel(rf"${ylabel} / L^\alpha$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(rf"$\alpha={alpha:.4g}$, $z={z:.4g}$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot p(t) and fit a Family-Vicsek-like collapse: y(t,L)=L^alpha F(t/L^z)."
    )
    parser.add_argument("bundles", nargs="+", type=Path, help="properties_dynamic_bundle.json[.gz|.xz] files")
    parser.add_argument("--p0", type=float, default=None)
    parser.add_argument("--P0", type=float, default=None)
    parser.add_argument("--order", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["p", "growth", "relaxation", "centered"],
        default="growth",
        help="Observable y(t): p, |p(0)-p(t)|, |p(t)-p_sat|, or p(t)-p_sat.",
    )
    parser.add_argument("--tail-fraction", type=float, default=0.2)
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument("--alpha-bounds", type=float, nargs=2, default=(-3.0, 3.0))
    parser.add_argument("--z-bounds", type=float, nargs=2, default=(0.05, 5.0))
    parser.add_argument("--grid-points", type=int, default=250)
    parser.add_argument("--beta-fit-range", type=float, nargs=2, default=(0.05, 0.35))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=Path, default=Path("results/FamilyVicsek"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    curves = []
    for bundle in args.bundles:
        curves.extend(iter_bundle_curves(bundle, p0=args.p0, P0=args.P0, order=args.order))

    if len(curves) < 2:
        raise SystemExit("Need at least two curves. Pass bundles for two or more L values.")

    prepared = prepare_curves(
        curves,
        mode=args.mode,
        tail_fraction=args.tail_fraction,
        max_points=args.max_points,
        t_min=args.t_min,
        t_max=args.t_max,
    )
    unique_L = sorted({curve.L for curve, _, _ in prepared})
    if len(unique_L) < 2:
        raise SystemExit("Need at least two distinct L values for a finite-size collapse.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ylabel = {
        "p": "p(t)",
        "growth": "|p(0)-p(t)|",
        "relaxation": "|p(t)-p_sat|",
        "centered": "p(t)-p_sat",
    }[args.mode]

    raw_path = args.out_dir / f"pt_raw_{args.mode}.png"
    collapse_path = args.out_dir / f"pt_collapse_{args.mode}.png"
    summary_path = args.out_dir / f"pt_family_vicsek_{args.mode}_summary.txt"

    beta_estimates = estimate_beta(prepared, tuple(args.beta_fit_range))
    alpha, z, score = fit_collapse(
        prepared,
        alpha_bounds=tuple(args.alpha_bounds),
        z_bounds=tuple(args.z_bounds),
        n_grid=args.grid_points,
        seed=args.seed,
    )

    plot_raw(prepared, raw_path, ylabel)
    plot_collapse(prepared, alpha, z, collapse_path, ylabel)

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"mode: {args.mode}\n")
        handle.write(f"alpha: {alpha:.12g}\n")
        handle.write(f"z: {z:.12g}\n")
        handle.write(f"collapse_score: {score:.12g}\n")
        handle.write(f"beta_from_alpha_over_z: {alpha / z:.12g}\n")
        handle.write("\nlog-log beta estimates by L:\n")
        for L, beta, r2 in beta_estimates:
            handle.write(f"L={L:g} beta={beta:.12g} r2={r2:.6g}\n")
        handle.write("\ncurves:\n")
        for curve, t, y in prepared:
            handle.write(
                f"L={curve.L:g} n={t.size} t=[{t.min():.6g},{t.max():.6g}] "
                f"y=[{np.nanmin(y):.6g},{np.nanmax(y):.6g}] path={curve.path}\n"
            )

    print(f"Saved raw plot: {raw_path}")
    print(f"Saved collapse plot: {collapse_path}")
    print(f"Saved summary: {summary_path}")
    print(f"alpha={alpha:.6g} z={z:.6g} beta=alpha/z={alpha / z:.6g} score={score:.6g}")
    if beta_estimates:
        print("beta log-log estimates:")
        for L, beta, r2 in beta_estimates:
            print(f"  L={L:g}: beta={beta:.6g}, r2={r2:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
