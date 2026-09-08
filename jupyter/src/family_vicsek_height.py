from __future__ import annotations

import gzip
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_COLUMNS = ["type_perc", "num_colors", "dim", "L", "f_T", "c", "rho", "color"]


def load_ensemble_series(path="../SOP_data/processed_height_timeseries/height_ensemble_timeseries_all.csv.gz"):
    return pd.read_csv(path)


def load_sample_summary(path="../SOP_data/processed_height_timeseries/height_group_summary.csv.gz"):
    return pd.read_csv(path)


def filter_params(df, *, type_perc=None, f_T=None, c=None, rho=None, color=0, L=None):
    out = df.copy()
    if type_perc is not None:
        out = out[out["type_perc"] == type_perc]
    if f_T is not None:
        out = out[np.isclose(out["f_T"].astype(float), float(f_T))]
    if c is not None:
        out = out[np.isclose(out["c"].astype(float), float(c))]
    if rho is not None:
        out = out[np.isclose(out["rho"].astype(float), float(rho))]
    if color is not None:
        out = out[out["color"].astype(int) == int(color)]
    if L is not None:
        values = [L] if np.isscalar(L) else list(L)
        out = out[out["L"].astype(int).isin([int(value) for value in values])]
    return out


def available_parameter_sets(df):
    cols = [col for col in ["type_perc", "f_T", "c", "rho", "color"] if col in df.columns]
    return df[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)


def _fit_range(time, values, t_min, t_max):
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(time) & np.isfinite(values) & (time >= t_min) & (time <= t_max) & (time > 0) & (values > 0)
    if keep.sum() < 2:
        return {
            "beta": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "n": int(keep.sum()),
            "t_min": float(t_min),
            "t_max": float(t_max),
            "beta_eff_mean": np.nan,
            "beta_eff_std": np.nan,
            "score": np.nan,
        }
    x = np.log(time[keep])
    y = np.log(values[keep])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "beta": float(slope),
        "intercept": float(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "n": int(keep.sum()),
        "t_min": float(t_min),
        "t_max": float(t_max),
        "beta_eff_mean": np.nan,
        "beta_eff_std": np.nan,
        "score": np.nan,
    }


def log_bin_series(time, values, max_points=500):
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
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
    return np.array(binned_t), np.array(binned_y)


def effective_exponent(time, values, max_points=500):
    btime, bvalues = log_bin_series(time, values, max_points=max_points)
    if btime.size < 3:
        return np.array([]), np.array([])
    return btime, np.gradient(np.log(bvalues), np.log(btime))


def _prefix(values):
    return np.concatenate(([0.0], np.cumsum(values, dtype=float)))


def _segment(prefix, start, end):
    return float(prefix[end] - prefix[start])


def auto_fit_power_law(
    time,
    values,
    *,
    max_points=500,
    min_points=24,
    min_decades=0.35,
    min_beta=0.03,
    max_beta_std=0.20,
    beta_std_weight=0.35,
):
    btime, bvalues = log_bin_series(time, values, max_points=max_points)
    if btime.size < min_points:
        return _fit_range(time, values, np.nanmin(time), np.nanmax(time))

    x = np.log(btime)
    y = np.log(bvalues)
    beta_eff = np.gradient(y, x)
    px, py = _prefix(x), _prefix(y)
    pxx, pyy, pxy = _prefix(x * x), _prefix(y * y), _prefix(x * y)
    pb, pbb = _prefix(beta_eff), _prefix(beta_eff * beta_eff)
    min_log_width = min_decades * math.log(10.0)
    best = None

    for start in range(0, x.size - min_points + 1):
        for end in range(start + min_points, x.size + 1):
            if x[end - 1] - x[start] < min_log_width:
                continue
            n = end - start
            sx, sy = _segment(px, start, end), _segment(py, start, end)
            sxx, syy, sxy = _segment(pxx, start, end), _segment(pyy, start, end), _segment(pxy, start, end)
            denom = n * sxx - sx * sx
            if denom <= 0:
                continue
            beta = (n * sxy - sx * sy) / denom
            if beta < min_beta:
                continue
            intercept = (sy - beta * sx) / n
            ss_tot = syy - sy * sy / n
            ss_res = syy + beta * beta * sxx + n * intercept * intercept
            ss_res += 2.0 * beta * intercept * sx - 2.0 * beta * sxy - 2.0 * intercept * sy
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            beta_mean = _segment(pb, start, end) / n
            beta_std = math.sqrt(max(_segment(pbb, start, end) / n - beta_mean * beta_mean, 0.0))
            if beta_std > max_beta_std:
                continue
            width_decades = (x[end - 1] - x[start]) / math.log(10.0)
            score = r2 - beta_std_weight * beta_std + 0.015 * math.log1p(width_decades)
            if best is None or score > best["score"]:
                best = {
                    "t_min": float(btime[start]),
                    "t_max": float(btime[end - 1]),
                    "beta_eff_mean": float(beta_mean),
                    "beta_eff_std": float(beta_std),
                    "score": float(score),
                }

    if best is None:
        return _fit_range(time, values, np.nanmin(btime), np.nanmax(btime))
    fit = _fit_range(time, values, best["t_min"], best["t_max"])
    fit.update(best)
    return fit


def _fit_loglog(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    keep = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0) & (y_values > 0)
    x = np.log(x_values[keep])
    y = np.log(y_values[keep])
    if x.size < 2:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(x.size)}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "n": int(x.size),
    }


def plot_w_mean_with_beta(
    ensemble_df,
    *,
    type_perc,
    f_T,
    c,
    L,
    rho=None,
    color=0,
    min_count=2,
    min_n_frac=0.5,
    ax=None,
    ax_eff=None,
    fit_kwargs=None,
):
    data = filter_params(ensemble_df, type_perc=type_perc, f_T=f_T, c=c, rho=rho, color=color, L=L)
    if data.empty:
        raise ValueError("No data found for the selected parameters.")
    data = data.sort_values("t")
    max_n = int(data["n"].max())
    min_n = max(int(min_count), int(math.ceil(max_n * min_n_frac)))
    data = data[data["n"] >= min_n]
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    fit = auto_fit_power_law(data["t"], data["y_width_mean"], **fit_kwargs)

    if ax is None or ax_eff is None:
        fig, (ax, ax_eff) = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True, height_ratios=(2.2, 1.0))
    else:
        fig = ax.figure

    t = data["t"].to_numpy(float)
    w = data["y_width_mean"].to_numpy(float)
    sem = data["y_width_sem"].to_numpy(float)
    y_for_limits = [w[np.isfinite(w) & (w > 0)]]
    ax.plot(t, w, color="#225ea8", lw=1.6, label=r"$\langle W(t)\rangle$")
    if np.isfinite(sem).any():
        step = max(1, math.ceil(t.size / 2500))
        tt_err = t[::step]
        lower = w[::step] - sem[::step]
        upper = w[::step] + sem[::step]
        valid_err = np.isfinite(tt_err) & np.isfinite(lower) & np.isfinite(upper) & (tt_err > 0) & (lower > 0) & (upper > 0)
        if np.any(valid_err):
            ax.fill_between(tt_err[valid_err], lower[valid_err], upper[valid_err], color="#225ea8", alpha=0.16, lw=0)
    if np.isfinite(fit["beta"]):
        tt = np.geomspace(fit["t_min"], fit["t_max"], 200)
        yy = np.exp(fit["intercept"]) * tt ** fit["beta"]
        y_for_limits.append(yy[np.isfinite(yy) & (yy > 0)])
        ax.axvspan(fit["t_min"], fit["t_max"], color="#d95f02", alpha=0.12, lw=0)
        ax.plot(tt, yy, color="#d95f02", lw=3.0, label=rf"fit: $\beta={fit['beta']:.3g}$")
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
    ax.set_title(f"{type_perc}, L={L}, f_T={float(f_T):.8g}, c={float(c):g}, color={color}")

    eff_t, eff_beta = effective_exponent(t, w)
    ax_eff.plot(eff_t, eff_beta, color="#756bb1", lw=1.2, label=r"$\beta_{\mathrm{eff}}(t)$")
    if np.isfinite(fit["beta"]):
        ax_eff.axvspan(fit["t_min"], fit["t_max"], color="#d95f02", alpha=0.12, lw=0)
        ax_eff.axhline(fit["beta"], color="#d95f02", lw=2.0)
    ax_eff.set_xscale("log")
    ax_eff.set_xlabel("t")
    ax_eff.set_ylabel(r"$d\log W/d\log t$")
    ax_eff.legend(frameon=False)
    fig.tight_layout()
    return fig, (ax, ax_eff), fit


def plot_alpha_estimate(summary_df, *, type_perc, f_T, c, rho=None, color=0, ax=None):
    data = filter_params(summary_df, type_perc=type_perc, f_T=f_T, c=c, rho=rho, color=color)
    data = data.sort_values("L")
    if data.empty:
        raise ValueError("No data found for the selected parameters.")
    fit = _fit_loglog(data["L"], data["y_width_sat_mean"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
    else:
        fig = ax.figure
    ax.errorbar(data["L"], data["y_width_sat_mean"], yerr=data.get("y_width_sat_sem"), fmt="o", color="#225ea8", capsize=3, label=r"$W_{\mathrm{sat}}(L)$")
    if np.isfinite(fit["slope"]):
        ll = np.geomspace(data["L"].min(), data["L"].max(), 200)
        yy = np.exp(fit["intercept"]) * ll ** fit["slope"]
        ax.plot(ll, yy, color="#d95f02", lw=2.8, label=rf"fit: $\alpha={fit['slope']:.3g}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$W_{\mathrm{sat}}$")
    ax.set_title(f"alpha: {type_perc}, f_T={float(f_T):.8g}, c={float(c):g}, color={color}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax, fit


def plot_z_estimate(summary_df, *, type_perc, f_T, c, rho=None, color=0, t_column="t_star_90_mean", ax=None):
    data = filter_params(summary_df, type_perc=type_perc, f_T=f_T, c=c, rho=rho, color=color)
    data = data.sort_values("L")
    if data.empty:
        raise ValueError("No data found for the selected parameters.")
    fit = _fit_loglog(data["L"], data[t_column])
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
    else:
        fig = ax.figure
    ax.plot(data["L"], data[t_column], "o", color="#225ea8", label=t_column)
    if np.isfinite(fit["slope"]):
        ll = np.geomspace(data["L"].min(), data["L"].max(), 200)
        yy = np.exp(fit["intercept"]) * ll ** fit["slope"]
        ax.plot(ll, yy, color="#d95f02", lw=2.8, label=rf"fit: $z={fit['slope']:.3g}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("L")
    ax.set_ylabel(t_column)
    ax.set_title(f"z: {type_perc}, f_T={float(f_T):.8g}, c={float(c):g}, color={color}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax, fit
