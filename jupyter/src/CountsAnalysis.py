from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OBSERVABLES = {
    "d_f": {
        "x": "L_over_epsilon",
        "y": "N_bulk",
        "axis": "epsilon",
        "xlabel": r"$L/\epsilon$",
        "ylabel": r"$N_{\mathrm{bulk}}(\epsilon)$",
    },
    "d_hull": {
        "x": "L_over_epsilon",
        "y": "N_hull",
        "axis": "epsilon",
        "xlabel": r"$L/\epsilon$",
        "ylabel": r"$N_{\mathrm{hull}}(\epsilon)$",
    },
    "d_hull_ext": {
        "x": "L_over_epsilon",
        "y": "N_hull_ext",
        "axis": "epsilon",
        "xlabel": r"$L/\epsilon$",
        "ylabel": r"$N_{\mathrm{hull,ext}}(\epsilon)$",
    },
    "d_min": {
        "x": "r",
        "y": "ell_min",
        "axis": "bin_index",
        "xlabel": r"$r$",
        "ylabel": r"$\ell_{\min}$",
    },
}


def rounded_value_error(value, error, max_decimals: int = 6) -> str:
    value = float(value)
    error = float(error)
    if not np.isfinite(value):
        return "nan"
    if not np.isfinite(error) or error <= 0:
        return f"{value:.{max_decimals}g}"

    exponent = int(np.floor(np.log10(abs(error))))
    first_digit = error / 10**exponent
    sig_digits = 2 if first_digit < 3 else 1
    decimals = max(0, -exponent + sig_digits - 1)
    decimals = min(decimals, max_decimals)
    return f"{value:.{decimals}f} $\\pm$ {error:.{decimals}f}"


def counts_root(path: str | Path = "../SOP_data/published_counts") -> Path:
    return Path(path).expanduser().resolve()


def list_counts_files(root: str | Path = "../SOP_data/published_counts") -> list[Path]:
    root = counts_root(root)
    return sorted(root.glob("**/counts_P0_*_p0_*.json"))


def read_counts_file(path: str | Path) -> dict:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def processed_counts_index(root: str | Path = "../SOP_data/published_counts") -> pd.DataFrame:
    root = counts_root(root)
    index_path = root / "split_outputs.csv"
    if index_path.exists():
        df = pd.read_csv(index_path)
        df["path"] = df["path"].map(lambda p: str(Path(p).expanduser().resolve()))
        return df

    rows = []
    for path in list_counts_files(root):
        payload = read_counts_file(path)
        meta = payload.get("meta", {})
        rows.append({
            **{k: meta.get(k) for k in [
                "type_percolation",
                "num_colors",
                "dim",
                "L",
                "f_T",
                "c",
                "rho",
                "gap_over_L",
                "P0",
                "p0",
            ]},
            "path": str(path.resolve()),
            "n_count_files": meta.get("n_count_files"),
            "n_samples": meta.get("n_samples", len(payload.get("samples", []))),
        })
    return pd.DataFrame(rows)


def filter_index(df: pd.DataFrame, **filters) -> pd.DataFrame:
    out = df.copy()
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            out = out[out[key].isin(value)]
        else:
            out = out[out[key] == value]
    return out.reset_index(drop=True)


def sample_series(payload: dict, sample_index: int, observable: str) -> pd.DataFrame:
    info = OBSERVABLES[observable]
    sample = payload["samples"][sample_index]
    series = sample["properties"][observable]
    x_name = info["x"]
    y_name = info["y"]
    axis_name = info["axis"]
    return pd.DataFrame({
        "sample_index": sample.get("sample_index", sample_index),
        "sample_id": sample.get("sample_id"),
        "seed": sample.get("seed"),
        "observable": observable,
        axis_name: series.get(axis_name, range(len(series[x_name]))),
        x_name: series[x_name],
        y_name: series[y_name],
        **({
            "ell_mean": series.get("ell_mean", series.get("ell")),
            "ell_min": series.get("ell_min"),
            "ell_max": series.get("ell_max"),
            "ell_std": series.get("ell_std"),
            "truncated": series.get("truncated"),
            "pairs": series.get("pairs"),
        } if observable == "d_min" else {}),
    })


def all_sample_series(payload: dict, observable: str) -> pd.DataFrame:
    frames = [
        sample_series(payload, i, observable)
        for i in range(len(payload.get("samples", [])))
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_series(payload: dict, observable: str) -> pd.DataFrame:
    info = OBSERVABLES[observable]
    df = all_sample_series(payload, observable)
    if df.empty:
        return df

    axis_name = info["axis"]
    x_name = info["x"]
    y_name = info["y"]
    grouped = df.groupby(axis_name, as_index=False).agg(
        x_mean=(x_name, "mean"),
        x_std=(x_name, "std"),
        y_mean=(y_name, "mean"),
        y_std=(y_name, "std"),
        n_samples=("sample_id", "nunique"),
    )
    grouped["x_sem"] = grouped["x_std"] / np.sqrt(grouped["n_samples"])
    grouped["y_sem"] = grouped["y_std"] / np.sqrt(grouped["n_samples"])
    return grouped


def _fit_arrays(x, y, x_min=None, x_max=None, y_min=None, y_max=None, keep=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if keep is not None:
        mask &= np.asarray(keep, dtype=bool)
    if x_min is not None:
        mask &= x >= x_min
    if x_max is not None:
        mask &= x <= x_max
    if y_min is not None:
        mask &= y >= y_min
    if y_max is not None:
        mask &= y <= y_max
    return x[mask], y[mask]


def fit_loglog(
    x,
    y,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    keep=None,
    log_base: float = np.e,
) -> dict:
    x, y = _fit_arrays(x, y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, keep=keep)
    if len(x) < 2:
        return {
            "slope": np.nan,
            "slope_err": np.nan,
            "intercept": np.nan,
            "intercept_err": np.nan,
            "r2": np.nan,
            "n_points": len(x),
            "x_log": np.array([]),
            "y_log": np.array([]),
            "y_fit_log": np.array([]),
        }

    log = np.log if log_base == np.e else lambda v: np.log(v) / np.log(log_base)
    x_log = log(x)
    y_log = log(y)
    slope, intercept = np.polyfit(x_log, y_log, 1)
    y_fit_log = slope * x_log + intercept
    ss_tot = np.sum((y_log - y_log.mean()) ** 2)
    ss_res = np.sum((y_log - y_fit_log) ** 2)
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    slope_err = np.nan
    intercept_err = np.nan
    if len(x_log) > 2:
        sxx = np.sum((x_log - x_log.mean()) ** 2)
        residual_var = ss_res / (len(x_log) - 2)
        if sxx > 0:
            slope_err = np.sqrt(residual_var / sxx)
            intercept_err = np.sqrt(residual_var * (1.0 / len(x_log) + x_log.mean() ** 2 / sxx))
    return {
        "slope": slope,
        "slope_err": slope_err,
        "intercept": intercept,
        "intercept_err": intercept_err,
        "r2": r2,
        "n_points": len(x),
        "x_log": x_log,
        "y_log": y_log,
        "y_fit_log": y_fit_log,
    }


def fit_samples(
    payload: dict,
    observable: str,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    y_variant: str | None = None,
    ignore_truncated: bool = False,
    log_base: float = np.e,
) -> pd.DataFrame:
    info = OBSERVABLES[observable]
    rows = []
    for i, sample in enumerate(payload.get("samples", [])):
        series = sample["properties"][observable]
        x_values = series[info["x"]]
        y_name = y_variant or info["y"]
        y_values = series[y_name]
        keep = None
        if observable == "d_min" and ignore_truncated and "truncated" in series:
            keep = [not value for value in series["truncated"]]
        fit = fit_loglog(
            x_values,
            y_values,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            keep=keep,
            log_base=log_base,
        )
        rows.append({
            **{k: payload.get("meta", {}).get(k) for k in [
                "type_percolation",
                "num_colors",
                "dim",
                "L",
                "f_T",
                "c",
                "rho",
                "gap_over_L",
                "P0",
                "p0",
            ]},
            "observable": observable,
            "sample_index": sample.get("sample_index", i),
            "sample_id": sample.get("sample_id"),
            "seed": sample.get("seed"),
            "x_name": info["x"],
            "y_name": y_name,
            "ignore_truncated": ignore_truncated,
            "slope": fit["slope"],
            "slope_err": fit["slope_err"],
            "intercept": fit["intercept"],
            "intercept_err": fit["intercept_err"],
            "r2": fit["r2"],
            "n_points": fit["n_points"],
        })
    return pd.DataFrame(rows)


def fit_all_observables(payload: dict, **fit_kwargs) -> pd.DataFrame:
    frames = []
    for observable in OBSERVABLES:
        kwargs = dict(fit_kwargs)
        if observable != "d_min":
            kwargs.pop("y_variant", None)
            kwargs.pop("ignore_truncated", None)
        frames.append(fit_samples(payload, observable, **kwargs))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_fits(fits: pd.DataFrame) -> pd.DataFrame:
    if fits.empty:
        return fits
    return fits.groupby("observable", as_index=False).agg(
        slope_mean=("slope", "mean"),
        slope_std=("slope", "std"),
        slope_sem=("slope", lambda x: x.std() / np.sqrt(x.notna().sum())),
        r2_mean=("r2", "mean"),
        n_samples=("slope", "count"),
    )


def plot_sample(
    payload: dict,
    observable: str,
    sample_index: int = 0,
    fit: bool = True,
    x_min=None,
    x_max=None,
    y_variant: str | None = None,
    ignore_truncated: bool = False,
    ax=None,
    log_base: float = np.e,
    **scatter_kwargs,
):
    info = OBSERVABLES[observable]
    series = payload["samples"][sample_index]["properties"][observable]
    x = np.asarray(series[info["x"]], dtype=float)
    y_name = y_variant or info["y"]
    y = np.asarray(series[y_name], dtype=float)
    keep = None
    if observable == "d_min" and ignore_truncated and "truncated" in series:
        keep = np.asarray([not value for value in series["truncated"]], dtype=bool)
    ax = ax or plt.gca()
    ax.scatter(x, y, label=f"sample {sample_index}", **scatter_kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(info["xlabel"])
    ax.set_ylabel(info["ylabel"])

    result = None
    if fit:
        result = fit_loglog(x, y, x_min=x_min, x_max=x_max, keep=keep, log_base=log_base)
        x_fit, y_fit = _fit_arrays(x, y, x_min=x_min, x_max=x_max, keep=keep)
        if len(x_fit) >= 2:
            order = np.argsort(x_fit)
            log = np.log if log_base == np.e else lambda v: np.log(v) / np.log(log_base)
            y_line_log = result["slope"] * log(x_fit[order]) + result["intercept"]
            y_line = log_base ** y_line_log if log_base != np.e else np.exp(y_line_log)
            slope_label = rounded_value_error(result["slope"], result["slope_err"])
            ax.plot(x_fit[order], y_line, color="black", lw=1.5, label=f"slope = {slope_label}")
    ax.legend()
    return ax, result


def plot_mean(
    payload: dict,
    observable: str,
    fit: bool = True,
    x_min=None,
    x_max=None,
    y_variant: str | None = None,
    ignore_truncated: bool = False,
    ax=None,
    log_base: float = np.e,
    **errorbar_kwargs,
):
    info = OBSERVABLES[observable]
    df = aggregate_series(payload, observable)
    y_col = "y_mean"
    if observable == "d_min" and y_variant in {"ell_mean", "ell_min", "ell_max"}:
        raw = all_sample_series(payload, observable)
        keep_cols = [info["axis"], "r", y_variant]
        if ignore_truncated and "truncated" in raw.columns:
            raw = raw[~raw["truncated"].astype(bool)]
        df = raw[keep_cols].groupby(info["axis"], as_index=False).agg(
            x_mean=("r", "mean"),
            y_mean=(y_variant, "mean"),
            y_std=(y_variant, "std"),
            n_samples=("r", "count"),
        )
        df["y_sem"] = df["y_std"] / np.sqrt(df["n_samples"])
    ax = ax or plt.gca()
    ax.errorbar(
        df["x_mean"],
        df["y_mean"],
        yerr=df["y_sem"],
        fmt="o",
        capsize=3,
        label="mean",
        **errorbar_kwargs,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(info["xlabel"])
    ax.set_ylabel(info["ylabel"])

    result = None
    if fit:
        result = fit_loglog(df["x_mean"], df[y_col], x_min=x_min, x_max=x_max, log_base=log_base)
        x_fit, y_fit = _fit_arrays(df["x_mean"], df[y_col], x_min=x_min, x_max=x_max)
        if len(x_fit) >= 2:
            order = np.argsort(x_fit)
            log = np.log if log_base == np.e else lambda v: np.log(v) / np.log(log_base)
            y_line_log = result["slope"] * log(x_fit[order]) + result["intercept"]
            y_line = log_base ** y_line_log if log_base != np.e else np.exp(y_line_log)
            slope_label = rounded_value_error(result["slope"], result["slope_err"])
            ax.plot(x_fit[order], y_line, color="black", lw=1.5, label=f"slope = {slope_label}")
    ax.legend()
    return ax, result


def load_first_dataset(root: str | Path = "../SOP_data/published_counts") -> tuple[Path, dict]:
    files = list_counts_files(root)
    if not files:
        raise FileNotFoundError(f"No processed counts files found in {counts_root(root)}")
    path = files[0]
    return path, read_counts_file(path)
