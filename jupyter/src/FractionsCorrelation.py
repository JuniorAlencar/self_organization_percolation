from __future__ import annotations

import gzip
import json
import lzma
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_IGNORED_CORRELATION_COLUMNS = {
    "sample_index",
    "source_seed",
    "source_file_index",
    "bundle_path",
    "dim",
    "L",
    "num_colors",
    "type_percolation",
    "type_perc",
    "N_total",
    "E_total",
    "sample_gap_layers",
    "sample_gap_over_L",
    "mode",
    "source_mode",
    "schema_version",
    "P0",
    "p0",
    "num_source_files",
    "total_samples",
}


def load_fraction_json_bundle(path: str | Path) -> dict[str, Any]:
    """Load a fractions JSON bundle from .json, .json.gz, or .json.xz."""
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    if bundle_path.suffix == ".xz":
        with lzma.open(bundle_path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    elif bundle_path.suffix == ".gz":
        with gzip.open(bundle_path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        with bundle_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid fractions JSON bundle: {bundle_path}")
    return data


def _published_fractions_roots(sop_root: str | Path) -> list[Path]:
    root = Path(sop_root)
    return [
        path
        for path in (root / "published_fractions", root / "fractions_published")
        if path.exists()
    ]


def discover_published_fraction_bundles(
    root: str | Path = "../SOP_data",
    pattern: str = "*.json.xz",
) -> list[Path]:
    """
    Discover published fraction bundles below SOP_data.

    `root` may be SOP_data, published_fractions/fractions_published, a parameter
    directory, or a single bundle file.
    """
    root_path = Path(root)
    if root_path.is_file():
        return [root_path]
    if root_path.name in {"published_fractions", "fractions_published"}:
        search_roots = [root_path]
    else:
        search_roots = _published_fractions_roots(root_path)
        if not search_roots and root_path.exists():
            search_roots = [root_path]
    bundles = []
    for search_root in search_roots:
        bundles.extend(path for path in search_root.rglob(pattern) if path.is_file())
    return sorted(bundles)


def published_fractions_dataframe(
    root: str | Path = "../SOP_data",
    measures: list[str] | tuple[str, ...] | None = None,
    pattern: str = "*.json.xz",
) -> pd.DataFrame:
    """
    Load SOP_data published fraction bundles into one row per fraction sample.

    Metadata from `meta` is repeated in each row. Columns from `data` are aligned
    by sample index and may be filtered with `measures`.
    """
    rows: list[dict[str, Any]] = []
    for bundle_path in discover_published_fraction_bundles(root, pattern=pattern):
        bundle = load_fraction_json_bundle(bundle_path)
        meta = bundle.get("meta", {})
        data = bundle.get("data", {})
        if not isinstance(meta, dict) or not isinstance(data, dict):
            continue

        data_lists = {key: value for key, value in data.items() if isinstance(value, list)}
        if measures is not None:
            wanted = set(measures) | {"source_seed", "source_file_index"}
            data_lists = {key: value for key, value in data_lists.items() if key in wanted}
        n_samples = max((len(values) for values in data_lists.values()), default=0)
        if n_samples == 0:
            continue

        meta_row = {
            key: value
            for key, value in meta.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        for idx in range(n_samples):
            row = {
                **meta_row,
                "bundle_path": bundle_path.as_posix(),
                "sample_index": idx,
            }
            for key, values in data_lists.items():
                row[key] = values[idx] if idx < len(values) else None
            rows.append(row)

    return pd.DataFrame(rows)


def correlation_between_fraction_samples(
    root: str | Path = "../SOP_data",
    measures: list[str] | tuple[str, ...] | None = None,
    method: str = "pearson",
    min_periods: int = 2,
    axis: str = "samples",
    group_cols: list[str] | tuple[str, ...] | None = ("bundle_path",),
    pattern: str = "*.json.xz",
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Calculate correlations from published fraction bundles.

    axis="samples" correlates samples using the selected measures as features.
    axis="measures" correlates measures using samples as observations.
    By default, returns one correlation matrix per bundle. Set group_cols=None
    to calculate a single matrix over all loaded samples.
    """
    if axis not in {"samples", "measures"}:
        raise ValueError("axis must be 'samples' or 'measures'")

    df = published_fractions_dataframe(root=root, measures=measures, pattern=pattern)
    if df.empty:
        return {} if group_cols else pd.DataFrame()

    if measures is None:
        numeric_cols = [
            col
            for col in df.columns
            if (
                col not in DEFAULT_IGNORED_CORRELATION_COLUMNS
                and pd.api.types.is_numeric_dtype(df[col])
            )
        ]
    else:
        numeric_cols = [
            col
            for col in measures
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

    def _corr(group: pd.DataFrame) -> pd.DataFrame:
        values = group[numeric_cols]
        if axis == "measures":
            return values.corr(method=method, min_periods=min_periods)
        labels = (
            group["source_seed"].astype("Int64").astype(str)
            + ":"
            + group["sample_index"].astype("Int64").astype(str)
            if "source_seed" in group.columns
            else group["sample_index"].astype("Int64").astype(str)
        )
        values = values.copy()
        values.index = labels
        return values.T.corr(method=method, min_periods=min_periods)

    if group_cols is None:
        return _corr(df)

    out: dict[str, pd.DataFrame] = {}
    available_group_cols = [col for col in group_cols if col in df.columns]
    if not available_group_cols:
        return {"all": _corr(df)}

    group_key: str | list[str]
    group_key = available_group_cols[0] if len(available_group_cols) == 1 else available_group_cols
    for key, group in df.groupby(group_key, dropna=False):
        if isinstance(key, str):
            label = key
        elif isinstance(key, tuple):
            label = "|".join(map(str, key))
        else:
            label = str(key)
        out[label] = _corr(group)
    return out


def _as_list_or_none(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _matches_filter(value: Any, allowed: list[Any] | None, *, atol: float = 1e-12) -> bool:
    if allowed is None:
        return True
    for item in allowed:
        try:
            if np.isclose(float(value), float(item), atol=atol, rtol=0.0):
                return True
        except (TypeError, ValueError):
            if value == item:
                return True
    return False


def _normal_two_sided_pvalue(z: float) -> float:
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def _fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r):
        return (np.nan, np.nan)
    r_clip = float(np.clip(r, -0.999999999999, 0.999999999999))
    z = np.arctanh(r_clip)
    se = 1.0 / math.sqrt(n - 3)
    zcrit = 1.959963984540054 if alpha == 0.05 else 1.959963984540054
    return (
        float(np.tanh(z - zcrit * se)),
        float(np.tanh(z + zcrit * se)),
    )


def lagged_sample_correlation(values: Any, lag: int = 1, min_pairs: int = 2) -> dict[str, float | int]:
    """
    Correlate a sample sequence with itself shifted by `lag`.

    For lag=1, this is corr(sample_k, sample_{k+1}) for one measure.
    """
    if lag <= 0:
        raise ValueError("lag must be positive")
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if arr.size <= lag:
        return {
            "correlation": np.nan,
            "abs_correlation": np.nan,
            "p_value": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_pairs": 0,
            "lag": int(lag),
        }

    x = arr[:-lag]
    y = arr[lag:]
    valid = np.isfinite(x) & np.isfinite(y)
    n_pairs = int(valid.sum())
    if n_pairs < min_pairs:
        return {
            "correlation": np.nan,
            "abs_correlation": np.nan,
            "p_value": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_pairs": n_pairs,
            "lag": int(lag),
        }
    if np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return {
            "correlation": np.nan,
            "abs_correlation": np.nan,
            "p_value": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_pairs": n_pairs,
            "lag": int(lag),
        }
    corr = float(np.corrcoef(x[valid], y[valid])[0, 1])
    if n_pairs > 3 and np.isfinite(corr):
        z = np.arctanh(float(np.clip(corr, -0.999999999999, 0.999999999999))) * math.sqrt(n_pairs - 3)
        p_value = _normal_two_sided_pvalue(z)
    else:
        p_value = np.nan
    ci95_low, ci95_high = _fisher_ci(corr, n_pairs)
    return {
        "correlation": corr,
        "abs_correlation": abs(corr),
        "p_value": p_value,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "n_pairs": n_pairs,
        "lag": int(lag),
    }


def measure_correlations_by_L_and_gap(
    root: str | Path = "../SOP_data",
    measure: str = "p_stab_bond",
    L_values: int | list[int] | tuple[int, ...] | None = None,
    gaps: float | list[float] | tuple[float, ...] | None = (0.5, 1.0, 1.5, 2.0),
    lags: int | list[int] | tuple[int, ...] = (1,),
    min_pairs: int = 2,
    pattern: str = "*.json.xz",
) -> pd.DataFrame:
    """
    Compute lagged sample correlations for one measure, grouped by L and gap.

    Returns one row per bundle and lag. The most common use is lag=1:
    corr(measure_sample_k, measure_sample_{k+1}) for each parameter set.
    """
    L_allowed = _as_list_or_none(L_values)
    gap_allowed = _as_list_or_none(gaps)
    lag_values = [int(lag) for lag in _as_list_or_none(lags)]
    rows: list[dict[str, Any]] = []

    for bundle_path in discover_published_fraction_bundles(root, pattern=pattern):
        bundle = load_fraction_json_bundle(bundle_path)
        meta = bundle.get("meta", {})
        data = bundle.get("data", {})
        if not isinstance(meta, dict) or not isinstance(data, dict):
            continue
        if measure not in data:
            continue

        L = meta.get("L")
        gap = meta.get("sample_gap_over_L")
        if not _matches_filter(L, L_allowed):
            continue
        if not _matches_filter(gap, gap_allowed):
            continue

        series = data.get(measure)
        if not isinstance(series, list):
            continue

        base_row = {
            "L": L,
            "gap": gap,
            "measure": measure,
            "dim": meta.get("dim"),
            "num_colors": meta.get("num_colors"),
            "type_percolation": meta.get("type_percolation"),
            "f_T": meta.get("f_T"),
            "c": meta.get("c"),
            "rho": meta.get("rho"),
            "P0": meta.get("P0"),
            "p0": meta.get("p0"),
            "n_samples": len(series),
            "bundle_path": bundle_path.as_posix(),
        }
        for lag in lag_values:
            rows.append({
                **base_row,
                **lagged_sample_correlation(series, lag=lag, min_pairs=min_pairs),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["L", "gap", "f_T", "c", "P0", "p0", "lag"],
        ignore_index=True,
    )


def read_fraction_values_by_L_and_gap(
    root: str | Path = "../SOP_data",
    measures: str | list[str] | tuple[str, ...] | None = None,
    L_values: int | list[int] | tuple[int, ...] | None = None,
    gaps: float | list[float] | tuple[float, ...] | None = None,
    pattern: str = "*.json.xz",
    return_format: str = "dict",
) -> dict[tuple[Any, Any], list[dict[str, Any]]] | pd.DataFrame:
    """
    Read published fraction bundles and return values grouped by (L, gap).

    return_format="dict" returns:
        {(L, gap): [{"meta": ..., "data": ..., "bundle_path": ...}, ...]}

    return_format="dataframe" returns one row per sample, including L, gap,
    sample_index, bundle_path, metadata, and the selected measures.
    """
    if return_format not in {"dict", "dataframe"}:
        raise ValueError("return_format must be 'dict' or 'dataframe'")

    if isinstance(measures, str):
        measure_names = [measures]
    else:
        measure_names = _as_list_or_none(measures)
    L_allowed = _as_list_or_none(L_values)
    gap_allowed = _as_list_or_none(gaps)

    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    dataframe_rows: list[dict[str, Any]] = []

    for bundle_path in discover_published_fraction_bundles(root, pattern=pattern):
        bundle = load_fraction_json_bundle(bundle_path)
        meta = bundle.get("meta", {})
        data = bundle.get("data", {})
        if not isinstance(meta, dict) or not isinstance(data, dict):
            continue

        L = meta.get("L")
        gap = meta.get("sample_gap_over_L")
        if not _matches_filter(L, L_allowed):
            continue
        if not _matches_filter(gap, gap_allowed):
            continue

        if measure_names is None:
            selected_data = {
                key: value
                for key, value in data.items()
                if isinstance(value, list)
            }
        else:
            selected_data = {
                key: data[key]
                for key in measure_names
                if key in data and isinstance(data[key], list)
            }
        if not selected_data:
            continue

        meta_row = {
            key: value
            for key, value in meta.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        item = {
            "meta": meta,
            "data": selected_data,
            "bundle_path": bundle_path.as_posix(),
        }
        grouped.setdefault((L, gap), []).append(item)

        if return_format == "dataframe":
            n_samples = max(len(values) for values in selected_data.values())
            for idx in range(n_samples):
                row = {
                    **meta_row,
                    "L": L,
                    "gap": gap,
                    "sample_index": idx,
                    "bundle_path": bundle_path.as_posix(),
                }
                for key, values in selected_data.items():
                    row[key] = values[idx] if idx < len(values) else None
                dataframe_rows.append(row)

    if return_format == "dataframe":
        return pd.DataFrame(dataframe_rows)
    return grouped


def summarize_gap_independence(
    correlations: pd.DataFrame,
    *,
    group_cols: list[str] | tuple[str, ...] = ("L", "f_T", "c", "P0", "p0"),
) -> pd.DataFrame:
    """
    Summarize autocorrelation results and rank gaps for each parameter set.

    The best gap is the row with the smallest max_abs_correlation across the
    lags present in `correlations`.
    """
    if correlations.empty:
        return pd.DataFrame()

    available_group_cols = [col for col in group_cols if col in correlations.columns]
    summary_cols = available_group_cols + ["gap"]
    summary = (
        correlations.groupby(summary_cols, dropna=False)
        .agg(
            max_abs_correlation=("abs_correlation", "max"),
            mean_abs_correlation=("abs_correlation", "mean"),
            max_p_value=("p_value", "max"),
            min_p_value=("p_value", "min"),
            n_lags=("lag", "nunique"),
            min_n_pairs=("n_pairs", "min"),
            n_samples=("n_samples", "max"),
            measure=("measure", "first"),
            bundle_path=("bundle_path", "first"),
        )
        .reset_index()
    )
    if available_group_cols:
        summary = summary.sort_values(
            available_group_cols + ["max_abs_correlation", "mean_abs_correlation", "gap"],
            ignore_index=True,
        )
        summary["gap_rank"] = summary.groupby(available_group_cols, dropna=False).cumcount() + 1
    else:
        summary = summary.sort_values(
            ["max_abs_correlation", "mean_abs_correlation", "gap"],
            ignore_index=True,
        )
        summary["gap_rank"] = np.arange(1, len(summary) + 1)
    summary["is_best_gap"] = summary["gap_rank"] == 1
    return summary
