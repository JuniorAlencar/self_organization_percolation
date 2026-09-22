"""Data preparation for the initial-condition analysis notebooks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


BASE_COLUMNS = {
    "type_perc",
    "dim",
    "L",
    "f_T",
    "c",
    "nc",
    "rho",
    "p0",
    "P0",
    "order",
    "N_samples",
    "N_samples_perc",
    "p_mean",
}

HEATMAP_METRICS = (
    "p_star_at_f_T_min",
    "p_star_at_one_third",
    "p_star_at_two_thirds",
    "p_star_at_f_T_max",
)


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _close(series: pd.Series, value: float) -> np.ndarray:
    return np.isclose(series.to_numpy(dtype=float), value, rtol=1e-9, atol=1e-12)


def filter_initial_condition_data(
    df: pd.DataFrame,
    L: int,
    *,
    dim: int = 2,
    c: float = 0.1,
    nc: int = 1,
    rho: float = 1.0,
    order: int = 0,
    rule_update: str = "relative",
    type_perc: str | None = None,
) -> pd.DataFrame:
    """Select one model setup while retaining every available ``(p0, P0)`` pair."""
    _require_columns(df, BASE_COLUMNS)

    mask = (
        (df["L"] == L)
        & (df["dim"] == dim)
        & (df["nc"] == nc)
        & (df["order"] == order)
        & _close(df["c"], c)
        & _close(df["rho"], rho)
        & (df["control_rule"] == rule_update)
    )
    if type_perc is not None:
        mask &= df["type_perc"].eq(type_perc)

    return df.loc[mask].copy().sort_values(
        ["type_perc", "P0", "p0", "f_T"], ignore_index=True
    )


def get_initial_condition_combinations(
    df: pd.DataFrame,
    L: int,
    **filter_kwargs,
) -> list[tuple[float, float]]:
    """Return sorted unique ``(p0, P0)`` pairs available for a model setup."""
    selected = filter_initial_condition_data(df, L, **filter_kwargs)
    pairs = selected.loc[:, ["p0", "P0"]].drop_duplicates().sort_values(["p0", "P0"])
    return list(pairs.itertuples(index=False, name=None))


def get_missing_initial_condition_combinations(
    df: pd.DataFrame,
    L: int,
    all_combinations: Iterable[tuple[float, float]],
    **filter_kwargs,
) -> list[tuple[float, float]]:
    """Return possible pairs that have no row at all in the selected setup.

    A pair counts as executed as soon as it is present in the dataframe. Sample
    quality is deliberately ignored here: an executed pair with fewer than 95%
    percolating samples is not reported as missing.
    """
    executed = get_initial_condition_combinations(df, L, **filter_kwargs)
    executed_keys = {(round(float(p0), 12), round(float(P0), 12)) for p0, P0 in executed}

    missing = []
    for p0, P0 in all_combinations:
        pair = (float(p0), float(P0))
        key = (round(pair[0], 12), round(pair[1], 12))
        if key not in executed_keys:
            missing.append(pair)
    return missing


def prepare_initial_condition_maps(
    df: pd.DataFrame,
    L: int,
    *,
    dim: int = 2,
    c: float = 0.1,
    nc: int = 1,
    rho: float = 1.0,
    order: int = 0,
    types: tuple[str, ...] = ("node", "bond"),
    min_percolating_samples: int = 5,
    min_percolating_fraction: float = 0.95,
    p_star_min: float = 0.0,
    p_star_max: float = 0.9,
    f_T_max: float | None = 0.4,
    rule_update: str = "relative",
) -> dict[str, dict[str, object]]:
    """Prepare curve summaries and heatmap matrices for each percolation type.

    Every heatmap contains p-star. The four maps sample each valid f_T interval
    at normalized positions 0, 1/3, 2/3 and 1. Linear interpolation is used at
    the two internal positions. Missing initial-condition pairs remain NaN.
    By default, a point is valid when at least 95% of its samples percolate.
    """
    selected = filter_initial_condition_data(
        df, L, dim=dim, c=c, nc=nc, rho=rho, order=order, rule_update=rule_update,
    )
    selected = selected[selected["type_perc"].isin(types)].copy()

    if not 0.0 <= min_percolating_fraction <= 1.0:
        raise ValueError("min_percolating_fraction must be between 0 and 1")

    valid = (
        selected["p_mean"].between(p_star_min, p_star_max, inclusive="left")
        & (selected["N_samples_perc"] >= min_percolating_samples)
        & (
            selected["N_samples_perc"]
            >= min_percolating_fraction * selected["N_samples"]
        )
    )
    if f_T_max is not None:
        valid &= selected["f_T"] <= f_T_max

    selected = selected.loc[valid].copy()
    result: dict[str, dict[str, object]] = {}

    for type_perc in types:
        curves = selected[selected["type_perc"] == type_perc].copy()
        rows = []

        for (p0, P0), curve in curves.groupby(["p0", "P0"], sort=True):
            curve = curve.sort_values("f_T")
            f_values = curve["f_T"].to_numpy(dtype=float)
            p_values = curve["p_mean"].to_numpy(dtype=float)
            f_min = f_values[0]
            f_max = f_values[-1]
            delta_f = f_max - f_min
            f_one_third = f_min + delta_f / 3.0
            f_two_thirds = f_min + 2.0 * delta_f / 3.0
            min_row = curve.loc[curve["p_mean"].idxmin()]
            rows.append(
                {
                    "type_perc": type_perc,
                    "p0": p0,
                    "P0": P0,
                    "f_T_min": f_min,
                    "f_T_one_third": f_one_third,
                    "f_T_two_thirds": f_two_thirds,
                    "f_T_max": f_max,
                    "delta_f_T": delta_f,
                    "p_star_at_f_T_min": p_values[0],
                    "p_star_at_one_third": np.interp(f_one_third, f_values, p_values),
                    "p_star_at_two_thirds": np.interp(f_two_thirds, f_values, p_values),
                    "p_star_at_f_T_max": p_values[-1],
                    "p_star_min": min_row["p_mean"],
                    "f_T_at_p_star_min": min_row["f_T"],
                    "p_star_min_err": min_row.get("p_err", np.nan),
                    "n_f_T": curve["f_T"].nunique(),
                }
            )

        summary = pd.DataFrame(rows)
        all_pairs = get_initial_condition_combinations(
            df,
            L,
            dim=dim,
            c=c,
            nc=nc,
            rho=rho,
            order=order,
            type_perc=type_perc,
            rule_update=rule_update
        )
        p0_values = sorted({pair[0] for pair in all_pairs})
        P0_values = sorted({pair[1] for pair in all_pairs})

        maps = {}
        for metric in HEATMAP_METRICS:
            if summary.empty:
                maps[metric] = pd.DataFrame(index=P0_values, columns=p0_values, dtype=float)
            else:
                maps[metric] = (
                    summary.pivot(index="P0", columns="p0", values=metric)
                    .reindex(index=P0_values, columns=p0_values)
                )
                maps[metric].index.name = "P0"
                maps[metric].columns.name = "p0"

        result[type_perc] = {
            "curves": curves,
            "summary": summary,
            "combinations": all_pairs,
            "maps": maps,
        }

    return result
