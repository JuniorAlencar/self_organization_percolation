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

HEATMAP_METRICS = ("f_T_min", "f_T_max", "delta_f_T", "p_star_min")


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
    require_all_samples: bool = True,
    p_star_min: float = 0.0,
    p_star_max: float = 0.9,
    f_T_max: float | None = 0.4,
) -> dict[str, dict[str, object]]:
    """Prepare curve summaries and heatmap matrices for each percolation type.

    The four heatmap observables are ``f_T_min``, ``f_T_max``, ``delta_f_T``
    and ``p_star_min``. Missing initial-condition pairs remain NaN in the
    matrices. The validity defaults reproduce the cuts used in the exploratory
    plots in ``1Color_2D.ipynb``.
    """
    selected = filter_initial_condition_data(
        df, L, dim=dim, c=c, nc=nc, rho=rho, order=order
    )
    selected = selected[selected["type_perc"].isin(types)].copy()

    valid = (
        selected["p_mean"].between(p_star_min, p_star_max, inclusive="left")
        & (selected["N_samples_perc"] >= min_percolating_samples)
    )
    if require_all_samples:
        valid &= selected["N_samples_perc"] >= selected["N_samples"]
    if f_T_max is not None:
        valid &= selected["f_T"] <= f_T_max

    selected = selected.loc[valid].copy()
    result: dict[str, dict[str, object]] = {}

    for type_perc in types:
        curves = selected[selected["type_perc"] == type_perc].copy()
        rows = []

        for (p0, P0), curve in curves.groupby(["p0", "P0"], sort=True):
            curve = curve.sort_values("f_T")
            min_row = curve.loc[curve["p_mean"].idxmin()]
            rows.append(
                {
                    "type_perc": type_perc,
                    "p0": p0,
                    "P0": P0,
                    "f_T_min": curve["f_T"].min(),
                    "f_T_max": curve["f_T"].max(),
                    "delta_f_T": curve["f_T"].max() - curve["f_T"].min(),
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
