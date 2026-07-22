import json
import gzip
import lzma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def _resolve_json_bundle_path(fn):
    path = Path(fn)
    if path.exists():
        return path
    if path.name.endswith(".json"):
        for suffix in (".xz", ".gz"):
            candidate = Path(str(path) + suffix)
            if candidate.exists():
                return candidate
    return path


def _load_json_bundle(fn):
    path = _resolve_json_bundle_path(fn)
    if path.suffix == ".xz":
        with lzma.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_properties_bundle(fn):
    return _load_json_bundle(fn)


def get_group_by_p0_P0(bundle, p0_target, P0_target, tol=1e-12):
    """
    Seleciona o bloco correto usando simultaneamente p0_value e P0_value.
    """
    for group in bundle["p0_groups"]:
        p0_value = float(group["p0_value"])
        P0_value = float(group["P0_value"])

        if abs(p0_value - p0_target) < tol and abs(P0_value - P0_target) < tol:
            return group

    raise ValueError(f"Não encontrei grupo com p0={p0_target} e P0={P0_target}")

def build_pt_dataframe(group, y_key="pt_mean"):
    """
    Retorna um DataFrame com:

    t, order_0, order_1, ..., order_{nc-1}, p_all

    onde p_all é a média entre todas as ordens no maior intervalo
    comum de tempo.
    """
    dfs = []

    for order in group["orders"]:
        order_mean = int(order["order_percolation"])
        data = order["data"]

        df_order = pd.DataFrame({
            "t": np.asarray(data["time"], dtype=float),
            f"order_{order_mean}": np.asarray(data[y_key], dtype=float),
        })

        dfs.append(df_order)

    df = dfs[0]

    # Inner join garante apenas tempos comuns a todas as ordens
    for df_order in dfs[1:]:
        df = df.merge(df_order, on="t", how="inner")

    order_cols = [c for c in df.columns if c.startswith("order_")]
    order_cols = sorted(order_cols, key=lambda x: int(x.split("_")[1]))

    df = df[["t"] + order_cols].copy()

    # Série média sobre todas as ordens
    df["p_all"] = df[order_cols].mean(axis=1)

    return df

def rolling_mean(y, window_roll, center=True, min_periods=1):
    """
    Média deslizante da série y.
    """
    y = pd.Series(np.asarray(y, dtype=float))

    return y.rolling(
        window=window_roll,
        center=center,
        min_periods=min_periods
    ).mean().to_numpy()


def block_mean_regular_time(t, y, window_block, drop_last=True):
    """
    Calcula j_w(t), isto é, a média temporal de y em blocos
    de tamanho window_block.

    Como seus dados têm dt = 1, window_block = 20 significa blocos
    com 20 pontos de tempo.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if window_block < 1:
        raise ValueError("window_block deve ser >= 1.")

    n = len(y)
    n_blocks = n // window_block

    if n_blocks == 0:
        raise ValueError("window_block é maior que o tamanho da série.")

    if drop_last:
        n_use = n_blocks * window_block
    else:
        n_use = n

    rows = []

    for k, i0 in enumerate(range(0, n_use, window_block)):
        i1 = min(i0 + window_block, n)

        if drop_last and (i1 - i0) < window_block:
            continue

        t_block = t[i0:i1]
        y_block = y[i0:i1]

        rows.append({
            "block": k,
            "t_start": t_block[0],
            "t_end": t_block[-1],
            "t_center": t_block.mean(),
            "j_w": y_block.mean(),
            "n_points": len(y_block),
        })

    return pd.DataFrame(rows)


def analyze_one_series(
    t,
    p,
    window_roll=15,
    window_block=20,
    use_rolling_for_blocks=True,
    center=True,
    drop_last=True,
):
    """
    Para uma série p(t), calcula:

    1. média deslizante: pt_roll
    2. média em blocos: j_w(t)
    3. diferença entre blocos consecutivos:
       s = |j_w(t_{k+1}) - j_w(t_k)|
    4. derivada discreta:
       s_prime = ds/dt
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)

    pt_roll = rolling_mean(
        p,
        window_roll=window_roll,
        center=center,
        min_periods=1
    )

    if use_rolling_for_blocks:
        y_for_blocks = pt_roll
    else:
        y_for_blocks = p

    df_j = block_mean_regular_time(
        t=t,
        y=y_for_blocks,
        window_block=window_block,
        drop_last=drop_last,
    )

    j = df_j["j_w"].to_numpy()
    t_j = df_j["t_center"].to_numpy()

    s = np.abs(np.diff(j))
    t_s = 0.5 * (t_j[:-1] + t_j[1:])

    if len(s) >= 2:
        s_prime = np.gradient(s, t_s)
    else:
        s_prime = np.full_like(s, np.nan, dtype=float)

    df_pt = pd.DataFrame({
        "t": t,
        "pt": p,
        "pt_roll": pt_roll,
    })

    df_s = pd.DataFrame({
        "t_s": t_s,
        "s": s,
        "s_prime": s_prime,
    })

    return {
        "df_pt": df_pt,
        "df_j": df_j,
        "df_s": df_s,
    }

def analyze_pt_group(
    group,
    window_roll=15,
    window_block=20,
    y_key="pt_mean",
    use_rolling_for_blocks=True,
    center=True,
    drop_last=True,
):
    """
    Aplica toda a análise para:

    - cada order_i
    - p_all, que é a média entre todas as ordens
    """
    df_pt_all = build_pt_dataframe(group, y_key=y_key)

    results = {}

    series_cols = [c for c in df_pt_all.columns if c.startswith("order_")]
    series_cols = sorted(series_cols, key=lambda x: int(x.split("_")[1]))

    # adiciona a série média entre todas as ordens
    series_cols.append("p_all")

    for col in series_cols:
        results[col] = analyze_one_series(
            t=df_pt_all["t"].to_numpy(),
            p=df_pt_all[col].to_numpy(),
            window_roll=window_roll,
            window_block=window_block,
            use_rolling_for_blocks=use_rolling_for_blocks,
            center=center,
            drop_last=drop_last,
        )

    return df_pt_all, results

def plot_single_analysis(results, label="p_all", figsize=(18, 5), fs=14):
    """
    Plota, para uma série específica:

    1. p(t) e média deslizante
    2. j_w(t)
    3. s(t) e s'(t)
    """
    res = results[label]

    fig, axes = plt.subplots(
        1, 3,
        figsize=figsize,
        constrained_layout=True
    )

    ax = axes[0]
    ax.plot(
        res["df_pt"]["t"],
        res["df_pt"]["pt"],
        lw=1.0,
        alpha=0.45,
        label=r"$p(t)$"
    )
    ax.plot(
        res["df_pt"]["t"],
        res["df_pt"]["pt_roll"],
        lw=2.0,
        label=r"$\langle p(t)\rangle_{\mathrm{roll}}$"
    )
    ax.set_xlabel(r"$t$", fontsize=fs)
    ax.set_ylabel(r"$p(t)$", fontsize=fs)
    ax.set_title(label, fontsize=fs)
    ax.legend(fontsize=fs-2)

    ax = axes[1]
    ax.plot(
        res["df_j"]["t_center"],
        res["df_j"]["j_w"],
        marker="o",
        lw=1.5
    )
    ax.set_xlabel(r"$t_{\mathrm{center}}$", fontsize=fs)
    ax.set_ylabel(r"$j_w(t)$", fontsize=fs)

    ax = axes[2]
    ax.plot(
        res["df_s"]["t_s"],
        res["df_s"]["s"],
        marker="o",
        lw=1.5,
        label=r"$s$"
    )
    ax.plot(
        res["df_s"]["t_s"],
        res["df_s"]["s_prime"],
        marker="s",
        lw=1.5,
        label=r"$s'$"
    )
    ax.set_xlabel(r"$t_{\mathrm{interval}}$", fontsize=fs)
    ax.set_ylabel(r"$s,\ s'$", fontsize=fs)
    ax.legend(fontsize=fs-2)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=fs-2)

    return fig, axes

def plot_s_all_orders(results, include_p_all=True, figsize=(10, 7), fs=14):
    """
    Plota s(t) para todas as ordens e, opcionalmente, para p_all.
    """
    labels = [k for k in results.keys() if k.startswith("order_")]
    labels = sorted(labels, key=lambda x: int(x.split("_")[1]))

    if include_p_all:
        labels.append("p_all")

    fig, ax = plt.subplots(
        1, 1,
        figsize=figsize,
        constrained_layout=True
    )

    for label in labels:
        df_s = results[label]["df_s"]

        lw = 2.5 if label == "p_all" else 1.2

        ax.plot(
            df_s["t_s"],
            df_s["s"],
            lw=lw,
            label=label
        )

    ax.set_xlabel(r"$t_{\mathrm{interval}}$", fontsize=fs)
    ax.set_ylabel(
        r"$s = |j_w(t_{k+1}) - j_w(t_k)|$",
        fontsize=fs
    )
    ax.tick_params(axis="both", labelsize=fs-2)
    ax.legend(fontsize=fs-3, ncols=2)

    return fig, ax


def read_dynamic_bundle(bundle_path):
    bundle = _load_json_bundle(bundle_path)

    meta = bundle.get("meta", {})
    rows = []

    def data_value(data, key, default=None):
        value = data.get(key, default)
        return default if value is None else value

    for p0_group in bundle.get("p0_groups", []):
        P0 = p0_group.get("P0_value")
        p0 = p0_group.get("p0_value")
        N_samples = p0_group.get("num_samples_total")

        colors = p0_group.get("colors", {})

        for order_block in p0_group.get("orders", []):
            data = order_block.get("data", {})

            t_eq_stats = order_block.get("t_eq_species", {})
            p_stats = order_block.get("p", {})
            f_stats = order_block.get("f", {})
            z_stats = order_block.get("z_max", {})
            z_stat_stats = order_block.get("z_stat", {})

            rows.append({
                "type_perc": meta.get("type_perc"),
                "dim": meta.get("dim"),
                "L": meta.get("L"),
                "f_T": meta.get("f_T"),
                "c": meta.get("c"),
                "nc": meta.get("nc"),
                "rho": meta.get("rho"),
                "stat_window": meta.get("stat_window"),
                "series_mode": meta.get("series_mode", "full"),

                "P0": P0,
                "p0": p0,
                "order": order_block.get("order"),

                "N_samples": N_samples,
                "N_samples_perc": order_block.get("N_samples_perc"),

                "nc_mean": colors.get("nc"),
                "nc_err": colors.get("nc_err"),
                "nc_std": colors.get("nc_std"),

                "t_eq": data.get("t_eq"),
                "t_eq_mean": data.get("t_eq_mean"),
                "t_eq_min": data.get("t_eq_min"),
                "t_eq_max": data.get("t_eq_max"),
                "t_eq_species_mean": t_eq_stats.get("mean"),
                "t_eq_species_err": t_eq_stats.get("err"),

                "p_star": p_stats.get("mean"),
                "p_star_err": p_stats.get("err"),
                "f_star": f_stats.get("mean"),
                "f_star_err": f_stats.get("err"),

                "p_tail_mean": data.get("p_tail_mean"),
                "p_tail_err": data.get("p_tail_err"),
                "p_tail_sample_values": data_value(data, "p_tail_sample_values", []),
                "p_tail_estimator": data.get("p_tail_estimator"),
                "f_tail_mean": data.get("f_tail_mean"),
                "f_tail_err": data.get("f_tail_err"),
                "f_tail_sample_values": data_value(data, "f_tail_sample_values", []),
                "f_tail_estimator": data.get("f_tail_estimator"),

                "z_max_mean": z_stats.get("mean"),
                "z_max_err": z_stats.get("err"),
                "z_max_std": z_stats.get("std"),
                "z_max_values": data_value(data, "z_max_values", z_stats.get("values", [])),
                "z_stat_mean": z_stat_stats.get("mean"),
                "z_stat_err": z_stat_stats.get("err"),
                "z_stat_std": z_stat_stats.get("std"),

                "time": data_value(data, "time", []),
                "pt_mean": data_value(data, "pt_mean", []),
                "pt_std": data_value(data, "pt_std", []),
                "pt_sem": data_value(data, "pt_sem", []),
                "pt_N_per_t": data_value(data, "pt_N_per_t", []),
                "n_seeds_pt": data.get("n_seeds_pt"),
                "pt_common_time": data_value(data, "pt_common_time", []),
                "pt_common_mean": data_value(data, "pt_common_mean", []),
                "pt_common_std": data_value(data, "pt_common_std", []),
                "pt_common_sem": data_value(data, "pt_common_sem", []),
                "pt_common_N_per_t": data_value(data, "pt_common_N_per_t", []),
                "pt_supported_time": data_value(data, "pt_supported_time", []),
                "pt_supported_mean": data_value(data, "pt_supported_mean", []),
                "pt_supported_std": data_value(data, "pt_supported_std", []),
                "pt_supported_sem": data_value(data, "pt_supported_sem", []),
                "pt_supported_N_per_t": data_value(data, "pt_supported_N_per_t", []),
                "pt_min_support_count": data.get("pt_min_support_count"),
                "pt_support_policy": data.get("pt_support_policy"),
                "pt_common_support_policy": data.get("pt_common_support_policy"),
                "pt_supported_support_policy": data.get("pt_supported_support_policy"),

                "ft_time": data_value(data, "ft_time", data_value(data, "time", [])),
                "ft_mean": data_value(data, "ft_mean", []),
                "ft_std": data_value(data, "ft_std", []),
                "ft_sem": data_value(data, "ft_sem", []),
                "ft_N_per_t": data_value(data, "ft_N_per_t", []),
                "n_seeds_ft": data.get("n_seeds_ft"),
                "ft_common_time": data_value(data, "ft_common_time", []),
                "ft_common_mean": data_value(data, "ft_common_mean", []),
                "ft_common_std": data_value(data, "ft_common_std", []),
                "ft_common_sem": data_value(data, "ft_common_sem", []),
                "ft_common_N_per_t": data_value(data, "ft_common_N_per_t", []),
                "ft_supported_time": data_value(data, "ft_supported_time", []),
                "ft_supported_mean": data_value(data, "ft_supported_mean", []),
                "ft_supported_std": data_value(data, "ft_supported_std", []),
                "ft_supported_sem": data_value(data, "ft_supported_sem", []),
                "ft_supported_N_per_t": data_value(data, "ft_supported_N_per_t", []),
                "ft_min_support_count": data.get("ft_min_support_count"),
                "ft_support_policy": data.get("ft_support_policy"),
                "ft_common_support_policy": data.get("ft_common_support_policy"),
                "ft_supported_support_policy": data.get("ft_supported_support_policy"),

                "fL_z_z": data_value(data, "fL_z_z", []),
                "fL_z_mean": data_value(data, "fL_z_mean", []),
                "fL_z_std": data_value(data, "fL_z_std", []),
                "fL_z_sem": data_value(data, "fL_z_sem", []),
                "fL_z_N_per_z": data_value(data, "fL_z_N_per_z", []),
                "n_seeds_fL_z": data.get("n_seeds_fL_z"),
                "fL_z_common_z": data_value(data, "fL_z_common_z", []),
                "fL_z_common_mean": data_value(data, "fL_z_common_mean", []),
                "fL_z_common_std": data_value(data, "fL_z_common_std", []),
                "fL_z_common_sem": data_value(data, "fL_z_common_sem", []),
                "fL_z_common_N_per_z": data_value(data, "fL_z_common_N_per_z", []),
                "fL_z_supported_z": data_value(data, "fL_z_supported_z", []),
                "fL_z_supported_mean": data_value(data, "fL_z_supported_mean", []),
                "fL_z_supported_std": data_value(data, "fL_z_supported_std", []),
                "fL_z_supported_sem": data_value(data, "fL_z_supported_sem", []),
                "fL_z_supported_N_per_z": data_value(data, "fL_z_supported_N_per_z", []),
                "fL_z_min_support_count": data.get("fL_z_min_support_count"),
                "fL_z_support_policy": data.get("fL_z_support_policy"),
                "fL_z_common_support_policy": data.get("fL_z_common_support_policy"),
                "fL_z_supported_support_policy": data.get("fL_z_supported_support_policy"),
                "min_support_fraction": data.get("min_support_fraction"),
            })

    return pd.DataFrame(rows)
