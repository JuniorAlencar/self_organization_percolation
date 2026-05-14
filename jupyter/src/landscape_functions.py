
import json
import math
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _parse_float_part(path: Union[str, Path], prefix: str) -> Optional[float]:
    """Read values from path components like fT_1.000000e-01, c_1.500000e-01, rho_2.5000e-01."""
    for part in Path(path).parts:
        m = re.match(rf"^{re.escape(prefix)}({_FLOAT})$", part)
        if m:
            return float(m.group(1))
    return None


def _parse_int_part(path: Union[str, Path], prefix: str) -> Optional[int]:
    for part in Path(path).parts:
        m = re.match(rf"^{re.escape(prefix)}([-+]?\d+)$", part)
        if m:
            return int(m.group(1))
    return None


def _isclose(value, target, rtol=1e-9, atol=1e-12) -> bool:
    if value is None or target is None:
        return False
    return math.isclose(float(value), float(target), rel_tol=rtol, abs_tol=atol)


def _hist_density(values, p_bins):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    counts, _ = np.histogram(values, bins=p_bins)
    widths = np.diff(p_bins)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(p_bins) - 1, dtype=float)
    return counts / (total * widths)


def _centers_to_edges(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 1:
        dx = 0.5 * max(abs(x[0]), 1e-3)
        return np.array([x[0] - dx, x[0] + dx])
    edges = np.empty(len(x) + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges


def _read_one_json_stationary(path: Union[str, Path], t_stat_override=None) -> pd.DataFrame:
    """Return one row per stationary value p^i(t) from one SOP JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        js = json.load(f)

    meta = js.get("meta", {})
    t_stat = int(t_stat_override) if t_stat_override is not None else int(meta.get("t_stat", meta.get("t_eq", 0)))

    f_T = meta.get("f_T", meta.get("fT", None))
    if f_T is None:
        f_T = _parse_float_part(path, "fT_")
    if f_T is None:
        raise ValueError(f"Could not infer f_T from JSON meta or path: {path}")
    f_T = float(f_T)

    c_path = _parse_float_part(path, "c_")
    rho_path = _parse_float_part(path, "rho_")
    L_path = _parse_int_part(path, "L_")
    dim_path = _parse_int_part(path, "dim_")
    nc_path = _parse_int_part(path, "num_colors_")

    seed = None
    m_seed = re.search(r"seed_([-+]?\d+)", path.name)
    if m_seed:
        seed = int(m_seed.group(1))

    rows = []
    for key, val in js.get("results", {}).items():
        block = val.get("data", {})
        if "pt" not in block:
            continue

        color = int(block.get("color", -1))
        p_arr = np.asarray(block["pt"], dtype=float)
        t_arr = np.asarray(block.get("time", np.arange(len(p_arr))), dtype=int)

        if len(p_arr) != len(t_arr):
            raise ValueError(f"len(pt) != len(time) in {path}, block {key}")

        mask = t_arr >= t_stat
        for t, p in zip(t_arr[mask], p_arr[mask]):
            if np.isfinite(p):
                rows.append({
                    "file": str(path),
                    "seed": seed,
                    "f_T": f_T,
                    "c": c_path,
                    "rho": rho_path,
                    "L": L_path,
                    "dim": dim_path,
                    "num_colors": nc_path,
                    "color": color,
                    "time": int(t),
                    "p": float(p),
                    "t_stat": t_stat,
                })

    if not rows:
        raise ValueError(f"No stationary pt values found in: {path}")

    return pd.DataFrame(rows)


def _find_sop_json_files(
    c: float,
    rho: float,
    L: int,
    dim: int,
    num_colors: int,
    base_raw: Union[str, Path] = "../SOP_data/raw",
    type_perc: str = "bond_percolation",
):
    """
    Search all JSON files in:
    base_raw/type_perc/num_colors_<num_colors>/dim_<dim>/L_<L>/fT_constant/fT_*/c_*/rho_*/data/*.json
    """
    root = (
        Path(base_raw)
        / type_perc
        / f"num_colors_{num_colors}"
        / f"dim_{dim}"
        / f"L_{L}"
        / "fT_constant"
    )

    files_all = sorted(root.glob("fT_*/c_*/rho_*/data/*.json"))

    files = []
    for path in files_all:
        c_path = _parse_float_part(path, "c_")
        rho_path = _parse_float_part(path, "rho_")
        if _isclose(c_path, c) and _isclose(rho_path, rho):
            files.append(path)

    return root, files


def _build_P(df: pd.DataFrame, p_bins: np.ndarray, weight_mode: str):
    fT_values = np.array(sorted(df["f_T"].unique()), dtype=float)
    p_centers = 0.5 * (p_bins[:-1] + p_bins[1:])
    P = np.zeros((len(fT_values), len(p_centers)), dtype=float)

    for idx, fT in enumerate(fT_values):
        df_ft = df[np.isclose(df["f_T"].to_numpy(dtype=float), fT)]

        if weight_mode == "raw":
            P[idx] = _hist_density(df_ft["p"].to_numpy(), p_bins)

        elif weight_mode == "equal_file":
            hist_list = [_hist_density(g["p"].to_numpy(), p_bins) for _, g in df_ft.groupby("file")]
            P[idx] = np.mean(hist_list, axis=0)

        elif weight_mode == "equal_file_color":
            hist_list = [
                _hist_density(g["p"].to_numpy(), p_bins)
                for _, g in df_ft.groupby(["file", "color"])
            ]
            P[idx] = np.mean(hist_list, axis=0)

        else:
            raise ValueError("weight_mode must be 'raw', 'equal_file', or 'equal_file_color'.")

        # Normalize conditional distribution: sum_b P(p_b | fT) Delta p = 1.
        integral = np.sum(P[idx] * np.diff(p_bins))
        if integral > 0:
            P[idx] /= integral

    return fT_values, p_centers, P


def _make_plots(fT_values, p_bins, p_centers, P, V):
    figs = {}
    fT_edges = _centers_to_edges(fT_values)
    p_plot_min = 0.2
    p_plot_max = 0.3
    p_bin_mask = (p_bins >= p_plot_min) & (p_bins <= p_plot_max)
    p_center_mask = (p_centers >= p_plot_min) & (p_centers <= p_plot_max)
    p_plot_bins = p_bins[p_bin_mask]
    p_plot_centers = p_centers[p_center_mask]
    P_plot = P[:, p_center_mask]
    V_plot = V[:, p_center_mask]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    mesh = ax.pcolormesh(fT_edges, p_plot_bins, P_plot.T, shading="auto")
    fig.colorbar(mesh, ax=ax, label=r"$P(p\mid f_T)$")
    ax.set_xlabel(r"$f_T$")
    ax.set_ylabel(r"$p$")
    ax.set_ylim(p_plot_min, p_plot_max)
    ax.set_title(r"Stationary distribution $P(p\mid f_T)$")
    fig.tight_layout()
    figs["P_heatmap"] = fig
    plt.show()

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    mesh = ax.pcolormesh(fT_edges, p_plot_bins, V_plot.T, shading="auto")
    fig.colorbar(mesh, ax=ax, label=r"$V_{\mathrm{eff}}$")
    ax.set_xlabel(r"$f_T$")
    ax.set_ylabel(r"$p$")
    ax.set_ylim(p_plot_min, p_plot_max)
    ax.set_title(r"Effective landscape $V_{\mathrm{eff}}(f_T,p)$")
    fig.tight_layout()
    figs["V_heatmap"] = fig
    plt.show()

    if len(fT_values) >= 2:
        FT, PP = np.meshgrid(fT_values, p_plot_centers, indexing="ij")
        fig = plt.figure(figsize=(8.0, 5.8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(FT, PP, V_plot, linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.10, label=r"$V_{\mathrm{eff}}$")
        ax.set_xlabel(r"$f_T$")
        ax.set_ylabel(r"$p$")
        ax.set_zlabel(r"$V_{\mathrm{eff}}$")
        ax.set_ylim(p_plot_min, p_plot_max)
        ax.set_title(r"Effective stationary landscape")
        ax.view_init(elev=28, azim=-55)
        fig.tight_layout()
        figs["V_surface"] = fig
        plt.show()

    return figs


def build_sop_landscape(
    c: float,
    rho: float,
    L: int,
    dim: int,
    num_colors: int,
    base_raw: Union[str, Path] = "../SOP_data/raw",
    type_perc: str = "bond_percolation",
    value_mode: str = "species",
    weight_mode: str = "equal_file_color",
    p_min: float = 0.0,
    p_max: float = 1.0,
    n_bins: int = 120,
    eps_factor: float = 1e-12,
    clip_percentile: Optional[float] = 99.0,
    t_stat_override: Optional[int] = None,
    make_plots: bool = True,
    save_outputs: bool = False,
    output_prefix: Optional[Union[str, Path]] = None,
):
    """
    Build P(p | f_T) and V_eff(f_T, p) from SOP JSON files.

    Parameters
    ----------
    c, rho, L, dim, num_colors:
        Simulation parameters used to select folders.

    base_raw:
        Usually "../SOP_data/raw" when the notebook is one folder inside the Python repository.

    type_perc:
        Example: "bond_percolation".

    value_mode:
        "species" -> uses all p^i(t) separately, estimating P(p^i | f_T).
        "mean"    -> uses pbar(t) = mean_i p^i(t), estimating P(pbar | f_T).

    weight_mode:
        "equal_file_color" -> recommended for value_mode="species".
        "equal_file"       -> recommended for value_mode="mean".
        "raw"              -> pools all stationary values directly.

    Returns
    -------
    dict with:
        samples, summary, fT_values, p_bins, p_centers, P, V, figures, files, root
    """

    root, files = _find_sop_json_files(
        c=c,
        rho=rho,
        L=L,
        dim=dim,
        num_colors=num_colors,
        base_raw=base_raw,
        type_perc=type_perc,
    )

    print(f"Root used: {root}")
    print(f"JSON files found after c/rho filter: {len(files)}")

    if len(files) == 0:
        raise FileNotFoundError(
            "No JSON files found. Check base_raw, type_perc, L, dim, num_colors, c and rho."
        )

    frames = []
    failed = []
    for path in files:
        try:
            frames.append(_read_one_json_stationary(path, t_stat_override=t_stat_override))
        except Exception as exc:
            failed.append((str(path), str(exc)))

    if failed:
        print(f"Warning: {len(failed)} files failed and were ignored. First failures:")
        for path, msg in failed[:5]:
            print(" -", path, "::", msg)

    if not frames:
        raise RuntimeError("No valid JSON file was loaded.")

    samples = pd.concat(frames, ignore_index=True)

    # Keep exact selected group.
    samples = samples[
        samples["c"].apply(lambda x: _isclose(x, c))
        & samples["rho"].apply(lambda x: _isclose(x, rho))
        & (samples["L"] == L)
        & (samples["dim"] == dim)
        & (samples["num_colors"] == num_colors)
    ].reset_index(drop=True)

    if samples.empty:
        raise RuntimeError("No samples remained after filtering metadata.")

    if value_mode == "mean":
        # Mean over species at fixed file/time.
        group_cols = ["file", "seed", "f_T", "c", "rho", "L", "dim", "num_colors", "time", "t_stat"]
        samples_for_hist = (
            samples.groupby(group_cols, dropna=False, as_index=False)["p"]
            .mean()
            .sort_values(["f_T", "file", "time"])
            .reset_index(drop=True)
        )
        if weight_mode == "equal_file_color":
            weight_mode = "equal_file"
    elif value_mode == "species":
        samples_for_hist = samples.copy()
    else:
        raise ValueError("value_mode must be 'species' or 'mean'.")

    summary = (
        samples_for_hist.groupby("f_T", as_index=False)
        .agg(
            n_files=("file", "nunique"),
            n_rows=("p", "size"),
            p_mean=("p", "mean"),
            p_std=("p", "std"),
            p_min=("p", "min"),
            p_max=("p", "max"),
        )
        .sort_values("f_T")
        .reset_index(drop=True)
    )

    display(summary)

    p_bins = np.linspace(p_min, p_max, n_bins + 1)
    fT_values, p_centers, P = _build_P(samples_for_hist, p_bins, weight_mode=weight_mode)

    Pmax = np.nanmax(P)
    eps = eps_factor * Pmax if Pmax > 0 else eps_factor
    V = -np.log(P + eps)
    V -= np.nanmin(V)

    if clip_percentile is not None:
        vmax = np.nanpercentile(V, clip_percentile)
        V = np.minimum(V, vmax)

    figures = None
    if make_plots:
        figures = _make_plots(fT_values, p_bins, p_centers, P, V)

    if save_outputs:
        if output_prefix is None:
            output_prefix = (
                Path("../SOP_data/analysis")
                / f"landscape_{type_perc}_L{L}_dim{dim}_ns{num_colors}_c{c:.3e}_rho{rho:.4e}_{value_mode}"
            )
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)

        samples_for_hist.to_csv(output_prefix.with_suffix(".stationary_samples.csv"), index=False)
        summary.to_csv(output_prefix.with_suffix(".summary.csv"), index=False)
        np.savez(
            output_prefix.with_suffix(".npz"),
            fT_values=fT_values,
            p_bins=p_bins,
            p_centers=p_centers,
            P=P,
            V=V,
        )
        print(f"Saved outputs with prefix: {output_prefix}")

    return {
        "samples": samples_for_hist,
        "summary": summary,
        "fT_values": fT_values,
        "p_bins": p_bins,
        "p_centers": p_centers,
        "P": P,
        "V": V,
        "figures": figures,
        "files": files,
        "root": root,
    }
