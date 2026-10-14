"""
Module for occupation fraction and spanning cluster statistics
from SOP_data/published_fractions bundles.

Designed for publication-quality figures in international scientific journals:
  - Formal English terminology
  - Concise mathematical variable labels (L, P(k), S_i/L^2, E_i/E_total, etc.)
  - Subplot panel labels (a), (b), (c) in bold at loc='left'
  - Independent axis scaling across cluster regimes to eliminate squashing
  - Non-overlapping legends with generous headroom
  - Clean integer notation for L on multiplicity plots (no mixed 2^p and integers)
  - Enlarged x-axis labels and legend font sizes
"""

from __future__ import annotations

import json
import lzma
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "load_published_fractions_df",
    "get_cluster_count_table",
    "plot_cluster_count_distribution_all_L",
    "plot_largest_cluster_bond_combined",
    "plot_largest_cluster_node",
    "plot_largest_cluster_node_combined",
    "plot_cluster_regimes_separated",
    "plot_relative_cluster_mass_fractions",
]


def load_published_fractions_df(
    root: str | Path = "../SOP_data/published_fractions",
) -> pd.DataFrame:
    """
    Load all bundles from SOP_data/published_fractions into a consolidated DataFrame.

    Each row represents a single stabilized slab sample.
    Extracts:
      - System parameters: L, f_T, c, percolation type ('bond' or 'node'), N_total, E_total
      - Number of spanning clusters present (k = 1, 2, 3, ...)
      - Absolute sizes and fractions of largest spanning cluster (c1)
      - Absolute sizes and fractions of other spanning clusters (c2, c3, ...)
      - Slab-level fractions (p_stab_node, p_stab_bond)
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        alt_path = Path("SOP_data/published_fractions").resolve()
        if alt_path.exists():
            root_path = alt_path
        else:
            raise FileNotFoundError(f"Directory not found: {root_path}")

    files = sorted(root_path.glob("**/*.json.xz")) + [
        p for p in sorted(root_path.glob("**/*.json")) if not p.name.endswith(".xz")
    ]
    seen_keys = set()
    unique_files: list[Path] = []
    for f in files:
        key = f.parent / f.name.replace(".xz", "")
        if key not in seen_keys:
            seen_keys.add(key)
            unique_files.append(f)

    rows: list[dict[str, Any]] = []

    for f in unique_files:
        if f.suffix == ".xz":
            with lzma.open(f, "rt", encoding="utf-8") as h:
                bundle = json.load(h)
        else:
            with open(f, "r", encoding="utf-8") as h:
                bundle = json.load(h)

        meta = bundle.get("meta", {})
        data = bundle.get("data", {})
        L = int(meta.get("L", 0))
        perc_type = str(meta.get("type_percolation", ""))
        N_tot = int(meta.get("N_total", L * L))
        E_tot = int(meta.get("E_total", 2 * L * L))
        rel_dir = meta.get("relative_parameter_dir", "")

        # Parse f_T
        f_T = np.nan
        for part in Path(rel_dir).parts:
            if part.startswith("fT_") and part != "fT_constant":
                try:
                    f_T = float(part.replace("fT_", ""))
                except ValueError:
                    pass

        # Parse c
        c_val = 0.05
        for part in Path(rel_dir).parts:
            if part.startswith("c_"):
                try:
                    c_val = float(part.replace("c_", ""))
                except ValueError:
                    pass

        num_clusters_list = data.get("num_spanning_clusters", [])
        lsc_list = data.get("largest_spanning_cluster", [])
        osc_list = data.get("other_spanning_clusters", [])
        p_bond_list = data.get("p_stab_bond", [])
        p_node_list = data.get("p_stab_node", [])

        n_samples = len(num_clusters_list)
        for idx in range(n_samples):
            k = num_clusters_list[idx]
            row: dict[str, Any] = {
                "type": perc_type,
                "L": L,
                "f_T": f_T,
                "c": c_val,
                "sample_index": idx,
                "k": k,
                "N_total": N_tot,
                "E_total": E_tot,
                "p_stab_bond": p_bond_list[idx] if idx < len(p_bond_list) else np.nan,
                "p_stab_node": p_node_list[idx] if idx < len(p_node_list) else np.nan,
            }

            # Cluster 1 (largest)
            c1 = lsc_list[idx] if idx < len(lsc_list) else None
            if c1 is not None and "component" in c1:
                comp1 = c1["component"]
                tot_occ_s = comp1.get("total_occupied_sites", 0)
                tot_act_b = comp1.get("total_active_bonds", 0)
                ns1 = comp1.get("num_sites", 0)
                nb1 = comp1.get("num_bonds", 0)

                row["c1_sites"] = ns1
                row["c1_bonds"] = nb1
                row["c1_frac_site"] = ns1 / N_tot if N_tot > 0 else np.nan
                row["c1_frac_bond"] = nb1 / E_tot if E_tot > 0 else np.nan
                row["c1_frac_rel_site"] = ns1 / tot_occ_s if tot_occ_s > 0 else np.nan
                row["c1_frac_rel_bond"] = nb1 / tot_act_b if tot_act_b > 0 else np.nan
                row["total_occupied_sites"] = tot_occ_s
                row["total_active_bonds"] = tot_act_b
            else:
                row["c1_sites"] = 0
                row["c1_bonds"] = 0
                row["c1_frac_site"] = 0.0
                row["c1_frac_bond"] = 0.0
                row["c1_frac_rel_site"] = 0.0
                row["c1_frac_rel_bond"] = 0.0
                row["total_occupied_sites"] = 0
                row["total_active_bonds"] = 0

            for rank in (2, 3):
                row[f"c{rank}_sites"] = np.nan
                row[f"c{rank}_bonds"] = np.nan
                row[f"c{rank}_frac_site"] = np.nan
                row[f"c{rank}_frac_bond"] = np.nan
                row[f"c{rank}_frac_rel_site"] = np.nan
                row[f"c{rank}_frac_rel_bond"] = np.nan

            others = osc_list[idx] if idx < len(osc_list) else []
            if others:
                for j, oc in enumerate(others):
                    rank = j + 2
                    if rank > 3:
                        continue
                    comp = oc.get("component", {})
                    tot_occ_s = comp.get("total_occupied_sites", 0)
                    tot_act_b = comp.get("total_active_bonds", 0)
                    ns = comp.get("num_sites", 0)
                    nb = comp.get("num_bonds", 0)

                    row[f"c{rank}_sites"] = ns
                    row[f"c{rank}_bonds"] = nb
                    row[f"c{rank}_frac_site"] = ns / N_tot if N_tot > 0 else np.nan
                    row[f"c{rank}_frac_bond"] = nb / E_tot if E_tot > 0 else np.nan
                    row[f"c{rank}_frac_rel_site"] = ns / tot_occ_s if tot_occ_s > 0 else np.nan
                    row[f"c{rank}_frac_rel_bond"] = nb / tot_act_b if tot_act_b > 0 else np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(by=["type", "L", "sample_index"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_cluster_count_table(df: pd.DataFrame, perc_type: str = "bond") -> pd.DataFrame:
    """
    Return summary table of spanning cluster counts (k = 1, 2, 3) across lattice sizes L.
    """
    sub = df[df["type"] == perc_type]
    counts = pd.crosstab(sub["L"], sub["k"]).reindex(columns=[1, 2, 3], fill_value=0)
    totals = counts.sum(axis=1)
    pcts = counts.div(totals, axis=0) * 100.0

    return pd.DataFrame({
        "L": counts.index,
        "Samples": totals.values,
        "k=1 (count)": counts[1].values,
        "k=1 (%)": pcts[1].round(2).values,
        "k=2 (count)": counts[2].values,
        "k=2 (%)": pcts[2].round(2).values,
        "k=3 (count)": counts[3].values,
        "k=3 (%)": pcts[3].round(2).values,
    }).reset_index(drop=True)


def plot_cluster_count_distribution_all_L(
    df: pd.DataFrame,
    perc_type: str = "bond",
    figsize: tuple[float, float] = (14.5, 5.2),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot cluster multiplicity statistics across all L:
      - Left: P(k) discrete bar distribution for each L (ticks strictly in integer values).
      - Right: P(k >= 2) line plot with log2(L) scale (ticks strictly in integer values).
    """
    sub = df[df["type"] == perc_type]
    L_vals = sorted(sub["L"].unique())
    counts = pd.crosstab(sub["L"], sub["k"]).reindex(columns=[1, 2, 3], fill_value=0)
    totals = counts.sum(axis=1)
    props = counts.div(totals, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    x = np.arange(len(L_vals))
    width = 0.26

    c1 = "#1f77b4"
    c2 = "#ff7f0e"
    c3 = "#2ca02c"

    # Integer labels for all L values (no mixed powers of 2)
    x_tick_labels = [str(L) for L in L_vals]

    # Panel 1: Bar chart of P(k)
    ax1.bar(x - width, props[1] * 100, width=width, color=c1, label=r"$k = 1$", alpha=0.9, edgecolor="black", linewidth=0.8)
    ax1.bar(x, props[2] * 100, width=width, color=c2, label=r"$k = 2$", alpha=0.9, edgecolor="black", linewidth=0.8)
    if props[3].max() > 0:
        ax1.bar(x + width, props[3] * 100, width=width, color=c3, label=r"$k = 3$", alpha=0.9, edgecolor="black", linewidth=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_tick_labels, rotation=45, fontsize=15)
    ax1.set_xlabel(r"$L$", fontsize=22)
    ax1.set_ylabel(r"$N_k(L)$", fontsize=20)
    ax1.set_ylim(0, 130)
    type_title = r"$\mathrm{Bond\ Percolation}$" if perc_type == "bond" else r"$\mathrm{Site\ Percolation}$"
    ax1.set_title("(a)", loc="left", fontsize=22, fontweight="bold")
    ax1.set_title(type_title, loc="center", fontsize=22)
    ax1.legend(frameon=False, loc="upper right", fontsize=16)
    #ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel 2: P(k = 2) and P(k = 3) line plot
    ax2.plot(L_vals, props[2] * 100, marker="o", markersize=7, color="#d62728", linewidth=2.0, label=r"$k = 2$")
    if props[3].max() > 0:
        ax2.plot(L_vals, props[3] * 100, marker="s", markersize=7, color=c3, linewidth=1.6, linestyle="--", label=r"$k = 3$")

    ax2.set_xscale("log", base=2)
    major_ticks = [512, 1024, 2048, 4096, 8192, 16384]
    major_labels = [str(t) for t in major_ticks]
    ax2.set_xticks(major_ticks)
    ax2.set_xticklabels(major_labels, fontsize=15)
    ax2.set_xlabel(r"$L$", fontsize=22)
    ax2.set_ylabel(r"$P(k \ge 2|L)$", fontsize=20)

    if perc_type == "bond":
        ax2.set_ylim(-1.0, 26.0)
    else:
        ax2.set_ylim(-0.1, 2.2)

    ax2.set_title("(b)", loc="left", fontsize=22, fontweight="bold")
    #ax2.set_title(r"$P(k)$", loc="center", fontsize=22)
    ax2.legend(frameon=False, loc="upper right", fontsize=16)
    #ax2.grid(False, linestyle=":", alpha=0.5)

    return fig, (ax1, ax2)


def plot_largest_cluster_bond_combined(
    df: pd.DataFrame,
    L_subset: list[int] | None = None,
    figsize: tuple[float, float] = (14.5, 5.5),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Combined 1x2 subplots of largest spanning cluster fraction across different L for bond percolation:
      - Left: Site fraction S_1 / L^2 with panel label (a) and title 'Bond percolation'
      - Right: Bond fraction E_1 / E_total with panel label (b) and title 'E_1 / E_total'
    """
    sub = df[df["type"] == "bond"]
    if L_subset is None:
        L_subset = [512, 1024, 2048, 4096, 8192, 16384]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(L_subset)))

    max_count_1 = 0.0
    max_count_2 = 0.0

    bins_s = np.linspace(0.1, 0.7, 31)
    bins_e = np.linspace(0.05, 0.35, 31)

    for idx, L in enumerate(L_subset):
        sub_L = sub[sub["L"] == L]

        # Site fraction
        s_data = sub_L["c1_frac_site"].dropna()
        if not s_data.empty:
            mean_s = s_data.mean()
            sem_s = s_data.std() / np.sqrt(len(s_data))
            lbl_s = f"$L = {L}$ ($\\mu = {mean_s:.3f} \\pm {sem_s:.3f}$)"
            counts1, _, _ = ax1.hist(
                s_data,
                bins=bins_s,
                alpha=0.35,
                color=colors[idx],
                edgecolor=colors[idx],
                linewidth=1.2,
                histtype="stepfilled",
                label=lbl_s,
            )
            max_count_1 = max(max_count_1, float(np.max(counts1)))

        # Bond fraction
        e_data = sub_L["c1_frac_bond"].dropna()
        if not e_data.empty:
            mean_e = e_data.mean()
            sem_e = e_data.std() / np.sqrt(len(e_data))
            lbl_e = f"$L = {L}$ ($\\mu = {mean_e:.3f} \\pm {sem_e:.3f}$)"
            counts2, _, _ = ax2.hist(
                e_data,
                bins=bins_e,
                alpha=0.35,
                color=colors[idx],
                edgecolor=colors[idx],
                linewidth=1.2,
                histtype="stepfilled",
                label=lbl_e,
            )
            max_count_2 = max(max_count_2, float(np.max(counts2)))

    # Labels and titles
    ax1.set_title("(a)", loc="left", fontsize=22, fontweight="bold")
    ax1.set_title(r"$\mathrm{Bond\ percolation}$", loc="center", fontsize=22)
    ax1.set_xlabel(r"$S_1 / L^2$", fontsize=22)
    ax1.set_ylabel(r"$N(S_1 / L^2)$", fontsize=20)
    ax1.set_ylim(0, max_count_1 * 1.85)
    ax1.legend(frameon=False, loc="upper right", fontsize=15)
    #ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.set_title("(b)", loc="left", fontsize=22, fontweight="bold")
    #ax2.set_title(r"$E_1 / E_{\mathrm{total}}$", loc="center", fontsize=22)
    ax2.set_xlabel(r"$E_1 / L(2L-1)$", fontsize=22)
    ax2.set_ylabel(r"$N(E_1 / L(2L-1))$", fontsize=20)
    ax2.set_ylim(0, max_count_2 * 1.85)
    ax2.legend(frameon=False, loc="upper right", fontsize=15)
    #ax2.grid(True, linestyle=":", alpha=0.5)

    return fig, (ax1, ax2)


def plot_largest_cluster_node(
    df: pd.DataFrame,
    L_subset: list[int] | None = None,
    figsize: tuple[float, float] = (8.0, 5.5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot largest spanning cluster fraction across different L for node (site) percolation.
    """
    sub = df[df["type"] == "node"]
    if L_subset is None:
        L_subset = [512, 1024, 2048, 4096, 8192, 16384]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(L_subset)))

    max_count = 0.0
    bins_s = np.linspace(0.15, 0.55, 31)

    for idx, L in enumerate(L_subset):
        s_data = sub[sub["L"] == L]["c1_frac_site"].dropna()
        if not s_data.empty:
            mean_s = s_data.mean()
            sem_s = s_data.std() / np.sqrt(len(s_data))
            lbl = f"$L = {L}$ ($\\mu = {mean_s:.3f} \\pm {sem_s:.3f}$)"
            counts, _, _ = ax.hist(
                s_data,
                bins=bins_s,
                alpha=0.35,
                color=colors[idx],
                edgecolor=colors[idx],
                linewidth=1.2,
                histtype="stepfilled",
                label=lbl,
            )
            max_count = max(max_count, float(np.max(counts)))

    #ax.set_title("(a)", loc="left", fontsize=22, fontweight="bold")
    ax.set_title(r"$\mathrm{Site\ percolation}$", loc="center", fontsize=22)
    ax.set_xlabel(r"$S_1 / L^2$", fontsize=22)
    ax.set_ylabel(r"$N(S_1 / L^2)$", fontsize=20)
    ax.set_ylim(0, max_count * 1.85)
    ax.legend(frameon=False, loc="upper right", fontsize=15)
    #ax.grid(True, linestyle=":", alpha=0.5)

    return fig, ax


def plot_largest_cluster_node_combined(
    df: pd.DataFrame,
    L_subset: list[int] | None = None,
    figsize: tuple[float, float] = (8.0, 5.5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Combined/standalone plot of largest spanning cluster fraction across different L
    for node (site) percolation.
    Convenience alias matching plot_largest_cluster_bond_combined naming convention.
    """
    return plot_largest_cluster_node(df, L_subset=L_subset, figsize=figsize)


def plot_cluster_regimes_separated(
    df: pd.DataFrame,
    L: int | list[int] | None = None,
    L_list: list[int] | None = None,
    perc_type: str = "bond",
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, Any]:
    """
    Separate spanning cluster distributions into distinct subplots according to multiplicity k:
      - Column 1: k = 1 (single spanning cluster P_1)
      - Column 2: k = 2 (two spanning clusters P_1 and P_2)
      - Column 3: k = 3 (three spanning clusters P_1, P_2, P_3)
    By default, plots a single 2x3 figure with L = 512 above (row 0) and L = 1024 below (row 1).
    """
    if L_list is not None:
        target_L = L_list if isinstance(L_list, (list, tuple)) else [L_list]
    elif L is not None:
        target_L = L if isinstance(L, (list, tuple)) else [L]
    else:
        target_L = [512, 1024]

    n_rows = len(target_L)
    if n_rows == 1:
        fig_size = figsize if figsize is not None else (15.5, 4.8)
        fig, axes = plt.subplots(1, 3, figsize=fig_size, constrained_layout=True)
        axes_grid = np.array([axes])
        panel_letters = [["(a)", "(b)", "(c)"]]
    else:
        fig_size = figsize if figsize is not None else (15.5, 9.2)
        fig, axes_grid = plt.subplots(n_rows, 3, figsize=fig_size, constrained_layout=True)
        flat_letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
        panel_letters = [flat_letters[i * 3 : (i + 1) * 3] for i in range(n_rows)]

    for row_idx, current_L in enumerate(target_L):
        sub = df[(df["type"] == perc_type) & (df["L"] == current_L)]
        if sub.empty:
            continue
        ax1, ax2, ax3 = axes_grid[row_idx]
        p_row = panel_letters[row_idx]

        # 1. Regime k = 1
        sub_k1 = sub[sub["k"] == 1]
        s1_k1 = sub_k1["c1_frac_site"].dropna()
        if not s1_k1.empty:
            mean_s1_k1 = s1_k1.mean()
            sem_s1_k1 = s1_k1.std() / np.sqrt(len(s1_k1))
            bins_k1 = np.linspace(s1_k1.min(), s1_k1.max(), 26)
            c1, _, _ = ax1.hist(
                s1_k1,
                bins=bins_k1,
                color="#1f77b4",
                alpha=0.55,
                edgecolor="#1f77b4",
                linewidth=1.2,
            )
            ax1.axvline(
                mean_s1_k1,
                color="#1f77b4",
                linestyle="--",
                linewidth=2.2,
                label=rf"$P_1\ (\mu = {mean_s1_k1:.3f} \pm {sem_s1_k1:.3f})$",
            )
            ax1.set_ylim(0, float(np.max(c1)) * 1.50)
        ax1.set_title(p_row[0], loc="left", fontsize=22, fontweight="bold")
        ax1.set_title(rf"$L = {current_L},\ k = 1$", loc="center", fontsize=22)
        ax1.set_xlabel(r"$S_1 / L^2$", fontsize=22)
        ax1.set_ylabel(r"$N(S_1 / L^2)$", fontsize=20)
        ax1.legend(frameon=False, loc="upper right", fontsize=17)
        #ax1.grid(True, linestyle=":", alpha=0.5)

        # 2. Regime k = 2
        sub_k2 = sub[sub["k"] == 2]
        if not sub_k2.empty:
            s1_k2 = sub_k2["c1_frac_site"].dropna()
            s2_k2 = sub_k2["c2_frac_site"].dropna()
            mean_s1_k2 = s1_k2.mean()
            sem_s1_k2 = s1_k2.std() / np.sqrt(len(s1_k2))
            mean_s2_k2 = s2_k2.mean()
            sem_s2_k2 = s2_k2.std() / np.sqrt(len(s2_k2))

            min_k2 = min(s1_k2.min(), s2_k2.min())
            max_k2 = max(s1_k2.max(), s2_k2.max())
            bins_k2 = np.linspace(min_k2, max_k2, 25)

            c1_k2, _, _ = ax2.hist(
                s1_k2,
                bins=bins_k2,
                color="#1f77b4",
                alpha=0.55,
                edgecolor="#1f77b4",
                linewidth=1.2,
            )
            c2_k2, _, _ = ax2.hist(
                s2_k2,
                bins=bins_k2,
                color="#ff7f0e",
                alpha=0.55,
                edgecolor="#ff7f0e",
                linewidth=1.2,
            )
            ax2.axvline(
                mean_s1_k2,
                color="#1f77b4",
                linestyle="--",
                linewidth=2.2,
                label=rf"$P_1\ (\mu = {mean_s1_k2:.3f} \pm {sem_s1_k2:.3f})$",
            )
            ax2.axvline(
                mean_s2_k2,
                color="#ff7f0e",
                linestyle="--",
                linewidth=2.2,
                label=rf"$P_2\ (\mu = {mean_s2_k2:.3f} \pm {sem_s2_k2:.3f})$",
            )
            max_h2 = max(float(np.max(c1_k2)), float(np.max(c2_k2)))
            ax2.set_ylim(0, max_h2 * 1.50)
        ax2.set_title(p_row[1], loc="left", fontsize=22, fontweight="bold")
        ax2.set_title(rf"$L = {current_L},\ k = 2$", loc="center", fontsize=22)
        ax2.set_xlabel(r"$S_i / L^2$", fontsize=22)
        ax2.set_ylabel(r"$N(S_i / L^2)$", fontsize=20)
        ax2.legend(frameon=False, loc="upper right", fontsize=17)
#        ax2.grid(True, linestyle=":", alpha=0.5)

        # 3. Regime k = 3
        sub_k3 = sub[sub["k"] == 3]
        if not sub_k3.empty:
            c1_3 = sub_k3["c1_frac_site"].values
            c2_3 = sub_k3["c2_frac_site"].values
            c3_3 = sub_k3["c3_frac_site"].values
            mean_c1_3 = c1_3.mean()
            mean_c2_3 = c2_3.mean()
            mean_c3_3 = c3_3.mean()

            # N <= 2 -> omit SEM to avoid overestimation / zero variance
            ax3.axvline(
                mean_c1_3,
                color="#1f77b4",
                linestyle="--",
                linewidth=2.2,
                label=rf"$P_1\ (\mu = {mean_c1_3:.3f})$",
            )
            ax3.axvline(
                mean_c2_3,
                color="#ff7f0e",
                linestyle="--",
                linewidth=2.2,
                label=rf"$P_2\ (\mu = {mean_c2_3:.3f})$",
            )
            ax3.axvline(
                mean_c3_3,
                color="#2ca02c",
                linestyle="--",
                linewidth=2.2,
                label=rf"$P_3\ (\mu = {mean_c3_3:.3f})$",
            )

            # Each observation represents 1 sample at y = 1
            y_pos = np.ones(len(c1_3))
            ax3.scatter(c1_3, y_pos, color="#1f77b4", s=90, zorder=5, edgecolor="black", linewidth=0.8)
            ax3.scatter(c2_3, y_pos, color="#ff7f0e", s=90, zorder=5, edgecolor="black", linewidth=0.8)
            ax3.scatter(c3_3, y_pos, color="#2ca02c", s=90, zorder=5, edgecolor="black", linewidth=0.8)

            ax3.set_ylim(0, 2.5)
            ax3.set_yticks([0, 1, 2])
            all_vals = np.concatenate([c1_3, c2_3, c3_3])
            margin = (all_vals.max() - all_vals.min()) * 0.4
            ax3.set_xlim(max(0.0, all_vals.min() - margin), all_vals.max() + margin)
        else:
            ax3.text(0.5, 0.5, r"$N = 0$", ha="center", va="center", transform=ax3.transAxes, fontsize=18)
            ax3.set_ylim(0, 2.5)
            ax3.set_yticks([0, 1, 2])

        ax3.set_title(p_row[2], loc="left", fontsize=22, fontweight="bold")
        ax3.set_title(rf"$L = {current_L},\ k = 3$", loc="center", fontsize=22)
        ax3.set_xlabel(r"$S_i / L^2$", fontsize=22)
        ax3.set_ylabel(r"$N(S_i / L^2)$", fontsize=20)
        ax3.legend(frameon=False, loc="upper right", fontsize=17)
        #ax3.grid(True, linestyle=":", alpha=0.5)

    if n_rows == 1:
        return fig, tuple(axes_grid[0])
    return fig, axes_grid


def plot_relative_cluster_mass_fractions(
    df: pd.DataFrame,
    L_list: list[int] | None = None,
    figsize: tuple[float, float] = (14.5, 5.2),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot relative mass fraction s_i = S_i / S_active for k=1 and k=2 regimes
    with panel labels (a), (b) and without zoom-out or squashing.
    """
    if L_list is None:
        L_list = [512, 1024]

    fig, axes = plt.subplots(1, len(L_list), figsize=figsize, constrained_layout=True)
    if len(L_list) == 1:
        axes = [axes]

    panel_letters = ["(a)", "(b)", "(c)"]

    for idx, (ax, L) in enumerate(zip(axes, L_list)):
        sub_L = df[(df["type"] == "bond") & (df["L"] == L)]

        # Common bins for direct comparison of sample counts N
        bins = np.linspace(0.18, 1.0, 31)

        # k = 1
        s1_k1 = sub_L[sub_L["k"] == 1]["c1_frac_rel_site"].dropna()
        c1, _, _ = ax.hist(
            s1_k1,
            bins=bins,
            alpha=0.45,
            color="#1f77b4",
            edgecolor="#1f77b4",
            label=r"$k = 1:\ S_1$",
        )

        # k = 2
        sub_k2 = sub_L[sub_L["k"] == 2]
        max_h = float(np.max(c1))
        if not sub_k2.empty:
            c1_k2 = sub_k2["c1_frac_rel_site"].dropna()
            c2_k2 = sub_k2["c2_frac_rel_site"].dropna()
            h2_1, _, _ = ax.hist(
                c1_k2,
                bins=bins,
                alpha=0.45,
                color="#d62728",
                edgecolor="#d62728",
                label=r"$k = 2\ (\mathrm{pair}):\ S_1\ (1^{\mathrm{st}}\ \mathrm{cluster})$",
            )
            h2_2, _, _ = ax.hist(
                c2_k2,
                bins=bins,
                alpha=0.45,
                color="#e377c2",
                edgecolor="#e377c2",
                label=r"$k = 2\ (\mathrm{pair}):\ S_2\ (2^{\mathrm{nd}}\ \mathrm{cluster})$",
            )
            max_h = max(max_h, float(np.max(h2_1)), float(np.max(h2_2)))

        ax.set_title(panel_letters[idx], loc="left", fontsize=22, fontweight="bold")
        ax.set_title(rf"$L = {L}$", loc="center", fontsize=18)
        ax.set_xlabel(r"$S_i / S_{\mathrm{active}}$", fontsize=22)
        ax.set_ylabel(r"$N(S_i / S_{\mathrm{active}})$", fontsize=20)
        ax.set_ylim(0, max_h * 1.45)
        ax.legend(frameon=False, loc="upper left", fontsize=15)
        

    return fig, tuple(axes)
