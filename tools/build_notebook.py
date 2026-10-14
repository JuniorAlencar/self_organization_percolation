#!/usr/bin/env python3
"""Build and execute jupyter/Occupation_Fraction_Analysis.ipynb with nbclient."""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

def build_notebook():
    nb = nbf.v4.new_notebook()

    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.12.0'
        }
    }

    cells = []

    # Cell 0 - Title and Context
    cells.append(nbf.v4.new_markdown_cell("""# Occupation Fraction and Spanning Cluster Statistics at $f_{T,\\mathrm{min}}$

Analysis of the **occupation fraction** and the **spanning cluster statistics** for stabilized Self-Organized Percolation (SOP) networks from `SOP_data/published_fractions`.

All configurations in this dataset correspond to the characteristic threshold **$f_{T,\\mathrm{min}}(L)$** for each lattice size $L \\in \\{512, 645, 813, 1024, 1448, 2048, 2896, 4096, 5793, 8192, 11585, 16384\\}$ ($c = 0.05$, $p_0 = 0.8$, $P_0 = 0.2$, $\\rho = 1.0$, and $\\mathrm{gap} = 1.000L$ in $2\\mathrm{D}$).

---

### Objectives:
1. **Cluster Multiplicity Distribution ($k = 1, 2, 3$)**:
   - Histograms and probabilities $P(k)$ of spanning cluster counts across all lattice sizes $L$.
   - Probabilities $P(k = 2)$ and $P(k = 3)$ as a function of $L$.
2. **Occupation Fraction of Spanning Clusters**:
   - Scaling of the largest spanning cluster fraction ($P_1$) across lattice sizes $L$.
   - Separation of cluster fractions across multiplicity regimes ($k = 1$, $k = 2$, and $k = 3$).
"""))

    # Cell 1 - Setup
    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path

# Add repository paths
repo_root = next(
    p for p in (Path.cwd(), *Path.cwd().parents)
    if (p / "SOP_data" / "published_fractions").exists()
)
jupyter_dir = repo_root / "jupyter"
if str(jupyter_dir) not in sys.path:
    sys.path.insert(0, str(jupyter_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if "src.FractionsClusterAnalysis" in sys.modules:
    import importlib
    importlib.reload(sys.modules["src.FractionsClusterAnalysis"])

from src.FractionsClusterAnalysis import (
    load_published_fractions_df,
    get_cluster_count_table,
    plot_cluster_count_distribution_all_L,
    plot_largest_cluster_bond_combined,
    plot_largest_cluster_node,
    plot_largest_cluster_node_combined,
    plot_cluster_regimes_separated,
    plot_relative_cluster_mass_fractions,
)

from src.TimeSeriesAnalysis import (
    create_folder
)

# Plotting style
mplstyle_path = jupyter_dir / "properties.mplstyle"
if mplstyle_path.exists():
    try:
        plt.style.use(str(mplstyle_path))
        print(f"Loaded style: {mplstyle_path.name}")
    except Exception as e:
        print(f"Style fallback: {e}")
        plt.style.use("default")
else:
    plt.style.use("default")

dim = 2
nc = 1
results_folder = create_folder(f"../results/{nc}Color/{dim}D/Occupation_Fractions_Analysis")
"""))

    # Cell 2 - Load Data
    cells.append(nbf.v4.new_code_cell("""fractions_dir = repo_root / "SOP_data" / "published_fractions"
df_clusters = load_published_fractions_df(fractions_dir)

n_samples = len(df_clusters)
n_L = df_clusters['L'].nunique()
print(f"Loaded {n_samples:,} samples across {n_L} lattice sizes (L).")
print(f"Sample count by percolation type:\\n{df_clusters['type'].value_counts().to_string()}")
df_clusters[['type', 'L', 'f_T', 'sample_index', 'k', 'c1_frac_site', 'c2_frac_site', 'c3_frac_site']].head(10)
"""))

    # Cell 3 - Section 1 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 1. Number of Spanning Clusters vs Lattice Size $L$

Count and occurrence probability of the number of simultaneous spanning clusters ($k = 1, 2, 3$) for each lattice size $L$.
"""))

    # Cell 4 - Tables
    cells.append(nbf.v4.new_code_cell("""print("=" * 60)
print("BOND PERCOLATION - CLUSTER MULTIPLICITY")
print("=" * 60)
tbl_bond = get_cluster_count_table(df_clusters, perc_type="bond")
display(tbl_bond)

print("\\n" + "=" * 60)
print("NODE PERCOLATION - CLUSTER MULTIPLICITY")
print("=" * 60)
tbl_node = get_cluster_count_table(df_clusters, perc_type="node")
display(tbl_node)
"""))

    # Cell 5 - Figure 1: Bond Multiplicity
    cells.append(nbf.v4.new_code_cell("""fig_bond, axes_bond = plot_cluster_count_distribution_all_L(
    df_clusters, perc_type="bond", figsize=(14, 5.0)
)
plt.show()
"""))

    # Cell 6 - Figure 2: Node Multiplicity
    cells.append(nbf.v4.new_code_cell("""fig_node, axes_node = plot_cluster_count_distribution_all_L(
    df_clusters, perc_type="node", figsize=(14, 5.0)
)
plt.show()
"""))

    # Cell 7 - Section 2 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 2. Largest Spanning Cluster Fraction across Different $L$

Finite-size scaling of the largest spanning cluster:
- Site fraction: $S_1 / L^2$
- Bond fraction: $E_1 / E_{\\mathrm{total}}$
"""))

    # Cell 8 - Figure 3: Combined 1x2 Bond Percolation
    cells.append(nbf.v4.new_code_cell("""fig_bond_comb, axes_bc = plot_largest_cluster_bond_combined(
    df_clusters,
    L_subset=[512, 1024, 2048, 4096, 8192, 16384],
    figsize=(14, 5.2)
)
plt.show()
"""))

    # Cell 9 - Figure 4: Site Percolation
    cells.append(nbf.v4.new_code_cell("""fig_node_comb, ax_nc = plot_largest_cluster_node_combined(
    df_clusters,
    L_subset=[512, 1024, 2048, 4096, 8192, 16384],
    figsize=(7.5, 5.2)
)
plt.show()
"""))

    # Cell 10 - Section 3 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 3. Cluster Fractions by Multiplicity Regime ($k = 1, 2, 3$)

To avoid squashing and overlay confusion, distributions are separated into distinct subplots for:
- **$k = 1$**: Single spanning cluster ($P_1$)
- **$k = 2$**: Two coexisting spanning clusters ($P_1$ and $P_2$)
- **$k = 3$**: Three coexisting spanning clusters ($P_1, P_2, P_3$)
"""))

    # Cell 11 - Figure 5: Regimes k=1, 2, 3 (2x3: L=512 above, L=1024 below)
    cells.append(nbf.v4.new_code_cell("""fig_sep, axes_sep = plot_cluster_regimes_separated(
    df_clusters,
    L_list=[512, 1024],
    perc_type="bond",
    figsize=(15.5, 9.2)
)
plt.show()
"""))

    # Cell 13 - Summary Table by k
    cells.append(nbf.v4.new_code_cell("""stats_by_k = (
    df_clusters[df_clusters["type"] == "bond"]
    .groupby(["L", "k"])[
        [
            "c1_frac_site",
            "c2_frac_site",
            "c3_frac_site",
            "c1_frac_rel_site",
            "c2_frac_rel_site",
            "c3_frac_rel_site",
        ]
    ]
    .mean()
    .round(4)
)

print("Mean cluster fractions by lattice size L and multiplicity k (Bond Percolation):")
display(stats_by_k.loc[[512, 1024, 2048]])
"""))

    # Cell 14 - Section 4 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 4. Relative Mass Partition: $S_i / S_{\\mathrm{active}}$

Fraction of active mass in the slab occupied by each cluster:
$$s_i = \\frac{S_i}{S_{\\mathrm{active}}}$$
"""))

    # Cell 15 - Figure 7: Relative mass fractions
    cells.append(nbf.v4.new_code_cell("""fig_rel, axes_rel = plot_relative_cluster_mass_fractions(
    df_clusters,
    L_list=[512, 1024],
    figsize=(14, 5.0)
)
plt.show()
"""))

    # Cell 16 - Conclusions
    cells.append(nbf.v4.new_markdown_cell("""## 5. Summary

1. **Multiplicity Distribution**:
   - In bond percolation, $k = 1$ accounts for $84\\%-97\\%$ of samples, $k = 2$ accounts for $3\\%-16\\%$, and $k = 3$ appears in small lattices ($L = 512, 1024$).
   - In node percolation, configurations are overwhelmingly single-cluster ($k = 1$, $>99\\%$), with rare $k = 2$ and no $k = 3$.

2. **Cluster Mass Partitioning**:
   - When multiple spanning clusters coexist, the active mass is split competitively between them rather than being dominated by a single giant component.
"""))

    nb.cells = cells

    nb_path = Path("jupyter/Occupation_Fraction_Analysis.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("Executing notebook with nbclient...")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": "jupyter"}}
    )
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("Notebook successfully executed and saved!")

if __name__ == "__main__":
    build_notebook()
