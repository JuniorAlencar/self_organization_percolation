#!/usr/bin/env python3
"""Generate topological-properties shell scripts from editable parameter lists."""

import os
from pathlib import Path

from src.run_samples_functions import shell_data


# =============================================================================
# PARAMETROS DO RUN
# Edite as listas e os fors abaixo; cada chamada a shell_data cria um .sh.
# =============================================================================
seed = -1
dim = 2

type_lst = ["bond", "node"]
L_lst = [512, 645, 813, 1024, 1448, 2048, 2896, 4096, 5793, 8192, 11585, 16384]
#num_runs_lst = [29, 27, 25, 23, 21, 19, 16, 13, 11, 9, 7, 5]  # one value per L in L_lst
num_runs_lst = [15, 14, 13, 12, 11, 10, 8, 7, 6, 5, 4, 3]  # one value per L in L_lst
num_runs_por_L = dict(zip(L_lst, num_runs_lst))

nc = 1
c = 0.05

# =============================================================================
# RELAÇÃO DE ESCALONAMENTO: f_T(L) = lambda * f_{T,min}(L)
# Valores de f_{T,min}(L) obtidos dos ajustes refinados (c = 0.05, 2D)
# salvos em SOP_data/ft_min_max_2D_{type_perc}_smooth_refined.csv
# =============================================================================
ft_min_por_L = {
    "bond": {
        512: 0.2400650,
        645: 0.2244347,
        813: 0.2097881,
        1024: 0.1961397,
        1448: 0.1772954,
        2048: 0.1602516,
        2896: 0.1448552,
        4096: 0.1309299,
        5793: 0.1183448,
        8192: 0.1069734,
        11585: 0.09669337,
        16384: 0.08740020,
    },
    "node": {
        512: 0.1893358,
        645: 0.1775234,
        813: 0.1664222,
        1024: 0.1560474,
        1448: 0.1416711,
        2048: 0.1286116,
        2896: 0.1167630,
        4096: 0.1059995,
        5793: 0.09622943,
        8192: 0.08736305,
        11585: 0.07931259,
        16384: 0.07200316,
    },
}

# Fator multiplicativo lambda (escalonamento f_T(L) = lambda * f_{T,min}(L))
# Pode ser um valor escalar (ex: 1.0 para f_T = f_{T,min}(L)), uma lista (ex: [1.0]),
# ou None para usar as listas empíricas padrão por tipo (bond: [1.53...], node: [1.92...])
lambda_lst = [1.0]

fraction_samples_lst = 30
fraction_gap = 1.0  # spacing in units of L

p0 = 0.8
P0 = 0.2
rho = [1.0 / nc]
multi = True
Equilibration = "false"
Properties = "false"
Mode = "fractions"
InitialLayout = "random"
ControlRule = "linear"  # "relative" or "linear"
ForceTopologicalCounts = True


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def get_ft_min(type_p: str, l_val: int) -> float:
    """Retorna f_{T,min} para o tipo e tamanho L especificados."""
    if type_p in ft_min_por_L and l_val in ft_min_por_L[type_p]:
        return ft_min_por_L[type_p][l_val]
    csv_p = Path(__file__).resolve().parent.parent / f"SOP_data/ft_min_max_2D_{type_p}_smooth_refined.csv"
    if csv_p.exists():
        import pandas as pd
        df = pd.read_csv(csv_p)
        match = df[df["L"] == l_val]
        if not match.empty:
            return float(match.iloc[0]["f_T"])
    raise KeyError(f"f_{{T,min}} para L={l_val} e tipo '{type_p}' não encontrado.")


if len(num_runs_lst) != len(L_lst):
    raise ValueError("num_runs_lst must have one value per L in L_lst")
if dim not in (2, 3):
    raise ValueError("dim must be 2 or 3")
if nc <= 0:
    raise ValueError("nc must be positive")

if lambda_lst is not None:
    lambda_values = [float(value) for value in as_list(lambda_lst)]
    if any(value <= 0 for value in lambda_values):
        raise ValueError("lambda_lst values must be positive")
else:
    lambda_values = None

fraction_samples_values = [int(value) for value in as_list(fraction_samples_lst)]
fraction_gap_values = [float(value) for value in as_list(fraction_gap)]
if any(value <= 0 for value in fraction_samples_values):
    raise ValueError("fraction_samples_lst values must be positive")
if any(value < 0 for value in fraction_gap_values):
    raise ValueError("fraction_gap values must be non-negative")

os.chdir(Path(__file__).resolve().parent)

count_generated = 0
for L in L_lst:
    num_runs = num_runs_por_L[L]

    for type_perc in type_lst:
        ft_min_base = get_ft_min(type_perc, L)
        if lambda_values is not None:
            current_lambdas = lambda_values
        elif type_perc == "bond":
            current_lambdas = [1.53, 1.42, 1.32, 1.21, 1.11]
        elif type_perc == "node":
            current_lambdas = [1.92, 1.74, 1.55, 1.37, 1.18]
        else:
            current_lambdas = [1.0]

        for lam in current_lambdas:
            ft = round(float(lam * ft_min_base), 7)

            for fraction_samples in fraction_samples_values:
                for gap_over_L in fraction_gap_values:
                    mode_tag = "" if Mode == "sop" else f"_{Mode}"
                    layout_tag = (
                        "" if InitialLayout == "random" else f"_{InitialLayout}"
                    )
                    exec_name = (
                        f"topological_L_{L}_ft_{ft:.7g}_c_{c}_nc_{nc}"
                        f"_dim_{dim}_p0_{p0}_P0_{P0}_M_{fraction_samples}"
                        f"_gap_{gap_over_L:.3f}L{mode_tag}"
                        f"{layout_tag}_type_{type_perc}.sh"
                    )
                    
                    shell_data(
                        L,
                        type_perc,
                        p0,
                        seed,
                        c,
                        ft,
                        dim,
                        nc,
                        num_runs,
                        rho,
                        exec_name,
                        P0,
                        Equilibration,
                        multi,
                        properties=Properties,
                        mode=Mode,
                        initial_layout=InitialLayout,
                        control_rule=ControlRule,
                        fraction_samples=fraction_samples,
                        fraction_gap_over_L=gap_over_L,
                        force_topological_counts=ForceTopologicalCounts,
                    )
                    count_generated += 1
                    print(
                        f"[{count_generated:02d}] {exec_name} -> "
                        f"L={L:5d}, {type_perc}, lambda={lam:g}, ft={ft:.7g}, runs={num_runs:2d}"
                    )

print(f"\nConcluído: {count_generated} shell scripts gerados em shells/")
