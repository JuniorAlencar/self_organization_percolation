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
L_lst = [16384]
num_runs_lst = [1]  # one value per L in L_lst
num_runs_por_L = dict(zip(L_lst, num_runs_lst))

nc = 1
c_lst = [0.01]
ft_lst = [0.1041379]
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


if len(num_runs_lst) != len(L_lst):
    raise ValueError("num_runs_lst must have one value per L in L_lst")
if dim not in (2, 3):
    raise ValueError("dim must be 2 or 3")
if nc <= 0:
    raise ValueError("nc must be positive")

fraction_samples_values = [int(value) for value in as_list(fraction_samples_lst)]
fraction_gap_values = [float(value) for value in as_list(fraction_gap)]
if any(value <= 0 for value in fraction_samples_values):
    raise ValueError("fraction_samples_lst values must be positive")
if any(value < 0 for value in fraction_gap_values):
    raise ValueError("fraction_gap values must be non-negative")

os.chdir(Path(__file__).resolve().parent)

for L in L_lst:
    num_runs = num_runs_por_L[L]

    for type_perc in type_lst:
        for ft in ft_lst:
            for c in c_lst:
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
