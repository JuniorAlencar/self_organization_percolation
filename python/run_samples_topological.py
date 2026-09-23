#!/usr/bin/env python3
"""Generate topological-properties run scripts from parameter lists."""

import os
from itertools import product
from pathlib import Path

from src.run_samples_functions import shell_data


# Edit these lists to generate one shell for each parameter combination.
seed = -1
dim = 2
type_lst = ["bond", "node"]
L_lst = [16384]
num_runs_lst = [1]  # one value per L in L_lst; a scalar is broadcast


nc = 1
c_lst = [0.01]
ft_lst = [0.1041379]
fraction_samples_lst = 30  # scalar or list; list values form a Cartesian product
fraction_gap = 1.0  # scalar or list, measured in units of L

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


def as_values(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)

    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    if nc <= 0:
        raise ValueError("nc must be positive")

    lengths = [int(value) for value in as_values(L_lst)]
    run_values = [int(value) for value in as_values(num_runs_lst)]
    if len(run_values) == 1:
        run_values *= len(lengths)
    elif len(run_values) != len(lengths):
        raise ValueError("num_runs_lst must have one value or one value per L in L_lst")
    if any(value <= 0 for value in run_values):
        raise ValueError("num_runs_lst values must be positive")

    samples_values = [int(value) for value in as_values(fraction_samples_lst)]
    gaps = [float(value) for value in as_values(fraction_gap)]
    if any(value <= 0 for value in samples_values):
        raise ValueError("fraction_samples_lst values must be positive")
    if any(value < 0 for value in gaps):
        raise ValueError("fraction_gap values must be non-negative")

    rho_values = [1.0 / nc] if rho is None else as_values(rho)
    if len(rho_values) not in (1, nc):
        raise ValueError("rho must contain one value or one value per color")

    paired_lengths_and_runs = list(zip(lengths, run_values))
    combinations = product(
        paired_lengths_and_runs,
        as_values(type_lst),
        as_values(ft_lst),
        as_values(c_lst),
        samples_values,
        gaps,
    )
    for (L, num_runs), type_perc, ft, c, fraction_samples, gap_over_L in combinations:
        mode_tag = "" if Mode == "sop" else f"_{Mode}"
        layout_tag = "" if InitialLayout == "random" else f"_{InitialLayout}"
        exec_name = (
            f"topological_L_{L}_ft_{float(ft):.7g}_c_{c}_nc_{nc}"
            f"_dim_{dim}_p0_{p0}_P0_{P0}_M_{fraction_samples}"
            f"_gap_{float(gap_over_L):.3f}L{mode_tag}"
            f"{layout_tag}_type_{type_perc}.sh"
        )
        print(f"[topological] generating shells/{exec_name}")
        shell_data(
            L,
            str(type_perc),
            p0,
            seed,
            float(c),
            float(ft),
            dim,
            nc,
            num_runs,
            rho_values,
            exec_name,
            P0,
            Equilibration,
            multi,
            properties=Properties,
            mode=Mode,
            initial_layout=InitialLayout,
            control_rule=ControlRule,
            fraction_samples=fraction_samples,
            fraction_gap_over_L=float(gap_over_L),
            force_topological_counts=ForceTopologicalCounts,
        )


if __name__ == "__main__":
    main()
