import argparse
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from src.run_samples_functions import shell_data


# =============================================================================
# PARAMETROS DO RUN
# =============================================================================
# Uso padrao:
#   1. Edite os parametros abaixo.
#   2. Rode: python3 python/run_samples.py
#   3. Os scripts sao gerados em shells/.
#
# No modo fractions:
#   - num_runs controla quantas execucoes/seeds entram no shell.
#   - fraction_samples_lst controla quantas amostras M sao coletadas dentro de
#     cada seed, par-a-par com L_lst.
#   - fraction_gap_over_L_lst controla o espacamento entre janelas, em unidades
#     de L. Todos os gaps sao usados para cada par L/ft/c.
# =============================================================================

use_csv = False
csv_path = "../SOP_data/run_values.csv"
csv_limit = None

seed = -1
dim = 2
#nc=2

#type_perc = 'node'
type_lst = ['node', 'bond']
L_lst = [512, 1024, 2048, 4096, 8192, 16384]
num_runs_lst = [700, 500, 400, 200, 100, 50]
#L_lst = [16384]
#um_runs_lst = []
#L_lst = [8192, 16384]
#num_runs_lst = [100, 50]
#L_lst = [16384]
#num_runs_lst = [50]

# L_lst = [1024]
# num_runs_lst = [500]
num_runs_por_L = dict(zip(L_lst, num_runs_lst))
#L_lst = [1024]
#num_runs = [400]
# nc = 4
#L_lst = [128, 256, 512, 1024]
#num_runs = [300, 150, 50, 5]

#L_lst = [256]
#num_runs = [150]
nc = 1
#c_lst = [0.01, 0.05, 0.1, 0.15, 0.2]
#c_lst = [0.02, 0.03, 0.04, 0.06, 0.07, 0.8, 0.9]
#c_lst = np.round(np.arange(0.01, 0.21, 0.01), 2)
c = 0.01
#P0 = 0.2
multi=True
Equilibration = 'false'
Properties = 'false'
Mode = 'growth_test'  # use 'sop' for the original fixed-height SOP run
InitialLayout = 'clustered'  # 'clustered', 'random', 'blocks', or 'alternating'
ControlRule = 'linear'     # 'relative' or 'linear'

# step = 0.01 * abs(ft)

# left_points = ft - step * np.arange(7, 0, -1)
# right_points = ft + step * np.arange(1, 8)

# ft_lst = np.concatenate([
#     left_points,
#     right_points
# ])

#df = pd.read_csv(f"../SOP_data/ft_min_max_2D_{type_perc}.csv", sep=',')
# ft = 0.3178947
rho = 1/nc
p0 = 0.8
P0 = 0.2
#ft = 0.2394655
ft_lst = np.linspace(0.01, 0.3, 20)
#p0 = 0.4


#P0_lst = [0.8]
for L in L_lst:
    
    #print(L)
    #df_sub = df[(df["L"]==L) & (df["nc"]==nc) & (df['c']==c)]

    #ft = df_sub['f_T_min'].values[0]

    #for ft in ft_lst:
    for type_perc in type_lst:

        # df = pd.read_csv(f"../SOP_data/ft_min_max_2D_{type_perc}.csv", sep=',')
        # ftmin = df[(df["L"]==L) & (df["nc"]==nc) & (df['c']==c)]['f_T_min'].values[0]
        # ftmax = df[(df["L"]==L) & (df["nc"]==nc) & (df['c']==c)]['f_T_max'].values[0]
        # ft_lst = [ftmin, ftmax]
        for ft in ft_lst:
            #P0 = min(1.0, 1.2 * ft)
        #    for ft in ft_lst:
                #P0 = min(1.0, 1.2 * ft)
            num_runs = num_runs_por_L[L]
            mode_tag = "" if Mode == "sop" else f"_{Mode}"
            layout_tag = "" if InitialLayout == "clustered" else f"_{InitialLayout}"
            exec_name = f"L_{L}_ft_{ft:.3f}_c_{c}_nc_{nc}_dim_{dim}_p0_{p0}_P0_{P0:.3f}{mode_tag}{layout_tag}_type_{type_perc}.sh"

            shell_data(L, type_perc, p0, seed, c, ft, dim,
                    nc, num_runs, [1/nc], exec_name, P0, Equilibration, multi,
                    properties=Properties, mode=Mode,
                    initial_layout=InitialLayout, control_rule=ControlRule)   

# for L in L_lst:
# #    ft_lst = np.linspace(0.4, 0.6, 20)
#
#     for ft in ft_lst:
#         P0 = min(1.0, 1.2 * ft)
#         rho = 1/nc
#         num_runs = num_runs_por_L[L]
#         mode_tag = "" if Mode == "sop" else f"_{Mode}"
#         layout_tag = "" if InitialLayout == "clustered" else f"_{InitialLayout}"
#         exec_name = f"L_{L}_ft_{ft:.3f}_c_{c}_nc_{nc}_dim_{dim}_p0_{p0}_P0_{P0:.3f}{mode_tag}{layout_tag}.sh"
#
#         shell_data(L, type_perc, p0, seed, c, ft, dim,
#                 nc, num_runs, [1/nc], exec_name, P0, Equilibration, multi,
#                 properties=Properties, mode=Mode,
#                 initial_layout=InitialLayout, control_rule=ControlRule)    


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate SOP shell scripts. Defaults target the dynamic-memory "
            "raw_fractions output."
        )
    )
    parser.add_argument("--L", type=int, nargs="+", default=[8192])
    parser.add_argument("--ft", type=float, nargs="+", default=[0.1041379])
    parser.add_argument("--c", type=float, nargs="+", default=[0.01])
    parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--type-perc", default="bond", choices=["bond", "node"])
    parser.add_argument("--nc", type=int, default=1)
    parser.add_argument("--rho", type=float, nargs="+", default=None)
    parser.add_argument("--p0", type=float, default=0.8)
    parser.add_argument("--P0", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of executable runs/seeds written to the generated shell.",
    )
    parser.add_argument(
        "--fraction-samples",
        type=int,
        nargs="+",
        default=[30],
        help=(
            "M samples collected inside each seed when mode=fractions. "
            "Pass one value or one value per L/ft/c row."
        ),
    )
    parser.add_argument(
        "--fraction-gap-over-L",
        type=float,
        nargs="+",
        default=[1.0],
        help=(
            "Gap between consecutive L-sized windows, in units of L. "
            "All values are generated for each L/ft/c row."
        ),
    )
    parser.add_argument(
        "--mode",
        default="fractions",
        choices=["fractions", "raw_fractions", "growth_test", "sop"],
    )
    parser.add_argument("--equilibration", default="false")
    parser.add_argument("--properties", default="false")
    parser.add_argument(
        "--initial-layout",
        default="random",
        choices=["random", "blocks", "quadrants", "quadrantes", "alternating", "alternado"],
    )
    parser.add_argument("--surface-observables", type=parse_bool_text, default=False)
    parser.add_argument("--multi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--exec-prefix",
        default="",
        help="Optional prefix for generated shell filenames.",
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with columns L,c,f_T and optionally N_samples, "
            "fraction_samples and fraction_gap_over_L. Relative paths are "
            "resolved from python/."
        ),
    )
    parser.add_argument("--csv-limit", type=int, default=None)
    return parser


def config_args():
    return SimpleNamespace(
        L=L_lst,
        ft=ft_lst,
        c=c_lst,
        dim=dim,
        type_perc=type_perc,
        nc=nc,
        rho=rho,
        p0=p0,
        P0=P0,
        seed=seed,
        num_runs=num_runs,
        fraction_samples=fraction_samples_lst,
        fraction_gap_over_L=fraction_gap_over_L_lst,
        mode=Mode,
        equilibration=Equilibration,
        properties=Properties,
        initial_layout=InitialLayout,
        surface_observables=SurfaceObservables,
        multi=multi,
        exec_prefix=exec_prefix,
        from_csv=Path(csv_path) if use_csv else None,
        csv_limit=csv_limit,
    )


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def expand_per_row(name, value, n_rows, cast):
    values = [cast(v) for v in as_list(value)]
    if len(values) == 1:
        return values * n_rows
    if len(values) != n_rows:
        raise ValueError(
            f"{name} must have 1 value or {n_rows} values; got {len(values)}"
        )
    return values


def output_name(args, L, ft, c, fraction_samples, gap_over_L):
    mode_tag = "" if args.mode == "sop" else f"_{args.mode}"
    layout_tag = "" if args.initial_layout == "random" else f"_{args.initial_layout}"
    frac_tag = (
        f"_M_{fraction_samples}_gap_{gap_over_L:.3f}L"
        if args.mode in {"fractions", "raw_fractions"}
        else ""
    )
    prefix = f"{args.exec_prefix}_" if args.exec_prefix else ""
    return (
        f"{prefix}L_{L}_ft_{ft:.7g}_c_{c}_nc_{args.nc}_dim_{args.dim}"
        f"_type_{args.type_perc}"
        f"_p0_{args.p0}_P0_{args.P0}{mode_tag}{frac_tag}{layout_tag}.sh"
    )


def iter_parameter_rows(args):
    if args.from_csv is not None:
        csv_path = args.from_csv
        if not csv_path.is_absolute():
            csv_path = Path(__file__).resolve().parent / csv_path
        df = pd.read_csv(csv_path)
        if args.csv_limit is not None:
            df = df.iloc[: args.csv_limit]
        for _, row in df.iterrows():
            row_fraction_samples = row.get(
                "fraction_samples",
                row.get("FractionSamples", as_list(args.fraction_samples)[0]),
            )
            yield (
                int(row["L"]),
                float(row["f_T"]),
                float(row["c"]),
                int(row.get("N_samples", args.num_runs[0])),
                int(row_fraction_samples),
                float(row.get("fraction_gap_over_L", args.fraction_gap_over_L[0])),
            )
        return

    raw_L_values = as_list(args.L)
    raw_ft_values = as_list(args.ft)
    raw_c_values = as_list(args.c)
    n_rows = max(len(raw_L_values), len(raw_ft_values), len(raw_c_values))

    L_values = expand_per_row("L", raw_L_values, n_rows, int)
    ft_values = expand_per_row("ft", raw_ft_values, n_rows, float)
    c_values = expand_per_row("c", raw_c_values, n_rows, float)

    num_runs_values = expand_per_row("num_runs", args.num_runs, n_rows, int)
    fraction_samples_values = expand_per_row(
        "fraction_samples", args.fraction_samples, n_rows, int
    )
    fraction_gap_values = [float(v) for v in as_list(args.fraction_gap_over_L)]

    for L, ft, c, num_runs, fraction_samples in zip(
        L_values,
        ft_values,
        c_values,
        num_runs_values,
        fraction_samples_values,
    ):
        for gap_over_L in fraction_gap_values:
            yield L, ft, c, num_runs, fraction_samples, gap_over_L


def main():
    os.chdir(Path(__file__).resolve().parent)
    args = build_parser().parse_args() if len(sys.argv) > 1 else config_args()
    args.fraction_samples = [int(v) for v in as_list(args.fraction_samples)]
    for fraction_samples in args.fraction_samples:
        if fraction_samples <= 0:
            raise ValueError("--fraction-samples values must be positive")
    args.num_runs = [int(v) for v in as_list(args.num_runs)]
    for row_num_runs in args.num_runs:
        if row_num_runs <= 0:
            raise ValueError("--num-runs values must be positive")
    if args.rho is None:
        rho = [1.0 / args.nc]
    else:
        rho = args.rho

    if isinstance(args.fraction_gap_over_L, (int, float)):
        args.fraction_gap_over_L = [float(args.fraction_gap_over_L)]
    args.fraction_gap_over_L = [float(v) for v in args.fraction_gap_over_L]
    for gap_over_L in args.fraction_gap_over_L:
        if gap_over_L < 0:
            raise ValueError("fraction_gap_over_L values must be non-negative")

    for L, ft, c, num_runs, fraction_samples, gap_over_L in iter_parameter_rows(args):
        exec_name = output_name(args, L, ft, c, fraction_samples, gap_over_L)
        shell_data(
            L,
            args.type_perc,
            args.p0,
            args.seed,
            c,
            ft,
            args.dim,
            args.nc,
            num_runs,
            rho,
            exec_name,
            args.P0,
            args.equilibration,
            args.multi,
            properties=args.properties,
            mode=args.mode,
            initial_layout=args.initial_layout,
            surface_observables=args.surface_observables,
            fraction_samples=fraction_samples,
            fraction_gap_over_L=gap_over_L,
        )


if __name__ == "__main__":
    main()
