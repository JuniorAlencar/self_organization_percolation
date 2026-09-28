import numpy as np

from src.run_samples_functions import shell_data
import pandas as pd
# =============================================================================
# PARAMETROS DO RUN
# =============================================================================
# Uso padrao:
#   1. Edite os parametros abaixo.
#   2. Rode: python3 python/run_samples.py
#   3. Os scripts sao gerados em shells/.
#
# Este gerador cria shells para as simulacoes SOP tradicionais.
# Para gerar os shells de propriedades topologicas, use
# python3 python/run_samples_topological.py.
# =============================================================================

seed = -1
dim = 2
#nc=2

#type_perc = 'node'
type_lst = ['node', 'bond']
# L_lst = [512, 1024, 2048, 4096, 8192, 16384]
# num_runs_lst = [700, 500, 400, 200, 100, 50]
L_lst = [16384]
num_runs_lst = [50]
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
c = 0.05
#P0 = 0.2
multi=True
Equilibration = 'false'
Properties = 'false'
Mode = 'growth_test'  # use 'sop' for the original fixed-height SOP run
InitialLayout = 'clustered'  # 'clustered', 'random', 'blocks', or 'alternating'
ControlRule = 'linear'     # 'relative' or 'linear'


#ft = 0.14
#step = 0.01 * abs(ft)
#ft_lst = ft - step * np.arange(7, 0, -1)
# right_points = ft + step * np.arange(1, 8)

# ft_lst = np.concatenate([
#     left_points,
#     right_points
# ])

#df = pd.read_csv(f"../SOP_data/ft_min_max_2D_{type_perc}.csv", sep=',')
# ft = 0.3178947
rho = 1/nc
#p0 = 0.8
p0 = 0.8
P0_lst = [0.6, 0.8, 1.0]
#P0 = 0.2
#ft = 0.2394655
#ft_lst = np.linspace(0.01, 0.3, 20)
#p0 = 0.4


#P0_lst = [0.8]

ftmin = 0.1
for type_perc in type_lst:
    #csv_path = f"../SOP_data/ft_min_max_2D_{type_perc}_{ControlRule}.csv"
    #df = pd.read_csv(csv_path, sep=',')
    # for index, row in ftmin.iterrows():
    #     c = row['c']
    #     #nc = row['nc']
    #     P0 = row['f0']
    #     p0 = row['p0']
    #     fT = row['f_T_min']
    #     step = 0.01 * abs(fT)

    #     left_points = fT - step * np.arange(5, 0, -1)
    #     right_points = fT + step * np.arange(1, 6)

    #     ft_lst = np.concatenate([
    #         left_points,
    #         right_points
    #     ])
    num_runs = num_runs_por_L[L_lst[0]]
    for P0 in P0_lst:
        mode_tag = "" if Mode == "sop" else f"_{Mode}"
        layout_tag = "" if InitialLayout == "clustered" else f"_{InitialLayout}"
        exec_name = f"L_{L_lst[0]}_ft_{ftmin:.3f}_c_{c}_nc_{nc}_dim_{dim}_p0_{p0}_P0_{P0:.3f}{mode_tag}{layout_tag}_type_{type_perc}.sh"
        
        
        shell_data(L_lst[0], type_perc, p0, seed, c, ftmin, dim,
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
