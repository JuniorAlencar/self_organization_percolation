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

type_lst = ['node', 'bond']
# L_lst =        [512, 1024, 2048, 4096, 8192, 16384]
# num_runs_lst = [700, 500,  400,   200, 100,   50]
#L_lst =        [645, 813, 1448, 2896, 5793, 11585] 
L_lst =        [512, 645, 813, 1024, 1448, 2048, 2896, 4096, 5793, 8192, 11585, 16384]
num_runs_lst = [700, 600, 550, 500, 450, 400, 200, 150, 100, 75, 60, 50]

num_runs_por_L = dict(zip(L_lst, num_runs_lst))

nc = 1
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
P0 = 0.2
#P0 = 0.2
#ft = 0.2394655
#ft_lst = np.linspace(0.01, 0.3, 20)
#ft_lst_min_site = [0.211382,0.211382,0.160550,0.143685, 0.112168, .091675]
#p0 = 0.4


#P0_lst = [0.8]

#ftmin = 0.1
#L_lst_sub = L_lst[0:7]
L_lst_sub = [5793]
num_runs_lst = [100]
num_runs_por_L = dict(zip(L_lst_sub, num_runs_lst))

for idx, L in enumerate(L_lst_sub):
    for type_perc in type_lst:
        csv_path = f"../SOP_data/ft_min_max_2D_{type_perc}_{ControlRule}.csv"
        df = pd.read_csv(csv_path, sep=',')
        df_sub = df[(df['L'] == L) & (df['c'] == c) & (df['f0'] == P0)]
        num_runs = num_runs_por_L[L]
        for index, row in df_sub.iterrows():
            c = row['c']
            P0 = row['f0']
            p0 = row['p0']
            fT = row['f_T_min']
            step = 0.01 * abs(fT)

            ft_lst = fT - step * np.arange(5, 0, -1)
        
            
            for ft in ft_lst:
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
