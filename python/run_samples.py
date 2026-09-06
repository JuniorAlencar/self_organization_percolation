from src.run_samples_functions import shell_data, custom_range
from src.SOP_parms import *
import numpy as np
import pandas as pd
# L = 128 => Ns = 700
# L = 192 => Ns = 600
# L = 256 => Ns = 500
# L = 384 => Ns = 300
# L = 512 => Ns = 100
# L = 768 => Ns = 50
# L = 1024 => Ns = 20

seed = -1
dim = 2
#nc=2

type_perc = 'node'
L_lst = [1024, 2048, 4096, 8192, 16384]
num_runs_lst = [500, 400, 200, 100, 50]
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
multi=True
Equilibration = 'false'
Properties = 'false'
Mode = 'growth_test'  # use 'sop' for the original fixed-height SOP run
InitialLayout = 'random'  # 'random', 'blocks', or 'alternating'
p0 = 0.8
P0 = 0.2
ft = 0.06873726
# step = 0.01 * abs(ft)

# left_points = ft - step * np.arange(7, 0, -1)
# right_points = ft + step * np.arange(1, 8)

# ft_lst = np.concatenate([
#     left_points,
#     right_points
# ])

df = pd.read_csv(f"../SOP_data/ft_min_max_2D_{type_perc}.csv", sep=',')
# ft = 0.3178947
rho = 1/nc
p0 = 0.8
for L in L_lst:
    
    print(L)
    df_sub = df[(df["L"]==L) & (df["nc"]==nc) & (df['c']==c)]

    ft = df_sub['f_T_min'].values[0]

    #for ft in ft_lst:

    num_runs = num_runs_por_L[L]
    mode_tag = "" if Mode == "sop" else f"_{Mode}"
    layout_tag = "" if InitialLayout == "random" else f"_{InitialLayout}"
    exec_name = f"L_{L}_ft_{ft:.3f}_c_{c}_nc_{nc}_dim_{dim}_p0_{p0}_P0_{P0}{mode_tag}{layout_tag}.sh"

    shell_data(L, type_perc, p0, seed, c, ft, dim,
            nc, num_runs, [1/nc], exec_name, P0, Equilibration, multi,
            properties=Properties, mode=Mode,
            initial_layout=InitialLayout)   

# for L in L_lst:
# #    ft_lst = np.linspace(0.4, 0.6, 20)

#     for ft in ft_lst:
#         rho = 1/nc
#         num_runs = num_runs_por_L[L]
#         mode_tag = "" if Mode == "sop" else f"_{Mode}"
#         layout_tag = "" if InitialLayout == "random" else f"_{InitialLayout}"
#         exec_name = f"L_{L}_ft_{ft:.3f}_c_{c}_nc_{nc}_dim_{dim}_p0_{p0}_P0_{P0}{mode_tag}{layout_tag}.sh"

#         shell_data(L, type_perc, p0, seed, c, ft, dim,
#                 nc, num_runs, [1/nc], exec_name, P0, Equilibration, multi,
#                 properties=Properties, mode=Mode,
#                 initial_layout=InitialLayout)    


#for idx, L in enumerate(L_lst):
