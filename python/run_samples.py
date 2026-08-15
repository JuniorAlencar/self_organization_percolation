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

type_perc = 'bond'
L_lst = [1024, 2048, 4096, 8192, 16384]
num_runs_lst = [500, 400, 200, 100, 50]
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
#df = pd.read_csv("../SOP_data/ft_min_max_2D.csv")


#multiply = 1.612903226
#c= 0.2
#df_sub = df[(df["c"]==c) & (df["nc"]==nc)]
# for index, row in df.iterrows():
#     L = int(row["L"])
#     c = row["c"]
#     nc = row["nc"]
#     ft = row["f_T_min"]
#     #fT = ft * multiply
#     print(L,c)
#     if(L==1024 and c==0.02):
#         fT = 0.40
#         num_runs = num_runs_por_L[L]
        
#         rho = 1/nc
#         mode_tag = "" if Mode == "sop" else f"_{Mode}"
#         layout_tag = "" if InitialLayout == "random" else f"_{InitialLayout}"
#         exec_name = f"L_{L}_ft_{fT:.3f}_c_{c}_nc_{nc}_dim_{dim}_p0_{p0}_P0_{P0}{mode_tag}{layout_tag}.sh"

#         shell_data(L, type_perc, p0, seed, c, fT, dim,
#                 nc, num_runs, [1/nc], exec_name, P0, Equilibration, multi,
#                 properties=Properties, mode=Mode,
#                 initial_layout=InitialLayout)
#     else:
#         pass
parms = [(1024, 0.02), (1024, 0.03), (1024, 0.04), (1024, 0.05), (1024, 0.06), (1024, 0.07), (1024, 0.08), (1024, 0.09), (1024, 0.1), (8192, 0.03), (8192, 0.04), (8192, 0.06), (8192, 0.07), (8192, 0.08), (8192, 0.09), (16384, 0.02), (16384, 0.03), (16384, 0.04), (16384, 0.05), (16384, 0.06), (16384, 0.07), (16384, 0.08), (16384, 0.09), (16384, 0.1)]

for L, c in parms:
        if(L==1024):
            ft_max = 0.3
        elif(L==2048):
            ft_max = 0.22
        elif(L==4096):
            ft_max = 0.18
        elif(L==8192):
            ft_max = 0.12
        elif(L==16384):
            ft_max = 0.1
        ft_lst = np.linspace(0.01, ft_max, 20)
        for ft in ft_lst:
            rho = 1/nc
            num_runs = num_runs_por_L[L]
            mode_tag = "" if Mode == "sop" else f"_{Mode}"
            layout_tag = "" if InitialLayout == "random" else f"_{InitialLayout}"
            exec_name = f"L_{L}_ft_{ft:.3f}_c_{c}_nc_{nc}_dim_{dim}_p0_{p0}_P0_{P0}{mode_tag}{layout_tag}.sh"

            shell_data(L, type_perc, p0, seed, c, ft, dim,
                    nc, num_runs, [1/nc], exec_name, P0, Equilibration, multi,
                    properties=Properties, mode=Mode,
                    initial_layout=InitialLayout)    


#for idx, L in enumerate(L_lst):
