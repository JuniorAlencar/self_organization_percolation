from src.run_samples_functions import shell_data, custom_range
from src.SOP_parms import *
import numpy as np
# L = 128 => Ns = 700
# L = 192 => Ns = 600
# L = 256 => Ns = 500
# L = 384 => Ns = 300
# L = 512 => Ns = 100
# L = 768 => Ns = 50
# L = 1024 => Ns = 20

# num_runs = [500, 75 , 100,  50,  30, 10]
# L_lst =    [128, 192, 256, 384, 512, 768, 1024]

#L_lst = [256, 384, 512, 768, 1024]
# nc = 8

# if nc == 2:
#     f = 0.03
# elif nc == 4:
#     f = 0.02
# elif nc == 8:
#     f = 0.01

#Nt = [[int(fraction*L**2) for fraction in f] for L in L_lst]
#f0 = 0.02
#rho = 1/nc
#p0 = 1.0
# seed = -1
# dim = 3
# #nc=2
# P0 = 0.1
# type_perc = 'bond'
# L_lst =       [256, 304, 362, 430, 512, 608, 724, 861, 1024]
# num_runs =    [300, 250, 150, 150, 80 , 50 , 25 , 15 , 10]
# num_threads = [11 , 11 , 11 , 11 , 9  , 11 , 11 , 6  , 3]
# multi=True
# Equilibration = 'true'

# for idx, L in enumerate(L_lst):
#         base = L**(dim-1)
#         NT = int(f*base)
#         # start = 1/(P0*L**2)
#         # print(start)
#         # stop = 1/(2*nc)
#         # n_points = 50
#         # rho = custom_range(start, stop, n_points)
#         rho = [1/nc]
#         exec_name = f"NT_{NT}_L_{L}_k_{k}_nc_{nc}_dim_{dim}.sh"
        
#         shell_data(L, type_perc, p0, seed, k, NT, dim,
#                 nc, num_runs[idx], rho, exec_name, P0, Equilibration, num_threads[idx],multi)


# f_T = np.linspace(0.001, 0.30, 25)
# c_lst = [0.01, 0.05, 0.25, 0.50]
#L =       1024
seed = -1
dim = 3
#nc=2

type_perc = 'bond'
# L_lst = [1024, 2048, 4096, 8192]
# num_runs = [400, 200, 100, 50]
# nc = 4
#L_lst = [128, 256, 512, 1024]
#num_runs = [300, 150, 50, 5]

L_lst = [512]
num_runs = [50]
nc = 1
c_lst = [0.01, 0.05, 0.1, 0.15, 0.2]

multi=True
Equilibration = 'false'
Properties = 'false'
Mode = 'growth_test'  # use 'sop' for the original fixed-height SOP run
InitialLayout = 'random'  # 'random', 'blocks', or 'alternating'
p0 = 0.6
P0 = 0.2
ft_lst = np.linspace(0.001, 0.4, 20)
#P0_lst = [round(i,1) for i in np.arange(start=0.1,stop=1.1,step=0.1)]

for idx, L in enumerate(L_lst):
        for c in c_lst:
                for ft in ft_lst:
                                rho = 1/nc
                                mode_tag = "" if Mode == "sop" else f"_{Mode}"
                                layout_tag = "" if InitialLayout == "random" else f"_{InitialLayout}"
                                exec_name = f"L_{L}_ft_{ft:.3f}_c_{c}_nc_{nc}_dim_{dim}_p0_{p0}_P0_{P0}{mode_tag}{layout_tag}.sh"

                                shell_data(L, type_perc, p0, seed, c, ft, dim,
                                        nc, num_runs[idx], [1/nc], exec_name, P0, Equilibration, multi,
                                        properties=Properties, mode=Mode,
                                        initial_layout=InitialLayout)
