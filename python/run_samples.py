from src.run_samples_functions import shell_data
from src.SOP_parms import *
import numpy as np
# L = 128 => Ns = 700
# L = 192 => Ns = 600
# L = 256 => Ns = 500
# L = 384 => Ns = 300
# L = 512 => Ns = 100
# L = 768 => Ns = 50
# L = 1024 => Ns = 20

# num_runs = [500, 75 ,100, 50, 30, 10]
# L_lst = [128, 192, 256, 384, 512, 768, 1024]



# start = 0.001
# stop = 1/nc
# n_points = 100
# rho = custom_range(start, stop, n_points)
nc_lst = [4]
L_lst = [128, 256, 512, 1024]
f = [round(i, 2) for i in np.arange(0.01, 0.26, 0.01)]
Nt = [[int(fraction*L**2) for fraction in f] for L in L_lst]
#k_lst = [1.0e-04, 1.0e-05, 1.0e-06, 1.0e-07]
k_lst = 1.0e-06
rho_lst = [0.001, 0.05, 0.10, 0.15, 0.20]
num_runs = [300, 100, 20, 10]
type_perc = 'bond'
p0 = 1.0
seed = -1
dim = 3

# =========================

#Nt = 3000
k = 1.0e-06

#nc=2
P0 = 0.1
num_threads = [19, 19, 19, 2]
multi=True

#rho = custom_range(start, stop, n_points=n_points)
#rho = [0.125]
#num_threads = [10, 10]

for idx, L in enumerate(L_lst):
        f0 = [round(i, 4) for i in np.arange(0.005, 0.2, 0.005)]
        NT_lst = [int(L**2 * f) for f in f0]
        
        for Nt in NT_lst:
                exec_name = f"NT_{Nt}_L_{L}.sh"
                shell_data(L, type_perc, p0, seed, k, Nt, dim,
                        nc_lst[0], num_runs[idx], rho_lst, exec_name, P0, num_threads[0], multi)
