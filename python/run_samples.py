from src.run_samples_functions import *
from src.SOP_parms import *

num_runs = 50
stop = 0.25
start = 0.001
n_points = 200
rho = custom_range(start, stop, n_points)

L = 512
nc = [2, 4, 8]
dim = 3

p0 = 1.0
seed = -1
multi=True
num_threads=[20, 18, 10]
DCU_prop=False
type_perc = 'bond'

for i in range(len(nc)):
        p = sop_choose_params(L=L, n_c=nc[i], mantissa_decimals=1)
        Nt, k = p.NT, p.k_raw
        exec_name = f"data_{dim}D_L_{L}_num_colors_{nc[i]}.sh"
        shell_data(L, type_perc, p0, seed, k, Nt, dim,
                nc[i], num_runs, rho, exec_name, DCU_prop, num_threads[i], multi)