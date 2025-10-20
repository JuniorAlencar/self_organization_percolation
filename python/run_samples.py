from src.run_samples_functions import *
from src.SOP_parms import *

num_runs = 1000
stop = 1
start = 1
n_points = 1
#rho = custom_range(start, stop, n_points)
rho = [1.0]
L = 2028
nc = 1
dim = 2

p0 = [0.3, 0.7, 1.0]
seed = -1
multi=True
num_threads=12
DCU_prop=False
type_perc = 'bond'

for p_ in p0:
        p = sop_choose_params(L=L, n_c=nc,d=dim ,mantissa_decimals=1)
        Nt, k = p.NT, p.k_raw
        exec_name = f"data_{dim}D_L_{L}_p0_{p_}.sh"
        shell_data(L, type_perc, p_, seed, k, Nt, dim,
                nc, num_runs, rho, exec_name, DCU_prop, num_threads, multi)