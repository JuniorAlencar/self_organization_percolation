from src.run_samples_functions import *

num_runs = 100
stop = 10**(-3)
start = 10**(-5)
n_points = 200
rho = custom_range(start, stop, n_points)


type_perc = 'bond'
num_colors = 2
dim = 3
L = 512 
NT = 26000
k = 8.0e-07
p0 = 1.0
seed = -1

multi=True
num_threads=20
DCU_prop=False

exec_name = f"data_{dim}D_L_{L}_num_colors_{num_colors}.sh"
shell_data(L, type_perc, p0, seed, k, NT, dim,
        num_colors, num_runs, rho, exec_name, DCU_prop, num_threads, multi)
