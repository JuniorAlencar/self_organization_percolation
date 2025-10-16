from src.run_samples_functions import *

num_runs = 1000
stop = 1
start = 1
n_points = 1
rho = custom_range(start, stop, n_points)


type_perc = 'bond'
num_colors = 1
dim = 2
L_lst = [512, 1024]
NT_lst = [50, 100]
k_lst = [4.0e-04, 2.0e-04]
p0 = 1.0
seed = -1

multi=True
num_threads=20
DCU_prop=False
for L, k, NT in zip(L_lst, NT_lst, k_lst):
        exec_name = f"data_{dim}D_L_{L}_num_colors_{num_colors}.sh"
        shell_data(L, type_perc, p0, seed, k, NT, dim,
                num_colors, num_runs, rho, exec_name, DCU_prop, num_threads, multi)
