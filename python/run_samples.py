from src.run_samples_functions import *

num_runs_lst = [1000, 500, 250]   # number of external repetitions
stop = 0.25
start = 0.0001
# stop = 1
# start = 1
n_points = 40
rho = custom_range(start, stop, n_points)

# Range 1 -> 0.0001, 0.25 OK
# Range 2 -> 0.21 0.23

type_perc = 'bond'
num_colors = 4
dim = 3
L = [128, 256, 512]
NT = [1600, 6500, 26000]
k_lst = [1.0e-05, 3.0e-06, 8.0e-07]
p0 = 1.0
seed = -1

multi=True
num_threads=20

for i in range(len(L)):    
    exec_name = f"data_{dim}D_L_{L[i]}_num_colors_{num_colors}.sh"
    shell_data(L[i], type_perc, p0, seed, k_lst[i], NT[i], dim,
            num_colors, num_runs_lst[i], rho, exec_name, num_threads, multi)
