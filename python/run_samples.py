from src.run_samples_functions import *

num_runs = 100
stop = 1
start = 1
# stop = 1
# start = 1
n_points = 1
#rho = custom_range(start, stop, n_points)
rho = [0.001]
# Range 1 -> 0.0001, 0.25 OK
# Range 2 -> 0.21 0.23

type_perc = 'bond'
num_colors = 4
dim = 3
L = [128, 512]
NT = [1600, 6500]
k = [1.0e-05, 3.0e-06]
p0 = 1.0
seed = -1

multi=True
num_threads=20

for l in L:
    exec_name = f"data_{dim}D_L_{l}_num_colors_{num_colors}.sh"
    shell_data(l, type_perc, p0, seed, k, NT, dim,
            num_colors, num_runs, rho, exec_name, num_threads, multi)
