from src.run_samples_functions import *

num_runs_lst = [500, 300, 100]   # number of external repetitions
#stop = 0.25
#start = 0.001
stop = 0.23
start = 0.21
n_points = 100
rho = custom_range(start, stop, n_points)

# Range 1 -> 0.0001, 0.25 OK
# Range 2 -> 0.21 0.23

#0.21902, 0.23078 l=128, k=1.0e-04, nt= 205
type_perc = 'bond'
num_colors = 4
dim = 3
L_lst = [128, 256, 512]
NT_lst = [200, 800, 3500]
k_lst = [1.0e-04,1.0e-05, 1.0e-06]
p0 = 1.0
seed = -1

multi=True
num_threads=12
for l in range(len(L_lst)):
    exec_name = f"data_{dim}D_L_{L_lst[l]}.sh"
    shell_data(L_lst[l], type_perc, p0, seed, k_lst[l], NT_lst[l], dim,
            num_colors, num_runs_lst[l], rho, exec_name, num_threads ,multi)
