from src.run_samples_cluster import *

start = 0.001
stop = 0.25
n_steps = 200
rho_lst = custom_range(start, stop, n_steps)

type_perc = 'bond'
num_colors = 2
dim = 3
L = 512
NT = 26000
k = 8.0e-07
p0 = 1.0
seed = -1
N_samples = 10
DCU_prop=False

# Example to use IN CLUSTER
# OBS: execute the python with this code in \python folder, with functions um \python\src
# run_multi_rho(L, p0, seed, type_perc, k, NT, dim, num_colors, rho_lst, N_samples, DCU_prop)