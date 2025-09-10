from src.run_samples_functions import *

num_runs = 200   # number of external repetitions

stop = 0.24
start = 0.21
n_points = 50
rho = custom_range(start, stop, n_points)

# rho_total = rho_1 + rho_2 + rho_3
# Range 1 -> 0.0001, 0.25
# Range 2 -> 0.19, 0.25
# Range 3 -> 0.21, 0.24

L = 128
p0 = 1.0
seed = -1
type_perc = "bond"
k = 1.0e-5
NT =  205 # 12000 para L=10³
dim = 3
num_colors = 4
exec_name = f"data_{dim}D.sh"
multi=True
num_threads=12

shell_data(L, type_perc, p0, seed, k, NT, dim,
           num_colors, num_runs, rho, exec_name, num_threads ,multi)
