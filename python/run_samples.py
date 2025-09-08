from src.run_samples_functions import *

num_runs = 10   # number of external repetitions

stop = 0.25
start = 0.0001
n_points = 50
rho = custom_range(start, stop, n_points)

L = 128
NumSamples = 15000
p0 = 1.0
seed = -1
type_perc = "bond"
k = 1.0e-5
NT =  205# 12000 para L=10³
dim = 3
num_colors = 4
exec_name = f"data_{dim}D.sh"

shell_data(L, NumSamples, type_perc, p0, seed, k, NT, dim,
           num_colors, num_runs, rho, exec_name)
