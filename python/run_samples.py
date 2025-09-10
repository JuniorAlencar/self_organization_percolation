from src.run_samples_functions import *

num_runs = 20   # number of external repetitions
stop = 0.23078
start = 0.21902
# stop = 0.25
# start = 0.0001
n_points = 50
rho = custom_range(start, stop, n_points)

# Range 1 -> 0.0001, 0.25 OK
# Range 2 -> 0.21 0.23

#0.21902, 0.23078 l=128, k=1.0e-04, nt= 205
type_perc = 'bond'
num_colors = 4
dim = 3
L = 256
NT = 820
k = 1.0e-05
p0 = 1.0
seed = -1
exec_name = f"data_{dim}D.sh"
multi=True
num_threads=20

shell_data(L, type_perc, p0, seed, k, NT, dim,
           num_colors, num_runs, rho, exec_name, num_threads ,multi)
