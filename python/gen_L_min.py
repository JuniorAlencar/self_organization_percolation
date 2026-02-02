from src.run_samples_functions import *

L = np.arange(210, 235, 5)
nc = 2
rho = [1/nc]
p0 = 1.0
dim = 3
seed = -1
type_perc = 'bond'
Nt = 3000
k=1.0e-06
P0 = 1.0
multi=True
num_threads = 16
num_runs = 10

for l in L:
    exec_name = f"data_{dim}D_L_{l}_p0_{p0}_nc_{nc}_props.sh"
    shell_data(l, type_perc, p0, seed, k, Nt, dim,
                nc, num_runs, rho, exec_name, P0, num_threads, multi)