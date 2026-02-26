from src.run_samples_functions import *
from src.SOP_parms import *

# L = 128 => Ns = 500
# L = 192 => Ns = 300
# L = 256 => Ns = 100
# L = 384 => Ns = 75
# L = 512 => Ns = 50
# L = 768 => Ns = 30
# L = 1024 => Ns = 10

# num_runs = [500, 75 ,100, 50, 30, 10]
# L_lst = [128, 192, 256, 384, 512, 768, 1024]

nc_lst = [2] 
rho = [0.5]

# start = 0.001
# stop = 1/nc
# n_points = 100
# rho = custom_range(start, stop, n_points)
L_lst = [272, 296, 312, 336, 360, 424, 464, 849, 928]
#nc = 2
num_runs = [100, 100, 100, 100, 75, 75, 75, 30, 10]
p0 = 1.0
dim = 3
type_perc = 'bond'
seed = -1
Nt = 3000
k=1.0e-06
#nc=2
P0 = 0.1
num_threads = [11, 11, 11, 11, 11, 11, 11, 3, 3]
multi=True

#rho = custom_range(start, stop, n_points=n_points)
#rho = [0.125]
#num_threads = [10, 10]
for i in range(len(L_lst)):                
        exec_name = f"data_{dim}D_L_{L_lst[i]}_p0_{p0}_nc_{nc_lst[0]}_props.sh"
        shell_data(L_lst[i], type_perc, p0, seed, k, Nt, dim,
                nc_lst[0], num_runs[i], rho, exec_name, P0, num_threads[0], multi)
