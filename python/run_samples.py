from src.run_samples_functions import *
from src.SOP_parms import *

num_runs = [500, 100, 50]

start = 0.001
#stop = 0.50
n_points = 200

rho = [0.125]

L_lst = [128, 256, 512]
nc_ = [2, 4, 8]
dim = 3

p0 = 1.0
seed = -1
multi=True
num_threads = 20
DCU_prop=True
type_perc = 'bond'
k = 1.0e-06
Nt = 3000
P0 = 0.5

for i in range(len(L_lst)):                
        for nc in nc_:
                if(nc == 8 and L_lst[i] == 512):
                        num_threads = 11
                
                stop = round(1/nc, 3)
                rho = custom_range(start, stop, n_points)
                exec_name = f"data_{dim}D_L_{L_lst[i]}_nc_{nc}_P0_{P0}_props.sh"
                shell_data(L_lst[i], type_perc, p0, seed, k, Nt, dim,
                         nc, num_runs[i], rho, exec_name, P0, num_threads, multi)

