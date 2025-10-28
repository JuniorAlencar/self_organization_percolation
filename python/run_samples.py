from src.run_samples_functions import *
from src.SOP_parms import *

num_runs = 50

start = 0.001
stop = 0.50
n_points = 200

L_lst = [512]
nc_ = [2, 4, 8]
dim = 3

p0 = 1.0
seed = -1
multi=True
num_threads = 20
DCU_prop=True
type_perc = 'bond'

for i in range(len(L_lst)):                
        for nc in nc_:
                if(nc==2):
                        start = 10**(-5)
                        stop = 10**(-3)
                else:
                        stop = round(1/nc,2)
                rho = custom_range(start, stop, n_points)
                p = sop_choose_params(L=L_lst[i], n_c=nc, d=dim, mantissa_decimals=1)
                Nt, k = p.NT, p.k_raw        
                exec_name = f"data_{dim}D_L_{L_lst[i]}_nc_{nc}_props.sh"
                shell_data(L_lst[i], type_perc, p0, seed, k, Nt, dim,
                        nc, num_runs, rho, exec_name, num_threads, multi)
