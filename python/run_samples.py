from src.run_samples_functions import *
from src.SOP_parms import *

# L = 128 => Ns = 700
# L = 192 => Ns = 600
# L = 256 => Ns = 500
# L = 384 => Ns = 300
# L = 512 => Ns = 100
# L = 768 => Ns = 50
# L = 1024 => Ns = 20

# num_runs = [500, 75 ,100, 50, 30, 10]
# L_lst = [128, 192, 256, 384, 512, 768, 1024]



# start = 0.001
# stop = 1/nc
# n_points = 100
# rho = custom_range(start, stop, n_points)
nc_lst = [4]
rho = [0.25]
L_lst = [128]
num_runs = [300]
p0 = 1.0
dim = 3
type_perc = 'bond'
seed = -1
f0 = [round(i, 2) for i in np.arange(0.01, 0.26, 0.01)]
NT_lst = [int(L_lst[0]**2 * f) for f in f0]
#Nt = 3000
K_lst = [5.0e-04, 1.0e-05, 5.0e-05]
#nc=2
P0 = 0.1
num_threads = [11]
multi=True

#rho = custom_range(start, stop, n_points=n_points)
#rho = [0.125]
#num_threads = [10, 10]

for k in K_lst:
        if(k==5.0e-05):
                NT_lst_filter = NT_lst[:-7]
        elif(k==1.0e-05):
                NT_lst_filter = NT_lst[-7:]
        elif(k==5.0e-04):
                NT_lst_filter = NT_lst
        for Nt in NT_lst_filter:
                exec_name = f"k_{k:.1e}_NT_{Nt}_props.sh"
                shell_data(L_lst[0], type_perc, p0, seed, k, Nt, dim,
                        nc_lst[0], num_runs[0], rho, exec_name, P0, num_threads[0], multi)
