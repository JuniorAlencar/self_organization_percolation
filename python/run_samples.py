from src.run_samples_functions import shell_data, custom_range
from src.SOP_parms import *
import numpy as np
# L = 128 => Ns = 700
# L = 192 => Ns = 600
# L = 256 => Ns = 500
# L = 384 => Ns = 300
# L = 512 => Ns = 100
# L = 768 => Ns = 50
# L = 1024 => Ns = 20

# num_runs = [500, 75 ,100, 50, 30, 10]
# L_lst = [128, 192, 256, 384, 512, 768, 1024]

#L_lst = [256, 384, 512, 768, 1024]
nc = 8

if nc == 2:
    f = 0.03
elif nc == 4:
    f = 0.02
elif nc == 8:
    f = 0.01

#Nt = [[int(fraction*L**2) for fraction in f] for L in L_lst]
#f0 = 0.02
#rho = 1/nc
k = 1.0e-06
p0 = 1.0
seed = -1
dim = 3
#nc=2
P0 = 0.1
type_perc = 'bond'
L_lst =       [304, 362, 430, 608, 724, 861]
num_runs =    [250, 150, 150, 50, 25, 15]
num_threads = [11 , 11 , 11 , 11, 9, 8]
multi=True

for idx, L in enumerate(L_lst):
        base = L**(dim-1)
        NT = int(f*base)
        # start = 1/(P0*L**2)
        # print(start)
        # stop = 1/(2*nc)
        # n_points = 50
        # rho = custom_range(start, stop, n_points)
        rho = [1/nc]
        exec_name = f"NT_{NT}_L_{L}_k_{k}_nc_{nc}_dim_{dim}.sh"
        
        shell_data(L, type_perc, p0, seed, k, NT, dim,
                nc, num_runs[idx], rho, exec_name, P0, num_threads[idx], multi)
