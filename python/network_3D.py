import pandas as pd
from mayavi import mlab
import numpy as np
import os

import matplotlib.pyplot as plt
from src.network_functions import convert_positions, convert_positions_sp, plot_3D_cut, plot_3D_full, plot_3D_full_with_planes, plot_projection

# plot network with 7 colors
#L = 256
#dim = 3
#nc = 7
#rho = 1/nc
#k = 1.0e-06
#NT = 3000
#z_level = 225
#z_planes = [25, 125, 225]
#plot_projection(dim, L, nc, rho, k, NT ,z_level)



# plot network with 4 colors
L = 300
dim = 3
nc = 10
rho = 1/nc
k = 1.0e-06
NT = 3000
P0 = 1.00
seed = 1
path_dir = f"../network/{dim}D_L{L}_nc{nc}_rho{rho:.2f}_k{k:.1e}_Nt{NT}/"
filename = f"P0_{P0:.2f}_p0_1.00_seed_{seed}.npz"
output_fn = f"network_positions_P0_{P0:.2f}.csv"
#plot_3D_full(dim, L, nc, rho, k, NT, specific_color=None)
#plot_3D_full_with_planes(dim, L, nc, rho, k, NT)
convert_positions(path_dir,filename, output_fn, dim)
plot_3D_full(dim, L, nc, rho, k, NT, seed, P0, specific_color=None)
