import pandas as pd
from mayavi import mlab
import numpy as np
import os

import matplotlib.pyplot as plt
from src.network_functions import convert_positions, convert_positions_sp, plot_3D_cut, plot_3D_full, plot_3D_full_with_planes, plot_projection, read_network

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
L = 256
dim = 3
nc = 8
rho = 1/nc
k = 1.0e-06

p0 = 1.0
seed = 42
z_level = 0


P0 = 0.1
NT = 3276
path_dir = f"../network/{dim}D_L{L}_nc{nc}_rho{rho:.3f}_k{k:.1e}_Nt{NT}/"
#filename = f"complex_seed_42_ts_20260407T083457_P0_0.10_p0_1.00.npz"
filename = f"complex_seed_42_ts_20260407T083857_P0_0.10_p0_1.00.npz"
file_positions = f"network_positions_P0_{P0:.2f}.parquet"
plot_3D_full(path_dir, file_positions, P0, L, nc, seed, filename, specific_color=None, show_base=False)
#convert_positions(path_dir, filename, file_positions , dim)
#====
# NT = 3276
# path_dir = f"../network/{dim}D_L{L}_nc{nc}_rho{rho:.3f}_k{k:.1e}_Nt{NT}/"
# filename = f"complex_seed_42_ts_20260407T083857_P0_0.10_p0_1.00.npz"
# file_positions = f"network_positions_P0_{P0:.2f}.parquet"
#convert_positions(path_dir, filename, file_positions , dim)

#plot_projection(path_dir, file_positions, L, P0, seed, filename, z_level)
plot_3D_full(path_dir, file_positions, P0, L, nc, seed, filename, specific_color=None, show_base=False)

# P0 = 0.1
# filename = f"light_seed_1_ts_20260325T213653_P0_0.10_p0_1.00.npz"
# file_positions = f"network_positions_P0_{P0:.2f}.parquet"
# plot_projection(path_dir, file_positions, L, P0, seed, filename, z_level)
# plot_3D_full(path_dir, file_positions, P0, L, nc, seed, filename, specific_color=None, show_base=False)


#convert_positions(path_dir, filename, output_fn , dim)
# filename = f"light_seed_523_ts_20260325T153417_P0_0.10_p0_1.00.npz"

#convert_positions(path_dir, filename, output_fn , dim)


#plot_projection(path_dir, output_fn, L, P0, seed, filename, z_level)


#plot_3D_full_with_planes(dim, L, nc, rho, k, NT)
#convert_positions(path_dir,filename, output_fn, dim)
#plot_3D_full(dim, L, nc, rho, k, NT, seed, P0, specific_color=None)
