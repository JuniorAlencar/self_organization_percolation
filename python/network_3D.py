import pandas as pd
from mayavi import mlab
import numpy as np
import os

import matplotlib.pyplot as plt
from src.network_functions import create_folder, plot_3D_full_codec, plot_3D_preteq_posteq, check_codification
import glob

L = 256
dim = 3
nc = 4
rho = 1/nc
k = 1.0e-06
NT = 655

path_dir = f"../SOP_data/raw/bond_percolation/num_colors_{nc}/dim_{dim}/L_{L}/NT_constant/NT_{NT}/k_{k:.1e}/rho_{rho:.4e}"
filename = "light_seed_44_ts_20260422T155829_P0_0.10_p0_1.00.npz"

output_dir = f"../network/{dim}D_L{L}_nc{nc}_rho{rho:.4f}_k{k:.1e}_NT{NT}"
create_folder(output_dir)

plot_3D_full_codec(
    path_dir=path_dir + "/network/",
    filename=filename,
    path_out=os.path.join(output_dir, f"Network_FULL_nc{nc}_L{L}.png"),
    figure_name=f"network{dim}D_L_{L}_nc_{nc}",
    positions_file=os.path.join(output_dir, "Network_FULL.parquet"),
    L=L,
    nc=nc
)

plot_3D_preteq_posteq(
    path_dir_pre=path_dir + "/network_preteq/",
    filename_pre=filename,
    path_dir_post=path_dir + "/network_posteq/",
    filename_post=filename,
    L=L,
    nc=nc,
    path_out_pre=output_dir + f"/Network_PRETEQ_nc{nc}_L{L}.png",
    path_out_post=output_dir + f"/Network_POSTEQ_nc{nc}_L{L}.png",
    figure_name_pre=f"Network_PRETEQ_nc{nc}_L{L}",
    figure_name_post=f"Network_POSTEQ_nc{nc}_L{L}",
    positions_file_pre=output_dir + f"/preteq_positions_nc{nc}_L{L}.parquet",
    positions_file_post=output_dir + f"/posteq_positions_nc{nc}_L{L}.parquet"
)