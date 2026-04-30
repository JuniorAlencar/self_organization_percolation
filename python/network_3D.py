import pandas as pd
from mayavi import mlab
import numpy as np
import os

import matplotlib.pyplot as plt
from src.network_functions import create_folder, plot_3D_full_codec, plot_3D_preteq_posteq, plot_3D_full_codec_by_species, plot_network_edges
from src.network_functions import TIME_BASE_3D, load_or_create_positions_codec
import glob



#filename = "light_seed_44_ts_20260428T133536_P0_0.10_p0_1.00_PERCOLATION.npz"
#filename = "light_seed_44_ts_20260428T133536_P0_0.10_p0_1.00_PERCOLATION.npz"

#output_dir = f"../network/{dim}D_L{L}_nc{nc}_rho{rho:.4f}_c{c:.2f}_fT{fT:.2f}"
#create_folder(output_dir)

# plot_3D_full_codec(
#     path_dir=path_dir + "/network/",
#     filename=filename,
#     path_out=os.path.join(output_dir, f"Network_FULL_nc{nc}_L{L}.png"),
#     figure_name=f"network{dim}D_L_{L}_nc_{nc}",
#     positions_file=os.path.join(output_dir, "Network_FULL.parquet"),
#     L=L,
#     nc=nc
# )

#L=256, c=0.01, ft=0.07, ns=1, dim=3
#"light_seed_44_ts_20260428T142421_P0_0.10_p0_1.00_PERCOLATION.npz"
#L=512, ''
#light_seed_44_ts_20260428T142755_P0_0.10_p0_1.00_PERCOLATION.npz
L = 128
DIM = 3
nc = 1
rho = 1/nc
c = 0.01
fT = 0.06
path_dir = f"../SOP_data/raw/bond_percolation/num_colors_{nc}/dim_{DIM}/L_{L}/fT_constant/fT_{fT:.6e}/c_{c:.6e}/rho_{rho:.4e}/network/"
filename = "light_seed_1488959839_ts_20260430T125300_P0_0.10_p0_1.00_PERCOLATION.bin"


# create output dir and save both active-site plot and edge plot
output_dir = f"../network/Connections_L{L}_nc{nc}/"
create_folder(output_dir)

# plot active sites (percolated) as cubes/points
# plot_3D_full_codec(
#     path_dir=path_dir,
#     filename=filename,
#     path_out=os.path.join(output_dir, f"Network_ACTIVE_nc{nc}_L{L}.png"),
#     figure_name=f"network_active_{L}",
#     positions_file=output_dir + "positions.parquet",
#     L=L,
#     nc=nc,
#     time_base=TIME_BASE_3D,
#     show_base=False
# )

# plot edges colored by site color and save to PNG (limits edges for performance)
edges_out = os.path.join(output_dir, f"Network_EDGES_nc{nc}_L{L}.png")
plot_network_edges(
    path_dir=path_dir,
    filename=filename,
    figure_name=f"network_edges_{L}",
    line_width=0.6,
    alpha=0.9,
    edge_color_rule="source",
    max_edges=200000,
    nc=nc,
    show_base=False,
    path_out=edges_out
)
print(f"Plots saved in {output_dir}")

# plot_3D_full_codec_by_species(
#     path_dir=path_dir + "/network/",
#     filename=filename,
#     path_out_dir=output_dir,
#     figure_name="network_species",
#     L=L,
#     nc=nc,
#     time_base=TIME_BASE_3D,
#     show_base=False,
#     outline_mode="full"
# )

# plot_3D_preteq_posteq(
#     path_dir_pre=path_dir + "/network_preteq/",
#     filename_pre=filename,
#     path_dir_post=path_dir + "/network_posteq/",
#     filename_post=filename,
#     L=L,
#     nc=nc,
#     path_out_pre=output_dir + f"/Network_PRETEQ_nc{nc}_L{L}.png",
#     path_out_post=output_dir + f"/Network_POSTEQ_nc{nc}_L{L}.png",
#     figure_name_pre=f"Network_PRETEQ_nc{nc}_L{L}",
#     figure_name_post=f"Network_POSTEQ_nc{nc}_L{L}",
#     positions_file_pre=output_dir + f"/preteq_positions_nc{nc}_L{L}.parquet",
#     positions_file_post=output_dir + f"/posteq_positions_nc{nc}_L{L}.parquet"
# )