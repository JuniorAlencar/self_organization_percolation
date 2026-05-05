import os
from src.network_functions import create_folder, plot_run_network_blocks, TIME_BASE_3D



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
L = 256
DIM = 3
nc = 4
rho = 1/nc
c = 0.1
fT = 0.01
path_dir = f"../SOP_data/raw/bond_percolation/num_colors_{nc}/dim_{DIM}/L_{L}/fT_constant/fT_{fT:.6e}/c_{c:.6e}/rho_{rho:.4e}/"
filename = "light_seed_444_ts_20260504T121926_P0_0.10_p0_1.00_PERCOLATION.bin"

network_folder = os.path.join(path_dir, "network")
if not os.path.exists(os.path.join(network_folder, filename)):
    candidates = sorted(
        f for f in os.listdir(network_folder)
        if f.lower().endswith(".bin") and "percolation" in f.lower()
    )
    if candidates:
        filename = candidates[-1]
    else:
        raise FileNotFoundError(
            f"Nenhum arquivo de rede PERCOLATION encontrado em {network_folder}"
        )

# create output dir and save both active-site plot and edge + surface plots
output_dir = f"../network/Connections_L{L}_nc{nc}/"
create_folder(output_dir)

results = plot_run_network_blocks(
    path_dir=path_dir,
    network_filename=filename,
    output_dir=output_dir,
    nc=nc,
    blocks="04b",  # exemplos: "01", "02", "03", "04", "04b", "05", ["02", "05"]
    show_base=False,
    outline_mode="full",
    visual_profile="full",
    edge_color_rule="source",
    max_edges=200000,
)

print("Plots saved in:")
print(f"  output_dir = {output_dir}")
for key, value in results.items():
    print(f"  {key} = {value}")

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
