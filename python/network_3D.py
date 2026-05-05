import os
from src.network_functions import create_folder, plot_run_network_blocks, TIME_BASE_3D


def choose_percolation_network_file(path_dir):
    network_folder = os.path.join(path_dir, "network")
    candidates = sorted(
        f for f in os.listdir(network_folder)
        if f.endswith("_PERCOLATION.bin")
    )

    if not candidates:
        raise FileNotFoundError(
            f"Nenhum arquivo *_PERCOLATION.bin encontrado em {network_folder}"
        )

    return candidates[-1]


L = 512
DIM = 3
nc = 4
rho = 1/nc
c = 0.1
fT = 0.01
path_dir = f"../SOP_data/raw/bond_percolation/num_colors_{nc}/dim_{DIM}/L_{L}/fT_constant/fT_{fT:.6e}/c_{c:.6e}/rho_{rho:.4e}/"
filename = choose_percolation_network_file(path_dir)
print(f"Selected network file: {filename}")

# create output dir and save both active-site plot and edge + surface plots
output_dir = f"../network/Connections_L{L}_nc{nc}/"
create_folder(output_dir)

results = plot_run_network_blocks(
    path_dir=path_dir,
    network_filename=filename,
    output_dir=output_dir,
    nc=nc,
    blocks=["03", "04", "04b"],  # exemplos: "01", "02", "03", "04", "04b", "05", ["02", "05"]
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
