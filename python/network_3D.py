import os
from src.network_functions import (
    create_folder,
    plot_run_network_blocks,
    save_dynamic_height_cumulative_frames,
    write_gimp_crop_frames_script,
    TIME_BASE_3D,
)


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


L = 256
DIM = 3
nc = 2
rho = 1/nc
c = 0.03
fT = 0.05
window = 300
path_dir = f"../SOP_data/raw_growth_test_dynamic/bond_percolation/num_colors_{nc}/dim_{DIM}/L_{L}/fT_constant/fT_{fT:.6e}/c_{c:.6e}/rho_{rho:.4e}/stationary_window_{window}/"

#types_list = ['random', 'alternating', 'blocks']
types_list = ['random', 'alternating', 'blocks']

for type_base in types_list:
    output_dir = f"../animate/L_{L}_nc{nc}/type_base_{type_base}/"
    create_folder(output_dir)

    tb = type_base
    if tb == 'random':
        filename = "light_seed_1765624488_ts_20260605T140006_P0_0.50_p0_0.50.bin"
    elif tb == 'alternating':
        filename = "light_seed_1765624488_ts_20260605T140011_P0_0.50_p0_0.50_base_alternating.bin"
    elif tb == 'blocks':
        filename = "light_seed_1765624488_ts_20260605T140017_P0_0.50_p0_0.50_base_blocks.bin"
    
    save_dynamic_height_cumulative_frames(
        path_dir=path_dir,
        output_dir=output_dir,
        network_filename=filename,
        nc=nc,
        L=L,
        dim=DIM,
        frame_stride=1,
        stop_when_front_reaches_height=True,
        resume=True,
        #height_limit=256,
    )

    write_gimp_crop_frames_script(
        output_dir=output_dir,
        run=True,
    )

#print(" ".join(gimp_job["command"]))

# Flags possíveis em calculations:
#   "active_sites_3d", "network_edges", "surface_heatmap",
#   "surface_3d", "surface_posteq_3d", "active_sites_by_color"
# results = plot_run_network_blocks(
#     path_dir=path_dir,
#     network_filename=filename,
#     output_dir=output_dir,
#     nc=nc,
#     calculations=[
#         "network_edges"
#     ],
#     show_base=False,
#     outline_mode="full",
#     visual_profile="full",
#     edge_color_rule="source",
#     max_edges=200000,
# )

# print("Plots saved in:")
# print(f"  output_dir = {output_dir}")
# for key, value in results.items():
#     print(f"  {key} = {value}")

# RANDOM_BASE - DYNAMIC HEIGHT - 3D
# L = 256, p0=0.5, seed=129798198, c= 0.03, 
# fT=0.05, nc =8, rho=0.125, P0=0.5
# window = 300

# Exemplo para frames cumulativos da altura dinâmica:
#
# L = 256
# DIM = 3
# nc = 8
# rho = 1 / nc
# c = 0.03
# fT = 0.05
# path_dir = (
#     f"../SOP_data/raw_growth_test_dynamic/bond_percolation/"
#     f"num_colors_{nc}/dim_{DIM}/L_{L}/fT_constant/"
#     f"fT_{fT:.6e}/c_{c:.6e}/rho_{rho:.4e}/stationary_window_300/"
# )
# output_dir = f"../network/dynamic_frames_L{L}_nc{nc}_c{c}_fT{fT}/"
# frames = save_dynamic_height_cumulative_frames(
#     path_dir=path_dir,
#     output_dir=output_dir,
#     nc=nc,
#     L=L,
#     dim=DIM,
#     frame_stride=1,
#     max_frames=None,
#     height_limit=None,  # use, por exemplo, 256 no caso 2D para truncar y<=256
#     stop_when_front_reaches_height=True,
#     show_base=False,
#     visual_profile="full",
# )
# print(frames["output_dir"])
#
# gimp_job = write_gimp_crop_frames_script(
#     output_dir=frames["output_dir"],
#     cropped_dir=os.path.join(frames["output_dir"], "cropped"),
#     threshold=8,
#     run=False,
# )
# print("Rode no terminal:")
# print(" ".join(gimp_job["command"]))
