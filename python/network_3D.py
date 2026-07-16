import os
from src.network_functions import (
    create_folder,
    plot_run_network_blocks,
    save_dynamic_height_cumulative_frames,
    write_gimp_crop_frames_script,
    TIME_BASE_3D,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


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


L = 64
DIM = 3
nc = 2

if nc==1:
    fT = 0.0480975
elif nc==2:
    fT= 0.02780737

rho = 1/nc
c = 0.05
#fT = 0.15
FRAME_WORKERS = min(3, os.cpu_count() or 1)

path_dir = os.path.join(
    PROJECT_ROOT,
    "SOP_data",
    "raw_growth_test_dynamic",
    "bond_percolation",
    f"num_colors_{nc}",
    f"dim_{DIM}",
    f"L_{L}",
    "fT_constant",
    f"fT_{fT:.6e}",
    f"c_{c:.6e}",
    f"rho_{rho:.4e}",
)

#types_list = ['random', 'alternating', 'blocks']
types_list = ['random', 'alternating', 'blocks']

for type_base in types_list:
    output_dir = os.path.join(PROJECT_ROOT, "animate", f"L_{L}_nc{nc}", f"type_base_{type_base}")
    create_folder(output_dir)

    tb = type_base
    if(nc==2):
        if tb == 'random':
            filename = "light_seed_44_ts_20260711T105243_P0_0.20_p0_0.60.bin"
        elif tb == 'alternating':
            filename = "light_seed_44_ts_20260711T105255_P0_0.20_p0_0.60_base_alternating.bin"
        elif tb == 'blocks':
            filename = "light_seed_44_ts_20260711T105259_P0_0.20_p0_0.60_base_blocks.bin"
    else:
        if tb == 'random':
            filename = "light_seed_44_ts_20260711T105553_P0_0.20_p0_0.60.bin"
        elif tb == 'alternating':
            filename = "light_seed_44_ts_20260711T105547_P0_0.20_p0_0.60_base_alternating.bin"
        elif tb == 'blocks':
            filename = "light_seed_44_ts_20260711T105543_P0_0.20_p0_0.60_base_blocks.bin"
    
    save_dynamic_height_cumulative_frames(
        path_dir=path_dir,
        output_dir=output_dir,
        network_filename=filename,
        nc=nc,
        L=L,
        dim=DIM,
        frame_stride=1,
        stop_when_front_reaches_height=True,
        highlight_growth_front=True,
        frame_workers=FRAME_WORKERS,
        resume=True,
        overwrite_existing=False,
        #height_limit=256,
    )

    write_gimp_crop_frames_script(
        output_dir=output_dir,
        run=True,
    )
