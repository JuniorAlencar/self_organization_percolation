import pandas as pd
from mayavi import mlab
import numpy as np
import os
from src.network_functions import convert_positions

L = 256
dim = 3
nc = 7
rho = 0.14
k = 1.0e-06
NT = 3000

path_dir = f"../network/{dim}D_L{L}_nc{nc}_rho{rho}_k{k:.1e}_Nt{NT}/"
filename = "P0_0.10_p0_1.00_seed_1.npz"

file_positions = f"../network/{dim}D_L{L}_nc{nc}_rho{rho}_k{k:.1e}_Nt{NT}/" + "network_positions.csv" 
seed = 1

# Create file positions, if dont exist
if not os.path.exists(file_positions):
    print("file positions don't exist, create it...")
    convert_positions(path_dir, "P0_0.10_p0_1.00_seed_1.npz", dim)
    print("File with positions created")

a = 0
b = 0
seed = 1

df = pd.read_csv(path_dir + "network_positions.csv")

cut = L * (1/2)

# Ajuste as coordenadas x e y para levar em conta a e b
df['x'] = (df['x'] + a) % L
df['y'] = (df['y'] + b) % L

figure_size = (800, 800)
mlab.figure(size=figure_size, bgcolor=(1, 1, 1))

colors = [2, 3, 4, 5, 6, 7, 8]
colors_used = [
    (0.9, 0.1, 0.1),  # 2 - red
    (1.0, 0.5, 0.0),  # 3 - orange
    (0.1, 0.9, 0.1),  # 4 - green
    (0.1, 0.1, 0.9),  # 5 - blue
    (0.8, 0.2, 0.8),  # 6 - purple
    (0.2, 0.8, 0.8),  # 7 - teal
    (1.0, 1.0, 0.0)   # 8 - yellow
]

for idx, color in enumerate(colors):
    # NÃO sobrescrever df
    df_color = df[df['color'] == color]

    mask_oct = (df_color['x'] <= cut) | (df_color['y'] <= cut) | (df_color['z'] <= cut)
    x = df_color['x'][mask_oct]
    y = df_color['y'][mask_oct]
    z = df_color['z'][mask_oct]

    if len(x) > 0:
        mlab.points3d(x, y, z, color=colors_used[idx], scale_factor=1, mode='cube')

mlab.view(azimuth=60, elevation=60, distance=3.25*L)
path_out_network = path_dir + f"L{L}_seed{seed}_{cut}_better.png" 
mlab.savefig(path_out_network, magnification=4)
print(f"network save in {path_out_network}")
mlab.show()


