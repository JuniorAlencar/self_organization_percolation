import pandas as pd
from mayavi import mlab
import numpy as np
import os
from src.network_functions import convert_positions_sp

L = 256
dim = 3
nc = 7
rho = 1/nc
k = 1.0e-06
NT = 3000

path_dir = f"../network/{dim}D_L{L}_nc{nc}_rho{rho:.2f}_k{k:.1e}_Nt{NT}/"

filename = "map_shortest_P0_0.10_p0_1.00_seed_1.npz"

file_positions = f"../network/{dim}D_L{L}_nc{nc}_rho{rho:.2f}_k{k:.1e}_Nt{NT}/network_positions_sp.csv"
seed = 1

# Create file positions, if dont exist
if not os.path.exists(file_positions):
    print("file positions don't exist, create it...")
    convert_positions_sp(path_dir, "map_shortest_P0_0.10_p0_1.00_seed_1.npz", "network_positions_sp.csv", dim)
    print("File with positions created")

a = 0
b = 0
seed = 1

df = pd.read_csv(path_dir + "network_positions_sp.csv")

cut = L * (1/2)

# Ajuste as coordenadas x e y para levar em conta a e b
df['x'] = (df['x'] + a) % L
df['y'] = (df['y'] + b) % L

figure_size = (800, 800)
mlab.figure(size=figure_size, bgcolor=(1, 1, 1))

colors = [i+2 for i in range(nc)]
colors_used = [
    (0.9, 0.1, 0.1),  # 2 - red
    (1.0, 0.5, 0.0),  # 3 - orange
    (0.1, 0.9, 0.1),  # 4 - green
    (0.1, 0.1, 0.9),  # 5 - blue
    (0.8, 0.2, 0.8),  # 6 - purple
    (0.2, 0.8, 0.8),  # 7 - teal
    (1.0, 1.0, 0.0),  # 8 - yellow
    (0.6, 0.4, 0.2),  # 9 - brown
]

for idx, color in enumerate(colors):    
    df_color = df[df['color'] == color]

    x = df_color['x']
    y = df_color['y']
    z = df_color['z']
    
    if len(x) > 0:
        pts = mlab.points3d(
        x, y, z,
        color=colors_used[idx],
        scale_factor=3,
        opacity=1.0,
        mode='cube'
    )
    # Ativar contorno preto nos cubos
    pts.actor.property.edge_visibility = True
    pts.actor.property.edge_color = (0, 0, 0)   # preto
    pts.actor.property.line_width = 0.00         # espessura da borda (ajuste se quiser)
#    else:
        #pass

# Depois de plotar os pontos3d, lá no final, antes do mlab.show()
mlab.outline(
    extent=[0, L, 0, L, 0, L],  # [xmin, xmax, ymin, ymax, zmin, zmax]
    color=(0, 0, 0),            # contorno preto
    line_width=2.0              # espessura da linha
)

center = (L/2, L/2, L/2)

mlab.view(
    azimuth=0,
    elevation=90,
    distance=2.4 * L,      # bem mais perto que 3.25*L
    focalpoint=center      # olha pro centro da rede
)

path_out_network = path_dir + f"L{L}_seed{seed}_sp.png" 
mlab.savefig(path_out_network, magnification=4)
print(f"network save in {path_out_network}")
mlab.show()

