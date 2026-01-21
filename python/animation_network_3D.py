import os
import numpy as np
import pandas as pd
from mayavi import mlab
from src.network_functions import convert_positions_animation, read_network

dim = 3
L = 256
nc = 4
rho = 1/nc
k = 1.0e-06
Nt = 3000

path_dir  = f"../animation/{dim}D_L{L}_nc{nc}_rho{rho:.2f}_k{k:.1e}_Nt{Nt}/"
filename = "P0_0.10_p0_1.00_seed_1.npz"
data_positions = "network_positions_time.csv"

# Se o CSV com posições não existir, cria
if not os.path.exists(path_dir + data_positions):
    print("file positions don't exist, create it...")
    convert_positions_animation(path_dir, filename, dim)
    print("File with positions created")

# Pasta de saída
folder_with_figures = path_dir + "positions_vs_time/"
os.makedirs(folder_with_figures, exist_ok=True)

a = 0
b = 0
seed = 1

df = pd.read_csv(path_dir + data_positions)

# Ajuste PBC em x,y
df['x'] = (df['x'] + a) % L
df['y'] = (df['y'] + b) % L

# Lista de tempos (ordenados, únicos)
frame_times = np.sort(df['time'].unique())
print("N_frames =", len(frame_times), " | t_min =", frame_times[0], "t_max =", frame_times[-1])

colors = [1, 2, 3, 4]
colors_used = [
    (0.9, 0.1, 0.1),  # 1 - red
    (0.1, 0.9, 0.1),  # 2 - green
    (0.1, 0.1, 0.9),  # 3 - blue
    (0.2, 0.8, 0.8),  # 4 - teal
]

# Opcional: ativar modo offscreen (depende do ambiente)
# mlab.options.offscreen = True

# Cria UMA figura e reutiliza em todos os frames
figure_size = (800, 800)
fig = mlab.figure(size=figure_size, bgcolor=(1, 1, 1))
time = 0
for i in range(2):
#for i, time in enumerate(frame_times[420:]):

    # Limpa cena ANTES de desenhar o frame atual
    mlab.clf()

    for idx, color in enumerate(colors):
        df_color = df[(df['color'] == color) & (df['time'] <= time)]

        x = df_color['x'].values
        y = df_color['y'].values
        z = df_color['z'].values

        if x.size > 0:
            pts = mlab.points3d(
            x, y, z,
            color=colors_used[idx],
            scale_factor=1,
            opacity=1.0,
            mode='cube'
        )
        pts.actor.property.edge_visibility = True
        pts.actor.property.edge_color = (0, 0, 0)   # preto
        pts.actor.property.line_width = 0.00         # espessura da borda (ajuste se quiser)
        
    mlab.outline(
        extent=[0, L, 0, L, 0, L],  # [xmin, xmax, ymin, ymax, zmin, zmax]
        color=(0, 0, 0),            # contorno preto
        line_width=2.0              # espessura da linha
    )
    center = (L/2, L/2, L/2)
    mlab.view(
        azimuth=70,
        elevation=65,
        distance=3.1 * L,      # bem mais perto que 3.25*L
        focalpoint=center      # olha pro centro da rede
    )
    path_out_network = os.path.join(folder_with_figures, f"frame_{time:04d}.png")   
    mlab.savefig(path_out_network, magnification=4)

    print(f"network save in {path_out_network}")
    if (i % 10) == 0 or i == len(frame_times) - 1:
        print(f"[{i+1:5d}/{len(frame_times)}]  t = {time}")
    
# Fecha figura ao final
mlab.close(fig)
print("Frames salvos em:", folder_with_figures)
