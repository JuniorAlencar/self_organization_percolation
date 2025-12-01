import os
import numpy as np
import pandas as pd
from mayavi import mlab

def read_network(path_dir, filename):
    fname = path_dir + filename
    
    data = np.load(fname)

    print("chaves dentro do arquivo:", data.files)

    dim        = int(data["dim"])
    num_colors = int(data["num_colors"])
    seed       = int(data["seed"])
    shape      = tuple(data["shape"])
    rho        = data["rho"]

    # matriz 3D (estava salva achatada em 'data')
    mat3d = data["data"].reshape(shape)  # shape = (L, L, L)
    return mat3d

TIME_BASE_3D = 100_000_000  # fator da codificação: C * 100000000 + t

def convert_positions_3D(path_dir, filename, dim, time_base=TIME_BASE_3D):
    """
    Lê o arquivo da rede (2D ou 3D) e gera um CSV com:
      - 3D: x, y, z, color, time
      - 2D: x, y, color, time

    Supõe que os valores ativos estejam codificados como:
        valor = C * time_base + t
    e que sítios inativos tenham valor <= 0 (0 ou negativo).
    """
    fname = path_dir + filename
    network = read_network(path_dir, filename)  # deve retornar np.ndarray

    # conferência rápida
    print("network.shape:", network.shape)
    valores_unicos, contagens = np.unique(network, return_counts=True)
    print("alguns valores únicos:", valores_unicos[:10])

    # máscara: pega apenas sítios ativos (valor > 0)
    mask_active = network > 0

    # índices dos sítios ativos
    coords = np.argwhere(mask_active)  # (z,y,x) se 3D; (y,x) se 2D

    # valores codificados
    encoded_vals = network[mask_active].astype(np.int64, copy=False)

    # decodificação cor e tempo
    colors = encoded_vals // time_base
    times  = encoded_vals %  time_base

    # mapeando para o sistema físico
    if dim == 3:
        # coords: (z, y, x)
        z = coords[:, 0]
        y = coords[:, 1]
        x = coords[:, 2]

        df_points = pd.DataFrame({
            "x": x,
            "y": y,
            "z": z,
            "color": colors,
            "time": times,
        })
    else:
        # dim == 2: coords: (y, x)
        y = coords[:, 0]
        x = coords[:, 1]

        df_points = pd.DataFrame({
            "x": x,
            "y": y,
            "color": colors,
            "time": times,
        })

    save_out = path_dir + "network_positions_time.csv"
    df_points = df_points.sort_values("time").reset_index(drop=True)
    print(df_points.head())
    print("Total de pontos ativos:", len(df_points))
    df_points.to_csv(save_out, sep=',', index=False)
    print("CSV salvo em:", save_out)

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
    convert_positions_3D(path_dir, filename, dim)
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

for i, time in enumerate(frame_times):
    # Limpa cena ANTES de desenhar o frame atual
    mlab.clf()

    for idx, color in enumerate(colors):
        df_color = df[(df['color'] == color) & (df['time'] <= time)]

        x = df_color['x'].values
        y = df_color['y'].values
        z = df_color['z'].values

        if x.size > 0:
            mlab.points3d(
                x, y, z,
                color=colors_used[idx],
                scale_factor=1,
                mode='cube'
            )

    # Câmera
    mlab.view(azimuth=60, elevation=60, distance=3.25 * L)

    # Salva frame
    path_out_network = os.path.join(folder_with_figures, f"frame_{time:04d}.png")
    mlab.savefig(path_out_network, magnification=2)

    if (i % 10) == 0 or i == len(frame_times) - 1:
        print(f"[{i+1:5d}/{len(frame_times)}]  t = {time}")

# Fecha figura ao final
mlab.close(fig)
print("Frames salvos em:", folder_with_figures)
