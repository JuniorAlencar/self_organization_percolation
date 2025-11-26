import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from src.network_functions import convert_positions

# Parâmetros
L   = 2048
dim = 2
nc  = 3
rho = 0.33
k   = 1.0e-05
NT  = 400

path_dir = f"../network/{dim}D_L{L}_nc{nc}_rho{rho}_k{k:.1e}_Nt{NT}/"
file_positions = f"../network/{dim}D_L{L}_nc{nc}_rho{rho}_k{k:.1e}_Nt{NT}/" + "network_positions.csv" 
seed = 1

# Create file positions, if dont exist
if not os.path.exists(file_positions):
    print("file positions don't exist, create it...")
    convert_positions(path_dir, "P0_0.10_p0_1.00_seed_1.npz", dim)
    print("File with positions created")

# Load positions
df = pd.read_csv(file_positions)

# Se você usa PBC no plano:
a, b = 0, 0
df["x"] = (df["x"] + a) % L
df["y"] = (df["y"] + b) % L

# Garante que são inteiros (se o seu código salva assim)
df["x"] = df["x"].astype(int)
df["y"] = df["y"].astype(int)

# Valores de cor presentes (ex.: [2,3,4,...])
color_values = sorted(df["color"].unique())

# Paleta de cores (ajuste se tiver mais cores)
colors_used = [
    (0.9, 0.1, 0.1),  # vermelho
    (1.0, 0.5, 0.0),  # laranja
    (0.1, 0.9, 0.1),  # verde
    (0.1, 0.1, 0.9),  # azul
    (0.8, 0.2, 0.8),  # roxo
    (0.2, 0.8, 0.8),  # ciano
    (1.0, 1.0, 0.0),  # amarelo
]

# Cria matriz da imagem:
# 0 = fundo branco; 1..N = cores
img = np.zeros((L, L), dtype=np.uint8)

for j, c in enumerate(color_values):
    idx = j + 1  # começa em 1
    mask = (df["color"] == c)
    xs = df.loc[mask, "x"].to_numpy()
    ys = df.loc[mask, "y"].to_numpy()

    # Se quiser y crescente para cima: origin='lower' no imshow
    img[ys, xs] = idx

# Colormap: fundo branco + cores usadas
cmap = ListedColormap([(1.0, 1.0, 1.0)] + colors_used[:len(color_values)])

# Plot
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

im = ax.imshow(
    img,
    origin="lower",      # y=0 embaixo
    cmap=cmap,
    interpolation="nearest"
)

ax.set_axis_off()
plt.tight_layout(pad=0)

out = path_dir + f"L{L}_seed{seed}_2D_imshow.png"
plt.savefig(out, bbox_inches="tight", pad_inches=0)
plt.close(fig)
print(f"Figure saved in {out}")