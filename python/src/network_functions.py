import numpy as np
import pandas as pd

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

def convert_positions(path_dir, filename, dim):
    fname = path_dir + filename
    network = read_network(path_dir, filename)

    # valores únicos só pra conferência (opcional)
    valores_unicos, contagens = np.unique(network, return_counts=True)
    print(valores_unicos)
    # índices de todos os sítios ativos (valor != -1)
    coords_zyx = np.argwhere(network != -1)  # (z, y, x)

    # valores (cores) correspondentes
    colors = network[network != -1]

    # mapeando para o sistema físico: x,y base; z altura

    if(dim==3):
        x = coords_zyx[:, 2]   # eixo 2 -> x
        y = coords_zyx[:, 1]   # eixo 1 -> y
        z = coords_zyx[:, 0]   # eixo 0 -> z

        df_points = pd.DataFrame({
            "x": x,
            "y": y,
            "z": z,
            "color": colors
        })
    
    else:
        y = coords_zyx[:, 0]
        x = coords_zyx[:, 1]
        df_points = pd.DataFrame({
            "x": x,
            "y": y,
            "color": colors
        })
    save_out = path_dir + "network_positions.csv"

    print(df_points.head())
    print("Total de pontos ativos:", len(df_points))
    df_points.to_csv(save_out, sep=',', index=False)