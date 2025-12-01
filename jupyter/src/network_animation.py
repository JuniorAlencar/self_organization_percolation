import numpy as np
import pandas as pd
import os

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