import numpy as np
import pandas as pd
from mayavi import mlab
import os 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import patches  # coloque no topo do arquivo

def read_network(path_dir, filename, return_metadata=False):
    fn = os.path.join(path_dir, filename)

    with np.load(fn, allow_pickle=True) as npz:
        keys = list(npz.keys())
        print("chaves dentro do arquivo:", keys)

        required = ("dim", "shape", "data")
        missing = [k for k in required if k not in npz]
        if missing:
            raise KeyError(f"Arquivo {fn} sem as chaves obrigatórias: {missing}")

        dim = int(np.asarray(npz["dim"]).item())
        shape = tuple(np.asarray(npz["shape"], dtype=np.int64).tolist())

        # IMPORTANTE:
        # data.npy foi salvo com fortran_order=False e shape=net.shape,
        # mas o buffer linear net.data segue idx = x + SX*(y + SY*z).
        # Então a forma segura é recuperar o buffer linear original:
        raw = np.asarray(npz["data"], dtype=np.int32).ravel(order="C")

        if dim == 2:
            if len(shape) != 2:
                raise ValueError(f"dim=2, mas shape={shape}")

            SX, SY = map(int, shape)

            expected = SX * SY
            if raw.size != expected:
                raise ValueError(
                    f"Tamanho inconsistente em {fn}: raw.size={raw.size}, esperado={expected}"
                )

            # idx = x + SX*y
            # array C-order com shape (SY, SX) dá arr[y, x]
            # depois transpomos para obter network[x, y]
            network = raw.reshape((SY, SX), order="C").T

        elif dim == 3:
            if len(shape) != 3:
                raise ValueError(f"dim=3, mas shape={shape}")

            SX, SY, SZ = map(int, shape)

            expected = SX * SY * SZ
            if raw.size != expected:
                raise ValueError(
                    f"Tamanho inconsistente em {fn}: raw.size={raw.size}, esperado={expected}"
                )

            # idx = x + SX*(y + SY*z)
            # array C-order com shape (SZ, SY, SX) dá arr[z, y, x]
            # depois transpomos para obter network[x, y, z]
            network = raw.reshape((SZ, SY, SX), order="C").transpose(2, 1, 0)

        else:
            raise ValueError(f"dim inválido em {fn}: {dim}")

        metadata = {
            "dim": dim,
            "shape": shape,
            "keys": keys,
        }

        for extra_key in ("num_colors", "seed", "rho"):
            if extra_key in npz:
                metadata[extra_key] = npz[extra_key]

    if return_metadata:
        return network, metadata
    return network

def convert_positions(path_dir, filename, output_filename, dim=None):
    network, meta = read_network(path_dir, filename, return_metadata=True)

    dim_file = meta["dim"]
    if dim is not None and dim != dim_file:
        raise ValueError(
            f"dim informado ({dim}) difere do dim do arquivo ({dim_file})"
        )

    dim = dim_file

    valores_unicos, contagens = np.unique(network, return_counts=True)
    print("valores únicos:", valores_unicos)
    print("contagens:", contagens)

    mask = network > 0
    coords = np.argwhere(mask)
    colors = network[mask]

    if dim == 3:
        # read_network já devolve network[x, y, z]
        df_points = pd.DataFrame({
            "x": coords[:, 0].astype(np.int32),
            "y": coords[:, 1].astype(np.int32),
            "z": coords[:, 2].astype(np.int32),
            "color": colors.astype(np.int16),
        })
    else:
        # read_network já devolve network[x, y]
        df_points = pd.DataFrame({
            "x": coords[:, 0].astype(np.int32),
            "y": coords[:, 1].astype(np.int32),
            "color": colors.astype(np.int16),
        })

    save_out = os.path.join(path_dir, output_filename)
    if not save_out.endswith(".parquet"):
        save_out += ".parquet"

    print(df_points.head())
    print("Total de pontos salvos:", len(df_points))

    df_points.to_parquet(
        save_out,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    print(f"Arquivo salvo em: {save_out}")

def convert_positions_sp(path_dir, filename, output_filename, dim):
    fname = path_dir + filename
    network = read_network(path_dir, filename)  # deve retornar um np.ndarray

    # valores únicos só pra conferência (opcional)
    valores_unicos, contagens = np.unique(network, return_counts=True)
    print("Valores únicos na rede:", dict(zip(valores_unicos, contagens)))

    # índices de todos os sítios NÃO NULOS (valor != 0)
    coords_zyx = np.argwhere(network != 0)  # (z, y, x) em 3D; (y, x) em 2D

    # valores (cores) correspondentes: exatamente o valor da matriz em (i,j,k)
    colors = network[network != 0]

    if dim == 3:
        # mapeando para o sistema físico: x,y base; z altura
        x = coords_zyx[:, 0]   # eixo 2 -> x
        y = coords_zyx[:, 1]   # eixo 1 -> y
        z = coords_zyx[:, 2]   # eixo 0 -> z

        df_points = pd.DataFrame({
            "x": x,
            "y": y,
            "z": z,
            "color": colors
        })
    else:
        # 2D: network esperado como (y, x)
        y = coords_zyx[:, 0]
        x = coords_zyx[:, 1]

        df_points = pd.DataFrame({
            "x": x,
            "y": y,
            "color": colors
        })

    save_out = path_dir + output_filename

    print(df_points.head())
    print("Total de pontos não nulos:", len(df_points))
    df_points.to_csv(save_out, sep=',', index=False)


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


def plot_3D_cut(dim, L, nc, rho, k, NT):
    path_dir = f"../network/{dim}D_L{L}_nc{nc}_rho{rho}_k{k:.1e}_Nt{NT}/"
    
    file_positions = f"../network/{dim}D_L{L}_nc{nc}_rho{rho}_k{k:.1e}_Nt{NT}/network_positions.csv"
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

        mask_oct = (df_color['x'] <= cut) | (df_color['y'] <= cut) | (df_color['z'] <= cut)
        x = df_color['x'][mask_oct]
        y = df_color['y'][mask_oct]
        z = df_color['z'][mask_oct]

        if len(x) > 0:
            pts = mlab.points3d(
            x, y, z,
            color=colors_used[idx],
            scale_factor=1,
            opacity=1.0,
            mode='cube'
        )
        #pts.actor.property.edge_visibility = True
        #pts.actor.property.edge_color = (0, 0, 0)   # preto
        #pts.actor.property.line_width = 0.00         # espessura da borda (ajuste se quiser)


    mlab.outline(
        extent=[0, L, 0, L, 0, L],  # [xmin, xmax, ymin, ymax, zmin, zmax]
        color=(0, 0, 0),            # contorno preto
        line_width=2.0              # espessura da linha
    )

    mlab.view(azimuth=60, elevation=60, distance=3.25*L)
    path_out_network = path_dir + f"L{L}_seed{seed}_{cut}_better.png" 
    mlab.savefig(path_out_network, magnification=4)
    print(f"network save in {path_out_network}")
    mlab.show()

def plot_3D_full(path_dir, file_positions, p0, P0, L, nc, seed, filename,
                 specific_color=None, show_base=False):
    fn = path_dir + file_positions

    if not os.path.exists(fn):
        print("file positions don't exist, create it...")
        convert_positions(
            path_dir,
            filename,
            f"network_positions_p0_{p0:.1f}_P0_{P0:.2f}.parquet",
            3
        )
        print("File with positions created")

    a = 0
    b = 0

    df = pd.read_parquet(fn).copy()

    # garante inteiros e periodicidade transversal
    df["x"] = ((df["x"].astype(int) + a) % L).astype(int)
    df["y"] = ((df["y"].astype(int) + b) % L).astype(int)
    df["z"] = df["z"].astype(int)
    df["color"] = df["color"].astype(int)

    # mantém apenas sítios ativos
    df = df[df["color"] > 0].copy()

    figure_size = (800, 800)

    # ========= mudança importante =========
    # em vez de mlab.clf() solto ou mlab.close(all=False),
    # cria/reutiliza uma figura nomeada e limpa essa figura
    fig = mlab.figure(
        figure="network3d",
        size=figure_size,
        bgcolor=(1, 1, 1),
        fgcolor=(0, 0, 0)
    )
    mlab.clf(figure=fig)
    # =====================================

    colors = [i + 2 for i in range(nc)]

    colors_used = [
        (0.9, 0.1, 0.1),    # 2 - red
        (1.0, 0.5, 0.0),    # 3 - orange
        (0.1, 0.9, 0.1),    # 4 - green
        (0.1, 0.1, 0.9),    # 5 - blue
        (0.8, 0.2, 0.8),    # 6 - purple
        (0.2, 0.8, 0.8),    # 7 - teal
        (1.0, 1.0, 0.0),    # 8 - yellow
        (0.6, 0.4, 0.2),    # 9 - brown
        (0.0, 0.0, 0.0),    # 10 - black
        (0.65, 0.65, 0.65)  # 11 - gray
    ]

    if specific_color is None:
        for idx, color in enumerate(colors):
            df_color = df[df["color"] == color]

            if df_color.empty:
                continue

            x = df_color["x"].to_numpy()
            y = df_color["y"].to_numpy()
            z = df_color["z"].to_numpy()

            pts = mlab.points3d(
                x, y, z,
                np.ones_like(x),
                color=colors_used[idx],
                scale_factor=1.0,
                opacity=1.0,
                mode="cube"
            )
            pts.actor.property.edge_visibility = True
            pts.actor.property.edge_color = (0, 0, 0)
            pts.actor.property.line_width = 0.00

        if show_base:
            path_out_network = path_dir + f"L{L}_seed{seed}_p0_{p0:.1f}_P0_{P0:.2f}base.png"
        else:
            path_out_network = path_dir + f"L{L}_seed{seed}_p0_{p0:.1f}_P0_{P0:.2f}all.png"

    else:
        if specific_color not in colors:
            print(f"color not accept, please enter with any color in list {colors}")
            return

        idx = colors.index(specific_color)
        df_color = df[df["color"] == specific_color]

        if df_color.empty:
            print(f"No points found for color {specific_color}")
            return

        x = df_color["x"].to_numpy()
        y = df_color["y"].to_numpy()
        z = df_color["z"].to_numpy()

        pts = mlab.points3d(
            x, y, z,
            np.ones_like(x),
            color=colors_used[idx],
            scale_factor=1.0,
            opacity=1.0,
            mode="cube"
        )
        pts.actor.property.edge_visibility = True
        pts.actor.property.edge_color = (0, 0, 0)
        pts.actor.property.line_width = 0.00

        if show_base:
            path_out_network = path_dir + f"L{L}_seed{seed}_color_{specific_color}_base.png"
        else:
            path_out_network = path_dir + f"L{L}_seed{seed}_color_{specific_color}.png"

    mlab.outline(
        extent=[0, L, 0, L, 0, L],
        color=(0, 0, 0),
        line_width=2.0
    )

    # ========= mantém EXATAMENTE o ângulo original =========
    if show_base:
        mlab.view(
            azimuth=0,
            elevation=-90,
            distance=2.8 * L,
            focalpoint=(L / 2, L / 2, 0)
        )
    else:
        mlab.view(
            azimuth=70,
            elevation=65,
            distance=3.1 * L,
            focalpoint=(L / 2, L / 2, L / 2)
        )
    # ======================================================

    fig.scene.render()
    
    mlab.savefig(path_out_network, magnification=4)
    print(f"network save in {path_out_network}")
    mlab.close()

def plot_3D_full_with_planes(path_dir, file_positions, specific_color=None):
    # Create file positions, if dont exist
    if not os.path.exists(file_positions):
        print("file positions don't exist, create it...")
        convert_positions(path_dir, "P0_0.10_p0_1.00_seed_1.npz", "network_positions_.csv", dim)
        print("File with positions created")

    a = 0
    b = 0
    seed = 1

    df = pd.read_csv(file_positions)

    cut = L * (1/2)

    # Ajuste as coordenadas x e y para levar em conta a e b
    df['x'] = (df['x'] + a) % L
    df['y'] = (df['y'] + b) % L

    figure_size = (800, 800)
    mlab.figure(size=figure_size, bgcolor=(1, 1, 1))

    colors = [i + 2 for i in range(nc)]
    
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

    if specific_color is None:
        for idx, color in enumerate(colors):
            df_color = df[df['color'] == color]

            x = df_color['x']
            y = df_color['y']
            z = df_color['z']
            
            if len(x) > 0:
                pts = mlab.points3d(
                    x, y, z,
                    color=colors_used[idx],
                    scale_factor=1,
                    opacity=1.0,   # cubos totalmente opacos
                    mode='cube'
                )
                # Ativar contorno preto nos cubos
                pts.actor.property.edge_visibility = True
                pts.actor.property.edge_color = (0, 0, 0)   # preto
                pts.actor.property.line_width = 0.00        # espessura da borda
        
        path_out_network = path_dir + f"L{L}_seed{seed}_all_planes.png"
    
    else:
        if specific_color not in colors:
            print(f"color not accept, please enter with any color in list {colors}")
        else:
            df_color = df[df['color'] == specific_color]
            x = df_color['x']
            y = df_color['y']
            z = df_color['z']

            if len(x) > 0:
                # índice da cor correta no vetor colors_used
                color_idx = colors.index(specific_color)
                pts = mlab.points3d(
                    x, y, z,
                    color=colors_used[color_idx],
                    scale_factor=1,
                    opacity=1.0,
                    mode='cube'
                )
                # Ativar contorno preto nos cubos
                pts.actor.property.edge_visibility = True
                pts.actor.property.edge_color = (0, 0, 0)   # preto
                pts.actor.property.line_width = 0.00        # espessura da borda

                prop = pts.actor.property
                prop.ambient = 0.2       # componente ambiente (luz geral)
                prop.diffuse = 0.9       # quanto o objeto responde à luz difusa
                prop.specular = 0.3      # brilho especular (reflexo)
                prop.specular_power = 20 # quão concentrado é esse brilho

            path_out_network = path_dir + f"L{L}_seed{seed}_color_{specific_color}_planes.png"

    # ------------------------------------------------------
    # Planos de corte em z = 25, 125, 225 atravessando o cubo
    # ------------------------------------------------------
    z_planes = [25, 125, 225]
    margin = 0.1 * L  # quanto o plano "sai" para fora da caixa

    for z0 in z_planes:
        if 0 <= z0 <= L:
            # plano maior que o cubo: [-margin, L+margin] em x e y
            x_plane, y_plane = np.mgrid[
                -margin : L + margin : 2j,
                -margin : L + margin : 2j
            ]
            z_plane = np.full_like(x_plane, float(z0))

            mlab.mesh(
                x_plane, y_plane, z_plane,
                color=(0.3, 0.3, 0.3),   # cinza
                opacity=0.3              # semi-transparente
            )

    # Caixa de contorno do sistema
    mlab.outline(
        extent=[0, L, 0, L, 0, L],  # [xmin, xmax, ymin, ymax, zmin, zmax]
        color=(0, 0, 0),            # contorno preto
        line_width=2.0              # espessura da linha
    )

    center = (L / 2, L / 2, L / 2)

    mlab.view(
        azimuth=50,
        elevation=60,
        distance=3.50 * L,
        focalpoint=center
    )

    mlab.savefig(path_out_network, magnification=4)
    print(f"network save in {path_out_network}")
    mlab.show()




TIME_BASE_3D = 100_000_000  # fator da codificação: C * 100000000 + t

def convert_positions_animation(path_dir, filename, dim, time_base=TIME_BASE_3D):
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


from matplotlib import patches  # coloque no topo do arquivo

def plot_projection(path_dir, file_positions, L, P0, seed, filename, z_level):
    fn = path_dir + file_positions

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
    if not os.path.exists(fn):
        print("file positions don't exist, create it...")
        convert_positions(path_dir, filename, f"network_positions_p0_{p0}_P0_{P0:.2f}.parquet", 3)
        print("File with positions created")

    df = pd.read_parquet(fn)
    df_sub = df[df["z"] == z_level].copy()

    a, b = 0, 0
    df_sub.loc[:, "x"] = ((df_sub["x"] + a) % L).astype(int)
    df_sub.loc[:, "y"] = ((df_sub["y"] + b) % L).astype(int)

    color_values = sorted(df_sub["color"].unique())

    img = np.zeros((L, L), dtype=np.uint8)

    for j, c in enumerate(color_values):
        idx = j + 1
        mask = (df_sub["color"] == c)
        xs = df_sub.loc[mask, "x"].to_numpy()
        ys = df_sub.loc[mask, "y"].to_numpy()
        img[xs, ys] = idx

    cmap = ListedColormap([(1.0, 1.0, 1.0)] + colors_used[:len(color_values)])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.imshow(
        img.T,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=len(color_values)
    )

    rect = patches.Rectangle(
        (-0.5, -0.5),
        L,
        L,
        linewidth=1.5,
        edgecolor="black",
        facecolor="none"
    )
    ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(
        path_dir + f"L{L}_seed{seed}_P0_{P0:.2f}_projectionz_{z_level}.png",
        bbox_inches="tight",
        pad_inches=0
    )
    plt.show()

    
