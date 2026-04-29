import numpy as np
import pandas as pd
from mayavi import mlab
import os 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import patches  # coloque no topo do arquivo


def create_folder(folder_path):
    """
    Creates the folder if it does not already exist.

    Args:
        folder_path (str): Path to the folder to be created.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
        
    else:
        print(f"Folder already exists: {folder_path}")


TIME_BASE_3D = 10_000_000

def _get_colors_used():
    return [
        (0.9, 0.1, 0.1),    # 2 - red
        (1.0, 0.5, 0.0),    # 3 - orange
        (0.1, 0.9, 0.1),    # 4 - green
        (0.1, 0.1, 0.9),    # 5 - blue
        (0.8, 0.2, 0.8),    # 6 - purple
        (0.2, 0.8, 0.8),    # 7 - teal
        (1.0, 1.0, 0.0),    # 8 - yellow
        (0.6, 0.4, 0.2),    # 9 - brown
        (0.0, 0.0, 0.0),    # 10 - black
        (0.65, 0.65, 0.65), # 11 - gray
    ]

def _build_fixed_color_map(unique_colors, nc):
    """
    Mantém o mesmo padrão visual da rede completa.

    Casos suportados:
    - cores 1..nc
    - cores 2..nc+1

    Se vier outro esquema, cai num fallback controlado.
    """
    colors_used = _get_colors_used()
    unique_colors = sorted(int(c) for c in unique_colors)

    if not unique_colors:
        return {}

    cmin = min(unique_colors)
    cmax = max(unique_colors)

    # Caso 1: labels 1..nc
    if cmin >= 1 and cmax <= nc:
        return {c: colors_used[c - 1] for c in unique_colors}

    # Caso 2: labels 2..nc+1
    if cmin >= 2 and cmax <= nc + 1:
        return {c: colors_used[c - 2] for c in unique_colors}

    # Fallback: mantém ordem crescente, mas avisa
    if len(unique_colors) > len(colors_used):
        raise ValueError(
            f"Há {len(unique_colors)} cores no arquivo, mas apenas "
            f"{len(colors_used)} cores foram definidas na paleta."
        )

    print(
        "[WARN] Esquema de cores fora do padrão esperado. "
        "Aplicando fallback por ordem crescente."
    )
    return {c: colors_used[i] for i, c in enumerate(unique_colors)}

def _darken_rgb(rgb, factor=0.82):
    return tuple(max(0.0, min(1.0, c * factor)) for c in rgb)

def _apply_full_cube_style(pts, edge_width=1.4):
    prop = pts.actor.property
    prop.edge_visibility = True
    prop.edge_color = (0, 0, 0)
    prop.line_width = edge_width

    prop.opacity = 1.0
    prop.ambient = 0.10
    prop.diffuse = 0.90
    prop.specular = 0.0

def _draw_points3d_cube_cloud(fig, x, y, z, rgb, darken_factor=1.0, edge_width=1.0):
    rgb_use = _darken_rgb(rgb, darken_factor)

    pts = mlab.points3d(
        x, y, z,
        np.ones_like(x),
        color=rgb_use,
        scale_factor=1.0,
        opacity=1.0,
        mode="cube",
        figure=fig
    )
    _apply_full_cube_style(pts, edge_width=edge_width)
    return pts

def _read_positions_table(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".parquet":
        return pd.read_parquet(file_path)
    if ext == ".csv":
        return pd.read_csv(file_path)

    raise ValueError(f"Formato não suportado para posições: {file_path}")


def _write_positions_table(df, file_path):
    out_dir = os.path.dirname(file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".parquet":
        df.to_parquet(file_path, index=False)
    elif ext == ".csv":
        df.to_csv(file_path, index=False)
    else:
        raise ValueError(
            f"output_data deve terminar com .parquet ou .csv. Recebido: {file_path}"
        )


def _read_codec_metadata(path_dir, filename):
    fn = os.path.join(path_dir, filename)

    with np.load(fn, allow_pickle=False) as npz:
        meta = {
            "dim": int(np.asarray(npz["dim"]).item()),
            "shape": tuple(np.asarray(npz["shape"], dtype=np.int64).tolist()),
            "keys": list(npz.keys()),
        }

        for extra_key in ("num_colors", "seed", "rho"):
            if extra_key in npz:
                meta[extra_key] = npz[extra_key]

    return meta

def _camera_plot_3D_full(fig, L, show_base=False):
    if show_base:
        mlab.view(
            azimuth=0,
            elevation=-90,
            distance=2.8 * L,
            focalpoint=(L / 2, L / 2, 0),
            figure=fig
        )
    else:
        mlab.view(
            azimuth=70,
            elevation=65,
            distance=3.1 * L,
            focalpoint=(L / 2, L / 2, L / 2),
            figure=fig
        )


def _new_figure_3d(figure_name="network3d"):
    fig = mlab.figure(
        figure=figure_name,
        size=(800, 800),
        bgcolor=(1, 1, 1),
        fgcolor=(0, 0, 0),
    )
    mlab.clf(figure=fig)
    return fig


def read_network_codec(path_dir, filename, fill_value=0):
    """
    Lê arquivo .npz codificado, aceitando:
      1) formato denso:  dim, shape, data
      2) formato esparso: dim, shape, active_idx, active_val

    Convenção inferida do struct C++:
      shape[0] = Ny
      shape[1] = Nx
      shape[2] = Nz

    Retorna:
      network[x, y, z]
    """
    fn = os.path.join(path_dir, filename)

    with np.load(fn, allow_pickle=False) as npz:
        keys = list(npz.keys())
        dim = int(np.asarray(npz["dim"]).item())
        shape = tuple(np.asarray(npz["shape"], dtype=np.int64).tolist())

        metadata = {
            "dim": dim,
            "shape": shape,
            "keys": keys,
        }

        for extra_key in ("num_colors", "seed", "rho"):
            if extra_key in npz:
                metadata[extra_key] = npz[extra_key]

        if dim == 2:
            if len(shape) != 2:
                raise ValueError(f"dim=2, mas shape={shape}")

            Ny, Nx = map(int, shape)
            expected = Ny * Nx

            if "data" in npz:
                raw = np.asarray(npz["data"], dtype=np.int64).ravel(order="C")
            elif "active_idx" in npz and "active_val" in npz:
                raw = np.full(expected, fill_value, dtype=np.int64)
                active_idx = np.asarray(npz["active_idx"], dtype=np.int64)
                active_val = np.asarray(npz["active_val"], dtype=np.int64)
                raw[active_idx] = active_val
            else:
                raise KeyError(
                    f"Arquivo {fn} não possui nem 'data' nem ('active_idx', 'active_val'). "
                    f"Chaves encontradas: {keys}"
                )

            if raw.size != expected:
                raise ValueError(
                    f"Tamanho inconsistente em {fn}: raw.size={raw.size}, esperado={expected}"
                )

            # raw[y, x] -> network[x, y]
            network = raw.reshape((Ny, Nx), order="C").T

        elif dim == 3:
            if len(shape) != 3:
                raise ValueError(f"dim=3, mas shape={shape}")

            Ny, Nx, Nz = map(int, shape)
            expected = Ny * Nx * Nz

            if "data" in npz:
                raw = np.asarray(npz["data"], dtype=np.int64).ravel(order="C")
            elif "active_idx" in npz and "active_val" in npz:
                raw = np.full(expected, fill_value, dtype=np.int64)
                active_idx = np.asarray(npz["active_idx"], dtype=np.int64)
                active_val = np.asarray(npz["active_val"], dtype=np.int64)
                raw[active_idx] = active_val
            else:
                raise KeyError(
                    f"Arquivo {fn} não possui nem 'data' nem ('active_idx', 'active_val'). "
                    f"Chaves encontradas: {keys}"
                )

            if raw.size != expected:
                raise ValueError(
                    f"Tamanho inconsistente em {fn}: raw.size={raw.size}, esperado={expected}"
                )

            # raw[z, y, x] -> network[x, y, z]
            network = raw.reshape((Nz, Ny, Nx), order="C").transpose(2, 1, 0)

        else:
            raise ValueError(f"dim inválido em {fn}: {dim}")

    return network, metadata


TIME_BASE_3D = 10_000_000

def positions_from_codec_npz(path_dir, filename, output_data=None, time_base=None):
    """
    Lê o .npz codificado e devolve diretamente um DataFrame com
    x, y, z, color, time.

    Caso esparso:
        usa active_idx / active_val e decodifica x,y,z diretamente
        do índice linear antigo:
            idx = x + SX * (y + SY * z)

    Caso denso:
        reconstrói como no read_network antigo.

    Se output_data for informado, salva o dataframe final em .parquet ou .csv.
    """
    fn = os.path.join(path_dir, filename)

    with np.load(fn, allow_pickle=False) as npz:
        keys = list(npz.keys())
        dim = int(np.asarray(npz["dim"]).item())
        shape = tuple(np.asarray(npz["shape"], dtype=np.int64).tolist())

        meta = {
            "dim": dim,
            "shape": shape,
            "keys": keys,
        }
        for extra_key in ("num_colors", "seed", "rho"):
            if extra_key in npz:
                meta[extra_key] = npz[extra_key]

        if dim != 3:
            raise ValueError("Esta função foi escrita para o caso 3D.")

        if time_base is None:
            if "active_val" in npz:
                vmax = int(np.asarray(npz["active_val"]).max())
            elif "data" in npz:
                vmax = int(np.asarray(npz["data"]).max())
            else:
                raise KeyError("Arquivo sem 'data' e sem 'active_val'.")

            if vmax >= 100_000_000:
                time_base = 100_000_000
            elif vmax >= 10_000_000:
                time_base = 10_000_000
            else:
                raise ValueError(f"Não foi possível inferir time_base. vmax={vmax}")

        if "active_idx" in npz and "active_val" in npz:
            active_idx = np.asarray(npz["active_idx"], dtype=np.int64)
            encoded_vals = np.asarray(npz["active_val"], dtype=np.int64)

            SX, SY, SZ = map(int, shape)

            x = active_idx % SX
            y = (active_idx // SX) % SY
            z = active_idx // (SX * SY)

            colors = encoded_vals // time_base
            times = encoded_vals % time_base

            df = pd.DataFrame({
                "x": x.astype(np.int32),
                "y": y.astype(np.int32),
                "z": z.astype(np.int32),
                "color": colors.astype(np.int32),
                "time": times.astype(np.int64),
            })

        elif "data" in npz:
            raw = np.asarray(npz["data"], dtype=np.int64).ravel(order="C")
            SX, SY, SZ = map(int, shape)

            expected = SX * SY * SZ
            if raw.size != expected:
                raise ValueError(
                    f"Tamanho inconsistente em {fn}: raw.size={raw.size}, esperado={expected}"
                )

            network = raw.reshape((SZ, SY, SX), order="C").transpose(2, 1, 0)

            mask_active = network > 0
            coords = np.argwhere(mask_active)
            encoded_vals = network[mask_active].astype(np.int64, copy=False)

            colors = encoded_vals // time_base
            times = encoded_vals % time_base

            df = pd.DataFrame({
                "x": coords[:, 0].astype(np.int32),
                "y": coords[:, 1].astype(np.int32),
                "z": coords[:, 2].astype(np.int32),
                "color": colors.astype(np.int32),
                "time": times.astype(np.int64),
            })

        else:
            raise KeyError(
                f"Arquivo {fn} não possui nem ('active_idx','active_val') nem 'data'. "
                f"Chaves encontradas: {keys}"
            )

    df = df[df["color"] > 0].copy()
    df = df.sort_values("time").reset_index(drop=True)

    if output_data is not None:
        _write_positions_table(df, output_data)

    print(f"time_base usado = {time_base}")
    print(f"cores únicas decodificadas = {np.unique(df['color'])}")
    print(df[["x", "y", "z"]].agg(["min", "max"]))

    return df, meta

def load_or_create_positions_codec(path_dir, filename, output_data, time_base=None, force_rebuild=False):
    """
    Se output_data existir, lê e retorna.
    Se não existir, converte do .npz, salva e retorna.
    """
    if (not force_rebuild) and os.path.exists(output_data):
        df = _read_positions_table(output_data)
        meta = _read_codec_metadata(path_dir, filename)
        return df, meta

    return positions_from_codec_npz(
        path_dir=path_dir,
        filename=filename,
        output_data=output_data,
        time_base=time_base
    )

def _plot_points_df_same_style(
    df,
    L,
    nc,
    path_out,
    figure_name=None,
    specific_color=None,
    show_base=False,
    outline_mode="full",
    visual_profile="full"
):
    df = df.copy()

    for col in ("x", "y", "z", "color"):
        if col not in df.columns:
            raise ValueError(f"DataFrame sem coluna obrigatória: {col}")

    df["x"] = df["x"].astype(int)
    df["y"] = df["y"].astype(int)
    df["z"] = df["z"].astype(int)
    df["color"] = df["color"].astype(int)

    df = df[df["color"] > 0].copy()
    if df.empty:
        raise ValueError("Nenhum ponto ativo para plotar.")

    if figure_name is None:
        figure_name = os.path.splitext(os.path.basename(path_out))[0]

    fig = _new_figure_3d(figure_name=figure_name)

    unique_colors = sorted(df["color"].unique().tolist())
    color_map = _build_fixed_color_map(unique_colors, nc)

    if visual_profile == "full":
        darken_factor = 0.90
        edge_width = 0.4
    elif visual_profile == "cut":
        darken_factor = 0.90
        edge_width = 0.5
    else:
        raise ValueError("visual_profile deve ser 'full' ou 'cut'.")

    if specific_color is None:
        colors_to_plot = unique_colors
    else:
        if specific_color not in unique_colors:
            raise ValueError(
                f"Cor {specific_color} não encontrada. Disponíveis: {unique_colors}"
            )
        colors_to_plot = [specific_color]

    for color in colors_to_plot:
        df_color = df[df["color"] == color]
        if df_color.empty:
            continue

        x = df_color["x"].to_numpy()
        y = df_color["y"].to_numpy()
        z = df_color["z"].to_numpy()

        _draw_points3d_cube_cloud(
            fig=fig,
            x=x,
            y=y,
            z=z,
            rgb=color_map[color],
            darken_factor=darken_factor,
            edge_width=edge_width
        )

    if outline_mode == "full":
        extent = [0, L, 0, L, 0, L]
        focal = (L / 2, L / 2, L / 2 if not show_base else 0)
        Lref = L
    elif outline_mode == "tight":
        xmin, xmax = df["x"].min(), df["x"].max() + 1
        ymin, ymax = df["y"].min(), df["y"].max() + 1
        zmin, zmax = df["z"].min(), df["z"].max() + 1
        extent = [xmin, xmax, ymin, ymax, zmin, zmax]
        focal = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
        Lref = max(xmax - xmin, ymax - ymin, zmax - zmin)
    else:
        raise ValueError("outline_mode deve ser 'full' ou 'tight'.")

    mlab.outline(
        extent=extent,
        color=(0, 0, 0),
        line_width=2.0,
        figure=fig
    )

    if show_base:
        mlab.view(
            azimuth=0,
            elevation=-90,
            distance=2.8 * Lref,
            focalpoint=(focal[0], focal[1], extent[4]),
            figure=fig
        )
    else:
        mlab.view(
            azimuth=70,
            elevation=65,
            distance=3.1 * Lref,
            focalpoint=focal,
            figure=fig
        )

    fig.scene.render()
    mlab.savefig(path_out, magnification=4, figure=fig)
    print(f"network save in {path_out}")
    mlab.close(fig)

def plot_3D_full_codec(path_dir, filename, path_out, figure_name, L, nc, seed=None,
                       time_base=TIME_BASE_3D,
                       specific_color=None,
                       show_base=False,
                       save_name=None,
                       positions_file=None,
                       force_rebuild_positions=False):
    """
    Plota a rede original codificada.
    Usa um arquivo de posições cacheado (.parquet ou .csv) se ele existir.
    """
    out_dir = os.path.dirname(path_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if positions_file is None:
        base = os.path.splitext(filename)[0]
        positions_file = os.path.join(out_dir, f"{base}_positions.parquet")

    df, meta = load_or_create_positions_codec(
        path_dir=path_dir,
        filename=filename,
        output_data=positions_file,
        time_base=time_base,
        force_rebuild=force_rebuild_positions
    )

    if seed is None:
        seed = int(meta.get("seed", -1))

    _plot_points_df_same_style(
        df=df,
        L=L,
        nc=nc,
        path_out=path_out,
        specific_color=specific_color,
        show_base=show_base,
        outline_mode="full",
        figure_name=figure_name,
        visual_profile="full"
    )

def _read_plain_npz_metadata(path_dir, filename):
    fn = os.path.join(path_dir, filename)

    with np.load(fn, allow_pickle=False) as npz:
        meta = {
            "dim": int(np.asarray(npz["dim"]).item()),
            "shape": tuple(np.asarray(npz["shape"], dtype=np.int64).tolist()),
            "keys": list(npz.keys()),
        }

        for extra_key in ("num_colors", "seed", "rho"):
            if extra_key in npz:
                meta[extra_key] = npz[extra_key]

    return meta

def positions_from_plain_npz(path_dir, filename, output_data=None):
    """
    Lê um .npz NÃO codificado e devolve um DataFrame com:
        x, y, z, color

    Aceita:
      1) denso:  dim, shape, data
      2) esparso: dim, shape, active_idx, active_val

    Assume o índice linear:
        idx = x + SX * (y + SY * z)
    """
    fn = os.path.join(path_dir, filename)

    with np.load(fn, allow_pickle=False) as npz:
        keys = list(npz.keys())
        dim = int(np.asarray(npz["dim"]).item())
        shape = tuple(np.asarray(npz["shape"], dtype=np.int64).tolist())

        meta = {
            "dim": dim,
            "shape": shape,
            "keys": keys,
        }
        for extra_key in ("num_colors", "seed", "rho"):
            if extra_key in npz:
                meta[extra_key] = npz[extra_key]

        if dim != 3:
            raise ValueError("Esta função foi escrita para o caso 3D.")

        # ----------------------------------
        # CASO ESPARSO: active_idx/active_val
        # ----------------------------------
        if "active_idx" in npz and "active_val" in npz:
            active_idx = np.asarray(npz["active_idx"], dtype=np.int64)
            active_val = np.asarray(npz["active_val"], dtype=np.int64)

            SX, SY, SZ = map(int, shape)

            x = active_idx % SX
            y = (active_idx // SX) % SY
            z = active_idx // (SX * SY)

            df = pd.DataFrame({
                "x": x.astype(np.int32),
                "y": y.astype(np.int32),
                "z": z.astype(np.int32),
                "color": active_val.astype(np.int32),
            })

        # ---------------------------
        # CASO DENSO: data completo
        # ---------------------------
        elif "data" in npz:
            raw = np.asarray(npz["data"], dtype=np.int64).ravel(order="C")

            SX, SY, SZ = map(int, shape)
            expected = SX * SY * SZ

            if raw.size != expected:
                raise ValueError(
                    f"Tamanho inconsistente em {fn}: raw.size={raw.size}, esperado={expected}"
                )

            # mesma lógica da read_network antiga:
            # raw[z, y, x] -> network[x, y, z]
            network = raw.reshape((SZ, SY, SX), order="C").transpose(2, 1, 0)

            mask_active = network > 0
            coords = np.argwhere(mask_active)
            colors = network[mask_active]

            df = pd.DataFrame({
                "x": coords[:, 0].astype(np.int32),
                "y": coords[:, 1].astype(np.int32),
                "z": coords[:, 2].astype(np.int32),
                "color": colors.astype(np.int32),
            })

        else:
            raise KeyError(
                f"Arquivo {fn} não possui nem ('active_idx','active_val') nem 'data'. "
                f"Chaves encontradas: {keys}"
            )

    df = df[df["color"] > 0].copy()
    df = df.sort_values(["z", "y", "x"]).reset_index(drop=True)

    if output_data is not None:
        _write_positions_table(df, output_data)

    print(f"cores únicas = {np.unique(df['color'])}")
    print(df[["x", "y", "z"]].agg(["min", "max"]))

    return df, meta

def load_or_create_positions_plain(path_dir, filename, output_data, force_rebuild=False):
    """
    Se output_data existir, lê e retorna.
    Se não existir, converte do .npz, salva e retorna.
    """
    if (not force_rebuild) and os.path.exists(output_data):
        df = _read_positions_table(output_data)
        meta = _read_plain_npz_metadata(path_dir, filename)
        return df, meta

    return positions_from_plain_npz(
        path_dir=path_dir,
        filename=filename,
        output_data=output_data
    )

def plot_3D_plain_npz(path_dir, filename, path_out, figure_name, L, nc,
                      positions_file=None,
                      seed=None,
                      specific_color=None,
                      show_base=False,
                      outline_mode="tight",
                      force_rebuild_positions=False):
    """
    Plota uma rede .npz NÃO codificada.
    Cria o arquivo de posições se ele não existir.
    """
    out_dir = os.path.dirname(path_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if positions_file is None:
        base = os.path.splitext(filename)[0]
        positions_file = os.path.join(out_dir, f"{base}_positions.parquet")

    df, meta = load_or_create_positions_plain(
        path_dir=path_dir,
        filename=filename,
        output_data=positions_file,
        force_rebuild=force_rebuild_positions
    )

    if seed is None:
        seed = int(meta.get("seed", -1))

    _plot_points_df_same_style(
        df=df,
        L=L,
        nc=nc,
        path_out=path_out,
        specific_color=specific_color,
        show_base=show_base,
        outline_mode=outline_mode,
        figure_name=figure_name
    )

def plot_3D_preteq_posteq(path_dir_pre, filename_pre,
                          path_dir_post, filename_post,
                          L, nc,
                          path_out_pre, path_out_post,
                          figure_name_pre=None, figure_name_post=None,
                          positions_file_pre=None, positions_file_post=None,
                          specific_color=None,
                          show_base=False,
                          outline_mode="tight",
                          force_rebuild_positions=False):
    """
    Plota os recortes preteq e posteq a partir dos .npz NÃO codificados.

    Para cada um:
      - se o parquet de posições existir -> lê
      - se não existir -> cria a partir do .npz e então lê/usa
    """
    if figure_name_pre is None:
        figure_name_pre = "network3d_preteq"

    if figure_name_post is None:
        figure_name_post = "network3d_posteq"

    out_dir_pre = os.path.dirname(path_out_pre)
    if out_dir_pre:
        os.makedirs(out_dir_pre, exist_ok=True)

    out_dir_post = os.path.dirname(path_out_post)
    if out_dir_post:
        os.makedirs(out_dir_post, exist_ok=True)

    if positions_file_pre is None:
        base_pre = os.path.splitext(filename_pre)[0]
        positions_file_pre = os.path.join(out_dir_pre, f"{base_pre}_positions.parquet")

    if positions_file_post is None:
        base_post = os.path.splitext(filename_post)[0]
        positions_file_post = os.path.join(out_dir_post, f"{base_post}_positions.parquet")

    df_pre, _ = load_or_create_positions_plain(
        path_dir=path_dir_pre,
        filename=filename_pre,
        output_data=positions_file_pre,
        force_rebuild=force_rebuild_positions
    )

    df_post, _ = load_or_create_positions_plain(
        path_dir=path_dir_post,
        filename=filename_post,
        output_data=positions_file_post,
        force_rebuild=force_rebuild_positions
    )

    _plot_points_df_same_style(
    df=df_pre,
    L=L,
    nc=nc,
    path_out=path_out_pre,
    specific_color=specific_color,
    show_base=show_base,
    outline_mode=outline_mode,
    figure_name=figure_name_pre,
    visual_profile="cut"
    )

    _plot_points_df_same_style(
        df=df_post,
        L=L,
        nc=nc,
        path_out=path_out_post,
        specific_color=specific_color,
        show_base=show_base,
        outline_mode=outline_mode,
        figure_name=figure_name_post,
        visual_profile="cut"
    )


def check_codification(arquivo):
    with np.load(arquivo, allow_pickle=False) as npz:
        print("chaves:", npz.files)

        if "active_val" in npz:
            vals = npz["active_val"]
        elif "data" in npz:
            vals = npz["data"].ravel()
        else:
            print("Arquivo sem active_val e sem data.")
            return

        vals = vals[vals > 0]
        print("primeiros valores positivos:", vals[:20])

        if len(vals) == 0:
            print("Sem valores ativos.")
        elif vals.max() >= 10_000_000:
            print("Rede codificada.")
        else:
            print("Rede não codificada.")

def plot_3D_full_codec_by_species(path_dir, filename, path_out_dir, figure_name,
                                  L, nc, seed=None,
                                  time_base=TIME_BASE_3D,
                                  show_base=False,
                                  positions_file=None,
                                  force_rebuild_positions=False,
                                  colors_to_plot=None,
                                  outline_mode="full",
                                  visual_profile="full",
                                  filename_prefix=None):
    """
    Plota a rede codificada separando uma figura para cada espécie/cor.

    Uso típico:
        - Se o arquivo de entrada já contém apenas os clusters percolantes,
          cada figura mostrará o cluster percolante daquela espécie.
        - Se o arquivo contém a rede completa, cada figura mostrará toda a
          espécie correspondente.

    Args:
        path_dir: pasta onde está o .npz codificado.
        filename: nome do arquivo .npz codificado.
        path_out_dir: pasta onde as figuras serão salvas.
        figure_name: nome-base da figura Mayavi.
        L: tamanho linear da rede.
        nc: número de cores/espécies.
        seed: opcional.
        time_base: base usada na codificação cor/tempo.
        show_base: se True, câmera olhando pela base.
        positions_file: cache .parquet/.csv das posições.
        force_rebuild_positions: força reconstrução do cache.
        colors_to_plot: lista opcional de cores específicas. Ex: [2, 3, 4].
        outline_mode: "full" mantém caixa LxLxL; "tight" ajusta ao cluster.
        visual_profile: "full" ou "cut".
        filename_prefix: prefixo opcional dos arquivos salvos.

    Returns:
        dict {color: path_out}
    """
    os.makedirs(path_out_dir, exist_ok=True)

    if positions_file is None:
        base = os.path.splitext(filename)[0]
        positions_file = os.path.join(path_out_dir, f"{base}_positions.parquet")

    df, meta = load_or_create_positions_codec(
        path_dir=path_dir,
        filename=filename,
        output_data=positions_file,
        time_base=time_base,
        force_rebuild=force_rebuild_positions
    )

    if seed is None:
        seed = int(meta.get("seed", -1))

    df = df.copy()
    df = df[df["color"] > 0].copy()

    if df.empty:
        raise ValueError("Nenhum ponto ativo encontrado para plotar.")

    unique_colors = sorted(df["color"].astype(int).unique().tolist())

    if colors_to_plot is None:
        colors_to_plot = unique_colors
    else:
        colors_to_plot = [int(c) for c in colors_to_plot]

    missing_colors = sorted(set(colors_to_plot) - set(unique_colors))
    if missing_colors:
        print(
            f"[WARN] As cores {missing_colors} não foram encontradas no arquivo. "
            f"Cores disponíveis: {unique_colors}"
        )

    if filename_prefix is None:
        filename_prefix = os.path.splitext(filename)[0]

    saved_paths = {}

    for color in colors_to_plot:
        if color not in unique_colors:
            continue

        df_color = df[df["color"].astype(int) == color].copy()

        if df_color.empty:
            continue

        path_out = os.path.join(
            path_out_dir,
            f"{filename_prefix}_species_{color}.png"
        )

        fig_name_color = f"{figure_name}_species_{color}"

        _plot_points_df_same_style(
            df=df_color,
            L=L,
            nc=nc,
            path_out=path_out,
            specific_color=color,
            show_base=show_base,
            outline_mode=outline_mode,
            figure_name=fig_name_color,
            visual_profile=visual_profile
        )

        saved_paths[color] = path_out

    print("Figuras salvas por espécie:")
    for color, path in saved_paths.items():
        print(f"  color={color}: {path}")

    return saved_paths