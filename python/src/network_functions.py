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
    # Prefer compact .bin metadata
    fn = os.path.join(path_dir, filename)
    base, ext = os.path.splitext(fn)
    if ext.lower() == ".npz":
        binf = base + ".bin"
        if os.path.exists(binf):
            info = _read_compact_bin(binf)
            N = info["N"]
            L3 = int(round(float(N) ** (1.0 / 3.0)))
            if L3 * L3 * L3 == N:
                return {"dim": 3, "shape": (L3, L3, L3)}
            else:
                L2 = int(round(np.sqrt(N)))
                return {"dim": 2, "shape": (L2, L2)}

    # try direct .bin
    if ext.lower() == ".bin" and os.path.exists(fn):
        info = _read_compact_bin(fn)
        N = info["N"]
        L3 = int(round(float(N) ** (1.0 / 3.0)))
        if L3 * L3 * L3 == N:
            return {"dim": 3, "shape": (L3, L3, L3)}
        else:
            L2 = int(round(np.sqrt(N)))
            return {"dim": 2, "shape": (L2, L2)}

    raise FileNotFoundError(f"No compact .bin metadata available for {fn}")

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
    # Read the compact .bin (preferred) and return a dense network array plus metadata
    fn = os.path.join(path_dir, filename)
    base, ext = os.path.splitext(fn)
    if ext.lower() == ".npz":
        candidate = base + ".bin"
        if os.path.exists(candidate):
            fn_use = candidate
        else:
            fn_use = fn
    else:
        fn_use = fn

    info = _read_compact_bin(fn_use)
    N = info["N"]
    species = info["species"].astype(np.int64)
    pos = info["pos_flat"].astype(np.int64)

    # infer shape heuristics: prefer cube then square
    L3 = int(round(float(N) ** (1.0 / 3.0)))
    if L3 * L3 * L3 == N:
        dim = 3
        SX = SY = SZ = L3
    else:
        L2 = int(round(np.sqrt(N)))
        if L2 * L2 == N:
            dim = 2
            SX = SY = L2
            SZ = 1
        else:
            raise ValueError(f"Cannot infer shape from N={N} in {fn_use}")

    if dim == 2:
        network = np.full((SX, SY), fill_value, dtype=np.int64)
        mask = species > 0
        idxs = pos[mask]
        vals = species[mask]
        x = (idxs % SX).astype(np.int64)
        y = ((idxs // SX) % SY).astype(np.int64)
        network[x, y] = vals
    else:
        network = np.full((SX, SY, SZ), fill_value, dtype=np.int64)
        mask = species > 0
        idxs = pos[mask]
        vals = species[mask]
        x = (idxs % SX).astype(np.int64)
        y = ((idxs // SX) % SY).astype(np.int64)
        z = (idxs // (SX * SY)).astype(np.int64)
        network[x, y, z] = vals

    meta = {"dim": dim, "shape": (SX, SY, SZ) if dim == 3 else (SY, SX)}
    return network, meta


TIME_BASE_3D = 10_000_000



def load_or_create_positions_codec(path_dir, filename, output_data, time_base=None, force_rebuild=False):
    """
    Se output_data existir, lê e retorna.
    Se não existir, converte do .npz, salva e retorna.
    """
    fn = _prefer_bin(filename)
    out_path = output_data

    if (not force_rebuild) and out_path is not None and os.path.exists(out_path):
        df = _read_positions_table(out_path)
        # attempt to read metadata from companion .bin
        base = os.path.splitext(os.path.join(path_dir, filename))[0]
        binf = base + ".bin"
        if os.path.exists(binf):
            info = _read_compact_bin(binf)
            N = info["N"]
            L3 = int(round(float(N) ** (1.0 / 3.0)))
            if L3 * L3 * L3 == N:
                meta = {"dim": 3, "shape": (L3, L3, L3)}
            else:
                L2 = int(round(np.sqrt(N)))
                meta = {"dim": 2, "shape": (L2, L2)}
        else:
            meta = {"dim": None, "shape": None}
        return df, meta

    return positions_from_compact_bin(path_dir, filename, output_data=output_data, time_base=time_base)
    
def _read_compact_bin(fn):
    """Read compact NetworkCompact .bin created by the C++ code.

    Binary layout (as produced by NetworkCompact::write_binary):
      - magic uint32 (0x4E455447 'NETG')
      - N uint32
      - E uint64
      - pos_flat: N x uint32
      - species: N x uint8
      - activation_time: N x uint32
      - edge_offsets: (N+1) x uint32
      - edges: E x uint32

    Returns dict with keys: N, E, pos_flat (np.uint64), species (np.uint8), activation_time (np.uint32), edge_offsets, edges
    """
    import struct

    with open(fn, "rb") as f:
        hdr = f.read(4)
        if len(hdr) < 4:
            raise ValueError(f"File too small: {fn}")
        magic = struct.unpack('<I', hdr)[0]
        if magic != 0x4E455447:
            raise ValueError(f"Bad magic in {fn}: {hex(magic)}")

        N = struct.unpack('<I', f.read(4))[0]
        E = struct.unpack('<Q', f.read(8))[0]

        # read arrays
        pos_flat = np.fromfile(f, dtype=np.uint32, count=N).astype(np.int64)
        species = np.fromfile(f, dtype=np.uint8, count=N)
        activation_time = np.fromfile(f, dtype=np.uint32, count=N).astype(np.int64)

        edge_offsets = np.fromfile(f, dtype=np.uint32, count=(N + 1)).astype(np.int64)
        edges = np.fromfile(f, dtype=np.uint32, count=E).astype(np.int64)

    return {
        "N": int(N),
        "E": int(E),
        "pos_flat": pos_flat,
        "species": species,
        "activation_time": activation_time,
        "edge_offsets": edge_offsets,
        "edges": edges,
    }


def _infer_shape_from_pos(pos):
    """Infer grid shape from maximum linear index in pos array.

    Returns (dim, SX, SY, SZ) where SZ==1 for 2D.
    """
    if pos.size == 0:
        raise ValueError("Empty pos array; cannot infer shape")
    max_idx = int(pos.max())
    # minimal candidate size is max_idx+1
    tot = int(max_idx) + 1

    # try cube
    L3 = int(round(float(tot) ** (1.0 / 3.0)))
    if L3 * L3 * L3 >= tot:
        return 3, L3, L3, L3

    # try square
    L2 = int(round(np.sqrt(tot)))
    if L2 * L2 >= tot:
        return 2, L2, L2, 1

    # fallback: assume 1D-ish
    return 1, tot, 1, 1


def positions_from_compact_bin(path_dir, filename, output_data=None, time_base=None):
    """Convert compact .bin into positions DataFrame (x,y,z,color,time).

    Heuristics to infer shape like the C++ loader: prefer cube then square.
    """
    fn_npz = os.path.join(path_dir, filename)
    # prefer .bin next to requested filename
    base, ext = os.path.splitext(fn_npz)
    if ext.lower() == ".npz":
        candidate = base + ".bin"
        if os.path.exists(candidate):
            fn = candidate
        elif os.path.exists(fn_npz):
            fn = fn_npz
        else:
            raise FileNotFoundError(f"Neither {candidate} nor {fn_npz} exist")
    else:
        fn = fn_npz

    info = _read_compact_bin(fn)
    N = info["N"]
    species = info["species"].astype(np.int64)
    activation = info["activation_time"].astype(np.int64)
    pos = info["pos_flat"].astype(np.int64)

    # Infer shape from positions (supports active-only .bin)
    dim, SX, SY, SZ = _infer_shape_from_pos(pos)

    if time_base is None:
        vmax = int(activation.max() if activation.size else 0)
        if vmax >= 100_000_000:
            time_base = 100_000_000
        elif vmax >= 10_000_000:
            time_base = 10_000_000
        else:
            time_base = 10_000_000

    # active nodes are those with species > 0
    mask = species > 0
    idxs = pos[mask]
    cols = species[mask]
    times = activation[mask]

    if dim == 3:
        x = (idxs % SX).astype(np.int32)
        y = ((idxs // SX) % SY).astype(np.int32)
        z = (idxs // (SX * SY)).astype(np.int32)
    elif dim == 2:
        x = (idxs % SX).astype(np.int32)
        y = ((idxs // SX) % SY).astype(np.int32)
        z = np.zeros_like(x)
    else:
        x = idxs.astype(np.int32)
        y = np.zeros_like(x)
        z = np.zeros_like(x)

    df = pd.DataFrame({
        "x": x,
        "y": y,
        "z": z,
        "color": cols.astype(np.int32),
        "time": times.astype(np.int64),
    })

    df = df[df["color"] > 0].copy()
    df = df.sort_values("time").reset_index(drop=True)

    if output_data is not None:
        _write_positions_table(df, output_data)

    print(f"time_base used = {time_base}")
    print(f"unique colors = {np.unique(df['color'])}")
    print(df[["x", "y", "z"]].agg(["min", "max"]))

    meta = {"dim": (3 if SZ > 1 else 2), "shape": (SZ, SY, SX)}
    return df, meta


def _prefer_bin(filename):
    base, ext = os.path.splitext(filename)
    if ext.lower() == ".npz":
        binf = base + ".bin"
        return binf
    return filename


def convert_positions_animation(path_dir, filename, dim, time_base=None):
    # compatibility wrapper used by scripts
    fn = _prefer_bin(filename)
    return positions_from_compact_bin(path_dir, fn, output_data=os.path.join(path_dir, "network_positions_time.csv"), time_base=time_base)


def convert_positions(path_dir, filename, dim, time_base=None):
    fn = _prefer_bin(filename)
    out = os.path.join(path_dir, "network_positions.csv")
    return positions_from_compact_bin(path_dir, fn, output_data=out, time_base=time_base)


def convert_positions_sp(path_dir, filename, output_csv, dim, time_base=None):
    fn = _prefer_bin(filename)
    return positions_from_compact_bin(path_dir, fn, output_data=os.path.join(path_dir, output_csv), time_base=time_base)

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

    df_pre, _ = load_or_create_positions_codec(
        path_dir=path_dir_pre,
        filename=filename_pre,
        output_data=positions_file_pre,
        force_rebuild=force_rebuild_positions
    )

    df_post, _ = load_or_create_positions_codec(
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
    base, ext = os.path.splitext(arquivo)
    if ext.lower() == ".bin":
        info = _read_compact_bin(arquivo)
        print(f"compact .bin: N={info['N']} E={info['E']}")
        print(f"sample species (first 20): {info['species'][:20]}")
        print(f"sample activation_time (first 20): {info['activation_time'][:20]}")
        return

    # Do not inspect .npz files. Prefer compact .bin.
    if ext.lower() == ".npz":
        base = os.path.splitext(arquivo)[0]
        binf = base + ".bin"
        if os.path.exists(binf):
            info = _read_compact_bin(binf)
            print(f"Found companion .bin: N={info['N']} E={info['E']}")
            return
        raise ValueError(f".npz inspection is deprecated; create or provide {base}.bin")

    raise ValueError(f"Formato não suportado pelo check_codification: {arquivo}")


def plot_network_edges(path_dir, filename, figure_name="network_edges",
                       line_width=0.6, alpha=0.8, edge_color_rule="source",
                       max_edges=None, nc=None, show_base=False, path_out=None):
    """
    Plot network edges from compact .bin. Each edge is drawn as a segment colored
    according to the incident site's color. Parameters:
      - edge_color_rule: 'source' (color of u) or 'target' or 'average'
      - max_edges: if set, limits number of edges plotted (for performance)
      - nc: number of species (optional, used to build palette)
    """
    fn = os.path.join(path_dir, filename)
    base, ext = os.path.splitext(fn)
    if ext.lower() == ".npz":
        candidate = base + ".bin"
        if os.path.exists(candidate):
            fn = candidate

    info = _read_compact_bin(fn)
    N = info["N"]
    species = info["species"].astype(np.int32)
    pos = info["pos_flat"].astype(np.int64)
    offs = info["edge_offsets"].astype(np.int64)
    edges = info["edges"].astype(np.int64)

    # infer shape from positions (supports active-only .bin)
    dim, SX, SY, SZ = _infer_shape_from_pos(pos)

    if dim == 3:
        xs = (pos % SX).astype(np.float32)
        ys = ((pos // SX) % SY).astype(np.float32)
        zs = (pos // (SX * SY)).astype(np.float32)
    else:
        xs = (pos % SX).astype(np.float32)
        ys = ((pos // SX) % SY).astype(np.float32)
        zs = np.zeros_like(xs, dtype=np.float32)

    unique_colors = sorted(int(c) for c in np.unique(species) if c > 0)
    if nc is None:
        nc_guess = max(unique_colors) if unique_colors else 1
    else:
        nc_guess = nc
    cmap = _build_fixed_color_map(unique_colors, nc_guess)

    fig = _new_figure_3d(figure_name=figure_name)

    count = 0
    for u in range(N):
        start = offs[u]
        end = offs[u + 1]
        for idx in range(start, end):
            v = int(edges[idx])
            if v < 0 or v >= N:
                continue
            if u >= v:
                # avoid duplicate undirected plotting
                continue

            xu, yu, zu = float(xs[u]), float(ys[u]), float(zs[u])
            xv, yv, zv = float(xs[v]), float(ys[v]), float(zs[v])

            if edge_color_rule == "source":
                cidx = int(species[u])
            elif edge_color_rule == "target":
                cidx = int(species[v])
            else:
                cidx = int((int(species[u]) + int(species[v])) // 2)

            if cidx <= 0:
                color = (0.5, 0.5, 0.5)
            else:
                color = cmap.get(cidx, (0.2, 0.2, 0.2))

            mlab.plot3d([xu, xv], [yu, yv], [zu, zv], color=color,
                        tube_radius=line_width * 0.02, opacity=alpha, figure=fig)

            count += 1
            if max_edges is not None and count >= max_edges:
                break
        if max_edges is not None and count >= max_edges:
            break

    _camera_plot_3D_full(fig, max(SX, SY, SZ), show_base=show_base)

    if path_out is not None:
        out_dir = os.path.dirname(path_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        mlab.savefig(path_out, figure=fig)

    return fig

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