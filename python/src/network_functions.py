import json
import numpy as np
import pandas as pd
from mayavi import mlab
mlab.options.offscreen = True
import os 
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
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

_FRAME_RENDER_CONTEXT = {}


def _slice_cumulative_frame(df, t, time_values=None):
    if time_values is None:
        time_values = df["time"].to_numpy()
    end = int(np.searchsorted(time_values, int(t), side="right"))
    return df.iloc[:end]


def _render_dynamic_height_frame_worker(item):
    ctx = _FRAME_RENDER_CONTEXT
    df = ctx["df"]
    t = int(item["time"])
    df_frame = _slice_cumulative_frame(df, t, ctx["time_values"])

    _plot_points_df_dynamic_height(
        df=df_frame,
        shape=ctx["shape"],
        nc=ctx["nc"],
        path_out=item["path"],
        figure_name=f"{ctx['prefix']}_{int(item['frame_idx']):06d}",
        specific_color=ctx["specific_color"],
        show_base=ctx["show_base"],
        visual_profile=ctx["visual_profile"],
        height_limit=ctx["height_limit"],
        current_time=t,
        highlight_growth_front=ctx["highlight_growth_front"],
    )

    return item


def _render_dynamic_height_frame_serial(df, meta, nc, prefix, item,
                                        time_values,
                                        specific_color, show_base,
                                        visual_profile, height_limit,
                                        highlight_growth_front):
    t = int(item["time"])
    df_frame = _slice_cumulative_frame(df, t, time_values)

    _plot_points_df_dynamic_height(
        df=df_frame,
        shape=meta["shape"],
        nc=nc,
        path_out=item["path"],
        figure_name=f"{prefix}_{int(item['frame_idx']):06d}",
        specific_color=specific_color,
        show_base=show_base,
        visual_profile=visual_profile,
        height_limit=height_limit,
        current_time=t,
        highlight_growth_front=highlight_growth_front,
    )

    return item


def _slice_front_frame(df, t):
    return df[df["time"] == int(t)].copy()


def _render_dynamic_front_frame_worker(item):
    ctx = _FRAME_RENDER_CONTEXT
    df = ctx["df"]
    t = int(item["time"])
    df_frame = _slice_front_frame(df, t)

    _plot_points_df_dynamic_height(
        df=df_frame,
        shape=ctx["shape"],
        nc=ctx["nc"],
        path_out=item["path"],
        figure_name=f"{ctx['prefix']}_{int(item['frame_idx']):06d}",
        specific_color=ctx["specific_color"],
        show_base=ctx["show_base"],
        visual_profile=ctx["visual_profile"],
        height_limit=ctx["height_limit"],
        current_time=None,
        highlight_growth_front=False,
    )

    return item


def _render_dynamic_front_frame_serial(df, meta, nc, prefix, item,
                                        time_values,
                                        specific_color, show_base,
                                        visual_profile, height_limit):
    t = int(item["time"])
    df_frame = _slice_front_frame(df, t)

    _plot_points_df_dynamic_height(
        df=df_frame,
        shape=meta["shape"],
        nc=nc,
        path_out=item["path"],
        figure_name=f"{prefix}_{int(item['frame_idx']):06d}",
        specific_color=specific_color,
        show_base=show_base,
        visual_profile=visual_profile,
        height_limit=height_limit,
        current_time=None,
        highlight_growth_front=False,
    )

    return item


def _get_colors_used():
    return [
        (0.90, 0.05, 0.05), # red
        (0.05, 0.20, 0.95), # blue
        (0.00, 0.65, 0.18), # green
        (0.95, 0.65, 0.00), # amber
        (0.58, 0.00, 0.82), # purple
        (0.00, 0.75, 0.85), # cyan
        (0.98, 0.00, 0.45), # magenta
        (0.15, 0.15, 0.15), # near black
        (0.60, 0.32, 0.05), # brown
        (0.55, 0.55, 0.55), # gray
    ]


def _lighten_rgb(rgb, amount=0.58):
    return tuple(
        max(0.0, min(1.0, c + (1.0 - c) * amount))
        for c in rgb
    )


def _build_front_highlight_color_map(colors_to_plot, color_map):
    colors_to_plot = sorted(int(c) for c in colors_to_plot)
    return {
        color: _lighten_rgb(color_map[color])
        for color in colors_to_plot
    }


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


def _strip_percolation_suffix(name):
    if name.lower().endswith("_percolation"):
        return name[:-len("_percolation")]
    return name

def _find_percolation_json_for_network(path_dir, filename):
    network_parent = os.path.dirname(os.path.abspath(path_dir))
    data_dir = os.path.join(network_parent, "data")
    base = os.path.splitext(os.path.basename(filename))[0]
    base = _strip_percolation_suffix(base)
    json_path = os.path.join(data_dir, base + ".json")
    return json_path if os.path.exists(json_path) else None

def _read_percolation_order_by_color(path_json):
    if path_json is None:
        return {}

    with open(path_json, "r") as fh:
        data = json.load(fh)

    results = data.get("results", {})
    if not isinstance(results, dict):
        return {}

    order_by_color = {}
    for key, block in results.items():
        if not isinstance(key, str) or "order_percolation" not in key:
            continue

        try:
            order = int(key.split()[-1])
        except (TypeError, ValueError):
            continue

        payload = block.get("data", block) if isinstance(block, dict) else {}
        color = payload.get("color") if isinstance(payload, dict) else None

        try:
            order_by_color[int(color)] = order
        except (TypeError, ValueError):
            continue

    return order_by_color

def _darken_rgb(rgb, factor=0.82):
    return tuple(max(0.0, min(1.0, c * factor)) for c in rgb)

def _apply_full_cube_style(pts, edge_width=1.4, opacity=1.0, ambient=0.10):
    prop = pts.actor.property
    prop.edge_visibility = True
    prop.edge_color = (0, 0, 0)
    prop.line_width = edge_width

    prop.opacity = opacity
    prop.ambient = ambient
    prop.diffuse = 0.90
    prop.specular = 0.0

def _draw_points3d_cube_cloud(
    fig,
    x,
    y,
    z,
    rgb,
    scale_factor=0.82,
    darken_factor=1.0,
    edge_width=0.25,
    opacity=1.0,
    ambient=0.10,
):
    rgb_use = _darken_rgb(rgb, darken_factor)

    pts = mlab.points3d(
        x, y, z,
        np.ones_like(x),
        color=rgb_use,
        scale_factor=scale_factor,
        opacity=opacity,
        mode="cube",
        figure=fig
    )
    _apply_full_cube_style(
        pts,
        edge_width=edge_width,
        opacity=opacity,
        ambient=ambient,
    )
    return pts


def _dynamic_height_extent(df, SX, SY, SZ):
    if int(SZ) > 1:
        zmax = int(df["z"].max()) + 1
        z_extent = max(1, zmax)
        return [0, int(SX), 0, int(SY), 0, z_extent], z_extent

    ymax = int(df["y"].max()) + 1
    y_extent = max(1, ymax)
    return [0, int(SX), 0, y_extent, 0, 1], y_extent


def _apply_regular_lattice_camera(fig, extent, show_base=False):
    Lx = float(extent[1] - extent[0])
    Ly = float(extent[3] - extent[2])
    Lz = float(extent[5] - extent[4])
    Lref = max(Lx, Ly, Lz, 1.0)
    focal = (
        (extent[0] + extent[1]) / 2,
        (extent[2] + extent[3]) / 2,
        (extent[4] + extent[5]) / 2,
    )

    if show_base:
        mlab.view(
            azimuth=0,
            elevation=-90,
            distance=3.2 * Lref,
            focalpoint=(focal[0], focal[1], extent[4]),
            figure=fig,
        )
    else:
        mlab.view(
            azimuth=45,
            elevation=60,
            distance=3.4 * Lref,
            focalpoint=focal,
            figure=fig,
        )

    fig.scene.camera.parallel_projection = True
    fig.scene.camera.parallel_scale = 0.82 * Lref
    fig.scene.reset_zoom()
    fig.scene.camera.parallel_scale = 0.82 * Lref
    fig.scene.render()

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
        darken_factor = 1.0
        edge_width = 0.25
        marker_scale = 0.82
    elif visual_profile == "cut":
        darken_factor = 1.0
        edge_width = 0.30
        marker_scale = 0.82
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
            scale_factor=marker_scale,
            darken_factor=darken_factor,
            edge_width=edge_width
        )

    if outline_mode == "full":
        z_extent = max(1, int(df["z"].max()) + 1)
        extent = [0, L, 0, L, 0, z_extent]
    elif outline_mode == "tight":
        xmin, xmax = df["x"].min(), df["x"].max() + 1
        ymin, ymax = df["y"].min(), df["y"].max() + 1
        zmin, zmax = df["z"].min(), df["z"].max() + 1
        extent = [xmin, xmax, ymin, ymax, zmin, zmax]
    else:
        raise ValueError("outline_mode deve ser 'full' ou 'tight'.")

    mlab.outline(
        extent=extent,
        color=(0, 0, 0),
        line_width=2.0,
        figure=fig
    )

    _apply_regular_lattice_camera(fig, extent, show_base=show_base)

    mlab.savefig(path_out, magnification=4, figure=fig)
    print(f"network save in {path_out}")
    mlab.close(fig)


def _plot_points_df_dynamic_height(
    df,
    shape,
    nc,
    path_out,
    figure_name=None,
    specific_color=None,
    show_base=False,
    visual_profile="full",
    height_limit=None,
    current_time=None,
    highlight_growth_front=True,
):
    df = df.copy()
    for col in ("x", "y", "z", "color"):
        if col not in df.columns:
            raise ValueError(f"DataFrame sem coluna obrigatória: {col}")

    df["x"] = df["x"].astype(int)
    df["y"] = df["y"].astype(int)
    df["z"] = df["z"].astype(int)
    df["color"] = df["color"].astype(int)
    if "time" in df.columns:
        df["time"] = df["time"].astype(int)
    df = df[df["color"] > 0].copy()
    if df.empty:
        raise ValueError("Nenhum ponto ativo para plotar.")

    if figure_name is None:
        figure_name = os.path.splitext(os.path.basename(path_out))[0]

    SX, SY, SZ = (int(shape[0]), int(shape[1]), int(shape[2]))
    if height_limit is not None:
        hlim = int(height_limit)
        if SZ > 1:
            df = df[df["z"] <= hlim].copy()
            SZ = min(SZ, hlim + 1)
        else:
            df = df[df["y"] <= hlim].copy()
            SY = min(SY, hlim + 1)
        if df.empty:
            raise ValueError("Nenhum ponto ativo dentro do limite de altura.")

    fig = _new_figure_3d(figure_name=figure_name)

    unique_colors = sorted(df["color"].unique().tolist())
    color_map = _build_fixed_color_map(unique_colors, nc)

    if visual_profile == "full":
        darken_factor = 1.0
        edge_width = 0.25
        marker_scale = 0.82
    elif visual_profile == "cut":
        darken_factor = 1.0
        edge_width = 0.30
        marker_scale = 0.82
    else:
        raise ValueError("visual_profile deve ser 'full' ou 'cut'.")

    if specific_color is None:
        colors_to_plot = unique_colors
    else:
        specific_color = int(specific_color)
        if specific_color not in unique_colors:
            mlab.close(fig)
            raise ValueError(f"Cor {specific_color} não encontrada no frame.")
        colors_to_plot = [specific_color]

    highlight_enabled = (
        bool(highlight_growth_front)
        and current_time is not None
        and "time" in df.columns
    )
    current_time = None if current_time is None else int(current_time)

    for color in colors_to_plot:
        df_color = df[df["color"] == color]
        if highlight_enabled:
            df_color = df_color[df_color["time"] != current_time]
        if df_color.empty:
            continue

        _draw_points3d_cube_cloud(
            fig=fig,
            x=df_color["x"].to_numpy(),
            y=df_color["y"].to_numpy(),
            z=df_color["z"].to_numpy(),
            rgb=color_map[color],
            scale_factor=marker_scale,
            darken_factor=darken_factor,
            edge_width=edge_width,
        )

    if highlight_enabled:
        front_color_map = _build_front_highlight_color_map(colors_to_plot, color_map)

        for color in colors_to_plot:
            df_front = df[(df["color"] == color) & (df["time"] == current_time)]
            if df_front.empty:
                continue

            _draw_points3d_cube_cloud(
                fig=fig,
                x=df_front["x"].to_numpy(),
                y=df_front["y"].to_numpy(),
                z=df_front["z"].to_numpy(),
                rgb=front_color_map[color],
                scale_factor=1.34,
                darken_factor=1.0,
                edge_width=1.45,
                opacity=0.95,
                ambient=0.35,
            )
            _draw_points3d_cube_cloud(
                fig=fig,
                x=df_front["x"].to_numpy(),
                y=df_front["y"].to_numpy(),
                z=df_front["z"].to_numpy(),
                rgb=color_map[color],
                scale_factor=0.90,
                darken_factor=1.0,
                edge_width=0.65,
                opacity=1.0,
                ambient=0.25,
            )

    extent, _ = _dynamic_height_extent(df, SX, SY, SZ)
    mlab.outline(
        extent=extent,
        color=(0, 0, 0),
        line_width=2.0,
        figure=fig,
    )

    _apply_regular_lattice_camera(fig, extent, show_base=show_base)

    out_dir = os.path.dirname(path_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mlab.savefig(path_out, magnification=4, figure=fig)
    mlab.close(fig)

def save_dynamic_height_cumulative_frames(
    path_dir,
    network_filename=None,
    output_dir=None,
    nc=4,
    L=None,
    dim=None,
    frame_stride=1,
    max_frames=None,
    specific_color=None,
    show_base=False,
    visual_profile="full",
    positions_file=None,
    force_rebuild_positions=False,
    prefix="frame",
    height_limit=None,
    stop_when_front_reaches_height=True,
    highlight_growth_front=True,
    resume=True,
    frame_workers=1,
    memory_file=None,
    overwrite_existing=False,
):
    """
    Salva frames cumulativos da rede dinâmica.

    Frame em t contém todos os sítios com activation_time <= t, ou seja:
      t=0 -> base inicial;
      t=1 -> base + sítios ativados em 1;
      ...

    Args:
        path_dir: pasta da execução, contendo subpasta `network/`.
        network_filename: arquivo `.bin`; se None escolhe a rede completa.
        output_dir: pasta para PNGs; default `<path_dir>/dynamic_frames`.
        frame_stride: usa todos os tempos se 1; use 5,10,... para reduzir frames.
        max_frames: limite opcional de frames.
        height_limit: limite opcional da direção de crescimento. Em dim=2 filtra
            y <= height_limit; em dim=3 filtra z <= height_limit.
        stop_when_front_reaches_height: se True, não gera tempos posteriores ao
            primeiro instante em que a frente cumulativa alcança height_limit.

        resume: se True, pula frames já existentes e continua a geração.
        frame_workers: número de processos para renderizar frames em paralelo.
            Use 1 para modo serial. Em Linux, valores 2-4 costumam ser um bom
            compromisso para Mayavi/VTK.
        memory_file: arquivo JSON opcional para registrar o progresso.
        overwrite_existing: se True, recria os frames mesmo que já existam.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride deve ser >= 1")
    if frame_workers < 1:
        raise ValueError("frame_workers deve ser >= 1")

    network_dir = os.path.join(path_dir, "network")
    network_filename = _choose_network_file(path_dir, network_filename)

    if output_dir is None:
        output_dir = os.path.join(path_dir, "dynamic_frames")

    os.makedirs(output_dir, exist_ok=True)

    if memory_file is None:
        memory_file = os.path.join(output_dir, f"{prefix}_resume_memory.json")

    if positions_file is None:
        positions_file = os.path.join(
            output_dir,
            f"{os.path.splitext(network_filename)[0]}_dynamic_positions.csv",
        )

    def _valid_frame(path_out):
        return os.path.exists(path_out) and os.path.getsize(path_out) > 0

    def _write_memory(done_items, total_frames):
        payload = {
            "path_dir": path_dir,
            "output_dir": output_dir,
            "network_file": os.path.join(network_dir, network_filename),
            "positions_file": positions_file,
            "prefix": prefix,
            "frame_stride": int(frame_stride),
            "max_frames": None if max_frames is None else int(max_frames),
            "height_limit": None if height_limit is None else int(height_limit),
            "stop_when_front_reaches_height": bool(stop_when_front_reaches_height),
            "highlight_growth_front": bool(highlight_growth_front),
            "frame_workers": int(frame_workers),
            "specific_color": specific_color,
            "show_base": bool(show_base),
            "visual_profile": visual_profile,
            "total_frames": int(total_frames),
            "done_frames": [
                {
                    "frame_idx": int(item["frame_idx"]),
                    "time": int(item["time"]),
                    "path": item["path"],
                }
                for item in done_items
            ],
        }

        tmp_file = f"{memory_file}.tmp"

        with open(tmp_file, "w") as f:
            json.dump(payload, f, indent=2)

        os.replace(tmp_file, memory_file)

    if (not force_rebuild_positions) and os.path.exists(positions_file):
        df = _read_positions_table(positions_file)

        if L is None:
            L = _infer_L_from_path(path_dir)

        if dim is None:
            dim = _infer_dim_from_path(path_dir)

        info = _read_compact_bin(os.path.join(network_dir, network_filename))

        if dim == 3:
            shape = (int(L), int(L), int(info["N"]) // (int(L) * int(L)))
        else:
            shape = (int(L), int(info["N"]) // int(L), 1)

        meta = {
            "dim": dim,
            "shape": shape,
            "N": int(info["N"]),
            "E": int(info["E"]),
        }
    else:
        df, meta = positions_from_dynamic_height_compact_bin(
            path_dir=network_dir,
            filename=network_filename,
            L=L,
            dim=dim,
            output_data=positions_file,
        )

    if df.empty:
        raise ValueError("A rede não possui sítios ativos para animar.")

    if height_limit is not None:
        hlim = int(height_limit)
        growth_col = "z" if int(meta["dim"]) == 3 else "y"

        df = df[df[growth_col] <= hlim].copy()

        if df.empty:
            raise ValueError("Nenhum sítio ativo dentro do limite de altura.")

        shape = tuple(int(v) for v in meta["shape"])

        if int(meta["dim"]) == 3:
            meta["shape"] = (shape[0], shape[1], min(shape[2], hlim + 1))
        else:
            meta["shape"] = (shape[0], min(shape[1], hlim + 1), shape[2])

        if stop_when_front_reaches_height:
            reached = df[df[growth_col] >= hlim]

            if not reached.empty:
                t_stop = int(reached["time"].min())
                df = df[df["time"] <= t_stop].copy()

    df = df.sort_values("time").reset_index(drop=True)
    time_values = df["time"].to_numpy()
    times = np.array(sorted(df["time"].astype(int).unique()), dtype=int)
    times = times[::frame_stride]

    if max_frames is not None:
        times = times[:int(max_frames)]

    expected_frames = []

    for frame_idx, t in enumerate(times):
        path_out = os.path.join(
            output_dir,
            f"{prefix}_{frame_idx:06d}_t{int(t):06d}.png",
        )

        expected_frames.append(
            {
                "frame_idx": int(frame_idx),
                "time": int(t),
                "path": path_out,
            }
        )

    saved = []
    done_items = []

    start_idx = 0

    if resume and not overwrite_existing:
        while start_idx < len(expected_frames):
            item = expected_frames[start_idx]

            if not _valid_frame(item["path"]):
                break

            saved.append(item["path"])
            done_items.append(item)
            start_idx += 1

        if start_idx > 0:
            print(
                f"[resume] {start_idx}/{len(expected_frames)} frames já existem. "
                f"Continuando a partir do frame {start_idx}."
            )

        _write_memory(done_items, len(expected_frames))

    render_items = []

    for item in expected_frames[start_idx:]:
        path_out = item["path"]

        if resume and not overwrite_existing and _valid_frame(path_out):
            saved.append(path_out)
            done_items.append(item)

            print(
                f"[skip {item['frame_idx'] + 1}/{len(expected_frames)}] "
                f"t={int(item['time'])} já existe -> {path_out}"
            )

            _write_memory(done_items, len(expected_frames))
            continue

        render_items.append(item)

    workers = int(frame_workers)
    if workers > 1 and "fork" not in mp.get_all_start_methods():
        print("[WARN] multiprocessing por fork não disponível. Usando frame_workers=1.")
        workers = 1
    if render_items:
        workers = min(workers, len(render_items))

    if workers > 1 and render_items:
        global _FRAME_RENDER_CONTEXT
        _FRAME_RENDER_CONTEXT = {
            "df": df,
            "time_values": time_values,
            "shape": meta["shape"],
            "nc": nc,
            "prefix": prefix,
            "specific_color": specific_color,
            "show_base": show_base,
            "visual_profile": visual_profile,
            "height_limit": height_limit,
            "highlight_growth_front": highlight_growth_front,
        }

        print(
            f"[multiprocessing] renderizando {len(render_items)} frames "
            f"com {workers} processos."
        )

        mp_context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
            futures = [
                executor.submit(_render_dynamic_height_frame_worker, item)
                for item in render_items
            ]

            for future in as_completed(futures):
                item = future.result()
                saved.append(item["path"])
                done_items.append(item)
                done_items_sorted = sorted(done_items, key=lambda done: done["frame_idx"])

                _write_memory(done_items_sorted, len(expected_frames))

                print(
                    f"[frame {item['frame_idx'] + 1}/{len(expected_frames)}] "
                    f"t={int(item['time'])} -> {item['path']}"
                )
    else:
        for item in render_items:
            item = _render_dynamic_height_frame_serial(
                df=df,
                meta=meta,
                nc=nc,
                prefix=prefix,
                item=item,
                time_values=time_values,
                specific_color=specific_color,
                show_base=show_base,
                visual_profile=visual_profile,
                height_limit=height_limit,
                highlight_growth_front=highlight_growth_front,
            )

            saved.append(item["path"])
            done_items.append(item)

            _write_memory(done_items, len(expected_frames))

            print(
                f"[frame {item['frame_idx'] + 1}/{len(expected_frames)}] "
                f"t={int(item['time'])} -> {item['path']}"
            )

    done_items = sorted(done_items, key=lambda done: done["frame_idx"])
    saved = [item["path"] for item in done_items]

    return {
        "frames": saved,
        "output_dir": output_dir,
        "positions_file": positions_file,
        "network_file": os.path.join(network_dir, network_filename),
        "shape": meta["shape"],
        "times": times.tolist(),
        "memory_file": memory_file,
        "resume": resume,
        "overwrite_existing": overwrite_existing,
    }


def save_dynamic_height_front_frames(
    path_dir,
    network_filename=None,
    output_dir=None,
    nc=4,
    L=None,
    dim=None,
    frame_stride=1,
    max_frames=None,
    specific_color=None,
    show_base=False,
    visual_profile="full",
    positions_file=None,
    force_rebuild_positions=False,
    prefix="frame",
    height_limit=None,
    stop_when_front_reaches_height=True,
    resume=True,
    frame_workers=1,
    memory_file=None,
    overwrite_existing=False,
):
    """
    Salva frames contendo apenas a frente ativa (sítios ativados no tempo t).

    Cada frame é gerado com os pontos cujo `time` corresponde ao instante t.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride deve ser >= 1")
    if frame_workers < 1:
        raise ValueError("frame_workers deve ser >= 1")

    network_dir = os.path.join(path_dir, "network")
    network_filename = _choose_network_file(path_dir, network_filename)

    if output_dir is None:
        output_dir = os.path.join(path_dir, "dynamic_front_frames")

    os.makedirs(output_dir, exist_ok=True)

    if memory_file is None:
        memory_file = os.path.join(output_dir, f"{prefix}_resume_memory.json")

    if positions_file is None:
        positions_file = os.path.join(
            output_dir,
            f"{os.path.splitext(network_filename)[0]}_dynamic_positions.csv",
        )

    def _valid_frame(path_out):
        return os.path.exists(path_out) and os.path.getsize(path_out) > 0

    def _write_memory(done_items, total_frames):
        payload = {
            "path_dir": path_dir,
            "output_dir": output_dir,
            "network_file": os.path.join(network_dir, network_filename),
            "positions_file": positions_file,
            "prefix": prefix,
            "frame_stride": int(frame_stride),
            "max_frames": None if max_frames is None else int(max_frames),
            "height_limit": None if height_limit is None else int(height_limit),
            "stop_when_front_reaches_height": bool(stop_when_front_reaches_height),
            "frame_workers": int(frame_workers),
            "specific_color": specific_color,
            "show_base": bool(show_base),
            "visual_profile": visual_profile,
            "total_frames": int(total_frames),
            "done_frames": [
                {
                    "frame_idx": int(item["frame_idx"]),
                    "time": int(item["time"]),
                    "path": item["path"],
                }
                for item in done_items
            ],
        }

        tmp_file = f"{memory_file}.tmp"

        with open(tmp_file, "w") as f:
            json.dump(payload, f, indent=2)

        os.replace(tmp_file, memory_file)

    if (not force_rebuild_positions) and os.path.exists(positions_file):
        df = _read_positions_table(positions_file)

        if L is None:
            L = _infer_L_from_path(path_dir)

        if dim is None:
            dim = _infer_dim_from_path(path_dir)

        info = _read_compact_bin(os.path.join(network_dir, network_filename))

        if dim == 3:
            shape = (int(L), int(L), int(info["N"]) // (int(L) * int(L)))
        else:
            shape = (int(L), int(info["N"]) // int(L), 1)

        meta = {
            "dim": dim,
            "shape": shape,
            "N": int(info["N"]),
            "E": int(info["E"]),
        }
    else:
        df, meta = positions_from_dynamic_height_compact_bin(
            path_dir=network_dir,
            filename=network_filename,
            L=L,
            dim=dim,
            output_data=positions_file,
        )

    if df.empty:
        raise ValueError("A rede não possui sítios ativos para animar.")

    if height_limit is not None:
        hlim = int(height_limit)
        growth_col = "z" if int(meta["dim"]) == 3 else "y"

        df = df[df[growth_col] <= hlim].copy()

        if df.empty:
            raise ValueError("Nenhum sítio ativo dentro do limite de altura.")

        shape = tuple(int(v) for v in meta["shape"])

        if int(meta["dim"]) == 3:
            meta["shape"] = (shape[0], shape[1], min(shape[2], hlim + 1))
        else:
            meta["shape"] = (shape[0], min(shape[1], hlim + 1), shape[2])

        if stop_when_front_reaches_height:
            reached = df[df[growth_col] >= hlim]

            if not reached.empty:
                t_stop = int(reached["time"].min())
                df = df[df["time"] <= t_stop].copy()

    df = df.sort_values("time").reset_index(drop=True)
    times = np.array(sorted(df["time"].astype(int).unique()), dtype=int)
    times = times[::frame_stride]

    if max_frames is not None:
        times = times[:int(max_frames)]

    expected_frames = []

    for frame_idx, t in enumerate(times):
        path_out = os.path.join(
            output_dir,
            f"{prefix}_{frame_idx:06d}_t{int(t):06d}.png",
        )

        expected_frames.append(
            {
                "frame_idx": int(frame_idx),
                "time": int(t),
                "path": path_out,
            }
        )

    saved = []
    done_items = []

    start_idx = 0

    if resume and not overwrite_existing:
        while start_idx < len(expected_frames):
            item = expected_frames[start_idx]

            if not _valid_frame(item["path"]):
                break

            saved.append(item["path"])
            done_items.append(item)
            start_idx += 1

        if start_idx > 0:
            print(
                f"[resume] {start_idx}/{len(expected_frames)} frames já existem. "
                f"Continuando a partir do frame {start_idx}."
            )

        _write_memory(done_items, len(expected_frames))

    render_items = []

    for item in expected_frames[start_idx:]:
        path_out = item["path"]

        if resume and not overwrite_existing and _valid_frame(path_out):
            saved.append(path_out)
            done_items.append(item)

            print(
                f"[skip {item['frame_idx'] + 1}/{len(expected_frames)}] "
                f"t={int(item['time'])} já existe -> {path_out}"
            )

            _write_memory(done_items, len(expected_frames))
            continue

        render_items.append(item)

    workers = int(frame_workers)
    if workers > 1 and "fork" not in mp.get_all_start_methods():
        print("[WARN] multiprocessing por fork não disponível. Usando frame_workers=1.")
        workers = 1
    if render_items:
        workers = min(workers, len(render_items))

    if workers > 1 and render_items:
        global _FRAME_RENDER_CONTEXT
        _FRAME_RENDER_CONTEXT = {
            "df": df,
            "shape": meta["shape"],
            "nc": nc,
            "prefix": prefix,
            "specific_color": specific_color,
            "show_base": show_base,
            "visual_profile": visual_profile,
            "height_limit": height_limit,
        }

        print(
            f"[multiprocessing] renderizando {len(render_items)} frames "
            f"com {workers} processos."
        )

        mp_context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
            futures = [
                executor.submit(_render_dynamic_front_frame_worker, item)
                for item in render_items
            ]

            for future in as_completed(futures):
                item = future.result()
                saved.append(item["path"])
                done_items.append(item)
                done_items_sorted = sorted(done_items, key=lambda done: done["frame_idx"])

                _write_memory(done_items_sorted, len(expected_frames))

                print(
                    f"[frame {item['frame_idx'] + 1}/{len(expected_frames)}] "
                    f"t={int(item['time'])} -> {item['path']}"
                )
    else:
        for item in render_items:
            item = _render_dynamic_front_frame_serial(
                df=df,
                meta=meta,
                nc=nc,
                prefix=prefix,
                item=item,
                time_values=None,
                specific_color=specific_color,
                show_base=show_base,
                visual_profile=visual_profile,
                height_limit=height_limit,
            )

            saved.append(item["path"])
            done_items.append(item)

            _write_memory(done_items, len(expected_frames))

            print(
                f"[frame {item['frame_idx'] + 1}/{len(expected_frames)}] "
                f"t={int(item['time'])} -> {item['path']}"
            )

    done_items = sorted(done_items, key=lambda done: done["frame_idx"])
    saved = [item["path"] for item in done_items]

    return {
        "frames": saved,
        "output_dir": output_dir,
        "positions_file": positions_file,
        "network_file": os.path.join(network_dir, network_filename),
        "shape": meta["shape"],
        "times": times.tolist(),
        "memory_file": memory_file,
        "resume": resume,
        "overwrite_existing": overwrite_existing,
    }


def write_gimp_crop_frames_script(
    output_dir,
    cropped_dir=None,
    script_path=None,
    frame_prefix="frame_",
    background_rgb=(255, 255, 255),
    threshold=8,
    crop_padding=32,
    gimp_executable="gimp",
    run=False,
):
    """
    Gera um script batch do GIMP para limpar fundo e recortar cada frame.

    Para cada PNG em `output_dir` cujo nome começa com `frame_prefix`, o script:
      1. abre a imagem;
      2. adiciona canal alfa;
      3. seleciona o fundo branco por cor;
      4. deleta o fundo;
      5. recorta ao conteúdo;
      6. adiciona uma margem transparente;
      7. salva em `cropped_dir`.

    Retorna um dicionário com o caminho do script, comando sugerido e arquivos.
    """
    output_dir = os.path.abspath(output_dir)
    if cropped_dir is None:
        cropped_dir = os.path.join(output_dir, "cropped")
    cropped_dir = os.path.abspath(cropped_dir)
    os.makedirs(cropped_dir, exist_ok=True)

    if script_path is None:
        script_path = os.path.join(output_dir, "gimp_crop_frames.scm")
    script_path = os.path.abspath(script_path)

    frames = [
        f for f in sorted(os.listdir(output_dir))
        if f.lower().endswith(".png") and f.startswith(frame_prefix)
    ]
    if not frames:
        raise FileNotFoundError(
            f"Nenhum PNG começando com {frame_prefix!r} encontrado em {output_dir}"
        )

    def scm_string(path):
        return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'

    r, g, b = (int(background_rgb[0]), int(background_rgb[1]), int(background_rgb[2]))

    lines = [
        "(define (sop-crop-frame input-path output-path threshold padding)",
        "  (let* (",
        "      (image (car (gimp-file-load RUN-NONINTERACTIVE input-path input-path)))",
        "      (layer (car (gimp-image-get-active-layer image))))",
        "    (gimp-layer-add-alpha layer)",
        f"    (gimp-by-color-select layer '({r} {g} {b}) threshold CHANNEL-OP-REPLACE TRUE FALSE 0 FALSE)",
        "    (gimp-edit-clear layer)",
        "    (gimp-selection-none image)",
        "    (plug-in-autocrop RUN-NONINTERACTIVE image layer)",
        "    (let* (",
        "        (width (car (gimp-image-width image)))",
        "        (height (car (gimp-image-height image))))",
        "      (gimp-image-resize image (+ width (* 2 padding)) (+ height (* 2 padding)) padding padding))",
        "    (set! layer (car (gimp-image-get-active-layer image)))",
        "    (gimp-file-save RUN-NONINTERACTIVE image layer output-path output-path)",
        "    (gimp-image-delete image)))",
        "",
    ]

    for frame in frames:
        in_path = os.path.join(output_dir, frame)
        out_path = os.path.join(cropped_dir, frame)
        lines.append(
            f"(sop-crop-frame {scm_string(in_path)} {scm_string(out_path)} "
            f"{int(threshold)} {int(crop_padding)})"
        )

    lines.append("(gimp-quit 0)")
    lines.append("")

    with open(script_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    command = [
        gimp_executable,
        "-i",
        "-b",
        f"(load {scm_string(script_path)})",
    ]

    if run:
        subprocess.run(command, check=True)

    return {
        "script_path": script_path,
        "cropped_dir": cropped_dir,
        "crop_padding": int(crop_padding),
        "frames": [os.path.join(output_dir, f) for f in frames],
        "command": command,
    }

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


def load_surface_json(path_json):
    """Load surface JSON data from a run folder data_surfaces file."""
    if not os.path.exists(path_json):
        raise FileNotFoundError(f"Arquivo de superfície não encontrado: {path_json}")

    with open(path_json, "r") as f:
        data = json.load(f)

    surfaces = {}
    for key in ("surface_preteq", "surface_posteq"):
        if key not in data:
            continue

        arr = np.asarray(data[key], dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(
                f"Formato inesperado em {key}: cada registro deve ter pelo menos 4 valores"
            )

        surfaces[key] = pd.DataFrame(
            arr[:, :4],
            columns=["x", "y", "z", "color"]
        )

    return surfaces


def _build_surface_grid(x, y, z, c):
    nx = int(x.max()) + 1
    ny = int(y.max()) + 1

    Z = np.full((ny, nx), np.nan, dtype=float)
    C = np.full((ny, nx), np.nan, dtype=float)

    Z[y, x] = z
    C[y, x] = c

    return Z, C


def _fill_surface_grid_nans(Z, C):
    """Fill small holes in Plotly surface grids for display."""
    if not (np.isnan(Z).any() or np.isnan(C).any()):
        return Z, C

    Z_filled = pd.DataFrame(Z).interpolate(
        axis=0,
        limit_direction="both",
    ).interpolate(
        axis=1,
        limit_direction="both",
    ).to_numpy()

    C_filled = pd.DataFrame(C).ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1).to_numpy()

    if np.isnan(Z_filled).any():
        Z_filled = np.nan_to_num(Z_filled, nan=float(np.nanmean(Z)))
    if np.isnan(C_filled).any():
        C_filled = np.nan_to_num(C_filled, nan=float(np.nanmin(C)))

    return Z_filled, C_filled


def _save_surface_heightmaps(surface_dfs, path_out, cmap="viridis"):
    pre = surface_dfs.get("surface_preteq")
    post = surface_dfs.get("surface_posteq")
    if pre is None and post is None:
        raise ValueError("Nenhum dado de superfície encontrado para plotar.")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    if pre is None:
        axes = [None, axes[0]]
    elif post is None:
        axes = [axes[0], None]

    def plot_panel(ax, df, title):
        x = df["x"].to_numpy().astype(int)
        y = df["y"].to_numpy().astype(int)
        z = df["z"].to_numpy().astype(float)
        c = df["color"].to_numpy().astype(int)
        Z, C = _build_surface_grid(x, y, z, c)

        im = ax.imshow(Z, origin="lower", cmap=cmap, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    if pre is not None:
        plot_panel(axes[0], pre, "Surface PRETEQ")
    else:
        axes[0].axis("off")

    if post is not None:
        plot_panel(axes[1], post, "Surface POSTEQ")
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(path_out, dpi=150)
    plt.close(fig)
    print(f"surface save in {path_out}")


def plot_surfaces_from_json(path_surface_json,
                            path_out_dir=None,
                            cmap="viridis",
                            figure_name=None):
    """Plot surfaces defined in a data_surfaces JSON file using matplotlib."""
    if path_out_dir is None:
        path_out_dir = os.path.dirname(path_surface_json)
    os.makedirs(path_out_dir, exist_ok=True)

    surfaces = load_surface_json(path_surface_json)
    if not surfaces:
        raise ValueError(f"Nenhum dado de superfície válido em {path_surface_json}")

    if figure_name is None:
        base_name = os.path.splitext(os.path.basename(path_surface_json))[0]
        figure_name = f"{base_name}_surfaces"

    path_out = os.path.join(path_out_dir, f"{figure_name}.png")
    _save_surface_heightmaps(surfaces, path_out, cmap=cmap)
    return {key: path_out for key in surfaces.keys()}


def _infer_L_from_path(path_dir):
    basename = os.path.basename(os.path.normpath(path_dir))
    if basename.startswith("L_"):
        try:
            return int(basename.split("L_")[1].split("/")[0])
        except ValueError:
            pass

    for part in path_dir.split(os.sep):
        if part.startswith("L_"):
            try:
                return int(part.split("L_")[1])
            except ValueError:
                pass

    raise ValueError(f"Não foi possível inferir L a partir do caminho: {path_dir}")


def _infer_dim_from_path(path_dir):
    for part in path_dir.split(os.sep):
        if part.startswith("dim_"):
            try:
                return int(part.split("dim_")[1])
            except ValueError:
                pass
    raise ValueError(f"Não foi possível inferir dim a partir do caminho: {path_dir}")


def positions_from_dynamic_height_compact_bin(path_dir,
                                              filename,
                                              L=None,
                                              dim=None,
                                              output_data=None):
    """
    Lê uma rede compacta `.bin` gerada pelo growth_test com altura dinâmica.

    O formato C++ salva `pos_flat`, `species` e `activation_time`. Aqui a forma
    não é inferida como cubo: para dim=3 usa L x L x H, e para dim=2 usa L x H.
    Retorna um DataFrame com x, y, z, color, time e meta com shape=(SX,SY,SZ).
    """
    network_path = os.path.join(path_dir, filename)
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"Arquivo de rede não encontrado: {network_path}")

    if L is None:
        L = _infer_L_from_path(path_dir)
    if dim is None:
        dim = _infer_dim_from_path(path_dir)

    info = _read_compact_bin(network_path)
    N = int(info["N"])
    if dim == 3:
        base = int(L) * int(L)
        if N % base != 0:
            raise ValueError(f"N={N} não é múltiplo de L*L={base}")
        SX, SY, SZ = int(L), int(L), N // base
    elif dim == 2:
        if N % int(L) != 0:
            raise ValueError(f"N={N} não é múltiplo de L={L}")
        SX, SY, SZ = int(L), N // int(L), 1
    else:
        raise ValueError("dim deve ser 2 ou 3")

    species = info["species"].astype(np.int64)
    activation = info["activation_time"].astype(np.int64)
    pos = info["pos_flat"].astype(np.int64)

    mask = species > 0
    idxs = pos[mask]
    cols = species[mask]
    times = activation[mask]

    x = (idxs % SX).astype(np.int32)
    y = ((idxs // SX) % SY).astype(np.int32)
    if dim == 3:
        z = (idxs // (SX * SY)).astype(np.int32)
    else:
        z = np.zeros_like(x)

    df = pd.DataFrame({
        "x": x,
        "y": y,
        "z": z,
        "color": cols.astype(np.int32),
        "time": times.astype(np.int64),
    }).sort_values("time").reset_index(drop=True)

    if output_data is not None:
        _write_positions_table(df, output_data)

    meta = {
        "dim": dim,
        "shape": (SX, SY, SZ),
        "N": N,
        "E": int(info["E"]),
        "L": int(L),
    }
    return df, meta


def plot_active_sites_from_folder(path_dir,
                                  filename=None,
                                  path_out=None,
                                  figure_name="network_active",
                                  L=None,
                                  nc=None,
                                  time_base=TIME_BASE_3D,
                                  show_base=False,
                                  positions_file=None,
                                  force_rebuild_positions=False):
    """Plot active network sites from a run folder with a network subfolder."""
    network_dir = os.path.join(path_dir, "network")
    if filename is None:
        files = [f for f in os.listdir(network_dir) if f.lower().endswith(".bin") or f.lower().endswith(".npz")]
        if not files:
            raise FileNotFoundError(f"Nenhum arquivo de rede encontrado em {network_dir}")
        files = sorted(files)
        filename = next((f for f in files if "percolation" not in f.lower()), files[0])

    if path_out is None:
        path_out = os.path.join(path_dir, f"{figure_name}.png")

    if L is None:
        L = _infer_L_from_path(path_dir)

    plot_3D_full_codec(
        path_dir=network_dir,
        filename=filename,
        path_out=path_out,
        figure_name=figure_name,
        L=L,
        nc=nc,
        time_base=time_base,
        show_base=show_base,
        positions_file=positions_file,
        force_rebuild_positions=force_rebuild_positions,
    )
    return path_out


def _choose_network_file(path_dir, filename=None):
    network_dir = os.path.join(path_dir, "network")
    if filename is not None:
        return filename

    files = [f for f in os.listdir(network_dir) if f.lower().endswith(".bin") or f.lower().endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo de rede encontrado em {network_dir}")

    # prefer non-percolation file if present
    non_perc = [f for f in files if "percolation" not in f.lower()]
    return sorted(non_perc)[0] if non_perc else sorted(files)[0]


def _choose_full_network_file_for_surface(path_dir, filename=None):
    network_dir = os.path.join(path_dir, "network")

    if filename is not None:
        base, ext = os.path.splitext(filename)
        candidates = []
        if base.endswith("_PERCOLATION"):
            candidates.append(base[:-len("_PERCOLATION")] + ext)
        if base.endswith("_percolation"):
            candidates.append(base[:-len("_percolation")] + ext)

        for candidate in candidates:
            if os.path.exists(os.path.join(network_dir, candidate)):
                return candidate

    return _choose_network_file(path_dir, None)


def plot_network_links_from_folder(path_dir,
                                   filename=None,
                                   path_out=None,
                                   figure_name="network_edges",
                                   line_width=0.6,
                                   alpha=0.8,
                                   edge_color_rule="source",
                                   max_edges=None,
                                   nc=None,
                                   show_base=False):
    """Plot network edges from a run folder with a network subfolder."""
    network_dir = os.path.join(path_dir, "network")
    filename = _choose_network_file(path_dir, filename)

    if path_out is None:
        path_out = os.path.join(path_dir, f"{figure_name}.png")

    fn = os.path.join(network_dir, filename)
    info = _read_compact_bin(fn)
    if info["E"] == 0:
        print(f"[WARN] O arquivo {fn} não contém arestas (E=0); pulando plot de edges.")
        return None

    plot_network_edges(
        path_dir=network_dir,
        filename=filename,
        figure_name=figure_name,
        line_width=line_width,
        alpha=alpha,
        edge_color_rule=edge_color_rule,
        max_edges=max_edges,
        nc=nc,
        show_base=show_base,
        path_out=path_out,
    )
    return path_out


def plot_run_surfaces_and_network(path_dir,
                                  surface_filename=None,
                                  network_filename=None,
                                  data_filename=None,
                                  output_dir=None,
                                  nc=4,
                                  L=None,
                                  show_base=False,
                                  outline_mode="full",
                                  visual_profile="full",
                                  edge_color_rule="source",
                                  max_edges=None):
    """Plot surfaces, active sites and network edges for a given run folder structure."""
    if output_dir is None:
        output_dir = path_dir
    os.makedirs(output_dir, exist_ok=True)

    surface_dir = os.path.join(path_dir, "data_surfaces")
    if surface_filename is None:
        surface_files = [f for f in os.listdir(surface_dir) if f.lower().endswith(".json")]
        if not surface_files:
            raise FileNotFoundError(f"Nenhum arquivo JSON de superfície encontrado em {surface_dir}")
        surface_filename = sorted(surface_files)[0]
    surface_json_path = os.path.join(surface_dir, surface_filename)

    data_dir = os.path.join(path_dir, "data")
    if data_filename is None:
        data_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".json")]
        data_filename = None
        for candidate in data_files:
            if os.path.splitext(candidate)[0] == os.path.splitext(surface_filename)[0]:
                data_filename = candidate
                break
        if data_filename is None and data_files:
            data_filename = sorted(data_files)[0]
    metadata = None
    if data_filename is not None:
        data_path = os.path.join(data_dir, data_filename)
        with open(data_path, "r") as f:
            metadata = json.load(f)
        print(f"Loaded data metadata from {data_path}")
    else:
        print(f"Nenhum arquivo JSON encontrado em {data_dir}")

    active_sites_path = os.path.join(output_dir, "network_active.png")
    active_sites_fig = plot_active_sites_from_folder(
        path_dir=path_dir,
        filename=network_filename,
        path_out=active_sites_path,
        figure_name="network_active",
        L=L,
        nc=nc,
        show_base=show_base,
    )

    network_path = None
    network_path = plot_network_links_from_folder(
        path_dir=path_dir,
        filename=network_filename,
        path_out=os.path.join(output_dir, "network_edges.png"),
        figure_name="network_edges",
        line_width=0.6,
        alpha=0.8,
        edge_color_rule=edge_color_rule,
        max_edges=max_edges,
        nc=nc,
        show_base=show_base,
    )

    surface_paths = plot_surfaces_from_json(
        path_surface_json=surface_json_path,
        path_out_dir=output_dir,
        cmap="viridis",
        figure_name="surfaces",
    )

    return {
        "active_sites": active_sites_fig,
        "network_edges": network_path,
        "surfaces": surface_paths,
        "metadata": metadata,
    }


def _rgb_tuple_to_plotly(rgb):
    vals = [int(round(255 * max(0.0, min(1.0, float(c))))) for c in rgb]
    return f"rgb({vals[0]},{vals[1]},{vals[2]})"


def _build_discrete_plotly_colorscale(color_values, nc):
    color_values = sorted(int(c) for c in color_values)
    if not color_values:
        return [[0.0, "rgb(128,128,128)"], [1.0, "rgb(128,128,128)"]]

    if min(color_values) == 0 and max(color_values) <= nc - 1:
        surface_palette = [
            (0.9, 0.1, 0.1),  # 0 - red
            (0.1, 0.1, 0.9),  # 1 - blue
            (0.1, 0.9, 0.1),  # 2 - green
            (1.0, 0.5, 0.0),  # 3 - orange
            (0.8, 0.2, 0.8),
            (0.2, 0.8, 0.8),
            (1.0, 1.0, 0.0),
            (0.6, 0.4, 0.2),
        ]
        if len(color_values) > len(surface_palette):
            raise ValueError(
                f"Há {len(color_values)} cores na superfície, mas apenas "
                f"{len(surface_palette)} cores foram definidas na paleta Plotly."
            )
        color_map = {c: surface_palette[c] for c in color_values}
    else:
        color_map = _build_fixed_color_map(color_values, nc)

    if len(color_values) == 1:
        color = _rgb_tuple_to_plotly(color_map[color_values[0]])
        return [[0.0, color], [1.0, color]]

    colorscale = []
    n_colors = len(color_values)
    for i, value in enumerate(color_values):
        color = _rgb_tuple_to_plotly(color_map[value])
        left = i / n_colors
        right = (i + 1) / n_colors
        colorscale.append([left, color])
        colorscale.append([right, color])

    return colorscale


def _normalize_surface_colors(C, color_values):
    color_values = sorted(int(c) for c in color_values)
    if not color_values:
        return np.zeros_like(C, dtype=float)

    if len(color_values) == 1:
        return np.zeros_like(C, dtype=float)

    idx_by_color = {value: i for i, value in enumerate(color_values)}
    C_norm = np.full(C.shape, np.nan, dtype=float)

    for value, idx in idx_by_color.items():
        C_norm[C == value] = idx / (len(color_values) - 1)

    return C_norm


def _surface_df_to_jupyter_grid(surface_df):
    for col in ("x", "y", "z"):
        if col not in surface_df.columns:
            raise ValueError(f"DataFrame de superfície sem coluna obrigatória: {col}")

    x = surface_df["x"].to_numpy().astype(int)
    y = surface_df["y"].to_numpy().astype(int)
    z = surface_df["z"].to_numpy().astype(float)

    Z = np.full((y.max() + 1, x.max() + 1), np.nan, dtype=float)
    Z[y, x] = z

    x_axis = np.arange(Z.shape[1])
    y_axis = np.arange(Z.shape[0])

    return x_axis, y_axis, Z


def _species_colorscale_like_notebook():
    return [
        [0.00, "red"],
        [0.249999, "red"],
        [0.25, "blue"],
        [0.499999, "blue"],
        [0.50, "green"],
        [0.749999, "green"],
        [0.75, "orange"],
        [1.00, "orange"],
    ]


def _normalize_species_like_notebook(C):
    cmin = np.nanmin(C)
    cmax = np.nanmax(C)
    if cmax == cmin:
        return np.zeros_like(C, dtype=float)
    return (C - cmin) / (cmax - cmin)


def plot_surface_3d_from_df(surface_df, nc=4, title="Network Surface 3D",
                            path_out_html=None, width=850, height=750,
                            showscale=False, fill_missing=True,
                            color_by="height", colorscale="Plasma"):
    """
    Plota uma superfície 3D interativa estilo Plotly usando z como altura e
    color como heatmap/cor discreta da espécie.
    """
    import plotly.graph_objects as go

    x_axis, y_axis, Z = _surface_df_to_jupyter_grid(surface_df)

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_axis,
                y=y_axis,
                z=Z,
            )
        ]
    )
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
        ),
    )

    if path_out_html is not None:
        out_dir = os.path.dirname(path_out_html)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.write_html(path_out_html)

    return fig


def plot_surface_3d_from_json_key(path_surface_json, surface_key="surface_posteq",
                                  path_out_html=None, nc=4, title=None,
                                  width=850, height=750, showscale=False,
                                  fill_missing=True, color_by="height",
                                  colorscale="Plasma"):
    """Plota uma única superfície 3D a partir de uma key do JSON em data_surfaces."""
    surfaces = load_surface_json(path_surface_json)

    if surface_key == "surface_preqteq":
        surface_key = "surface_preteq"

    if surface_key not in surfaces:
        available = ", ".join(sorted(surfaces))
        raise ValueError(
            f"Superfície {surface_key!r} não encontrada em {path_surface_json}. "
            f"Disponíveis: {available}"
        )

    if title is None:
        title = "Surface PRETEQ" if surface_key == "surface_preteq" else "Surface POSTEQ"

    return plot_surface_3d_from_df(
        surface_df=surfaces[surface_key],
        nc=nc,
        title=title,
        path_out_html=path_out_html,
        width=width,
        height=height,
        showscale=showscale,
        fill_missing=fill_missing,
        color_by=color_by,
        colorscale=colorscale,
    )


def _surface_array_to_jupyter_grid(surface):
    x, y, z, c = surface.T
    x = x.astype(int)
    y = y.astype(int)
    z = z.astype(float)

    Z = np.full((y.max() + 1, x.max() + 1), np.nan)
    Z[y, x] = z

    x_axis = np.arange(Z.shape[1])
    y_axis = np.arange(Z.shape[0])

    return x_axis, y_axis, Z


def _write_surface_png_fallback_matplotlib(x_axis, y_axis, Z, path_out_png, title):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    X, Y = np.meshgrid(x_axis, y_axis)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(surf, shrink=0.6, aspect=12)
    plt.tight_layout()
    plt.savefig(path_out_png, dpi=150)
    plt.close(fig)


def _write_surface_webgl_html(x_axis, y_axis, Z, path_out_html, title):
    Z_filled = pd.DataFrame(Z).interpolate(
        axis=0,
        limit_direction="both",
    ).interpolate(
        axis=1,
        limit_direction="both",
    ).to_numpy()

    x_vals = np.asarray(x_axis, dtype=float)
    y_vals = np.asarray(y_axis, dtype=float)
    X, Y = np.meshgrid(x_vals, y_vals)

    x_norm = (X - np.nanmean(x_vals)) / max(1.0, float(np.nanmax(x_vals) - np.nanmin(x_vals))) * 2.0
    y_norm = (Y - np.nanmean(y_vals)) / max(1.0, float(np.nanmax(y_vals) - np.nanmin(y_vals))) * 2.0
    z_norm = (Z_filled - float(np.nanmean(Z_filled))) / max(1.0, float(np.nanmax(Z_filled) - np.nanmin(Z_filled))) * 1.2

    vertices = np.column_stack([
        x_norm.ravel(),
        y_norm.ravel(),
        z_norm.ravel(),
        Z_filled.ravel(),
    ]).astype(float)

    ny, nx = Z_filled.shape
    triangles = []
    for yy in range(ny - 1):
        row = yy * nx
        next_row = (yy + 1) * nx
        for xx in range(nx - 1):
            a = row + xx
            b = row + xx + 1
            c = next_row + xx
            d = next_row + xx + 1
            triangles.extend([a, c, b, b, c, d])

    payload = {
        "title": title,
        "vertices": vertices.tolist(),
        "indices": triangles,
        "zmin": float(np.nanmin(Z_filled)),
        "zmax": float(np.nanmax(Z_filled)),
    }

    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #ffffff; }
    #title { position: fixed; top: 10px; left: 14px; z-index: 2; font: 16px sans-serif; color: #1f2937; }
    #hint { position: fixed; bottom: 10px; left: 14px; z-index: 2; font: 12px sans-serif; color: #4b5563; }
    canvas { width: 100vw; height: 100vh; display: block; }
  </style>
</head>
<body>
  <div id="title">__TITLE__</div>
  <div id="hint">drag: rotate | wheel: zoom</div>
  <canvas id="canvas"></canvas>
  <script id="surface-data" type="application/json">__DATA__</script>
  <script>
const data = JSON.parse(document.getElementById("surface-data").textContent);
const canvas = document.getElementById("canvas");
const gl = canvas.getContext("webgl2");
if (!gl) {
  document.body.innerHTML = "<p style='font:16px sans-serif;padding:24px'>WebGL2 não disponível neste navegador.</p>";
  throw new Error("WebGL2 unavailable");
}

const vsSource = `#version 300 es
in vec4 a_position;
uniform mat4 u_matrix;
uniform float u_zmin;
uniform float u_zmax;
out float v_t;
void main() {
  gl_Position = u_matrix * vec4(a_position.xyz, 1.0);
  v_t = clamp((a_position.w - u_zmin) / max(0.0001, u_zmax - u_zmin), 0.0, 1.0);
}`;

const fsSource = `#version 300 es
precision highp float;
in float v_t;
out vec4 outColor;
vec3 palette(float t) {
  vec3 a = vec3(0.267, 0.005, 0.329);
  vec3 b = vec3(0.128, 0.567, 0.551);
  vec3 c = vec3(0.993, 0.906, 0.144);
  if (t < 0.5) return mix(a, b, t * 2.0);
  return mix(b, c, (t - 0.5) * 2.0);
}
void main() {
  outColor = vec4(palette(v_t), 1.0);
}`;

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

const program = gl.createProgram();
gl.attachShader(program, compile(gl.VERTEX_SHADER, vsSource));
gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fsSource));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));

const vertices = new Float32Array(data.vertices.flat());
const indices = new Uint32Array(data.indices);

const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
const vbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
const ibo = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

const posLoc = gl.getAttribLocation(program, "a_position");
gl.enableVertexAttribArray(posLoc);
gl.vertexAttribPointer(posLoc, 4, gl.FLOAT, false, 16, 0);

const matrixLoc = gl.getUniformLocation(program, "u_matrix");
const zminLoc = gl.getUniformLocation(program, "u_zmin");
const zmaxLoc = gl.getUniformLocation(program, "u_zmax");

function mat4mul(a, b) {
  const out = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    out[c + r * 4] = a[r * 4] * b[c] + a[r * 4 + 1] * b[c + 4] + a[r * 4 + 2] * b[c + 8] + a[r * 4 + 3] * b[c + 12];
  }
  return out;
}
function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2), nf = 1 / (near - far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,(2*far*near)*nf,0]);
}
function translate(z) { return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,z,1]); }
function rotX(a) { const c=Math.cos(a), s=Math.sin(a); return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]); }
function rotZ(a) { const c=Math.cos(a), s=Math.sin(a); return new Float32Array([c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1]); }

let rx = -0.95, rz = -0.75, zoom = -3.4;
let dragging = false, lx = 0, ly = 0;
canvas.addEventListener("mousedown", e => { dragging = true; lx = e.clientX; ly = e.clientY; });
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", e => {
  if (!dragging) return;
  rz += (e.clientX - lx) * 0.01;
  rx += (e.clientY - ly) * 0.01;
  lx = e.clientX; ly = e.clientY;
  draw();
});
canvas.addEventListener("wheel", e => { e.preventDefault(); zoom += e.deltaY * 0.002; draw(); }, {passive:false});

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  gl.viewport(0, 0, canvas.width, canvas.height);
}

function draw() {
  resize();
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(1,1,1,1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(program);
  gl.bindVertexArray(vao);
  const p = perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100);
  let m = mat4mul(p, translate(zoom));
  m = mat4mul(m, rotX(rx));
  m = mat4mul(m, rotZ(rz));
  gl.uniformMatrix4fv(matrixLoc, false, m);
  gl.uniform1f(zminLoc, data.zmin);
  gl.uniform1f(zmaxLoc, data.zmax);
  gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_INT, 0);
}
window.addEventListener("resize", draw);
draw();
  </script>
</body>
</html>
"""

    html = html.replace("__TITLE__", title)
    html = html.replace("__DATA__", json.dumps(payload))

    out_dir = os.path.dirname(path_out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path_out_html, "w") as f:
        f.write(html)


def plot_jupyter_style_surface_from_json_key(path_surface_json,
                                             surface_key,
                                             path_out_html=None,
                                             path_out_png=None,
                                             title=None,
                                             width=800,
                                             height=700,
                                             show=False):
    """
    Reproduz diretamente o bloco do notebook Tests_Equilibration para uma key:
      with open(fn) ...
      surface = np.asarray(data[key])
      x, y, z, c = surface.T
      Z[y, x] = z
      go.Surface(x=x_axis, y=y_axis, z=Z)
    """
    import plotly.graph_objects as go

    with open(path_surface_json, "r") as f:
        data = json.load(f)

    if surface_key == "surface_preqteq":
        surface_key = "surface_preteq"

    if surface_key not in data:
        available = ", ".join(sorted(data.keys()))
        raise ValueError(
            f"Superfície {surface_key!r} não encontrada em {path_surface_json}. "
            f"Disponíveis: {available}"
        )

    surface = np.asarray(data[surface_key])
    x_axis, y_axis, Z = _surface_array_to_jupyter_grid(surface)

    if title is None:
        title = "Surface PRETEQ" if surface_key == "surface_preteq" else "Surface POSTEQ"

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_axis,
                y=y_axis,
                z=Z,
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
        ),
        width=width,
        height=height,
    )

    png_status = None

    if path_out_html is not None:
        out_dir = os.path.dirname(path_out_html)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        _write_surface_webgl_html(
            x_axis=x_axis,
            y_axis=y_axis,
            Z=Z,
            path_out_html=path_out_html,
            title=title,
        )

    if path_out_png is not None:
        out_dir = os.path.dirname(path_out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        try:
            fig.write_image(path_out_png)
            png_status = "plotly"
        except Exception as exc:
            _write_surface_png_fallback_matplotlib(
                x_axis=x_axis,
                y_axis=y_axis,
                Z=Z,
                path_out_png=path_out_png,
                title=title,
            )
            png_status = f"matplotlib_fallback: {exc}"

    if show:
        fig.show()

    return fig, png_status


def plot_jupyter_style_surfaces_from_json(path_surface_json,
                                          path_out_dir,
                                          width=800,
                                          height=700,
                                          show=False):
    os.makedirs(path_out_dir, exist_ok=True)

    outputs = {}

    for surface_key, stem, title in (
        ("surface_preteq", "04_surface_preteq_3d", "Surface PRETEQ"),
        ("surface_posteq", "04_surface_posteq_3d", "Surface POSTEQ"),
    ):
        path_out_html = os.path.join(path_out_dir, f"{stem}.html")
        path_out_png = os.path.join(path_out_dir, f"{stem}.png")

        _, png_status = plot_jupyter_style_surface_from_json_key(
            path_surface_json=path_surface_json,
            surface_key=surface_key,
            path_out_html=path_out_html,
            path_out_png=path_out_png,
            title=title,
            width=width,
            height=height,
            show=show,
        )

        outputs[surface_key] = {
            "html": path_out_html,
            "png": path_out_png,
            "png_status": png_status,
        }

    return outputs


def plot_surfaces_3d_from_json(path_surface_json, path_out_html=None, nc=4,
                               width=1500, height=700, showscale=False,
                               fill_missing=True, color_by="height",
                               colorscale="Plasma"):
    """
    Versão em função do bloco NETWORK SURFACE 3D do notebook:
    plota PRETEQ e POSTEQ como duas superfícies 3D com heatmap discreto.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    surfaces = load_surface_json(path_surface_json)
    if not surfaces:
        raise ValueError(f"Nenhum dado de superfície válido em {path_surface_json}")

    fig = make_subplots(
        rows=1,
        cols=len(surfaces),
        specs=[[{"type": "surface"} for _ in surfaces]],
        subplot_titles=[
            "Surface PRETEQ" if key == "surface_preteq" else "Surface POSTEQ"
            for key in surfaces.keys()
        ],
        horizontal_spacing=0.03,
    )

    for col, (key, df) in enumerate(surfaces.items(), start=1):
        x_axis, y_axis, Z = _surface_df_to_jupyter_grid(df)

        fig.add_trace(
            go.Surface(
                x=x_axis,
                y=y_axis,
                z=Z,
            ),
            row=1,
            col=col,
        )

    scene_common = dict(
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        zaxis=dict(title="Z"),
        aspectmode="cube",
    )
    layout = dict(width=width, height=height, margin=dict(l=0, r=0, b=0, t=40))
    for i in range(1, len(surfaces) + 1):
        layout["scene" if i == 1 else f"scene{i}"] = scene_common
    fig.update_layout(**layout)

    if path_out_html is not None:
        out_dir = os.path.dirname(path_out_html)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.write_html(path_out_html)

    return fig


def build_top_surface_from_active_sites(df):
    """
    Cria a malha z=max(z) para cada par (x,y), mantendo a cor do sítio no topo.
    Este é o análogo funcional do código do notebook com groupby(["y", "x"]).
    """
    for col in ("x", "y", "z", "color"):
        if col not in df.columns:
            raise ValueError(f"DataFrame sem coluna obrigatória: {col}")

    df_active = df[df["color"] > 0].copy()
    if df_active.empty:
        raise ValueError("Nenhum sítio ativo encontrado para montar a superfície.")

    top_idx = df_active.groupby(["y", "x"])["z"].idxmax()
    return df_active.loc[top_idx, ["x", "y", "z", "color"]].reset_index(drop=True)


def plot_top_surface_from_network(path_dir, filename, path_out_html=None,
                                  nc=4, positions_file=None,
                                  time_base=TIME_BASE_3D,
                                  force_rebuild_positions=False,
                                  title="Top Surface From Active Sites"):
    """Plota a superfície z=max(z) extraída diretamente dos sítios ativos."""
    df, _ = load_or_create_positions_codec(
        path_dir=path_dir,
        filename=filename,
        output_data=positions_file,
        time_base=time_base,
        force_rebuild=force_rebuild_positions,
    )
    surface_df = build_top_surface_from_active_sites(df)
    return plot_surface_3d_from_df(
        surface_df=surface_df,
        nc=nc,
        title=title,
        path_out_html=path_out_html,
    )


NETWORK_CALCULATIONS = {
    "active_sites_3d",
    "network_edges",
    "surface_heatmap",
    "surface_3d",
    "surface_posteq_3d",
    "active_sites_by_color",
}


def _normalize_network_calculations(calculations):
    aliases = {
        "1": "active_sites_3d",
        "01": "active_sites_3d",
        "active": "active_sites_3d",
        "active_sites": "active_sites_3d",
        "active_sites_3d": "active_sites_3d",
        "sitios": "active_sites_3d",
        "sítios": "active_sites_3d",
        "2": "network_edges",
        "02": "network_edges",
        "edges": "network_edges",
        "links": "network_edges",
        "network_edges": "network_edges",
        "ligacoes": "network_edges",
        "ligações": "network_edges",
        "3": "surface_heatmap",
        "03": "surface_heatmap",
        "heatmap": "surface_heatmap",
        "surface_heatmap": "surface_heatmap",
        "4": "surface_3d",
        "04": "surface_3d",
        "surface": "surface_3d",
        "surface3d": "surface_3d",
        "surface_3d": "surface_3d",
        "network_surface_3d": "surface_3d",
        "04b": "surface_posteq_3d",
        "posteq": "surface_posteq_3d",
        "surface_posteq": "surface_posteq_3d",
        "surface_posteq_3d": "surface_posteq_3d",
        "top": "surface_posteq_3d",
        "top_surface": "surface_posteq_3d",
        "active_top_surface": "surface_posteq_3d",
        "5": "active_sites_by_color",
        "05": "active_sites_by_color",
        "species": "active_sites_by_color",
        "colors": "active_sites_by_color",
        "clusters": "active_sites_by_color",
        "cores": "active_sites_by_color",
        "active_sites_by_color": "active_sites_by_color",
    }

    if calculations is None:
        return set(NETWORK_CALCULATIONS)

    if isinstance(calculations, str):
        calculations_clean = calculations.strip().lower()
        if calculations_clean in ("all", "todos", "*"):
            return set(NETWORK_CALCULATIONS)
        raw_calculations = [part.strip().lower() for part in calculations_clean.split(",")]
    elif isinstance(calculations, (int, np.integer)):
        raw_calculations = [str(int(calculations))]
    else:
        raw_calculations = []
        for calculation in calculations:
            if isinstance(calculation, (int, np.integer)):
                raw_calculations.append(str(int(calculation)))
            else:
                raw_calculations.append(str(calculation).strip().lower())

    selected = set()
    invalid = []
    for calculation in raw_calculations:
        if not calculation:
            continue
        normalized = aliases.get(calculation)
        if normalized is None:
            invalid.append(calculation)
        else:
            selected.add(normalized)

    if invalid:
        valid = ", ".join(sorted(NETWORK_CALCULATIONS))
        raise ValueError(f"Cálculo(s) inválido(s): {invalid}. Use: {valid} ou 'all'.")

    if not selected:
        raise ValueError("Nenhum cálculo selecionado.")

    return selected


def _normalize_network_blocks(blocks):
    return _normalize_network_calculations(blocks)


def plot_run_network_blocks(path_dir,
                            surface_filename=None,
                            network_filename=None,
                            output_dir=None,
                            nc=4,
                            blocks=None,
                            L=None,
                            time_base=TIME_BASE_3D,
                            show_base=False,
                            outline_mode="full",
                            visual_profile="full",
                            edge_color_rule="source",
                            max_edges=None,
                            colors_to_plot=None,
                            force_rebuild_positions=False,
                            calculations=None):
    """
    Separa a visualização de uma execução em cálculos:
      - active_sites_3d: plot 3D dos sítios ativados;
      - network_edges: rede de ligações;
      - surface_heatmap: surface com heatmap 2D;
      - surface_3d: surface 3D interativa análoga ao NETWORK SURFACE 3D do notebook;
      - surface_posteq_3d: surface 3D do topo POSTEQ, lida de data_surfaces/surface_posteq;
      - active_sites_by_color: um plot 3D por cor/cluster ativo.

    Args:
        calculations: cálculo(s) a executar. Exemplos:
            None ou "all" -> todos;
            "active_sites_3d" -> apenas sítios ativos;
            ["network_edges", "active_sites_by_color"] -> ligações e um plot por cor/cluster.
        blocks: nome antigo de calculations, mantido por compatibilidade.

    Returns:
        dict com os caminhos salvos apenas para os cálculos executados.
    """
    if blocks is not None and calculations is not None:
        raise ValueError("Use apenas 'calculations'. 'blocks' foi mantido só por compatibilidade.")

    selected_calculations = _normalize_network_calculations(
        calculations if calculations is not None else blocks
    )

    if output_dir is None:
        output_dir = os.path.join(path_dir, "network_blocks")
    os.makedirs(output_dir, exist_ok=True)

    needs_l = bool(selected_calculations & {"active_sites_3d", "active_sites_by_color"})
    needs_network = bool(selected_calculations & {"active_sites_3d", "network_edges", "active_sites_by_color"})
    needs_surface = bool(selected_calculations & {"surface_heatmap", "surface_3d", "surface_posteq_3d"})

    if L is None and needs_l:
        L = _infer_L_from_path(path_dir)

    network_dir = None
    positions_file = None
    if needs_network:
        network_dir = os.path.join(path_dir, "network")
        network_filename = _choose_network_file(path_dir, network_filename)
        positions_file = os.path.join(
            output_dir,
            f"{os.path.splitext(network_filename)[0]}_positions.parquet",
        )

    surface_json_path = None
    if needs_surface:
        surface_dir = os.path.join(path_dir, "data_surfaces")
        if surface_filename is None:
            surface_files = [f for f in os.listdir(surface_dir) if f.lower().endswith(".json")]
            if not surface_files:
                raise FileNotFoundError(f"Nenhum arquivo JSON de superfície encontrado em {surface_dir}")
            surface_filename = sorted(surface_files)[0]
        surface_json_path = os.path.join(surface_dir, surface_filename)

    results = {}

    if "active_sites_3d" in selected_calculations:
        active_sites_path = os.path.join(output_dir, "01_active_sites_3d.png")
        plot_3D_full_codec(
            path_dir=network_dir,
            filename=network_filename,
            path_out=active_sites_path,
            figure_name="block_01_active_sites_3d",
            L=L,
            nc=nc,
            time_base=time_base,
            show_base=show_base,
            positions_file=positions_file,
            force_rebuild_positions=force_rebuild_positions,
        )
        results["01_active_sites_3d"] = active_sites_path

    if "network_edges" in selected_calculations:
        network_edges_path = plot_network_links_from_folder(
            path_dir=path_dir,
            filename=network_filename,
            path_out=os.path.join(output_dir, "02_network_edges.png"),
            figure_name="block_02_network_edges",
            edge_color_rule=edge_color_rule,
            max_edges=max_edges,
            nc=nc,
            show_base=show_base,
        )
        results["02_network_edges"] = network_edges_path

    if "surface_heatmap" in selected_calculations:
        surface_heatmap_paths = plot_surfaces_from_json(
            path_surface_json=surface_json_path,
            path_out_dir=output_dir,
            cmap="viridis",
            figure_name="03_surface_heatmap",
        )
        results["03_surface_heatmap"] = surface_heatmap_paths

    if "surface_3d" in selected_calculations:
        surface_3d_paths = plot_jupyter_style_surfaces_from_json(
            path_surface_json=surface_json_path,
            path_out_dir=output_dir,
            width=800,
            height=700,
        )
        results["04_jupyter_style_surfaces_3d"] = surface_3d_paths

    if "surface_posteq_3d" in selected_calculations:
        top_surface_path = os.path.join(output_dir, "04b_surface_posteq_3d.html")
        fig_posteq = plot_surface_3d_from_json_key(
            path_surface_json=surface_json_path,
            surface_key="surface_posteq",
            path_out_html=top_surface_path,
            nc=nc,
            title="Surface POSTEQ",
        )
        legacy_top_surface_path = os.path.join(output_dir, "04b_active_sites_top_surface_3d.html")
        fig_posteq.write_html(legacy_top_surface_path)
        results["04b_surface_posteq_3d"] = top_surface_path
        results["04b_legacy_top_surface_3d"] = legacy_top_surface_path

    if "active_sites_by_color" in selected_calculations:
        species_paths = plot_3D_full_codec_by_species(
            path_dir=network_dir,
            filename=network_filename,
            path_out_dir=output_dir,
            figure_name="block_05_active_sites_by_color",
            L=L,
            nc=nc,
            time_base=time_base,
            show_base=show_base,
            positions_file=positions_file,
            force_rebuild_positions=force_rebuild_positions,
            colors_to_plot=colors_to_plot,
            outline_mode=outline_mode,
            visual_profile=visual_profile,
            filename_prefix="05_active_cluster",
            filename_by_percolation_order=True,
        )
        results["05_active_sites_by_color"] = species_paths

    if positions_file is not None:
        results["positions"] = positions_file

    return results


def plot_3D_full_codec_by_species(path_dir, filename, path_out_dir, figure_name,
                                  L, nc, seed=None,
                                  time_base=TIME_BASE_3D,
                                  show_base=False,
                                  positions_file=None,
                                  force_rebuild_positions=False,
                                  colors_to_plot=None,
                                  outline_mode="full",
                                  visual_profile="full",
                                  filename_prefix=None,
                                  filename_by_percolation_order=False):
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
        filename_by_percolation_order: se True, salva como i_{order}.png
            usando o JSON em ../data correspondente ao arquivo da rede.

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

    order_by_color = {}
    if filename_by_percolation_order:
        percolation_json = _find_percolation_json_for_network(path_dir, filename)
        order_by_color = _read_percolation_order_by_color(percolation_json)
        if not order_by_color:
            print(
                "[WARN] Não foi possível ler a ordem de percolação no JSON. "
                "Usando nomes por espécie/cor."
            )

    saved_paths = {}

    for color in colors_to_plot:
        if color not in unique_colors:
            continue

        df_color = df[df["color"].astype(int) == color].copy()

        if df_color.empty:
            continue

        order = order_by_color.get(color)
        if filename_by_percolation_order and order is not None:
            out_filename = f"i_{order}.png"
        else:
            out_filename = f"{filename_prefix}_species_{color}.png"

        path_out = os.path.join(path_out_dir, out_filename)

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


def assemble_animation_from_frames(
    frames_dir,
    output_video_path=None,
    fps=10,
    frame_pattern="frame_*.png",
    codec="libx264",
    crf=18,
    preset="slow",
    scale=None,
    verbose=False,
):
    """
    Monta uma animação a partir de frames PNG usando FFmpeg.

    Características:
    - Processa frames em stream (eficiente de memória)
    - Suporta diferentes codecs (libx264, libx265, libvpx-vp9)
    - CRF ajustável (0-51: 0=lossless, 18=visualmente lossless, 51=pior)
    - Preset de velocidade/qualidade (ultrafast...veryslow)
    - Escalamento opcional
    - Sem limite de RAM independentemente da quantidade de frames

    Args:
        frames_dir: diretório contendo os frames PNG
        output_video_path: caminho do vídeo de saída (default: frames_dir/output.mp4)
        fps: frames por segundo (default: 10)
        frame_pattern: glob pattern para encontrar frames (default: frame_*.png)
        codec: 'libx264' (H.264, default), 'libx265' (H.265), 'libvpx-vp9' (VP9)
        crf: 0-51 para H.264/H.265 (18=visualmente lossless, default)
               0-63 para VP9 (15=visualmente lossless)
        preset: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        scale: tuple (width, height) ou None para manter resolução original
        verbose: True para mostrar saída do FFmpeg

    Returns:
        dict com informações sobre o vídeo criado
    """
    import glob

    frames_dir = os.path.abspath(frames_dir)
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Diretório de frames não existe: {frames_dir}")

    # Encontrar frames
    pattern = os.path.join(frames_dir, frame_pattern)
    frames = sorted(glob.glob(pattern))

    if not frames:
        raise FileNotFoundError(f"Nenhum frame encontrado com padrão: {pattern}")

    print(f"[assemble_animation] Encontrados {len(frames)} frames em {frames_dir}")

    if output_video_path is None:
        base_name = f"output_{codec}_{crf}_{preset}"
        ext = ".webm" if codec == "libvpx-vp9" else ".mp4"
        output_video_path = os.path.join(frames_dir, f"{base_name}{ext}")

    output_video_path = os.path.abspath(output_video_path)
    out_dir = os.path.dirname(output_video_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Validar codec
    valid_codecs = {"libx264", "libx265", "libvpx-vp9"}
    if codec not in valid_codecs:
        raise ValueError(f"Codec inválido: {codec}. Deve ser um de: {valid_codecs}")

    # Validar CRF
    if codec == "libvpx-vp9":
        crf_max = 63
    else:
        crf_max = 51
    crf = int(crf)
    if crf < 0 or crf > crf_max:
        raise ValueError(f"CRF deve estar entre 0 e {crf_max}, recebido: {crf}")

    # Construir comando FFmpeg
    # Usar stdin para passar a lista de frames (mais robusto que glob)
    first_frame = frames[0]
    frame_dir_abs = os.path.dirname(first_frame)
    frame_base = os.path.basename(first_frame)

    # Extrair padrão do nome do frame (ex: "frame_000000_t000000.png")
    # Assumir formato "frame_XXXXXX_tYYYYYY.png" e usar glob no FFmpeg
    frame_spec = os.path.join(frame_dir_abs, frame_pattern)

    cmd = [
        "ffmpeg",
        "-framerate", str(int(fps)),
        "-pattern_type", "glob",
        "-i", frame_spec,
        "-vf", "format=rgb8",
        "-c:v", codec,
    ]

    # Adicionar opções específicas do codec
    if codec == "libx264":
        cmd.extend(["-crf", str(crf), "-preset", preset])
    elif codec == "libx265":
        cmd.extend(["-crf", str(crf), "-preset", preset])
    elif codec == "libvpx-vp9":
        cmd.extend(["-crf", str(crf), "-b:v", "0", "-deadline", "good"])

    # Adicionar escala se especificada
    if scale is not None:
        w, h = int(scale[0]), int(scale[1])
        cmd.extend(["-vf", f"scale={w}:{h}:flags=lanczos"])

    # Opções gerais
    cmd.extend([
        "-y",  # sobrescrever sem perguntar
        output_video_path,
    ])

    if verbose:
        print(f"[ffmpeg] Comando: {' '.join(cmd)}")
    else:
        progress_insert_idx = cmd.index(output_video_path)
        cmd_with_progress = (
            cmd[:progress_insert_idx]
            + ["-nostats", "-progress", "pipe:1", "-loglevel", "error"]
            + cmd[progress_insert_idx:]
        )
        cmd_without_progress = (
            cmd[:progress_insert_idx]
            + ["-loglevel", "error"]
            + cmd[progress_insert_idx:]
        )
        cmd = cmd_with_progress

    print(f"[assemble_animation] Codificando {len(frames)} frames com {codec} (CRF={crf})...")
    print(f"[assemble_animation] Progresso: 0/{len(frames)} frames processados (0.0%) - faltam {len(frames)}")

    try:
        if verbose:
            subprocess.run(cmd, check=True)
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            last_reported = 0
            try:
                for line in process.stdout:
                    if not line:
                        continue
                    if line.startswith("frame="):
                        try:
                            frame_num = int(line.split("=", 1)[1].split()[0])
                        except ValueError:
                            continue

                        processed = min(frame_num, len(frames))
                        remaining = max(len(frames) - processed, 0)
                        percent = (processed / len(frames) * 100.0) if len(frames) else 100.0

                        if processed != last_reported:
                            print(
                                f"[assemble_animation] Progresso: {processed}/{len(frames)} frames processados "
                                f"({percent:.1f}%) - faltam {remaining}",
                                flush=True,
                            )
                            last_reported = processed
            finally:
                return_code = process.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
    except subprocess.CalledProcessError as e:
        if not verbose:
            print("[assemble_animation] O monitoramento de progresso causou falha no FFmpeg; tentando novamente sem a saída de progresso.")
            subprocess.run(cmd_without_progress, check=True, capture_output=True)
        else:
            raise RuntimeError(f"Falha ao executar FFmpeg: {e}")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg não encontrado. Instale com: sudo apt-get install ffmpeg"
        )

    file_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)

    result = {
        "video_path": output_video_path,
        "frames_count": len(frames),
        "fps": int(fps),
        "codec": codec,
        "crf": int(crf),
        "preset": preset,
        "file_size_mb": round(file_size_mb, 2),
        "duration_seconds": round(len(frames) / int(fps), 2),
    }

    print(f"[assemble_animation] Vídeo criado: {output_video_path}")
    print(f"  - Tamanho: {result['file_size_mb']} MB")
    print(f"  - Duração: {result['duration_seconds']} s")
    print(f"  - Frames: {len(frames)} @ {fps} FPS")

    return result


def create_multiple_quality_videos(
    frames_dir,
    output_base_dir=None,
    fps=10,
    frame_pattern="frame_*.png",
    quality_presets=None,
):
    """
    Cria múltiplas versões da animação com diferentes qualidades.

    Útil para ter uma versão de alta qualidade (arquivo grande) e versões
    comprimidas para compartilhamento rápido.

    Args:
        frames_dir: diretório contendo os frames PNG
        output_base_dir: diretório base para os vídeos (default: frames_dir)
        fps: frames por segundo
        frame_pattern: glob pattern para encontrar frames
        quality_presets: lista de dicts com configurações
            Default:
            [
                {"name": "hq", "codec": "libx265", "crf": 15, "preset": "slow"},
                {"name": "mq", "codec": "libx264", "crf": 23, "preset": "medium"},
                {"name": "lq", "codec": "libx264", "crf": 28, "preset": "fast"},
            ]

    Returns:
        dict com caminhos dos vídeos criados por qualidade
    """
    if quality_presets is None:
        quality_presets = [
            {"name": "hq", "codec": "libx265", "crf": 15, "preset": "slow"},
            {"name": "mq", "codec": "libx264", "crf": 23, "preset": "medium"},
            {"name": "lq", "codec": "libx264", "crf": 28, "preset": "fast"},
        ]

    if output_base_dir is None:
        output_base_dir = frames_dir

    videos = {}

    for preset_cfg in quality_presets:
        name = preset_cfg.get("name", "video")
        codec = preset_cfg.get("codec", "libx264")
        crf = preset_cfg.get("crf", 23)
        preset = preset_cfg.get("preset", "medium")

        ext = ".webm" if codec == "libvpx-vp9" else ".mp4"
        output_path = os.path.join(output_base_dir, f"{name}{ext}")

        print(f"\n[quality_preset] Criando versão '{name}'...")
        result = assemble_animation_from_frames(
            frames_dir=frames_dir,
            output_video_path=output_path,
            fps=fps,
            frame_pattern=frame_pattern,
            codec=codec,
            crf=crf,
            preset=preset,
            verbose=False,
        )

        videos[name] = result

    print(f"\n[quality_presets] Resumo de vídeos criados:")
    for name, info in videos.items():
        print(f"  {name:8s}: {info['file_size_mb']:8.2f} MB - {info['video_path']}")

    return videos
