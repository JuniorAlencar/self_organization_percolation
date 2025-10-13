import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
#from matplotlib.colors import ListedColormap

def list_npz_files(folder):
    """
    Lists all '.npz' files in the specified directory.
    Returns:
        List of filenames ending with '.npz'
    Raises:
        FileNotFoundError: If the target directory does not exist.
    """
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    npz_files = [f for f in os.listdir(folder) if f.endswith('.npz')]
    return npz_files

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_bond_network(
    filepath,
    num_colors: int,
    savepath=None,
    dpi=600,
    min_density=1,
    color_map=None,          # dict[int,str] opcional: {valor_ativo: cor}
    linewidth=0.25,
    figsize=(8, 10),
    show_legend=True
):
    """
    Plota ligações entre sítios ativos de MESMA cor, conforme num_colors.

    Regras:
      - num_colors = 1 -> ativos = {+1} (cor cinza por padrão)
      - num_colors >= 2 -> ativos = {+2, +3, ..., +(num_colors+1)}
      - valores negativos e -1 são ignorados
    """
    if num_colors < 1:
        raise ValueError("num_colors deve ser >= 1")

    # --------- valores ativos (positivos) ---------
    if num_colors == 1:
        active_values = (1,)
    else:
        active_values = tuple(range(2, 2 + num_colors))

    # --------- carregar arquivo ---------
    def _load_network_2d(path):
        """
        Retorna matriz 2D (nrows, ncols) de ints.
        Aceita:
          - .npz com chaves: data(1D), shape(1D), dim(esc.)   [novo formato]
          - .npz com 'network' 2D (legado)
          - .npy 2D
        """
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
            # Novo formato (.npz) com {data, shape, dim}
            if {"data", "shape", "dim"} <= keys:
                dim = int(np.array(z["dim"]).item())
                if dim != 2:
                    raise ValueError(f"plot_bond_network: dim={dim} não suportado para este plot (apenas 2D).")
                shape = np.asarray(z["shape"]).astype(int)
                if shape.ndim != 1 or shape.size != 2:
                    raise ValueError(f"shape inválido: {shape}")
                data = np.asarray(z["data"]).astype(np.int32, copy=False)
                if data.size != int(shape[0]) * int(shape[1]):
                    raise ValueError("Tamanho de 'data' inconsistente com 'shape'.")
                net = data.reshape(int(shape[0]), int(shape[1]))  # C-order
                return net  # já em (nrows, ncols)

            # Legado: .npz com 'network' 2D
            if "network" in keys:
                net = np.array(z["network"])
                if net.ndim != 2:
                    raise ValueError("network não é 2D.")
                # Seu código antigo transpunha: network = data["network"].T
                # Mantemos o mesmo comportamento para compatibilidade:
                return net.T

            # Caso peculiar: .npz com um único array não padronizado
            if len(keys) == 1:
                arr = np.array(z[list(keys)[0]])
                if arr.ndim == 2:
                    return arr
                raise ValueError("Arquivo .npz não possui chaves esperadas (data/shape/dim ou network).")

        # Se for .npy (np.load em caminho .npy não retorna dict, cai fora do with)
        # Tenta como .npy:
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError("Arquivo .npy precisa ser 2D para este plot.")
        return arr

    network = _load_network_2d(filepath)
    nrows, ncols = network.shape

    # --------- paleta ---------
    if color_map is None:
        if num_colors == 1:
            color_map = {1: "0.4"}  # cinza
        else:
            base = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple",
                    "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
            color_map = {val: base[(i % len(base))] for i, val in enumerate(active_values)}

    segments_by_color = {val: [] for val in active_values}
    density_per_row = np.zeros(nrows, dtype=int)

    # --------- varredura e criação de segmentos (apenas ligações MESMA cor ativa) ---------
    # Observação: ligações horizontais e verticais (4-vizinhança)
    for i in range(nrows):
        row = network[i]
        for j in range(ncols):
            v = row[j]
            if v in active_values:
                # horizontal (direita)
                if j + 1 < ncols and row[j + 1] == v:
                    segments_by_color[v].append([(j, i), (j + 1, i)])
                    density_per_row[i] += 1
                # vertical (baixo)
                if i + 1 < nrows and network[i + 1, j] == v:
                    segments_by_color[v].append([(j, i), (j, i + 1)])
                    density_per_row[i] += 1

    # --------- recorte por densidade mínima ---------
    active_rows = np.where(density_per_row >= min_density)[0]
    if active_rows.size == 0:
        print("[!] No row found with minimum density.")
        return
    row_start, row_end = active_rows[0], active_rows[-1]

    # filtra segmentos dentro do recorte vertical
    for val in active_values:
        segs = segments_by_color[val]
        if segs:
            segments_by_color[val] = [
                [(x0, y0), (x1, y1)]
                for (x0, y0), (x1, y1) in segs
                if row_start <= y0 <= row_end and row_start <= y1 <= row_end
            ]

    # --------- plot ---------
    fig, ax = plt.subplots(figsize=figsize)
    handles, labels = [], []
    for val in active_values:
        segs = segments_by_color[val]
        if not segs:
            continue
        lc = LineCollection(segs, colors=color_map.get(val, "black"),
                            linewidths=linewidth, label=f"{val:+d}")
        ax.add_collection(lc)
        handles.append(lc)
        labels.append(f"{val:+d}")

    ax.set_xlim(0, ncols)
    ax.set_ylim(row_end, row_start)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    # if show_legend and handles:
    #     ax.legend(handles=handles, labels=labels, loc="upper right",
    #               frameon=False, fontsize=8)

    plt.tight_layout(pad=0)
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight", pad_inches=0)
        print(f"[✔] Image saved to: {savepath}")
    else:
        plt.show()


import os, json
import numpy as np

def read_orders_one_file(file_path):
    """
    Lê o JSON e retorna lista de tuplas (order, pt_array, nt_array_ou_None).
    Aceita arquivos com 1..N ordens.
    """
    with open(file_path, "r") as f:
        obj = json.load(f)

    out = []
    results = obj.get("results", []) if isinstance(obj, dict) else []
    for item in results:
        order = item.get("order_percolation")
        d = item.get("data", {}) or {}
        if order is None or "pt" not in d:
            continue
        p = np.asarray(d["pt"], dtype=float)
        if p.size == 0:
            continue
        n_arr = np.asarray(d["nt"], dtype=float) if "nt" in d else None
        if n_arr is not None:
            n = min(len(p), len(n_arr))
            p = p[:n]
            n_arr = n_arr[:n]
        out.append((int(order), p, n_arr))
    return out

def data_single_sample(type_perc, num_colors, dim, L, Nt, k, rho, p0, seed):
    """
    Retorna dict com 't' e chaves 'p_i'/'N_i' APENAS para as ordens existentes no arquivo.
    Não assume que existem 4 ordens.
    """
    path = (f"/home/junior/Documents/self_organization_percolation/Data/"
            f"{type_perc}_percolation/num_colors_{int(num_colors)}/dim_{dim}/L_{L}/"
            f"NT_constant/NT_{Nt}/k_{k:.1e}/rho_{rho:.4e}/data/")
    filename = f"P0_0.10_p0_{p0:.2f}_seed_{seed}.json"
    file_path = os.path.join(path, filename)

    data_list = read_orders_one_file(file_path)
    if not data_list:
        raise ValueError(f"Nenhuma ordem encontrada em: {file_path}")

    # ordena por ordem e usa o comprimento da primeira série para o 't'
    data_list.sort(key=lambda x: x[0])
    T = len(data_list[0][1])
    out = {"t": list(range(T))}

    for order, p_arr, n_arr in data_list:
        out[f"p_{order}"] = [float(x) for x in p_arr[:T]]
        if n_arr is not None:
            # 'nt' pode vir como float; converta com segurança
            out[f"N_{order}"] = [int(x) for x in np.asarray(n_arr[:T]).round()]

    return out
