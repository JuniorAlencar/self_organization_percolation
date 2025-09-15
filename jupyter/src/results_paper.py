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

    # Define o conjunto de valores ativos conforme a regra solicitada
    if num_colors == 1:
        active_values = (1,)
    else:
        active_values = tuple(range(2, 2 + num_colors))

    data = np.load(filepath)
    network = data["network"].T  # “em pé”
    nrows, ncols = network.shape

    # Paleta padrão
    if color_map is None:
        if num_colors == 1:
            color_map = {1: "0.4"}  # cinza
        else:
            base = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple",
                    "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
            color_map = {val: base[(i % len(base))] for i, val in enumerate(active_values)}

    segments_by_color = {val: [] for val in active_values}
    density_per_row = np.zeros(nrows, dtype=int)

    # Varre e cria segmentos apenas se vizinho tem o MESMO valor ativo
    for i in range(nrows):
        for j in range(ncols):
            v = network[i, j]
            if v in active_values:
                # horizontal (direita)
                if j + 1 < ncols and network[i, j + 1] == v:
                    segments_by_color[v].append([(j, i), (j + 1, i)])
                    density_per_row[i] += 1
                # vertical (cima)
                if i + 1 < nrows and network[i + 1, j] == v:
                    segments_by_color[v].append([(j, i), (j, i + 1)])
                    density_per_row[i] += 1

    active_rows = np.where(density_per_row >= min_density)[0]
    if active_rows.size == 0:
        print("[!] No row found with minimum density.")
        return
    row_start, row_end = active_rows[0], active_rows[-1]

    # Recorte vertical
    for val in active_values:
        segments_by_color[val] = [
            [(x0, y0), (x1, y1)]
            for (x0, y0), (x1, y1) in segments_by_color[val]
            if row_start <= y0 <= row_end and row_start <= y1 <= row_end
        ]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    handles, labels = [], []
    for val in active_values:
        segs = segments_by_color[val]
        if not segs:
            continue
        lc = LineCollection(segs, colors=color_map.get(val, "black"),
                            linewidths=linewidth, label=f"+{val}")
        ax.add_collection(lc)
        handles.append(lc)
        labels.append(f"+{val}")

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

def plot_N_vs_t(npy_path, savepath=None):
    """
    Plota N(t) versus t a partir de um arquivo .npy contendo duas colunas: [t, N_t]
    
    Parâmetros:
        npy_path (str): caminho do arquivo .npy
        savepath (str, opcional): caminho para salvar a figura (formato .png, .pdf, etc.)
    """
    try:
        data = np.load(npy_path)
        
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"Formato inválido: esperado array (n, 2), recebido {data.shape}")
        
        t = data[:, 0]
        N_t = data[:, 1]
        
        plt.figure(figsize=(6, 4))
        plt.plot(t, N_t, '-o', markersize=2, linewidth=1, color='black')
        plt.xlabel("t")
        plt.ylabel("N(t)")
        plt.title("Crescimento do número de sítios ativos")
        plt.grid(True)
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=300)
            print(f"Figura salva em: {savepath}")
        else:
            plt.show()
    
    except Exception as e:
        print(f"[Erro] Falha ao processar o arquivo '{npy_path}':\n{e}")

def list_npy_files(type_percolation:str, dim:int, L:int, N_samples:int, type_Nt:int, Nt:int, k:float, prop:str):
    """
    Lists all '.npy' files in the specified directory.
    Returns:
        List of filenames ending with '.npy'
    Raises:
        FileNotFoundError: If the target directory does not exist.
    """
    if type_Nt == 0:
        typeNt = "constant"
    elif type_Nt == 1:
        typeNt = "variable"
    
    if prop == "Nt":
        folder_prop = "N_versus_t"
    elif prop == "Pt":
        folder_prop = "p_t"
    else:
        raise ValueError(f"[prop] must be 'Nt' or 'Pt'. Received: '{prop}'")
    
    folder = f"../Data/{type_percolation}_percolation/num_colors_1/dim_{dim}/L_{L}/NT_{typeNt}/NT_{Nt}/k_{k:.1e}/{folder_prop}/"

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    npy_files = [folder + f for f in os.listdir(folder) if f.endswith('.csv')]
    return npy_files

def load_t_pt(filepath):
    """
    Carrega um arquivo .npy contendo duas colunas: t e p_t.

    Parâmetros:
        filepath (str): Caminho para o arquivo .npy.

    Retorna:
        t (np.ndarray): Vetor de tempos.
        p_t (np.ndarray): Vetor de probabilidades p(t).
    """
    data = np.load(filepath)
    
    if data.shape[1] != 2:
        raise ValueError(f"Esperado 2 colunas (t, p_t), mas encontrado {data.shape[1]}")
    
    t = data[:, 0]
    p_t = data[:, 1]
    return t, p_t


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
