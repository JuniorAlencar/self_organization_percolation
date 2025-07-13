import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
#from matplotlib.colors import ListedColormap

def list_npz_files():
    """
    Lists all '.npz' files in the specified directory.
    Returns:
        List of filenames ending with '.npz'
    Raises:
        FileNotFoundError: If the target directory does not exist.
    """
    folder = "../Data/bond_percolation/dim_2/L_1000_N_samples_1000/NT_constant/NT_200/k_1.0e-05/network/"
    
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

def plot_bond_network(filepath, savepath=None, dpi=600, min_density=1):
    """
    Plots only the bond connections (black segments) between active sites in the lattice.
    The image is vertically cropped to include only rows with density ≥ min_density.
    The y-axis is inverted (matching the orientation in the original paper).
    """
    data = np.load(filepath)
    network = data["network"]
    nrows, ncols = network.shape

    segments = []
    density_per_row = np.zeros(nrows, dtype=int)

    for i in range(nrows):
        for j in range(ncols):
            if network[i, j] == 1:
                # right bond
                if j + 1 < ncols and network[i, j + 1] == 1:
                    segments.append([(j, i), (j + 1, i)])
                    density_per_row[i] += 1
                # upward bond
                if i + 1 < nrows and network[i + 1, j] == 1:
                    segments.append([(j, i), (j, i + 1)])
                    density_per_row[i] += 1

    active_rows = np.where(density_per_row >= min_density)[0]
    if active_rows.size == 0:
        print("[!] No row found with minimum density.")
        return
    row_start = active_rows[0]
    row_end = active_rows[-1]

    # Filter visible bonds
    segments_filtered = [
        [(x0, y0), (x1, y1)]
        for (x0, y0), (x1, y1) in segments
        if row_start <= y0 <= row_end and row_start <= y1 <= row_end
    ]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    lc = LineCollection(segments_filtered, colors="black", linewidths=0.25)
    ax.add_collection(lc)
    ax.set_xlim(0, ncols)
    ax.set_ylim(row_end, row_start + 1)
    ax.invert_yaxis()  # ensure top-to-bottom growth
    ax.set_aspect("equal")
    ax.axis("off")
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
    
    folder = f"../Data/{type_percolation}_percolation/dim_{dim}/L_{L}_N_samples_{N_samples}/NT_{typeNt}/NT_{Nt}/k_{k:.1e}/{folder_prop}/"

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    npy_files = [folder + f for f in os.listdir(folder) if f.endswith('.npy')]
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