import os

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