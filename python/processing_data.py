from src.process_data import *

# ====== Simulation parameters ======
type_perc = 'bond'
num_colors = 3
dim = 2
L = 128
Nt = 1500
k = 1.0e-04

# ===========================

# List all values of rho inside folder with parameters above
rho_values = list_rho_values(type_perc, num_colors, dim, L, Nt, k, base_root="../Data")

# Creating file with all data processing
for rho in rho_values:
    path_data = f"../Data/{type_perc}_percolation/num_colors_{num_colors}/dim_{dim}/L_{L}/NT_constant/NT_{Nt}/k_{k:.1e}/rho_{rho:.1e}/data"

    all_data = glob.glob(os.path.join(path_data, "*.json"))

    out_dir = Path(path_data)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dat = out_dir / "all_data.dat"
    out_txt = out_dir / "process_names.txt"

    df, processed = saving_data(
        all_data=all_data,
        output_data=out_dat,
        output_names=out_txt,
        burn_in_frac=0.20,
        verbose=True,
        path_hint=path_data,
        force_recompute=True,   # <<<<<<<<<< força recalcular
    )

    print("Salvo em:", out_dat.resolve())
    print("Processados:", len(processed))
print("All files processing")

# ----------------------
print("CREATING DATAFRAME...")

df_all = join_all_data(type_perc, num_colors, dim, L, Nt, k, base_root="../Data")

print("DataFrame Create!")