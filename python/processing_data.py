from src.process_data import *

# ====== Simulation parameters ======
type_perc = 'bond'
dim = 2
num_colors = 4
L_lst  = [512, 1024, 2048, 4096]
Nt_lst = [50, 100, 200, 400]
k_lst  = [4.0e-04, 2.0e-04, 9.0e-05, 5.0e-05]
# ===========================

# List all values of rho inside folder with parameters above
for L, Nt, k in zip(L_lst, Nt_lst, k_lst):
    rho_values = list_rho_values(type_perc, num_colors, dim, L, Nt, k, base_root="../Data")
    
    # Creating file with all data processing
    for rho in rho_values:
        path_data = f"../Data/{type_perc}_percolation/num_colors_{num_colors}/dim_{dim}/L_{L}/NT_constant/NT_{Nt}/k_{k:.1e}/rho_{rho:.4e}/data/"

        all_data = glob.glob(os.path.join(path_data, "*.json"))
        print(len(all_data))
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
# for num_colors in num_colors_lst:
#     for L, k, Nt in zip(L_lst, k_lst, Nt_lst):
#df_all = join_all_data(type_perc, num_colors, dim, L, Nt, k, base_root="../Data")

#df_all_colors = processing_data_nc(L_lst, Nt_lst, k_lst, num_colors_lst[2], dim[0], type_perc)



print("DataFrame Create!")



