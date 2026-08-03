from src.run_samples_functions import shell_data, custom_range
from src.SOP_parms import *
import numpy as np
import pandas as pd
# L = 128 => Ns = 700
# L = 192 => Ns = 600
# L = 256 => Ns = 500
# L = 384 => Ns = 300
# L = 512 => Ns = 100
# L = 768 => Ns = 50
# L = 1024 => Ns = 20

seed = -1
dim = 2
#nc=2

type_perc = 'bond'
L_lst = [2048, 4096, 8192, 16384]
num_runs = [400, 200, 100, 50]
#L_lst = [1024]
#num_runs = [400]
# nc = 4
#L_lst = [128, 256, 512, 1024]
#num_runs = [300, 150, 50, 5]

#L_lst = [256]
#num_runs = [150]
nc = 1
#c_lst = [0.01, 0.05, 0.1, 0.15, 0.2]
c_lst = [0.02, 0.03, 0.04, 0.06, 0.07, 0.8, 0.9]
multi=True
Equilibration = 'false'
Properties = 'false'
Mode = 'growth_test'  # use 'sop' for the original fixed-height SOP run
InitialLayout = 'random'  # 'random', 'blocks', or 'alternating'
p0 = 0.8
P0 = 0.2
#ft_lst = np.linspace(0.001, 0.4, 10)
#P0_lst = [round(i,1) for i in np.arange(start=0.1,stop=1.1,step=0.1)]


# for idx, L in enumerate(L_lst):
#         for c in c_lst:
#                 ft_max = df_filter[(df_filter["L"]==L) & (df_filter["c"]==c) & (df_filter["nc"]==nc)]['f_T_min'].values[0]
#                 ft_lst = np.linspace(0.01, ft_max, 20)

#                 for ft in ft_lst:
#                                 rho = 1/nc
#                                 mode_tag = "" if Mode == "sop" else f"_{Mode}"
#                                 layout_tag = "" if InitialLayout == "random" else f"_{InitialLayout}"
#                                 exec_name = f"L_{L}_ft_{ft:.3f}_c_{c}_nc_{nc}_dim_{dim}_p0_{p0}_P0_{P0}{mode_tag}{layout_tag}.sh"

#                                 shell_data(L, type_perc, p0, seed, c, ft, dim,
#                                         nc, num_runs[idx], [1/nc], exec_name, P0, Equilibration, multi,
#                                         properties=Properties, mode=Mode,
#                                         initial_layout=InitialLayout)

import numpy as np
import pandas as pd


type_perc = "bond"

L_lst = [2048, 4096, 8192, 16384]
num_runs = [400, 200, 100, 50]

c_lst = [round(0.01 * i, 2) for i in range(1, 21)]

nc = 1

multi = True
Equilibration = "false"
Properties = "false"

Mode = "growth_test"
InitialLayout = "random"

p0 = 0.8
P0 = 0.2


# Associa cada L ao respectivo número de runs
num_runs_by_L = dict(zip(L_lst, num_runs))


# Dataframe contendo as simulações já realizadas
df = pd.read_csv("../SOP_data/ft_min_max_2D.csv")


# Garante tipos numéricos
df["L"] = pd.to_numeric(df["L"], errors="coerce")
df["c"] = pd.to_numeric(df["c"], errors="coerce")


# Cria um conjunto com todos os pares (L, c) existentes
# O round evita problemas de ponto flutuante
existing_pairs = {
    (int(L), round(float(c), 2))
    for L, c in zip(df["L"], df["c"])
    if pd.notna(L) and pd.notna(c)
}


generated_combinations = 0
skipped_combinations = 0
generated_shells = 0


for L in L_lst:

    runs = num_runs_by_L[L]

    for c in c_lst:

        pair = (L, round(c, 2))

        # Se o par (L, c) já existe no dataframe, ignora
        if pair in existing_pairs:
            print(f"Ignorando L={L}, c={c:.2f}: já existe no dataframe.")
            skipped_combinations += 1
            continue

        print(f"Gerando shells para L={L}, c={c:.2f}.")

        ft_lst = np.linspace(0.01, 0.4, 20)

        rho = 1.0 / nc

        mode_tag = "" if Mode == "sop" else f"_{Mode}"

        layout_tag = (
            ""
            if InitialLayout == "random"
            else f"_{InitialLayout}"
        )

        for ft in ft_lst:

            exec_name = (
                f"L_{L}"
                f"_ft_{ft:.6f}"
                f"_c_{c:.2f}"
                f"_nc_{nc}"
                f"_dim_{dim}"
                f"_p0_{p0:g}"
                f"_P0_{P0:g}"
                f"{mode_tag}"
                f"{layout_tag}.sh"
            )

            shell_data(
                L,
                type_perc,
                p0,
                seed,
                c,
                float(ft),
                dim,
                nc,
                runs,
                [rho],
                exec_name,
                P0,
                Equilibration,
                multi,
                properties=Properties,
                mode=Mode,
                initial_layout=InitialLayout,
            )

            generated_shells += 1

        generated_combinations += 1


print("\nResumo:")
print(f"Combinações ignoradas: {skipped_combinations}")
print(f"Combinações geradas:   {generated_combinations}")
print(f"Shells gerados:        {generated_shells}")