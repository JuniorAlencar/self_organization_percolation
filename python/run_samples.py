import os
from pathlib import Path
import numpy as np
import pandas as pd

from src.run_samples_functions import (
    shell_data,
    custom_range,
    find_dynamic_file,
    load_filtered_dynamic,
    resolve_existing_ft_and_samples,
    build_all_convergence_jobs,
    generate_convergence_shells,
)

# Garante que os scripts shell sejam gerados na pasta shells/ do projeto
os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# PARAMETROS DA SIMULACAO
# =============================================================================

seed = -1
dim = 2
nc = 1                      # Numero de cores (ex: 1, 2, 3, 4...)

type_lst = ['node', 'bond']  # 'node' (site) e/ou 'bond' percolation

# Lista de tamanhos L e orcamento de amostras por tamanho
# L_lst = [512, 1024, 2048, 4096, 8192, 16384]
# num_runs_lst = [700, 500, 400, 200, 100, 50]
L_lst = [16384]
num_runs_lst = [50]
num_runs_por_L = dict(zip(L_lst, num_runs_lst))

c = 0.05
multi = True
Equilibration = 'false'
Properties = 'false'
Mode = 'growth_test'         # 'growth_test' ou 'sop'
InitialLayout = 'clustered'  # 'clustered', 'random', 'blocks', ou 'alternating'
ControlRule = 'linear'       # 'linear' ou 'relative'

# Fracao de densidade por cor (para nc cores simetricas: 1/nc)
rho = [1.0 / nc]

p0 = 0.8
P0 = 0.6

# Grade de valores de f_T (utilizando np.linspace ou lista explicita)
#ft_lst = np.linspace(0.01, 0.3, 20)
ft = 0.1
# =============================================================================
# GERACAO DOS SCRIPTS SHELL
# =============================================================================

for L in L_lst:
    num_runs = num_runs_por_L.get(L, 100)
    for type_perc in type_lst:
        for ft in [ft]:
            mode_tag = "" if Mode == "sop" else f"_{Mode}"
            layout_tag = "" if InitialLayout == "clustered" else f"_{InitialLayout}"
            exec_name = (
                f"L_{L}_ft_{ft:.3f}_c_{c}_nc_{nc}_dim_{dim}"
                f"_p0_{p0}_P0_{P0:.3f}{mode_tag}{layout_tag}_type_{type_perc}.sh"
            )

            shell_data(
                L=L,
                type_perc=type_perc,
                p0=p0,
                seed=seed,
                c=c,
                f_T=ft,
                dim=dim,
                num_colors=nc,
                num_runs=num_runs,
                rho=rho,
                exec_name=exec_name,
                P0=P0,
                equlibration=Equilibration,
                multi=multi,
                properties=Properties,
                mode=Mode,
                initial_layout=InitialLayout,
                control_rule=ControlRule,
            )
